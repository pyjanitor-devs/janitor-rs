//! Rust boundary for single range predicates and range-first residual joins.
//!
//! PyJanitor owns pandas preparation: null removal, right-anchor sorting, and
//! construction of physical position maps. This module owns the public PyO3
//! boundary, binary-search windows, `keep` materialization, residual dispatch,
//! and range-first aggregation.
//!
//! A position array always contains a physical row position in the original
//! Python array. It is not a sorted offset and is never compacted. The value
//! arrays may be compact and the right values may be sorted, so every sorted
//! right slot must be translated through `right_index` before it is returned
//! or used to index a full-layout aggregation array.
//!
//! The ordinary single-range ABI is:
//!
//! ```text
//! (left_index, left_values, right_index, right_values,
//!  right_index_is_ordered, operator, keep, return_building_blocks)
//! ```
//!
//! The range-first index ABI is a list whose first tuple is:
//!
//! ```text
//! (left_values, left_positions, right_values, right_positions, operator)
//! ```
//!
//! Range-first aggregation extends that first tuple with the full source
//! lengths and uses the complete source aggregation arrays:
//!
//! ```text
//! (left_values, left_positions, right_values, right_positions,
//!  left_full_len, right_full_len, operator)
//! ```
//!
//! Later tuples are residual predicates. They are evaluated using compact
//! offsets, then successful candidates are translated through the anchor
//! position maps before updating full-layout aggregation state.

use numpy::ndarray::ArrayView1;
use numpy::{Element, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::common::{range_window, range_window_bounds};
use crate::join_aggregation_helpers::{aggregate_range_windows, aggregation_windows};
use crate::join_candidate_materialization::materialize_range_candidates;
use crate::join_common::{result_dict, Keep, SingleJoinResult};
use crate::op::CompareOp;
use crate::predicate::{check_predicate_lengths, parse_predicates_with_nulls_strings};
use crate::range_predicate::parse_full_layout_aggregation_anchor;

fn prefix_extreme(values: &[i64], minimum: bool) -> Vec<i64> {
    // Prefix extrema are used only when the right physical positions are not
    // ordered. `first` and `last` refer to physical output order, not to the
    // order in which sorted right values happen to be searched.
    let initial = if minimum { i64::MAX } else { i64::MIN };
    values
        .iter()
        .copied()
        .scan(initial, |current, value| {
            *current = if minimum {
                (*current).min(value)
            } else {
                (*current).max(value)
            };
            Some(*current)
        })
        .collect()
}

fn suffix_extreme(values: &[i64], minimum: bool) -> Vec<i64> {
    // The suffix table is the reverse-direction counterpart of
    // `prefix_extreme`; it lets each binary-search window select its physical
    // minimum or maximum without rescanning the window.
    let mut result = vec![0_i64; values.len()];
    let initial = if minimum { i64::MAX } else { i64::MIN };
    let mut current = initial;
    for (offset, &value) in values.iter().enumerate().rev() {
        current = if minimum {
            current.min(value)
        } else {
            current.max(value)
        };
        result[offset] = current;
    }
    result
}

/// Materialize one range predicate from already-built candidate windows.
///
/// `starts` and `ends` are offsets into the sorted right-value layout. The
/// returned right positions are always values from `right_index`, never those
/// offsets. When `right_index_is_ordered` is false, `first` and `last` use
/// prefix/suffix extrema over physical positions; `any` may select the first
/// binary-search hit because it has no ordering guarantee.
///
/// # Arguments
///
/// * `left_index` - Physical left positions, one per left search value.
/// * `right_index` - Physical right positions in sorted right-value order.
/// * `starts` / `ends` - Half-open right-window boundaries per left value.
/// * `op` - Range operator that produced the windows.
/// * `right_index_is_ordered` - Whether physical right positions are ordered.
/// * `keep` - Output selection policy.
///
/// # Returns
///
/// Physical left/right position pairs, or an empty pair when no window has a
/// candidate.
fn materialize(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    starts: &[usize],
    ends: &[usize],
    op: CompareOp,
    right_index_is_ordered: bool,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let mut output_len = 0_usize;
    for (&start, &end) in starts.iter().zip(ends) {
        if keep == Keep::All {
            output_len = output_len
                .checked_add(end - start)
                .ok_or("single range result size exceeds platform capacity")?;
        } else if start < end {
            output_len = output_len
                .checked_add(1)
                .ok_or("single range result size exceeds platform capacity")?;
        }
    }
    if output_len == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let mut output_left = Vec::with_capacity(output_len);
    let mut output_right = Vec::with_capacity(output_len);
    let labels = right_index.to_vec();
    let prefix_min =
        (!right_index_is_ordered && keep == Keep::First).then(|| prefix_extreme(&labels, true));
    let prefix_max =
        (!right_index_is_ordered && keep == Keep::Last).then(|| prefix_extreme(&labels, false));
    let suffix_min =
        (!right_index_is_ordered && keep == Keep::First).then(|| suffix_extreme(&labels, true));
    let suffix_max =
        (!right_index_is_ordered && keep == Keep::Last).then(|| suffix_extreme(&labels, false));

    for (row, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start == end {
            continue;
        }
        if keep == Keep::All {
            for &right_position in &labels[start..end] {
                output_left.push(left_index[row]);
                output_right.push(right_position);
            }
            continue;
        }
        let suffix = matches!(op, CompareOp::Lt | CompareOp::Le);
        let selected = match keep {
            Keep::Any => labels[start],
            Keep::First if right_index_is_ordered => labels[start],
            Keep::Last if right_index_is_ordered => labels[end - 1],
            Keep::First if suffix => suffix_min.as_ref().ok_or("missing suffix extrema")?[start],
            Keep::Last if suffix => suffix_max.as_ref().ok_or("missing suffix extrema")?[start],
            Keep::First => prefix_min.as_ref().ok_or("missing prefix extrema")?[end - 1],
            Keep::Last => prefix_max.as_ref().ok_or("missing prefix extrema")?[end - 1],
            Keep::All => unreachable!(),
        };
        output_left.push(left_index[row]);
        output_right.push(selected);
    }
    Ok((output_left, output_right))
}

#[allow(clippy::too_many_arguments)]
fn run_single_range<'py, T: PartialOrd + Copy + Element>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, T>,
    left_index: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    right_index_is_ordered: bool,
    operator: &str,
    keep: &str,
    return_building_blocks: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // This function is the typed implementation behind every public
    // dtype-specialised single-range index function. It validates the
    // operator, constructs binary-search windows, and only then either
    // exposes those windows or applies `keep`.
    let op = CompareOp::try_from_str(operator)?;
    if !matches!(
        op,
        CompareOp::Gt | CompareOp::Ge | CompareOp::Lt | CompareOp::Le
    ) {
        return Err(PyValueError::new_err(
            "single range predicate requires <, <=, >, or >=",
        ));
    }
    let left = left.as_array();
    let left_index = left_index.as_array();
    let right = right.as_array();
    let right_index = right_index.as_array();
    let (starts, ends) = range_window_bounds(left, left_index, right, right_index, op)
        .map_err(PyValueError::new_err)?;
    let has_match = starts.iter().zip(&ends).any(|(start, end)| start < end);
    if !has_match {
        return Ok(None);
    }
    if return_building_blocks {
        return Ok(Some(result_dict(
            py,
            left_index.to_vec(),
            right_index.to_vec(),
            Some(starts),
            Some(ends),
        )?));
    }
    let keep = Keep::parse(keep)?;
    let (output_left, output_right) = materialize(
        left_index,
        right_index,
        &starts,
        &ends,
        op,
        right_index_is_ordered,
        keep,
    )
    .map_err(PyValueError::new_err)?;
    if output_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(
        py,
        output_left,
        output_right,
        None,
        None,
    )?))
}

macro_rules! registered_single_range {
    ($name:ident, $type:ty) => {
        /// Dtype-specialised Python entry point for one range predicate.
        ///
        /// The Python caller supplies value arrays and physical position
        /// arrays. The right values must already be sorted in ascending order;
        /// `right_index_is_ordered` describes only the physical position order
        /// used by `first` and `last`.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<'py>(
            py: Python<'py>,
            left_index: PyReadonlyArray1<'py, i64>,
            left: PyReadonlyArray1<'py, $type>,
            right_index: PyReadonlyArray1<'py, i64>,
            right: PyReadonlyArray1<'py, $type>,
            right_index_is_ordered: bool,
            operator: &str,
            keep: &str,
            return_building_blocks: bool,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            run_single_range(
                py,
                left,
                left_index,
                right,
                right_index,
                right_index_is_ordered,
                operator,
                keep,
                return_building_blocks,
            )
        }
    };
}

registered_single_range!(single_range_predicate_indices_int64, i64);
registered_single_range!(single_range_predicate_indices_int32, i32);
registered_single_range!(single_range_predicate_indices_int16, i16);
registered_single_range!(single_range_predicate_indices_int8, i8);
registered_single_range!(single_range_predicate_indices_uint64, u64);
registered_single_range!(single_range_predicate_indices_uint32, u32);
registered_single_range!(single_range_predicate_indices_uint16, u16);
registered_single_range!(single_range_predicate_indices_uint8, u8);
registered_single_range!(single_range_predicate_indices_f64, f64);
registered_single_range!(single_range_predicate_indices_f32, f32);

/// Aggregate a single range predicate using layout-aligned source arrays.
///
/// Unlike the range-first extended aggregation path below, this function
/// receives aggregation arrays already aligned to the compact predicate
/// layouts. That alignment permits the specialised starts/ends aggregation
/// implementations to index directly without a physical-position map.
///
/// # Arguments
///
/// * `left_index` / `right_index` - Physical positions paired with the
///   prepared left/right value layouts.
/// * `left` / `right` - Non-null values; `right` is sorted for binary search.
/// * `operator` - One of `<`, `<=`, `>`, or `>=`.
/// * `aggregations` - Rust aggregation requests prepared by Python.
/// * `return_matched` - Include one match flag per output slot.
/// * `reverse` - Aggregate left source values into right output slots.
///
#[allow(clippy::too_many_arguments)]
fn aggregate_single_range<'py, T: PartialOrd + Copy + Element>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    left: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, T>,
    operator: &str,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let op = CompareOp::try_from_str(operator)?;
    if !matches!(
        op,
        CompareOp::Gt | CompareOp::Ge | CompareOp::Lt | CompareOp::Le
    ) {
        return Err(PyValueError::new_err(
            "single range aggregation requires <, <=, >, or >=",
        ));
    }
    let left = left.as_array();
    let left_index = left_index.as_array();
    let right = right.as_array();
    let right_index = right_index.as_array();
    if left.len() != left_index.len() {
        return Err(PyValueError::new_err(
            "left values and left index must have equal lengths",
        ));
    }
    if right.len() != right_index.len() {
        return Err(PyValueError::new_err(
            "right values and right index must have equal lengths",
        ));
    }

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let output_len = if reverse {
        right_index.len()
    } else {
        left_index.len()
    };
    let source_len = if reverse {
        left_index.len()
    } else {
        right_index.len()
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    let mut boundaries = Vec::with_capacity(left.len());
    for &left_value in left {
        let (start, end) = range_window(left_value, right, op);
        let boundary = if matches!(op, CompareOp::Lt | CompareOp::Le) {
            start
        } else {
            end
        };
        boundaries.push(i64::try_from(boundary).map_err(|_| {
            PyValueError::new_err("single range aggregation boundary exceeds int64")
        })?);
    }

    let boundaries = ArrayView1::from(&boundaries);
    if matches!(op, CompareOp::Lt | CompareOp::Le) {
        if reverse {
            set.aggregate_reverse_starts(boundaries);
        } else {
            set.aggregate_starts(boundaries);
        }
    } else if reverse {
        set.aggregate_reverse_ends(boundaries);
    } else {
        set.aggregate_ends(boundaries);
    }

    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results_with_positions(
        py,
        set,
        Some(if reverse { right_index } else { left_index }),
        output_len,
        return_matched,
    )?))
}

macro_rules! registered_single_range_aggregation {
    ($forward:ident, $reverse:ident, $type:ty) => {
        /// Forward single-range aggregation entry point for one numeric dtype.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $forward<'py>(
            py: Python<'py>,
            left_index: PyReadonlyArray1<'py, i64>,
            left: PyReadonlyArray1<'py, $type>,
            right_index: PyReadonlyArray1<'py, i64>,
            right: PyReadonlyArray1<'py, $type>,
            operator: &str,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            aggregate_single_range(
                py,
                left_index,
                left,
                right_index,
                right,
                operator,
                aggregations,
                return_matched,
                false,
            )
        }

        /// Reverse single-range aggregation entry point for one numeric dtype.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $reverse<'py>(
            py: Python<'py>,
            left_index: PyReadonlyArray1<'py, i64>,
            left: PyReadonlyArray1<'py, $type>,
            right_index: PyReadonlyArray1<'py, i64>,
            right: PyReadonlyArray1<'py, $type>,
            operator: &str,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            aggregate_single_range(
                py,
                left_index,
                left,
                right_index,
                right,
                operator,
                aggregations,
                return_matched,
                true,
            )
        }
    };
}

registered_single_range_aggregation!(
    single_range_aggregate_int64,
    single_range_aggregate_reverse_int64,
    i64
);
registered_single_range_aggregation!(
    single_range_aggregate_int32,
    single_range_aggregate_reverse_int32,
    i32
);
registered_single_range_aggregation!(
    single_range_aggregate_int16,
    single_range_aggregate_reverse_int16,
    i16
);
registered_single_range_aggregation!(
    single_range_aggregate_int8,
    single_range_aggregate_reverse_int8,
    i8
);
registered_single_range_aggregation!(
    single_range_aggregate_uint64,
    single_range_aggregate_reverse_uint64,
    u64
);
registered_single_range_aggregation!(
    single_range_aggregate_uint32,
    single_range_aggregate_reverse_uint32,
    u32
);
registered_single_range_aggregation!(
    single_range_aggregate_uint16,
    single_range_aggregate_reverse_uint16,
    u16
);
registered_single_range_aggregation!(
    single_range_aggregate_uint8,
    single_range_aggregate_reverse_uint8,
    u8
);
registered_single_range_aggregation!(
    single_range_aggregate_f64,
    single_range_aggregate_reverse_f64,
    f64
);
registered_single_range_aggregation!(
    single_range_aggregate_f32,
    single_range_aggregate_reverse_f32,
    f32
);

/// Materialize a range-first candidate stream after residual predicates.
///
/// The first range predicate builds one half-open window per compact left
/// value. Later predicates are parsed as residuals and evaluated for every
/// candidate in that window before `keep` is applied. This ordering is
/// essential: `first` and `last` describe the first/last surviving candidate,
/// not merely the first/last candidate from the anchor window.
///
/// # Arguments
///
/// * `predicates` - At least two tuples: a five-field range anchor followed by
///   residual comparison tuples.
/// * `keep` - Final selection policy after residual filtering.
/// * `left_index` / `right_index` - Physical position arrays for the compact
///   anchor layouts.
/// * `left` / `right` - Typed anchor values; right values are sorted.
/// * `operator` - Parsed range operator for the first predicate.
#[allow(clippy::too_many_arguments)]
fn materialize_range_first_indices<'py, T: Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    left_index: PyReadonlyArray1<'py, i64>,
    left: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, T>,
    operator: CompareOp,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range-first extended join requires at least two predicates",
        ));
    }
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    check_predicate_lengths(&parsed, left.as_array().len(), right.as_array().len())?;
    let (all_starts, all_ends) = range_window_bounds(
        left.as_array(),
        left_index.as_array(),
        right.as_array(),
        right_index.as_array(),
        operator,
    )
    .map_err(PyValueError::new_err)?;
    let mut windows = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: right_index.as_array().to_vec(),
        starts: Vec::new(),
        ends: Vec::new(),
    };
    for (left_position, (&start, &end)) in all_starts.iter().zip(&all_ends).enumerate() {
        if start == end {
            continue;
        }
        windows.left_positions.push(left_position);
        windows
            .left_index
            .push(left_index.as_array()[left_position]);
        windows.starts.push(start);
        windows.ends.push(end);
    }
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    let (output_left, output_right) =
        materialize_range_candidates(&windows, &parsed, metadata.as_deref(), Keep::parse(keep)?)
            .map_err(PyValueError::new_err)?;
    if output_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(
        py,
        output_left,
        output_right,
        None,
        None,
    )?))
}

macro_rules! registered_range_first_extended_indices {
    ($name:ident, $type:ty, $export:literal) => {
        /// Dtype-specialised Python entry point for a range anchor followed
        /// by one or more residual predicates.
        ///
        /// The first tuple must be the five-field range anchor; all later
        /// tuples are evaluated against the anchor's compact layouts. The
        /// `keep` policy is applied only after residual filtering.
        #[pyfunction(name = $export)]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            keep: &str,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            if first.len() != 5 {
                return Err(PyValueError::new_err(
                    "range-first anchor must contain 5 elements",
                ));
            }
            let operator = CompareOp::try_from_str(first.get_item(4)?.extract()?)?;
            if !matches!(
                operator,
                CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
            ) {
                return Err(PyValueError::new_err(
                    "range-first anchor requires <, <=, >, or >=",
                ));
            }
            materialize_range_first_indices::<$type>(
                py,
                predicates,
                keep,
                first.get_item(1)?.extract()?,
                first.get_item(0)?.extract()?,
                first.get_item(3)?.extract()?,
                first.get_item(2)?.extract()?,
                operator,
            )
        }
    };
}

registered_range_first_extended_indices!(
    range_anchor_extended_indices_int64,
    i64,
    "range_anchor_extended_indices_int64"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_int32,
    i32,
    "range_anchor_extended_indices_int32"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_int16,
    i16,
    "range_anchor_extended_indices_int16"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_int8,
    i8,
    "range_anchor_extended_indices_int8"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_uint64,
    u64,
    "range_anchor_extended_indices_uint64"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_uint32,
    u32,
    "range_anchor_extended_indices_uint32"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_uint16,
    u16,
    "range_anchor_extended_indices_uint16"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_uint8,
    u8,
    "range_anchor_extended_indices_uint8"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_f64,
    f64,
    "range_anchor_extended_indices_f64"
);
registered_range_first_extended_indices!(
    range_anchor_extended_indices_f32,
    f32,
    "range_anchor_extended_indices_f32"
);

/// Forward range-first aggregation: right source values are aggregated into
/// one output slot per full-layout left row.
#[pyfunction]
pub fn range_anchor_extended_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_anchor(py, predicates, aggregations, return_matched, false)
}

/// Reverse range-first aggregation: left source values are aggregated into
/// one output slot per full-layout right row.
#[pyfunction]
pub fn range_anchor_extended_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_anchor(py, predicates, aggregations, return_matched, true)
}

fn aggregate_range_anchor<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // The first tuple is parsed separately from residuals because it both
    // defines the binary-search windows and carries the full source lengths.
    // Residuals continue to use their ordinary compact predicate ABI.
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range-first extended aggregation requires at least two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let first = first_item.cast::<PyTuple>()?;
    let anchor = parse_full_layout_aggregation_anchor(first)?;
    let (parsed, metadata) =
        crate::join_aggregation_helpers::residuals(py, predicates, false, false)?;
    let (starts, ends) = anchor.range.bounds().map_err(PyValueError::new_err)?;
    let windows =
        aggregation_windows(&anchor.range, starts, ends, true).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    let output_len = if reverse {
        anchor.right_len
    } else {
        anchor.left_len
    };
    let source_len = if reverse {
        anchor.left_len
    } else {
        anchor.right_len
    };
    aggregate_range_windows(
        py,
        windows,
        &parsed,
        metadata.as_deref(),
        aggregations,
        None,
        output_len,
        source_len,
        return_matched,
        reverse,
        true,
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Keep all single-range exports together. Python dispatch tables import
    // these names directly, so changing a registration name is an ABI change
    // even when the underlying Rust implementation is unchanged.
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_predicate_indices_f32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_f32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_range_aggregate_reverse_f32, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_indices_f32, m)?)?;
    m.add_function(wrap_pyfunction!(range_anchor_extended_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(
        range_anchor_extended_aggregate_reverse,
        m
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array1, PyArray1, PyArrayMethods};

    #[test]
    fn windows_cover_each_range_operator() {
        let right = Array1::from_vec(vec![1_i64, 3, 5]);
        assert_eq!(range_window(3, right.view(), CompareOp::Gt), (0, 1));
        assert_eq!(range_window(3, right.view(), CompareOp::Ge), (0, 2));
        assert_eq!(range_window(3, right.view(), CompareOp::Lt), (2, 3));
        assert_eq!(range_window(3, right.view(), CompareOp::Le), (1, 3));
    }

    #[test]
    fn unordered_index_uses_physical_extrema() {
        let left_index = Array1::from_vec(vec![90_i64]);
        let right_index = Array1::from_vec(vec![40_i64, 10, 30]);
        let (out_left, out_right) = materialize(
            left_index.view(),
            right_index.view(),
            &[0],
            &[1],
            CompareOp::Gt,
            false,
            Keep::First,
        )
        .unwrap();
        assert_eq!(out_left, vec![90]);
        assert_eq!(out_right, vec![40]);
    }

    #[test]
    fn single_indices_cover_all_operators_and_equal_boundaries() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            for (operator, expected) in [
                (">", vec![40_i64]),
                (">=", vec![40_i64, 10]),
                ("<", vec![30_i64, 20]),
                ("<=", vec![10_i64, 30, 20]),
            ] {
                let result = single_range_predicate_indices_int64(
                    py,
                    PyArray1::from_vec(py, vec![100_i64]).readonly(),
                    PyArray1::from_vec(py, vec![4_i64]).readonly(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).readonly(),
                    PyArray1::from_vec(py, vec![1_i64, 4, 5, 7]).readonly(),
                    false,
                    operator,
                    "all",
                    false,
                )?
                .expect("the operator should produce matches");
                assert_eq!(
                    read_extended_pair(&result),
                    (vec![100; expected.len()], expected)
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn single_indices_keep_respects_ordered_and_unordered_right_indexes() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let args = |ordered, keep| {
                single_range_predicate_indices_int64(
                    py,
                    PyArray1::from_vec(py, vec![100_i64]).readonly(),
                    PyArray1::from_vec(py, vec![4_i64]).readonly(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).readonly(),
                    PyArray1::from_vec(py, vec![1_i64, 4, 5, 7]).readonly(),
                    ordered,
                    "<=",
                    keep,
                    false,
                )
            };

            assert_eq!(
                read_extended_pair(&args(false, "any")?.unwrap()),
                (vec![100], vec![10])
            );
            assert_eq!(
                read_extended_pair(&args(false, "first")?.unwrap()),
                (vec![100], vec![10])
            );
            assert_eq!(
                read_extended_pair(&args(false, "last")?.unwrap()),
                (vec![100], vec![30])
            );
            assert_eq!(
                read_extended_pair(&args(true, "first")?.unwrap()),
                (vec![100], vec![10])
            );
            assert_eq!(
                read_extended_pair(&args(true, "last")?.unwrap()),
                (vec![100], vec![20])
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn single_indices_return_building_blocks_before_keep_materialization() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let result = single_range_predicate_indices_int64(
                py,
                PyArray1::from_vec(py, vec![100_i64]).readonly(),
                PyArray1::from_vec(py, vec![4_i64]).readonly(),
                PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).readonly(),
                PyArray1::from_vec(py, vec![1_i64, 4, 5, 7]).readonly(),
                false,
                "<=",
                "first",
                true,
            )?
            .expect("the operator should produce a window");
            assert_eq!(
                result
                    .get_item("left_index")?
                    .unwrap()
                    .extract::<Vec<i64>>()?,
                vec![100]
            );
            assert_eq!(
                result
                    .get_item("right_index")?
                    .unwrap()
                    .extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            assert_eq!(
                result.get_item("starts")?.unwrap().extract::<Vec<i64>>()?,
                vec![1]
            );
            assert_eq!(
                result.get_item("ends")?.unwrap().extract::<Vec<i64>>()?,
                vec![4]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn single_indices_return_none_for_empty_or_non_matching_inputs() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let no_match = single_range_predicate_indices_int64(
                py,
                PyArray1::from_vec(py, vec![100_i64]).readonly(),
                PyArray1::from_vec(py, vec![0_i64]).readonly(),
                PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).readonly(),
                PyArray1::from_vec(py, vec![1_i64, 4, 5, 7]).readonly(),
                false,
                ">",
                "all",
                false,
            )?;
            assert!(no_match.is_none());

            let empty_left = single_range_predicate_indices_int64(
                py,
                PyArray1::from_vec(py, Vec::<i64>::new()).readonly(),
                PyArray1::from_vec(py, Vec::<i64>::new()).readonly(),
                PyArray1::from_vec(py, vec![40_i64]).readonly(),
                PyArray1::from_vec(py, vec![1_i64]).readonly(),
                true,
                "<",
                "all",
                false,
            )?;
            assert!(empty_left.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn single_indices_reject_invalid_operator_and_lengths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let invalid_operator = single_range_predicate_indices_int64(
                py,
                PyArray1::from_vec(py, vec![100_i64]).readonly(),
                PyArray1::from_vec(py, vec![4_i64]).readonly(),
                PyArray1::from_vec(py, vec![40_i64]).readonly(),
                PyArray1::from_vec(py, vec![1_i64]).readonly(),
                true,
                "==",
                "all",
                false,
            );
            assert!(invalid_operator.is_err());

            let mismatched_left = single_range_predicate_indices_int64(
                py,
                PyArray1::from_vec(py, vec![100_i64, 200]).readonly(),
                PyArray1::from_vec(py, vec![4_i64]).readonly(),
                PyArray1::from_vec(py, vec![40_i64]).readonly(),
                PyArray1::from_vec(py, vec![1_i64]).readonly(),
                true,
                "<",
                "all",
                false,
            );
            assert!(mismatched_left.is_err());

            let mismatched_right = single_range_predicate_indices_int64(
                py,
                PyArray1::from_vec(py, vec![100_i64]).readonly(),
                PyArray1::from_vec(py, vec![4_i64]).readonly(),
                PyArray1::from_vec(py, vec![40_i64, 10]).readonly(),
                PyArray1::from_vec(py, vec![1_i64]).readonly(),
                true,
                "<",
                "all",
                false,
            );
            assert!(mismatched_right.is_err());
            Ok(())
        })
        .unwrap();
    }

    fn extended_range_predicates<'py>(
        py: Python<'py>,
        operator: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let predicates = PyList::empty(py);
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, vec![4_i64]).into_any(),
                PyArray1::from_vec(py, vec![100_i64]).into_any(),
                PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                operator.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, vec![0_i64]).into_any(),
                PyArray1::from_vec(py, vec![1_i64, 1, 1, 1]).into_any(),
                "<".into_pyobject(py)?.into_any(),
            ],
        )?)?;
        Ok(predicates)
    }

    fn read_extended_pair<'py>(result: &Bound<'py, PyDict>) -> (Vec<i64>, Vec<i64>) {
        let left = result
            .get_item("left_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        let right = result
            .get_item("right_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        (left, right)
    }

    #[test]
    fn extended_indices_cover_all_range_operators() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            for (operator, expected) in [
                (">", vec![40_i64, 10]),
                (">=", vec![40_i64, 10]),
                ("<", vec![30_i64, 20]),
                ("<=", vec![30_i64, 20]),
            ] {
                let predicates = extended_range_predicates(py, operator)?;
                let result = range_anchor_extended_indices_int64(py, &predicates, "all")?
                    .expect("range anchor should produce matches");
                assert_eq!(read_extended_pair(&result), (vec![100, 100], expected));
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_indices_apply_keep_after_residuals() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            for (keep, expected) in [
                ("any", vec![30_i64]),
                ("first", vec![20_i64]),
                ("last", vec![30_i64]),
                ("all", vec![30_i64, 20]),
            ] {
                let predicates = extended_range_predicates(py, "<")?;
                let result = range_anchor_extended_indices_int64(py, &predicates, keep)?
                    .expect("range anchor should produce matches");
                assert_eq!(
                    read_extended_pair(&result),
                    (vec![100; expected.len()], expected)
                );
            }
            Ok(())
        })
        .unwrap();
    }

    fn sum_request<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyTuple>> {
        let mask = vec![false; values.len()];
        PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, values).into_any(),
                PyArray1::from_vec(py, mask).into_any(),
                "sum".into_pyobject(py)?.into_any(),
            ],
        )
    }

    #[test]
    fn forward_aggregation_uses_sorted_aligned_source_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // The physical right values [10, 20, 30, 40] have been aligned to
            // the sorted right layout [0, 1, 3] before crossing the ABI.
            let aggregations = PyList::new(py, [sum_request(py, vec![10, 20, 40])?])?;
            let result = single_range_aggregate_int64(
                py,
                PyArray1::from_vec(py, vec![2_i64]).readonly(),
                PyArray1::from_vec(py, vec![4_i64]).readonly(),
                PyArray1::from_vec(py, vec![0_i64, 1, 3]).readonly(),
                PyArray1::from_vec(py, vec![1_i64, 3, 5]).readonly(),
                "<",
                &aggregations,
                true,
            )?
            .expect("the range has a matching right row");
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let values_item = result.get_item(2)?;
            let values = values_item.cast::<PyList>()?;
            assert_eq!(values.get_item(0)?.extract::<Vec<i64>>()?, vec![40]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_aggregation_maps_compact_anchor_positions_to_full_layouts() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 0, 2]).into_any(),
                    3_usize.into_pyobject(py)?.into_any(),
                    4_usize.into_pyobject(py)?.into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64; 4]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            // The source right values use their original physical order, not
            // the sorted anchor order [1, 3, 0, 2]. The matching sorted
            // offsets are 2 and 3, which map to physical positions 0 and 2.
            let right_aggregations = PyList::new(py, [sum_request(py, vec![50_i64, 10, 70, 30])?])?;
            let forward =
                range_anchor_extended_aggregate(py, &predicates, &right_aggregations, true)?
                    .expect("the range has matching right rows");
            assert_eq!(forward.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 1, 2]);
            assert_eq!(
                forward.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true]
            );
            let forward_values_item = forward.get_item(2)?;
            let forward_values = forward_values_item.cast::<PyList>()?;
            assert_eq!(
                forward_values.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 120]
            );

            let left_aggregations = PyList::new(py, [sum_request(py, vec![100_i64, 200, 300])?])?;
            let reverse =
                range_anchor_extended_aggregate_reverse(py, &predicates, &left_aggregations, true)?
                    .expect("the range has matching right rows");
            assert_eq!(
                reverse.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, false, true, false]
            );
            let reverse_values_item = reverse.get_item(2)?;
            let reverse_values = reverse_values_item.cast::<PyList>()?;
            assert_eq!(
                reverse_values.get_item(0)?.extract::<Vec<i64>>()?,
                vec![300, 0, 300, 0]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn reverse_aggregation_uses_left_layout_and_sorted_right_output() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // The left predicate layout contains only physical row 2, so the
            // source aggregation is aligned to that one-row layout.
            let aggregations = PyList::new(py, [sum_request(py, vec![300])?])?;
            let result = single_range_aggregate_reverse_int64(
                py,
                PyArray1::from_vec(py, vec![2_i64]).readonly(),
                PyArray1::from_vec(py, vec![4_i64]).readonly(),
                PyArray1::from_vec(py, vec![0_i64, 1, 3]).readonly(),
                PyArray1::from_vec(py, vec![1_i64, 3, 5]).readonly(),
                "<",
                &aggregations,
                true,
            )?
            .expect("the range has a matching right row");
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true]
            );
            let values_item = result.get_item(2)?;
            let values = values_item.cast::<PyList>()?;
            assert_eq!(values.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 0, 300]);
            Ok(())
        })
        .unwrap();
    }
}
