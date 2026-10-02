//! Rust boundary for one non-equality range predicate.
//!
//! PyJanitor supplies non-null arrays. The right values are already sorted,
//! and `right_index` is the companion physical-position array for that sorted
//! layout. Index positions are unique by contract, but are not necessarily
//! ordered; `right_index_is_ordered` therefore controls only `first`/`last`
//! selection.

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
use crate::range_predicate::{parse_any_range_parts, AnyParsedRangePredicate};

fn prefix_extreme(values: &[i64], minimum: bool) -> Vec<i64> {
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
/// `left` and `right` are the same layouts used for the predicate search:
/// `right` is sorted, and both index arrays contain the physical positions for
/// their corresponding layouts. Aggregation requests are already realigned by
/// PyJanitor to those layouts, so their lengths are derived from the index
/// arrays. This lets the existing prefix/suffix aggregation implementations
/// operate without per-match physical-position updates.
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

/// Build dense half-open candidate windows for each left value.
///
/// Both the normal single-range path and the extended residual path use this
/// traversal. The extended path additionally compacts non-empty windows into
/// a `SingleJoinResult` while preserving the corresponding left positions.
/// Materialize a range-first candidate stream after all residual predicates
/// have been evaluated.
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

#[pyfunction]
pub fn range_anchor_extended_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_anchor(py, predicates, aggregations, return_matched, false)
}

#[pyfunction]
pub fn range_anchor_extended_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_anchor(py, predicates, aggregations, return_matched, true)
}

/// The full-layout metadata carried by the range-first aggregation anchor.
///
/// The value and position arrays remain compact because they are the layouts
/// used by the binary search. The two lengths, however, describe the original
/// left and right arrays used by Python to build aggregation inputs.
struct FullLayoutRangeAnchor<'py> {
    range: AnyParsedRangePredicate<'py>,
    left_len: usize,
    right_len: usize,
}

/// Parse the seven-field range-first aggregation ABI.
///
/// The tuple is:
///
/// ```text
/// (left_values, left_positions, right_values, right_positions,
///  left_full_len, right_full_len, operator)
/// ```
///
/// `left_positions` and `right_positions` contain physical positions in the
/// original Python arrays. They are deliberately not sorted or compacted.
fn parse_full_layout_range_anchor<'py>(
    tuple: &Bound<'py, PyTuple>,
) -> PyResult<FullLayoutRangeAnchor<'py>> {
    if tuple.len() != 7 {
        return Err(PyValueError::new_err(
            "range-first aggregation anchor must contain 7 elements",
        ));
    }
    let left_len = tuple.get_item(4)?.extract::<usize>()?;
    let right_len = tuple.get_item(5)?.extract::<usize>()?;
    let range = parse_any_range_parts(
        &tuple.get_item(0)?,
        &tuple.get_item(1)?,
        &tuple.get_item(2)?,
        &tuple.get_item(3)?,
        &tuple.get_item(6)?,
    )?;
    range
        .validate_range_operator()
        .map_err(PyValueError::new_err)?;
    range.validate_lengths().map_err(PyValueError::new_err)?;

    macro_rules! validate_positions {
        ($predicate:expr) => {{
            let predicate = $predicate;
            let invalid_left = predicate.left_index.as_array().iter().any(|&position| {
                position < 0
                    || usize::try_from(position).map_or(true, |position| position >= left_len)
            });
            let invalid_right = predicate.right_index.as_array().iter().any(|&position| {
                position < 0
                    || usize::try_from(position).map_or(true, |position| position >= right_len)
            });
            if invalid_left || invalid_right {
                return Err(PyValueError::new_err(
                    "range-first aggregation positions must address the full input arrays",
                ));
            }
        }};
    }
    match &range {
        AnyParsedRangePredicate::I64(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::I32(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::I16(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::I8(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::U64(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::U32(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::U16(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::U8(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::F64(predicate) => validate_positions!(predicate),
        AnyParsedRangePredicate::F32(predicate) => validate_positions!(predicate),
    }

    Ok(FullLayoutRangeAnchor {
        range,
        left_len,
        right_len,
    })
}

fn aggregate_range_anchor<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range-first extended aggregation requires at least two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let first = first_item.cast::<PyTuple>()?;
    let anchor = parse_full_layout_range_anchor(first)?;
    let (parsed, metadata) =
        crate::join_aggregation_helpers::residuals(py, predicates, false, false)?;
    let windows = aggregation_windows(&anchor.range, true).map_err(PyValueError::new_err)?;
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
