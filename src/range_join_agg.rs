//! Fused aggregation for exactly two range predicates.
//!
//! The two range predicates are intersected into one half-open window per
//! left row. The windows are then passed to the existing optimized
//! `AggregationSet` range machinery; no pair indices are materialized.
//!
//! This module contains two related public entry-point families:
//!
//! * `range_join_aggregate_*` handles exactly two range predicates and uses
//!   the optimized prefix/suffix range kernels directly.
//! * `range_join_extended_aggregate_*` handles the same two range anchors and
//!   then evaluates any later predicates as residual filters before updating
//!   aggregation state.
//!
//! PyJanitor is responsible for removing null rows, aligning each predicate's
//! arrays, and sorting the right-hand range arrays in ascending order before
//! calling these functions. Rust validates the tuple shape and lengths but
//! does not sort the input.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::join_aggregation_helpers::{aggregate_range_windows, check_residual_lengths, residuals};
use crate::op::CompareOp;
use crate::range_join::{
    build_windows, parse_range_predicate, ParsedRangePredicate, RangePredicate,
};

/// Aggregate a range-led extended join with two confirmed range anchors.
///
/// The first two predicates are parsed as ascending range predicates. Their
/// half-open windows are intersected before predicates three onward are
/// evaluated as residual filters. Aggregation state is updated only after all
/// residual predicates pass; no intermediate pair index is materialized.
///
/// This belongs beside the basic two-range aggregation entry points because
/// its defining operation is dual-range window construction. The single
/// extended aggregation path uses `run_range` for one anchor and treats every
/// later predicate as a residual instead.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to parse residual tuples and
///   construct the returned aggregation tuple.
/// * `predicates` - At least two aligned predicates. The first predicate may
///   use the six-element range tuple
///   `(left, left_index, right, right_index, right_index_is_ordered,
///   comparator)`, or the eight-element form with output-position arrays in
///   fields four and five. The second predicate uses the five-element
///   extended-anchor form `(left, left_index, right, right_index,
///   comparator)` because its ordering flag is not needed by aggregation.
///   Predicates after the first two are residual filters evaluated in order.
/// * `first` - The already-parsed first range anchor, including its borrowed
///   arrays and aligned index labels.
/// * `aggregations` - Non-empty aggregation requests over the source layout.
/// * `output_positions` - Optional compact-to-physical output mapping from the
///   eight-element first-anchor contract. The selected map is the left map in
///   forward mode and the right map in reverse mode; each entry identifies the
///   original physical row represented by that compact aggregation slot.
/// * `output_len` - Number of compact output slots in the selected forward or
///   reverse layout. It must equal the length of the selected output map when
///   that map is supplied.
/// * `return_matched` - Whether to include the per-output matched mask.
/// * `reverse` - If true, aggregate left source values into right output
///   slots; otherwise aggregate right source values into left output slots.
///
/// # Returns
///
/// Returns the standard aggregation tuple
/// `(output_positions, matched, aggregation_arrays)` when `return_matched` is
/// true, or `(output_positions, aggregation_arrays)` otherwise. Returns
/// `None` when no candidate survives both range anchors and all residual
/// predicates.
///
/// # Errors
///
/// Returns a Python `ValueError` for malformed range tuples, non-range second
/// comparators, mismatched residual layouts, invalid aggregation requests, or
/// invalid output-position metadata.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_range_extended<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    first: ParsedRangePredicate<'py, T>,
    aggregations: &Bound<'py, PyList>,
    output_positions: Option<ArrayView1<'_, i64>>,
    output_len: usize,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let second_object = predicates.get_item(1)?;
    let second_tuple = second_object.cast::<PyTuple>()?;
    // The second anchor is already known to be part of the dual-range
    // extended contract, so it uses the compact five-field form. The ordinary
    // six-field parser belongs to the basic two-range API, where the shared
    // ordering flag is retained in every predicate tuple.
    let second = crate::range_join::parse_extended_range_predicate::<T>(second_tuple)?;
    if !second.op.is_range() {
        return Err(PyValueError::new_err(
            "the second range predicate must use <, <=, >, or >=",
        ));
    }
    let (parsed, metadata) = residuals(py, predicates, false, true)?;
    let left = first.left.as_array();
    let right = first.right.as_array();
    check_residual_lengths(&parsed, left.len(), right.len())?;
    let windows = build_windows(
        RangePredicate {
            left,
            left_index: first.left_index.as_array(),
            right,
            right_index: first.right_index.as_array(),
            op: first.op,
        },
        RangePredicate {
            left: second.left.as_array(),
            left_index: second.left_index.as_array(),
            right: second.right.as_array(),
            right_index: second.right_index.as_array(),
            op: second.op,
        },
    )
    .map_err(PyValueError::new_err)?;
    aggregate_range_windows(
        py,
        windows,
        &parsed,
        metadata.as_deref(),
        aggregations,
        output_positions,
        output_len,
        if reverse { left.len() } else { right.len() },
        return_matched,
        reverse,
    )
}

/// Parse and dispatch the range-led extended aggregation contract.
///
/// This is the Python-boundary adapter for [`aggregate_range_extended`]. The
/// first tuple may use the ordinary six-element range form or the eight-
/// element form that additionally carries trimmed output-position maps. The
/// second tuple must use the ordinary six-element range form. Any predicates
/// after those two anchors remain residual filters. The
/// `right_index_is_ordered` field is accepted as part of the shared tuple
/// contract. Aggregation does not use its value: PyJanitor has already
/// established ascending order before this function is called.
///
/// ELI5: this function is the adapter between Python's tuple-shaped API and
/// Rust's typed aggregation code. It checks the tuple, borrows the arrays,
/// identifies which side is the output side, and then hands the typed values
/// to [`aggregate_range_extended`]. It does not calculate windows itself and
/// it does not update aggregation state itself.
///
/// The first tuple has one of these layouts:
///
/// ```text
/// 6 fields:
/// (left, left_index, right, right_index,
///  right_index_is_ordered, comparator)
///
/// 8 fields:
/// (left, left_index, right, right_index,
///  left_output_positions, right_output_positions,
///  right_index_is_ordered, comparator)
/// ```
///
/// The second tuple is always the five-field extended-anchor form. Predicates
/// after the first two are not used for binary search; they are residual
/// predicates evaluated against each candidate inside the intersected window.
///
/// Keeping this adapter beside the range aggregation implementation prevents
/// `anchor_non_equi_join_agg.rs` from owning dual-range tuple semantics. That
/// module is limited to one-anchor extended aggregation and all-`!=`
/// aggregation.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two aligned predicate tuples. The first two are
///   range anchors; later tuples are residual filters.
/// * `first` - The first tuple, already extracted from `predicates`. It must
///   contain six or eight fields, with the comparator in the final field.
/// * `aggregations` - Non-empty aggregation requests in the shared Rust
///   aggregation-input format.
/// * `return_matched` - Whether to include one boolean match flag per output
///   slot in the returned tuple.
/// * `reverse` - Whether to aggregate into right-oriented output slots using
///   left-side source values.
///
/// # Returns
///
/// Returns `(output_positions, matched, aggregation_arrays)` when
/// `return_matched` is true, or `(output_positions, aggregation_arrays)` when
/// it is false. Returns `None` when no complete match survives both range
/// anchors and all residual predicates.
///
/// # Errors
///
/// Returns a Python `ValueError` for too few predicates, malformed tuple
/// layouts, invalid comparators, non-range anchors, mismatched output maps,
/// mismatched residual lengths, or invalid aggregation requests.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_range_extended_aggregation<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    first: &Bound<'py, PyTuple>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // This entry point is specifically for a dual-range anchor. A list with
    // fewer than two predicates cannot provide the second window that must be
    // intersected with the first one.
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range extended aggregation requires at least two predicates",
        ));
    }

    // Keep the Python tuple separate from the typed `ParsedRangePredicate`
    // created below. The tuple is still needed to read optional output maps,
    // while the parsed value owns the borrowed NumPy array handles used by
    // the window builder.
    let first_tuple = first;

    // A six-field tuple is the normal range contract. The eight-field form
    // adds two complete maps from compact aggregation slots to original
    // physical rows. No other shape can tell this adapter where its
    // comparator or output layout is located.
    if !matches!(first_tuple.len(), 6 | 8) {
        return Err(PyValueError::new_err(
            "the first range aggregation predicate must contain 6 or 8 elements",
        ));
    }

    // In the six-field form the comparator is field five. In the eight-field
    // form fields five and six are output maps, so the comparator moves to
    // field seven.
    let comparator_position = if first_tuple.len() == 8 { 7 } else { 5 };

    // The ordering flag is part of the shared predicate tuple contract.
    // Aggregation does not use its value: PyJanitor has already sorted the
    // right-hand arrays before calling Rust. In the six-field form it is at
    // field four; in the eight-field form fields four and five are the output
    // maps, so the flag moves to field six. Extract it only to validate the
    // tuple shape and field type.
    let ordering_position = if first_tuple.len() == 8 { 6 } else { 4 };
    first_tuple.get_item(ordering_position)?.extract::<bool>()?;

    // Extract the four aligned value/label arrays and decode the comparator.
    // `ParsedRangePredicate` keeps the Python array owners alive while the
    // range windows and residual filters borrow their views.
    let first = ParsedRangePredicate {
        left: first_tuple
            .get_item(0)?
            .extract::<PyReadonlyArray1<'py, T>>()?,
        left_index: first_tuple
            .get_item(1)?
            .extract::<PyReadonlyArray1<'py, i64>>()?,
        right: first_tuple
            .get_item(2)?
            .extract::<PyReadonlyArray1<'py, T>>()?,
        right_index: first_tuple
            .get_item(3)?
            .extract::<PyReadonlyArray1<'py, i64>>()?,
        op: CompareOp::try_from_str(
            first_tuple
                .get_item(comparator_position)?
                .extract::<&str>()?,
        )?,
    };

    // The dual-range implementation only supports inequality range anchors.
    // Equality and `!=` use different upstream/kernel contracts and must not
    // enter this window-intersection path.
    if !first.op.is_range() {
        return Err(PyValueError::new_err(
            "range extended aggregation requires a range comparator first",
        ));
    }

    // Only the eight-field form carries output maps. Each map is ordered by
    // compact aggregation slot and stores the corresponding original
    // physical row. In this form the maps occupy fields four and five.
    // Forward aggregation writes one result per left slot, so it uses the
    // left map; reverse aggregation writes one result per right slot, so it
    // uses the right map.
    let (left_output_positions, right_output_positions) = if first_tuple.len() == 8 {
        (
            Some(
                first_tuple
                    .get_item(4)?
                    .extract::<PyReadonlyArray1<'py, i64>>()?,
            ),
            Some(
                first_tuple
                    .get_item(5)?
                    .extract::<PyReadonlyArray1<'py, i64>>()?,
            ),
        )
    } else {
        (None, None)
    };

    // The output map describes the trimmed layout, when present. Without a
    // map, the value arrays themselves already define an identity layout.
    // Keep both lengths because forward and reverse aggregation have
    // different output domains.
    let (left_output_len, right_output_len) = match (
        left_output_positions.as_ref(),
        right_output_positions.as_ref(),
    ) {
        (Some(left), Some(right)) => (left.len()?, right.len()?),
        _ => (first.left.len()?, first.right.len()?),
    };

    // Select the map and output length for the requested orientation. The
    // aggregation core receives only one output map because it writes into
    // one output domain per call. The other map remains available only for a
    // separate reverse call.
    aggregate_range_extended(
        py,
        predicates,
        first,
        aggregations,
        if reverse {
            right_output_positions
                .as_ref()
                .map(|values| values.as_array())
        } else {
            left_output_positions
                .as_ref()
                .map(|values| values.as_array())
        },
        if reverse {
            right_output_len
        } else {
            left_output_len
        },
        return_matched,
        reverse,
    )
}

/// Expand sparse surviving windows into dense aggregation boundaries.
///
/// `build_windows` omits left rows whose intersection is empty because that is
/// ideal for index generation. The optimized aggregation API instead expects
/// one range per source left row. Empty rows are therefore represented as
/// `[0, 0)`, which contributes nothing and remains unmatched.
///
/// # Arguments
///
/// * `windows` - Sparse windows produced by [`build_windows`].
/// * `left_len` - Number of rows in the complete left aggregation layout.
///
/// # Returns
///
/// Two int64 vectors with one boundary pair per left output slot. Their
/// positions match the supplied left value layout, including rows whose
/// window is empty.
///
/// # Errors
///
/// Returns an error if a positional `usize` boundary cannot be represented as
/// int64. The caller validates the sparse left positions while constructing
/// the windows.
fn dense_boundaries(
    windows: &crate::join_common::SingleJoinResult,
    left_len: usize,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let mut starts = vec![0_i64; left_len];
    let mut ends = vec![0_i64; left_len];
    for (row, &left_position) in windows.left_positions.iter().enumerate() {
        let start = i64::try_from(windows.starts[row]).map_err(|_| "range start exceeds int64")?;
        let end = i64::try_from(windows.ends[row]).map_err(|_| "range end exceeds int64")?;
        starts[left_position] = start;
        ends[left_position] = end;
    }
    Ok((starts, ends))
}

/// Run forward or reverse aggregation for exactly two range predicates.
///
/// The predicates use the ordinary six-element range contract. The local
/// aggregation slots already match the supplied left/right layouts, so this
/// basic path returns identity output positions rather than accepting separate
/// output maps. Aggregation consumes every pair in the intersection; there is
/// deliberately no `keep` parameter.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to build the result tuple.
/// * `predicates` - Exactly two aligned six-element range predicate tuples.
/// * `aggregations` - Non-empty aggregation requests. Every intersected pair
///   contributes to the selected aggregation state.
/// * `return_matched` - Include one boolean flag per output slot when true.
/// * `reverse` - Aggregate left values into right output slots when true;
///   otherwise aggregate right values into left output slots.
///
/// # Returns
///
/// Returns the standard aggregation tuple, with identity output positions, or
/// `None` when the two windows have no intersection.
///
/// # Errors
///
/// Returns a Python `ValueError` when the predicate count, tuple layout,
/// comparator, array lengths, or aggregation requests are invalid.
fn run_two_range<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_range_predicate::<T>(first_tuple)?;
    let second = parse_range_predicate::<T>(second_tuple)?;
    let left = first.left.as_array();
    let right = first.right.as_array();

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    let windows = build_windows(
        RangePredicate {
            left,
            left_index: first.left_index.as_array(),
            right,
            right_index: first.right_index.as_array(),
            op: first.op,
        },
        RangePredicate {
            left: second.left.as_array(),
            left_index: second.left_index.as_array(),
            right: second.right.as_array(),
            right_index: second.right_index.as_array(),
            op: second.op,
        },
    )
    .map_err(PyValueError::new_err)?;
    let (starts, ends) = dense_boundaries(&windows, left.len()).map_err(PyValueError::new_err)?;
    let starts = ArrayView1::from(&starts[..]);
    let ends = ArrayView1::from(&ends[..]);
    let output_len = if reverse { right.len() } else { left.len() };
    let source_len = if reverse { left.len() } else { right.len() };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    if reverse {
        set.aggregate_reverse_starts_ends(starts, ends);
    } else {
        set.aggregate_starts_ends(starts, ends);
    }
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results_with_positions(
        py,
        set,
        None,
        output_len,
        return_matched,
    )?))
}

macro_rules! range_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Aggregate over the intersection of two ascending range predicates.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Exactly two six-element tuples of the form
        ///   `(left, left_index, right, right_index,
        ///   right_index_is_ordered, comparator)`. PyJanitor supplies
        ///   null-free arrays and sorts both right value arrays before calling
        ///   Rust. The two predicates must use the same aligned layouts.
        /// * `aggregations` - Non-empty aggregation requests. Each request is
        ///   parsed by the shared aggregation input parser and is evaluated
        ///   for every pair in the intersected range window.
        /// * `return_matched` - Include the per-left-row matched mask in the
        ///   returned tuple when true.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair is present. Otherwise returns the
        /// standard aggregation result with identity output positions and
        /// either `(output_positions, matched, arrays)` or
        /// `(output_positions, arrays)` depending on `return_matched`.
        /// There is no `keep` parameter: aggregation consumes all pairs.
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            run_two_range::<$ty>(py, predicates, aggregations, return_matched, false)
        }

        /// Aggregate the same two range predicates in reverse orientation.
        ///
        /// The predicate and aggregation arguments have the same contract as
        /// the forward function, but source values come from the left layout
        /// and output slots are indexed by right rows. Reverse aggregation
        /// uses the optimized boundary-start/end kernels already provided by
        /// `AggregationSet`.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair is present; otherwise returns the
        /// output positions, optional matched mask, and aggregation arrays.
        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            run_two_range::<$ty>(py, predicates, aggregations, return_matched, true)
        }
    };
}

range_aggregation_functions!(
    range_join_aggregate_int64,
    range_join_aggregate_reverse_int64,
    i64
);
range_aggregation_functions!(
    range_join_aggregate_int32,
    range_join_aggregate_reverse_int32,
    i32
);
range_aggregation_functions!(
    range_join_aggregate_int16,
    range_join_aggregate_reverse_int16,
    i16
);
range_aggregation_functions!(
    range_join_aggregate_int8,
    range_join_aggregate_reverse_int8,
    i8
);
range_aggregation_functions!(
    range_join_aggregate_uint64,
    range_join_aggregate_reverse_uint64,
    u64
);
range_aggregation_functions!(
    range_join_aggregate_uint32,
    range_join_aggregate_reverse_uint32,
    u32
);
range_aggregation_functions!(
    range_join_aggregate_uint16,
    range_join_aggregate_reverse_uint16,
    u16
);
range_aggregation_functions!(
    range_join_aggregate_uint8,
    range_join_aggregate_reverse_uint8,
    u8
);
range_aggregation_functions!(
    range_join_aggregate_f64,
    range_join_aggregate_reverse_f64,
    f64
);
range_aggregation_functions!(
    range_join_aggregate_f32,
    range_join_aggregate_reverse_f32,
    f32
);

/// Export the two-range-plus-residual aggregation contract.
///
/// These wrappers live with the range aggregation implementation because the
/// first two predicates are always consumed as range windows. The general
/// extended aggregation module owns the separate one-anchor and all-`!=`
/// wrappers; it does not own this dual-range API.
macro_rules! range_extended_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Aggregate a range-led extended join without materializing pairs.
        ///
        /// The first two predicates must be ascending range anchors. The
        /// first tuple may use the six-element range form or the eight-element
        /// form carrying output-position maps; the second tuple uses the
        /// six-element form. Remaining predicates are residual filters and
        /// are evaluated in their supplied order. They do not need sorted
        /// right arrays because they are checked by aligned physical
        /// position.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - At least two aligned range-anchor/residual tuples.
        /// * `aggregations` - Non-empty aggregation requests over the source
        ///   layout selected by the direction of the call.
        /// * `return_matched` - Include one matched flag per output slot when
        ///   true.
        ///
        /// # Returns
        ///
        /// Returns `None` when no candidate survives both range anchors and
        /// all residual predicates. Otherwise returns output positions,
        /// optionally the matched mask, and one result array per aggregation.
        ///
        /// # Errors
        ///
        /// Returns a Python `ValueError` when fewer than two predicates are
        /// supplied or when the predicate, layout, or aggregation contract is
        /// invalid.
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            if predicates.len() < 2 {
                return Err(PyValueError::new_err(
                    "range extended aggregation requires at least two predicates",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch_range_extended_aggregation::<$ty>(
                py,
                predicates,
                &first,
                aggregations,
                return_matched,
                false,
            )
        }

        /// Aggregate the same range-led extended join in reverse orientation.
        ///
        /// The first two right arrays must already be ascending because
        /// PyJanitor performs sorting before calling Rust. Later predicates
        /// are residual filters evaluated by aligned physical position. The
        /// function aggregates left-side source values into right-oriented
        /// output slots and uses the optimized reverse boundary kernels only
        /// for the range-window portion.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - At least two aligned range-anchor/residual tuples.
        /// * `aggregations` - Non-empty aggregation requests over the left
        ///   source layout.
        /// * `return_matched` - Include one matched flag per right output slot
        ///   when true.
        ///
        /// # Returns
        ///
        /// Returns `None` when no complete pair survives. Otherwise returns
        /// output positions, optionally the matched mask, and aggregation
        /// arrays.
        ///
        /// # Errors
        ///
        /// Returns a Python `ValueError` when fewer than two predicates are
        /// supplied or when the predicate, layout, or aggregation contract is
        /// invalid.
        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            if predicates.len() < 2 {
                return Err(PyValueError::new_err(
                    "range extended aggregation requires at least two predicates",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch_range_extended_aggregation::<$ty>(
                py,
                predicates,
                &first,
                aggregations,
                return_matched,
                true,
            )
        }
    };
}

range_extended_aggregation_functions!(
    range_join_extended_aggregate_int64,
    range_join_extended_aggregate_reverse_int64,
    i64
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_int32,
    range_join_extended_aggregate_reverse_int32,
    i32
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_int16,
    range_join_extended_aggregate_reverse_int16,
    i16
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_int8,
    range_join_extended_aggregate_reverse_int8,
    i8
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_uint64,
    range_join_extended_aggregate_reverse_uint64,
    u64
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_uint32,
    range_join_extended_aggregate_reverse_uint32,
    u32
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_uint16,
    range_join_extended_aggregate_reverse_uint16,
    u16
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_uint8,
    range_join_extended_aggregate_reverse_uint8,
    u8
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_f64,
    range_join_extended_aggregate_reverse_f64,
    f64
);
range_extended_aggregation_functions!(
    range_join_extended_aggregate_f32,
    range_join_extended_aggregate_reverse_f32,
    f32
);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        range_join_aggregate_int64,
        range_join_aggregate_reverse_int64,
        range_join_aggregate_int32,
        range_join_aggregate_reverse_int32,
        range_join_aggregate_int16,
        range_join_aggregate_reverse_int16,
        range_join_aggregate_int8,
        range_join_aggregate_reverse_int8,
        range_join_aggregate_uint64,
        range_join_aggregate_reverse_uint64,
        range_join_aggregate_uint32,
        range_join_aggregate_reverse_uint32,
        range_join_aggregate_uint16,
        range_join_aggregate_reverse_uint16,
        range_join_aggregate_uint8,
        range_join_aggregate_reverse_uint8,
        range_join_aggregate_f64,
        range_join_aggregate_reverse_f64,
        range_join_aggregate_f32,
        range_join_aggregate_reverse_f32,
        range_join_extended_aggregate_int64,
        range_join_extended_aggregate_reverse_int64,
        range_join_extended_aggregate_int32,
        range_join_extended_aggregate_reverse_int32,
        range_join_extended_aggregate_int16,
        range_join_extended_aggregate_reverse_int16,
        range_join_extended_aggregate_int8,
        range_join_extended_aggregate_reverse_int8,
        range_join_extended_aggregate_uint64,
        range_join_extended_aggregate_reverse_uint64,
        range_join_extended_aggregate_uint32,
        range_join_extended_aggregate_reverse_uint32,
        range_join_extended_aggregate_uint16,
        range_join_extended_aggregate_reverse_uint16,
        range_join_extended_aggregate_uint8,
        range_join_extended_aggregate_reverse_uint8,
        range_join_extended_aggregate_f64,
        range_join_extended_aggregate_reverse_f64,
        range_join_extended_aggregate_f32,
        range_join_extended_aggregate_reverse_f32,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    #[test]
    fn forward_two_range_aggregation_uses_intersected_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left = PyArray1::from_vec(py, vec![4_i64]);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    left_index.clone().into_any(),
                    right.clone().into_any(),
                    right_index.clone().into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    left_index.into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    right_index.into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let mask = PyArray1::from_vec(py, vec![false, false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = range_join_aggregate_int64(py, &predicates, &aggregations, true)?
                .expect("the range intersection has matches");
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![40]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn empty_two_range_intersection_returns_none() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            for op in ["<", ">"] {
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                        PyArray1::from_vec(py, vec![10_i64, 11, 12]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        op.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            assert!(range_join_aggregate_int64(py, &predicates, &aggregations, true)?.is_none());
            Ok(())
        })
        .unwrap();
    }
}
