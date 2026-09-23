//! Fused aggregation for multiple conditional-join predicates.
//!
//! The first predicate supplies either a range window or the complete
//! null-aware `!=` candidate stream. Later predicates filter candidates before
//! `AggregationSet` is updated. No intermediate index pairs are built.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::aggs::ensure_equal_lengths_core;
use crate::op::CompareOp;
use crate::predicate::{
    null_metadata_views, parse_predicates_with_nulls_strings, predicates_match_dispatch,
    PredicateView,
};
use crate::single_join::{range_bounds, visit_not_equal_pairs_core};

/// Visit a range-led extended join and update aggregations for survivors.
///
/// The first predicate has already been separated from `residuals`. For each
/// left value, [`range_bounds`] supplies the contiguous right-side window
/// that satisfies that first predicate. Every candidate in that window is
/// then checked against the remaining predicates with
/// [`predicates_match_dispatch`]. Aggregation state is updated only after all
/// residual predicates pass.
///
/// This function intentionally does not use prefix/suffix aggregation tables.
/// A residual predicate may reject arbitrary candidates inside the first
/// predicate's window, so the surviving positions are no longer guaranteed to
/// form a complete prefix or suffix.
///
/// # Arguments
///
/// * `left` - Non-null left values for the first range predicate, in physical
///   left-row order.
/// * `right` - Sorted, non-null right values for the first range predicate.
/// * `op` - The first predicate's range operator: `<`, `<=`, `>`, or `>=`.
/// * `residuals` - Parsed predicates after the first predicate. Their arrays
///   must be aligned to the same physical left and right positions.
/// * `residual_metadata` - Optional authoritative null metadata for residual
///   `!=` predicates.
/// * `set` - Aggregation state updated for every fully matching pair.
/// * `reverse` - Whether source values come from the left and output slots
///   are indexed by the right side.
///
/// # Example
///
/// With `right = [2, 5, 8]`, `left = [4]`, and `left < right`, the first
/// predicate produces the candidate window `right[1..] = [5, 8]`. A residual
/// predicate can remove either candidate before `set.update` is called.
fn aggregate_range<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    op: CompareOp,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
) {
    for (left_position, &left_value) in left.iter().enumerate() {
        let (start, end) = range_bounds(left_value, right, op);
        for right_position in start..end {
            if predicates_match_dispatch(
                residuals,
                residual_metadata,
                left_position,
                right_position,
            ) {
                if reverse {
                    set.update(left_position, right_position);
                } else {
                    set.update(right_position, left_position);
                }
            }
        }
    }
}

/// Visit null-aware `!=` candidates for an extended join.
///
/// [`visit_not_equal_pairs_core`] generates the strict less-than and
/// greater-than candidates from the first `!=` predicate and adds null
/// candidates according to the NumPy or pandas extension-array contract.
/// Each candidate is passed through the residual predicates before it updates
/// the aggregation state.
///
/// The filtered value arrays contain only non-null values. `left_positions`
/// and `right_positions` map those values to the complete physical domains;
/// the optional null-position arrays complete those domains when nulls exist.
/// Consequently, `left_full_len` and `right_full_len` describe the original
/// physical layouts, not the filtered value-array lengths.
///
/// # Arguments
///
/// * `left` / `right` - Filtered, non-null first-predicate values.
/// * `left_full_len` / `right_full_len` - Full physical lengths used for
///   aggregation output and residual indexing.
/// * `left_positions` / `right_positions` - Physical positions of the
///   filtered values in the original layouts.
/// * `left_null_positions` / `right_null_positions` - Optional physical
///   positions of null rows.
/// * `is_extension_array` - Selects pandas nullable versus NumPy null
///   comparison behavior.
/// * `residuals` - Parsed predicates after the first `!=` predicate.
/// * `residual_metadata` - Optional null metadata for residual predicates.
/// * `set` - Aggregation state updated for each surviving pair.
/// * `reverse` - Selects forward or reverse source/output slot mapping.
///
/// # Errors
///
/// Returns a string error when the physical position partitions are malformed.
#[allow(clippy::too_many_arguments)]
fn aggregate_not_equal<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_full_len: usize,
    left_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right: ArrayView1<'_, T>,
    right_full_len: usize,
    right_positions: ArrayView1<'_, i64>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
) -> Result<(), String> {
    visit_not_equal_pairs_core(
        left,
        left_full_len,
        left_positions,
        right,
        right_full_len,
        right_positions,
        left_null_positions,
        right_null_positions,
        is_extension_array,
        |left_position, right_position| {
            if predicates_match_dispatch(
                residuals,
                residual_metadata,
                left_position,
                right_position,
            ) {
                if reverse {
                    set.update(left_position, right_position);
                } else {
                    set.update(right_position, left_position);
                }
            }
        },
    )
}

/// Parse every predicate after the first extended-join predicate.
///
/// The first predicate is consumed by the range or `!=` candidate generator.
/// This helper copies the remaining Python tuple references into a temporary
/// list because the shared predicate parser expects a list containing exactly
/// the predicates it should evaluate. The underlying NumPy arrays are still
/// borrowed; their values are not copied.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to create the temporary list.
/// * `predicates` - Full extended predicate list, including the first anchor
///   predicate at position zero.
///
/// # Returns
///
/// Parsed residual predicates and optional null metadata in their original
/// order.
fn residuals<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
) -> PyResult<(
    Vec<crate::predicate::Predicate<'py>>,
    Option<Vec<crate::predicate::NullMetadata<'py>>>,
)> {
    let values = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        values.append(item)?;
    }
    parse_predicates_with_nulls_strings(py, &values)
}

/// Validate that residual arrays share the anchor predicate's physical shape.
///
/// Candidate positions generated by the first predicate are reused directly
/// for every residual predicate. A mismatched residual array would therefore
/// make a physical position refer to a different row or fall outside the
/// array. This check happens once before the hot candidate loop.
///
/// # Arguments
///
/// * `predicates` - Parsed residual predicates to validate.
/// * `left_len` / `right_len` - Full physical lengths established by the
///   first predicate.
fn check_residual_lengths(
    predicates: &[crate::predicate::Predicate<'_>],
    left_len: usize,
    right_len: usize,
) -> PyResult<()> {
    for predicate in predicates {
        ensure_equal_lengths_core(
            "first left predicate array",
            left_len,
            "residual left predicate array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core(
            "first right predicate array",
            right_len,
            "residual right predicate array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    Ok(())
}

/// Run fused aggregation for a range-led extended join.
///
/// `predicates` contains the first range tuple followed by residual tuples.
/// The first tuple supplies the binary-searchable candidate window; residual
/// tuples are evaluated for every candidate in that window. The result has
/// one matched slot per left row in forward mode or per right row in reverse
/// mode, and one aggregation array per request.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - First range predicate plus aligned residual predicates.
/// * `left` / `right` - First-predicate value arrays. The right array must
///   already be sorted by PyJanitor.
/// * `op` - First predicate's range comparator.
/// * `aggregations` - Parsed at the shared boundary as value/mask/operation
///   requests. Arrays must use the complete source-side physical layout.
/// * `reverse` - Aggregate left values into right output slots when true.
///
/// # Returns
///
/// Returns `None` when no candidate survives all predicates. Otherwise
/// returns the common `(matched, aggregation_arrays)` tuple.
#[allow(clippy::too_many_arguments)]
fn run_range<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left: PyReadonlyArray1<'py, T>,
    right: PyReadonlyArray1<'py, T>,
    op: CompareOp,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (parsed, metadata) = residuals(py, predicates)?;
    let left = left.as_array();
    let right = right.as_array();
    check_residual_lengths(&parsed, left.len(), right.len())?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut set = AggregationSet::new(
        if reverse { right.len() } else { left.len() },
        if reverse { left.len() } else { right.len() },
        &inputs,
    )?;
    let views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_range(
        left,
        right,
        op,
        &views,
        metadata_views.as_deref(),
        &mut set,
        reverse,
    );
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

/// Run fused aggregation for an all-`!=` extended join.
///
/// The first tuple uses the eleven-element null-aware position contract. Its
/// filtered value arrays and physical position partitions generate candidate
/// pairs. Every later tuple is a residual predicate over the full physical
/// layouts. No pair tape is materialized; successful candidates update the
/// aggregation state immediately.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - First null-aware `!=` tuple followed by residual tuples.
/// * `left_index` / `right_index` - Complete physical index-label arrays;
///   their lengths define the output domains for forward and reverse modes.
/// * `left_positions` / `right_positions` - Physical maps for filtered
///   non-null first-predicate values.
/// * `left_null_positions` / `right_null_positions` - Optional physical null
///   partitions.
/// * `is_extension_array` - Selects pandas extension-array null semantics.
/// * `aggregations` - Full-layout aggregation requests.
/// * `reverse` - Selects right-oriented output and left-side source values.
///
/// # Returns
///
/// Returns `None` when no pair survives every predicate; otherwise returns the
/// standard matched-mask and aggregation-array tuple.
#[allow(clippy::too_many_arguments)]
fn run_not_equal<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left: PyReadonlyArray1<'py, T>,
    left_index: PyReadonlyArray1<'py, i64>,
    left_positions: PyReadonlyArray1<'py, i64>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    right_positions: PyReadonlyArray1<'py, i64>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (parsed, metadata) = residuals(py, predicates)?;
    check_residual_lengths(&parsed, left_index.len()?, right_index.len()?)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let left_full_len = left_index.len()?;
    let right_full_len = right_index.len()?;
    let mut set = AggregationSet::new(
        if reverse {
            right_full_len
        } else {
            left_full_len
        },
        if reverse {
            left_full_len
        } else {
            right_full_len
        },
        &inputs,
    )?;
    let views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_not_equal(
        left.as_array(),
        left_full_len,
        left_positions.as_array(),
        left_null_positions.as_ref().map(|values| values.as_array()),
        right.as_array(),
        right_full_len,
        right_positions.as_array(),
        right_null_positions
            .as_ref()
            .map(|values| values.as_array()),
        is_extension_array,
        &views,
        metadata_views.as_deref(),
        &mut set,
        reverse,
    )
    .map_err(PyValueError::new_err)?;
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

/// Validate the first extended predicate and dispatch to its fused traversal.
///
/// The first tuple is the algorithm anchor. A six-element tuple is a range
/// anchor; an eleven-element tuple is the null-aware all-`!=` anchor. The
/// tuple shape is deliberately checked here so malformed Python input fails
/// before any aggregation state or candidate loop is created.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Complete extended predicate list. It must contain at
///   least the anchor and one residual predicate.
/// * `first` - The first predicate tuple, already extracted from `predicates`.
/// * `aggregations` - Non-empty aggregation requests.
/// * `reverse` - Selects forward or reverse output mapping.
///
/// # Returns
///
/// Returns the common `(matched, aggregation_arrays)` tuple, or `None` when no
/// candidate survives.
#[allow(clippy::too_many_arguments)]
fn dispatch<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    first: &Bound<'py, PyTuple>,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended aggregation requires at least two predicates",
        ));
    }
    let op_position = match first.len() {
        6 => 5,
        11 => 10,
        _ => {
            return Err(PyValueError::new_err(
                "the first extended predicate must contain 6 or 11 elements",
            ))
        }
    };
    let op = CompareOp::try_from_str(first.get_item(op_position)?.extract::<&str>()?)?;
    if op != CompareOp::Ne {
        if first.len() != 6 {
            return Err(PyValueError::new_err(
                "the first range predicate must contain 6 elements",
            ));
        }
        if !matches!(
            op,
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
        ) {
            return Err(PyValueError::new_err(
                "single extended aggregation requires a range predicate first",
            ));
        }
        first.get_item(4)?.extract::<bool>()?;
        return run_range(
            py,
            predicates,
            first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
            first.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?,
            op,
            aggregations,
            reverse,
        );
    }
    if first.len() != 11 {
        return Err(PyValueError::new_err(
            "the first != predicate must contain 11 elements",
        ));
    }
    let left_null_positions = if first.get_item(3)?.is_none() {
        None
    } else {
        Some(first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    let right_null_positions = if first.get_item(7)?.is_none() {
        None
    } else {
        Some(first.get_item(7)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    run_not_equal(
        py,
        predicates,
        first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
        first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        first.get_item(2)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        left_null_positions,
        first.get_item(4)?.extract::<PyReadonlyArray1<'py, T>>()?,
        first.get_item(5)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        first.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        right_null_positions,
        first.get_item(9)?.extract::<bool>()?,
        aggregations,
        reverse,
    )
}

macro_rules! extended_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Fused forward aggregation for multiple conditional-join
        /// predicates.
        ///
        /// The first predicate must be either a range comparator (`<`, `<=`,
        /// `>`, `>=`) or the null-aware eleven-element `!=` form. Remaining
        /// predicates are aligned residual filters. Aggregation is performed
        /// as candidates pass the first predicate and all residual filters;
        /// no intermediate pair indices are returned or allocated.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Python list of aligned predicate tuples. The
        ///   first tuple is the range or `!=` anchor.
        /// * `aggregations` - Non-empty `(values, null_mask, operation)`
        ///   requests, or wildcard count requests. Values use the complete
        ///   right-side physical layout in forward mode.
        ///
        /// # Example
        ///
        /// Conceptually, for `left < right` followed by `left2 != right2`,
        /// Rust first finds the sorted-right range and then updates the
        /// aggregation only for candidates passing `left2 != right2`.
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch::<$ty>(py, predicates, &first, aggregations, false)
        }

        /// Fused reverse aggregation for multiple conditional-join
        /// predicates.
        ///
        /// This has the same predicate contract as the forward entry point,
        /// but aggregation values come from the left physical layout and
        /// output slots are indexed by right rows. Reverse range anchors use
        /// the optimized dense reverse prefix/suffix aggregation machinery
        /// only for single joins; residual-filtered extended candidates are
        /// updated individually.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Python list of aligned anchor and residual tuples.
        /// * `aggregations` - Non-empty aggregation requests over the complete
        ///   left-side physical layout.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair survives, otherwise a matched mask and
        /// one result array per aggregation request.
        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch::<$ty>(py, predicates, &first, aggregations, true)
        }
    };
}

extended_aggregation_functions!(
    single_join_extended_aggregate_int64,
    single_join_extended_aggregate_reverse_int64,
    i64
);
extended_aggregation_functions!(
    single_join_extended_aggregate_int32,
    single_join_extended_aggregate_reverse_int32,
    i32
);
extended_aggregation_functions!(
    single_join_extended_aggregate_int16,
    single_join_extended_aggregate_reverse_int16,
    i16
);
extended_aggregation_functions!(
    single_join_extended_aggregate_int8,
    single_join_extended_aggregate_reverse_int8,
    i8
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint64,
    single_join_extended_aggregate_reverse_uint64,
    u64
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint32,
    single_join_extended_aggregate_reverse_uint32,
    u32
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint16,
    single_join_extended_aggregate_reverse_uint16,
    u16
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint8,
    single_join_extended_aggregate_reverse_uint8,
    u8
);
extended_aggregation_functions!(
    single_join_extended_aggregate_f64,
    single_join_extended_aggregate_reverse_f64,
    f64
);
extended_aggregation_functions!(
    single_join_extended_aggregate_f32,
    single_join_extended_aggregate_reverse_f32,
    f32
);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        single_join_extended_aggregate_int64,
        single_join_extended_aggregate_reverse_int64,
        single_join_extended_aggregate_int32,
        single_join_extended_aggregate_reverse_int32,
        single_join_extended_aggregate_int16,
        single_join_extended_aggregate_reverse_int16,
        single_join_extended_aggregate_int8,
        single_join_extended_aggregate_reverse_int8,
        single_join_extended_aggregate_uint64,
        single_join_extended_aggregate_reverse_uint64,
        single_join_extended_aggregate_uint32,
        single_join_extended_aggregate_reverse_uint32,
        single_join_extended_aggregate_uint16,
        single_join_extended_aggregate_reverse_uint16,
        single_join_extended_aggregate_uint8,
        single_join_extended_aggregate_reverse_uint8,
        single_join_extended_aggregate_f64,
        single_join_extended_aggregate_reverse_f64,
        single_join_extended_aggregate_f32,
        single_join_extended_aggregate_reverse_f32,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    #[test]
    fn range_residuals_filter_before_forward_aggregation() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![3_i64, 7, 9, 7]).into_any(),
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
            let result = single_join_extended_aggregate_int64(py, &predicates, &aggregations)?
                .expect("the filtered range has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_value = result.get_item(1)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn all_not_equal_residuals_aggregate_without_materializing_pairs() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64, 21]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3]).into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![100_i64, 200]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = single_join_extended_aggregate_int64(py, &predicates, &aggregations)?
                .expect("the not-equal join has matches");
            assert_eq!(
                result.get_item(0)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(1)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![200, 300]);
            Ok(())
        })
        .unwrap();
    }
}
