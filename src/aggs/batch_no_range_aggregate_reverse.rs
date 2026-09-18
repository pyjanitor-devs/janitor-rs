//! Fused reverse aggregation for aligned no-range candidates.

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{checked_index, ensure_equal_lengths, ensure_nonempty_core, ensure_unique_index};
use crate::multi_join_indices::predicate::{
    null_metadata_views, parse_predicates_with_nulls, predicates_match_dispatch,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// Compare one aligned candidate per left row and aggregate into right slots.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to create NumPy results.
/// * `predicates` - Non-empty comparison tuple list with aligned left/right
///   arrays.
/// * `positions` - One right candidate position for each left row. Invalid or
///   negative positions are treated as non-matches.
/// * `right_index` - Unique right labels. Its positions define output slots,
///   even when the labels are not sorted.
/// * `aggregations` - Non-empty `(array, null_mask, operation)` tuples aligned
///   to the left rows. The mask is authoritative: `true` means the
///   corresponding value is null and is skipped by `sum`, `product`, `min`,
///   and `max`; `false` means the value is valid. Nullness is never inferred
///   from the value array, including from `NaN`. A three-element `count`
///   request counts only non-null source values; `size` and the two-element
///   `("*", "count")` shorthand count every successful comparison.
///
/// # Returns
///
/// `Some((matched, list))`, where `matched` is a boolean array aligned to the
/// right positions and `list` contains right-aligned arrays in request order.
/// Returns `None` when no candidate succeeds anywhere.
///
/// # Errors
///
/// Returns a Python exception for empty or mismatched inputs, duplicate labels,
/// unsupported inputs, or invalid masks.
#[pyfunction]
pub fn aggregate_batch_no_range_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    positions: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err("at least one comparison is required"));
    }
    let left_len = predicates[0].left_len();
    let right_len = predicates[0].right_len();
    ensure_nonempty_core("left predicate array", left_len).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right predicate array", right_len).map_err(PyValueError::new_err)?;
    ensure_equal_lengths(
        "positions",
        positions.len()?,
        "left predicate array",
        left_len,
    )?;
    ensure_equal_lengths(
        "right index",
        right_index.len()?,
        "right predicate array",
        right_len,
    )?;
    ensure_unique_index("right index", right_index.as_array())?;
    for predicate in predicates.iter().skip(1) {
        ensure_equal_lengths(
            "first left predicate array",
            left_len,
            "current left predicate array",
            predicate.left_len(),
        )?;
        ensure_equal_lengths(
            "first right predicate array",
            right_len,
            "current right predicate array",
            predicate.right_len(),
        )?;
    }
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let views: Vec<_> = predicates
        .iter()
        .map(|predicate| predicate.view())
        .collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let mut set = AggregationSet::new(right_len, left_len, &inputs)?;
    for (row, position) in positions.as_array().iter().enumerate() {
        let Some(candidate) = checked_index(*position, right_len) else {
            continue;
        };
        if predicates_match_dispatch(&views, metadata_views.as_deref(), row, candidate) {
            set.update(row, candidate);
        }
    }
    // Invalid positions and failed predicates do not set the shared success
    // flag. Once any aligned candidate succeeds, however, return the full
    // right-aligned tuple so `matched` can identify which slots were touched.
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_batch_no_range_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn writes_multiple_aggregations_to_unordered_right_slots() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let right = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let positions = PyArray1::from_vec(py, vec![0_i64, 1, -1]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 10, 30]);
            let predicate = PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.into_any(),
                    4_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let predicates = PyList::new(py, [predicate]).unwrap();
            let values = PyArray1::from_vec(py, vec![5_i64, 7, 11]);
            let mask = PyArray1::from_vec(py, vec![false, true, false]);
            let sum = PyTuple::new(
                py,
                [
                    values.clone().into_any(),
                    mask.clone().into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let count = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            // These two requests deliberately exercise the shared
            // count-all accumulator. They should produce identical arrays,
            // while the column-based count below must honor the null mask.
            let count_all = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let size_all = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    "size".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [sum, count, count_all, size_all]).unwrap();
            let result = aggregate_batch_no_range_reverse(
                py,
                &predicates,
                positions.readonly(),
                right_index.readonly(),
                &aggregations,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, true, false]
            );
            let result_list = result.get_item(1).unwrap();
            let result = result_list.cast::<PyList>().unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![5, 0, 0]
            );
            assert_eq!(
                result.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 0, 0]
            );
            assert_eq!(
                result.get_item(2).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 1, 0]
            );
            assert_eq!(
                result.get_item(3).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 1, 0]
            );
        });
    }
}
