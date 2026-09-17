//! Fused forward batch comparison and aggregation.
//!
//! This module deliberately does not call the index-producing batch kernels.
//! It owns the candidate loop and updates all requested aggregations at the
//! moment a candidate passes every predicate.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::common::checked_bounds;
use super::predicate::{
    null_metadata_views, parse_predicates_with_nulls, predicates_match_dispatch,
};
use crate::aggs::aggregation::{parse_inputs, AggregationSet};
use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core};

/// Compare heterogeneous predicates over optional per-row bounds and update
/// all requested forward aggregations in the same traversal.
///
/// `aggregations` contains `(array, null_mask, operation)` tuples. The arrays
/// are indexed by candidate position and must therefore all have the same
/// length. Results have one slot per left predicate row. A completely empty
/// match result is represented by `None`.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to borrow arrays and create
///   result objects.
/// * `predicates` - Non-empty list of comparison tuples. Each tuple contains
///   a left array, right array, and comparison operation. All tuples must have
///   matching left and right lengths.
/// * `starts` - Optional inclusive right-side start bound for each left row.
///   When omitted, every row starts at zero.
/// * `ends` - Optional exclusive right-side end bound for each left row. When
///   omitted, every row ends at the right-array length.
/// * `aggregations` - Non-empty list of `(array, null_mask, operation)` tuples.
///   Every value array and mask must have the same length as the predicate
///   right arrays. The value arrays must be null-free. The mask is the sole
///   null-tracking mechanism: `true` marks a null to skip for value-based
///   operations, while `false` asserts that the corresponding value is valid.
///   Callers, principally pyjanitor, are responsible for supplying the aligned
///   array and correct mask; this function does not infer nulls from values.
///
/// # Returns
///
/// `Some(list)` containing one output NumPy array per requested aggregation,
/// in input order. Every output has one slot per left row. Returns `None` when
/// no candidate passes the comparisons.
///
/// # Errors
///
/// Returns a Python exception for empty inputs, unsupported dtypes/operations,
/// mismatched lengths, or invalid bounds.
#[pyfunction]
pub fn compare_batch_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err("at least one comparison is required"));
    }
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let left_len = predicates[0].left_len();
    let right_len = predicates[0].right_len();
    ensure_nonempty_core("left predicate array", left_len).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right predicate array", right_len).map_err(PyValueError::new_err)?;
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
    if let Some(values) = &starts {
        ensure_equal_lengths("starts", values.len()?, "left predicate array", left_len)?;
    }
    if let Some(values) = &ends {
        ensure_equal_lengths("ends", values.len()?, "left predicate array", left_len)?;
    }

    let views: Vec<_> = predicates
        .iter()
        .map(|predicate| predicate.view())
        .collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let starts = starts.as_ref().map(|values| values.as_array());
    let ends = ends.as_ref().map(|values| values.as_array());
    let mut set = AggregationSet::new(left_len, right_len, &inputs)?;
    for row in 0..left_len {
        let start = starts.map_or(0, |values| values[row]);
        let end = ends.map_or(right_len as i64, |values| values[row]);
        let Some((start, end)) = checked_bounds(start, end, right_len) else {
            continue;
        };
        for candidate in start..end {
            if predicates_match_dispatch(&views, metadata_views.as_deref(), row, candidate) {
                set.update(candidate, row);
            }
        }
    }
    if set.is_empty() {
        return Ok(None);
    }
    let results = set.into_results(py);
    Ok(Some(PyList::new(py, results)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_batch_aggregate, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::types::PyTuple;

    #[test]
    fn updates_multiple_forward_aggregations_in_one_pass() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let predicate = PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    3_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let predicates = PyList::new(py, [predicate]).unwrap();
            let aggregation = |name: &str| {
                PyTuple::new(
                    py,
                    [
                        values.clone().into_any(),
                        mask.clone().into_any(),
                        name.into_pyobject(py).unwrap().into_any(),
                    ],
                )
                .unwrap()
            };
            let aggregations = PyList::new(
                py,
                [
                    aggregation("sum"),
                    aggregation("count"),
                    aggregation("prod"),
                    aggregation("min"),
                    aggregation("max"),
                ],
            )
            .unwrap();
            let result = compare_batch_aggregate(py, &predicates, None, None, &aggregations)
                .unwrap()
                .unwrap();
            assert_eq!(result.len(), 5);
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![50, 30]
            );
            assert_eq!(
                result.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![2, 1]
            );
            assert_eq!(
                result.get_item(2).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![600, 30]
            );
            assert_eq!(
                result.get_item(3).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 2]
            );
            assert_eq!(
                result.get_item(4).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![2, 2]
            );
        });
    }

    #[test]
    fn returns_none_when_no_candidate_succeeds() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![5_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2]);
            let values = PyArray1::from_vec(py, vec![10_i64, 20]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
            let predicate = PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    3_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let predicates = PyList::new(py, [predicate]).unwrap();
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [aggregation]).unwrap();
            assert!(
                compare_batch_aggregate(py, &predicates, None, None, &aggregations)
                    .unwrap()
                    .is_none()
            );
        });
    }

    #[test]
    fn fused_float_sum_uses_the_existing_kahan_result_contract() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![0_i64]);
            let right = PyArray1::from_vec(py, vec![0_i64, 0, 0]);
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
            let values = PyArray1::from_vec(py, vec![1.0e16_f64, 1.0, -1.0e16]);
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [aggregation]).unwrap();
            let result = compare_batch_aggregate(py, &predicates, None, None, &aggregations)
                .unwrap()
                .unwrap();
            let values = result.get_item(0).unwrap().extract::<Vec<f64>>().unwrap();
            assert_eq!(values, vec![0.0]);
        });
    }

    #[test]
    fn fused_float_sum_preserves_positive_and_negative_infinity() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![0_i64]);
            let right = PyArray1::from_vec(py, vec![0_i64, 0]);
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
            let mask = PyArray1::from_vec(py, vec![false, false]);

            // Both aggregations receive an infinity first and then a finite
            // value. The mask is entirely false: this checks floating-point
            // compensation state, not null inference.
            let positive = PyArray1::from_vec(py, vec![f64::INFINITY, 1.0]);
            let negative = PyArray1::from_vec(py, vec![f64::NEG_INFINITY, 1.0]);
            let positive_sum = PyTuple::new(
                py,
                [
                    positive.into_any(),
                    mask.clone().into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let negative_sum = PyTuple::new(
                py,
                [
                    negative.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [positive_sum, negative_sum]).unwrap();

            let result = compare_batch_aggregate(py, &predicates, None, None, &aggregations)
                .unwrap()
                .unwrap();
            let positive_result = result.get_item(0).unwrap().extract::<Vec<f64>>().unwrap();
            let negative_result = result.get_item(1).unwrap().extract::<Vec<f64>>().unwrap();
            assert!(positive_result[0].is_infinite() && positive_result[0].is_sign_positive());
            assert!(negative_result[0].is_infinite() && negative_result[0].is_sign_negative());
        });
    }
}
