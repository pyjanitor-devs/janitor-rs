//! Fused reverse batch comparison and aggregation.

use crate::aggs::aggregation::{parse_inputs, AggregationSet};
use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core, ensure_unique_index};
use crate::compare::common::checked_bounds;
use crate::compare::predicate::{
    null_metadata_views, parse_predicates_with_nulls, predicates_match_dispatch,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Compare bounded candidate ranges and aggregate successful pairs into the
/// corresponding right-side output positions.
///
/// The aggregation arrays are aligned to the left predicate rows. A successful
/// comparison reads the source value at the left row and writes to the output
/// slot identified by the right candidate position.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to borrow inputs and create
///   NumPy result arrays.
/// * `predicates` - Non-empty list of comparison tuples. All left arrays must
///   have the same length, and all right arrays must have the same length.
/// * `starts` - Optional inclusive right-position start bound for each left
///   row. Omitted bounds start at zero.
/// * `ends` - Optional exclusive right-position end bound for each left row.
///   Omitted bounds end at the right-array length.
/// * `right_index` - Unique right labels in output-position order. Labels may
///   be unordered; slot `n` corresponds to `right_index[n]`.
/// * `aggregations` - Non-empty `(array, null_mask, operation)` tuples. Value
///   arrays and masks must be aligned to the left predicate rows.
///
/// # Returns
///
/// `Some(list)` of arrays, one per requested aggregation, each with
/// `right_index.len()` entries. Returns `None` when no comparison succeeds.
///
/// # Errors
///
/// Returns a Python exception for empty or mismatched inputs, duplicate right
/// labels, invalid bounds, unsupported dtypes/operations, or invalid masks.
#[pyfunction]
pub fn aggregate_batch_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err("at least one comparison is required"));
    }
    let left_len = predicates[0].left_len();
    let right_len = predicates[0].right_len();
    ensure_nonempty_core("left predicate array", left_len).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right predicate array", right_len).map_err(PyValueError::new_err)?;
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
    if let Some(values) = &starts {
        ensure_equal_lengths("starts", values.len()?, "left predicate array", left_len)?;
    }
    if let Some(values) = &ends {
        ensure_equal_lengths("ends", values.len()?, "left predicate array", left_len)?;
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
    let starts = starts.as_ref().map(|values| values.as_array());
    let ends = ends.as_ref().map(|values| values.as_array());
    let mut set = AggregationSet::new(right_len, left_len, &inputs)?;
    for row in 0..left_len {
        let start = starts.map_or(0, |values| values[row]);
        let end = ends.map_or(right_len as i64, |values| values[row]);
        let Some((start, end)) = checked_bounds(start, end, right_len) else {
            continue;
        };
        for candidate in start..end {
            if predicates_match_dispatch(&views, metadata_views.as_deref(), row, candidate) {
                set.update(row, candidate);
            }
        }
    }
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(PyList::new(py, set.into_results(py))?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_batch_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn writes_bounded_matches_to_unordered_right_slots() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![0_i64, 0]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            // Opcode 2 is `<`; both left rows match every right candidate.
            let predicate = PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.into_any(),
                    2_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let predicates = PyList::new(py, [predicate]).unwrap();
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![3_i64, 3]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 10, 30]);
            let values = PyArray1::from_vec(py, vec![5_i64, 7]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
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
            let aggregations = PyList::new(py, [sum, count]).unwrap();

            let result = aggregate_batch_reverse(
                py,
                &predicates,
                Some(starts.readonly()),
                Some(ends.readonly()),
                right_index.readonly(),
                &aggregations,
            )
            .unwrap()
            .unwrap();

            // The first row contributes 5 to slots 0, 1, and 2. The second
            // row contributes 7 only to slots 1 and 2 because its range is
            // [1, 3). Output order follows right_index, not label sorting.
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![5, 12, 12]
            );
            assert_eq!(
                result.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 2, 2]
            );
        });
    }

    #[test]
    fn float_sum_preserves_positive_and_negative_infinity() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![0_i64, 0]);
            let right = PyArray1::from_vec(py, vec![0_i64]);
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
            let right_index = PyArray1::from_vec(py, vec![99_i64]);
            let mask = PyArray1::from_vec(py, vec![false, false]);

            // Each aggregation receives an infinity first and then a finite
            // value. The mask is entirely false: this test exercises IEEE
            // floating-point state, not null-mask handling.
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

            let result = aggregate_batch_reverse(
                py,
                &predicates,
                None,
                None,
                right_index.readonly(),
                &aggregations,
            )
            .unwrap()
            .unwrap();

            let positive_result = result.get_item(0).unwrap().extract::<Vec<f64>>().unwrap();
            let negative_result = result.get_item(1).unwrap().extract::<Vec<f64>>().unwrap();
            assert!(positive_result[0].is_infinite() && positive_result[0].is_sign_positive());
            assert!(negative_result[0].is_infinite() && negative_result[0].is_sign_negative());
        });
    }
}
