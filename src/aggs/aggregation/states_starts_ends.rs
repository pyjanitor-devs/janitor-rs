//! Forward aggregation for explicit starts/ends ranges.
//!
//! The comparison phase supplies half-open ranges. This module only walks
//! those ranges and delegates every operation to the shared aggregation state.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::{make_results, parse_inputs, AggregationSet};
use crate::aggs::ensure_equal_lengths;
use crate::aggs::ensure_nonempty_core;

/// Aggregate every valid right-side half-open range.
///
/// A valid row aggregates `right[start[row]..end[row]]`. Empty, inverted,
/// negative, oversized, and sentinel ranges contribute no matches. The
/// output arrays and `matched` indicator remain aligned to `starts.len()`.
/// Value-backed requests use an authoritative boolean mask: `true` marks a
/// null and is skipped by value-based operations, while `false` marks a valid
/// value. Nullness is never inferred from the value array, including NaN; the
/// caller is responsible for supplying null-free values and correct metadata.
///
#[pyfunction]
pub fn aggregate_starts_ends<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    right_len: usize,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_nonempty_core("starts", starts.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right side", right_len).map_err(PyValueError::new_err)?;
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut state = AggregationSet::new(starts.len(), right_len, &inputs, true)?;
    state.aggregate_starts_ends(starts, ends);
    if state.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, state)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_starts_ends, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    fn direct_sum(values: &[i64], start: usize, end: usize) -> i64 {
        let mut total = 0_i64;
        for value in &values[start..end] {
            total = total.wrapping_add(*value);
        }
        total
    }

    fn direct_product(values: &[i64], start: usize, end: usize) -> i64 {
        let mut total = 1_i64;
        for value in &values[start..end] {
            total = total.wrapping_mul(*value);
        }
        total
    }

    fn direct_min_position(values: &[i64], start: usize, end: usize) -> i64 {
        let mut winner = -1_i64;
        for position in start..end {
            if winner < 0 || values[position] < values[winner as usize] {
                winner = position as i64;
            }
        }
        winner
    }

    fn direct_max_position(values: &[i64], start: usize, end: usize) -> i64 {
        let mut winner = -1_i64;
        for position in start..end {
            if winner < 0 || values[position] > values[winner as usize] {
                winner = position as i64;
            }
        }
        winner
    }

    #[test]
    fn aggregates_half_open_ranges_and_counts_only_valid_values() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3, 2]);
            let values = PyArray1::from_vec(py, vec![2_i64, 3, 5, 7]);
            let mask = PyArray1::from_vec(py, vec![false, true, false, false]);
            let count = PyTuple::new(
                py,
                [
                    values.clone().into_any(),
                    mask.clone().into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let product = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "product".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [count, product]).unwrap();
            let result =
                aggregate_starts_ends(py, starts.readonly(), ends.readonly(), 4, &aggregations)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, true, false]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 1, 0]
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![2, 5, 1]
            );
        });
    }

    #[test]
    fn no_valid_ranges_return_none() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![0_i64, 3]);
            let ends = PyArray1::from_vec(py, vec![0_i64, 2]);
            let aggregations = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        "*".into_pyobject(py).unwrap().into_any(),
                        "count".into_pyobject(py).unwrap().into_any(),
                    ],
                )
                .unwrap()],
            )
            .unwrap();
            assert!(aggregate_starts_ends(
                py,
                starts.readonly(),
                ends.readonly(),
                4,
                &aggregations
            )
            .unwrap()
            .is_none());
        });
    }

    #[test]
    fn broad_batches_use_range_tables_for_overlapping_ranges() {
        Python::initialize();
        Python::attach(|py| {
            let values_vec = vec![2_i64, 3, 4, 5, 1, 6, 7, 8];
            let starts_vec: Vec<i64> = vec![0_i64, 1, 0].into_iter().cycle().take(100).collect();
            let ends_vec: Vec<i64> = vec![8_i64, 8, 8].into_iter().cycle().take(100).collect();
            let starts = PyArray1::from_vec(py, starts_vec.clone());
            let ends = PyArray1::from_vec(py, ends_vec.clone());

            // Use four independent requests so the optimized state path is
            // checked for every associative integer operation and for both
            // position-returning extrema.
            let values_sum = PyArray1::from_vec(py, values_vec.clone());
            let values_product = PyArray1::from_vec(py, values_vec.clone());
            let values_min = PyArray1::from_vec(py, values_vec.clone());
            let values_max = PyArray1::from_vec(py, values_vec.clone());
            let mask_sum = PyArray1::from_vec(py, vec![false; 8]);
            let mask_product = PyArray1::from_vec(py, vec![false; 8]);
            let mask_min = PyArray1::from_vec(py, vec![false; 8]);
            let mask_max = PyArray1::from_vec(py, vec![false; 8]);
            let sum = PyTuple::new(
                py,
                [
                    values_sum.into_any(),
                    mask_sum.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let product = PyTuple::new(
                py,
                [
                    values_product.into_any(),
                    mask_product.into_any(),
                    "product".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let min = PyTuple::new(
                py,
                [
                    values_min.into_any(),
                    mask_min.into_any(),
                    "min".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let max = PyTuple::new(
                py,
                [
                    values_max.into_any(),
                    mask_max.into_any(),
                    "max".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [sum, product, min, max]).unwrap();
            let result =
                aggregate_starts_ends(py, starts.readonly(), ends.readonly(), 8, &aggregations)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true; 100]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();

            let mut expected_sum = Vec::with_capacity(100);
            let mut expected_product = Vec::with_capacity(100);
            let mut expected_min = Vec::with_capacity(100);
            let mut expected_max = Vec::with_capacity(100);
            for row in 0..100 {
                let start = starts_vec[row] as usize;
                let end = ends_vec[row] as usize;
                expected_sum.push(direct_sum(&values_vec, start, end));
                expected_product.push(direct_product(&values_vec, start, end));
                expected_min.push(direct_min_position(&values_vec, start, end));
                expected_max.push(direct_max_position(&values_vec, start, end));
            }
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                expected_sum
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                expected_product
            );
            assert_eq!(
                arrays.get_item(2).unwrap().extract::<Vec<i64>>().unwrap(),
                expected_min
            );
            assert_eq!(
                arrays.get_item(3).unwrap().extract::<Vec<i64>>().unwrap(),
                expected_max
            );
        });
    }
}
