//! Reverse fused aggregation for starts-only ranges.
//!
//! The comparison phase has already reduced each source row to a suffix
//! boundary. This wrapper performs no predicate comparison and no match-tape
//! reconstruction. It delegates dense right-position aggregation to the
//! shared state implementation.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{ensure_nonempty_core, ensure_unique_index};

/// Aggregate reverse suffixes into right-index-aligned arrays.
///
/// For source row `row`, `starts[row]` is an inclusive right-side position
/// and the successful range is `right[starts[row]..right_len]`. Every output
/// array has `right_index.len()` entries; output position `n` corresponds to
/// `right_index[n]`, even when labels are unordered.
///
/// `true` in a value request's null mask means null and skips sum, product,
/// min, max, and column-based count. `false` means valid. The mask is
/// authoritative: nullness is never inferred from values, including NaN.
/// Count-all (`("*", "count")` or `("*", "size")`) ignores the mask and
/// counts every successful range contribution.
///
/// Integer starts are reduced with boundary events and a left-to-right sweep.
/// Floating-point reductions update each suffix in source-row order so their
/// encounter order is preserved. Min/max return source-row positions; tied
/// positions are interchangeable as long as the pointed-to value is the
/// actual extreme.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to create NumPy outputs.
/// * `starts` - One inclusive suffix start per source/left row.
/// * `right_index` - Unique right labels in positional output order. Labels
///   may be unordered; only its length and uniqueness define output slots.
/// * `aggregations` - Non-empty aggregation requests in
///   `(array, null_mask, operation)` or wildcard count form. Value arrays and
///   masks must be aligned to `starts`.
///
/// # Returns
///
/// `Some((matched, aggregation_arrays))` when at least one valid suffix is
/// non-empty. `matched` and every aggregation array have
/// `right_index.len()` entries. Returns `None` when no valid suffix exists.
#[pyfunction]
pub fn aggregate_starts_reverse<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let starts = starts.as_array();
    let right_index = right_index.as_array();
    ensure_nonempty_core("starts", starts.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right index", right_index.len()).map_err(PyValueError::new_err)?;
    ensure_unique_index("right index", right_index)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut state = AggregationSet::new(right_index.len(), starts.len(), &inputs, true)?;
    state.aggregate_reverse_starts(starts);
    if state.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, state)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_starts_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    #[test]
    fn writes_dense_suffix_results_and_match_flags() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![1_i64, 0]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 10, 40, 30]);
            let values = PyArray1::from_vec(py, vec![2_i64, 3]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
            let requests = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            mask.clone().into_any(),
                            "sum".into_pyobject(py).unwrap().into_any(),
                        ],
                    )
                    .unwrap(),
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            mask.clone().into_any(),
                            "product".into_pyobject(py).unwrap().into_any(),
                        ],
                    )
                    .unwrap(),
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            mask.clone().into_any(),
                            "min".into_pyobject(py).unwrap().into_any(),
                        ],
                    )
                    .unwrap(),
                    PyTuple::new(
                        py,
                        [
                            values.into_any(),
                            mask.into_any(),
                            "max".into_pyobject(py).unwrap().into_any(),
                        ],
                    )
                    .unwrap(),
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py).unwrap().into_any(),
                            "count".into_pyobject(py).unwrap().into_any(),
                        ],
                    )
                    .unwrap(),
                ],
            )
            .unwrap();
            let result =
                aggregate_starts_reverse(py, starts.readonly(), right_index.readonly(), &requests)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, true, true, true]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![3, 5, 5, 5]
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![3, 6, 6, 6]
            );
            assert_eq!(
                arrays.get_item(2).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 0, 0, 0]
            );
            assert_eq!(
                arrays.get_item(3).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 1, 1, 1]
            );
            assert_eq!(
                arrays.get_item(4).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 2, 2, 2]
            );
        });
    }

    #[test]
    fn preserves_infinite_float_sum_and_separates_masked_count() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let values = PyArray1::from_vec(py, vec![f64::INFINITY, 1.0]);
            let mask = PyArray1::from_vec(py, vec![false, true]);
            let sum = PyTuple::new(
                py,
                [
                    values.clone().into_any(),
                    mask.clone().into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let non_null_count = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let all_count = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let requests = PyList::new(py, [sum, non_null_count, all_count]).unwrap();
            let result =
                aggregate_starts_reverse(py, starts.readonly(), right_index.readonly(), &requests)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, true]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            let sums = arrays.get_item(0).unwrap().extract::<Vec<f64>>().unwrap();
            assert!(sums
                .iter()
                .all(|value| value.is_infinite() && value.is_sign_positive()));
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 1]
            );
            assert_eq!(
                arrays.get_item(2).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![2, 2]
            );
        });
    }

    #[test]
    fn broad_suffix_batches_use_boundary_sweep_without_changing_results() {
        Python::initialize();
        Python::attach(|py| {
            // Four full-width suffixes cross the adaptive cutoff:
            // query_count = 4 and total_width = 16 > 3 * right_len = 12.
            let starts = PyArray1::from_vec(py, vec![0_i64, 0, 0, 0]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let values = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4]);
            let mask = PyArray1::from_vec(py, vec![false, false, false, false]);
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
                    "*".into_pyobject(py).unwrap().into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let requests = PyList::new(py, [sum, count]).unwrap();
            let result =
                aggregate_starts_reverse(py, starts.readonly(), right_index.readonly(), &requests)
                    .unwrap()
                    .unwrap();
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![10, 10, 10, 10]
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![4, 4, 4, 4]
            );
        });
    }
}
