//! Reverse fused aggregation for ends-only ranges.
//!
//! The comparison phase has already produced exclusive prefix boundaries.
//! This module validates the public reverse API and delegates the dense
//! aggregation traversal to the shared state implementation.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{ensure_nonempty_core, ensure_unique_index};

/// Aggregate reverse prefixes into right-index-aligned arrays.
///
/// For source row `row`, `ends[row]` is exclusive and the successful range is
/// `right[..ends[row]]`. Every output array has `right_index.len()` entries;
/// output position `n` corresponds to `right_index[n]`, regardless of label
/// ordering.
///
/// The boolean null mask is authoritative. `true` means null and skips
/// value-based operations; `false` means valid. Nullness is never inferred
/// from values, including NaN. Wildcard count-all requests count every range
/// contribution regardless of the mask. Min/max return source-row positions,
/// and any valid tied winner is acceptable.
///
/// Integer reductions use end-boundary events and a right-to-left sweep.
/// Floating-point reductions update prefixes directly in source-row encounter
/// order to preserve numerical behavior.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to create NumPy outputs.
/// * `ends` - One exclusive prefix end per source/left row.
/// * `right_index` - Unique right labels in positional output order.
/// * `aggregations` - Non-empty `(array, null_mask, operation)` requests or
///   wildcard count requests. Value arrays and masks must align to `ends`.
///
/// # Returns
///
/// `Some((matched, aggregation_arrays))` when at least one valid non-empty
/// prefix exists; otherwise `None`. All returned arrays have
/// `right_index.len()` entries.
#[pyfunction]
pub fn aggregate_ends_reverse<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let ends = ends.as_array();
    let right_index = right_index.as_array();
    ensure_nonempty_core("ends", ends.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right index", right_index.len()).map_err(PyValueError::new_err)?;
    ensure_unique_index("right index", right_index)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut state = AggregationSet::new(right_index.len(), ends.len(), &inputs)?;
    state.aggregate_reverse_ends(ends);
    if state.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, state)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_ends_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    #[test]
    fn writes_dense_prefix_results_and_match_flags() {
        Python::initialize();
        Python::attach(|py| {
            let ends = PyArray1::from_vec(py, vec![3_i64, 1]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 10, 40, 30]);
            let values = PyArray1::from_vec(py, vec![2_i64, 3]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
            let sum = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let count = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    "size".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let requests = PyList::new(py, [sum, count]).unwrap();
            let result =
                aggregate_ends_reverse(py, ends.readonly(), right_index.readonly(), &requests)
                    .unwrap()
                    .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, true, true, false]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![5, 2, 2, 0]
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![2, 1, 1, 0]
            );
        });
    }

    #[test]
    fn broad_prefix_batches_use_boundary_sweep_without_changing_results() {
        Python::initialize();
        Python::attach(|py| {
            // Four full-width prefixes cross the adaptive cutoff:
            // query_count = 4 and total_width = 16 > 3 * right_len = 12.
            let ends = PyArray1::from_vec(py, vec![4_i64, 4, 4, 4]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let values = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4]);
            let mask = PyArray1::from_vec(py, vec![false, false, false, false]);
            let sum = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let requests = PyList::new(py, [sum]).unwrap();
            let result =
                aggregate_ends_reverse(py, ends.readonly(), right_index.readonly(), &requests)
                    .unwrap()
                    .unwrap();
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![10, 10, 10, 10]
            );
        });
    }
}
