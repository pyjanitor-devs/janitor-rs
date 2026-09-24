//! Fused forward aggregation for dual non-equi regions.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::BTreeMap;

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core};
use crate::multi_join_indices::common::{add_right_region, checked_region_start, GroupState};

/// Aggregate successful candidates from the dual-region traversal without
/// materialising the intermediate index arrays.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to create NumPy results.
/// * `left_region` - Left-side region labels, one label per output row.
/// * `right_region` - Right-side region labels, indexed by candidate position.
/// * `starts` - Non-increasing right-side start boundary for each left row.
/// * `aggregations` - Non-empty list of `(array, null_mask, operation)` tuples
///   whose arrays are indexed by right-side candidate position. The value
///   arrays must be null-free. The mask is the sole null-tracking mechanism:
///   `true` marks a null for value-based operations and `false` asserts a valid
///   value. The caller is responsible for keeping each array and mask aligned.
///   A three-element `count` request counts non-null values; `size` or the
///   two-element `("*", "count")` shorthand counts every successful
///   comparison without requiring a value column.
///
/// # Returns
///
/// `Some((matched, list))`, where `matched` is a boolean array aligned to the
/// left rows and `list` contains aggregation result arrays, each with
/// `left_region.len()` entries. Returns `None` when no region comparison
/// succeeds.
///
/// # Errors
///
/// Returns a Python exception for empty or mismatched inputs, invalid region
/// boundaries, unsupported aggregation dtypes/operations, or invalid masks.
#[pyfunction]
pub fn aggregate_dual_regions<'py>(
    py: Python<'py>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left = left_region.as_array();
    let right = right_region.as_array();
    let starts = starts.as_array();
    ensure_nonempty_core("left_region", left.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right_region", right.len()).map_err(PyValueError::new_err)?;
    ensure_equal_lengths("left region", left.len(), "starts", starts.len())?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut set = AggregationSet::new(left.len(), right.len(), &inputs, true)?;
    let mut next = vec![-1_i64; right.len()];
    let mut groups = BTreeMap::<i64, GroupState>::new();
    let mut previous_end = right.len();
    for row in 0..left.len() {
        let Some(start) = checked_region_start(starts[row], right.len(), previous_end)
            .map_err(PyValueError::new_err)?
        else {
            continue;
        };
        add_right_region(right, start, previous_end, &mut next, &mut groups);
        previous_end = start;
        for (_, state) in groups.range(left[row]..) {
            let mut position = state.head;
            while position >= 0 {
                set.update(position as usize, row);
                position = next[position as usize];
            }
        }
    }
    // Do not infer match status from identities such as sum=0 or product=1.
    // The state tracks comparison success separately and exposes it through
    // the `matched` array in the returned `(matched, aggregation_arrays)`
    // tuple.
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_dual_regions, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn maps_each_successful_region_candidate_to_its_left_output_row() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    mask.into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [aggregation]).unwrap();
            let result = aggregate_dual_regions(
                py,
                left.readonly(),
                right.readonly(),
                starts.readonly(),
                &aggregations,
            )
            .unwrap()
            .unwrap();
            let result_list = result.get_item(1).unwrap();
            let result = result_list.cast::<PyList>().unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![2, 1]
            );
        });
    }
}
