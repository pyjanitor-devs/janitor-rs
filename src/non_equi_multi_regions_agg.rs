//! Fused forward aggregation for multi-condition non-equi regions.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::BTreeMap;

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core};
use crate::compare::common::{add_right_region, checked_region_start, GroupState};
use crate::compare::predicate::{
    null_metadata_views, parse_predicates_with_nulls, predicates_match_dispatch,
};

/// Aggregate candidates from region bounds after applying all residual
/// predicates. The function owns its comparison loop and never calls the
/// existing index-producing multi-region functions.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to borrow arrays and create
///   NumPy results.
/// * `predicates` - Non-empty list of residual comparison tuples. Their left
///   and right arrays must align with the region arrays.
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
/// left rows and `list` contains aggregation result arrays in the same order
/// as the input requests. Returns `None` when no candidate passes both the
/// region and residual predicate checks.
///
/// # Errors
///
/// Returns a Python exception for empty or mismatched inputs, invalid region
/// boundaries, unsupported predicate/aggregation dtypes or operations, or
/// invalid masks.
#[pyfunction]
pub fn aggregate_multi_regions<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err("at least one predicate is required"));
    }
    let left = left_region.as_array();
    let right = right_region.as_array();
    let starts = starts.as_array();
    ensure_nonempty_core("left_region", left.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right_region", right.len()).map_err(PyValueError::new_err)?;
    ensure_equal_lengths("left region", left.len(), "starts", starts.len())?;
    for predicate in &predicates {
        ensure_equal_lengths(
            "left region",
            left.len(),
            "predicate left array",
            predicate.left_len(),
        )?;
        ensure_equal_lengths(
            "right region",
            right.len(),
            "predicate right array",
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
    let mut set = AggregationSet::new(left.len(), right.len(), &inputs)?;
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
                let candidate = position as usize;
                if predicates_match_dispatch(&views, metadata_views.as_deref(), row, candidate) {
                    set.update(candidate, row);
                }
                position = next[candidate];
            }
        }
    }
    // Region traversal may produce output positions with identities when no
    // value survives its mask. Only a pass with zero successful comparisons
    // returns `None`; otherwise `matched` carries the per-left-row status.
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_multi_regions, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn filters_region_candidates_before_updating_aggregations() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let left_region = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right_region = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
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
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
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
            let result = aggregate_multi_regions(
                py,
                &predicates,
                left_region.readonly(),
                right_region.readonly(),
                starts.readonly(),
                &aggregations,
            )
            .unwrap()
            .unwrap();
            let result_list = result.get_item(1).unwrap();
            let result = result_list.cast::<PyList>().unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![50, 30]
            );
        });
    }
}
