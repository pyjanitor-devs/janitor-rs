//! Fused reverse aggregation for dual non-equi regions.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::collections::BTreeMap;

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core, ensure_unique_index};
use crate::multi_join_indices::common::{add_right_region, checked_region_start, GroupState};

/// Aggregate successful dual-region candidates into right-position slots.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to create NumPy outputs.
/// * `left_region` - Left region values, one per source row.
/// * `right_region` - Right region values, one per candidate position.
/// * `starts` - Non-increasing right-position suffix start per left row.
/// * `right_index` - Unique right labels in output order; labels may be
///   unordered, but output slot `n` corresponds to `right_index[n]`.
/// * `aggregations` - Non-empty `(array, null_mask, operation)` tuples aligned
///   to left rows. The mask is authoritative: `true` means the corresponding
///   value is null and is skipped by `sum`, `product`, `min`, and `max`;
///   `false` means the value is valid. Nullness is never inferred from the
///   value array, including from `NaN`. A three-element `count` request
///   counts only non-null source values; `size` and the two-element
///   `("*", "count")` shorthand count every successful comparison without
///   requiring a value column.
///
/// # Returns
///
/// `Some((matched, list))`, where `matched` is a boolean array aligned to the
/// right positions and `list` contains right-aligned aggregation arrays in
/// request order. Unmatched slots retain identities or `-1` for min/max.
/// Returns `None` when no region comparison succeeds.
///
/// # Errors
///
/// Returns an exception for empty/mismatched inputs, duplicate right labels,
/// invalid boundaries, or invalid aggregation inputs.
#[pyfunction]
pub fn aggregate_dual_regions_reverse<'py>(
    py: Python<'py>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left = left_region.as_array();
    let right = right_region.as_array();
    let starts = starts.as_array();
    let right_index = right_index.as_array();
    ensure_nonempty_core("left_region", left.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right_region", right.len()).map_err(PyValueError::new_err)?;
    ensure_equal_lengths("left region", left.len(), "starts", starts.len())?;
    ensure_equal_lengths(
        "right region",
        right.len(),
        "right index",
        right_index.len(),
    )?;
    ensure_unique_index("right index", right_index)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut set = AggregationSet::new(right.len(), left.len(), &inputs)?;
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
                set.update(row, position as usize);
                position = next[position as usize];
            }
        }
    }
    // Reverse output is indexed by right position. A slot that was never
    // reached is distinguishable from a matched slot with an identity result
    // only through the tuple's `matched` boolean array.
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_dual_regions_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn writes_region_matches_to_right_positions() {
        Python::initialize();
        Python::attach(|py| {
            let left_region = PyArray1::from_vec(py, vec![2_i64]);
            let right_region = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![30_i64, 10, 20]);
            let values = PyArray1::from_vec(py, vec![7_i64]);
            let mask = PyArray1::from_vec(py, vec![false]);
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
            let result = aggregate_dual_regions_reverse(
                py,
                left_region.readonly(),
                right_region.readonly(),
                starts.readonly(),
                right_index.readonly(),
                &aggregations,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![false, true, true]
            );
            let result_list = result.get_item(1).unwrap();
            let result = result_list.cast::<PyList>().unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![0, 7, 7]
            );
        });
    }
}
