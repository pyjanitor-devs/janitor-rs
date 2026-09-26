//! Forward aggregation for ends-only ranges.
//!
//! The comparison phase supplies one exclusive prefix boundary per left row.
//! This module aggregates `right[..end]` without re-running comparisons or
//! consulting a match tape.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::{make_results, parse_inputs, AggregationSet};
use crate::aggs::ensure_nonempty_core;

/// Aggregate every right-side prefix described by `ends`.
///
/// A valid `ends[row]` aggregates `right[..ends[row]]` into output slot
/// `row`. `matched[row]` is true exactly when the prefix is non-empty and
/// valid. Null masks affect value-based operations, but not count-all.
///
/// `right_len` is explicit because a wildcard count-all request has no value
/// array from which the source length could otherwise be inferred. For every
/// value-backed request, `true` in its mask means null and is skipped by
/// value-based operations; `false` means valid. The mask is authoritative and
/// nullness is never inferred from values, including NaN. The caller is
/// responsible for supplying a null-free value array plus its mask.
#[pyfunction]
pub fn aggregate_ends<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    right_len: usize,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let ends = ends.as_array();
    ensure_nonempty_core("ends", ends.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right side", right_len).map_err(PyValueError::new_err)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut state = AggregationSet::new(ends.len(), right_len, &inputs, true)?;
    state.aggregate_ends(ends);
    if state.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, state)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_ends, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn aggregates_prefixes_and_uses_the_explicit_right_length_for_count_all() {
        Python::initialize();
        Python::attach(|py| {
            let ends = PyArray1::from_vec(py, vec![3_i64, 0]);
            let count_all = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    "size".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [count_all]).unwrap();
            let result = aggregate_ends(py, ends.readonly(), 5, &aggregations)
                .unwrap()
                .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, false]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![3, 0]
            );
        });
    }

    #[test]
    fn broad_batches_use_prefix_tables_without_changing_results() {
        Python::initialize();
        Python::attach(|py| {
            let ends = PyArray1::from_vec(py, vec![4_i64, 4, 4, 4]);
            let values = PyArray1::from_vec(py, vec![2_i64, 3, 5, 7]);
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
            let aggregations = PyList::new(py, [sum]).unwrap();
            let result = aggregate_ends(py, ends.readonly(), 4, &aggregations)
                .unwrap()
                .unwrap();
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![17; 4]
            );
        });
    }
}
