//! Forward aggregation for starts-only ranges.
//!
//! The comparison phase has already reduced each left row to a suffix
//! boundary. This module therefore performs no predicate comparison and does
//! not consume a match tape. It simply walks `right[start..right_len]` and
//! feeds each source position into the shared aggregation state.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::{make_results, parse_inputs, AggregationSet};
use crate::aggs::ensure_nonempty_core;

/// Aggregate every right-side suffix described by `starts`.
///
/// `starts[row]` is inclusive. A valid row aggregates
/// `right[start[row]..right_len]` into output slot `row`. The comparison
/// phase is complete before this function is called; every position in the
/// valid range is therefore a successful comparison.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to create NumPy results.
/// * `starts` - One inclusive suffix boundary per left output row.
/// * `right_len` - Number of positional rows on the right side. This is
///   explicit so count-all can work even when every request is `("*",
///   "count")` and therefore has no value array.
/// * `aggregations` - Aggregation requests parsed by [`parse_inputs`]. Value
///   arrays and masks must be aligned to `right_len`; `true` mask entries are
///   null and are skipped by value-based operations. `false` means valid. The
///   mask is authoritative: nullness is never inferred from the value array,
///   including from NaN or other sentinel-looking values. Callers own the
///   contract that the value array itself contains no null representation.
///
/// # Returns
///
/// `Some((matched, aggregation_arrays))` when at least one suffix is
/// non-empty. `matched` has one entry per `starts` row and is retained for
/// consistency with the other fused aggregation APIs. Returns `None` when
/// every boundary is invalid or equal to `right_len`.
#[pyfunction]
pub fn aggregate_starts<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    right_len: usize,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, pyo3::types::PyTuple>>> {
    let starts = starts.as_array();
    ensure_nonempty_core("starts", starts.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right side", right_len).map_err(PyValueError::new_err)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut state = AggregationSet::new(starts.len(), right_len, &inputs)?;
    state.aggregate_starts(starts);
    if state.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, state)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_starts, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn aggregates_suffixes_and_keeps_output_left_aligned() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![1_i64, 3]);
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let mask = PyArray1::from_vec(py, vec![false, false, true, false]);
            let sum = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let count_all = PyTuple::new(
                py,
                [
                    "*".into_pyobject(py).unwrap().into_any(),
                    "count".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let aggregations = PyList::new(py, [sum, count_all]).unwrap();
            let result = aggregate_starts(py, starts.readonly(), 4, &aggregations)
                .unwrap()
                .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true, true]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![60, 40]
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![3, 1]
            );
        });
    }

    #[test]
    fn broad_batches_use_suffix_tables_without_changing_results() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![0_i64, 0, 0, 0]);
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
            let result = aggregate_starts(py, starts.readonly(), 4, &aggregations)
                .unwrap()
                .unwrap();
            assert_eq!(
                result.get_item(0).unwrap().extract::<Vec<bool>>().unwrap(),
                vec![true; 4]
            );
            let arrays_item = result.get_item(1).unwrap();
            let arrays = arrays_item.cast::<PyList>().unwrap();
            assert_eq!(
                arrays.get_item(0).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![17; 4]
            );
        });
    }
}
