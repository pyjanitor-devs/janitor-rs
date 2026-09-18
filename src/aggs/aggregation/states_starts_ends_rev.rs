//! Reverse fused aggregation for arbitrary starts/ends ranges.
//!
//! This path receives successful half-open ranges and performs dense ordinal
//! updates. It does not reconstruct a match tape and does not use a HashMap:
//! every right ordinal already has a required output slot.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::{make_results, parse_inputs, AggregationSet};
use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core, ensure_unique_index};

/// Aggregate reverse arbitrary half-open ranges into dense right slots.
///
/// Source row `row` contributes to:
///
/// ```text
/// right[starts[row]..ends[row]]
/// ```
///
/// Ranges are half-open. Negative, inverted, zero-width, and oversized
/// ranges contribute no output coverage. Output position `n` corresponds to
/// `right_index[n]`, and all output arrays have `right_index.len()` entries.
///
/// `true` in a null mask means null and is skipped by sum, product, min, max,
/// and column-based count. `false` means valid. The mask is authoritative and
/// nullness is never inferred from values, including NaN. Wildcard count-all
/// ignores the mask. Min/max return source-row positions; ties do not need a
/// stable position, only a position containing the true extreme value.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to create NumPy outputs.
/// * `starts` - Inclusive start boundary per source/left row.
/// * `ends` - Exclusive end boundary per source/left row.
/// * `right_index` - Unique right labels in positional output order.
/// * `aggregations` - Non-empty value/mask/operation requests or wildcard
///   count requests aligned to the source rows.
///
/// # Returns
///
/// `Some((matched, aggregation_arrays))` when at least one valid range exists;
/// otherwise `None`. All arrays are dense and right-index aligned.
#[pyfunction]
pub fn aggregate_starts_ends_reverse<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let right_index = right_index.as_array();
    ensure_nonempty_core("starts", starts.len()).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right index", right_index.len()).map_err(PyValueError::new_err)?;
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    ensure_unique_index("right index", right_index)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut state = AggregationSet::new(right_index.len(), starts.len(), &inputs)?;
    state.aggregate_reverse_starts_ends(starts, ends);
    if state.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, state)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aggregate_starts_ends_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    #[test]
    fn writes_dense_interval_results_without_hashmap_ordering() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_vec(py, vec![1_i64, 0]);
            let ends = PyArray1::from_vec(py, vec![4_i64, 2]);
            let right_index = PyArray1::from_vec(py, vec![30_i64, 10, 40, 20]);
            let values = PyArray1::from_vec(py, vec![2_i64, 3]);
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
            let min = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "min".into_pyobject(py).unwrap().into_any(),
                ],
            )
            .unwrap();
            let requests = PyList::new(py, [sum, min]).unwrap();
            let result = aggregate_starts_ends_reverse(
                py,
                starts.readonly(),
                ends.readonly(),
                right_index.readonly(),
                &requests,
            )
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
                vec![3, 5, 2, 2]
            );
            assert_eq!(
                arrays.get_item(1).unwrap().extract::<Vec<i64>>().unwrap(),
                vec![1, 0, 0, 0]
            );
        });
    }
}
