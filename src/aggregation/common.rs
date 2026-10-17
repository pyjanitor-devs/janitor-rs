//! Shared aggregation boundary used by the surviving join kernels.
//!
//! The former aggregation tree has been collapsed into
//! [`crate::aggregation::input`] and [`crate::aggregation::state`]. This small
//! module keeps the validation and result-shape
//! helpers in one place without recreating the deleted per-operation kernel
//! hierarchy.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

pub(crate) mod adaptive {
    const RUNNING_WORK_FACTOR: usize = 3;
    pub(crate) const MAX_DIRECT_QUERY_COUNT: usize = 3;

    pub(crate) fn should_use_running_aggregation(
        query_count: usize,
        total_width: usize,
        array_len: usize,
    ) -> bool {
        query_count > MAX_DIRECT_QUERY_COUNT
            && total_width > array_len.saturating_mul(RUNNING_WORK_FACTOR)
    }

    /// ELI5: build one running total per packed source value, then read two
    /// entries per query. Keep tiny batches direct and require saved visits
    /// to pay for both passes. Saturation makes this cost estimate safe.
    pub(crate) fn should_use_prefix_difference(
        query_count: usize,
        total_width: usize,
        packed_len: usize,
    ) -> bool {
        query_count > MAX_DIRECT_QUERY_COUNT
            && packed_len > 0
            && total_width > packed_len.saturating_add(query_count.saturating_mul(2))
    }

    pub(crate) fn should_use_segment_tree(
        query_count: usize,
        total_width: usize,
        array_len: usize,
    ) -> bool {
        if query_count <= 3 || array_len == 0 {
            return false;
        }
        let height = (usize::BITS - (array_len - 1).leading_zeros()) as usize;
        let build = array_len.saturating_mul(5);
        let queries = query_count.saturating_mul(height).saturating_mul(2);
        total_width > build.saturating_add(queries)
    }
}

pub(crate) mod aggregation {
    pub(crate) use crate::aggregation::input::parse_inputs;
    pub(crate) use crate::aggregation::state::AggregationSet;

    use super::*;

    /// Build the stable tuple returned by every fused aggregation path.
    pub(crate) fn make_results_with_positions<'py>(
        py: Python<'py>,
        set: AggregationSet<'_>,
        output_positions: Option<ArrayView1<'_, i64>>,
        output_len: usize,
        return_matched: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let output_positions = output_positions
            .map(|values| values.to_owned())
            .unwrap_or_else(|| Array1::from_iter((0..output_len).map(|p| p as i64)))
            .into_pyarray(py)
            .unbind()
            .into_any();
        let (matched, results) = set.into_results(py);
        let results = PyList::new(py, results)?;
        if return_matched {
            let matched = matched.expect("matched metadata was requested");
            PyTuple::new(py, [output_positions, matched, results.into_any().unbind()])
        } else {
            PyTuple::new(py, [output_positions, results.into_any().unbind()])
        }
    }
}

pub(crate) fn ensure_equal_lengths(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> PyResult<()> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{left_name} and {right_name} must have equal lengths; got {left_len} and {right_len}"
        )))
    }
}

pub(crate) fn ensure_equal_lengths_core(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> Result<(), String> {
    if left_len == right_len {
        Ok(())
    } else {
        Err(format!(
            "{left_name} and {right_name} must have equal lengths; got {left_len} and {right_len}"
        ))
    }
}
