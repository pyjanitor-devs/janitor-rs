//! Shared result and output types for non-equi join kernels.

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Requested output selection for a join window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Select the smallest matching original right index label.
    First,
    /// Select the largest matching original right index label.
    Last,
    /// Select any one matching right index label without computing extrema.
    Any,
    /// Emit every matching pair in physical right-array order.
    All,
}

impl Keep {
    /// Parse the public string representation used by the PyO3 wrappers.
    ///
    /// Keeping this conversion at the Python boundary means the kernel code
    /// works with a closed enum and cannot silently accept a misspelled mode.
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            "any" => Ok(Self::Any),
            "all" => Ok(Self::All),
            other => Err(PyValueError::new_err(format!(
                "invalid keep value: {other} (expected one of first, last, any, all)"
            ))),
        }
    }
}

/// Own the positional windows and labels produced by a non-equi join.
///
/// `starts` and `ends` describe half-open right-array windows for each retained
/// left row. The vectors are owned so the result can outlive borrowed NumPy
/// views and can be passed to either index materialization or aggregation.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SingleJoinResult {
    /// Physical left positions corresponding to `left_index`.
    pub(crate) left_positions: Vec<usize>,
    /// Original left index labels for rows with a non-empty match window.
    pub left_index: Vec<i64>,
    /// Complete original right index-label array in supplied right-array order.
    pub right_index: Vec<i64>,
    /// Inclusive start positions of each retained row's right match window.
    pub starts: Vec<usize>,
    /// Exclusive end positions of each retained row's right match window.
    pub ends: Vec<usize>,
}

/// Build the Python dictionary returned by index-building kernels.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `left` / `right` - Materialized original index labels.
/// * `starts` / `ends` - Optional positional half-open windows. When both are
///   present, internal `usize` bounds are exported as int64.
///
/// # Returns
///
/// A dictionary containing `left_index` and `right_index`, plus `starts` and
/// `ends` when building blocks were requested.
///
/// # Errors
///
/// Returns `ValueError` if a bound cannot be represented as int64 or Python
/// object construction fails.
pub(crate) fn result_dict<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
    starts: Option<Vec<usize>>,
    ends: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    // Ordinary selected results contain only flattened left/right labels.
    // Range building-block results additionally expose positional windows.
    let result = PyDict::new(py);
    result.set_item("left_index", left.into_pyarray(py))?;
    result.set_item("right_index", right.into_pyarray(py))?;
    if let (Some(starts), Some(ends)) = (starts, ends) {
        // Keep bounds as `usize` while Rust constructs windows, then normalize
        // the Python-facing dtype to int64. This avoids platform-dependent
        // NumPy `usize` output and matches downstream aggregation kernels.
        let starts = positions_to_i64(starts)?;
        let ends = positions_to_i64(ends)?;
        result.set_item("starts", starts.into_pyarray(py))?;
        result.set_item("ends", ends.into_pyarray(py))?;
    }
    Ok(result)
}

/// Convert internal positional bounds to the stable Python-facing int64 type.
fn positions_to_i64(values: Vec<usize>) -> PyResult<Vec<i64>> {
    values
        .into_iter()
        .map(|value| {
            i64::try_from(value)
                .map_err(|_| PyValueError::new_err("single join position exceeds int64 capacity"))
        })
        .collect()
}
