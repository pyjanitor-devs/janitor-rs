//! Forward aggregation building blocks for fused comparison kernels.
//!
//! This module separates the Python input boundary from the accumulator
//! implementation:
//!
//! 1. [`input`] parses Python tuples, operation names, and supported dtypes.
//! 2. [`state`] owns the complete accumulator state and all update rules.
//!
//! Keeping the numerical behavior in one state module makes the operation
//! contracts easy to review together. A contributor adding an operation can
//! update the enum and parser in `input.rs`, then add its state and update
//! behavior in `state.rs`.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

mod input;
mod state;
pub(crate) mod states_ends;
pub(crate) mod states_ends_rev;
pub(crate) mod states_starts;
pub(crate) mod states_starts_ends;
pub(crate) mod states_starts_ends_rev;
pub(crate) mod states_starts_rev;

// The parent module exposes the input parser, shared state, and three
// range-shaped entry points needed by the forward aggregation paths.
// The concrete input and operation types stay private to this directory, so
// adding a dtype does not enlarge the crate's public API.
pub(crate) use input::parse_inputs;
pub(crate) use state::AggregationSet;

/// Register reverse range-only fused aggregation entry points.
pub(crate) fn register_reverse(m: &Bound<'_, PyModule>) -> PyResult<()> {
    states_ends_rev::register(m)?;
    states_starts_rev::register(m)?;
    states_starts_ends_rev::register(m)?;
    Ok(())
}

/// Register the three forward range-only aggregation entry points.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    states_ends::register(m)?;
    states_starts::register(m)?;
    states_starts_ends::register(m)?;
    Ok(())
}

/// Build the common Python result shape for fused aggregations.
///
/// The first tuple item is a boolean array indicating which output positions
/// received at least one successful comparison. The second item is the list
/// of requested aggregation arrays, in request order. Keeping this assembly
/// here ensures forward and reverse wrappers expose exactly the same shape.
pub(crate) fn make_results<'py>(
    py: Python<'py>,
    set: AggregationSet<'_>,
) -> PyResult<Bound<'py, PyTuple>> {
    let (matched, results) = set.into_results(py);
    let matched = matched.expect("legacy aggregation results always request matched metadata");
    let results = PyList::new(py, results)?;
    PyTuple::new(py, [matched, results.into_any().unbind()])
}

/// Build the result used by single and extended fused joins.
///
/// `output_positions` describes the trimmed physical output layout. The
/// aggregation state is already in that same calculation order; this helper
/// only preserves the map in the returned tuple. It deliberately does not
/// scatter or reorder any accumulator buffer.
pub(crate) fn make_results_with_positions<'py>(
    py: Python<'py>,
    set: AggregationSet<'_>,
    output_positions: Option<ArrayView1<'_, i64>>,
    output_len: usize,
    return_matched: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let output_positions = output_positions
        .map(|values| values.to_owned())
        .unwrap_or_else(|| Array1::from_iter((0..output_len).map(|position| position as i64)))
        .into_pyarray(py)
        .unbind()
        .into_any();
    let (matched, results) = set.into_results(py);
    let results = PyList::new(py, results)?;
    if return_matched {
        let matched = matched.expect("matched metadata was requested but not allocated");
        PyTuple::new(py, [output_positions, matched, results.into_any().unbind()])
    } else {
        PyTuple::new(py, [output_positions, results.into_any().unbind()])
    }
}
