use pyo3::prelude::*;

pub mod computes;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    computes::register(m)?;
    Ok(())
}
