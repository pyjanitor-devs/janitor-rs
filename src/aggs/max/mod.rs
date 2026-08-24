use pyo3::prelude::*;

pub mod max_ends;
pub mod max_ends_matches;
pub mod max_positions;
pub mod max_starts;
pub mod max_starts_ends;
pub mod max_starts_ends_matches;
pub mod max_starts_matches;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    max_ends::register(m)?;
    max_ends_matches::register(m)?;
    max_positions::register(m)?;
    max_starts::register(m)?;
    max_starts_ends::register(m)?;
    max_starts_ends_matches::register(m)?;
    max_starts_matches::register(m)?;
    Ok(())
}
