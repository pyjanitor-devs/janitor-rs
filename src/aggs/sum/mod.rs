use pyo3::prelude::*;

pub mod sum_ends;
pub mod sum_ends_matches;
pub mod sum_positions;
pub mod sum_starts;
pub mod sum_starts_ends;
pub mod sum_starts_ends_matches;
pub mod sum_starts_matches;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    sum_ends::register(m)?;
    sum_ends_matches::register(m)?;
    sum_positions::register(m)?;
    sum_starts::register(m)?;
    sum_starts_ends::register(m)?;
    sum_starts_ends_matches::register(m)?;
    sum_starts_matches::register(m)?;
    Ok(())
}
