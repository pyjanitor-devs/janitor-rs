use pyo3::prelude::*;

pub mod prod_ends;
pub mod prod_ends_matches;
pub mod prod_positions;
pub mod prod_starts;
pub mod prod_starts_ends;
pub mod prod_starts_ends_matches;
pub mod prod_starts_matches;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    prod_ends::register(m)?;
    prod_ends_matches::register(m)?;
    prod_positions::register(m)?;
    prod_starts::register(m)?;
    prod_starts_ends::register(m)?;
    prod_starts_ends_matches::register(m)?;
    prod_starts_matches::register(m)?;
    Ok(())
}
