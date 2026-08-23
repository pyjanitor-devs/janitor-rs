use pyo3::prelude::*;

pub mod min_ends;
pub mod min_ends_matches;
pub mod min_no_range;
pub mod min_positions;
pub mod min_starts;
pub mod min_starts_ends;
pub mod min_starts_ends_matches;
pub mod min_starts_matches;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    min_ends::register(m)?;
    min_ends_matches::register(m)?;
    min_no_range::register(m)?;
    min_positions::register(m)?;
    min_starts::register(m)?;
    min_starts_ends::register(m)?;
    min_starts_ends_matches::register(m)?;
    min_starts_matches::register(m)?;
    Ok(())
}
