use pyo3::prelude::*;

pub mod comp;
pub mod comp_ends;
pub mod comp_first;
pub mod comp_first_ends;
pub mod comp_first_starts;
pub mod comp_ne;
pub mod comp_ne_1st;
pub mod comp_ne_ends;
pub mod comp_ne_ends_1st;
pub mod comp_ne_starts;
pub mod comp_ne_starts_1st;
pub mod comp_no_range;
pub mod comp_no_range_ne;
pub mod comp_posns;
pub mod comp_posns_ne;
pub mod comp_starts;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    comp::register(m)?;
    comp_ends::register(m)?;
    comp_first::register(m)?;
    comp_first_ends::register(m)?;
    comp_first_starts::register(m)?;
    comp_ne::register(m)?;
    comp_ne_1st::register(m)?;
    comp_ne_ends::register(m)?;
    comp_ne_ends_1st::register(m)?;
    comp_ne_starts::register(m)?;
    comp_ne_starts_1st::register(m)?;
    comp_no_range::register(m)?;
    comp_no_range_ne::register(m)?;
    comp_posns::register(m)?;
    comp_posns_ne::register(m)?;
    comp_starts::register(m)?;
    Ok(())
}
