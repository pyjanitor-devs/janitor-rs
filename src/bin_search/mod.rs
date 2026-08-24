use pyo3::prelude::*;

pub mod bin_search_ge;
pub mod bin_search_ge_first;
pub mod bin_search_ge_regions;
pub mod bin_search_gt;
pub mod bin_search_gt_first;
pub mod bin_search_gt_regions;
pub mod bin_search_le;
pub mod bin_search_le_first;
pub mod bin_search_lt;
pub mod bin_search_lt_first;

/// Registers every export from this family's submodules with the
/// PyO3 module.
///
/// ELI5: a department manager collects the guest lists from each of
/// their teams and hands one combined list up the chain, instead of
/// the front door needing to know every team by name.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bin_search_ge::register(m)?;
    bin_search_ge_first::register(m)?;
    bin_search_ge_regions::register(m)?;
    bin_search_gt::register(m)?;
    bin_search_gt_first::register(m)?;
    bin_search_gt_regions::register(m)?;
    bin_search_le::register(m)?;
    bin_search_le_first::register(m)?;
    bin_search_lt::register(m)?;
    bin_search_lt_first::register(m)?;
    Ok(())
}
