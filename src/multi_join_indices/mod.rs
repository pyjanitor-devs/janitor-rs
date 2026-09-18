use pyo3::prelude::*;

pub mod batch_indices;
pub mod batch_no_range_indices;
pub(crate) mod common;
pub mod dual_regions;
pub mod multi_regions;
pub mod op;
pub(crate) mod predicate;

/// Registers the retained fused comparison and aggregation APIs.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    batch_indices::register(m)?;
    batch_no_range_indices::register(m)?;
    dual_regions::register(m)?;
    multi_regions::register(m)?;
    Ok(())
}
