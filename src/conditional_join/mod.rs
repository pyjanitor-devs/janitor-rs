use pyo3::prelude::*;

mod compare_op;
mod equi_join;
mod join_aggregation_helpers;
mod join_candidate_materialization;
mod join_search;
mod join_types;
mod not_equals_only;
pub(crate) mod packed_spans;
mod predicate;
mod range_join;
mod range_predicate;
mod region_kd_tree;
mod regions;
mod single_range_predicate;

pub use equi_join::{equi_aggregate_bench, equi_range_and_residual_indices};
pub use range_join::{
    adaptive_extreme_positions, range_join_extended_aggregate,
    range_join_extended_aggregate_reverse,
};
pub use regions::{
    dual_regions_aggregate, dual_regions_and_residual_aggregate, n_region_aggregate,
};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    equi_join::register(m)?;
    not_equals_only::register(m)?;
    range_join::register(m)?;
    single_range_predicate::register(m)?;
    regions::register(m)?;
    Ok(())
}
