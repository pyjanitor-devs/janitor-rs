use pyo3::prelude::*;
mod aggregation;
mod conditional_join;

/// Narrow Rust-only surface used by criterion benchmarks.
#[doc(hidden)]
pub mod bench_support {
    pub use crate::conditional_join::adaptive_extreme_positions;
    pub use crate::conditional_join::n_region_aggregate;
    pub use crate::conditional_join::{
        dual_regions_aggregate, dual_regions_and_residual_aggregate,
    };
    pub use crate::conditional_join::{equi_aggregate_bench, equi_range_and_residual_indices};
    pub use crate::conditional_join::{
        range_join_extended_aggregate, range_join_extended_aggregate_reverse,
    };
}

/// Top-level composition point: each family owns and registers its own
/// exports, so this function only has to know the surviving family names, not
/// the ~900 individual dtype-specialized functions they expose.
///
/// ELI5: instead of one giant guest list at the front door, each
/// department (equality, inequality, range, or region joins)
/// keeps its own short list and reports up through its `register`
/// function; the front door just asks each department to check its own
/// guests in.
#[pymodule]
fn janitor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    conditional_join::register(m)
}

#[cfg(test)]
mod registration_tests {
    use super::janitor_rs;
    use pyo3::prelude::*;

    #[test]
    fn surviving_join_families_register_exports() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "janitor_rs_registration_test")
                .expect("module creation must not fail");
            janitor_rs(&module).expect("registration must not fail");
            for name in [
                "equi_range_indices",
                "equi_range_and_residual_indices",
                "equi_uniq_residual_indices",
                "equi_aggregate",
                "not_equals_aggregate_int64",
                "range_join_indices",
                "single_range_predicate_indices_int64",
                "dual_regions_indices",
                "dual_regions_and_residual_indices",
                "n_region_indices",
                "n_region_and_residual_indices",
            ] {
                assert!(module.getattr(name).is_ok(), "missing export: {name}");
            }
        });
    }
}
