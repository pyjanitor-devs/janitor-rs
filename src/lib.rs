use pyo3::prelude::*;
mod aggs;
mod bin_search;
mod compare;
mod index_builder;
mod left_le_right;

/// Narrow Rust-only surface used by `benches/kernels.rs`.
///
/// ELI5: Criterion benchmarks are separate Rust programs, so they need a
/// public door into this library. This door exposes only the small set of
/// algorithms they time instead of opening every implementation module and
/// hundreds of Python wrappers.
#[doc(hidden)]
pub mod bench_support {
    pub use crate::aggs::sum::sum_ends::sum_end_core;
    pub use crate::aggs::sum::sum_starts::{sum_start_core, sum_start_u32_core};
    pub use crate::aggs::sum::sum_starts_ends::sum_start_end_core;
    pub use crate::bin_search::bin_search_lt::binary_search_lt_core;
    pub use crate::compare::comp::compare_start_end_core;
    pub use crate::index_builder::{repeat_index_core, trim_index_core};
}

/// Top-level composition point: each family owns and registers its own
/// exports, so this function only has to know the five family names, not
/// the ~900 individual dtype-specialized functions they expose.
///
/// ELI5: instead of one giant guest list at the front door, each
/// department (binary search, comparison, index building, aggregation)
/// keeps its own short list and reports up through its `register`
/// function; the front door just asks each department to check its own
/// guests in.
#[pymodule]
fn janitor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    bin_search::register(m)?;
    compare::register(m)?;
    index_builder::register(m)?;
    left_le_right::register(m)?;
    aggs::register(m)?;
    Ok(())
}

/// ELI5: instead of re-checking all ~900 dtype-specialized exports (that's
/// what `cargo test` running every family's own tests already does), this
/// builds the real module once and asks it for one guest per department --
/// enough to prove each family's `register` actually got wired into
/// `janitor_rs`, not just that it compiles.
#[cfg(test)]
mod registration_tests {
    use super::janitor_rs;
    use pyo3::prelude::*;

    #[test]
    fn every_family_registers_a_representative_export() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "janitor_rs_registration_test")
                .expect("module creation must not fail");
            janitor_rs(&module).expect("registration must not fail");

            let representative_exports = [
                "binary_search_lt_int64",            // bin_search
                "compare_start_end_int64",           // compare
                "repeat_index",                      // index_builder
                "get_positions_where_left_le_right", // left_le_right
                "compute_sum_start_int64",           // aggs::sum
                "compute_sum_rev_start_int64",       // aggs::sum_rev
                "compute_min_start_int64",           // aggs::min
                "compute_min_rev_start_int64",       // aggs::min_rev
                "compute_max_start_int64",           // aggs::max
                "compute_max_rev_start_int64",       // aggs::max_rev
                "compute_prod_start_int64",          // aggs::prod
                "compute_prod_rev_start_int64",      // aggs::prod_rev
                "compute_size_rev_start",            // aggs::size_rev
            ];

            for name in representative_exports {
                assert!(
                    module.getattr(name).is_ok(),
                    "expected `{name}` to be registered on the janitor_rs module"
                );
            }
        });
    }
}
