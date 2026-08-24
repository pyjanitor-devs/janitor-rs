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
    pub use crate::bin_search::bin_search_ge_first::binary_search_ge_first_core;
    pub use crate::bin_search::bin_search_gt_first::binary_search_gt_first_core;
    pub use crate::bin_search::bin_search_le_first::binary_search_le_first_core;
    pub use crate::bin_search::bin_search_lt::binary_search_lt_core;
    pub use crate::bin_search::bin_search_lt_first::binary_search_lt_first_core;
    pub use crate::compare::comp::compare_start_end_core;
    pub use crate::compare::op::CompareOp;
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

    /// Total `m.add_function(...)` call count across every family's
    /// `register`, as of this PR (884 dtype-specialized exports across 89
    /// leaf modules). Bump this alongside any PR that intentionally adds
    /// or removes an export.
    const EXPECTED_EXPORT_COUNT: usize = 884;

    /// ELI5: the representative-export test above only proves each
    /// department's guest list reports up the chain at all -- it would
    /// still pass even if one department quietly dropped a single guest
    /// from an otherwise-still-reporting list. This test instead counts
    /// heads: every dunder-free (non-`__x__`) name on a freshly registered
    /// module must be one of our own exports (Python/PyO3 module
    /// machinery -- `__name__`, `__all__`, etc. -- all use `__`-wrapped
    /// names), so counting just those catches a missing or duplicate
    /// export without spelling out all 884 names here.
    #[test]
    fn total_registered_export_count_matches_expected() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "janitor_rs_export_count_test")
                .expect("module creation must not fail");
            janitor_rs(&module).expect("registration must not fail");

            let export_count = module
                .dir()
                .expect("dir() must not fail")
                .iter()
                .filter(|name| {
                    let name = name.to_string();
                    !(name.starts_with("__") && name.ends_with("__"))
                })
                .count();

            assert_eq!(
                export_count, EXPECTED_EXPORT_COUNT,
                "expected exactly {EXPECTED_EXPORT_COUNT} registered exports; got \
                 {export_count}. If this PR intentionally added or removed an export, update \
                 EXPECTED_EXPORT_COUNT to match -- otherwise a register() call went missing \
                 somewhere in the chain."
            );
        });
    }
}
