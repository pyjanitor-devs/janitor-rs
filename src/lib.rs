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
    use numpy::ndarray::ArrayView1;
    use pyo3::prelude::*;

    pub use crate::aggs::max_rev::max_ends::max_rev_ends_core;
    pub use crate::aggs::max_rev::max_ends_matches::compute_max_rev_end_match_int64;
    pub use crate::aggs::max_rev::max_ends_matches::max_rev_end_match_core;
    pub use crate::aggs::max_rev::max_positions::max_positions_core;
    pub use crate::aggs::max_rev::max_starts::max_rev_starts_core;
    pub use crate::aggs::max_rev::max_starts_ends::max_rev_start_end_core;
    pub use crate::aggs::min_rev::min_ends::min_rev_ends_core;
    pub use crate::aggs::min_rev::min_ends_matches::compute_min_rev_end_match_int64;
    pub use crate::aggs::min_rev::min_positions::min_positions_core;
    pub use crate::aggs::min_rev::min_starts::min_rev_starts_core;
    pub use crate::aggs::min_rev::min_starts_ends::min_rev_start_end_core;
    pub use crate::aggs::prod_rev::prod_ends::prod_rev_ends_int_core;
    pub use crate::aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_int64;
    pub use crate::aggs::prod_rev::prod_starts::prod_rev_starts_int_core;
    pub use crate::aggs::size_rev::computes::compute_size_rev_end_matches;
    pub use crate::aggs::size_rev::computes::{
        size_rev_ends_core, size_rev_start_end_core, size_rev_starts_core,
    };
    pub use crate::aggs::sum::sum_ends::sum_end_core;
    pub use crate::aggs::sum::sum_starts::{sum_start_core, sum_start_u32_core};
    pub use crate::aggs::sum::sum_starts_ends::sum_start_end_core;
    pub use crate::aggs::sum_rev::sum_ends::sum_rev_ends_int_core;
    pub use crate::aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_int64;
    pub use crate::aggs::sum_rev::sum_starts::sum_rev_starts_int_core;
    pub use crate::bin_search::bin_search_ge_first::binary_search_ge_first_core;
    pub use crate::bin_search::bin_search_gt_first::binary_search_gt_first_core;
    pub use crate::bin_search::bin_search_le_first::binary_search_le_first_core;
    pub use crate::bin_search::bin_search_lt::binary_search_lt_core;
    pub use crate::bin_search::bin_search_lt_first::binary_search_lt_first_core;
    pub use crate::compare::comp::{compare_start_end_core, compare_start_end_in_place_core};
    pub use crate::compare::comp_direct::select_start_end_core;
    pub use crate::compare::comp_ends::{compare_end_allocating_core, compare_end_in_place_core};
    pub use crate::compare::comp_ne::{
        compare_ne_start_end_allocating_core, compare_ne_start_end_in_place_core,
    };
    pub use crate::compare::comp_ne_ends::{
        compare_ne_end_allocating_core, compare_ne_end_in_place_core,
    };
    pub use crate::compare::comp_ne_starts::{
        compare_ne_start_allocating_core, compare_ne_start_in_place_core,
    };
    pub use crate::compare::comp_starts::{
        compare_start_allocating_core, compare_start_in_place_core,
    };
    pub use crate::compare::op::CompareOp;
    pub use crate::index_builder::{repeat_index_core, trim_index_core};

    pub fn sum_rev_start_end_i64(
        arr: ArrayView1<'_, i64>,
        starts: ArrayView1<'_, i64>,
        ends: ArrayView1<'_, i64>,
        index: ArrayView1<'_, i64>,
        booleans: ArrayView1<'_, bool>,
    ) -> Result<(Vec<i64>, Vec<i64>), String> {
        crate::aggs::sum_rev::sum_starts_ends::sum_rev_start_end_int_core(
            arr,
            starts,
            ends,
            index,
            booleans,
            |value| value,
        )
    }

    pub fn sum_rev_start_end_f64(
        arr: ArrayView1<'_, f64>,
        starts: ArrayView1<'_, i64>,
        ends: ArrayView1<'_, i64>,
        index: ArrayView1<'_, i64>,
        booleans: ArrayView1<'_, bool>,
    ) -> Result<(Vec<i64>, Vec<f64>), String> {
        crate::aggs::sum_rev::sum_starts_ends::sum_rev_start_end_float_core(
            arr,
            starts,
            ends,
            index,
            booleans,
            |value| value,
        )
    }

    pub fn prod_rev_start_end_i64(
        arr: ArrayView1<'_, i64>,
        starts: ArrayView1<'_, i64>,
        ends: ArrayView1<'_, i64>,
        index: ArrayView1<'_, i64>,
        booleans: ArrayView1<'_, bool>,
    ) -> Result<(Vec<i64>, Vec<i64>), String> {
        crate::aggs::prod_rev::prod_starts_ends::prod_rev_start_end_int_core(
            arr,
            starts,
            ends,
            index,
            booleans,
            |value| value,
        )
    }

    pub fn prod_rev_start_end_f64(
        arr: ArrayView1<'_, f64>,
        starts: ArrayView1<'_, i64>,
        ends: ArrayView1<'_, i64>,
        index: ArrayView1<'_, i64>,
        booleans: ArrayView1<'_, bool>,
    ) -> Result<(Vec<i64>, Vec<f64>), String> {
        crate::aggs::prod_rev::prod_starts_ends::prod_rev_start_end_float_core(
            arr,
            starts,
            ends,
            index,
            booleans,
            |value| value,
        )
    }

    /// Build the fully registered Python module for wrapper benchmarks.
    pub fn registered_module<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
        let module = PyModule::new(py, "janitor_rs_bench")?;
        super::janitor_rs(&module)?;
        Ok(module)
    }
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
    /// `register`, as of this PR (894 dtype-specialized exports across 90
    /// leaf modules). Bump this alongside any PR that intentionally adds
    /// or removes an export.
    const EXPECTED_EXPORT_COUNT: usize = 896;

    /// ELI5: the representative-export test above only proves each
    /// department's guest list reports up the chain at all -- it would
    /// still pass even if one department quietly dropped a single guest
    /// from an otherwise-still-reporting list. This test instead counts
    /// heads: every dunder-free (non-`__x__`) name on a freshly registered
    /// module must be one of our own exports (Python/PyO3 module
    /// machinery -- `__name__`, `__all__`, etc. -- all use `__`-wrapped
    /// names), so counting just those catches a missing or duplicate
    /// export without spelling out all 894 names here.
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
