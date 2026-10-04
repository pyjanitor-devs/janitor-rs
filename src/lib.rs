use pyo3::prelude::*;
mod aggregation_common;
mod aggregation_input;
mod aggregation_state;
mod compare_op;
mod equi_join;
mod join_aggregation_helpers;
mod join_candidate_materialization;
mod join_search;
mod join_types;
mod not_equals_only;
mod predicate;
mod range_join;
mod range_predicate;
mod regions;
mod single_range_predicate;

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
    equi_join::register(m)?;
    not_equals_only::register(m)?;
    range_join::register(m)?;
    single_range_predicate::register(m)?;
    regions::register(m)?;
    Ok(())
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
                "equi_join_indices",
                "not_equals_aggregate_int64",
                "range_join_indices",
                "single_range_predicate_indices_int64",
                "region_indices",
            ] {
                assert!(module.getattr(name).is_ok(), "missing export: {name}");
            }
        });
    }
}
