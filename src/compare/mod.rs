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
pub mod op;

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

/// A representative sample of the `compare` family's `#[pyfunction]`
/// wrappers, proving `CompareOp::try_from_code`'s validation is actually
/// wired into the generated Python-facing functions -- not just into
/// `CompareOp` itself (covered exhaustively by `op::tests`) or into
/// whichever function a manual smoke test happened to poke.
///
/// ELI5: every one of the ~160 dtype-specialized functions across these
/// 16 files decodes `op` with the exact same one-line
/// `CompareOp::try_from_code(op)?` the macro expands into, so a handful of
/// functions -- one per structurally distinct shape (plain start/end
/// range, the `i64`-typed no-range single-position lookup, the booleans/
/// extension-array `!=` variant, and the open-ended-range variant) -- is
/// enough to catch a regression in how that line got wired in, without
/// needing all ~160 combinations to prove the same one-line pattern
/// four hundred times over.
#[cfg(test)]
mod wrapper_op_validation_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::{PyErr, Python};

    fn assert_rejects_invalid_op(py: Python<'_>, result: Result<impl std::fmt::Debug, PyErr>) {
        let error = result
            .expect_err("an invalid operator code must be rejected, not silently treated as `!=`");
        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(
            error.value(py).to_string(),
            "invalid comparison operator code: 9 (expected 0..=5)"
        );
    }

    #[test]
    fn invalid_op_code_is_rejected_across_representative_wrapper_shapes() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // Plain start/end range with a `matches` tape (comp.rs).
            let left = PyArray1::from_vec(py, vec![5_i64]);
            let right = PyArray1::from_vec(py, vec![5_i64, 4, 6]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            assert_rejects_invalid_op(
                py,
                super::comp::compare_start_end_int64(
                    py,
                    left.readonly(),
                    right.readonly(),
                    starts.readonly(),
                    ends.readonly(),
                    matches.readonly(),
                    9,
                ),
            );

            // `i64`-typed no-range single-position lookup (comp_no_range.rs).
            let left = PyArray1::from_vec(py, vec![5_i64]);
            let right = PyArray1::from_vec(py, vec![5_i64]);
            let positions = PyArray1::from_vec(py, vec![0_i64]);
            assert_rejects_invalid_op(
                py,
                super::comp_no_range::compare_no_range_int64(
                    py,
                    left.readonly(),
                    right.readonly(),
                    positions.readonly(),
                    9,
                ),
            );

            // Booleans/extension-array `!=` variant (comp_ne.rs).
            let left = PyArray1::from_vec(py, vec![5_i64]);
            let right = PyArray1::from_vec(py, vec![5_i64, 4, 6]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let left_booleans = PyArray1::from_vec(py, vec![false]);
            let right_booleans = PyArray1::from_vec(py, vec![false, false, false]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            assert_rejects_invalid_op(
                py,
                super::comp_ne::compare_start_end_ne_int64(
                    py,
                    left.readonly(),
                    right.readonly(),
                    starts.readonly(),
                    ends.readonly(),
                    left_booleans.readonly(),
                    right_booleans.readonly(),
                    matches.readonly(),
                    false,
                    9,
                ),
            );

            // Open-ended range variant (comp_first_starts.rs).
            let left = PyArray1::from_vec(py, vec![5_i64]);
            let right = PyArray1::from_vec(py, vec![5_i64, 4, 6]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            assert_rejects_invalid_op(
                py,
                super::comp_first_starts::compare_first_start_int64(
                    py,
                    left.readonly(),
                    right.readonly(),
                    starts.readonly(),
                    9,
                ),
            );
        });
    }
}
