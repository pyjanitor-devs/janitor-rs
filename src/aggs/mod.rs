pub mod min;

pub mod prod;

pub mod max;
pub mod max_rev;
pub mod min_rev;
pub mod prod_rev;
pub mod size_rev;
pub mod sum;
pub mod sum_rev;

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Reject parallel arrays that cannot describe the same number of rows.
///
/// ELI5: `zip` stops when either list runs out, so unequal lists can silently
/// leave work undone. Check the two ticket books once before the hot loop and
/// give Python a normal `ValueError` instead of plausible partial results.
pub(crate) fn ensure_equal_lengths(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> PyResult<()> {
    if left_len == right_len {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "{left_name} and {right_name} must have equal lengths; got {left_len} and {right_len}"
    )))
}

/// Convert a signed position only when it names a real element.
///
/// ELI5: negative sentinels and positions past the end never become huge
/// `usize` values; they are rejected while they are still signed. Uses `<`
/// (not `<=`) because this is an *index* -- `index == len` is out of
/// bounds, unlike an exclusive end bound (see `checked_end`).
pub(crate) fn checked_index(index: i64, len: usize) -> Option<usize> {
    usize::try_from(index).ok().filter(|&index| index < len)
}

/// Convert an exclusive signed end bound in the inclusive range `0..=len`.
///
/// ELI5: uses `<=` (not `<`) because `end` is a slice bound, not an index --
/// `end == len` legitimately means "up to and including the last element".
pub(crate) fn checked_end(end: i64, len: usize) -> Option<usize> {
    usize::try_from(end).ok().filter(|&end| end <= len)
}

/// Convert a non-empty signed half-open range contained in `0..len`.
///
/// ELI5: `start` only needs `usize::try_from` (no upper-bound check of its
/// own) because it's compared against the already-validated `end` next;
/// `start < end <= len` proves `start < len` for free.
pub(crate) fn checked_range(start: i64, end: i64, len: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(start).ok()?;
    let end = checked_end(end, len)?;
    (start < end).then_some((start, end))
}

/// Reject a flat `matches` tape too short for the candidate positions every
/// row's `(start, end)` range implies it must cover.
///
/// ELI5: unlike `ensure_equal_lengths`, `matches.len()` isn't compared
/// against any *single* other array's length -- it has to be at least the
/// **sum of every row's own interval width**, which isn't known until every
/// row has been looked at. Callers sum that total themselves (respecting
/// whatever per-row rejection -- `checked_range`/`checked_index`/no
/// rejection at all -- their own loop already applies, so a row that
/// contributes zero tape entries in the main loop also contributes zero
/// here) and hand it to this helper as `expected_width`, once, before the
/// loop that actually indexes `matches[n]`. A `matches` tape *longer* than
/// needed is harmless (unused trailing entries) and not rejected here --
/// only a too-short one, which is what walks `n` past `matches.len()`.
pub(crate) fn ensure_tape_width(expected_width: usize, matches_len: usize) -> PyResult<()> {
    if expected_width <= matches_len {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "matches must have length at least {expected_width} to cover every candidate position; got {matches_len}"
    )))
}

/// Shared function-pointer shape for the `_positions` family's `#[cfg(test)]`
/// dtype-signature checks, parameterized over the input element type `T`
/// and the (already-fixed-per-macro) result element type `R`.
///
/// ELI5: a regression test assigns a generated wrapper to
/// `PositionsFn<i8, i64>` (etc.); that only compiles if the macro was
/// instantiated with the input type the function's name promises, so a
/// regression back to the wrong type is a compile error, not a runtime
/// surprise. One shared alias instead of a copy in each `_positions.rs`
/// file, so a future signature change (e.g. a new parameter) only needs
/// updating here.
#[cfg(test)]
pub(crate) type PositionsFn<T, R> =
    for<'py> fn(
        pyo3::Python<'py>,
        numpy::PyReadonlyArray1<'py, T>,
        numpy::PyReadonlyArray1<'py, i64>,
        numpy::PyReadonlyArray1<'py, i64>,
        numpy::PyReadonlyArray1<'py, i64>,
        numpy::PyReadonlyArray1<'py, bool>,
    ) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray1<R>>>;

#[cfg(test)]
mod adversarial_bounds_tests {
    use numpy::ndarray::array;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::Python;

    use super::ensure_equal_lengths;
    use super::ensure_tape_width;
    use super::max::max_ends::max_end_core;
    use super::max::max_ends_matches::max_end_match_core;
    use super::max::max_positions::max_positions_core;
    use super::max::max_starts::max_start_core;
    use super::max::max_starts_ends::max_start_end_core;
    use super::max::max_starts_ends_matches::max_start_end_match_core;
    use super::max::max_starts_matches::max_start_match_core;
    use super::min::min_ends::min_end_core;
    use super::min::min_ends_matches::min_end_match_core;
    use super::min::min_positions::min_positions_core;
    use super::min::min_starts::min_start_core;
    use super::min::min_starts_ends::min_start_end_core;
    use super::min::min_starts_ends_matches::min_start_end_match_core;
    use super::min::min_starts_matches::min_start_match_core;

    #[test]
    fn equal_length_validation_accepts_empty_and_non_empty_pairs() {
        assert!(ensure_equal_lengths("starts", 0, "ends", 0).is_ok());
        assert!(ensure_equal_lengths("starts", 3, "ends", 3).is_ok());
    }

    #[test]
    fn equal_length_validation_rejects_both_mismatch_directions() {
        Python::initialize();
        for (starts_len, ends_len) in [(2, 1), (1, 2)] {
            let error = ensure_equal_lengths("starts", starts_len, "ends", ends_len)
                .expect_err("unequal parallel arrays must be rejected");
            Python::attach(|py| {
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(
                    error.value(py).to_string(),
                    format!(
                        "starts and ends must have equal lengths; got {starts_len} and {ends_len}"
                    )
                );
            });
        }
    }

    #[test]
    fn tape_width_validation_accepts_exact_and_longer_tapes() {
        assert!(ensure_tape_width(0, 0).is_ok());
        assert!(ensure_tape_width(5, 5).is_ok());
        assert!(ensure_tape_width(5, 8).is_ok()); // longer than needed is fine
    }

    #[test]
    fn tape_width_validation_rejects_a_too_short_tape() {
        Python::initialize();
        let error = ensure_tape_width(5, 4)
            .expect_err("a matches tape shorter than the total candidate width must be rejected");
        Python::attach(|py| {
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 4"
            );
        });
    }

    #[test]
    fn representative_python_wrappers_reject_mismatched_lengths() {
        Python::initialize();
        Python::attach(|py| {
            // The ordinary Rust test job does not install Python's NumPy
            // module. Run this boundary test when NumPy is available, while
            // keeping the core-only test suite usable in that lean setup.
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            for (starts_values, ends_values) in
                [(vec![0_i64, 0], vec![1_i64]), (vec![0_i64], vec![1_i64, 1])]
            {
                let expected = format!(
                    "starts and ends must have equal lengths; got {} and {}",
                    starts_values.len(),
                    ends_values.len()
                );
                let starts = PyArray1::from_vec(py, starts_values);
                let ends = PyArray1::from_vec(py, ends_values);
                let arr = PyArray1::from_vec(py, vec![1_i64, 2]);
                let booleans = PyArray1::from_vec(py, vec![false, false]);
                let index = PyArray1::from_vec(py, vec![0_i64, 1]);

                let error = super::sum::sum_starts_ends::compute_sum_start_end_int64(
                    py,
                    arr.readonly(),
                    starts.readonly(),
                    ends.readonly(),
                    booleans.readonly(),
                )
                .expect_err("forward wrapper must reject unequal lengths");
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), expected);

                let error = super::min_rev::min_starts_ends::compute_min_rev_start_end_int64(
                    py,
                    arr.readonly(),
                    starts.readonly(),
                    ends.readonly(),
                    index.readonly(),
                    booleans.readonly(),
                    2,
                )
                .expect_err("reverse wrapper must reject unequal lengths");
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), expected);

                let error = super::size_rev::computes::compute_size_rev_start_end(
                    py,
                    starts.readonly(),
                    ends.readonly(),
                    index.readonly(),
                    2,
                )
                .expect_err("reverse-size wrapper must reject unequal lengths");
                assert!(error.is_instance_of::<PyValueError>(py));
                assert_eq!(error.value(py).to_string(), expected);
            }
        });
    }

    #[test]
    fn representative_python_wrappers_reject_a_too_short_matches_tape() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // issue #41's own repro: a single row whose candidate range is
            // [0, 5), against a `matches` tape only 1 wide. Before the fix
            // this panicked (`pyo3_runtime.PanicException: index out of
            // bounds`) instead of raising a catchable Python exception.
            let arr = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4, 5]);
            let ends = PyArray1::from_vec(py, vec![5_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false; 5]);
            let error = super::max::max_ends_matches::compute_max_end_match_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect_err("a matches tape shorter than the candidate range must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 1"
            );

            // Same shape of bug, unguarded single-bound family (no
            // `checked_range` at all -- `sum`/`prod` forward `_ends_matches`
            // never validated `end` against anything before this fix).
            let arr = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4, 5]);
            let ends = PyArray1::from_vec(py, vec![5_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false; 5]);
            let error = super::sum::sum_ends_matches::compute_sum_end_match_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect_err("a matches tape shorter than the candidate range must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 1"
            );

            // `index_builder.rs`'s 9 functions had no `matches` length
            // check at all -- not even one comparable to `ensure_equal_lengths`.
            let index = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![5_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let error = crate::index_builder::index_starts_and_ends(
                py,
                index.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                5,
            )
            .expect_err("a matches tape shorter than the candidate range must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                error.value(py).to_string(),
                "matches must have length at least 5 to cover every candidate position; got 1"
            );

            // A tape at least as long as the total candidate width must
            // still succeed -- this fix only rejects too-short tapes.
            let arr = PyArray1::from_vec(py, vec![1_i64, 9, 4]);
            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let booleans = PyArray1::from_vec(py, vec![false; 3]);
            super::max::max_ends_matches::compute_max_end_match_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect("an exactly-sized matches tape must not be rejected");
        });
    }

    #[test]
    fn every_forward_core_rejects_signed_and_one_past_bounds() {
        let arr = array![5_i64, 1, 4];
        let booleans = array![false, false, false];
        let invalid_starts = array![-1_i64, 3, 4];
        let invalid_ends = array![-1_i64, 4];
        let zero_counts = array![0_i64, 0, 0];
        let empty_matches = array![];

        assert_eq!(
            min_start_core(arr.view(), invalid_starts.view(), booleans.view()),
            array![-1, -1, -1]
        );
        assert_eq!(
            max_start_core(arr.view(), invalid_starts.view(), booleans.view()),
            array![-1, -1, -1]
        );
        assert_eq!(
            min_end_core(arr.view(), invalid_ends.view(), booleans.view()),
            array![-1, -1]
        );
        assert_eq!(
            max_end_core(arr.view(), invalid_ends.view(), booleans.view()),
            array![-1, -1]
        );

        let starts = array![0_i64, 0];
        assert_eq!(
            min_start_end_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_start_end_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );

        let positions = array![0_i64, 1, 2];
        let position_starts = array![-1_i64, 0];
        let position_ends = array![1_i64, 4];
        assert_eq!(
            min_positions_core(
                arr.view(),
                position_starts.view(),
                position_ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_positions_core(
                arr.view(),
                position_starts.view(),
                position_ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );

        assert_eq!(
            min_start_match_core(
                arr.view(),
                invalid_starts.view(),
                zero_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1, -1]
        );
        assert_eq!(
            max_start_match_core(
                arr.view(),
                invalid_starts.view(),
                zero_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1, -1]
        );

        let invalid_end_counts = array![0_i64, 0];
        assert_eq!(
            min_end_match_core(
                arr.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_end_match_core(
                arr.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            min_start_end_match_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
        assert_eq!(
            max_start_end_match_core(
                arr.view(),
                starts.view(),
                invalid_ends.view(),
                invalid_end_counts.view(),
                empty_matches.view(),
                booleans.view(),
            ),
            array![-1, -1]
        );
    }

    #[test]
    fn zero_count_rows_return_minus_one_without_shifting_the_match_tape() {
        let arr = array![5_i64, 1, 4];
        let booleans = array![false, false, false];
        let counts = array![0_i64, 1];

        let starts = array![0_i64, 1];
        let start_matches = array![0_i8, 0, 0, 1, 0];
        assert_eq!(
            min_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let ends = array![3_i64, 2];
        let end_matches = array![0_i8, 0, 0, 0, 1];
        assert_eq!(
            min_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let interval_ends = array![3_i64, 3];
        assert_eq!(
            min_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
    }

    #[test]
    fn invalid_rows_contribute_zero_slots_to_the_match_tape() {
        let arr = array![5_i64, 1, 4];
        let booleans = array![false, false, false];
        let counts = array![0_i64, 1];

        let starts = array![-1_i64, 1];
        let start_matches = array![1_i8, 0];
        assert_eq!(
            min_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_match_core(
                arr.view(),
                starts.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let ends = array![-1_i64, 2];
        let end_matches = array![0_i8, 1];
        assert_eq!(
            min_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_end_match_core(
                arr.view(),
                ends.view(),
                counts.view(),
                end_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );

        let interval_ends = array![2_i64, 3];
        assert_eq!(
            min_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
        assert_eq!(
            max_start_end_match_core(
                arr.view(),
                starts.view(),
                interval_ends.view(),
                counts.view(),
                start_matches.view(),
                booleans.view(),
            ),
            array![-1, 1]
        );
    }

    #[test]
    fn positions_outside_the_value_array_are_skipped() {
        let arr = array![5_i64, 1, 4];
        let starts = array![0_i64];
        let ends = array![2_i64];
        let positions = array![-2_i64, 3];
        let booleans = array![false, false, false];

        assert_eq!(
            min_positions_core(
                arr.view(),
                starts.view(),
                ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1]
        );
        assert_eq!(
            max_positions_core(
                arr.view(),
                starts.view(),
                ends.view(),
                positions.view(),
                booleans.view(),
            ),
            array![-1]
        );
    }
}
