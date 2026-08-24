use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_tape_width;
use crate::compare::op::CompareOp;

/// For every left row `i`, compare `left[i]` against `right[starts[i]..ends[i])`
/// under `op`, but only at positions the caller has already flagged as
/// live in `matches` (a *flat* array covering every row's candidate range
/// back to back, not one array per row).
///
/// ELI5: `matches` is one long tape shared by every row. Each tape entry is
/// one tiny yes/no flag for one candidate position: `1` means "this position
/// is still in the running" and `0` means "an earlier condition already
/// ruled it out". As we scan row `i`'s slice of `right`, we walk the tape
/// forward one tick (`n += 1`) for every candidate position, even when the
/// flag says to skip it. That keeps two rows lined up with the single flat
/// tape even when their candidate ranges have different widths.
///
/// Returns `(result, counts_array, total)`: `result[n]` is `1`/`0` for
/// whether position `n` on the tape satisfies `op` (left as `0` when the
/// tape said skip), `counts_array[i]` is how many positions row `i`
/// matched, and `total` is the grand total across every row.
///
/// A row with an invalid/inverted range (`start` negative, `end` the `-1`
/// sentinel, `start >= end`, or `end` beyond `right.len()`) contributes
/// zero ticks to the tape, matching a genuinely empty range.
///
/// ELI5 (why negativity is checked before the cast, not after): a `usize`
/// can't represent a negative number, so casting one doesn't keep it
/// negative -- it wraps around to a huge positive `usize` instead. A
/// `start_ >= end_` check done only *after* casting would miss exactly
/// the case where *both* bounds are negative but still satisfy
/// `start < end` in `i64` space (e.g. `start=-3, end=-2`): both wrap to
/// huge-but-still-ordered `usize` values, so the post-cast comparison
/// would see a small, genuine-looking non-empty range and walk
/// `right`/`matches` far past their real length instead of recognizing
/// the row as invalid. Requiring both bounds non-negative *before* the
/// cast closes that gap -- once `start >= 0` and `start < end` both hold,
/// `end` is provably positive too, so no separate `end < 0` check is
/// needed to reach that guarantee.
///
/// ELI5 (`end as usize > right_len`, issue #53): a *positive* but oversized
/// `end` (e.g. `end = right.len() + 1`) survives every check above
/// unchanged -- it's non-negative and `start < end` holds fine -- and then
/// indexes `right[nn]` out of bounds once the loop reaches `nn ==
/// right_len`. This one extra comparison, using the same already-cast
/// `start_`/`end_` this function needs regardless, is deliberately kept as
/// a plain condition here rather than routed through `checked_range` (used
/// for the same `0 <= start <= end <= len` contract elsewhere, e.g.
/// `index_builder`): benchmarking this specific per-row hot loop showed
/// `checked_range`'s `Option`-returning `usize::try_from` calls cost
/// ~25-30% more wall time here than the equivalent plain comparisons (see
/// `benches/kernels.rs`'s `compare_start_end` group) -- consistent with
/// `compare::op`'s own doc comment on indirection being measurably costly
/// in this same per-row-comparison hot path.
pub fn compare_start_end_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    op: CompareOp,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(matches.len());
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut total: i64 = 0;
    let mut n: usize = 0;
    let right_len = right.len();
    let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
    for (position, (left_val, start, end)) in zipped.enumerate() {
        if *start < 0 || *end == -1 || *start >= *end || *end as usize > right_len {
            continue;
        }
        let start_ = *start as usize;
        let end_ = *end as usize;
        let mut counter: i64 = 0;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let right_val = right[nn];
            let compare = op.apply(left_val, &right_val);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }
        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let starts_view = starts.as_array();
            let ends_view = ends.as_array();
            // ELI5: mirrors `compare_start_end_core`'s own row-rejection
            // condition exactly (same four checks, same plain-comparison
            // style -- see that function's doc comment for why this isn't
            // routed through `checked_range` here either) -- a row the
            // core skips contributes zero ticks there, so it must also
            // contribute zero here, not its raw (possibly huge or
            // negative) `e - s`.
            let right_len = right.as_array().len();
            let expected_matches_width: usize = starts_view
                .iter()
                .zip(ends_view.iter())
                .filter(|(s, e)| **s >= 0 && **e != -1 && **s < **e && **e as usize <= right_len)
                .map(|(s, e)| (*e as usize) - (*s as usize))
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let op = CompareOp::try_from_code(op)?;
            let (result, counts_array, total) = compare_start_end_core(
                left.as_array(),
                right.as_array(),
                starts_view,
                ends_view,
                matches.as_array(),
                op,
            );
            Ok((
                result.into_pyarray(py),
                counts_array.into_pyarray(py),
                total,
            ))
        }
    };
}

generic_compare!(compare_start_end_int64, i64);
generic_compare!(compare_start_end_int32, i32);
generic_compare!(compare_start_end_int16, i16);
generic_compare!(compare_start_end_int8, i8);
generic_compare!(compare_start_end_uint64, u64);
generic_compare!(compare_start_end_uint32, u32);
generic_compare!(compare_start_end_uint16, u16);
generic_compare!(compare_start_end_uint8, u8);
generic_compare!(compare_start_end_f64, f64);
generic_compare!(compare_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    use CompareOp::{Eq as EQ, Ge as GE, Gt as GT, Le as LE, Lt as LT, Ne as NE};

    #[test]
    fn both_bounds_negative_but_start_less_than_end_contributes_nothing_not_a_panic() {
        // start=-3, end=-2 aren't the `-1` sentinel, and -3 < -2 holds in
        // i64 space, so the old `start == -1`-only check let this row
        // through. Both bounds then cast to *huge*, still-ordered usize
        // values (start_ < end_ survives the cast), so the loop walked
        // `right`/`matches` far out of bounds instead of contributing
        // zero ticks.
        let left = array![5_i64];
        let right = array![1_i64];
        let starts = array![-3_i64];
        let ends = array![-2_i64];
        let matches = array![1_i8];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, Array1::<i8>::zeros(1));
        assert_eq!(counts, array![0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn each_op_code_matches_its_operator() {
        let left = array![5_i64];
        let right = array![5_i64, 4, 6];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let matches = array![1_i8, 1, 1]; // nothing pre-filtered

        let cases = [
            (GT, [0_i8, 1, 0]), // 5>5 f, 5>4 t, 5>6 f
            (GE, [1, 1, 0]),
            (LT, [0, 0, 1]),
            (LE, [1, 0, 1]),
            (EQ, [1, 0, 0]),
            (NE, [0, 1, 1]),
        ];
        for (op, expected) in cases {
            let (result, counts, total) = compare_start_end_core(
                left.view(),
                right.view(),
                starts.view(),
                ends.view(),
                matches.view(),
                op,
            );
            assert_eq!(result, Array1::from_vec(expected.to_vec()), "op={op:?}");
            let expected_count: i64 = expected.iter().map(|&v| v as i64).sum();
            assert_eq!(counts, array![expected_count], "op={op:?}");
            assert_eq!(total, expected_count, "op={op:?}");
        }
    }

    #[test]
    fn sentinel_range_contributes_zero_ticks_not_a_panic() {
        // -1 is the crate's "invalid/no match" sentinel. Cast naively to
        // usize alone it becomes usize::MAX, and the inner loop would
        // walk `right`/`matches` straight out of bounds instead of
        // contributing zero ticks to the shared tape.
        let left = array![3_i64, 5_i64];
        let right = array![1_i64, 4, 9];
        let starts = array![0_i64, 1_i64];
        let ends = array![3_i64, -1_i64];
        let matches = array![1_i8, 1, 1];

        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            LT,
        );
        // row 0: 3<1 f, 3<4 t, 3<9 t
        assert_eq!(result, array![0, 1, 1]);
        assert_eq!(counts, array![2, 0]);
        assert_eq!(total, 2);
    }

    #[test]
    fn matches_mask_skips_positions_but_still_advances_the_tape() {
        // right has 3 candidates for the single row; the middle one is
        // pre-filtered out via matches=0, so it must not be compared even
        // though 4 > 3 would otherwise be true.
        let left = array![3_i64];
        let right = array![1_i64, 4, 9];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let matches = array![1_i8, 0, 1];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, array![1_i8, 0, 0]); // middle stays 0, not evaluated
        assert_eq!(counts, array![1]);
        assert_eq!(total, 1);
    }

    #[test]
    fn empty_range_contributes_nothing() {
        let left = array![3_i64];
        let right: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let matches: Array1<i8> = array![];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, Array1::<i8>::zeros(0));
        assert_eq!(counts, array![0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn zero_matches_gives_zero_total_and_zero_counts() {
        let left = array![1_i64, 2];
        let right = array![100_i64, 200];
        let starts = array![0_i64, 0];
        let ends = array![2_i64, 2];
        let matches = array![1_i8, 1, 1, 1]; // flat tape: 2 rows x 2 candidates
        let (_, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT, // left values are far smaller than right values -> never `>`
        );
        assert_eq!(counts, array![0, 0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn duplicate_values_all_count_under_ge_and_le() {
        let left = array![5_i64];
        let right = array![5_i64, 5, 5];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let matches = array![1_i8, 1, 1];
        let (_, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GE,
        );
        assert_eq!(counts, array![3]);
        assert_eq!(total, 3);
    }

    #[test]
    fn multiple_rows_share_one_flat_matches_tape() {
        // two rows, each with a 2-wide candidate range: the tape length is
        // the sum of both rows' widths, not per-row.
        let left = array![1_i64, 10];
        let right = array![0_i64, 2, 9, 11];
        let starts = array![0_i64, 2];
        let ends = array![2_i64, 4];
        let matches = array![1_i8, 1, 1, 1];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        // row 0: 1>0 t, 1>2 f ; row 1: 10>9 t, 10>11 f
        assert_eq!(result, array![1_i8, 0, 1, 0]);
        assert_eq!(counts, array![1, 1]);
        assert_eq!(total, 2);
    }

    #[test]
    fn non_sentinel_negative_start_does_not_underflow_the_width_precheck() {
        use numpy::{PyArray1, PyArrayMethods};
        use pyo3::exceptions::PyValueError;
        use pyo3::Python;

        // starts=[-2], ends=[1]: -2 isn't the `-1` sentinel, and -2 < 1
        // holds in i64 space, so the old `!= -1`-only filter let this row
        // through and then underflowed `(1usize) - ((-2i64) as usize)`
        // before `ensure_tape_width` ever ran (panicking in debug builds,
        // silently producing a bogus width in release). It must now be
        // rejected as cleanly as any other malformed range, not panic.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let left = PyArray1::from_vec(py, vec![3_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64]);
            let starts = PyArray1::from_vec(py, vec![-2_i64]);
            let ends = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let result = compare_start_end_int64(
                py,
                left.readonly(),
                right.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                // `compare_start_end_int64` is the `#[pyfunction]` wrapper,
                // which still takes `CompareOp::try_from_code`'s raw i8
                // code (not the `GT`/`LT`/... `CompareOp` aliases this
                // module's other tests use against `compare_start_end_core`
                // directly) -- 0 is `CompareOp::Gt`, matching the mapping
                // in `wrapper_op_validation_tests` in `compare/mod.rs`.
                0,
            );
            // Whether this is accepted (because the row is now correctly
            // excluded from the width sum) or rejected with a clean
            // PyValueError is both fine -- what must never happen is a
            // panic (an unrecoverable `pyo3_runtime.PanicException` on the
            // Python side instead of a catchable exception).
            if let Err(error) = result {
                assert!(
                    error.is_instance_of::<PyValueError>(py),
                    "expected a PyValueError, got {error:?}"
                );
            }
        });
    }

    #[test]
    fn end_equal_to_right_len_is_a_valid_inclusive_bound() {
        // end == right.len() is not "one past the end of what exists" --
        // it is the ordinary exclusive bound of a range that reaches the
        // last element (`0..right.len()` covers every element). This must
        // stay accepted, not get pulled into the `end > right.len()`
        // rejection alongside issue #53's actual out-of-bounds case below.
        let left = array![3_i64];
        let right = array![1_i64];
        let starts = array![0_i64];
        let ends = array![1_i64]; // == right.len()
        let matches = array![1_i8];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, array![1_i8]); // 3 > 1
        assert_eq!(counts, array![1]);
        assert_eq!(total, 1);
    }

    #[test]
    fn end_beyond_right_len_contributes_nothing_not_a_panic() {
        // Issue #53's reproduction: right has length 1, but end=2 asks the
        // loop to walk `0..2` and index `right[1]`, which is out of
        // bounds. `end` is positive and `start < end` holds, so the
        // pre-#53 checks (start >= 0, start < end) both passed this row
        // through unchanged -- only bounding `end` against `right.len()`
        // (via `checked_range`) catches it.
        let left = array![1_i64];
        let right = array![1_i64];
        let starts = array![0_i64];
        let ends = array![2_i64]; // right.len() + 1
        let matches = array![1_i8, 1];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, Array1::<i8>::zeros(2));
        assert_eq!(counts, array![0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn end_beyond_right_len_does_not_panic_through_the_python_wrapper() {
        use numpy::{PyArray1, PyArrayMethods};
        use pyo3::exceptions::PyValueError;
        use pyo3::Python;

        // Same reproduction as `end_beyond_right_len_contributes_nothing_not_a_panic`,
        // but through the `#[pyfunction]` wrapper end-to-end -- issue #53
        // was reported as a Rust panic/`PanicException` observed from the
        // Python side, not from the plain-Rust core directly, so this is
        // the case that actually needs to stop panicking.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let left = PyArray1::from_vec(py, vec![1_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![2_i64]); // right.len() + 1
            let matches = PyArray1::from_vec(py, vec![1_i8, 1]);
            let result = compare_start_end_int64(
                py,
                left.readonly(),
                right.readonly(),
                starts.readonly(),
                ends.readonly(),
                matches.readonly(),
                0, // CompareOp::Gt, see the sibling test above for why
            );
            // As with the negative-start case: accepted (row now correctly
            // excluded) or a clean PyValueError are both fine. A panic is
            // the one outcome issue #53 is about closing off.
            if let Err(error) = result {
                assert!(
                    error.is_instance_of::<PyValueError>(py),
                    "expected a PyValueError, got {error:?}"
                );
            }
        });
    }
}
