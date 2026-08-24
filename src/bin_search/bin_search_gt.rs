use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// A row with a malformed range (`start` negative, `end` the `-1` sentinel,
/// `start >= end`, or `end` beyond `right.len()`) resolves to `-1`, matching
/// a genuinely empty range, instead of casting a negative bound to `usize`
/// and indexing `right` out of bounds.
///
/// ELI5 (why `start < 0`, not just `start == -1`): a `usize` can't represent
/// a negative number, so casting one doesn't keep it negative -- it wraps
/// around to a huge positive `usize` instead. Checking only for the `-1`
/// sentinel would miss e.g. `start=-3, end=-2`: both wrap to huge-but-still
/// -ordered `usize` values (so `start_ < end_` survives the cast), and the
/// loop walks `right` far out of bounds instead of recognizing the row as
/// invalid. Requiring `start >= 0` up front closes that gap.
pub fn binary_search_gt_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
) -> Array1<i64> {
    // ELI5: `.as_slice()` only returns `Some` when `right` is contiguous in
    // standard (C) order -- true for a plain NumPy array, false for one
    // sliced with a non-1 step. Checked once per call, not per row, since
    // contiguity doesn't change mid-call.
    let right_slice = right.as_slice();
    let mut result = Array1::<i64>::zeros(left.len());
    // Widen right.len() up to i64 instead of narrowing `end` down to
    // usize: on a 32-bit target, `*end as usize` truncates rather than
    // saturates, so an oversized `end` (e.g. usize::MAX + 2, which still
    // fits comfortably in i64) could wrap back into a small, seemingly
    // in-bounds usize and slip past this guard. right.len() casts up
    // losslessly on every real target (an array can never hold i64::MAX
    // elements), so comparing in i64 space is always correct.
    let right_len = right.len() as i64;
    let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
    for (pos, (left_value, start, end)) in zipped.enumerate() {
        if *start < 0 || *end == -1 || *start >= *end || *end > right_len {
            result[pos] = -1;
            continue;
        }
        // ELI5: `partition_point` uses the same predicate, same
        // direction, as the manual `while` loop below -- but that alone
        // only guarantees a matching result for a genuinely sorted,
        // NaN-free `right` (this function's documented precondition); see
        // bin_search_lt.rs's core for why `partition_point`'s branchless
        // search can still probe differently than the fallback for a
        // `right` that violates it. Within that precondition, the
        // predicate itself must still be written as `!(*v >= *left_value)`
        // -- the literal negation of the manual loop's "max side"
        // condition below -- not the algebraically-equivalent
        // `*v < *left_value`: those two differ whenever *either* compared
        // value is NaN (a NaN query against an otherwise sorted `right`,
        // say), even though `right` itself may be perfectly sorted in
        // that case.
        let min_idx = if let Some(slice) = right_slice {
            // clippy's neg_cmp_op_on_partial_ord exists to flag exactly
            // this kind of NaN footgun -- but here the negation *is* the
            // fix, deliberately mirroring the fallback loop's `else`
            // branch (`!(current_value >= left_value)`) rather than the
            // algebraically-"cleaner" `<`, which is what silently
            // diverged from it for a NaN query in the first place.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let rel =
                slice[*start as usize..*end as usize].partition_point(|v| !(*v >= *left_value));
            *start + rel as i64
        } else {
            let mut min_idx = *start;
            let mut max_idx = *end;
            while min_idx < max_idx {
                // to avoid overflow
                // adapted from numba's implementation
                let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
                let current_value = right[mid_idx as usize];
                if current_value >= *left_value {
                    max_idx = mid_idx;
                } else {
                    min_idx = mid_idx + 1;
                }
            }
            min_idx
        };
        if min_idx == *start {
            result[pos] = -1;
            continue;
        }
        let mid_idx = min_idx - 1;
        let current_value = right[mid_idx as usize];
        if current_value == *left_value {
            result[pos] = -1;
            continue;
        }
        result[pos] = min_idx;
    }
    result
}

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
        ) -> Bound<'py, PyArray1<i64>> {
            let result = binary_search_gt_core(
                left.as_array(),
                right.as_array(),
                starts.as_array(),
                ends.as_array(),
            );
            result.into_pyarray(py)
        }
    };
}

bin_search!(binary_search_gt_int64, i64);
bin_search!(binary_search_gt_int32, i32);
bin_search!(binary_search_gt_int16, i16);
bin_search!(binary_search_gt_int8, i8);
bin_search!(binary_search_gt_uint64, u64);
bin_search!(binary_search_gt_uint32, u32);
bin_search!(binary_search_gt_uint16, u16);
bin_search!(binary_search_gt_uint8, u8);
bin_search!(binary_search_gt_f64, f64);
bin_search!(binary_search_gt_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_gt_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s};

    #[test]
    fn contiguous_and_strided_paths_agree() {
        // `right_dense` is contiguous (fast path); `right_strided` reads
        // the identical values through a stride-2 view over a padded
        // array, so `.as_slice()` returns `None` and the fallback loop
        // runs instead. Comparing the two directly proves they agree
        // without needing to hand-derive an expected value independently
        // of either.
        let right_dense = array![10_i64, 20, 20, 30];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![10_i64, -1, 20, -1, 20, -1, 30, -1];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![5_i64, 20, 35];
        let starts = array![0_i64, 0, 0];
        let ends = array![4_i64, 4, 4];
        let fast =
            binary_search_gt_core(left.view(), right_dense.view(), starts.view(), ends.view());
        let fallback =
            binary_search_gt_core(left.view(), right_strided, starts.view(), ends.view());
        assert_eq!(fast, fallback);
    }

    #[test]
    fn contiguous_and_strided_paths_agree_when_the_query_is_nan() {
        // Regression test: the fast path used to be written as
        // `*v < *left_value`, which is only equal to the fallback's
        // literal `!(*v >= *left_value)` for totally-ordered values --
        // any comparison with NaN is `false`, so `!(a >= b)` is `true`
        // while `a < b` is `false` when either side is NaN. `right`
        // itself stays genuinely sorted here (only the query is NaN),
        // which is in-contract input, unlike a `right` that itself
        // contains NaN (see `nan_in_right_does_not_panic_but_parity_is_not_guaranteed`
        // below). Comparing the two paths directly (not against a
        // hand-derived "expected" value, since NaN ordering has no
        // independent ground truth) proves the fix restores parity for
        // this in-contract case.
        let right_dense = array![1.0_f64, 2.0, 3.0, 4.0];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![1.0_f64, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![f64::NAN];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let fast =
            binary_search_gt_core(left.view(), right_dense.view(), starts.view(), ends.view());
        let fallback =
            binary_search_gt_core(left.view(), right_strided, starts.view(), ends.view());
        assert_eq!(fast, fallback);
    }

    #[test]
    fn nan_in_right_does_not_panic_but_parity_is_not_guaranteed() {
        // See bin_search_lt.rs's core for the full explanation: a `right`
        // containing NaN violates "sorted ascending," and the fast path's
        // branchless partition_point can then probe different elements
        // than the manual fallback loop even with an identical predicate,
        // landing on a genuinely different (but never out-of-bounds)
        // answer. Deliberately not asserting fast == fallback -- that's
        // exactly the parity this out-of-contract input isn't guaranteed
        // to have.
        let right_dense = array![-1.0_f64, f64::NAN, 0.0, 1.0, 2.0, 3.0, 4.0];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![
            -1.0_f64,
            -99.0,
            f64::NAN,
            -99.0,
            0.0,
            -99.0,
            1.0,
            -99.0,
            2.0,
            -99.0,
            3.0,
            -99.0,
            4.0,
            -99.0
        ];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![0.0_f64];
        let starts = array![0_i64];
        let ends = array![7_i64];
        let fast =
            binary_search_gt_core(left.view(), right_dense.view(), starts.view(), ends.view());
        let fallback =
            binary_search_gt_core(left.view(), right_strided, starts.view(), ends.view());
        for got in [fast[0], fallback[0]] {
            assert!(
                got == -1 || (0..=7).contains(&got),
                "expected -1 or an index in 0..=7, got {got}"
            );
        }
    }

    #[test]
    fn both_bounds_negative_but_start_less_than_end_returns_minus_one_not_a_panic() {
        // start=-3, end=-2 aren't the `-1` sentinel, and -3 < -2 holds in
        // i64 space, so a `start == -1`-only check would let this row
        // through, cast both bounds to huge-but-still-ordered `usize`
        // values, and index `right` far out of bounds.
        let left = array![5_i64];
        let right = array![1_i64];
        let starts = array![-3_i64];
        let ends = array![-2_i64];
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn non_sentinel_negative_start_returns_minus_one() {
        let left = array![3_i64];
        let right = array![1_i64, 2, 3];
        let starts = array![-2_i64];
        let ends = array![2_i64];
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn end_beyond_right_len_returns_minus_one_not_a_panic() {
        // right has length 1; end=2 would ask the loop to consider index 1,
        // out of bounds.
        let left = array![1_i64];
        let right = array![1_i64];
        let starts = array![0_i64];
        let ends = array![2_i64]; // right.len() + 1
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn end_near_a_32_bit_usize_wraparound_point_is_still_rejected() {
        // On a 32-bit target, `usize` is 32 bits wide, so a narrowing
        // `*end as usize` truncates rather than saturates: this exact
        // `end` (u32::MAX + 2) would cast down to `1`, which is `<=`
        // right.len() and would wrongly slip past a guard phrased as
        // `*end as usize > right_len`. Comparing in `i64` space (widening
        // `right.len()` up instead of narrowing `end` down) rejects it on
        // every target, including this 64-bit test host.
        let left = array![1_i64];
        let right = array![1_i64];
        let starts = array![0_i64];
        let ends = array![(u32::MAX as i64) + 2];
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn end_equal_to_right_len_is_a_valid_inclusive_bound() {
        // every element in [0, right.len()) is < left_value, so the
        // "one past the last strictly-less element" answer is right.len()
        // itself -- this must not be confused with the out-of-bounds
        // rejection above.
        let left = array![5_i64];
        let right = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let ends = array![3_i64]; // == right.len()
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![3]);
    }

    #[test]
    fn sentinel_starts_and_ends_return_minus_one() {
        let left = array![1_i64, 2, 3];
        let right = array![10_i64, 20, 30];
        let starts = array![-1_i64, 0, 2];
        let ends = array![3_i64, -1, 1]; // last: start(2) >= end(1)
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1, -1, -1]);
    }

    #[test]
    fn restricted_subrange_only_searches_within_start_end() {
        // left_value=6: within [2, 5), right[2..5] = [2, 5, 9]; both 2 and
        // 5 are < 6, so the answer is one past the last of them (index 4).
        let left = array![6_i64];
        let right = array![9_i64, 9, 2, 5, 9];
        let starts = array![2_i64];
        let ends = array![5_i64];
        let got = binary_search_gt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![4]);
    }
}
