use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Find, for every `left[i]`, the first position in `right[starts[i]..ends[i])`
/// whose value is strictly greater than `left[i]` -- the start of the match
/// region for a `<` join (`right` is assumed sorted ascending within each
/// `[start, end)` slice, which in particular means NaN-free -- see the
/// contiguous-fast-path note below for what that precondition rules out
/// and why it matters). Returns `-1` for an
/// invalid/empty range (`start` negative, `end` the `-1` sentinel, an
/// inverted `start`/`end` pair, or `end` beyond `right.len()`), when no
/// element in the range is greater than `left[i]`, or when the first such
/// element is equal to `left[i]` (defensive: cannot happen when `right` is
/// truly sorted, since the search converges on the first strictly-greater
/// element).
///
/// ELI5: binary search cuts the candidate range in half each step instead of
/// scanning it one item at a time, so it costs O(log width) per query
/// instead of O(width).
///
/// ELI5 (why `start < 0`, not just `start == -1`): a `usize` can't
/// represent a negative number, so casting one doesn't keep it negative --
/// it wraps around to a huge positive `usize` instead. Checking only for
/// the `-1` sentinel would miss e.g. `start=-3, end=-2`: both wrap to
/// huge-but-still-ordered `usize` values (so `start_ < end_` survives the
/// cast), and the loop walks `right` far out of bounds instead of
/// recognizing the row as invalid. Requiring `start >= 0` up front closes
/// that gap, and rejecting `end > right.len()` closes the separate
/// oversized-positive-`end` gap.
///
/// ELI5 (why `right.len()` is widened to `i64`, not `end` narrowed to
/// `usize`): on a 32-bit target, `usize` is only 32 bits wide, so `*end as
/// usize` *truncates* an oversized `i64` instead of saturating --
/// `end = usize::MAX as i64 + 2` (which still fits easily in `i64`) casts
/// down to `1`, silently passing an `end > right.len()` check phrased as
/// `*end as usize > right_len`. Casting `right.len()` up to `i64` instead
/// never loses information on any real target (an array can't hold
/// `i64::MAX` elements), so comparing `*end > right_len` in `i64` space is
/// correct everywhere, not just on 64-bit hosts.
pub fn binary_search_lt_core<T: PartialOrd + Copy>(
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
    let right_len = right.len() as i64;
    let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
    for (pos, (left_value, start, end)) in zipped.enumerate() {
        if *start < 0 || *end == -1 || *start >= *end || *end > right_len {
            result[pos] = -1;
            continue;
        }
        // ELI5: `partition_point` uses the same predicate, in the same
        // direction, as the manual `while` loop below -- but that alone
        // does *not* guarantee an identical result. `slice::partition_point`
        // is std's "branchless" binary search: it shrinks the search width
        // by a fixed `size / 2` every step regardless of the comparison
        // outcome, whereas the manual loop below shrinks by whatever the
        // comparison decides (`mid - min` or `max - mid`, generally
        // unequal). For a genuinely sorted, NaN-free `right` (this
        // function's documented precondition) both converge on the same
        // unique partition point regardless of which width-shrinking
        // strategy got them there. For a `right` that violates that
        // precondition -- most notably one containing NaN, which cannot
        // occupy a valid position in "sorted ascending" -- the two
        // strategies can probe different elements and land on genuinely
        // different (but never out-of-bounds) answers; see
        // `nan_in_right_does_not_panic_but_is_not_guaranteed_to_match_fallback`
        // below for a concrete case. This is why the fast path is worth
        // having at all: a `&[T]` slice both elides bounds checks a manual
        // `ArrayView1` index can't *and* unlocks this branchless algorithm,
        // which benchmarks ~2x faster than an equivalent manual loop over
        // the same slice at n=100,000 -- rewriting it as a manual loop to
        // guarantee fallback parity on malformed input would give up most
        // of that, for a case this function's contract already excludes.
        let min_idx = if let Some(slice) = right_slice {
            let rel = slice[*start as usize..*end as usize].partition_point(|v| *v <= *left_value);
            *start + rel as i64
        } else {
            let mut min_idx = *start;
            let mut max_idx = *end;
            while min_idx < max_idx {
                // to avoid overflow
                // adapted from numba's implementation
                let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
                let current_value = right[mid_idx as usize];
                if current_value <= *left_value {
                    min_idx = mid_idx + 1;
                } else {
                    max_idx = mid_idx;
                }
            }
            min_idx
        };
        if min_idx == *end {
            result[pos] = -1;
            continue;
        }
        let current_value = right[min_idx as usize];
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
            let result = binary_search_lt_core(
                left.as_array(),
                right.as_array(),
                starts.as_array(),
                ends.as_array(),
            );
            result.into_pyarray(py)
        }
    };
}

bin_search!(binary_search_lt_int64, i64);
bin_search!(binary_search_lt_int32, i32);
bin_search!(binary_search_lt_int16, i16);
bin_search!(binary_search_lt_int8, i8);
bin_search!(binary_search_lt_uint64, u64);
bin_search!(binary_search_lt_uint32, u32);
bin_search!(binary_search_lt_uint16, u16);
bin_search!(binary_search_lt_uint8, u8);
bin_search!(binary_search_lt_f64, f64);
bin_search!(binary_search_lt_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_lt_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s};

    #[test]
    fn strided_input_falls_back_to_manual_loop_and_still_finds_the_right_answer() {
        // Every other element is real data; the odd-indexed slots are junk
        // that a stride-2 view skips entirely. Slicing with a non-1 step
        // makes the view non-contiguous, so `.as_slice()` returns `None`
        // and the fallback loop runs instead of the fast path.
        let right_padded = array![1_i64, 999, 3, 999, 7, 999, 9, 999];
        let right = right_padded.slice(s![..;2]);
        assert!(
            right.as_slice().is_none(),
            "test setup bug: this view should be non-contiguous"
        );
        let left = array![5_i64];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right, starts.view(), ends.view());
        assert_eq!(got, array![2]); // first strictly-greater element (7) is at index 2
    }

    #[test]
    fn contiguous_and_strided_paths_agree_when_the_query_is_nan() {
        // `right` itself stays genuinely sorted ascending here (this
        // function's documented precondition) -- only the *query*
        // (`left`) is NaN. `*v <= NaN` is `false` for every element, a
        // valid (if degenerate) partition of the whole range into the
        // "false" side, so both the fast path (`slice::partition_point`)
        // and the fallback (`ArrayView1` manual bisection) still agree:
        // this is in-contract input, unlike a `right` that itself
        // contains NaN (see `nan_in_right_does_not_panic_but_parity_is_not_guaranteed`
        // below for that out-of-contract case, and this file's core doc
        // comment for why the two can genuinely differ there).
        let right_dense = array![1.0_f64, 2.0, 3.0, 7.0];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![1.0_f64, -1.0, 2.0, -1.0, 3.0, -1.0, 7.0, -1.0];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![f64::NAN];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let fast =
            binary_search_lt_core(left.view(), right_dense.view(), starts.view(), ends.view());
        let fallback =
            binary_search_lt_core(left.view(), right_strided, starts.view(), ends.view());
        assert_eq!(fast, fallback);
    }

    #[test]
    fn nan_in_right_does_not_panic_but_parity_is_not_guaranteed() {
        // Issue found in review of PR #54: a `right` that itself contains
        // NaN violates "sorted ascending" (NaN has no valid position in a
        // sort order), which is out of this function's documented
        // contract -- unlike the query-is-NaN case above, `slice::partition_point`
        // (branchless, shrinks the search width by a fixed `size / 2`
        // every step) and the manual fallback loop (shrinks by whatever
        // the comparison decides) can probe different elements and land
        // on genuinely different answers here, even though both use the
        // identical `<=` predicate. Concretely, right=[-1, NaN, 0, 1, 2,
        // 3, 4], left=0: the fast path returns 3, the fallback returns 1.
        // Both are "some in-bounds index," neither panics -- that's the
        // only guarantee this out-of-contract case gets.
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
            binary_search_lt_core(left.view(), right_dense.view(), starts.view(), ends.view());
        let fallback =
            binary_search_lt_core(left.view(), right_strided, starts.view(), ends.view());
        // Deliberately not asserting fast == fallback: that's exactly the
        // parity this out-of-contract input isn't guaranteed to have. Both
        // must still be valid (in-bounds-or-sentinel) results.
        for got in [fast[0], fallback[0]] {
            assert!(
                got == -1 || (0..=7).contains(&got),
                "expected -1 or an index in 0..=7, got {got}"
            );
        }
    }

    #[test]
    fn empty_right_range_returns_minus_one() {
        let left = array![5_i64];
        let right: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_starts_and_ends_return_minus_one() {
        let left = array![1_i64, 2, 3];
        let right = array![10_i64, 20, 30];
        let starts = array![-1_i64, 0, 2];
        let ends = array![3_i64, -1, 1]; // last: start(2) >= end(1)
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1, -1, -1]);
    }

    #[test]
    fn both_bounds_negative_but_start_less_than_end_returns_minus_one_not_a_panic() {
        // Issue #57: start=-3, end=-2 aren't the `-1` sentinel, and
        // -3 < -2 holds in i64 space, so a `start == -1`-only check would
        // let this row through, cast both bounds to huge-but-still-ordered
        // `usize` values, and index `right` far out of bounds.
        let left = array![5_i64];
        let right = array![1_i64];
        let starts = array![-3_i64];
        let ends = array![-2_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn non_sentinel_negative_start_returns_minus_one() {
        let left = array![3_i64];
        let right = array![1_i64, 2, 3];
        let starts = array![-2_i64];
        let ends = array![2_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
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
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
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
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn end_equal_to_right_len_is_a_valid_inclusive_bound() {
        let left = array![2_i64];
        let right = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let ends = array![3_i64]; // == right.len()
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![2]);
    }

    #[test]
    fn finds_first_strictly_greater_element() {
        // right is sorted ascending; left=5 -> first element > 5 is 7 at index 2
        let left = array![5_i64];
        let right = array![1_i64, 3, 7, 9];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![2]);
    }

    #[test]
    fn no_element_greater_than_left_returns_minus_one() {
        let left = array![100_i64];
        let right = array![1_i64, 3, 7, 9];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn duplicate_values_in_right_find_first_greater_past_the_run() {
        // duplicates of the target value must be skipped entirely
        let left = array![5_i64];
        let right = array![1_i64, 5, 5, 5, 9];
        let starts = array![0_i64];
        let ends = array![5_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![4]);
    }

    #[test]
    fn boundary_first_and_last_positions() {
        let right = array![10_i64, 20, 30];
        // left value below everything -> first position (0) is the answer
        let left_low = array![0_i64];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let got_low =
            binary_search_lt_core(left_low.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got_low, array![0]);

        // left value equal to the last element -> no strictly-greater element exists
        let left_high = array![30_i64];
        let got_high =
            binary_search_lt_core(left_high.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got_high, array![-1]);
    }

    #[test]
    fn generic_over_float_dtype() {
        let left = array![2.5_f64];
        let right = array![1.0_f64, 2.0, 3.0, 4.0];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![2]);
    }

    #[test]
    fn restricted_subrange_only_searches_within_start_end() {
        // value 2 exists in right, but the searched slice [2, 5) excludes it;
        // within that slice the first element > 2 is at index 3
        let left = array![2_i64];
        let right = array![2_i64, 2, 2, 5, 9];
        let starts = array![2_i64];
        let ends = array![5_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![3]);
    }
}
