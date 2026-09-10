use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp::Ordering;

use crate::aggs::adaptive::{should_use_segment_tree, MAX_DIRECT_QUERY_COUNT};
use crate::aggs::{checked_range, ensure_equal_lengths_core, ensure_nonempty_core};

/// For every `(starts[i], ends[i])`, find the position (not the value) of
/// the largest element in `arr[starts[i]..ends[i]]`, skipping positions
/// flagged `true` in `booleans`. Returns `-1` for an empty or inverted
/// range (`start < 0`, `end < 0`, `start >= end`, or `end > arr.len()`),
/// or when every candidate is null.
///
/// ELI5 (the guard): same reasoning as `max_start_core` -- `max` needs a
/// real array element to seed its comparison, so the range must be
/// checked *before* that seed read, not after. See issue #27.
///
/// Null-mask contract: `booleans[nn] == true` is the source of truth for a
/// missing value. For floating-point inputs, pyjanitor marks `NaN` entries in
/// this mask before calling Rust; the kernel does not infer nullness from the
/// value itself. Direct callers must preserve the same invariant.
///
/// ELI5: the value array may still contain the empty box's old contents, but
/// the boolean mask puts a red X on that box. The tree ignores every box with
/// a red X before comparing winners.
///
/// Input contract: `arr`, `starts`, and `ends` must be non-empty, and
/// `starts` and `ends` must have equal lengths. The Python wrapper raises
/// `ValueError` when this contract is violated.
pub fn max_start_end_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Result<Array1<i64>, String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<i64>::from_elem(starts.len(), -1);

    // ELI5: a few questions are cheaper to answer by walking their ranges.
    // For larger batches, first estimate the total direct-scan work; build a
    // tree only when its one-time construction and logarithmic queries should
    // cost less than rereading all requested ranges.
    let use_segment_tree = if starts.len() <= MAX_DIRECT_QUERY_COUNT {
        false
    } else {
        let mut total_width = 0_usize;
        for (start, end) in starts.iter().zip(ends.iter()) {
            if let Some((start_, end_)) = checked_range(*start, *end, arr.len()) {
                total_width = total_width.saturating_add(end_ - start_);
            }
        }
        should_use_segment_tree(starts.len(), total_width, arr.len())
    };

    if use_segment_tree {
        // ELI5: each tree node remembers the largest non-null item in its
        // block. Overlapping ranges then reuse those block winners.
        // `tree_size` is exactly the number of input leaves. The half-open
        // iterative walk works for non-power-of-two lengths, so padding is
        // unnecessary; checked ranges keep every leaf access below 2*n.
        let tree_size = arr.len();
        // The vectors use the conventional 1-based heap layout: leaves live
        // at `[tree_size, 2 * tree_size)`, internal nodes at
        // `[1, tree_size)`, and slot 0 is intentionally unused. The internal
        // value slots are overwritten by the bottom-up build immediately
        // below, but `Vec<T>` still requires every element to be initialized
        // before it can be indexed. `T` is only `Copy`, not `Default`, so
        // `arr[0]` is the only generally available safe initializer. The
        // corresponding positions are initialized to `-1` because that is
        // the real no-candidate sentinel used for null leaves and empty tree
        // branches. Avoiding this initialization with `Option<T>` would add
        // branching to every tree access; using `MaybeUninit<T>` would add
        // unsafe invariants for a one-time build allocation. The small
        // initialization cost is therefore deliberate and keeps the query
        // path simple and safe.
        let mut values = vec![arr[0]; tree_size * 2];
        let mut positions = vec![-1_i64; tree_size * 2];
        for nn in 0..arr.len() {
            if !booleans[nn] {
                values[tree_size + nn] = arr[nn];
                positions[tree_size + nn] = nn as i64;
            }
        }
        for node in (1..tree_size).rev() {
            (values[node], positions[node]) = max_node(
                values[node * 2],
                positions[node * 2],
                values[node * 2 + 1],
                positions[node * 2 + 1],
            );
        }

        for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
            let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
                continue;
            };
            let mut left = start_ + tree_size;
            let mut right = end_ + tree_size;
            let mut left_best = (arr[0], -1_i64);
            let mut right_best = (arr[0], -1_i64);
            while left < right {
                if left % 2 == 1 {
                    left_best = max_node(left_best.0, left_best.1, values[left], positions[left]);
                    left += 1;
                }
                if right % 2 == 1 {
                    right -= 1;
                    right_best =
                        max_node(values[right], positions[right], right_best.0, right_best.1);
                }
                left /= 2;
                right /= 2;
            }
            result[pos] = max_node(left_best.0, left_best.1, right_best.0, right_best.1).1;
        }
        return Ok(result);
    }

    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut base: i64 = -1;
        let mut base_val = arr[start_];
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            let current = arr[nn];
            // ELI5: `base == -1` does double duty -- it's both "no candidate
            // accepted yet" during the scan and the final "no match"
            // sentinel if nothing ever qualifies. The strict `>` means an
            // exact tie keeps the earliest position found, not the latest.
            if (base == -1) || (current > base_val) {
                base_val = current;
                base = nn as i64;
            }
        }
        result[pos] = base;
    }
    Ok(result)
}

fn max_node<T: PartialOrd + Copy>(
    left_value: T,
    left_position: i64,
    right_value: T,
    right_position: i64,
) -> (T, i64) {
    // ELI5: a position of `-1` means that side of the tree has no valid
    // ticket, so the other side wins automatically. When both sides have a
    // ticket, equal values keep the smaller position—the first occurrence.
    if left_position == -1 {
        (right_value, right_position)
    } else if right_position == -1 {
        (left_value, left_position)
    } else {
        // `right_value` is compared against `left_value` because this
        // function returns the larger value. On an equal-value tie, the
        // smaller original array position wins; this preserves the same
        // earliest-position contract as the direct scan, regardless of how
        // the segment tree grouped the leaves.
        match right_value.partial_cmp(&left_value) {
            Some(Ordering::Greater) => {
                // The right child contains the larger value, so propagate
                // its value and its original position to the parent node.
                (right_value, right_position)
            }
            Some(Ordering::Equal) if right_position < left_position => {
                // Values are equal, but the right child refers to an earlier
                // input position. Keep it so tree results match a left-to-
                // right direct scan's tie-breaking behavior.
                (right_value, right_position)
            }
            Some(Ordering::Equal) | Some(Ordering::Less) => {
                // The left child is larger, or it wins the equal-value tie.
                (left_value, left_position)
            }
            None => {
                // ELI5: `partial_cmp` returns no answer for a NaN. Check
                // each value against itself to identify that invalid ticket;
                // discard it when the other side is a real candidate instead
                // of letting it poison the whole parent subtree. If both
                // sides are incomparable, retain the left one deterministically.
                // For IEEE floating-point values, a NaN is not comparable
                // even with itself, while ordinary finite values and
                // infinities compare equal to themselves. This gives us a
                // generic way to recognize the invalid floating-point case
                // without adding a floating-point-only bound to this kernel.
                let left_invalid = left_value.partial_cmp(&left_value).is_none();
                let right_invalid = right_value.partial_cmp(&right_value).is_none();
                match (left_invalid, right_invalid) {
                    (true, false) => {
                        // The left value is invalid, so the real right
                        // candidate must survive into the parent node.
                        (right_value, right_position)
                    }
                    (false, true) => {
                        // The right value is invalid, so retain the valid
                        // left candidate.
                        (left_value, left_position)
                    }
                    _ => {
                        // Both values are incomparable (or the type has an
                        // unusual partial-order implementation). Keep the
                        // left candidate deterministically; there is no
                        // ordering information with which to prefer either.
                        (left_value, left_position)
                    }
                }
            }
        }
    }
}

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        /// Finds the position of the largest value in each half-open
        /// `arr[start..end]` range. `arr`, `starts`, and `ends` must be
        /// non-empty; invalid ranges return `-1`.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            let starts = starts.as_array();
            let ends = ends.as_array();
            let result = max_start_end_core(arr.as_array(), starts, ends, booleans.as_array())
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute!(compute_max_start_end_int64, i64);
generic_compute!(compute_max_start_end_int32, i32);
generic_compute!(compute_max_start_end_int16, i16);
generic_compute!(compute_max_start_end_int8, i8);
generic_compute!(compute_max_start_end_uint64, u64);
generic_compute!(compute_max_start_end_uint32, u32);
generic_compute!(compute_max_start_end_uint16, u16);
generic_compute!(compute_max_start_end_uint8, u8);
generic_compute!(compute_max_start_end_f64, f64);
generic_compute!(compute_max_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn start_equal_to_end_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![2_i64];
        let ends = array![2_i64];
        let booleans = array![false, false, false];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn inverted_range_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![2_i64];
        let ends = array![0_i64];
        let booleans = array![false, false, false];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_start_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![-1_i64];
        let ends = array![2_i64];
        let booleans = array![false, false, false];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_in_interior_slice() {
        let arr = array![5_i64, 1, 9, 2, 3];
        let starts = array![1_i64];
        let ends = array![4_i64]; // slice [1, 9, 2]
        let booleans = array![false, false, false, false, false];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![2]); // position of value 9
    }

    #[test]
    fn segment_tree_preserves_maximum_positions_and_nulls() {
        let arr = array![5_i64, 1, 9, 2, 3];
        let starts = array![0_i64, 0, 1, 0];
        let ends = array![5_i64, 5, 5, 4];
        let booleans = array![false, false, false, false, false];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![2, 2, 2, 2]);

        let booleans = array![true, true, true, true, true];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1, -1, -1, -1]);
    }

    #[test]
    fn validation_checks_nonempty_before_parallel_lengths() {
        let arr = array![1_i64];
        let starts = array![0_i64];
        let ends = array![1_i64, 1];
        let ends_short = array![1_i64];
        let booleans = array![false];
        let error = max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view())
            .unwrap_err();
        assert_eq!(
            error,
            "starts and ends must have equal lengths; got 1 and 2"
        );

        let empty_arr = Array1::<i64>::zeros(0);
        let error = max_start_end_core(
            empty_arr.view(),
            starts.view(),
            ends_short.view(),
            Array1::<bool>::from_vec(vec![]).view(),
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");

        let error = max_start_end_core(
            arr.view(),
            Array1::<i64>::zeros(0).view(),
            ends_short.view(),
            booleans.view(),
        )
        .unwrap_err();
        assert_eq!(error, "starts cannot be empty");

        let error = max_start_end_core(
            arr.view(),
            starts.view(),
            Array1::<i64>::zeros(0).view(),
            booleans.view(),
        )
        .unwrap_err();
        assert_eq!(error, "ends cannot be empty");
    }

    #[test]
    fn segment_tree_keeps_earliest_position_for_maximum_ties() {
        // Sixteen broad queries over a length-17 array cross the segment-tree
        // threshold while keeping several equal maxima in every range.
        let arr = Array1::from_iter(std::iter::once(4_i64).chain(std::iter::repeat_n(9, 16)));
        let starts = Array1::from_elem(16, 1_i64);
        let ends = Array1::from_elem(16, 17_i64);
        let booleans = Array1::from_elem(17, false);
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, Array1::from_elem(16, 1_i64));
    }

    #[test]
    fn segment_tree_does_not_let_unmasked_nan_poison_maximum() {
        // Nine full-width queries over eight values cross the real tree
        // cutoff: total width 72 is greater than build/query cost 64.
        let arr = array![1.0_f64, 0.5, 0.4, 0.3, f64::NAN, 2.0, 6.0, 0.2];
        let starts = Array1::from_elem(9, 0_i64);
        let ends = Array1::from_elem(9, 8_i64);
        let booleans = Array1::from_elem(8, false);
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, Array1::from_elem(9, 6_i64));
    }

    #[test]
    fn adaptive_tree_skips_nan_values_marked_null() {
        let arr = array![1.0_f64, f64::NAN, 2.0];
        let starts = array![0_i64, 0, 0, 0];
        let ends = array![3_i64, 3, 3, 3];
        let booleans = array![false, true, false];
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![2, 2, 2, 2]);
    }

    #[test]
    fn segment_tree_handles_non_power_of_two_length() {
        let arr = numpy::ndarray::Array1::from_iter(0_i64..17);
        let starts = numpy::ndarray::Array1::from_elem(16, 0_i64);
        let ends = numpy::ndarray::Array1::from_elem(16, 17_i64);
        let booleans = numpy::ndarray::Array1::from_elem(17, false);
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, numpy::ndarray::Array1::from_elem(16, 16_i64));
    }
}
