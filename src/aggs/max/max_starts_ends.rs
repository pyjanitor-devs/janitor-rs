use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp::Ordering;

use crate::aggs::adaptive::should_use_running_aggregation;
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

    let mut total_width = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start_, end_)) = checked_range(*start, *end, arr.len()) {
            total_width = total_width.saturating_add(end_ - start_);
        }
    }

    if should_use_running_aggregation(starts.len(), total_width, arr.len()) {
        // ELI5: each tree node remembers the largest non-null item in its
        // block. Overlapping ranges then reuse those block winners.
        let tree_size = arr.len().next_power_of_two();
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
            let mut best = (arr[0], -1_i64);
            while left < right {
                if left % 2 == 1 {
                    best = max_node(best.0, best.1, values[left], positions[left]);
                    left += 1;
                }
                if right % 2 == 1 {
                    right -= 1;
                    best = max_node(best.0, best.1, values[right], positions[right]);
                }
                left /= 2;
                right /= 2;
            }
            result[pos] = best.1;
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
    if left_position == -1 {
        (right_value, right_position)
    } else if right_position == -1 {
        (left_value, left_position)
    } else {
        match right_value.partial_cmp(&left_value) {
            Some(Ordering::Greater) => (right_value, right_position),
            Some(Ordering::Equal) if right_position < left_position => {
                (right_value, right_position)
            }
            Some(Ordering::Equal) | Some(Ordering::Less) => (left_value, left_position),
            None => {
                // `booleans` must mark NaN/null values invalid before they
                // reach the tree. If a direct caller violates that contract,
                // retain the earlier candidate deterministically.
                (left_value, left_position)
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
        let arr = array![4_i64, 9, 9, 9, 9, 9, 9, 4];
        let starts = array![1_i64, 1, 1, 1];
        let ends = array![7_i64, 7, 7, 7];
        let booleans = Array1::from_elem(8, false);
        let got =
            max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![1, 1, 1, 1]);
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
}
