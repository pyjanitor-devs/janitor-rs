use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::cmp::Ordering;

use crate::aggs::adaptive::should_use_running_aggregation;
use crate::aggs::checked_index;
use crate::aggs::{ensure_equal_lengths_core, ensure_nonempty_core};

/// For every `starts[i]`, find the position (not the value) of the
/// largest element in `arr[starts[i]..]`, skipping positions flagged
/// `true` in `booleans` (a null mask). Returns `-1` when the range is
/// empty or invalid (`starts[i] < 0` or `starts[i] >= arr.len()`) or every
/// candidate is null. An empty `arr` is rejected with
/// `Err("arr cannot be empty")`.
///
/// Null-mask contract: `booleans[nn] == true` is the source of truth for a
/// missing value. For floating-point inputs, pyjanitor marks `NaN` entries in
/// this mask before calling Rust; the kernel does not infer nullness from the
/// value itself. Direct callers must preserve the same invariant.
///
/// # Arguments
///
/// * `arr` - Values to inspect.
/// * `starts` - Inclusive suffix boundaries, one per output.
/// * `booleans` - Null mask aligned with `arr`; `true` values are skipped.
///
/// ELI5 (the guard): unlike `sum`, which can start a running total at `0`
/// with no data read, `max` needs an actual array element to compare
/// against first. Reading `arr[start_]` unconditionally before checking
/// it's a valid index is exactly the bug this guard closes -- see issue
/// #27.
pub fn max_start_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Result<Array1<i64>, String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<i64>::from_elem(starts.len(), -1);
    let end_ = arr.len();

    let mut total_width = 0_usize;
    for start in starts.iter() {
        if let Ok(start_) = usize::try_from(*start) {
            total_width = total_width.saturating_add(end_.saturating_sub(start_));
        }
    }
    if should_use_running_aggregation(starts.len(), total_width, end_) {
        let mut suffix = vec![-1_i64; end_];
        let mut winner = -1_i64;
        for nn in (0..end_).rev() {
            if !booleans[nn]
                && (winner == -1
                    || arr[winner as usize].partial_cmp(&arr[nn]) != Some(Ordering::Greater))
            {
                winner = nn as i64;
            }
            suffix[nn] = winner;
        }
        for (pos, start) in starts.iter().enumerate() {
            if let Some(start_) = checked_index(*start, end_) {
                result[pos] = suffix[start_];
            }
        }
        return Ok(result);
    }

    for (pos, start) in starts.iter().enumerate() {
        let Some(start_) = checked_index(*start, end_) else {
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

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        /// Return the positions of the maximum non-null values in each
        /// suffix. `arr` is the value array, `starts` contains inclusive
        /// boundaries, and `booleans` marks null values to skip. Invalid or
        /// all-null suffixes return `-1`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Values to inspect.
        /// * `starts` - Inclusive suffix boundaries.
        /// * `booleans` - Null mask aligned with `arr`.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            let result = max_start_core(arr.as_array(), starts.as_array(), booleans.as_array());
            Ok(result
                .map_err(pyo3::exceptions::PyValueError::new_err)?
                .into_pyarray(py))
        }
    };
}

generic_compute!(compute_max_start_int64, i64);
generic_compute!(compute_max_start_int32, i32);
generic_compute!(compute_max_start_int16, i16);
generic_compute!(compute_max_start_int8, i8);
generic_compute!(compute_max_start_uint64, u64);
generic_compute!(compute_max_start_uint32, u32);
generic_compute!(compute_max_start_uint16, u16);
generic_compute!(compute_max_start_uint8, u8);
generic_compute!(compute_max_start_f64, f64);
generic_compute!(compute_max_start_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array_is_rejected() {
        let arr = Array1::<i64>::zeros(0);
        let starts = array![0_i64];
        let booleans = Array1::<bool>::default(0);
        let error = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap_err();
        assert_eq!(error, "arr cannot be empty");
    }

    #[test]
    fn start_equal_to_len_returns_minus_one_not_a_panic() {
        // The exact reproduction from issue #27.
        let arr = array![1_i64, 2, 3];
        let starts = array![3_i64]; // == arr.len()
        let booleans = array![false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_start_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![-1_i64];
        let booleans = array![false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_in_suffix() {
        let arr = array![5_i64, 1, 9, 2, 3];
        let starts = array![1_i64]; // suffix [1, 9, 2, 3]
        let booleans = array![false, false, false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![2]); // position of value 9
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![3_i64, 2, 1];
        let starts = array![0_i64];
        let booleans = array![true, false, false]; // largest (3) is null
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![1]); // position of value 2
    }

    #[test]
    fn all_null_range_returns_minus_one() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let booleans = array![true, true, true];
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn broad_suffix_batch_uses_running_winners() {
        let arr = array![5_i64, 1, 4, 2, 3, 0];
        let starts = array![0_i64, 1, 2, 3, 4, 5];
        let booleans = array![false, false, false, false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0, 2, 2, 4, 4, 5]);
    }

    #[test]
    fn adaptive_suffix_skips_nan_values_marked_null() {
        let arr = array![1.0_f64, f64::NAN];
        let starts = array![0_i64, 0, 0, 0];
        let booleans = array![false, true];
        let got = max_start_core(arr.view(), starts.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0, 0, 0, 0]);
    }
}
