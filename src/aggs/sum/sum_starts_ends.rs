use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_equal_lengths_core, ensure_nonempty_core,
};

/// For every `(starts[i], ends[i])`, sum `arr[starts[i]..ends[i]]`,
/// skipping any position flagged `true` in `booleans` (a null mask).
///
/// ELI5: an arbitrary `[start, end)` slice instead of "to the end" or
/// "from the beginning" -- same null-skip/overflow-wrap contract as
/// `sum_start_core`. `checked_range(start, end, arr.len())` rejects an
/// inverted/empty range, the `-1` "no match" sentinel (which would
/// otherwise wrap to `usize::MAX` when cast), *and* an `end` that's
/// simply too large for `arr` -- a plain `start == -1 || end == -1 ||
/// start >= end` check (this function's original guard) caught the first
/// two but not the third, so a valid-looking but oversized `end` still
/// walked `arr`/`booleans` out of bounds. Any rejected row contributes
/// `0`.
pub fn sum_start_end_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Result<Array1<i64>, String> {
    sum_start_end_core_with_cast(arr, starts, ends, booleans, |value| value)
}

fn sum_start_end_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Result<Array1<i64>, String>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<i64>::zeros(starts.len());
    let zipped = starts.into_iter().zip(ends);

    let use_prefix = if starts.len() <= 3 {
        false
    } else {
        let total_width = starts
            .iter()
            .zip(ends.iter())
            .fold(0_usize, |total, (start, end)| {
                let width = checked_range(*start, *end, arr.len())
                    .map_or(0, |(start_, end_)| end_ - start_);
                total.saturating_add(width)
            });
        total_width > arr.len().saturating_mul(3)
    };

    if use_prefix {
        // ELI5: one running prefix total turns every valid interval into two
        // lookups and a wrapped subtraction instead of another full scan.
        let mut prefix = vec![0_i64; arr.len() + 1];
        for nn in 0..arr.len() {
            prefix[nn + 1] = prefix[nn];
            if !booleans[nn] {
                prefix[nn + 1] = prefix[nn + 1].wrapping_add(to_i64(arr[nn]));
            }
        }
        for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
            if let Some((start_, end_)) = checked_range(*start, *end, arr.len()) {
                result[pos] = prefix[end_].wrapping_sub(prefix[start_]);
            }
        }
        return Ok(result);
    }

    for (pos, (start, end)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue; // result[pos] is already 0
        };
        let mut total: i64 = 0;
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            total = total.wrapping_add(to_i64(arr[nn]));
        }
        result[pos] = total;
    }
    Ok(result)
}

fn sum_start_end_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::zeros(starts.len());
    for (pos, (start, end)) in starts.into_iter().zip(ends).enumerate() {
        // ELI5: validate the range ticket once, before either dtype-specific
        // path turns its signed numbers into array positions. That keeps a
        // "no match" ticket worth zero for both integer and float columns.
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut total = 0.0;
        let mut compensation = 0.0;
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            let current = to_f64(arr[nn]);
            let difference = current - compensation;
            let increment = total + difference;
            compensation = (increment - total) - difference;
            total = increment;
        }
        result[pos] = total;
    }
    result
}

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
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
            let result = sum_start_end_core_with_cast(
                arr.as_array(),
                starts,
                ends,
                booleans.as_array(),
                |value| value as i64,
            );
            Ok(result
                .map_err(pyo3::exceptions::PyValueError::new_err)?
                .into_pyarray(py))
        }
    };
}

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>>
        // The macro will expand into the contents of this block.
        {
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let result = sum_start_end_float_core_with_cast(
                arr.as_array(),
                starts,
                ends,
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}
generic_compute_ints!(compute_sum_start_end_int64, i64);
generic_compute_ints!(compute_sum_start_end_int32, i32);
generic_compute_ints!(compute_sum_start_end_int16, i16);
generic_compute_ints!(compute_sum_start_end_int8, i8);
generic_compute_ints!(compute_sum_start_end_uint64, u64);
generic_compute_ints!(compute_sum_start_end_uint32, u32);
generic_compute_ints!(compute_sum_start_end_uint16, u16);
generic_compute_ints!(compute_sum_start_end_uint8, u8);
generic_compute_floats!(compute_sum_start_end_f32, f32);
generic_compute_floats!(compute_sum_start_end_f64, f64);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array_is_rejected() {
        let arr: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let booleans: Array1<bool> = array![];
        let error = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view())
            .unwrap_err();
        assert_eq!(error, "arr cannot be empty");
    }

    #[test]
    fn full_array_range() {
        let arr = array![1_i64, 2, 3, 4];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let booleans = array![false, false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![10]);
    }

    #[test]
    fn arbitrary_interior_slice() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![1_i64];
        let ends = array![4_i64]; // [2, 3, 4]
        let booleans = array![false, false, false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![9]);
    }

    #[test]
    fn inverted_range_is_zero() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![3_i64];
        let ends = array![1_i64]; // start > end
        let booleans = array![false, false, false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0]);
    }

    #[test]
    fn equal_start_and_end_is_zero() {
        let arr = array![1_i64, 2, 3];
        let starts = array![1_i64];
        let ends = array![1_i64];
        let booleans = array![false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0]);
    }

    #[test]
    fn sentinel_end_is_zero_not_a_panic() {
        // -1 is the crate's "invalid/no match" sentinel. Cast naively to
        // usize alone it becomes usize::MAX -- larger than any real
        // `start`, so a post-cast `start_ >= end_` check would miss it
        // and the loop would walk straight off the end of `arr`.
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![2_i64];
        let ends = array![-1_i64];
        let booleans = array![false, false, false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0]);
    }

    #[test]
    fn sentinel_start_is_zero_not_a_panic() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![-1_i64];
        let ends = array![3_i64];
        let booleans = array![false, false, false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0]);
    }

    #[test]
    fn oversized_non_sentinel_end_is_zero_not_a_panic() {
        // Found during an adversarial review of PR #37: the original guard
        // here (`start == -1 || end == -1 || start >= end`) only caught the
        // sentinel and inverted-range cases, not a plain positive `end`
        // that's simply larger than `arr.len()`. `checked_range` closes
        // that gap by validating the upper bound too.
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![0_i64];
        let ends = array![1000_i64];
        let booleans = array![false, false, false, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![0]);
    }

    #[test]
    fn float_sentinel_bounds_are_zero_not_a_panic() {
        let arr = array![1.0_f64, 2.0, 3.0];
        let starts = array![-1_i64, 0_i64];
        let ends = array![2_i64, -1_i64];
        let booleans = array![false, false, false];
        let got = sum_start_end_float_core_with_cast(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got, array![0.0, 0.0]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![1_i64, 2, 3, 4];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let booleans = array![false, true, false, true];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![1 + 3]);
    }

    #[test]
    fn repeated_broad_ranges_use_wrapping_prefix_differences() {
        let arr = array![1_i64, 2, 3, 4];
        let starts = array![0_i64, 0, 1, 2];
        let ends = array![4_i64, 3, 4, 4];
        let booleans = array![false, true, false, false];
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got, array![8, 4, 7, 7]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let value = i64::MAX / 2;
        let arr = Array1::<i64>::from_elem(100, value);
        let starts = array![0_i64];
        let ends = array![100_i64];
        let booleans = Array1::<bool>::from_elem(100, false);
        let got =
            sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view()).unwrap();
        assert_eq!(got[0], -100_i64);
    }

    #[test]
    fn casts_only_values_in_requested_interval() {
        let arr = array![1_i32, 2, 3, 4];
        let starts = array![2_i64];
        let ends = array![3_i64];
        let booleans = array![false, false, false, false];
        let mut casts = 0;
        let got = sum_start_end_core_with_cast(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| {
                casts += 1;
                value as i64
            },
        )
        .unwrap();
        assert_eq!(got, array![3]);
        assert_eq!(casts, 1);
    }
}
