use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `(starts[i], ends[i])`, sum `arr[starts[i]..ends[i]]`,
/// skipping any position flagged `true` in `booleans` (a null mask).
///
/// ELI5: an arbitrary `[start, end)` slice instead of "to the end" or
/// "from the beginning" -- same null-skip/overflow-wrap contract as
/// `sum_start_core`. An inverted or empty range (`start >= end`, checked
/// in `i64` space before either bound is cast to `usize`) contributes `0`.
///
/// `start`/`end` of `-1` (the crate's sentinel for "invalid/no match")
/// also contributes `0`, rather than being cast to `usize::MAX` and
/// walked off the end of `arr`.
///
/// ELI5 (why the check has to happen *before* the cast): `-1 as usize`
/// wraps around to the *largest* possible `usize` instead of staying
/// negative, so it's bigger than any real `start`. A `start_ >= end_`
/// check done *after* casting would see a huge `end_` and conclude the
/// range is fine, when the original `i64` value actually meant "no
/// match" -- checking `-1` explicitly, before the cast, is the only way
/// to catch it.
pub fn sum_start_end_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    sum_start_end_core_with_cast(arr, starts, ends, booleans, |value| value)
}

fn sum_start_end_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::zeros(starts.len());
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
        if *start == -1 || *end == -1 || *start >= *end {
            continue; // result[pos] is already 0
        }
        let mut total: i64 = 0;
        let start_ = *start as usize;
        let end_ = *end as usize;
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            total = total.wrapping_add(to_i64(arr[nn]));
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
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = sum_start_end_core_with_cast(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            result.into_pyarray(py)
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
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(starts.len());
            let zipped = starts.into_iter().zip(ends);
            for (pos, (start, end)) in zipped.enumerate() {
                let mut total: f64 = 0.0;
                let mut compensation: f64 = 0.0;
                let start_ = *start as usize;
                let end_ = *end as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current: f64 = arr[nn] as f64;
                    let difference = current - compensation;
                    let increment = total + difference;
                    compensation = (increment - total) - difference;
                    total = increment;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array() {
        let arr: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let booleans: Array1<bool> = array![];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn full_array_range() {
        let arr = array![1_i64, 2, 3, 4];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let booleans = array![false, false, false, false];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![10]);
    }

    #[test]
    fn arbitrary_interior_slice() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![1_i64];
        let ends = array![4_i64]; // [2, 3, 4]
        let booleans = array![false, false, false, false, false];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![9]);
    }

    #[test]
    fn inverted_range_is_zero() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![3_i64];
        let ends = array![1_i64]; // start > end
        let booleans = array![false, false, false, false, false];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn equal_start_and_end_is_zero() {
        let arr = array![1_i64, 2, 3];
        let starts = array![1_i64];
        let ends = array![1_i64];
        let booleans = array![false, false, false];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
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
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn sentinel_start_is_zero_not_a_panic() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![-1_i64];
        let ends = array![3_i64];
        let booleans = array![false, false, false, false, false];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![1_i64, 2, 3, 4];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let booleans = array![false, true, false, true];
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1 + 3]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let value = i64::MAX / 2;
        let arr = Array1::<i64>::from_elem(100, value);
        let starts = array![0_i64];
        let ends = array![100_i64];
        let booleans = Array1::<bool>::from_elem(100, false);
        let got = sum_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
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
        );
        assert_eq!(got, array![3]);
        assert_eq!(casts, 1);
    }
}
