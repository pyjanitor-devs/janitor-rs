use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `(starts[i], ends[i])`, sum `arr[starts[i]..ends[i]]`,
/// skipping any position flagged `true` in `booleans` (a null mask).
///
/// ELI5: an arbitrary `[start, end)` slice instead of "to the end" or
/// "from the beginning" -- same null-skip/overflow-wrap contract as
/// `sum_start_core`. An inverted or empty range (`start >= end`)
/// contributes `0` for free: Rust's `start_..end_` range simply produces
/// no items to iterate when `start_ >= end_`, no special-casing needed.
pub fn sum_start_end_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(starts.len());
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
        let mut total: i64 = 0;
        let start_ = *start as usize;
        let end_ = *end as usize;
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            total += arr[nn];
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
            let widened = arr.as_array().mapv(|v| v as i64);
            let result = sum_start_end_core(
                widened.view(),
                starts.as_array(),
                ends.as_array(),
                booleans.as_array(),
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
}
