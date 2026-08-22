use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `starts[i]`, find the position (not the value) of the
/// largest element in `arr[starts[i]..]`, skipping positions flagged
/// `true` in `booleans` (a null mask). Returns `-1` when the range is
/// empty (`starts[i] >= arr.len()`, including a `-1` sentinel cast to
/// `usize`, which wraps past `arr.len()`) or every candidate is null.
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
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(starts.len());
    let end_ = arr.len();
    for (pos, start) in starts.indexed_iter() {
        let start_ = *start as usize;
        if start_ >= end_ {
            result[pos] = -1;
            continue;
        }
        let mut base: i64 = -1;
        let mut base_val = arr[start_];
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            let current = arr[nn];
            if (base == -1) || (current > base_val) {
                base_val = current;
                base = nn as i64;
            }
        }
        result[pos] = base;
    }
    result
}

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = max_start_core(arr.as_array(), starts.as_array(), booleans.as_array());
            result.into_pyarray(py)
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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn start_equal_to_len_returns_minus_one_not_a_panic() {
        // The exact reproduction from issue #27.
        let arr = array![1_i64, 2, 3];
        let starts = array![3_i64]; // == arr.len()
        let booleans = array![false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_start_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![-1_i64];
        let booleans = array![false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_in_suffix() {
        let arr = array![5_i64, 1, 9, 2, 3];
        let starts = array![1_i64]; // suffix [1, 9, 2, 3]
        let booleans = array![false, false, false, false, false];
        let got = max_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![2]); // position of value 9
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![3_i64, 2, 1];
        let starts = array![0_i64];
        let booleans = array![true, false, false]; // largest (3) is null
        let got = max_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1]); // position of value 2
    }

    #[test]
    fn all_null_range_returns_minus_one() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let booleans = array![true, true, true];
        let got = max_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }
}
