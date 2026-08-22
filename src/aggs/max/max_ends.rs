use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_end;

/// For every `ends[i]`, find the position (not the value) of the largest
/// element in `arr[..ends[i]]`, skipping positions flagged `true` in
/// `booleans` (a null mask). Returns `-1` when `end` is negative or past
/// `arr.len()`, `arr` is empty, or every candidate is null.
///
/// ELI5 (the guard): validate the signed end before either casting it or
/// reading `arr[0]`; otherwise `-1` becomes a huge index and an empty array
/// has no seed value. See issue #27.
pub fn max_end_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::from_elem(ends.len(), -1);
    for (pos, end) in ends.indexed_iter() {
        let Some(end_) = checked_end(*end, arr.len()) else {
            continue;
        };
        if arr.is_empty() {
            continue;
        }
        let mut base: i64 = -1;
        let mut base_val = arr[0];
        for nn in 0..end_ {
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
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = max_end_core(arr.as_array(), ends.as_array(), booleans.as_array());
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_max_end_int64, i64);
generic_compute!(compute_max_end_int32, i32);
generic_compute!(compute_max_end_int16, i16);
generic_compute!(compute_max_end_int8, i8);
generic_compute!(compute_max_end_uint64, u64);
generic_compute!(compute_max_end_uint32, u32);
generic_compute!(compute_max_end_uint16, u16);
generic_compute!(compute_max_end_uint8, u8);
generic_compute!(compute_max_end_f64, f64);
generic_compute!(compute_max_end_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array_returns_minus_one_not_a_panic() {
        let arr: Array1<i64> = array![];
        let ends = array![0_i64];
        let booleans: Array1<bool> = array![];
        let got = max_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_in_prefix() {
        let arr = array![3_i64, 1, 9, 2, 5];
        let ends = array![3_i64]; // prefix [3, 1, 9]
        let booleans = array![false, false, false, false, false];
        let got = max_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![2]); // position of value 9
    }

    #[test]
    fn end_at_zero_returns_minus_one() {
        let arr = array![1_i64, 2, 3];
        let ends = array![0_i64];
        let booleans = array![false, false, false];
        let got = max_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![3_i64, 2, 1];
        let ends = array![3_i64];
        let booleans = array![true, false, false]; // largest (3) is null
        let got = max_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]); // position of value 2
    }
}
