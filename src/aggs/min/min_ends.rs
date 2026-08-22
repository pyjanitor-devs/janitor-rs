use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `ends[i]`, find the position (not the value) of the smallest
/// element in `arr[..ends[i]]`, skipping positions flagged `true` in
/// `booleans` (a null mask). Returns `-1` when `arr` is empty (there's
/// nothing to seed the running comparison from) or every candidate is
/// null.
///
/// ELI5 (the guard): the mirror image of `min_start_core`'s guard --
/// instead of the *requested range* being invalid, it's that `arr` itself
/// has nothing in it, so the unconditional seed read `arr[0]` has no
/// index `0` to read. See issue #27.
pub fn min_end_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(ends.len());
    for (pos, end) in ends.indexed_iter() {
        if arr.is_empty() {
            result[pos] = -1;
            continue;
        }
        let mut base: i64 = -1;
        let end_ = *end as usize;
        let mut base_val = arr[0];
        for nn in 0..end_ {
            if booleans[nn] {
                continue;
            }
            let current = arr[nn];
            if (base == -1) || (current < base_val) {
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
            let result = min_end_core(arr.as_array(), ends.as_array(), booleans.as_array());
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_min_end_int64, i64);
generic_compute!(compute_min_end_int32, i32);
generic_compute!(compute_min_end_int16, i16);
generic_compute!(compute_min_end_int8, i8);
generic_compute!(compute_min_end_uint64, u64);
generic_compute!(compute_min_end_uint32, u32);
generic_compute!(compute_min_end_uint16, u16);
generic_compute!(compute_min_end_uint8, u8);
generic_compute!(compute_min_end_f64, f64);
generic_compute!(compute_min_end_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array_returns_minus_one_not_a_panic() {
        let arr: Array1<i64> = array![];
        let ends = array![0_i64];
        let booleans: Array1<bool> = array![];
        let got = min_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_smallest_in_prefix() {
        let arr = array![3_i64, 1, 4, 2, 5];
        let ends = array![3_i64]; // prefix [3, 1, 4]
        let booleans = array![false, false, false, false, false];
        let got = min_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]); // position of value 1
    }

    #[test]
    fn end_at_zero_returns_minus_one() {
        let arr = array![1_i64, 2, 3];
        let ends = array![0_i64];
        let booleans = array![false, false, false];
        let got = min_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![1_i64, 2, 3];
        let ends = array![3_i64];
        let booleans = array![true, false, false]; // smallest (1) is null
        let got = min_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]); // position of value 2
    }
}
