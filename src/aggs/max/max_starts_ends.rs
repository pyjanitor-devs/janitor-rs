use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

use crate::aggs::checked_range;

/// For every `(starts[i], ends[i])`, find the position (not the value) of
/// the largest element in `arr[starts[i]..ends[i]]`, skipping positions
/// flagged `true` in `booleans`. Returns `-1` for an empty or inverted
/// range (`start < 0`, `end < 0`, `start >= end`, or `end > arr.len()`),
/// or when every candidate is null.
///
/// ELI5 (the guard): same reasoning as `max_start_core` -- `max` needs a
/// real array element to seed its comparison, so the range must be
/// checked *before* that seed read, not after. See issue #27.
pub fn max_start_end_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::from_elem(starts.len(), -1);
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
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
    result
}

macro_rules! generic_compute {
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
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let result = max_start_end_core(arr.as_array(), starts, ends, booleans.as_array());
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
        let got = max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn inverted_range_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![2_i64];
        let ends = array![0_i64];
        let booleans = array![false, false, false];
        let got = max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_start_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![-1_i64];
        let ends = array![2_i64];
        let booleans = array![false, false, false];
        let got = max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_in_interior_slice() {
        let arr = array![5_i64, 1, 9, 2, 3];
        let starts = array![1_i64];
        let ends = array![4_i64]; // slice [1, 9, 2]
        let booleans = array![false, false, false, false, false];
        let got = max_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![2]); // position of value 9
    }
}
