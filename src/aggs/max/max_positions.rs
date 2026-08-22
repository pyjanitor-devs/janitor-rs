use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, checked_range};

/// For every `(starts[i], ends[i])`, find the position (not the value) of
/// the largest `arr[positions[nn]]` over `nn` in `[starts[i], ends[i])`,
/// skipping `nn` where `positions[nn] == -1` (no candidate at that slot)
/// or where the candidate's own position is null. Returns `-1` when the
/// slot range is invalid, `arr` is empty, or every candidate is skipped.
///
/// ELI5 (the guard): validate both the slot range and every indirect
/// position before indexing; signed sentinels must never become huge
/// `usize` values. See issue #27.
pub fn max_positions_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::from_elem(starts.len(), -1);
    let zipped = izip!(starts.into_iter(), ends.into_iter());
    for (pos, (start, end)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        if arr.is_empty() {
            continue;
        }
        let mut base: i64 = -1;
        let mut base_val = arr[0];
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], arr.len()) else {
                continue;
            };
            if booleans[indexer_] {
                continue;
            }
            let current = arr[indexer_];
            if (base == -1) || (current > base_val) {
                base_val = current;
                base = indexer_ as i64;
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
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = max_positions_core(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                positions.as_array(),
                booleans.as_array(),
            );
            result.into_pyarray(py)
        }
    };
}

// ELI5: `$type` above only picks the dtype of the *input* array (`arr`) --
// the result is always `i64` because this function returns the *position*
// of the max element, not its value, so positions stay `i64` regardless of
// what dtype the values themselves are. `$type` must still match the numpy
// dtype the function's name promises, or pyo3 rejects the array at the
// Python boundary: `compute_max_positions_int8` once had `i64` here (a
// leftover copy-paste from a wider sibling) even though its name promises
// `i8` input -- see issue #30.
generic_compute!(compute_max_positions_int64, i64);
generic_compute!(compute_max_positions_int32, i32);
generic_compute!(compute_max_positions_int16, i16);
generic_compute!(compute_max_positions_int8, i8); // fixed: was `i64`, see issue #30
generic_compute!(compute_max_positions_uint64, u64);
generic_compute!(compute_max_positions_uint32, u32);
generic_compute!(compute_max_positions_uint16, u16);
generic_compute!(compute_max_positions_uint8, u8);
generic_compute!(compute_max_positions_f64, f64);
generic_compute!(compute_max_positions_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    type Int8PositionsFn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, i8>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
    ) -> Bound<'py, PyArray1<i64>>;

    #[test]
    fn int8_wrapper_accepts_an_int8_array() {
        // ELI5: the typed slot only accepts a wrapper whose `arr` is really
        // `i8`; changing the macro argument back to `i64` breaks compilation.
        let _wrapper: Int8PositionsFn = compute_max_positions_int8;
    }

    #[test]
    fn empty_array_returns_minus_one_not_a_panic() {
        let arr: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let positions: Array1<i64> = array![];
        let booleans: Array1<bool> = array![];
        let got = max_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_via_indirection() {
        let arr = array![10_i64, 30, 20];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, 1, 2];
        let booleans = array![false, false, false];
        let got = max_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]); // arr[1] == 30 is largest
    }

    #[test]
    fn skips_negative_one_position_sentinel() {
        let arr = array![10_i64, 30, 20];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, -1, 2]; // slot 1 has no candidate
        let booleans = array![false, false, false];
        let got = max_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2]); // arr[2] == 20 is largest of the remaining
    }
}
