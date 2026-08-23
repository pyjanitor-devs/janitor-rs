use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_index;
use crate::aggs::ensure_equal_lengths;

/// For every `starts[i]`, find the position (not the value) of the
/// smallest element in `arr[starts[i]..]` among positions the caller has
/// flagged live in `matches` (a flat tape covering every row's candidate
/// range back to back -- see `compare_start_end_core` for the tape
/// convention). Returns `-1` when the range is empty/invalid (including
/// a negative sentinel `start`), when the row has zero matches, or when
/// every candidate is skipped/null.
///
/// ELI5 (the guard): `min` needs a real array element to seed its
/// comparison, so an invalid `start` must be caught *before* that seed
/// read, not folded into the `count == 0` branch below it (which only
/// handles "no matches", not "no valid range" -- those are different
/// conditions). See issue #27.
pub fn min_start_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::from_elem(starts.len(), -1);
    let mut n: usize = 0;
    let end_: usize = arr.len();
    let zipped = starts.into_iter().zip(counts);
    for (pos, (start, count)) in zipped.enumerate() {
        let Some(start_) = checked_index(*start, end_) else {
            continue;
        };
        let mut base: i64 = -1;
        let mut base_val = arr[start_];
        if *count == 0 {
            let size = end_ - start_;
            n += size;
            continue;
        }
        for nn in start_..end_ {
            if matches[n] == 0 || booleans[nn] {
                n += 1;
                continue;
            }
            let current = arr[nn];
            // ELI5: `base == -1` does double duty -- it's both "no candidate
            // accepted yet" during the scan and the final "no match"
            // sentinel if nothing ever qualifies. The strict `<` means an
            // exact tie keeps the earliest position found, not the latest.
            if (base == -1) || (current < base_val) {
                base_val = current;
                base = nn as i64;
            }
            n += 1;
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
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let result = min_start_match_core(
                arr.as_array(),
                starts.as_array(),
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
            );
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute!(compute_min_start_match_int64, i64);
generic_compute!(compute_min_start_match_int32, i32);
generic_compute!(compute_min_start_match_int16, i16);
generic_compute!(compute_min_start_match_int8, i8);
generic_compute!(compute_min_start_match_uint64, u64);
generic_compute!(compute_min_start_match_uint32, u32);
generic_compute!(compute_min_start_match_uint16, u16);
generic_compute!(compute_min_start_match_uint8, u8);
generic_compute!(compute_min_start_match_f64, f64);
generic_compute!(compute_min_start_match_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn start_equal_to_len_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![3_i64]; // == arr.len()
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans = array![false, false, false];
        let got = min_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_start_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![-1_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans = array![false, false, false];
        let got = min_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_smallest_among_matched_positions() {
        let arr = array![5_i64, 1, 4];
        let starts = array![0_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1]; // position 1 (value 1) excluded
        let booleans = array![false, false, false];
        let got = min_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2]); // position of value 4 (5 vs 4, 1 excluded)
    }

    #[test]
    fn zero_count_returns_minus_one() {
        // A valid (non-empty) range with count == 0 must not panic on the
        // seed read `arr[start_]`, even though this row's own loop never
        // runs; it's a distinct code path from the start_ >= end_ guard.
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![0, 0, 0];
        let booleans = array![false, false, false];
        let got = min_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-1]);
    }
}
