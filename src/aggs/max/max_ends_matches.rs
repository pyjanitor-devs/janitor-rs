use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `ends[i]`, find the position (not the value) of the
/// largest element in `arr[..ends[i]]` among positions the caller has
/// flagged live in `matches` (a flat tape covering every row's candidate
/// range back to back). Returns `-1` when `arr` is empty (nothing to
/// seed the running comparison from) or every candidate is skipped/null.
///
/// ELI5 (the guard): the mirror image of `max_start_match_core`'s guard
/// -- `arr` itself, not the range, is what can be invalid here, since the
/// seed read is the fixed `arr[0]`. See issue #27.
pub fn max_end_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(ends.len());
    let mut n: usize = 0;
    let zipped = ends.into_iter().zip(counts);
    for (pos, (end, count)) in zipped.enumerate() {
        let end_ = *end as usize;
        if arr.is_empty() {
            n += end_;
            result[pos] = -1;
            continue;
        }
        let mut base: i64 = -1;
        let mut base_val = arr[0];
        if *count == 0 {
            n += end_;
            continue;
        }
        for nn in 0..end_ {
            if matches[n] == 0 || booleans[nn] {
                n += 1;
                continue;
            }
            let current = arr[nn];
            if (base == -1) || (current > base_val) {
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
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = max_end_match_core(
                arr.as_array(),
                ends.as_array(),
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
            );
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_max_end_match_int64, i64);
generic_compute!(compute_max_end_match_int32, i32);
generic_compute!(compute_max_end_match_int16, i16);
generic_compute!(compute_max_end_match_int8, i8);
generic_compute!(compute_max_end_match_uint64, u64);
generic_compute!(compute_max_end_match_uint32, u32);
generic_compute!(compute_max_end_match_uint16, u16);
generic_compute!(compute_max_end_match_uint8, u8);
generic_compute!(compute_max_end_match_f64, f64);
generic_compute!(compute_max_end_match_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array_returns_minus_one_not_a_panic() {
        let arr: Array1<i64> = array![];
        let ends = array![0_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans: Array1<bool> = array![];
        let got = max_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn finds_position_of_largest_among_matched_positions() {
        let arr = array![5_i64, 9, 4];
        let ends = array![3_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1]; // position 1 (value 9) excluded
        let booleans = array![false, false, false];
        let got = max_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0]); // position of value 5 (5 vs 4, 9 excluded)
    }

    #[test]
    fn zero_count_does_not_panic() {
        let arr = array![1_i64, 2, 3];
        let ends = array![3_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![0, 0, 0];
        let booleans = array![false, false, false];
        let got = max_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0]);
    }
}
