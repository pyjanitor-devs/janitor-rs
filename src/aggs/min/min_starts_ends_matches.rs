use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

use crate::aggs::checked_range;
use crate::aggs::ensure_tape_width;

/// For every `(starts[i], ends[i])`, find the position (not the value) of
/// the smallest element in `arr[starts[i]..ends[i]]` among positions the
/// caller has flagged live in `matches` (a flat tape covering every row's
/// candidate range back to back). Returns `-1` for an empty or inverted
/// range, a zero match count, or when every candidate is skipped/null.
///
/// ELI5 (the guard): same reasoning as the other `_matches` guards --
/// `start >= end` must be checked before either the seed read `arr[start_]`
/// or the `end_ - start_` subtraction in the `count == 0` branch, not
/// folded into either of them. See issue #27.
pub fn min_start_end_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::from_elem(starts.len(), -1);
    let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
    let mut n: usize = 0;
    for (pos, (start, end, count)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut base: i64 = -1;
        if *count == 0 {
            let size = end_ - start_;
            n += size;
            continue;
        }
        let mut base_val = arr[start_];
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
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
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
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width: usize = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| {
                    checked_range(*s, *e, arr.as_array().len()).map(|(s_, e_)| e_ - s_)
                })
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let result = min_start_end_match_core(
                arr.as_array(),
                starts,
                ends,
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
            );
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute!(compute_min_start_end_match_int64, i64);
generic_compute!(compute_min_start_end_match_int32, i32);
generic_compute!(compute_min_start_end_match_int16, i16);
generic_compute!(compute_min_start_end_match_int8, i8);
generic_compute!(compute_min_start_end_match_uint64, u64);
generic_compute!(compute_min_start_end_match_uint32, u32);
generic_compute!(compute_min_start_end_match_uint16, u16);
generic_compute!(compute_min_start_end_match_uint8, u8);
generic_compute!(compute_min_start_end_match_f64, f64);
generic_compute!(compute_min_start_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_start_end_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn start_equal_to_end_returns_minus_one_not_a_panic() {
        let arr = array![1_i64, 2, 3];
        let starts = array![2_i64];
        let ends = array![2_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans = array![false, false, false];
        let got = min_start_end_match_core(
            arr.view(),
            starts.view(),
            ends.view(),
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
        let ends = array![2_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans = array![false, false, false];
        let got = min_start_end_match_core(
            arr.view(),
            starts.view(),
            ends.view(),
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
        let ends = array![3_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1]; // position 1 (value 1) excluded
        let booleans = array![false, false, false];
        let got = min_start_end_match_core(
            arr.view(),
            starts.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2]); // position of value 4 (5 vs 4, 1 excluded)
    }

    #[test]
    fn zero_count_returns_minus_one() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![0, 0, 0];
        let booleans = array![false, false, false];
        let got = min_start_end_match_core(
            arr.view(),
            starts.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-1]);
    }
}
