use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_range;
use crate::aggs::{ensure_equal_lengths, ensure_tape_width};

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`prod_end_match_core_with_cast`] at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn prod_end_match_core(
    arr: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    prod_end_match_core_with_cast(arr, ends, counts, matches, booleans, |value| value)
}

/// For every `ends[i]`, multiply `arr[..ends[i]]` (from the start of the
/// array), but only at positions the caller has already flagged as live
/// in `matches` (a *flat* tape shared by every row -- see
/// `prod_start_match_core`'s doc comment for the same shape).
/// `counts[i] == 0` short-circuits straight to the identity without
/// reading `matches` for that row's span. Returns `1` (the multiplicative
/// identity) for a row with nothing live, or one whose `end` is negative
/// or past `arr.len()`.
///
/// ELI5 (the guard, unlike `prod_start_match_core`): unlike a `start`,
/// which produces an *empty* range on its own when it wraps to a huge
/// `usize`, an out-of-range `end` here (`0..end_`) does not self-limit --
/// a negative `end` (e.g. the `-1` sentinel) cast to `usize` becomes
/// `usize::MAX`, and the tape-width sum below would try to add that in
/// directly, overflowing `usize` before the loop is even reached.
/// `checked_range(0, end, arr.len())` rejects that (and any `end >
/// arr.len()`) up front, for both the width sum and the main loop, so a
/// rejected row contributes zero to `n` on both sides and the tape
/// pointer never desyncs. See issue #63.
fn prod_end_match_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::from_elem(ends.len(), 1);
    let mut n: usize = 0;
    let zipped = ends.into_iter().zip(counts);
    for (pos, (end, count)) in zipped.enumerate() {
        let Some((_, end_)) = checked_range(0, *end, arr.len()) else {
            continue;
        };
        let mut total: i64 = 1;
        if *count == 0 {
            n += end_;
            continue;
        }
        for nn in 0..end_ {
            if matches[n] == 0 || booleans[nn] {
                n += 1;
                continue;
            }
            total = total.wrapping_mul(to_i64(arr[nn]));
            n += 1;
        }
        result[pos] = total;
    }
    result
}

fn prod_end_match_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::from_elem(ends.len(), 1.0);
    let mut n: usize = 0;
    let zipped = ends.into_iter().zip(counts);
    for (pos, (end, count)) in zipped.enumerate() {
        let Some((_, end_)) = checked_range(0, *end, arr.len()) else {
            continue;
        };
        let mut total: f64 = 1.0;
        if *count == 0 {
            n += end_;
            continue;
        }
        for nn in 0..end_ {
            if matches[n] == 0 || booleans[nn] {
                n += 1;
                continue;
            }
            total *= to_f64(arr[nn]);
            n += 1;
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
            ends: PyReadonlyArray1<'py, i64>,
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
            let ends = ends.as_array();
            let arr_len = arr.as_array().len();
            let expected_matches_width: usize = ends
                .iter()
                .filter_map(|e| checked_range(0, *e, arr_len).map(|(_, e_)| e_))
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let result = prod_end_match_core_with_cast(
                arr.as_array(),
                ends,
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let ends = ends.as_array();
            let arr_len = arr.as_array().len();
            let expected_matches_width: usize = ends
                .iter()
                .filter_map(|e| checked_range(0, *e, arr_len).map(|(_, e_)| e_))
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let result = prod_end_match_float_core_with_cast(
                arr.as_array(),
                ends,
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute_ints!(compute_prod_end_match_int64, i64);
generic_compute_ints!(compute_prod_end_match_int32, i32);
generic_compute_ints!(compute_prod_end_match_int16, i16);
generic_compute_ints!(compute_prod_end_match_int8, i8);
generic_compute_ints!(compute_prod_end_match_uint64, u64);
generic_compute_ints!(compute_prod_end_match_uint32, u32);
generic_compute_ints!(compute_prod_end_match_uint16, u16);
generic_compute_ints!(compute_prod_end_match_uint8, u8);

generic_compute_floats!(compute_prod_end_match_f64, f64);
generic_compute_floats!(compute_prod_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_f64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_match_int64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array() {
        let arr: Array1<i64> = array![];
        let ends = array![0_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans: Array1<bool> = array![];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn zero_count_short_circuits_to_identity() {
        let arr = array![2_i64, 3, 4];
        let ends = array![3_i64];
        let counts = array![0_i64];
        let matches = array![1_i8, 1, 1];
        let booleans = array![false, false, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn matches_mask_skips_positions_but_still_advances_the_tape() {
        let arr = array![2_i64, 3, 4];
        let ends = array![3_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1]; // middle position not live
        let booleans = array![false, false, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![2_i64, 3, 4];
        let ends = array![3_i64];
        let counts = array![3_i64];
        let matches = array![1_i8, 1, 1];
        let booleans = array![false, true, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn sentinel_end_is_identity_and_does_not_desync_the_tape() {
        // -1 cast naively to usize becomes usize::MAX; summing that
        // straight into `expected_matches_width` (the pre-extraction
        // behavior) overflows usize before the loop even runs.
        // `checked_range` rejects the row instead, contributing 0 to both
        // the width sum and `n`.
        let arr = array![2_i64, 3];
        let ends = array![-1_i64, 2];
        let counts = array![5_i64, 2];
        let matches = array![1_i8, 1];
        let booleans = array![false, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1, 2 * 3]);
    }

    #[test]
    fn end_beyond_len_is_identity_not_a_panic() {
        let arr = array![2_i64, 3];
        let ends = array![1000_i64];
        let counts = array![5_i64];
        let matches: Array1<i8> = array![];
        let booleans = array![false, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn multiple_rows_share_one_flat_matches_tape() {
        // Each row's window is `0..end_i`, so different `end`s produce
        // different-width, non-shared tape slices: row 0 (end=2) owns
        // tape[0..2], row 1 (end=4) owns tape[2..6].
        let arr = array![2_i64, 3, 4, 5];
        let ends = array![2_i64, 4];
        let counts = array![2_i64, 4];
        let matches = array![1_i8, 1, 1, 1, 1, 1];
        let booleans = array![false, false, false, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2 * 3, 2 * 3 * 4 * 5]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 2];
        let ends = array![2_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 1];
        let booleans = array![false, false];
        let got = prod_end_match_core(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-2]);
    }

    #[test]
    fn zero_and_negative_values_follow_product_semantics() {
        let ends = array![2_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 1];
        let booleans = array![false, false];
        assert_eq!(
            prod_end_match_core(
                array![0_i64, 3].view(),
                ends.view(),
                counts.view(),
                matches.view(),
                booleans.view(),
            ),
            array![0]
        );
        assert_eq!(
            prod_end_match_core(
                array![-2_i64, 3].view(),
                ends.view(),
                counts.view(),
                matches.view(),
                booleans.view(),
            ),
            array![-6]
        );
    }

    #[test]
    fn float_variant_multiplies_only_live_positions() {
        let arr = array![2.0_f64, 3.0, 4.0];
        let ends = array![3_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1];
        let booleans = array![false, false, false];
        let got = prod_end_match_float_core_with_cast(
            arr.view(),
            ends.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got, array![8.0]);
    }
}
