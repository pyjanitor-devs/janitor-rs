use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ensure_equal_lengths, ensure_tape_width};

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`prod_start_match_core_with_cast`] at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn prod_start_match_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    prod_start_match_core_with_cast(arr, starts, counts, matches, booleans, |value| value)
}

/// For every `starts[i]`, multiply `arr[starts[i]..]` (to the end of the
/// array), but only at positions the caller has already flagged as live
/// in `matches` (a *flat* tape shared by every row, one entry per
/// candidate position; see `compare_start_end_core`'s doc comment for the
/// same tape shape). `counts[i] == 0` short-circuits straight to the
/// identity without even reading `matches` for that row's span. Returns
/// `1` (the multiplicative identity) for a row with nothing live.
///
/// ELI5 (no bounds guard needed): same reasoning as `prod_start_core` --
/// the tape pointer `n` and the running product both only advance inside
/// `start_..arr.len()`, which is an *empty* Rust range (zero iterations,
/// `n` untouched) for any `start` that wraps to a huge or otherwise
/// out-of-range `usize`. `expected_matches_width` (computed by the
/// caller-side wrapper) uses the same saturating arithmetic, so a
/// rejected-by-emptiness row contributes zero to both sides and `n` never
/// desyncs from the tape. See issue #63.
fn prod_start_match_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::from_elem(starts.len(), 1);
    let mut n: usize = 0;
    let end_: usize = arr.len();
    let zipped = starts.into_iter().zip(counts);
    for (pos, (start, count)) in zipped.enumerate() {
        let start_ = *start as usize;
        let mut total: i64 = 1;
        if *count == 0 {
            n += end_.saturating_sub(start_);
            continue;
        }
        for nn in start_..end_ {
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

fn prod_start_match_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::from_elem(starts.len(), 1.0);
    let mut n: usize = 0;
    let end_: usize = arr.len();
    let zipped = starts.into_iter().zip(counts);
    for (pos, (start, count)) in zipped.enumerate() {
        let start_ = *start as usize;
        let mut total: f64 = 1.0;
        if *count == 0 {
            n += end_.saturating_sub(start_);
            continue;
        }
        for nn in start_..end_ {
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
            let starts = starts.as_array();
            let end_: usize = arr.as_array().len();
            let expected_matches_width: usize = starts
                .iter()
                .map(|s| end_.saturating_sub(*s as usize))
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let result = prod_start_match_core_with_cast(
                arr.as_array(),
                starts,
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
            starts: PyReadonlyArray1<'py, i64>,
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
            let starts = starts.as_array();
            let end_: usize = arr.as_array().len();
            let expected_matches_width: usize = starts
                .iter()
                .map(|s| end_.saturating_sub(*s as usize))
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let result = prod_start_match_float_core_with_cast(
                arr.as_array(),
                starts,
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute_ints!(compute_prod_start_match_int64, i64);
generic_compute_ints!(compute_prod_start_match_int32, i32);
generic_compute_ints!(compute_prod_start_match_int16, i16);
generic_compute_ints!(compute_prod_start_match_int8, i8);
generic_compute_ints!(compute_prod_start_match_uint64, u64);
generic_compute_ints!(compute_prod_start_match_uint32, u32);
generic_compute_ints!(compute_prod_start_match_uint16, u16);
generic_compute_ints!(compute_prod_start_match_uint8, u8);
generic_compute_floats!(compute_prod_start_match_f64, f64);
generic_compute_floats!(compute_prod_start_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_start_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array() {
        let arr: Array1<i64> = array![];
        let starts = array![0_i64];
        let counts = array![0_i64];
        let matches: Array1<i8> = array![];
        let booleans: Array1<bool> = array![];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn zero_count_short_circuits_to_identity() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64];
        let counts = array![0_i64];
        let matches = array![1_i8, 1, 1]; // ignored: count is 0
        let booleans = array![false, false, false];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn matches_mask_skips_positions_but_still_advances_the_tape() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1]; // middle position not live
        let booleans = array![false, false, false];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64];
        let counts = array![3_i64];
        let matches = array![1_i8, 1, 1];
        let booleans = array![false, true, false];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn multiple_rows_share_one_flat_matches_tape() {
        // Each row's window is `start_i..arr.len()`, not `count`-sized, so
        // rows with different `start`s revisit overlapping array
        // positions through *different* tape slots: row 0 (start=0) owns
        // tape[0..4] for positions 0..4, row 1 (start=2) owns tape[4..6]
        // for positions 2..4. Total tape width is the sum of both
        // (4 + 2 = 6), not `arr.len()`.
        let arr = array![2_i64, 3, 4, 5];
        let starts = array![0_i64, 2];
        let counts = array![4_i64, 2];
        let matches = array![1_i8, 1, 1, 1, 1, 1];
        let booleans = array![false, false, false, false];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2 * 3 * 4 * 5, 4 * 5]);
    }

    #[test]
    fn negative_start_is_identity_and_does_not_desync_the_tape() {
        // A negative `start` wraps to a huge `usize` on cast, so
        // `start_..arr.len()` is empty and the tape pointer for this row
        // must not advance at all -- matching
        // `expected_matches_width`'s own `saturating_sub` contributing 0
        // for the same row.
        let arr = array![2_i64, 3];
        let starts = array![-5_i64, 0];
        let counts = array![5_i64, 2]; // first row's count is irrelevant once start wraps
        let matches = array![1_i8, 1];
        let booleans = array![false, false];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1, 2 * 3]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 2];
        let starts = array![0_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 1];
        let booleans = array![false, false];
        let got = prod_start_match_core(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-2]);
    }

    #[test]
    fn zero_and_negative_values_follow_product_semantics() {
        let starts = array![0_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 1];
        let booleans = array![false, false];
        assert_eq!(
            prod_start_match_core(
                array![0_i64, 3].view(),
                starts.view(),
                counts.view(),
                matches.view(),
                booleans.view(),
            ),
            array![0]
        );
        assert_eq!(
            prod_start_match_core(
                array![-2_i64, 3].view(),
                starts.view(),
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
        let starts = array![0_i64];
        let counts = array![2_i64];
        let matches = array![1_i8, 0, 1];
        let booleans = array![false, false, false];
        let got = prod_start_match_float_core_with_cast(
            arr.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got, array![8.0]);
    }
}
