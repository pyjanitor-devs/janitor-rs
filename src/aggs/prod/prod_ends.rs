use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_range;
use crate::aggs::ensure_equal_lengths;

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`prod_end_core_with_cast`] at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn prod_end_core(
    arr: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    prod_end_core_with_cast(arr, ends, booleans, |value| value)
}

/// For every `ends[i]`, multiply `arr[..ends[i]]` (from the start of the
/// array), skipping any position flagged `true` in `booleans` (a null
/// mask). Returns `1` (the multiplicative identity) when `end` is negative
/// or past `arr.len()`.
///
/// ELI5 (the guard, unlike `prod_start_core`): here the loop always walks
/// `0..end_`, so a negative `end` cast to `usize` (e.g. the `-1` "no
/// match" sentinel, which wraps to `usize::MAX`) does *not* land on an
/// empty range the way an out-of-range `start` does in `prod_start_core`
/// -- it instead walks far past `arr.len()` and indexes out of bounds.
/// `checked_range(0, end, arr.len())` rejects that (and any `end >
/// arr.len()`) before the cast, matching `min_end_core`'s same-shaped
/// guard. See issue #63.
fn prod_end_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::from_elem(ends.len(), 1);
    for (pos, end) in ends.indexed_iter() {
        let Some((_, end_)) = checked_range(0, *end, arr.len()) else {
            continue;
        };
        let mut total: i64 = 1;
        for nn in 0..end_ {
            if booleans[nn] {
                continue;
            }
            total = total.wrapping_mul(to_i64(arr[nn]));
        }
        result[pos] = total;
    }
    result
}

fn prod_end_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::from_elem(ends.len(), 1.0);
    for (pos, end) in ends.indexed_iter() {
        let Some((_, end_)) = checked_range(0, *end, arr.len()) else {
            continue;
        };
        let mut total: f64 = 1.0;
        for nn in 0..end_ {
            if booleans[nn] {
                continue;
            }
            total *= to_f64(arr[nn]);
        }
        result[pos] = total;
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
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let result = prod_end_core_with_cast(
                arr.as_array(),
                ends.as_array(),
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
            let result = prod_end_float_core_with_cast(
                arr.as_array(),
                ends.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute!(compute_prod_end_int64, i64);
generic_compute!(compute_prod_end_int32, i32);
generic_compute!(compute_prod_end_int16, i16);
generic_compute!(compute_prod_end_int8, i8);
generic_compute!(compute_prod_end_uint64, u64);
generic_compute!(compute_prod_end_uint32, u32);
generic_compute!(compute_prod_end_uint16, u16);
generic_compute!(compute_prod_end_uint8, u8);
generic_compute_floats!(compute_prod_end_f32, f32);
generic_compute_floats!(compute_prod_end_f64, f64);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_end_f64, m)?)?;
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
        let booleans: Array1<bool> = array![];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn end_at_zero_is_identity() {
        let arr = array![2_i64, 3, 4];
        let ends = array![0_i64]; // boundary: nothing yet
        let booleans = array![false, false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn end_at_len_multiplies_everything() {
        let arr = array![2_i64, 3, 4];
        let ends = array![3_i64]; // boundary: whole array
        let booleans = array![false, false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![24]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![2_i64, 3, 4, 5];
        let ends = array![4_i64];
        let booleans = array![false, true, false, true];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn all_null_range_is_identity() {
        let arr = array![2_i64, 3, 4];
        let ends = array![3_i64];
        let booleans = array![true, true, true];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn sentinel_end_is_identity_not_a_panic() {
        // -1 is the crate's "invalid/no match" sentinel. Cast naively to
        // usize it becomes usize::MAX, and the loop would walk straight
        // off the end of `arr`; this must return 1 instead of panicking.
        let arr = array![2_i64, 3, 4];
        let ends = array![-1_i64];
        let booleans = array![false, false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn end_beyond_len_is_identity_not_a_panic() {
        let arr = array![2_i64, 3, 4];
        let ends = array![1000_i64];
        let booleans = array![false, false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 2];
        let ends = array![2_i64];
        let booleans = array![false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-2]);
    }

    #[test]
    fn float_sentinel_end_is_identity_not_a_panic() {
        let arr = array![2.0_f64, 3.0];
        let ends = array![-1_i64];
        let booleans = array![false, false];
        let got =
            prod_end_float_core_with_cast(arr.view(), ends.view(), booleans.view(), |value| value);
        assert_eq!(got, array![1.0]);
    }

    #[test]
    fn casts_only_values_in_requested_prefix() {
        let arr = array![2_i32, 3, 4, 5];
        let ends = array![1_i64];
        let booleans = array![false, false, false, false];
        let mut casts = 0;
        let got = prod_end_core_with_cast(arr.view(), ends.view(), booleans.view(), |value| {
            casts += 1;
            value as i64
        });
        assert_eq!(got, array![2]);
        assert_eq!(casts, 1);
    }
}
