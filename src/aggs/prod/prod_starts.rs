use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`prod_start_core_with_cast`] at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn prod_start_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    prod_start_core_with_cast(arr, starts, booleans, |value| value)
}

/// For every `starts[i]`, multiply `arr[starts[i]..]` (to the end of the
/// array), skipping any position flagged `true` in `booleans` (a null
/// mask). Returns `1` (the multiplicative identity) when the range is
/// empty or `starts[i]` is negative or past `arr.len()`.
///
/// ELI5 (no bounds guard needed, unlike `min`/`max`): `prod` never reads
/// `arr[start_]` unconditionally before checking the range -- the loop
/// body itself is the only place `arr` is indexed, and `start_..arr.len()`
/// is simply an *empty* Rust range (not a panic) whenever `start_` is
/// wrapped-huge (a negative `start` cast to `usize`) or otherwise past
/// `arr.len()`. So the running product starts at `1` and never changes,
/// which is exactly the identity a caller expects for an empty/invalid
/// range. Contrast `min_start_core`, which needs `checked_index` because
/// it must read a real element to seed its comparison. See issue #63.
fn prod_start_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::from_elem(starts.len(), 1);
    let end_: usize = arr.len();
    for (pos, start) in starts.indexed_iter() {
        let mut total: i64 = 1;
        let start_ = *start as usize;
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            total = total.wrapping_mul(to_i64(arr[nn]));
        }
        result[pos] = total;
    }
    result
}

fn prod_start_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::from_elem(starts.len(), 1.0);
    let end_: usize = arr.len();
    for (pos, start) in starts.indexed_iter() {
        let mut total: f64 = 1.0;
        let start_ = *start as usize;
        for nn in start_..end_ {
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
            starts: PyReadonlyArray1<'py, i64>,
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
            let result = prod_start_core_with_cast(
                arr.as_array(),
                starts.as_array(),
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
            let result = prod_start_float_core_with_cast(
                arr.as_array(),
                starts.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}
generic_compute!(compute_prod_start_int64, i64);
generic_compute!(compute_prod_start_int32, i32);
generic_compute!(compute_prod_start_int16, i16);
generic_compute!(compute_prod_start_int8, i8);
generic_compute!(compute_prod_start_uint64, u64);
generic_compute!(compute_prod_start_uint32, u32);
generic_compute!(compute_prod_start_uint16, u16);
generic_compute!(compute_prod_start_uint8, u8);
generic_compute_floats!(compute_prod_start_f32, f32);
generic_compute_floats!(compute_prod_start_f64, f64);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_f64, m)?)?;
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
        let booleans: Array1<bool> = array![];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn start_at_end_is_identity() {
        let arr = array![2_i64, 3, 4];
        let starts = array![3_i64]; // boundary: nothing left to multiply
        let booleans = array![false, false, false];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn start_at_zero_multiplies_everything() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64]; // boundary: whole array
        let booleans = array![false, false, false];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![24]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![2_i64, 3, 4, 5];
        let starts = array![0_i64];
        // position 1 (value 3) and position 3 (value 5) are "null"
        let booleans = array![false, true, false, true];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn all_null_range_is_identity() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64];
        let booleans = array![true, true, true];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn negative_start_is_identity_not_a_panic() {
        // A negative `start` cast to `usize` wraps to a huge value, but
        // `start_..arr.len()` is then simply an empty Rust range -- no
        // unconditional read happens before that check, unlike
        // `min`/`max`, so this needs no explicit bounds guard to stay
        // safe. See the doc comment on `prod_start_core_with_cast`.
        let arr = array![2_i64, 3, 4];
        let starts = array![-5_i64];
        let booleans = array![false, false, false];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn start_past_len_is_identity_not_a_panic() {
        let arr = array![2_i64, 3, 4];
        let starts = array![1000_i64];
        let booleans = array![false, false, false];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 2];
        let starts = array![0_i64];
        let booleans = array![false, false];
        let got = prod_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![-2]);
    }

    #[test]
    fn float_variant_multiplies_from_start_to_end() {
        let arr = array![2.0_f64, 3.0, 4.0];
        let starts = array![1_i64];
        let booleans = array![false, false, false];
        let got =
            prod_start_float_core_with_cast(arr.view(), starts.view(), booleans.view(), |value| {
                value
            });
        assert_eq!(got, array![12.0]);
    }

    #[test]
    fn casts_only_values_in_requested_suffix() {
        let arr = array![2_i32, 3, 4, 5];
        let starts = array![3_i64];
        let booleans = array![false, false, false, false];
        let mut casts = 0;
        let got = prod_start_core_with_cast(arr.view(), starts.view(), booleans.view(), |value| {
            casts += 1;
            value as i64
        });
        assert_eq!(got, array![5]);
        assert_eq!(casts, 1);
    }
}
