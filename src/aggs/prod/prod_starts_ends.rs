use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths};

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`prod_start_end_core_with_cast`] at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn prod_start_end_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    prod_start_end_core_with_cast(arr, starts, ends, booleans, |value| value)
}

/// For every `(starts[i], ends[i])`, multiply `arr[starts[i]..ends[i]]`,
/// skipping any position flagged `true` in `booleans` (a null mask).
/// Returns `1` (the multiplicative identity) for an empty or invalid range
/// (`start < 0`, `end < 0`, `start >= end`, or `end > arr.len()`).
///
/// ELI5 (the guard): `checked_range(start, end, arr.len())` rejects a
/// negative, inverted, or too-large range before it's cast to `usize`; an
/// unguarded `-1` "no match" sentinel would otherwise wrap to
/// `usize::MAX` and walk `arr`/`booleans` out of bounds. Same shape as
/// `sum_start_end_core`'s guard, just preserving product's identity (`1`)
/// instead of sum's (`0`) for a rejected row.
fn prod_start_end_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::from_elem(starts.len(), 1);
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut total: i64 = 1;
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

fn prod_start_end_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::from_elem(starts.len(), 1.0);
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut total: f64 = 1.0;
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

macro_rules! generic_compute_ints {
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
            let result = prod_start_end_core_with_cast(
                arr.as_array(),
                starts,
                ends,
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
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>>
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
            let result = prod_start_end_float_core_with_cast(
                arr.as_array(),
                starts,
                ends,
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}
generic_compute_ints!(compute_prod_start_end_int64, i64);
generic_compute_ints!(compute_prod_start_end_int32, i32);
generic_compute_ints!(compute_prod_start_end_int16, i16);
generic_compute_ints!(compute_prod_start_end_int8, i8);
generic_compute_ints!(compute_prod_start_end_uint64, u64);
generic_compute_ints!(compute_prod_start_end_uint32, u32);
generic_compute_ints!(compute_prod_start_end_uint16, u16);
generic_compute_ints!(compute_prod_start_end_uint8, u8);
generic_compute_floats!(compute_prod_start_end_f32, f32);
generic_compute_floats!(compute_prod_start_end_f64, f64);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_start_end_f64, m)?)?;
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
        let ends = array![0_i64];
        let booleans: Array1<bool> = array![];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn full_array_range() {
        let arr = array![2_i64, 3, 4, 5];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let booleans = array![false, false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![120]);
    }

    #[test]
    fn arbitrary_interior_slice() {
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![1_i64];
        let ends = array![4_i64]; // [3, 4, 5]
        let booleans = array![false, false, false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![60]);
    }

    #[test]
    fn inverted_range_is_identity() {
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![3_i64];
        let ends = array![1_i64]; // start > end
        let booleans = array![false, false, false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn equal_start_and_end_is_identity() {
        let arr = array![2_i64, 3, 4];
        let starts = array![1_i64];
        let ends = array![1_i64];
        let booleans = array![false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn sentinel_end_is_identity_not_a_panic() {
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![2_i64];
        let ends = array![-1_i64];
        let booleans = array![false, false, false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn sentinel_start_is_identity_not_a_panic() {
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![-1_i64];
        let ends = array![3_i64];
        let booleans = array![false, false, false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn oversized_non_sentinel_end_is_identity_not_a_panic() {
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![0_i64];
        let ends = array![1000_i64];
        let booleans = array![false, false, false, false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1]);
    }

    #[test]
    fn float_sentinel_bounds_are_identity_not_a_panic() {
        let arr = array![2.0_f64, 3.0, 4.0];
        let starts = array![-1_i64, 0_i64];
        let ends = array![2_i64, -1_i64];
        let booleans = array![false, false, false];
        let got = prod_start_end_float_core_with_cast(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got, array![1.0, 1.0]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![2_i64, 3, 4, 5];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let booleans = array![false, true, false, true];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![2 * 4]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 2];
        let starts = array![0_i64];
        let ends = array![2_i64];
        let booleans = array![false, false];
        let got = prod_start_end_core(arr.view(), starts.view(), ends.view(), booleans.view());
        assert_eq!(got, array![-2]);
    }

    #[test]
    fn zero_and_negative_values_follow_product_semantics() {
        let booleans = array![false, false];
        assert_eq!(
            prod_start_end_core(
                array![0_i64, 3].view(),
                array![0_i64].view(),
                array![2_i64].view(),
                booleans.view(),
            ),
            array![0]
        );
        assert_eq!(
            prod_start_end_core(
                array![-2_i64, 3].view(),
                array![0_i64].view(),
                array![2_i64].view(),
                booleans.view(),
            ),
            array![-6]
        );
    }

    #[test]
    fn casts_only_values_in_requested_interval() {
        let arr = array![2_i32, 3, 4, 5];
        let starts = array![2_i64];
        let ends = array![3_i64];
        let booleans = array![false, false, false, false];
        let mut casts = 0;
        let got = prod_start_end_core_with_cast(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| {
                casts += 1;
                value as i64
            },
        );
        assert_eq!(got, array![4]);
        assert_eq!(casts, 1);
    }
}
