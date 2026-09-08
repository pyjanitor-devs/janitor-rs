use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::adaptive::should_use_segment_tree;
use crate::aggs::{checked_range, ensure_equal_lengths_core, ensure_nonempty_core};

/// Compute integer products over arbitrary half-open ranges.
///
/// ELI5: narrow workloads multiply each requested slice directly. For many
/// broad, overlapping slices, the local segment tree stores each block's
/// product once and combines only the blocks covering a query. Null values
/// contribute the multiplicative identity, `1`.
///
/// Input contract: `arr`, `starts`, and `ends` must be non-empty, and
/// `starts` and `ends` must have equal lengths. The Python wrapper raises
/// `ValueError` when this contract is violated.
pub fn prod_start_end_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Result<Array1<i64>, String>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<i64>::from_elem(starts.len(), 1);
    let mut total_width = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start_, end_)) = checked_range(*start, *end, arr.len()) {
            total_width = total_width.saturating_add(end_ - start_);
        }
    }

    if should_use_segment_tree(starts.len(), total_width, arr.len()) {
        let tree_size = arr.len().next_power_of_two();
        let mut tree = vec![1_i64; tree_size * 2];
        for nn in 0..arr.len() {
            if !booleans[nn] {
                tree[tree_size + nn] = to_i64(arr[nn]);
            }
        }
        for node in (1..tree_size).rev() {
            tree[node] = tree[node * 2].wrapping_mul(tree[node * 2 + 1]);
        }

        for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
            let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
                continue;
            };
            let mut left = start_ + tree_size;
            let mut right = end_ + tree_size;
            let mut total = 1_i64;
            while left < right {
                if left % 2 == 1 {
                    total = total.wrapping_mul(tree[left]);
                    left += 1;
                }
                if right % 2 == 1 {
                    right -= 1;
                    total = total.wrapping_mul(tree[right]);
                }
                left /= 2;
                right /= 2;
            }
            result[pos] = total;
        }
        return Ok(result);
    }

    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut total = 1_i64;
        for nn in start_..end_ {
            if !booleans[nn] {
                total = total.wrapping_mul(to_i64(arr[nn]));
            }
        }
        result[pos] = total;
    }
    Ok(result)
}

/// Compute floating-point products over arbitrary half-open ranges.
///
/// This intentionally keeps the direct left-to-right loop: regrouping
/// floating-point multiplications in a tree can change rounding, signed-zero,
/// infinity, and NaN behavior.
pub fn prod_start_end_float_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Result<Array1<f64>, String>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<f64>::from_elem(starts.len(), 1.0);
    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
            continue;
        };
        let mut total = 1.0;
        for nn in start_..end_ {
            if !booleans[nn] {
                total *= to_f64(arr[nn]);
            }
        }
        result[pos] = total;
    }
    Ok(result)
}

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        /// Computes products for each half-open `arr[start..end]` range.
        /// `arr`, `starts`, and `ends` must be non-empty; invalid ranges
        /// return the multiplicative identity, `1`.
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
            let result =
                prod_start_end_core(arr.as_array(), starts, ends, booleans.as_array(), |value| {
                    value as i64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok(result.into_pyarray(py))
        }
    };
}

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        /// Computes floating-point products for each half-open range.
        /// `arr`, `starts`, and `ends` must be non-empty; invalid ranges
        /// return the multiplicative identity, `1.0`.
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
            let result = prod_start_end_float_core(
                arr.as_array(),
                starts,
                ends,
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
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
    use super::prod_start_end_core;
    use numpy::ndarray::array;

    #[test]
    fn segment_tree_preserves_wrapping_products_and_nulls() {
        let arr = array![2_i64, 3, 5, 7];
        let starts = array![0_i64, 0, 1, 0];
        let ends = array![4_i64, 3, 4, 2];
        let booleans = array![false, false, true, false];
        let got = prod_start_end_core(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(got, array![42, 6, 21, 6]);
    }

    #[test]
    fn validation_checks_nonempty_before_parallel_lengths() {
        let arr = array![2_i64];
        let starts = array![0_i64];
        let ends = array![1_i64, 1];
        let ends_short = array![1_i64];
        let booleans = array![false];
        let error = prod_start_end_core(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap_err();
        assert_eq!(
            error,
            "starts and ends must have equal lengths; got 1 and 2"
        );

        let empty_arr = numpy::ndarray::Array1::<i64>::zeros(0);
        let error = prod_start_end_core(
            empty_arr.view(),
            starts.view(),
            ends_short.view(),
            numpy::ndarray::Array1::<bool>::from_vec(vec![]).view(),
            |value| value,
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");

        let error = super::prod_start_end_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            booleans.view(),
            |value| value as f64,
        )
        .unwrap_err();
        assert_eq!(
            error,
            "starts and ends must have equal lengths; got 1 and 2"
        );
    }
}
