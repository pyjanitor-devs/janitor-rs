use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::adaptive::should_use_running_aggregation;
use crate::aggs::{ensure_equal_lengths_core, ensure_nonempty_core};

/// Computes the product of every prefix selected by `ends` for an integer
/// input array. `ends` contains exclusive zero-based boundaries, and `true`
/// entries in `booleans` mark null values that contribute the identity `1`.
/// Integer products use fixed-width wrapping multiplication.
///
/// # Arguments
///
/// * `arr` - Values to multiply.
/// * `ends` - Exclusive prefix boundaries.
/// * `booleans` - Null mask aligned with `arr`.
pub fn prod_end_core<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut convert: F,
) -> Result<Array1<i64>, String>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<i64>::from_elem(ends.len(), 1);
    let mut total_width = 0_usize;
    for end in ends.iter() {
        if let Ok(end_) = usize::try_from(*end) {
            total_width = total_width.saturating_add(end_.min(arr.len()));
        }
    }
    if should_use_running_aggregation(ends.len(), total_width, arr.len()) {
        // ELI5: when many prefix questions together would walk the array
        // repeatedly, multiply each prefix once and answer the questions by
        // lookup. Null entries contribute the multiplicative identity `1`.
        let mut prefix = vec![1_i64; arr.len() + 1];
        for nn in 0..arr.len() {
            prefix[nn + 1] = prefix[nn];
            if !booleans[nn] {
                prefix[nn + 1] = prefix[nn + 1].wrapping_mul(convert(arr[nn]));
            }
        }
        for (pos, end) in ends.iter().enumerate() {
            if let Ok(end_) = usize::try_from(*end) {
                if end_ <= arr.len() {
                    result[pos] = prefix[end_];
                }
            }
        }
        return Ok(result);
    }
    for (pos, end) in ends.iter().enumerate() {
        let mut total = 1_i64;
        let end_ = *end as usize;
        for nn in 0..end_ {
            if !booleans[nn] {
                total = total.wrapping_mul(convert(arr[nn]));
            }
        }
        result[pos] = total;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn broad_prefix_batch_uses_running_products() {
        let arr = array![2_i64, 3, 4, 5, 6, 7];
        let ends = array![1_i64, 2, 3, 4, 5, 6];
        let booleans = array![false, false, false, false, false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view(), |value| value).unwrap();
        assert_eq!(got, array![2, 6, 24, 120, 720, 5040]);
    }

    #[test]
    fn invalid_adaptive_prefixes_keep_product_identity() {
        let arr = array![2_i64, 3, 4];
        let ends = array![-1_i64, 3, 3, 3, 3];
        let booleans = array![false, false, false];
        let got = prod_end_core(arr.view(), ends.view(), booleans.view(), |value| value).unwrap();
        assert_eq!(got, array![1, 24, 24, 24, 24]);
    }
}

/// Computes floating-point products for prefix queries described by `ends`.
/// This core is separate from the integer version so IEEE-754 behavior is
/// preserved for zero, infinity, NaN, overflow, and underflow. The running
/// prefix path preserves the multiplication order of each prefix.
///
/// # Arguments
///
/// * `arr` - Values to multiply.
/// * `ends` - Exclusive prefix boundaries.
/// * `booleans` - Null mask aligned with `arr`.
pub fn prod_end_float_core<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut convert: F,
) -> Result<Array1<f64>, String>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<f64>::from_elem(ends.len(), 1.0);
    let mut total_width = 0_usize;
    for end in ends.iter() {
        if let Ok(end_) = usize::try_from(*end) {
            total_width = total_width.saturating_add(end_.min(arr.len()));
        }
    }
    if should_use_running_aggregation(ends.len(), total_width, arr.len()) {
        let mut prefix = vec![1.0_f64; arr.len() + 1];
        for nn in 0..arr.len() {
            prefix[nn + 1] = prefix[nn];
            if !booleans[nn] {
                prefix[nn + 1] *= convert(arr[nn]);
            }
        }
        for (pos, end) in ends.iter().enumerate() {
            if let Ok(end_) = usize::try_from(*end) {
                if end_ <= arr.len() {
                    result[pos] = prefix[end_];
                }
            }
        }
        return Ok(result);
    }
    for (pos, end) in ends.iter().enumerate() {
        let mut total = 1.0_f64;
        let end_ = *end as usize;
        for nn in 0..end_ {
            if !booleans[nn] {
                total *= convert(arr[nn]);
            }
        }
        result[pos] = total;
    }
    Ok(result)
}

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        /// Compute products over prefixes of `arr` for integer-compatible
        /// values. `ends` supplies exclusive boundaries and `booleans` marks
        /// null values to skip; the returned array follows `ends`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Values to multiply.
        /// * `ends` - Exclusive prefix boundaries.
        /// * `booleans` - Null mask aligned with `arr`.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            let result = prod_end_core(
                arr.as_array(),
                ends.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            Ok(result
                .map_err(pyo3::exceptions::PyValueError::new_err)?
                .into_pyarray(py))
        }
    };
}

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        /// Compute floating-point products over prefixes of `arr`.
        /// `ends` supplies exclusive boundaries and `booleans` marks null
        /// values to skip; the returned array follows `ends`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Values to multiply.
        /// * `ends` - Exclusive prefix boundaries.
        /// * `booleans` - Null mask aligned with `arr`.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>>
        // The macro will expand into the contents of this block.
        {
            let result = prod_end_float_core(
                arr.as_array(),
                ends.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result
                .map_err(pyo3::exceptions::PyValueError::new_err)?
                .into_pyarray(py))
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
