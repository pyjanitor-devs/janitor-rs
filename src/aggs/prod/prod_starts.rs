use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::adaptive::should_use_running_aggregation;
use crate::aggs::{ensure_equal_lengths_core, ensure_nonempty_core};

/// Computes the product of every suffix selected by `starts` for an integer
/// input array.
///
/// `arr` contains the values, `starts` contains zero-based inclusive suffix
/// boundaries, and `booleans` marks null values (`true` values are skipped).
/// The result has one `i64` product per entry in `starts`; null-only suffixes
/// therefore return the multiplicative identity, `1`.
///
/// Integer multiplication uses `wrapping_mul`, matching the explicit
/// fixed-width wrapping behavior used by the integer sum cores.
///
/// # Arguments
///
/// * `arr` - Values to multiply.
/// * `starts` - Inclusive suffix boundaries.
/// * `booleans` - Null mask aligned with `arr`.
pub fn prod_start_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut convert: F,
) -> Result<Array1<i64>, String>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<i64>::zeros(starts.len());
    let end_ = arr.len();
    let mut total_width = 0_usize;
    for start in starts.iter() {
        if let Ok(start_) = usize::try_from(*start) {
            total_width = total_width.saturating_add(end_.saturating_sub(start_));
        }
    }
    if should_use_running_aggregation(starts.len(), total_width, end_) {
        // ELI5: when many suffix questions together would walk the array
        // repeatedly, multiply each suffix once and answer the questions by
        // lookup. Null entries contribute the multiplicative identity `1`.
        let mut suffix = vec![1_i64; end_ + 1];
        for nn in (0..end_).rev() {
            suffix[nn] = suffix[nn + 1];
            if !booleans[nn] {
                suffix[nn] = suffix[nn].wrapping_mul(convert(arr[nn]));
            }
        }
        for (pos, start) in starts.iter().enumerate() {
            if let Ok(start_) = usize::try_from(*start) {
                if start_ < end_ {
                    result[pos] = suffix[start_];
                }
            }
        }
        return Ok(result);
    }
    for (pos, start) in starts.iter().enumerate() {
        let mut total = 1_i64;
        let start_ = *start as usize;
        for nn in start_..end_ {
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
    fn broad_suffix_batch_uses_running_products() {
        let arr = array![2_i64, 3, 4, 5, 6, 7];
        let starts = array![0_i64, 1, 2, 3, 4, 5];
        let booleans = array![false, false, false, false, false, false];
        let got =
            prod_start_core(arr.view(), starts.view(), booleans.view(), |value| value).unwrap();
        assert_eq!(got, array![5040, 2520, 840, 210, 42, 7]);
    }
}

/// Computes floating-point products for suffix queries described by `starts`.
///
/// Floating-point multiplication is kept separate from the integer core so it
/// follows IEEE-754 behavior for zero, infinity, NaN, overflow, and underflow
/// instead of applying integer wrapping semantics. The running suffix path
/// is valid because it preserves the multiplication order of each suffix.
///
/// # Arguments
///
/// * `arr` - Values to multiply.
/// * `starts` - Inclusive suffix boundaries.
/// * `booleans` - Null mask aligned with `arr`.
pub fn prod_start_float_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut convert: F,
) -> Result<Array1<f64>, String>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut result = Array1::<f64>::zeros(starts.len());
    let end_ = arr.len();
    let mut total_width = 0_usize;
    for start in starts.iter() {
        if let Ok(start_) = usize::try_from(*start) {
            total_width = total_width.saturating_add(end_.saturating_sub(start_));
        }
    }
    if should_use_running_aggregation(starts.len(), total_width, end_) {
        let mut suffix = vec![1.0_f64; end_ + 1];
        for nn in (0..end_).rev() {
            suffix[nn] = suffix[nn + 1];
            if !booleans[nn] {
                suffix[nn] *= convert(arr[nn]);
            }
        }
        for (pos, start) in starts.iter().enumerate() {
            if let Ok(start_) = usize::try_from(*start) {
                if start_ < end_ {
                    result[pos] = suffix[start_];
                }
            }
        }
        return Ok(result);
    }
    for (pos, start) in starts.iter().enumerate() {
        let mut total = 1.0_f64;
        let start_ = *start as usize;
        for nn in start_..end_ {
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
        /// Compute products over suffixes of `arr` for integer-compatible
        /// values. `starts` supplies inclusive boundaries and `booleans`
        /// marks null values to skip; the returned array follows `starts`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Values to multiply.
        /// * `starts` - Inclusive suffix boundaries.
        /// * `booleans` - Null mask aligned with `arr`.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            let result = prod_start_core(
                arr.as_array(),
                starts.as_array(),
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
        /// Compute floating-point products over suffixes of `arr`.
        /// `starts` supplies inclusive boundaries and `booleans` marks null
        /// values to skip; the returned array follows `starts`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Values to multiply.
        /// * `starts` - Inclusive suffix boundaries.
        /// * `booleans` - Null mask aligned with `arr`.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>>
        // The macro will expand into the contents of this block.
        {
            let result = prod_start_float_core(
                arr.as_array(),
                starts.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result
                .map_err(pyo3::exceptions::PyValueError::new_err)?
                .into_pyarray(py))
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
