use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths};
use std::collections::HashMap;

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter(),
            );
            for (current, start, end, boolean) in zipped {
                let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as i64;
                for item in start_..end_ {
                    let pos = index[item];
                    let total = dictionary.entry(pos).or_insert(0);
                    if *boolean {
                        continue;
                    }
                    *total += current_;
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<i64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_sum_rev_start_end_int64, i64);
compute_ints!(compute_sum_rev_start_end_int32, i32);
compute_ints!(compute_sum_rev_start_end_int16, i16);
compute_ints!(compute_sum_rev_start_end_int8, i8);
compute_ints!(compute_sum_rev_start_end_uint64, u64);
compute_ints!(compute_sum_rev_start_end_uint32, u32);
compute_ints!(compute_sum_rev_start_end_uint16, u16);
compute_ints!(compute_sum_rev_start_end_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let length = length as usize;
            let mut dictionary: HashMap<i64, f64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, f64> = HashMap::with_capacity(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter(),
            );
            for (current, start, end, boolean) in zipped {
                let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as f64;
                for item in start_..end_ {
                    let pos = index[item];
                    let total = dictionary.entry(pos).or_insert(0.);
                    let compensation = mapping.entry(pos).or_insert(0.);
                    if *boolean {
                        continue;
                    }
                    let difference = current_ - *compensation;
                    let increment = *total + difference;
                    *compensation = (increment - *total) - difference;
                    // adapted from pandas' cython code
                    // # GH#53606; GH#60303
                    // # If val is +/- infinity compensation is NaN
                    // # which would lead to results being NaN instead
                    // # of +/- infinity. We cannot use util.is_nan
                    // # because of no gil
                    if !compensation.is_finite() {
                        *compensation = 0.;
                    }
                    *total = increment;
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<f64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_sum_rev_start_end_f64, f64);
compute_floats!(compute_sum_rev_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ELI5: both macros above used to hardcode `arr`'s PyO3 type to `i64`
    // regardless of `$type`, so every non-i64 int export and *both* float
    // exports actually demanded an i64 numpy array at the Python boundary
    // -- silently for ints, with a TypeError for floats. These fn-pointer
    // typedefs make that a compile error again: reintroducing the hardcoded
    // `i64` breaks compilation instead of only failing at runtime.
    type Int8Fn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, i8>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
        i64,
    )
        -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

    type F32Fn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, f32>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
        i64,
    )
        -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>;

    type F64Fn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, f64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
        i64,
    )
        -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>;

    #[test]
    fn int8_wrapper_accepts_an_int8_array() {
        let _wrapper: Int8Fn = compute_sum_rev_start_end_int8;
    }

    #[test]
    fn f32_wrapper_accepts_an_f32_array() {
        let _wrapper: F32Fn = compute_sum_rev_start_end_f32;
    }

    #[test]
    fn f64_wrapper_accepts_an_f64_array() {
        let _wrapper: F64Fn = compute_sum_rev_start_end_f64;
    }
}
