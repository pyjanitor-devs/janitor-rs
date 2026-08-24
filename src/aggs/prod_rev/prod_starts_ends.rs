use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_range, ensure_equal_lengths};

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
            let mut slots: DenseSlots<i64> = DenseSlots::new(length);
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
                    let pos = index[item] as usize;
                    let total = slots.touch(pos, 1);
                    if *boolean {
                        continue;
                    }
                    *total *= current_;
                }
            }
            let (indexers, result) = slots.to_arrays(|value| *value);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_prod_rev_start_end_int64, i64);
compute_ints!(compute_prod_rev_start_end_int32, i32);
compute_ints!(compute_prod_rev_start_end_int16, i16);
compute_ints!(compute_prod_rev_start_end_int8, i8);
compute_ints!(compute_prod_rev_start_end_uint64, u64);
compute_ints!(compute_prod_rev_start_end_uint32, u32);
compute_ints!(compute_prod_rev_start_end_uint16, u16);
compute_ints!(compute_prod_rev_start_end_uint8, u8);

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
            let mut slots: DenseSlots<f64> = DenseSlots::new(length);
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
                    let pos = index[item] as usize;
                    let total = slots.touch(pos, 1.);
                    if *boolean {
                        continue;
                    }
                    *total *= current_;
                }
            }
            let (indexers, result) = slots.to_arrays(|value| *value);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_prod_rev_start_end_f64, f64);
compute_floats!(compute_prod_rev_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::compute_prod_rev_start_end_int64;

    #[test]
    fn touched_row_positions_are_emitted_ascending_with_products() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [4, 1, 8]; row0 (0..2) reaches index[0..2] =
            // {4, 1}, row1 (1..3) reaches index[1..3] = {1, 8}. Position
            // 1 gets both rows' values: 3*7=21.
            let arr = PyArray1::from_vec(py, vec![3_i64, 7]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3]);
            let index = PyArray1::from_vec(py, vec![4_i64, 1, 8]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let (indexers, result) = compute_prod_rev_start_end_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                booleans.readonly(),
                9,
            )
            .expect("valid equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![1, 4, 8]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![21, 3, 7]);
        });
    }
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
        let _wrapper: Int8Fn = compute_prod_rev_start_end_int8;
    }

    #[test]
    fn f32_wrapper_accepts_an_f32_array() {
        let _wrapper: F32Fn = compute_prod_rev_start_end_f32;
    }

    #[test]
    fn f64_wrapper_accepts_an_f64_array() {
        let _wrapper: F64Fn = compute_prod_rev_start_end_f64;
    }
}
