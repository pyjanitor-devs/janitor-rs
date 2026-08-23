use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths};

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
            let arr = arr.as_array();
            let booleans = booleans.as_array();
            // ELI5: `1`, not `0` -- an empty or rejected range must
            // preserve product's multiplicative identity, or a bounds
            // guard would silently change the result for rows it rejects.
            let mut result = Array1::<i64>::from_elem(starts.len(), 1);
            let zipped = starts.into_iter().zip(ends.into_iter());
            for (pos, (start, end)) in zipped.enumerate() {
                // ELI5 (the guard): `checked_range` rejects a negative,
                // inverted, or too-large range before it's cast to `usize`;
                // an unguarded `-1` "no match" sentinel would otherwise
                // wrap to `usize::MAX` and walk `arr`/`booleans` out of
                // bounds.
                let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
                    continue;
                };
                let mut total: i64 = 1;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    total *= current as i64;
                }
                result[pos] = total;
            }
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
            let arr = arr.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::from_elem(starts.len(), 1.0);
            let zipped = starts.into_iter().zip(ends.into_iter());
            for (pos, (start, end)) in zipped.enumerate() {
                let Some((start_, end_)) = checked_range(*start, *end, arr.len()) else {
                    continue;
                };
                let mut total: f64 = 1.0;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    total *= current as f64;
                }
                result[pos] = total;
            }
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
