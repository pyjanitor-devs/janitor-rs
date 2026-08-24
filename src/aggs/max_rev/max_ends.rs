use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_range, ensure_equal_lengths};

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let length = length as usize;
            let mut slots: DenseSlots<(i64, $type)> = DenseSlots::new(length);
            let zipped = izip!(arr.into_iter(), ends.into_iter(), booleans.into_iter());
            for (posn, (current, end, boolean)) in zipped.enumerate() {
                // ELI5 (the guard): `end` indexes into `index`, not `arr`, so
                // the bound to check against is `index.len()`; an unguarded
                // cast of the `-1` "no match" sentinel wraps to `usize::MAX`
                // and walks `index` out of bounds. See issue #34.
                let Some((_, end_)) = checked_range(0, *end, index.len()) else {
                    continue;
                };
                for item in 0..end_ {
                    let pos = index[item] as usize;
                    let (base, base_val) = slots.touch(pos, (-1, *current));
                    if *boolean {
                        continue;
                    }
                    if (*base == -1) || (*current > *base_val) {
                        *base_val = *current;
                        *base = posn as i64;
                    }
                }
            }
            let (indexers, result) = slots.to_arrays(|(base, _base_val)| *base);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_max_rev_end_int64, i64);
compute!(compute_max_rev_end_int32, i32);
compute!(compute_max_rev_end_int16, i16);
compute!(compute_max_rev_end_int8, i8);
compute!(compute_max_rev_end_uint64, u64);
compute!(compute_max_rev_end_uint32, u32);
compute!(compute_max_rev_end_uint16, u16);
compute!(compute_max_rev_end_uint8, u8);
compute!(compute_max_rev_end_f64, f64);
compute!(compute_max_rev_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::compute_max_rev_end_int64;

    #[test]
    fn touched_row_positions_are_emitted_ascending_with_winning_row_index() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [5, 3]; row0 (end=1) only reaches index[0]=5, row1
            // (end=2) reaches index[0..2] = {5, 3}. Row1's value (30)
            // beats row0's (10) at position 5.
            let arr = PyArray1::from_vec(py, vec![10_i64, 30]);
            let ends = PyArray1::from_vec(py, vec![1_i64, 2]);
            let index = PyArray1::from_vec(py, vec![5_i64, 3]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let (indexers, result) = compute_max_rev_end_int64(
                py,
                arr.readonly(),
                ends.readonly(),
                index.readonly(),
                booleans.readonly(),
                6,
            )
            .expect("valid equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![3, 5]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![1, 1]);
        });
    }
}
