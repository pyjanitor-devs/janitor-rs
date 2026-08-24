use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_index, checked_range, ensure_equal_lengths};

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
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
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let length = length as usize;
            let mut slots: DenseSlots<i64> = DenseSlots::new(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (current, start, end, boolean) in zipped {
                let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                    continue;
                };
                let current_ = *current as i64;
                for nn in start_..end_ {
                    let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                        continue;
                    };
                    let pos = index[indexer_] as usize;
                    let total = slots.touch(pos, 0);
                    if *boolean {
                        continue;
                    }
                    *total += current_;
                }
            }
            let (indexers, result) = slots.to_arrays(|value| *value);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_sum_rev_positions_int64, i64);
compute_ints!(compute_sum_rev_positions_int32, i32);
compute_ints!(compute_sum_rev_positions_int16, i16);
compute_ints!(compute_sum_rev_positions_int8, i8);
compute_ints!(compute_sum_rev_positions_uint64, u64);
compute_ints!(compute_sum_rev_positions_uint32, u32);
compute_ints!(compute_sum_rev_positions_uint16, u16);
compute_ints!(compute_sum_rev_positions_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
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
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let length = length as usize;
            let mut slots: DenseSlots<(f64, f64)> = DenseSlots::new(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (current, start, end, boolean) in zipped {
                let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                    continue;
                };
                let current_ = *current as f64;
                for nn in start_..end_ {
                    let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                        continue;
                    };
                    let pos = index[indexer_] as usize;
                    let (total, compensation) = slots.touch(pos, (0., 0.));
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
            let (indexers, result) = slots.to_arrays(|(total, _compensation)| *total);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_sum_rev_positions_f64, f64);
compute_floats!(compute_sum_rev_positions_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_positions_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::compute_sum_rev_positions_int64;

    #[test]
    fn touched_row_positions_are_emitted_ascending_with_summed_values() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // positions = [1, 0, 2], index = [6, 3, 9]. row0 (0..2) walks
            // positions[0..2] = {1, 0} -> index positions {3, 6}; row1
            // (1..3) walks positions[1..3] = {0, 2} -> index positions
            // {6, 9}.
            let arr = PyArray1::from_vec(py, vec![4_i64, 5]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3]);
            let index = PyArray1::from_vec(py, vec![6_i64, 3, 9]);
            let positions = PyArray1::from_vec(py, vec![1_i64, 0, 2]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let (indexers, result) = compute_sum_rev_positions_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                positions.readonly(),
                booleans.readonly(),
                10,
            )
            .expect("valid equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![3, 6, 9]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![4, 9, 5]);
        });
    }
}
