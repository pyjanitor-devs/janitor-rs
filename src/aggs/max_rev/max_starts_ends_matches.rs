use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_range, ensure_equal_lengths, ensure_tape_width};

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
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
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let matches = matches.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width: usize = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| checked_range(*s, *e, index.len()).map(|(s_, e_)| e_ - s_))
                .sum();
            ensure_tape_width(expected_matches_width, matches.len())?;
            let length = length as usize;
            let mut slots: DenseSlots<(i64, $type)> = DenseSlots::new(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter(),
            );
            let mut n: usize = 0;
            for (posn, (current, start, end, count, boolean)) in zipped.enumerate() {
                let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                    continue;
                };
                for item in start_..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    let Some((base, base_val)) = slots.touch(pos, (-1, *current)) else {
                        n += 1;
                        continue;
                    };
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    if (*base == -1) || (*current > *base_val) {
                        *base_val = *current;
                        *base = posn as i64;
                    }
                    n += 1;
                }
            }
            let (indexers, result) = slots.to_arrays(|(base, _base_val)| *base);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_max_rev_start_end_match_int64, i64);
compute!(compute_max_rev_start_end_match_int32, i32);
compute!(compute_max_rev_start_end_match_int16, i16);
compute!(compute_max_rev_start_end_match_int8, i8);
compute!(compute_max_rev_start_end_match_uint64, u64);
compute!(compute_max_rev_start_end_match_uint32, u32);
compute!(compute_max_rev_start_end_match_uint16, u16);
compute!(compute_max_rev_start_end_match_uint8, u8);
compute!(compute_max_rev_start_end_match_f64, f64);
compute!(compute_max_rev_start_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::compute_max_rev_start_end_match_int64;

    #[test]
    fn touched_row_positions_are_emitted_ascending_with_winning_row_index() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [1, 4, 9]; row0 (0..2) walks index[0..2] = {1, 4}
            // (tape width 2), row1 (1..3) walks index[1..3] = {4, 9}
            // (tape width 2) -- total tape width 4, all matching. Row1's
            // value (5) beats row0's (2) at the shared position 4.
            let arr = PyArray1::from_vec(py, vec![2_i64, 5]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3]);
            let index = PyArray1::from_vec(py, vec![1_i64, 4, 9]);
            let counts = PyArray1::from_vec(py, vec![1_i64, 1]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let (indexers, result) = compute_max_rev_start_end_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
                10,
            )
            .expect("valid equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![1, 4, 9]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0, 1, 1]);
        });
    }
}
