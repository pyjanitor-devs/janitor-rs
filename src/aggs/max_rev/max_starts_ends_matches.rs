use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches,
};

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// `matches` must be non-empty and must contain exactly one entry for
        /// every candidate position. pyjanitor supplies the per-row counts
        /// and binary mask from the same comparison stage. pyjanitor is
        /// responsible for ensuring each mask value is 0 or 1; Rust does not
        /// scan the tape to enforce that value-level contract. Normally
        /// `counts_array.sum() == matches.sum()`, while `matches.len()` is the
        /// full candidate-tape width.
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
            ensure_nonempty_matches(matches.len())?;
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
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            // The fold below finds the smallest enclosing valid slice.
            // ELI5: it stretches one ruler just enough to cover every range.
            // Ranges point directly at right-side rows, so each row in
            // the smallest enclosing slice gets a bucket. `seen` preserves a
            // matched label even when its row is null or has zero count.
            let (min_start, max_end) = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| checked_range(*s, *e, index.len()))
                .fold((usize::MAX, 0), |(min_start, max_end), (start, end)| {
                    (min_start.min(start), max_end.max(end))
                });
            let width = max_end.saturating_sub(min_start);
            let mut seen = vec![false; width];
            let mut touched = Vec::new();
            let mut best_positions = vec![-1_i64; width];
            let mut best_values: Vec<Option<$type>> = vec![None; width];
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
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                        best_values[slot] = Some(*current);
                    }
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    if (best_positions[slot] == -1)
                        || (*current > best_values[slot].expect("seen slot has a value"))
                    {
                        best_values[slot] = Some(*current);
                        best_positions[slot] = posn as i64;
                    }
                    n += 1;
                }
            }
            let indexers = Array1::from_iter(touched.iter().map(|&slot| index[min_start + slot]));
            let result = Array1::from_iter(touched.iter().map(|&slot| best_positions[slot]));
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
mod tests {
    use super::compute_max_rev_start_end_match_int64;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::Python;

    #[test]
    fn wrapper_rejects_empty_matches_tape_for_zero_width_range() {
        Python::initialize();
        Python::attach(|py| {
            let arr = PyArray1::from_vec(py, vec![5_i64]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![0_i64]);
            let index = PyArray1::from_vec(py, vec![10_i64]);
            let counts = PyArray1::from_vec(py, vec![0_i64]);
            let matches = PyArray1::from_vec(py, Vec::<i8>::new());
            let booleans = PyArray1::from_vec(py, vec![false]);

            let error = compute_max_rev_start_end_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap_err();

            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("matches cannot be empty"));
        });
    }
}
