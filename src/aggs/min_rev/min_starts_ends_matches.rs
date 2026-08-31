use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches,
};
use std::collections::HashMap;

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// `matches` must be non-empty and must contain exactly one entry for
        /// every candidate position. pyjanitor supplies the per-row counts
        /// and binary mask from the same comparison stage. pyjanitor is
        /// responsible for ensuring each mask value is 0 or 1; Rust does not
        /// scan the tape to enforce that value-level contract. Normally
        /// `counts_array.sum() == matches.sum()`, while `matches.len()` is the
        /// full candidate-tape width.
        #[allow(clippy::too_many_arguments)]
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
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(index.len());
            let mut mapping: HashMap<i64, $type> = HashMap::with_capacity(index.len());
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
                    let base = dictionary.entry(pos).or_insert(-1);
                    let base_val = mapping.entry(pos).or_insert(*current);
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    if (*base == -1) || (*current < *base_val) {
                        *base_val = *current;
                        *base = posn as i64;
                    }
                    n += 1;
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

compute!(compute_min_rev_start_end_match_int64, i64);
compute!(compute_min_rev_start_end_match_int32, i32);
compute!(compute_min_rev_start_end_match_int16, i16);
compute!(compute_min_rev_start_end_match_int8, i8);
compute!(compute_min_rev_start_end_match_uint64, u64);
compute!(compute_min_rev_start_end_match_uint32, u32);
compute!(compute_min_rev_start_end_match_uint16, u16);
compute!(compute_min_rev_start_end_match_uint8, u8);
compute!(compute_min_rev_start_end_match_f64, f64);
compute!(compute_min_rev_start_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_match_f64, m)?)?;
    Ok(())
}
