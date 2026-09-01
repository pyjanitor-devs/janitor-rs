use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::{hash_map::Entry, HashMap};

use crate::aggs::{
    checked_range, ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};

pub fn max_rev_start_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_nonempty_core("matches", matches.len())?;

    let mut expected = 0_usize;
    let mut min_start = index.len();
    for start in starts.iter() {
        if let Some((start_, end_)) = checked_range(*start, index.len() as i64, index.len()) {
            expected += end_ - start_;
            min_start = min_start.min(start_);
        }
    }
    ensure_exact_tape_width_core(expected, matches.len())?;
    let width = index.len().saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut touched = Vec::with_capacity(width);
    let mut tape = 0_usize;

    if dense {
        let mut seen = vec![false; width];
        let mut states = vec![(arr[0], -1_i64); width];
        for (row, (current, start, count, boolean)) in
            izip!(arr.iter(), starts.iter(), counts.iter(), booleans.iter()).enumerate()
        {
            let Some((start_, end_)) = checked_range(*start, index.len() as i64, index.len())
            else {
                continue;
            };
            for item in start_..end_ {
                if matches[tape] == 0 {
                    tape += 1;
                    continue;
                }
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                tape += 1;
                if *boolean || *count == 0 {
                    continue;
                }
                if states[slot].1 == -1 || *current > states[slot].0 {
                    states[slot] = (*current, row as i64);
                }
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut result = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            result.push(states[slot].1);
        }
        return Ok((labels, result));
    }

    let mut states: HashMap<usize, (T, i64)> = HashMap::with_capacity(width);
    for (row, (current, start, count, boolean)) in
        izip!(arr.iter(), starts.iter(), counts.iter(), booleans.iter()).enumerate()
    {
        let Some((start_, end_)) = checked_range(*start, index.len() as i64, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[tape] == 0 {
                tape += 1;
                continue;
            }
            let slot = item - min_start;
            if let Entry::Vacant(entry) = states.entry(slot) {
                touched.push(slot);
                entry.insert((*current, -1));
            }
            tape += 1;
            if *boolean || *count == 0 {
                continue;
            }
            let state = states.get_mut(&slot).expect("inserted above");
            if state.1 == -1 || *current > state.0 {
                *state = (*current, row as i64);
            }
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut result = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        result.push(states[&slot].1);
    }
    Ok((labels, result))
}

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
            counts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let matches = matches.as_array();
            let counts = counts.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let (indexers, result) =
                max_rev_start_match_core(arr, starts, counts, index, matches, booleans)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(indexers).into_pyarray(py),
                Array1::from_vec(result).into_pyarray(py),
            ))
        }
    };
}

compute!(compute_max_rev_start_match_int64, i64);
compute!(compute_max_rev_start_match_int32, i32);
compute!(compute_max_rev_start_match_int16, i16);
compute!(compute_max_rev_start_match_int8, i8);
compute!(compute_max_rev_start_match_uint64, u64);
compute!(compute_max_rev_start_match_uint32, u32);
compute!(compute_max_rev_start_match_uint16, u16);
compute!(compute_max_rev_start_match_uint8, u8);
compute!(compute_max_rev_start_match_f64, f64);
compute!(compute_max_rev_start_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_f64, m)?)?;
    Ok(())
}
