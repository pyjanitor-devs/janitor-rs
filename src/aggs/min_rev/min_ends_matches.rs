use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_range, ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};

pub fn min_rev_end_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    index: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_nonempty_core("matches", matches.len())?;
    let mut expected = 0_usize;
    let mut max_end = 0_usize;
    for end in ends.iter() {
        if let Some((_, end_)) = checked_range(0, *end, index.len()) {
            expected += end_;
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected, matches.len())?;
    let dense = should_use_dense_match_storage(index.len(), max_end);
    let mut touched = Vec::with_capacity(max_end);
    let mut tape = 0_usize;
    if dense {
        let mut seen = vec![false; max_end];
        let mut states = vec![(arr[0], -1_i64); max_end];
        for (row, (current, end, count, boolean)) in
            izip!(arr.iter(), ends.iter(), counts.iter(), booleans.iter()).enumerate()
        {
            let Some((_, end_)) = checked_range(0, *end, index.len()) else {
                continue;
            };
            for item in 0..end_ {
                if matches[tape] == 0 {
                    tape += 1;
                    continue;
                }
                if !seen[item] {
                    seen[item] = true;
                    touched.push(item);
                }
                tape += 1;
                if *boolean || *count == 0 {
                    continue;
                }
                if states[item].1 == -1 || *current < states[item].0 {
                    states[item] = (*current, row as i64);
                }
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut result = Vec::with_capacity(touched.len());
        for item in touched {
            labels.push(index[item]);
            result.push(states[item].1);
        }
        return Ok((labels, result));
    }
    let mut states: HashMap<usize, (T, i64)> = HashMap::with_capacity(max_end);
    for (row, (current, end, count, boolean)) in
        izip!(arr.iter(), ends.iter(), counts.iter(), booleans.iter()).enumerate()
    {
        let Some((_, end_)) = checked_range(0, *end, index.len()) else {
            continue;
        };
        for item in 0..end_ {
            if matches[tape] == 0 {
                tape += 1;
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = states.entry(item) {
                touched.push(item);
                entry.insert((*current, -1));
            }
            tape += 1;
            if *boolean || *count == 0 {
                continue;
            }
            let state = states.get_mut(&item).expect("inserted above");
            if state.1 == -1 || *current < state.0 {
                *state = (*current, row as i64);
            }
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut result = Vec::with_capacity(touched.len());
    for item in touched {
        labels.push(index[item]);
        result.push(states[&item].1);
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
            index: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let index = index.as_array();
            let ends = ends.as_array();
            let matches = matches.as_array();
            let counts = counts.as_array();
            let booleans = booleans.as_array();
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let (indexers, result) =
                min_rev_end_match_core(arr, index, ends, counts, matches, booleans)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(indexers).into_pyarray(py),
                Array1::from_vec(result).into_pyarray(py),
            ))
        }
    };
}

compute!(compute_min_rev_end_match_int64, i64);
compute!(compute_min_rev_end_match_int32, i32);
compute!(compute_min_rev_end_match_int16, i16);
compute!(compute_min_rev_end_match_int8, i8);
compute!(compute_min_rev_end_match_uint64, u64);
compute!(compute_min_rev_end_match_uint32, u32);
compute!(compute_min_rev_end_match_uint16, u16);
compute!(compute_min_rev_end_match_uint8, u8);
compute!(compute_min_rev_end_match_f64, f64);
compute!(compute_min_rev_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_f64, m)?)?;
    Ok(())
}
