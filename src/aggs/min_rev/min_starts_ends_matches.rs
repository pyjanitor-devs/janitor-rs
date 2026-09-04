use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};
use std::collections::HashMap;

/// Finds the row containing the minimum value for each right-side label that
/// survives the reverse interval-range match tape.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive ordinal start of each interval.
/// * `ends` - Exclusive ordinal end of each interval.
/// * `index` - Right-side labels in ordinal position order.
/// * `counts` - Number of matching candidates for each row.
/// * `matches` - Flat per-candidate match mask; must have the exact tape width.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn min_rev_start_end_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("matches", matches.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;

    let mut expected = 0_usize;
    let mut min_start = index.len();
    let mut max_end = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start_, end_)) = checked_range(*start, *end, index.len()) {
            expected += end_ - start_;
            min_start = min_start.min(start_);
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected, matches.len())?;
    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut tape = 0_usize;

    if dense {
        let mut seen = vec![false; width];
        let mut states = vec![(arr[0], -1_i64); width];
        for (row, (current, start, end, count, boolean)) in izip!(
            arr.iter(),
            starts.iter(),
            ends.iter(),
            counts.iter(),
            booleans.iter()
        )
        .enumerate()
        {
            let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            for item in start_..end_ {
                if matches[tape] == 0 {
                    tape += 1;
                    continue;
                }
                let slot = item - min_start;
                seen[slot] = true;
                tape += 1;
                if *boolean || *count == 0 {
                    continue;
                }
                if states[slot].1 == -1 || *current < states[slot].0 {
                    states[slot] = (*current, row as i64);
                }
            }
        }
        let mut labels = Vec::new();
        let mut result = Vec::new();
        for (slot, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[min_start + slot]);
                result.push(states[slot].1);
            }
        }
        return Ok((labels, result));
    }

    let mut states: HashMap<usize, (T, i64)> = HashMap::new();
    for (row, (current, start, end, count, boolean)) in izip!(
        arr.iter(),
        starts.iter(),
        ends.iter(),
        counts.iter(),
        booleans.iter()
    )
    .enumerate()
    {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[tape] == 0 {
                tape += 1;
                continue;
            }
            let slot = item - min_start;
            let state = states.entry(slot).or_insert((*current, -1));
            tape += 1;
            if *boolean || *count == 0 {
                continue;
            }
            if state.1 == -1 || *current < state.0 {
                *state = (*current, row as i64);
            }
        }
    }
    let mut labels = Vec::with_capacity(states.len());
    let mut result = Vec::with_capacity(states.len());
    for (slot, (_, best_position)) in states {
        labels.push(index[min_start + slot]);
        result.push(best_position);
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
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive ordinal start of each interval.
        /// * `ends` - Exclusive ordinal end of each interval.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `counts` - Number of matching candidates for each row.
        /// * `matches` - Flat per-candidate match mask.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
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
            let (indexers, result) = min_rev_start_end_match_core(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                index.as_array(),
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(indexers).into_pyarray(py),
                Array1::from_vec(result).into_pyarray(py),
            ))
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

#[cfg(test)]
mod precedence_tests {
    use super::min_rev_start_end_match_core;
    use numpy::ndarray::{array, Array1};

    #[test]
    fn empty_input_precedes_shape_mismatch() {
        let error = min_rev_start_end_match_core(
            Array1::<i64>::zeros(0).view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![1_i64].view(),
            array![1_i8].view(),
            array![false].view(),
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");
    }
}
