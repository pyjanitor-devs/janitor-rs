use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_index, checked_range, ensure_equal_lengths_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};
use std::collections::{hash_map::Entry, HashMap};

/// Find the minimum value for each label reached through `positions`.
/// `index` is expected to contain unique labels, as guaranteed by the
/// pyjanitor producer for this path.
///
/// ELI5: `positions` already gives us a validated ordinal into `index`, so the
/// HashMap can use that ordinal directly. We only look up the original label
/// while emitting the result; a null row can create state but cannot win. The
/// returned label/state pairs follow HashMap iteration order.
pub fn min_positions_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let dense = should_use_dense_match_storage(index.len(), positions.len());
    Ok(min_positions_core_with_storage(
        arr, starts, ends, index, positions, booleans, dense,
    ))
}

/// Run minimum positional aggregation with an explicit storage mode.
///
/// This Rust-only entry point is used by benchmarks; production callers should
/// use [`min_positions_core`] so the dense/sparse heuristic remains automatic.
pub fn min_positions_core_with_storage<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    dense: bool,
) -> (Vec<i64>, Vec<i64>) {
    // `positions.len()` is only a cheap upper-bound heuristic here: it counts
    // tape entries, not distinct ordinals. Dense storage trades memory for
    // direct indexing, while the sparse fallback remains safer for short
    // tapes. A repeated-ordinal tape can still make this heuristic overselect
    // dense storage; adaptive promotion is a future refinement.
    if dense {
        let mut seen = vec![false; index.len()];
        let mut states = vec![(arr[0], -1_i64); index.len()];
        for (posn, (current, start, end, boolean)) in izip!(
            arr.into_iter(),
            starts.into_iter(),
            ends.into_iter(),
            booleans.into_iter()
        )
        .enumerate()
        {
            let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                continue;
            };
            for nn in start_..end_ {
                let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                    continue;
                };
                seen[indexer_] = true;
                let state = &mut states[indexer_];
                // Insert state before checking the mask: null-only labels
                // must still be emitted with the missing-position sentinel.
                if !*boolean && (state.1 == -1 || *current < state.0) {
                    *state = (*current, posn as i64);
                }
            }
        }
        let mut labels = Vec::new();
        let mut best_positions = Vec::new();
        for (ordinal, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[ordinal]);
                best_positions.push(states[ordinal].1);
            }
        }
        return (labels, best_positions);
    }

    let mut states: HashMap<usize, (T, i64)> = HashMap::new();

    for (posn, (current, start, end, boolean)) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    )
    .enumerate()
    {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let state = match states.entry(indexer_) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert((*current, -1)),
            };
            // Insert state before checking the mask: null-only labels must
            // still be emitted with the missing-position sentinel.
            if !*boolean && (state.1 == -1 || *current < state.0) {
                *state = (*current, posn as i64);
            }
        }
    }
    let mut labels = Vec::with_capacity(states.len());
    let mut best_positions = Vec::with_capacity(states.len());
    for (ordinal, (_, best_position)) in states {
        labels.push(index[ordinal]);
        best_positions.push(best_position);
    }
    (labels, best_positions)
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// Find the minimum contributing row position for each label reached
        /// through the positional candidate tape.
        ///
        /// # Arguments
        /// * `arr` - Left-side values; must not be empty.
        /// * `starts` - Inclusive positional range starts.
        /// * `ends` - Exclusive positional range ends.
        /// * `index` - Right-side labels addressed by `positions`.
        /// * `positions` - Positional candidate tape.
        /// * `booleans` - Null mask; `True` rows are ignored.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let (labels, best_positions) =
                min_positions_core(arr, starts, ends, index, positions, booleans)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(best_positions);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_min_rev_positions_int64, i64);
compute!(compute_min_rev_positions_int32, i32);
compute!(compute_min_rev_positions_int16, i16);
compute!(compute_min_rev_positions_int8, i8);
compute!(compute_min_rev_positions_uint64, u64);
compute!(compute_min_rev_positions_uint32, u32);
compute!(compute_min_rev_positions_uint16, u16);
compute!(compute_min_rev_positions_uint8, u8);
compute!(compute_min_rev_positions_f64, f64);
compute!(compute_min_rev_positions_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::min_positions_core;
    use numpy::ndarray::array;

    #[test]
    fn positions_keep_labels_and_minimum_rows() {
        let arr = array![5_i64, 2, 8];
        let starts = array![0_i64, 2, 4];
        let ends = array![2_i64, 4, 5];
        let index = array![10_i64, 20, 30];
        let positions = array![0_i64, 1, 2, 1, -1];
        let booleans = array![false, false, false];

        let (labels, rows) = min_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
        )
        .unwrap();

        let mut got: Vec<_> = labels.into_iter().zip(rows).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(10, 0), (20, 1), (30, 1)]);
    }

    #[test]
    fn null_rows_create_labels_but_do_not_select_a_minimum() {
        let arr = array![7_i64, 3];
        let starts = array![0_i64, 1];
        let ends = array![1_i64, 2];
        let index = array![4_i64];
        let positions = array![0_i64, 0];
        let booleans = array![true, false];

        let (labels, rows) = min_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
        )
        .unwrap();

        assert_eq!(labels, vec![4]);
        assert_eq!(rows, vec![1]);
    }
}
