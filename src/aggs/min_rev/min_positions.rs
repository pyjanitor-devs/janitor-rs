use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, checked_range, ensure_equal_lengths};
use std::collections::{hash_map::Entry, HashMap};

/// Find the minimum value for each label reached through `positions`.
///
/// ELI5: each label gets a small slot number. The label, best row position,
/// and best value are kept in Vecs, while the HashMap only translates a label
/// to its slot. A null row can create a label, but cannot become its winner.
pub fn min_positions_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    capacity: usize,
) -> (Vec<i64>, Vec<i64>) {
    let capacity = capacity.min(index.len()).min(positions.len());
    let mut slots: HashMap<i64, usize> = HashMap::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut best_positions = Vec::with_capacity(capacity);
    let mut best_values = Vec::with_capacity(capacity);

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
            let label = index[indexer_];
            let slot = match slots.entry(label) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    best_positions.push(-1);
                    best_values.push(*current);
                    slot
                }
            };
            if *boolean {
                continue;
            }
            if (best_positions[slot] == -1) || (*current < best_values[slot]) {
                best_values[slot] = *current;
                best_positions[slot] = posn as i64;
            }
        }
    }

    (labels, best_positions)
}

macro_rules! compute {
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
            if arr.is_empty() || index.is_empty() || positions.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr, starts, ends, booleans, index, and positions cannot be empty",
                ));
            }
            let (labels, best_positions) = min_positions_core(
                arr,
                starts,
                ends,
                index,
                positions,
                booleans,
                index.len().min(positions.len()),
            );
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
    fn positions_keep_first_seen_labels_and_minimum_rows() {
        let arr = array![5_i64, 2, 8];
        let starts = array![0_i64, 2, 4];
        let ends = array![2_i64, 4, 5];
        let index = array![10_i64, 20, 10];
        let positions = array![0_i64, 1, 2, 1, -1];
        let booleans = array![false, false, false];

        let (labels, rows) = min_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            100,
        );

        assert_eq!(labels, vec![10, 20]);
        assert_eq!(rows, vec![1, 1]);
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
            1,
        );

        assert_eq!(labels, vec![4]);
        assert_eq!(rows, vec![1]);
    }
}
