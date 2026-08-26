use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, checked_range, ensure_equal_lengths};
use std::collections::HashMap;

/// Accumulate integer values for the indirect ranges in `positions`.
///
/// ELI5: the HashMap stores only a small slot number for each label. The
/// actual labels and totals live side-by-side in Vecs, so duplicate labels do
/// not require a separate hash entry or a second lookup when producing the
/// result. Integer totals use wrapping arithmetic, so overflow has the same
/// deterministic result in debug and release builds.
pub fn sum_positions_int_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    capacity: usize,
    to_i64: F,
) -> (Vec<i64>, Vec<i64>)
where
    T: Copy,
    F: Fn(T) -> i64,
{
    let capacity = capacity.min(index.len()).min(positions.len());
    let mut slots: HashMap<i64, usize> = HashMap::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut totals = Vec::with_capacity(capacity);

    for (current, start, end, boolean) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    ) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let current_ = to_i64(*current);
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let label = index[indexer_];
            let slot = match slots.entry(label) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    totals.push(0_i64);
                    slot
                }
            };
            if !*boolean {
                // ELI5: once a label's bucket is found, add in the same
                // wraparound style as the forward kernel. This keeps a very
                // large integer from panicking only in debug/test builds.
                totals[slot] = totals[slot].wrapping_add(current_);
            }
        }
    }

    (labels, totals)
}

/// Accumulate floating-point values while retaining pandas-style compensated
/// summation. The labels and both pieces of state use the same compact slot.
///
/// ELI5: instead of looking up a label in two dictionaries, we find its slot
/// once and update the total and its rounding-error correction in Vecs.
pub fn sum_positions_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    capacity: usize,
    to_f64: F,
) -> (Vec<i64>, Vec<f64>)
where
    T: Copy,
    F: Fn(T) -> f64,
{
    let capacity = capacity.min(index.len()).min(positions.len());
    let mut slots: HashMap<i64, usize> = HashMap::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut totals = Vec::with_capacity(capacity);
    let mut compensations = Vec::with_capacity(capacity);

    for (current, start, end, boolean) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    ) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let current_ = to_f64(*current);
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let label = index[indexer_];
            let slot = match slots.entry(label) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    totals.push(0.);
                    compensations.push(0.);
                    slot
                }
            };
            if *boolean {
                continue;
            }
            let difference = current_ - compensations[slot];
            let increment = totals[slot] + difference;
            compensations[slot] = (increment - totals[slot]) - difference;
            // Adapted from pandas' cython code. Infinite values should not
            // turn the compensation term into NaN and poison later sums.
            if !compensations[slot].is_finite() {
                compensations[slot] = 0.;
            }
            totals[slot] = increment;
        }
    }

    (labels, totals)
}

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
            let (labels, totals) = sum_positions_int_core(
                arr,
                starts,
                ends,
                index,
                positions,
                booleans,
                index.len().min(positions.len()),
                |value| value as i64,
            );
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(totals);
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
            if arr.is_empty() || index.is_empty() || positions.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr, starts, ends, booleans, index, and positions cannot be empty",
                ));
            }
            let (labels, totals) = sum_positions_float_core(
                arr,
                starts,
                ends,
                index,
                positions,
                booleans,
                index.len().min(positions.len()),
                |value| value as f64,
            );
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(totals);
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
mod tests {
    use super::{sum_positions_float_core, sum_positions_int_core};
    use numpy::ndarray::array;

    #[test]
    fn integer_positions_keep_first_seen_labels_and_skip_invalid_entries() {
        let arr = array![10_i64, 20, 30];
        let starts = array![0_i64, 2, 4];
        let ends = array![2_i64, 4, 5];
        let index = array![100_i64, 200, 300];
        let positions = array![1_i64, 0, 1, 2, -1];
        let booleans = array![false, true, false];

        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            100,
            |value| value,
        );

        assert_eq!(labels, vec![200, 100, 300]);
        // The null row still creates slots, but contributes no value.
        assert_eq!(totals, vec![10, 10, 0]);
    }

    #[test]
    fn integer_positions_skip_empty_and_inverted_ranges() {
        let arr = array![7_i64, 11];
        let starts = array![1_i64, 3];
        let ends = array![1_i64, 2];
        let index = array![42_i64];
        let positions = array![0_i64];
        let booleans = array![false, false];

        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            1,
            |value| value,
        );

        assert!(labels.is_empty());
        assert!(totals.is_empty());
    }

    #[test]
    fn floating_positions_use_compensated_sum_for_duplicate_labels() {
        let arr = array![0.1_f64, 0.2, 0.3];
        let starts = array![0_i64, 1, 2];
        let ends = array![1_i64, 2, 3];
        let index = array![5_i64];
        let positions = array![0_i64, 0, 0];
        let booleans = array![false, false, false];

        let (labels, totals) = sum_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            1,
            |value| value,
        );

        assert_eq!(labels, vec![5]);
        assert!((totals[0] - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn integer_positions_wrap_on_overflow() {
        let arr = array![i64::MAX, 1];
        let starts = array![0_i64, 1];
        let ends = array![1_i64, 2];
        let index = array![5_i64];
        let positions = array![0_i64, 0];
        let booleans = array![false, false];

        let (labels, totals) = sum_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            1,
            |value| value,
        );

        assert_eq!(labels, vec![5]);
        assert_eq!(totals, vec![i64::MIN]);
    }
}
