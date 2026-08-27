use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths};
use std::collections::{hash_map::Entry, HashMap};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if starts.len() != ends.len() {
        return Err("starts and ends must have equal lengths");
    }
    if arr.len() != starts.len() {
        return Err("arr, starts, and ends must have equal lengths");
    }
    if arr.len() != booleans.len() {
        return Err("arr and booleans must have equal lengths");
    }
    if arr.is_empty() || index.is_empty() {
        return Err("arr, starts, booleans, and index cannot be empty");
    }
    Ok(())
}

fn capacity_hint(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    right_len: usize,
) -> usize {
    // ELI5: every covered index position can introduce at most one new label.
    // This gives the map a useful upper bound without trusting a Python-side
    // length value or reserving more buckets than the right index contains.
    starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| checked_range(*start, *end, right_len))
        .fold(0_usize, |total, (start, end)| {
            total.saturating_add(end - start)
        })
        .min(right_len)
}

/// Find the row containing the minimum value for each distinct right label.
///
/// ELI5: each label gets one numbered drawer. The drawer stores the best
/// row seen so far and its value; repeated labels reuse that drawer instead
/// of maintaining two separate dictionaries.
pub fn min_rev_start_end_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, starts, ends, index, booleans)?;

    let hint = capacity_hint(starts, ends, index.len());
    let mut slots = HashMap::with_capacity(hint);
    let mut labels = Vec::new();
    let mut values = Vec::new();
    let mut rows = Vec::new();

    for (row, (current, start, end, boolean)) in
        izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter()).enumerate()
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start..end {
            let label = index[item];
            let slot = match slots.entry(label) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    values.push(*current);
                    rows.push(-1_i64);
                    slot
                }
            };
            if *boolean {
                continue;
            }
            if rows[slot] == -1 || *current < values[slot] {
                values[slot] = *current;
                rows[slot] = row as i64;
            }
        }
    }

    Ok((Array1::from_vec(labels), Array1::from_vec(rows)))
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
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let (indexers, result) = min_rev_start_end_core(arr, starts, ends, index, booleans)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_min_rev_start_end_int64, i64);
compute!(compute_min_rev_start_end_int32, i32);
compute!(compute_min_rev_start_end_int16, i16);
compute!(compute_min_rev_start_end_int8, i8);
compute!(compute_min_rev_start_end_uint64, u64);
compute!(compute_min_rev_start_end_uint32, u32);
compute!(compute_min_rev_start_end_uint16, u16);
compute!(compute_min_rev_start_end_uint8, u8);
compute!(compute_min_rev_start_end_f64, f64);
compute!(compute_min_rev_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every department's
/// exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn compact_slots_find_minimum_for_duplicate_labels() {
        let got = min_rev_start_end_core(
            array![5_i64, 2, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((array![10, 20], array![1, 1])));
    }

    #[test]
    fn null_rows_emit_labels_but_not_minimum_rows() {
        let got = min_rev_start_end_core(
            array![5_i64].view(),
            array![0_i64].view(),
            array![2_i64].view(),
            array![10_i64, 20].view(),
            array![true].view(),
        );
        assert_eq!(got, Ok((array![10, 20], array![-1, -1])));
    }

    #[test]
    fn invalid_or_zero_width_ranges_are_skipped() {
        let got = min_rev_start_end_core(
            array![1_i64, 2, 3].view(),
            array![2_i64, -1, 1].view(),
            array![2_i64, 1, 4].view(),
            array![10_i64, 20].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((array![], array![])));
    }

    #[test]
    fn validation_rejects_shape_mismatches_and_empty_inputs() {
        assert!(min_rev_start_end_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false].view(),
        )
        .is_err());
        assert!(min_rev_start_end_core(
            (array![] as Array1<i64>).view(),
            array![].view(),
            array![].view(),
            array![10_i64].view(),
            array![].view(),
        )
        .is_err());
    }
}
