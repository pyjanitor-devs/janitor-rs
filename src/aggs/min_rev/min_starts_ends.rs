use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_equal_lengths_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};

/// Find the row containing the minimum value for each distinct right label.
///
/// janitor-rs is primarily called by pyjanitor. Its conditional-join path
/// resets the right DataFrame index to unique row labels before sorting or
/// filtering, so labels can be reordered or gapped but are not duplicated.
/// `item`, the ordinal position in `index`, is the state slot; `index[item]`
/// is the output label. `starts` and `ends` describe half-open ranges
/// `[start, end)`. Invalid or zero-width ranges are skipped by `checked_range`,
/// and invalid or zero-width ranges are skipped.
///
/// # Preconditions
///
/// `index` must contain unique labels in positional order. pyjanitor provides
/// this by normalizing the right side to `range(len(right))`; direct callers
/// must provide it themselves. Duplicate labels are unsupported and are not
/// merged by the positional accumulator.
///
/// `arr`, `starts`, `ends`, `index`, `booleans` cannot be empty.
///
/// ELI5: each right-row position gets one numbered drawer. The number printed
/// on the row may be 42, 7, or 100, but the drawer is still its position on
/// the shelf. We remember the best left row in that drawer.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive ordinal start of each interval.
/// * `ends` - Exclusive ordinal end of each interval.
/// * `index` - Right-side labels in ordinal position order.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn min_rev_start_end_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;

    let mut min_start = index.len();
    let mut max_end = 0_usize;
    for (&start, &end) in starts.iter().zip(ends.iter()) {
        if let Some((start, end)) = checked_range(start, end, index.len()) {
            min_start = min_start.min(start);
            max_end = max_end.max(end);
        }
    }
    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut touched = if dense {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };

    if dense {
        let mut seen = vec![false; width];
        let mut states = vec![(arr[0], -1_i64); width];
        for (row, (current, start, end, boolean)) in
            izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter()).enumerate()
        {
            let Some((start, end)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            for item in start..end {
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                if *boolean {
                    continue;
                }
                if states[slot].1 == -1 || *current < states[slot].0 {
                    states[slot] = (*current, row as i64);
                }
            }
        }

        let mut result = Vec::with_capacity(touched.len());
        let mut labels = Vec::with_capacity(touched.len());
        for slot in touched {
            result.push(states[slot].1);
            labels.push(index[min_start + slot]);
        }
        return Ok((labels, result));
    }

    let mut states: HashMap<usize, (T, i64)> = HashMap::new();
    for (row, (current, start, end, boolean)) in
        izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter()).enumerate()
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start..end {
            let slot = item - min_start;
            let state = states.entry(slot).or_insert_with(|| {
                touched.push(slot);
                (*current, -1)
            });
            if *boolean {
                continue;
            }
            if state.1 == -1 || *current < state.0 {
                *state = (*current, row as i64);
            }
        }
    }
    let mut result = Vec::with_capacity(touched.len());
    let mut labels = Vec::with_capacity(touched.len());
    for slot in touched {
        result.push(states[&slot].1);
        labels.push(index[min_start + slot]);
    }
    Ok((labels, result))
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// Finds the row containing the minimum value for each right-side
        /// label covered by the reverse interval ranges.
        ///
        /// Null rows, identified by `booleans`, do not participate in the
        /// minimum. The returned positions use `-1` when no non-null row
        /// covers a label.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive ordinal start of each interval.
        /// * `ends` - Exclusive ordinal end of each interval.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        ///
        /// `index` must contain unique labels. Positions in the array are
        /// ordinal state slots; direct callers must preserve that contract.
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
            Ok((
                Array1::from_vec(indexers).into_pyarray(py),
                Array1::from_vec(result).into_pyarray(py),
            ))
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
    fn dense_slots_find_minimum_for_unique_gapped_labels() {
        let got = min_rev_start_end_core(
            array![5_i64, 2, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![42_i64, 7, 100].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((vec![42, 7, 100], vec![2, 1, 1])));
    }

    #[test]
    fn duplicate_index_labels_are_explicitly_unsupported() {
        let got = min_rev_start_end_core(
            array![5_i64, 2, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
        );
        // Ordinal slots are independent; duplicate labels are not merged.
        assert_eq!(got, Ok((vec![10, 20, 10], vec![2, 1, 1])));
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
        assert_eq!(got, Ok((vec![10, 20], vec![-1, -1])));
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
        assert_eq!(got, Ok((vec![], vec![])));
    }

    #[test]
    fn validation_rejects_shape_mismatches() {
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
        assert!(min_rev_start_end_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![].view(),
            array![false].view(),
        )
        .is_err());
    }
}
