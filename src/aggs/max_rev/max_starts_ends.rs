use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_equal_lengths_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};

/// Find the row containing the maximum value for each distinct right label.
///
/// janitor-rs is primarily called by pyjanitor. Its conditional-join path
/// resets the right DataFrame index to unique row labels before sorting or
/// filtering, so labels can be reordered or gapped but are not duplicated.
/// `item`, the ordinal position in `index`, is the state slot; `index[item]`
/// is the output label. `starts` and `ends` describe half-open ranges
/// `[start, end)`. Invalid or zero-width ranges are skipped by `checked_range`.
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
pub fn max_rev_start_end_core<T: PartialOrd + Copy>(
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
    let mut total_width = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start, end)) = checked_range(*start, *end, index.len()) {
            min_start = min_start.min(start);
            max_end = max_end.max(end);
            total_width = total_width.saturating_add(end - start);
        }
    }
    let width = max_end.saturating_sub(min_start);
    // ELI5: when the requested ranges cover enough of the right-hand domain,
    // a numbered row of drawers is cheaper than repeatedly looking positions
    // up in a dictionary. `total_width` is the estimated amount of work; it
    // is intentionally only a dispatch hint, not the allocated width.
    let dense = should_use_dense_match_storage(index.len(), total_width);
    if dense {
        let mut seen = vec![false; width];
        let mut states = vec![(arr[0], -1_i64); width];
        for (row, (current, (start, end), boolean)) in
            izip!(arr.iter(), starts.iter().zip(ends.iter()), booleans.iter()).enumerate()
        {
            let Some((start, end)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            let current = *current;
            let boolean = *boolean;
            for item in start..end {
                // `checked_range` guarantees `item >= start`, and the
                // pre-pass guarantees `min_start <= start`. Therefore this
                // unsigned subtraction cannot underflow: it translates an
                // absolute right-row position into the compact domain.
                let slot = item - min_start;
                seen[slot] = true;
                if boolean {
                    continue;
                }
                if states[slot].1 == -1 || current > states[slot].0 {
                    states[slot] = (current, row as i64);
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
    for (row, (current, (start, end), boolean)) in
        izip!(arr.iter(), starts.iter().zip(ends.iter()), booleans.iter()).enumerate()
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current = *current;
        let boolean = *boolean;
        for item in start..end {
            // `min_start` is the smallest validated start, so every valid
            // item is at least `min_start`; subtraction is a safe compact-slot
            // translation rather than an unchecked signed-boundary operation.
            let slot = item - min_start;
            let state = states.entry(slot).or_insert((current, -1));

            if boolean {
                continue;
            }

            if state.1 == -1 || current > state.0 {
                *state = (current, row as i64);
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
        /// Finds the row containing the maximum value for each right-side
        /// label covered by the reverse interval ranges.
        ///
        /// Null rows, identified by `booleans`, do not participate in the
        /// maximum. The returned positions use `-1` when no non-null row
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
            let (indexers, result) = max_rev_start_end_core(arr, starts, ends, index, booleans)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(indexers).into_pyarray(py),
                Array1::from_vec(result).into_pyarray(py),
            ))
        }
    };
}

compute!(compute_max_rev_start_end_int64, i64);
compute!(compute_max_rev_start_end_int32, i32);
compute!(compute_max_rev_start_end_int16, i16);
compute!(compute_max_rev_start_end_int8, i8);
compute!(compute_max_rev_start_end_uint64, u64);
compute!(compute_max_rev_start_end_uint32, u32);
compute!(compute_max_rev_start_end_uint16, u16);
compute!(compute_max_rev_start_end_uint8, u8);
compute!(compute_max_rev_start_end_f64, f64);
compute!(compute_max_rev_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every department's
/// exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn dense_slots_find_maximum_for_unique_gapped_labels() {
        let got = max_rev_start_end_core(
            array![5_i64, 2, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![42_i64, 7, 100].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((vec![42, 7, 100], vec![0, 0, 1])));
    }

    #[test]
    fn duplicate_index_labels_are_explicitly_unsupported() {
        let got = max_rev_start_end_core(
            array![5_i64, 2, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
        );
        // Ordinal slots are independent; duplicate labels are not merged.
        assert_eq!(got, Ok((vec![10, 20, 10], vec![0, 0, 1])));
    }

    #[test]
    fn null_rows_emit_labels_but_not_maximum_rows() {
        let got = max_rev_start_end_core(
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
        let got = max_rev_start_end_core(
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
        assert!(max_rev_start_end_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false].view(),
        )
        .is_err());
        assert!(max_rev_start_end_core(
            (array![] as Array1<i64>).view(),
            array![].view(),
            array![].view(),
            array![10_i64].view(),
            array![].view(),
        )
        .is_err());
        assert!(max_rev_start_end_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![].view(),
            array![false].view(),
        )
        .is_err());
    }
}
