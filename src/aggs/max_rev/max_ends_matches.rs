use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_end, ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    should_use_dense_match_storage,
};

/// Aggregate the survivor tape by right-position ordinal.
///
/// `index[item]` is the label returned to Python; `item` is the state key.
/// Pyjanitor normalizes the right index to unique labels, but those labels are
/// not required to equal their positions or form a contiguous integer range.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `index` - Right-side labels in ordinal position order.
/// * `ends` - Exclusive ordinal end of each prefix range.
/// * `counts` - Number of matching candidates for each row.
/// * `matches` - Flat per-candidate match mask; must have the exact tape width.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn max_rev_end_match_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    index: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("matches", matches.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;

    let mut expected_matches_width = 0_usize;
    let mut max_end = 0_usize;
    for end in ends.iter() {
        if let Some(end_) = checked_end(*end, index.len()) {
            expected_matches_width += end_;
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())?;

    if should_use_dense_match_storage(index.len(), max_end) {
        let mut seen = vec![false; max_end];
        let mut states = vec![(arr[0], -1_i64); max_end];
        let mut touched = Vec::with_capacity(max_end);
        let mut tape = 0_usize;
        for (row, (current, end, count, boolean)) in
            izip!(arr.iter(), ends.iter(), counts.iter(), booleans.iter(),).enumerate()
        {
            let Some(end_) = checked_end(*end, index.len()) else {
                continue;
            };
            for item in 0..end_ {
                let tape_value = matches[tape];
                if tape_value == 0_i8 {
                    tape += 1;
                    continue;
                }
                if !seen[item] {
                    seen[item] = true;
                    touched.push(item);
                }
                tape += 1;
                if *boolean || *count == 0_i64 {
                    continue;
                }
                let state = &mut states[item];
                if state.1 == -1 || *current > state.0 {
                    *state = (*current, row as i64);
                }
            }
        }
        let mut indexers = Vec::with_capacity(touched.len());
        let mut result = Vec::with_capacity(touched.len());
        for item in touched {
            indexers.push(index[item]);
            result.push(states[item].1);
        }
        return Ok((indexers, result));
    }

    let mut states: HashMap<usize, (T, i64)> = HashMap::new();
    let mut touched = Vec::new();
    let mut tape = 0_usize;
    for (row, (current, end, count, boolean)) in
        izip!(arr.iter(), ends.iter(), counts.iter(), booleans.iter(),).enumerate()
    {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in 0..end_ {
            let tape_value = matches[tape];
            if tape_value == 0_i8 {
                tape += 1;
                continue;
            }
            let state = states.entry(item).or_insert_with(|| {
                touched.push(item);
                (*current, -1)
            });
            tape += 1;
            if *boolean || *count == 0_i64 {
                continue;
            }
            if state.1 == -1 || *current > state.0 {
                *state = (*current, row as i64);
            }
        }
    }
    let mut indexers = Vec::with_capacity(touched.len());
    let mut result = Vec::with_capacity(touched.len());
    for item in touched {
        indexers.push(index[item]);
        result.push(states[&item].1);
    }
    Ok((indexers, result))
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
        /// * `index` - Right-side labels in ordinal position order.
        /// * `ends` - Exclusive ordinal end of each prefix range.
        /// * `counts` - Number of matching candidates for each row.
        /// * `matches` - Flat per-candidate match mask.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
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
            let (indexers, result) =
                max_rev_end_match_core(arr, index, ends, counts, matches, booleans)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(indexers).into_pyarray(py),
                Array1::from_vec(result).into_pyarray(py),
            ))
        }
    };
}

compute!(compute_max_rev_end_match_int64, i64);
compute!(compute_max_rev_end_match_int32, i32);
compute!(compute_max_rev_end_match_int16, i16);
compute!(compute_max_rev_end_match_int8, i8);
compute!(compute_max_rev_end_match_uint64, u64);
compute!(compute_max_rev_end_match_uint32, u32);
compute!(compute_max_rev_end_match_uint16, u16);
compute!(compute_max_rev_end_match_uint8, u8);
compute!(compute_max_rev_end_match_f64, f64);
compute!(compute_max_rev_end_match_f32, f32);

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn ordinal_state_emits_unique_non_positional_labels() {
        let got = max_rev_end_match_core(
            array![5_i64, 9, 4].view(),
            array![100_i64, 20, 900].view(),
            array![3_i64, 2, 1].view(),
            array![2_i64, 1, 1].view(),
            array![1_i8, 1, 1, 1, 1, 1].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((vec![100, 20, 900], vec![1, 1, 0])));
    }

    #[test]
    fn null_and_zero_count_rows_create_labels_without_winners() {
        let got = max_rev_end_match_core(
            array![5_i64, 9].view(),
            array![100_i64, 20, 900].view(),
            array![3_i64, 2].view(),
            array![0_i64, 0].view(),
            array![1_i8, 1, 1, 1, 1].view(),
            array![true, false].view(),
        );
        assert_eq!(got, Ok((vec![100, 20, 900], vec![-1, -1, -1])));
    }

    #[test]
    fn dead_tape_entries_are_skipped_before_state_work() {
        let got = max_rev_end_match_core(
            array![5_i64].view(),
            array![100_i64, 20].view(),
            array![2_i64].view(),
            array![1_i64].view(),
            array![0_i8, 1].view(),
            array![false].view(),
        );
        assert_eq!(got, Ok((vec![20], vec![0])));
    }
}

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_match_f64, m)?)?;
    Ok(())
}
