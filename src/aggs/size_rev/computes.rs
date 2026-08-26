use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

use crate::aggs::{
    checked_end, checked_index, checked_range, ensure_equal_lengths, ensure_exact_tape_width,
    ensure_nonempty_matches,
};

type SizeRevResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

fn size_rev_ends_core(
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    if ends.is_empty() || index.is_empty() {
        return Err("ends and index cannot be empty");
    }
    if ends.iter().any(|end| {
        usize::try_from(*end)
            .map(|end| end == 0 || end > index.len())
            .unwrap_or(true)
    }) {
        return Err("ends must satisfy 0 < end <= right_len");
    }
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut result = vec![0_i64; max_end];
    for end in ends {
        for value in result.iter_mut().take(*end as usize) {
            *value += 1;
        }
    }
    let indexers = (0..max_end).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(result)))
}

fn size_rev_starts_core(
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    if starts.is_empty() || index.is_empty() {
        return Err("starts and index cannot be empty");
    }
    if starts.iter().any(|start| {
        usize::try_from(*start)
            .map(|start| start >= index.len())
            .unwrap_or(true)
    }) {
        return Err("starts must satisfy 0 <= start < right_len");
    }
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let mut result = vec![0_i64; index.len() - min_start];
    for start in starts {
        for value in result.iter_mut().skip(*start as usize - min_start) {
            *value += 1;
        }
    }
    let indexers = (min_start..index.len()).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(result)))
}

#[pyfunction]
pub fn compute_size_rev_end<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let _ = length;
    let (indexers, result) = size_rev_ends_core(ends.as_array(), index.as_array())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_start<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let _ = length;
    let (indexers, result) = size_rev_starts_core(starts.as_array(), index.as_array())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position. pyjanitor guarantees this with `counts_array.sum() == matches.len()`.
pub fn compute_size_rev_end_matches<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> SizeRevResult<'py> {
    let ends = ends.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    ensure_nonempty_matches(matches.len())?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = ends
        .iter()
        .filter_map(|e| checked_end(*e, index.len()))
        .sum();
    ensure_exact_tape_width(expected_matches_width, matches.len())?;
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let start_: usize = 0_usize;
    let mut n: usize = 0;
    for end in ends.into_iter() {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item];
            let total = dictionary.entry(pos).or_insert(0);
            *total += 1;
            n += 1;
        }
    }
    let length = dictionary.len();
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);
    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position. pyjanitor guarantees this with `counts_array.sum() == matches.len()`.
pub fn compute_size_rev_start_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    ensure_nonempty_matches(matches.len())?;
    let end_: usize = index.len();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .map(|s| end_.saturating_sub(*s as usize))
        .sum();
    ensure_exact_tape_width(expected_matches_width, matches.len())?;
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let mut n: usize = 0;
    for start in starts.into_iter() {
        let start_ = *start as usize;
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item];
            let total = dictionary.entry(pos).or_insert(0);
            *total += 1;
            n += 1;
        }
    }
    let length = dictionary.len();
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);
    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position. pyjanitor guarantees this with `counts_array.sum() == matches.len()`.
pub fn compute_size_rev_start_end_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let matches = matches.as_array();
    ensure_nonempty_matches(matches.len())?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(s, e)| checked_range(*s, *e, index.len()).map(|(s_, e_)| e_ - s_))
        .sum();
    ensure_exact_tape_width(expected_matches_width, matches.len())?;
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let mut n: usize = 0;
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item];
            let total = dictionary.entry(pos).or_insert(0);
            *total += 1;
            n += 1;
        }
    }
    let length = dictionary.len();
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);

    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_start_end<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let (indexers, result) = size_rev_start_end_core(starts, ends, index)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

/// Count explicit range coverage for each distinct right-hand label.
///
/// ELI5: each label gets one numbered counter. Repeated labels reuse that
/// counter, so the state vectors grow with distinct labels rather than with
/// every repeated occurrence.
pub fn size_rev_start_end_core(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    if starts.len() != ends.len() {
        return Err("starts and ends must have equal lengths");
    }
    if starts.is_empty() || index.is_empty() {
        return Err("starts, ends, and index cannot be empty");
    }
    let hint = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| checked_range(*start, *end, index.len()))
        .fold(0_usize, |total, (start, end)| {
            total.saturating_add(end - start)
        })
        .min(index.len());
    // ELI5: this width is only a safe upper bound. Duplicate labels do not
    // create duplicate vector entries because vectors grow on first sight.
    let mut slots = HashMap::with_capacity(hint);
    let mut labels = Vec::new();
    let mut counts = Vec::new();
    for (start, end) in starts.iter().zip(ends.iter()) {
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
                    counts.push(0_i64);
                    slot
                }
            };
            counts[slot] += 1;
        }
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(counts)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn counts_prefixes_and_suffixes_in_compact_slots() {
        let index = array![50_i64, 10, 90];
        assert_eq!(
            size_rev_ends_core(array![2_i64, 3, 1].view(), index.view()),
            Ok((array![50, 10, 90], array![3, 2, 1]))
        );
        assert_eq!(
            size_rev_starts_core(array![1_i64, 0, 2].view(), index.view()),
            Ok((array![50, 10, 90], array![1, 2, 3]))
        );
    }

    #[test]
    fn rejects_empty_and_invalid_boundaries() {
        let index = array![10_i64];
        assert!(size_rev_ends_core(array![0_i64].view(), index.view()).is_err());
        assert!(size_rev_starts_core(array![-1_i64].view(), index.view()).is_err());
        assert!(size_rev_ends_core(array![1_i64].view(), array![].view()).is_err());
    }

    #[test]
    fn explicit_ranges_count_duplicate_labels_in_compact_slots() {
        assert_eq!(
            size_rev_start_end_core(
                array![0_i64, 1, 0].view(),
                array![2_i64, 3, 1].view(),
                array![10_i64, 20, 10].view(),
            ),
            Ok((array![10, 20], array![3, 2]))
        );
    }

    #[test]
    fn explicit_ranges_skip_invalid_and_accept_zero_width_rows() {
        assert_eq!(
            size_rev_start_end_core(
                array![2_i64, -1, 0, 1].view(),
                array![2_i64, 1, 1, 4].view(),
                array![10_i64, 20].view(),
            ),
            Ok((array![10], array![1]))
        );
        assert_eq!(
            size_rev_start_end_core(
                array![2_i64].view(),
                array![2_i64].view(),
                array![10_i64, 20].view(),
            ),
            Ok((array![], array![]))
        );
    }

    #[test]
    fn explicit_range_validation_rejects_empty_and_mismatched_inputs() {
        assert!(size_rev_start_end_core(
            array![0_i64].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
        )
        .is_err());
        assert!(
            size_rev_start_end_core(array![].view(), array![].view(), array![10_i64].view(),)
                .is_err()
        );
    }
}

#[pyfunction]
pub fn compute_size_rev_positions<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let positions = positions.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        for item in start_..end_ {
            let Some(indexer_) = checked_index(positions[item], index.len()) else {
                continue;
            };
            let pos = index[indexer_];
            let total = dictionary.entry(pos).or_insert(0);
            *total += 1;
        }
    }
    let length = dictionary.len();
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);
    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_size_rev_start, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_end, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_positions, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_start_matches, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_end_matches, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_start_end, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_start_end_matches, m)?)?;
    Ok(())
}
