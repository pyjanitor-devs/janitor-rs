use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_end, checked_index, checked_range, ends_domain, ends_labels, ensure_equal_lengths,
    ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    into_starts_ends_result, materialize_labels, range_reduce, starts_domain, starts_labels,
    sweep_reduce,
};

type SizeRevResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

/// Count how many reverse-end rows cover each compact prefix position.
///
/// ELI5: each `end` says “this row covers everything before here.” Instead of
/// walking that whole prefix for every row, the implementation activates rows
/// once during a right-to-left sweep and carries the running count.
pub fn size_rev_ends_core(
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    let max_end = ends_domain(ends, index.len())?;
    // ELI5: a prefix row is active to the left of its end. Count how many
    // rows end at each boundary, then sweep from right to left and carry the
    // active-row count across the output. `end == 0` is an empty prefix and
    // naturally has no activation bucket visited by the output loop. Counts
    // are enough here, so memory depends on the compact output width—not the
    // number of rows in the batch.
    // Events are streamed directly from `ends`; `sweep_reduce` stores both the
    // boundary totals and the final running counts in one width-sized bucket
    // vector. `end == 0` is an empty prefix and has no event to record.
    let events = ends
        .iter()
        .filter(|end| **end > 0)
        .map(|end| (*end as usize - 1, 1_i64));
    let result = sweep_reduce(max_end, 0_i64, events, (0..max_end).rev(), |left, right| {
        left + right
    })?;
    Ok((ends_labels(max_end, index), result))
}

/// Count how many reverse-start rows cover each compact suffix position.
///
/// ELI5: each `start` says “this row covers everything from here onward.” The
/// implementation groups starts by boundary and sweeps those boundaries once,
/// so wide suffixes do not cause repeated work.
pub fn size_rev_starts_core(
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    let (min_start, width) = starts_domain(starts, index.len())?;
    // ELI5: a suffix row is active from its start onward. The shared reducer
    // counts rows at each start boundary, then carries the active-row count
    // forward. The valid `start == index.len()` bucket has no emitted slot and
    // is naturally outside the compact output domain.
    let events = starts
        .iter()
        .filter(|start| **start < index.len() as i64)
        .map(|start| (*start as usize - min_start, 1_i64));
    let result = sweep_reduce(width, 0_i64, events, 0..width, |left, right| left + right)?;
    Ok((starts_labels(min_start, index), result))
}

#[pyfunction]
pub fn compute_size_rev_end<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    into_starts_ends_result(py, size_rev_ends_core(ends.as_array(), index.as_array()))
}

#[pyfunction]
pub fn compute_size_rev_start<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    into_starts_ends_result(
        py,
        size_rev_starts_core(starts.as_array(), index.as_array()),
    )
}

#[pyfunction]
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position. `counts_array.sum()` normally equals `matches.sum()`; the tape
/// width is `matches.len()`.
pub fn compute_size_rev_end_matches<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
) -> SizeRevResult<'py> {
    let ends = ends.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    ensure_nonempty_core("ends", ends.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("index", index.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("matches", matches.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let mut expected_matches_width = 0_usize;
    let mut max_end = 0_usize;
    for end in ends.iter() {
        if let Some(end_) = checked_end(*end, index.len()) {
            expected_matches_width += end_;
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let dense = crate::aggs::should_use_dense_match_storage(index.len(), max_end);
    let mut touched = Vec::with_capacity(max_end);
    let mut n = 0_usize;
    if dense {
        let mut seen = vec![false; max_end];
        let mut result = vec![0_i64; max_end];
        for end in ends.iter() {
            let Some(end_) = checked_end(*end, index.len()) else {
                continue;
            };
            for item in 0..end_ {
                if matches[n] != 0 {
                    if !seen[item] {
                        seen[item] = true;
                        touched.push(item);
                    }
                    result[item] += 1;
                }
                n += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for item in touched {
            labels.push(index[item]);
            values.push(result[item]);
        }
        return Ok((labels.into_pyarray(py), values.into_pyarray(py)));
    }
    let mut dictionary: HashMap<usize, i64> = HashMap::with_capacity(max_end);
    for end in ends.into_iter() {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in 0..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let value = dictionary.entry(item).or_insert_with(|| {
                touched.push(item);
                0
            });
            *value += 1;
            n += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for item in touched {
        labels.push(index[item]);
        values.push(dictionary[&item]);
    }
    Ok((labels.into_pyarray(py), values.into_pyarray(py)))
}

#[pyfunction]
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position. `counts_array.sum()` normally equals `matches.sum()`; the tape
/// width is `matches.len()`.
pub fn compute_size_rev_start_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    ensure_nonempty_core("starts", starts.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("index", index.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("matches", matches.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let end_: usize = index.len();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let mut expected_matches_width = 0_usize;
    let mut min_start = index.len();
    for start in starts.iter() {
        if let Some((start_, end_)) = checked_range(*start, end_ as i64, index.len()) {
            expected_matches_width += end_ - start_;
            min_start = min_start.min(start_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let width = index.len().saturating_sub(min_start);
    let dense = crate::aggs::should_use_dense_match_storage(index.len(), width);
    let mut touched = Vec::with_capacity(width);
    let mut n = 0_usize;
    if dense {
        let mut seen = vec![false; width];
        let mut values = vec![0_i64; width];
        for start in starts.iter() {
            let Some((start_, _)) = checked_range(*start, end_ as i64, index.len()) else {
                continue;
            };
            for item in start_..index.len() {
                if matches[n] != 0 {
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    values[slot] += 1;
                }
                n += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut result = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            result.push(values[slot]);
        }
        return Ok((labels.into_pyarray(py), result.into_pyarray(py)));
    }
    let mut dictionary: HashMap<usize, i64> = HashMap::with_capacity(width);
    for start in starts.into_iter() {
        let Some((start_, _)) = checked_range(*start, end_ as i64, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let value = dictionary.entry(item).or_insert_with(|| {
                touched.push(item);
                0
            });
            *value += 1;
            n += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut result = Vec::with_capacity(touched.len());
    for item in touched {
        labels.push(index[item]);
        result.push(dictionary[&item]);
    }
    Ok((labels.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position. `counts_array.sum()` normally equals `matches.sum()`; the tape
/// width is `matches.len()`.
pub fn compute_size_rev_start_end_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    ensure_nonempty_core("starts", starts.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("ends", ends.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("index", index.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("matches", matches.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let mut expected_matches_width = 0_usize;
    let mut min_start = index.len();
    let mut max_end = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start_, end_)) = checked_range(*start, *end, index.len()) {
            expected_matches_width += end_ - start_;
            min_start = min_start.min(start_);
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let width = max_end.saturating_sub(min_start);
    let dense = crate::aggs::should_use_dense_match_storage(index.len(), width);
    let mut touched = Vec::with_capacity(width);
    let mut n = 0_usize;
    if dense {
        let mut seen = vec![false; width];
        let mut values = vec![0_i64; width];
        for (start, end) in starts.iter().zip(ends.iter()) {
            let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            for item in start_..end_ {
                if matches[n] != 0 {
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    values[slot] += 1;
                }
                n += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut result = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            result.push(values[slot]);
        }
        return Ok((labels.into_pyarray(py), result.into_pyarray(py)));
    }
    let mut dictionary: HashMap<usize, i64> = HashMap::with_capacity(width);
    for (start, end) in starts.into_iter().zip(ends) {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[n] != 0 {
                let value = dictionary.entry(item).or_insert_with(|| {
                    touched.push(item);
                    0
                });
                *value += 1;
            }
            n += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut result = Vec::with_capacity(touched.len());
    for item in touched {
        labels.push(index[item]);
        result.push(dictionary[&item]);
    }
    Ok((labels.into_pyarray(py), result.into_pyarray(py)))
}

/// Count covered right-row positions for each distinct right-row identity.
///
/// janitor-rs is primarily called by pyjanitor. Its conditional-join path
/// resets the right DataFrame index to unique row labels before sorting or
/// filtering, so labels can be reordered or gapped but are not duplicated.
/// `item`, the ordinal position in `index`, is the state slot; `index[item]`
/// is the output label. `starts` and `ends` describe half-open ranges
/// `[start, end)`. Invalid or zero-width ranges are skipped by `checked_range`;
/// empty `starts`, `ends`, or `index` inputs are rejected.
///
/// # Preconditions
///
/// `index` must contain unique labels in positional order. pyjanitor provides
/// this by normalizing the right side to `range(len(right))`; direct callers
/// must provide it themselves. Duplicate labels are unsupported and are not
/// merged by the positional accumulator.
///
/// ELI5: each right-row position gets a drawer. Every range adds one to each
/// covered drawer, then we print the row identity stored in `index[item]`.
pub fn size_rev_start_end_core(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    let (touched, result) = range_reduce(starts, ends, index.len(), 0_i64, |_row, _item, count| {
        *count += 1
    });
    let indexers = materialize_labels(&touched, index);
    Ok((indexers, result.into_iter().collect()))
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
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let result = size_rev_start_end_core(starts, ends, index)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::array, PyArrayMethods};

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
    fn duplicate_index_labels_are_explicitly_unsupported() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_array(py, &array![0_i64, 1, 0]);
            let ends = PyArray1::from_array(py, &array![2_i64, 3, 1]);
            let index = PyArray1::from_array(py, &array![10_i64, 20, 10]);
            let (labels, counts) = compute_size_rev_start_end(
                py,
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_array(), array![10, 20, 10].view());
            assert_eq!(counts.readonly().as_array(), array![2, 2, 1].view());
        });
    }

    #[test]
    fn rejects_empty_and_invalid_boundaries() {
        let index = array![10_i64];
        assert_eq!(
            size_rev_ends_core(array![0_i64].view(), index.view()),
            Ok((array![], array![]))
        );
        assert!(size_rev_starts_core(array![-1_i64].view(), index.view()).is_err());
        assert!(size_rev_ends_core(array![1_i64].view(), array![].view()).is_err());
    }

    #[test]
    fn skips_zero_width_rows_without_affecting_other_rows() {
        let index = array![10_i64, 20, 30];
        assert_eq!(
            size_rev_starts_core(array![1_i64, 3].view(), index.view()),
            Ok((array![20, 30], array![1, 1]))
        );
        assert_eq!(
            size_rev_ends_core(array![0_i64, 2].view(), index.view()),
            Ok((array![10, 20], array![1, 1]))
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
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let positions = positions.as_array();
    ensure_nonempty_core("starts", starts.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("ends", ends.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("index", index.len()).map_err(pyo3::exceptions::PyValueError::new_err)?;
    ensure_nonempty_core("positions", positions.len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(index.len());
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
