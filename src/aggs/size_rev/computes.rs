use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_end, checked_index, checked_range, ends_domain, ends_labels, ensure_equal_lengths,
    ensure_exact_tape_width, ensure_nonempty_matches, into_starts_ends_result, starts_domain,
    starts_labels,
};

type SizeRevResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

fn size_rev_ends_core(
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    let max_end = ends_domain(ends, index.len())?;
    let mut result = vec![0_i64; max_end];
    for end in ends {
        for value in result.iter_mut().take(*end as usize) {
            *value += 1;
        }
    }
    Ok((ends_labels(max_end, index), Array1::from_vec(result)))
}

fn size_rev_starts_core(
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    let (min_start, width) = starts_domain(starts, index.len())?;
    let mut result = vec![0_i64; width];
    for start in starts {
        for value in result.iter_mut().skip(*start as usize - min_start) {
            *value += 1;
        }
    }
    Ok((starts_labels(min_start, index), Array1::from_vec(result)))
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
    if ends.is_empty() || index.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ends and index cannot be empty",
        ));
    }
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
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(index.len());
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
    if starts.is_empty() || index.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "starts and index cannot be empty",
        ));
    }
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
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(index.len());
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
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let matches = matches.as_array();
    if starts.is_empty() || index.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "starts, ends, and index cannot be empty",
        ));
    }
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
    // Find the smallest contiguous slice that contains every valid range.
    // Invalid ranges are deliberately ignored here because `checked_range`
    // gives them the existing reverse-kernel "skip this row" behavior.
    //
    // ELI5: if the valid ranges are [2, 5) and [4, 9), we only need buckets
    // for positions [2, 9), not for the entire right index. A bucket's local
    // slot is `position - min_start`; `seen` records matched labels for the
    // compact output.
    // The fold carries the current smallest start and largest end as it visits
    // each valid range; it is equivalent to a simple running min/max loop.
    let (min_start, max_end) = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(s, e)| checked_range(*s, *e, index.len()))
        .fold((usize::MAX, 0), |(min_start, max_end), (start, end)| {
            (min_start.min(start), max_end.max(end))
        });
    let width = max_end.saturating_sub(min_start);
    let mut seen = vec![false; width];
    let mut touched = Vec::new();
    let mut counts = vec![0_i64; width];
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
            let slot = item - min_start;
            if !seen[slot] {
                seen[slot] = true;
                touched.push(slot);
            }
            counts[slot] += 1;
            n += 1;
        }
    }
    let indexers = Array1::from_iter(touched.iter().map(|&slot| index[min_start + slot]));
    let result = Array1::from_iter(touched.iter().map(|&slot| counts[slot]));
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
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    if starts.is_empty() || index.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "starts, ends, and index cannot be empty",
        ));
    }
    // There can be at most one output slot per right-side row. This local
    // bound replaces the old caller-supplied capacity hint and cannot change
    // the result when labels are duplicated.
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(index.len());
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            let pos = index[item];
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
        assert_eq!(
            size_rev_ends_core(array![0_i64].view(), index.view()),
            Ok((array![], array![]))
        );
        assert!(size_rev_starts_core(array![-1_i64].view(), index.view()).is_err());
        assert!(size_rev_ends_core(array![1_i64].view(), array![].view()).is_err());
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
    if starts.is_empty() || index.is_empty() || positions.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "starts, ends, index, and positions cannot be empty",
        ));
    }
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
