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

/// Count how many reverse-end rows cover each compact prefix position.
///
/// ELI5: each `end` says “this row covers everything before here.” Instead of
/// walking that whole prefix for every row, the implementation activates rows
/// once during a right-to-left sweep and carries the running count.
pub fn size_rev_ends_core(
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    if ends.is_empty() || index.is_empty() {
        return Err("ends and index cannot be empty");
    }
    if ends.iter().any(|end| {
        usize::try_from(*end)
            .map(|end| end > index.len())
            .unwrap_or(true)
    }) {
        return Err("ends must satisfy 0 <= end <= right_len");
    }
    let max_end = ends.iter().copied().max().unwrap_or(0) as usize;
    // ELI5: a prefix row is active to the left of its end. Count how many
    // rows end at each boundary, then sweep from right to left and carry the
    // active-row count across the output. `end == 0` is an empty prefix and
    // naturally has no activation bucket visited by the output loop. Counts
    // are enough here, so memory depends on the compact output width—not the
    // number of rows in the batch.
    let mut events = vec![0_i64; max_end + 1];
    for end in ends {
        events[*end as usize] += 1;
    }

    let mut running = 0_i64;
    let mut result = vec![0_i64; max_end];
    for position in (0..max_end).rev() {
        running += events[position + 1];
        result[position] = running;
    }
    Ok((ends_labels(max_end, index), Array1::from_vec(result)))
}

/// Count how many reverse-start rows cover each compact suffix position.
///
/// ELI5: each `start` says “this row covers everything from here onward.” The
/// implementation groups starts by boundary and sweeps those boundaries once,
/// so wide suffixes do not cause repeated work.
pub fn size_rev_starts_core(
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    if starts.is_empty() || index.is_empty() {
        return Err("starts and index cannot be empty");
    }
    if starts.iter().any(|start| {
        usize::try_from(*start)
            .map(|start| start > index.len())
            .unwrap_or(true)
    }) {
        return Err("starts must satisfy 0 <= start <= right_len");
    }
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let width = index.len() - min_start;
    // ELI5: a suffix row is active from its start onward. Count rows at each
    // start boundary, then sweep those boundaries once while carrying the
    // active-row count. The terminal bucket represents `start ==
    // index.len()` and is intentionally not emitted. Counts are enough, so
    // the temporary memory is proportional to the compact output width.
    let mut events = vec![0_i64; width + 1];
    for start in starts {
        events[*start as usize - min_start] += 1;
    }

    let mut running = 0_i64;
    let mut result = Vec::with_capacity(width);
    for event in events.iter().take(width) {
        running += event;
        result.push(running);
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
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(index.len());
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
