use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_tape_width;

/// Replicates `numpy.repeat(index, counts)`: emit `index[i]`, `counts[i]`
/// times, for every `i`, back to back.
///
/// ELI5: like turning a compact "3 apples, 0 pears, 2 plums" tally into
/// the flat list "apple, apple, apple, plum, plum" -- each label repeated
/// as many times as its count says.
///
/// `length` must equal `counts.sum()`; the caller (the Python side)
/// computes it once and passes it in so this function doesn't need to
/// scan `counts` twice just to size its output.
pub fn repeat_index_core(
    index: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    length: i64,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut val: i64;
    for (i, number) in counts.indexed_iter() {
        val = index[i];
        let num: usize = *number as usize;
        for _ in 0..num {
            result[n] = val;
            n += 1;
        }
    }
    result
}

/// This function replicates numpy.repeat
#[pyfunction]
#[pyo3(signature = (*, index, counts, length))]
pub fn repeat_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let result = repeat_index_core(index.as_array(), counts.as_array(), length);
    result.into_pyarray(py)
}

/// This function replicates index[positions>-1]
#[pyfunction]
#[pyo3(signature = (*, index, positions, length))]
pub fn index_trim_positions<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let positions = positions.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut val: i64;
    let mut pos: usize = 0;
    for (i, number) in positions.indexed_iter() {
        if *number < 0 {
            continue;
        }
        val = index[i];
        result[pos] = val;
        pos += 1;
    }
    result.into_pyarray(py)
}

/// Replicates `index[counts > 0]`: keep only the entries of `index` whose
/// paired `counts` entry is non-zero, in order.
///
/// ELI5: walk the tally sheet and cross off every label that matched
/// nothing (`count == 0`); what's left, in the same order, is the answer.
/// `length` must equal `(counts != 0).sum()`.
pub fn trim_index_core(
    index: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    length: i64,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut val: i64;
    let mut pos: usize = 0;
    for (i, number) in counts.indexed_iter() {
        if *number == 0 {
            continue;
        }
        val = index[i];
        result[pos] = val;
        pos += 1;
    }
    result
}

/// This function replicates index[counts>0]
#[pyfunction]
#[pyo3(signature = (*, index, counts, length))]
pub fn trim_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let result = trim_index_core(index.as_array(), counts.as_array(), length);
    result.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (*, index, starts, matches, length))]
pub fn index_starts_only<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let matches = matches.as_array();
    let end: usize = index.len();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize =
        starts.iter().map(|s| end.saturating_sub(*s as usize)).sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for start in starts.into_iter() {
        if pos == length as usize {
            break;
        }
        let start_: usize = *start as usize;
        for nn in start_..end {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            result[pos] = val;
            pos += 1;
            n += 1;
        }
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, starts, counts, matches, length))]
pub fn index_starts_only_keep_first<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    let end: usize = index.len();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize =
        starts.iter().map(|s| end.saturating_sub(*s as usize)).sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let zipped = starts.into_iter().zip(counts);
    for (start, count_) in zipped {
        let start_: usize = *start as usize;
        if *count_ == 0 {
            let size = end - start_;
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }
        let mut base: i64 = -1;
        for nn in start_..end {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if (base < 0) || (val < base) {
                base = val;
            }
            n += 1;
        }
        result[pos] = base;
        pos += 1
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, starts, counts, matches, length))]
pub fn index_starts_only_keep_last<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    let end: usize = index.len();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize =
        starts.iter().map(|s| end.saturating_sub(*s as usize)).sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for (start, count) in starts.into_iter().zip(counts) {
        let start_: usize = *start as usize;
        if *count == 0 {
            let size = end - start_;
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }
        let mut base: i64 = -1;
        for nn in start_..end {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if base < val {
                base = val;
            }
            n += 1;
        }
        result[pos] = base;
        pos += 1
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, ends, matches, length))]
pub fn index_ends_only<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = ends.iter().map(|e| *e as usize).sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for end in ends.into_iter() {
        if pos == length as usize {
            break;
        }
        let end_: usize = *end as usize;
        for nn in 0..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            result[pos] = val;
            pos += 1;
            n += 1;
        }
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, ends, counts, matches, length))]
pub fn index_ends_only_keep_first<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = ends.iter().map(|e| *e as usize).sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let start_: usize = 0;
    for (end, count) in ends.into_iter().zip(counts) {
        let end_: usize = *end as usize;
        if *count == 0 {
            let size = end_ - start_;
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }
        let mut base: i64 = -1;

        for nn in 0..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if (base < 0) || (val < base) {
                base = val;
            }
            n += 1;
        }
        result[pos] = base;
        pos += 1
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, ends, counts, matches, length))]
pub fn index_ends_only_keep_last<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = ends.iter().map(|e| *e as usize).sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let start_: usize = 0;
    for (end, count) in ends.into_iter().zip(counts) {
        let end_: usize = *end as usize;
        if *count == 0 {
            let size = end_ - start_;
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }
        let mut base: i64 = -1;

        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if base < val {
                base = val;
            }
            n += 1;
        }
        result[pos] = base;
        pos += 1
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, starts,ends, matches, length))]
pub fn index_starts_and_ends<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .map(|(s, e)| (*e as usize).saturating_sub(*s as usize))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let start_: usize = *start as usize;
        let end_: usize = *end as usize;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            result[pos] = val;
            pos += 1;
            n += 1;
        }
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, starts,ends, counts,matches, length))]
pub fn index_starts_and_ends_keep_first<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .map(|(s, e)| (*e as usize).saturating_sub(*s as usize))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
    for (start, end, count_) in zipped {
        let start_: usize = *start as usize;
        let end_: usize = *end as usize;
        if *count_ == 0 {
            let size = end_ - start_;
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }
        let mut base: i64 = -1;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if (base < 0) || (val < base) {
                base = val;
            }
            n += 1;
        }
        result[pos] = base;
        pos += 1;
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, index, starts,ends, counts,matches, length))]
pub fn index_starts_and_ends_keep_last<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .map(|(s, e)| (*e as usize).saturating_sub(*s as usize))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
    for (start, end, count_) in zipped {
        let start_: usize = *start as usize;
        let end_: usize = *end as usize;
        if *count_ == 0 {
            let size = end_ - start_;
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }

        let mut base: i64 = -1;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if base < val {
                base = val;
            }
            n += 1;
        }
        result[pos] = base;
        pos += 1;
    }
    Ok(result.into_pyarray(py))
}

/// Build index based on positions
// here we jump between starts and ends
// to get positions, before getting the index
// this is unlike the true range join
// or previous iterations above
// where starts and ends point directly to the index
#[pyfunction]
#[pyo3(signature = (*, index, positions, length))]
pub fn build_positional_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let positions = positions.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    for position in positions.into_iter() {
        if *position < 0 {
            continue;
        }
        let val: i64 = index[*position as usize];
        result[n] = val;
        n += 1;
    }
    result.into_pyarray(py)
}

/// Build index based on positions
// here we jump between starts and ends
// to get positions, before getting the index
// this is unlike the true range join
// where starts and ends point directly to the index
#[pyfunction]
#[pyo3(signature = (*, index, starts,ends, counts,positions, length))]
pub fn build_positional_index_first<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let positions = positions.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut pos: usize = 0;
    let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
    for (start, end, count_) in zipped.into_iter() {
        if *count_ == 0 {
            continue;
        }
        if pos == length as usize {
            break;
        }
        let start_ = *start as usize;
        let end_ = *end as usize;
        let mut base: i64 = -1;
        for nn in start_..end_ {
            let indexer = positions[nn];
            if indexer == -1 {
                continue;
            }
            let indexer_: usize = indexer as usize;
            let val: i64 = index[indexer_];
            if (base < 0) || (val < base) {
                base = val;
            }
        }
        result[pos] = base;
        pos += 1;
    }
    result.into_pyarray(py)
}

/// Build index based on positions
// here we jump between starts and ends
// to get positions, before getting the index
// this is unlike the true range join
// where starts and ends point directly to the index
#[pyfunction]
#[pyo3(signature = (*, index, starts,ends,counts, positions, length))]
pub fn build_positional_index_last<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let counts = counts.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let positions = positions.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut pos: usize = 0;
    let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
    for (start, end, count_) in zipped.into_iter() {
        if *count_ == 0 {
            continue;
        }
        if pos == length as usize {
            break;
        }
        let start_ = *start as usize;
        let end_ = *end as usize;
        let mut base: i64 = -1;
        for nn in start_..end_ {
            let indexer = positions[nn];
            if indexer == -1 {
                continue;
            }
            let indexer_: usize = indexer as usize;
            let val: i64 = index[indexer_];
            if base < val {
                base = val;
            }
        }
        result[pos] = base;
        pos += 1;
    }
    result.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (*, positions, starts))]
pub fn reorder_index<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let positions = positions.as_array();
    let starts = starts.as_array();
    let mut result = Array1::<i64>::zeros(positions.len());
    let mut counts: Array1<i64> = Array1::zeros(starts.len());
    for (index, val) in positions.indexed_iter() {
        let mut pos = starts[*val as usize];
        pos += counts[*val as usize];
        counts[*val as usize] += 1;
        result[pos as usize] = index as i64;
    }
    result.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn repeat_index_expands_each_entry_by_its_count() {
        let index = array![10_i64, 20, 30];
        let counts = array![3_i64, 0, 2];
        let got = repeat_index_core(index.view(), counts.view(), 5);
        assert_eq!(got, array![10, 10, 10, 30, 30]);
    }

    #[test]
    fn repeat_index_all_zero_counts_gives_empty_output() {
        let index = array![1_i64, 2, 3];
        let counts = array![0_i64, 0, 0];
        let got = repeat_index_core(index.view(), counts.view(), 0);
        assert_eq!(got, Array1::<i64>::zeros(0));
    }

    #[test]
    fn repeat_index_empty_input() {
        let index: Array1<i64> = array![];
        let counts: Array1<i64> = array![];
        let got = repeat_index_core(index.view(), counts.view(), 0);
        assert_eq!(got, Array1::<i64>::zeros(0));
    }

    #[test]
    fn repeat_index_single_large_count() {
        let index = array![7_i64];
        let counts = array![5_i64];
        let got = repeat_index_core(index.view(), counts.view(), 5);
        assert_eq!(got, array![7, 7, 7, 7, 7]);
    }

    #[test]
    fn trim_index_drops_zero_count_entries() {
        let index = array![10_i64, 20, 30, 40];
        let counts = array![1_i64, 0, 0, 2];
        let got = trim_index_core(index.view(), counts.view(), 2);
        assert_eq!(got, array![10, 40]);
    }

    #[test]
    fn trim_index_all_zero_counts_gives_empty_output() {
        let index = array![1_i64, 2, 3];
        let counts = array![0_i64, 0, 0];
        let got = trim_index_core(index.view(), counts.view(), 0);
        assert_eq!(got, Array1::<i64>::zeros(0));
    }

    #[test]
    fn trim_index_empty_input() {
        let index: Array1<i64> = array![];
        let counts: Array1<i64> = array![];
        let got = trim_index_core(index.view(), counts.view(), 0);
        assert_eq!(got, Array1::<i64>::zeros(0));
    }

    #[test]
    fn trim_index_no_zero_counts_keeps_everything() {
        let index = array![1_i64, 2, 3];
        let counts = array![1_i64, 1, 1];
        let got = trim_index_core(index.view(), counts.view(), 3);
        assert_eq!(got, array![1, 2, 3]);
    }
}
