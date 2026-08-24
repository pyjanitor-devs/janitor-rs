use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::aggs::{checked_end, checked_index, ensure_equal_lengths, ensure_tape_width};

/// Deliberate variant of `crate::aggs::checked_range`: allows `start == end`
/// (an empty range) instead of rejecting it, because a candidate row with
/// no matches is a valid "nothing here" result in this module, not an error.
fn checked_range_or_none(start: i64, end: i64, len: usize) -> Option<(usize, usize)> {
    usize::try_from(start)
        .ok()
        .zip(checked_end(end, len))
        .filter(|(start, end)| start <= end)
}

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
    // here, before the loop below ever indexes into the tape. `starts` is
    // a cheap-to-reiterate view, so `checked_end` is recomputed in each
    // pass instead of materializing a `Vec` of validated rows just to
    // reuse it once.
    let expected_matches_width: usize = starts
        .iter()
        .filter_map(|start| checked_end(*start, end).map(|start| end - start))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for start in starts.iter() {
        if pos == length as usize {
            break;
        }
        let Some(start_) = checked_end(*start, end) else {
            continue;
        };
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
    // here, before the loop below ever indexes into the tape. `starts` is
    // a cheap-to-reiterate view, so `checked_end` is recomputed in each
    // pass instead of materializing a `Vec` of validated rows just to
    // reuse it once.
    let expected_matches_width: usize = starts
        .iter()
        .filter_map(|start| checked_end(*start, end).map(|start| end - start))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for (start, count_) in starts.iter().zip(counts) {
        let start = checked_end(*start, end);
        if *count_ == 0 {
            let size = start.map_or(0, |start| end - start);
            n += size;
            continue;
        }
        let start_ = start.unwrap_or(end);
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
    // here, before the loop below ever indexes into the tape. `starts` is
    // a cheap-to-reiterate view, so `checked_end` is recomputed in each
    // pass instead of materializing a `Vec` of validated rows just to
    // reuse it once.
    let expected_matches_width: usize = starts
        .iter()
        .filter_map(|start| checked_end(*start, end).map(|start| end - start))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for (start, count) in starts.iter().zip(counts) {
        let start = checked_end(*start, end);
        if *count == 0 {
            let size = start.map_or(0, |start| end - start);
            n += size;
            continue;
        }
        let start_ = start.unwrap_or(end);
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
    // here, before the loop below ever indexes into the tape. `ends` is
    // a cheap-to-reiterate view, so `checked_end` is recomputed in each
    // pass instead of materializing a `Vec` of validated rows just to
    // reuse it once.
    let expected_matches_width: usize = ends
        .iter()
        .filter_map(|end| checked_end(*end, index.len()))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for end in ends.iter() {
        if pos == length as usize {
            break;
        }
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
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
    // here, before the loop below ever indexes into the tape. `ends` is
    // a cheap-to-reiterate view, so `checked_end` is recomputed in each
    // pass instead of materializing a `Vec` of validated rows just to
    // reuse it once.
    let expected_matches_width: usize = ends
        .iter()
        .filter_map(|end| checked_end(*end, index.len()))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let start_: usize = 0;
    for (end, count) in ends.iter().zip(counts) {
        let end = checked_end(*end, index.len());
        if *count == 0 {
            let size = end.unwrap_or(start_);
            n += size;
            continue;
        }
        let end_ = end.unwrap_or(start_);
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
    // here, before the loop below ever indexes into the tape. `ends` is
    // a cheap-to-reiterate view, so `checked_end` is recomputed in each
    // pass instead of materializing a `Vec` of validated rows just to
    // reuse it once.
    let expected_matches_width: usize = ends
        .iter()
        .filter_map(|end| checked_end(*end, index.len()))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let start_: usize = 0;
    for (end, count) in ends.iter().zip(counts) {
        let end = checked_end(*end, index.len());
        if *count == 0 {
            let size = end.unwrap_or(start_);
            n += size;
            continue;
        }
        let end_ = end.unwrap_or(start_);
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
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape. `starts`/
    // `ends` are cheap-to-reiterate views, so `checked_range_or_none` is
    // recomputed in each pass instead of materializing a `Vec` of
    // validated rows just to reuse it once.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| {
            checked_range_or_none(*start, *end, index.len()).map(|(start, end)| end - start)
        })
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for (start, end) in starts.iter().zip(ends.iter()) {
        let Some((start_, end_)) = checked_range_or_none(*start, *end, index.len()) else {
            continue;
        };
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
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths("starts", starts.len(), "counts", counts.len())?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape. `starts`/
    // `ends` are cheap-to-reiterate views, so `checked_range_or_none` is
    // recomputed in each pass instead of materializing a `Vec` of
    // validated rows just to reuse it once.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| {
            checked_range_or_none(*start, *end, index.len()).map(|(start, end)| end - start)
        })
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for ((start, end), count_) in starts.iter().zip(ends.iter()).zip(counts) {
        let range = checked_range_or_none(*start, *end, index.len());
        if *count_ == 0 {
            let size = range.map_or(0, |(start, end)| end - start);
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }
        let (start_, end_) = range.unwrap_or((0, 0));
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
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths("starts", starts.len(), "counts", counts.len())?;
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape. `starts`/
    // `ends` are cheap-to-reiterate views, so `checked_range_or_none` is
    // recomputed in each pass instead of materializing a `Vec` of
    // validated rows just to reuse it once.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| {
            checked_range_or_none(*start, *end, index.len()).map(|(start, end)| end - start)
        })
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    for ((start, end), count_) in starts.iter().zip(ends.iter()).zip(counts) {
        let range = checked_range_or_none(*start, *end, index.len());
        if *count_ == 0 {
            let size = range.map_or(0, |(start, end)| end - start);
            n += size;
            continue;
        }
        if pos == length as usize {
            break;
        }

        let (start_, end_) = range.unwrap_or((0, 0));
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
        // ELI5: `n` only ever grows, so once it reaches `result`'s
        // capacity (the caller-supplied `length`, independent of how many
        // in-bounds entries `positions` actually yields) every further
        // write would also be out of bounds -- break instead of walking
        // `result[n]` out of bounds below, matching the equivalent
        // capacity guard on the sibling `index_*_only` functions above.
        if n == result.len() {
            break;
        }
        // ELI5: `position` is a raw index read straight from the
        // caller-supplied `positions` array, not derived from a validated
        // `start..end` range. The old `< 0` check only caught the `-1`
        // sentinel; a positive value `>= index.len()` fell straight into
        // `index[...]` unchecked. `checked_index` catches both.
        let Some(pos) = checked_index(*position, index.len()) else {
            continue;
        };
        let val: i64 = index[pos];
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
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let positions = positions.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths("starts", starts.len(), "counts", counts.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut pos: usize = 0;
    // ELI5: `starts`/`ends` are cheap-to-reiterate views, so there's no
    // need to precompute a `Vec` of validated ranges before this loop --
    // unlike the `index_*` family above, this function has no `matches`
    // tape to width-check up front, so a single pass suffices.
    for ((start, end), count_) in starts.iter().zip(ends.iter()).zip(counts) {
        if *count_ == 0 {
            continue;
        }
        if pos == length as usize {
            break;
        }
        let (start_, end_) = checked_range_or_none(*start, *end, positions.len()).unwrap_or((0, 0));
        let mut base: i64 = -1;
        for nn in start_..end_ {
            let indexer = positions[nn];
            let Some(indexer) = checked_index(indexer, index.len()) else {
                continue;
            };
            let val: i64 = index[indexer];
            if (base < 0) || (val < base) {
                base = val;
            }
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
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let index = index.as_array();
    let counts = counts.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let positions = positions.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    ensure_equal_lengths("starts", starts.len(), "counts", counts.len())?;
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut pos: usize = 0;
    // ELI5: `starts`/`ends` are cheap-to-reiterate views, so there's no
    // need to precompute a `Vec` of validated ranges before this loop --
    // unlike the `index_*` family above, this function has no `matches`
    // tape to width-check up front, so a single pass suffices.
    for ((start, end), count_) in starts.iter().zip(ends.iter()).zip(counts) {
        if *count_ == 0 {
            continue;
        }
        if pos == length as usize {
            break;
        }
        let (start_, end_) = checked_range_or_none(*start, *end, positions.len()).unwrap_or((0, 0));
        let mut base: i64 = -1;
        for nn in start_..end_ {
            let indexer = positions[nn];
            let Some(indexer) = checked_index(indexer, index.len()) else {
                continue;
            };
            let val: i64 = index[indexer];
            if base < val {
                base = val;
            }
        }
        result[pos] = base;
        pos += 1;
    }
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (*, positions, starts))]
pub fn reorder_index<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let positions = positions.as_array();
    let starts = starts.as_array();
    // ELI5: a well-formed call fills every slot in `result` exactly once
    // -- `starts`/`counts` partition all of `positions` into contiguous,
    // non-overlapping runs that together span `0..positions.len()`. So
    // unlike most `-1`-sentinel guards elsewhere in this crate (which
    // gracefully skip a malformed *row* and leave its own slot as "no
    // match"), a bucket/position that fails to resolve here means the
    // *whole call's* input was malformed, not just one row: pyjanitor's
    // only caller does an unfiltered `right.iloc[reordered_positions]` on
    // this output, and pandas treats `-1` as a real (last-row) position,
    // not a "no match" marker. Silently leaving a `-1` in `result` would
    // therefore surface as a wrong row being duplicated into the output,
    // not as an error -- so this raises instead of skipping.
    let mut result = Array1::<i64>::from_elem(positions.len(), -1);
    let mut counts: Array1<i64> = Array1::zeros(starts.len());
    for (index, val) in positions.indexed_iter() {
        let bucket = checked_index(*val, starts.len()).ok_or_else(|| {
            PyValueError::new_err(format!(
                "positions[{index}] = {val} does not name a valid starts/counts bucket \
                 (starts has length {})",
                starts.len()
            ))
        })?;
        // ELI5: `starts[bucket]` and `counts[bucket]` are both caller-
        // controlled (indirectly, via `positions`/`starts` values), so
        // their sum could in principle overflow `i64` -- `checked_add`
        // turns that into the same reported error as any other malformed
        // mapping, instead of panicking (debug) or silently wrapping to a
        // bogus position (release).
        let pos = starts[bucket].checked_add(counts[bucket]).ok_or_else(|| {
            PyValueError::new_err(format!(
                "computed position for positions[{index}] (bucket {bucket}) overflowed i64"
            ))
        })?;
        counts[bucket] += 1;
        let pos = checked_index(pos, result.len()).ok_or_else(|| {
            PyValueError::new_err(format!(
                "computed position {pos} for positions[{index}] is out of bounds for a result \
                 of length {}",
                result.len()
            ))
        })?;
        if result[pos] != -1 {
            return Err(PyValueError::new_err(format!(
                "computed position {pos} for positions[{index}] is already occupied"
            )));
        }
        result[pos] = index as i64;
    }
    if let Some(pos) = result.iter().position(|value| *value == -1) {
        return Err(PyValueError::new_err(format!(
            "reorder_index left result position {pos} unassigned"
        )));
    }
    Ok(result.into_pyarray(py))
}

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(repeat_index, m)?)?;
    m.add_function(wrap_pyfunction!(trim_index, m)?)?;
    m.add_function(wrap_pyfunction!(reorder_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_trim_positions, m)?)?;
    m.add_function(wrap_pyfunction!(build_positional_index, m)?)?;
    m.add_function(wrap_pyfunction!(build_positional_index_first, m)?)?;
    m.add_function(wrap_pyfunction!(build_positional_index_last, m)?)?;
    m.add_function(wrap_pyfunction!(index_starts_only, m)?)?;
    m.add_function(wrap_pyfunction!(index_starts_only_keep_first, m)?)?;
    m.add_function(wrap_pyfunction!(index_starts_only_keep_last, m)?)?;
    m.add_function(wrap_pyfunction!(index_ends_only, m)?)?;
    m.add_function(wrap_pyfunction!(index_ends_only_keep_first, m)?)?;
    m.add_function(wrap_pyfunction!(index_ends_only_keep_last, m)?)?;
    m.add_function(wrap_pyfunction!(index_starts_and_ends, m)?)?;
    m.add_function(wrap_pyfunction!(index_starts_and_ends_keep_first, m)?)?;
    m.add_function(wrap_pyfunction!(index_starts_and_ends_keep_last, m)?)?;
    Ok(())
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
