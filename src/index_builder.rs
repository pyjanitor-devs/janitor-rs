use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// This function replicates numpy.repeat
#[pyfunction]
#[pyo3(signature = (*, index, counts, length))]
pub fn repeat_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let counts = counts.as_array();
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

/// This function replicates index[counts>0]
#[pyfunction]
#[pyo3(signature = (*, index, counts, length))]
pub fn trim_index<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let counts = counts.as_array();
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let matches = matches.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let end: usize = index.len();
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let end: usize = index.len();
    let zipped = starts.into_iter().zip(counts.into_iter());
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let end: usize = index.len();
    for (start, count) in starts.into_iter().zip(counts.into_iter()) {
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
    result.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (*, index, ends, matches, length))]
pub fn index_ends_only<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let start_: usize = 0;
    for (end, count) in ends.into_iter().zip(counts.into_iter()) {
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let start_: usize = 0;
    for (end, count) in ends.into_iter().zip(counts.into_iter()) {
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let zipped = starts.into_iter().zip(ends.into_iter());
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
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
    result.into_pyarray(py)
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
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let counts = counts.as_array();
    let matches = matches.as_array();
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
    result.into_pyarray(py)
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
