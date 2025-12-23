use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn repeat_index(
    index: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
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
#[pyfunction(name = "repeat_index")]
#[pyo3(signature = (*, index, counts, length))]
pub fn index_repeat<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let counts = counts.as_array();
    let result = repeat_index(index, counts, length);
    result.into_pyarray(py)
}

fn trim_index(index: ArrayView1<'_, i64>, counts: ArrayView1<'_, i64>, length: i64) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut val: i64;
    let mut pos: usize = 0;
    for (i, number) in counts.indexed_iter() {
        val = index[i];
        if *number == 0 {
            continue;
        }
        result[pos] = val;
        pos += 1;
    }
    result
}

/// This function replicates index[counts>0]
#[pyfunction(name = "trim_index")]
#[pyo3(signature = (*, index, counts, length))]
pub fn index_trim<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let counts = counts.as_array();
    let result = trim_index(index, counts, length);
    result.into_pyarray(py)
}

fn index_starts_only(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_starts_only")]
#[pyo3(signature = (*, index, starts, matches, length))]
pub fn index_starts<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let matches = matches.as_array();
    let result = index_starts_only(index, starts, matches, length);
    result.into_pyarray(py)
}

fn index_starts_only_first(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_starts_only_keep_first")]
#[pyo3(signature = (*, index, starts, counts, matches, length))]
pub fn index_starts_1st<'py>(
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
    let result = index_starts_only_first(index, starts, counts, matches, length);
    result.into_pyarray(py)
}

fn index_starts_only_last(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_starts_only_keep_last")]
#[pyo3(signature = (*, index, starts, counts, matches, length))]
pub fn index_starts_last<'py>(
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
    let result = index_starts_only_last(index, starts, counts, matches, length);
    result.into_pyarray(py)
}

fn index_ends_only(
    index: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_ends_only")]
#[pyo3(signature = (*, index, ends, matches, length))]
pub fn index_ends<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let result = index_ends_only(index, ends, matches, length);
    result.into_pyarray(py)
}

fn index_ends_only_first(
    index: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_ends_only_keep_first")]
#[pyo3(signature = (*, index, ends, counts, matches, length))]
pub fn index_ends_1st<'py>(
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
    let result = index_ends_only_first(index, ends, counts, matches, length);
    result.into_pyarray(py)
}

fn index_ends_only_last(
    index: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_ends_only_keep_last")]
#[pyo3(signature = (*, index, ends, counts, matches, length))]
pub fn index_ends_last<'py>(
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
    let result = index_ends_only_last(index, ends, counts, matches, length);
    result.into_pyarray(py)
}

fn index_starts_and_ends(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_starts_and_ends")]
#[pyo3(signature = (*, index, starts,ends, matches, length))]
pub fn index_starts_ends<'py>(
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
    let result = index_starts_and_ends(index, starts, ends, matches, length);
    result.into_pyarray(py)
}

fn index_starts_and_ends_first(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_starts_and_ends_keep_first")]
#[pyo3(signature = (*, index, starts,ends, counts,matches, length))]
pub fn index_starts_ends_1st<'py>(
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
    let result = index_starts_and_ends_first(index, starts, ends, counts, matches, length);
    result.into_pyarray(py)
}

fn index_starts_and_ends_last(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
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
    result
}

#[pyfunction(name = "index_starts_and_ends_keep_last")]
#[pyo3(signature = (*, index, starts,ends, counts,matches, length))]
pub fn index_starts_ends_last<'py>(
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
    let result = index_starts_and_ends_last(index, starts, ends, counts, matches, length);
    result.into_pyarray(py)
}

// here we jump between starts and ends
// to get positions, before getting the index
// this is unlike the true range join
// or previous iterations above
// where starts and ends point directly to the index
fn build_positional_index(
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    length: i64,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;

    for position in positions.into_iter() {
        if *position < 0 {
            continue;
        }
        let pos = *position as usize;
        let val: i64 = index[pos];
        result[n] = val;
        n += 1;
    }

    result
}

/// Build index based on positions
#[pyfunction(name = "build_positional_index")]
#[pyo3(signature = (*, index, positions, length))]
pub fn index_positions<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let positions = positions.as_array();
    let result = build_positional_index(index, positions, length);
    result.into_pyarray(py)
}

// here we jump between starts and ends
// to get positions, before getting the index
// this is unlike the true range join
// where starts and ends point directly to the index
fn build_positional_index_first(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    length: i64,
) -> Array1<i64> {
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

    result
}

/// Build index based on positions
#[pyfunction(name = "build_positional_index_first")]
#[pyo3(signature = (*, index, starts,ends, counts,positions, length))]
pub fn index_positions_first<'py>(
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
    let result = build_positional_index_first(index, starts, ends, counts, positions, length);
    result.into_pyarray(py)
}

// here we jump between starts and ends
// to get positions, before getting the index
// this is unlike the true range join
// where starts and ends point directly to the index
fn build_positional_index_last(
    index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    length: i64,
) -> Array1<i64> {
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

    result
}

/// Build index based on positions
#[pyfunction(name = "build_positional_index_last")]
#[pyo3(signature = (*, index, starts,ends,counts, positions, length))]
pub fn index_positions_last<'py>(
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
    let result = build_positional_index_last(index, starts, ends, counts, positions, length);
    result.into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_repeat_index() {
        let index = Array1::from_vec(vec![10, 20, 30]);
        let counts = Array1::from_vec(vec![2, 3, 1]);
        let length = 6;

        let result = repeat_index(index.view(), counts.view(), length);

        assert_eq!(result, Array1::from_vec(vec![10, 10, 20, 20, 20, 30]));
    }

    #[test]
    fn test_repeat_index_with_zero_counts() {
        let index = Array1::from_vec(vec![10, 20, 30]);
        let counts = Array1::from_vec(vec![2, 0, 3]);
        let length = 5;

        let result = repeat_index(index.view(), counts.view(), length);

        assert_eq!(result, Array1::from_vec(vec![10, 10, 30, 30, 30]));
    }

    #[test]
    fn test_trim_index() {
        let index = Array1::from_vec(vec![10, 20, 30, 40]);
        let counts = Array1::from_vec(vec![1, 0, 2, 0]);
        let length = 2;

        let result = trim_index(index.view(), counts.view(), length);

        assert_eq!(result, Array1::from_vec(vec![10, 30]));
    }

    #[test]
    fn test_trim_index_all_nonzero() {
        let index = Array1::from_vec(vec![10, 20, 30]);
        let counts = Array1::from_vec(vec![1, 2, 3]);
        let length = 3;

        let result = trim_index(index.view(), counts.view(), length);

        assert_eq!(result, Array1::from_vec(vec![10, 20, 30]));
    }

    #[test]
    fn test_index_starts_only() {
        let index = Array1::from_vec(vec![100, 200, 300, 400]);
        let starts = Array1::from_vec(vec![0, 2]);
        let matches = Array1::from_vec(vec![1, 1, 1, 1]);
        let length = 4;

        let result = index_starts_only(index.view(), starts.view(), matches.view(), length);

        assert_eq!(result, Array1::from_vec(vec![100, 200, 300, 400]));
    }

    #[test]
    fn test_index_starts_only_with_skip() {
        let index = Array1::from_vec(vec![100, 200, 300, 400]);
        let starts = Array1::from_vec(vec![0, 2]);
        let matches = Array1::from_vec(vec![1, 0, 1, 1]);
        let length = 3;

        let result = index_starts_only(index.view(), starts.view(), matches.view(), length);

        assert_eq!(result, Array1::from_vec(vec![100, 300, 400]));
    }

    #[test]
    fn test_index_starts_only_first() {
        let index = Array1::from_vec(vec![100, 200, 150, 250]);
        let starts = Array1::from_vec(vec![0, 2]);
        let counts = Array1::from_vec(vec![2, 1]);
        let matches = Array1::from_vec(vec![1, 0, 0, 0, 1, 0]);
        let length = 2;

        let result = index_starts_only_first(
            index.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            length,
        );

        assert_eq!(result, Array1::from_vec(vec![100, 150]));
    }

    #[test]
    fn test_index_starts_only_first_with_zero_count() {
        let index = Array1::from_vec(vec![100, 200, 150]);
        let starts = Array1::from_vec(vec![0, 1]);
        let counts = Array1::from_vec(vec![1, 0]);
        let matches = Array1::from_vec(vec![1, 1, 1]);
        let length = 1;

        let result = index_starts_only_first(
            index.view(),
            starts.view(),
            counts.view(),
            matches.view(),
            length,
        );

        assert_eq!(result, Array1::from_vec(vec![100]));
    }
}
