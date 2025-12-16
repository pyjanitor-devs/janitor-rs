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
        let mut base: i64 = -1;
        for nn in start_..end {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if (base == -1) || (val < base) {
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
#[pyo3(signature = (*, index, starts, matches, length))]
pub fn index_starts_1st<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let matches = matches.as_array();
    let result = index_starts_only_first(index, starts, matches, length);
    result.into_pyarray(py)
}

fn index_starts_only_last(
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
#[pyo3(signature = (*, index, starts, matches, length))]
pub fn index_starts_last<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let starts = starts.as_array();
    let matches = matches.as_array();
    let result = index_starts_only_last(index, starts, matches, length);
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
        let mut base: i64 = -1;
        let end_: usize = *end as usize;
        for nn in 0..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if (base == -1) || (val < base) {
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
#[pyo3(signature = (*, index, ends, matches, length))]
pub fn index_ends_1st<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let result = index_ends_only_first(index, ends, matches, length);
    result.into_pyarray(py)
}

fn index_ends_only_last(
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
        let mut base: i64 = -1;
        let end_: usize = *end as usize;
        for nn in 0..end_ {
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
#[pyo3(signature = (*, index, ends, matches, length))]
pub fn index_ends_last<'py>(
    py: Python<'py>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let result = index_ends_only_last(index, ends, matches, length);
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
    matches: ArrayView1<'_, i8>,
    length: i64,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut pos: usize = 0;
    let mut val: i64;
    let zipped = starts.into_iter().zip(ends.into_iter());
    for (start, end) in zipped {
        if pos == length as usize {
            break;
        }
        let start_: usize = *start as usize;
        let end_: usize = *end as usize;
        let mut base: i64 = -1;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            val = index[nn];
            if (base == -1) || (val < base) {
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
#[pyo3(signature = (*, index, starts,ends, matches, length))]
pub fn index_starts_ends_1st<'py>(
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
    let result = index_starts_and_ends_first(index, starts, ends, matches, length);
    result.into_pyarray(py)
}

fn index_starts_and_ends_last(
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
        if pos == length as usize {
            break;
        }
        let start_: usize = *start as usize;
        let end_: usize = *end as usize;
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
#[pyo3(signature = (*, index, starts,ends, matches, length))]
pub fn index_starts_ends_last<'py>(
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
    let result = index_starts_and_ends_last(index, starts, ends, matches, length);
    result.into_pyarray(py)
}
