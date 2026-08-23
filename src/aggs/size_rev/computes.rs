use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{checked_end, checked_index, checked_range, ensure_equal_lengths};

type SizeRevResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

#[pyfunction]
pub fn compute_size_rev_end<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let ends = ends.as_array();
    let index = index.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let start_: usize = 0_usize;
    for end in ends.into_iter() {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            let pos = index[item];
            let total = dictionary.entry(pos).or_insert(0);
            *total += 1;
        }
    }
    // ELI5: `length` above is only a capacity hint for `with_capacity`; the
    // real output size is however many distinct keys the loop actually
    // found. Sizing `indexers`/`result` from the hint instead of
    // `dictionary.len()` panics (out-of-bounds write below) whenever more
    // keys were found than the hint promised, and silently pads with
    // bogus zero entries whenever fewer were found. Every sibling
    // `_rev`/`size_rev` function re-derives `length` from the dictionary
    // for exactly this reason.
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
pub fn compute_size_rev_start<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let starts = starts.as_array();
    let index = index.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let end_: usize = index.len();
    for start in starts.into_iter() {
        let start_ = *start as usize;
        for item in start_..end_ {
            let pos = index[item];
            let total = dictionary.entry(pos).or_insert(0);
            *total += 1;
        }
    }
    // See the matching comment in `compute_size_rev_end`: the output must
    // be sized from the dictionary's actual key count, not the capacity
    // hint, or writing `indexers[pos]`/`result[pos]` below can walk off
    // the end of a too-small array.
    let length = dictionary.len();
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);
    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction]
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
pub fn compute_size_rev_start_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let starts = starts.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let end_: usize = index.len();
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
    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction]
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
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
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
