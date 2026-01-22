use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;



#[pyfunction]
pub fn compute_size_rev_end<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let ends = ends.as_array();
    let index = index.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let start_: usize = 0 as usize;
    for end in ends.into_iter() {
        let end_ = *end as usize;
        for item in start_..end_ {
            let pos = index[item];
            *dictionary.entry(pos).or_insert(0) += 1;
        }
    }
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);
    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    (indexers.into_pyarray(py), result.into_pyarray(py))
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
            *dictionary.entry(pos).or_insert(0) += 1;
        }
    }
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
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let ends = ends.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let start_: usize = 0 as usize;
    let mut n: usize = 0;
    for end in ends.into_iter() {
        let end_ = *end as usize;
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item];
            *dictionary.entry(pos).or_insert(0) += 1;
            n += 1;
        }
    }
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);

    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    (indexers.into_pyarray(py), result.into_pyarray(py))
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
            *dictionary.entry(pos).or_insert(0) += 1;
            n += 1;
        }
    }
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
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let mut n: usize = 0;
    let zipped = starts.into_iter().zip(ends.into_iter());
    for (start, end) in zipped {
        let start_ = *start as usize;
        let end_ = *end as usize;
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item];
            *dictionary.entry(pos).or_insert(0) += 1;
            n += 1;
        }
    }
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);

    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction]
pub fn compute_size_rev_start_end<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let zipped = starts.into_iter().zip(ends.into_iter());
    for (start, end) in zipped {
        let start_ = *start as usize;
        let end_ = *end as usize;
        for item in start_..end_ {
            let pos = index[item];
            *dictionary.entry(pos).or_insert(0) += 1;
        }
    }
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);

    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction]
pub fn compute_size_rev_positions<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();
    let length = length as usize;
    let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
    let zipped = starts.into_iter().zip(ends.into_iter());
    for (start, end) in zipped {
        let start_ = *start as usize;
        let end_ = *end as usize;
        for item in start_..end_ {
            let pos = positions[item];
            if pos == -1 {
                continue;
            }
            let pos = index[pos as usize];
            *dictionary.entry(pos).or_insert(0) += 1;
        }
    }
    let mut indexers = Array1::<i64>::zeros(length);
    let mut result = Array1::<i64>::zeros(length);

    for (pos, (key, val)) in dictionary.iter().enumerate() {
        indexers[pos] = *key;
        result[pos] = *val;
    }
    (indexers.into_pyarray(py), result.into_pyarray(py))
}
