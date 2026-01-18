use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            index: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            counts: ArrayView1<'_, i64>,
            matches: ArrayView1<'_, i8>,
            booleans: ArrayView1<'_, bool>,
        ) -> (Array1<i64>, Array1<i64>)
        // The macro will expand into the contents of this block.
        {
            let mut dictionary: HashMap<i64, i64> = HashMap::new();
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (current, end, count, boolean) in zipped {
                let end_ = *end as usize;
                if *boolean || (*count == 0) {
                    n += end_;
                    continue;
                }
                let current_ = *current as i64;
                for item in 0..end_ {
                    if matches[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    *dictionary.entry(pos).or_insert(0) += current_;
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
            (indexers, result)
        }
    };
}

generic_compute_ints!(array_compute_int64, i64);
generic_compute_ints!(array_compute_int32, i32);
generic_compute_ints!(array_compute_int16, i16);
generic_compute_ints!(array_compute_int8, i8);
generic_compute_ints!(array_compute_uint64, u64);
generic_compute_ints!(array_compute_uint32, u32);
generic_compute_ints!(array_compute_uint16, u16);
generic_compute_ints!(array_compute_uint8, u8);

#[pyfunction(name = "compute_sum_rev_end_match_int64")]
pub fn compute_int64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int64(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_int32")]
pub fn compute_int32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int32(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_int16")]
pub fn compute_int16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i16>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int16(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_int8")]
pub fn compute_int8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i8>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int8(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_uint64")]
pub fn compute_uint64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint64(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_uint32")]
pub fn compute_uint32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint32(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_uint16")]
pub fn compute_uint16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u16>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint16(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_uint8")]
pub fn compute_uint8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u8>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint8(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

/// kahan sum_revmation
macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            index: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            counts: ArrayView1<'_, i64>,
            matches: ArrayView1<'_, i8>,
            booleans: ArrayView1<'_, bool>,
        ) -> (Array1<i64>, Array1<f64>)
        // The macro will expand into the contents of this block.
        {
            let mut dictionary: HashMap<i64, f64> = HashMap::new();
            let mut mapping: HashMap<i64, f64> = HashMap::new();
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (current, end, count, boolean) in zipped {
                let end_ = *end as usize;
                if *boolean || (*count == 0) {
                    n += end_;
                    continue;
                }
                let current_ = *current as f64;
                for item in 0..end_ {
                    if matches[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    let mut total = dictionary.get(&pos).copied().unwrap_or(0.);
                    let mut compensation = mapping.get(&pos).copied().unwrap_or(0.);
                    let difference = current_ - compensation;
                    let increment = total + difference;
                    compensation = (increment - total) - difference;
                    total = increment;
                    *dictionary.entry(pos).or_insert(0.) = total;
                    *mapping.entry(pos).or_insert(0.) = compensation;
                    n += 1;
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<f64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            (indexers, result)
        }
    };
}

generic_compute_floats!(array_compute_f64, f64);
generic_compute_floats!(array_compute_f32, f32);

#[pyfunction(name = "compute_sum_rev_end_match_f32")]
pub fn compute_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_f32(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_end_match_f64")]
pub fn compute_f64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    index: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let arr = arr.as_array();
    let index = index.as_array();
    let ends = ends.as_array();
    let matches = matches.as_array();
    let counts = counts.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_f64(arr, index, ends, counts, matches, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}
