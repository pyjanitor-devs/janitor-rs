use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            starts: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            index: ArrayView1<'_, i64>,
            positions: ArrayView1<'_, i64>,
            booleans: ArrayView1<'_, bool>,
        ) -> (Array1<i64>, Array1<i64>)
        // The macro will expand into the contents of this block.
        {
            let mut dictionary: HashMap<i64, i64> = HashMap::new();
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (current, start, end, boolean) in zipped {
                if *boolean {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let current_ = *current as i64;
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let pos = index[indexer as usize];
                    *dictionary.entry(pos).or_insert(0) += current_;
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
generic_compute_ints!(array_compute_int8, i64);
generic_compute_ints!(array_compute_uint64, u64);
generic_compute_ints!(array_compute_uint32, u32);
generic_compute_ints!(array_compute_uint16, u16);
generic_compute_ints!(array_compute_uint8, u8);

#[pyfunction(name = "compute_sum_rev_positions_int64")]
pub fn compute_int64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int64(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_int32")]
pub fn compute_int32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int32(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_int16")]
pub fn compute_int16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int16(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_int8")]
pub fn compute_int8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_int8(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_uint64")]
pub fn compute_uint64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint64(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_uint32")]
pub fn compute_uint32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint32(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_uint16")]
pub fn compute_uint16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint16(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_uint8")]
pub fn compute_uint8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_uint8(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

/// kahan sum_revmation
macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            starts: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            index: ArrayView1<'_, i64>,
            positions: ArrayView1<'_, i64>,
            booleans: ArrayView1<'_, bool>,
        ) -> (Array1<i64>, Array1<f64>)
        // The macro will expand into the contents of this block.
        {
            let mut dictionary: HashMap<i64, f64> = HashMap::new();
            let mut mapping: HashMap<i64, f64> = HashMap::new();
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (current, start, end, boolean) in zipped {
                if *boolean {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let current_ = *current as f64;
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let pos = index[indexer as usize];
                    let mut total = dictionary.get(&pos).copied().unwrap_or(0.);
                    let mut compensation = mapping.get(&pos).copied().unwrap_or(0.);
                    let difference = current_ - compensation;
                    let increment = total + difference;
                    compensation = (increment - total) - difference;
                    total = increment;
                    *dictionary.entry(pos).or_insert(0.) = total;
                    *mapping.entry(pos).or_insert(0.) = compensation;
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

#[pyfunction(name = "compute_sum_rev_positions_f32")]
pub fn compute_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_f32(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction(name = "compute_sum_rev_positions_f64")]
pub fn compute_f64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,

    positions: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();

    let booleans = booleans.as_array();
    let (indexers, result) = array_compute_f64(arr, starts, ends, index, positions, booleans);

    (indexers.into_pyarray(py), result.into_pyarray(py))
}
