use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            ends: ArrayView1<'_, i64>,
            booleans: ArrayView1<'_, bool>,
        ) -> Array1<i64>
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i64>::zeros(ends.len());
            let start_: usize = 0;
            for (pos, end) in ends.indexed_iter() {
                let mut total: i64 = 0;
                let end_ = *end as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    total += current as i64;
                }
                result[pos] = total;
            }
            result
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

#[pyfunction(name = "compute_sum_end_int64")]
pub fn compute_int64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int64(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_int32")]
pub fn compute_int32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int32(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_int16")]
pub fn compute_int16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i16>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int16(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_int8")]
pub fn compute_int8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i8>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int8(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_uint64")]
pub fn compute_uint64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint64(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_uint32")]
pub fn compute_uint32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint32(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_uint16")]
pub fn compute_uint16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u16>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint16(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_uint8")]
pub fn compute_uint8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u8>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint8(arr, ends, booleans);

    result.into_pyarray(py)
}

/// kahan summation
macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            ends: ArrayView1<'_, i64>,
            booleans: ArrayView1<'_, bool>,
        ) -> Array1<f64>
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<f64>::zeros(ends.len());
            let start_: usize = 0;
            for (pos, end) in ends.indexed_iter() {
                let mut total: f64 = 0.0;
                let mut compensation: f64 = 0.0;
                let end_ = *end as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current: f64 = arr[nn] as f64;
                    let difference = current - compensation;
                    let increment = total + difference;
                    compensation = (increment - total) - difference;
                    total = increment;
                }
                result[pos] = total;
            }
            result
        }
    };
}

generic_compute_floats!(array_compute_f64, f64);
generic_compute_floats!(array_compute_f32, f32);

#[pyfunction(name = "compute_sum_end_f32")]
pub fn compute_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<f64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_f32(arr, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_sum_end_f64")]
pub fn compute_f64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<f64>> {
    let arr = arr.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_f64(arr, ends, booleans);

    result.into_pyarray(py)
}
