use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        fn $fname(
            arr: ArrayView1<'_, $type>,
            starts: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
            booleans: ArrayView1<'_, bool>,
        ) -> Array1<i64>
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i64>::zeros(starts.len());
            let zipped = starts.into_iter().zip(ends.into_iter());
            for (pos, (start, end)) in zipped.enumerate() {
                let mut base: i64 = -1;
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut base_val = arr[start_];
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    if (base == -1) || (current > base_val) {
                        base_val = current;
                        base = nn as i64;
                    }
                }
                result[pos] = base;
            }
            result
        }
    };
}

generic_compute!(array_compute_int64, i64);
generic_compute!(array_compute_int32, i32);
generic_compute!(array_compute_int16, i16);
generic_compute!(array_compute_int8, i8);
generic_compute!(array_compute_uint64, u64);
generic_compute!(array_compute_uint32, u32);
generic_compute!(array_compute_uint16, u16);
generic_compute!(array_compute_uint8, u8);
generic_compute!(array_compute_f64, f64);
generic_compute!(array_compute_f32, f32);

#[pyfunction(name = "compute_max_start_end_int64")]
pub fn compute_int64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int64(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_int32")]
pub fn compute_int32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int32(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_int16")]
pub fn compute_int16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int16(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_int8")]
pub fn compute_int8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, i8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_int8(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_uint64")]
pub fn compute_uint64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint64(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_uint32")]
pub fn compute_uint32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint32(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_uint16")]
pub fn compute_uint16<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint16(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_uint8")]
pub fn compute_uint8<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_uint8(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_f32")]
pub fn compute_f32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_f32(arr, starts, ends, booleans);

    result.into_pyarray(py)
}

#[pyfunction(name = "compute_max_start_end_f64")]
pub fn compute_f64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    let arr = arr.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let booleans = booleans.as_array();
    let result = array_compute_f64(arr, starts, ends, booleans);

    result.into_pyarray(py)
}
