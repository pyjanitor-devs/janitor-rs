use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        fn $fname(
            left: ArrayView1<'_, $type>,
            right: ArrayView1<'_, $type>,
            starts: ArrayView1<'_, i64>,
            ends: ArrayView1<'_, i64>,
        ) -> Array1<i64>
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i64>::zeros(left.len());
            let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
            for (pos, (left_value, start, end)) in zipped.enumerate() {
                if *start == -1 || *end == -1 {
                    result[pos as usize] = -1;
                    continue;
                }
                let mut min_idx = *start;
                let mut max_idx = *end;
                while min_idx < max_idx {
                    // to avoid overflow
                    // adapted from numba's implementation
                    let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
                    let current_value = right[mid_idx as usize];
                    if current_value >= *left_value {
                        max_idx = mid_idx;
                    } else {
                        min_idx = mid_idx + 1;
                    }
                }
                if min_idx == 0 {
                    result[pos as usize] = -1;
                    continue;
                }
                let mid_idx = min_idx - 1;
                let current_value = right[mid_idx as usize];
                if current_value == *left_value {
                    result[pos as usize] = -1;
                    continue;
                }
                result[pos as usize] = min_idx;
            }
            result
        }
    };
}

bin_search!(array_compare_int64, i64);
bin_search!(array_compare_int32, i32);
bin_search!(array_compare_int16, i16);
bin_search!(array_compare_int8, i8);
bin_search!(array_compare_uint64, u64);
bin_search!(array_compare_uint32, u32);
bin_search!(array_compare_uint16, u16);
bin_search!(array_compare_uint8, u8);
bin_search!(array_compare_float64, f64);
bin_search!(array_compare_float32, f32);

#[pyfunction(name = "binary_search_gt_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_int64(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_int32(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_int16(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_int8(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_float64(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_float32(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_uint64(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_uint32(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_uint16(left, right, starts, ends);
    result.into_pyarray(py)
}

#[pyfunction(name = "binary_search_gt_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
) -> Bound<'py, PyArray1<i64>> {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = array_compare_uint8(left, right, starts, ends);
    result.into_pyarray(py)
}
