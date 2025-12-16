use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn binary_compare<T: std::cmp::PartialOrd>(left: &T, right: &T, op: i8) -> bool {
    match op {
        0 => left > right,
        1 => left >= right,
        2 => left < right,
        3 => left <= right,
        4 => left == right,
        _ => left != right,
    }
}

fn array_compare_int64(
    left: ArrayView1<'_, i64>,
    right: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: i64;
    let mut compare: bool;
    let mut total: i64 = 0;
    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;
        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }
        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int64(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_int32(
    left: ArrayView1<'_, i32>,
    right: ArrayView1<'_, i32>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: i32;

    let mut compare: bool;
    let mut total: i64 = 0;
    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int32(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_int16(
    left: ArrayView1<'_, i16>,
    right: ArrayView1<'_, i16>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: i16;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int16(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_int8(
    left: ArrayView1<'_, i8>,
    right: ArrayView1<'_, i8>,

    ends: ArrayView1<'_, i64>,
    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: i8;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    ends: PyReadonlyArray1<'py, i64>,
    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_int8(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_float32(
    left: ArrayView1<'_, f32>,
    right: ArrayView1<'_, f32>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: f32;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_float32(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_float64(
    left: ArrayView1<'_, f64>,
    right: ArrayView1<'_, f64>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: f64;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_float64(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_uint64(
    left: ArrayView1<'_, u64>,
    right: ArrayView1<'_, u64>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: u64;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint64(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_uint32(
    left: ArrayView1<'_, u32>,
    right: ArrayView1<'_, u32>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: u32;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint32(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_uint16(
    left: ArrayView1<'_, u16>,
    right: ArrayView1<'_, u16>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: u16;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint16(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

fn array_compare_uint8(
    left: ArrayView1<'_, u8>,
    right: ArrayView1<'_, u8>,
    ends: ArrayView1<'_, i64>,

    length: i64,
    op: i8,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(length as usize);
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut n: usize = 0;
    let mut right_val: u8;
    let mut compare: bool;
    let mut total: i64 = 0;

    let zipped = izip!(left.into_iter(), ends.into_iter(),);
    for (position, (left_val, end)) in zipped.enumerate() {
        let end_ = *end as usize;

        let mut counter: i64 = 0;
        for nn in 0..end_ {
            right_val = right[nn];
            compare = binary_compare(left_val, &right_val, op);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }

        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

#[pyfunction(name = "compare_first_end_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    ends: PyReadonlyArray1<'py, i64>,

    length: i64,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let ends = ends.as_array();

    let (result, counts_array, total) = array_compare_uint8(left, right, ends, length, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}
