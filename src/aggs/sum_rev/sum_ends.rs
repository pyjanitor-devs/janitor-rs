use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths};

fn validate_ends_inputs(
    arr_len: usize,
    ends: numpy::ndarray::ArrayView1<'_, i64>,
    right_len: usize,
    booleans_len: usize,
) -> Result<(), &'static str> {
    if arr_len != ends.len() || arr_len != booleans_len {
        return Err("arr, ends, and booleans must have equal lengths");
    }
    if arr_len == 0 || right_len == 0 {
        return Err("arr, ends, booleans, and index cannot be empty");
    }
    if ends.iter().any(|end| {
        usize::try_from(*end)
            .map(|end| end == 0 || end > right_len)
            .unwrap_or(true)
    }) {
        return Err("ends must satisfy 0 < end <= right_len");
    }
    Ok(())
}

/// Accumulate reverse-sum `ends` rows in compact candidate-ordinal slots.
///
/// ELI5: `item` addresses the accumulator and `index[item]` is only the
/// original right-row label returned at the end, so sparse labels never
/// inflate the accumulator or become an out-of-bounds address.
pub fn sum_rev_ends_int_core<T, F>(
    arr: numpy::ndarray::ArrayView1<T>,
    ends: numpy::ndarray::ArrayView1<i64>,
    index: numpy::ndarray::ArrayView1<i64>,
    booleans: numpy::ndarray::ArrayView1<bool>,
    mut to_i64: F,
) -> Result<(numpy::ndarray::Array1<i64>, numpy::ndarray::Array1<i64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    validate_ends_inputs(arr.len(), ends, index.len(), booleans.len())?;
    let max_end = ends
        .iter()
        .filter_map(|end| checked_range(0, *end, index.len()).map(|(_, end)| end))
        .max()
        .unwrap();
    let mut values = vec![0_i64; max_end];
    for (current, end, boolean) in izip!(arr, ends, booleans) {
        let current_ = to_i64(*current);
        for value in values.iter_mut().take(*end as usize) {
            if *boolean {
                continue;
            }
            *value = value.wrapping_add(current_);
        }
    }
    let mut indexers = Vec::new();
    let mut result = Vec::new();
    for (item, value) in values.iter().enumerate().take(max_end) {
        indexers.push(index[item]);
        result.push(*value);
    }
    Ok((
        numpy::ndarray::Array1::from_vec(indexers),
        numpy::ndarray::Array1::from_vec(result),
    ))
}

pub fn sum_rev_ends_float_core<T, F>(
    arr: numpy::ndarray::ArrayView1<T>,
    ends: numpy::ndarray::ArrayView1<i64>,
    index: numpy::ndarray::ArrayView1<i64>,
    booleans: numpy::ndarray::ArrayView1<bool>,
    mut to_f64: F,
) -> Result<(numpy::ndarray::Array1<i64>, numpy::ndarray::Array1<f64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    validate_ends_inputs(arr.len(), ends, index.len(), booleans.len())?;
    let max_end = ends
        .iter()
        .filter_map(|end| checked_range(0, *end, index.len()).map(|(_, end)| end))
        .max()
        .unwrap();
    let mut slots = vec![(0.0_f64, 0.0_f64); max_end];
    for (current, end, boolean) in izip!(arr, ends, booleans) {
        let current_ = to_f64(*current);
        for (total, compensation) in slots.iter_mut().take(*end as usize) {
            if *boolean {
                continue;
            }
            let difference = current_ - *compensation;
            let increment = *total + difference;
            *compensation = (increment - *total) - difference;
            if !compensation.is_finite() {
                *compensation = 0.;
            }
            *total = increment;
        }
    }
    let mut indexers = Vec::new();
    let mut result = Vec::new();
    for (item, (total, _)) in slots.iter().enumerate().take(max_end) {
        indexers.push(index[item]);
        result.push(*total);
    }
    Ok((
        numpy::ndarray::Array1::from_vec(indexers),
        numpy::ndarray::Array1::from_vec(result),
    ))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let _ = length;
            let (indexers, result) =
                sum_rev_ends_int_core(arr, ends, index, booleans, |value| value as i64)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_sum_rev_end_int64, i64);
compute_ints!(compute_sum_rev_end_int32, i32);
compute_ints!(compute_sum_rev_end_int16, i16);
compute_ints!(compute_sum_rev_end_int8, i8);
compute_ints!(compute_sum_rev_end_uint64, u64);
compute_ints!(compute_sum_rev_end_uint32, u32);
compute_ints!(compute_sum_rev_end_uint16, u16);
compute_ints!(compute_sum_rev_end_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let _ = length;
            let (indexers, result) =
                sum_rev_ends_float_core(arr, ends, index, booleans, |value| value as f64)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_sum_rev_end_f64, f64);
compute_floats!(compute_sum_rev_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn uses_compact_candidate_slots_in_prefix_order() {
        let arr = array![5_i64, 7, 99];
        let ends = array![2_i64, 1, 2];
        let index = array![10_i64, 30, 20];
        let booleans = array![false, true, true];

        let (indexers, result) = sum_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert_eq!(indexers, array![10, 30]);
        assert_eq!(result, array![5, 5]);
    }

    #[test]
    fn rejects_zero_end_before_allocation() {
        let arr = array![5_i64];
        let ends = array![0_i64];
        let index = array![0_i64];
        let booleans = array![false];

        let error = sum_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap_err();

        assert_eq!(error, "ends must satisfy 0 < end <= right_len");
    }
}
