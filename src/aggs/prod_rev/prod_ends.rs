use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ensure_equal_lengths, ensure_nonempty, ensure_valid_ends};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    ensure_equal_lengths("arr", arr.len(), "ends", ends.len())
        .map_err(|_| "arr and ends must have equal lengths")?;
    ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())
        .map_err(|_| "arr and booleans must have equal lengths")?;
    ensure_nonempty("arr", arr.len())?;
    ensure_nonempty("index", index.len())?;
    ensure_nonempty("ends", ends.len())?;
    ensure_valid_ends("ends", ends.iter().copied(), index.len())
}

pub fn prod_rev_ends_int_core<T: Copy, F: FnMut(T) -> i64>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_i64: F,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, ends, index, booleans)?;
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut values = vec![1_i64; max_end];
    for ((current, end), boolean) in arr.iter().zip(ends.iter()).zip(booleans.iter()) {
        if *boolean {
            continue;
        }
        let current = to_i64(*current);
        for value in values.iter_mut().take(*end as usize) {
            *value = value.wrapping_mul(current);
        }
    }
    let indexers = (0..max_end).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(values)))
}

pub fn prod_rev_ends_float_core<T: Copy, F: FnMut(T) -> f64>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str> {
    validate_inputs(arr, ends, index, booleans)?;
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut values = vec![1_f64; max_end];
    for ((current, end), boolean) in arr.iter().zip(ends.iter()).zip(booleans.iter()) {
        if *boolean {
            continue;
        }
        let current = to_f64(*current);
        for value in values.iter_mut().take(*end as usize) {
            *value *= current;
        }
    }
    let indexers = (0..max_end).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(values)))
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
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            let _ = length;
            let (indexers, result) = prod_rev_ends_int_core(
                arr.as_array(),
                ends.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as i64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}
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
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            let _ = length;
            let (indexers, result) = prod_rev_ends_float_core(
                arr.as_array(),
                ends.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_prod_rev_end_int64, i64);
compute_ints!(compute_prod_rev_end_int32, i32);
compute_ints!(compute_prod_rev_end_int16, i16);
compute_ints!(compute_prod_rev_end_int8, i8);
compute_ints!(compute_prod_rev_end_uint64, u64);
compute_ints!(compute_prod_rev_end_uint32, u32);
compute_ints!(compute_prod_rev_end_uint16, u16);
compute_ints!(compute_prod_rev_end_uint8, u8);
compute_floats!(compute_prod_rev_end_f64, f64);
compute_floats!(compute_prod_rev_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn integer_core_groups_prefixes_and_preserves_identity() {
        let got = prod_rev_ends_int_core(
            array![2_i64, 3].view(),
            array![2_i64, 3].view(),
            array![20_i64, 10, 90].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![20, 10, 90], array![6, 6, 3])));

        let got = prod_rev_ends_int_core(
            array![2_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10], array![1])));
    }

    #[test]
    fn integer_core_wraps_and_float_core_handles_infinity() {
        let got = prod_rev_ends_int_core(
            array![i64::MAX, 2].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10], array![-2])));

        let got = prod_rev_ends_float_core(
            array![f64::INFINITY, 2.0].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        );
        assert!(got.unwrap().1[0].is_infinite());
    }

    #[test]
    fn rejects_invalid_bounds() {
        assert_eq!(
            prod_rev_ends_int_core(
                array![1_i64].view(),
                array![0_i64].view(),
                array![10_i64].view(),
                array![false].view(),
                |value| value,
            ),
            Ok((array![], array![]))
        );
    }
}
