use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if arr.len() != starts.len() || arr.len() != booleans.len() {
        return Err("arr, starts, and booleans must have equal lengths");
    }
    if arr.is_empty() || index.is_empty() {
        return Err("arr, starts, booleans, and index cannot be empty");
    }
    if starts.iter().any(|start| {
        usize::try_from(*start)
            .map(|start| start > index.len())
            .unwrap_or(true)
    }) {
        return Err("starts must satisfy 0 <= start <= right_len");
    }
    Ok(())
}

pub fn prod_rev_starts_int_core<T: Copy, F: FnMut(T) -> i64>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_i64: F,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, starts, index, booleans)?;
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let width = index.len() - min_start;
    let mut values = vec![1_i64; width];
    for ((current, start), boolean) in arr.iter().zip(starts.iter()).zip(booleans.iter()) {
        if *boolean {
            continue;
        }
        let current = to_i64(*current);
        for value in values.iter_mut().skip(*start as usize - min_start) {
            *value = value.wrapping_mul(current);
        }
    }
    let indexers = (min_start..index.len()).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(values)))
}

pub fn prod_rev_starts_float_core<T: Copy, F: FnMut(T) -> f64>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str> {
    validate_inputs(arr, starts, index, booleans)?;
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let width = index.len() - min_start;
    let mut values = vec![1_f64; width];
    for ((current, start), boolean) in arr.iter().zip(starts.iter()).zip(booleans.iter()) {
        if *boolean {
            continue;
        }
        let current = to_f64(*current);
        for value in values.iter_mut().skip(*start as usize - min_start) {
            *value *= current;
        }
    }
    let indexers = (min_start..index.len()).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(values)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            let _ = length;
            let (indexers, result) = prod_rev_starts_int_core(
                arr.as_array(),
                starts.as_array(),
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
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            let _ = length;
            let (indexers, result) = prod_rev_starts_float_core(
                arr.as_array(),
                starts.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_prod_rev_start_int64, i64);
compute_ints!(compute_prod_rev_start_int32, i32);
compute_ints!(compute_prod_rev_start_int16, i16);
compute_ints!(compute_prod_rev_start_int8, i8);
compute_ints!(compute_prod_rev_start_uint64, u64);
compute_ints!(compute_prod_rev_start_uint32, u32);
compute_ints!(compute_prod_rev_start_uint16, u16);
compute_ints!(compute_prod_rev_start_uint8, u8);
compute_floats!(compute_prod_rev_start_f64, f64);
compute_floats!(compute_prod_rev_start_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn integer_core_groups_suffixes_and_preserves_identity() {
        let arr = array![2_i64, 3];
        let starts = array![0_i64, 1];
        let index = array![20_i64, 10, 90];
        let booleans = array![false, false];
        let got = prod_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![20, 10, 90], array![2, 6, 6])));
    }

    #[test]
    fn integer_core_wraps_and_null_rows_leave_identity() {
        let got = prod_rev_starts_int_core(
            array![i64::MAX, 2].view(),
            array![0_i64, 0].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10], array![-2])));

        let got = prod_rev_starts_int_core(
            array![2_i64].view(),
            array![0_i64].view(),
            array![10_i64].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10], array![1])));
    }

    #[test]
    fn float32_core_promotes_output_and_rejects_invalid_bounds() {
        let got = prod_rev_starts_float_core(
            array![2.5_f32, 4.0].view(),
            array![0_i64, 1].view(),
            array![20_i64, 10].view(),
            array![false, false].view(),
            |value| value as f64,
        );
        assert_eq!(got, Ok((array![20, 10], array![2.5, 10.0])));

        assert!(prod_rev_starts_int_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![10_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
    }
}
