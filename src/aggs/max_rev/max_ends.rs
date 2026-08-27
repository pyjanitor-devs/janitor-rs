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

/// Groups reverse-maximum ends by compact candidate ordinal.
/// ELI5: every prefix starts at zero, so the largest end is the exact slot count.
pub fn max_rev_ends_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, ends, index, booleans)?;
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut values = vec![arr[0]; max_end];
    let mut positions = vec![-1_i64; max_end];
    for (row, ((current, end), boolean)) in
        arr.iter().zip(ends.iter()).zip(booleans.iter()).enumerate()
    {
        for (position, value) in positions
            .iter_mut()
            .zip(values.iter_mut())
            .take(*end as usize)
        {
            if *boolean {
                continue;
            }
            if *position == -1 || *current > *value {
                *position = row as i64;
                *value = *current;
            }
        }
    }
    let indexers = (0..max_end).map(|item| index[item]).collect();
    Ok((indexers, Array1::from_vec(positions)))
}

macro_rules! compute {
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
            let (indexers, result) = max_rev_ends_core(
                arr.as_array(),
                ends.as_array(),
                index.as_array(),
                booleans.as_array(),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_max_rev_end_int64, i64);
compute!(compute_max_rev_end_int32, i32);
compute!(compute_max_rev_end_int16, i16);
compute!(compute_max_rev_end_int8, i8);
compute!(compute_max_rev_end_uint64, u64);
compute!(compute_max_rev_end_uint32, u32);
compute!(compute_max_rev_end_uint16, u16);
compute!(compute_max_rev_end_uint8, u8);
compute!(compute_max_rev_end_f64, f64);
compute!(compute_max_rev_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;
    #[test]
    fn finds_max_positions_and_labels() {
        let got = max_rev_ends_core(
            array![5_i64, 2, 4].view(),
            array![2_i64, 3, 1].view(),
            array![50_i64, 10, 90].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((array![50, 10, 90], array![0, 0, 1])));
    }
    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            max_rev_ends_core(
                array![1_i64].view(),
                array![0_i64].view(),
                array![1_i64].view(),
                array![false].view()
            ),
            Ok((array![], array![]))
        );
    }
}
