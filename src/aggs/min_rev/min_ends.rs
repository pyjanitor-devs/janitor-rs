use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ends_domain, ends_labels, into_starts_ends_result};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if arr.len() != ends.len() || arr.len() != booleans.len() {
        return Err("arr, ends, and booleans must have equal lengths");
    }
    Ok(())
}

/// Groups reverse-minimum ends by compact candidate ordinal.
///
/// ELI5: every prefix touches the same contiguous range beginning at zero,
/// so the largest end tells us exactly how many accumulator slots are needed.
pub fn min_rev_ends_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, ends, booleans)?;
    let max_end = ends_domain(ends, index.len())?;
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
            if *position == -1 || *current < *value {
                *position = row as i64;
                *value = *current;
            }
        }
    }

    Ok((ends_labels(max_end, index), Array1::from_vec(positions)))
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
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            into_starts_ends_result(
                py,
                min_rev_ends_core(
                    arr.as_array(),
                    ends.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                ),
            )
        }
    };
}

compute!(compute_min_rev_end_int64, i64);
compute!(compute_min_rev_end_int32, i32);
compute!(compute_min_rev_end_int16, i16);
compute!(compute_min_rev_end_int8, i8);
compute!(compute_min_rev_end_uint64, u64);
compute!(compute_min_rev_end_uint32, u32);
compute!(compute_min_rev_end_uint16, u16);
compute!(compute_min_rev_end_uint8, u8);
compute!(compute_min_rev_end_f64, f64);
compute!(compute_min_rev_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn groups_prefixes_by_position_and_emits_original_labels() {
        let arr = array![5_i64, 2, 4];
        let ends = array![2_i64, 3, 1];
        let index = array![50_i64, 10, 90];
        let booleans = array![false, false, false];
        let got = min_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((array![50, 10, 90], array![1, 1, 1])));
    }

    #[test]
    fn null_and_all_null_groups_use_minus_one_position() {
        let arr = array![1_i64, 2, 3];
        let ends = array![3_i64, 2, 1];
        let index = array![10_i64, 20, 30];
        let booleans = array![true, false, true];
        let got = min_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((array![10, 20, 30], array![1, 1, -1])));
    }

    #[test]
    fn rejects_empty_inputs_mismatches_and_invalid_bounds() {
        let arr = array![1_i64];
        let index = array![10_i64];
        let booleans = array![false];
        assert_eq!(
            min_rev_ends_core(
                arr.view(),
                array![0_i64].view(),
                index.view(),
                booleans.view()
            ),
            Ok((array![], array![]))
        );
        assert!(min_rev_ends_core(
            arr.view(),
            array![-1_i64].view(),
            index.view(),
            booleans.view()
        )
        .is_err());
        assert!(min_rev_ends_core(
            arr.view(),
            array![1_i64].view(),
            index.view(),
            array![].view()
        )
        .is_err());
    }
}
