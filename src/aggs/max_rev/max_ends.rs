use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ends_domain, ends_labels, into_starts_ends_result, should_sweep, sweep_winner};

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

/// Groups reverse-maximum ends by compact candidate ordinal.
/// ELI5: every prefix starts at zero, so the largest end is the exact slot count.
pub fn max_rev_ends_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, ends, booleans)?;
    let max_end = ends_domain(ends, index.len())?;

    if !should_sweep(arr.len(), max_end, std::mem::size_of::<T>()) {
        let mut values = vec![arr[0]; max_end];
        let mut positions = vec![-1_i64; max_end];
        for (row, (current, end, boolean)) in
            izip!(arr.iter(), ends.iter(), booleans.iter()).enumerate()
        {
            if *boolean {
                continue;
            }
            // Example: `end = 3` means this prefix covers output slots 0, 1,
            // and 2. `take(3)` visits exactly those three paired position and
            // value slots, then stops before slot 3.
            for (position, value) in positions
                .iter_mut()
                .zip(values.iter_mut())
                .take(*end as usize)
            {
                if *position == -1 || *current > *value {
                    *position = row as i64;
                    *value = *current;
                }
            }
        }
        return Ok((ends_labels(max_end, index), Array1::from_vec(positions)));
    }

    let positions = sweep_winner(
        arr,
        booleans,
        max_end,
        |row| ends[row] as usize,
        |position| position + 1,
        (0..max_end).rev(),
        |current, winner| current > winner,
    );
    Ok((ends_labels(max_end, index), positions))
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
                max_rev_ends_core(
                    arr.as_array(),
                    ends.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                ),
            )
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
        assert!(max_rev_ends_core(
            Array1::<i64>::zeros(0).view(),
            Array1::<i64>::zeros(0).view(),
            Array1::<i64>::zeros(0).view(),
            Array1::<bool>::default(0).view()
        )
        .is_err());
        assert!(max_rev_ends_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![1_i64].view(),
            array![false].view()
        )
        .is_err());
        assert!(max_rev_ends_core(
            array![1_i64].view(),
            array![2_i64].view(),
            array![1_i64].view(),
            array![false].view()
        )
        .is_err());
    }

    #[test]
    fn sweep_preserves_first_tie_and_skips_zero_width_rows() {
        let mut arr = Array1::from_elem(100, 0_i64);
        arr[0] = 5;
        arr[1] = 7;
        let mut ends = Array1::from_elem(100, 1000_i64);
        ends[1] = 500;
        ends[99] = 0;
        let index = Array1::from_iter(0_i64..1000);
        let mut booleans = Array1::from_elem(100, false);
        booleans[99] = true;
        let got = max_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view());
        let (labels, positions) = got.unwrap();
        assert_eq!(labels, index);
        assert_eq!(positions[0], 1);
        assert_eq!(positions[499], 1);
        assert_eq!(positions[500], 0);
    }

    #[test]
    fn sweep_preserves_smallest_row_on_equal_maximum() {
        let mut arr = Array1::from_elem(20, 99_i64);
        arr[2] = 7;
        arr[18] = 7;
        let mut ends = Array1::zeros(20);
        ends[2] = 20;
        ends[18] = 1;
        let index = Array1::from_iter(0..20_i64);
        let mut booleans = Array1::from_elem(20, true);
        booleans[2] = false;
        booleans[18] = false;
        let got = max_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((index, Array1::from_elem(20, 2_i64))));
    }

    #[test]
    fn sweep_matches_direct_reference_with_null_and_nonuniform_ends() {
        let arr = Array1::from_iter((0..20).map(|row| (row % 7) as i64));
        let ends = Array1::from_iter((0..20).map(|row| (row * 3 % 21) as i64));
        let index = Array1::from_iter(0..20_i64);
        let mut booleans = Array1::from_elem(20, false);
        booleans[4] = true;
        booleans[17] = true;

        let max_end = ends.iter().copied().max().unwrap() as usize;
        let mut values = vec![arr[0]; max_end];
        let mut expected = vec![-1_i64; max_end];
        for row in 0..arr.len() {
            if booleans[row] {
                continue;
            }
            for position in 0..ends[row] as usize {
                if expected[position] == -1 || arr[row] > values[position] {
                    expected[position] = row as i64;
                    values[position] = arr[row];
                }
            }
        }

        let (labels, positions) =
            max_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view()).unwrap();
        assert_eq!(labels, Array1::from_iter(0..max_end as i64));
        assert_eq!(positions, Array1::from_vec(expected));
    }
}
