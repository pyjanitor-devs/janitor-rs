use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    ensure_equal_lengths_core, into_starts_ends_result, should_sweep, starts_domain, starts_labels,
    sweep_winner,
};

/// Groups reverse-maximum starts by compact candidate ordinal.
/// ELI5: all suffixes share a contiguous union, so one slot per ordinal replaces the HashMaps.
pub fn max_rev_starts_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;

    if !should_sweep(arr.len(), width, std::mem::size_of::<T>()) {
        let mut values = vec![arr[0]; width];
        let mut positions = vec![-1_i64; width];
        for (row, (current, start, boolean)) in
            izip!(arr.iter(), starts.iter(), booleans.iter()).enumerate()
        {
            if *boolean {
                continue;
            }
            // Example: with `min_start = 2` and `start = 5`, the compact
            // output begins at offset `5 - 2 = 3`. Skipping three paired
            // slots means this row updates positions 5 through the suffix
            // end, while `positions` and `values` stay aligned.
            for (position, value) in positions
                .iter_mut()
                .zip(values.iter_mut())
                .skip(*start as usize - min_start)
            {
                if *position == -1 || *current > *value {
                    *position = row as i64;
                    *value = *current;
                }
            }
        }
        return Ok((starts_labels(min_start, index), Array1::from_vec(positions)));
    }

    let positions = sweep_winner(
        arr,
        booleans,
        width,
        |row| starts[row] as usize - min_start,
        |position| position,
        0..width,
        |current, winner| current > winner,
    );
    Ok((starts_labels(min_start, index), positions))
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            into_starts_ends_result(
                py,
                max_rev_starts_core(
                    arr.as_array(),
                    starts.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                ),
            )
        }
    };
}

compute!(compute_max_rev_start_int64, i64);
compute!(compute_max_rev_start_int32, i32);
compute!(compute_max_rev_start_int16, i16);
compute!(compute_max_rev_start_int8, i8);
compute!(compute_max_rev_start_uint64, u64);
compute!(compute_max_rev_start_uint32, u32);
compute!(compute_max_rev_start_uint16, u16);
compute!(compute_max_rev_start_uint8, u8);
compute!(compute_max_rev_start_f64, f64);
compute!(compute_max_rev_start_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;
    #[test]
    fn finds_max_positions_and_labels() {
        let got = max_rev_starts_core(
            array![5_i64, 2, 4].view(),
            array![1_i64, 0, 2].view(),
            array![50_i64, 10, 90].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((array![50, 10, 90], array![1, 0, 0])));
    }
    #[test]
    fn rejects_invalid_inputs() {
        assert!(max_rev_starts_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![1_i64].view(),
            array![false].view()
        )
        .is_err());
    }

    #[test]
    fn sweep_preserves_first_tie_and_skips_null_rows() {
        let mut arr = Array1::from_elem(100, 7_i64);
        arr[0] = 5;
        let mut starts = Array1::zeros(100);
        starts[99] = 500;
        let index = Array1::from_iter(0_i64..1000);
        let mut booleans = Array1::from_elem(100, false);
        booleans[99] = true;
        let got = max_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        let (labels, positions) = got.unwrap();
        assert_eq!(labels, index);
        assert!(positions.iter().all(|position| *position == 1));
    }

    #[test]
    fn sweep_preserves_smallest_row_on_equal_maximum() {
        let mut arr = Array1::from_elem(20, 99_i64);
        arr[2] = 7;
        arr[18] = 7;
        let mut starts = Array1::from_elem(20, 20_i64);
        starts[2] = 19;
        starts[18] = 0;
        let index = Array1::from_iter(0..20_i64);
        let mut booleans = Array1::from_elem(20, true);
        booleans[2] = false;
        booleans[18] = false;
        let got = max_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        let expected =
            Array1::from_iter((0..20).map(|position| if position == 19 { 2 } else { 18 }));
        assert_eq!(got, Ok((Array1::from_iter(0..20_i64), expected)));
    }

    #[test]
    fn sweep_matches_direct_reference_with_null_and_nonuniform_starts() {
        let arr = Array1::from_iter((0..20).map(|row| (row % 7) as i64));
        let starts = Array1::from_iter((0..20).map(|row| (row * 3 % 21) as i64));
        let index = Array1::from_iter(0..20_i64);
        let mut booleans = Array1::from_elem(20, false);
        booleans[4] = true;
        booleans[17] = true;

        let min_start = starts.iter().copied().min().unwrap() as usize;
        let width = index.len() - min_start;
        let mut values = vec![arr[0]; width];
        let mut expected = vec![-1_i64; width];
        for row in 0..arr.len() {
            if booleans[row] {
                continue;
            }
            for position in (starts[row] as usize - min_start)..width {
                if expected[position] == -1 || arr[row] > values[position] {
                    expected[position] = row as i64;
                    values[position] = arr[row];
                }
            }
        }

        let (labels, positions) =
            max_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view()).unwrap();
        assert_eq!(labels, index);
        assert_eq!(positions, Array1::from_vec(expected));
    }
}
