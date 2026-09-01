use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    ensure_equal_lengths_core, into_starts_ends_result, should_sweep, starts_domain, starts_labels,
    sweep_winner,
};

/// Groups reverse-minimum starts by compact candidate ordinal.
///
/// ELI5: every suffix touches a contiguous tail of `index`, so one slot per
/// position in the union of those tails replaces both key/value HashMaps.
pub fn min_rev_starts_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;

    if should_sweep(arr.len(), width, std::mem::size_of::<T>()) {
        // ELI5: a suffix row becomes eligible when the sweep reaches its
        // start. Bucket each row at that boundary, then compare it with the
        // current champion only once. Equal values are explicitly resolved by
        // the smallest input row index in `sweep_winner`, matching the direct
        // path regardless of bucket traversal order.
        let positions = sweep_winner(
            arr,
            booleans,
            width,
            |row| starts[row] as usize - min_start,
            |position| position,
            0..width,
            |current, winner| current < winner,
        );
        return Ok((starts_labels(min_start, index), positions));
    }

    let mut values = vec![arr[0]; width];
    let mut positions = vec![-1_i64; width];

    for (row, (current, start, boolean)) in izip!(arr, starts, booleans).enumerate() {
        if *boolean {
            continue;
        }
        // Example: with `min_start = 2` and `start = 5`, the compact offset
        // is `5 - 2 = 3`. Skipping three paired slots leaves this row's
        // suffix, positions 5 onward, for the minimum comparison.
        for (position, value) in positions
            .iter_mut()
            .zip(values.iter_mut())
            .skip(*start as usize - min_start)
        {
            if *position == -1 || *current < *value {
                *position = row as i64;
                *value = *current;
            }
        }
    }

    Ok((starts_labels(min_start, index), Array1::from_vec(positions)))
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
                min_rev_starts_core(
                    arr.as_array(),
                    starts.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                ),
            )
        }
    };
}

compute!(compute_min_rev_start_int64, i64);
compute!(compute_min_rev_start_int32, i32);
compute!(compute_min_rev_start_int16, i16);
compute!(compute_min_rev_start_int8, i8);
compute!(compute_min_rev_start_uint64, u64);
compute!(compute_min_rev_start_uint32, u32);
compute!(compute_min_rev_start_uint16, u16);
compute!(compute_min_rev_start_uint8, u8);
compute!(compute_min_rev_start_f64, f64);
compute!(compute_min_rev_start_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn groups_suffixes_by_position_and_emits_original_labels() {
        let arr = array![5_i64, 2, 4];
        let starts = array![1_i64, 0, 2];
        let index = array![50_i64, 10, 90];
        let booleans = array![false, false, false];
        let got = min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((array![50, 10, 90], array![1, 1, 1])));
    }

    #[test]
    fn null_and_all_null_groups_use_minus_one_position() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64, 1, 2];
        let index = array![10_i64, 20, 30];
        let booleans = array![true, false, true];
        let got = min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((array![10, 20, 30], array![-1, 1, 1])));
    }

    #[test]
    fn sweep_preserves_smallest_row_on_equal_minimum() {
        let mut arr = Array1::from_elem(20, 99_i64);
        arr[2] = 7;
        arr[18] = 7;
        let mut starts = Array1::from_elem(20, 20_i64);
        starts[2] = 19;
        starts[18] = 0;
        let index = Array1::from_iter(0..20_i64);
        let booleans = Array1::from_elem(20, false);
        let got = min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        let expected =
            Array1::from_iter((0..20).map(|position| if position == 19 { 2 } else { 18 }));
        assert_eq!(got, Ok((Array1::from_iter(0..20_i64), expected)));
    }

    #[test]
    fn sweep_skips_null_rows() {
        let mut arr = Array1::from_elem(20, 99_i64);
        arr[2] = 7;
        arr[18] = 0;
        let mut starts = Array1::from_elem(20, 20_i64);
        starts[2] = 19;
        starts[18] = 0;
        let index = Array1::from_iter(0..20_i64);
        let mut booleans = Array1::from_elem(20, true);
        booleans[2] = false;
        let got = min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        let expected =
            Array1::from_iter((0..20).map(|position| if position == 19 { 2 } else { -1 }));
        assert_eq!(got, Ok((Array1::from_iter(0..20_i64), expected)));
    }

    #[test]
    fn rejects_empty_inputs_mismatches_and_invalid_bounds() {
        let arr = array![1_i64];
        let index = array![10_i64];
        let booleans = array![false];
        assert_eq!(
            min_rev_starts_core(
                arr.view(),
                array![1_i64].view(),
                index.view(),
                booleans.view()
            ),
            Ok((array![], array![]))
        );
        assert!(min_rev_starts_core(
            arr.view(),
            array![-1_i64].view(),
            index.view(),
            booleans.view()
        )
        .is_err());
        assert!(min_rev_starts_core(
            arr.view(),
            array![0_i64].view(),
            index.view(),
            array![].view()
        )
        .is_err());
    }
}
