use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ensure_equal_lengths_core, ensure_nonempty_core, starts_domain};

/// Groups reverse-minimum starts by compact candidate ordinal.
///
/// ELI5: every suffix touches a contiguous tail of `index`, so one slot per
/// position in the union of those tails replaces both key/value HashMaps.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive ordinal start of each suffix range.
/// * `index` - Right-side labels in ordinal position order.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn min_rev_starts_core<T: PartialOrd + Copy>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;

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

    let labels = index.iter().skip(min_start).copied().collect();
    Ok((labels, positions))
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// Finds the row containing the minimum value for each right-side
        /// label covered by the reverse suffix ranges.
        ///
        /// Null rows, identified by `booleans`, do not participate in the
        /// minimum. The returned positions use `-1` when no non-null row
        /// covers a label.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive ordinal start of each suffix range.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            let (labels, positions) = min_rev_starts_core(
                arr.as_array(),
                starts.as_array(),
                index.as_array(),
                booleans.as_array(),
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(labels).into_pyarray(py),
                Array1::from_vec(positions).into_pyarray(py),
            ))
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
        assert_eq!(got, Ok((vec![50, 10, 90], vec![1, 1, 1])));
    }

    #[test]
    fn null_and_all_null_groups_use_minus_one_position() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64, 1, 2];
        let index = array![10_i64, 20, 30];
        let booleans = array![true, false, true];
        let got = min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view());
        assert_eq!(got, Ok((vec![10, 20, 30], vec![-1, 1, 1])));
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
        assert_eq!(got, Ok((Vec::from_iter(0..20_i64), expected.to_vec())));
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
        assert_eq!(got, Ok((Vec::from_iter(0..20_i64), expected.to_vec())));
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
            Ok((vec![], vec![]))
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
