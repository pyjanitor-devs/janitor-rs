use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, ensure_equal_lengths_core, ensure_nonempty_core};
use std::collections::HashMap;

/// Find the maximum contributing row position for each right-side label
/// without range metadata. Null rows create labels but cannot win.
/// Output label/position pairs are aligned, but their order is unspecified.
pub fn max_rev_no_range_core<T: Copy + PartialOrd>(
    arr: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    // Empty inputs are rejected before shape mismatches intentionally. This
    // is the compatibility contract for the shared core validation helpers.
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("left_index", left_index.len())?;
    ensure_nonempty_core("right_index", right_index.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_equal_lengths_core(
        "left_index",
        left_index.len(),
        "right_index",
        right_index.len(),
    )?;
    // ELI5: grow one lookup table only for labels actually observed. The
    // compact layout still avoids the second map used by the old reducer.
    // This saves memory for repeated labels, at the cost of rehashing when
    // nearly every match has a unique label.
    let mut states = HashMap::<i64, (T, i64)>::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let current = arr[left];
        let boolean = booleans[left];
        let state = states.entry(*index_right).or_insert((current, -1));
        if !boolean && (state.1 == -1 || current > state.0) {
            *state = (current, left as i64);
        }
    }

    let mut labels = Vec::with_capacity(states.len());
    let mut positions = Vec::with_capacity(states.len());
    for (label, (_, position)) in states {
        labels.push(label);
        positions.push(position);
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(positions)))
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// Find the maximum contributing row position for each right-side
        /// label without range metadata.
        ///
        /// `arr`, `left_index`, and `right_index` must not be empty. Null
        /// rows create labels but cannot win the maximum.
        ///
        /// # Arguments
        /// * `arr` - Left-side values.
        /// * `left_index` - Positions into `arr`.
        /// * `right_index` - Output label for each joined row.
        /// * `booleans` - Null mask; `True` rows are ignored.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            let booleans = booleans.as_array();
            let (indexers, result) = max_rev_no_range_core(arr, left_index, right_index, booleans)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_max_rev_no_range_int64, i64);
compute!(compute_max_rev_no_range_int32, i32);
compute!(compute_max_rev_no_range_int16, i16);
compute!(compute_max_rev_no_range_int8, i8);
compute!(compute_max_rev_no_range_uint64, u64);
compute!(compute_max_rev_no_range_uint32, u32);
compute!(compute_max_rev_no_range_uint16, u16);
compute!(compute_max_rev_no_range_uint8, u8);
compute!(compute_max_rev_no_range_f64, f64);
compute!(compute_max_rev_no_range_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn core_returns_correct_label_position_pairs() {
        let got = max_rev_no_range_core(
            array![5_i64, 2, 7].view(),
            array![0_i64, 1, 2, 1].view(),
            array![20_i64, 40, 20, 40].view(),
            array![false, false, false].view(),
        );
        let (labels, positions) = got.unwrap();
        let mut got: Vec<_> = labels
            .iter()
            .zip(positions.iter())
            .map(|(&label, &position)| (label, position))
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![(20, 2), (40, 1)]);
    }

    #[test]
    fn core_supports_every_dtype() {
        macro_rules! assert_max {
            ($type:ty) => {
                let got = max_rev_no_range_core(
                    array![5 as $type, 2 as $type, 7 as $type].view(),
                    array![0_i64, 1, 2, 1].view(),
                    array![20_i64, 40, 20, 40].view(),
                    array![false, false, false].view(),
                );
                let (labels, positions) = got.unwrap();
                let mut got: Vec<_> = labels
                    .iter()
                    .zip(positions.iter())
                    .map(|(&label, &position)| (label, position))
                    .collect();
                got.sort_unstable();
                assert_eq!(got, vec![(20, 2), (40, 1)]);
            };
        }

        assert_max!(i64);
        assert_max!(i32);
        assert_max!(i16);
        assert_max!(i8);
        assert_max!(u64);
        assert_max!(u32);
        assert_max!(u16);
        assert_max!(u8);
        assert_max!(f64);
        assert_max!(f32);
    }

    #[test]
    fn core_preserves_null_and_tie_behavior() {
        let got = max_rev_no_range_core(
            array![5_i64, 2].view(),
            array![0_i64, 1, 0].view(),
            array![20_i64, 40, 40].view(),
            array![true, false].view(),
        );
        let (labels, positions) = got.unwrap();
        let mut got: Vec<_> = labels
            .iter()
            .zip(positions.iter())
            .map(|(&label, &position)| (label, position))
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![(20, -1), (40, 1)]);

        let got = max_rev_no_range_core(
            array![5_i64, 5, 4].view(),
            array![0_i64, 1, 2].view(),
            array![20_i64, 20, 20].view(),
            array![false, false, false].view(),
        );
        assert_eq!(got, Ok((array![20], array![0])));
    }

    #[test]
    fn core_rejects_mismatches_and_invalid_positions() {
        assert!(max_rev_no_range_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![20_i64, 40].view(),
            array![false].view(),
        )
        .is_err());
        assert!(max_rev_no_range_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![20_i64].view(),
            array![false].view(),
        )
        .is_err());
        assert!(max_rev_no_range_core(
            array![1_i64].view(),
            array![i64::from(u32::MAX) + 1].view(),
            array![20_i64].view(),
            array![false].view(),
        )
        .is_err());
    }

    #[test]
    fn empty_input_error_precedes_shape_mismatch() {
        let error = max_rev_no_range_core(
            Array1::<i64>::zeros(0).view(),
            array![].view(),
            array![20_i64].view(),
            array![false].view(),
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");
    }
}
