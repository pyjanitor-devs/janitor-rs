use crate::aggs::checked_index;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if arr.len() != booleans.len() {
        return Err("arr and booleans must have equal lengths");
    }
    if left_index.len() != right_index.len() {
        return Err("left_index and right_index must have equal lengths");
    }
    if arr.is_empty() || left_index.is_empty() || right_index.is_empty() {
        return Err("arr, left_index, and right_index cannot be empty");
    }
    Ok(())
}

/// Sums no-range rows with arbitrary right labels using compact label slots.
///
/// ELI5: the map stores one total per arbitrary label, while the label vector
/// records first-seen order so output is deterministic.
pub fn sum_rev_no_range_int_core<T: Copy, F: FnMut(T) -> i64>(
    arr: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_i64: F,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    validate_inputs(arr, left_index, right_index, booleans)?;
    // ELI5: reserve the lookup table for the full join, but let output state
    // grow only as distinct labels appear; duplicate-heavy inputs should not
    // preallocate one result slot per matched row.
    let capacity = right_index.len();
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::new();
    let mut totals = Vec::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let slot = match slots.entry(*index_right) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = labels.len();
                labels.push(*index_right);
                totals.push(0_i64);
                entry.insert(slot);
                slot
            }
        };
        if !booleans[left] {
            totals[slot] = totals[slot].wrapping_add(to_i64(arr[left]));
        }
    }

    Ok((Array1::from_vec(labels), Array1::from_vec(totals)))
}

/// `u64`-native counterpart to `sum_rev_no_range_int_core`.
///
/// ELI5: `uint64` values `>= 2**63` don't fit in `i64`; funneling them
/// through the shared `i64` accumulator wraps them to a negative number.
/// This keeps the accumulator and the returned totals in `u64` end-to-end
/// so large unsigned sums come back correct instead of sign-flipped.
pub fn sum_rev_no_range_u64_core(
    arr: ArrayView1<'_, u64>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<u64>), &'static str> {
    validate_inputs(arr, left_index, right_index, booleans)?;
    let capacity = right_index.len();
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::new();
    let mut totals = Vec::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let slot = match slots.entry(*index_right) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = labels.len();
                labels.push(*index_right);
                totals.push(0_u64);
                entry.insert(slot);
                slot
            }
        };
        if !booleans[left] {
            totals[slot] = totals[slot].wrapping_add(arr[left]);
        }
    }

    Ok((Array1::from_vec(labels), Array1::from_vec(totals)))
}

/// Float counterpart to `sum_rev_no_range_int_core`.
///
/// ELI5: one compact slot owns both the running total and its compensation,
/// so the float path no longer maintains two HashMaps with the same labels.
pub fn sum_rev_no_range_float_core<T: Copy, F: FnMut(T) -> f64>(
    arr: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str> {
    validate_inputs(arr, left_index, right_index, booleans)?;
    let capacity = right_index.len();
    let mut slots = HashMap::<i64, usize>::with_capacity(capacity);
    let mut labels = Vec::new();
    let mut values = Vec::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let slot = match slots.entry(*index_right) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let slot = labels.len();
                labels.push(*index_right);
                values.push((0.0, 0.0));
                entry.insert(slot);
                slot
            }
        };
        if booleans[left] {
            continue;
        }
        let current = to_f64(arr[left]);
        let (total, compensation) = &mut values[slot];
        let difference = current - *compensation;
        let increment = *total + difference;
        *compensation = (increment - *total) - difference;
        if !compensation.is_finite() {
            *compensation = 0.;
        }
        *total = increment;
    }

    let result = values.into_iter().map(|(total, _)| total).collect();
    Ok((Array1::from_vec(labels), result))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        /// Sum joined values by right-side label without range metadata.
        /// Integer accumulation wraps on overflow; null rows are skipped.
        ///
        /// # Arguments
        /// * `arr` - Left-side values; must not be empty.
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
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            let (indexers, result) = sum_rev_no_range_int_core(
                arr.as_array(),
                left_index.as_array(),
                right_index.as_array(),
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
        /// Sum joined floating-point values by right-side label without range
        /// metadata, using compensated accumulation.
        ///
        /// # Arguments
        /// * `arr` - Left-side values; must not be empty.
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
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            let (indexers, result) = sum_rev_no_range_float_core(
                arr.as_array(),
                left_index.as_array(),
                right_index.as_array(),
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

/// `uint64` export: returns `u64` totals so values `>= 2**63` survive
/// the round trip to Python instead of wrapping to a negative `i64`.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn compute_sum_rev_no_range_uint64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<u64>>)> {
    let (indexers, result) = sum_rev_no_range_u64_core(
        arr.as_array(),
        left_index.as_array(),
        right_index.as_array(),
        booleans.as_array(),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

compute_ints!(compute_sum_rev_no_range_int64, i64);
compute_ints!(compute_sum_rev_no_range_int32, i32);
compute_ints!(compute_sum_rev_no_range_int16, i16);
compute_ints!(compute_sum_rev_no_range_int8, i8);
compute_ints!(compute_sum_rev_no_range_uint32, u32);
compute_ints!(compute_sum_rev_no_range_uint16, u16);
compute_ints!(compute_sum_rev_no_range_uint8, u8);
compute_floats!(compute_sum_rev_no_range_f64, f64);
compute_floats!(compute_sum_rev_no_range_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn integer_core_compresses_arbitrary_labels_deterministically() {
        let got = sum_rev_no_range_int_core(
            array![5_i64, 2, 7].view(),
            array![0_i64, 1, 2, 1].view(),
            array![20_i64, 40, 20, 40].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![20, 40], array![12, 4])));
    }

    #[test]
    fn u64_core_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let got = sum_rev_no_range_u64_core(
            array![value].view(),
            array![0_i64].view(),
            array![20_i64].view(),
            array![false].view(),
        );
        assert_eq!(got, Ok((array![20], array![value])));
    }

    #[test]
    fn integer_core_wraps_and_skips_null_values() {
        let got = sum_rev_no_range_int_core(
            array![i64::MAX, 1, 4].view(),
            array![0_i64, 1, 2].view(),
            array![20_i64, 20, 40].view(),
            array![false, false, true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![20, 40], array![i64::MIN, 0])));
    }

    #[test]
    fn float_core_handles_infinity_all_null_and_f32_promotion() {
        let got = sum_rev_no_range_float_core(
            array![f64::INFINITY, 2.0].view(),
            array![0_i64, 1].view(),
            array![20_i64, 20].view(),
            array![false, false].view(),
            |value| value,
        )
        .unwrap();
        assert!(got.1[0].is_infinite());

        let got = sum_rev_no_range_float_core(
            array![1.5_f32].view(),
            array![0_i64].view(),
            array![40_i64].view(),
            array![true].view(),
            |value| value as f64,
        );
        assert_eq!(got, Ok((array![40], array![0.0])));
    }

    #[test]
    fn rejects_mismatched_pair_lengths() {
        assert!(sum_rev_no_range_int_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![20_i64, 40].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_left_positions_before_aggregation() {
        assert!(sum_rev_no_range_int_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
        assert!(sum_rev_no_range_int_core(
            array![1_i64].view(),
            array![1_i64].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
    }

    #[test]
    fn rejects_left_positions_that_do_not_fit_usize() {
        let oversized = i64::from(u32::MAX) + 1;
        assert!(sum_rev_no_range_int_core(
            array![1_i64].view(),
            array![oversized].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
        assert!(sum_rev_no_range_float_core(
            array![1.0_f64].view(),
            array![oversized].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
    }
}
