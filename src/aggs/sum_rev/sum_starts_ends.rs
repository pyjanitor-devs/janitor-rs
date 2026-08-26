use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_range;
use std::collections::{hash_map::Entry, HashMap};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if starts.len() != ends.len() {
        return Err("starts and ends must have equal lengths");
    }
    if arr.len() != starts.len() {
        return Err("arr, starts, and ends must have equal lengths");
    }
    if arr.len() != booleans.len() {
        return Err("arr and booleans must have equal lengths");
    }
    if arr.is_empty() || index.is_empty() {
        return Err("arr, starts, booleans, and index cannot be empty");
    }
    Ok(())
}

fn capacity_hint(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    right_len: usize,
) -> usize {
    // ELI5: one interval can introduce at most one new label per position it
    // covers. Summing those widths gives a safe, cheap ceiling for the
    // number of distinct labels, while `min` prevents a malformed/overlapping
    // workload from asking the map for more buckets than the index contains.
    starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| checked_range(*start, *end, right_len))
        .fold(0_usize, |total, (start, end)| {
            total.saturating_add(end - start)
        })
        .min(right_len)
}

/// Sum values into one compact state slot for each distinct right-hand label.
///
/// ELI5: `starts` and `ends` describe little windows into `index`. We keep a
/// small numbered drawer for each label we actually encounter, rather than
/// carrying a separate dictionary for labels and totals. Duplicate labels
/// therefore share one drawer and do not allocate duplicate aggregate state.
pub fn sum_rev_start_end_int_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_i64: F,
) -> Result<(Array1<i64>, Array1<i64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    validate_inputs(arr, starts, ends, index, booleans)?;

    // ELI5: reserving up to the total number of positions covered avoids
    // repeated rehashing when most labels are unique, but does not reserve
    // the whole right index when only a few narrow ranges can be touched.
    let hint = capacity_hint(starts, ends, index.len());
    let mut slots = HashMap::with_capacity(hint);
    let mut labels = Vec::new();
    let mut totals = Vec::new();

    for (current, start, end, boolean) in
        izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter())
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start..end {
            let label = index[item];
            let slot = match slots.entry(label) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    totals.push(0_i64);
                    slot
                }
            };
            if *boolean {
                continue;
            }
            totals[slot] = totals[slot].wrapping_add(to_i64(*current));
        }
    }

    Ok((Array1::from_vec(labels), Array1::from_vec(totals)))
}

pub fn sum_rev_start_end_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    validate_inputs(arr, starts, ends, index, booleans)?;

    let hint = capacity_hint(starts, ends, index.len());
    let mut slots = HashMap::with_capacity(hint);
    let mut labels = Vec::new();
    let mut totals = Vec::new();
    let mut compensations = Vec::new();

    for (current, start, end, boolean) in
        izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter())
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current = to_f64(*current);
        for item in start..end {
            let label = index[item];
            let slot = match slots.entry(label) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    totals.push(0.0);
                    compensations.push(0.0);
                    slot
                }
            };
            if *boolean {
                continue;
            }
            let difference = current - compensations[slot];
            let increment = totals[slot] + difference;
            compensations[slot] = (increment - totals[slot]) - difference;
            // ELI5: compensation remembers tiny rounding crumbs. If an
            // infinity makes that crumb NaN, discard the crumb so the actual
            // infinity remains the result, matching pandas' summation rules.
            if !compensations[slot].is_finite() {
                compensations[slot] = 0.0;
            }
            totals[slot] = increment;
        }
    }

    Ok((Array1::from_vec(labels), Array1::from_vec(totals)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let (indexers, result) = sum_rev_start_end_int_core(
                arr.as_array(),
                starts.as_array(),
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

compute_ints!(compute_sum_rev_start_end_int64, i64);
compute_ints!(compute_sum_rev_start_end_int32, i32);
compute_ints!(compute_sum_rev_start_end_int16, i16);
compute_ints!(compute_sum_rev_start_end_int8, i8);
compute_ints!(compute_sum_rev_start_end_uint64, u64);
compute_ints!(compute_sum_rev_start_end_uint32, u32);
compute_ints!(compute_sum_rev_start_end_uint16, u16);
compute_ints!(compute_sum_rev_start_end_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let (indexers, result) = sum_rev_start_end_float_core(
                arr.as_array(),
                starts.as_array(),
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

compute_floats!(compute_sum_rev_start_end_f64, f64);
compute_floats!(compute_sum_rev_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    // ELI5: both macros above used to hardcode `arr`'s PyO3 type to `i64`
    // regardless of `$type`, so every non-i64 int export and *both* float
    // exports actually demanded an i64 numpy array at the Python boundary
    // -- silently for ints, with a TypeError for floats. These fn-pointer
    // typedefs make that a compile error again: reintroducing the hardcoded
    // `i64` breaks compilation instead of only failing at runtime.
    type Int8Fn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, i8>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
    )
        -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

    type F32Fn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, f32>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
    )
        -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>;

    type F64Fn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, f64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
    )
        -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>;

    #[test]
    fn int8_wrapper_accepts_an_int8_array() {
        let _wrapper: Int8Fn = compute_sum_rev_start_end_int8;
    }

    #[test]
    fn f32_wrapper_accepts_an_f32_array() {
        let _wrapper: F32Fn = compute_sum_rev_start_end_f32;
    }

    #[test]
    fn f64_wrapper_accepts_an_f64_array() {
        let _wrapper: F64Fn = compute_sum_rev_start_end_f64;
    }

    #[test]
    fn compact_slots_sum_duplicate_labels_and_arbitrary_ranges() {
        let got = sum_rev_start_end_int_core(
            array![1_i64, 2, 3].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10, 20], array![6, 3])));
    }

    #[test]
    fn null_rows_still_emit_touched_labels_with_zero_totals() {
        let got = sum_rev_start_end_int_core(
            array![5_i64].view(),
            array![0_i64].view(),
            array![2_i64].view(),
            array![10_i64, 20].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10, 20], array![0, 0])));
    }

    #[test]
    fn invalid_or_zero_width_ranges_are_skipped_without_panicking() {
        let got = sum_rev_start_end_int_core(
            array![1_i64, 2, 3].view(),
            array![2_i64, -1, 1].view(),
            array![2_i64, 1, 4].view(),
            array![10_i64, 20].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![], array![])));
    }

    #[test]
    fn validation_rejects_shape_mismatches_and_empty_inputs() {
        let index = array![10_i64];
        let booleans = array![false];
        assert!(sum_rev_start_end_int_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64, 1].view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .is_err());
        assert!(sum_rev_start_end_int_core(
            array![].view(),
            array![].view(),
            array![].view(),
            index.view(),
            array![].view(),
            |value: i64| value,
        )
        .is_err());
    }

    #[test]
    fn float_core_keeps_compensated_sum_state_per_label() {
        let got = sum_rev_start_end_float_core(
            array![0.1_f64, 0.2].view(),
            array![0_i64, 0].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(got.0, array![10]);
        assert!((got.1[0] - 0.3).abs() < f64::EPSILON);
    }
}
