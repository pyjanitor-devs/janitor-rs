use itertools::izip;
use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_equal_lengths_core, ensure_nonempty_core,
    should_use_dense_match_storage, WrapAdd,
};

/// Sum values into one compact state slot for each distinct right-hand label.
///
/// janitor-rs is primarily called by pyjanitor. Its conditional-join path
/// resets the right DataFrame index to unique row labels before sorting or
/// filtering, so labels can be reordered or gapped but are not duplicated.
/// `item`, the ordinal position in `index`, is the state slot; `index[item]` is
/// the output label for that slot.
///
/// ELI5: `starts` and `ends` describe little windows into `index`. We use the
/// shelf position as the drawer number, even when the printed labels have
/// gaps, then print the original label from that drawer.
///
/// Invalid or zero-width ranges are skipped by `checked_range`, and valid
/// ranges are half-open `[start, end)`.
///
/// # Preconditions
///
/// `arr` and `index` cannot be empty.
///
/// `index` must contain unique labels in positional order. This is not checked
/// here: pyjanitor supplies unique right-side labels, while a direct caller is
/// responsible for this correctness precondition. The array position is the
/// ordinal state slot; label values need not be positional. Duplicate labels
/// are unsupported and are accumulated independently by ordinal.
/// `A` is the accumulator type: every integer dtype instantiates this with
/// `A = i64`, except `uint64`, which instantiates it with `A = u64` so
/// values `>= 2**63` don't get sign-flipped by a forced `i64` cast (see
/// `WrapAdd`).
pub fn sum_rev_start_end_int_core<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Vec<i64>, Vec<A>), String>
where
    T: Copy,
    A: WrapAdd,
    F: FnMut(T) -> A,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut min_start = index.len();
    let mut max_end = 0_usize;
    let mut total_width = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start, end)) = checked_range(*start, *end, index.len()) {
            min_start = min_start.min(start);
            max_end = max_end.max(end);
            total_width = total_width.saturating_add(end - start);
        }
    }
    let width = max_end.saturating_sub(min_start);
    // ELI5: if the ranges collectively represent enough positional work,
    // direct array indexing is cheaper than dictionary lookups. This value is
    // only a dispatch estimate; `width` remains the actual dense allocation.
    let dense = should_use_dense_match_storage(index.len(), total_width);
    let mut touched = if dense {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };

    if dense {
        let mut seen = vec![false; width];
        let mut totals = vec![A::ZERO; width];
        for (current, (start, end), boolean) in
            izip!(arr.iter(), starts.iter().zip(ends.iter()), booleans.iter())
        {
            let Some((start, end)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            let current = *current;
            let boolean = *boolean;
            for item in start..end {
                // `min_start` is the smallest validated start, so every
                // valid item is at least it. This subtraction therefore maps
                // an absolute position into the compact array safely.
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                if !boolean {
                    totals[slot] = totals[slot].wrap_add(convert(current));
                }
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            values.push(totals[slot]);
        }
        return Ok((labels, values));
    }

    let mut totals: HashMap<usize, A> = HashMap::new();
    for (current, (start, end), boolean) in
        izip!(arr.iter(), starts.iter().zip(ends.iter()), booleans.iter())
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current = *current;
        let boolean = *boolean;
        for item in start..end {
            let slot = item - min_start;
            let total = totals.entry(slot).or_insert_with(|| {
                touched.push(slot);
                A::ZERO
            });
            if !boolean {
                *total = total.wrap_add(convert(current));
            }
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        values.push(totals[&slot]);
    }
    Ok((labels, values))
}

pub fn sum_rev_start_end_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Vec<i64>, Vec<f64>), String>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("index", index.len())?;
    // Integer reverse sums use `WrapAdd` because the project defines their
    // overflow behavior as modular wrapping. Floats do not have an
    // equivalent wrapping operation: IEEE-754 addition deliberately produces
    // infinities or NaN for overflow/invalid arithmetic. Keep the existing
    // compensated floating-point sequence instead of forcing integer rules
    // onto floating-point values.
    //
    // ELI5: an integer odometer rolls back to zero after its last digit; a
    // float thermometer goes to infinity (or becomes NaN) instead. They need
    // different arithmetic rules.
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let mut min_start = index.len();
    let mut max_end = 0_usize;
    let mut total_width = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start, end)) = checked_range(*start, *end, index.len()) {
            min_start = min_start.min(start);
            max_end = max_end.max(end);
            total_width = total_width.saturating_add(end - start);
        }
    }
    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), total_width);
    let mut touched = if dense {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };

    if dense {
        let mut seen = vec![false; width];
        let mut slots = vec![(0.0_f64, 0.0_f64); width];
        for (current, (start, end), boolean) in
            izip!(arr.iter(), starts.iter().zip(ends.iter()), booleans.iter())
        {
            let Some((start, end)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            let current = *current;
            let boolean = *boolean;
            let current = if boolean { None } else { Some(to_f64(current)) };
            for item in start..end {
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                if let Some(current) = current {
                    let difference = current - slots[slot].1;
                    let increment = slots[slot].0 + difference;
                    slots[slot].1 = (increment - slots[slot].0) - difference;
                    // ELI5: compensation tracks the tiny rounding remainder.
                    // Once arithmetic produces infinity or NaN, that
                    // remainder is no longer meaningful, so discard it and
                    // continue with the ordinary running total.
                    if !slots[slot].1.is_finite() {
                        slots[slot].1 = 0.0;
                    }
                    slots[slot].0 = increment;
                }
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            values.push(slots[slot].0);
        }
        return Ok((labels, values));
    }

    let mut slots: HashMap<usize, (f64, f64)> = HashMap::new();
    for (current, (start, end), boolean) in
        izip!(arr.iter(), starts.iter().zip(ends.iter()), booleans.iter())
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current = *current;
        let boolean = *boolean;
        let current = if boolean { None } else { Some(to_f64(current)) };
        for item in start..end {
            // `min_start` is the smallest validated start, so this unsigned
            // subtraction cannot underflow while translating to a slot.
            let slot = item - min_start;
            let state = slots.entry(slot).or_insert_with(|| {
                touched.push(slot);
                (0.0, 0.0)
            });
            if let Some(current) = current {
                let difference = current - state.1;
                let increment = state.0 + difference;
                state.1 = (increment - state.0) - difference;
                // ELI5: after overflow or an invalid float operation, the
                // compensation remainder is no longer usable. Reset it so a
                // stale NaN/ infinity does not poison later updates.
                if !state.1.is_finite() {
                    state.1 = 0.0;
                }
                state.0 = increment;
            }
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        values.push(slots[&slot].0);
    }
    Ok((labels, values))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// `index` must contain unique labels. Positions in the array are the
        /// ordinal state slots; direct callers must preserve that contract.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let (indexers, result) =
                sum_rev_start_end_int_core(arr, starts, ends, index, booleans, |value| {
                    value as $acc
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapAdd`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_sum_rev_start_end_int64, i64, i64);
compute_ints!(compute_sum_rev_start_end_int32, i32, i64);
compute_ints!(compute_sum_rev_start_end_int16, i16, i64);
compute_ints!(compute_sum_rev_start_end_int8, i8, i64);
compute_ints!(compute_sum_rev_start_end_uint64, u64, u64);
compute_ints!(compute_sum_rev_start_end_uint32, u32, i64);
compute_ints!(compute_sum_rev_start_end_uint16, u16, i64);
compute_ints!(compute_sum_rev_start_end_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// `index` must contain unique labels. Positions in the array are the
        /// ordinal state slots; direct callers must preserve that contract.
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
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let (indexers, result) =
                sum_rev_start_end_float_core(arr, starts, ends, index, booleans, |value| {
                    value as f64
                })
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
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let got = sum_rev_start_end_int_core(
            array![value].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![false].view(),
            |v: u64| v,
        );
        assert_eq!(got, Ok((vec![10], vec![value])));
    }

    #[test]
    fn dense_slots_sum_unique_gapped_labels_and_arbitrary_ranges() {
        let got = sum_rev_start_end_int_core(
            array![1_i64, 2, 3].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![42_i64, 7, 100].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((vec![42, 7, 100], vec![4, 3, 2])));
    }

    #[test]
    fn duplicate_index_labels_are_explicitly_unsupported() {
        // The dense reducer is positional, not label-keyed. This input is
        // intentionally unsupported: equal labels remain separate slots
        // rather than being merged as they were by the old HashMap path.
        let got = sum_rev_start_end_int_core(
            array![1_i64, 2, 3].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((vec![10, 20, 10], vec![4, 3, 2])));
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
        assert_eq!(got, Ok((vec![10, 20], vec![0, 0])));
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
        assert_eq!(got, Ok((vec![], vec![])));
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
        assert!(sum_rev_start_end_int_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![].view(),
            booleans.view(),
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
        assert_eq!(got.0, vec![10]);
        assert!((got.1[0] - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn float_null_row_still_emits_its_label_with_zero() {
        let got = sum_rev_start_end_float_core(
            array![5.0_f64].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![true].view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(got, (vec![10], vec![0.0]));
    }

    #[test]
    fn float_null_rows_emit_disjoint_labels_with_zero() {
        let got = sum_rev_start_end_float_core(
            array![5.0_f64, 7.0].view(),
            array![0_i64, 2].view(),
            array![1_i64, 3].view(),
            array![10_i64, 20, 30].view(),
            array![true, true].view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(got, (vec![10, 30], vec![0.0, 0.0]));
    }

    #[test]
    fn float_mixed_rows_preserve_null_only_labels() {
        let got = sum_rev_start_end_float_core(
            array![2.0_f64, 5.0].view(),
            array![0_i64, 1].view(),
            array![1_i64, 2].view(),
            array![10_i64, 20].view(),
            array![false, true].view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(got, (vec![10, 20], vec![2.0, 0.0]));
    }
}
