use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths, WrapAdd};

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

/// Sum values into one compact state slot for each distinct right-hand label.
///
/// ELI5: `starts` and `ends` describe little windows into `index`. We keep a
/// numbered drawer for each right-hand row position, rather than asking a
/// dictionary which drawer owns the position on every visit. `index[item]` is
/// the output label for that drawer. The caller contract guarantees that
/// `index` contains each right-row identity at most once; identities may be
/// reordered or have gaps, because the drawer number is `item`, not the value
/// stored in `index[item]`.
///
/// Empty `arr`/`index` inputs are rejected. Invalid or zero-width ranges are
/// skipped by `checked_range`, and valid ranges are half-open `[start, end)`.
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
) -> Result<(Array1<i64>, Array1<A>), &'static str>
where
    T: Copy,
    A: WrapAdd,
    F: FnMut(T) -> A,
{
    validate_inputs(arr, starts, ends, index, booleans)?;

    // ELI5: because each right row is unique, its position in `index` is
    // already a perfect drawer number. Gaps in the identity values do not
    // matter: `[7, 3, 11]` still has drawers `0`, `1`, and `2`.
    let mut seen = vec![false; index.len()];
    let mut touched = Vec::new();
    let mut totals = vec![A::ZERO; index.len()];

    for (current, start, end, boolean) in
        izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter())
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start..end {
            if !seen[item] {
                seen[item] = true;
                touched.push(item);
            }
            if *boolean {
                continue;
            }
            totals[item] = totals[item].wrap_add(convert(*current));
        }
    }

    let labels = touched.iter().map(|&item| index[item]).collect();
    let totals = touched.iter().map(|&item| totals[item]).collect();
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

    let mut seen = vec![false; index.len()];
    let mut touched = Vec::new();
    let mut totals = vec![0.0; index.len()];
    let mut compensations = vec![0.0; index.len()];

    for (current, start, end, boolean) in
        izip!(arr.iter(), starts.iter(), ends.iter(), booleans.iter())
    {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current = to_f64(*current);
        for item in start..end {
            if !seen[item] {
                seen[item] = true;
                touched.push(item);
            }
            if *boolean {
                continue;
            }
            let difference = current - compensations[item];
            let increment = totals[item] + difference;
            compensations[item] = (increment - totals[item]) - difference;
            // ELI5: compensation remembers tiny rounding crumbs. If an
            // infinity makes that crumb NaN, discard the crumb so the actual
            // infinity remains the result, matching pandas' summation rules.
            if !compensations[item].is_finite() {
                compensations[item] = 0.0;
            }
            totals[item] = increment;
        }
    }

    let labels = touched.iter().map(|&item| index[item]).collect();
    let totals = touched.iter().map(|&item| totals[item]).collect();
    Ok((Array1::from_vec(labels), Array1::from_vec(totals)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
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
        assert_eq!(got, Ok((array![10], array![value])));
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
        assert_eq!(got, Ok((array![42, 7, 100], array![4, 3, 2])));
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
