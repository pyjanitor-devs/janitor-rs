use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    ensure_equal_lengths, into_starts_ends_result, starts_domain, starts_labels, WrapAdd,
};

fn validate_starts_inputs(
    arr_len: usize,
    starts_len: usize,
    booleans_len: usize,
) -> Result<(), &'static str> {
    if arr_len != starts_len || arr_len != booleans_len {
        return Err("arr, starts, and booleans must have equal lengths");
    }
    Ok(())
}

/// Accumulate reverse-sum `starts` rows in slots for the touched candidate
/// suffix, then emit the original right labels from `index`.
///
/// ELI5: `item` is the compact candidate ordinal used to address the
/// accumulator, while `index[item]` is only the original right-row label to
/// return. The slots cover the union of all suffixes, so their count is the
/// touched width rather than the full right domain. `A` is the accumulator
/// type: every integer dtype instantiates this with `A = i64`, except
/// `uint64`, which instantiates it with `A = u64` so values `>= 2**63`
/// don't get sign-flipped by a forced `i64` cast (see `WrapAdd`).
pub fn sum_rev_starts_int_core<T, A, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut convert: F,
) -> Result<(Array1<i64>, Array1<A>), &'static str>
where
    T: Copy,
    A: WrapAdd,
    F: FnMut(T) -> A,
{
    validate_starts_inputs(arr.len(), starts.len(), booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;
    let mut values = vec![A::ZERO; width];
    let zipped = izip!(arr.into_iter(), starts.into_iter(), booleans.into_iter());
    for (current, start, boolean) in zipped {
        let current_ = convert(*current);
        for value in values.iter_mut().skip(*start as usize - min_start) {
            // Every valid suffix position is already part of the contiguous
            // output domain; a null left value contributes zero to its slot.
            if *boolean {
                continue;
            }
            *value = value.wrap_add(current_);
        }
    }
    Ok((starts_labels(min_start, index), Array1::from_vec(values)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "starts",
                starts.as_array().len(),
            )?;
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            into_starts_ends_result(
                py,
                sum_rev_starts_int_core(
                    arr.as_array(),
                    starts.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                    |value| value as $acc,
                ),
            )
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapAdd`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_sum_rev_start_int64, i64, i64);
compute_ints!(compute_sum_rev_start_int32, i32, i64);
compute_ints!(compute_sum_rev_start_int16, i16, i64);
compute_ints!(compute_sum_rev_start_int8, i8, i64);
compute_ints!(compute_sum_rev_start_uint64, u64, u64);
compute_ints!(compute_sum_rev_start_uint32, u32, i64);
compute_ints!(compute_sum_rev_start_uint16, u16, i64);
compute_ints!(compute_sum_rev_start_uint8, u8, i64);

/// Pure-Rust reverse-sum core for the float path.
///
/// Each compact candidate slot carries a Neumaier-compensated
/// `(total, compensation)` pair in one `Vec`, so the pair cannot desync.
pub fn sum_rev_starts_float_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    validate_starts_inputs(arr.len(), starts.len(), booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;
    let mut slots = vec![(0.0_f64, 0.0_f64); width];
    let zipped = izip!(arr.into_iter(), starts.into_iter(), booleans.into_iter());
    for (current, start, boolean) in zipped {
        let current_ = to_f64(*current);
        for (total, compensation) in slots.iter_mut().skip(*start as usize - min_start) {
            if *boolean {
                continue;
            }
            let difference = current_ - *compensation;
            let increment = *total + difference;
            *compensation = (increment - *total) - difference;
            // adapted from pandas' cython code
            // # GH#53606; GH#60303
            // # If val is +/- infinity compensation is NaN
            // # which would lead to results being NaN instead
            // # of +/- infinity. We cannot use util.is_nan
            // # because of no gil
            if !compensation.is_finite() {
                *compensation = 0.;
            }
            *total = increment;
        }
    }
    let result = slots.into_iter().map(|(total, _)| total).collect();
    Ok((starts_labels(min_start, index), result))
}

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "starts",
                starts.as_array().len(),
            )?;
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            into_starts_ends_result(
                py,
                sum_rev_starts_float_core(
                    arr.as_array(),
                    starts.as_array(),
                    index.as_array(),
                    booleans.as_array(),
                    |value| value as f64,
                ),
            )
        }
    };
}

compute_floats!(compute_sum_rev_start_f64, f64);
compute_floats!(compute_sum_rev_start_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn overlapping_left_rows_accumulate_and_emit_candidate_order() {
        // index is a permutation of 0..3 -- the only shape `index` can
        // legitimately take for this join type.
        let arr = array![5_i64, 6];
        let starts = array![1_i64, 0];
        let index = array![2_i64, 0, 1];
        let booleans = array![false, false];
        let (indexers, result) = sum_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        // row0 (start=1) touches index[1..3] = {0, 1}; row1 (start=0)
        // touches index[0..3] = {2, 0, 1}.
        assert_eq!(indexers, array![2, 0, 1]);
        assert_eq!(result, array![6, 11, 11]);
    }

    #[test]
    fn a_touched_position_can_exceed_the_length_capacity_hint() {
        // Regression test for the reasoning correction made while writing
        // this file: `length = ends - starts.min()` is an exact *count* of
        // touched positions, not a bound on their *values*. Here
        // `starts.min() == 6`, so `length` would be `8 - 6 == 2`, but the
        // two positions actually touched are 6 and 7 -- indexing a `Vec`
        // sized `length` (2) by either would panic. The kernel must size
        // its accumulator by `index.len()` (8), not by `length`.
        let arr = array![10_i64];
        let starts = array![6_i64];
        let index = array![0_i64, 1, 2, 3, 4, 5, 7, 6];
        let booleans = array![false];
        let (indexers, result) = sum_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(indexers, array![7, 6]);
        assert_eq!(result, array![10, 10]);
    }

    #[test]
    fn a_null_left_row_still_touches_its_slot_at_zero() {
        // Mirrors the old `HashMap::entry(...).or_insert(...)` call
        // happening before the null-mask `continue`: the row still "shows
        // up" for every right position it matches, just at the identity
        // value (0 for sum), instead of disappearing from the output.
        let arr = array![99_i64];
        let starts = array![0_i64];
        let index = array![0_i64];
        let booleans = array![true];
        let (indexers, result) = sum_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(indexers, array![0]);
        assert_eq!(result, array![0]);
    }

    #[test]
    fn float_path_merges_total_and_compensation_in_one_slot() {
        // Three left rows all feed the same right position -- exercises
        // the merged (total, compensation) tuple (the shape issue #48
        // targets) across repeated summation.
        let arr = array![0.1_f64, 0.2, 0.3];
        let starts = array![0_i64, 0, 0];
        let index = array![0_i64];
        let booleans = array![false, false, false];
        let (indexers, result) = sum_rev_starts_float_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        assert_eq!(indexers, array![0]);
        assert!((result[0] - 0.6).abs() < 1e-9);
    }

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let (indexers, result) = sum_rev_starts_int_core(
            array![value].view(),
            array![0_i64].view(),
            array![20_i64].view(),
            array![false].view(),
            |v: u64| v,
        )
        .unwrap();
        assert_eq!(indexers, array![20]);
        assert_eq!(result, array![value]);
    }

    #[test]
    fn rejects_invalid_start_before_allocation() {
        let arr = array![5_i64];
        let starts = array![-1_i64];
        let index = array![0_i64];
        let booleans = array![false];

        let error = sum_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap_err();

        assert_eq!(error, "starts must satisfy 0 <= start <= right_len");
    }
}
