use itertools::izip;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    ends_domain, ends_labels, ensure_equal_lengths, ensure_equal_lengths_core,
    into_starts_ends_result, sweep_reduce, WrapAdd,
};

/// Accumulate reverse-sum `ends` rows in compact candidate-ordinal slots.
///
/// ELI5: `item` addresses the accumulator and `index[item]` is only the
/// original right-row label returned at the end, so sparse labels never
/// inflate the accumulator or become an out-of-bounds address.
/// `A` is the accumulator type: every integer dtype instantiates this with
/// `A = i64`, except `uint64`, which instantiates it with `A = u64` so
/// values `>= 2**63` don't get sign-flipped by a forced `i64` cast (see
/// `WrapAdd`).
#[allow(private_bounds)]
pub fn sum_rev_ends_int_core<T, A, F>(
    arr: numpy::ndarray::ArrayView1<T>,
    ends: numpy::ndarray::ArrayView1<i64>,
    index: numpy::ndarray::ArrayView1<i64>,
    booleans: numpy::ndarray::ArrayView1<bool>,
    mut convert: F,
) -> Result<(numpy::ndarray::Array1<i64>, numpy::ndarray::Array1<A>), &'static str>
where
    T: Copy,
    A: WrapAdd,
    F: FnMut(T) -> A,
{
    ensure_equal_lengths_core(
        arr.len(),
        ends.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    ensure_equal_lengths_core(
        arr.len(),
        booleans.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    let max_end = ends_domain(ends, index.len())?;

    // ELI5: an end-bound row covers a prefix. Put its value at the last slot
    // it covers, then sweep right-to-left and carry the running sum leftward.
    // Integer wrapping addition is associative, so grouping values at a
    // boundary preserves results. Using `end - 1` means each event can be
    // replaced by its final value after it is read; `end == 0` is an empty
    // prefix and has no event.
    let events = izip!(arr, ends, booleans)
        .filter(|(_, end, boolean)| !**boolean && **end > 0)
        .map(|(current, end, _)| (*end as usize - 1, convert(*current)));
    let values = sweep_reduce(
        max_end,
        A::ZERO,
        events,
        (0..max_end).rev(),
        |left, right| left.wrap_add(right),
    )?;
    Ok((ends_labels(max_end, index), values))
}

pub fn sum_rev_ends_float_core<T, F>(
    arr: numpy::ndarray::ArrayView1<T>,
    ends: numpy::ndarray::ArrayView1<i64>,
    index: numpy::ndarray::ArrayView1<i64>,
    booleans: numpy::ndarray::ArrayView1<bool>,
    mut to_f64: F,
) -> Result<(numpy::ndarray::Array1<i64>, numpy::ndarray::Array1<f64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    ensure_equal_lengths_core(
        arr.len(),
        ends.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    ensure_equal_lengths_core(
        arr.len(),
        booleans.len(),
        "arr, ends, and booleans must have equal lengths",
    )?;
    let max_end = ends_domain(ends, index.len())?;
    // Keep the existing per-position Neumaier accumulation for floats for
    // the same row-order reason as the starts path.
    let mut slots = vec![(0.0_f64, 0.0_f64); max_end];
    for (current, end, boolean) in izip!(arr, ends, booleans) {
        if *boolean {
            continue;
        }
        let current_ = to_f64(*current);
        // Example: `end = 3` selects slots 0, 1, and 2. The row contributes
        // to that prefix and cannot change slot 3 or anything after it.
        for (total, compensation) in slots.iter_mut().take(*end as usize) {
            let difference = current_ - *compensation;
            let increment = *total + difference;
            *compensation = (increment - *total) - difference;
            if !compensation.is_finite() {
                *compensation = 0.;
            }
            *total = increment;
        }
    }
    let result = slots.into_iter().map(|(total, _)| total).collect();
    Ok((ends_labels(max_end, index), result))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            into_starts_ends_result(
                py,
                sum_rev_ends_int_core(arr, ends, index, booleans, |value| value as $acc),
            )
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapAdd`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_sum_rev_end_int64, i64, i64);
compute_ints!(compute_sum_rev_end_int32, i32, i64);
compute_ints!(compute_sum_rev_end_int16, i16, i64);
compute_ints!(compute_sum_rev_end_int8, i8, i64);
compute_ints!(compute_sum_rev_end_uint64, u64, u64);
compute_ints!(compute_sum_rev_end_uint32, u32, i64);
compute_ints!(compute_sum_rev_end_uint16, u16, i64);
compute_ints!(compute_sum_rev_end_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            into_starts_ends_result(
                py,
                sum_rev_ends_float_core(arr, ends, index, booleans, |value| value as f64),
            )
        }
    };
}

compute_floats!(compute_sum_rev_end_f64, f64);
compute_floats!(compute_sum_rev_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn uses_compact_candidate_slots_in_prefix_order() {
        let arr = array![5_i64, 7, 99];
        let ends = array![2_i64, 1, 2];
        let index = array![10_i64, 30, 20];
        let booleans = array![false, true, true];

        let (indexers, result) = sum_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert_eq!(indexers, array![10, 30]);
        assert_eq!(result, array![5, 5]);
    }

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let (indexers, result) = sum_rev_ends_int_core(
            array![value].view(),
            array![1_i64].view(),
            array![20_i64].view(),
            array![false].view(),
            |v: u64| v,
        )
        .unwrap();
        assert_eq!(indexers, array![20]);
        assert_eq!(result, array![value]);
    }

    #[test]
    fn float_core_accepts_all_zero_ends_instead_of_panicking() {
        // Regression test: the pre-refactor `.max().unwrap()` on a
        // zero-width-filtered iterator panicked here because every `end`
        // was 0, leaving nothing to take a max of. `ends_domain` takes the
        // max of the raw `ends` array instead, so an all-zero domain
        // resolves to `0` rather than panicking.
        let arr = array![5.0_f64];
        let ends = array![0_i64];
        let index = array![0_i64];
        let booleans = array![false];

        let result = sum_rev_ends_float_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert!(result.0.is_empty());
        assert!(result.1.is_empty());
    }

    #[test]
    fn accepts_zero_end_as_an_empty_prefix() {
        let arr = array![5_i64];
        let ends = array![0_i64];
        let index = array![0_i64];
        let booleans = array![false];

        let result = sum_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert!(result.0.is_empty());
        assert!(result.1.is_empty());
    }

    #[test]
    fn in_place_sweep_handles_zero_one_and_full_width_ends() {
        let arr = array![5_i64, 7, 11];
        let ends = array![0_i64, 1, 3];
        let index = array![10_i64, 30, 20];
        let booleans = array![false, false, false];

        let (indexers, result) = sum_rev_ends_int_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        // The end=0 row covers nothing; end=1 covers only slot 0; end=3
        // covers all three slots. The labels remain in right-index order.
        assert_eq!(indexers, array![10, 30, 20]);
        assert_eq!(result, array![18, 11, 11]);
    }

    #[test]
    fn float_core_returns_zero_for_all_null_rows() {
        let arr = array![1.0_f64, 2.0];
        let ends = array![2_i64, 1];
        let index = array![20_i64, 10];
        let booleans = array![true, true];

        let (indexers, result) = sum_rev_ends_float_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert_eq!(indexers, array![20, 10]);
        assert_eq!(result, array![0.0, 0.0]);
    }

    #[test]
    fn float_core_preserves_infinity_without_compensation_nan() {
        let arr = array![f64::INFINITY, 1.0];
        let ends = array![1_i64, 1];
        let index = array![20_i64];
        let booleans = array![false, false];

        let (_, result) = sum_rev_ends_float_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert!(result[0].is_infinite() && result[0].is_sign_positive());
    }

    #[test]
    fn float_core_overflow_adjacent_values_follow_float_sum() {
        let arr = array![f64::MAX, f64::MAX];
        let ends = array![1_i64, 2];
        let index = array![20_i64, 10];
        let booleans = array![false, false];

        let (_, result) = sum_rev_ends_float_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        assert_eq!(result[0], f64::INFINITY);
        assert_eq!(result[1], f64::MAX);
    }

    #[test]
    fn float32_core_promotes_values_to_float64_output() {
        let arr = array![1.5_f32, 2.25];
        let ends = array![2_i64, 1];
        let index = array![20_i64, 10];
        let booleans = array![false, false];

        let (indexers, result) = sum_rev_ends_float_core(
            arr.view(),
            ends.view(),
            index.view(),
            booleans.view(),
            |value| value as f64,
        )
        .unwrap();

        assert_eq!(indexers, array![20, 10]);
        assert_eq!(result, array![3.75, 1.5]);
    }
}
