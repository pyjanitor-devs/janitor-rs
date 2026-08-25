use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

/// For every left row `i`, add `arr[i]` into every right-row position in
/// `index[starts[i]..]` (open-ended to the end of `index` -- that's what
/// makes this the "starts" shape). Multiple left rows commonly land on the
/// same right position, so positions accumulate rather than overwrite.
///
/// ELI5: `index` is always a permutation of `0..index.len()` (pyjanitor
/// resets both frames to a plain positional `RangeIndex` before any
/// matching runs, and this shape never drops right rows, only reorders
/// them by the join column's value) -- so `pos = index[item]` is always a
/// valid position in a `Vec` sized `index.len()`. That bound comes from
/// `index` itself, not from the `length` pyjanitor passes in: `length` is
/// `ends - starts.min()`, an exact *count* of how many positions end up
/// touched, but not a bound on their *values* -- a left row whose own
/// `start` is greater than 0 can still touch a position numerically past
/// `length` (see the crate's regression test for a concrete case). Trusting
/// a passed-in count as an index bound is exactly what issue #69 got
/// burned by elsewhere; here we sidestep it by deriving the bound from the
/// array we're actually indexing into, not from a Python-supplied number.
pub fn sum_rev_starts_int_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> (Array1<i64>, Array1<i64>)
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let end_ = index.len();
    let mut values = vec![0_i64; end_];
    let mut seen = vec![false; end_];
    let zipped = izip!(arr.into_iter(), starts.into_iter(), booleans.into_iter());
    for (current, start, boolean) in zipped {
        let start_ = *start as usize;
        let current_ = to_i64(*current);
        for item in start_..end_ {
            let pos = index[item] as usize;
            // Touch the slot before checking the null mask: a null left
            // value still means "this right position was matched," it
            // just doesn't contribute to the running total -- matching
            // pandas' `sum(skipna=True)`, which drops the value, not the
            // group.
            seen[pos] = true;
            if *boolean {
                continue;
            }
            values[pos] += current_;
        }
    }
    let mut indexers = Vec::new();
    let mut result = Vec::new();
    for pos in 0..end_ {
        if seen[pos] {
            indexers.push(pos as i64);
            result.push(values[pos]);
        }
    }
    (Array1::from_vec(indexers), Array1::from_vec(result))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
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
            // `length` (== `ends - starts.min()`) is only a row-count the
            // caller already knows; the kernel derives its own safe bound
            // from `index.len()` instead of trusting this number -- see
            // the doc comment on `sum_rev_starts_int_core`.
            let _ = length;
            let (indexers, result) = sum_rev_starts_int_core(
                arr.as_array(),
                starts.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_sum_rev_start_int64, i64);
compute_ints!(compute_sum_rev_start_int32, i32);
compute_ints!(compute_sum_rev_start_int16, i16);
compute_ints!(compute_sum_rev_start_int8, i8);
compute_ints!(compute_sum_rev_start_uint64, u64);
compute_ints!(compute_sum_rev_start_uint32, u32);
compute_ints!(compute_sum_rev_start_uint16, u16);
compute_ints!(compute_sum_rev_start_uint8, u8);

/// Pure-Rust reverse-sum core for the float path: same touch/accumulate
/// structure as `sum_rev_starts_int_core`, but each position's slot
/// carries a Neumaier-compensated `(total, compensation)` pair in a single
/// `Vec`, instead of the old code's two independently-keyed `HashMap`s
/// (issue #48: nothing enforced those two maps staying in sync -- a single
/// paired slot can't desync).
pub fn sum_rev_starts_float_core<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> (Array1<i64>, Array1<f64>)
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let end_ = index.len();
    let mut slots = vec![(0.0_f64, 0.0_f64); end_];
    let mut seen = vec![false; end_];
    let zipped = izip!(arr.into_iter(), starts.into_iter(), booleans.into_iter());
    for (current, start, boolean) in zipped {
        let start_ = *start as usize;
        let current_ = to_f64(*current);
        for item in start_..end_ {
            let pos = index[item] as usize;
            seen[pos] = true;
            if *boolean {
                continue;
            }
            let (total, compensation) = &mut slots[pos];
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
    let mut indexers = Vec::new();
    let mut result = Vec::new();
    for pos in 0..end_ {
        if seen[pos] {
            indexers.push(pos as i64);
            result.push(slots[pos].0);
        }
    }
    (Array1::from_vec(indexers), Array1::from_vec(result))
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
            length: i64,
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
            let _ = length;
            let (indexers, result) = sum_rev_starts_float_core(
                arr.as_array(),
                starts.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
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
    fn overlapping_left_rows_accumulate_and_emit_ascending() {
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
        );
        // row0 (start=1) touches index[1..3] = {0, 1}; row1 (start=0)
        // touches index[0..3] = {2, 0, 1}.
        assert_eq!(indexers, array![0, 1, 2]);
        assert_eq!(result, array![11, 11, 6]);
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
        );
        assert_eq!(indexers, array![6, 7]);
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
        );
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
        );
        assert_eq!(indexers, array![0]);
        assert!((result[0] - 0.6).abs() < 1e-9);
    }
}
