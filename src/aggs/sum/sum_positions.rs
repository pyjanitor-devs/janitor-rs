use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

use crate::aggs::{checked_index, checked_range};

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`sum_positions_core_with_cast`] (see its doc comment for the guard
/// rationale) at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn sum_positions_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    sum_positions_core_with_cast(arr, starts, ends, positions, booleans, |value| value)
}

/// For every `(starts[i], ends[i])`, sum `arr[positions[nn]]` over `nn` in
/// `[starts[i], ends[i])`, skipping `nn` where `positions[nn]` is not a
/// valid index into `arr` (including the `-1` "no candidate" sentinel) or
/// where the candidate's own position is null. Returns `0` (the additive
/// identity) when the slot range is invalid or every candidate is skipped.
///
/// Caller contract: `starts` and `ends` are parallel arrays with equal
/// lengths. As in the other aggregation kernels, this low-level core does not
/// validate that whole-call invariant inside its hot loop.
///
/// ELI5 (the guard): `checked_range(start, end, positions.len())` rejects a
/// negative or out-of-bounds slot range *before* it's used to index
/// `positions`; a row rejected here (e.g. `end == -1`, this crate's "no
/// match" sentinel, cast to `usize` without a guard) would otherwise wrap
/// to a huge `usize` and walk `positions` out of bounds. See issue #32.
fn sum_positions_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::zeros(starts.len());
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let mut total: i64 = 0;
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], arr.len()) else {
                continue;
            };
            if booleans[indexer_] {
                continue;
            }
            total = total.wrapping_add(to_i64(arr[indexer_]));
        }
        result[pos] = total;
    }
    result
}

#[cfg(test)]
pub(crate) fn sum_positions_float_core(
    arr: ArrayView1<f64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<f64> {
    sum_positions_float_core_with_cast(arr, starts, ends, positions, booleans, |value| value)
}

fn sum_positions_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::zeros(starts.len());
    for (pos, (start, end)) in starts.into_iter().zip(ends).enumerate() {
        // ELI5: validate the shared slot range before either float dtype
        // touches `positions`; `-1` means "no ticket", not a huge index.
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let mut total = 0.0;
        let mut compensation = 0.0;
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], arr.len()) else {
                continue;
            };
            if booleans[indexer_] {
                continue;
            }
            let current = to_f64(arr[indexer_]);
            let difference = current - compensation;
            let increment = total + difference;
            compensation = (increment - total) - difference;
            total = increment;
        }
        result[pos] = total;
    }
    result
}

// ELI5: `$type` below only picks the dtype of the *input* array (`arr`) --
// the accumulator and result are always `i64`/`f64`, hardcoded in the macro
// body a few lines down, regardless of `$type`. That widening is
// intentional (it's how `sum`/`prod` avoid overflowing a narrow dtype), but
// it means `$type` must still match the numpy dtype the function's name
// promises, or pyo3 rejects the array at the Python boundary with a dtype
// mismatch instead of computing anything. `compute_sum_positions_int8` once
// had `i64` here (a leftover copy-paste from a wider sibling) even though
// its name promises `i8` input -- see issue #30.
macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<i64>>>
        // The macro will expand into the contents of this block.
        {
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            let result = sum_positions_core_with_cast(
                arr.as_array(),
                starts,
                ends,
                positions.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>>
        // The macro will expand into the contents of this block.
        {
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            let result = sum_positions_float_core_with_cast(
                arr.as_array(),
                starts,
                ends,
                positions.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            Ok(result.into_pyarray(py))
        }
    };
}

generic_compute_ints!(compute_sum_positions_int64, i64);
generic_compute_ints!(compute_sum_positions_int32, i32);
generic_compute_ints!(compute_sum_positions_int16, i16);
generic_compute_ints!(compute_sum_positions_int8, i8); // fixed: was `i64`, see issue #30
generic_compute_ints!(compute_sum_positions_uint64, u64);
generic_compute_ints!(compute_sum_positions_uint32, u32);
generic_compute_ints!(compute_sum_positions_uint16, u16);
generic_compute_ints!(compute_sum_positions_uint8, u8);
generic_compute_floats!(compute_sum_positions_f32, f32);
generic_compute_floats!(compute_sum_positions_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggs::PositionsFn;
    use numpy::ndarray::array;

    type Int8PositionsFn = PositionsFn<i8, i64>;

    #[test]
    fn int8_wrapper_accepts_an_int8_array() {
        // ELI5: the typed slot only accepts a wrapper whose `arr` is really
        // `i8`; changing the macro argument back to `i64` breaks compilation.
        let _wrapper: Int8PositionsFn = compute_sum_positions_int8;
    }

    #[test]
    fn end_sentinel_returns_zero_not_a_panic() {
        // The exact reproduction from issue #32: `end == -1` used to cast
        // to `usize::MAX` and walk `positions` out of bounds.
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![0_i64];
        let ends = array![-1_i64];
        let positions = array![0_i64, 1, 2, 3, 4];
        let booleans = array![false, false, false, false, false];
        let got = sum_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0]);
    }

    #[test]
    fn start_sentinel_returns_zero_not_a_panic() {
        let arr = array![1_i64, 2, 3, 4, 5];
        let starts = array![-1_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, 1, 2, 3, 4];
        let booleans = array![false, false, false, false, false];
        let got = sum_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0]);
    }

    #[test]
    fn sums_via_indirection() {
        let arr = array![10_i64, 20, 30];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, 1, 2];
        let booleans = array![false, false, false];
        let got = sum_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![60]);
    }

    #[test]
    fn skips_negative_one_position_sentinel_and_null_mask() {
        let arr = array![10_i64, 20, 30];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, -1, 2]; // slot 1 has no candidate
        let booleans = array![false, false, true]; // arr[2] is null
        let got = sum_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![10]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 1];
        let starts = array![0_i64];
        let ends = array![2_i64];
        let positions = array![0_i64, 1];
        let booleans = array![false, false];
        let got = sum_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![i64::MIN]);
    }

    #[test]
    fn float_end_sentinel_returns_zero_not_a_panic() {
        let arr = array![1.0_f64, 2.0];
        let starts = array![0_i64];
        let ends = array![-1_i64];
        let positions = array![0_i64, 1];
        let booleans = array![false, false];
        let got = sum_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0.0]);
    }

    #[test]
    fn integer_range_and_candidate_boundaries_are_safe() {
        let arr = array![2_i64, 3, 4];
        let booleans = array![false, false, false];
        let positions = array![0_i64, 1, 2];
        let starts = array![-2_i64, -1, 0, 0, 2, 3, 4, 2];
        let ends = array![1_i64, 1, -2, -1, 2, 3, 4, 4];
        let got = sum_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0, 0, 0, 0, 0, 0, 0, 0]);

        let candidate_positions = array![-2_i64, -1, 0, 2, 3, 4];
        let candidate_starts = array![0_i64];
        let candidate_ends = array![candidate_positions.len() as i64];
        let got = sum_positions_core(
            arr.view(),
            candidate_starts.view(),
            candidate_ends.view(),
            candidate_positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![6]);
    }

    #[test]
    fn float_range_and_candidate_boundaries_are_safe() {
        let arr = array![2.0_f64, 3.0, 4.0];
        let booleans = array![false, false, false];
        let positions = array![0_i64, 1, 2];
        let starts = array![-2_i64, -1, 0, 0, 2, 3, 4, 2];
        let ends = array![1_i64, 1, -2, -1, 2, 3, 4, 4];
        let got = sum_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let candidate_positions = array![-2_i64, -1, 0, 2, 3, 4];
        let candidate_starts = array![0_i64];
        let candidate_ends = array![candidate_positions.len() as i64];
        let got = sum_positions_float_core(
            arr.view(),
            candidate_starts.view(),
            candidate_ends.view(),
            candidate_positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![6.0]);
    }
}
