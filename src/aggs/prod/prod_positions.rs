use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_equal_lengths;

use crate::aggs::{checked_index, checked_range};

/// `#[cfg(test)]`-only entry point for direct, Python-free testing of
/// [`prod_positions_core_with_cast`] (see its doc comment for the guard
/// rationale) at the representative `i64` dtype.
#[cfg(test)]
pub(crate) fn prod_positions_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    prod_positions_core_with_cast(arr, starts, ends, positions, booleans, |value| value)
}

/// For every `(starts[i], ends[i])`, multiply `arr[positions[nn]]` over `nn`
/// in `[starts[i], ends[i])`, skipping `nn` where `positions[nn]` is not a
/// valid index into `arr` (including the `-1` "no candidate" sentinel) or
/// where the candidate's own position is null. Returns `1` (the
/// multiplicative identity) when the slot range is empty or invalid, or
/// when every candidate is skipped.
///
/// Caller contract: `starts` and `ends` are parallel arrays with equal
/// lengths. As in the other aggregation kernels, this low-level core does not
/// validate that whole-call invariant inside its hot loop.
///
/// ELI5 (the guard): `checked_range(start, end, positions.len())` rejects a
/// negative or out-of-bounds slot range *before* it's used to index
/// `positions`; a row rejected here (e.g. `end == -1`, this crate's "no
/// match" sentinel, cast to `usize` without a guard) would otherwise wrap
/// to a huge `usize` and walk `positions` out of bounds. The result starts at
/// `1`, so rejecting a bad ticket preserves the same product identity that an
/// empty range returned before this guard was added. See issue #32.
fn prod_positions_core_with_cast<T, F>(
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
    let mut result = Array1::<i64>::from_elem(starts.len(), 1);
    let zipped = starts.into_iter().zip(ends);
    for (pos, (start, end)) in zipped.enumerate() {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let mut total: i64 = 1;
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], arr.len()) else {
                continue;
            };
            if booleans[indexer_] {
                continue;
            }
            total = total.wrapping_mul(to_i64(arr[indexer_]));
        }
        result[pos] = total;
    }
    result
}

#[cfg(test)]
pub(crate) fn prod_positions_float_core(
    arr: ArrayView1<f64>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    positions: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<f64> {
    prod_positions_float_core_with_cast(arr, starts, ends, positions, booleans, |value| value)
}

fn prod_positions_float_core_with_cast<T, F>(
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
    let mut result = Array1::<f64>::from_elem(starts.len(), 1.0);
    for (pos, (start, end)) in starts.into_iter().zip(ends).enumerate() {
        // ELI5: validate the shared slot range before either float dtype
        // touches `positions`; a rejected ticket keeps product's identity.
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let mut total = 1.0;
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], arr.len()) else {
                continue;
            };
            if booleans[indexer_] {
                continue;
            }
            total *= to_f64(arr[indexer_]);
        }
        result[pos] = total;
    }
    result
}

// ELI5: `$type` below only picks the dtype of the *input* array (`arr`) --
// the accumulator and result are always `i64`, hardcoded in the macro body
// a few lines down, regardless of `$type`. That widening is intentional
// (it's how `prod` avoids overflowing a narrow dtype), but `$type` must
// still match the numpy dtype the function's name promises, or pyo3
// rejects the array at the Python boundary with a dtype mismatch instead
// of computing anything. `compute_prod_positions_int8` once had `i64` here
// (a leftover copy-paste from a wider sibling) even though its name
// promises `i8` input -- see issue #30.
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
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let result = prod_positions_core_with_cast(
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

generic_compute_ints!(compute_prod_positions_int64, i64);
generic_compute_ints!(compute_prod_positions_int32, i32);
generic_compute_ints!(compute_prod_positions_int16, i16);
generic_compute_ints!(compute_prod_positions_int8, i8); // fixed: was `i64`, see issue #30
generic_compute_ints!(compute_prod_positions_uint64, u64);
generic_compute_ints!(compute_prod_positions_uint32, u32);
generic_compute_ints!(compute_prod_positions_uint16, u16);
generic_compute_ints!(compute_prod_positions_uint8, u8);

/// kahan summation
// ELI5: same story as `generic_compute_ints!` above -- `$type` only picks
// the *input* array's dtype; the accumulator/result are always `f64`,
// hardcoded a few lines down. `compute_prod_positions_f32` once had `f64`
// here instead of `f32`, so a real `float32` numpy array failed the pyo3
// dtype check even though the function's name promises it accepts one --
// see issue #30.
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
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let result = prod_positions_float_core_with_cast(
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

generic_compute_floats!(compute_prod_positions_f32, f32); // fixed: was `f64`, see issue #30
generic_compute_floats!(compute_prod_positions_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggs::PositionsFn;
    use numpy::ndarray::array;

    type Int8PositionsFn = PositionsFn<i8, i64>;
    type F32PositionsFn = PositionsFn<f32, f64>;

    #[test]
    fn int8_wrapper_accepts_an_int8_array() {
        // ELI5: the typed slot only accepts a wrapper whose `arr` is really
        // `i8`; changing the macro argument back to `i64` breaks compilation.
        let _wrapper: Int8PositionsFn = compute_prod_positions_int8;
    }

    #[test]
    fn f32_wrapper_accepts_an_f32_array() {
        // ELI5: the typed slot only accepts a wrapper whose `arr` is really
        // `f32`; changing the macro argument back to `f64` breaks compilation.
        let _wrapper: F32PositionsFn = compute_prod_positions_f32;
    }

    #[test]
    fn end_sentinel_returns_one_not_a_panic() {
        // The exact reproduction from issue #32: `end == -1` used to cast
        // to `usize::MAX` and walk `positions` out of bounds. A rejected
        // row keeps the product identity instead of changing the established
        // empty-range result while fixing the panic.
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![0_i64];
        let ends = array![-1_i64];
        let positions = array![0_i64, 1, 2, 3, 4];
        let booleans = array![false, false, false, false, false];
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn start_sentinel_returns_one_not_a_panic() {
        let arr = array![2_i64, 3, 4, 5, 6];
        let starts = array![-1_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, 1, 2, 3, 4];
        let booleans = array![false, false, false, false, false];
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn multiplies_via_indirection() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, 1, 2];
        let booleans = array![false, false, false];
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![24]);
    }

    #[test]
    fn skips_negative_one_position_sentinel_and_null_mask() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let positions = array![0_i64, -1, 2]; // slot 1 has no candidate
        let booleans = array![false, false, true]; // arr[2] is null
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![2]);
    }

    #[test]
    fn empty_range_returns_multiplicative_identity() {
        let arr = array![2_i64, 3];
        let starts = array![1_i64];
        let ends = array![1_i64];
        let positions = array![0_i64, 1];
        let booleans = array![false, false];
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let arr = array![i64::MAX, 2];
        let starts = array![0_i64];
        let ends = array![2_i64];
        let positions = array![0_i64, 1];
        let booleans = array![false, false];
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![-2]);
    }

    #[test]
    fn float_end_sentinel_returns_one_not_a_panic() {
        let arr = array![2.0_f64, 3.0];
        let starts = array![0_i64];
        let ends = array![-1_i64];
        let positions = array![0_i64, 1];
        let booleans = array![false, false];
        let got = prod_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1.0]);
    }

    #[test]
    fn integer_range_and_candidate_boundaries_are_safe() {
        let arr = array![2_i64, 3, 4];
        let booleans = array![false, false, false];
        let positions = array![0_i64, 1, 2];
        let starts = array![-2_i64, -1, 0, 0, 2, 3, 4, 2];
        let ends = array![1_i64, 1, -2, -1, 2, 3, 4, 4];
        let got = prod_positions_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1, 1, 1, 1, 1, 1, 1, 1]);

        let candidate_positions = array![-2_i64, -1, 0, 2, 3, 4];
        let candidate_starts = array![0_i64];
        let candidate_ends = array![candidate_positions.len() as i64];
        let got = prod_positions_core(
            arr.view(),
            candidate_starts.view(),
            candidate_ends.view(),
            candidate_positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![8]);
    }

    #[test]
    fn float_range_and_candidate_boundaries_are_safe() {
        let arr = array![2.0_f64, 3.0, 4.0];
        let booleans = array![false, false, false];
        let positions = array![0_i64, 1, 2];
        let starts = array![-2_i64, -1, 0, 0, 2, 3, 4, 2];
        let ends = array![1_i64, 1, -2, -1, 2, 3, 4, 4];
        let got = prod_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let candidate_positions = array![-2_i64, -1, 0, 2, 3, 4];
        let candidate_starts = array![0_i64];
        let candidate_ends = array![candidate_positions.len() as i64];
        let got = prod_positions_float_core(
            arr.view(),
            candidate_starts.view(),
            candidate_ends.view(),
            candidate_positions.view(),
            booleans.view(),
        );
        assert_eq!(got, array![8.0]);
    }
}
