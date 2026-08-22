use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, checked_range};

/// For every `(starts[i], ends[i])`, multiply `arr[positions[nn]]` over `nn`
/// in `[starts[i], ends[i])`, skipping `nn` where `positions[nn]` is not a
/// valid index into `arr` (including the `-1` "no candidate" sentinel) or
/// where the candidate's own position is null. Returns `1` (the
/// multiplicative identity) when the slot range is valid but every
/// candidate is skipped, or `0` (the result array's zero-initialized
/// default) when the slot range itself is rejected outright.
///
/// ELI5 (the guard): `checked_range(start, end, positions.len())` rejects a
/// negative or out-of-bounds slot range *before* it's used to index
/// `positions`; a row rejected here (e.g. `end == -1`, this crate's "no
/// match" sentinel, cast to `usize` without a guard) would otherwise wrap
/// to a huge `usize` and walk `positions` out of bounds. A rejected row's
/// `continue` skips the `result[pos] = total` write entirely, so it's left
/// at `0`, not `1` -- the two "nothing happened" cases are distinguishable
/// in principle, but this function (like existing sibling kernels, e.g.
/// `min_start_match_core`) doesn't currently distinguish them in its
/// return value. See issue #32.
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
    let mut result = Array1::<i64>::zeros(starts.len());
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
            total *= to_i64(arr[indexer_]);
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
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = prod_positions_core_with_cast(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                positions.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            result.into_pyarray(py)
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
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(starts.len());
            let zipped = starts.into_iter().zip(ends);
            for (pos, (start, end)) in zipped.enumerate() {
                // ELI5 (the guard): same reasoning as the int path's
                // `checked_range` call above -- a row with an invalid or
                // sentinel-cast slot range must be rejected before it's
                // used to index `positions`. See issue #32.
                let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                    continue;
                };
                let mut total: f64 = 1.;
                for nn in start_..end_ {
                    let Some(indexer_) = checked_index(positions[nn], arr.len()) else {
                        continue;
                    };
                    if booleans[indexer_] {
                        continue;
                    }
                    let current: f64 = arr[indexer_] as f64;
                    total *= current;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
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
    fn end_sentinel_returns_zero_not_a_panic() {
        // The exact reproduction from issue #32: `end == -1` used to cast
        // to `usize::MAX` and walk `positions` out of bounds. A rejected
        // row's `result[pos] = total` write is skipped entirely, leaving
        // the zero-initialized array default (not `1`, the multiplicative
        // identity a valid-but-empty range would produce).
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
        assert_eq!(got, array![0]);
    }

    #[test]
    fn start_sentinel_returns_zero_not_a_panic() {
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
        assert_eq!(got, array![0]);
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
}
