use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `starts[i]`, sum `arr[starts[i]..]` (to the end of the array),
/// skipping any position flagged `true` in `booleans` (a null mask).
///
/// ELI5: `booleans[nn] == true` means "this value is missing, treat it as
/// absent, not as zero" -- we skip it in the running total rather than
/// adding it in.
///
/// Overflow note: `wrapping_add` makes two's-complement wraparound explicit,
/// matching NumPy `i64` arithmetic in debug, test, and release builds.
pub fn sum_start_core(
    arr: ArrayView1<i64>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    sum_start_core_with_cast(arr, starts, booleans, |value| value)
}

/// `u32` benchmark entry point that follows the same cast-on-access path as
/// the corresponding Python wrapper.
///
/// ELI5: this lets the benchmark count the cost of opening only the eight
/// boxes a tiny query asks for, rather than first copying and relabelling
/// every box in a large warehouse.
pub fn sum_start_u32_core(
    arr: ArrayView1<u32>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    sum_start_core_with_cast(arr, starts, booleans, |value| value as i64)
}

fn sum_start_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    starts: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::zeros(starts.len());
    let end_: usize = arr.len();
    for (pos, start) in starts.indexed_iter() {
        let mut total: i64 = 0;
        let start_ = *start as usize;
        for nn in start_..end_ {
            if booleans[nn] {
                continue;
            }
            total = total.wrapping_add(to_i64(arr[nn]));
        }
        result[pos] = total;
    }
    result
}

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            // Cast only values inside the requested ranges. Widening the
            // whole column would make a tiny suffix query scan and copy it.
            //
            // Note (applies to the uint64 instantiation only): `value as
            // i64` is a bit-reinterpretation (two's complement), not a
            // widen, for any u64 value >= 2^63 - e.g. 2^63 becomes
            // i64::MIN. This is pre-existing behavior this PR doesn't
            // change; see `u64_values_above_i63_reinterpret_bits` below,
            // which locks it in and documents it now that it's
            // independently testable without a Python interpreter.
            //
            // ELI5: an i64 and a u64 are the same 64 bits of memory, just
            // read with a different rulebook for what those bits mean.
            // For a u64 value small enough that both rulebooks agree
            // (< 2^63), the cast is a normal widen. Past that point the
            // top bit means "huge positive number" under one rulebook and
            // "negative number" under the other, so re-reading the same
            // bits as i64 can flip the sign instead of preserving the
            // value.
            let result = sum_start_core_with_cast(
                arr.as_array(),
                starts.as_array(),
                booleans.as_array(),
                |value| value as i64,
            );
            result.into_pyarray(py)
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
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(starts.len());
            let end_: usize = arr.len();
            for (pos, start) in starts.indexed_iter() {
                let mut total: f64 = 0.0;
                let mut compensation: f64 = 0.0;
                let start_ = *start as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current: f64 = arr[nn] as f64;
                    let difference = current - compensation;
                    let increment = total + difference;
                    compensation = (increment - total) - difference;
                    total = increment;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_sum_start_int64, i64);
generic_compute!(compute_sum_start_int32, i32);
generic_compute!(compute_sum_start_int16, i16);
generic_compute!(compute_sum_start_int8, i8);
generic_compute!(compute_sum_start_uint64, u64);
generic_compute!(compute_sum_start_uint16, u16);
generic_compute!(compute_sum_start_uint8, u8);
generic_compute_floats!(compute_sum_start_f32, f32);
generic_compute_floats!(compute_sum_start_f64, f64);

// ELI5: this representative wrapper walks through the exact same door as
// the sparse benchmark, so timing that door also protects Python callers
// from an accidental whole-column copy.
#[pyfunction]
pub fn compute_sum_start_uint32<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> Bound<'py, PyArray1<i64>> {
    sum_start_u32_core(arr.as_array(), starts.as_array(), booleans.as_array()).into_pyarray(py)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array() {
        let arr: Array1<i64> = array![];
        let starts = array![0_i64];
        let booleans: Array1<bool> = array![];
        let got = sum_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn start_at_end_is_zero() {
        let arr = array![1_i64, 2, 3];
        let starts = array![3_i64]; // boundary: nothing left to sum
        let booleans = array![false, false, false];
        let got = sum_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn start_at_zero_sums_everything() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64]; // boundary: whole array
        let booleans = array![false, false, false];
        let got = sum_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![6]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![1_i64, 2, 3, 4];
        let starts = array![0_i64];
        // position 1 (value 2) and position 3 (value 4) are "null"
        let booleans = array![false, true, false, true];
        let got = sum_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![1 + 3]);
    }

    #[test]
    fn all_null_range_is_zero() {
        let arr = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let booleans = array![true, true, true];
        let got = sum_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        // 100 copies of i64::MAX / 2 overflows i64 many times over; this
        // must wrap (two's complement), matching the NumPy/pyjanitor side
        // of the boundary, not panic.
        let value = i64::MAX / 2;
        let arr = Array1::<i64>::from_elem(100, value);
        let starts = array![0_i64];
        let booleans = Array1::<bool>::from_elem(100, false);
        let got = sum_start_core(arr.view(), starts.view(), booleans.view());
        assert_eq!(got[0], -100_i64);
    }

    #[test]
    fn u64_values_above_i63_reinterpret_bits() {
        // Documents pre-existing, unchanged behavior of the uint64
        // instantiation's `|value| value as i64` cast (see the note on
        // `generic_compute!` above): for a u64 value >= 2^63, this is a
        // bit-reinterpretation (two's complement), not a widen -
        // matching NumPy's own `.astype(np.int64)` unsafe-cast semantics
        // on the pyjanitor side of this same boundary, not a bug to fix
        // in this crate independently of that Python-side contract.
        let arr = Array1::<u64>::from_elem(1, 2u64.pow(63));
        let starts = array![0_i64];
        let booleans = array![false];
        let got = sum_start_core_with_cast(arr.view(), starts.view(), booleans.view(), |value| {
            value as i64
        });
        assert_eq!(got, array![i64::MIN]);
    }

    #[test]
    fn casts_only_values_in_requested_suffix() {
        let arr = array![1_i32, 2, 3, 4];
        let starts = array![3_i64];
        let booleans = array![false, false, false, false];
        let mut casts = 0;
        let got = sum_start_core_with_cast(arr.view(), starts.view(), booleans.view(), |value| {
            casts += 1;
            value as i64
        });
        assert_eq!(got, array![4]);
        assert_eq!(casts, 1);
    }
}
