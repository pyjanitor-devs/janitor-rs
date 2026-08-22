use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn is_empty_sentinel_end(end: i64) -> bool {
    end == -1
}

/// For every `ends[i]`, sum `arr[..ends[i]]` (from the start of the array),
/// skipping any position flagged `true` in `booleans` (a null mask). An
/// `end` of `-1` (the crate's sentinel for "invalid/no match", e.g. as
/// returned by `binary_search_lt_core`) contributes `0`, matching that
/// sentinel's meaning elsewhere rather than being cast to `usize` and
/// walked off the end of `arr`.
///
/// ELI5: the mirror image of `sum_start_core` -- instead of "everything
/// from here to the end", it's "everything from the beginning up to here".
/// Same null-skip and overflow-wrap contract; see `sum_start_core` for the
/// full explanation.
///
/// ELI5 (the `-1` guard): `-1 as usize` isn't "negative one" any more, a
/// `usize` can't hold that -- it wraps around to the *largest* possible
/// number instead (`usize::MAX`). So without checking `-1` first, "no
/// match" would silently turn into "sum almost the whole address space",
/// which reads past the array and crashes rather than just giving `0`.
pub fn sum_end_core(
    arr: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
) -> Array1<i64> {
    sum_end_core_with_cast(arr, ends, booleans, |value| value)
}

fn sum_end_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_i64: F,
) -> Array1<i64>
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut result = Array1::<i64>::zeros(ends.len());
    let start_: usize = 0;
    for (pos, end) in ends.indexed_iter() {
        if is_empty_sentinel_end(*end) {
            continue; // result[pos] is already 0
        }
        let mut total: i64 = 0;
        let end_ = *end as usize;
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

fn sum_end_float_core_with_cast<T, F>(
    arr: ArrayView1<T>,
    ends: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    mut to_f64: F,
) -> Array1<f64>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut result = Array1::<f64>::zeros(ends.len());
    for (pos, end) in ends.indexed_iter() {
        // ELI5: integers and floats receive the same list of slice ends.
        // Check the "no match" card before either path turns it into an
        // array position, so dtype cannot decide whether it returns 0 or
        // crashes.
        if is_empty_sentinel_end(*end) {
            continue;
        }
        let mut total = 0.0;
        let end_ = *end as usize;
        for nn in 0..end_ {
            if booleans[nn] {
                continue;
            }
            total += to_f64(arr[nn]);
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
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let result = sum_end_core_with_cast(
                arr.as_array(),
                ends.as_array(),
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
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let result = sum_end_float_core_with_cast(
                arr.as_array(),
                ends.as_array(),
                booleans.as_array(),
                |value| value as f64,
            );
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_sum_end_int64, i64);
generic_compute!(compute_sum_end_int32, i32);
generic_compute!(compute_sum_end_int16, i16);
generic_compute!(compute_sum_end_int8, i8);
generic_compute!(compute_sum_end_uint64, u64);
generic_compute!(compute_sum_end_uint32, u32);
generic_compute!(compute_sum_end_uint16, u16);
generic_compute!(compute_sum_end_uint8, u8);
generic_compute_floats!(compute_sum_end_f32, f32);
generic_compute_floats!(compute_sum_end_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_array() {
        let arr: Array1<i64> = array![];
        let ends = array![0_i64];
        let booleans: Array1<bool> = array![];
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn end_at_zero_is_empty() {
        let arr = array![1_i64, 2, 3];
        let ends = array![0_i64]; // boundary: nothing yet
        let booleans = array![false, false, false];
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn end_at_len_sums_everything() {
        let arr = array![1_i64, 2, 3];
        let ends = array![3_i64]; // boundary: whole array
        let booleans = array![false, false, false];
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![6]);
    }

    #[test]
    fn null_mask_skips_flagged_positions() {
        let arr = array![1_i64, 2, 3, 4];
        let ends = array![4_i64];
        let booleans = array![false, true, false, true];
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![1 + 3]);
    }

    #[test]
    fn all_null_range_is_zero() {
        let arr = array![1_i64, 2, 3];
        let ends = array![3_i64];
        let booleans = array![true, true, true];
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn sentinel_end_is_zero_not_a_panic() {
        // -1 is the crate's "invalid/no match" sentinel (see
        // binary_search_lt_core). Cast naively to usize it becomes
        // usize::MAX, and the loop would walk straight off the end of
        // `arr`; this must return 0 instead of panicking.
        let arr = array![1_i64, 2, 3];
        let ends = array![-1_i64];
        let booleans = array![false, false, false];
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got, array![0]);
    }

    #[test]
    fn float_sentinel_end_is_zero_not_a_panic() {
        let arr = array![1.0_f64, 2.0, 3.0];
        let ends = array![-1_i64];
        let booleans = array![false, false, false];
        let got =
            sum_end_float_core_with_cast(arr.view(), ends.view(), booleans.view(), |value| value);
        assert_eq!(got, array![0.0]);
    }

    #[test]
    fn accumulation_overflow_wraps_instead_of_panicking() {
        let value = i64::MAX / 2;
        let arr = Array1::<i64>::from_elem(100, value);
        let ends = array![100_i64];
        let booleans = Array1::<bool>::from_elem(100, false);
        let got = sum_end_core(arr.view(), ends.view(), booleans.view());
        assert_eq!(got[0], -100_i64);
    }

    #[test]
    fn casts_only_values_in_requested_prefix() {
        let arr = array![1_i32, 2, 3, 4];
        let ends = array![1_i64];
        let booleans = array![false, false, false, false];
        let mut casts = 0;
        let got = sum_end_core_with_cast(arr.view(), ends.view(), booleans.view(), |value| {
            casts += 1;
            value as i64
        });
        assert_eq!(got, array![1]);
        assert_eq!(casts, 1);
    }
}
