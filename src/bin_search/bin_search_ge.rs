use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// A row with a malformed range (`start` negative, `end` the `-1` sentinel,
/// `start >= end`, or `end` beyond `right.len()`) resolves to `-1`, matching
/// a genuinely empty range, instead of casting a negative bound to `usize`
/// and indexing `right` out of bounds.
///
/// ELI5 (why `start < 0`, not just `start == -1`): a `usize` can't represent
/// a negative number, so casting one doesn't keep it negative -- it wraps
/// around to a huge positive `usize` instead. Checking only for the `-1`
/// sentinel would miss e.g. `start=-3, end=-2`: both wrap to huge-but-still
/// -ordered `usize` values (so `start_ < end_` survives the cast), and the
/// loop walks `right` far out of bounds instead of recognizing the row as
/// invalid. Requiring `start >= 0` up front closes that gap.
pub fn binary_search_ge_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(left.len());
    let right_len = right.len();
    let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
    for (pos, (left_value, start, end)) in zipped.enumerate() {
        if *start < 0 || *end == -1 || *start >= *end || *end as usize > right_len {
            result[pos] = -1;
            continue;
        }
        let mut min_idx = *start;
        let mut max_idx = *end;
        while min_idx < max_idx {
            // to avoid overflow
            // adapted from numba's implementation
            let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            let current_value = right[mid_idx as usize];
            if current_value > *left_value {
                max_idx = mid_idx;
            } else {
                min_idx = mid_idx + 1;
            }
        }
        if min_idx == *start {
            result[pos] = -1;
            continue;
        }
        result[pos] = min_idx;
    }
    result
}

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
        ) -> Bound<'py, PyArray1<i64>> {
            let result = binary_search_ge_core(
                left.as_array(),
                right.as_array(),
                starts.as_array(),
                ends.as_array(),
            );
            result.into_pyarray(py)
        }
    };
}

bin_search!(binary_search_ge_int64, i64);
bin_search!(binary_search_ge_int32, i32);
bin_search!(binary_search_ge_int16, i16);
bin_search!(binary_search_ge_int8, i8);
bin_search!(binary_search_ge_uint64, u64);
bin_search!(binary_search_ge_uint32, u32);
bin_search!(binary_search_ge_uint16, u16);
bin_search!(binary_search_ge_uint8, u8);
bin_search!(binary_search_ge_f64, f64);
bin_search!(binary_search_ge_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_ge_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn both_bounds_negative_but_start_less_than_end_returns_minus_one_not_a_panic() {
        // start=-3, end=-2 aren't the `-1` sentinel, and -3 < -2 holds in
        // i64 space, so a `start == -1`-only check would let this row
        // through, cast both bounds to huge-but-still-ordered `usize`
        // values, and index `right` far out of bounds.
        let left = array![5_i64];
        let right = array![1_i64];
        let starts = array![-3_i64];
        let ends = array![-2_i64];
        let got = binary_search_ge_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn non_sentinel_negative_start_returns_minus_one() {
        let left = array![3_i64];
        let right = array![1_i64, 2, 3];
        let starts = array![-2_i64];
        let ends = array![2_i64];
        let got = binary_search_ge_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn end_beyond_right_len_returns_minus_one_not_a_panic() {
        let left = array![1_i64];
        let right = array![1_i64];
        let starts = array![0_i64];
        let ends = array![2_i64]; // right.len() + 1
        let got = binary_search_ge_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn end_equal_to_right_len_is_a_valid_inclusive_bound() {
        // every element in [0, right.len()) is <= left_value, so the
        // "first position > left_value" answer is right.len() itself --
        // this must not be confused with the out-of-bounds rejection above.
        let left = array![5_i64];
        let right = array![1_i64, 2, 3];
        let starts = array![0_i64];
        let ends = array![3_i64]; // == right.len()
        let got = binary_search_ge_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![3]);
    }

    #[test]
    fn sentinel_starts_and_ends_return_minus_one() {
        let left = array![1_i64, 2, 3];
        let right = array![10_i64, 20, 30];
        let starts = array![-1_i64, 0, 2];
        let ends = array![3_i64, -1, 1]; // last: start(2) >= end(1)
        let got = binary_search_ge_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1, -1, -1]);
    }

    #[test]
    fn restricted_subrange_only_searches_within_start_end() {
        let left = array![6_i64];
        let right = array![9_i64, 9, 2, 5, 9];
        let starts = array![2_i64];
        let ends = array![5_i64];
        let got = binary_search_ge_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![4]);
    }
}
