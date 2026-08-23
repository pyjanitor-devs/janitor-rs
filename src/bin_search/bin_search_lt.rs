use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Find, for every `left[i]`, the first position in `right[starts[i]..ends[i])`
/// whose value is strictly greater than `left[i]` -- the start of the match
/// region for a `<` join (`right` is assumed sorted ascending within each
/// `[start, end)` slice). Returns `-1` for an invalid/empty range (`start`
/// or `end` is `-1`, or `start >= end`), when no element in the range is
/// greater than `left[i]`, or when the first such element is equal to
/// `left[i]` (defensive: cannot happen when `right` is truly sorted, since
/// the search converges on the first strictly-greater element).
///
/// ELI5: binary search cuts the candidate range in half each step instead of
/// scanning it one item at a time, so it costs O(log width) per query
/// instead of O(width).
pub fn binary_search_lt_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(left.len());
    let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
    for (pos, (left_value, start, end)) in zipped.enumerate() {
        if *start == -1 || *end == -1 || *start >= *end {
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
            if current_value <= *left_value {
                min_idx = mid_idx + 1;
            } else {
                max_idx = mid_idx;
            }
        }
        if min_idx == *end {
            result[pos] = -1;
            continue;
        }
        let current_value = right[min_idx as usize];
        if current_value == *left_value {
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
            let result = binary_search_lt_core(
                left.as_array(),
                right.as_array(),
                starts.as_array(),
                ends.as_array(),
            );
            result.into_pyarray(py)
        }
    };
}

bin_search!(binary_search_lt_int64, i64);
bin_search!(binary_search_lt_int32, i32);
bin_search!(binary_search_lt_int16, i16);
bin_search!(binary_search_lt_int8, i8);
bin_search!(binary_search_lt_uint64, u64);
bin_search!(binary_search_lt_uint32, u32);
bin_search!(binary_search_lt_uint16, u16);
bin_search!(binary_search_lt_uint8, u8);
bin_search!(binary_search_lt_f64, f64);
bin_search!(binary_search_lt_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_lt_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_right_range_returns_minus_one() {
        let left = array![5_i64];
        let right: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn sentinel_starts_and_ends_return_minus_one() {
        let left = array![1_i64, 2, 3];
        let right = array![10_i64, 20, 30];
        let starts = array![-1_i64, 0, 2];
        let ends = array![3_i64, -1, 1]; // last: start(2) >= end(1)
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1, -1, -1]);
    }

    #[test]
    fn finds_first_strictly_greater_element() {
        // right is sorted ascending; left=5 -> first element > 5 is 7 at index 2
        let left = array![5_i64];
        let right = array![1_i64, 3, 7, 9];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![2]);
    }

    #[test]
    fn no_element_greater_than_left_returns_minus_one() {
        let left = array![100_i64];
        let right = array![1_i64, 3, 7, 9];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![-1]);
    }

    #[test]
    fn duplicate_values_in_right_find_first_greater_past_the_run() {
        // duplicates of the target value must be skipped entirely
        let left = array![5_i64];
        let right = array![1_i64, 5, 5, 5, 9];
        let starts = array![0_i64];
        let ends = array![5_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![4]);
    }

    #[test]
    fn boundary_first_and_last_positions() {
        let right = array![10_i64, 20, 30];
        // left value below everything -> first position (0) is the answer
        let left_low = array![0_i64];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let got_low =
            binary_search_lt_core(left_low.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got_low, array![0]);

        // left value equal to the last element -> no strictly-greater element exists
        let left_high = array![30_i64];
        let got_high =
            binary_search_lt_core(left_high.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got_high, array![-1]);
    }

    #[test]
    fn generic_over_float_dtype() {
        let left = array![2.5_f64];
        let right = array![1.0_f64, 2.0, 3.0, 4.0];
        let starts = array![0_i64];
        let ends = array![4_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![2]);
    }

    #[test]
    fn restricted_subrange_only_searches_within_start_end() {
        // value 2 exists in right, but the searched slice [2, 5) excludes it;
        // within that slice the first element > 2 is at index 3
        let left = array![2_i64];
        let right = array![2_i64, 2, 2, 5, 9];
        let starts = array![2_i64];
        let ends = array![5_i64];
        let got = binary_search_lt_core(left.view(), right.view(), starts.view(), ends.view());
        assert_eq!(got, array![3]);
    }
}
