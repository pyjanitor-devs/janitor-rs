use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `left[i]`, find the first position in all of `right` whose
/// value is strictly greater than `left[i]` (`right` is assumed sorted
/// ascending) -- the one-pass sibling of `binary_search_lt_core`, which
/// searches a per-row `[start, end)` slice instead of the whole array.
/// Rows with no such position (or, defensively, one that happens to equal
/// `left[i]` -- cannot happen when `right` is truly sorted) are dropped
/// entirely rather than represented with a sentinel; only the surviving
/// `(search index, left_index[i])` pairs are returned, in row order.
///
/// ELI5: the old shape searched every row into a full `left.len()`-sized
/// array using an internal "no match" marker (a value no real match could
/// ever produce), then made a second pass over that array to copy out only
/// the real matches. Pushing straight into two `Vec`s sized to the only
/// upper bound that exists -- at most one match per row -- does the same
/// work in one pass, with no throwaway marker and no second scan.
pub fn binary_search_lt_first_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    left_index: ArrayView1<i64>,
) -> (Vec<i64>, Vec<i64>) {
    let len_right = right.len();
    let mut search_indices = Vec::with_capacity(left.len());
    let mut index_left = Vec::with_capacity(left.len());
    for (pos, left_value) in left.into_iter().enumerate() {
        let mut min_idx = 0;
        let mut max_idx = len_right;
        while min_idx < max_idx {
            // to avoid overflow
            // adapted from numba's implementation
            let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            let current_value = right[mid_idx];
            if current_value <= *left_value {
                min_idx = mid_idx + 1;
            } else {
                max_idx = mid_idx;
            }
        }
        if min_idx == len_right {
            continue;
        }
        let current_value = right[min_idx];
        if current_value == *left_value {
            continue;
        }
        search_indices.push(min_idx as i64);
        index_left.push(left_index[pos]);
    }
    (search_indices, index_left)
}

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
            let (search_indices, index_left) = binary_search_lt_first_core(
                left.as_array(),
                right.as_array(),
                left_index.as_array(),
            );
            let total = search_indices.len() as i64;
            (
                Array1::from_vec(search_indices).into_pyarray(py),
                Array1::from_vec(index_left).into_pyarray(py),
                total,
            )
        }
    };
}

bin_search!(binary_search_lt_first_int64, i64);
bin_search!(binary_search_lt_first_int32, i32);
bin_search!(binary_search_lt_first_int16, i16);
bin_search!(binary_search_lt_first_int8, i8);
bin_search!(binary_search_lt_first_uint64, u64);
bin_search!(binary_search_lt_first_uint32, u32);
bin_search!(binary_search_lt_first_uint16, u16);
bin_search!(binary_search_lt_first_uint8, u8);
bin_search!(binary_search_lt_first_f64, f64);
bin_search!(binary_search_lt_first_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_lt_first_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_lt_first_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn empty_input_returns_nothing() {
        let left: Array1<i64> = array![];
        let right: Array1<i64> = array![];
        let left_index: Array1<i64> = array![];
        let (indices, index_left) =
            binary_search_lt_first_core(left.view(), right.view(), left_index.view());
        assert!(indices.is_empty());
        assert!(index_left.is_empty());
    }

    #[test]
    fn no_element_greater_than_left_is_dropped_not_sentineled() {
        let left = array![100_i64];
        let right = array![1_i64, 3, 7, 9];
        let left_index = array![42_i64];
        let (indices, index_left) =
            binary_search_lt_first_core(left.view(), right.view(), left_index.view());
        assert!(indices.is_empty());
        assert!(index_left.is_empty());
    }

    #[test]
    fn mixed_rows_keep_only_matches_with_their_own_left_index() {
        let right = array![1_i64, 3, 7, 9];
        // row 0 (left=5, index=10): first > 5 is 7 at position 2 -> kept
        // row 1 (left=100, index=20): nothing greater -> dropped
        // row 2 (left=0, index=30): first > 0 is 1 at position 0 -> kept
        let left = array![5_i64, 100, 0];
        let left_index = array![10_i64, 20, 30];
        let (indices, index_left) =
            binary_search_lt_first_core(left.view(), right.view(), left_index.view());
        assert_eq!(indices, vec![2, 0]);
        assert_eq!(index_left, vec![10, 30]);
    }

    #[test]
    fn duplicate_values_in_right_skip_the_whole_run() {
        let left = array![5_i64];
        let right = array![1_i64, 5, 5, 5, 9];
        let left_index = array![7_i64];
        let (indices, index_left) =
            binary_search_lt_first_core(left.view(), right.view(), left_index.view());
        assert_eq!(indices, vec![4]);
        assert_eq!(index_left, vec![7]);
    }

    #[test]
    fn boundary_first_and_last_positions() {
        let right = array![10_i64, 20, 30];
        // left value below everything -> first position (0) matches
        let left_low = array![0_i64];
        let left_index = array![1_i64];
        let (indices, index_left) =
            binary_search_lt_first_core(left_low.view(), right.view(), left_index.view());
        assert_eq!(indices, vec![0]);
        assert_eq!(index_left, vec![1]);

        // left value equal to the last element -> no strictly-greater element
        let left_high = array![30_i64];
        let (indices, index_left) =
            binary_search_lt_first_core(left_high.view(), right.view(), left_index.view());
        assert!(indices.is_empty());
        assert!(index_left.is_empty());
    }

    #[test]
    fn generic_over_float_dtype() {
        let left = array![2.5_f64];
        let right = array![1.0_f64, 2.0, 3.0, 4.0];
        let left_index = array![9_i64];
        let (indices, index_left) =
            binary_search_lt_first_core(left.view(), right.view(), left_index.view());
        assert_eq!(indices, vec![2]);
        assert_eq!(index_left, vec![9]);
    }
}
