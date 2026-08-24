use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// For every `left[i]`, find the last position in all of `right` whose
/// value is less than or equal to `left[i]`, returned as one past that
/// position (`right` is assumed sorted ascending, which in particular
/// means NaN-free -- see the contiguous-fast-path note below for what
/// that rules out and why it matters). Rows with no such
/// position are dropped entirely rather than represented with a sentinel;
/// only the surviving `(search index, left_index[i])` pairs are returned,
/// in row order.
///
/// ELI5: see `binary_search_lt_first_core`'s comment for why a single pass
/// pushing into two `Vec`s replaces the old "full-length array with an
/// internal no-match marker, then a second filtering pass" shape. Here `0`
/// is the value no genuine match can ever produce (a real match always
/// leaves `min_idx >= 1`), which is why the old code used it as that
/// marker; the one-pass version does not need a marker at all.
pub fn binary_search_ge_first_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    left_index: ArrayView1<i64>,
) -> (Vec<i64>, Vec<i64>) {
    let len_right = right.len();
    let right_slice = right.as_slice();
    let mut search_indices = Vec::new();
    let mut index_left = Vec::new();
    for (pos, left_value) in left.into_iter().enumerate() {
        let min_idx = if let Some(slice) = right_slice {
            // Written as `!(*v > *left_value)`, the literal negation of
            // the manual loop's "max side" condition below, not the
            // algebraically-equivalent `*v <= *left_value` -- those two
            // differ for NaN (see bin_search_ge.rs's core for the full
            // explanation), and this function accepts float dtypes
            // without validating them as NaN-free. The negation is
            // deliberate (clippy::neg_cmp_op_on_partial_ord suppressed
            // below), not the anti-pattern that lint normally flags.
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            let p = slice.partition_point(|v| !(*v > *left_value));
            p
        } else {
            let mut min_idx = 0;
            let mut max_idx = len_right;
            while min_idx < max_idx {
                // to avoid overflow
                // adapted from numba's implementation
                let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
                let current_value = right[mid_idx];
                if current_value > *left_value {
                    max_idx = mid_idx;
                } else {
                    min_idx = mid_idx + 1;
                }
            }
            min_idx
        };
        if min_idx == 0 {
            continue;
        }
        search_indices.push(min_idx as i64);
        index_left.push(left_index[pos]);
    }
    (search_indices, index_left)
}

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        /// For every `left[i]`, the last position in all of `right`
        /// whose value is less than or equal to `left[i]`, returned as
        /// one past that position. `right` is assumed sorted ascending,
        /// which in particular means NaN-free: for a `right` that
        /// contains NaN, this function will not panic, but its result is
        /// not guaranteed to be the same for a contiguous `right` array
        /// as for a non-contiguous (e.g. sliced with a step) one. See
        /// `binary_search_ge_first_core`'s doc comment (in the Rust
        /// source) for why.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
            let (search_indices, index_left) = binary_search_ge_first_core(
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

bin_search!(binary_search_ge_first_int64, i64);
bin_search!(binary_search_ge_first_int32, i32);
bin_search!(binary_search_ge_first_int16, i16);
bin_search!(binary_search_ge_first_int8, i8);
bin_search!(binary_search_ge_first_uint64, u64);
bin_search!(binary_search_ge_first_uint32, u32);
bin_search!(binary_search_ge_first_uint16, u16);
bin_search!(binary_search_ge_first_uint8, u8);
bin_search!(binary_search_ge_first_f64, f64);
bin_search!(binary_search_ge_first_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_ge_first_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_ge_first_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s};

    #[test]
    fn contiguous_and_strided_paths_agree() {
        let right_dense = array![1_i64, 3, 5, 5, 5, 9];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![1_i64, -1, 3, -1, 5, -1, 5, -1, 5, -1, 9, -1];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![0_i64, 5, 100];
        let left_index = array![10_i64, 20, 30];
        let fast = binary_search_ge_first_core(left.view(), right_dense.view(), left_index.view());
        let fallback = binary_search_ge_first_core(left.view(), right_strided, left_index.view());
        assert_eq!(fast, fallback);
    }

    #[test]
    fn contiguous_and_strided_paths_agree_when_the_query_is_nan() {
        // Regression test: the fast path used to be written as
        // `*v <= *left_value`, which is only equal to the fallback's
        // literal `!(*v > *left_value)` for totally-ordered values -- see
        // bin_search_ge.rs's core for the full explanation. `right`
        // itself stays genuinely sorted here (only the query is NaN),
        // which is in-contract input, unlike a `right` that itself
        // contains NaN (see `nan_in_right_does_not_panic_but_parity_is_not_guaranteed`
        // below).
        let right_dense = array![1.0_f64, 2.0, 3.0, 4.0];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![1.0_f64, -1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![f64::NAN];
        let left_index = array![0_i64];
        let fast = binary_search_ge_first_core(left.view(), right_dense.view(), left_index.view());
        let fallback = binary_search_ge_first_core(left.view(), right_strided, left_index.view());
        assert_eq!(fast, fallback);
    }

    #[test]
    fn nan_in_right_does_not_panic_but_parity_is_not_guaranteed() {
        // See bin_search_lt.rs's core for the full explanation: a `right`
        // containing NaN violates "sorted ascending," and the fast path's
        // branchless partition_point can then probe different elements
        // than the manual fallback loop even with an identical predicate,
        // landing on a genuinely different (but still in-bounds) answer.
        // Deliberately not asserting fast == fallback -- that's exactly
        // the parity this out-of-contract input isn't guaranteed to have.
        let right_dense = array![-1.0_f64, f64::NAN, 0.0, 1.0, 2.0, 3.0, 4.0];
        assert!(right_dense.view().as_slice().is_some());
        let right_padded = array![
            -1.0_f64,
            -99.0,
            f64::NAN,
            -99.0,
            0.0,
            -99.0,
            1.0,
            -99.0,
            2.0,
            -99.0,
            3.0,
            -99.0,
            4.0,
            -99.0
        ];
        let right_strided = right_padded.slice(s![..;2]);
        assert!(right_strided.as_slice().is_none());

        let left = array![0.0_f64];
        let left_index = array![0_i64];
        let (fast_indices, _) =
            binary_search_ge_first_core(left.view(), right_dense.view(), left_index.view());
        let (fallback_indices, _) =
            binary_search_ge_first_core(left.view(), right_strided, left_index.view());
        for indices in [&fast_indices, &fallback_indices] {
            for &idx in indices {
                assert!(
                    (0..7).contains(&idx),
                    "expected an index in 0..7, got {idx}"
                );
            }
        }
    }

    #[test]
    fn empty_input_returns_nothing() {
        let left: Array1<i64> = array![];
        let right: Array1<i64> = array![];
        let left_index: Array1<i64> = array![];
        let (indices, index_left) =
            binary_search_ge_first_core(left.view(), right.view(), left_index.view());
        assert!(indices.is_empty());
        assert!(index_left.is_empty());
    }

    #[test]
    fn left_value_below_the_first_element_is_dropped() {
        let right = array![10_i64, 20, 30];
        let left = array![5_i64];
        let left_index = array![1_i64];
        let (indices, index_left) =
            binary_search_ge_first_core(left.view(), right.view(), left_index.view());
        assert!(indices.is_empty());
        assert!(index_left.is_empty());
    }

    #[test]
    fn mixed_rows_keep_only_matches_with_their_own_left_index() {
        let right = array![10_i64, 20, 30];
        // row 0 (left=5, index=100): min_idx==0 -> dropped
        // row 1 (left=10, index=200): kept, search index 1
        // row 2 (left=35, index=300): kept, search index 3 (one past the end)
        let left = array![5_i64, 10, 35];
        let left_index = array![100_i64, 200, 300];
        let (indices, index_left) =
            binary_search_ge_first_core(left.view(), right.view(), left_index.view());
        assert_eq!(indices, vec![1, 3]);
        assert_eq!(index_left, vec![200, 300]);
    }

    #[test]
    fn generic_over_float_dtype() {
        let left = array![2.0_f64];
        let right = array![1.0_f64, 2.0, 3.0, 4.0];
        let left_index = array![9_i64];
        let (indices, index_left) =
            binary_search_ge_first_core(left.view(), right.view(), left_index.view());
        assert_eq!(indices, vec![2]);
        assert_eq!(index_left, vec![9]);
    }
}
