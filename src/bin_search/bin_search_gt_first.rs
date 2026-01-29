use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
            let left = left.as_array();
            let right = right.as_array();
            let left_index = left_index.as_array();
            let mut result = Array1::<i64>::zeros(left.len());
            let mut total: usize = left.len();
            for (pos, left_value) in left.into_iter().enumerate() {
                let mut min_idx = 0;
                let mut max_idx = right.len();
                while min_idx < max_idx {
                    // to avoid overflow
                    // adapted from numba's implementation
                    let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
                    let current_value = right[mid_idx as usize];
                    if current_value >= *left_value {
                        max_idx = mid_idx;
                    } else {
                        min_idx = mid_idx + 1;
                    }
                }
                if min_idx == 0 {
                    total -= 1;
                    continue;
                }
                let mid_idx = min_idx - 1;
                let current_value = right[mid_idx as usize];
                if current_value == *left_value {
                    result[pos as usize] = 0 as i64;
                    total -= 1;
                    continue;
                }
                result[pos as usize] = min_idx as i64;
            }
            let mut index_left = Array1::<i64>::zeros(total as usize);
            let mut search_indices = Array1::<i64>::zeros(total as usize);
            let mut n = 0;
            for (pos, item) in result.into_iter().enumerate() {
                if item == 0 {
                    continue;
                }
                search_indices[n] = item;
                let ind = left_index[pos];
                index_left[n] = ind;
                n += 1;
            }
            (
                search_indices.into_pyarray(py),
                index_left.into_pyarray(py),
                total as i64,
            )
        }
    };
}

bin_search!(binary_search_gt_first_int64, i64);
bin_search!(binary_search_gt_first_int32, i32);
bin_search!(binary_search_gt_first_int16, i16);
bin_search!(binary_search_gt_first_int8, i8);
bin_search!(binary_search_gt_first_uint64, u64);
bin_search!(binary_search_gt_first_uint32, u32);
bin_search!(binary_search_gt_first_uint16, u16);
bin_search!(binary_search_gt_first_uint8, u8);
bin_search!(binary_search_gt_first_f64, f64);
bin_search!(binary_search_gt_first_f32, f32);
