use itertools::izip;
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
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
        ) -> Bound<'py, PyArray1<i64>> {
            let left = left.as_array();
            let right = right.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let mut result = Array1::<i64>::zeros(left.len());
            let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
            for (pos, (left_value, start, end)) in zipped.enumerate() {
                if *start == -1 || *end == -1 || *start >= *end {
                    result[pos as usize] = -1;
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
                    result[pos as usize] = -1;
                    continue;
                }
                result[pos as usize] = min_idx;
            }
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
