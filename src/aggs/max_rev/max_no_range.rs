use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, $type> = HashMap::with_capacity(length);
            let zipped = izip!(left_index.into_iter(), right_index.into_iter(), booleans.into_iter());
            for (index_left, index_right, boolean) in zipped {                
                let current = arr[*index_left as usize];
                let base = dictionary.entry(*index_right).or_insert(-1);
                let base_val = mapping.entry(*index_right).or_insert(current);
                if *boolean {
                    continue;
                }
                if (*base == -1) || (current > *base_val) {
                    *base_val = current;
                    *base = *index_left as i64;
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<i64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            (indexers.into_pyarray(py), result.into_pyarray(py))
        }
    };
}

compute!(compute_max_rev_no_range_int64, i64);
compute!(compute_max_rev_no_range_int32, i32);
compute!(compute_max_rev_no_range_int16, i16);
compute!(compute_max_rev_no_range_int8, i8);
compute!(compute_max_rev_no_range_uint64, u64);
compute!(compute_max_rev_no_range_uint32, u32);
compute!(compute_max_rev_no_range_uint16, u16);
compute!(compute_max_rev_no_range_uint8, u8);
compute!(compute_max_rev_no_range_f64, f64);
compute!(compute_max_rev_no_range_f32, f32);
