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
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let zipped = izip!(arr.into_iter(), ends.into_iter(), booleans.into_iter());
            let mut base_val = arr[0];
            for (posn, (current, end, boolean)) in zipped.enumerate() {
                if *boolean {
                    continue;
                }
                let end_ = *end as usize;
                let mut base: i64 = -1;
                for item in 0..end_ {
                    let pos = index[item];
                    if (base == -1) || (*current > base_val) {
                        base_val = *current;
                        base = posn as i64;
                    }
                    dictionary.insert(pos, base);
                }
            }
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

compute!(compute_max_rev_end_int64, i64);
compute!(compute_max_rev_end_int32, i32);
compute!(compute_max_rev_end_int16, i16);
compute!(compute_max_rev_end_int8, i8);
compute!(compute_max_rev_end_uint64, u64);
compute!(compute_max_rev_end_uint32, u32);
compute!(compute_max_rev_end_uint16, u16);
compute!(compute_max_rev_end_uint8, u8);
compute!(compute_max_rev_end_f64, f64);
compute!(compute_max_rev_end_f32, f32);
