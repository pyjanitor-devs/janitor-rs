use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
            let arr = arr.as_array();
            let ends = ends.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<i64>::zeros(ends.len());
            for (pos, end) in ends.indexed_iter() {
                let mut base: i64 = -1;
                let mut base_val = arr[0];
                let end_ = *end as usize;
                for nn in 0..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    if (base == -1) || (current > base_val) {
                        base_val = current;
                        base = nn as i64;
                    }
                }
                result[pos] = base;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_max_end_int64, i64);
generic_compute!(compute_max_end_int32, i32);
generic_compute!(compute_max_end_int16, i16);
generic_compute!(compute_max_end_int8, i8);
generic_compute!(compute_max_end_uint64, u64);
generic_compute!(compute_max_end_uint32, u32);
generic_compute!(compute_max_end_uint16, u16);
generic_compute!(compute_max_end_uint8, u8);
generic_compute!(compute_max_end_f64, f64);
generic_compute!(compute_max_end_f32, f32);
