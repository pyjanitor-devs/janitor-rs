use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<i64>::zeros(starts.len());
            let end_ = arr.len();
            for (pos, start) in starts.indexed_iter() {
                let mut base: i64 = -1;
                let start_ = *start as usize;
                let mut base_val = arr[start_];
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    if (base == -1) || (current < base_val) {
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

generic_compute!(compute_min_start_int64, i64);
generic_compute!(compute_min_start_int32, i32);
generic_compute!(compute_min_start_int16, i16);
generic_compute!(compute_min_start_int8, i8);
generic_compute!(compute_min_start_uint64, u64);
generic_compute!(compute_min_start_uint32, u32);
generic_compute!(compute_min_start_uint16, u16);
generic_compute!(compute_min_start_uint8, u8);
generic_compute!(compute_min_start_f64, f64);
generic_compute!(compute_min_start_f32, f32);
