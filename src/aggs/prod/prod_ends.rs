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
            let start_: usize = 0;
            for (pos, end) in ends.indexed_iter() {
                let mut total: i64 = 1;
                let end_ = *end as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    total *= current as i64;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
        }
    };
}

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(ends.len());
            for (pos, end) in ends.indexed_iter() {
                let mut total: f64 = 1.;
                let end_ = *end as usize;
                for nn in 0..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    total *= current as f64;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_prod_end_int64, i64);
generic_compute!(compute_prod_end_int32, i32);
generic_compute!(compute_prod_end_int16, i16);
generic_compute!(compute_prod_end_int8, i8);
generic_compute!(compute_prod_end_uint64, u64);
generic_compute!(compute_prod_end_uint32, u32);
generic_compute!(compute_prod_end_uint16, u16);
generic_compute!(compute_prod_end_uint8, u8);
generic_compute_floats!(compute_prod_end_f32, f32);
generic_compute_floats!(compute_prod_end_f64, f64);
