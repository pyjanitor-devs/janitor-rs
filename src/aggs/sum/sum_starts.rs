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
            let end_: usize = arr.len();
            for (pos, start) in starts.indexed_iter() {
                let mut total: i64 = 0;
                let start_ = *start as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current = arr[nn];
                    total += current as i64;
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
            starts: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(starts.len());
            let end_: usize = arr.len();
            for (pos, start) in starts.indexed_iter() {
                let mut total: f64 = 0.0;
                let mut compensation: f64 = 0.0;
                let start_ = *start as usize;
                for nn in start_..end_ {
                    if booleans[nn] {
                        continue;
                    }
                    let current: f64 = arr[nn] as f64;
                    let difference = current - compensation;
                    let increment = total + difference;
                    compensation = (increment - total) - difference;
                    total = increment;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_sum_start_int64, i64);
generic_compute!(compute_sum_start_int32, i32);
generic_compute!(compute_sum_start_int16, i16);
generic_compute!(compute_sum_start_int8, i8);
generic_compute!(compute_sum_start_uint64, u64);
generic_compute!(compute_sum_start_uint32, u32);
generic_compute!(compute_sum_start_uint16, u16);
generic_compute!(compute_sum_start_uint8, u8);
generic_compute_floats!(compute_sum_start_f32, f32);
generic_compute_floats!(compute_sum_start_f64, f64);
