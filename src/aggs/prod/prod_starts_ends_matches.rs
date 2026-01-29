use itertools::izip;
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
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let counts = counts.as_array();
            let matches = matches.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<i64>::zeros(starts.len());
            let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
            let mut n: usize = 0;
            for (pos, (start, end, count)) in zipped.enumerate() {
                let mut total: i64 = 1;
                let start_ = *start as usize;
                let end_ = *end as usize;
                if *count == 0 {
                    let size = end_ - start_;
                    n += size;
                    continue;
                }
                for nn in start_..end_ {
                    if matches[n] == 0 || booleans[nn] {
                        n += 1;
                        continue;
                    }
                    let current = arr[nn];
                    total *= current as i64;
                    n += 1;
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
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let counts = counts.as_array();
            let matches = matches.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(starts.len());
            let zipped = izip!(starts.into_iter(), ends.into_iter(), counts.into_iter());
            let mut n: usize = 0;
            for (pos, (start, end, count)) in zipped.enumerate() {
                let mut total: f64 = 1.;
                let start_ = *start as usize;
                let end_ = *end as usize;
                if *count == 0 {
                    let size = end_ - start_;
                    n += size;
                    continue;
                }
                for nn in start_..end_ {
                    if matches[n] == 0 || booleans[nn] {
                        n += 1;
                        continue;
                    }
                    let current = arr[nn];
                    total *= current as f64;
                    n += 1;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_prod_start_end_match_int64, i64);
generic_compute!(compute_prod_start_end_match_int32, i32);
generic_compute!(compute_prod_start_end_match_int16, i16);
generic_compute!(compute_prod_start_end_match_int8, i8);
generic_compute!(compute_prod_start_end_match_uint64, u64);
generic_compute!(compute_prod_start_end_match_uint32, u32);
generic_compute!(compute_prod_start_end_match_uint16, u16);
generic_compute!(compute_prod_start_end_match_uint8, u8);
generic_compute_floats!(compute_prod_start_end_match_f64, f64);
generic_compute_floats!(compute_prod_start_end_match_f32, f32);
