use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            let matches = matches.as_array();
            let counts = counts.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<i64>::zeros(ends.len());
            let mut n: usize = 0;
            let zipped = ends.into_iter().zip(counts.into_iter());
            for (pos, (end, count)) in zipped.enumerate() {
                let end_ = *end as usize;
                let mut total: i64 = 1;
                if *count == 0 {
                    n += end_;
                    continue;
                }
                for nn in 0..end_ {
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

generic_compute_ints!(compute_prod_end_match_int64, i64);
generic_compute_ints!(compute_prod_end_match_int32, i32);
generic_compute_ints!(compute_prod_end_match_int16, i16);
generic_compute_ints!(compute_prod_end_match_int8, i8);
generic_compute_ints!(compute_prod_end_match_uint64, u64);
generic_compute_ints!(compute_prod_end_match_uint32, u32);
generic_compute_ints!(compute_prod_end_match_uint16, u16);
generic_compute_ints!(compute_prod_end_match_uint8, u8);

macro_rules! generic_compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            let matches = matches.as_array();
            let counts = counts.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(ends.len());
            let mut n: usize = 0;
            let zipped = ends.into_iter().zip(counts.into_iter());
            for (pos, (end, count)) in zipped.enumerate() {
                let end_ = *end as usize;
                let mut total: f64 = 1.;
                if *count == 0 {
                    n += end_;
                    continue;
                }
                for nn in 0..end_ {
                    if matches[n] == 0 || booleans[nn] {
                        n += 1;
                        continue;
                    }
                    let current: f64 = arr[nn] as f64;
                    total *= current;
                    n += 1;
                }
                result[pos] = total;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute_floats!(compute_prod_end_match_f64, f64);
generic_compute_floats!(compute_prod_end_match_f32, f32);
