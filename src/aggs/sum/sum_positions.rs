use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! generic_compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<i64>::zeros(starts.len());
            let zipped = starts.into_iter().zip(ends.into_iter());
            for (pos, (start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut total: i64 = 0;
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let indexer_: usize = indexer as usize;
                    if booleans[indexer_] {
                        continue;
                    }
                    let current = arr[indexer_];
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
            ends: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<f64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<f64>::zeros(starts.len());
            let zipped = starts.into_iter().zip(ends.into_iter());
            for (pos, (start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut total: f64 = 0.0;
                let mut compensation: f64 = 0.0;
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let indexer_: usize = indexer as usize;
                    if booleans[indexer_] {
                        continue;
                    }
                    let current: f64 = arr[indexer_] as f64;
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

generic_compute_ints!(compute_sum_positions_int64, i64);
generic_compute_ints!(compute_sum_positions_int32, i32);
generic_compute_ints!(compute_sum_positions_int16, i16);
generic_compute_ints!(compute_sum_positions_int8, i64);
generic_compute_ints!(compute_sum_positions_uint64, u64);
generic_compute_ints!(compute_sum_positions_uint32, u32);
generic_compute_ints!(compute_sum_positions_uint16, u16);
generic_compute_ints!(compute_sum_positions_uint8, u8);
generic_compute_floats!(compute_sum_positions_f32, f32);
generic_compute_floats!(compute_sum_positions_f64, f64);
