use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn binary_compare<T: std::cmp::PartialOrd>(left: &T, right: &T, op: i8) -> bool {
    match op {
        0 => left > right,
        1 => left >= right,
        2 => left < right,
        3 => left <= right,
        4 => left == right,
        _ => left != right,
    }
}

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            op: i8,
        ) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let starts_array = starts.as_array();
            let ends_array = ends.as_array();
            let matches_array = matches.as_array();

            let mut result = Array1::<i8>::zeros(matches_array.len());
            let mut counts_array = Array1::<i64>::zeros(left_array.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left_array.into_iter(),
                starts_array.into_iter(),
                ends_array.into_iter()
            );
            for (position, (left_val, start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    if matches_array[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let right_val = right_array[nn];
                    let compare = binary_compare(left_val, &right_val, op);
                    counter += compare as i64;
                    total += compare as i64;
                    result[n] = compare as i8;
                    n += 1;
                }
                counts_array[position] = counter;
            }
            (
                result.into_pyarray(py),
                counts_array.into_pyarray(py),
                total,
            )
        }
    };
}

generic_compare!(compare_start_end_int64, i64);
generic_compare!(compare_start_end_int32, i32);
generic_compare!(compare_start_end_int16, i16);
generic_compare!(compare_start_end_int8, i8);
generic_compare!(compare_start_end_uint64, u64);
generic_compare!(compare_start_end_uint32, u32);
generic_compare!(compare_start_end_uint16, u16);
generic_compare!(compare_start_end_uint8, u8);
generic_compare!(compare_start_end_float64, f64);
generic_compare!(compare_start_end_float32, f32);
