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
            op: i8,
        ) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
            let left = left.as_array();
            let right = right.as_array();
            let starts = starts.as_array();
            let end: usize = right.len();
            let length: usize = starts.iter().map(|x| end - (*x as usize)).sum();
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(starts.into_iter());
            for (position, (left_val, start)) in zipped.enumerate() {
                let start_ = *start as usize;
                let mut counter: i64 = 0;
                for nn in start_..end {
                    let right_val = right[nn];
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

generic_compare!(compare_first_start_int64, i64);
generic_compare!(compare_first_start_int32, i32);
generic_compare!(compare_first_start_int16, i16);
generic_compare!(compare_first_start_int8, i8);
generic_compare!(compare_first_start_uint64, u64);
generic_compare!(compare_first_start_uint32, u32);
generic_compare!(compare_first_start_uint16, u16);
generic_compare!(compare_first_start_uint8, u8);
generic_compare!(compare_first_start_f64, f64);
generic_compare!(compare_first_start_f32, f32);
