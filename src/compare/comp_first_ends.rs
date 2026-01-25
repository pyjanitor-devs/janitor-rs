/// compare rows where only ends exist (usually a >/>= join)
/// and matches does not exist yet
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
            ends: PyReadonlyArray1<'py, i64>,
            op: i8,
        ) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
            let left = left.as_array();
            let right = right.as_array();
            let ends = ends.as_array();
            let length = ends.sum();
            let mut result = Array1::<i8>::zeros(length as usize);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(ends.into_iter());
            for (position, (left_val, end)) in zipped.enumerate() {
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in 0..end_ {
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

generic_compare!(compare_first_end_int64, i64);
generic_compare!(compare_first_end_int32, i32);
generic_compare!(compare_first_end_int16, i16);
generic_compare!(compare_first_end_int8, i8);
generic_compare!(compare_first_end_uint64, u64);
generic_compare!(compare_first_end_uint32, u32);
generic_compare!(compare_first_end_uint16, u16);
generic_compare!(compare_first_end_uint8, u8);
generic_compare!(compare_first_end_float64, f64);
generic_compare!(compare_first_end_float32, f32);
