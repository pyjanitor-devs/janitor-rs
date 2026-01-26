use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn binary_compare<T: std::cmp::PartialOrd>(left: &T, right: &T, op: i64) -> bool {
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
            positions: PyReadonlyArray1<'py, i64>,
            op: i64,
        ) -> (Bound<'py, PyArray1<i64>>, i64)
        // The macro will expand into the contents of this block.
        {
            let left = left.as_array();
            let right = right.as_array();
            let positions = positions.as_array();
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(positions.into_iter());
            for (left_val, right_pos) in zipped {
                if *right_pos == -1 {
                    result[n] = -1;
                    n += 1;
                    continue;
                }
                let right_val = right[*right_pos as usize];
                let compare = binary_compare(left_val, &right_val, op);
                total += compare as i64;
                result[n] = if compare { *right_pos } else { -1 };
                n += 1;
            }
            (result.into_pyarray(py), total)
        }
    };
}

generic_compare!(compare_no_range_int64, i64);
generic_compare!(compare_no_range_int32, i32);
generic_compare!(compare_no_range_int16, i16);
generic_compare!(compare_no_range_int8, i8);
generic_compare!(compare_no_range_uint64, u64);
generic_compare!(compare_no_range_uint32, u32);
generic_compare!(compare_no_range_uint16, u16);
generic_compare!(compare_no_range_uint8, u8);
generic_compare!(compare_no_range_f64, f64);
generic_compare!(compare_no_range_f32, f32);
