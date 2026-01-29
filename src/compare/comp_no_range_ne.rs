use itertools::izip;
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
            left_booleans: PyReadonlyArray1<'py, bool>,
            right_booleans: PyReadonlyArray1<'py, bool>,
            is_extension_array: bool,
            op: i64,
        ) -> (Bound<'py, PyArray1<i64>>, i64)
        // The macro will expand into the contents of this block.
        {
            let left = left.as_array();
            let right = right.as_array();
            let positions = positions.as_array();
            let left_booleans = left_booleans.as_array();
            let right_booleans = right_booleans.as_array();
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left.into_iter(),
                left_booleans.into_iter(),
                positions.into_iter(),
            );
            for (left_val, left_bool, right_pos) in zipped {
                //pd.NA != pd.NA returns pd.NA, which defaults to False
                // pd.NA != anything returns pd.NA, which defaults to False
                // whereas np.nan != np.nan returns True
                // np.nan != anything returns True
                if (*right_pos == -1) || (*left_bool && is_extension_array) {
                    result[n] = -1;
                    n += 1;
                    continue;
                }
                if *left_bool && !is_extension_array {
                    result[n] = *right_pos;
                    n += 1;
                    total += 1;
                    continue;
                }
                let right_bool = right_booleans[*right_pos as usize];
                if right_bool && is_extension_array {
                    result[n] = -1;
                    n += 1;
                    continue;
                }
                if right_bool && !is_extension_array {
                    result[n] = *right_pos;
                    n += 1;
                    total += 1;
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

generic_compare!(compare_no_range_ne_int64, i64);
generic_compare!(compare_no_range_ne_int32, i32);
generic_compare!(compare_no_range_ne_int16, i16);
generic_compare!(compare_no_range_ne_int8, i8);
generic_compare!(compare_no_range_ne_uint64, u64);
generic_compare!(compare_no_range_ne_uint32, u32);
generic_compare!(compare_no_range_ne_uint16, u16);
generic_compare!(compare_no_range_ne_uint8, u8);
generic_compare!(compare_no_range_ne_f64, f64);
generic_compare!(compare_no_range_ne_f32, f32);
