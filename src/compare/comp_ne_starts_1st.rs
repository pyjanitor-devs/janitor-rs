/// compare rows where only starts exist - for !=
/// and matches does not exist
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
            left_booleans: PyReadonlyArray1<'py, bool>,
            right_booleans: PyReadonlyArray1<'py, bool>,
            is_extension_array: bool,
            op: i8,
        ) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let starts_array = starts.as_array();
            let left_booleans_array = left_booleans.as_array();
            let right_booleans_array = right_booleans.as_array();
            let end_: usize = right_array.len();
            let length: usize = starts_array.iter().map(|x| end_ - (*x as usize)).sum();
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left_array.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left_array.into_iter(),
                left_booleans_array.into_iter(),
                starts_array.into_iter(),
            );
            for (position, (left_val, left_bool, start)) in zipped.enumerate() {
                let start_ = *start as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let right_bool_ = right_booleans_array[nn];
                    // pd.NA != pd.NA returns pd.NA, which defaults to False
                    // pd.NA != anything returns pd.NA, which defaults to False
                    // whereas np.nan != np.nan returns True
                    // np.nan != anything returns True
                    if (*left_bool || right_bool_) && is_extension_array {
                        n += 1;
                        continue;
                    }
                    if (*left_bool || right_bool_) && !is_extension_array {
                        result[n] = 1;
                        counter += 1;
                        total += 1;
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

generic_compare!(compare_start_ne_1st_int64, i64);
generic_compare!(compare_start_ne_1st_int32, i32);
generic_compare!(compare_start_ne_1st_int16, i16);
generic_compare!(compare_start_ne_1st_int8, i8);
generic_compare!(compare_start_ne_1st_uint64, u64);
generic_compare!(compare_start_ne_1st_uint32, u32);
generic_compare!(compare_start_ne_1st_uint16, u16);
generic_compare!(compare_start_ne_1st_uint8, u8);
generic_compare!(compare_start_ne_1st_float64, f64);
generic_compare!(compare_start_ne_1st_float32, f32);
