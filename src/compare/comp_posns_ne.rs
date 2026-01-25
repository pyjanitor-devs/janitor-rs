/// comparison on !=, based on positions
/// handles comparisions where nulls exist
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
            positions: PyReadonlyArray1<'py, i64>,
            left_booleans: PyReadonlyArray1<'py, bool>,
            right_booleans: PyReadonlyArray1<'py, bool>,
            is_extension_array: bool,
            op: i8,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
            let left = left.as_array();
            let right = right.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let positions = positions.as_array();
            let left_booleans = left_booleans.as_array();
            let right_booleans = right_booleans.as_array();
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left.into_iter(),
                left_booleans.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
            );
            for (position, (left_val, left_bool, start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let mut indexer = positions[nn];
                    //pd.NA != pd.NA returns pd.NA, which defaults to False
                    // pd.NA != anything returns pd.NA, which defaults to False
                    // whereas np.nan != np.nan returns True
                    // np.nan != anything returns True
                    if (indexer == -1) || (*left_bool && is_extension_array) {
                        result[n] = -1;
                        n += 1;
                        continue;
                    }
                    if *left_bool && !is_extension_array {
                        result[n] = indexer;
                        n += 1;
                        counter += 1;
                        total += 1;
                        continue;
                    }
                    let right_bool = right_booleans[indexer as usize];
                    if right_bool && is_extension_array {
                        result[n] = -1;
                        n += 1;
                        continue;
                    }
                    if right_bool && !is_extension_array {
                        result[n] = indexer;
                        n += 1;
                        counter += 1;
                        total += 1;
                        continue;
                    }
                    let right_val = right[indexer as usize];
                    let compare = binary_compare(left_val, &right_val, op);
                    counter += compare as i64;
                    total += compare as i64;
                    indexer = if compare { indexer } else { -1 };
                    result[n] = indexer;
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

generic_compare!(compare_posns_ne_int64, i64);
generic_compare!(compare_posns_ne_int32, i32);
generic_compare!(compare_posns_ne_int16, i16);
generic_compare!(compare_posns_ne_int8, i8);
generic_compare!(compare_posns_ne_uint64, u64);
generic_compare!(compare_posns_ne_uint32, u32);
generic_compare!(compare_posns_ne_uint16, u16);
generic_compare!(compare_posns_ne_uint8, u8);
generic_compare!(compare_posns_ne_float64, f64);
generic_compare!(compare_posns_ne_float32, f32);
