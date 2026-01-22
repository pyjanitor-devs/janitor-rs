use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            index: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let index = index.as_array();
            let ends = ends.as_array();
            let matches = matches.as_array();
            let counts = counts.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (current, end, count, boolean) in zipped {
                let end_ = *end as usize;
                if *boolean || (*count == 0) {
                    n += end_;
                    continue;
                }
                let current_ = *current as i64;
                for item in 0..end_ {
                    if matches[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    *dictionary.entry(pos).or_insert(0) += current_;
                    n += 1;
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<i64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            (indexers.into_pyarray(py), result.into_pyarray(py))
        }
    };
}

compute_ints!(compute_sum_rev_end_match_int64, i64);
compute_ints!(compute_sum_rev_end_match_int32, i32);
compute_ints!(compute_sum_rev_end_match_int16, i16);
compute_ints!(compute_sum_rev_end_match_int8, i8);
compute_ints!(compute_sum_rev_end_match_uint64, u64);
compute_ints!(compute_sum_rev_end_match_uint32, u32);
compute_ints!(compute_sum_rev_end_match_uint16, u16);
compute_ints!(compute_sum_rev_end_match_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            index: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let index = index.as_array();
            let ends = ends.as_array();
            let matches = matches.as_array();
            let counts = counts.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, f64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, f64> = HashMap::with_capacity(length);
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (current, end, count, boolean) in zipped {
                let end_ = *end as usize;
                if *boolean || (*count == 0) {
                    n += end_;
                    continue;
                }
                let current_ = *current as f64;
                for item in 0..end_ {
                    if matches[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    let total = dictionary.entry(pos).or_insert(0.);
                    let compensation = mapping.entry(pos).or_insert(0.);
                    let difference = current_ - *compensation;
                    let increment = *total + difference;
                    *compensation = (increment - *total) - difference;
                    *total = increment;
                    n += 1;
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<f64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            (indexers.into_pyarray(py), result.into_pyarray(py))
        }
    };
}

compute_floats!(compute_sum_rev_end_match_f64, f64);
compute_floats!(compute_sum_rev_end_match_f32, f32);
