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
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let zipped = izip!(arr.into_iter(), ends.into_iter(), booleans.into_iter());
            for (current, end, boolean) in zipped {
                let end_ = *end as usize;
                let current_ = *current as i64;
                for item in 0..end_ {
                    let pos = index[item];
                    let total = dictionary.entry(pos).or_insert(0);
                    if *boolean {
                    continue;
                }
                    *total += current_;
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

compute_ints!(compute_sum_rev_end_int64, i64);
compute_ints!(compute_sum_rev_end_int32, i32);
compute_ints!(compute_sum_rev_end_int16, i16);
compute_ints!(compute_sum_rev_end_int8, i8);
compute_ints!(compute_sum_rev_end_uint64, u64);
compute_ints!(compute_sum_rev_end_uint32, u32);
compute_ints!(compute_sum_rev_end_uint16, u16);
compute_ints!(compute_sum_rev_end_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, f64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, f64> = HashMap::with_capacity(length);
            let zipped = izip!(arr.into_iter(), ends.into_iter(), booleans.into_iter());
            for (current, end, boolean) in zipped {                
                let end_ = *end as usize;
                let current_ = *current as f64;
                for item in 0..end_ {
                    let pos = index[item];
                    let total = dictionary.entry(pos).or_insert(0.);
                    let compensation = mapping.entry(pos).or_insert(0.);
                    if *boolean {
                    continue;
                }
                    let difference = current_ - *compensation;
                    let increment = *total + difference;
                    // adapted from pandas' cython code
                    // # GH#53606; GH#60303
                    // # If val is +/- infinity compensation is NaN
                    // # which would lead to results being NaN instead
                    // # of +/- infinity. We cannot use util.is_nan
                    // # because of no gil
                    *compensation = (increment - *total) - difference;
                    if !compensation.is_finite() {
                        *compensation = 0.;
                    }
                    *total = increment;
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

compute_floats!(compute_sum_rev_end_f64, f64);
compute_floats!(compute_sum_rev_end_f32, f32);
