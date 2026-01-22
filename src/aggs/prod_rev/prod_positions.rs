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
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (current, start, end, boolean) in zipped {
                if *boolean {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let current_ = *current as i64;
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let pos = index[indexer as usize];
                    *dictionary.entry(pos).or_insert(1) *= current_;
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

compute_ints!(compute_prod_rev_positions_int64, i64);
compute_ints!(compute_prod_rev_positions_int32, i32);
compute_ints!(compute_prod_rev_positions_int16, i16);
compute_ints!(compute_prod_rev_positions_int8, i8);
compute_ints!(compute_prod_rev_positions_uint64, u64);
compute_ints!(compute_prod_rev_positions_uint32, u32);
compute_ints!(compute_prod_rev_positions_uint16, u16);
compute_ints!(compute_prod_rev_positions_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, f64> = HashMap::with_capacity(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (current, start, end, boolean) in zipped {
                if *boolean {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let current_ = *current as f64;
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let pos = index[indexer as usize];
                    *dictionary.entry(pos).or_insert(1.) *= current_;
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

compute_floats!(compute_prod_rev_positions_f64, f64);
compute_floats!(compute_prod_rev_positions_f32, f32);
