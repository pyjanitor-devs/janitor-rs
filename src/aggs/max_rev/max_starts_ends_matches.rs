use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let counts = counts.as_array();
            let matches = matches.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, $type> = HashMap::with_capacity(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter(),
            );
            let mut n: usize = 0;
            for (posn, (current, start, end, count, boolean)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                if *boolean || (*count == 0) {
                    let size = end_ - start_;
                    n += size;
                    continue;
                }
                for item in start_..end_ {
                    if matches[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    let base = dictionary.entry(pos).or_insert(-1);
                    let base_val = mapping.entry(pos).or_insert(*current);
                    if (*base == -1) || (*current > *base_val) {
                        *base_val = *current;
                        *base = posn as i64;
                    }
                    n += 1;
                }
            }
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

compute!(compute_max_rev_start_end_match_int64, i64);
compute!(compute_max_rev_start_end_match_int32, i32);
compute!(compute_max_rev_start_end_match_int16, i16);
compute!(compute_max_rev_start_end_match_int8, i8);
compute!(compute_max_rev_start_end_match_uint64, u64);
compute!(compute_max_rev_start_end_match_uint32, u32);
compute!(compute_max_rev_start_end_match_uint16, u16);
compute!(compute_max_rev_start_end_match_uint8, u8);
compute!(compute_max_rev_start_end_match_f64, f64);
compute!(compute_max_rev_start_end_match_f32, f32);
