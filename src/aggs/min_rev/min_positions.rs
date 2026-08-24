use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, checked_range, ensure_equal_lengths};
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
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, $type> = HashMap::with_capacity(length);
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                booleans.into_iter()
            );
            for (posn, (current, start, end, boolean)) in zipped.enumerate() {
                let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                    continue;
                };
                for nn in start_..end_ {
                    let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                        continue;
                    };
                    let pos = index[indexer_];
                    let base = dictionary.entry(pos).or_insert(-1);
                    let base_val = mapping.entry(pos).or_insert(*current);
                    if *boolean {
                        continue;
                    }
                    if (*base == -1) || (*current < *base_val) {
                        *base_val = *current;
                        *base = posn as i64;
                    }
                }
            }
            let length = dictionary.len();
            let mut indexers = Array1::<i64>::zeros(length);
            let mut result = Array1::<i64>::zeros(length);
            for (pos, (key, val)) in dictionary.iter().enumerate() {
                indexers[pos] = *key;
                result[pos] = *val;
            }
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_min_rev_positions_int64, i64);
compute!(compute_min_rev_positions_int32, i32);
compute!(compute_min_rev_positions_int16, i16);
compute!(compute_min_rev_positions_int8, i8);
compute!(compute_min_rev_positions_uint64, u64);
compute!(compute_min_rev_positions_uint32, u32);
compute!(compute_min_rev_positions_uint16, u16);
compute!(compute_min_rev_positions_uint8, u8);
compute!(compute_min_rev_positions_f64, f64);
compute!(compute_min_rev_positions_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_positions_f64, m)?)?;
    Ok(())
}
