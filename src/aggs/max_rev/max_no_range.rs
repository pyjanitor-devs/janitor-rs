use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, ensure_equal_lengths};
use std::collections::HashMap;

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let arr = arr.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            let booleans = booleans.as_array();
            let length = length as usize;
            let mut dictionary: HashMap<i64, i64> = HashMap::with_capacity(length);
            let mut mapping: HashMap<i64, $type> = HashMap::with_capacity(length);
            let zipped = left_index.into_iter().zip(right_index.into_iter());
            for (index_left, index_right) in zipped {
                // ELI5: `index_left` names a position in `arr`/`booleans`,
                // read straight from the caller-supplied `left_index`
                // array -- unlike a `start..end` range, there's no natural
                // "empty" fallback here, so an out-of-bounds or negative
                // value must be rejected before it's used to index
                // anything. `right_index` is never used to index an array
                // (only as a `HashMap` key), so it needs no such guard.
                let Some(left) = checked_index(*index_left, arr.len()) else {
                    continue;
                };
                let current = arr[left];
                let boolean = booleans[left];
                let base = dictionary.entry(*index_right).or_insert(-1);
                let base_val = mapping.entry(*index_right).or_insert(current);
                if boolean {
                    continue;
                }
                if (*base == -1) || (current > *base_val) {
                    *base_val = current;
                    *base = left as i64;
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

compute!(compute_max_rev_no_range_int64, i64);
compute!(compute_max_rev_no_range_int32, i32);
compute!(compute_max_rev_no_range_int16, i16);
compute!(compute_max_rev_no_range_int8, i8);
compute!(compute_max_rev_no_range_uint64, u64);
compute!(compute_max_rev_no_range_uint32, u32);
compute!(compute_max_rev_no_range_uint16, u16);
compute!(compute_max_rev_no_range_uint8, u8);
compute!(compute_max_rev_no_range_f64, f64);
compute!(compute_max_rev_no_range_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_f64, m)?)?;
    Ok(())
}
