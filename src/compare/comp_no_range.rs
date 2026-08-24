use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::op::CompareOp;
use crate::aggs::checked_index;

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            positions: PyReadonlyArray1<'py, i64>,
            op: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, i64)>
        // The macro will expand into the contents of this block.
        {
            let left = left.as_array();
            let right = right.as_array();
            let positions = positions.as_array();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(positions.into_iter());
            for (left_val, right_pos) in zipped {
                // ELI5: `right_pos` is a raw position read straight from
                // the caller-supplied `positions` array, not a `start..end`
                // range with a natural empty case. The `-1` sentinel was
                // already handled here, but any other out-of-bounds value
                // (negative-not-`-1`, or `>= right.len()`) fell straight
                // into `right[...]` unchecked; treat every unresolvable
                // position the same way the sentinel already was, as "no
                // match".
                let Some(pos) = checked_index(*right_pos, right.len()) else {
                    result[n] = -1;
                    n += 1;
                    continue;
                };
                let right_val = right[pos];
                let compare = op.apply(left_val, &right_val);
                total += compare as i64;
                result[n] = if compare { *right_pos } else { -1 };
                n += 1;
            }
            Ok((result.into_pyarray(py), total))
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

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_no_range_f64, m)?)?;
    Ok(())
}
