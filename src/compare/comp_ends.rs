/// compare rows where only ends exist (usually a >/>= join)
/// and matches already exist
use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::op::CompareOp;

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let ends_array = ends.as_array();
            let matches_array = matches.as_array();
            let counts = counts.as_array();
            // `matches`/`counts` are supplied by the caller, already sized
            // to this row layout's real (unclamped) widths -- unlike
            // `compare_start_end_core` (comp.rs), which owns its own tape
            // end to end, this function can't silently treat an
            // out-of-bounds `end` as "contributes zero" without
            // desynchronizing `n` from every row after it. Reject the
            // whole call up front instead: a negative `end` wraps to a
            // huge `usize` and a too-large one survives unchanged, either
            // way walking `right_array[nn]` out of bounds once the main
            // loop reaches it.
            let right_len = right_array.len();
            // Comparing in i64 space (`right_len as i64`, a lossless
            // widening cast) rather than casting `end` down to `usize`
            // first: on a 32-bit target `usize` is 32 bits, so a
            // genuinely oversized `end` (e.g. 2^32 + 1) would truncate to
            // a small value *before* this check ever saw it, silently
            // passing validation instead of being rejected.
            if let Some(bad_end) = ends_array
                .iter()
                .find(|e| **e < 0 || **e > right_len as i64)
            {
                return Err(PyValueError::new_err(format!(
                    "end must be within 0..={right_len}; got {bad_end}"
                )));
            }
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i8>::zeros(matches_array.len());
            let mut counts_array = Array1::<i64>::zeros(left_array.len());
            let mut total: i64 = 0;
            let start = 0;
            let mut n = 0;
            let zipped = izip!(
                left_array.into_iter(),
                ends_array.into_iter(),
                counts.into_iter()
            );
            for (position, (left_val, end, count)) in zipped.enumerate() {
                let end_ = *end as usize;
                if *count == 0 {
                    let size = end_ - start;
                    n += size;
                    continue;
                }
                let mut counter: i64 = 0;
                for nn in start..end_ {
                    if matches_array[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let right_val = right_array[nn];
                    let compare = op.apply(left_val, &right_val);
                    counter += compare as i64;
                    total += compare as i64;
                    result[n] = compare as i8;
                    n += 1;
                }

                counts_array[position] = counter;
            }
            Ok((
                result.into_pyarray(py),
                counts_array.into_pyarray(py),
                total,
            ))
        }
    };
}

generic_compare!(compare_end_int64, i64);
generic_compare!(compare_end_int32, i32);
generic_compare!(compare_end_int16, i16);
generic_compare!(compare_end_int8, i8);
generic_compare!(compare_end_uint64, u64);
generic_compare!(compare_end_uint32, u32);
generic_compare!(compare_end_uint16, u16);
generic_compare!(compare_end_uint8, u8);
generic_compare!(compare_end_f64, f64);
generic_compare!(compare_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    type CompareResult<'py> = PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)>;

    fn run(py: Python<'_>, end: i64) -> CompareResult<'_> {
        let left = PyArray1::from_vec(py, vec![1_i64]);
        let right = PyArray1::from_vec(py, vec![1_i64]);
        let ends = PyArray1::from_vec(py, vec![end]);
        let counts = PyArray1::from_vec(py, vec![1_i64]);
        let matches = PyArray1::from_vec(py, vec![1_i8]);
        compare_end_int64(
            py,
            left.readonly(),
            right.readonly(),
            ends.readonly(),
            counts.readonly(),
            matches.readonly(),
            0, // CompareOp::Gt
        )
    }

    #[test]
    fn end_beyond_right_len_is_rejected_not_a_panic() {
        // right has length 1; end=2 used to walk 0..2 and index
        // right_array[1], out of bounds.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let error = run(py, 2).expect_err("end beyond right.len() must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(
                error.value(py).to_string().contains("0..=1"),
                "expected the valid range in the message, got {error:?}"
            );
        });
    }

    #[test]
    fn negative_end_is_rejected_not_a_panic() {
        // A negative end (not the crate's -1 sentinel elsewhere -- this
        // file has no sentinel case at all) casts to a huge usize if
        // unchecked, walking right_array/matches_array far out of bounds.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let error = run(py, -2).expect_err("a negative end must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn end_equal_to_right_len_is_accepted() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 1).expect("end == right.len() is valid");
            // left=[1], right=[1], op=Gt: 1 > 1 is false.
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i8]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
