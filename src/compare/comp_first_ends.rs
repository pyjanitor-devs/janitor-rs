/// compare rows where only ends exist (usually a >/>= join)
/// and matches does not exist yet
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left = left.as_array();
            let right = right.as_array();
            let ends = ends.as_array();
            let right_len = right.len();
            // Unlike `comp_ends.rs`, this file owns its own `result`
            // array end to end (no caller-supplied `matches` tape to stay
            // in sync with), so an invalid row can be silently skipped --
            // same convention as `compare_start_end_core` (comp.rs) --
            // rather than rejecting the whole call. A negative `end`
            // wraps to a huge `usize` if used unchecked, and even a
            // non-negative `end > right.len()` walks `right[nn]` out of
            // bounds; the raw `ends.sum()` this `length` used to be also
            // let a negative row corrupt every other row's `Array1::zeros`
            // sizing.
            // Compares in i64 space (`right_len as i64`) rather than
            // casting `end` down to `usize` first -- on a 32-bit target a
            // genuinely oversized `end` would truncate to a small value
            // before either check below ever saw it, silently passing
            // validation instead of being skipped.
            let length: usize = ends
                .iter()
                .filter(|e| **e >= 0 && **e <= right_len as i64)
                .map(|e| *e as usize)
                .sum();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(ends.into_iter());
            for (position, (left_val, end)) in zipped.enumerate() {
                if *end < 0 || *end > right_len as i64 {
                    continue;
                }
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in 0..end_ {
                    let right_val = right[nn];
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

generic_compare!(compare_first_end_int64, i64);
generic_compare!(compare_first_end_int32, i32);
generic_compare!(compare_first_end_int16, i16);
generic_compare!(compare_first_end_int8, i8);
generic_compare!(compare_first_end_uint64, u64);
generic_compare!(compare_first_end_uint32, u32);
generic_compare!(compare_first_end_uint16, u16);
generic_compare!(compare_first_end_uint8, u8);
generic_compare!(compare_first_end_f64, f64);
generic_compare!(compare_first_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_first_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_end_f64, m)?)?;
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
        compare_first_end_int64(py, left.readonly(), right.readonly(), ends.readonly(), 0)
    }

    #[test]
    fn end_beyond_right_len_contributes_nothing_not_a_panic() {
        // right has length 1; end=2 used to walk 0..2 and index right[1].
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 2).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), Vec::<i8>::new());
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn negative_end_contributes_nothing_not_a_panic() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, -2).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), Vec::<i8>::new());
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn end_equal_to_right_len_is_valid() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 1).expect("must not panic");
            // left=[1], right=[1], op=Gt: 1 > 1 is false.
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i8]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
