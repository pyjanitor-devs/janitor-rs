/// compare rows where starts and ends exist
/// but no matches exist yet
use itertools::izip;
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
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let starts_array = starts.as_array();
            let ends_array = ends.as_array();
            let right_len = right_array.len();
            // Mirrors `compare_start_end_core` (comp.rs) exactly, plain
            // comparisons and all -- see that function's doc comment for
            // why (`checked_range` measurably regressed this same kind of
            // hot per-row loop there). The raw `e - s` sum this replaces
            // let an inverted or negative-start row (`e - s` negative)
            // wrap `length as usize` to a huge value and corrupt every
            // other row's `Array1::zeros` sizing; `e - s` alone also never
            // bounded `end` against `right.len()`.
            // Compares `right_len as i64` (a lossless widening cast)
            // against `end` rather than casting `end` down to `usize`
            // first -- on a 32-bit target a genuinely oversized `end`
            // would truncate to a small value before either check below
            // ever saw it, silently passing validation instead of being
            // skipped.
            let length: usize = starts_array
                .iter()
                .zip(ends_array.iter())
                .filter(|(s, e)| **s >= 0 && **e != -1 && **s < **e && **e <= right_len as i64)
                .map(|(s, e)| (*e as usize) - (*s as usize))
                .sum();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left_array.len());
            let mut n: usize = 0;
            let mut total: i64 = 0;
            let zipped = izip!(
                left_array.into_iter(),
                starts_array.into_iter(),
                ends_array.into_iter(),
            );
            for (position, (left_val, start, end)) in zipped.enumerate() {
                if *start < 0 || *end == -1 || *start >= *end || *end > right_len as i64 {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
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

generic_compare!(compare_first_start_end_int64, i64);
generic_compare!(compare_first_start_end_int32, i32);
generic_compare!(compare_first_start_end_int16, i16);
generic_compare!(compare_first_start_end_int8, i8);
generic_compare!(compare_first_start_end_uint64, u64);
generic_compare!(compare_first_start_end_uint32, u32);
generic_compare!(compare_first_start_end_uint16, u16);
generic_compare!(compare_first_start_end_uint8, u8);
generic_compare!(compare_first_start_end_f64, f64);
generic_compare!(compare_first_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_first_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    type CompareResult<'py> = PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)>;

    fn run(py: Python<'_>, start: i64, end: i64) -> CompareResult<'_> {
        let left = PyArray1::from_vec(py, vec![1_i64]);
        let right = PyArray1::from_vec(py, vec![1_i64]);
        let starts = PyArray1::from_vec(py, vec![start]);
        let ends = PyArray1::from_vec(py, vec![end]);
        compare_first_start_end_int64(
            py,
            left.readonly(),
            right.readonly(),
            starts.readonly(),
            ends.readonly(),
            0,
        )
    }

    #[test]
    fn end_beyond_right_len_contributes_nothing_not_a_panic() {
        // Issue #56's repro for this file: right has length 1, end=2
        // walks 0..2 and indexes right_array[1].
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 2).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), Vec::<i8>::new());
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn both_bounds_negative_contributes_nothing_not_a_panic() {
        // Same shape as comp.rs's `both_bounds_negative_but_start_less_than_end...`
        // test: start=-3, end=-2 satisfy start < end in i64 space but both
        // wrap to huge, still-ordered usize values if unchecked.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, -3, -2).expect("must not panic");
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
            let (result, counts, total) = run(py, 0, 1).expect("must not panic");
            // left=[1], right=[1], op=Gt: 1 > 1 is false.
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i8]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
