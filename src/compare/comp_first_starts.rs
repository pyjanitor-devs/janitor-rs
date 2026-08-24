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
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left = left.as_array();
            let right = right.as_array();
            let starts = starts.as_array();
            let end: usize = right.len();
            // `end - (*x as usize)` used to underflow for any `start` that
            // wraps to (or already is) a `usize` greater than `end` --
            // most obviously a negative `start`, but also a merely
            // oversized positive one. The `start_..end` loop below is
            // actually safe as-is for such a row (a `Range` with
            // `start > end` just iterates zero times), so contributing 0
            // to `length` here keeps `length` consistent with what the
            // loop will actually produce, without needing to touch the
            // loop itself.
            let length: usize = starts
                .iter()
                .map(|x| {
                    if *x < 0 {
                        return 0;
                    }
                    let start_ = *x as usize;
                    end.saturating_sub(start_)
                })
                .sum();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(starts.into_iter());
            for (position, (left_val, start)) in zipped.enumerate() {
                // No extra guard needed here: a negative `start` wraps to
                // a huge `usize`, and `start_..end` (a plain `Range`) is
                // already well-defined and empty whenever `start_ > end`
                // -- it's only the `length` precomputation above that
                // needed fixing.
                let start_ = *start as usize;
                let mut counter: i64 = 0;
                for nn in start_..end {
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

generic_compare!(compare_first_start_int64, i64);
generic_compare!(compare_first_start_int32, i32);
generic_compare!(compare_first_start_int16, i16);
generic_compare!(compare_first_start_int8, i8);
generic_compare!(compare_first_start_uint64, u64);
generic_compare!(compare_first_start_uint32, u32);
generic_compare!(compare_first_start_uint16, u16);
generic_compare!(compare_first_start_uint8, u8);
generic_compare!(compare_first_start_f64, f64);
generic_compare!(compare_first_start_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_first_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_first_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    type CompareResult<'py> = PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)>;

    fn run(py: Python<'_>, start: i64) -> CompareResult<'_> {
        let left = PyArray1::from_vec(py, vec![1_i64]);
        let right = PyArray1::from_vec(py, vec![1_i64]);
        let starts = PyArray1::from_vec(py, vec![start]);
        compare_first_start_int64(py, left.readonly(), right.readonly(), starts.readonly(), 0)
    }

    #[test]
    fn negative_start_contributes_nothing_not_a_panic() {
        // The old `end - (*x as usize)` length precomputation underflowed
        // for a negative start (wraps to a huge usize > end).
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
    fn start_beyond_right_len_contributes_nothing_not_a_panic() {
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
    fn start_equal_to_right_len_is_valid_empty_range() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 1).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), Vec::<i8>::new());
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
