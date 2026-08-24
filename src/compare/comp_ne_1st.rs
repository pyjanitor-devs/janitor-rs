/// compare rows where starts and ends exist - for !=
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
            left_booleans: PyReadonlyArray1<'py, bool>,
            right_booleans: PyReadonlyArray1<'py, bool>,
            is_extension_array: bool,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let starts_array = starts.as_array();
            let ends_array = ends.as_array();
            let left_booleans_array = left_booleans.as_array();
            let right_booleans_array = right_booleans.as_array();
            let right_len = right_array.len();
            // No caller-supplied `matches` tape here (unlike comp_ne.rs),
            // so -- mirroring comp_first.rs/comp.rs -- an invalid row can
            // be silently skipped rather than rejecting the whole call,
            // as long as the length precomputation and the main loop's
            // skip condition agree on which rows that is. The raw `e - s`
            // sum this replaces let a negative-start or inverted row wrap
            // `length as usize` to a huge value before any indexing even
            // started.
            let length: usize = starts_array
                .iter()
                .zip(ends_array.iter())
                .filter(|(s, e)| **s >= 0 && **s < **e && (**e as usize) <= right_len)
                .map(|(s, e)| (*e as usize) - (*s as usize))
                .sum();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left_array.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left_array.into_iter(),
                left_booleans_array.into_iter(),
                starts_array.into_iter(),
                ends_array.into_iter(),
            );
            for (position, (left_val, left_bool, start, end)) in zipped.enumerate() {
                if *start < 0 || *start >= *end || (*end as usize) > right_len {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let right_bool_ = right_booleans_array[nn];
                    // pd.NA != pd.NA returns pd.NA, which defaults to False
                    // pd.NA != anything returns pd.NA, which defaults to False
                    // whereas np.nan != np.nan returns True
                    // np.nan != anything returns True
                    if (*left_bool || right_bool_) && is_extension_array {
                        n += 1;
                        continue;
                    }
                    if (*left_bool || right_bool_) && !is_extension_array {
                        result[n] = 1;
                        n += 1;
                        counter += 1;
                        total += 1;
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

generic_compare!(compare_start_end_ne_1st_int64, i64);
generic_compare!(compare_start_end_ne_1st_int32, i32);
generic_compare!(compare_start_end_ne_1st_int16, i16);
generic_compare!(compare_start_end_ne_1st_int8, i8);
generic_compare!(compare_start_end_ne_1st_uint64, u64);
generic_compare!(compare_start_end_ne_1st_uint32, u32);
generic_compare!(compare_start_end_ne_1st_uint16, u16);
generic_compare!(compare_start_end_ne_1st_uint8, u8);
generic_compare!(compare_start_end_ne_1st_f64, f64);
generic_compare!(compare_start_end_ne_1st_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_1st_f64, m)?)?;
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
        let left_booleans = PyArray1::from_vec(py, vec![false]);
        let right_booleans = PyArray1::from_vec(py, vec![false]);
        compare_start_end_ne_1st_int64(
            py,
            left.readonly(),
            right.readonly(),
            starts.readonly(),
            ends.readonly(),
            left_booleans.readonly(),
            right_booleans.readonly(),
            false,
            5, // CompareOp::Ne
        )
    }

    #[test]
    fn end_beyond_right_len_contributes_nothing_not_a_panic() {
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
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i8]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
