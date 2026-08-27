/// compare rows where only starts exist (usually a </<= join)
/// and matches already exist
use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1, ArrayViewMut1};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::op::CompareOp;

pub fn compare_start_in_place_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    mut matches: ArrayViewMut1<'_, i8>,
    op: CompareOp,
) -> (Array1<i64>, i64) {
    let end = right.len();
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut total = 0;
    let mut n = 0;
    for (position, (left_val, start, count)) in
        izip!(left.into_iter(), starts.into_iter(), counts.into_iter()).enumerate()
    {
        let start_ = *start as usize;
        if *count == 0 {
            n += end - start_;
            continue;
        }
        let mut counter = 0;
        for nn in start_..end {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let compare = op.apply(left_val, &right[nn]);
            matches[n] = compare as i8;
            counter += compare as i64;
            total += compare as i64;
            n += 1;
        }
        counts_array[position] = counter;
    }
    for value in matches.iter_mut().skip(n) {
        *value = 0;
    }
    (counts_array, total)
}

pub fn compare_start_allocating_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    counts: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    op: CompareOp,
) -> (Array1<i8>, Array1<i64>, i64) {
    let end = right.len();
    let mut result = Array1::<i8>::zeros(matches.len());
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut total = 0;
    let mut n = 0;
    for (position, (left_val, start, count)) in
        izip!(left.into_iter(), starts.into_iter(), counts.into_iter()).enumerate()
    {
        let start_ = *start as usize;
        if *count == 0 {
            n += end - start_;
            continue;
        }
        let mut counter = 0;
        for nn in start_..end {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let compare = op.apply(left_val, &right[nn]);
            result[n] = compare as i8;
            counter += compare as i64;
            total += compare as i64;
            n += 1;
        }
        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: Bound<'py, PyArray1<i8>>,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let starts_array = starts.as_array();
            let mut matches_array = matches
                .try_readwrite()
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let counts = counts.as_array();
            let end = right_array.len();
            // Same reasoning as `comp_ends.rs`: `matches`/`counts` are
            // externally sized by the caller, so an out-of-bounds `start`
            // can't be silently skipped without desynchronizing `n` for
            // every subsequent row. A negative `start` wraps to a huge
            // `usize`, and `size = end - start_` (the `count == 0` fast
            // path below) underflows for it -- as does any `start >
            // right.len()`, even a non-negative one. Reject up front.
            //
            // Compares in i64 space (`end as i64`) rather than casting
            // `start` down to `usize` first -- on a 32-bit target a
            // genuinely oversized `start` would truncate to a small value
            // before this check ever saw it, silently passing validation.
            if let Some(bad_start) = starts_array.iter().find(|s| **s < 0 || **s > end as i64) {
                return Err(PyValueError::new_err(format!(
                    "start must be within 0..={end}; got {bad_start}"
                )));
            }
            let op = CompareOp::try_from_code(op)?;
            let (counts_array, total) = compare_start_in_place_core(
                left_array,
                right_array,
                starts_array,
                counts,
                matches_array.as_array_mut(),
                op,
            );
            Ok((matches, counts_array.into_pyarray(py), total))
        }
    };
}

generic_compare!(compare_start_int64, i64);
generic_compare!(compare_start_int32, i32);
generic_compare!(compare_start_int16, i16);
generic_compare!(compare_start_int8, i8);
generic_compare!(compare_start_uint64, u64);
generic_compare!(compare_start_uint32, u32);
generic_compare!(compare_start_uint16, u16);
generic_compare!(compare_start_uint8, u8);
generic_compare!(compare_start_f64, f64);
generic_compare!(compare_start_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_f64, m)?)?;
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
        let counts = PyArray1::from_vec(py, vec![1_i64]);
        let matches = PyArray1::from_vec(py, vec![1_i8]);
        compare_start_int64(
            py,
            left.readonly(),
            right.readonly(),
            starts.readonly(),
            counts.readonly(),
            matches.clone(),
            0, // CompareOp::Gt
        )
    }

    #[test]
    fn start_beyond_right_len_is_rejected_not_a_panic() {
        // right has length 1; start=2 makes `size = end - start_`
        // underflow in the `count == 0` fast path (not exercised by this
        // call since count=1, but the same start_ also feeds the main
        // loop's start_..end range with start_ > end).
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let error = run(py, 2).expect_err("start beyond right.len() must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(
                error.value(py).to_string().contains("0..=1"),
                "expected the valid range in the message, got {error:?}"
            );
        });
    }

    #[test]
    fn negative_start_is_rejected_not_a_panic() {
        // A negative start casts to a huge usize if unchecked; the
        // `count == 0` fast path's `size = end - start_` then underflows
        // before any indexing even happens.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let error = run(py, -2).expect_err("a negative start must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn start_equal_to_right_len_is_accepted() {
        // start == right.len() is the ordinary empty-range case (no
        // candidates for this row), not an error.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 1).expect("start == right.len() is valid");
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i8]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
