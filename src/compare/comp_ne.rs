/// compare rows where starts and ends exist - for !=
/// and matches exist
use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1, ArrayViewMut1};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::op::CompareOp;

#[allow(clippy::too_many_arguments)]
pub fn compare_ne_start_end_in_place_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    left_booleans: ArrayView1<bool>,
    right_booleans: ArrayView1<bool>,
    mut matches: ArrayViewMut1<'_, i8>,
    is_extension_array: bool,
    op: CompareOp,
) -> (Array1<i64>, i64) {
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut total = 0;
    let mut n = 0;
    for (position, (left_val, left_bool, start, end)) in izip!(
        left.into_iter(),
        left_booleans.into_iter(),
        starts.into_iter(),
        ends.into_iter()
    )
    .enumerate()
    {
        let start_ = *start as usize;
        let end_ = *end as usize;
        let mut counter = 0;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let right_bool = right_booleans[nn];
            if (*left_bool || right_bool) && is_extension_array {
                matches[n] = 0;
                n += 1;
                continue;
            }
            if (*left_bool || right_bool) && !is_extension_array {
                matches[n] = 1;
                n += 1;
                counter += 1;
                total += 1;
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

#[allow(clippy::too_many_arguments)]
pub fn compare_ne_start_end_allocating_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    left_booleans: ArrayView1<bool>,
    right_booleans: ArrayView1<bool>,
    matches: ArrayView1<i8>,
    is_extension_array: bool,
    op: CompareOp,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(matches.len());
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut total = 0;
    let mut n = 0;
    for (position, (left_val, left_bool, start, end)) in izip!(
        left.into_iter(),
        left_booleans.into_iter(),
        starts.into_iter(),
        ends.into_iter()
    )
    .enumerate()
    {
        let start_ = *start as usize;
        let end_ = *end as usize;
        let mut counter = 0;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let right_bool = right_booleans[nn];
            if (*left_bool || right_bool) && is_extension_array {
                n += 1;
                continue;
            }
            if (*left_bool || right_bool) && !is_extension_array {
                result[n] = 1;
                n += 1;
                counter += 1;
                total += 1;
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
        #[allow(clippy::too_many_arguments)]
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            left_booleans: PyReadonlyArray1<'py, bool>,
            right_booleans: PyReadonlyArray1<'py, bool>,
            matches: Bound<'py, PyArray1<i8>>,
            is_extension_array: bool,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left_array = left.as_array();
            let right_array = right.as_array();
            let starts_array = starts.as_array();
            let ends_array = ends.as_array();
            let left_booleans_array = left_booleans.as_array();
            let right_booleans_array = right_booleans.as_array();
            let mut matches_array = matches
                .try_readwrite()
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            // Same reasoning as `comp_ends.rs`/`comp_starts.rs`: `matches`
            // is externally sized by the caller, so an invalid row can't
            // be silently skipped without desynchronizing `n` from every
            // subsequent row. A negative `start`/`end` wraps to a huge
            // `usize`, and a positive but oversized `end` survives
            // unchanged -- either way walking `right_booleans_array[nn]`/
            // `right_array[nn]` out of bounds once the main loop reaches
            // it. `start >= 0 && start < end` already proves `end > 0`,
            // so no separate `end < 0` check is needed.
            let right_len = right_array.len();
            // Compares `right_len as i64` against `end` rather than
            // casting `end` down to `usize` first -- on a 32-bit target a
            // genuinely oversized `end` would truncate to a small value
            // before this check ever saw it, silently passing validation.
            if let Some((bad_start, bad_end)) = starts_array
                .iter()
                .zip(ends_array.iter())
                .find(|(s, e)| **s < 0 || **s >= **e || **e > right_len as i64)
            {
                return Err(PyValueError::new_err(format!(
                    "start ({bad_start}) and end ({bad_end}) must satisfy 0 <= start < end <= {right_len}"
                )));
            }
            let op = CompareOp::try_from_code(op)?;
            let (counts_array, total) = compare_ne_start_end_in_place_core(
                left_array,
                right_array,
                starts_array,
                ends_array,
                left_booleans_array,
                right_booleans_array,
                matches_array.as_array_mut(),
                is_extension_array,
                op,
            );
            Ok((matches, counts_array.into_pyarray(py), total))
        }
    };
}

generic_compare!(compare_start_end_ne_int64, i64);
generic_compare!(compare_start_end_ne_int32, i32);
generic_compare!(compare_start_end_ne_int16, i16);
generic_compare!(compare_start_end_ne_int8, i8);
generic_compare!(compare_start_end_ne_uint64, u64);
generic_compare!(compare_start_end_ne_uint32, u32);
generic_compare!(compare_start_end_ne_uint16, u16);
generic_compare!(compare_start_end_ne_uint8, u8);
generic_compare!(compare_start_end_ne_f64, f64);
generic_compare!(compare_start_end_ne_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_start_end_ne_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_ne_f64, m)?)?;
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
        let matches = PyArray1::from_vec(py, vec![1_i8]);
        compare_start_end_ne_int64(
            py,
            left.readonly(),
            right.readonly(),
            starts.readonly(),
            ends.readonly(),
            left_booleans.readonly(),
            right_booleans.readonly(),
            matches.clone(),
            false,
            5, // CompareOp::Ne
        )
    }

    #[test]
    fn end_beyond_right_len_is_rejected_not_a_panic() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let error = run(py, 0, 2).expect_err("end beyond right.len() must be rejected");
            assert!(error.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn negative_start_is_rejected_not_a_panic() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let error = run(py, -2, 1).expect_err("a negative start must be rejected");
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
            let (result, counts, total) = run(py, 0, 1).expect("end == right.len() is valid");
            // left=[1], right=[1], op=Ne: 1 != 1 is false.
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i8]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }
}
