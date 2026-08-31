use itertools::izip;
use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{ensure_equal_lengths, ensure_exact_tape_width};

fn expected_matches_width(starts: ArrayView1<'_, i64>, right_len: usize) -> PyResult<usize> {
    if starts.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "starts cannot be empty",
        ));
    }
    starts.iter().try_fold(0usize, |total, start| {
        let start = usize::try_from(*start)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("starts must be non-negative"))?;
        if start > right_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "starts must be less than index length",
            ));
        }
        total
            .checked_add(right_len - start)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matches tape width overflow"))
    })
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// `index`, `counts`, and `matches` are trusted outputs of the
        /// conditional-join boundary and are expected to be non-negative.
        /// `matches == 0` excludes a candidate; non-zero values are treated
        /// as live without a second validation pass over the tape.
        ///
        /// The accumulator type `$acc` is `i64` for every dtype except
        /// `uint64`, which uses `u64` so values `>= 2**63` don't get
        /// sign-flipped by a forced `i64` cast (issue #90's bug class).
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let matches = matches.as_array();
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let end_: usize = index.len();
            if arr.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr cannot be empty",
                ));
            }
            if index.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "index cannot be empty",
                ));
            }
            if matches.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "matches cannot be empty",
                ));
            }
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width = expected_matches_width(starts, end_)?;
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            let min_start = starts
                .iter()
                .filter_map(|s| usize::try_from(*s).ok())
                .min()
                .unwrap_or(end_);
            let width = end_.saturating_sub(min_start);
            let dense = crate::aggs::should_use_dense_match_storage(end_, width);
            let mut touched = Vec::with_capacity(width);
            let mut n = 0_usize;
            if dense {
                let mut seen = vec![false; width];
                let mut totals = vec![1 as $acc; width];
                for (current, start, count, boolean) in
                    izip!(arr.iter(), starts.iter(), counts.iter(), booleans.iter())
                {
                    let Some(start_) = usize::try_from(*start).ok().filter(|s| *s <= end_) else {
                        continue;
                    };
                    let current_ = *current as $acc;
                    for item in start_..end_ {
                        if matches[n] != 0 {
                            let slot = item - min_start;
                            if !seen[slot] {
                                seen[slot] = true;
                                touched.push(slot);
                            }
                            if !*boolean && *count != 0 {
                                totals[slot] = totals[slot].wrapping_mul(current_);
                            }
                        }
                        n += 1;
                    }
                }
                let mut labels = Vec::with_capacity(touched.len());
                let mut values = Vec::with_capacity(touched.len());
                for slot in touched {
                    labels.push(index[min_start + slot]);
                    values.push(totals[slot]);
                }
                return Ok((labels.into_pyarray(py), values.into_pyarray(py)));
            }
            let mut totals: HashMap<usize, $acc> = HashMap::with_capacity(width);
            for (current, start, count, boolean) in
                izip!(arr.iter(), starts.iter(), counts.iter(), booleans.iter())
            {
                let Some(start_) = usize::try_from(*start).ok().filter(|s| *s <= end_) else {
                    continue;
                };
                let current_ = *current as $acc;
                for item in start_..end_ {
                    if matches[n] != 0 {
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            totals.entry(item - min_start)
                        {
                            touched.push(item - min_start);
                            entry.insert(1 as $acc);
                        }
                        if !*boolean && *count != 0 {
                            let total =
                                totals.get_mut(&(item - min_start)).expect("inserted above");
                            *total = total.wrapping_mul(current_);
                        }
                    }
                    n += 1;
                }
            }
            let mut labels = Vec::with_capacity(touched.len());
            let mut values = Vec::with_capacity(touched.len());
            for slot in touched {
                labels.push(index[min_start + slot]);
                values.push(totals[&slot]);
            }
            Ok((labels.into_pyarray(py), values.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see the macro's doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_start_match_int64, i64, i64);
compute_ints!(compute_prod_rev_start_match_int32, i32, i64);
compute_ints!(compute_prod_rev_start_match_int16, i16, i64);
compute_ints!(compute_prod_rev_start_match_int8, i8, i64);
compute_ints!(compute_prod_rev_start_match_uint64, u64, u64);
compute_ints!(compute_prod_rev_start_match_uint32, u32, i64);
compute_ints!(compute_prod_rev_start_match_uint16, u16, i64);
compute_ints!(compute_prod_rev_start_match_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// `index`, `counts`, and `matches` are trusted outputs of the
        /// conditional-join boundary and are expected to be non-negative.
        /// `matches == 0` excludes a candidate; non-zero values are treated
        /// as live without a second validation pass over the tape.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let matches = matches.as_array();
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let mut n: usize = 0;
            let end_: usize = index.len();
            if arr.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr cannot be empty",
                ));
            }
            if index.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "index cannot be empty",
                ));
            }
            if matches.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "matches cannot be empty",
                ));
            }
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width = expected_matches_width(starts, end_)?;
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            let min_start = starts
                .iter()
                .filter_map(|s| usize::try_from(*s).ok())
                .min()
                .unwrap_or(end_);
            let width = end_.saturating_sub(min_start);
            let dense = crate::aggs::should_use_dense_match_storage(end_, width);
            let mut touched = Vec::with_capacity(width);
            if dense {
                let mut seen = vec![false; width];
                let mut totals = vec![1.0_f64; width];
                for (current, start, count, boolean) in
                    izip!(arr.iter(), starts.iter(), counts.iter(), booleans.iter())
                {
                    let start_ = usize::try_from(*start).expect("validated above");
                    let current_ = *current as f64;
                    for item in start_..end_ {
                        if matches[n] != 0 {
                            let slot = item - min_start;
                            if !seen[slot] {
                                seen[slot] = true;
                                touched.push(slot);
                            }
                            if !*boolean && *count != 0 {
                                totals[slot] *= current_;
                            }
                        }
                        n += 1;
                    }
                }
                let mut labels = Vec::with_capacity(touched.len());
                let mut values = Vec::with_capacity(touched.len());
                for slot in touched {
                    labels.push(index[min_start + slot]);
                    values.push(totals[slot]);
                }
                return Ok((labels.into_pyarray(py), values.into_pyarray(py)));
            }
            let mut totals: HashMap<usize, f64> = HashMap::with_capacity(width);
            for (current, start, count, boolean) in
                izip!(arr.iter(), starts.iter(), counts.iter(), booleans.iter())
            {
                let start_ = usize::try_from(*start).expect("validated above");
                let current_ = *current as f64;
                for item in start_..end_ {
                    if matches[n] != 0 {
                        let slot = item - min_start;
                        if let std::collections::hash_map::Entry::Vacant(entry) = totals.entry(slot)
                        {
                            touched.push(slot);
                            entry.insert(1.);
                        }
                        if !*boolean && *count != 0 {
                            *totals.get_mut(&slot).expect("inserted above") *= current_;
                        }
                    }
                    n += 1;
                }
            }
            let mut labels = Vec::with_capacity(touched.len());
            let mut values = Vec::with_capacity(touched.len());
            for slot in touched {
                labels.push(index[min_start + slot]);
                values.push(totals[&slot]);
            }
            Ok((labels.into_pyarray(py), values.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_prod_rev_start_match_f64, f64);
compute_floats!(compute_prod_rev_start_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        compute_prod_rev_start_match_int64, compute_prod_rev_start_match_uint64,
        expected_matches_width,
    };
    use numpy::ndarray::array;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let value = (i64::MAX as u64) + 5;
            let arr = PyArray1::from_vec(py, vec![value]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let index = PyArray1::from_vec(py, vec![10_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false]);
            let (labels, values) = compute_prod_rev_start_match_uint64(
                py,
                arr.readonly(),
                starts.readonly(),
                counts.readonly(),
                index.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_slice().unwrap(), &[10]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[value]);
        });
    }

    #[test]
    fn computes_zero_width_start_at_index_length() {
        assert_eq!(expected_matches_width(array![3_i64].view(), 3).unwrap(), 0);
    }

    #[test]
    fn rejects_extra_tape_entries() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let arr = PyArray1::from_vec(py, vec![2_i64]);
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 0]);
            let booleans = PyArray1::from_vec(py, vec![false]);
            assert!(compute_prod_rev_start_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                counts.readonly(),
                index.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .is_err());
        });
    }

    #[test]
    fn integer_kernel_handles_duplicate_labels() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let arr = PyArray1::from_vec(py, vec![2_i64, 3]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let counts = PyArray1::from_vec(py, vec![1_i64, 1]);
            let index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let (labels, values) = compute_prod_rev_start_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                counts.readonly(),
                index.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_slice().unwrap(), &[10, 20]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[2, 6]);
        });
    }

    #[test]
    fn integer_kernel_wraps_product_overflow() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let arr = PyArray1::from_vec(py, vec![i64::MAX, 2]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let counts = PyArray1::from_vec(py, vec![1_i64, 1]);
            // pyjanitor supplies unique right-index labels; they need not be
            // positional, so this also exercises the ordinal/label split.
            let index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 0, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);
            let (_, values) = compute_prod_rev_start_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                counts.readonly(),
                index.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();
            assert_eq!(values.readonly().as_slice().unwrap(), &[i64::MAX, 2]);
        });
    }
}
