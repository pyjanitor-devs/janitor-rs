use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches,
};

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// `matches` must be non-empty and must contain exactly one entry for
        /// every candidate position. pyjanitor supplies the per-row counts
        /// and binary mask from the same comparison stage. pyjanitor is
        /// responsible for ensuring each mask value is 0 or 1; Rust does not
        /// scan the tape to enforce that value-level contract. Normally
        /// `counts_array.sum() == matches.sum()`, while `matches.len()` is the
        /// full candidate-tape width.
        ///
        /// The accumulator type `$acc` is `i64` for every dtype except
        /// `uint64`, which uses `u64` so values `>= 2**63` don't get
        /// sign-flipped by a forced `i64` cast (issue #90's bug class).
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let matches = matches.as_array();
            ensure_nonempty_matches(matches.len())?;
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width: usize = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| checked_range(*s, *e, index.len()).map(|(s_, e_)| e_ - s_))
                .sum();
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            // The fold below finds the smallest enclosing valid slice.
            // ELI5: it stretches one ruler just enough to cover every range.
            // Each right-side ordinal in the smallest enclosing range
            // gets a product bucket. `seen` keeps matched labels visible even
            // when null or zero-count rows contribute the identity.
            let (min_start, max_end) = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| checked_range(*s, *e, index.len()))
                .fold((usize::MAX, 0), |(min_start, max_end), (start, end)| {
                    (min_start.min(start), max_end.max(end))
                });
            let width = max_end.saturating_sub(min_start);
            let mut seen = vec![false; width];
            let mut touched = Vec::new();
            let mut products = vec![1 as $acc; width];
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter(),
            );
            let mut n: usize = 0;
            for (current, start, end, count, boolean) in zipped {
                let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as $acc;
                for item in start_..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    if (*boolean) || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    // ELI5: integer products are allowed to wrap like NumPy
                    // integers; wrapping_mul keeps debug and release builds
                    // on the same path instead of panicking on overflow.
                    products[slot] = products[slot].wrapping_mul(current_);
                    n += 1;
                }
            }
            let indexers = Array1::from_iter(touched.iter().map(|&slot| index[min_start + slot]));
            let result = Array1::from_iter(touched.iter().map(|&slot| products[slot]));
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see the macro's doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_start_end_match_int64, i64, i64);
compute_ints!(compute_prod_rev_start_end_match_int32, i32, i64);
compute_ints!(compute_prod_rev_start_end_match_int16, i16, i64);
compute_ints!(compute_prod_rev_start_end_match_int8, i8, i64);
compute_ints!(compute_prod_rev_start_end_match_uint64, u64, u64);
compute_ints!(compute_prod_rev_start_end_match_uint32, u32, i64);
compute_ints!(compute_prod_rev_start_end_match_uint16, u16, i64);
compute_ints!(compute_prod_rev_start_end_match_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// `matches` must be non-empty and must contain exactly one entry for
        /// every candidate position. pyjanitor supplies the per-row counts
        /// and binary mask from the same comparison stage. pyjanitor is
        /// responsible for ensuring each mask value is 0 or 1; Rust does not
        /// scan the tape to enforce that value-level contract. Normally
        /// `counts_array.sum() == matches.sum()`, while `matches.len()` is the
        /// full candidate-tape width.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let matches = matches.as_array();
            ensure_nonempty_matches(matches.len())?;
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width: usize = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| checked_range(*s, *e, index.len()).map(|(s_, e_)| e_ - s_))
                .sum();
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            // The float path uses the same enclosing-slice calculation as the
            // integer path; only the product storage differs.
            let (min_start, max_end) = starts
                .iter()
                .zip(ends.iter())
                .filter_map(|(s, e)| checked_range(*s, *e, index.len()))
                .fold((usize::MAX, 0), |(min_start, max_end), (start, end)| {
                    (min_start.min(start), max_end.max(end))
                });
            let width = max_end.saturating_sub(min_start);
            let mut seen = vec![false; width];
            let mut touched = Vec::new();
            let mut products = vec![1.; width];
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            let mut n: usize = 0;
            for (current, start, end, count, boolean) in zipped {
                let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as f64;
                for item in start_..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    if (*boolean) || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    products[slot] *= current_;
                    n += 1;
                }
            }
            let indexers = Array1::from_iter(touched.iter().map(|&slot| index[min_start + slot]));
            let result = Array1::from_iter(touched.iter().map(|&slot| products[slot]));
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_prod_rev_start_end_match_f64, f64);
compute_floats!(compute_prod_rev_start_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        compute_prod_rev_start_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compute_prod_rev_start_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compute_prod_rev_start_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_prod_rev_start_end_match_int64, compute_prod_rev_start_end_match_uint64};
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
            let ends = PyArray1::from_vec(py, vec![1_i64]);
            let index = PyArray1::from_vec(py, vec![42_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false]);
            let (labels, values) = compute_prod_rev_start_end_match_uint64(
                py,
                arr.readonly(),
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_slice().unwrap(), &[42]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[value]);
        });
    }

    #[test]
    fn integer_kernel_wraps_product_overflow() {
        Python::initialize();
        Python::attach(|py| {
            let arr = PyArray1::from_vec(py, vec![i64::MAX, 2]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let ends = PyArray1::from_vec(py, vec![1_i64, 1]);
            let index = PyArray1::from_vec(py, vec![42_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64, 1]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);

            let (labels, values) = compute_prod_rev_start_end_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect("valid integer product input");

            assert_eq!(labels.readonly().as_slice().unwrap(), &[42]);
            assert_eq!(
                values.readonly().as_slice().unwrap(),
                &[i64::MAX.wrapping_mul(2)]
            );
        });
    }
}
