use itertools::izip;
use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches,
};
use std::collections::HashMap;

use crate::aggs::{
    ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_matches_core,
    should_use_dense_match_storage, WrapMul,
};

#[allow(clippy::too_many_arguments)]
fn prod_rev_start_end_match_int_core<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Vec<i64>, Vec<A>), String>
where
    T: Copy,
    A: Copy + WrapMul,
    F: FnMut(T) -> A,
{
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_nonempty_matches_core(matches.len())?;
    let mut expected = 0_usize;
    let mut min_start = index.len();
    let mut max_end = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((s, e)) = checked_range(*start, *end, index.len()) {
            expected += e - s;
            min_start = min_start.min(s);
            max_end = max_end.max(e);
        }
    }
    ensure_exact_tape_width_core(expected, matches.len())?;
    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut touched = Vec::with_capacity(width);
    let mut tape = 0_usize;
    if dense {
        let mut seen = vec![false; width];
        let mut totals = vec![A::ONE; width];
        for (current, start, end, count, boolean) in izip!(
            arr.iter(),
            starts.iter(),
            ends.iter(),
            counts.iter(),
            booleans.iter()
        ) {
            let Some((s, e)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            let current_ = convert(*current);
            for item in s..e {
                if matches[tape] != 0 {
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    if !*boolean && *count != 0 {
                        totals[slot] = totals[slot].wrap_mul(current_);
                    }
                }
                tape += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            values.push(totals[slot]);
        }
        return Ok((labels, values));
    }
    let mut totals: HashMap<usize, A> = HashMap::with_capacity(width);
    for (current, start, end, count, boolean) in izip!(
        arr.iter(),
        starts.iter(),
        ends.iter(),
        counts.iter(),
        booleans.iter()
    ) {
        let Some((s, e)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current_ = convert(*current);
        for item in s..e {
            if matches[tape] != 0 {
                let slot = item - min_start;
                if let std::collections::hash_map::Entry::Vacant(entry) = totals.entry(slot) {
                    touched.push(slot);
                    entry.insert(A::ONE);
                }
                if !*boolean && *count != 0 {
                    let total = totals.get_mut(&slot).expect("inserted above");
                    *total = total.wrap_mul(current_);
                }
            }
            tape += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        values.push(totals[&slot]);
    }
    Ok((labels, values))
}

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
        #[allow(clippy::too_many_arguments)]
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
            let (labels, values) = prod_rev_start_end_match_int_core(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                index.as_array(),
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
                |value| value as $acc,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            return Ok((labels.into_pyarray(py), values.into_pyarray(py)));
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
        #[allow(clippy::too_many_arguments)]
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
            let index = index.as_array();
            let counts = counts.as_array();
            let matches = matches.as_array();
            let booleans = booleans.as_array();
            let mut expected = 0_usize;
            let mut min_start = index.len();
            let mut max_end = 0_usize;
            for (start, end) in starts.iter().zip(ends.iter()) {
                if let Some((start_, end_)) = checked_range(*start, *end, index.len()) {
                    expected += end_ - start_;
                    min_start = min_start.min(start_);
                    max_end = max_end.max(end_);
                }
            }
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            ensure_nonempty_matches(matches.len())?;
            ensure_exact_tape_width(expected, matches.len())?;
            let width = max_end.saturating_sub(min_start);
            let dense = crate::aggs::should_use_dense_match_storage(index.len(), width);
            let mut touched = Vec::with_capacity(width);
            let mut tape = 0_usize;
            if dense {
                let mut seen = vec![false; width];
                let mut totals = vec![1.0_f64; width];
                for (current, start, end, count, boolean) in izip!(
                    arr.iter(),
                    starts.iter(),
                    ends.iter(),
                    counts.iter(),
                    booleans.iter()
                ) {
                    let Some((s, e)) = checked_range(*start, *end, index.len()) else {
                        continue;
                    };
                    let current_ = *current as f64;
                    for item in s..e {
                        if matches[tape] != 0 {
                            let slot = item - min_start;
                            if !seen[slot] {
                                seen[slot] = true;
                                touched.push(slot);
                            }
                            if !*boolean && *count != 0 {
                                totals[slot] *= current_;
                            }
                        }
                        tape += 1;
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
            for (current, start, end, count, boolean) in izip!(
                arr.iter(),
                starts.iter(),
                ends.iter(),
                counts.iter(),
                booleans.iter()
            ) {
                let Some((s, e)) = checked_range(*start, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as f64;
                for item in s..e {
                    if matches[tape] != 0 {
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
                    tape += 1;
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
