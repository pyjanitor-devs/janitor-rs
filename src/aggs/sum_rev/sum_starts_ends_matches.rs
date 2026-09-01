use itertools::izip;
use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_range;
use std::collections::{hash_map::Entry, HashMap};

use crate::aggs::{
    ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    should_use_dense_match_storage, WrapAdd,
};

#[allow(clippy::too_many_arguments)]
fn sum_rev_start_end_match_int_core<T, A, F>(
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
    A: Copy + WrapAdd,
    F: FnMut(T) -> A,
{
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_nonempty_core("matches", matches.len())?;
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
    ensure_exact_tape_width_core(expected, matches.len())?;
    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut touched = Vec::with_capacity(width);
    let mut tape = 0_usize;
    if dense {
        let mut seen = vec![false; width];
        let mut totals = vec![A::ZERO; width];
        for (current, start, end, count, boolean) in izip!(
            arr.iter(),
            starts.iter(),
            ends.iter(),
            counts.iter(),
            booleans.iter()
        ) {
            let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            let current_ = convert(*current);
            for item in start_..end_ {
                if matches[tape] != 0 {
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    if !*boolean && *count != 0 {
                        totals[slot] = totals[slot].wrap_add(current_);
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
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        let current_ = convert(*current);
        for item in start_..end_ {
            if matches[tape] != 0 {
                let slot = item - min_start;
                if let Entry::Vacant(entry) = totals.entry(slot) {
                    touched.push(slot);
                    entry.insert(A::ZERO);
                }
                if !*boolean && *count != 0 {
                    let total = totals.get_mut(&slot).expect("inserted above");
                    *total = total.wrap_add(current_);
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

#[allow(clippy::too_many_arguments)]
fn sum_rev_start_end_match_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    counts: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Vec<i64>, Vec<f64>), String>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "counts", counts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_nonempty_core("matches", matches.len())?;
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
        let mut states = vec![(0., 0.); width];
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
                        let state = &mut states[slot];
                        let d = current_ - state.1;
                        let inc = state.0 + d;
                        state.1 = (inc - state.0) - d;
                        if !state.1.is_finite() {
                            state.1 = 0.;
                        }
                        state.0 = inc;
                    }
                }
                tape += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            values.push(states[slot].0);
        }
        return Ok((labels, values));
    }
    let mut states: HashMap<usize, (f64, f64)> = HashMap::with_capacity(width);
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
                if let Entry::Vacant(entry) = states.entry(slot) {
                    touched.push(slot);
                    entry.insert((0., 0.));
                }
                if !*boolean && *count != 0 {
                    let state = states.get_mut(&slot).expect("inserted above");
                    let d = current_ - state.1;
                    let inc = state.0 + d;
                    state.1 = (inc - state.0) - d;
                    if !state.1.is_finite() {
                        state.1 = 0.;
                    }
                    state.0 = inc;
                }
            }
            tape += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        values.push(states[&slot].0);
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
            let (labels, values) = sum_rev_start_end_match_int_core(
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
compute_ints!(compute_sum_rev_start_end_match_int64, i64, i64);
compute_ints!(compute_sum_rev_start_end_match_int32, i32, i64);
compute_ints!(compute_sum_rev_start_end_match_int16, i16, i64);
compute_ints!(compute_sum_rev_start_end_match_int8, i8, i64);
compute_ints!(compute_sum_rev_start_end_match_uint64, u64, u64);
compute_ints!(compute_sum_rev_start_end_match_uint32, u32, i64);
compute_ints!(compute_sum_rev_start_end_match_uint16, u16, i64);
compute_ints!(compute_sum_rev_start_end_match_uint8, u8, i64);

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
            let (labels, values) = sum_rev_start_end_match_float_core(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                index.as_array(),
                counts.as_array(),
                matches.as_array(),
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            return Ok((labels.into_pyarray(py), values.into_pyarray(py)));
        }
    };
}

compute_floats!(compute_sum_rev_start_end_match_f64, f64);
compute_floats!(compute_sum_rev_start_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_start_end_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compute_sum_rev_start_end_match_uint64;
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
            let index = PyArray1::from_vec(py, vec![10_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false]);
            let (labels, values) = compute_sum_rev_start_end_match_uint64(
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
            assert_eq!(labels.readonly().as_slice().unwrap(), &[10]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[value]);
        });
    }
}
