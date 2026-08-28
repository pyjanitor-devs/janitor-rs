use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches,
};
use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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
        /// `index` contains unique right-row identities. They may be
        /// reordered or contain gaps; the ordinal `item` selects dense state,
        /// and `index[item]` is the output label.
        ///
        /// The accumulator type `$acc` is `i64` for every dtype except
        /// `uint64`, which uses `u64` so values `>= 2**63` don't get
        /// sign-flipped by a forced `i64` cast (issue #90's bug class).
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            index: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let index = index.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let matches = matches.as_array();
            ensure_nonempty_matches(matches.len())?;
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let (expected_matches_width, max_end) =
                ends.iter()
                    .fold((0_usize, 0_usize), |(width, max_end), end| {
                        checked_range(0, *end, index.len())
                            .map(|(_, end_)| (width + end_, max_end.max(end_)))
                            .unwrap_or((width, max_end))
                    });
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            // Prefix rows can only reach ordinals below their largest `end`.
            // ELI5: if every prefix stops at item 8, eight slots are enough;
            // allocating one slot per right-row label would waste memory when
            // the right frame is huge but the candidate prefixes are narrow.
            let mut seen = vec![false; max_end];
            let mut touched = Vec::new();
            let mut totals = vec![0 as $acc; max_end];
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (current, end, count, boolean) in zipped {
                // ELI5 (the guard): `end` indexes into `index`, not `arr`.
                // Unlike the dual-bound `_starts_ends` shape, this single-
                // bound producer (`src/compare/comp_ends.rs`) has no
                // invalid-row concept of its own -- every `end` reaching
                // here is already guaranteed `0 <= end <= index.len()`
                // because `bin_search_lt_first`/`bin_search_gt_first` drop
                // zero-width rows before `ends` is ever built. `end == 0`
                // is valid and simply contributes no tape entries. This
                // `checked_range` is defense in depth against that
                // cross-module invariant breaking, not a condition the
                // real pyjanitor call path can trigger; see issue #40 for
                // the full trace and issue #41 for the tape-width check
                // above, which is what actually guards `matches[n]`.
                let Some((_, end_)) = checked_range(0, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as $acc;
                for item in 0..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    if !seen[item] {
                        seen[item] = true;
                        touched.push(item);
                    }
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    totals[item] = totals[item].wrapping_add(current_);
                    n += 1;
                }
            }
            let indexers: Vec<i64> = touched.iter().map(|&item| index[item]).collect();
            let result: Vec<$acc> = touched.iter().map(|&item| totals[item]).collect();
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see the macro's doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_sum_rev_end_match_int64, i64, i64);
compute_ints!(compute_sum_rev_end_match_int32, i32, i64);
compute_ints!(compute_sum_rev_end_match_int16, i16, i64);
compute_ints!(compute_sum_rev_end_match_int8, i8, i64);
compute_ints!(compute_sum_rev_end_match_uint64, u64, u64);
compute_ints!(compute_sum_rev_end_match_uint32, u32, i64);
compute_ints!(compute_sum_rev_end_match_uint16, u16, i64);
compute_ints!(compute_sum_rev_end_match_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// `matches` must be non-empty and must contain exactly one entry for
        /// every candidate position. pyjanitor supplies the per-row counts
        /// and binary mask from the same comparison stage. pyjanitor is
        /// responsible for ensuring each mask value is 0 or 1; Rust does not
        /// scan the tape to enforce that value-level contract. Normally
        /// `counts_array.sum() == matches.sum()`, while `matches.len()` is the
        /// full candidate-tape width.
        ///
        /// `index` contains unique right-row identities. They may be
        /// reordered or contain gaps; the ordinal `item` selects dense state,
        /// and `index[item]` is the output label.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            index: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let index = index.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("arr", arr.len(), "ends", ends.len())?;
            let matches = matches.as_array();
            ensure_nonempty_matches(matches.len())?;
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let (expected_matches_width, max_end) =
                ends.iter()
                    .fold((0_usize, 0_usize), |(width, max_end), end| {
                        checked_range(0, *end, index.len())
                            .map(|(_, end_)| (width + end_, max_end.max(end_)))
                            .unwrap_or((width, max_end))
                    });
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            // See the integer path above: prefix bounds make `max_end` a safe
            // and substantially smaller dense-state capacity when applicable.
            let mut seen = vec![false; max_end];
            let mut touched = Vec::new();
            let mut totals = vec![0.0_f64; max_end];
            let mut mapping = vec![0.0_f64; max_end];
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (current, end, count, boolean) in zipped {
                let Some((_, end_)) = checked_range(0, *end, index.len()) else {
                    continue;
                };
                let current_ = *current as f64;
                for item in 0..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    if !seen[item] {
                        seen[item] = true;
                        touched.push(item);
                    }
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    let difference = current_ - mapping[item];
                    let increment = totals[item] + difference;
                    mapping[item] = (increment - totals[item]) - difference;
                    // https://github.com/pandas-dev/pandas/blob/fee326fa8338d57230f742a6d8a2ce527f3479bc/pandas/_libs/groupby.pyx#L801
                    // adapted from pandas' cython code
                    // # GH#53606; GH#60303
                    // # If val is +/- infinity compensation is NaN
                    // # which would lead to results being NaN instead
                    // # of +/- infinity. We cannot use util.is_nan
                    // # because of no gil
                    if !mapping[item].is_finite() {
                        mapping[item] = 0.;
                    }
                    totals[item] = increment;
                    n += 1;
                }
            }
            let indexers: Vec<i64> = touched.iter().map(|&item| index[item]).collect();
            let result: Vec<f64> = touched.iter().map(|&item| totals[item]).collect();
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_sum_rev_end_match_f64, f64);
compute_floats!(compute_sum_rev_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_end_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_sum_rev_end_match_int64, compute_sum_rev_end_match_uint64};
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
            let index = PyArray1::from_vec(py, vec![10_i64]);
            let ends = PyArray1::from_vec(py, vec![1_i64]);
            let counts = PyArray1::from_vec(py, vec![1_i64]);
            let matches = PyArray1::from_vec(py, vec![1_i8]);
            let booleans = PyArray1::from_vec(py, vec![false]);
            let (labels, values) = compute_sum_rev_end_match_uint64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_slice().unwrap(), &[10]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[value]);
        });
    }

    #[test]
    fn dense_slots_preserve_permuted_gapped_labels_and_sparse_matches() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // `index` contains unique row identities, but they are not
            // positional: labels are reordered and have gaps.  The dense
            // state must therefore be indexed by `item`, then translated
            // back through `index[item]` only when producing output.
            //
            // ELI5: each `end` describes a prefix of the right table, while
            // `matches` is one long tape containing those prefixes back to
            // back.  The mask selects item 0 and item 2 in the first prefix,
            // then item 0 in the second prefix.
            let arr = PyArray1::from_vec(py, vec![5_i64, 7]);
            let index = PyArray1::from_vec(py, vec![42_i64, 7, 100]);
            let ends = PyArray1::from_vec(py, vec![3_i64, 2]);
            let counts = PyArray1::from_vec(py, vec![2_i64, 1]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 0, 1, 1, 0]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);

            let (labels, values) = compute_sum_rev_end_match_int64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();

            assert_eq!(labels.readonly().as_slice().unwrap(), &[42, 100]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[12, 5]);
        });
    }
}
