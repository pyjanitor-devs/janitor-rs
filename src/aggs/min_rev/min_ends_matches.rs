use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_range, ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches,
};

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// `matches` must be non-empty and must contain exactly one entry for
        /// every candidate position. pyjanitor supplies the per-row counts
        /// and binary mask from the same comparison stage. pyjanitor is
        /// responsible for ensuring each mask value is 0 or 1; Rust does not
        /// scan the tape to enforce that value-level contract. Normally
        /// `counts_array.sum() == matches.sum()`, while `matches.len()` is the
        /// full candidate-tape width.
        ///
        /// `index` contains unique right-row identities. They may be reordered
        /// or contain gaps; the ordinal `item` selects dense state, and
        /// `index[item]` is the output label.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            index: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
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
            // ELI5: a prefix ending at 8 can only touch slots 0 through 7,
            // so reserving state for the whole right frame would waste memory.
            // Labels are translated through `index` only when output is built.
            let mut seen = vec![false; max_end];
            let mut touched = Vec::new();
            let mut rows = vec![-1_i64; max_end];
            let mut values = vec![<$type>::default(); max_end];
            let mut n: usize = 0;
            let zipped = izip!(
                arr.into_iter(),
                ends.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            for (posn, (current, end, count, boolean)) in zipped.enumerate() {
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
                    if rows[item] == -1 || *current < values[item] {
                        values[item] = *current;
                        rows[item] = posn as i64;
                    }
                    n += 1;
                }
            }
            let indexers: Vec<i64> = touched.iter().map(|&item| index[item]).collect();
            let result: Vec<i64> = touched.iter().map(|&item| rows[item]).collect();
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_min_rev_end_match_int64, i64);
compute!(compute_min_rev_end_match_int32, i32);
compute!(compute_min_rev_end_match_int16, i16);
compute!(compute_min_rev_end_match_int8, i8);
compute!(compute_min_rev_end_match_uint64, u64);
compute!(compute_min_rev_end_match_uint32, u32);
compute!(compute_min_rev_end_match_uint16, u16);
compute!(compute_min_rev_end_match_uint8, u8);
compute!(compute_min_rev_end_match_f64, f64);
compute!(compute_min_rev_end_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_end_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compute_min_rev_end_match_int64;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    #[test]
    fn dense_slots_preserve_permuted_gapped_labels_and_first_seen_order() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // ELI5: prefixes share the same ordinal slots even though the
            // labels are shuffled and have gaps. The result translates those
            // slots back to labels only after the scan.
            let arr = PyArray1::from_vec(py, vec![5_i64, 7]);
            let index = PyArray1::from_vec(py, vec![42_i64, 7, 100]);
            let ends = PyArray1::from_vec(py, vec![3_i64, 2]);
            let counts = PyArray1::from_vec(py, vec![2_i64, 2]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 0, 1, 1, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);

            let (labels, rows) = compute_min_rev_end_match_int64(
                py,
                arr.readonly(),
                index.readonly(),
                ends.readonly(),
                counts.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();

            assert_eq!(labels.readonly().as_slice().unwrap(), &[42, 100, 7]);
            assert_eq!(rows.readonly().as_slice().unwrap(), &[0, 0, 1]);
        });
    }
}
