use itertools::izip;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ensure_equal_lengths, ensure_exact_tape_width, ensure_nonempty_matches};

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
        /// or contain gaps; the suffix ordinal `item` selects dense state, and
        /// `index[item]` is the output label.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            counts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let matches = matches.as_array();
            ensure_nonempty_matches(matches.len())?;
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let end_: usize = index.len();
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let (expected_matches_width, min_start) =
                starts
                    .iter()
                    .fold((0_usize, end_), |(width, min_start), start| {
                        let Some(start_) = usize::try_from(*start).ok().filter(|&s| s <= end_)
                        else {
                            return (width, min_start);
                        };
                        (width + end_ - start_, min_start.min(start_))
                    });
            ensure_exact_tape_width(expected_matches_width, matches.len())?;
            // ELI5: if every suffix starts at 900 in a 1,000-row right frame,
            // only 100 ordinal slots can be reached. Allocate that suffix
            // domain, not one slot per label in the whole frame.
            let width = end_ - min_start;
            let mut seen = vec![false; width];
            let mut touched = Vec::new();
            let mut rows = vec![-1_i64; width];
            let mut values = vec![<$type>::default(); width];
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            let mut n: usize = 0;
            for (posn, (current, start, count, boolean)) in zipped.enumerate() {
                let Some(start_) = usize::try_from(*start).ok().filter(|&s| s <= end_) else {
                    continue;
                };
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
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    if rows[slot] == -1 || *current < values[slot] {
                        values[slot] = *current;
                        rows[slot] = posn as i64;
                    }
                    n += 1;
                }
            }
            let indexers: Vec<i64> = touched
                .iter()
                .map(|&slot| index[slot + min_start])
                .collect();
            let result: Vec<i64> = touched.iter().map(|&slot| rows[slot]).collect();
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_min_rev_start_match_int64, i64);
compute!(compute_min_rev_start_match_int32, i32);
compute!(compute_min_rev_start_match_int16, i16);
compute!(compute_min_rev_start_match_int8, i8);
compute!(compute_min_rev_start_match_uint64, u64);
compute!(compute_min_rev_start_match_uint32, u32);
compute!(compute_min_rev_start_match_uint16, u16);
compute!(compute_min_rev_start_match_uint8, u8);
compute!(compute_min_rev_start_match_f64, f64);
compute!(compute_min_rev_start_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_min_rev_start_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compute_min_rev_start_match_int64;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    #[test]
    fn dense_suffix_slots_preserve_permuted_gapped_labels() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }

            // Suffixes are [start, index.len()). The labels are reordered and
            // gapped, so `item` must index state and `index[item]` must label it.
            let arr = PyArray1::from_vec(py, vec![5_i64, 7]);
            let starts = PyArray1::from_vec(py, vec![1_i64, 0]);
            let counts = PyArray1::from_vec(py, vec![1_i64, 2]);
            let index = PyArray1::from_vec(py, vec![42_i64, 7, 100]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 0, 1, 1, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false]);

            let (labels, rows) = compute_min_rev_start_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                counts.readonly(),
                index.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .unwrap();

            assert_eq!(labels.readonly().as_slice().unwrap(), &[7, 42, 100]);
            assert_eq!(rows.readonly().as_slice().unwrap(), &[0, 1, 1]);
        });
    }
}
