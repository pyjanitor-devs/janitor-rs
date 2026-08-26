use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::aggs::{ensure_equal_lengths, ensure_tape_width};

fn expected_matches_width(ends: ArrayView1<'_, i64>, right_len: usize) -> PyResult<usize> {
    if ends.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ends cannot be empty",
        ));
    }
    ends.iter().try_fold(0usize, |total, end| {
        let end = usize::try_from(*end)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("ends must be non-negative"))?;
        if end > right_len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ends must satisfy 0 <= end <= index length",
            ));
        }
        total
            .checked_add(end)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("matches tape width overflow"))
    })
}

macro_rules! compute {
    ($fname:ident, $type:ty) => {
        /// `index`, `counts`, and `matches` are trusted outputs of the
        /// conditional-join boundary and are expected to be non-negative.
        /// `matches == 0` excludes a candidate; non-zero values are treated
        /// as live without a second validation pass over the tape.
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
            let counts = counts.as_array();
            ensure_equal_lengths("arr", arr.len(), "counts", counts.len())?;
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            if arr.is_empty() || index.is_empty() || matches.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr, ends, counts, booleans, index, and matches cannot be empty",
                ));
            }
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width = expected_matches_width(ends, index.len())?;
            ensure_tape_width(expected_matches_width, matches.len())?;
            let capacity = ends.iter().copied().max().unwrap() as usize;
            let mut slots: HashMap<i64, usize> = HashMap::with_capacity(capacity);
            let mut labels = Vec::with_capacity(capacity);
            let mut rows = Vec::with_capacity(capacity);
            let mut values = Vec::with_capacity(capacity);
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
                // here is guaranteed `0 <= end <= index.len()`. `end == 0`
                // A zero-width row contributes no tape entries. An entirely
                // empty `matches` tape is rejected before this loop. This
                // `checked_range` remains defense in depth against malformed
                // negative or oversized bounds reaching the kernel, not a
                // real pyjanitor call path can trigger; see issue #40 for
                // the full trace and issue #41 for the tape-width check
                // above, which is what actually guards `matches[n]`.
                let end_ = *end as usize;
                for item in 0..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    let slot = if let Some(slot) = slots.get(&pos) {
                        *slot
                    } else {
                        let slot = values.len();
                        slots.insert(pos, slot);
                        labels.push(pos);
                        rows.push(-1_i64);
                        values.push(*current);
                        slot
                    };
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
            Ok((
                Array1::from_vec(labels).into_pyarray(py),
                Array1::from_vec(rows).into_pyarray(py),
            ))
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
    use super::expected_matches_width;
    use numpy::ndarray::array;

    #[test]
    fn computes_zero_width_prefix_and_rejects_invalid_ends() {
        assert_eq!(
            expected_matches_width(array![0_i64, 3].view(), 3).unwrap(),
            3
        );
        assert!(expected_matches_width(array![-1_i64].view(), 3).is_err());
        assert!(expected_matches_width(array![4_i64].view(), 3).is_err());
    }
}
