use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::{hash_map::Entry, HashMap};

use crate::aggs::{ensure_equal_lengths, ensure_tape_width};

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
            // ELI5: `matches[n]` advances once per candidate position, summed
            // across every row -- not comparable to any single array's length.
            // Total that width up front and check it against `matches.len()`
            // here, before the loop below ever indexes into the tape.
            let expected_matches_width = expected_matches_width(starts, end_)?;
            ensure_tape_width(expected_matches_width, matches.len())?;
            let mut slots: HashMap<i64, usize> = HashMap::with_capacity(end_);
            let mut labels = Vec::new();
            let mut rows = Vec::new();
            let mut values = Vec::new();
            let zipped = izip!(
                arr.into_iter(),
                starts.into_iter(),
                counts.into_iter(),
                booleans.into_iter()
            );
            let mut n: usize = 0;
            for (posn, (current, start, count, boolean)) in zipped.enumerate() {
                let start_ = *start as usize;
                for item in start_..end_ {
                    if (matches[n] == 0) {
                        n += 1;
                        continue;
                    }
                    let pos = index[item];
                    // ELI5: ask the map for the label's slot once. `get`
                    // followed by `insert` would hash a first-seen label
                    // twice in this hot loop.
                    let slot = match slots.entry(pos) {
                        Entry::Occupied(entry) => *entry.get(),
                        Entry::Vacant(entry) => {
                            // ELI5: a label may be seen only through null or
                            // ignored values first, so reserve its slot now;
                            // `rows == -1` below records that no usable max
                            // has been found yet.
                            let slot = values.len();
                            entry.insert(slot);
                            labels.push(pos);
                            rows.push(-1_i64);
                            values.push(*current);
                            slot
                        }
                    };
                    if *boolean || (*count == 0) {
                        n += 1;
                        continue;
                    }
                    if rows[slot] == -1 || *current > values[slot] {
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

compute!(compute_max_rev_start_match_int64, i64);
compute!(compute_max_rev_start_match_int32, i32);
compute!(compute_max_rev_start_match_int16, i16);
compute!(compute_max_rev_start_match_int8, i8);
compute!(compute_max_rev_start_match_uint64, u64);
compute!(compute_max_rev_start_match_uint32, u32);
compute!(compute_max_rev_start_match_uint16, u16);
compute!(compute_max_rev_start_match_uint8, u8);
compute!(compute_max_rev_start_match_f64, f64);
compute!(compute_max_rev_start_match_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_start_match_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_max_rev_start_match_int64, expected_matches_width};
    use numpy::ndarray::array;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    #[test]
    fn tape_width_accepts_valid_starts() {
        assert_eq!(
            expected_matches_width(array![0_i64, 1].view(), 3).unwrap(),
            5
        );
    }

    #[test]
    fn tape_width_rejects_negative_and_one_past_starts() {
        assert!(expected_matches_width(array![-1_i64].view(), 3).is_err());
        assert_eq!(expected_matches_width(array![3_i64].view(), 3).unwrap(), 0);
    }

    #[test]
    fn tape_width_rejects_empty_starts() {
        assert!(expected_matches_width(array![].view(), 3).is_err());
    }

    #[test]
    fn integer_kernel_keeps_first_seen_labels_and_skips_null_values() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let arr = PyArray1::from_vec(py, vec![2_i64, 3, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let counts = PyArray1::from_vec(py, vec![1_i64, 1, 0]);
            let index = PyArray1::from_vec(py, vec![10_i64, 20, 10]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 0, 1, 1, 0, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false, false]);

            let (labels, values) = compute_max_rev_start_match_int64(
                py,
                arr.readonly(),
                starts.readonly(),
                counts.readonly(),
                index.readonly(),
                matches.readonly(),
                booleans.readonly(),
            )
            .expect("valid starts+matches input");

            assert_eq!(labels.readonly().as_slice().unwrap(), &[10, 20]);
            assert_eq!(values.readonly().as_slice().unwrap(), &[0, 1]);
        });
    }
}
