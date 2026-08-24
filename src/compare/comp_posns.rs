// comparisons based on positions
/// handles comparisions where nulls do not exist
use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::op::CompareOp;

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left = left.as_array();
            let right = right.as_array();
            let starts = starts.as_array();
            let positions = positions.as_array();
            let ends = ends.as_array();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let positions_len = positions.len();
            let right_len = right.len();
            let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter(),);
            for (position, (left_val, start, end)) in zipped.enumerate() {
                // `result` is presized to `positions.len()` (not derived
                // from this row's width), so an invalid row can be
                // silently skipped -- same convention as comp.rs/
                // comp_first.rs -- as long as `start`/`end` index into
                // `positions`, not `right` (a different array, and this
                // file's own outer range indexes `positions[nn]`, not
                // `right[nn]`, unlike every sibling file fixed so far).
                // Compares `positions_len as i64` against `end` rather
                // than casting `end` down to `usize` first -- on a
                // 32-bit target a genuinely oversized `end` would
                // truncate to a small value before this check ever saw
                // it, silently passing validation.
                if *start < 0 || *start >= *end || *end > positions_len as i64 {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let mut indexer = positions[nn];
                    // `-1` is the crate's existing "no match" sentinel for
                    // an individual position; broadened here (issue #56)
                    // to reject *any* out-of-bounds `indexer` the same
                    // way, not just the literal `-1` -- a `positions`
                    // entry with some other negative or oversized value
                    // used to survive unchanged into `right[indexer_]`,
                    // out of bounds. Mirrors `comp_no_range.rs`'s
                    // `checked_index` in spirit, as a plain comparison
                    // (this loop runs once per candidate position, the
                    // same hot-loop frequency #55 found checked_range
                    // costly at) -- and, like the check above, compares
                    // `right_len as i64` against `indexer` rather than
                    // casting `indexer` down to `usize` first, so a
                    // genuinely oversized indexer can't truncate to an
                    // in-range value and silently select the wrong row on
                    // a 32-bit target.
                    if indexer < 0 || indexer >= right_len as i64 {
                        result[n] = -1;
                        n += 1;
                        continue;
                    }
                    let indexer_ = indexer as usize;
                    let right_val = right[indexer_];
                    let compare = op.apply(left_val, &right_val);
                    counter += compare as i64;
                    total += compare as i64;
                    indexer = if compare { indexer } else { -1 };
                    result[n] = indexer;
                    n += 1;
                }
                counts_array[position] = counter;
            }
            Ok((
                result.into_pyarray(py),
                counts_array.into_pyarray(py),
                total,
            ))
        }
    };
}

generic_compare!(compare_posns_int64, i64);
generic_compare!(compare_posns_int32, i32);
generic_compare!(compare_posns_int16, i16);
generic_compare!(compare_posns_int8, i8);
generic_compare!(compare_posns_uint64, u64);
generic_compare!(compare_posns_uint32, u32);
generic_compare!(compare_posns_uint16, u16);
generic_compare!(compare_posns_uint8, u8);
generic_compare!(compare_posns_f64, f64);
generic_compare!(compare_posns_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_posns_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    type CompareResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64)>;

    fn run(py: Python<'_>, start: i64, end: i64, indexer: i64) -> CompareResult<'_> {
        let left = PyArray1::from_vec(py, vec![2_i64]);
        let right = PyArray1::from_vec(py, vec![1_i64]);
        let starts = PyArray1::from_vec(py, vec![start]);
        let positions = PyArray1::from_vec(py, vec![indexer]);
        let ends = PyArray1::from_vec(py, vec![end]);
        compare_posns_int64(
            py,
            left.readonly(),
            right.readonly(),
            starts.readonly(),
            positions.readonly(),
            ends.readonly(),
            0, // CompareOp::Gt
        )
    }

    #[test]
    fn out_of_bounds_indexer_is_treated_as_no_match_not_a_panic() {
        // right has length 1; positions=[5] used to index right[5] once
        // the row's start..end range reached it.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 1, 5).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), vec![-1_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn negative_non_sentinel_indexer_is_treated_as_no_match_not_a_panic() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 1, -5).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), vec![-1_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn end_beyond_positions_len_contributes_nothing_not_a_panic() {
        // positions has length 1; end=2 walks start_..end_ = 0..2 and
        // indexes positions[1], out of bounds.
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 2, 0).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn valid_indexer_compares_normally() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 1, 0).expect("must not panic");
            // left=[2], right=[1], op=Gt: 2 > 1 is true, indexer stays 0.
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![1_i64]);
            assert_eq!(total, 1);
        });
    }
}
