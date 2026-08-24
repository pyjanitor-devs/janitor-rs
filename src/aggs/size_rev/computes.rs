use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{
    checked_end, checked_index, checked_range, ensure_equal_lengths, ensure_tape_width,
};

type SizeRevResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

/// Pure-Rust reverse-size core for the "end" shape: for every row's
/// `0..end` span, count one hit per row position it touches in `index`,
/// skipping rows whose `end` doesn't name a valid range. Takes plain
/// `ArrayView1`s, not PyO3 types, so it can be tested and benchmarked
/// without a Python interpreter -- see `benches/kernels.rs`.
pub fn size_rev_end_core(
    ends: ArrayView1<i64>,
    index: ArrayView1<i64>,
    length: usize,
) -> (Array1<i64>, Array1<i64>) {
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let start_: usize = 0_usize;
    for end in ends.into_iter() {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            let pos = index[item] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
        }
    }
    slots.to_arrays(|value| *value)
}

#[pyfunction]
pub fn compute_size_rev_end<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let (indexers, result) = size_rev_end_core(ends.as_array(), index.as_array(), length as usize);
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_start<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>) {
    let starts = starts.as_array();
    let index = index.as_array();
    let length = length as usize;
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let end_: usize = index.len();
    for start in starts.into_iter() {
        let start_ = *start as usize;
        for item in start_..end_ {
            let pos = index[item] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
        }
    }
    let (indexers, result) = slots.to_arrays(|value| *value);
    (indexers.into_pyarray(py), result.into_pyarray(py))
}

#[pyfunction]
pub fn compute_size_rev_end_matches<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> SizeRevResult<'py> {
    let ends = ends.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = ends
        .iter()
        .filter_map(|e| checked_end(*e, index.len()))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let length = length as usize;
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let start_: usize = 0_usize;
    let mut n: usize = 0;
    for end in ends.into_iter() {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
            n += 1;
        }
    }
    let (indexers, result) = slots.to_arrays(|value| *value);
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_start_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let index = index.as_array();
    let matches = matches.as_array();
    let end_: usize = index.len();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .map(|s| end_.saturating_sub(*s as usize))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let length = length as usize;
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let mut n: usize = 0;
    for start in starts.into_iter() {
        let start_ = *start as usize;
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
            n += 1;
        }
    }
    let (indexers, result) = slots.to_arrays(|value| *value);
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_start_end_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let matches = matches.as_array();
    // ELI5: `matches[n]` advances once per candidate position, summed
    // across every row -- not comparable to any single array's length.
    // Total that width up front and check it against `matches.len()`
    // here, before the loop below ever indexes into the tape.
    let expected_matches_width: usize = starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(s, e)| checked_range(*s, *e, index.len()).map(|(s_, e_)| e_ - s_))
        .sum();
    ensure_tape_width(expected_matches_width, matches.len())?;
    let length = length as usize;
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let mut n: usize = 0;
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let pos = index[item] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
            n += 1;
        }
    }
    let (indexers, result) = slots.to_arrays(|value| *value);
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_start_end<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let length = length as usize;
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            let pos = index[item] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
        }
    }
    let (indexers, result) = slots.to_arrays(|value| *value);
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

#[pyfunction]
pub fn compute_size_rev_positions<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let positions = positions.as_array();
    let length = length as usize;
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let zipped = starts.into_iter().zip(ends);
    for (start, end) in zipped {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        for item in start_..end_ {
            let Some(indexer_) = checked_index(positions[item], index.len()) else {
                continue;
            };
            let pos = index[indexer_] as usize;
            let total = slots.touch(pos, 0);
            *total += 1;
        }
    }
    let (indexers, result) = slots.to_arrays(|value| *value);
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_size_rev_start, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_end, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_positions, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_start_matches, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_end_matches, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_start_end, m)?)?;
    m.add_function(wrap_pyfunction!(compute_size_rev_start_end_matches, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::*;

    #[test]
    fn size_rev_end_counts_touched_row_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [5, 3]; row0 (end=1) touches index[0..1] = {5},
            // row1 (end=2) touches index[0..2] = {5, 3} -- position 5 is
            // counted twice, position 3 once.
            let ends = PyArray1::from_vec(py, vec![1_i64, 2]);
            let index = PyArray1::from_vec(py, vec![5_i64, 3]);
            let (indexers, result) = compute_size_rev_end(py, ends.readonly(), index.readonly(), 6)
                .expect("valid inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![3, 5]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![1, 2]);
        });
    }

    #[test]
    fn size_rev_start_counts_touched_row_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [7, 2, 9]; row0 (start=1) touches index[1..3] =
            // {2, 9}, row1 (start=0) touches index[0..3] = {7, 2, 9}.
            let starts = PyArray1::from_vec(py, vec![1_i64, 0]);
            let index = PyArray1::from_vec(py, vec![7_i64, 2, 9]);
            let (indexers, result) =
                compute_size_rev_start(py, starts.readonly(), index.readonly(), 10);
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![2, 7, 9]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![2, 1, 2]);
        });
    }

    #[test]
    fn size_rev_end_matches_counts_only_matching_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [2, 5]; row0 (end=1) consumes 1 match-tape slot,
            // row1 (end=2) consumes 2 more -- tape width 3, all matching.
            let ends = PyArray1::from_vec(py, vec![1_i64, 2]);
            let index = PyArray1::from_vec(py, vec![2_i64, 5]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let (indexers, result) = compute_size_rev_end_matches(
                py,
                ends.readonly(),
                index.readonly(),
                matches.readonly(),
                6,
            )
            .expect("valid inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![2, 5]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![2, 1]);
        });
    }

    #[test]
    fn size_rev_start_matches_counts_only_matching_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [4, 7]; row0 (start=0) consumes tape width 2, row1
            // (start=1) consumes tape width 1 -- total tape width 3, all
            // matching.
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let index = PyArray1::from_vec(py, vec![4_i64, 7]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1]);
            let (indexers, result) = compute_size_rev_start_matches(
                py,
                starts.readonly(),
                index.readonly(),
                matches.readonly(),
                8,
            )
            .expect("valid inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![4, 7]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![1, 2]);
        });
    }

    #[test]
    fn size_rev_start_end_matches_counts_only_matching_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [1, 4, 9]; row0 (0..2) and row1 (1..3) each
            // contribute tape width 2 -- total tape width 4, all
            // matching.
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3]);
            let index = PyArray1::from_vec(py, vec![1_i64, 4, 9]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 1, 1, 1]);
            let (indexers, result) = compute_size_rev_start_end_matches(
                py,
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                matches.readonly(),
                10,
            )
            .expect("valid inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![1, 4, 9]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![1, 2, 1]);
        });
    }

    #[test]
    fn size_rev_start_end_counts_touched_row_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // index = [4, 1, 8]; row0 (0..2) touches index[0..2] =
            // {4, 1}, row1 (1..3) touches index[1..3] = {1, 8}.
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3]);
            let index = PyArray1::from_vec(py, vec![4_i64, 1, 8]);
            let (indexers, result) = compute_size_rev_start_end(
                py,
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                9,
            )
            .expect("valid inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![1, 4, 8]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![2, 1, 1]);
        });
    }

    #[test]
    fn size_rev_positions_counts_touched_row_positions_ascending() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // positions = [1, 0, 2], index = [6, 3, 9]. row0 (0..2) walks
            // positions[0..2] = {1, 0} -> index positions {3, 6}; row1
            // (1..3) walks positions[1..3] = {0, 2} -> index positions
            // {6, 9}.
            let starts = PyArray1::from_vec(py, vec![0_i64, 1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, 3]);
            let index = PyArray1::from_vec(py, vec![6_i64, 3, 9]);
            let positions = PyArray1::from_vec(py, vec![1_i64, 0, 2]);
            let (indexers, result) = compute_size_rev_positions(
                py,
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                positions.readonly(),
                10,
            )
            .expect("valid inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![3, 6, 9]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![1, 2, 1]);
        });
    }
}
