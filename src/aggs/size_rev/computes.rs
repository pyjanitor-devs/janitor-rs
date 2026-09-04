//! Reverse size aggregations.
//!
//! These kernels count how many rows or surviving match-tape candidates cover
//! each right-side position. State is addressed by ordinal position; `index`
//! supplies the original labels emitted in the result.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use std::collections::{hash_map::Entry, HashMap};

use crate::aggs::{
    checked_end, checked_index, checked_range, ends_domain, ensure_equal_lengths,
    ensure_equal_lengths_core, ensure_exact_tape_width_core, ensure_nonempty_core,
    should_use_dense_match_storage, starts_domain,
};

/// Common Python-facing result: output labels followed by integer counts.
type SizeRevResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

/// Count how many reverse-end rows cover each compact prefix position.
///
/// ELI5: each `end` says “this row covers everything before here.” Instead of
/// walking that whole prefix for every row, the implementation activates rows
/// once during a right-to-left sweep and carries the running count.
///
/// # Arguments
///
/// * `ends` - Exclusive prefix end for each row. Invalid or zero ends do not
///   contribute to the result.
/// * `index` - Right-side labels in ordinal position order.
///
/// # Returns
///
/// The emitted labels and one count per compact prefix position.
pub fn size_rev_ends_core(
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    ensure_nonempty_core("index", index.len()).map_err(|_| "index cannot be empty")?;
    let max_end = ends_domain(ends, index.len())?;
    // ELI5: a prefix row is active to the left of its end. Count how many
    // rows end at each boundary, then sweep from right to left and carry the
    // active-row count across the output. `end == 0` is an empty prefix and
    // naturally has no activation bucket visited by the output loop. Counts
    // are enough here, so memory depends on the compact output width—not the
    // number of rows in the batch.
    let mut result = vec![0_i64; max_end];
    for end in ends {
        if *end > 0 {
            result[*end as usize - 1] += 1;
        }
    }
    let mut running = 0_i64;
    for value in result.iter_mut().rev() {
        running += *value;
        *value = running;
    }
    let labels = Array1::from_iter(index.iter().take(max_end).copied());
    Ok((labels, Array1::from_vec(result)))
}

/// Count how many reverse-start rows cover each compact suffix position.
///
/// ELI5: each `start` says “this row covers everything from here onward.” The
/// implementation groups starts by boundary and sweeps those boundaries once,
/// so wide suffixes do not cause repeated work.
///
/// # Arguments
///
/// * `starts` - Inclusive suffix start for each row. Invalid starts and the
///   one-past-end sentinel do not contribute to the result.
/// * `index` - Right-side labels in ordinal position order.
///
/// # Returns
///
/// The emitted labels and one count per compact suffix position.
pub fn size_rev_starts_core(
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Array1<i64>, Array1<i64>), &'static str> {
    ensure_nonempty_core("index", index.len()).map_err(|_| "index cannot be empty")?;
    let (min_start, width) = starts_domain(starts, index.len())?;
    // ELI5: a suffix row is active from its start onward. The shared reducer
    // counts rows at each start boundary, then carries the active-row count
    // forward. The valid `start == index.len()` bucket has no emitted slot and
    // is naturally outside the compact output domain.
    let mut result = vec![0_i64; width];
    for start in starts {
        if *start < index.len() as i64 {
            result[*start as usize - min_start] += 1;
        }
    }
    let mut running = 0_i64;
    for value in &mut result {
        running += *value;
        *value = running;
    }
    let labels = Array1::from_iter(index.iter().skip(min_start).copied());
    Ok((labels, Array1::from_vec(result)))
}

/// Count rows covered by each compact right-side position for reverse prefix
/// ranges. Empty prefixes contribute no counts; `index` labels are emitted in
/// positional order.
///
/// # Arguments
///
/// * `ends` - Exclusive prefix end for each row.
/// * `index` - Right-side labels in ordinal position order.
#[pyfunction]
pub fn compute_size_rev_end<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    let (labels, counts) = size_rev_ends_core(ends.as_array(), index.as_array())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((labels.into_pyarray(py), counts.into_pyarray(py)))
}

/// Count rows covered by each compact right-side position for reverse suffix
/// ranges. Empty suffixes are omitted from the compact output.
///
/// # Arguments
///
/// * `starts` - Inclusive suffix start for each row.
/// * `index` - Right-side labels in ordinal position order.
#[pyfunction]
pub fn compute_size_rev_start<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    let (labels, counts) = size_rev_starts_core(starts.as_array(), index.as_array())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((labels.into_pyarray(py), counts.into_pyarray(py)))
}

/// Count surviving candidates for each right-side position in reverse prefix
/// ranges using a flat match tape.
///
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position in the valid ranges. Its length is the flattened tape width.
///
/// # Arguments
///
/// * `ends` - Exclusive prefix end for each row.
/// * `index` - Right-side labels in ordinal position order.
/// * `matches` - Flat per-candidate match mask.
///
/// # Returns
///
/// The labels with at least one surviving candidate and their counts.
pub fn size_rev_end_match_core(
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("matches", matches.len())?;

    let mut expected_matches_width = 0_usize;
    let mut max_end = 0_usize;
    for end in ends.iter() {
        if let Some(end_) = checked_end(*end, index.len()) {
            expected_matches_width += end_;
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())?;

    let dense = should_use_dense_match_storage(index.len(), max_end);
    let mut touched = if dense {
        Vec::with_capacity(max_end)
    } else {
        Vec::new()
    };
    let mut tape = 0_usize;

    if dense {
        let mut seen = vec![false; max_end];
        let mut counts = vec![0_i64; max_end];
        for end in ends.iter() {
            let Some(end_) = checked_end(*end, index.len()) else {
                continue;
            };
            for item in 0..end_ {
                if matches[tape] != 0 {
                    if !seen[item] {
                        seen[item] = true;
                        touched.push(item);
                    }
                    counts[item] += 1;
                }
                tape += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for item in touched {
            labels.push(index[item]);
            values.push(counts[item]);
        }
        return Ok((labels, values));
    }

    let mut counts = HashMap::<usize, i64>::new();
    for end in ends.iter() {
        let Some(end_) = checked_end(*end, index.len()) else {
            continue;
        };
        for item in 0..end_ {
            if matches[tape] != 0 {
                let count = counts.entry(item).or_insert_with(|| {
                    touched.push(item);
                    0
                });
                *count += 1;
            }
            tape += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for item in touched {
        labels.push(index[item]);
        values.push(counts[&item]);
    }
    Ok((labels, values))
}

/// # Returns
///
/// The labels with at least one surviving candidate and their counts.
#[pyfunction]
pub fn compute_size_rev_end_matches<'py>(
    py: Python<'py>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
) -> SizeRevResult<'py> {
    let (labels, counts) =
        size_rev_end_match_core(ends.as_array(), index.as_array(), matches.as_array())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((labels.into_pyarray(py), counts.into_pyarray(py)))
}

/// Count surviving candidates for each right-side position in reverse suffix
/// ranges using a flat match tape.
///
/// `matches` must be non-empty and contain exactly one entry per candidate
/// position in the valid ranges. Its length is the flattened tape width.
pub fn size_rev_start_match_core(
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("matches", matches.len())?;

    let mut expected_matches_width = 0_usize;
    let mut min_start = index.len();
    for start in starts.iter() {
        if let Some((start_, end_)) = checked_range(*start, index.len() as i64, index.len()) {
            expected_matches_width += end_ - start_;
            min_start = min_start.min(start_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())?;

    let width = index.len().saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut touched = if dense {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };
    let mut tape = 0_usize;

    if dense {
        let mut seen = vec![false; width];
        let mut counts = vec![0_i64; width];
        for start in starts.iter() {
            let Some((start_, end_)) = checked_range(*start, index.len() as i64, index.len())
            else {
                continue;
            };
            for item in start_..end_ {
                if matches[tape] != 0 {
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    counts[slot] += 1;
                }
                tape += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            values.push(counts[slot]);
        }
        return Ok((labels, values));
    }

    let mut counts = HashMap::<usize, i64>::new();
    for start in starts.iter() {
        let Some((start_, end_)) = checked_range(*start, index.len() as i64, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[tape] != 0 {
                let slot = item - min_start;
                let count = counts.entry(slot).or_insert_with(|| {
                    touched.push(slot);
                    0
                });
                *count += 1;
            }
            tape += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        values.push(counts[&slot]);
    }
    Ok((labels, values))
}

/// Count surviving candidates for each right-side position in reverse suffix
/// ranges using a flat match tape.
///
/// # Arguments
///
/// * `starts` - Inclusive suffix start for each row.
/// * `index` - Right-side labels in ordinal position order.
/// * `matches` - Flat per-candidate match mask with the exact tape width.
///
/// # Returns
///
/// The labels with at least one surviving candidate and their counts.
#[pyfunction]
pub fn compute_size_rev_start_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
) -> SizeRevResult<'py> {
    let (labels, counts) =
        size_rev_start_match_core(starts.as_array(), index.as_array(), matches.as_array())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((labels.into_pyarray(py), counts.into_pyarray(py)))
}

/// Count surviving candidates for each right-side position in reverse
/// interval ranges using a flat match tape.
pub fn size_rev_start_end_match_core(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    matches: ArrayView1<'_, i8>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("matches", matches.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;

    let mut expected_matches_width = 0_usize;
    let mut min_start = index.len();
    let mut max_end = 0_usize;
    for (start, end) in starts.iter().zip(ends.iter()) {
        if let Some((start_, end_)) = checked_range(*start, *end, index.len()) {
            expected_matches_width += end_ - start_;
            min_start = min_start.min(start_);
            max_end = max_end.max(end_);
        }
    }
    ensure_exact_tape_width_core(expected_matches_width, matches.len())?;

    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), width);
    let mut touched = if dense {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };
    let mut tape = 0_usize;

    if dense {
        let mut seen = vec![false; width];
        let mut counts = vec![0_i64; width];
        for (start, end) in starts.iter().zip(ends.iter()) {
            let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
                continue;
            };
            for item in start_..end_ {
                if matches[tape] != 0 {
                    let slot = item - min_start;
                    if !seen[slot] {
                        seen[slot] = true;
                        touched.push(slot);
                    }
                    counts[slot] += 1;
                }
                tape += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut values = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            values.push(counts[slot]);
        }
        return Ok((labels, values));
    }

    let mut counts = HashMap::<usize, i64>::new();
    for (start, end) in starts.iter().zip(ends.iter()) {
        let Some((start_, end_)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start_..end_ {
            if matches[tape] != 0 {
                let slot = item - min_start;
                let count = counts.entry(slot).or_insert_with(|| {
                    touched.push(slot);
                    0
                });
                *count += 1;
            }
            tape += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut values = Vec::with_capacity(touched.len());
    for slot in touched {
        labels.push(index[min_start + slot]);
        values.push(counts[&slot]);
    }
    Ok((labels, values))
}

/// `matches` must be non-empty and contain exactly one entry per candidate
/// position in the valid ranges. Its length is the flattened tape width.
///
/// # Arguments
///
/// * `starts` - Inclusive interval start for each row.
/// * `ends` - Exclusive interval end for each row.
/// * `index` - Right-side labels in ordinal position order.
/// * `matches` - Flat per-candidate match mask.
///
/// # Returns
///
/// The labels with at least one surviving candidate and their counts.
#[pyfunction]
pub fn compute_size_rev_start_end_matches<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    matches: PyReadonlyArray1<'py, i8>,
) -> SizeRevResult<'py> {
    let (labels, counts) = size_rev_start_end_match_core(
        starts.as_array(),
        ends.as_array(),
        index.as_array(),
        matches.as_array(),
    )
    .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((labels.into_pyarray(py), counts.into_pyarray(py)))
}

/// Count covered right-row positions for each distinct right-row identity.
///
/// janitor-rs is primarily called by pyjanitor. Its conditional-join path
/// resets the right DataFrame index to unique row labels before sorting or
/// filtering, so labels can be reordered or gapped but are not duplicated.
/// `item`, the ordinal position in `index`, is the state slot; `index[item]`
/// is the output label. `starts` and `ends` describe half-open ranges
/// `[start, end)`. Invalid or zero-width ranges are skipped by `checked_range`;
/// empty `starts`, `ends`, or `index` inputs are rejected.
///
/// # Preconditions
///
/// `index` must contain unique labels in positional order. pyjanitor provides
/// this by normalizing the right side to `range(len(right))`; direct callers
/// must provide it themselves. Duplicate labels are unsupported and are not
/// merged by the positional accumulator.
///
/// # Arguments
///
/// * `starts` - Inclusive start of each half-open interval.
/// * `ends` - Exclusive end of each half-open interval.
/// * `index` - Right-side labels in ordinal position order.
///
/// # Returns
///
/// The touched labels and the number of covering intervals for each label.
///
/// ELI5: each right-row position gets a drawer. Every range adds one to each
/// covered drawer, then we print the row identity stored in `index[item]`.
pub fn size_rev_start_end_core(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;

    let mut min_start = index.len();
    let mut max_end = 0_usize;
    let mut total_width = 0_usize;
    for (&start, &end) in starts.iter().zip(ends.iter()) {
        if let Some((start, end)) = checked_range(start, end, index.len()) {
            min_start = min_start.min(start);
            max_end = max_end.max(end);
            total_width = total_width.saturating_add(end - start);
        }
    }
    let width = max_end.saturating_sub(min_start);
    let dense = should_use_dense_match_storage(index.len(), total_width);
    let mut touched = if dense {
        Vec::with_capacity(width)
    } else {
        Vec::new()
    };

    if dense {
        let mut seen = vec![false; width];
        let mut counts = vec![0_i64; width];
        for (&start, &end) in starts.iter().zip(ends.iter()) {
            let Some((start, end)) = checked_range(start, end, index.len()) else {
                continue;
            };
            for item in start..end {
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                counts[slot] += 1;
            }
        }
        let mut labels = Vec::with_capacity(touched.len());
        let mut result = Vec::with_capacity(touched.len());
        for slot in touched {
            labels.push(index[min_start + slot]);
            result.push(counts[slot]);
        }
        return Ok((labels, result));
    }

    let mut counts = HashMap::<usize, i64>::new();
    for (&start, &end) in starts.iter().zip(ends.iter()) {
        let Some((start, end)) = checked_range(start, end, index.len()) else {
            continue;
        };
        for item in start..end {
            let count = counts.entry(item).or_insert_with(|| {
                touched.push(item);
                0
            });
            *count += 1;
        }
    }
    let mut labels = Vec::with_capacity(touched.len());
    let mut result = Vec::with_capacity(touched.len());
    for item in touched {
        labels.push(index[item]);
        result.push(counts[&item]);
    }
    Ok((labels, result))
}

/// Count rows covered by each compact right-side position for reverse interval
/// ranges. Ranges use half-open `[start, end)` semantics.
///
/// # Arguments
///
/// * `starts` - Inclusive interval start for each row.
/// * `ends` - Exclusive interval end for each row.
/// * `index` - Right-side labels in ordinal position order.
///
/// # Returns
///
/// The emitted labels and one count per touched interval position.
#[pyfunction]
pub fn compute_size_rev_start_end<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
    let index = index.as_array();
    let (labels, counts) = size_rev_start_end_core(starts, ends, index)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((labels.into_pyarray(py), counts.into_pyarray(py)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::array, PyArrayMethods};

    #[test]
    fn counts_prefixes_and_suffixes_in_compact_slots() {
        let index = array![50_i64, 10, 90];
        assert_eq!(
            size_rev_ends_core(array![2_i64, 3, 1].view(), index.view()),
            Ok((array![50, 10, 90], array![3, 2, 1]))
        );
        assert_eq!(
            size_rev_starts_core(array![1_i64, 0, 2].view(), index.view()),
            Ok((array![50, 10, 90], array![1, 2, 3]))
        );
    }

    #[test]
    fn duplicate_index_labels_are_explicitly_unsupported() {
        Python::initialize();
        Python::attach(|py| {
            let starts = PyArray1::from_array(py, &array![0_i64, 1, 0]);
            let ends = PyArray1::from_array(py, &array![2_i64, 3, 1]);
            let index = PyArray1::from_array(py, &array![10_i64, 20, 10]);
            let (labels, counts) = compute_size_rev_start_end(
                py,
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_array(), array![10, 20, 10].view());
            assert_eq!(counts.readonly().as_array(), array![2, 2, 1].view());
        });
    }

    #[test]
    fn dense_match_path_preserves_gapped_label_order() {
        Python::initialize();
        Python::attach(|py| {
            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let index = PyArray1::from_vec(py, vec![50_i64, 10, 90]);
            let matches = PyArray1::from_vec(py, vec![1_i8, 0, 1]);
            let (labels, counts) = compute_size_rev_end_matches(
                py,
                ends.readonly(),
                index.readonly(),
                matches.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_array(), array![50, 90].view());
            assert_eq!(counts.readonly().as_array(), array![1, 1].view());

            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let (labels, counts) = compute_size_rev_start_matches(
                py,
                starts.readonly(),
                index.readonly(),
                matches.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_array(), array![50, 90].view());
            assert_eq!(counts.readonly().as_array(), array![1, 1].view());

            let ends = PyArray1::from_vec(py, vec![3_i64]);
            let (labels, counts) = compute_size_rev_start_end_matches(
                py,
                starts.readonly(),
                ends.readonly(),
                index.readonly(),
                matches.readonly(),
            )
            .unwrap();
            assert_eq!(labels.readonly().as_array(), array![50, 90].view());
            assert_eq!(counts.readonly().as_array(), array![1, 1].view());
        });
    }

    #[test]
    fn rejects_empty_and_invalid_boundaries() {
        let index = array![10_i64];
        assert_eq!(
            size_rev_ends_core(array![0_i64].view(), index.view()),
            Ok((array![], array![]))
        );
        assert!(size_rev_starts_core(array![-1_i64].view(), index.view()).is_err());
        assert!(size_rev_ends_core(array![1_i64].view(), array![].view()).is_err());
    }

    #[test]
    fn skips_zero_width_rows_without_affecting_other_rows() {
        let index = array![10_i64, 20, 30];
        assert_eq!(
            size_rev_starts_core(array![1_i64, 3].view(), index.view()),
            Ok((array![20, 30], array![1, 1]))
        );
        assert_eq!(
            size_rev_ends_core(array![0_i64, 2].view(), index.view()),
            Ok((array![10, 20], array![1, 1]))
        );
    }

    #[test]
    fn positions_count_ordinals_and_emit_original_labels() {
        let starts = array![0_i64, 1];
        let ends = array![2_i64, 3];
        let index = array![100_i64, 900, 500];
        let positions = array![2_i64, 0, 1];
        let (labels, counts) =
            size_positions_core(starts.view(), ends.view(), index.view(), positions.view())
                .unwrap();
        let mut got: Vec<_> = labels.into_iter().zip(counts).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(100, 2), (500, 1), (900, 1)]);
    }

    #[test]
    fn positions_rejects_empty_starts_and_ends() {
        let starts = array![];
        let ends = array![];
        let index = array![10_i64];
        let positions = array![0_i64];

        assert!(
            size_positions_core(starts.view(), ends.view(), index.view(), positions.view())
                .is_err()
        );
    }
}

/// Count rows covered by each label reached through the positional candidate
/// tape.
/// `index` is expected to contain unique labels, as guaranteed by the
/// pyjanitor producer for this path.
///
/// # Arguments
///
/// * `starts` - Inclusive positional range starts.
/// * `ends` - Exclusive positional range ends.
/// * `index` - Right-side labels addressed by `positions`.
/// * `positions` - Positional candidate tape.
///
/// # Returns
///
/// The labels reached through `positions` and their coverage counts. `starts`,
/// `ends`, `index`, and `positions` must not be empty; `starts` and `ends` must
/// have equal lengths. Returned label/count pairs are aligned, but their
/// ordering is unspecified and follows HashMap iteration order in sparse mode.
pub fn size_positions_core(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    // `positions.len()` counts tape entries, not distinct ordinals. It is a
    // cheap upper-bound estimate only, so repeated ordinals can still cause
    // dense storage to be selected for sparse state.
    let dense = should_use_dense_match_storage(index.len(), positions.len());
    Ok(size_positions_core_with_storage_unchecked(
        starts, ends, index, positions, dense,
    ))
}

/// Run positional coverage counting with an explicit storage mode.
/// This is a Rust-only benchmark entry point; production callers should use
/// [`size_positions_core`] for automatic dispatch.
///
/// # Arguments
///
/// * `starts` - Inclusive start of each half-open interval; must not be empty.
/// * `ends` - Exclusive end of each half-open interval; must not be empty.
/// * `index` - Right-side labels addressed by ordinal; must not be empty.
/// * `positions` - Candidate tape of ordinals into `index`; must not be empty.
/// * `dense` - Selects vector storage when true and HashMap storage when false.
pub fn size_positions_core_with_storage(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    dense: bool,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("starts", starts.len(), "ends", ends.len())?;
    Ok(size_positions_core_with_storage_unchecked(
        starts, ends, index, positions, dense,
    ))
}

fn size_positions_core_with_storage_unchecked(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    dense: bool,
) -> (Vec<i64>, Vec<i64>) {
    // ELI5: a long candidate tape does not imply many distinct ordinals, so
    // avoid reserving a table for candidates that may never become state.
    if dense {
        let mut seen = vec![false; index.len()];
        let mut counts = vec![0_i64; index.len()];
        for (start, end) in starts.into_iter().zip(ends) {
            let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
                continue;
            };
            for item in start_..end_ {
                let Some(ordinal) = checked_index(positions[item], index.len()) else {
                    continue;
                };
                seen[ordinal] = true;
                counts[ordinal] += 1;
            }
        }
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for (ordinal, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[ordinal]);
                values.push(counts[ordinal]);
            }
        }
        return (labels, values);
    }

    let mut counts: HashMap<usize, i64> = HashMap::new();
    for (start, end) in starts.into_iter().zip(ends) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        for item in start_..end_ {
            let Some(ordinal) = checked_index(positions[item], index.len()) else {
                continue;
            };
            let count = match counts.entry(ordinal) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(0),
            };
            *count += 1;
        }
    }
    let mut labels = Vec::with_capacity(counts.len());
    let mut values = Vec::with_capacity(counts.len());
    for (ordinal, count) in counts {
        labels.push(index[ordinal]);
        values.push(count);
    }
    (labels, values)
}

/// Count positional-tape coverage for each reached right-side label.
///
/// # Arguments
///
/// * `starts` - Inclusive start of each positional tape range.
/// * `ends` - Exclusive end of each positional tape range.
/// * `index` - Right-side labels addressed by positional ordinals.
/// * `positions` - Positional candidate tape mapping to `index`.
///
/// `starts`, `ends`, `index`, and `positions` must not be empty, and `starts`
/// and `ends` must have equal lengths. Returned label/count pairs are aligned
/// but unordered.
#[pyfunction]
pub fn compute_size_rev_positions<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
) -> SizeRevResult<'py> {
    let starts = starts.as_array();
    let ends = ends.as_array();
    let index = index.as_array();
    let positions = positions.as_array();
    let (labels, counts) = size_positions_core(starts, ends, index, positions)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let indexers = Array1::from_vec(labels);
    let result = Array1::from_vec(counts);
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
