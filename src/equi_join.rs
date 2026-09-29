//! Standalone issue #218 equi positional core.
//!
//! PyJanitor owns scalar/MultiIndex construction, `get_indexer`, factorization,
//! null semantics, sorting, and the mapping from sorted physical positions to
//! original right positions. This file owns only the positional hot loop.
//!
//! The same core supports:
//!
//! * equi-only candidates using a full right window;
//! * equi plus `!=` or other residual predicates through the callback;
//! * one range window followed by equi and residual filtering;
//! * two intersected range windows when both use one shared right layout.
//!
//! PyJanitor supplies aligned left/right range arrays and their original-index
//! mappings. Rust builds the binary-search boundaries and intersects the two
//! windows when both arrays share one physical permutation. If they do not,
//! Rust uses the first range as the primary window and evaluates the second as
//! a residual predicate. PyJanitor owns only equi lookup: it first tries
//! `get_indexer`, then falls back to `factorize` when duplicate right equi keys
//! make the right key index non-unique.
//! Forward singleton aggregation can bypass candidate expansion when the right
//! equi keys are unique; reverse aggregation remains an existing consumer of
//! the emitted positional pairs.

use crate::anchor_non_equi_join::range_window;
use crate::join_common::Keep;
use crate::op::CompareOp;
use numpy::{ndarray::ArrayView1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Materialized labels from the sorted physical right layout.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EquiPairs {
    pub left: Vec<i64>,
    pub right: Vec<i64>,
}

#[derive(Debug)]
struct DenseRightMetadata {
    any: Vec<usize>,
    first: Vec<usize>,
    last: Vec<usize>,
    groups: Vec<Vec<usize>>,
}

fn build_dense_right_metadata(
    right_index: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
) -> Result<DenseRightMetadata, String> {
    if right_index.len() != right_codes.len() {
        return Err("right index and duplicate codes must have equal lengths".to_owned());
    }
    let code_count = right_codes
        .iter()
        .copied()
        .filter(|&code| code >= 0)
        .max()
        .map(|code| {
            usize::try_from(code)
                .map_err(|_| "right code exceeds the positional contract")?
                .checked_add(1)
                .ok_or("right code exceeds the positional contract")
        })
        .transpose()
        .map_err(|_| "right code exceeds the positional contract")?
        .unwrap_or(0);
    let mut any = vec![usize::MAX; code_count];
    let mut first = vec![usize::MAX; code_count];
    let mut last = vec![usize::MAX; code_count];
    let mut groups = vec![Vec::new(); code_count];
    for (position, &code) in right_codes.iter().enumerate() {
        if code < -1 {
            return Err("right codes must be greater than or equal to -1".to_owned());
        }
        if code == -1 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid right code")?;
        groups[code].push(position);
        if any[code] == usize::MAX {
            any[code] = position;
        }
        if first[code] == usize::MAX || right_index[position] < right_index[first[code]] {
            first[code] = position;
        }
        if last[code] == usize::MAX || right_index[position] > right_index[last[code]] {
            last[code] = position;
        }
    }
    debug_assert_eq!(any.len(), code_count);
    debug_assert_eq!(first.len(), code_count);
    debug_assert_eq!(last.len(), code_count);
    debug_assert_eq!(groups.len(), code_count);
    Ok(DenseRightMetadata {
        any,
        first,
        last,
        groups,
    })
}

fn build_duplicate_equi_pairs_core(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
    keep: Keep,
) -> Result<Option<EquiPairs>, String> {
    if left_index.len() != left_indexer.len() {
        return Err("left index and equi indexer must have equal lengths".to_owned());
    }
    let metadata = build_dense_right_metadata(right_index, right_codes)?;
    let mut output = EquiPairs::default();
    for row in 0..left_indexer.len() {
        let code = left_indexer[row];
        if code < -1 {
            return Err("left codes must be greater than or equal to -1".to_owned());
        }
        if code == -1 {
            continue;
        }
        let code = usize::try_from(code).map_err(|_| "invalid left code")?;
        if code >= metadata.groups.len() || metadata.groups[code].is_empty() {
            continue;
        }
        let positions = match keep {
            Keep::Any => std::slice::from_ref(&metadata.any[code]),
            Keep::First => std::slice::from_ref(&metadata.first[code]),
            Keep::Last => std::slice::from_ref(&metadata.last[code]),
            Keep::All => metadata.groups[code].as_slice(),
        };
        for &position in positions {
            output.left.push(left_index[row]);
            output.right.push(right_index[position]);
        }
    }
    if output.left.is_empty() {
        Ok(None)
    } else {
        Ok(Some(output))
    }
}

/// Build indices for a pure equi join with duplicate right keys.
///
/// PyJanitor handles the unique-right case directly. Here
/// `original_right_positions` contains one dense factorization code per right
/// row, aligned with `right_index`; `-1` entries are ignored.
#[pyfunction]
pub fn equi_join_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    left_indexer: PyReadonlyArray1<'py, i64>,
    original_right_positions: PyReadonlyArray1<'py, i64>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    let pairs = build_duplicate_equi_pairs_core(
        left_index.as_array(),
        left_indexer.as_array(),
        right_index.as_array(),
        original_right_positions.as_array(),
        keep,
    )
    .map_err(PyValueError::new_err)?;
    pairs
        .map(|pairs| crate::join_common::result_dict(py, pairs.left, pairs.right, None, None))
        .transpose()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equi_join_indices, m)?)?;
    Ok(())
}

/// Scan equi candidates inside per-left half-open physical windows.
///
/// `right_index` is aligned with `right_codes`; it carries original right
/// positions/labels through the sorted layout. `matches` represents residual
/// predicates such as `!=`, or may simply return `true` for equi-only joins.
pub fn build_equi_pairs_core<F>(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    original_right_positions: Option<ArrayView1<'_, i64>>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    keep: Keep,
    mut matches: F,
) -> Result<Option<EquiPairs>, String>
where
    F: FnMut(usize, usize) -> bool,
{
    if left_index.len() != left_indexer.len() {
        return Err("left index and equi indexer must have equal lengths".to_owned());
    }
    if let Some(right_codes) = original_right_positions {
        if right_codes.len() != right_index.len() {
            return Err("right index and duplicate codes must have equal lengths".to_owned());
        }
    }
    if starts.len() != left_indexer.len() || ends.len() != left_indexer.len() {
        return Err("equi windows must align with left rows".to_owned());
    }

    let mut output = EquiPairs::default();
    for row in 0..left_indexer.len() {
        let start = usize::try_from(starts[row]).map_err(|_| "invalid equi window start")?;
        let end = usize::try_from(ends[row]).map_err(|_| "invalid equi window end")?;
        if start > end || end > right_index.len() {
            return Err("equi window is out of bounds".to_owned());
        }

        let left_key = left_indexer[row];
        if left_key < 0 {
            continue;
        }
        let mut selected = None;
        if let Some(right_codes) = original_right_positions {
            for right_position in start..end {
                if right_codes[right_position] != left_key || !matches(row, right_position) {
                    continue;
                }
                select_match(
                    &mut output,
                    &mut selected,
                    left_index[row],
                    right_position,
                    right_index,
                    keep,
                );
            }
        } else {
            let right_position =
                usize::try_from(left_key).map_err(|_| "invalid unique-right position")?;
            if right_position >= start && right_position < end && matches(row, right_position) {
                select_match(
                    &mut output,
                    &mut selected,
                    left_index[row],
                    right_position,
                    right_index,
                    keep,
                );
            }
        }
        if let Some(right_position) = selected {
            output.left.push(left_index[row]);
            output.right.push(right_index[right_position]);
        }
    }

    if output.left.is_empty() {
        Ok(None)
    } else {
        Ok(Some(output))
    }
}

/// Run an equi-only join without making PyJanitor manufacture a window.
///
/// The full `[0, right_len)` candidate window is an internal Rust detail. It
/// is deliberately not part of the Python preparation contract.
pub fn build_equi_pairs_full_core<F>(
    left_index: ArrayView1<'_, i64>,
    left_indexer: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    original_right_positions: Option<ArrayView1<'_, i64>>,
    keep: Keep,
    matches: F,
) -> Result<Option<EquiPairs>, String>
where
    F: FnMut(usize, usize) -> bool,
{
    let starts = vec![0_i64; left_indexer.len()];
    let ends = vec![
        i64::try_from(right_index.len())
            .map_err(|_| "right length exceeds the positional contract")?;
        left_indexer.len()
    ];
    build_equi_pairs_core(
        left_index,
        left_indexer,
        right_index,
        original_right_positions,
        ArrayView1::from(&starts),
        ArrayView1::from(&ends),
        keep,
        matches,
    )
}

/// Build one or two half-open right windows from already aligned range arrays.
///
/// ELI5: each range predicate draws a slice of the sorted right array for
/// every left row; two predicates keep only the overlap of their slices.
pub fn build_range_windows_core<T: PartialOrd + Copy>(
    first_left: ArrayView1<'_, T>,
    first_right: ArrayView1<'_, T>,
    first_operator: CompareOp,
    second: Option<(ArrayView1<'_, T>, ArrayView1<'_, T>, CompareOp)>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if first_left.is_empty() || first_right.is_empty() {
        return Ok((vec![0; first_left.len()], vec![0; first_left.len()]));
    }
    let mut starts = Vec::with_capacity(first_left.len());
    let mut ends = Vec::with_capacity(first_left.len());
    for &value in first_left {
        let (start, end) = range_window(value, first_right, first_operator);
        starts.push(i64::try_from(start).map_err(|_| "range start exceeds int64")?);
        ends.push(i64::try_from(end).map_err(|_| "range end exceeds int64")?);
    }
    if let Some((second_left, second_right, second_operator)) = second {
        if second_left.len() != first_left.len() || second_right.len() != first_right.len() {
            return Err("aligned range arrays must have equal lengths".to_owned());
        }
        for row in 0..first_left.len() {
            let (second_start, second_end) =
                range_window(second_left[row], second_right, second_operator);
            starts[row] = starts[row]
                .max(i64::try_from(second_start).map_err(|_| "range start exceeds int64")?);
            ends[row] =
                ends[row].min(i64::try_from(second_end).map_err(|_| "range end exceeds int64")?);
        }
    }
    Ok((starts, ends))
}

fn select_match(
    output: &mut EquiPairs,
    selected: &mut Option<usize>,
    left_label: i64,
    right_position: usize,
    right_index: ArrayView1<'_, i64>,
    keep: Keep,
) {
    match keep {
        Keep::All => {
            output.left.push(left_label);
            output.right.push(right_index[right_position]);
        }
        Keep::Any => *selected = Some(right_position),
        Keep::First => {
            if selected.is_none() || right_index[right_position] < right_index[selected.unwrap()] {
                *selected = Some(right_position);
            }
        }
        Keep::Last => {
            if selected.is_none() || right_index[right_position] > right_index[selected.unwrap()] {
                *selected = Some(right_position);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn unique_right_indexer_is_checked_against_the_window() {
        let result = build_equi_pairs_core(
            array![10_i64, 11].view(),
            array![1_i64, -1].view(),
            array![100_i64, 101].view(),
            None,
            array![0_i64, 0].view(),
            array![2_i64, 1].view(),
            Keep::All,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(
            result,
            Some(EquiPairs {
                left: vec![10],
                right: vec![101],
            })
        );
    }

    #[test]
    fn full_core_uses_the_entire_unique_right_array() {
        let result = build_equi_pairs_full_core(
            array![10_i64].view(),
            array![1_i64].view(),
            array![100_i64, 101].view(),
            None,
            Keep::All,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(
            result,
            Some(EquiPairs {
                left: vec![10],
                right: vec![101],
            })
        );
    }

    #[test]
    fn one_range_window_uses_binary_search_boundaries() {
        let (starts, ends) = build_range_windows_core(
            array![2_i64, 8].view(),
            array![1_i64, 3, 7, 9].view(),
            CompareOp::Le,
            None,
        )
        .unwrap();
        assert_eq!(starts, vec![1, 3]);
        assert_eq!(ends, vec![4, 4]);
    }

    #[test]
    fn two_range_windows_are_intersected() {
        let (starts, ends) = build_range_windows_core(
            array![2_i64].view(),
            array![1_i64, 3, 7, 9].view(),
            CompareOp::Le,
            Some((
                array![8_i64].view(),
                array![1_i64, 3, 7, 9].view(),
                CompareOp::Ge,
            )),
        )
        .unwrap();
        assert_eq!(starts, vec![1]);
        assert_eq!(ends, vec![3]);
    }

    #[test]
    fn filters_duplicate_equi_codes_inside_a_range_window() {
        let result = build_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            array![100_i64, 101, 102, 103].view(),
            Some(array![1_i64, 2, 2, 3].view()),
            array![0_i64].view(),
            array![4_i64].view(),
            Keep::All,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(
            result,
            Some(EquiPairs {
                left: vec![10, 10],
                right: vec![101, 102],
            })
        );
    }

    #[test]
    fn residual_filter_runs_after_equi_code_matching() {
        let result = build_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            array![100_i64, 101, 102].view(),
            Some(array![2_i64, 2, 3].view()),
            array![0_i64].view(),
            array![3_i64].view(),
            Keep::All,
            |_, right| right == 1,
        )
        .unwrap();
        assert_eq!(
            result,
            Some(EquiPairs {
                left: vec![10],
                right: vec![101],
            })
        );
    }

    #[test]
    fn first_and_last_use_original_right_labels() {
        let first = build_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            array![200_i64, 100].view(),
            Some(array![2_i64, 2].view()),
            array![0_i64].view(),
            array![2_i64].view(),
            Keep::First,
            |_, _| true,
        )
        .unwrap()
        .unwrap();
        let last = build_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            array![200_i64, 100].view(),
            Some(array![2_i64, 2].view()),
            array![0_i64].view(),
            array![2_i64].view(),
            Keep::Last,
            |_, _| true,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.right, vec![100]);
        assert_eq!(last.right, vec![200]);
    }

    #[test]
    fn dense_metadata_tracks_codes_and_right_length() {
        let metadata = build_dense_right_metadata(
            array![12_i64, 5, 9, 7].view(),
            array![3_i64, 1, 3, -1].view(),
        )
        .unwrap();
        assert_eq!(metadata.groups.len(), 4);
        assert_eq!(metadata.groups[3], vec![0, 2]);
        assert_eq!(metadata.groups[1], vec![1]);
        assert_eq!(metadata.any[3], 0);
        assert_eq!(metadata.first[3], 2);
        assert_eq!(metadata.last[3], 0);
    }

    #[test]
    fn duplicate_equi_keep_modes_use_dense_codes() {
        let left_index = array![10_i64, 11, 12];
        let left_indexer = array![3_i64, 1, -1];
        let right_index = array![12_i64, 5, 9, 7];
        let right_codes = array![3_i64, 1, 3, -1];

        let any = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::Any,
        )
        .unwrap();
        assert_eq!(any.unwrap().right, vec![12, 5]);

        let first = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::First,
        )
        .unwrap();
        assert_eq!(first.unwrap().right, vec![9, 5]);

        let last = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::Last,
        )
        .unwrap();
        assert_eq!(last.unwrap().right, vec![12, 5]);

        let all = build_duplicate_equi_pairs_core(
            left_index.view(),
            left_indexer.view(),
            right_index.view(),
            right_codes.view(),
            Keep::All,
        )
        .unwrap();
        assert_eq!(all.unwrap().right, vec![12, 9, 5]);
    }

    #[test]
    fn duplicate_equi_rejects_codes_below_minus_one() {
        let error = build_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![-2_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            Keep::All,
        )
        .unwrap_err();
        assert_eq!(error, "left codes must be greater than or equal to -1");

        let error = build_duplicate_equi_pairs_core(
            array![10_i64].view(),
            array![0_i64].view(),
            array![100_i64].view(),
            array![-2_i64].view(),
            Keep::All,
        )
        .unwrap_err();
        assert_eq!(error, "right codes must be greater than or equal to -1");
    }
}
