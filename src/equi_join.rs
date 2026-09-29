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
use numpy::ndarray::ArrayView1;

/// Materialized labels from the sorted physical right layout.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EquiPairs {
    pub left: Vec<i64>,
    pub right: Vec<i64>,
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
}
