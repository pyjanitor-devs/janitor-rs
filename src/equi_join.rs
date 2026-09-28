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

use crate::join_common::Keep;
use ndarray::ArrayView1;

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
    left_codes: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    keep: Keep,
    mut matches: F,
) -> Result<Option<EquiPairs>, String>
where
    F: FnMut(usize, usize) -> bool,
{
    if left_index.len() != left_codes.len() {
        return Err("left index and equi codes must have equal lengths".to_owned());
    }
    if right_codes.len() != right_index.len() {
        return Err("right index and equi codes must have equal lengths".to_owned());
    }
    if starts.len() != left_codes.len() || ends.len() != left_codes.len() {
        return Err("equi windows must align with left rows".to_owned());
    }

    let mut output = EquiPairs::default();
    for row in 0..left_codes.len() {
        let start = usize::try_from(starts[row]).map_err(|_| "invalid equi window start")?;
        let end = usize::try_from(ends[row]).map_err(|_| "invalid equi window end")?;
        if start > end || end > right_codes.len() {
            return Err("equi window is out of bounds".to_owned());
        }

        let mut selected = None;
        for right_position in start..end {
            if right_codes[right_position] != left_codes[row] || !matches(row, right_position) {
                continue;
            }
            match keep {
                Keep::All => {
                    output.left.push(left_index[row]);
                    output.right.push(right_index[right_position]);
                }
                Keep::Any => {
                    selected = Some(right_position);
                    break;
                }
                Keep::First => {
                    if selected.is_none()
                        || right_index[right_position] < right_index[selected.unwrap()]
                    {
                        selected = Some(right_position);
                    }
                }
                Keep::Last => {
                    if selected.is_none()
                        || right_index[right_position] > right_index[selected.unwrap()]
                    {
                        selected = Some(right_position);
                    }
                }
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
    left_codes: ArrayView1<'_, i64>,
    right_codes: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    keep: Keep,
    matches: F,
) -> Result<Option<EquiPairs>, String>
where
    F: FnMut(usize, usize) -> bool,
{
    let starts = vec![0_i64; left_codes.len()];
    let ends = vec![
        i64::try_from(right_codes.len())
            .map_err(|_| "right length exceeds the positional contract")?;
        left_codes.len()
    ];
    build_equi_pairs_core(
        left_index,
        left_codes,
        right_codes,
        right_index,
        ArrayView1::from(&starts),
        ArrayView1::from(&ends),
        keep,
        matches,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn filters_duplicate_equi_codes_inside_a_range_window() {
        let result = build_equi_pairs_core(
            array![10_i64].view(),
            array![2_i64].view(),
            array![1_i64, 2, 2, 3].view(),
            array![100_i64, 101, 102, 103].view(),
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
            array![2_i64, 2, 3].view(),
            array![100_i64, 101, 102].view(),
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
            array![2_i64, 2].view(),
            array![200_i64, 100].view(),
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
            array![2_i64, 2].view(),
            array![200_i64, 100].view(),
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
