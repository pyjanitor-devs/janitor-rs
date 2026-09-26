//! Paper-guided region construction for dual and multi-predicate joins.
//!
//! `range_join` owns ordinary half-open windows. This module instead consumes
//! the two independent boundary sequences, aligns them by original IDs, and
//! exposes region labels for the paper's monotonic sweep.
//!
//! The region construction and sweep follow:
//! <https://www.scitepress.org/papers/2018/68268/68268.pdf>

use std::collections::{BTreeMap, HashMap};

use crate::anchor_non_equi_join::range_window;
use crate::join_common::{result_dict, Keep};
use crate::multi_join_indices::common::{add_right_region, GroupState};
use crate::op::CompareOp;
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, Predicate,
};
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::range_predicate::{parse_any_range_predicate, AnyParsedRangePredicate};
/// Search boundary sequence for one primary inequality.
///
/// For `<`/`<=`, each boundary starts the matching right suffix. For `>`/>=`
/// each boundary ends the matching right prefix and `reverse` records that
/// orientation for region construction.
pub(crate) struct RegionBoundary {
    /// Original left-row identifiers paired with `boundaries`.
    pub(crate) left_index: Vec<i64>,
    /// Original right-row identifiers paired with the sorted right values.
    pub(crate) right_index: Vec<i64>,
    /// Physical right positions returned by the typed binary searches.
    pub(crate) boundaries: Vec<usize>,
    /// Whether the anchor uses a greater-than orientation.
    pub(crate) reverse: bool,
}

/// Two primary region label sequences aligned by original row identifiers.
pub(crate) struct AlignedRegions {
    /// Original left-row identifiers in aligned traversal order.
    pub(crate) left_index: Vec<i64>,
    /// Original right-row identifiers in the first anchor's physical order.
    pub(crate) right_index: Vec<i64>,
    /// First primary region number for each left row.
    pub(crate) left_first: Vec<i64>,
    /// Second primary region number for each left row.
    pub(crate) left_second: Vec<i64>,
    /// First primary region number for each right row; monotonic increasing.
    pub(crate) right_first: Vec<i64>,
    /// Second primary region number for each right row; may be non-monotonic.
    pub(crate) right_second: Vec<i64>,
}

/// Find the first right-region position satisfying left <= right.
///
/// Region labels are integer encodings of eligible suffixes. The search is
/// delegated to the same range-window/partition-point implementation used by
/// the ordinary range kernels.
fn first_eligible(values: &[i64], left: i64) -> usize {
    // `values` is a monotonic region path. `range_window(..., Le)` returns
    // the first position where `left <= right`, which is the beginning of the
    // eligible right suffix.
    range_window(left, ArrayView1::from(values), CompareOp::Le).0
}

/// Order left queries for the paper sweep.
pub(crate) fn sweep_queries(regions: &AlignedRegions) -> Vec<(usize, usize)> {
    // Each query stores `(first_region_start, left_position)`. The start is
    // the first right position that can satisfy the first primary predicate.
    let mut queries = regions
        .left_first
        .iter()
        .enumerate()
        .map(|(left_position, &value)| (first_eligible(&regions.right_first, value), left_position))
        .collect::<Vec<_>>();
    // Process larger starts first. The active right suffix can then grow
    // leftward as later queries require more right rows.
    queries.sort_unstable_by(|left, right| right.0.cmp(&left.0));
    queries
}

/// Materialize an exact two-range `all` result directly from two sweeps.
///
/// The first sweep counts matches per left row and computes offsets. The
/// second sweep writes directly into the final index buffers, so no candidate
/// positions buffer is needed for the exact two-range case.
fn build_all_indices(regions: &AlignedRegions) -> Result<(Vec<i64>, Vec<i64>), String> {
    let queries = sweep_queries(regions);
    let mut counts = vec![0_usize; regions.left_index.len()];
    let mut overflow = false;
    let mut active = BTreeMap::<i64, GroupState>::new();
    let mut next = vec![-1_i64; regions.right_index.len()];
    let mut previous_end = regions.right_index.len();
    // First pass: count final output pairs. Since this is the exact two-range
    // path, there are no residual predicates to evaluate here.
    for (start, left_position) in queries.iter().copied() {
        if start >= regions.right_index.len() {
            continue;
        }
        // Grow the active suffix exactly as in the extended candidate sweep.
        add_right_region(
            ArrayView1::from(&regions.right_second[..]),
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;
        // Report all second-region values satisfying the second predicate.
        for (_, state) in active.range(regions.left_second[left_position]..) {
            let mut position = state.head;
            while position >= 0 {
                match counts[left_position].checked_add(1) {
                    Some(count) => counts[left_position] = count,
                    None => overflow = true,
                }
                position = next[position as usize];
            }
        }
    }
    if overflow {
        return Err("region index result size exceeds platform capacity".to_owned());
    }

    let mut offsets = vec![0_usize; counts.len() + 1];
    for (left_position, count) in counts.iter().copied().enumerate() {
        offsets[left_position + 1] = offsets[left_position]
            .checked_add(count)
            .ok_or("region index result size exceeds platform capacity")?;
    }
    let total = offsets[regions.left_index.len()];
    if total == 0 {
        // There are no matches at all, so the second pass is unnecessary.
        return Ok((Vec::new(), Vec::new()));
    }
    let mut left_index = vec![0_i64; total];
    let mut right_index = vec![0_i64; total];
    let mut cursors = offsets[..regions.left_index.len()].to_vec();
    active.clear();
    next.fill(-1);
    previous_end = regions.right_index.len();
    // Second pass: write directly into the final output arrays. The cursor
    // for each left row starts at that row's offset and advances independently.
    for (start, left_position) in queries.iter().copied() {
        if start >= regions.right_index.len() {
            continue;
        }
        add_right_region(
            ArrayView1::from(&regions.right_second[..]),
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;
        for (_, state) in active.range(regions.left_second[left_position]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                let output_position = cursors[left_position];
                left_index[output_position] = regions.left_index[left_position];
                right_index[output_position] = regions.right_index[right_position];
                cursors[left_position] += 1;
                position = next[right_position];
            }
        }
    }
    Ok((left_index, right_index))
}

/// Materialize extended `all` results by filtering inside two sweeps.
fn build_all_indices_extended<P>(
    regions: &AlignedRegions,
    mut predicates_pass: P,
) -> Result<(Vec<i64>, Vec<i64>), String>
where
    P: FnMut(usize, usize) -> bool,
{
    let queries = sweep_queries(regions);
    // `counts[left]` will count only complete matches: both primary regions
    // must pass and every residual predicate must pass too.
    let mut counts = vec![0_usize; regions.left_index.len()];
    let mut active = BTreeMap::<i64, GroupState>::new();
    let mut next = vec![-1_i64; regions.right_index.len()];
    let mut previous_end = regions.right_index.len();

    // First pass: count only candidates that pass every residual predicate.
    // The active map handles the primary predicates; the callback handles the
    // remaining predicates for each physical left/right pair.
    for (start, left_position) in queries.iter().copied() {
        if start >= regions.right_index.len() {
            continue;
        }
        add_right_region(
            ArrayView1::from(&regions.right_second[..]),
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;
        for (_, state) in active.range(regions.left_second[left_position]..) {
            // `state.head` starts a linked list of duplicate right positions.
            // Follow `next` so duplicates are counted individually.
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if predicates_pass(left_position, right_position) {
                    counts[left_position] = counts[left_position]
                        .checked_add(1)
                        .ok_or("region index result size exceeds platform capacity")?;
                }
                position = next[right_position];
            }
        }
    }

    let mut offsets = vec![0_usize; counts.len() + 1];
    // Turn counts into contiguous output ranges, one range per left row.
    for (left_position, count) in counts.iter().copied().enumerate() {
        offsets[left_position + 1] = offsets[left_position]
            .checked_add(count)
            .ok_or("region index result size exceeds platform capacity")?;
    }
    let total = offsets[regions.left_index.len()];
    if total == 0 {
        // No candidate survived the residual filters, so there is no second
        // sweep or output allocation to perform.
        return Ok((Vec::new(), Vec::new()));
    }

    let mut left_index = vec![0_i64; total];
    let mut right_index = vec![0_i64; total];
    let mut cursors = offsets[..regions.left_index.len()].to_vec();
    active.clear();
    next.fill(-1);
    previous_end = regions.right_index.len();

    // Second pass: rebuild the chains, reapply residuals, and write directly
    // into the exact output slice belonging to each left row. Residuals are
    // intentionally evaluated again instead of retaining every candidate.
    for (start, left_position) in queries.iter().copied() {
        if start >= regions.right_index.len() {
            continue;
        }
        add_right_region(
            ArrayView1::from(&regions.right_second[..]),
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;
        for (_, state) in active.range(regions.left_second[left_position]..) {
            // Walk every duplicate position in this qualifying region value.
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if predicates_pass(left_position, right_position) {
                    let output_position = cursors[left_position];
                    left_index[output_position] = regions.left_index[left_position];
                    right_index[output_position] = regions.right_index[right_position];
                    cursors[left_position] += 1;
                }
                position = next[right_position];
            }
        }
    }
    Ok((left_index, right_index))
}

/// Materialize extended `first`, `last`, or `any` results in one sweep.
fn build_selected_indices_extended<P>(
    regions: &AlignedRegions,
    keep: Keep,
    mut predicates_pass: P,
) -> (Vec<i64>, Vec<i64>)
where
    P: FnMut(usize, usize) -> bool,
{
    let queries = sweep_queries(regions);
    // A selected result needs at most one physical right position per left
    // row, so this vector is much smaller than an all-match result.
    let mut selected = vec![None; regions.left_index.len()];
    let mut active = BTreeMap::<i64, GroupState>::new();
    let mut next = vec![-1_i64; regions.right_index.len()];
    let mut previous_end = regions.right_index.len();

    // Filter candidates before applying first/last/any. This preserves the
    // contract that keep semantics see only complete predicate matches.
    for (start, left_position) in queries {
        if start >= regions.right_index.len() {
            continue;
        }
        add_right_region(
            ArrayView1::from(&regions.right_second[..]),
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;
        // The map range enforces the second primary predicate. Each GroupState
        // then supplies every physical right position for one region value.
        'candidate_groups: for (_, state) in active.range(regions.left_second[left_position]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if !predicates_pass(left_position, right_position) {
                    // A primary candidate that fails a residual predicate is
                    // invisible to first/last/any and cannot be selected.
                    position = next[right_position];
                    continue;
                }
                if keep == Keep::Any {
                    // Any can stop immediately after the first complete match
                    // for this left row.
                    selected[left_position] = Some(right_position);
                    break 'candidate_groups;
                }
                selected[left_position] = match selected[left_position] {
                    // First passing candidate becomes the initial selection.
                    None => Some(right_position),
                    Some(current) if keep == Keep::First => Some(
                        if regions.right_index[right_position] < regions.right_index[current] {
                            right_position
                        } else {
                            current
                        },
                    ),
                    Some(current) => Some(
                        if regions.right_index[right_position] > regions.right_index[current] {
                            right_position
                        } else {
                            current
                        },
                    ),
                };
                position = next[right_position];
            }
        }
    }

    let mut left_index = Vec::new();
    let mut right_index = Vec::new();
    // The sweep order is not left-row order, so materialize selected rows by
    // walking `selected` in canonical left order.
    for (left_position, right_position) in selected.into_iter().enumerate() {
        if let Some(right_position) = right_position {
            left_index.push(regions.left_index[left_position]);
            right_index.push(regions.right_index[right_position]);
        }
    }
    (left_index, right_index)
}

/// Materialize `first`, `last`, or `any` for exactly two range predicates.
///
/// This path has no residual predicate callback. `first` and `last` select by
/// original right index label, while `any` returns the first candidate found
/// by the sweep.
fn build_selected_indices(regions: &AlignedRegions, keep: Keep) -> (Vec<i64>, Vec<i64>) {
    let queries = sweep_queries(regions);
    let mut selected = vec![None; regions.left_index.len()];
    let mut active = BTreeMap::<i64, GroupState>::new();
    let mut next = vec![-1_i64; regions.right_index.len()];
    let mut previous_end = regions.right_index.len();

    // One sweep is enough for selected modes. Store only the chosen physical
    // right position for each left row; the final label lookup is O(1).
    for (start, left_position) in queries {
        if start >= regions.right_index.len() {
            continue;
        }
        add_right_region(
            ArrayView1::from(&regions.right_second[..]),
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;
        for (_, state) in active.range(regions.left_second[left_position]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                selected[left_position] = match (keep, selected[left_position]) {
                    (Keep::Any, None) => Some(right_position),
                    (Keep::Any, current) => current,
                    (Keep::First, None) => Some(right_position),
                    (Keep::First, Some(current)) => Some(
                        if regions.right_index[right_position] < regions.right_index[current] {
                            right_position
                        } else {
                            current
                        },
                    ),
                    (Keep::Last, None) => Some(right_position),
                    (Keep::Last, Some(current)) => Some(
                        if regions.right_index[right_position] > regions.right_index[current] {
                            right_position
                        } else {
                            current
                        },
                    ),
                    (Keep::All, _) => unreachable!(),
                };
                position = next[right_position];
            }
        }
    }

    let mut left_index = Vec::new();
    let mut right_index = Vec::new();
    for left_position in 0..regions.left_index.len() {
        if let Some(right_position) = selected[left_position] {
            left_index.push(regions.left_index[left_position]);
            right_index.push(regions.right_index[right_position]);
        }
    }
    (left_index, right_index)
}

/// Build original-index pairs from exactly two aligned primary regions.
///
/// No residual predicate callback is accepted because this builder is used
/// only by the exact two-range API.
///
/// # Arguments
///
/// * `regions` - The two primary region paths after alignment.
/// * `keep` - Selection semantics applied to primary matches.
///
/// # Returns
///
/// Paired original left and right index labels.
pub(crate) fn build_indices(
    regions: &AlignedRegions,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if keep == Keep::All {
        return build_all_indices(regions);
    }
    Ok(build_selected_indices(regions, keep))
}

/// Build original-index pairs from aligned regions and residual predicates.
///
/// The primary region constraints are traversed first. `predicates_pass` is
/// then called for each primary candidate so multi-predicate joins can apply
/// residual predicates before `keep` selects the result. The returned pairs
/// are emitted in aligned left-row order; for `Keep::All`, right-row order is
/// determined by the selected region traversal.
///
/// # Arguments
///
/// * `regions` - The two primary region paths after alignment.
/// * `keep` - Selection semantics applied after residual predicates pass.
/// * `predicates_pass` - Callback returning whether all residual predicates
///   pass for a positional left/right pair. Use a callback that always returns
///   `true` when there are no residual predicates.
///
/// # Returns
///
/// A pair of vectors containing original left and right index labels. The
/// vectors have equal length and are positionally paired.
pub(crate) fn build_indices_extended<P>(
    regions: &AlignedRegions,
    keep: Keep,
    predicates_pass: P,
) -> Result<(Vec<i64>, Vec<i64>), String>
where
    P: FnMut(usize, usize) -> bool,
{
    if keep == Keep::All {
        build_all_indices_extended(regions, predicates_pass)
    } else {
        Ok(build_selected_indices_extended(
            regions,
            keep,
            predicates_pass,
        ))
    }
}

/// Build one monotonic boundary sequence from a typed primary anchor.
///
/// # Arguments
///
/// * `anchor` - A parsed primary anchor whose right values are sorted in
///   ascending order and whose operator is `<`, `<=`, `>`, or `>=`.
///
/// # Returns
///
/// A [`RegionBoundary`] containing the original IDs, one binary-search
/// boundary per left row, and the traversal orientation.
///
/// # Errors
///
/// Returns an error for `==` or `!=`, because those operators do not define a
/// monotonic region boundary.
fn region_boundaries(anchor: &AnyParsedRangePredicate<'_>) -> Result<RegionBoundary, String> {
    macro_rules! build {
        ($predicate:expr) => {{
            let predicate = $predicate;
            if !matches!(
                predicate.op,
                CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
            ) {
                return Err("region anchors require an inequality operator".to_owned());
            }
            let reverse = matches!(predicate.op, CompareOp::Gt | CompareOp::Ge);
            let mut boundaries = Vec::with_capacity(predicate.left.as_array().len());
            for value in predicate.left.as_array().iter() {
                // `range_window` performs the typed binary search. For a
                // less-than anchor we keep the suffix start; for a
                // greater-than anchor we keep the prefix end.
                let (start, end) = range_window(*value, predicate.right.as_array(), predicate.op);
                boundaries.push(if reverse { end } else { start });
            }
            Ok(RegionBoundary {
                left_index: predicate.left_index.as_array().to_vec(),
                right_index: predicate.right_index.as_array().to_vec(),
                boundaries,
                reverse,
            })
        }};
    }
    match anchor {
        AnyParsedRangePredicate::I64(value) => build!(value),
        AnyParsedRangePredicate::I32(value) => build!(value),
        AnyParsedRangePredicate::I16(value) => build!(value),
        AnyParsedRangePredicate::I8(value) => build!(value),
        AnyParsedRangePredicate::U64(value) => build!(value),
        AnyParsedRangePredicate::U32(value) => build!(value),
        AnyParsedRangePredicate::U16(value) => build!(value),
        AnyParsedRangePredicate::U8(value) => build!(value),
        AnyParsedRangePredicate::F64(value) => build!(value),
        AnyParsedRangePredicate::F32(value) => build!(value),
    }
}

/// Convert one boundary sequence into left/right region numbers.
///
/// # Arguments
///
/// * `boundary` - The boundary sequence produced by [`region_boundaries`].
///   Its right IDs and boundary orientation are consumed.
///
/// # Returns
///
/// `(left_index, left_region, right_index, right_region)`, with empty-window
/// left rows removed and reverse-oriented right regions normalized for the
/// sweep.
fn labels(boundary: RegionBoundary) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let RegionBoundary {
        left_index: original_left_index,
        mut right_index,
        boundaries,
        reverse,
    } = boundary;
    // Each boundary marks where one left row's eligible right region begins
    // (or ends for a reversed anchor). Difference-style increments let us
    // construct all right region labels in one cumulative pass.
    let mut right_region = vec![0_i64; right_index.len()];
    for position in boundaries.iter().copied() {
        if position < right_region.len() {
            right_region[position] += 1;
        }
    }
    if !right_region.is_empty() {
        // The first element is the baseline. Subtracting one makes the
        // cumulative labels agree with the boundary convention: a right row
        // before a boundary belongs to the preceding region.
        right_region[0] -= 1;
        for position in 1..right_region.len() {
            right_region[position] += right_region[position - 1];
        }
    }
    let mut left_index = Vec::new();
    let mut left_region = Vec::new();
    for (position, boundary) in boundaries.iter().enumerate() {
        // A boundary at or beyond the right length has no matching right
        // position, so that left row is removed before alignment.
        if *boundary < right_region.len() {
            left_index.push(original_left_index[position]);
            left_region.push(right_region[*boundary]);
        }
    }
    if reverse {
        // Greater-than anchors were built as prefixes. Reverse their right
        // layout so the matching prefix becomes a matching suffix. Reversing
        // alone would make the region labels descend, so complement every
        // label around the largest right label. The left threshold needs one
        // extra step: this preserves the strict/inclusive distinction and
        // prevents an excluded equal boundary from becoming a match.
        let maximum = right_region.iter().copied().max().unwrap_or(0);
        right_index.reverse();
        right_region.reverse();
        for value in &mut right_region {
            *value = maximum - *value;
        }
        for value in &mut left_region {
            *value = maximum - *value + 1;
        }
    }
    // `sweep_queries` uses binary search over the first right-region path.
    // This is the invariant that makes that search valid for all four
    // inequality operators, including the normalized greater-than paths.
    debug_assert!(right_region.windows(2).all(|pair| pair[0] <= pair[1]));
    (left_index, left_region, right_index, right_region)
}

/// Align two primary region sets by their original left and right IDs.
///
/// # Arguments
///
/// * `first` - The first primary predicate's region boundary.
/// * `second` - The second primary predicate's region boundary. Its value
///   dtype may differ from `first`, but its original IDs identify the same
///   logical rows.
///
/// # Returns
///
/// An [`AlignedRegions`] value containing both region numbers for every
/// retained left and right identifier.
///
/// # Errors
///
/// Empty aligned sides are returned as empty vectors; they represent a valid
/// no-match result rather than a malformed join.
pub(crate) fn align(
    first: RegionBoundary,
    second: RegionBoundary,
) -> Result<AlignedRegions, String> {
    // `labels` removes impossible rows independently for each anchor. The
    // second anchor therefore has to be joined back to the first by original
    // identifiers rather than by vector position.
    let first = labels(first);
    let second = labels(second);
    // These maps turn original IDs into positions in the second anchor. They
    // are built once so alignment is linear instead of repeatedly scanning the
    // second anchor for every first-anchor row.
    let second_left = second
        .0
        .iter()
        .enumerate()
        .map(|(position, id)| (*id, position))
        .collect::<HashMap<_, _>>();
    let second_right = second
        .2
        .iter()
        .enumerate()
        .map(|(position, id)| (*id, position))
        .collect::<HashMap<_, _>>();
    let mut output = AlignedRegions {
        left_index: Vec::new(),
        right_index: Vec::new(),
        left_first: Vec::new(),
        left_second: Vec::new(),
        right_first: Vec::new(),
        right_second: Vec::new(),
    };
    // Keep the first anchor's left order as the canonical left-row order.
    // Look up the corresponding second region using the original left ID.
    for (position, id) in first.0.iter().enumerate() {
        if let Some(&other) = second_left.get(id) {
            output.left_index.push(*id);
            output.left_first.push(first.1[position]);
            output.left_second.push(second.1[other]);
        }
    }
    // Keep the first anchor's right order as the canonical physical right
    // layout. The second region is reordered to that same layout by ID.
    for (position, id) in first.2.iter().enumerate() {
        if let Some(&other) = second_right.get(id) {
            output.right_index.push(*id);
            output.right_first.push(first.3[position]);
            output.right_second.push(second.3[other]);
        }
    }
    Ok(output)
}

/// Parse and align the first two predicates in a dual or multi-predicate join.
///
/// # Arguments
///
/// * `predicates` - A list whose first two entries are five-element primary
///   range anchors. Later entries are residual predicates and are not parsed
///   here.
///
/// # Errors
///
/// Returns a Python `ValueError` when fewer than two anchors are supplied or
/// an anchor is malformed or uses `==`/`!=`.
pub(crate) fn parse_and_align<'py>(predicates: &Bound<'py, PyList>) -> PyResult<AlignedRegions> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "regions require two primary inequality predicates",
        ));
    }
    // Only the first two entries participate in region construction. Extended
    // callers leave later residual tuples untouched for the matcher.
    let first_item = predicates.get_item(0)?;
    let first = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second = second_item.cast::<PyTuple>()?;
    // `extended=true` selects the five-element anchor form:
    // (left values, left IDs, right values, right IDs, operator).
    let first = parse_any_range_predicate(&first, true)?;
    let second = parse_any_range_predicate(&second, true)?;
    align(
        region_boundaries(&first).map_err(PyValueError::new_err)?,
        region_boundaries(&second).map_err(PyValueError::new_err)?,
    )
    .map_err(PyValueError::new_err)
}

/// Build index pairs from aligned regions and optional residual predicates.
fn build_regions_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: Keep,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let regions = parse_and_align(predicates)?;

    // Copy only predicates after the two anchors. They may use any supported
    // operator and are evaluated against the already-aligned physical rows.
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(2) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    // Residual arrays must have the same physical left/right lengths as the
    // aligned regions because their positions are used directly in the hot
    // traversal loop.
    check_predicate_lengths(&parsed, regions.left_index.len(), regions.right_index.len())?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    // The matcher closure is called only for primary candidates. `keep` is
    // applied by traversal after this closure has exhausted all residuals.
    let (left_index, right_index) = build_indices_extended(&regions, keep, |left, right| {
        predicates_match_dispatch(&views, metadata_views.as_deref(), left, right)
    })
    .map_err(PyValueError::new_err)?;
    if left_index.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, left_index, right_index, None, None)?))
}

/// Build index pairs for exactly two primary region predicates.
///
/// This API performs only primary-region traversal and rejects residual
/// predicates. Use [`region_indices_extended`] when additional predicates are
/// present.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two five-element inequality anchors.
/// * `keep` - One of `all`, `first`, `last`, or `any`.
///
/// # Returns
///
/// `None` when no primary pair matches; otherwise a dictionary containing
/// paired `left_index` and `right_index` arrays.
///
/// # Errors
///
/// Returns a Python error when the predicate count is not exactly two, an
/// anchor is malformed, or `keep` is invalid.
#[pyfunction]
pub fn region_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "region_indices requires exactly two predicates",
        ));
    }
    let regions = parse_and_align(predicates)?;
    let keep = Keep::parse(keep)?;
    let (left_index, right_index) = build_indices(&regions, keep).map_err(PyValueError::new_err)?;
    if left_index.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, left_index, right_index, None, None)?))
}

/// Build index pairs for two primary regions followed by residual predicates.
///
/// Residual predicates may use any supported comparison operator and do not
/// need to be range predicates. They run after both primary regions pass and
/// before `keep` is applied. Nullable six-element `!=` residual tuples retain
/// their null-mask metadata through the shared matcher.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two five-element inequality anchors followed by
///   three-element residual predicates or six-element nullable `!=` tuples.
/// * `keep` - One of `all`, `first`, `last`, or `any`.
///
/// # Returns
///
/// `None` when no complete predicate combination matches; otherwise a
/// dictionary containing paired `left_index` and `right_index` arrays.
///
/// # Errors
///
/// Returns a Python error when fewer than two anchors are supplied, an anchor
/// or residual is malformed, lengths are misaligned, or `keep` is invalid.
#[pyfunction]
pub fn region_indices_extended<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "region_indices_extended requires at least two predicates",
        ));
    }
    build_regions_indices(py, predicates, Keep::parse(keep)?)
}

/// Register both regions index-generation entry points with the Python module.
///
/// # Arguments
///
/// * `m` - The parent `janitor_rs` Python module.
///
/// # Errors
///
/// Returns any PyO3 error raised while adding the function.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(region_indices, m)?)?;
    m.add_function(wrap_pyfunction!(region_indices_extended, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regions(right_second: Vec<i64>) -> AlignedRegions {
        AlignedRegions {
            left_index: vec![10, 11],
            right_index: vec![20, 21, 22, 23],
            left_first: vec![1, 2],
            left_second: vec![1, 2],
            right_first: vec![1, 2, 3, 4],
            right_second,
        }
    }

    #[test]
    fn every_inequality_orientation_produces_an_increasing_right_path() {
        // For right values [1, 2, 3, 4] and left values [1, 3], these are
        // the boundaries returned by `range_window`:
        //
        // * `<`  => first right value strictly greater: [1, 3]
        // * `<=` => first right value greater/equal:   [0, 2]
        // * `>`  => first right value greater/equal ends prefix: [0, 2]
        // * `>=` => first right value strictly greater ends prefix: [1, 3]
        //
        // The greater-than cases use the reverse normalization in `labels`.
        for (reverse, boundaries) in [
            (false, vec![1, 3]), // <
            (false, vec![0, 2]), // <=
            (true, vec![0, 2]),  // >
            (true, vec![1, 3]),  // >=
        ] {
            let (_, _, _, right_region) = labels(RegionBoundary {
                left_index: vec![10, 11],
                right_index: vec![20, 21, 22, 23],
                boundaries,
                reverse,
            });
            assert!(
                right_region.windows(2).all(|pair| pair[0] <= pair[1]),
                "right region path was not monotonic: {right_region:?}"
            );
        }
    }

    #[test]
    fn sweep_handles_monotonic_second_region() {
        let regions = regions(vec![1, 2, 3, 4]);
        let output = build_indices(&regions, Keep::All).expect("all pairs should build");
        assert_eq!(
            output,
            (
                vec![10, 10, 10, 10, 11, 11, 11],
                vec![20, 21, 22, 23, 21, 22, 23]
            )
        );
    }

    #[test]
    fn sweep_path_handles_duplicates_and_non_monotonic_values() {
        let regions = regions(vec![3, 1, 3, 2]);
        let output = build_indices(&regions, Keep::All).expect("all pairs should build");
        assert_eq!(
            output,
            (
                vec![10, 10, 10, 10, 11, 11, 11],
                vec![20, 21, 22, 23, 20, 22, 23]
            )
        );
    }

    #[test]
    fn keep_is_applied_after_residuals() {
        let regions = regions(vec![1, 2, 3, 4]);
        let mut output = Vec::new();
        let (left, right) = build_indices_extended(&regions, Keep::First, |_, right| right == 2)
            .expect("selection should succeed");
        output.extend(left.into_iter().zip(right));
        assert_eq!(output, vec![(10, 22), (11, 22)]);
    }
}
