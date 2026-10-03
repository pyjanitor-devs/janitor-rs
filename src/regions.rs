//! Paper-guided region construction for dual and multi-predicate joins.
//!
//! `range_join` owns ordinary half-open windows. This module instead consumes
//! the two independent boundary sequences, aligns them by original IDs, and
//! exposes region labels for the paper's monotonic sweep.
//!
//! The region construction and sweep follow:
//! <https://www.scitepress.org/papers/2018/68268/68268.pdf>
//!
//! ## Coordinate systems
//!
//! This file uses three different positions. Keeping them separate is
//! essential:
//!
//! 1. **Source positions** index the original value arrays supplied by
//!    PyJanitor.
//! 2. **Region positions** index the compact arrays after impossible left rows
//!    are removed and after a greater-than first anchor reverses the right
//!    traversal layout.
//! 3. **Index labels** are the `i64` values returned to the caller. They may be
//!    ordinary positions, dataframe index values, or another caller-defined
//!    identifier.
//!
//! `AlignedRegions::left_positions` and `right_positions` translate region
//! positions back to source positions. The sweep itself works only in region
//! coordinates; residual predicates and aggregation state use the mappings
//! before touching source arrays.
//!
//! ## Sweep overview
//!
//! The first inequality creates a monotonic `right_first` path. For each left
//! row, `sweep_queries` binary-searches that path to find the first eligible
//! right region. Queries are processed from the largest start to the smallest
//! start, so the active right suffix grows in one direction. The ordered map
//! groups active right rows by `right_second`; a map range then reports the
//! rows satisfying the second inequality. Duplicate labels are preserved by a
//! linked list stored in each map group.
//!
//! ## Python ABI
//!
//! Index entry points receive a list whose first two items are five-field
//! anchors:
//!
//! ```text
//! (left_values, left_positions, right_values, right_positions, operator)
//! ```
//!
//! `left_positions` and `right_positions` are physical positions in the
//! reset-index Python frames. They travel with the value arrays and are
//! returned as physical output positions; region offsets are never exposed as
//! dataframe labels. Later tuples are residual predicates and are evaluated
//! only after both primary region constraints pass.
//!
//! Aggregation entry points are registered by this module as well. They use
//! the same two primary anchors, with optional output maps on the first
//! anchor, and consume aggregation arrays aligned to that first-anchor
//! layout. Forward aggregation writes into left output slots from right source
//! values; reverse aggregation writes into right output slots from left source
//! values. The aggregation implementation is included below so index and
//! aggregation paths share the same region construction and sweep helpers.
//!
//! ## Position and label contract
//!
//! `left_positions` and `right_positions` are physical positions in the
//! reset-index Python layouts. They are not sorted offsets and are not pandas
//! labels. Binary searches return offsets into sorted right values; only the
//! physical position arrays cross back to Python. The first anchor supplies
//! the canonical layout. The second anchor may be searched independently and
//! is then aligned back to that layout by its unique physical positions.
//!
//! Region offsets and physical positions must never be mixed. The sweep uses
//! region offsets, while residual matching translates compact positions
//! through `left_positions` and `right_positions` before indexing full
//! predicate arrays. Aggregation uses the same translation for source values
//! and output slots.
//!
//! ## Selection and allocation
//!
//! `traverse_candidates` orders left queries by their first-region boundary,
//! adds newly exposed right rows to a `BTreeMap` keyed by the second-region
//! label, and preserves duplicate labels in linked chains. Exact `all` uses
//! two sweeps to allocate final arrays at their exact size. Extended `all`
//! evaluates residuals once and buffers only passing compact pairs. `first`
//! and `last` compare output labels; `any` stops after the first complete
//! candidate.

use std::collections::{BTreeMap, HashMap};

use crate::aggregation_common::aggregation::{
    make_results_with_positions, parse_inputs, AggregationSet,
};
use crate::aggregation_common::checked_end;
use crate::compare_op::CompareOp;
use crate::join_aggregation_helpers::residuals;
use crate::join_search::range_window;
use crate::join_types::{result_dict, Keep};
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings_from,
    predicates_match_dispatch, Predicate,
};
use crate::range_predicate::{
    parse_aggregation_range_anchor, parse_any_range_predicate, AnyParsedRangePredicate,
    ParsedAggregationRangeAnchor,
};
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// State for one right-region value's duplicate-position chain.
struct GroupState {
    head: i64,
    tail: i64,
}

impl Default for GroupState {
    fn default() -> Self {
        Self { head: -1, tail: -1 }
    }
}

/// Add newly exposed right positions to their duplicate-value chains.
fn add_right_region(
    right_region: &[i64],
    start: usize,
    previous_end: usize,
    next: &mut [i64],
    groups: &mut BTreeMap<i64, GroupState>,
) {
    for right_position in (start..previous_end).rev() {
        let state = groups.entry(right_region[right_position]).or_default();
        if state.head == -1 {
            state.head = right_position as i64;
        } else {
            debug_assert!(state.tail >= 0 && (state.tail as usize) < next.len());
            next[state.tail as usize] = right_position as i64;
        }
        state.tail = right_position as i64;
    }
}

/// Validate a half-open region interval against the right layout.
fn checked_bounds(start: i64, end: i64, right_len: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(start).ok()?;
    let end = checked_end(end, right_len)?;
    if start >= end {
        return None;
    }
    Some((start, end))
}

/// Validate a region boundary against the right layout and sweep invariant.
fn checked_region_start(
    start: i64,
    right_len: usize,
    previous_end: usize,
) -> Result<Option<usize>, String> {
    if start > previous_end as i64 {
        return Err("starts must be monotonically non-increasing".to_string());
    }
    Ok(checked_bounds(start, right_len as i64, right_len).map(|(start, _)| start))
}

/// The independently searched boundary sequence for one primary inequality.
///
/// A region join starts with two independent binary searches: one for each of
/// its first two predicates. This value is the owned hand-off between those
/// searches and the later label/alignment steps. Its vectors are parallel:
/// entry `n` in `boundaries` belongs to the left identifier at entry `n` in
/// `left_index`.
///
/// For `<`/`<=`, each boundary starts the matching right suffix. For `>`/`>=`,
/// `region_boundaries` converts the matching-prefix length returned by
/// `range_window` into the equivalent suffix start after the right side is
/// traversed in reverse. That normalization lets `labels` use one algorithm
/// for all four supported inequality operators.
///
/// The index arrays are intentionally owned rather than borrowed views. The
/// parser temporarily borrows NumPy data while it performs the typed binary
/// search, but `labels` may compact the left side and reverse the physical
/// right layout. Keeping the identifiers in owned vectors means the internal
/// region structs do not need Python lifetimes, and makes it explicit that
/// these arrays are the metadata carried into the sweep. A borrowed-view
/// optimization is possible later, but would thread lifetimes through the
/// entire boundary, labeling, and alignment pipeline for little benefit
/// relative to this short-lived copy.
pub(crate) struct RegionBoundary {
    /// Original left-row identifiers paired positionally with `boundaries`.
    pub(crate) left_index: Vec<i64>,
    /// Original right-row identifiers in the sorted physical order supplied
    /// by PyJanitor. `labels` may reverse this vector for a `>`/`>=` anchor.
    pub(crate) right_index: Vec<i64>,
    /// One suffix-start boundary for each original left row.
    pub(crate) boundaries: Vec<usize>,
    /// Whether the right layout must be reversed before labels are built.
    pub(crate) reverse: bool,
}

/// Two primary region label sequences aligned by original row identifiers.
pub(crate) struct AlignedRegions {
    /// The two primary predicates after label-based alignment.
    ///
    /// Rows that cannot participate in a primary region are absent from these
    /// compact vectors. The position mappings below preserve their locations
    /// in the original arrays.
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
    /// Original left-array position for each aligned left region position.
    pub(crate) left_positions: Vec<usize>,
    /// Original right-array position for each aligned right region position.
    /// Greater-than anchors reverse this mapping for traversal.
    pub(crate) right_positions: Vec<usize>,
    /// Number of left rows before impossible region rows are removed.
    pub(crate) left_len: usize,
    /// Number of right rows before region traversal reorders the layout.
    pub(crate) right_len: usize,
}

/// Find the first right-region position satisfying `left <= right`.
///
/// Region labels are integer encodings of eligible suffixes. This hot-path
/// search receives a contiguous slice, so it can use the standard slice
/// partition-point implementation directly without constructing an ndarray
/// view for every left query.
fn first_eligible(values: &[i64], left: i64) -> usize {
    // `values` is monotonic. The first value that is not less than `left` is
    // the beginning of the eligible right suffix.
    values.partition_point(|value| *value < left)
}

/// Convert each left row into a first-region query and order the queries for
/// the paper sweep.
///
/// # Arguments
///
/// * `regions` - Two primary region paths already aligned by original labels.
///   `regions.right_first` must be monotonic nondecreasing.
///
/// # Returns
///
/// A vector of `(right_start, left_position)` pairs sorted by descending
/// `right_start`. The left position is a compact region position, not a
/// source-array position.
///
/// # Complexity
///
/// Building the query vector is `O(left_rows log right_rows)` because each
/// query uses the existing binary-search window helper. Sorting costs
/// `O(left_rows log left_rows)`.
pub(crate) fn sweep_queries(regions: &AlignedRegions) -> Vec<(usize, usize)> {
    // A query is `(first_region_start, left_position)`. The start is a
    // position in the physical right layout, not a dataframe label. For
    // example, a start of 3 means “right positions 3..right_len may satisfy
    // the first anchor”; it says nothing about the right row's index value.
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
    queries.sort_unstable_by_key(|left| std::cmp::Reverse(left.0));
    queries
}

/// Visit every candidate reported by the two aligned primary regions.
///
/// This is the paper's right-region sweep. The first region determines the
/// earliest eligible right position for each left row. Queries are processed
/// from the largest boundary to the smallest boundary, so each right row is
/// added to the active map exactly once. The map is ordered by the second
/// region label; its range therefore applies the second primary predicate.
/// Duplicate second labels remain separate because each map value is a linked
/// chain of physical right positions.
///
/// Keeping this traversal in one place is important: index construction and
/// aggregation must agree on the boundary guard, duplicate handling, and
/// sentinel-chain termination. Their callbacks differ only in what they do
/// after a candidate has survived both primary regions.
///
/// # Arguments
///
/// * `regions` - Aligned compact region paths. `right_first` is monotonic;
///   `right_second` may contain duplicates and need not be monotonic.
/// * `visit` - Called with compact left and right region positions for every
///   candidate satisfying both primary inequalities. Return `Ok(false)` to
///   stop visiting candidates for the current left row; `Keep::Any` uses this
///   to stop after its first passing candidate. Return `Err` to abort the
///   entire sweep immediately.
///
/// # Errors
///
/// Returns an error if a sweep boundary is outside the right layout or is
/// greater than the previous boundary. The latter would cause a previously
/// linked right slice to be inserted twice and could create a cyclic chain.
pub(crate) fn traverse_candidates<F>(regions: &AlignedRegions, mut visit: F) -> Result<(), String>
where
    F: FnMut(usize, usize) -> Result<bool, String>,
{
    let queries = sweep_queries(regions);
    let mut active = BTreeMap::<i64, GroupState>::new();
    // `next` is a linked-list tape. Each right position points to the next
    // position with the same second-region label; -1 means chain end.
    let mut next = vec![-1_i64; regions.right_index.len()];
    let mut previous_end = regions.right_index.len();

    for (start, left_position) in queries {
        // `checked_region_start` validates both the array boundary and the
        // descending-order assumption required by `add_right_region`.
        let start = checked_region_start(
            i64::try_from(start).map_err(|_| "region start exceeds i64 capacity")?,
            regions.right_index.len(),
            previous_end,
        )?;
        let Some(start) = start else {
            // No right row can satisfy the first primary inequality for this
            // left row. There is no candidate to visit.
            continue;
        };
        add_right_region(
            &regions.right_second,
            start,
            previous_end,
            &mut next,
            &mut active,
        );
        previous_end = start;

        // Only groups at or above the left second-region label satisfy the
        // second inequality. Walk every duplicate in each qualifying chain.
        // ELI5: the B-tree tells us which labelled buckets qualify; the
        // linked list inside each bucket tells us which individual right rows
        // belong to that bucket. We need both because labels can repeat.
        'candidate_groups: for (_, state) in active.range(regions.left_second[left_position]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if !visit(left_position, right_position)? {
                    break 'candidate_groups;
                }
                position = next[right_position];
            }
        }
    }
    Ok(())
}

/// Materialize an `all` result directly from two sweeps.
///
/// # Arguments
///
/// * `regions` - Aligned primary regions.
/// * `predicates_pass` - Callback receiving compact region positions and
///   returning whether the candidate survives residual predicates. The exact
///   two-range path supplies a callback that always returns `true`.
///
/// # Returns
///
/// Two equal-length vectors of original left and right labels. Left rows are
/// emitted in canonical left order. Right-row order is the order produced by
/// the region traversal and is not promised for non-monotonic second paths.
/// Empty vectors mean that no pair passed both primary inequalities.
///
/// # Errors
///
/// Returns an error if the number of matching pairs cannot be represented by
/// the platform's `usize` capacity.
///
/// # Algorithm
///
/// The first pass counts matches per left row and computes prefix offsets. The
/// second pass repeats the same sweep and writes directly into the final
/// buffers. This avoids storing a nested candidate vector and guarantees that
/// allocation happens only after the exact result size is known.
///
/// The first sweep counts matches per left row and computes offsets. The
/// second sweep writes directly into the final index buffers, so no candidate
/// positions buffer is needed.
fn build_all_indices<P>(
    regions: &AlignedRegions,
    mut predicates_pass: P,
) -> Result<(Vec<i64>, Vec<i64>), String>
where
    P: FnMut(usize, usize) -> bool,
{
    let mut counts = vec![0_usize; regions.left_index.len()];
    // First pass: count final output pairs. Applying the callback here keeps
    // exact and extended paths on identical two-pass semantics.
    traverse_candidates(regions, |left_position, right_position| {
        if predicates_pass(left_position, right_position) {
            counts[left_position] = counts[left_position]
                .checked_add(1)
                .ok_or("region index result size exceeds platform capacity")?;
        }
        Ok(true)
    })?;

    let mut offsets = vec![0_usize; counts.len() + 1];
    // `offsets[row]..offsets[row + 1]` is the final output bucket for one
    // compact left row. The sweep visits rows by boundary order, so these
    // buckets are what restore the caller's canonical left-row order.
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
    // Each cursor starts at its row's bucket and advances only within that
    // bucket. Therefore no sorting is needed after the second sweep.
    // Second pass: write directly into the final output arrays. The cursor
    // for each left row starts at that row's offset and advances independently.
    traverse_candidates(regions, |left_position, right_position| {
        if predicates_pass(left_position, right_position) {
            let output_position = cursors[left_position];
            left_index[output_position] = regions.left_index[left_position];
            right_index[output_position] = regions.right_index[right_position];
            cursors[left_position] += 1;
        }
        Ok(true)
    })?;
    Ok((left_index, right_index))
}

/// Build an extended `Keep::All` result with one candidate sweep.
///
/// The exact two-range path uses [`build_all_indices`] because its second pass
/// only repeats cheap region traversal and array writes. Extended joins may
/// evaluate null masks, strings, and several residual predicates for every
/// candidate. Repeating that work would evaluate the residual callback twice.
///
/// This builder therefore records only the candidates that pass the residual
/// callback during one sweep. Counts and offsets still provide the final
/// canonical left-row order, while the flat pair buffer replaces the second
/// B-tree/linked-list traversal.
///
/// # Errors
///
/// Returns an error if the sweep boundary invariant is violated, the match
/// count overflows platform capacity, or the intermediate pair buffer cannot
/// reserve capacity.
fn build_all_indices_extended<P>(
    regions: &AlignedRegions,
    mut predicates_pass: P,
) -> Result<(Vec<i64>, Vec<i64>), String>
where
    P: FnMut(usize, usize) -> bool,
{
    let mut counts = vec![0_usize; regions.left_index.len()];
    let mut passing_pairs = Vec::<(usize, usize)>::new();

    // One sweep performs both primary-region traversal and residual filtering.
    // Store only passing candidates; rejected candidates never occupy the
    // intermediate buffer and never reach the output-sizing phase.
    traverse_candidates(regions, |left_position, right_position| {
        if predicates_pass(left_position, right_position) {
            counts[left_position] = counts[left_position]
                .checked_add(1)
                .ok_or("region index result size exceeds platform capacity")?;
            if passing_pairs.len() == passing_pairs.capacity() {
                passing_pairs
                    .try_reserve(1)
                    .map_err(|_| "region index result allocation failed".to_owned())?;
            }
            passing_pairs.push((left_position, right_position));
        }
        Ok(true)
    })?;

    if passing_pairs.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Prefix offsets assign each left row a contiguous output bucket. This is
    // what restores canonical left order after the sweep's descending query
    // order, without sorting the completed output pairs.
    let mut offsets = vec![0_usize; counts.len() + 1];
    for (left_position, count) in counts.iter().copied().enumerate() {
        offsets[left_position + 1] = offsets[left_position]
            .checked_add(count)
            .ok_or("region index result size exceeds platform capacity")?;
    }
    let total = offsets[regions.left_index.len()];
    let mut left_index = vec![0_i64; total];
    let mut right_index = vec![0_i64; total];
    let mut cursors = offsets[..regions.left_index.len()].to_vec();

    // This pass touches only the compact passing-pair buffer. It performs no
    // region-map updates, linked-list traversal, or residual predicate calls.
    for (left_position, right_position) in passing_pairs {
        let output_position = cursors[left_position];
        left_index[output_position] = regions.left_index[left_position];
        right_index[output_position] = regions.right_index[right_position];
        cursors[left_position] += 1;
    }
    Ok((left_index, right_index))
}

/// Materialize `first`, `last`, or `any` results in one sweep.
///
/// # Arguments
///
/// * `regions` - Aligned primary region paths.
/// * `keep` - Selection mode applied only after residual predicates pass.
/// * `predicates_pass` - Residual predicate callback over compact region
///   positions.
///
/// # Returns
///
/// At most one original-index pair per left row. `first` and `last` compare
/// original right labels, while `any` stops at the first passing traversal
/// candidate.
///
/// # Errors
///
/// Returns an error if the sweep boundary sequence violates the shared
/// `add_right_region` ordering invariant.
fn build_selected_indices<P>(
    regions: &AlignedRegions,
    keep: Keep,
    mut predicates_pass: P,
) -> Result<(Vec<i64>, Vec<i64>), String>
where
    P: FnMut(usize, usize) -> bool,
{
    // A selected result needs at most one physical right position per left
    // row, so this vector is much smaller than an all-match result.
    let mut selected = vec![None; regions.left_index.len()];

    // Filter candidates before applying first/last/any. This preserves the
    // contract that keep semantics see only complete predicate matches.
    traverse_candidates(regions, |left_position, right_position| {
        if !predicates_pass(left_position, right_position) {
            return Ok(true);
        }
        match keep {
            Keep::Any if selected[left_position].is_none() => {
                // Returning `Ok(false)` stops the shared traversal for this
                // left row only. The outer sweep still processes every other
                // left row.
                selected[left_position] = Some(right_position);
                Ok(false)
            }
            Keep::First => {
                // “First” means the smallest original right label, not the
                // first physical candidate encountered in the non-monotonic
                // second-region path.
                selected[left_position] = match selected[left_position] {
                    None => Some(right_position),
                    Some(current)
                        if regions.right_index[right_position] < regions.right_index[current] =>
                    {
                        Some(right_position)
                    }
                    Some(current) => Some(current),
                };
                Ok(true)
            }
            Keep::Last => {
                // “Last” follows the same rule in the opposite direction:
                // compare original labels, not traversal order.
                selected[left_position] = match selected[left_position] {
                    None => Some(right_position),
                    Some(current)
                        if regions.right_index[right_position] > regions.right_index[current] =>
                    {
                        Some(right_position)
                    }
                    Some(current) => Some(current),
                };
                Ok(true)
            }
            Keep::All => unreachable!("selected builder cannot receive Keep::All"),
            Keep::Any => Ok(true),
        }
    })?;

    let mut left_index = Vec::with_capacity(selected.len());
    let mut right_index = Vec::with_capacity(selected.len());
    // The sweep order is not left-row order, so materialize selected rows by
    // walking `selected` in canonical left order.
    for (left_position, right_position) in selected.into_iter().enumerate() {
        if let Some(right_position) = right_position {
            left_index.push(regions.left_index[left_position]);
            right_index.push(regions.right_index[right_position]);
        }
    }
    Ok((left_index, right_index))
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
/// The result is empty when no primary pair matches.
pub(crate) fn build_indices(
    regions: &AlignedRegions,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if keep == Keep::All {
        return build_all_indices(regions, |_, _| true);
    }
    build_selected_indices(regions, keep, |_, _| true)
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
///   pass for compact region positions. Use a callback that always returns
///   `true` when there are no residual predicates. The callback is invoked
///   after primary-region filtering and before `keep` selection.
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
        build_selected_indices(regions, keep, predicates_pass)
    }
}

/// Build one monotonic boundary sequence from a typed primary anchor.
///
/// For `<` and `<=`, each left value produces the beginning of an eligible
/// right suffix. For `>` and `>=`, [`range_window`] returns the number of
/// eligible values in a right-hand prefix. Subtracting that count from the
/// right length produces the suffix start in the reversed right layout. The
/// binary search is delegated to [`range_window`], so the four operator
/// variants retain the same boundary semantics as the existing range kernels.
///
/// # Arguments
///
/// * `anchor` - A parsed primary anchor whose right values are sorted in
///   ascending order and whose operator is `<`, `<=`, `>`, or `>=`.
///
/// The index arrays in `anchor` are expected to be parallel to their value
/// arrays. That contract is validated by the shared range-predicate parser
/// before this function is called. PyJanitor also guarantees that the index
/// labels are unique; alignment relies on that guarantee when it creates its
/// lookup maps.
///
/// # Returns
///
/// A [`RegionBoundary`] containing the original IDs, one binary-search
/// boundary per left row, and the traversal orientation. The identifiers are
/// copied into owned vectors deliberately: the returned object outlives the
/// borrowed NumPy views used during the search and is subsequently consumed by
/// `labels`, which may remove rows or reverse the right layout.
///
/// # Errors
///
/// Returns an error for `==` or `!=`, because those operators do not define a
/// monotonic region boundary.
fn region_boundaries(anchor: &AnyParsedRangePredicate<'_>) -> Result<RegionBoundary, String> {
    // Keep this validation at the boundary-construction function as well as
    // at the public parser boundary. The vectors below are parallel: labels
    // indexes `left_index[position]` for every boundary and later alignment
    // pairs each right label with a region value. A direct internal caller
    // must therefore receive a normal error rather than a panic or silently
    // shifted label pairing.
    anchor.validate_lengths()?;
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
            let right_len = predicate.right.as_array().len();
            for value in predicate.left.as_array().iter() {
                // `range_window` performs the typed binary search. Less-than
                // anchors already return a suffix start. Greater-than
                // anchors return an eligible prefix length; convert it to
                // the suffix start in the reversed right layout now. This
                // gives `labels` one uniform boundary convention:
                // boundary 0 means all right rows, and boundary right_len
                // means no right rows.
                let (start, end) = range_window(*value, predicate.right.as_array(), predicate.op);
                boundaries.push(if reverse { right_len - end } else { start });
            }
            // The parsed arrays are borrowed Python/NumPy views. Copy only
            // the identifier metadata here so the downstream region structs
            // can compact and reorder it without carrying Python lifetimes.
            // The value array itself is not copied: `range_window` searches
            // it directly, and only the resulting boundary positions survive.
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

/// Convert one boundary sequence into left/right region numbers and source
/// position mappings.
///
/// # Arguments
///
/// * `boundary` - The boundary sequence produced by [`region_boundaries`].
///   Its right IDs and boundary orientation are consumed.
///
/// # Returns
///
/// A [`LabeledRegions`] value with empty-window left rows removed,
/// reverse-oriented right regions normalized for the sweep, and mappings back
/// to the original value-array positions.
struct LabeledRegions {
    left_index: Vec<i64>,
    left_region: Vec<i64>,
    left_positions: Vec<usize>,
    right_index: Vec<i64>,
    right_region: Vec<i64>,
    right_positions: Vec<usize>,
    left_len: usize,
    right_len: usize,
}

///
/// # How labels are built
///
/// `boundaries` is treated as a set of suffix starts. An increment at position
/// `p` means that the right-side region number changes starting at `p`.
/// Cumulative summation turns those increments into one label per physical
/// right row. A left row reads the label at its boundary. A boundary equal to
/// the right length describes an empty suffix and is omitted from the compact
/// left path.
///
/// `region_boundaries` has already normalized greater-than anchors into this
/// same representation: a binary-search prefix length `p` becomes the
/// reversed-layout suffix start `right_len - p`, and `right_index` is reversed
/// before this function builds labels. Consequently, this function has one
/// boundary rule for all four inequality operators. `right_positions` records
/// that reversed position `0` came from original source position
/// `right_len - 1`.
fn labels(boundary: RegionBoundary) -> LabeledRegions {
    let RegionBoundary {
        left_index: original_left_index,
        mut right_index,
        boundaries,
        reverse,
    } = boundary;
    let right_len = right_index.len();
    let mut right_positions = (0..right_len).collect::<Vec<_>>();
    if reverse {
        // `region_boundaries` already converted the greater-than prefix
        // length into a suffix start. Here we only reverse the physical
        // identifiers and their source-position map so that the normalized
        // boundary refers to the reversed layout.
        //
        // In the shared convention, boundary 0 means "all right rows" and
        // boundary right_len means "no right rows". Having this convention
        // before the cumulative pass avoids a separate reverse-anchor branch
        // in every later sweep.
        right_index.reverse();
        right_positions.reverse();
    }
    // Each boundary marks where one left row's eligible right suffix begins.
    // Difference-style increments let us construct all right region labels
    // in one cumulative pass instead of visiting every right row once per
    // left row. If several left rows have the same boundary, their increments
    // naturally accumulate at the same position.
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
    let mut left_index = Vec::with_capacity(boundaries.len());
    let mut left_region = Vec::with_capacity(boundaries.len());
    let mut left_positions = Vec::with_capacity(boundaries.len());
    for (position, boundary) in boundaries.iter().enumerate() {
        // A boundary at or beyond the right length has no matching right
        // position, so that left row is removed before alignment.
        if *boundary < right_region.len() {
            left_index.push(original_left_index[position]);
            left_region.push(right_region[*boundary]);
            left_positions.push(position);
        }
    }
    let left_len = original_left_index.len();
    // `sweep_queries` uses binary search over the first right-region path.
    // This is the invariant that makes that search valid for all four
    // inequality operators, including the normalized greater-than paths.
    // This is a correctness-critical invariant: `sweep_queries` performs a
    // binary search over this path. Keep the assertion in release builds so
    // an internal construction bug cannot silently produce wrong matches.
    assert!(
        right_region.windows(2).all(|pair| pair[0] <= pair[1]),
        "right region path must be monotonic nondecreasing"
    );
    LabeledRegions {
        left_index,
        left_region,
        left_positions,
        right_index,
        right_region,
        right_positions,
        left_len,
        right_len,
    }
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
/// The first anchor supplies canonical traversal order. The second anchor is
/// looked up by original identifier and contributes only its corresponding
/// region labels. The returned source-position mappings always refer to the
/// first anchor's original value layout.
///
/// # Errors
///
/// Empty aligned sides are returned as empty vectors; they represent a valid
/// no-match result rather than a malformed join. Malformed parallel arrays
/// are rejected earlier by [`AnyParsedRangePredicate::validate_lengths`].
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
    // second anchor for every first-anchor row. The maps are lookup tables,
    // not a second ordering operation: the first anchor remains the canonical
    // output/traversal order, and the second anchor supplies labels by ID.
    //
    // PyJanitor guarantees unique index labels. Consequently each ID has one
    // position in these maps; this function intentionally trusts that public
    // contract rather than re-validating uniqueness in Rust.
    let mut second_left = HashMap::with_capacity(second.left_index.len());
    for (position, id) in second.left_index.iter().enumerate() {
        second_left.insert(*id, position);
    }
    let mut second_right = HashMap::with_capacity(second.right_index.len());
    for (position, id) in second.right_index.iter().enumerate() {
        second_right.insert(*id, position);
    }
    // Reserve the first anchor's sizes as upper bounds. Either anchor may have
    // removed rows, so the final vectors can be smaller, but they cannot be
    // larger than the first anchor's compact vectors.
    let mut output = AlignedRegions {
        left_index: Vec::with_capacity(first.left_index.len()),
        right_index: Vec::with_capacity(first.right_index.len()),
        left_first: Vec::with_capacity(first.left_index.len()),
        left_second: Vec::with_capacity(first.left_index.len()),
        right_first: Vec::with_capacity(first.right_index.len()),
        right_second: Vec::with_capacity(first.right_index.len()),
        left_positions: Vec::with_capacity(first.left_index.len()),
        right_positions: Vec::with_capacity(first.right_index.len()),
        left_len: first.left_len,
        right_len: first.right_len,
    };
    // Keep the first anchor's left order as the canonical left-row order.
    // Look up the corresponding second region using the original left ID.
    // The source-position mapping comes from the first anchor because residual
    // predicates must read the original left value at that position.
    for (position, id) in first.left_index.iter().enumerate() {
        if let Some(&other) = second_left.get(id) {
            output.left_index.push(*id);
            output.left_first.push(first.left_region[position]);
            output.left_second.push(second.left_region[other]);
            output.left_positions.push(first.left_positions[position]);
        }
    }
    // Keep the first anchor's right order as the canonical physical right
    // layout. The second region is reordered to that same layout by ID. This
    // is why `right_second` need not be monotonic: it is the second anchor's
    // labels projected onto the first anchor's physical order. The sweep uses
    // the first path for its monotonic query boundary and handles this second
    // path with the ordered active structure.
    for (position, id) in first.right_index.iter().enumerate() {
        if let Some(&other) = second_right.get(id) {
            output.right_index.push(*id);
            output.right_first.push(first.right_region[position]);
            output.right_second.push(second.right_region[other]);
            output.right_positions.push(first.right_positions[position]);
        }
    }
    Ok(output)
}

/// Build aligned regions from two anchors that were already parsed by a
/// caller-specific boundary parser.
///
/// Aggregation callers use this entry point so their named aggregation anchor
/// is parsed exactly once. Index callers can continue using
/// [`parse_and_align`], which owns parsing of the public five-field tuples.
pub(crate) fn align_parsed(
    first: &AnyParsedRangePredicate<'_>,
    second: &AnyParsedRangePredicate<'_>,
) -> Result<AlignedRegions, String> {
    align(region_boundaries(first)?, region_boundaries(second)?)
}

/// Parse and align the first two predicates in a dual or multi-predicate join.
///
/// This is the region boundary between Python-shaped data and the internal
/// sweep representation. It parses only the first two predicates; later
/// predicates are deliberately left to the residual matcher so they may use
/// any supported operator, including equality and inequality operators that
/// cannot define monotonic regions.
///
/// # Arguments
///
/// * `predicates` - A list whose first two entries are five-element primary
///   range anchors. Later entries are residual predicates and are not parsed
///   here.
///
/// # Errors
///
/// Returns a Python `ValueError` when fewer than two anchors are supplied, a
/// value/index pair is malformed, an anchor is malformed, or an anchor uses
/// `==`/`!=`.
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
    let first = parse_any_range_predicate(first, true)?;
    let second = parse_any_range_predicate(second, true)?;
    align(
        region_boundaries(&first).map_err(PyValueError::new_err)?,
        region_boundaries(&second).map_err(PyValueError::new_err)?,
    )
    .map_err(PyValueError::new_err)
}

/// Build index pairs from aligned regions and optional residual predicates.
///
/// The first two predicates have already been consumed by `parse_and_align`.
/// This helper copies the remaining Python tuples into the shared predicate
/// parser, translates compact region positions back to source positions for
/// residual evaluation, and delegates `keep` behavior to the region builders.
///
/// The parser receives the original Python list with an offset of two rather
/// than a copied residual list. This keeps the Python/Rust boundary positional:
/// the first two tuples remain region anchors, and every later tuple remains a
/// residual without an intermediate Python allocation. Residual arrays are
/// required to cover the full physical left and right layouts because the
/// callback indexes them with `AlignedRegions::left_positions` and
/// `AlignedRegions::right_positions`.
fn build_regions_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: Keep,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let regions = parse_and_align(predicates)?;

    // Parse only predicates after the two anchors. The offset-aware parser
    // avoids copying residual tuples into a temporary Python list; residuals
    // are evaluated against the already-aligned physical rows.
    let (parsed, metadata) = parse_predicates_with_nulls_strings_from(py, predicates, 2)?;
    // Residual arrays must have the same physical left/right lengths as the
    // aligned regions because their positions are used directly in the hot
    // traversal loop.
    check_predicate_lengths(&parsed, regions.left_len, regions.right_len)?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    // The matcher closure is called only for primary candidates. `keep` is
    // applied by traversal after this closure has exhausted all residuals.
    let (left_index, right_index) = build_indices_extended(&regions, keep, |left, right| {
        predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            regions.left_positions[left],
            regions.right_positions[right],
        )
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
    m.add_function(wrap_pyfunction!(region_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(region_aggregate_reverse, m)?)?;
    m.add_function(wrap_pyfunction!(region_extended_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(region_extended_aggregate_reverse, m)?)?;
    Ok(())
}

/// The two named region anchors parsed at the Python/Rust boundary.
struct PreparedPredicates<'py> {
    /// First anchor, including optional output-position maps.
    first: ParsedAggregationRangeAnchor<'py>,
    /// Second anchor, which uses the five-field region form.
    second: AnyParsedRangePredicate<'py>,
}

/// Parse the two region aggregation anchors once and expose named Rust fields.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used while borrowing the tuples.
/// * `predicates` - Full predicate list. The first tuple must use either the
///   six-field form or the established eight-field form; the second tuple
///   must use the five-field region form.
///
/// # Returns
///
/// A named anchor containing the parsed range predicate, operator, and output
/// maps. NumPy arrays are borrowed; no column values are copied.
///
/// # Errors
///
/// Returns `ValueError` for invalid tuple length, a non-boolean ordering flag,
/// invalid output maps, unsupported dtypes/operators, or mismatched
/// value/index lengths.
///
/// Parsing is intentionally separate from residual parsing. The first two
/// tuples define the paper sweep and must be parsed before any later tuple is
/// interpreted; later tuples may use operators such as `==` or `!=` that are
/// valid residual filters but cannot define a region boundary.
///
/// The returned request keeps the first anchor's output maps alongside its
/// typed range data and keeps the second typed range separately. Later
/// residual predicates remain in the original Python list and are parsed by
/// the shared residual parser only for extended aggregation.
fn parse_region_anchors<'py>(predicates: &Bound<'py, PyList>) -> PyResult<PreparedPredicates<'py>> {
    let first_item = predicates.get_item(0)?;
    let first = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second = second_item.cast::<PyTuple>()?;
    let parsed_first = parse_aggregation_range_anchor(first, true)?;
    let parsed_second = parse_aggregation_range_anchor(second, false)?;
    // Keep the first anchor's output metadata attached to its range. The
    // second anchor contributes only a region path; it must not overwrite the
    // canonical first-anchor physical layout during alignment.
    Ok(PreparedPredicates {
        first: parsed_first,
        second: parsed_second.range,
    })
}

/// Describe the output and source layouts for one aggregation direction.
///
/// Region positions are compact traversal coordinates. The optional map is
/// only output metadata; it does not change the physical source positions
/// used by [`AggregationSet::update`]. This remains local because returning
/// a borrowed view from a shared anchor method would borrow the entire named
/// anchor and prevent moving its independent typed range into dispatch.
fn aggregation_layout<'py>(
    first: &'py ParsedAggregationRangeAnchor<'py>,
    reverse: bool,
) -> (Option<ArrayView1<'py, i64>>, usize, usize) {
    // The ordering flag is validated by the parser and intentionally does not
    // affect region traversal: PyJanitor supplies the sorted right layout.
    let _ordering_flag_was_validated = first.ordered;
    let output_positions = if reverse {
        first
            .right_output_positions
            .as_ref()
            .map(|values| values.as_array())
    } else {
        first
            .left_output_positions
            .as_ref()
            .map(|values| values.as_array())
    };
    // Forward aggregation has one slot per left source row and reads right
    // values. Reverse aggregation swaps those roles. A supplied map changes
    // only the labels returned to Python; the accumulator remains indexed by
    // the compact output layout supplied by PyJanitor.
    let output_len = output_positions
        .map(|values| values.len())
        .unwrap_or(if reverse {
            first.range.right_len()
        } else {
            first.range.left_len()
        });
    let source_len = if reverse {
        first.range.left_len()
    } else {
        first.range.right_len()
    };
    (output_positions, output_len, source_len)
}

/// Parse and align the two primary region anchors once for an aggregation call.
///
/// Exact and extended aggregation intentionally keep separate candidate
/// consumers, because only the extended path evaluates residual predicates.
/// They should nevertheless share this setup: parsing the first anchor twice
/// or calculating output/source lengths differently would make the two public
/// entry points disagree on malformed input or mapped output layouts.
///
/// # Arguments
///
/// * `predicates` - Predicate list whose first two entries are region anchors.
/// # Returns
///
/// The parsed anchors and aligned region paths. Output/source layout is
/// calculated by each caller after this function returns so a borrowed output
/// map remains tied to the parsed anchor rather than to a temporary tuple.
fn prepare_region_aggregation<'py>(
    predicates: &Bound<'py, PyList>,
) -> PyResult<(PreparedPredicates<'py>, crate::regions::AlignedRegions)> {
    let prepared = parse_region_anchors(predicates)?;
    // Alignment happens once, before exact and extended consumers diverge.
    // This is what guarantees that residual predicates and aggregations see
    // the same compact-to-source position maps.
    let regions = crate::regions::align_parsed(&prepared.first.range, &prepared.second)
        .map_err(PyValueError::new_err)?;
    Ok((prepared, regions))
}

/// Execute exact dual-region aggregation and build the standard Python result.
///
/// Only the first two predicates are accepted. Both are converted into region
/// labels, and every candidate surviving both labels updates the aggregation
/// set. There is no residual predicate phase on this path.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used for parsing and result
///   construction.
/// * `predicates` - Exactly two inequality region anchors. The first anchor
///   may include compact output-position maps.
/// * `aggregations` - Non-empty aggregation requests.
/// * `return_matched` - Whether to include the matched output mask.
/// * `reverse` - Whether left values are aggregated into right output slots.
///
/// # Returns
///
/// `None` when no candidate updates the aggregation state; otherwise the
/// standard aggregation result tuple.
///
/// # Errors
///
/// Returns a Python error for invalid predicate counts, malformed anchors,
/// invalid aggregation requests, or invalid output metadata.
#[allow(clippy::too_many_arguments)]
fn aggregate_regions_exact<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // This exact path is intentionally separate from the extended path. With
    // no residual predicates, the sweep can update aggregation state directly
    // for every candidate and does not need to build predicate views.
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "region aggregation requires exactly two predicates",
        ));
    }

    let (prepared, regions) = prepare_region_aggregation(predicates)?;
    let (output_positions, output_len, source_len) = aggregation_layout(&prepared.first, reverse);
    if regions.left_index.is_empty() || regions.right_index.is_empty() {
        return Ok(None);
    }

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    // The optional maps label output slots. They do not replace the source
    // mappings in `regions`, which are needed when labels compact or reverse
    // the physical traversal layout.
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    // The shared traversal supplies compact region coordinates. Translate
    // them back to source coordinates only at the accumulator boundary.
    crate::regions::traverse_candidates(&regions, |left_position, right_position| {
        // `left_position` and `right_position` are compact region positions.
        // Convert them through the source maps exactly once, at the point
        // where an aggregation event is recorded.
        if reverse {
            set.update(
                regions.left_positions[left_position],
                regions.right_positions[right_position],
            );
        } else {
            set.update(
                regions.right_positions[right_position],
                regions.left_positions[left_position],
            );
        }
        Ok(true)
    })
    .map_err(PyValueError::new_err)?;

    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?))
}

/// Execute extended region aggregation and build the standard Python result.
///
/// This is the implementation behind the two extended Python functions:
///
/// ```text
/// region_extended_aggregate           residual filters, forward
/// region_extended_aggregate_reverse   residual filters, reverse
/// ```
///
/// `regions` are built from the first two predicates. Predicates after those
/// two anchors are evaluated for each primary candidate before aggregation.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used for parsing inputs and
///   constructing the result tuple.
/// * `predicates` - At least two region anchors, followed by optional residual
///   predicates. The first anchor may contain output-position maps in its
///   eight-field form.
/// * `aggregations` - Non-empty aggregation requests accepted by
///   `AggregationSet`, such as sum, min, max, product, size, or count.
/// * `return_matched` - Whether the result includes a boolean matched array.
/// * `reverse` - When false, aggregate right-side source values into left
///   output slots. When true, aggregate left-side source values into right
///   output slots.
///
/// # Returns
///
/// Returns `None` when no pair reaches an aggregation update. Otherwise
/// returns the established aggregation tuple: output positions, optionally
/// the matched mask, and the requested aggregation arrays.
///
/// # Errors
///
/// Returns a Python error for malformed anchors, invalid residual predicates,
/// misaligned predicate lengths, empty aggregation requests, invalid
/// aggregation inputs, or invalid output maps.
#[allow(clippy::too_many_arguments)]
fn aggregate_regions_extended<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // The extended path keeps the same region sweep as the exact path, but
    // inserts a residual-filter step between candidate discovery and the
    // aggregation update. This is why it cannot use the exact path's direct
    // update loop.
    // This implementation serves the two extended Python entry points.
    // `reverse` changes which side supplies values; residual predicates are
    // always part of this path.
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "region aggregation requires at least two predicates",
        ));
    }

    // Region construction reads only the first two anchors. It aligns their
    // left and right rows by original index labels, so the two independently
    // built region paths can be traversed together safely.
    let (prepared, regions) = prepare_region_aggregation(predicates)?;
    let (output_positions, output_len, source_len) = aggregation_layout(&prepared.first, reverse);
    if regions.left_index.is_empty() || regions.right_index.is_empty() {
        // At least one anchor has no surviving aligned rows. There can be no
        // aggregation event, so return the same no-match result as other
        // aggregation kernels.
        return Ok(None);
    }

    // Only predicates after the two region anchors are residual filters.
    // `residuals` also preserves nullable `!=` metadata for the shared
    // predicate matcher.
    let (parsed, metadata) = residuals(py, predicates, false, true)?;
    // Residual arrays retain the original source layout, including rows that
    // were removed from the compact region path. Validate against the source
    // lengths, then translate each compact candidate through the mappings
    // before invoking the shared matcher.
    check_predicate_lengths(&parsed, regions.left_len, regions.right_len)?;
    // Parse aggregation requests once before entering the sweep. The set
    // owns output accumulators while borrowing the source NumPy arrays.
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    // An eight-field first anchor carries compact output labels. These labels
    // are not the physical-to-local permutation used by `!=` aggregation.
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    // Convert parsed residual predicates into cheap Rust-side views once.
    // The hot loop then compares physical positions without repeatedly
    // touching Python objects.
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    // The shared traversal handles primary-region candidates. This callback
    // keeps the extended-only residual filtering directly before the update.
    crate::regions::traverse_candidates(&regions, |left_position, right_position| {
        // Residual predicates use source positions, not compact region
        // positions. Filtering here means `AggregationSet::update` sees only
        // candidates that passed every predicate, including null semantics.
        let passes = predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            regions.left_positions[left_position],
            regions.right_positions[right_position],
        );
        if passes {
            if reverse {
                set.update(
                    regions.left_positions[left_position],
                    regions.right_positions[right_position],
                );
            } else {
                set.update(
                    regions.right_positions[right_position],
                    regions.left_positions[left_position],
                );
            }
        }
        Ok(true)
    })
    .map_err(PyValueError::new_err)?;

    if set.is_empty() {
        // The region sweep may find candidates, but null masks or residual
        // filters can still reject every aggregation update.
        return Ok(None);
    }
    // Convert accumulator state into the established Python tuple shape and
    // attach the compact-to-original output map when one was supplied.
    Ok(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?))
}

/// Aggregate exactly two region predicates in the forward direction.
///
/// Every pair satisfying both inequality anchors updates the aggregation
/// state. There are no residual filter predicates on this path.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two region predicates. The first may use the
///   six-field form or the eight-field form with output-position maps.
/// * `aggregations` - Non-empty aggregation requests over right-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no pair matches; otherwise the standard aggregation result
/// tuple aligned to the left output layout.
///
/// # Errors
///
/// Returns an error when the predicate count, anchor shapes, lengths, or
/// aggregation requests are invalid.
#[pyfunction]
pub fn region_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_exact(py, predicates, aggregations, return_matched, false)
}

/// Aggregate exactly two region predicates in the reverse direction.
///
/// This uses the same two region anchors as forward aggregation, but treats
/// left-side values as the aggregation source and right-side rows as output
/// slots.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two region predicates.
/// * `aggregations` - Non-empty aggregation requests over left-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no pair matches; otherwise the standard aggregation result
/// tuple aligned to the right output layout.
///
/// # Errors
///
/// Returns an error when the predicate count, anchor shapes, lengths, or
/// aggregation requests are invalid.
#[pyfunction]
pub fn region_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_exact(py, predicates, aggregations, return_matched, true)
}

/// Aggregate two region predicates followed by residual filter predicates.
///
/// The first two predicates perform the efficient region sweep. Every later
/// predicate is evaluated against the physical left/right candidate before
/// the right-side source value updates a left-side aggregation slot.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two predicates: two inequality region anchors
///   followed by zero or more residual filters.
/// * `aggregations` - Non-empty aggregation requests over right-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no candidate survives the region anchors and residual filters;
/// otherwise the standard result tuple aligned to the left output layout.
///
/// # Errors
///
/// Returns an error for malformed predicates, misaligned arrays, unsupported
/// residual operators, or invalid aggregation requests.
#[pyfunction]
pub fn region_extended_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_extended(py, predicates, aggregations, return_matched, false)
}

/// Aggregate two region predicates plus residual filters in reverse direction.
///
/// The first two predicates define the candidate regions. Later predicates
/// filter those candidates. A passing left-side source value updates the
/// corresponding right-side aggregation slot.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two predicates: two inequality region anchors
///   followed by residual filters.
/// * `aggregations` - Non-empty aggregation requests over left-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no complete candidate survives; otherwise the standard result
/// tuple aligned to the right output layout.
///
/// # Errors
///
/// Returns an error for malformed predicates, misaligned arrays, unsupported
/// residual operators, or invalid aggregation requests.
#[pyfunction]
pub fn region_extended_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_extended(py, predicates, aggregations, return_matched, true)
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;
    use numpy::PyArray1;

    fn aggregation<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyList>> {
        let values = PyArray1::from_vec(py, values);
        let mask = PyArray1::from_vec(py, vec![false; values.len()?]);
        let request = PyTuple::new(
            py,
            [
                values.into_any(),
                mask.into_any(),
                "sum".into_pyobject(py)?.into_any(),
            ],
        )?;
        PyList::new(py, [request])
    }

    fn dual_predicates<'py>(
        py: Python<'py>,
        first_left: Vec<i64>,
        first_right: Vec<i64>,
        first_op: &str,
        second_op: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let predicates = PyList::empty(py);
        let left_index = PyArray1::from_vec(py, (0..first_left.len() as i64).collect());
        let right_index = PyArray1::from_vec(py, (0..first_right.len() as i64).collect());
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, first_left.clone()).into_any(),
                left_index.clone().into_any(),
                PyArray1::from_vec(py, first_right.clone()).into_any(),
                right_index.clone().into_any(),
                true.into_pyobject(py)?.to_owned().into_any(),
                first_op.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, first_left).into_any(),
                left_index.into_any(),
                PyArray1::from_vec(py, first_right).into_any(),
                right_index.into_any(),
                second_op.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        Ok(predicates)
    }

    fn result_parts<'py>(
        result: &Bound<'py, PyTuple>,
    ) -> PyResult<(Vec<i64>, Vec<bool>, Vec<i64>)> {
        let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
        let matched = result.get_item(1)?.extract::<Vec<bool>>()?;
        let outputs = result
            .get_item(2)?
            .cast::<PyList>()?
            .get_item(0)?
            .extract::<Vec<i64>>()?;
        Ok((positions, matched, outputs))
    }

    fn result_without_matched<'py>(
        result: &Bound<'py, PyTuple>,
    ) -> PyResult<(Vec<i64>, Vec<Vec<i64>>)> {
        let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
        let outputs_item = result.get_item(1)?;
        let outputs = outputs_item.cast::<PyList>()?;
        let outputs = outputs
            .iter()
            .map(|output| output.extract::<Vec<i64>>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok((positions, outputs))
    }

    #[test]
    fn greater_than_anchors_aggregate_forward_and_reverse() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1, 3], vec![1, 2, 3, 4], ">=", "<=")?;

            let forward_inputs = aggregation(py, vec![10, 20, 30, 40])?;
            let forward = region_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("greater-than forward aggregation should match");
            assert_eq!(
                result_parts(&forward)?,
                (vec![0, 1], vec![true, true], vec![10, 30])
            );

            let reverse_inputs = aggregation(py, vec![100, 300])?;
            let reverse = region_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                .expect("greater-than reverse aggregation should match");
            assert_eq!(
                result_parts(&reverse)?,
                (
                    vec![0, 1, 2, 3],
                    vec![true, false, true, false],
                    vec![100, 0, 300, 0]
                )
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn reverse_anchor_boundary_zero_and_full_prefix_are_handled() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // With right values [1, 2, 3], `>=` has an empty prefix for left
            // value 0 (boundary 0) and a full prefix for left value 3
            // (boundary right_len). Both rows must retain their correct
            // semantics after the reverse layout normalization.
            let predicates = dual_predicates(py, vec![0, 3], vec![1, 2, 3], ">=", "<=")?;
            let forward_inputs = aggregation(py, vec![10, 20, 30])?;
            let forward = region_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("the full-prefix row should aggregate");
            assert_eq!(
                result_parts(&forward)?,
                (vec![0, 1], vec![false, true], vec![0, 30])
            );

            let reverse_inputs = aggregation(py, vec![100, 300])?;
            let reverse = region_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                .expect("the full-prefix row should aggregate in reverse");
            assert_eq!(
                result_parts(&reverse)?,
                (vec![0, 1, 2], vec![false, false, true], vec![0, 0, 300])
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn eight_field_anchor_preserves_maps_when_a_left_row_is_dropped() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left = PyArray1::from_vec(py, vec![4_i64, 1]);
            let left_index = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let right_index = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    left_index.clone().into_any(),
                    right.clone().into_any(),
                    right_index.clone().into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20, 30]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    left_index.into_any(),
                    right.into_any(),
                    right_index.into_any(),
                    ">=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let forward_inputs = aggregation(py, vec![10, 20, 30])?;
            let forward = region_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("mapped forward aggregation should match");
            assert_eq!(
                result_parts(&forward)?,
                (vec![100, 200], vec![false, true], vec![0, 10])
            );

            let reverse_inputs = aggregation(py, vec![4, 8])?;
            let reverse = region_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                .expect("mapped reverse aggregation should match");
            assert_eq!(
                result_parts(&reverse)?,
                (vec![10, 20, 30], vec![true, false, false], vec![8, 0, 0])
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn residuals_use_source_positions_after_reversal() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Left row zero has no `>=` candidate and is removed from the
            // compact region path. Left row one matches right source position
            // zero. The residual values distinguish source position zero from
            // its reversed traversal position three.
            let predicates = dual_predicates(py, vec![0, 1], vec![1, 2, 3, 4], ">=", "<=")?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![200_i64, 99, 98, 97]).into_any(),
                    "==".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let forward_inputs = aggregation(py, vec![10, 20, 30, 40])?;
            let forward = region_extended_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("residual forward aggregation should match");
            assert_eq!(
                result_parts(&forward)?,
                (vec![0, 1], vec![false, true], vec![0, 10])
            );

            let reverse_inputs = aggregation(py, vec![100, 200])?;
            let reverse =
                region_extended_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                    .expect("residual reverse aggregation should match");
            assert_eq!(
                result_parts(&reverse)?,
                (
                    vec![0, 1, 2, 3],
                    vec![true, false, false, false],
                    vec![200, 0, 0, 0]
                )
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn exact_aggregation_supports_multiple_operations_masks_and_no_matched_output() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1, 2], vec![1, 2], "<=", ">=")?;
            let values = PyArray1::from_vec(py, vec![2_i64, 3]);
            let nulls = PyArray1::from_vec(py, vec![false, true]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "prod".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "min".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "max".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = region_aggregate(py, &predicates, &aggregations, false)?
                .expect("the two equality pairs should aggregate");
            let (positions, outputs) = result_without_matched(&result)?;
            assert_eq!(positions, vec![0, 1]);
            assert_eq!(
                outputs,
                vec![
                    vec![2, 0],
                    vec![2, 1],
                    vec![1, 0],
                    vec![1, 1],
                    vec![0, -1],
                    vec![0, -1]
                ]
            );

            let no_match = dual_predicates(py, vec![10], vec![1, 2], "<", "<")?;
            assert!(region_aggregate(py, &no_match, &aggregations, false)?.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn aggregation_supports_unsigned_and_float_sources() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1], vec![1], "<=", ">=")?;
            let unsigned = PyArray1::from_vec(py, vec![7_u64]);
            let floats = PyArray1::from_vec(py, vec![1.5_f64]);
            let mask = PyArray1::from_vec(py, vec![false]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            unsigned.into_any(),
                            mask.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            floats.into_any(),
                            mask.into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = region_aggregate(py, &predicates, &aggregations, false)?
                .expect("the equality pair should aggregate");
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<u64>>()?, vec![7]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<f64>>()?, vec![1.5_f64]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn aggregation_rejects_malformed_residuals_without_panicking() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1], vec![1], "<=", ">=")?;
            predicates.append(PyTuple::new(py, [1_i64.into_pyobject(py)?.into_any()])?)?;
            let aggregations = aggregation(py, vec![1])?;
            assert!(region_extended_aggregate(py, &predicates, &aggregations, true).is_err());
            Ok(())
        })
        .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::types::PyDict;
    use std::cell::Cell;

    fn regions(right_second: Vec<i64>) -> AlignedRegions {
        AlignedRegions {
            left_index: vec![10, 11],
            right_index: vec![20, 21, 22, 23],
            left_first: vec![1, 2],
            left_second: vec![1, 2],
            right_first: vec![1, 2, 3, 4],
            right_second,
            left_positions: vec![0, 1],
            right_positions: vec![0, 1, 2, 3],
            left_len: 2,
            right_len: 4,
        }
    }

    #[test]
    fn every_inequality_orientation_produces_an_increasing_right_path() {
        // These are normalized suffix starts, not raw greater-than prefix
        // ends. `region_boundaries` converts a reverse prefix length `p` to
        // `right_len - p` before calling `labels`.
        //
        // Every path is now built in the same suffix coordinate system, so
        // the monotonicity assertion is testing the actual post-refactor
        // representation used by the sweep.
        for (reverse, boundaries) in [
            (false, vec![1, 3]), // <
            (false, vec![0, 2]), // <=
            (true, vec![0, 2]),  // normalized >
            (true, vec![1, 3]),  // normalized >=
        ] {
            let labeled = labels(RegionBoundary {
                left_index: vec![10, 11],
                right_index: vec![20, 21, 22, 23],
                boundaries,
                reverse,
            });
            assert!(labeled
                .right_region
                .windows(2)
                .all(|pair| pair[0] <= pair[1]));
        }
    }

    #[test]
    fn labels_keep_original_positions_with_normalized_boundaries() {
        let forward = labels(RegionBoundary {
            left_index: vec![10, 11],
            right_index: vec![20, 21, 22],
            boundaries: vec![3, 1],
            reverse: false,
        });
        assert_eq!(forward.left_positions, vec![1]);
        assert_eq!(forward.right_positions, vec![0, 1, 2]);

        // Reverse boundaries are already `right_len - prefix_length`; labels
        // only reverses the physical right layout and does not transform the
        // boundary a second time.
        let reverse = labels(RegionBoundary {
            left_index: vec![10, 11],
            right_index: vec![20, 21, 22],
            boundaries: vec![0, 2],
            reverse: true,
        });
        assert_eq!(reverse.left_positions, vec![0, 1]);
        assert_eq!(reverse.right_positions, vec![2, 1, 0]);
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
                vec![10, 10, 10, 10, 11, 11],
                // Left rows remain canonical. For `all`, right-row order is
                // not part of the contract when the second region is
                // non-monotonic; the sweep emits each active region group
                // in label order and follows its physical linked list.
                vec![21, 23, 22, 20, 23, 22]
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

    #[test]
    fn extended_all_evaluates_each_candidate_once() {
        let regions = regions(vec![1, 2, 3, 4]);
        let evaluations = Cell::new(0_usize);
        let output = build_indices_extended(&regions, Keep::All, |_, right| {
            evaluations.set(evaluations.get() + 1);
            right == 2
        })
        .expect("extended all should build");

        // The primary sweep visits seven candidates in this fixture. Only two
        // pass the residual, but the residual callback must run once for each
        // candidate rather than once during a counting pass and again during
        // output materialization.
        assert_eq!(evaluations.get(), 7);
        assert_eq!(output, (vec![10, 11], vec![22, 22]));
    }

    /// Construct the public five-field representation used by the index
    /// wrappers.  The two anchors deliberately share the same arrays here so
    /// the expected result can be checked with a small brute-force matcher.
    fn public_predicates<'py>(
        py: Python<'py>,
        left: &[i64],
        right: &[i64],
        first_operator: &str,
        second_operator: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let predicates = PyList::empty(py);
        let left_index = PyArray1::from_vec(py, (0..left.len() as i64).collect());
        let right_index = PyArray1::from_vec(py, (0..right.len() as i64).collect());
        for operator in [first_operator, second_operator] {
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, left.to_vec()).into_any(),
                    left_index.clone().into_any(),
                    PyArray1::from_vec(py, right.to_vec()).into_any(),
                    right_index.clone().into_any(),
                    operator.into_pyobject(py)?.into_any(),
                ],
            )?)?;
        }
        Ok(predicates)
    }

    struct RightLayout<'a> {
        values: &'a [i64],
        labels: &'a [i64],
    }

    /// Build two anchors whose right values are sorted independently but whose
    /// right labels appear in different physical orders. This is the shape
    /// that exercises `align`: region positions cannot be joined by vector
    /// position, so the second anchor must be projected onto the first
    /// anchor's layout by its original labels.
    fn public_predicates_with_permuted_right_layouts<'py>(
        py: Python<'py>,
        left: &[i64],
        left_index: &[i64],
        first: RightLayout<'_>,
        second: RightLayout<'_>,
        first_operator: &str,
        second_operator: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let predicates = PyList::empty(py);
        let left_values = PyArray1::from_vec(py, left.to_vec());
        let left_labels = PyArray1::from_vec(py, left_index.to_vec());
        for (values, labels, operator) in [
            (first.values, first.labels, first_operator),
            (second.values, second.labels, second_operator),
        ] {
            predicates.append(PyTuple::new(
                py,
                [
                    left_values.clone().into_any(),
                    left_labels.clone().into_any(),
                    PyArray1::from_vec(py, values.to_vec()).into_any(),
                    PyArray1::from_vec(py, labels.to_vec()).into_any(),
                    operator.into_pyobject(py)?.into_any(),
                ],
            )?)?;
        }
        Ok(predicates)
    }

    /// Read the pair arrays returned by either public index wrapper.
    fn public_pairs(result: &Bound<'_, PyDict>) -> PyResult<Vec<(i64, i64)>> {
        let left = result
            .get_item("left_index")?
            .ok_or_else(|| PyValueError::new_err("result has no left_index"))?
            .extract::<Vec<i64>>()?;
        let right = result
            .get_item("right_index")?
            .ok_or_else(|| PyValueError::new_err("result has no right_index"))?
            .extract::<Vec<i64>>()?;
        Ok(left.into_iter().zip(right).collect())
    }

    fn comparison(left: i64, right: i64, operator: &str) -> bool {
        match operator {
            "<" => left < right,
            "<=" => left <= right,
            ">" => left > right,
            ">=" => left >= right,
            "==" => left == right,
            "!=" => left != right,
            _ => panic!("unsupported test operator"),
        }
    }

    fn expected_pairs(
        left: &[i64],
        right: &[i64],
        first_operator: &str,
        second_operator: &str,
    ) -> Vec<(i64, i64)> {
        left.iter()
            .enumerate()
            .flat_map(|(left_position, &left_value)| {
                right
                    .iter()
                    .enumerate()
                    .filter_map(move |(right_position, &right_value)| {
                        (comparison(left_value, right_value, first_operator)
                            && comparison(left_value, right_value, second_operator))
                        .then_some((left_position as i64, right_position as i64))
                    })
            })
            .collect()
    }

    /// Return the reference matches grouped by left row for every keep mode.
    ///
    /// The production kernel applies `first` and `last` to the original right
    /// labels, not to the right array's physical position. `any` has no
    /// prescribed winner, so callers validate it by checking membership in
    /// the complete candidate set.
    fn expected_labeled_pairs(
        left: &[i64],
        left_index: &[i64],
        first: RightLayout<'_>,
        second: RightLayout<'_>,
        first_operator: &str,
        second_operator: &str,
    ) -> Vec<Vec<(i64, i64)>> {
        left.iter()
            .enumerate()
            .map(|(left_position, &left_value)| {
                first
                    .labels
                    .iter()
                    .enumerate()
                    .filter_map(|(first_position, &right_label)| {
                        let second_position = second
                            .labels
                            .iter()
                            .position(|label| label == &right_label)
                            .expect("the two anchors share right labels");
                        (comparison(left_value, first.values[first_position], first_operator)
                            && comparison(
                                left_value,
                                second.values[second_position],
                                second_operator,
                            ))
                        .then_some((left_index[left_position], right_label))
                    })
                    .collect()
            })
            .collect()
    }

    fn expected_for_keep(candidates: &[Vec<(i64, i64)>], keep: &str) -> Vec<(i64, i64)> {
        candidates
            .iter()
            .flat_map(|row| match keep {
                "all" => row.clone(),
                "first" => row
                    .iter()
                    .min_by_key(|(_, right_label)| *right_label)
                    .copied()
                    .into_iter()
                    .collect(),
                "last" => row
                    .iter()
                    .max_by_key(|(_, right_label)| *right_label)
                    .copied()
                    .into_iter()
                    .collect(),
                "any" => Vec::new(),
                _ => panic!("unsupported keep mode"),
            })
            .collect()
    }

    #[test]
    fn public_indices_match_bruteforce_for_all_anchor_operator_pairs() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = [0, 1, 2, 3];
            let right = [0, 1, 2, 3];
            let operators = ["<", "<=", ">", ">="];
            for first_operator in operators {
                for second_operator in operators {
                    let predicates =
                        public_predicates(py, &left, &right, first_operator, second_operator)?;
                    let result = region_indices(py, &predicates, "all")?;
                    let mut actual = result
                        .as_ref()
                        .map(public_pairs)
                        .transpose()?
                        .unwrap_or_default();
                    let mut expected =
                        expected_pairs(&left, &right, first_operator, second_operator);
                    actual.sort_unstable();
                    expected.sort_unstable();
                    assert_eq!(actual, expected, "{first_operator} then {second_operator}");
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_indices_keep_modes_align_independently_permuted_right_layouts() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = [1, 3, 5];
            let left_index = [17, 42, 99];
            // Each right value array is sorted, as required by the binary
            // search. The labels, however, are deliberately arranged
            // differently. The second anchor therefore cannot be aligned by
            // physical position; it must use the shared right labels.
            let first_right = [0, 2, 4, 6];
            let first_right_index = [101, 503, 907, 1201];
            let second_right = [0, 2, 4, 6];
            let second_right_index = [907, 101, 1201, 503];
            let operators = ["<", "<=", ">", ">="];
            let keeps = ["all", "first", "last", "any"];

            for first_operator in operators {
                for second_operator in operators {
                    let candidates = expected_labeled_pairs(
                        &left,
                        &left_index,
                        RightLayout {
                            values: &first_right,
                            labels: &first_right_index,
                        },
                        RightLayout {
                            values: &second_right,
                            labels: &second_right_index,
                        },
                        first_operator,
                        second_operator,
                    );
                    for keep in keeps {
                        let predicates = public_predicates_with_permuted_right_layouts(
                            py,
                            &left,
                            &left_index,
                            RightLayout {
                                values: &first_right,
                                labels: &first_right_index,
                            },
                            RightLayout {
                                values: &second_right,
                                labels: &second_right_index,
                            },
                            first_operator,
                            second_operator,
                        )?;
                        let actual = region_indices(py, &predicates, keep)?
                            .as_ref()
                            .map(public_pairs)
                            .transpose()?
                            .unwrap_or_default();
                        let context = format!(
                            "first={first_operator}, second={second_operator}, keep={keep}"
                        );

                        if keep == "any" {
                            let all_candidates: Vec<_> =
                                candidates.iter().flatten().copied().collect();
                            assert!(
                                actual.iter().all(|pair| all_candidates.contains(pair)),
                                "{context}: any returned a non-matching pair: {actual:?}"
                            );
                            assert!(
                                actual.iter().all(|(left_label, _)| {
                                    actual
                                        .iter()
                                        .filter(|(label, _)| label == left_label)
                                        .count()
                                        == 1
                                }),
                                "{context}: any returned more than one pair for a left row"
                            );
                            assert_eq!(
                                actual.len(),
                                candidates.iter().filter(|row| !row.is_empty()).count(),
                                "{context}"
                            );
                        } else {
                            let mut actual = actual;
                            let mut expected = expected_for_keep(&candidates, keep);
                            actual.sort_unstable();
                            expected.sort_unstable();
                            assert_eq!(actual, expected, "{context}");
                        }
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_indices_keep_modes_handle_duplicate_right_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = public_predicates(py, &[1], &[1, 1, 2], "<=", ">=")?;
            let all = region_indices(py, &predicates, "all")?
                .map(|result| public_pairs(&result))
                .transpose()?
                .unwrap();
            assert_eq!(all.len(), 2);

            let first = region_indices(py, &predicates, "first")?
                .map(|result| public_pairs(&result))
                .transpose()?
                .unwrap();
            let last = region_indices(py, &predicates, "last")?
                .map(|result| public_pairs(&result))
                .transpose()?
                .unwrap();
            let any = region_indices(py, &predicates, "any")?
                .map(|result| public_pairs(&result))
                .transpose()?
                .unwrap();
            assert_eq!(first.len(), 1);
            assert_eq!(last.len(), 1);
            assert_eq!(any.len(), 1);
            assert!(all.contains(&(0, 0)) && all.contains(&(0, 1)));
            assert!(first[0].1 == 0 || first[0].1 == 1);
            assert!(last[0].1 == 0 || last[0].1 == 1);
            assert!(any[0].1 == 0 || any[0].1 == 1);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_indices_filter_after_region_candidates() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = public_predicates(py, &[1, 2], &[1, 2, 3], "<=", ">=")?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20, 30]).into_any(),
                    "==".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let result = region_indices_extended(py, &predicates, "all")?
                .map(|result| public_pairs(&result))
                .transpose()?
                .unwrap();
            assert_eq!(result, vec![(0, 0), (1, 1)]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_indices_reject_malformed_lengths_and_residuals() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let empty = public_predicates(py, &[], &[], "<=", ">=")?;
            assert!(region_indices(py, &empty, "all")?.is_none());

            let predicates = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                        "<=".into_pyobject(py)?.into_any(),
                    ],
                )?],
            )?;
            assert!(region_indices(py, &predicates, "all").is_err());

            let predicates = public_predicates(py, &[1], &[1], "<=", ">=")?;
            predicates.append(PyTuple::new(py, [1_i64.into_pyobject(py)?.into_any()])?)?;
            assert!(region_indices_extended(py, &predicates, "all").is_err());
            Ok(())
        })
        .unwrap();
    }
}
