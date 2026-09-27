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
    let queries = sweep_queries(regions);
    let mut counts = vec![0_usize; regions.left_index.len()];
    let mut active = BTreeMap::<i64, GroupState>::new();
    let mut next = vec![-1_i64; regions.right_index.len()];
    let mut previous_end = regions.right_index.len();
    // First pass: count final output pairs. Applying the callback here keeps
    // exact and extended paths on identical two-pass semantics.
    for (start, left_position) in queries.iter().copied() {
        if start >= regions.right_index.len() {
            continue;
        }
        // Grow the active suffix exactly as in the second pass.
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
fn build_selected_indices<P>(
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
/// The result is empty when no primary pair matches.
pub(crate) fn build_indices(
    regions: &AlignedRegions,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if keep == Keep::All {
        return build_all_indices(regions, |_, _| true);
    }
    Ok(build_selected_indices(regions, keep, |_, _| true))
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
        build_all_indices(regions, predicates_pass)
    } else {
        Ok(build_selected_indices(regions, keep, predicates_pass))
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
    let second_left = second
        .left_index
        .iter()
        .enumerate()
        .map(|(position, id)| (*id, position))
        .collect::<HashMap<_, _>>();
    let second_right = second
        .right_index
        .iter()
        .enumerate()
        .map(|(position, id)| (*id, position))
        .collect::<HashMap<_, _>>();
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
    first.validate_lengths().map_err(PyValueError::new_err)?;
    second.validate_lengths().map_err(PyValueError::new_err)?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::types::PyDict;

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
