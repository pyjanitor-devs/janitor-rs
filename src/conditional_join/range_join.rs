//! Shared range-join window construction.
//!
//! A basic range join has two range predicates. This module computes the
//! half-open `starts`/`ends` windows for those predicates. Extended joins
//! reuse the same primitive and perform their additional filtering elsewhere.
//!
//! PyJanitor performs all pandas-specific preparation before calling this
//! module: null removal, dtype selection, right-value sorting, and physical
//! position tracking. Consequently, the public PyO3 functions receive NumPy
//! arrays rather than pandas labels. A range anchor is a three-element tuple
//! `(left_values, right_values, operator)` and the shared physical maps are
//! explicit function arguments. Window offsets always address the sorted
//! right-value layout; returned right positions are obtained by indexing the
//! supplied `right_index` map.
//!
//! Aggregation functions use the same compact aligned value layout. Forward
//! aggregation writes to left output slots and consumes right source arrays;
//! reverse aggregation writes to right output slots and consumes left source
//! arrays. Residual predicates are evaluated only after the two anchor
//! windows have been intersected.
//!
//! # Layout contract
//!
//! The first predicate establishes the canonical right-side physical order.
//! The second predicate may have been searched in a different sorted value
//! order, but Python must provide its values and index map already aligned to
//! that canonical order before this module intersects windows. Rust treats a
//! window offset as a position in that prepared layout; it never sorts values
//! or reconstructs a pandas index.
//!
//! # Window contract
//!
//! Every window is half-open: `[start, end)`. Empty windows are represented by
//! `start == end` and are removed before index materialization or aggregation.
//! This convention is important because `end` is allowed to equal the right
//! length, while `start` may be zero.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use super::compare_op::CompareOp;
use super::join_aggregation_helpers::aggregate_range_windows;
use super::join_candidate_materialization::materialize_range_candidates;
#[cfg(test)]
use super::join_search::range_window;
use super::join_types::{result_dict, Keep, SingleJoinResult};
use super::predicate::{check_predicate_lengths, parse_predicates_with_nulls_strings};
use super::range_predicate::{parse_any_range_parts, AnchorSearch, AnyParsedRangePredicate};
use crate::aggregation::common::adaptive::should_use_segment_tree;
use crate::aggregation::common::ensure_equal_lengths_core;
use crate::aggregation::packed_spans::{
    assign_range_clusters, queried_source_spans, PackedSpanLayout,
};

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    // ELI5: each test thread keeps its own tally, so a neighboring test cannot
    // accidentally supply the branch evidence that this test is missing.
    pub(crate) static EXTREME_TREE_HITS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static EXTREME_HULL_TREE_HITS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static PACKING_PRECHECK_DIRECT_HITS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static ENVELOPE_RANK_SWEEP_HITS: Cell<usize> = const { Cell::new(0) };
    pub(crate) static ENVELOPE_KEEP_SWEEP_HITS: Cell<usize> = const { Cell::new(0) };
}

/// Select the smallest or largest physical label in each half-open window by scanning.
///
/// The returned values are positions into `labels`, not the labels themselves.
/// This helper searches each exact interior window rather than treating it as
/// a prefix or suffix of the right-hand layout. It is the simple reference
/// implementation used for small workloads and for checking the tree path in
/// tests.
///
/// # Arguments
///
/// * `labels` - Physical right-side labels in the prepared layout.
/// * `ranges` - Non-empty half-open windows `(start, end)` into `labels`.
/// * `minimum` - Select the smallest label when `true`; select the largest
///   label when `false`.
///
/// # Returns
///
/// One position in `labels` for every input range, preserving input order.
/// Equal labels keep the first position encountered.
///
/// # Panics
///
/// The caller must validate that every range is non-empty and in bounds.
/// Keeping that contract here avoids repeating checks inside the hot loop.
fn extreme_positions_direct(
    labels: ArrayView1<'_, i64>,
    ranges: &[(usize, usize)],
    minimum: bool,
) -> Vec<usize> {
    // ELI5: inspect only the exact shelf belonging to each left row. Looking
    // at a prefix or suffix would accidentally include right rows outside
    // the intersection of the two range predicates.
    let mut selected = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let mut best = start;
        for position in (start + 1)..end {
            let better = if minimum {
                labels[position] < labels[best]
            } else {
                labels[position] > labels[best]
            };
            if better {
                best = position;
            }
        }
        selected.push(best);
    }
    selected
}

/// Select extrema from arbitrary windows, choosing a direct scan or a packed tree.
///
/// The adaptive policy compares the estimated work of scanning every queried
/// element (`total_width`) with building and querying one segment tree over
/// the packed queried spans. Unqueried gaps are omitted, while every leaf
/// retains its original source position so callers receive the same result
/// coordinates as the direct scan.
///
/// ELI5: a short list of small shelves is faster to inspect by hand, but many
/// overlapping shelves are cheaper to answer from a shared index of summaries.
/// The tree stores the winning shelf position for each larger slice, so a
/// query combines a few stored answers instead of walking every item again.
///
/// # Arguments
///
/// * `labels` - Physical right-side labels addressed by the query ranges.
/// * `ranges` - Half-open `(start, end)` windows into `labels`. Every range
///   must be non-empty and satisfy `end <= labels.len()`.
/// * `minimum` - Select the smallest label when `true`; select the largest
///   label when `false`.
///
/// # Returns
///
/// One selected position in `labels` per input range, in input order. Equal
/// labels may select different offsets in the direct and tree paths, but the
/// returned label is identical. An empty `ranges` slice returns an empty vector.
///
/// # Errors
///
/// Returns an error for an empty, reversed, or out-of-bounds range, or when a
/// tree allocation cannot be represented by the platform's `usize` capacity.
pub fn adaptive_extreme_positions(
    labels: ArrayView1<'_, i64>,
    ranges: &[(usize, usize)],
    minimum: bool,
) -> Result<Vec<usize>, String> {
    let mut total_width = 0_usize;
    let mut max_width = 0_usize;
    let mut span_start = labels.len();
    let mut span_end = 0_usize;
    for &(start, end) in ranges {
        if start >= end || end > labels.len() {
            return Err("range window is empty or out of bounds".to_owned());
        }
        // ELI5: if the total workload is too large to count exactly, treating
        // it as "very large" safely favors the tree. This is only a speed
        // hint; it cannot change which position wins.
        let width = end - start;
        total_width = total_width.saturating_add(width);
        max_width = max_width.max(width);
        span_start = span_start.min(start);
        span_end = span_end.max(end);
    }
    if ranges.is_empty() {
        return Ok(Vec::new());
    }
    // The shared cutoff never selects a tree for three or fewer queries. Keep
    // those tiny batches on the direct path before paying for span sorting and
    // cluster mapping.
    if ranges.len() <= 3 {
        return Ok(extreme_positions_direct(labels, ranges, minimum));
    }
    // Keep the established adaptive cutoff exact for small batches; the
    // packing estimate is primarily needed for larger sparse batches where
    // sorting the query list can dominate the direct scan.
    if ranges.len() > PACKING_ESTIMATE_QUERY_THRESHOLD
        && !should_consider_packed_tree(ranges.len(), total_width, max_width)
    {
        #[cfg(test)]
        PACKING_PRECHECK_DIRECT_HITS.with(|hits| hits.set(hits.get() + 1));
        return Ok(extreme_positions_direct(labels, ranges, minimum));
    }

    // ELI5: sort the shelves once, then lay only the shelves that at least one
    // query visits end to end. Empty aisles between groups disappear from the
    // tree, but each query still maps back to its original source positions.
    // Hull-first gate: if removing every possible gap cannot repay sorting,
    // retain one hull span and avoid the packed metadata allocations.
    let hull_len = span_end - span_start;
    let packing_query_height =
        (usize::BITS - (ranges.len().saturating_sub(1)).leading_zeros()) as usize;
    let packing_cost = ranges.len().saturating_mul(packing_query_height + 1);
    let use_hull = hull_len.saturating_sub(max_width).saturating_mul(5) <= packing_cost;
    let (spans, clusters) = if use_hull {
        (
            vec![(span_start, span_end)],
            vec![Some(0_usize); ranges.len()],
        )
    } else {
        let packed_ranges: Vec<_> = ranges.iter().copied().map(Some).collect();
        let spans = queried_source_spans(&packed_ranges);
        let clusters = assign_range_clusters(&packed_ranges, &spans);
        (spans, clusters)
    };
    let layout = PackedSpanLayout::new(&spans, &clusters);
    if !should_use_segment_tree(ranges.len(), total_width, layout.length) {
        return Ok(extreme_positions_direct(labels, ranges, minimum));
    }

    #[cfg(test)]
    EXTREME_TREE_HITS.with(|hits| hits.set(hits.get() + 1));
    #[cfg(test)]
    if use_hull {
        EXTREME_HULL_TREE_HITS.with(|hits| hits.set(hits.get() + 1));
    }

    // ELI5: each tree node remembers the winning original position for its
    // slice, so a query combines a few summaries instead of rescanning rows.
    // Leaves live at `layout.length..2 * layout.length`; `usize::MAX` means no
    // candidate.
    let tree_len = layout
        .length
        .checked_mul(2)
        .ok_or("range-extreme tree size exceeds platform capacity")?;
    let mut tree = vec![usize::MAX; tree_len];
    for (cluster, &(span_start, span_end)) in layout.spans.iter().enumerate() {
        let offset = layout.offsets[cluster];
        for position in span_start..span_end {
            tree[layout.length + offset + position - span_start] = position;
        }
    }
    for node in (1..layout.length).rev() {
        let left = tree[node * 2];
        let right = tree[node * 2 + 1];
        tree[node] = choose_extreme_position(labels, left, right, minimum);
    }

    let mut selected = Vec::with_capacity(ranges.len());
    for (row, &(query_start, query_end)) in ranges.iter().enumerate() {
        // Translate right-layout coordinates into the compact tree's leaf
        // coordinates while preserving the half-open query convention.
        let (mut start, mut end) = layout
            .packed_range(row, Some((query_start, query_end)))
            .expect("validated range must belong to a packed span");
        let mut winner = usize::MAX;
        while start < end {
            if start & 1 == 1 {
                // An odd left boundary is the next complete slice to include.
                winner = choose_extreme_position(labels, winner, tree[start], minimum);
                start += 1;
            }
            if end & 1 == 1 {
                // An odd right boundary is the final complete slice to include.
                end -= 1;
                winner = choose_extreme_position(labels, winner, tree[end], minimum);
            }
            start /= 2;
            end /= 2;
        }
        selected.push(winner);
    }
    Ok(selected)
}

// Below this size, retaining the established cutoff is cheaper and keeps its
// exact boundary behavior; larger batches are where metadata sorting matters.
const PACKING_ESTIMATE_QUERY_THRESHOLD: usize = 16;

/// Decide whether it is worth constructing packed-span metadata at all.
///
/// `max_width` is a lower bound on the eventual packed length. If even a tree
/// over that smallest possible layout is unlikely to repay its build/query
/// work and a rough sort-and-map cost, scanning directly avoids paying for
/// packing only to choose the direct path afterward.
fn should_consider_packed_tree(query_count: usize, total_width: usize, max_width: usize) -> bool {
    let query_height = (usize::BITS - (query_count.saturating_sub(1)).leading_zeros()) as usize;
    let span_height = (usize::BITS - (max_width.saturating_sub(1)).leading_zeros()) as usize;
    let packing = query_count.saturating_mul(query_height.saturating_add(1));
    let smallest_tree = max_width
        .saturating_mul(5)
        .saturating_add(query_count.saturating_mul(span_height).saturating_mul(2));
    total_width > smallest_tree.saturating_add(packing)
}

/// Choose the better of two candidate positions for an extrema query.
///
/// `usize::MAX` is an internal empty-candidate sentinel. It cannot be a real
/// position because every valid position is smaller than `labels.len()`;
/// handling it here lets an iterative tree query combine partial slices
/// without inventing a fake label value. Strict comparison retains the first
/// argument on ties; tree query traversal need not visit positions in order.
///
/// ELI5: the marker is an empty seat, not a contestant. Only occupied seats
/// participate in the label comparison, so the marker is never used to index.
///
/// # Arguments
///
/// * `labels` - Labels used to compare the candidate positions.
/// * `left` - First candidate, or `usize::MAX` for no candidate.
/// * `right` - Second candidate, or `usize::MAX` for no candidate.
/// * `minimum` - Compare for the smallest label when `true`, otherwise the
///   largest label.
///
/// # Returns
///
/// The winning candidate, or `usize::MAX` if both candidates are empty.
fn choose_extreme_position(
    labels: ArrayView1<'_, i64>,
    left: usize,
    right: usize,
    minimum: bool,
) -> usize {
    if left == usize::MAX {
        return right;
    }
    if right == usize::MAX {
        return left;
    }
    let right_wins = if minimum {
        labels[right] < labels[left]
    } else {
        labels[right] > labels[left]
    };
    if right_wins {
        right
    } else {
        left
    }
}

/// Parse one basic dual-range anchor using shared physical index arrays.
///
/// Each anchor carries only `(left_values, right_values, operator)`. The
/// shared `left_index` and `right_index` are supplied once by
/// [`range_join_indices`], preventing the two anchors from carrying
/// inconsistent physical layouts.
///
/// # Arguments
///
/// * `tuple` - Three-element `(left_values, right_values, operator)` tuple.
/// * `left_index` - Physical positions paired with the left values.
/// * `right_index` - Physical positions paired with the sorted right values.
///
/// # Returns
///
/// A parsed range predicate borrowing the Python objects supplied by the
/// caller.
///
/// # Errors
///
/// Returns a Python `ValueError` when `tuple` does not contain exactly three
/// elements, or a parsing/type error when one of its fields is invalid.
fn parse_shared_index_range_predicate<'py>(
    tuple: &Bound<'py, PyTuple>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
) -> PyResult<AnyParsedRangePredicate<'py>> {
    // The shared-index ABI intentionally has only three fields. Physical
    // labels belong to the outer call, so malformed six/eight-field tuples
    // must not silently be interpreted as this contract.
    if tuple.len() != 3 {
        return Err(PyValueError::new_err(
            "shared-index range predicates must contain 3 elements",
        ));
    }
    parse_any_range_parts(
        &tuple.get_item(0)?,
        left_index,
        &tuple.get_item(1)?,
        right_index,
        &tuple.get_item(2)?,
    )
}

/// A typed range predicate used by the basic two-range kernel.
///
/// The value arrays describe the comparison layout, while the paired index
/// arrays carry the labels that must be returned to Python. PyJanitor owns
/// sorting and alignment before constructing this value; Rust only consumes
/// the already-aligned views.
#[cfg(test)]
pub(crate) struct RangePredicate<'a, T> {
    /// Left values in logical left-row order. Null rows have already been
    /// removed for range predicates.
    pub(crate) left: ArrayView1<'a, T>,
    /// Labels paired position-for-position with `left`.
    pub(crate) left_index: ArrayView1<'a, i64>,
    /// Right values in ascending value order.
    pub(crate) right: ArrayView1<'a, T>,
    /// Labels paired position-for-position with the sorted `right` values.
    pub(crate) right_index: ArrayView1<'a, i64>,
    /// Comparator applied to the paired left and right values.
    pub(crate) op: CompareOp,
}

/// Build the intersection window for exactly two aligned range predicates.
///
/// Each right-hand array must already be sorted in ascending value order by
/// PyJanitor. The two predicates are evaluated against the same logical left
/// rows and the same logical right positions; this function only intersects
/// their positional windows. It does not sort arrays or evaluate residual
/// predicates.
///
/// # Arguments
///
/// * `first` - The first range predicate, including its left/right values and
///   the corresponding original index labels.
/// * `second` - The second range predicate. Its left and right arrays must be
///   length-aligned with `first`; its right array must use the same sorted
///   physical layout as `first.right`.
///
/// # Returns
///
/// A [`SingleJoinResult`] containing one half-open `[start, end)` window for
/// each left row whose two predicates intersect. `left_index` retains input
/// order and `right_index` is copied from the first predicate's supplied
/// right-label array. A window's positions always refer to that copied right
/// layout.
///
/// # Errors
///
/// Returns an error when value/index lengths differ or either comparator is
/// not a range comparator.
#[cfg(test)]
pub(crate) fn build_windows<T: PartialOrd + Copy>(
    first: RangePredicate<'_, T>,
    second: RangePredicate<'_, T>,
) -> Result<SingleJoinResult, String> {
    fn build_test_window<T: PartialOrd + Copy>(
        predicate: RangePredicate<'_, T>,
    ) -> Result<SingleJoinResult, String> {
        ensure_equal_lengths_core(
            "left test values",
            predicate.left.len(),
            "left test labels",
            predicate.left_index.len(),
        )?;
        ensure_equal_lengths_core(
            "right test values",
            predicate.right.len(),
            "right test labels",
            predicate.right_index.len(),
        )?;
        let mut result = SingleJoinResult {
            left_positions: Vec::with_capacity(predicate.left.len()),
            left_index: predicate.left_index.to_vec(),
            right_index: predicate.right_index.to_vec(),
            starts: Vec::with_capacity(predicate.left.len()),
            ends: Vec::with_capacity(predicate.left.len()),
        };
        for (position, &left_value) in predicate.left.iter().enumerate() {
            let (start, end) = range_window(left_value, predicate.right, predicate.op);
            result.left_positions.push(position);
            result.starts.push(start);
            result.ends.push(end);
        }
        Ok(result)
    }

    let first = build_test_window(first)?;
    let second = build_test_window(second)?;
    intersect_windows(first, second)
}

/// Intersect two already-built positional windows.
///
/// The value dtypes are intentionally absent here. Each anchor has already
/// completed its own typed binary searches; this step only combines the
/// resulting positions. PyJanitor aligns the second right array to the first
/// right layout before calling Rust, so the two windows refer to the same
/// physical right positions even when their value dtypes differ.
///
/// The first anchor owns the right-label vector retained for output. The
/// second anchor deliberately does not carry a second copy of those labels.
/// Consequently, this function checks positional bounds but does not compare
/// every label element. A same-length, differently ordered second right layout
/// is outside the Rust contract and must be prevented by PyJanitor's alignment
/// step. This keeps the dual-range path from paying an O(right_len) defensive
/// scan on every call.
///
/// # Arguments
///
/// * `first` - The first anchor's windows, labels, and physical positions.
/// * `second` - The second anchor's windows, aligned to the first anchor's
///   left and right layouts.
///
/// # Returns
///
/// A new [`SingleJoinResult`] containing the row-wise intersection. Empty
/// intersections are retained as equal start/end boundaries for later stages
/// to remove.
///
/// # Errors
///
/// Returns an error when the window arrays or physical maps have incompatible
/// lengths, or when a window boundary is outside the corresponding right
/// layout.
#[cfg(test)]
pub(crate) fn intersect_windows(
    first: SingleJoinResult,
    second: SingleJoinResult,
) -> Result<SingleJoinResult, String> {
    // ELI5: two searches give two rulers over the same right rows; taking the
    // larger start and smaller end keeps only the part where both agree.
    ensure_equal_lengths_core(
        "first left window",
        first.left_positions.len(),
        "first left labels",
        first.left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "second left window",
        second.left_positions.len(),
        "second left labels",
        second.left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "first window starts",
        first.left_positions.len(),
        "first window ends",
        first.ends.len(),
    )?;
    ensure_equal_lengths_core(
        "second window starts",
        second.left_positions.len(),
        "second window ends",
        second.ends.len(),
    )?;
    if first.left_positions.len() != second.left_positions.len() {
        return Err(
            "dual range predicates must use aligned left rows and right positions".to_owned(),
        );
    }
    // The second anchor's right labels are deliberately not copied: PyJanitor
    // guarantees that both right value arrays use the same physical layout.
    // Boundary validation still catches a second array whose windows reach
    // beyond the first anchor's right domain.
    if second.ends.iter().any(|&end| end > first.right_index.len()) {
        return Err("dual range predicates must use aligned right positions".to_owned());
    }

    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: first.right_index,
        starts: Vec::new(),
        ends: Vec::new(),
    };
    for row in 0..first.left_positions.len() {
        let start = first.starts[row].max(second.starts[row]);
        let end = first.ends[row].min(second.ends[row]);
        if start < end {
            result.left_positions.push(first.left_positions[row]);
            result.left_index.push(first.left_index[row]);
            result.starts.push(start);
            result.ends.push(end);
        }
    }
    Ok(result)
}

/// Left rows whose windows are computed per batch by [`for_each_window`].
const WINDOW_CHUNK: usize = 4_096;

/// Evenly spaced row blocks, of [`SWEEP_WIDTH_SAMPLE_ROWS`] rows each, that
/// [`FirstAnchorSweep::new`] searches to estimate the candidate width.
const SWEEP_WIDTH_SAMPLE_BLOCKS: usize = 32;
const SWEEP_WIDTH_SAMPLE_ROWS: usize = 256;

/// Call `visit(row, (start, end))` for every left row's window, in row
/// order.
///
/// ELI5: work through the rows a few thousand at a time and measure each
/// batch in one tight typed loop, so the CPU can overlap neighbouring
/// rows' binary searches. Only one batch of windows is ever held; storing
/// every row's window first cost four `usize` per left row per anchor.
fn for_each_window(
    search: &AnchorSearch<'_>,
    left_len: usize,
    mut visit: impl FnMut(usize, (usize, usize)),
) {
    let chunk = WINDOW_CHUNK.min(left_len);
    let (mut starts, mut ends) = (vec![0_usize; chunk], vec![0_usize; chunk]);
    for chunk_start in (0..left_len).step_by(WINDOW_CHUNK) {
        let rows = WINDOW_CHUNK.min(left_len - chunk_start);
        search.windows_into(chunk_start, &mut starts[..rows], &mut ends[..rows]);
        for offset in 0..rows {
            visit(chunk_start + offset, (starts[offset], ends[offset]));
        }
    }
}

/// [`for_each_window`] over two aligned anchors at once, for the left rows
/// in `rows`: `visit(row, first_window, second_window)`.
fn for_each_window_pair(
    first: &AnchorSearch<'_>,
    second: &AnchorSearch<'_>,
    rows: std::ops::Range<usize>,
    mut visit: impl FnMut(usize, (usize, usize), (usize, usize)),
) {
    let chunk = WINDOW_CHUNK.min(rows.len());
    let (mut first_starts, mut first_ends) = (vec![0_usize; chunk], vec![0_usize; chunk]);
    let (mut second_starts, mut second_ends) = (vec![0_usize; chunk], vec![0_usize; chunk]);
    for chunk_start in rows.clone().step_by(WINDOW_CHUNK) {
        let rows = WINDOW_CHUNK.min(rows.end - chunk_start);
        first.windows_into(
            chunk_start,
            &mut first_starts[..rows],
            &mut first_ends[..rows],
        );
        second.windows_into(
            chunk_start,
            &mut second_starts[..rows],
            &mut second_ends[..rows],
        );
        for offset in 0..rows {
            visit(
                chunk_start + offset,
                (first_starts[offset], first_ends[offset]),
                (second_starts[offset], second_ends[offset]),
            );
        }
    }
}

/// Build dual-range windows while dispatching each anchor independently.
///
/// The two anchors may use different NumPy dtypes. Each is binary-searched in
/// its own typed value layout, then the resulting integer offsets are
/// intersected without converting either value array to a common dtype.
///
/// # Errors
///
/// Returns an error when either parsed anchor cannot produce valid windows or
/// when the two window collections do not share aligned left/right lengths.
///
/// # Arguments
///
/// * `first` - The first parsed range anchor and canonical right layout.
/// * `second` - The second parsed range anchor, aligned to `first`.
///
/// # Returns
///
/// The row-wise intersection of the two anchors as half-open windows.
pub(crate) fn build_any_windows(
    first: &AnyParsedRangePredicate<'_>,
    second: &AnyParsedRangePredicate<'_>,
) -> Result<SingleJoinResult, String> {
    // ELI5: measure both rulers for one row at a time and write down only
    // the overlap. Building each anchor's windows for every left row first
    // held eight left-length vectors at once (4 GiB for Arun's 40M rows)
    // just to keep the rows where they overlap.
    let left_len = first.left_len();
    let right_len = first.right_len();
    let (left_labels, right_labels) = first.labels();
    let (second_left_labels, second_right_labels) = second.labels();
    ensure_equal_lengths_core("left", left_len, "left_index", left_labels.len())?;
    ensure_equal_lengths_core("right", right_len, "right_index", right_labels.len())?;
    ensure_equal_lengths_core(
        "left",
        second.left_len(),
        "left_index",
        second_left_labels.len(),
    )?;
    ensure_equal_lengths_core(
        "right",
        second.right_len(),
        "right_index",
        second_right_labels.len(),
    )?;
    let (first_search, second_search) = (first.search(), second.search());
    for op in [first_search.op(), second_search.op()] {
        if !matches!(
            op,
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
        ) {
            return Err("single join range core requires a range comparator".to_owned());
        }
    }
    if second.left_len() != left_len {
        return Err(
            "dual range predicates must use aligned left rows and right positions".to_owned(),
        );
    }
    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: right_labels.to_vec(),
        starts: Vec::new(),
        ends: Vec::new(),
    };
    if left_len == 0 || right_len == 0 || second.right_len() == 0 {
        return Ok(result);
    }
    // ELI5: reserve room for every row once instead of letting the vectors
    // double as they fill. Doubling copies the data and can leave up to
    // twice the needed capacity; a reservation the rows never reach is
    // address space the operating system does not back with memory until
    // it is written.
    result.left_positions.reserve_exact(left_len);
    result.left_index.reserve_exact(left_len);
    result.starts.reserve_exact(left_len);
    result.ends.reserve_exact(left_len);
    // ELI5: work through the rows a few thousand at a time. For each batch,
    // measure all its rows against the first ruler, then all against the
    // second, then keep the overlaps. Each ruler's measurements run in their
    // own tight loop (as when every row was measured up front), but only one
    // batch of measurements is ever held.
    let mut misaligned = false;
    for_each_window_pair(
        &first_search,
        &second_search,
        0..left_len,
        |row, first, second| {
            // The second anchor's labels are deliberately not compared:
            // PyJanitor aligns both right value arrays to one physical layout.
            // Its windows must still stay inside the first anchor's layout.
            misaligned |= second.1 > right_len;
            let start = first.0.max(second.0);
            let end = first.1.min(second.1);
            if start < end {
                result.left_positions.push(row);
                result.left_index.push(left_labels[row]);
                result.starts.push(start);
                result.ends.push(end);
            }
        },
    );
    if misaligned {
        return Err("dual range predicates must use aligned right positions".to_owned());
    }
    Ok(result)
}

/// Select one label from each arbitrary dual-range window.
///
/// Unlike a single non-equi join, a dual-range join can produce an interior
/// window such as `[2, 4)`. Prefix and suffix extrema cannot answer that
/// shape because they include positions outside the intersection. The
/// extrema helper provides the correct arbitrary-window query and chooses
/// between direct scans and a segment tree using the shared adaptive workload
/// guard.
///
/// # Arguments
///
/// * `windows` - Intersected, non-empty half-open windows and their right
///   index labels.
/// * `keep` - Selection mode. `first` means the smallest right label,
///   `last` the largest, `any` the first physical label, and `all` every
///   label in each window.
/// * `right_index_is_ordered` - Whether labels increase within each prepared
///   window, allowing first/last selection from its endpoints.
///
/// # Returns
///
/// Materialized left and right labels in the same left-row order as
/// `windows`. For `all`, right labels remain in their supplied physical
/// right-array order within each window.
///
/// # Errors
///
/// Returns an error for an invalid extrema window, overflowing result/tree
/// size, failed result reservation, or an invalid selected position. Empty
/// windows are removed by [`build_any_windows`] before reaching this helper.
pub(crate) fn choose_range_windows(
    windows: &SingleJoinResult,
    keep: Keep,
    right_index_is_ordered: bool,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let labels = windows.right_index.as_slice();
    if windows.starts.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    if keep == Keep::All {
        // `all` does not need an extrema query: every position in every
        // half-open window is a result, and the physical right order is the
        // required output order. Because this basic range path has no
        // residual filters, the window widths give the exact final size.
        let mut output_capacity = 0_usize;
        for (&start, &end) in windows.starts.iter().zip(&windows.ends) {
            let width = end
                .checked_sub(start)
                .ok_or("range window has invalid bounds")?;
            output_capacity = output_capacity
                .checked_add(width)
                .ok_or("range join result size exceeds platform capacity")?;
        }
        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_capacity)
            .map_err(|_| "range join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_capacity)
            .map_err(|_| "range join result allocation failed")?;
        for (row, (&start, &end)) in windows.starts.iter().zip(&windows.ends).enumerate() {
            for &label in &labels[start..end] {
                output_left.push(windows.left_index[row]);
                output_right.push(label);
            }
        }
        return Ok((output_left, output_right));
    }

    if keep == Keep::Any {
        // Every stored window is non-empty, so its first physical position is
        // a valid arbitrary match.
        let mut output_left = Vec::with_capacity(windows.left_index.len());
        let mut output_right = Vec::with_capacity(windows.left_index.len());
        for (row, &start) in windows.starts.iter().enumerate() {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[start]);
        }
        return Ok((output_left, output_right));
    }

    if right_index_is_ordered {
        let mut output_left = Vec::with_capacity(windows.left_index.len());
        let mut output_right = Vec::with_capacity(windows.left_index.len());
        for (row, (&start, &end)) in windows.starts.iter().zip(&windows.ends).enumerate() {
            let position = if keep == Keep::First { start } else { end - 1 };
            output_left.push(windows.left_index[row]);
            output_right.push(labels[position]);
        }
        return Ok((output_left, output_right));
    }

    // ELI5: windows already use the helper's native offset format. Pair them
    // once instead of taking a detour through signed NumPy-style arrays.
    let ranges = windows
        .starts
        .iter()
        .copied()
        .zip(windows.ends.iter().copied())
        .collect::<Vec<_>>();
    let selected_positions =
        adaptive_extreme_positions(ArrayView1::from(labels), &ranges, keep == Keep::First)?;

    let mut output_left = Vec::with_capacity(windows.left_index.len());
    let mut output_right = Vec::with_capacity(windows.left_index.len());
    for (row, &position) in selected_positions.iter().enumerate() {
        let label = *labels
            .get(position)
            .ok_or("range window selection returned an out-of-bounds position")?;
        output_left.push(windows.left_index[row]);
        output_right.push(label);
    }
    Ok((output_left, output_right))
}

/// Build indices for a dual-range join from two per-row windows.
///
/// Each anchor produces one positional window for every logical left row. The
/// windows are intersected row by row; index generation is therefore defined
/// by the window intersection, not by the value dtype of either anchor.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to construct the result
///   dictionary.
/// * `predicates` - Exactly two three-element tuples of
///   `(left_values, right_values, operator)`. The value arrays in both tuples
///   must share the same left layout and the same sorted right layout.
/// * `left_index` - Physical positions paired with every left value. These
///   are original dataframe positions, not compact offsets.
/// * `right_index` - Physical positions paired with every sorted right value.
///   Window offsets are translated through this array before being returned.
/// * `right_index_is_ordered` - Whether the physical right positions are
///   ordered. This enables direct `first`/`last` selection; extrema kernels
///   are used otherwise.
/// * `keep` - Selection policy: `all`, `any`, `first`, or `last`.
/// * `return_building_blocks` - Return the physical maps plus `starts` and
///   `ends` instead of materialized pairs when true.
///
/// # Returns
///
/// Returns `None` when the intersected windows contain no candidate. Otherwise
/// returns a dictionary containing `left_index` and `right_index`, and, when
/// requested, the half-open window arrays.
///
/// # Errors
///
/// Raises `ValueError` when the predicate count, tuple shape, index lengths,
/// operator, keep value, or window boundaries violate the prepared-layout
/// contract. Raises `TypeError` when an index or predicate field is not a
/// one-dimensional NumPy-compatible value.
#[pyfunction]
pub fn range_join_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: Bound<'py, PyAny>,
    right_index: Bound<'py, PyAny>,
    right_index_is_ordered: bool,
    keep: &str,
    return_building_blocks: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // Python has already removed nulls, sorted the primary right anchor, and
    // aligned every second anchor to that physical layout. Rust's job here is
    // only to build/intersect positional windows; it must not sort again or
    // confuse a search offset with an original dataframe position.
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range join requires exactly two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let second_item = predicates.get_item(1)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_shared_index_range_predicate(first_tuple, &left_index, &right_index)?;
    let second = parse_shared_index_range_predicate(second_tuple, &left_index, &right_index)?;
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    if return_building_blocks {
        return Ok(Some(result_dict(
            py,
            windows.left_index,
            windows.right_index,
            Some(windows.starts),
            Some(windows.ends),
        )?));
    }
    let keep = Keep::parse(keep)?;
    let (left, right) = choose_range_windows(&windows, keep, right_index_is_ordered)
        .map_err(PyValueError::new_err)?;
    if left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, left, right, None, None)?))
}

/// Register the basic and extended range-index and aggregation entry points.
///
/// Registration lives with the implementation so adding or removing a range
/// kernel cannot leave the top-level module with a stale export list.
///
/// # Arguments
///
/// * `m` - Python module receiving the range-join functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        range_join_indices,
        range_join_extended_indices,
        range_join_aggregate,
        range_join_aggregate_reverse,
        range_join_extended_aggregate,
        range_join_extended_aggregate_reverse,
    );
    Ok(())
}

/// Execute the range-extended join with independently typed anchors.
///
/// The first two anchors are converted into intersected windows. Every later
/// predicate is retained as a residual filter and is evaluated only for a
/// candidate inside that intersection. This is the shared implementation for
/// the public index wrapper; aggregation has a parallel consumer because it
/// must update state for every passing candidate rather than apply `keep`.
///
/// # Arguments
///
/// * `py` - Python interpreter token used for residual parsing and results.
/// * `predicates` - At least two parsed predicate tuples.
/// * `keep` - Selection policy applied after all residual predicates pass.
/// * `first` - Parsed first range anchor defining the primary right layout.
/// * `second` - Parsed second range anchor used to intersect that layout.
///
/// # Returns
///
/// Returns materialized physical index pairs, or `None` when no candidate
/// survives the two anchors and residual filters.
///
/// # Errors
///
/// Returns an error for malformed residual predicates, length mismatches,
/// invalid keep values, or invalid window boundaries.
fn extended_join<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first: AnyParsedRangePredicate<'py>,
    second: AnyParsedRangePredicate<'py>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    let residuals = PyList::empty(py);
    // The first two predicates define the bounded candidate region. All later
    // predicates—including the original non-monotonic range predicate paired
    // with a cumulative envelope—are exact filters over that region.
    for item in predicates.iter().skip(2) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    ensure_equal_lengths_core(
        "first left predicate array",
        first.left_len(),
        "second left predicate array",
        second.left_len(),
    )
    .map_err(PyValueError::new_err)?;
    ensure_equal_lengths_core(
        "first right predicate array",
        first.right_len(),
        "second right predicate array",
        second.right_len(),
    )
    .map_err(PyValueError::new_err)?;
    check_predicate_lengths(&parsed, first.left_len(), first.right_len())?;
    // ELI5: `first`/`last` only need the smallest or largest label among a
    // row's matches. When the second anchor is just the residual's envelope,
    // a running scoreboard answers that without walking every match.
    if matches!(keep, Keep::First | Keep::Last) && metadata.is_none() && parsed.len() == 1 {
        if let Some((out_left, out_right)) =
            select_envelope_rank_sweep(&first, &second, &parsed[0], keep)
        {
            if out_left.is_empty() {
                return Ok(None);
            }
            return Ok(Some(result_dict(py, out_left, out_right, None, None)?));
        }
    }
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    let (out_left, out_right) =
        materialize_range_candidates(&windows, &parsed, metadata.as_deref(), keep)
            .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, out_left, out_right, None, None)?))
}

/// Build range-extended indices from two anchor windows and residual filters.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to construct the result.
/// * `predicates` - At least two tuples. The first two are three-element
///   `(left_values, right_values, operator)` range anchors; later tuples are
///   residual predicates parsed by the shared predicate machinery.
/// * `left_index` - Physical positions paired with every left anchor value.
/// * `right_index` - Physical positions paired with every sorted right value.
/// * `keep` - Selection policy applied after both anchors and all residuals
///   pass.
///
/// # Returns
///
/// Returns materialized physical pairs, or `None` when no candidate survives.
///
/// # Errors
///
/// Raises `ValueError` for fewer than two predicates, malformed anchor tuples,
/// incompatible lengths, unsupported operators, invalid keep values, or a
/// residual predicate whose arrays do not match the shared physical maps.
#[pyfunction]
pub fn range_join_extended_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: Bound<'py, PyAny>,
    right_index: Bound<'py, PyAny>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range extended join requires at least two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let second_item = predicates.get_item(1)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    if first_tuple.len() != 3 || second_tuple.len() != 3 {
        return Err(PyValueError::new_err(
            "extended range anchors must contain 3 elements",
        ));
    }
    let first = parse_shared_index_range_predicate(first_tuple, &left_index, &right_index)?;
    let second = parse_shared_index_range_predicate(second_tuple, &left_index, &right_index)?;
    extended_join(py, predicates, keep, first, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array1, PyArray1, PyArrayMethods};

    /// Select reference positions using the same public `keep` semantics as
    /// the production wrapper.
    ///
    /// Keeping this policy in one helper is important for test maintenance:
    /// the brute-force oracle should disagree with the implementation only
    /// when the implementation is wrong, not because one test copied a
    /// slightly different `first`/`last` tie-break rule. `first` and `last`
    /// are label-based, while `any` deliberately keeps the first physical
    /// candidate, matching the crate-wide join convention.
    fn select_reference_positions(
        passing: Vec<usize>,
        right_index: &[i64],
        keep: &str,
    ) -> Vec<usize> {
        match keep {
            "all" => passing,
            "any" => passing.into_iter().take(1).collect(),
            "first" | "last" => passing
                .into_iter()
                .min_by_key(|&position| {
                    if keep == "first" {
                        right_index[position]
                    } else {
                        -right_index[position]
                    }
                })
                .into_iter()
                .collect(),
            other => panic!("unexpected reference keep: {other}"),
        }
    }

    /// Advance the deterministic generator used by the randomized oracle.
    ///
    /// This is intentionally tiny and local to tests. It avoids a dependency
    /// on a random-number crate while giving future tests one named utility
    /// instead of each test inventing its own state transition.
    fn next_test_value(seed: &mut u64) -> i64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 32) % 9) as i64 - 4
    }

    struct ReferenceFixture<'a> {
        left_index: &'a [i64],
        first_left: &'a [i64],
        second_left: &'a [i64],
        right_index: &'a [i64],
        first_right: &'a [i64],
        second_right: &'a [i64],
    }

    /// The common i64 shape used by the public index tests.
    ///
    /// The production API is intentionally tuple-based for Python
    /// compatibility. Tests should not repeat the eight tuple-field
    /// positions everywhere, though: a field-order mistake in a test can make
    /// a regression look like a kernel failure. This small builder keeps the
    /// tuple layout in one documented place while malformed-tuple tests still
    /// construct their bad inputs explicitly.
    #[allow(dead_code)]
    struct IndexPredicate<'a> {
        left: &'a [i64],
        left_index: &'a [i64],
        right: &'a [i64],
        right_index: &'a [i64],
        ordered: bool,
        operator: &'a str,
    }

    fn append_index_predicate<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        predicate: IndexPredicate<'_>,
    ) -> PyResult<()> {
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, predicate.left.to_vec()).into_any(),
                PyArray1::from_vec(py, predicate.right.to_vec()).into_any(),
                predicate.operator.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        Ok(())
    }

    fn call_range_join_indices<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &[i64],
        right_index: &[i64],
        keep: &str,
        return_building_blocks: bool,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let left_index = PyArray1::from_vec(py, left_index.to_vec()).into_any();
        let right_index = PyArray1::from_vec(py, right_index.to_vec()).into_any();
        let right_index_is_ordered = right_index
            .cast::<PyArray1<i64>>()
            .map(|array| {
                array
                    .readonly()
                    .as_array()
                    .windows(2)
                    .into_iter()
                    .all(|pair| pair[0] <= pair[1])
            })
            .unwrap_or(false);
        range_join_indices(
            py,
            predicates,
            left_index,
            right_index,
            right_index_is_ordered,
            keep,
            return_building_blocks,
        )
    }

    fn call_range_join_extended_indices<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &[i64],
        right_index: &[i64],
        keep: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let left_index = PyArray1::from_vec(py, left_index.to_vec()).into_any();
        let right_index = PyArray1::from_vec(py, right_index.to_vec()).into_any();
        range_join_extended_indices(py, predicates, left_index, right_index, keep)
    }

    /// Build the public result expected from a pair-by-pair implementation.
    ///
    /// The optimized range path searches sorted values, but its public
    /// result is still defined by the original physical right labels.  This
    /// deliberately simple helper does not use windows or binary search: it
    /// is an independent oracle for the wrapper tests.
    fn reference_pairs(
        fixture: &ReferenceFixture<'_>,
        first_operator: &str,
        second_operator: &str,
        keep: &str,
    ) -> (Vec<i64>, Vec<i64>) {
        let mut output_left = Vec::new();
        let mut output_right = Vec::new();
        let first_op = CompareOp::try_from_str(first_operator).unwrap();
        let second_op = CompareOp::try_from_str(second_operator).unwrap();
        for left_position in 0..fixture.left_index.len() {
            let mut passing = Vec::new();
            for right_position in 0..fixture.right_index.len() {
                if first_op.apply(
                    &fixture.first_left[left_position],
                    &fixture.first_right[right_position],
                ) && second_op.apply(
                    &fixture.second_left[left_position],
                    &fixture.second_right[right_position],
                ) {
                    passing.push(right_position);
                }
            }
            for right_position in select_reference_positions(passing, fixture.right_index, keep) {
                output_left.push(fixture.left_index[left_position]);
                output_right.push(fixture.right_index[right_position]);
            }
        }
        (output_left, output_right)
    }

    fn assert_index_result<'py>(
        result: Option<Bound<'py, PyDict>>,
        expected: (Vec<i64>, Vec<i64>),
    ) {
        if expected.0.is_empty() {
            assert!(result.is_none(), "expected no matching pairs");
        } else {
            let result = result.expect("expected matching pairs");
            assert_eq!(read_pair(&result), expected);
        }
    }

    fn read_pair<'py>(result: &Bound<'py, PyDict>) -> (Vec<i64>, Vec<i64>) {
        let left = result
            .get_item("left_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        let right = result
            .get_item("right_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        (left, right)
    }

    #[test]
    fn two_range_windows_intersect_in_ascending_layout() {
        let left = Array1::from_vec(vec![2_i64, 5]);
        let left_second = Array1::from_vec(vec![8_i64, 9]);
        let right = Array1::from_vec(vec![1_i64, 2, 3, 4, 5, 6, 7, 8]);
        let right_second = Array1::from_vec(vec![0_i64, 1, 2, 3, 4, 5, 6, 7]);
        let left_index = Array1::from_vec(vec![10_i64, 11]);
        let right_index = Array1::from_vec(vec![20_i64, 21, 22, 23, 24, 25, 26, 27]);

        let result = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: left_index.view(),
                right: right.view(),
                right_index: right_index.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: left_index.view(),
                right: right_second.view(),
                right_index: right_index.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        assert_eq!(result.left_index, vec![10, 11]);
        assert_eq!(result.starts, vec![2, 5]);
        assert_eq!(result.ends, vec![8, 8]);
    }

    #[test]
    fn parsed_windows_dispatch_all_operators_and_boundary_cases() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = PyArray1::from_vec(py, vec![2_i64]);
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21, 22]);

            // `windows()` is the dtype-dispatch boundary used by the public
            // dual-range paths. These expected half-open intervals document
            // the binary-search convention for every supported operator:
            // strict/inclusive less-than creates a suffix, while
            // strict/inclusive greater-than creates a prefix.
            for (operator, expected_start, expected_end) in
                [("<", 2, 3), ("<=", 1, 3), (">", 0, 1), (">=", 0, 2)]
            {
                let predicate = PyTuple::new(
                    py,
                    [
                        left.clone().into_any(),
                        left_index.clone().into_any(),
                        right.clone().into_any(),
                        right_index.clone().into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?;
                let parsed = parse_any_range_parts(
                    &predicate.get_item(0)?,
                    &predicate.get_item(1)?,
                    &predicate.get_item(2)?,
                    &predicate.get_item(3)?,
                    &predicate.get_item(5)?,
                )?;
                let windows = parsed.windows(true).map_err(PyValueError::new_err)?;
                assert_eq!(windows.left_positions, vec![0], "operator {operator}");
                assert_eq!(windows.starts, vec![expected_start], "operator {operator}");
                assert_eq!(windows.ends, vec![expected_end], "operator {operator}");
                assert_eq!(windows.right_index, vec![20, 21, 22]);
            }

            // `retain_empty_windows` is intentional: alignment needs one
            // window per logical left row, even when that row has no match.
            // Check both the full suffix/prefix and empty suffix/prefix
            // boundaries, which are the off-by-one cases most likely to be
            // damaged by a dispatch refactor.
            let boundary_left = PyArray1::from_vec(py, vec![0_i64, 4]);
            let boundary_left_index = PyArray1::from_vec(py, vec![30_i64, 31]);
            for (operator, expected_starts, expected_ends) in [
                ("<", vec![0, 3], vec![3, 3]),
                (">=", vec![0, 0], vec![0, 3]),
            ] {
                let predicate = PyTuple::new(
                    py,
                    [
                        boundary_left.clone().into_any(),
                        boundary_left_index.clone().into_any(),
                        right.clone().into_any(),
                        right_index.clone().into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?;
                let parsed = parse_any_range_parts(
                    &predicate.get_item(0)?,
                    &predicate.get_item(1)?,
                    &predicate.get_item(2)?,
                    &predicate.get_item(3)?,
                    &predicate.get_item(5)?,
                )?;
                let windows = parsed.windows(true).map_err(PyValueError::new_err)?;
                assert_eq!(windows.left_positions, vec![0, 1], "operator {operator}");
                assert_eq!(windows.starts, expected_starts, "operator {operator}");
                assert_eq!(windows.ends, expected_ends, "operator {operator}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn two_range_windows_drop_empty_intersections() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_second = Array1::from_vec(vec![1_i64]);
        let right = Array1::from_vec(vec![1_i64, 2, 3]);
        let right_second = Array1::from_vec(vec![1_i64, 2, 3]);
        let labels = Array1::from_vec(vec![10_i64, 11, 12]);
        let one_left_index = Array1::from_vec(vec![0_i64]);

        let result = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: one_left_index.view(),
                right: right.view(),
                right_index: labels.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: one_left_index.view(),
                right: right_second.view(),
                right_index: labels.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        assert!(result.left_index.is_empty());
        assert!(result.starts.is_empty());
        assert!(result.ends.is_empty());
    }

    #[test]
    fn unordered_labels_are_selected_from_intersected_suffix_windows() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_second = Array1::from_vec(vec![6_i64]);
        let right = Array1::from_vec(vec![1_i64, 3, 5, 7]);
        let right_second = Array1::from_vec(vec![0_i64, 2, 4, 6]);
        let left_index = Array1::from_vec(vec![100_i64]);
        let right_index = Array1::from_vec(vec![40_i64, 10, 30, 20]);

        let windows = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: left_index.view(),
                right: right.view(),
                right_index: right_index.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: left_index.view(),
                right: right_second.view(),
                right_index: right_index.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        // P1 gives [1, 4), P2 gives [0, 3), so the intersection is [1, 3).
        // The labels in that physical window are [10, 30], even though the
        // complete right-index array [40, 10, 30, 20] is unordered.
        assert_eq!(windows.starts, vec![1]);
        assert_eq!(windows.ends, vec![3]);
        assert_eq!(
            choose_range_windows(&windows, Keep::First, false).unwrap(),
            (vec![100], vec![10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last, false).unwrap(),
            (vec![100], vec![30])
        );
    }

    #[test]
    fn arbitrary_windows_use_range_extrema_not_prefix_or_suffix_extrema() {
        let windows = SingleJoinResult {
            left_positions: vec![0, 1],
            left_index: vec![100, 101],
            right_index: vec![40, 10, 30, 20, 50],
            starts: vec![2, 1],
            ends: vec![4, 4],
        };

        // The first window is [2, 4) = [30, 20], and the second is
        // [1, 4) = [10, 30, 20]. Neither is a prefix or suffix. The
        // smallest and largest labels must be selected within each exact
        // interval, not from the surrounding right-index array.
        assert_eq!(
            choose_range_windows(&windows, Keep::First, false).unwrap(),
            (vec![100, 101], vec![20, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last, false).unwrap(),
            (vec![100, 101], vec![30, 30])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Any, false).unwrap(),
            (vec![100, 101], vec![30, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::All, false).unwrap(),
            (vec![100, 100, 101, 101, 101], vec![30, 20, 10, 30, 20])
        );
    }

    #[test]
    fn adaptive_extrema_tree_matches_direct_scan_for_broad_windows() {
        let labels = (0..1_024)
            .map(|position| ((position * 7919) % 1_024) as i64)
            .collect::<Vec<_>>();
        let ranges = (0..64)
            .map(|row| (row * 8, row * 8 + 512))
            .collect::<Vec<_>>();

        let before = EXTREME_TREE_HITS.get();
        let labels_view = ArrayView1::from(labels.as_slice());
        let minimum = adaptive_extreme_positions(labels_view, &ranges, true).unwrap();
        assert_eq!(EXTREME_TREE_HITS.get(), before + 1);
        let maximum = adaptive_extreme_positions(labels_view, &ranges, false).unwrap();
        assert_eq!(EXTREME_TREE_HITS.get(), before + 2);
        assert_eq!(
            minimum,
            extreme_positions_direct(labels_view, &ranges, true)
        );
        assert_eq!(
            maximum,
            extreme_positions_direct(labels_view, &ranges, false)
        );

        let windows = SingleJoinResult {
            left_index: (100..164).collect(),
            right_index: labels.clone(),
            starts: ranges.iter().map(|&(start, _)| start).collect(),
            ends: ranges.iter().map(|&(_, end)| end).collect(),
            ..Default::default()
        };
        for (keep, positions) in [(Keep::First, minimum), (Keep::Last, maximum)] {
            let before = EXTREME_TREE_HITS.get();
            let result = choose_range_windows(&windows, keep, false).unwrap();
            assert_eq!(EXTREME_TREE_HITS.get(), before + 1);
            assert_eq!(result.0, windows.left_index);
            assert_eq!(
                result.1,
                positions.iter().map(|&p| labels[p]).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn adaptive_extrema_packs_far_apart_query_groups() {
        let labels = Array1::from_iter(
            (0..1_000_000_usize).map(|position| ((position.wrapping_mul(7919)) % 1_000_000) as i64),
        );
        let mut ranges = Vec::new();
        for group in 0..16 {
            let group_start = group * 50_000;
            for query in 0..8 {
                let start = group_start + query * 64;
                ranges.push((start, start + 1_024));
            }
        }

        // The packed length is small enough for the tree cutoff, while the
        // hull between the first and last groups is large enough that the old
        // hull-based guard would have selected the direct scan.
        let before = EXTREME_TREE_HITS.get();
        let hull_before = EXTREME_HULL_TREE_HITS.get();
        for minimum in [true, false] {
            let actual = adaptive_extreme_positions(labels.view(), &ranges, minimum).unwrap();
            let expected = extreme_positions_direct(labels.view(), &ranges, minimum);
            assert_eq!(
                actual.iter().map(|&p| labels[p]).collect::<Vec<_>>(),
                expected.iter().map(|&p| labels[p]).collect::<Vec<_>>()
            );
        }
        assert_eq!(EXTREME_TREE_HITS.get(), before + 2);
        assert_eq!(EXTREME_HULL_TREE_HITS.get(), hull_before);
    }

    #[test]
    fn adaptive_extrema_precheck_keeps_sparse_batch_direct() {
        let labels = Array1::from_iter((0..10_000).map(|position| position as i64));
        let ranges = (0..64)
            .map(|row| (row * 100, row * 100 + 8))
            .collect::<Vec<_>>();
        let tree_before = EXTREME_TREE_HITS.get();
        let direct_before = PACKING_PRECHECK_DIRECT_HITS.get();

        let actual = adaptive_extreme_positions(labels.view(), &ranges, true).unwrap();
        let expected = extreme_positions_direct(labels.view(), &ranges, true);

        assert_eq!(actual, expected);
        assert_eq!(EXTREME_TREE_HITS.get(), tree_before);
        assert_eq!(PACKING_PRECHECK_DIRECT_HITS.get(), direct_before + 1);
    }

    #[test]
    fn adaptive_extrema_counter_is_thread_local() {
        let before = EXTREME_TREE_HITS.get();
        std::thread::spawn(|| {
            let labels = Array1::from_iter(0..64);
            let ranges = vec![(0, 64); 16];
            assert_eq!(EXTREME_TREE_HITS.get(), 0);
            assert_eq!(EXTREME_HULL_TREE_HITS.get(), 0);
            adaptive_extreme_positions(labels.view(), &ranges, true).unwrap();
            assert_eq!(EXTREME_TREE_HITS.get(), 1);
            assert_eq!(EXTREME_HULL_TREE_HITS.get(), 1);
        })
        .join()
        .unwrap();
        assert_eq!(EXTREME_TREE_HITS.get(), before);
    }

    #[test]
    fn adaptive_extrema_dispatch_boundaries_and_invalid_ranges() {
        let labels = Array1::from_iter((0..64).map(|p| (p * 17) % 64));
        // At span 64 and eight queries the guard requires total width > 416:
        // 5 * 64 build work + 2 * 8 * ceil(log2(64)) query work.
        for (width, tree_hits) in [(51, 0), (52, 0), (53, 1)] {
            let mut ranges = vec![(0, 64)];
            ranges.extend(vec![(0, 50); 6]);
            ranges.push((0, width));
            for minimum in [true, false] {
                let before = EXTREME_TREE_HITS.get();
                let result = adaptive_extreme_positions(labels.view(), &ranges, minimum).unwrap();
                assert_eq!(EXTREME_TREE_HITS.get(), before + tree_hits);
                assert_eq!(
                    result,
                    extreme_positions_direct(labels.view(), &ranges, minimum)
                );
            }
        }
        for ranges in [vec![], vec![(0, 1)], vec![(0, 64); 3]] {
            let before = EXTREME_TREE_HITS.get();
            assert_eq!(
                adaptive_extreme_positions(labels.view(), &ranges, true).unwrap(),
                extreme_positions_direct(labels.view(), &ranges, true)
            );
            assert_eq!(EXTREME_TREE_HITS.get(), before);
        }
        for range in [(0, 0), (64, 64), (65, 65), (1, 0), (0, 65), (usize::MAX, 1)] {
            assert_eq!(
                adaptive_extreme_positions(labels.view(), &[range], true),
                Err("range window is empty or out of bounds".to_owned())
            );
        }
        assert!(adaptive_extreme_positions(ArrayView1::from(&[]), &[], true)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn adaptive_extrema_handles_non_power_of_two_spans_ties_and_strides() {
        // Offset the queried span and reverse the view: neither contiguous
        // storage nor a power-of-two tree length is part of the contract.
        let labels = Array1::from_iter((0..1031).map(|p| ((p * 7919) % 97) as i64));
        let view = labels.slice(numpy::ndarray::s![..;-1]);
        let ranges = (0..64)
            .map(|p| (3 + p * 8, 516 + p * 8))
            .collect::<Vec<_>>();
        for minimum in [true, false] {
            let before = EXTREME_TREE_HITS.get();
            let result = adaptive_extreme_positions(view, &ranges, minimum).unwrap();
            assert_eq!(EXTREME_TREE_HITS.get(), before + 1);
            let expected = extreme_positions_direct(view, &ranges, minimum);
            for (actual, expected) in result.into_iter().zip(expected) {
                assert_eq!(view[actual], view[expected]);
            }
        }
        assert_eq!(
            choose_extreme_position(view, usize::MAX, usize::MAX, true),
            usize::MAX
        );
        assert_eq!(choose_extreme_position(view, usize::MAX, 0, true), 0);
        assert_eq!(choose_extreme_position(view, 0, usize::MAX, false), 0);
    }

    #[test]
    fn range_extended_entry_point_filters_residuals_after_two_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1, 5, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result = call_range_join_extended_indices(
                py,
                &predicates,
                &[100],
                &[40, 10, 30, 20],
                "all",
            )?
            .expect("the residual predicate should leave one candidate");
            assert_eq!(read_pair(&result), (vec![100], vec![30]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn mixed_range_anchors_search_with_their_own_dtypes() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6.0_f64]).into_any(),
                    PyArray1::from_vec(py, vec![0.0_f64, 2.0, 4.0, 6.0]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result =
                call_range_join_indices(py, &predicates, &[100], &[40, 10, 30, 20], "all", false)?
                    .expect("the mixed-dtype windows should intersect");
            assert_eq!(read_pair(&result), (vec![100, 100], vec![10, 30]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_range_indices_dispatch_every_anchor_dtype() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            macro_rules! check_dtype {
                ($ty:ty) => {{
                    let predicates = PyList::empty(py);
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, vec![2 as $ty]).into_any(),
                            PyArray1::from_vec(py, vec![1 as $ty, 3 as $ty, 5 as $ty]).into_any(),
                            "<".into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, vec![2 as $ty]).into_any(),
                            PyArray1::from_vec(py, vec![0 as $ty, 2 as $ty, 4 as $ty]).into_any(),
                            ">=".into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                    let result = call_range_join_indices(
                        py,
                        &predicates,
                        &[100],
                        &[10, 20, 30],
                        "all",
                        false,
                    )?
                    .expect("every supported dtype should dispatch");
                    assert_eq!(read_pair(&result), (vec![100], vec![20]));
                    Ok::<(), PyErr>(())
                }};
            }

            check_dtype!(i64)?;
            check_dtype!(i32)?;
            check_dtype!(i16)?;
            check_dtype!(i8)?;
            check_dtype!(u64)?;
            check_dtype!(u32)?;
            check_dtype!(u16)?;
            check_dtype!(u8)?;
            check_dtype!(f64)?;
            check_dtype!(f32)?;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_range_indices_handles_infinite_and_nan_anchor_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Infinity is an ordinary ordered endpoint: every finite value
            // is below positive infinity, and negative infinity is the first
            // value in an ascending right layout.  This checks the two
            // boundary values without relying on a finite sentinel.
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![f64::INFINITY]).into_any(),
                    PyArray1::from_vec(py, vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]).into_any(),
                    ">=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![f64::INFINITY]).into_any(),
                    PyArray1::from_vec(py, vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let result =
                call_range_join_indices(py, &predicates, &[100], &[10, 20, 30], "all", false)?
                    .expect("infinite endpoints should produce matches");
            assert_eq!(read_pair(&result), (vec![100, 100], vec![10, 20]));

            // NaN is not part of the sorted-range contract, but it can cross
            // the Python boundary.  The binary-search kernels deliberately
            // document that exact NaN parity is unspecified; this regression
            // only requires the public path to reject neither nor panic.
            let nan_predicates = PyList::empty(py);
            for operator in ["<", ">="] {
                nan_predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![f64::NAN]).into_any(),
                        PyArray1::from_vec(py, vec![-1.0_f64, 1.0]).into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let _ = call_range_join_indices(py, &nan_predicates, &[100], &[10, 20], "all", false)?;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_range_indices_match_reference_for_all_orientations_and_keeps() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let first_left = vec![0_i64, 3, 6];
            let second_left = vec![5_i64, 4, 7];
            let left_index = vec![101_i64, 99, 55];
            let first_right = vec![1_i64, 3, 3, 7];
            let second_right = vec![0_i64, 2, 4, 6];
            // The labels are intentionally not sorted by value. The right
            // values are sorted, as required by the binary-search contract,
            // while the returned labels retain this physical permutation.
            let right_index = vec![40_i64, 10, 30, 20];
            // Four representative pairs cover every operator in both anchor
            // positions, including strict/inclusive and complementary
            // directions. The randomized test below supplies additional
            // pairings. Keeping the deterministic matrix at 4 x 4 instead
            // of 4 x 4 x 4 makes failures easier to attribute while still
            // testing every keep mode against every selected pair.
            let operator_pairs = [("<", "<="), ("<=", ">"), (">", "<"), (">=", ">=")];
            let keeps = ["all", "first", "last", "any"];

            for (first_operator, second_operator) in operator_pairs {
                let predicates = PyList::empty(py);
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &first_left,
                        left_index: &left_index,
                        right: &first_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: first_operator,
                    },
                )?;
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &second_left,
                        left_index: &left_index,
                        right: &second_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: second_operator,
                    },
                )?;
                for keep in keeps {
                    let expected = reference_pairs(
                        &ReferenceFixture {
                            left_index: &left_index,
                            first_left: &first_left,
                            second_left: &second_left,
                            right_index: &right_index,
                            first_right: &first_right,
                            second_right: &second_right,
                        },
                        first_operator,
                        second_operator,
                        keep,
                    );
                    let result = call_range_join_indices(
                        py,
                        &predicates,
                        &left_index,
                        &right_index,
                        keep,
                        false,
                    )?;
                    assert_index_result(result, expected);
                }
            }

            // Building blocks are the unmaterialized public form. They retain
            // one row per surviving left window and expose positional bounds
            // rather than selected labels.
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64]).into_any(),
                    PyArray1::from_vec(py, first_right).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, second_right).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let blocks =
                call_range_join_indices(py, &predicates, &[101], &right_index, "all", true)?
                    .expect("the building-block windows should be non-empty");
            assert_eq!(
                blocks.get_item("starts")?.unwrap().extract::<Vec<i64>>()?,
                vec![1]
            );
            assert_eq!(
                blocks.get_item("ends")?.unwrap().extract::<Vec<i64>>()?,
                vec![2]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn randomized_range_indices_match_bruteforce_reference() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let operators = ["<", "<=", ">", ">="];
            let keeps = ["all", "first", "last", "any"];
            let left_index = vec![100_i64, 101, 102, 103];
            let right_index = vec![30_i64, 10, 40, 20, 50];
            let mut seed = 0x9e3779b97f4a7c15_u64;

            for case in 0..32 {
                let mut first_right = (0..5)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                let mut second_right = (0..5)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                first_right.sort_unstable();
                second_right.sort_unstable();
                let first_left = (0..4)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                let second_left = (0..4)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                let first_operator = operators[case % operators.len()];
                let second_operator = operators[(case * 3 + 1) % operators.len()];
                let keep = keeps[case % keeps.len()];
                let predicates = PyList::empty(py);
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &first_left,
                        left_index: &left_index,
                        right: &first_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: first_operator,
                    },
                )?;
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &second_left,
                        left_index: &left_index,
                        right: &second_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: second_operator,
                    },
                )?;
                let expected = reference_pairs(
                    &ReferenceFixture {
                        left_index: &left_index,
                        first_left: &first_left,
                        second_left: &second_left,
                        right_index: &right_index,
                        first_right: &first_right,
                        second_right: &second_right,
                    },
                    first_operator,
                    second_operator,
                    keep,
                );
                assert_index_result(
                    call_range_join_indices(
                        py,
                        &predicates,
                        &left_index,
                        &right_index,
                        keep,
                        false,
                    )?,
                    expected,
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_range_indices_filter_reference_candidates_before_keep() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let first_left = vec![0_i64, 3, 6];
            let second_left = vec![5_i64, 4, 7];
            let left_index = vec![101_i64, 99, 55];
            let first_right = vec![1_i64, 3, 3, 7];
            let second_right = vec![0_i64, 2, 4, 6];
            let right_index = vec![40_i64, 10, 30, 20];
            let residual_left = vec![0_i64, 3, 6];
            let residual_right = vec![0_i64, 1, 5, 8];
            let residual_operator = "<";
            let first_op = CompareOp::try_from_str("<").unwrap();
            let second_op = CompareOp::try_from_str(">").unwrap();
            let residual_op = CompareOp::try_from_str(residual_operator).unwrap();

            for keep in ["all", "first", "last", "any"] {
                let predicates = PyList::empty(py);
                for (left, right, operator) in [
                    (&first_left, &first_right, "<"),
                    (&second_left, &second_right, ">"),
                ] {
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, left.clone()).into_any(),
                            PyArray1::from_vec(py, right.clone()).into_any(),
                            operator.into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                }
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, residual_left.clone()).into_any(),
                        PyArray1::from_vec(py, residual_right.clone()).into_any(),
                        residual_operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;

                let mut expected = (Vec::new(), Vec::new());
                for left_position in 0..left_index.len() {
                    let mut passing = Vec::new();
                    for right_position in 0..right_index.len() {
                        if first_op.apply(&first_left[left_position], &first_right[right_position])
                            && second_op
                                .apply(&second_left[left_position], &second_right[right_position])
                            && residual_op.apply(
                                &residual_left[left_position],
                                &residual_right[right_position],
                            )
                        {
                            passing.push(right_position);
                        }
                    }
                    for position in select_reference_positions(passing, &right_index, keep) {
                        expected.0.push(left_index[left_position]);
                        expected.1.push(right_index[position]);
                    }
                }
                assert_index_result(
                    call_range_join_extended_indices(
                        py,
                        &predicates,
                        &left_index,
                        &right_index,
                        keep,
                    )?,
                    expected,
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_index_wrappers_reject_malformed_tuples_and_parallel_lengths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let malformed = PyList::empty(py);
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                ],
            )?)?;
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error =
                call_range_join_indices(py, &malformed, &[0], &[1], "all", false).unwrap_err();
            assert!(error
                .to_string()
                .contains("shared-index range predicates must contain 3 elements"));

            let mismatched = PyList::empty(py);
            for (_left_index, operator) in [(vec![0_i64, 1], "<"), (vec![0_i64], ">")] {
                mismatched.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let error =
                call_range_join_indices(py, &mismatched, &[0, 1], &[1], "all", false).unwrap_err();
            assert!(error.to_string().contains("left and left_index"));

            let extended_malformed = PyList::empty(py);
            extended_malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                ],
            )?)?;
            extended_malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error =
                call_range_join_extended_indices(py, &extended_malformed, &[0], &[1], "all")
                    .unwrap_err();
            assert!(error
                .to_string()
                .contains("extended range anchors must contain 3 elements"));

            let empty = PyList::empty(py);
            for operator in ["<", ">"] {
                empty.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            assert!(call_range_join_indices(py, &empty, &[], &[10, 11], "all", false)?.is_none());
            Ok(())
        })
        .unwrap();
    }
}
/// Build and aggregate the per-row windows for a dual-range join.
///
/// Each anchor performs its own typed boundary search. The resulting positional
/// windows are dtype-independent: they are intersected row by row, and the
/// aggregation helper consumes only the surviving windows. With no residual
/// predicates this is the basic two-range operation; with later predicates it
/// is the extended operation.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to parse residuals and create the
///   result tuple.
/// * `predicates` - At least two three-element range tuples. The first two
///   define the intersected windows; later tuples are residual predicates.
/// * `left_index` - Physical left labels and, in forward mode, output slots.
/// * `right_index` - Physical right labels and, in reverse mode, output
///   slots.
/// * `aggregations` - Non-empty source arrays, null masks, and operation
///   names. Forward requests must be right-aligned; reverse requests must be
///   left-aligned.
/// * `return_matched` - Whether the result includes a boolean matched array.
/// * `reverse` - Swap source/output sides while retaining the same candidates.
///
/// # Returns
///
/// Returns `(positions, matched?, results)` or `None` when no complete
/// candidate survives. Result positions retain the supplied physical labels.
///
/// # Errors
///
/// Returns an error for fewer than two predicates, malformed tuples, length
/// mismatches, unsupported residuals, invalid aggregation requests, or source
/// arrays whose lengths do not match the selected source side.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_range_extended<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range extended aggregation requires at least two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_shared_index_range_predicate(first_tuple, left_index, right_index)?;
    let second = parse_shared_index_range_predicate(second_tuple, left_index, right_index)?;
    let (parsed, metadata) =
        super::join_aggregation_helpers::residuals(py, predicates, false, true)?;
    super::predicate::check_predicate_lengths(&parsed, first.left_len(), first.right_len())?;

    let left_index = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let output_positions = if reverse {
        Some(right_index.as_array())
    } else {
        Some(left_index.as_array())
    };
    let output_len = if reverse {
        first.right_len()
    } else {
        first.left_len()
    };
    let source_len = if reverse {
        first.left_len()
    } else {
        first.right_len()
    };
    if !reverse && metadata.is_none() && parsed.len() == 1 {
        if let Some(result) = aggregate_envelope_rank_sweep(
            py,
            &first,
            &second,
            &parsed[0],
            aggregations,
            output_positions,
            output_len,
            source_len,
            return_matched,
        )? {
            return Ok(result);
        }
    }
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    aggregate_range_windows(
        py,
        windows,
        &parsed,
        metadata.as_deref(),
        aggregations,
        output_positions,
        output_len,
        source_len,
        return_matched,
        reverse,
        false,
    )
}

/// What the envelope sweeps need from the first anchor, without storing
/// its windows.
///
/// A `<`/`<=` first anchor gives every left row a suffix of the right
/// layout and a `>`/`>=` anchor a prefix, so the sweep only needs each row's
/// window boundary, which it searches when it builds its queries. Storing
/// the first anchor's windows for every left row cost three vectors of
/// left length (about 1 GB for Arun's 40M rows).
struct FirstAnchorSweep<'a> {
    search: AnchorSearch<'a>,
    /// Every window is a suffix (`<`/`<=`); otherwise every one is a prefix.
    suffix: bool,
    left_len: usize,
    right_len: usize,
    /// Total size of the two anchors' intersected windows: the work the
    /// residual traversal would do, which the sweep's cost gate compares.
    candidate_width: usize,
}

impl<'a> FirstAnchorSweep<'a> {
    /// Search both anchors row by row and total the intersected widths.
    ///
    /// Returns `None` when the second anchor's windows reach beyond the
    /// first anchor's right layout; the caller's window builder reports
    /// that misalignment.
    fn new(
        first: &'a AnyParsedRangePredicate<'_>,
        second: &AnyParsedRangePredicate<'_>,
    ) -> Option<Self> {
        let search = first.search();
        let second_search = second.search();
        let (left_len, right_len) = (first.left_len(), first.right_len());
        // Rows, positions and thresholds are stored as `u32` below.
        if second.left_len() != left_len
            || u32::try_from(left_len).is_err()
            || u32::try_from(right_len).is_err()
        {
            return None;
        }
        // ELI5: the width only feeds a cost guess, so estimate it from a
        // few evenly spaced batches of rows instead of searching every row.
        // An exact pass searched both anchors for every left row, then the
        // fallback searched them all again whenever the sweep declined;
        // sparse joins ran up to 1.5x slower. Misaligned anchors cannot
        // reach the sweep: the envelope proof requires equal right lengths,
        // and the fallback window builder checks every row.
        let sampled = SWEEP_WIDTH_SAMPLE_BLOCKS * SWEEP_WIDTH_SAMPLE_ROWS;
        let mut candidate_width = 0_usize;
        let mut count_width = |_: usize, first: (usize, usize), second: (usize, usize)| {
            candidate_width = candidate_width
                .saturating_add(first.1.min(second.1).saturating_sub(first.0.max(second.0)));
        };
        if left_len <= sampled {
            for_each_window_pair(&search, &second_search, 0..left_len, &mut count_width);
        } else {
            let stride = left_len / SWEEP_WIDTH_SAMPLE_BLOCKS;
            for block in 0..SWEEP_WIDTH_SAMPLE_BLOCKS {
                let start = block * stride;
                let rows = start..start + SWEEP_WIDTH_SAMPLE_ROWS;
                for_each_window_pair(&search, &second_search, rows, &mut count_width);
            }
            candidate_width = (candidate_width as f64 * left_len as f64 / sampled as f64) as usize;
        }
        let suffix = matches!(search.op(), CompareOp::Lt | CompareOp::Le);
        Some(Self {
            search,
            suffix,
            left_len,
            right_len,
            candidate_width,
        })
    }

    /// Right positions visible to a row whose first-anchor window is
    /// `(start, end)`, counted in insertion order: from the end for a suffix
    /// window, from the start for a prefix.
    #[inline]
    fn visible(&self, (start, end): (usize, usize)) -> usize {
        if self.suffix {
            self.right_len.saturating_sub(start)
        } else {
            end
        }
    }

    /// Left rows that can match, bucketed by visible count.
    ///
    /// Returns `(queries, bin_ends)`: every query packs `row` in its low 32
    /// bits and the row's threshold in the high 32, and the queries for
    /// visible count `v` are `queries[bin_ends[v - 1]..bin_ends[v]]`.
    ///
    /// ELI5: sort the left rows into numbered bins, one per visible count,
    /// then read the bins in order. A visible count is at most the right
    /// length, so this counting sort is O(left + right). Packing the
    /// threshold next to the row keeps the sweep reading one array in order
    /// (8 bytes per row) instead of hopping between per-row arrays; sorting
    /// `(visible, threshold, row)` tuples held 24 bytes per row (1.5 GiB of
    /// capacity for 40M rows).
    fn bucket_queries(&self, thresholds: &[u32], rank_count: usize) -> (Vec<u64>, Vec<usize>) {
        let mut visible = Vec::with_capacity(self.left_len);
        let mut bin_ends = vec![0_usize; self.right_len + 1];
        for_each_window(&self.search, self.left_len, |row, window| {
            let count = if (thresholds[row] as usize) < rank_count {
                self.visible(window)
            } else {
                0
            };
            visible.push(count as u32);
            bin_ends[count] += 1;
        });
        // Bin 0 holds rows that cannot match; it stays out of the queries.
        bin_ends[0] = 0;
        for bin in 1..bin_ends.len() {
            bin_ends[bin] += bin_ends[bin - 1];
        }
        // `cursor[v]` is where bin `v` starts: the previous bin's end.
        let mut cursor: Vec<usize> = std::iter::once(0)
            .chain(bin_ends.iter().copied())
            .take(bin_ends.len())
            .collect();
        let mut queries = vec![0_u64; bin_ends[self.right_len]];
        for (row, &count) in visible.iter().enumerate() {
            if count > 0 {
                let slot = &mut cursor[count as usize];
                queries[*slot] = row as u64 | (u64::from(thresholds[row]) << 32);
                *slot += 1;
            }
        }
        (queries, bin_ends)
    }
}

/// Rank thresholds for a residual range predicate over one right layout.
///
/// Returns the rank of every right position and, for every left row, the
/// smallest rank that satisfies `left op right`. Ranks are ordered so that
/// the matching right values are always the ranks at or above the
/// threshold: ascending value order for `<`/`<=`, descending for `>`/`>=`.
/// A threshold equal to the rank count means the row matches nothing.
///
/// ELI5: line the right values up so that "good enough" always means "this
/// far along the line or further"; each left row then only needs to know
/// where its "good enough" begins.
fn residual_rank_thresholds<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    op: super::compare_op::CompareOp,
) -> (Vec<usize>, usize, Vec<u32>) {
    use super::compare_op::CompareOp;
    use std::cmp::Ordering;
    let mut distinct: Vec<T> = right.iter().copied().collect();
    // NaN is outside the range-join contract (AGENTS.md); treating it as
    // equal keeps the sort total instead of panicking for direct callers.
    distinct.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    distinct.dedup_by(|a, b| a.partial_cmp(&b) == Some(Ordering::Equal));
    let count = distinct.len();
    let descending = matches!(op, CompareOp::Gt | CompareOp::Ge);
    let rank_of = |value: T| {
        let ascending = distinct.partition_point(|candidate| *candidate < value);
        if descending {
            count - 1 - ascending
        } else {
            ascending
        }
    };
    let ranks = right.iter().map(|&value| rank_of(value)).collect();
    let thresholds = left
        .iter()
        .map(|&value| match op {
            // left < right: right values above `value`.
            CompareOp::Lt => distinct.partition_point(|candidate| *candidate <= value),
            CompareOp::Le => distinct.partition_point(|candidate| *candidate < value),
            // left > right: right values below `value`, which are the
            // highest descending ranks.
            CompareOp::Gt => count - distinct.partition_point(|candidate| *candidate < value),
            CompareOp::Ge => count - distinct.partition_point(|candidate| *candidate <= value),
            CompareOp::Eq | CompareOp::Ne => unreachable!("caller checks range operators"),
        })
        // The caller declines right layouts longer than `u32::MAX`, so a
        // threshold (at most the distinct count) always fits.
        .map(|threshold| threshold as u32)
        .collect();
    (ranks, count, thresholds)
}

/// Report whether `anchor` is exactly the cumulative envelope of `residual`.
///
/// PyJanitor replaces an opposing second range predicate whose right values
/// are unordered in the first anchor's layout with a monotonic envelope
/// (suffix minimum for `>`/`>=`, prefix maximum for `<`/`<=`) and keeps the
/// original predicate as a residual. When that holds, the envelope window
/// contains every position the residual accepts, so dropping the envelope
/// and filtering the first anchor's window by the residual alone gives the
/// same match set. This check proves it from the values instead of trusting
/// the call shape: equal operators, equal left values, and an anchor right
/// array equal to the envelope recomputed from the residual's right values.
fn anchor_is_envelope_of<T: PartialOrd + Copy>(
    anchor_left: ArrayView1<'_, T>,
    anchor_right: ArrayView1<'_, T>,
    anchor_op: super::compare_op::CompareOp,
    residual_left: ArrayView1<'_, T>,
    residual_right: ArrayView1<'_, T>,
    residual_op: super::compare_op::CompareOp,
) -> bool {
    use super::compare_op::CompareOp;
    if anchor_op != residual_op
        || anchor_left.len() != residual_left.len()
        || anchor_right.len() != residual_right.len()
        || anchor_left.iter().zip(residual_left).any(|(a, b)| a != b)
    {
        return false;
    }
    let len = residual_right.len();
    match residual_op {
        CompareOp::Gt | CompareOp::Ge => {
            // Suffix minimum, read right to left.
            let mut current = None::<T>;
            for position in (0..len).rev() {
                let value = residual_right[position];
                let envelope = match current {
                    Some(previous) if previous < value => previous,
                    _ => value,
                };
                current = Some(envelope);
                if anchor_right[position] != envelope {
                    return false;
                }
            }
            true
        }
        CompareOp::Lt | CompareOp::Le => {
            // Prefix maximum, read left to right.
            let mut current = None::<T>;
            for position in 0..len {
                let value = residual_right[position];
                let envelope = match current {
                    Some(previous) if previous > value => previous,
                    _ => value,
                };
                current = Some(envelope);
                if anchor_right[position] != envelope {
                    return false;
                }
            }
            true
        }
        CompareOp::Eq | CompareOp::Ne => false,
    }
}

/// The sweep has a materially larger fixed build cost than a segment tree.
const RANK_SWEEP_BUILD_FACTOR: usize = 64;
const MIN_RANK_SWEEP_MEMORY_BUDGET: usize = 512 * 1024 * 1024;
const RANK_SWEEP_MEMORY_MULTIPLIER: usize = 3;

fn should_use_rank_sweep(
    query_count: usize,
    total_width: usize,
    source_len: usize,
    rank_count: usize,
) -> bool {
    if query_count <= 3 || source_len == 0 || rank_count == 0 {
        return false;
    }
    let height = (usize::BITS - (rank_count - 1).leading_zeros()) as usize;
    let build = source_len.saturating_mul(RANK_SWEEP_BUILD_FACTOR);
    let queries = query_count.saturating_mul(height).saturating_mul(2);
    total_width > build.saturating_add(queries)
}

fn rank_sweep_workspace_bytes(
    query_count: usize,
    source_len: usize,
    rank_count: usize,
    state_bytes: usize,
) -> Option<usize> {
    let tree_size = rank_count.checked_next_power_of_two()?;
    let tree_nodes = tree_size.checked_mul(2)?;
    let tree_bytes = tree_nodes.checked_mul(state_bytes)?;
    let right_workspace = source_len.checked_mul(3 * std::mem::size_of::<usize>())?;
    let left_workspace = query_count.checked_mul(7 * std::mem::size_of::<usize>())?;
    tree_bytes
        .checked_add(right_workspace)?
        .checked_add(left_workspace)
}

fn rank_sweep_memory_budget(
    source_len: usize,
    query_count: usize,
    operation_count: usize,
) -> usize {
    let input_bytes = source_len
        .saturating_mul(32 + operation_count.saturating_mul(8))
        .saturating_add(query_count.saturating_mul(32));
    input_bytes
        .saturating_mul(RANK_SWEEP_MEMORY_MULTIPLIER)
        .max(MIN_RANK_SWEEP_MEMORY_BUDGET)
}

fn rank_sweep_state_bytes(
    inputs: &[crate::aggregation::input::AggregationInput<'_>],
) -> Option<usize> {
    use crate::aggregation::input::{AggregationInput, AggregationOp};

    let mut bytes: usize = 1;
    let mut has_count_all = false;
    for input in inputs {
        match input {
            AggregationInput::CountAll => has_count_all = true,
            AggregationInput::CountNonNull(_) => bytes = bytes.checked_add(8)?,
            AggregationInput::I64(_, _, op)
            | AggregationInput::I32(_, _, op)
            | AggregationInput::I16(_, _, op)
            | AggregationInput::I8(_, _, op)
            | AggregationInput::U64(_, _, op)
            | AggregationInput::U32(_, _, op)
            | AggregationInput::U16(_, _, op)
            | AggregationInput::U8(_, _, op) => {
                if *op == AggregationOp::CountAll {
                    has_count_all = true;
                    continue;
                }
                if !matches!(
                    op,
                    AggregationOp::Sum | AggregationOp::Min | AggregationOp::Max
                ) {
                    return None;
                }
                bytes = bytes.checked_add(8)?;
            }
            AggregationInput::F64(_, _, op) | AggregationInput::F32(_, _, op) => {
                if *op == AggregationOp::CountAll {
                    has_count_all = true;
                    continue;
                }
                if !matches!(op, AggregationOp::Min | AggregationOp::Max) {
                    return None;
                }
                bytes = bytes.checked_add(8)?;
            }
        }
    }
    if has_count_all {
        bytes = bytes.checked_add(8)?;
    }
    Some(bytes)
}

/// Sample residual survivors to estimate the work left for the ordinary scan.
///
/// ELI5: a rank tree is worthwhile when many cards in the windows survive the
/// residual. The caller compares the estimated survivor count with the rank
/// tree's fixed build cost.
fn residual_density_sample<T: PartialOrd + Copy>(
    windows: &SingleJoinResult,
    left_positions: &[usize],
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    op: super::compare_op::CompareOp,
) -> Option<(usize, usize)> {
    const SAMPLE_ROWS: usize = 32;
    const SAMPLE_POSITIONS: usize = 32;
    let row_count = windows.starts.len();
    if row_count == 0 {
        return None;
    }
    let row_step = row_count.div_ceil(SAMPLE_ROWS).max(1);
    let mut sampled = 0_usize;
    let mut surviving = 0_usize;
    for row in (0..row_count).step_by(row_step).take(SAMPLE_ROWS) {
        let start = windows.starts[row];
        let end = windows.ends[row];
        if start >= end {
            continue;
        }
        let width = end - start;
        let probes = width.min(SAMPLE_POSITIONS);
        for probe in 0..probes {
            let position = start + probe * width / probes;
            sampled += 1;
            surviving += usize::from(op.apply(&left[left_positions[row]], &right[position]));
        }
    }
    (sampled > 0).then_some((sampled, surviving))
}

fn residual_density_sample_dispatch(
    windows: &SingleJoinResult,
    left_positions: &[usize],
    residual: &super::predicate::Predicate<'_>,
) -> Option<(usize, usize)> {
    use super::predicate::PredicateView;
    match residual.view() {
        PredicateView::I64(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::I32(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::I16(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::I8(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::U64(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::U32(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::U16(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::U8(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::F64(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
        PredicateView::F32(left, right, op) => {
            residual_density_sample(windows, left_positions, left, right, op)
        }
    }
}

/// Residual ranks and per-left thresholds when `second` is provably the
/// cumulative envelope of `residual` (see [`anchor_is_envelope_of`]).
///
/// Returns `None` when the residual is not a range predicate of the same
/// dtype or the anchor is not its envelope; callers then keep the general
/// traversal. On success the tuple is `(rank of every right position, rank
/// count, threshold of every left position)` from
/// [`residual_rank_thresholds`].
fn envelope_residual_ranks(
    second: &AnyParsedRangePredicate<'_>,
    residual: &super::predicate::Predicate<'_>,
) -> Option<(Vec<usize>, usize, Vec<u32>)> {
    use super::compare_op::CompareOp;
    use super::predicate::PredicateView;

    macro_rules! ranked {
        ($variant:ident, $anchor:expr) => {{
            let anchor = $anchor;
            let PredicateView::$variant(residual_left, residual_right, residual_op) =
                residual.view()
            else {
                return None;
            };
            if matches!(residual_op, CompareOp::Eq | CompareOp::Ne)
                || !anchor_is_envelope_of(
                    anchor.left.as_array(),
                    anchor.right.as_array(),
                    anchor.op,
                    residual_left,
                    residual_right,
                    residual_op,
                )
            {
                return None;
            }
            Some(residual_rank_thresholds(
                residual_left,
                residual_right,
                residual_op,
            ))
        }};
    }
    match second {
        AnyParsedRangePredicate::I64(anchor) => ranked!(I64, anchor),
        AnyParsedRangePredicate::I32(anchor) => ranked!(I32, anchor),
        AnyParsedRangePredicate::I16(anchor) => ranked!(I16, anchor),
        AnyParsedRangePredicate::I8(anchor) => ranked!(I8, anchor),
        AnyParsedRangePredicate::U64(anchor) => ranked!(U64, anchor),
        AnyParsedRangePredicate::U32(anchor) => ranked!(U32, anchor),
        AnyParsedRangePredicate::U16(anchor) => ranked!(U16, anchor),
        AnyParsedRangePredicate::U8(anchor) => ranked!(U8, anchor),
        AnyParsedRangePredicate::F64(anchor) => ranked!(F64, anchor),
        AnyParsedRangePredicate::F32(anchor) => ranked!(F32, anchor),
    }
}

/// Select `keep="first"` or `"last"` for an envelope-backed dual range
/// without visiting every match.
///
/// Same shape as [`aggregate_envelope_rank_sweep`]: the second anchor is
/// provably the cumulative envelope of the only residual, and every
/// first-anchor window is a suffix (or every one a prefix) of the right
/// layout. Each row then wants the smallest (or largest) right label among
/// the positions in its first-anchor window whose residual rank clears the
/// row's threshold. Right positions are inserted in window-growth order into
/// a Fenwick tree indexed by reversed residual rank, each node keeping the
/// best label below it; a row's answer is one prefix query once its window
/// is fully inserted. That is O((left + right) log right), against the
/// residual scan's one check per candidate: on Arun's 1M x 33K dominance
/// join the scan visited 5.5 billion matches.
///
/// ELI5: right rows walk onto a stage one by one. Each spot on a ladder
/// (ordered by how "good" the residual value is) remembers the smallest
/// ticket number standing at or below it. When a left row's turn comes, it
/// reads one ladder spot instead of checking every ticket.
///
/// Returns `None` when the shape does not apply or the candidate width is
/// too small to repay the tree, so the caller keeps the residual scan.
/// Otherwise the pairs are emitted in left-row order, as the scan does.
fn select_envelope_rank_sweep(
    first: &AnyParsedRangePredicate<'_>,
    second: &AnyParsedRangePredicate<'_>,
    residual: &super::predicate::Predicate<'_>,
    keep: Keep,
) -> Option<(Vec<i64>, Vec<i64>)> {
    let sweep = FirstAnchorSweep::new(first, second)?;
    let (left_len, right_len, suffix) = (sweep.left_len, sweep.right_len, sweep.suffix);
    if !should_use_segment_tree(left_len, sweep.candidate_width, right_len) {
        return None;
    }
    let (sampled, surviving) =
        residual_density_sample_dispatch(windows, &windows.left_positions, residual)?;
    if (surviving as u128) * (candidate_width as u128) / (sampled as u128)
        <= (right_len as u128) * (RANK_SWEEP_BUILD_FACTOR as u128)
    {
        return None;
    }
    let (ranks, rank_count, thresholds) = envelope_residual_ranks(second, residual)?;
    if rank_sweep_workspace_bytes(first_starts.len(), right_len, rank_count, 8)
        .is_none_or(|bytes| bytes > rank_sweep_memory_budget(right_len, first_starts.len(), 1))
    {
        return None;
    }

    #[cfg(test)]
    ENVELOPE_KEEP_SWEEP_HITS.with(|hits| hits.set(hits.get() + 1));
    let (left_labels, labels) = first.labels();
    let better = |candidate: i64, current: i64| match keep {
        Keep::First => candidate < current,
        _ => candidate > current,
    };
    // A row's window holds the first `visible` inserted positions, and
    // positions with rank at or above its threshold sit in reversed-rank
    // slots `1..=rank_count - threshold`.
    let (queries, bin_ends) = sweep.bucket_queries(&thresholds, rank_count);
    drop(thresholds);

    // Tree nodes and selections hold right positions (`NONE` when empty)
    // rather than `Option<i64>` labels: 4 bytes per left row instead of 16.
    const NONE: u32 = u32::MAX;
    let wins = |candidate: u32, current: u32| {
        current == NONE || better(labels[candidate as usize], labels[current as usize])
    };
    let mut tree = vec![NONE; rank_count + 1];
    let mut selected = vec![NONE; left_len];
    for visible in 1..bin_ends.len() {
        // Insert the one position that becomes visible at this count.
        let inserted = visible - 1;
        let position = if suffix {
            right_len - 1 - inserted
        } else {
            inserted
        };
        let mut slot = rank_count - ranks[position];
        while slot <= rank_count {
            if wins(position as u32, tree[slot]) {
                tree[slot] = position as u32;
            }
            slot += slot.isolate_lowest_one();
        }
        for &query in &queries[bin_ends[visible - 1]..bin_ends[visible]] {
            let (row, threshold) = ((query as u32) as usize, (query >> 32) as usize);
            let mut best = NONE;
            let mut slot = rank_count - threshold;
            while slot > 0 {
                if tree[slot] != NONE && wins(tree[slot], best) {
                    best = tree[slot];
                }
                slot -= slot.isolate_lowest_one();
            }
            selected[row] = best;
        }
    }

    // Pairs in left-row order, as the residual scan emits them.
    let matched = selected
        .iter()
        .filter(|&&position| position != NONE)
        .count();
    let mut out_left = Vec::with_capacity(matched);
    let mut out_right = Vec::with_capacity(matched);
    for (row, position) in selected.into_iter().enumerate() {
        if position != NONE {
            out_left.push(left_labels[row]);
            out_right.push(labels[position as usize]);
        }
    }
    Some((out_left, out_right))
}

/// Aggregate an envelope-backed dual range without visiting every pair.
///
/// Applies when the second anchor is the cumulative envelope of the only
/// residual (see [`anchor_is_envelope_of`]), every first-anchor window is a
/// suffix (or every one a prefix) of the right layout, the aggregation is
/// forward, and every requested operation is one the insert-only rank tree
/// supports (counts, integer sums, min, max). Each left row's matches are
/// then "positions in a growing prefix or suffix whose residual rank clears
/// a threshold", which [`AggregationSet::aggregate_rank_threshold_sweep`]
/// answers in O((left + right) log right) regardless of how many pairs
/// match. The old traversal visits every pair in the envelope window, so it
/// is kept whenever the candidate width is too small to repay the sweep.
///
/// ELI5: instead of shaking hands with every match, each left row asks a
/// running scoreboard "how many have arrived that beat my mark?".
///
/// # Returns
///
/// `Ok(None)` when the shape does not apply, so the caller runs the general
/// traversal; otherwise the finished aggregation result.
#[allow(clippy::too_many_arguments)]
fn aggregate_envelope_rank_sweep<'py>(
    py: Python<'py>,
    first: &AnyParsedRangePredicate<'_>,
    second: &AnyParsedRangePredicate<'_>,
    residual: &super::predicate::Predicate<'_>,
    aggregations: &Bound<'py, PyList>,
    output_positions: Option<ArrayView1<'_, i64>>,
    output_len: usize,
    source_len: usize,
    return_matched: bool,
) -> PyResult<Option<Option<Bound<'py, PyTuple>>>> {
    use crate::aggregation::common::aggregation::{
        make_results_with_positions, parse_inputs, AggregationSet,
    };

    let Some(sweep) = FirstAnchorSweep::new(first, second) else {
        return Ok(None);
    };
    let suffix = sweep.suffix;
    // The sweep replaces a traversal of the intersected windows; only take
    // it when that traversal is large next to building and querying a tree.
    if !should_use_segment_tree(sweep.left_len, sweep.candidate_width, source_len) {
        return Ok(None);
    }
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;
    if !set.supports_dual_rank_tree() {
        return Ok(None);
    }

    let Some((ranks, rank_count, thresholds)) = envelope_residual_ranks(second, residual) else {
        return Ok(None);
    };

    if !should_use_rank_sweep(first_starts.len(), candidate_width, source_len, rank_count)
        || rank_sweep_workspace_bytes(first_starts.len(), source_len, rank_count, state_bytes)
            .is_none_or(|bytes| {
                bytes > rank_sweep_memory_budget(source_len, first_starts.len(), inputs.len())
            })
    {
        return Ok(None);
    }

    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;
    if !set.supports_dual_rank_tree() {
        return Ok(None);
    }

    #[cfg(test)]
    ENVELOPE_RANK_SWEEP_HITS.with(|hits| hits.set(hits.get() + 1));
    // ELI5: a suffix window grows as its start moves left, so feed right
    // positions from the end; a prefix window grows as its end moves right.
    let insertion_order: Vec<usize> = if suffix {
        (0..source_len).rev().collect()
    } else {
        (0..source_len).collect()
    };
    let (queries, bin_ends) = sweep.bucket_queries(&thresholds, rank_count);
    drop(thresholds);
    // Every left row is its own compact output slot here, so the packed
    // `(row, threshold)` queries are already `(slot, threshold)`.
    set.aggregate_rank_threshold_sweep(
        &insertion_order,
        &ranks,
        rank_count,
        &queries,
        &bin_ends,
        &inputs,
    );
    if set.is_empty() {
        return Ok(Some(None));
    }
    Ok(Some(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?)))
}

/// Aggregate exactly two range predicates in the forward direction.
///
/// The first two predicates each produce a half-open window over their own
/// sorted right-hand value layout. The Rust range kernel intersects those
/// windows row by row, then aggregates every pair in the intersection. No
/// residual predicates are accepted by this entry point; callers that have
/// additional predicates must use [`range_join_extended_aggregate`].
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two three-element tuples of
///   `(left_values, right_values, operator)` using the shared physical maps.
/// * `left_index` - Physical positions paired with the left anchor values.
/// * `right_index` - Physical positions paired with the sorted right anchor
///   values.
/// * `aggregations` - Non-empty aggregation requests over aligned source
///   arrays. Their offsets match the compact value layouts in `predicates`.
/// * `return_matched` - Include the boolean match array in the returned
///   tuple. It does not change aggregation semantics.
///
/// # Returns
///
/// Returns the standard aggregation tuple, or `None` when the two windows have
/// no intersection. The output remains aligned to the supplied trimmed output
/// layout.
///
/// # Errors
///
/// Returns an error for malformed anchors, mismatched index/value lengths,
/// empty or invalid aggregation requests, or source arrays that are not
/// aligned to the selected aggregation direction.
#[pyfunction]
pub fn range_join_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        false,
    )
}

/// Aggregate exactly two range predicates in the reverse direction.
///
/// Reverse aggregation uses the same intersected windows as forward
/// aggregation, but writes into right-side output slots while consuming
/// left-side source values.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two three-element range-anchor tuples using the
///   shared physical maps.
/// * `left_index` - Physical positions paired with left anchor values.
/// * `right_index` - Physical positions paired with sorted right values.
/// * `aggregations` - Non-empty aggregation requests over aligned left source
///   arrays.
/// * `return_matched` - Whether to include the boolean match array.
///
/// # Returns
///
/// Returns the standard reverse aggregation tuple, or `None` when no pair
/// satisfies both range predicates.
///
/// # Errors
///
/// Returns an error for malformed anchors, mismatched index/value lengths,
/// empty or invalid aggregation requests, or left-side source arrays that are
/// not aligned with the supplied left index.
#[pyfunction]
pub fn range_join_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        true,
    )
}

/// Aggregate two range anchors followed by residual predicates.
///
/// The first two predicates are the only predicates used for binary search.
/// Their windows are intersected row by row. Predicates three onward are
/// evaluated in their original user order for every candidate in that
/// intersection. Aggregation state is updated only after all residuals pass.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two tuples. The first two are three-element
///   range anchors; later tuples are residual filters evaluated in order.
/// * `left_index` - Physical positions paired with the left anchor values.
/// * `right_index` - Physical positions paired with sorted right values.
/// * `aggregations` - Non-empty aggregation requests over aligned source
///   arrays.
/// * `return_matched` - Whether to include one match flag for every output
///   slot.
///
/// # Returns
///
/// Returns the standard forward aggregation tuple, or `None` when no complete
/// candidate survives both range anchors and every residual predicate.
///
/// # Errors
///
/// Returns an error for malformed anchors or residual predicates, mismatched
/// lengths, empty or invalid aggregation requests, or misaligned source
/// arrays. Residual rejection is represented by `None`, not an error.
#[pyfunction]
pub fn range_join_extended_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        false,
    )
}

/// Aggregate two range anchors plus residual predicates in reverse direction.
///
/// The first two predicates build and intersect right-oriented windows. Later
/// predicates are residual filters. Surviving candidates update left-source
/// aggregation state in right output slots.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two tuples. The first two are three-element
///   range anchors; later tuples are residual filters evaluated in order.
/// * `left_index` - Physical positions paired with left anchor values.
/// * `right_index` - Physical positions paired with sorted right values.
/// * `aggregations` - Non-empty aggregation requests over aligned left-side
///   source arrays.
/// * `return_matched` - Whether to include the per-output match mask.
///
/// # Returns
///
/// Returns the standard reverse aggregation tuple, or `None` when no candidate
/// survives the anchors and residual filters.
///
/// # Errors
///
/// Returns an error for malformed anchors or residual predicates, mismatched
/// lengths, empty or invalid aggregation requests, or left-side source arrays
/// that do not match the reverse source layout.
#[pyfunction]
pub fn range_join_extended_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        true,
    )
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;
    use numpy::PyArray1;

    fn call_range_join_aggregate<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_aggregate(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    fn call_range_join_aggregate_reverse<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_aggregate_reverse(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    fn call_range_join_extended_aggregate<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_extended_aggregate(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    fn call_range_join_extended_aggregate_reverse<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_extended_aggregate_reverse(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    #[test]
    fn forward_two_range_aggregation_uses_intersected_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left = PyArray1::from_vec(py, vec![4_i64]);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.clone().into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let mask = PyArray1::from_vec(py, vec![false, false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the range intersection has matches");
            // Both anchors share the physical index arrays passed separately
            // to the entry point. The second anchor uses a different value
            // layout while referring to the same physical right positions.
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100]);
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![40]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn empty_two_range_intersection_returns_none() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11, 12]);
            for op in ["<", ">"] {
                let mut fields = vec![
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                ];
                fields.push(op.into_pyobject(py)?.into_any());
                predicates.append(PyTuple::new(py, fields)?)?;
            }
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            assert!(call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .is_none());

            let empty = PyList::empty(py);
            empty.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            empty.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let empty_left_index = PyArray1::from_vec(py, Vec::<i64>::new());
            let empty_right_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            assert!(call_range_join_aggregate(
                py,
                &empty,
                &empty_left_index,
                &empty_right_index,
                &aggregations,
                false,
            )?
            .is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn reverse_two_range_aggregation_uses_intersected_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            // P1 selects right positions [2, 4): values 5 and 7.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            // P2 also selects a suffix beginning at position two, so the
            // intersection remains right positions [2, 4).
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![4_i64]);
            let mask = PyArray1::from_vec(py, vec![false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = call_range_join_aggregate_reverse(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the reverse range intersection has matches");

            // Reverse aggregation creates one output slot per right row. The
            // left value contributes to the two right positions in the
            // intersected window, not to the source-left slot itself.
            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 4, 4]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_covers_all_operations_maps_nulls_and_return_modes() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            // The first anchor is integer-valued and the second is float-
            // valued.  Each right value array is independently sorted, but
            // both use the same physical right labels.  The intersection is
            // right positions [2, 4) for left row 0 and [3, 4) for row 1.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3.0_f64, 5.0]).into_any(),
                    PyArray1::from_vec(py, vec![0.0_f64, 2.0, 4.0, 6.0]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let nulls = PyArray1::from_vec(py, vec![false, true, false, false]);
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
                ],
            )?;

            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the exact range aggregation has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100, 101]);
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70, 40]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![1200, 40]);
            // min/max return source positions, not source values.
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![2, 3]);
            assert_eq!(outputs.get_item(3)?.extract::<Vec<i64>>()?, vec![3, 3]);
            assert_eq!(outputs.get_item(4)?.extract::<Vec<i64>>()?, vec![2, 1]);
            assert_eq!(outputs.get_item(5)?.extract::<Vec<i64>>()?, vec![2, 1]);

            // Reverse aggregation consumes left values and writes one slot
            // for every right row.  The nontrivial right output map must be
            // returned unchanged, while null left value 6 is skipped by
            // value-based operations but still counts for `size`.
            let left_values = PyArray1::from_vec(py, vec![2_i64, 6]);
            let left_nulls = PyArray1::from_vec(py, vec![false, true]);
            let reverse_aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "prod".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "min".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "max".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            left_nulls.clone().into_any(),
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
                ],
            )?;
            let result = call_range_join_aggregate_reverse(
                py,
                &predicates,
                &left_index,
                &right_index,
                &reverse_aggregations,
                false,
            )?
            .expect("the reverse range aggregation has matches");
            assert_eq!(result.len(), 2);
            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 2, 2]
            );
            assert_eq!(
                outputs.get_item(1)?.extract::<Vec<i64>>()?,
                vec![1, 1, 2, 2]
            );
            assert_eq!(
                outputs.get_item(2)?.extract::<Vec<i64>>()?,
                vec![-1, -1, 0, 0]
            );
            assert_eq!(
                outputs.get_item(3)?.extract::<Vec<i64>>()?,
                vec![-1, -1, 0, 0]
            );
            assert_eq!(
                outputs.get_item(4)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 1]
            );
            assert_eq!(
                outputs.get_item(5)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 2]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_range_aggregation_filters_before_forward_and_reverse_updates() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            // The two range anchors produce the same candidates as the
            // exact test above. The residual `<` removes the row-1 candidate
            // and is evaluated before any aggregation update.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64, 5]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 3, 5, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let nulls = PyArray1::from_vec(py, vec![false, true, false, false]);
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
                ],
            )?;

            let result = call_range_join_extended_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                false,
            )?
            .expect("the residual leaves one left row with matches");
            assert_eq!(result.len(), 2);
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100, 101]);
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70, 0]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![2, 0]);
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![2, 0]);

            let left_values = PyArray1::from_vec(py, vec![2_i64, 6]);
            let left_nulls = PyArray1::from_vec(py, vec![false, true]);
            let reverse_aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            left_nulls.clone().into_any(),
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
                ],
            )?;
            let result = call_range_join_extended_aggregate_reverse(
                py,
                &predicates,
                &left_index,
                &right_index,
                &reverse_aggregations,
                true,
            )?
            .expect("the residual leaves reverse matches");
            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 2, 2]
            );
            assert_eq!(
                outputs.get_item(1)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 1]
            );
            assert_eq!(
                outputs.get_item(2)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 1]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_supports_unsigned_and_float_sources() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let unsigned = PyArray1::from_vec(py, vec![1_u32, 2, 3]);
            let floats = PyArray1::from_vec(py, vec![1.5_f64, 2.5, 3.5]);
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
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("mixed aggregation sources should match");
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<u64>>()?, vec![3]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<f64>>()?, vec![3.5]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_rejects_malformed_tuples_and_parallel_lengths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11, 12]);
            let malformed = PyList::empty(py);
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let aggregation = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, vec![false]).into_any(),
                        "sum".into_pyobject(py)?.into_any(),
                    ],
                )?],
            )?;
            let error = call_range_join_aggregate(
                py,
                &malformed,
                &left_index,
                &right_index,
                &aggregation,
                true,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("shared-index range predicates must contain 3 elements"));

            let mismatched = PyList::empty(py);
            mismatched.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            mismatched.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = call_range_join_aggregate(
                py,
                &mismatched,
                &left_index,
                &right_index,
                &aggregation,
                true,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("right and right_index must have equal lengths"));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_preserves_integer_overflow_semantics() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Keep the original tiny direct case alongside a batch that
            // exercises prefix sums. Both anchors select every right row.
            for (left_rows, right_rows, prefix_hits) in [(1, 2, 0), (4, 5, 1)] {
                let predicates = PyList::empty(py);
                let left_index = PyArray1::from_vec(py, (100..100 + left_rows as i64).collect());
                let right_index =
                    PyArray1::from_vec(py, (1..=right_rows as i64).map(|v| v * 10).collect());
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![0_i64; left_rows]).into_any(),
                        PyArray1::from_vec(py, (0..right_rows as i64).collect()).into_any(),
                        "<=".into_pyobject(py)?.into_any(),
                    ],
                )?)?;
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![0_i64; left_rows]).into_any(),
                        PyArray1::from_vec(py, (0..right_rows as i64).collect()).into_any(),
                        "<=".into_pyobject(py)?.into_any(),
                    ],
                )?)?;

                let mut source = vec![i64::MAX, 2];
                source.resize(right_rows, 1);
                let values = PyArray1::from_vec(py, source);
                let nulls = PyArray1::from_vec(py, vec![false; right_rows]);
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
                                values.into_any(),
                                nulls.into_any(),
                                "prod".into_pyobject(py)?.into_any(),
                            ],
                        )?,
                    ],
                )?;
                let prefix_before = crate::aggregation::state::PREFIX_SUM_HITS.get();
                let result = call_range_join_aggregate(
                    py,
                    &predicates,
                    &left_index,
                    &right_index,
                    &aggregations,
                    false,
                )?
                .expect("the overflow fixture has matches");
                assert_eq!(
                    crate::aggregation::state::PREFIX_SUM_HITS.get(),
                    prefix_before + prefix_hits
                );
                let outputs_item = result.get_item(1)?;
                let outputs = outputs_item.cast::<PyList>()?;
                assert_eq!(
                    outputs.get_item(0)?.extract::<Vec<i64>>()?,
                    vec![i64::MAX.wrapping_add(right_rows as i64); left_rows]
                );
                assert_eq!(
                    outputs.get_item(1)?.extract::<Vec<i64>>()?,
                    vec![i64::MAX.wrapping_mul(2); left_rows]
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn forward_aggregation_skips_a_null_inside_a_matching_window() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            // Both anchors include right position one.  Its source value is
            // null, so it must count toward `size` and matching metadata but
            // must not contribute to `sum` or `count`.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let nulls = PyArray1::from_vec(py, vec![false, true, false]);
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
                ],
            )?;
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the null lies inside the matching window");
            // Forward aggregation writes one slot per left row and returns
            // the source physical label rather than a compact offset.
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100]);
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![30]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![1]);
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![2]);
            Ok(())
        })
        .unwrap();
    }
}

/// The default range path's envelope residual answered by a skip scan.
///
/// These fixtures mimic pyjanitor's default dual-range preparation for an
/// interval overlap: the right rows are sorted by the first anchor, the
/// second range is replaced by its cumulative envelope, and the original
/// second range follows as a residual. A few very long intervals keep the
/// envelope low, so the windows are wide and the skip scan engages.
#[cfg(test)]
mod default_skip_scan_tests {
    use super::*;
    use crate::conditional_join::range_skip_scan::{
        ANY_PROBE_PENDING, ANY_PROBE_RESOLVED, SECOND_RANGE_SUPER_SKIPS, WINDOW_RESIDUAL_MASK_RUNS,
        WINDOW_RESIDUAL_SKIP_DENSE, WINDOW_RESIDUAL_SKIP_PLANS,
    };
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::IntoPyObject;
    use std::collections::BTreeMap;

    struct Overlap {
        left_start: Vec<i64>,
        left_end: Vec<i64>,
        left_tag: Vec<i64>,
        left_tag_mask: Vec<bool>,
        right_start: Vec<i64>,
        right_end: Vec<i64>,
        right_tag: Vec<i64>,
        right_tag_mask: Vec<bool>,
        right_value: Vec<i64>,
        /// Physical labels of the right rows after sorting by `right_end`.
        right_index: Vec<i64>,
        left_index: Vec<i64>,
    }

    fn next(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    /// Short intervals plus a few long ones, right side sorted by end.
    fn overlap_fixture(left_rows: usize, right_rows: usize, seed: u64) -> Overlap {
        let mut seed = seed;
        let intervals = |rows: usize, long_every: u64, seed: &mut u64| {
            (0..rows)
                .map(|_| {
                    let start = (next(seed) % 100_000) as i64;
                    let length = if next(seed).is_multiple_of(long_every) {
                        (next(seed) % 100_000) as i64
                    } else {
                        (next(seed) % 40) as i64
                    };
                    (start, start + length)
                })
                .collect::<Vec<_>>()
        };
        let left = intervals(left_rows, 1_000, &mut seed);
        let mut right = intervals(right_rows, 300, &mut seed);
        let mut labels: Vec<i64> = (0..right_rows as i64)
            .map(|label| label * 7 % 10_007)
            .collect();
        let mut order: Vec<usize> = (0..right_rows).collect();
        order.sort_by_key(|&position| right[position].1);
        right = order.iter().map(|&position| right[position]).collect();
        labels = order.iter().map(|&position| labels[position]).collect();
        let right_tag: Vec<i64> = (0..right_rows)
            .map(|_| (next(&mut seed) % 5) as i64)
            .collect();
        Overlap {
            left_start: left.iter().map(|interval| interval.0).collect(),
            left_end: left.iter().map(|interval| interval.1).collect(),
            left_tag: (0..left_rows)
                .map(|_| (next(&mut seed) % 5) as i64)
                .collect(),
            left_tag_mask: (0..left_rows)
                .map(|_| next(&mut seed).is_multiple_of(11))
                .collect(),
            right_tag_mask: (0..right_rows)
                .map(|_| next(&mut seed).is_multiple_of(13))
                .collect(),
            right_value: (0..right_rows)
                .map(|_| (next(&mut seed) % 1_000) as i64)
                .collect(),
            right_start: right.iter().map(|interval| interval.0).collect(),
            right_end: right.iter().map(|interval| interval.1).collect(),
            right_tag,
            right_index: labels,
            left_index: (0..left_rows as i64).map(|label| 1_000 + label).collect(),
        }
    }

    impl Overlap {
        /// `left.start <= right.end`, `left.end >= envelope`,
        /// `left.end >= right.start`, and optionally a masked
        /// `left.tag != right.tag`.
        fn predicates<'py>(
            &self,
            py: Python<'py>,
            as_float: bool,
            with_tag: bool,
        ) -> PyResult<Bound<'py, PyList>> {
            let array = |values: &[i64]| -> Bound<'py, PyAny> {
                if as_float {
                    PyArray1::from_vec(py, values.iter().map(|&value| value as f64).collect())
                        .into_any()
                } else {
                    PyArray1::from_vec(py, values.to_vec()).into_any()
                }
            };
            let mut envelope = self.right_start.clone();
            for position in (0..envelope.len().saturating_sub(1)).rev() {
                envelope[position] = envelope[position].min(envelope[position + 1]);
            }
            let predicates = PyList::empty(py);
            for (left, right, op) in [
                (&self.left_start, &self.right_end, "<="),
                (&self.left_end, &envelope, ">="),
                (&self.left_end, &self.right_start, ">="),
            ] {
                predicates.append(PyTuple::new(
                    py,
                    [array(left), array(right), op.into_pyobject(py)?.into_any()],
                )?)?;
            }
            if with_tag {
                predicates.append(PyTuple::new(
                    py,
                    [
                        array(&self.left_tag),
                        PyArray1::from_vec(py, self.left_tag_mask.clone()).into_any(),
                        array(&self.right_tag),
                        PyArray1::from_vec(py, self.right_tag_mask.clone()).into_any(),
                        false.into_pyobject(py)?.to_owned().into_any(),
                        "!=".into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            Ok(predicates)
        }

        /// Matching right positions for every left row, ascending.
        fn reference(&self, with_tag: bool) -> Vec<Vec<usize>> {
            (0..self.left_start.len())
                .map(|row| {
                    (0..self.right_start.len())
                        .filter(|&position| {
                            self.left_start[row] <= self.right_end[position]
                                && self.left_end[row] >= self.right_start[position]
                                // NumPy-null semantics: a masked side
                                // passes `!=`.
                                && (!with_tag
                                    || self.left_tag_mask[row]
                                    || self.right_tag_mask[position]
                                    || self.left_tag[row] != self.right_tag[position])
                        })
                        .collect()
                })
                .collect()
        }
    }

    fn plans() -> usize {
        WINDOW_RESIDUAL_SKIP_PLANS.with(|plans| plans.get())
    }

    #[test]
    fn envelope_residual_skip_scan_matches_bruteforce_for_every_keep() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let data = overlap_fixture(500, 5_000, 17);
            let super_skips_before = SECOND_RANGE_SUPER_SKIPS.with(|skips| skips.get());
            for as_float in [false, true] {
                for with_tag in [false, true] {
                    let reference = data.reference(with_tag);
                    assert!(reference.iter().map(Vec::len).sum::<usize>() > 500);
                    for keep in ["all", "any", "first", "last"] {
                        let predicates = data.predicates(py, as_float, with_tag)?;
                        let plans_before = plans();
                        let result = range_join_extended_indices(
                            py,
                            &predicates,
                            PyArray1::from_vec(py, data.left_index.clone()).into_any(),
                            PyArray1::from_vec(py, data.right_index.clone()).into_any(),
                            keep,
                        )?
                        .expect("the overlap has matches");
                        // The envelope rank sweep intentionally supersedes the
                        // skip scan for first/last on this fixture. Those modes
                        // retain dedicated skip-scan coverage in the blocky
                        // residual test below; keep this test focused on the
                        // scan path for all/any.
                        if matches!(keep, "all" | "any") {
                            assert!(plans() > plans_before, "skip scan did not run for {keep}");
                        }
                        let read = |name: &str| -> PyResult<Vec<i64>> {
                            Ok(result
                                .get_item(name)?
                                .expect("index output")
                                .cast::<PyArray1<i64>>()?
                                .readonly()
                                .as_array()
                                .to_vec())
                        };
                        let observed: Vec<(i64, i64)> = read("left_index")?
                            .into_iter()
                            .zip(read("right_index")?)
                            .collect();
                        let mut expected = Vec::new();
                        for (row, positions) in reference.iter().enumerate() {
                            let labels =
                                positions.iter().map(|&position| data.right_index[position]);
                            let selected: Vec<i64> = match keep {
                                "all" => labels.collect(),
                                "any" => labels.take(1).collect(),
                                "first" => labels.min().into_iter().collect(),
                                _ => labels.max().into_iter().collect(),
                            };
                            expected.extend(
                                selected
                                    .into_iter()
                                    .map(|label| (data.left_index[row], label)),
                            );
                        }
                        assert_eq!(
                            observed, expected,
                            "keep={keep} float={as_float} tag={with_tag}"
                        );
                    }
                }
            }
            assert!(SECOND_RANGE_SUPER_SKIPS.with(|skips| skips.get()) > super_skips_before);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn envelope_residual_skip_scan_matches_bruteforce_aggregation() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let data = overlap_fixture(500, 5_000, 29);
            for with_tag in [false, true] {
                let reference = data.reference(with_tag);
                for reverse in [false, true] {
                    // Integer-valued floats keep the float sum exact, so the
                    // brute-force total is directly comparable.
                    let (values, output_index) = if reverse {
                        (
                            (0..data.left_start.len())
                                .map(|row| (row % 17) as f64)
                                .collect::<Vec<_>>(),
                            &data.right_index,
                        )
                    } else {
                        (
                            data.right_value.iter().map(|&value| value as f64).collect(),
                            &data.left_index,
                        )
                    };
                    let mut expected_size: BTreeMap<i64, i64> = BTreeMap::new();
                    let mut expected_sum: BTreeMap<i64, f64> = BTreeMap::new();
                    for (row, positions) in reference.iter().enumerate() {
                        for &position in positions {
                            let (slot, source) = if reverse {
                                (position, row)
                            } else {
                                (row, position)
                            };
                            *expected_size.entry(output_index[slot]).or_default() += 1;
                            *expected_sum.entry(output_index[slot]).or_default() += values[source];
                        }
                    }
                    let aggregations = PyList::empty(py);
                    for name in ["size", "sum"] {
                        aggregations.append(PyTuple::new(
                            py,
                            [
                                PyArray1::from_vec(py, values.clone()).into_any(),
                                PyArray1::from_vec(py, vec![false; values.len()]).into_any(),
                                name.into_pyobject(py)?.into_any(),
                            ],
                        )?)?;
                    }
                    let predicates = data.predicates(py, false, with_tag)?;
                    let left_index = PyArray1::from_vec(py, data.left_index.clone()).into_any();
                    let right_index = PyArray1::from_vec(py, data.right_index.clone()).into_any();
                    let plans_before = plans();
                    let result = if reverse {
                        range_join_extended_aggregate_reverse(
                            py,
                            &predicates,
                            &left_index,
                            &right_index,
                            &aggregations,
                            true,
                        )?
                    } else {
                        range_join_extended_aggregate(
                            py,
                            &predicates,
                            &left_index,
                            &right_index,
                            &aggregations,
                            true,
                        )?
                    }
                    .expect("the overlap has matches");
                    assert!(
                        plans() > plans_before,
                        "skip scan did not run (reverse={reverse})"
                    );
                    let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
                    let matched = result.get_item(1)?.extract::<Vec<bool>>()?;
                    let outputs_item = result.get_item(2)?;
                    let outputs = outputs_item.cast::<PyList>()?;
                    let sizes = outputs.get_item(0)?.extract::<Vec<i64>>()?;
                    let sums = outputs.get_item(1)?.extract::<Vec<f64>>()?;
                    let mut observed_size = BTreeMap::new();
                    let mut observed_sum = BTreeMap::new();
                    for (slot, &label) in positions.iter().enumerate() {
                        if matched[slot] {
                            observed_size.insert(label, sizes[slot]);
                            observed_sum.insert(label, sums[slot]);
                        }
                    }
                    assert_eq!(
                        observed_size, expected_size,
                        "reverse={reverse} tag={with_tag}"
                    );
                    assert_eq!(
                        observed_sum, expected_sum,
                        "reverse={reverse} tag={with_tag}"
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn dense_wide_windows_keep_the_residual_checklist() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Every right interval starts near zero: the windows are wide and
            // almost every candidate matches, so the summaries cannot skip
            // anything and the plan must decline after sampling.
            let mut data = overlap_fixture(300, 5_000, 53);
            for position in 0..data.right_start.len() {
                data.right_start[position] = (position % 3) as i64;
            }
            let reference = data.reference(true);
            for keep in ["all", "first"] {
                let predicates = data.predicates(py, false, true)?;
                let (plans_before, dense_before) = (
                    plans(),
                    WINDOW_RESIDUAL_SKIP_DENSE.with(|dense| dense.get()),
                );
                let result = range_join_extended_indices(
                    py,
                    &predicates,
                    PyArray1::from_vec(py, data.left_index.clone()).into_any(),
                    PyArray1::from_vec(py, data.right_index.clone()).into_any(),
                    keep,
                )?
                .expect("dense join has matches");
                assert_eq!(
                    plans(),
                    plans_before,
                    "dense windows must not use the skip scan"
                );
                assert!(WINDOW_RESIDUAL_SKIP_DENSE.with(|dense| dense.get()) > dense_before);
                let observed = result
                    .get_item("right_index")?
                    .expect("index output")
                    .len()?;
                let expected: usize = if keep == "all" {
                    reference.iter().map(Vec::len).sum()
                } else {
                    reference
                        .iter()
                        .filter(|positions| !positions.is_empty())
                        .count()
                };
                assert_eq!(observed, expected, "keep={keep}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn dense_opened_blocks_mask_the_other_residuals() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // About a third of the 32-row right blocks match the residual on
            // every row and the rest on none: the plan runs, and since the
            // sampled opened blocks all pass, the masked `!=` is checked with
            // the window mask over opened runs.
            let mut data = overlap_fixture(300, 5_000, 61);
            let mut seed = 7_u64;
            for block in 0..data.right_start.len().div_ceil(32) {
                let start = if next(&mut seed).is_multiple_of(3) {
                    0
                } else {
                    1_000_000
                };
                for position in block * 32..((block + 1) * 32).min(data.right_start.len()) {
                    data.right_start[position] = start;
                }
            }
            let reference = data.reference(true);
            for keep in ["all", "any", "first", "last"] {
                let predicates = data.predicates(py, false, true)?;
                let (plans_before, runs_before) =
                    (plans(), WINDOW_RESIDUAL_MASK_RUNS.with(|runs| runs.get()));
                let result = range_join_extended_indices(
                    py,
                    &predicates,
                    PyArray1::from_vec(py, data.left_index.clone()).into_any(),
                    PyArray1::from_vec(py, data.right_index.clone()).into_any(),
                    keep,
                )?
                .expect("blocky join has matches");
                assert!(
                    plans() > plans_before,
                    "keep={keep}: the skip plan must run"
                );
                assert!(
                    WINDOW_RESIDUAL_MASK_RUNS.with(|runs| runs.get()) > runs_before,
                    "keep={keep}: opened runs must be masked"
                );
                let column = |name: &str| -> PyResult<Vec<i64>> {
                    result
                        .get_item(name)?
                        .expect("index output")
                        .extract::<Vec<i64>>()
                };
                let mut observed: Vec<(i64, i64)> = column("left_index")?
                    .into_iter()
                    .zip(column("right_index")?)
                    .collect();
                observed.sort_unstable();
                let mut expected: Vec<(i64, i64)> = Vec::new();
                for (row, positions) in reference.iter().enumerate() {
                    let labels = positions.iter().map(|&position| data.right_index[position]);
                    let left = data.left_index[row];
                    match keep {
                        "all" => expected.extend(labels.map(|label| (left, label))),
                        "any" => expected.extend(labels.take(1).map(|label| (left, label))),
                        "first" => expected.extend(labels.min().map(|label| (left, label))),
                        _ => expected.extend(labels.max().map(|label| (left, label))),
                    }
                }
                expected.sort_unstable();
                assert_eq!(observed, expected, "keep={keep}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn narrow_windows_keep_the_residual_checklist() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Without long right intervals the envelope is tight and the
            // windows hold fewer candidates than twice the right rows.
            let mut data = overlap_fixture(50, 5_000, 41);
            for position in 0..data.right_start.len() {
                data.right_start[position] =
                    data.right_start[position].max(data.right_end[position] - 39);
            }
            let reference = data.reference(false);
            let predicates = data.predicates(py, false, false)?;
            let plans_before = plans();
            let result = range_join_extended_indices(
                py,
                &predicates,
                PyArray1::from_vec(py, data.left_index.clone()).into_any(),
                PyArray1::from_vec(py, data.right_index.clone()).into_any(),
                "all",
            )?;
            assert_eq!(
                plans(),
                plans_before,
                "narrow windows must not build summaries"
            );
            let expected: usize = reference.iter().map(Vec::len).sum();
            let observed = result.map_or(0, |result| {
                result
                    .get_item("left_index")
                    .unwrap()
                    .unwrap()
                    .len()
                    .unwrap()
            });
            assert_eq!(observed, expected);
            Ok(())
        })
        .unwrap();
    }

    #[cfg(test)]
    mod envelope_rank_sweep_tests {
        use super::*;
        use crate::conditional_join::compare_op::CompareOp;
        use numpy::PyArray1;
        use pyo3::IntoPyObject;

        struct XorShift(u64);

        impl XorShift {
            fn below(&mut self, bound: u64) -> i64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (self.0 % bound) as i64
            }
        }

        fn sweep_hits() -> usize {
            ENVELOPE_RANK_SWEEP_HITS.with(|hits| hits.get())
        }

        fn op_str(op: CompareOp) -> &'static str {
            match op {
                CompareOp::Lt => "<",
                CompareOp::Le => "<=",
                CompareOp::Gt => ">",
                CompareOp::Ge => ">=",
                CompareOp::Eq => "==",
                CompareOp::Ne => "!=",
            }
        }

        /// Envelope of `values` for the second operator, as pyjanitor builds it.
        fn envelope(values: &[i64], op: CompareOp) -> Vec<i64> {
            let mut out = values.to_vec();
            if matches!(op, CompareOp::Gt | CompareOp::Ge) {
                for position in (0..out.len().saturating_sub(1)).rev() {
                    out[position] = out[position].min(out[position + 1]);
                }
            } else {
                for position in 1..out.len() {
                    out[position] = out[position].max(out[position - 1]);
                }
            }
            out
        }

        struct Fixture {
            left_first: Vec<i64>,
            right_first: Vec<i64>,
            left_second: Vec<i64>,
            right_second: Vec<i64>,
            values: Vec<i64>,
            nulls: Vec<bool>,
        }

        fn fixture(rng: &mut XorShift, left_len: usize, right_len: usize) -> Fixture {
            let mut right_first: Vec<i64> = (0..right_len).map(|_| rng.below(500)).collect();
            right_first.sort_unstable();
            Fixture {
                left_first: (0..left_len).map(|_| rng.below(500)).collect(),
                right_first,
                left_second: (0..left_len).map(|_| rng.below(500)).collect(),
                right_second: (0..right_len).map(|_| rng.below(500)).collect(),
                values: (0..right_len).map(|_| rng.below(1_000) - 500).collect(),
                nulls: (0..right_len).map(|_| rng.below(10) == 0).collect(),
            }
        }

        fn tuple<'py>(
            py: Python<'py>,
            left: Vec<i64>,
            right: Vec<i64>,
            op: CompareOp,
            as_float: bool,
        ) -> PyResult<Bound<'py, PyTuple>> {
            let (left, right) = if as_float {
                (
                    PyArray1::from_vec(py, left.iter().map(|&v| v as f64 / 2.0).collect())
                        .into_any(),
                    PyArray1::from_vec(py, right.iter().map(|&v| v as f64 / 2.0).collect())
                        .into_any(),
                )
            } else {
                (
                    PyArray1::from_vec(py, left).into_any(),
                    PyArray1::from_vec(py, right).into_any(),
                )
            };
            PyTuple::new(py, [left, right, op_str(op).into_pyobject(py)?.into_any()])
        }

        fn aggregations<'py>(
            py: Python<'py>,
            fixture: &Fixture,
            float_sum: bool,
        ) -> PyResult<Bound<'py, PyList>> {
            let values = PyArray1::from_vec(py, fixture.values.clone()).into_any();
            let nulls = PyArray1::from_vec(py, fixture.nulls.clone()).into_any();
            let mut items = vec![
                PyTuple::new(
                    py,
                    [
                        values.clone(),
                        nulls.clone(),
                        "sum".into_pyobject(py)?.into_any(),
                    ],
                )?,
                PyTuple::new(
                    py,
                    [
                        values.clone(),
                        nulls.clone(),
                        "min".into_pyobject(py)?.into_any(),
                    ],
                )?,
                PyTuple::new(
                    py,
                    [
                        values.clone(),
                        nulls.clone(),
                        "max".into_pyobject(py)?.into_any(),
                    ],
                )?,
                PyTuple::new(
                    py,
                    [
                        "*".into_pyobject(py)?.into_any(),
                        nulls.clone(),
                        "count".into_pyobject(py)?.into_any(),
                    ],
                )?,
                PyTuple::new(
                    py,
                    [
                        values.clone(),
                        nulls.clone(),
                        "size".into_pyobject(py)?.into_any(),
                    ],
                )?,
            ];
            if float_sum {
                let floats =
                    PyArray1::from_vec(py, fixture.values.iter().map(|&v| v as f64).collect());
                items.push(PyTuple::new(
                    py,
                    [
                        floats.into_any(),
                        nulls,
                        "sum".into_pyobject(py)?.into_any(),
                    ],
                )?);
            }
            PyList::new(py, items)
        }

        /// Kernel output: (positions, matched, integer results, float sum).
        type Observed = (Vec<i64>, Vec<bool>, Vec<Vec<i64>>, Option<Vec<f64>>);

        /// Per left row: (matched, sum, min value, max value, non-null count, size).
        type Expected = Vec<(bool, i64, Option<i64>, Option<i64>, i64, i64)>;

        /// Brute force over every pair with the anchor and residual applied as
        /// given, so it also describes fixtures whose anchor is not an envelope.
        #[allow(clippy::needless_range_loop)]
        fn bruteforce_with_anchor(
            fixture: &Fixture,
            first_op: CompareOp,
            anchor_right: &[i64],
            anchor_op: CompareOp,
            residual_op: CompareOp,
        ) -> Expected {
            (0..fixture.left_first.len())
                .map(|row| {
                    let mut want = (false, 0_i64, None::<i64>, None::<i64>, 0, 0);
                    for position in 0..fixture.right_first.len() {
                        let left_second = fixture.left_second[row];
                        if !first_op.apply(&fixture.left_first[row], &fixture.right_first[position])
                            || !anchor_op.apply(&left_second, &anchor_right[position])
                            || !residual_op.apply(&left_second, &fixture.right_second[position])
                        {
                            continue;
                        }
                        want.0 = true;
                        want.5 += 1;
                        if fixture.nulls[position] {
                            continue;
                        }
                        let value = fixture.values[position];
                        want.4 += 1;
                        want.1 = want.1.wrapping_add(value);
                        want.2 = Some(want.2.map_or(value, |current: i64| current.min(value)));
                        want.3 = Some(want.3.map_or(value, |current: i64| current.max(value)));
                    }
                    want
                })
                .collect()
        }

        fn bruteforce(fixture: &Fixture, first_op: CompareOp, second_op: CompareOp) -> Expected {
            // The residual alone is the second predicate; the envelope anchor
            // can only admit a superset, so it is omitted from the reference.
            let always = vec![i64::MIN; fixture.right_first.len()];
            bruteforce_with_anchor(fixture, first_op, &always, CompareOp::Gt, second_op)
        }

        #[allow(clippy::too_many_arguments)]
        fn run(
            py: Python<'_>,
            fixture: &Fixture,
            first_op: CompareOp,
            second_op: CompareOp,
            anchor_right: Vec<i64>,
            residual_op: CompareOp,
            as_float: bool,
            float_sum: bool,
        ) -> PyResult<Observed> {
            let left_len = fixture.left_first.len();
            let right_len = fixture.right_first.len();
            let predicates = PyList::new(
                py,
                [
                    tuple(
                        py,
                        fixture.left_first.clone(),
                        fixture.right_first.clone(),
                        first_op,
                        as_float,
                    )?,
                    tuple(
                        py,
                        fixture.left_second.clone(),
                        anchor_right,
                        second_op,
                        as_float,
                    )?,
                    tuple(
                        py,
                        fixture.left_second.clone(),
                        fixture.right_second.clone(),
                        residual_op,
                        as_float,
                    )?,
                ],
            )?;
            let left_index = PyArray1::from_vec(py, (0..left_len as i64).collect()).into_any();
            let right_index = PyArray1::from_vec(py, (0..right_len as i64).collect()).into_any();
            let result = range_join_extended_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations(py, fixture, float_sum)?,
                true,
            )?
            .expect("fixture has matches");
            let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
            let matched = result.get_item(1)?.extract::<Vec<bool>>()?;
            let outputs = result.get_item(2)?.cast::<PyList>()?.clone();
            let integer = (0..5)
                .map(|item| outputs.get_item(item)?.extract::<Vec<i64>>())
                .collect::<PyResult<Vec<_>>>()?;
            let floats = if float_sum {
                Some(outputs.get_item(5)?.extract::<Vec<f64>>()?)
            } else {
                None
            };
            Ok((positions, matched, integer, floats))
        }

        fn check(
            fixture: &Fixture,
            expected: &Expected,
            (positions, matched, outputs, floats): &Observed,
            context: &str,
        ) {
            assert_eq!(
                positions,
                &(0..expected.len() as i64).collect::<Vec<_>>(),
                "{context}"
            );
            for (row, want) in expected.iter().enumerate() {
                assert_eq!(matched[row], want.0, "{context} matched row {row}");
                if !want.0 {
                    continue;
                }
                assert_eq!(outputs[0][row], want.1, "{context} sum row {row}");
                // Min/max return source positions; only the value is a contract.
                let value_at =
                    |position: i64| (position >= 0).then(|| fixture.values[position as usize]);
                assert_eq!(value_at(outputs[1][row]), want.2, "{context} min row {row}");
                assert_eq!(value_at(outputs[2][row]), want.3, "{context} max row {row}");
                assert_eq!(outputs[3][row], want.4, "{context} count row {row}");
                assert_eq!(outputs[4][row], want.5, "{context} size row {row}");
                if let Some(floats) = floats {
                    assert_eq!(floats[row], want.1 as f64, "{context} float sum row {row}");
                }
            }
        }

        const SHAPES: [(CompareOp, CompareOp); 8] = [
            (CompareOp::Lt, CompareOp::Gt),
            (CompareOp::Lt, CompareOp::Ge),
            (CompareOp::Le, CompareOp::Gt),
            (CompareOp::Le, CompareOp::Ge),
            (CompareOp::Gt, CompareOp::Lt),
            (CompareOp::Gt, CompareOp::Le),
            (CompareOp::Ge, CompareOp::Lt),
            (CompareOp::Ge, CompareOp::Le),
        ];

        #[test]
        fn envelope_rank_sweep_matches_bruteforce() {
            Python::initialize();
            Python::attach(|py| -> PyResult<()> {
                let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
                for (first_op, second_op) in SHAPES {
                    for as_float in [false, true] {
                        let fixture = fixture(&mut rng, 400, 300);
                        let expected = bruteforce(&fixture, first_op, second_op);
                        let hits = sweep_hits();
                        let result = run(
                            py,
                            &fixture,
                            first_op,
                            second_op,
                            envelope(&fixture.right_second, second_op),
                            second_op,
                            as_float,
                            false,
                        )?;
                        assert_eq!(
                            sweep_hits(),
                            hits + 1,
                            "{first_op:?}/{second_op:?} float={as_float} skipped the sweep"
                        );
                        check(
                            &fixture,
                            &expected,
                            &result,
                            &format!("{first_op:?}/{second_op:?} float={as_float}"),
                        );
                    }
                }
                Ok(())
            })
            .unwrap();
        }

        #[test]
        fn keep_any_probes_windows_before_planning_a_skip_scan() {
            Python::initialize();
            Python::attach(|py| -> PyResult<()> {
                let counters = || {
                    (
                        ANY_PROBE_RESOLVED.with(|rows| rows.get()),
                        ANY_PROBE_PENDING.with(|rows| rows.get()),
                        plans(),
                    )
                };
                // The sparse overlap leaves some rows unmatched after the direct
                // probe, and their remaining windows still repay the skip scan.
                // The dense fixture matches every row within the probe, so no
                // summaries are built at all.
                let sparse = overlap_fixture(500, 5_000, 17);
                let mut dense = overlap_fixture(300, 5_000, 53);
                for position in 0..dense.right_start.len() {
                    dense.right_start[position] = (position % 3) as i64;
                }
                for (data, expect_plan) in [(&sparse, true), (&dense, false)] {
                    for with_tag in [false, true] {
                        let reference = data.reference(with_tag);
                        let predicates = data.predicates(py, false, with_tag)?;
                        let (resolved, pending, plans_before) = counters();
                        let result = range_join_extended_indices(
                            py,
                            &predicates,
                            PyArray1::from_vec(py, data.left_index.clone()).into_any(),
                            PyArray1::from_vec(py, data.right_index.clone()).into_any(),
                            "any",
                        )?
                        .expect("the overlap has matches");
                        let (resolved_after, pending_after, plans_after) = counters();
                        assert!(resolved_after > resolved, "no row resolved by the probe");
                        if expect_plan {
                            assert!(pending_after > pending, "no row left for the skip scan");
                            assert!(plans_after > plans_before, "skip scan did not run");
                        } else {
                            assert_eq!(pending_after, pending, "dense rows left the probe");
                            assert_eq!(plans_after, plans_before, "dense join built summaries");
                        }
                        let read = |name: &str| -> PyResult<Vec<i64>> {
                            Ok(result
                                .get_item(name)?
                                .expect("index output")
                                .cast::<PyArray1<i64>>()?
                                .readonly()
                                .as_array()
                                .to_vec())
                        };
                        let observed: Vec<(i64, i64)> = read("left_index")?
                            .into_iter()
                            .zip(read("right_index")?)
                            .collect();
                        // `any` reports each row's first match in position order.
                        let expected: Vec<(i64, i64)> = reference
                            .iter()
                            .enumerate()
                            .filter_map(|(row, positions)| {
                                positions.first().map(|&position| {
                                    (data.left_index[row], data.right_index[position])
                                })
                            })
                            .collect();
                        assert_eq!(observed, expected, "plan={expect_plan} tag={with_tag}");
                    }
                }
                Ok(())
            })
            .unwrap();
        }
        #[test]
        fn envelope_rank_sweep_falls_back_when_not_provable_or_unsupported() {
            Python::initialize();
            Python::attach(|py| -> PyResult<()> {
                let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
                let (first_op, second_op) = (CompareOp::Le, CompareOp::Gt);
                let fixture = fixture(&mut rng, 400, 300);
                let expected = bruteforce(&fixture, first_op, second_op);
                let exact_envelope = envelope(&fixture.right_second, second_op);

                // A genuinely tighter second anchor: its window is not a
                // superset of the residual's matches, so it cannot be dropped.
                let mut tighter = exact_envelope.clone();
                tighter[150] += 1;
                let tighter_expected =
                    bruteforce_with_anchor(&fixture, first_op, &tighter, second_op, second_op);
                let hits = sweep_hits();
                let result = run(
                    py, &fixture, first_op, second_op, tighter, second_op, false, false,
                )?;
                assert_eq!(sweep_hits(), hits, "a non-envelope anchor took the sweep");
                check(&fixture, &tighter_expected, &result, "tighter anchor");

                // A different residual operator is not the anchor's predicate.
                let ge_expected = bruteforce_with_anchor(
                    &fixture,
                    first_op,
                    &exact_envelope,
                    second_op,
                    CompareOp::Ge,
                );
                let hits = sweep_hits();
                let result = run(
                    py,
                    &fixture,
                    first_op,
                    second_op,
                    exact_envelope.clone(),
                    CompareOp::Ge,
                    false,
                    false,
                )?;
                assert_eq!(sweep_hits(), hits, "a mismatched residual took the sweep");
                check(
                    &fixture,
                    &ge_expected,
                    &result,
                    "mismatched residual operator",
                );

                // Floating-point sums keep encounter-order traversal.
                let hits = sweep_hits();
                let result = run(
                    py,
                    &fixture,
                    first_op,
                    second_op,
                    exact_envelope,
                    second_op,
                    false,
                    true,
                )?;
                assert_eq!(sweep_hits(), hits, "a float sum took the sweep");
                check(&fixture, &expected, &result, "float sum");
                Ok(())
            })
            .unwrap();
        }

        #[test]
        fn envelope_rank_sweep_skips_sparse_windows() {
            Python::initialize();
            Python::attach(|py| -> PyResult<()> {
                // Few candidates per row: the traversal is cheaper than a tree.
                let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
                let mut fixture = fixture(&mut rng, 50, 3_000);
                fixture.left_first = (0..50).map(|_| 490 + rng.below(10)).collect();
                let (first_op, second_op) = (CompareOp::Le, CompareOp::Ge);
                let expected = bruteforce(&fixture, first_op, second_op);
                let hits = sweep_hits();
                let result = run(
                    py,
                    &fixture,
                    first_op,
                    second_op,
                    envelope(&fixture.right_second, second_op),
                    second_op,
                    false,
                    false,
                )?;
                assert_eq!(sweep_hits(), hits, "a sparse join took the sweep");
                check(&fixture, &expected, &result, "sparse");
                Ok(())
            })
            .unwrap();
        }

        #[test]
        fn rank_sweep_gate_charges_build_and_workspace() {
            let query_count: usize = 10_000;
            let source_len: usize = 10_000;
            let rank_count: usize = 10_000;
            let height = (usize::BITS - (rank_count - 1).leading_zeros()) as usize;
            let query_cost = query_count * height * 2;
            let crossover = source_len * RANK_SWEEP_BUILD_FACTOR + query_cost;

            assert!(!should_use_rank_sweep(
                query_count,
                crossover,
                source_len,
                rank_count,
            ));
            assert!(should_use_rank_sweep(
                query_count,
                crossover + 1,
                source_len,
                rank_count,
            ));
            assert!(!should_use_rank_sweep(
                3,
                crossover + 1,
                source_len,
                rank_count
            ));

            let bytes = rank_sweep_workspace_bytes(10_000_000, 40_000_000, 10_000_000, 8).unwrap();
            assert!(bytes > 40_000_000 * 3 * std::mem::size_of::<usize>());
            assert!(bytes > 10_000_000 * 7 * std::mem::size_of::<usize>());
            let estimated_input = 10_000_000 * (32 + 4 * 8) + 1_000_000 * 32;
            assert_eq!(
                rank_sweep_memory_budget(10_000_000, 1_000_000, 4),
                estimated_input * RANK_SWEEP_MEMORY_MULTIPLIER
            );
        }
    }
}

/// `keep="first"`/`"last"` on envelope-backed dual ranges via the rank sweep.
#[cfg(test)]
mod envelope_keep_sweep_tests {
    use super::*;
    use crate::conditional_join::compare_op::CompareOp;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::IntoPyObject;

    struct XorShift(u64);

    impl XorShift {
        fn below(&mut self, bound: u64) -> i64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound) as i64
        }
    }

    fn keep_hits() -> usize {
        ENVELOPE_KEEP_SWEEP_HITS.with(|hits| hits.get())
    }

    fn op_str(op: CompareOp) -> &'static str {
        match op {
            CompareOp::Lt => "<",
            CompareOp::Le => "<=",
            CompareOp::Gt => ">",
            CompareOp::Ge => ">=",
            CompareOp::Eq => "==",
            CompareOp::Ne => "!=",
        }
    }

    /// Envelope of `values` for the second operator, as pyjanitor builds it.
    fn envelope(values: &[i64], op: CompareOp) -> Vec<i64> {
        let mut out = values.to_vec();
        if matches!(op, CompareOp::Gt | CompareOp::Ge) {
            for position in (0..out.len().saturating_sub(1)).rev() {
                out[position] = out[position].min(out[position + 1]);
            }
        } else {
            for position in 1..out.len() {
                out[position] = out[position].max(out[position - 1]);
            }
        }
        out
    }

    struct Fixture {
        left_first: Vec<i64>,
        right_first: Vec<i64>,
        left_second: Vec<i64>,
        right_second: Vec<i64>,
        left_extra: Vec<i64>,
        right_extra: Vec<i64>,
        left_index: Vec<i64>,
        right_index: Vec<i64>,
    }

    fn fixture(rng: &mut XorShift, left_len: usize, right_len: usize) -> Fixture {
        let mut right_first: Vec<i64> = (0..right_len).map(|_| rng.below(500)).collect();
        right_first.sort_unstable();
        // Shuffled, unique right labels so first/last are not just the
        // window ends.
        let mut right_index: Vec<i64> = (0..right_len as i64).map(|label| label * 3 + 7).collect();
        for position in (1..right_len).rev() {
            right_index.swap(position, rng.below(position as u64 + 1) as usize);
        }
        Fixture {
            left_first: (0..left_len).map(|_| rng.below(500)).collect(),
            right_first,
            left_second: (0..left_len).map(|_| rng.below(500)).collect(),
            right_second: (0..right_len).map(|_| rng.below(500)).collect(),
            left_extra: (0..left_len).map(|_| rng.below(4)).collect(),
            right_extra: (0..right_len).map(|_| rng.below(4)).collect(),
            left_index: (0..left_len as i64).map(|label| 10_000 + label).collect(),
            right_index,
        }
    }

    fn tuple<'py>(
        py: Python<'py>,
        left: &[i64],
        right: &[i64],
        op: CompareOp,
        as_float: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let (left, right) = if as_float {
            (
                PyArray1::from_vec(py, left.iter().map(|&v| v as f64 / 2.0).collect()).into_any(),
                PyArray1::from_vec(py, right.iter().map(|&v| v as f64 / 2.0).collect()).into_any(),
            )
        } else {
            (
                PyArray1::from_vec(py, left.to_vec()).into_any(),
                PyArray1::from_vec(py, right.to_vec()).into_any(),
            )
        };
        PyTuple::new(py, [left, right, op_str(op).into_pyobject(py)?.into_any()])
    }

    /// Every pair checked directly, with the anchor applied as given.
    #[allow(clippy::too_many_arguments)]
    fn bruteforce(
        fixture: &Fixture,
        first_op: CompareOp,
        anchor_right: &[i64],
        second_op: CompareOp,
        with_extra: bool,
        keep: &str,
    ) -> (Vec<i64>, Vec<i64>) {
        let mut out = (Vec::new(), Vec::new());
        for row in 0..fixture.left_first.len() {
            let labels = (0..fixture.right_first.len())
                .filter(|&position| {
                    first_op.apply(&fixture.left_first[row], &fixture.right_first[position])
                        && second_op.apply(&fixture.left_second[row], &anchor_right[position])
                        && second_op
                            .apply(&fixture.left_second[row], &fixture.right_second[position])
                        && (!with_extra || fixture.left_extra[row] != fixture.right_extra[position])
                })
                .map(|position| fixture.right_index[position]);
            let best = if keep == "first" {
                labels.min()
            } else {
                labels.max()
            };
            if let Some(label) = best {
                out.0.push(fixture.left_index[row]);
                out.1.push(label);
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        py: Python<'_>,
        fixture: &Fixture,
        first_op: CompareOp,
        anchor_right: &[i64],
        second_op: CompareOp,
        with_extra: bool,
        as_float: bool,
        keep: &str,
    ) -> PyResult<(Vec<i64>, Vec<i64>)> {
        let mut items = vec![
            tuple(
                py,
                &fixture.left_first,
                &fixture.right_first,
                first_op,
                as_float,
            )?,
            tuple(py, &fixture.left_second, anchor_right, second_op, as_float)?,
            tuple(
                py,
                &fixture.left_second,
                &fixture.right_second,
                second_op,
                as_float,
            )?,
        ];
        if with_extra {
            items.push(tuple(
                py,
                &fixture.left_extra,
                &fixture.right_extra,
                CompareOp::Ne,
                as_float,
            )?);
        }
        let predicates = PyList::new(py, items)?;
        let result = range_join_extended_indices(
            py,
            &predicates,
            PyArray1::from_vec(py, fixture.left_index.clone()).into_any(),
            PyArray1::from_vec(py, fixture.right_index.clone()).into_any(),
            keep,
        )?;
        let Some(result) = result else {
            return Ok((Vec::new(), Vec::new()));
        };
        let read = |name: &str| -> PyResult<Vec<i64>> {
            Ok(result
                .get_item(name)?
                .expect("index output")
                .cast::<PyArray1<i64>>()?
                .readonly()
                .as_array()
                .to_vec())
        };
        Ok((read("left_index")?, read("right_index")?))
    }

    const RANGE_OPS: [CompareOp; 4] = [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge];

    #[test]
    fn envelope_keep_sweep_matches_bruteforce() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let mut rng = XorShift(0x5EED_F00D);
            for first_op in RANGE_OPS {
                for second_op in RANGE_OPS {
                    let fixture = fixture(&mut rng, 400, 300);
                    let anchor = envelope(&fixture.right_second, second_op);
                    // The opposing shapes are the ones pyjanitor builds. The
                    // same-direction shapes are exact too: an envelope always
                    // admits a superset of the residual's matches, whatever
                    // the first window's direction.
                    for as_float in [false, true] {
                        for keep in ["first", "last"] {
                            let before = keep_hits();
                            let observed = run(
                                py, &fixture, first_op, &anchor, second_op, false, as_float, keep,
                            )?;
                            let expected =
                                bruteforce(&fixture, first_op, &anchor, second_op, false, keep);
                            assert!(!expected.0.is_empty());
                            assert_eq!(
                                observed, expected,
                                "{first_op:?} {second_op:?} float={as_float} keep={keep}"
                            );
                            assert!(
                                keep_hits() > before,
                                "sweep did not run for {first_op:?} {second_op:?} keep={keep}"
                            );
                        }
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn envelope_keep_sweep_estimates_width_on_large_left() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // More left rows than the width sample covers, so the gate runs
            // on the sampled estimate rather than an exact pass.
            let mut rng = XorShift(0x1A26_E5A3);
            let fixture = fixture(&mut rng, 12_000, 300);
            let (first_op, second_op) = (CompareOp::Le, CompareOp::Gt);
            let anchor = envelope(&fixture.right_second, second_op);
            for keep in ["first", "last"] {
                let before = keep_hits();
                let observed = run(
                    py, &fixture, first_op, &anchor, second_op, false, false, keep,
                )?;
                assert!(keep_hits() > before, "sweep did not run for keep={keep}");
                assert_eq!(
                    observed,
                    bruteforce(&fixture, first_op, &anchor, second_op, false, keep),
                    "keep={keep}"
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn envelope_keep_sweep_falls_back_when_not_provable() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let mut rng = XorShift(0xFA11_BAC4);
            let fixture = fixture(&mut rng, 400, 300);
            let (first_op, second_op) = (CompareOp::Le, CompareOp::Gt);
            let envelope = envelope(&fixture.right_second, second_op);
            let mut tighter = envelope.clone();
            tighter[150] += 1;
            for (anchor, with_extra, why) in [
                (&tighter, false, "anchor is not the envelope"),
                (&envelope, true, "a second residual"),
            ] {
                for keep in ["first", "last"] {
                    let before = keep_hits();
                    let observed = run(
                        py, &fixture, first_op, anchor, second_op, with_extra, false, keep,
                    )?;
                    assert_eq!(keep_hits(), before, "sweep ran with {why}");
                    assert_eq!(
                        observed,
                        bruteforce(&fixture, first_op, anchor, second_op, with_extra, keep),
                        "{why} keep={keep}"
                    );
                }
            }
            // `any` and `all` keep the residual scan.
            for keep in ["any", "all"] {
                let before = keep_hits();
                run(
                    py, &fixture, first_op, &envelope, second_op, false, false, keep,
                )?;
                assert_eq!(keep_hits(), before, "sweep ran for keep={keep}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn envelope_keep_sweep_skips_sparse_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let mut rng = XorShift(0x005B_A45E);
            let mut fixture = fixture(&mut rng, 400, 300);
            // Left values near the top of the first anchor leave each row a
            // few candidates, too few to repay the tree.
            for value in &mut fixture.left_first {
                *value = 490 + *value % 10;
            }
            let (first_op, second_op) = (CompareOp::Le, CompareOp::Gt);
            let anchor = envelope(&fixture.right_second, second_op);
            let before = keep_hits();
            let observed = run(
                py, &fixture, first_op, &anchor, second_op, false, false, "first",
            )?;
            assert_eq!(keep_hits(), before, "sparse windows must keep the scan");
            assert_eq!(
                observed,
                bruteforce(&fixture, first_op, &anchor, second_op, false, "first")
            );
            Ok(())
        })
        .unwrap();
    }
}

/// Streaming window intersection against the stored-window reference.
#[cfg(test)]
mod streaming_window_tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::IntoPyObject;

    struct XorShift(u64);

    impl XorShift {
        fn below(&mut self, bound: u64) -> i64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound) as i64
        }
    }

    fn anchor<'py>(
        py: Python<'py>,
        left: &[i64],
        right: &[i64],
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        op: &str,
        as_float: bool,
    ) -> PyResult<AnyParsedRangePredicate<'py>> {
        let (left, right) = if as_float {
            (
                PyArray1::from_vec(py, left.iter().map(|&v| v as f64 / 4.0).collect()).into_any(),
                PyArray1::from_vec(py, right.iter().map(|&v| v as f64 / 4.0).collect()).into_any(),
            )
        } else {
            (
                PyArray1::from_vec(py, left.to_vec()).into_any(),
                PyArray1::from_vec(py, right.to_vec()).into_any(),
            )
        };
        parse_any_range_parts(
            &left,
            left_index,
            &right,
            right_index,
            &op.into_pyobject(py)?.into_any(),
        )
    }

    #[test]
    fn streaming_windows_match_stored_intersection() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let mut rng = XorShift(0xC0FF_EE11);
            for (left_len, right_len) in [(0, 5), (5, 0), (1, 1), (300, 200), (1_000, 37)] {
                let mut first_right: Vec<i64> = (0..right_len).map(|_| rng.below(100)).collect();
                first_right.sort_unstable();
                // A sorted second layout, as an envelope anchor would be.
                let mut second_right: Vec<i64> = (0..right_len).map(|_| rng.below(100)).collect();
                second_right.sort_unstable();
                let first_left: Vec<i64> = (0..left_len).map(|_| rng.below(110) - 5).collect();
                let second_left: Vec<i64> = (0..left_len).map(|_| rng.below(110) - 5).collect();
                let left_index =
                    PyArray1::from_vec(py, (0..left_len as i64).map(|v| v * 2 + 1).collect())
                        .into_any();
                let right_index =
                    PyArray1::from_vec(py, (0..right_len as i64).map(|v| 500 - v).collect())
                        .into_any();
                for first_op in ["<", "<=", ">", ">="] {
                    for second_op in ["<", "<=", ">", ">="] {
                        for as_float in [false, true] {
                            let first = anchor(
                                py,
                                &first_left,
                                &first_right,
                                &left_index,
                                &right_index,
                                first_op,
                                as_float,
                            )?;
                            let second = anchor(
                                py,
                                &second_left,
                                &second_right,
                                &left_index,
                                &right_index,
                                second_op,
                                as_float,
                            )?;
                            let streamed = build_any_windows(&first, &second)
                                .map_err(PyValueError::new_err)?;
                            let stored = intersect_windows(
                                first.windows(true).map_err(PyValueError::new_err)?,
                                second.windows(false).map_err(PyValueError::new_err)?,
                            )
                            .map_err(PyValueError::new_err)?;
                            let context = format!(
                                "{left_len}x{right_len} {first_op} {second_op} float={as_float}"
                            );
                            assert_eq!(streamed.left_positions, stored.left_positions, "{context}");
                            assert_eq!(streamed.left_index, stored.left_index, "{context}");
                            assert_eq!(streamed.starts, stored.starts, "{context}");
                            assert_eq!(streamed.ends, stored.ends, "{context}");
                            if left_len > 0 {
                                assert_eq!(streamed.right_index, stored.right_index, "{context}");
                            }
                        }
                    }
                }
            }
            Ok(())
        })
        .unwrap();
    }
}

#[cfg(test)]
mod review_probe_tests {
    use super::*;
    use numpy::PyArray1;
    use pyo3::IntoPyObject;

    fn t<'py>(py: Python<'py>, l: Vec<i64>, r: Vec<i64>, op: &str) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [PyArray1::from_vec(py, l).into_any(), PyArray1::from_vec(py, r).into_any(), op.into_pyobject(py)?.into_any()])
    }

    fn probe(short_left: bool, keep: &str) -> String {
        Python::initialize();
        Python::attach(|py| {
            let (n, m) = (12_000_usize, 300_usize);
            let lf: Vec<i64> = (0..n as i64).map(|v| v % 500).collect();
            let rf: Vec<i64> = (0..m as i64).map(|v| v * 500 / m as i64).collect();
            let ls: Vec<i64> = (0..n as i64).map(|v| (v * 7) % 500).collect();
            let rs: Vec<i64> = (0..m as i64).map(|v| (v * 13) % 500).collect();
            let mut env = rs.clone();
            for p in (0..m - 1).rev() { env[p] = env[p].min(env[p + 1]); }
            let preds = PyList::new(py, [
                t(py, lf, rf, "<=").unwrap(),
                t(py, ls.clone(), env, ">").unwrap(),
                t(py, ls, rs, ">").unwrap(),
            ]).unwrap();
            let mut li: Vec<i64> = (0..n as i64).collect();
            let mut ri: Vec<i64> = (0..m as i64).collect();
            if short_left { li.pop(); } else { ri.pop(); }
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                range_join_extended_indices(py, &preds, PyArray1::from_vec(py, li).into_any(), PyArray1::from_vec(py, ri).into_any(), keep)
            }));
            match res {
                Err(_) => "RUST PANIC".to_owned(),
                Ok(Err(e)) => format!("PyErr: {e}"),
                Ok(Ok(_)) => "Ok".to_owned(),
            }
        })
    }

    #[test]
    fn review_probe_label_length_mismatch() {
        for keep in ["first", "all"] {
            eprintln!("PROBE short left_index keep={keep}: {}", probe(true, keep));
            eprintln!("PROBE short right_index keep={keep}: {}", probe(false, keep));
        }
    }
}
