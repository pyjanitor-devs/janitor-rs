use std::collections::{BTreeMap, HashMap};

use crate::aggregation_common::adaptive::should_use_segment_tree;
use crate::aggregation_common::aggregation::{
    make_results_with_positions, parse_inputs, AggregationSet,
};
use crate::aggregation_common::ensure_equal_lengths_core;
use crate::compare_op::CompareOp;
use crate::join_search::partition_point;
use crate::join_types::{result_dict, Keep};
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicate_tuple,
    parse_predicates_with_nulls_strings, predicates_match_dispatch, Predicate, PredicateView,
};
use numpy::{ndarray::ArrayView1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// The second predicate's paths together with the first-predicate starts.
pub(crate) struct DualRegionPaths {
    pub(crate) left_region: Vec<i64>,
    pub(crate) right_region: Vec<i64>,
    pub(crate) starts: Vec<usize>,
    pub(crate) end: usize,
    pub(crate) left_order: Vec<usize>,
}

type RegionPaths = (Vec<i64>, Vec<i64>);
/// Flat storage layout for positions grouped by a right-region label.
///
/// The map stores `(count, offset)` for each label. `offset..offset + count`
/// is the slice in the accompanying flat position buffer for that label; the
/// second tuple item is the total buffer length required by all labels.
pub(crate) struct RankedRegionPositionLayout {
    labels: Vec<i64>,
    offsets: Vec<usize>,
    ranks: Vec<usize>,
    positions: Vec<usize>,
}

/// Point-update/suffix-query tree for the best right index at each rank.
///
/// ELI5: as the sweep exposes another right row, update its region's bucket.
/// A suffix query then skips all ineligible buckets in O(log R), instead of
/// scanning every active region for every left row.
struct RankedExtremeTree {
    min: Vec<Option<i64>>,
    max: Vec<Option<i64>>,
    base: usize,
}

impl RankedExtremeTree {
    fn new(rank_count: usize) -> Self {
        let base = rank_count.max(1).next_power_of_two();
        Self {
            min: vec![None; base * 2],
            max: vec![None; base * 2],
            base,
        }
    }

    fn update(&mut self, rank: usize, value: i64) {
        let mut node = self.base + rank;
        self.min[node] = Some(self.min[node].map_or(value, |current| current.min(value)));
        self.max[node] = Some(self.max[node].map_or(value, |current| current.max(value)));
        node /= 2;
        while node > 0 {
            self.min[node] = min_optional(self.min[node * 2], self.min[node * 2 + 1]);
            self.max[node] = max_optional(self.max[node * 2], self.max[node * 2 + 1]);
            node /= 2;
        }
    }

    fn suffix_extreme(&self, start: usize, keep: Keep) -> Option<i64> {
        let mut left = self.base + start;
        let mut right = self.base * 2;
        let mut result = None;
        while left < right {
            if left % 2 == 1 {
                result = match keep {
                    Keep::First => min_optional(result, self.min[left]),
                    Keep::Last => max_optional(result, self.max[left]),
                    _ => unreachable!("ranked extrema only supports first/last"),
                };
                left += 1;
            }
            if right % 2 == 1 {
                right -= 1;
                result = match keep {
                    Keep::First => min_optional(result, self.min[right]),
                    Keep::Last => max_optional(result, self.max[right]),
                    _ => unreachable!("ranked extrema only supports first/last"),
                };
            }
            left /= 2;
            right /= 2;
        }
        result
    }
}

fn min_optional(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    left.into_iter().chain(right).min()
}

fn max_optional(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    left.into_iter().chain(right).max()
}

/// Counts produced by the first pass of `keep="all"` materialization.
pub(crate) struct KeepAllFirstPass {
    pub(crate) left_match_counts: Vec<usize>,
    pub(crate) total_size: usize,
    /// Final count of every nonzero right-region label.
    pub(crate) right_region_layout: RankedRegionPositionLayout,
}

/// Shared candidate/layout data produced while counting a residual pass.
///
/// For `keep="all"`, `left_match_counts` contains exact counts. For the
/// selected modes it contains only zero-or-one counts because the first pass
/// stops after the first match for each left row.
struct DualResidualCandidatePass {
    left_match_counts: Vec<usize>,
    total_matches: usize,
    selected_right_positions: Vec<Option<usize>>,
    right_region_layout: RankedRegionPositionLayout,
}

/// Materialized index pairs produced by the second `keep="all"` pass.
pub(crate) struct MaterializedIndices {
    pub(crate) left_index: Vec<i64>,
    pub(crate) right_index: Vec<i64>,
}

/// Count the matching left rows for `keep="any"`.
pub(crate) struct MatchCountPass {
    pub(crate) match_count: usize,
    pub(crate) left_has_match: Vec<bool>,
}

/// First-pass results for `keep="any"`.
pub(crate) struct AnyMatchPass {
    pub(crate) match_count: usize,
    pub(crate) selected_right_positions: Vec<Option<usize>>,
}

/// Find left rows whose second region can possibly match the current window.
///
/// ELI5: the first predicate gives each left row a right-side window. If the
/// largest second-region label inside that window is still smaller than the
/// left label, searching the ranked labels cannot find a match.
///
/// # Arguments
///
/// * `regions` - Dual paths in descending-left search order with monotonic
///   first-predicate starts.
///
/// # Returns
///
/// Returns one cheap possibility flag per reordered left row and a summary
/// flag indicating whether any row can match. This does not residual-filter
/// or enumerate candidate positions.
///
/// # Errors
///
/// Returns an error when starts are not monotonically non-increasing.
fn dual_region_match_filter(regions: &DualRegionPaths) -> Result<(Vec<bool>, bool), String> {
    let pass = count_matching_left_rows(regions)?;
    Ok((pass.left_has_match, pass.match_count != 0))
}

/// Parse a three-element region predicate using shared left/right indices.
///
/// The Python-facing predicate is `(left_values, right_values, operator)`.
/// Index arrays are shared by both primary predicates and are supplied as
/// separate arguments to the outer entry point.
fn parse_region_predicate<'py>(predicate: &Bound<'py, PyTuple>) -> PyResult<Predicate<'py>> {
    parse_predicate_tuple(predicate)
}

/// Validate that a parsed predicate is a strict range predicate.
fn validate_region_predicate(predicate: &Predicate<'_>) -> PyResult<()> {
    let op = match predicate.view() {
        PredicateView::I64(_, _, op)
        | PredicateView::I32(_, _, op)
        | PredicateView::I16(_, _, op)
        | PredicateView::I8(_, _, op)
        | PredicateView::U64(_, _, op)
        | PredicateView::U32(_, _, op)
        | PredicateView::U16(_, _, op)
        | PredicateView::U8(_, _, op)
        | PredicateView::F64(_, _, op)
        | PredicateView::F32(_, _, op) => op,
    };
    if matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return Err(PyValueError::new_err(
            "region predicates must use <, <=, >, or >=",
        ));
    }
    Ok(())
}

/// Return the suffix boundary for a range predicate.
///
/// ELI5: right values for `<`/`<=` arrive ascending, while right values for
/// `>`/`>=` arrive descending. In both layouts, matching values form a suffix,
/// so every operator can return one suffix boundary.
fn range_boundary<T: PartialOrd + Copy>(
    left_value: T,
    right: numpy::ndarray::ArrayView1<'_, T>,
    op: CompareOp,
) -> usize {
    match op {
        CompareOp::Lt => partition_point(right, |value| value <= left_value),
        CompareOp::Le => partition_point(right, |value| value < left_value),
        CompareOp::Gt => partition_point(right, |value| value >= left_value),
        CompareOp::Ge => partition_point(right, |value| value > left_value),
        CompareOp::Eq | CompareOp::Ne => {
            unreachable!("range_boundary only handles range operators")
        }
    }
}

/// Reindex a region path from sorted-right order into canonical-right order.
///
/// `alignment[global_position]` contains the source position to read from the
/// predicate-specific right layout. Rust applies the gather permutation, while
/// PyJanitor remains responsible for choosing and supplying the sorting
/// contract.
///
/// # Arguments
///
/// * `right_region` - Labels in predicate-specific sorted-right order.
/// * `alignment` - Optional global-to-source permutation; `None` means the
///   path is already in global order.
///
/// # Returns
///
/// Returns a new vector in global right order.
///
/// # Errors
///
/// Returns an error for mismatched lengths, negative/out-of-bounds positions,
/// or duplicate alignment positions.
fn realign_right_region(
    right_region: Vec<i64>,
    alignment: Option<&ArrayView1<'_, i64>>,
) -> Result<Vec<i64>, String> {
    let Some(alignment) = alignment else {
        return Ok(right_region);
    };

    if alignment.len() != right_region.len() {
        return Err("right alignment and right region must have equal lengths".to_owned());
    }

    let mut realigned = vec![0_i64; right_region.len()];
    let mut seen = vec![false; right_region.len()];
    for (global_position, &source_position) in alignment.iter().enumerate() {
        let source_position = usize::try_from(source_position)
            .map_err(|_| "right alignment contains a negative position".to_owned())?;
        if source_position >= realigned.len() {
            return Err("right alignment contains an out-of-bounds position".to_owned());
        }
        if seen[source_position] {
            return Err("right alignment must be a permutation".to_owned());
        }
        seen[source_position] = true;
        realigned[global_position] = right_region[source_position];
    }

    Ok(realigned)
}

/// Clear second-predicate right positions that the first predicate cannot use.
///
/// ELI5: after the second path has been put into global right-index order,
/// copy the first path's zeroes onto it so both paths agree on impossible
/// right rows.
fn mask_right_region(
    right_region: &mut [i64],
    first_right_region: Option<&[i64]>,
) -> Result<(), String> {
    let Some(first_right_region) = first_right_region else {
        return Ok(());
    };

    ensure_equal_lengths_core(
        "right_region",
        right_region.len(),
        "first_right_region",
        first_right_region.len(),
    )?;

    for (region, &first_region) in right_region.iter_mut().zip(first_right_region) {
        if first_region == 0 {
            *region = 0;
        }
    }

    Ok(())
}

/// Propagate zero regions across a collection of aligned region paths.
///
/// A zero in any path means that the corresponding position cannot satisfy
/// the full predicate set. Every path therefore receives the same zero mask.
///
/// # Arguments
///
/// * `regions` - Mutable position-aligned left or right region paths.
///
/// # Errors
///
/// Returns an error when the paths have different lengths.
fn mask_zero_regions(regions: &mut [Vec<i64>]) -> Result<(), String> {
    let Some(first_region) = regions.first() else {
        return Ok(());
    };

    for region in regions.iter().skip(1) {
        ensure_equal_lengths_core("region", region.len(), "first region", first_region.len())?;
    }

    for position in 0..first_region.len() {
        if regions.iter().any(|region| region[position] == 0) {
            for region in regions.iter_mut() {
                region[position] = 0;
            }
        }
    }

    Ok(())
}

/// Find the first right position whose first-region label reaches each left
/// first-region label.
///
/// ELI5: the right path is sorted by region number, so binary search finds the
/// first right row that belongs to the same-or-later region. A zero label from
/// either predicate means that left row has no final match, so its start is
/// placed at `end` and its suffix is empty. This avoids searching for a start
/// that will never be used.
///
/// # Arguments
///
/// * `first_left_region` - First-path labels in descending left search order.
/// * `second_left_region` - Second-path labels in the same left order.
/// * `right_region` - Monotonic first-path labels in global right order.
///
/// # Returns
///
/// Returns one right-window start per left row, or `None` if every row has an
/// empty intersection.
///
/// # Errors
///
/// Returns an error when the two left paths have different lengths.
fn first_region_starts(
    first_left_region: &[i64],
    second_left_region: &[i64],
    right_region: &[i64],
) -> Result<Option<Vec<usize>>, String> {
    ensure_equal_lengths_core(
        "first_left_region",
        first_left_region.len(),
        "second_left_region",
        second_left_region.len(),
    )?;

    debug_assert!(
        right_region.windows(2).all(|pair| pair[0] <= pair[1]),
        "first right region must be monotonically nondecreasing"
    );

    let mut has_match = false;
    let starts = first_left_region
        .iter()
        .zip(second_left_region)
        .map(|(&first_left_value, &second_left_value)| {
            let start = if first_left_value == 0 || second_left_value == 0 {
                right_region.len()
            } else {
                right_region.partition_point(|&right_value| right_value < first_left_value)
            };
            if start < right_region.len() {
                has_match = true;
            }
            start
        })
        .collect::<Vec<_>>();

    if !has_match {
        return Ok(None);
    }

    Ok(Some(starts))
}

/// Find n-region search starts from the first predicate alone.
///
/// Only the first right path participates in this initial binary search.
/// Later paths may be non-monotonic and are checked directly for each
/// candidate position by the n-region visitor.
///
/// # Arguments
///
/// * `first_left_region` - First-path labels in descending left order.
/// * `first_right_region` - Monotonic first-path labels in global right order.
///
/// # Returns
///
/// Returns one start per left row, or `None` when no row has a non-empty
/// first-predicate window.
fn n_region_starts(
    first_left_region: &[i64],
    first_right_region: &[i64],
) -> Result<Option<Vec<usize>>, String> {
    // The first right path is the only path used for the initial binary
    // search. Later paths are checked candidate-by-candidate, so they do not
    // need to be monotonic.
    debug_assert!(
        first_right_region.windows(2).all(|pair| pair[0] <= pair[1]),
        "first right region must be monotonically nondecreasing"
    );

    let mut has_match = false;
    let starts = first_left_region
        .iter()
        .map(|&left_region| {
            let start = if left_region == 0 {
                first_right_region.len()
            } else {
                first_right_region.partition_point(|&right_region| right_region < left_region)
            };
            if start < first_right_region.len() {
                has_match = true;
            }
            start
        })
        .collect::<Vec<_>>();

    if !has_match {
        return Ok(None);
    }

    Ok(Some(starts))
}

/// Return original left positions ordered by descending first-region label.
///
/// The returned permutation controls the sweep order without changing the
/// public left index. Callers use it to translate a search-order row back to
/// its original position before emitting results or updating aggregation.
fn descending_left_order(first_left_region: &[i64]) -> Vec<usize> {
    let mut left_order = (0..first_left_region.len()).collect::<Vec<_>>();
    left_order
        .sort_unstable_by(|&left, &right| first_left_region[right].cmp(&first_left_region[left]));
    left_order
}

/// Reorder one left-region path using an existing original-position map.
///
/// The result is in search order; `left_order` remains the map back to the
/// caller's original left positions.
fn reorder_left_region(region: &[i64], left_order: &[usize]) -> Vec<i64> {
    left_order
        .iter()
        .map(|&position| region[position])
        .collect()
}

/// Count left rows that have at least one eligible right position.
///
/// ELI5: starts move from right to left, so the search window grows one slice
/// at a time. We only need the largest right-region label seen so far; if it
/// reaches the left label, that left row has an `any` match.
pub(crate) fn count_matching_left_rows(
    regions: &DualRegionPaths,
) -> Result<MatchCountPass, String> {
    let mut max_right_region = i64::MIN;
    let mut previous_end = regions.end;
    let mut match_count = 0usize;
    let mut left_has_match = vec![false; regions.left_region.len()];

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }

        for &right_region in &regions.right_region[start..previous_end] {
            max_right_region = max_right_region.max(right_region);
        }
        previous_end = start;

        if left_region != 0 && left_region <= max_right_region {
            left_has_match[left_position] = true;
            match_count += 1;
        }
    }

    Ok(MatchCountPass {
        match_count,
        left_has_match,
    })
}

/// Find and retain one matching right position for each left row.
///
/// ELI5: `keep="any"` does not need a second search. While finding whether a
/// left row matches, remember the right position that currently has the
/// largest region value.
pub(crate) fn count_any_matches(regions: &DualRegionPaths) -> Result<AnyMatchPass, String> {
    let mut max_right_region = i64::MIN;
    let mut max_right_position = None;
    let mut previous_end = regions.end;
    let mut match_count = 0usize;
    let mut selected_right_positions = vec![None; regions.left_region.len()];

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }

        for right_position in start..previous_end {
            let right_region = regions.right_region[right_position];
            if right_region > max_right_region {
                max_right_region = right_region;
                max_right_position = Some(right_position);
            }
        }
        previous_end = start;

        if left_region != 0 && left_region <= max_right_region {
            selected_right_positions[left_position] = max_right_position;
            match_count += 1;
        }
    }

    Ok(AnyMatchPass {
        match_count,
        selected_right_positions,
    })
}

/// Materialize one matching pair for each left row that has an `any` match.
pub(crate) fn keep_any_second_pass(
    regions: &DualRegionPaths,
    first_pass: &AnyMatchPass,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
) -> Result<Option<MaterializedIndices>, String> {
    if first_pass.match_count == 0 {
        return Ok(None);
    }

    let mut left_output = Vec::with_capacity(first_pass.match_count);
    let mut right_output = Vec::with_capacity(first_pass.match_count);

    for (left_position, right_position) in first_pass.selected_right_positions.iter().enumerate() {
        let Some(right_position) = right_position else {
            continue;
        };
        let original_left_position = regions.left_order[left_position];
        left_output.push(left_index[original_left_position]);
        right_output.push(right_index[*right_position]);
    }

    if left_output.len() != first_pass.match_count {
        return Err("keep-any second pass count disagrees with first pass".to_owned());
    }

    Ok(Some(MaterializedIndices {
        left_index: left_output,
        right_index: right_output,
    }))
}

/// Materialize the smallest or largest right-index value for each matching
/// left row.
///
/// ELI5: the map remembers the best right index seen for each region while
/// the search window grows. `first` keeps the smallest index and `last` keeps
/// the largest; physical right-array position is not used for selection.
pub(crate) fn keep_first_last_second_pass(
    regions: &DualRegionPaths,
    first_pass: &MatchCountPass,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    keep: Keep,
) -> Result<Option<MaterializedIndices>, String> {
    if first_pass.match_count == 0 {
        return Ok(None);
    }

    let right_region_layout =
        build_ranked_region_position_layout(&regions.right_region, &regions.starts, regions.end)?;
    let mut active_best = RankedExtremeTree::new(right_region_layout.labels.len());
    let mut left_output = Vec::with_capacity(first_pass.match_count);
    let mut right_output = Vec::with_capacity(first_pass.match_count);
    let mut previous_end = regions.end;

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }

        for right_position in start..previous_end {
            let rank = right_region_layout.ranks[right_position];
            if rank == usize::MAX {
                continue;
            }

            active_best.update(rank, right_index[right_position]);
        }
        previous_end = start;

        if !first_pass.left_has_match[left_position] || left_region == 0 {
            continue;
        }

        let first_rank = right_region_layout
            .labels
            .partition_point(|&label| label < left_region);
        let selected = active_best.suffix_extreme(first_rank, keep);
        let Some(right_value) = selected else {
            continue;
        };

        let original_left_position = regions.left_order[left_position];
        left_output.push(left_index[original_left_position]);
        right_output.push(right_value);
    }

    if left_output.len() != first_pass.match_count {
        return Err("keep first/last second pass count disagrees with first pass".to_owned());
    }

    Ok(Some(MaterializedIndices {
        left_index: left_output,
        right_index: right_output,
    }))
}

/// Count all matching right positions for each ordered left row.
///
/// ELI5: starts move from right to left, so each right position becomes active
/// once. Sorted labels let us jump to the first eligible label, while the
/// active counts tell us how many positions from each label are visible.
///
/// The returned per-left counts are authoritative for the range predicates.
/// A later residual predicate may reduce them, but must never increase them.
/// This makes the counts suitable for `count(*)` and for sizing a later index
/// materialization without repeating the range search.
pub(crate) fn keep_all_first_pass(regions: &DualRegionPaths) -> Result<KeepAllFirstPass, String> {
    let (left_has_match, any_left_match) = dual_region_match_filter(regions)?;
    if !any_left_match {
        return Ok(KeepAllFirstPass {
            left_match_counts: vec![0; regions.left_region.len()],
            total_size: 0,
            right_region_layout: RankedRegionPositionLayout {
                labels: Vec::new(),
                offsets: Vec::new(),
                ranks: Vec::new(),
                positions: Vec::new(),
            },
        });
    }

    let right_region_layout =
        build_ranked_region_position_layout(&regions.right_region, &regions.starts, regions.end)?;
    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut left_match_counts = Vec::with_capacity(regions.starts.len());
    let mut total_matches = 0usize;
    let mut previous_end = regions.end;

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            left_match_counts.push(0);
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        activate_dual_ranked_positions(
            &right_region_layout,
            start,
            previous_end,
            &mut active_counts,
        )?;
        previous_end = start;

        if !left_has_match[left_position] {
            left_match_counts.push(0);
            continue;
        }

        let first_rank = right_region_layout
            .labels
            .partition_point(|&label| label < left_region);
        let matches = active_counts
            .range(first_rank..)
            .map(|(_, &count)| count)
            .sum::<usize>();
        left_match_counts.push(matches);
        total_matches = total_matches
            .checked_add(matches)
            .ok_or("region result size exceeds platform capacity")?;
    }
    Ok(KeepAllFirstPass {
        left_match_counts,
        total_size: total_matches,
        right_region_layout,
    })
}

/// Materialize every matching pair using the exact sizes from the first pass.
pub(crate) fn keep_all_second_pass(
    regions: &DualRegionPaths,
    first_pass: &KeepAllFirstPass,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
) -> Result<Option<MaterializedIndices>, String> {
    if first_pass.total_size == 0 {
        return Ok(None);
    }
    let mut offsets = vec![0usize; first_pass.left_match_counts.len()];
    let mut offset = 0usize;
    for (position, &size) in first_pass.left_match_counts.iter().enumerate() {
        offsets[position] = offset;
        offset += size;
    }
    debug_assert_eq!(offset, first_pass.total_size);

    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut positions = vec![0usize; first_pass.total_size];
    let mut left_output = vec![0_i64; first_pass.total_size];
    let mut right_output = vec![0_i64; first_pass.total_size];
    let mut previous_end = regions.end;

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        activate_dual_ranked_positions(
            &first_pass.right_region_layout,
            start,
            previous_end,
            &mut active_counts,
        )?;
        previous_end = start;

        let match_count = first_pass.left_match_counts[left_position];
        if match_count == 0 {
            continue;
        }

        let output_start = offsets[left_position];
        let output_end = output_start + match_count;
        let mut output_position = output_start;
        let first_rank = first_pass
            .right_region_layout
            .labels
            .partition_point(|&label| label < left_region);
        for (&rank, &count) in active_counts.range(first_rank..) {
            let region_offset = first_pass.right_region_layout.offsets[rank];
            for &right_position in
                &first_pass.right_region_layout.positions[region_offset..region_offset + count]
            {
                positions[output_position] = right_position;
                output_position += 1;
            }
        }
        if output_position != output_end {
            return Err("keep-all second pass count disagrees with first pass".to_owned());
        }
    }
    for (left_position, &match_count) in first_pass.left_match_counts.iter().enumerate() {
        let output_start = offsets[left_position];
        let output_end = output_start + match_count;
        let original_left_position = regions.left_order[left_position];
        for output_position in output_start..output_end {
            left_output[output_position] = left_index[original_left_position];
            right_output[output_position] = right_index[positions[output_position]];
        }
    }

    Ok(Some(MaterializedIndices {
        left_index: left_output,
        right_index: right_output,
    }))
}

/// Count the positions needed for each label that enters a search window.
///
/// ELI5: the second right region is the lookup key. We reserve one contiguous
/// slice for every key, then reuse those slices during both passes instead of
/// allocating a vector for every key.
///
/// # Arguments
///
/// * `second_right_region` - Aligned second-path labels in global right order.
/// * `starts` - Descending-left window starts.
/// * `end` - Exclusive right-side length shared by all paths.
///
/// # Returns
///
/// Returns a flat label/rank/position layout. Position slices are allocated
/// only for labels that enter at least one search window.
///
/// # Errors
///
/// Returns an error for non-monotonic starts, arithmetic overflow, or an
/// internal activation-count disagreement.
fn build_ranked_region_position_layout(
    second_right_region: &[i64],
    starts: &[usize],
    end: usize,
) -> Result<RankedRegionPositionLayout, String> {
    let mut counts = HashMap::<i64, usize>::new();
    for &region in second_right_region {
        if region != 0 {
            let count = counts.entry(region).or_default();
            *count = (*count)
                .checked_add(1)
                .ok_or("region result size exceeds platform capacity")?;
        }
    }

    let mut labels = counts.keys().copied().collect::<Vec<_>>();
    labels.sort_unstable();

    let mut label_to_rank = HashMap::<i64, usize>::with_capacity(labels.len());
    for (rank, &label) in labels.iter().enumerate() {
        label_to_rank.insert(label, rank);
    }

    let mut ranks = vec![usize::MAX; second_right_region.len()];
    for (right_position, &region) in second_right_region.iter().enumerate() {
        if region == 0 {
            continue;
        }
        let rank = label_to_rank[&region];
        ranks[right_position] = rank;
    }

    // Count only positions that the sweep can actually see. Some right rows
    // may never enter a first-predicate window, so allocating for every
    // nonzero label would make the layout invariant too strong.
    let mut active_totals = vec![0usize; labels.len()];
    let mut previous_end = end;
    for &start in starts {
        if start == end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        for &rank in &ranks[start..previous_end] {
            if rank != usize::MAX {
                active_totals[rank] = active_totals[rank]
                    .checked_add(1)
                    .ok_or("region result size exceeds platform capacity")?;
            }
        }
        previous_end = start;
    }

    let mut offsets = vec![0usize; labels.len()];
    let mut position_total = 0usize;
    for (rank, &count) in active_totals.iter().enumerate() {
        offsets[rank] = position_total;
        position_total = position_total
            .checked_add(count)
            .ok_or("region result size exceeds platform capacity")?;
    }

    // Reproduce the sweep's insertion order once. This lets each label's
    // active positions remain a prefix of its precomputed slice without
    // changing the output order established by the original B-tree path.
    let mut positions = vec![0usize; position_total];
    let mut position_counts = vec![0usize; labels.len()];
    previous_end = end;
    for &start in starts {
        if start == end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        for (right_position, &rank) in ranks
            .iter()
            .enumerate()
            .skip(start)
            .take(previous_end - start)
        {
            if rank == usize::MAX {
                continue;
            }
            let position = offsets[rank] + position_counts[rank];
            positions[position] = right_position;
            position_counts[rank] += 1;
        }
        previous_end = start;
    }

    if position_counts != active_totals {
        return Err("second right region layout count disagrees with the sweep".to_owned());
    }

    Ok(RankedRegionPositionLayout {
        labels,
        offsets,
        ranks,
        positions,
    })
}

/// Add dual-region positions to the encountered-rank map.
///
/// # Arguments
///
/// * `layout` - Precomputed right-label ranks and flat position slices.
/// * `start` / `previous_end` - The newly visible half-open window.
/// * `active_counts` - Per-rank counts accumulated while the sweep moves
///   leftward.
///
/// # Errors
///
/// Returns an error if the window moves in the wrong direction or a count
/// overflows platform capacity.
fn activate_dual_ranked_positions(
    layout: &RankedRegionPositionLayout,
    start: usize,
    previous_end: usize,
    active_counts: &mut BTreeMap<usize, usize>,
) -> Result<(), String> {
    if start > previous_end {
        return Err("starts must be monotonically non-increasing".to_owned());
    }

    for &rank in &layout.ranks[start..previous_end] {
        if rank == usize::MAX {
            continue;
        }
        let count = active_counts.entry(rank).or_default();
        *count = count
            .checked_add(1)
            .ok_or("region result size exceeds platform capacity")?;
    }
    Ok(())
}

/// Add the right positions in one newly visible sweep window.
///
/// ELI5: each label owns a shelf in `positions`. `active_counts` records how
/// many items on each shelf are visible for the current left row.
///
/// # Arguments
///
/// * `layout` - Precomputed rank and flat-position storage.
/// * `start` / `previous_end` - Newly visible half-open right window.
/// * `active_counts` - Mutable count prefix for each label rank.
///
/// # Errors
///
/// Returns an error for non-monotonic windows or count overflow.
fn activate_ranked_positions(
    layout: &RankedRegionPositionLayout,
    start: usize,
    previous_end: usize,
    active_counts: &mut BTreeMap<usize, usize>,
) -> Result<(), String> {
    if start > previous_end {
        return Err("starts must be monotonically non-increasing".to_owned());
    }

    for &rank in &layout.ranks[start..previous_end] {
        if rank == usize::MAX {
            continue;
        }
        let count = active_counts.entry(rank).or_default();
        *count = count
            .checked_add(1)
            .ok_or("region result size exceeds platform capacity")?;
    }

    Ok(())
}

/// Visit n-region candidates in descending first-region order.
///
/// The first two region paths define the search window. The remaining paths
/// are progressively checked for each candidate, and `.all(...)` stops at the
/// first failed predicate. `visit` returns `false` when the caller has enough
/// information for the current left row (for example, `keep="any"`).
///
/// The second right-region positions are grouped once by sorted label and in
/// the same activation order as the search sweep. The sweep only increments
/// encountered ranks, so candidate traversal does not scan labels that have
/// not entered the current window.
///
/// This function deliberately knows nothing about index materialization or
/// aggregation. Its callback receives the reordered left position and the
/// global right position, allowing callers to count, select, materialize, or
/// update aggregate accumulators without duplicating the range traversal.
///
/// # Arguments
///
/// * `left_regions` / `right_regions` - Equal-length aligned region paths.
/// * `starts` - One descending-left search start per left row.
/// * `end` - Exclusive right-side length.
/// * `right_region_layout` - Ranked layout for the second right region.
/// * `visit` - Callback receiving search-order left and global-right positions;
///   return `false` to stop candidates for the current left row.
///
/// # Returns
///
/// Returns the number of range candidates accepted by the callback.
///
/// # Errors
///
/// Returns an error for path-length mismatches, invalid starts, or result-size
/// overflow. A callback rejection is normal control flow, not an error.
fn visit_n_region_matches<F>(
    left_regions: &[Vec<i64>],
    right_regions: &[Vec<i64>],
    starts: &[usize],
    end: usize,
    right_region_layout: &RankedRegionPositionLayout,
    mut visit: F,
) -> Result<usize, String>
where
    F: FnMut(usize, usize) -> bool,
{
    if left_regions.len() < 3 || right_regions.len() != left_regions.len() {
        return Err("n-region paths must contain at least three aligned regions".to_owned());
    }
    if starts.len() != left_regions[0].len() {
        return Err("n-region starts must match the left-region length".to_owned());
    }
    for region in left_regions.iter().skip(1) {
        ensure_equal_lengths_core(
            "left region",
            region.len(),
            "first left region",
            left_regions[0].len(),
        )?;
    }
    for region in right_regions {
        ensure_equal_lengths_core("right region", region.len(), "first right region", end)?;
    }

    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut right_region_maxima = vec![0_i64; right_regions.len()];
    let mut previous_end = end;
    let mut total_matches = 0usize;

    for (left_position, &start) in starts.iter().enumerate() {
        if start == end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }

        for right_position in start..previous_end {
            for (maximum, region) in right_region_maxima.iter_mut().zip(right_regions) {
                *maximum = (*maximum).max(region[right_position]);
            }
        }
        activate_ranked_positions(right_region_layout, start, previous_end, &mut active_counts)?;
        previous_end = start;

        let impossible = left_regions
            .iter()
            .zip(&right_region_maxima)
            .any(|(left_region, &right_maximum)| left_region[left_position] > right_maximum);
        if impossible {
            continue;
        }

        let second_left_region = left_regions[1][left_position];
        let first_rank = right_region_layout
            .labels
            .partition_point(|&label| label < second_left_region);
        'candidate_regions: for (&rank, &region_count) in active_counts.range(first_rank..) {
            let region_offset = right_region_layout.offsets[rank];
            for &right_position in
                &right_region_layout.positions[region_offset..region_offset + region_count]
            {
                let matches_remaining = left_regions
                    .iter()
                    .skip(2)
                    .zip(right_regions.iter().skip(2))
                    .all(|(left_region, right_region)| {
                        left_region[left_position] <= right_region[right_position]
                    });
                if matches_remaining {
                    total_matches = total_matches
                        .checked_add(1)
                        .ok_or("region result size exceeds platform capacity")?;
                    if !visit(left_position, right_position) {
                        break 'candidate_regions;
                    }
                }
            }
        }
    }

    Ok(total_matches)
}

/// Count residual-passing candidates for each left row.
///
/// ELI5: the first pass uses one flat position buffer. Sorted labels identify
/// the eligible region suffix, and active counts limit each label to the
/// positions visible in the current sweep window.
///
/// `stop_after_first` is an index-selection optimization. It is appropriate
/// for `any`/`first`/`last` selection, but aggregate callers must leave it
/// disabled so every residual-passing candidate can update aggregate state.
///
/// # Arguments
///
/// * `regions` - Prepared dual paths.
/// * `predicates_pass` - Residual callback receiving search-order positions.
/// * `stop_after_first` - Stop each left row after its first residual match
///   when selecting one result; must be false for `keep="all"`.
///
/// # Returns
///
/// Returns per-left counts, total matches, optional selected positions, and
/// the reusable ranked right layout.
///
/// # Errors
///
/// Returns an error for invalid sweep order or count overflow.
fn collect_dual_residual_candidates<F>(
    regions: &DualRegionPaths,
    predicates_pass: F,
    stop_after_first: bool,
) -> Result<DualResidualCandidatePass, String>
where
    F: Fn(usize, usize) -> bool,
{
    let (left_has_match, any_left_match) = dual_region_match_filter(regions)?;
    if !any_left_match {
        return Ok(DualResidualCandidatePass {
            left_match_counts: vec![0; regions.left_region.len()],
            total_matches: 0,
            selected_right_positions: Vec::new(),
            right_region_layout: RankedRegionPositionLayout {
                labels: Vec::new(),
                offsets: Vec::new(),
                ranks: Vec::new(),
                positions: Vec::new(),
            },
        });
    }

    let right_region_layout =
        build_ranked_region_position_layout(&regions.right_region, &regions.starts, regions.end)?;
    let mut match_counts = vec![0usize; regions.left_region.len()];
    let mut selected_right_positions = if stop_after_first {
        vec![None; regions.left_region.len()]
    } else {
        Vec::new()
    };
    let mut total_matches = 0usize;
    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut previous_end = regions.end;

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        activate_dual_ranked_positions(
            &right_region_layout,
            start,
            previous_end,
            &mut active_counts,
        )?;
        previous_end = start;

        if !left_has_match[left_position] || left_region == 0 {
            continue;
        }

        let first_rank = right_region_layout
            .labels
            .partition_point(|&label| label < left_region);
        'candidates: for (&rank, &region_count) in active_counts.range(first_rank..) {
            let region_start = right_region_layout.offsets[rank];
            for &right_position in
                &right_region_layout.positions[region_start..region_start + region_count]
            {
                if predicates_pass(left_position, right_position) {
                    match_counts[left_position] = match_counts[left_position]
                        .checked_add(1)
                        .ok_or("region result size exceeds platform capacity")?;
                    if stop_after_first {
                        selected_right_positions[left_position] = Some(right_position);
                    }
                    total_matches = total_matches
                        .checked_add(1)
                        .ok_or("region result size exceeds platform capacity")?;
                    if stop_after_first {
                        break 'candidates;
                    }
                }
            }
        }
    }

    Ok(DualResidualCandidatePass {
        left_match_counts: match_counts,
        total_matches,
        selected_right_positions,
        right_region_layout,
    })
}

/// Count every residual-passing candidate for `keep="all"`.
///
/// The returned counts are exact and are used to size the second materializing
/// pass. This function never applies a one-result keep optimization.
fn count_dual_residual_matches<F>(
    regions: &DualRegionPaths,
    predicates_pass: F,
) -> Result<DualResidualCandidatePass, String>
where
    F: Fn(usize, usize) -> bool,
{
    collect_dual_residual_candidates(regions, predicates_pass, false)
}

/// Count whether each left row has at least one residual-passing candidate.
///
/// The result is intentionally per-left rather than a total-only count because
/// first/last selection must retain one output slot for each matching left row.
fn count_dual_residual_left_matches<F>(
    regions: &DualRegionPaths,
    predicates_pass: F,
) -> Result<DualResidualCandidatePass, String>
where
    F: Fn(usize, usize) -> bool,
{
    collect_dual_residual_candidates(regions, predicates_pass, true)
}

/// Materialize dual-region candidates after residual predicates pass.
///
/// This is a two-pass operation: the first pass counts exact output size and
/// the second pass fills preallocated left/right index buffers.
fn materialize_dual_residual_indices<F>(
    regions: &DualRegionPaths,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    predicates_pass: F,
) -> Result<Option<MaterializedIndices>, String>
where
    F: Fn(usize, usize) -> bool,
{
    let first_pass = count_dual_residual_matches(regions, &predicates_pass)?;

    if first_pass.total_matches == 0 {
        return Ok(None);
    }

    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut previous_end = regions.end;
    let mut left_output = vec![0_i64; first_pass.total_matches];
    let mut right_output = vec![0_i64; first_pass.total_matches];
    let mut output_position = 0usize;

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        activate_dual_ranked_positions(
            &first_pass.right_region_layout,
            start,
            previous_end,
            &mut active_counts,
        )?;
        previous_end = start;

        if left_region == 0 || first_pass.left_match_counts[left_position] == 0 {
            continue;
        }

        let first_rank = first_pass
            .right_region_layout
            .labels
            .partition_point(|&label| label < left_region);
        for (&rank, &region_count) in active_counts.range(first_rank..) {
            let region_start = first_pass.right_region_layout.offsets[rank];
            for &right_position in
                &first_pass.right_region_layout.positions[region_start..region_start + region_count]
            {
                if predicates_pass(left_position, right_position) {
                    left_output[output_position] = left_index[regions.left_order[left_position]];
                    right_output[output_position] = right_index[right_position];
                    output_position += 1;
                }
            }
        }
    }

    if output_position != first_pass.total_matches {
        return Err("dual residual second pass count disagrees with first pass".to_owned());
    }

    Ok(Some(MaterializedIndices {
        left_index: left_output,
        right_index: right_output,
    }))
}

/// Materialize one selected matching pair per left row after residuals pass.
///
/// `any` stops at the first passing candidate. `first` and `last` inspect all
/// passing candidates for the left row and compare physical right labels.
fn materialize_dual_residual_selected_indices<F>(
    regions: &DualRegionPaths,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    predicates_pass: F,
    keep: Keep,
) -> Result<Option<MaterializedIndices>, String>
where
    F: Fn(usize, usize) -> bool,
{
    let first_pass = count_dual_residual_left_matches(regions, &predicates_pass)?;

    if first_pass.total_matches == 0 {
        return Ok(None);
    }

    if keep == Keep::Any {
        let mut left_output = Vec::with_capacity(first_pass.total_matches);
        let mut right_output = Vec::with_capacity(first_pass.total_matches);
        for (left_position, right_position) in
            first_pass.selected_right_positions.iter().enumerate()
        {
            let Some(right_position) = right_position else {
                continue;
            };
            left_output.push(left_index[regions.left_order[left_position]]);
            right_output.push(right_index[*right_position]);
        }
        return Ok(Some(MaterializedIndices {
            left_index: left_output,
            right_index: right_output,
        }));
    }

    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut previous_end = regions.end;
    let mut left_output = vec![0_i64; first_pass.total_matches];
    let mut right_output = vec![0_i64; first_pass.total_matches];
    let mut output_position = 0usize;

    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }
        activate_dual_ranked_positions(
            &first_pass.right_region_layout,
            start,
            previous_end,
            &mut active_counts,
        )?;
        previous_end = start;

        if left_region == 0 || first_pass.left_match_counts[left_position] == 0 {
            continue;
        }

        let mut selected: Option<(i64, usize)> = None;
        let first_rank = first_pass
            .right_region_layout
            .labels
            .partition_point(|&label| label < left_region);
        'candidates: for (&rank, &region_count) in active_counts.range(first_rank..) {
            let region_start = first_pass.right_region_layout.offsets[rank];
            for &right_position in
                &first_pass.right_region_layout.positions[region_start..region_start + region_count]
            {
                if !predicates_pass(left_position, right_position) {
                    continue;
                }

                let right_value = right_index[right_position];
                match keep {
                    Keep::Any => {
                        selected = Some((right_value, right_position));
                        break 'candidates;
                    }
                    Keep::First => {
                        if selected.is_none_or(|(value, _)| right_value < value) {
                            selected = Some((right_value, right_position));
                        }
                    }
                    Keep::Last => {
                        if selected.is_none_or(|(value, _)| right_value > value) {
                            selected = Some((right_value, right_position));
                        }
                    }
                    Keep::All => unreachable!("selected materialization excludes keep='all'"),
                }
            }
        }

        if let Some((_, right_position)) = selected {
            left_output[output_position] = left_index[regions.left_order[left_position]];
            right_output[output_position] = right_index[right_position];
            output_position += 1;
        }
    }

    if output_position != first_pass.total_matches {
        return Err("dual residual selection count disagrees with first pass".to_owned());
    }

    Ok(Some(MaterializedIndices {
        left_index: left_output,
        right_index: right_output,
    }))
}

/// Build the boundary sequence for one typed range predicate.
///
/// ELI5: each left value asks where its matching right-side window begins.
/// Successful boundaries mark events in the right-side layout. A cumulative
/// sum turns those events into right-region labels, and each successful left
/// row reads its label at its boundary.
///
/// # Arguments
///
/// * `left` - Left predicate values in logical left-row order.
/// * `right` - Sorted right predicate values in that predicate's search order.
/// * `op` - Strict or inclusive inequality operator.
/// * `alignment` - Optional permutation used to gather the right labels into
///   global right order.
/// * `left_region_filter` - Optional zero mask from an earlier predicate; a
///   zero skips the binary search and remains unsuccessful.
///
/// # Returns
///
/// Returns `(left_region, right_region)` or `None` when no left row produces a
/// successful window. Region zero is reserved for unsuccessful rows.
///
/// # Errors
///
/// Returns an error when the filter length or alignment length is invalid, or
/// when an alignment is not a safe permutation.
fn build_region_paths_typed<'py, T>(
    left: ArrayView1<'py, T>,
    right: ArrayView1<'py, T>,
    op: CompareOp,
    alignment: Option<&ArrayView1<'_, i64>>,
    left_region_filter: Option<&[i64]>,
) -> Result<Option<RegionPaths>, String>
where
    T: numpy::Element + PartialOrd + Copy,
{
    let (left_region, right_region) = {
        if let Some(left_region_filter) = left_region_filter {
            ensure_equal_lengths_core(
                "left",
                left.len(),
                "left_region_filter",
                left_region_filter.len(),
            )?;
        }

        let mut boundaries = Vec::with_capacity(left.len());
        let mut successful = Vec::with_capacity(left.len());
        let mut successful_count = 0;

        for (position, &left_value) in left.iter().enumerate() {
            // ELI5: if the first predicate already found no right match for
            // this left row, the second predicate cannot rescue it. Leave a
            // zero marker and avoid doing another binary search.
            if left_region_filter.is_some_and(|filter| filter[position] == 0) {
                boundaries.push(right.len());
                successful.push(false);
                continue;
            }

            let boundary = range_boundary(left_value, right, op);
            let valid = boundary < right.len();
            boundaries.push(boundary);
            successful.push(valid);
            if valid {
                successful_count += 1;
            }
        }

        if successful_count == 0 {
            return Ok(None);
        }

        // Each successful left window contributes one event at its right-side
        // boundary. The cumulative sum labels every right row by the set of
        // successful left windows whose boundary has been reached.
        let mut right_region = vec![0_i64; right.len()];
        for (boundary, &valid) in boundaries.iter().zip(&successful) {
            if valid && *boundary < right_region.len() {
                right_region[*boundary] += 1;
            }
        }
        for position in 1..right_region.len() {
            right_region[position] += right_region[position - 1];
        }

        // Zero is reserved for an unsuccessful left window. Successful region
        // labels begin at one because every successful boundary contributes an
        // event before the cumulative sum reaches that boundary.
        let mut left_region = vec![0_i64; left.len()];
        for (position, (boundary, &valid)) in boundaries.iter().zip(&successful).enumerate() {
            if valid && *boundary < right_region.len() {
                left_region[position] = right_region[*boundary];
            }
        }

        let right_region = realign_right_region(right_region, alignment)?;
        (left_region, right_region)
    };

    Ok(Some((left_region, right_region)))
}

/// Dispatch one parsed Python predicate to its concrete NumPy dtype.
///
/// `AnyRegionPredicate` keeps the dtype-specific value arrays behind an
/// enum. The actual boundary builder stays generic, so this is the only place
/// that needs to enumerate the supported dtypes.
///
/// # Arguments
///
/// * `predicate` - Parsed typed predicate containing one supported NumPy dtype.
/// * `alignment` - Optional global-right gather permutation.
/// * `left_region_filter` - Optional zero mask from an earlier left path.
/// * `first_right_region` - Optional first-path zero mask for the right path.
///
/// # Returns
///
/// Returns region paths in the requested global layout, or `None` when no
/// successful left row remains.
///
/// # Errors
///
/// Returns an error for equality/non-equality operators, invalid lengths, or
/// invalid alignment positions.
pub(crate) fn build_region_paths<'a>(
    predicate: &'a Predicate<'a>,
    alignment: Option<&ArrayView1<'_, i64>>,
    left_region_filter: Option<&[i64]>,
    first_right_region: Option<&[i64]>,
) -> Result<Option<RegionPaths>, String> {
    let view = predicate.view();
    let op = match &view {
        PredicateView::I64(_, _, op)
        | PredicateView::I32(_, _, op)
        | PredicateView::I16(_, _, op)
        | PredicateView::I8(_, _, op)
        | PredicateView::U64(_, _, op)
        | PredicateView::U32(_, _, op)
        | PredicateView::U16(_, _, op)
        | PredicateView::U8(_, _, op)
        | PredicateView::F64(_, _, op)
        | PredicateView::F32(_, _, op) => *op,
    };
    if matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return Err("region predicates must use <, <=, >, or >=".to_owned());
    }

    let paths = match view {
        PredicateView::I64(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::I32(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::I16(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::I8(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::U64(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::U32(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::U16(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::U8(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::F64(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
        PredicateView::F32(left, right, op) => {
            build_region_paths_typed(left, right, op, alignment, left_region_filter)
        }
    }?;

    let Some((left_region, mut right_region)) = paths else {
        return Ok(None);
    };
    mask_right_region(&mut right_region, first_right_region)?;
    Ok(Some((left_region, right_region)))
}

/// Build both dual-region paths and derive the shared search starts.
///
/// The first predicate establishes the global right order. The second path is
/// realigned into that order and masked by the first path before both left
/// paths are reordered into descending search order.
///
/// # Arguments
///
/// * `first` - Global-order primary inequality predicate.
/// * `second` - Secondary inequality predicate.
/// * `left_index` / `right_index` - Physical maps whose lengths must match
///   every predicate array.
/// * `alignment` - Global-to-secondary-right permutation.
///
/// # Returns
///
/// Returns prepared dual paths, or `None` when no row can satisfy both range
/// predicates.
///
/// # Errors
///
/// Returns an error for length mismatches, invalid alignments, unsupported
/// operators, or invalid monotonic/start invariants.
pub(crate) fn build_dual_region_paths<'py>(
    first: Predicate<'py>,
    second: Predicate<'py>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    alignment: &ArrayView1<'_, i64>,
) -> Result<Option<DualRegionPaths>, String> {
    let first_left_len = first.left_len();
    let first_right_len = first.right_len();
    let second_left_len = second.left_len();
    let second_right_len = second.right_len();
    ensure_equal_lengths_core("first left", first_left_len, "left_index", left_index.len())?;
    ensure_equal_lengths_core(
        "first right",
        first_right_len,
        "right_index",
        right_index.len(),
    )?;
    ensure_equal_lengths_core(
        "second left",
        second_left_len,
        "left_index",
        left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "second right",
        second_right_len,
        "right_index",
        right_index.len(),
    )?;

    let Some(first) = build_region_paths(&first, None, None, None)? else {
        return Ok(None);
    };
    let (first_left_region, first_right_region) = first;

    let Some(second) = build_region_paths(
        &second,
        Some(alignment),
        Some(&first_left_region),
        Some(&first_right_region),
    )?
    else {
        return Ok(None);
    };
    let (second_left_region, second_right_region) = second;

    let left_order = descending_left_order(&first_left_region);
    let first_left_region = reorder_left_region(&first_left_region, &left_order);
    let second_left_region = reorder_left_region(&second_left_region, &left_order);
    let Some(starts) =
        first_region_starts(&first_left_region, &second_left_region, &first_right_region)?
    else {
        return Ok(None);
    };
    let end = first_right_region.len();
    Ok(Some(DualRegionPaths {
        left_region: second_left_region,
        right_region: second_right_region,
        starts,
        end,
        left_order,
    }))
}

/// Build index pairs from two three-element inequality predicates.
///
/// Each predicate is `(left_values, right_values, operator)`. The predicate
/// arrays describe comparisons; `left_index` and `right_index` contain the
/// physical labels returned to Python. The first predicate defines the global
/// right-side order. `aligner` maps positions in the second predicate's sorted
/// right-side array into that global order.
///
/// ELI5: the first predicate draws the search window, and the second predicate
/// removes the right-side rows that do not belong to that window. The aligner
/// tells us which second-predicate row corresponds to each global right row.
///
/// `keep` controls index selection only. It has no meaning for aggregation;
/// aggregation entry points should visit every candidate that passes all
/// predicates rather than selecting one candidate per left row.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to build the result dictionary.
/// * `left_index` - One physical output label for each left predicate row.
/// * `right_index` - Physical output labels in the first predicate's global
///   right-side order.
/// * `first_predicate` - Primary three-item range predicate.
/// * `second_predicate` - Secondary three-item range predicate.
/// * `aligner` - Permutation mapping each global right position to the
///   corresponding position in `second_predicate`'s right values.
/// * `keep` - Index selection policy: `"all"`, `"any"`, `"first"`, or
///   `"last"`.
///
/// # Returns
///
/// `None` when no pair satisfies both predicates. Otherwise returns a
/// dictionary containing equal-length `left_index` and `right_index` arrays.
///
/// # Errors
///
/// Returns `ValueError` for invalid operators, lengths, alignments, keep
/// values, or internal pass-count disagreements. Returns `TypeError` when a
/// predicate, index, or aligner has an incompatible Python type or shape.
#[pyfunction]
pub fn dual_regions_indices<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    second_predicate: &Bound<'py, PyTuple>,
    aligner: PyReadonlyArray1<'py, i64>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;

    let first = parse_region_predicate(first_predicate)?;
    let second = parse_region_predicate(second_predicate)?;
    let left_index = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let aligner = aligner.as_array();

    let left_index_view = left_index.as_array();
    let right_index_view = right_index.as_array();
    let Some(regions) =
        build_dual_region_paths(first, second, left_index_view, right_index_view, &aligner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?
    else {
        return Ok(None);
    };

    let indices = match keep {
        Keep::All => {
            let first_pass =
                keep_all_first_pass(&regions).map_err(pyo3::exceptions::PyValueError::new_err)?;
            keep_all_second_pass(&regions, &first_pass, left_index_view, right_index_view)
        }
        Keep::Any => {
            let first_pass =
                count_any_matches(&regions).map_err(pyo3::exceptions::PyValueError::new_err)?;
            keep_any_second_pass(&regions, &first_pass, left_index_view, right_index_view)
        }
        Keep::First | Keep::Last => {
            let first_pass = count_matching_left_rows(&regions)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            keep_first_last_second_pass(
                &regions,
                &first_pass,
                left_index_view,
                right_index_view,
                keep,
            )
        }
    }
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let Some(indices) = indices else {
        return Ok(None);
    };

    Ok(Some(result_dict(
        py,
        indices.left_index,
        indices.right_index,
        None,
        None,
    )?))
}

/// Build index pairs from two primary region predicates and residual filters.
///
/// The two range predicates generate candidate positions first. Every
/// residual predicate is then evaluated against each candidate before the
/// `keep` policy is applied. Residual predicates therefore cannot widen the
/// range candidate set.
///
/// ELI5: the range regions are a coarse sieve and the residual predicates are
/// the final inspection. A row is emitted only after it passes both stages.
/// `keep` selects index pairs after that complete predicate check; it is not an
/// aggregation option.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to build the result dictionary.
/// * `left_index` - Physical output labels for the left predicate rows.
/// * `right_index` - Physical output labels in the first predicate's global
///   right-side order.
/// * `first_predicate` - Primary range predicate that defines the global
///   right-side search layout.
/// * `second_predicate` - Secondary range predicate used to narrow candidates.
/// * `aligner` - Permutation from the secondary predicate's right layout to
///   the first predicate's global right layout.
/// * `residual_predicates` - Additional Python predicate tuples evaluated only
///   for candidates that pass both range predicates.
/// * `keep` - Index selection policy applied after residual filtering.
///
/// # Returns
///
/// `None` when no candidate passes all predicates. Otherwise returns equal-
/// length `left_index` and `right_index` result arrays.
///
/// # Errors
///
/// Returns `ValueError` for invalid range operators, mismatched lengths,
/// invalid alignments, unsupported keep values, or internal pass-count
/// failures. Returns `TypeError` for malformed Python arrays or predicates.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn dual_regions_and_residual_indices<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    second_predicate: &Bound<'py, PyTuple>,
    aligner: PyReadonlyArray1<'py, i64>,
    residual_predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    let first = parse_region_predicate(first_predicate)?;
    let second = parse_region_predicate(second_predicate)?;
    let left_index = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let aligner = aligner.as_array();
    let left_index_view = left_index.as_array();
    let right_index_view = right_index.as_array();

    let Some(regions) =
        build_dual_region_paths(first, second, left_index_view, right_index_view, &aligner)
            .map_err(pyo3::exceptions::PyValueError::new_err)?
    else {
        return Ok(None);
    };

    let (residuals, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&residuals, left_index_view.len(), right_index_view.len())?;
    let views = residuals.iter().map(Predicate::view).collect::<Vec<_>>();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let predicates_pass = |left_position, right_position| {
        predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            regions.left_order[left_position],
            right_position,
        )
    };
    let indices = match keep {
        Keep::All => materialize_dual_residual_indices(
            &regions,
            left_index_view,
            right_index_view,
            predicates_pass,
        ),
        Keep::Any | Keep::First | Keep::Last => materialize_dual_residual_selected_indices(
            &regions,
            left_index_view,
            right_index_view,
            predicates_pass,
            keep,
        ),
    }
    .map_err(pyo3::exceptions::PyValueError::new_err)?;

    let Some(indices) = indices else {
        return Ok(None);
    };

    Ok(Some(result_dict(
        py,
        indices.left_index,
        indices.right_index,
        None,
        None,
    )?))
}

/// All immutable data needed by the n-region candidate sweep.
///
/// ELI5: this is the prepared map for a treasure hunt. The region vectors say
/// which rows can match, `starts` says where each search window begins, and
/// the ranked layout lets the sweep find the second region without rebuilding
/// a map for every left row.
struct NRegionCandidatePaths {
    left_regions: Vec<Vec<i64>>,
    right_regions: Vec<Vec<i64>>,
    starts: Vec<usize>,
    end: usize,
    left_order: Vec<usize>,
    right_region_layout: RankedRegionPositionLayout,
}

/// Build the shared candidate state for three or more range predicates.
///
/// This helper deliberately stops before `keep` selection or aggregation.
/// Both consumers must see the same masked, reordered, and aligned candidates
/// or their results could disagree on which pairs are valid.
///
/// # Arguments
///
/// * `first_predicate` - Global first range anchor.
/// * `range_predicates` - At least two additional anchors.
/// * `left_index` / `right_index` - Shared physical maps.
/// * `aligners` - One global-to-source-right permutation per additional
///   predicate.
///
/// # Returns
///
/// Returns reusable n-region paths or `None` when an earlier stage proves that
/// no row can match.
///
/// # Errors
///
/// Returns an error for mismatched predicate/index/aligner lengths, invalid
/// alignments, unsupported operators, or invalid search invariants.
fn build_n_region_candidate_paths(
    first_predicate: &Predicate<'_>,
    range_predicates: &[Predicate<'_>],
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    aligners: &[ArrayView1<'_, i64>],
) -> Result<Option<NRegionCandidatePaths>, String> {
    ensure_equal_lengths_core(
        "first left predicate",
        first_predicate.left_len(),
        "left_index",
        left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "first right predicate",
        first_predicate.right_len(),
        "right_index",
        right_index.len(),
    )?;
    if range_predicates.len() < 2 {
        return Err("n-region paths require at least three predicates".to_owned());
    }
    if range_predicates.len() != aligners.len() {
        return Err("range predicates and aligners must have equal lengths".to_owned());
    }
    for predicate in range_predicates {
        ensure_equal_lengths_core(
            "range left predicate",
            predicate.left_len(),
            "left_index",
            left_index.len(),
        )?;
        ensure_equal_lengths_core(
            "range right predicate",
            predicate.right_len(),
            "right_index",
            right_index.len(),
        )?;
    }

    let Some(first_regions) = build_region_paths(first_predicate, None, None, None)? else {
        return Ok(None);
    };
    let (first_left_region, first_right_region) = first_regions;
    let mut left_regions = Vec::with_capacity(range_predicates.len() + 1);
    let mut right_regions = Vec::with_capacity(range_predicates.len() + 1);
    left_regions.push(first_left_region);
    right_regions.push(first_right_region);

    for (predicate, aligner) in range_predicates.iter().zip(aligners) {
        ensure_equal_lengths_core("aligner", aligner.len(), "right_index", right_index.len())?;
        let Some((left_region, right_region)) =
            build_region_paths(predicate, Some(aligner), None, None)?
        else {
            return Ok(None);
        };
        left_regions.push(left_region);
        right_regions.push(right_region);
    }

    mask_zero_regions(&mut left_regions)?;
    if left_regions[0].iter().all(|&region| region == 0) {
        return Ok(None);
    }

    let left_order = descending_left_order(&left_regions[0]);
    for region in &mut left_regions {
        *region = reorder_left_region(region, &left_order);
    }

    // Starts are derived from the first predicate before masking its right
    // region. That preserves the monotonic input required by partition_point.
    let Some(starts) = n_region_starts(&left_regions[0], &right_regions[0])? else {
        return Ok(None);
    };
    let end = right_regions[0].len();
    mask_zero_regions(&mut right_regions)?;
    if right_regions[0].iter().all(|&region| region == 0) {
        return Ok(None);
    }

    let right_region_layout = build_ranked_region_position_layout(&right_regions[1], &starts, end)?;
    Ok(Some(NRegionCandidatePaths {
        left_regions,
        right_regions,
        starts,
        end,
        left_order,
        right_region_layout,
    }))
}

/// Build index pairs for three or more range predicates after Python-side
/// parsing and validation.
///
/// `range_predicates` and `aligners` are parallel slices. Keeping them
/// separate from the first predicate makes the pairing explicit and prevents
/// an aligner from accidentally being applied to the wrong right-value array.
/// This helper assumes that Python has already parsed the tuples and checked
/// their operators; it performs the shared length and region-path validation
/// needed by the Rust implementation.
///
/// ELI5: this is the no-residual wrapper around the common n-region engine.
/// The engine finds every range candidate before applying the requested index
/// `keep` policy.
fn n_region_indices_core(
    first_predicate: &Predicate<'_>,
    range_predicates: &[Predicate<'_>],
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    aligners: &[ArrayView1<'_, i64>],
    keep: Keep,
) -> Result<Option<MaterializedIndices>, String> {
    n_region_indices_with_filter_core(
        first_predicate,
        range_predicates,
        left_index,
        right_index,
        aligners,
        keep,
        |_, _| true,
    )
}

/// Build n-region index pairs while applying an additional candidate filter.
///
/// The filter receives the original left position and the global right
/// position. Range regions are still checked first, so residual predicates
/// only run for candidates that survived every range predicate.
///
/// ELI5: the range sweep narrows the haystack first; `candidate_pass` checks
/// each remaining needle. A candidate is emitted only when that filter passes.
///
/// # Arguments
///
/// The predicate, index, range, aligner, and `keep` arguments define the
/// prepared n-region search. `candidate_pass` receives original left and
/// global right positions and is evaluated only after all range predicates
/// pass.
///
/// # Returns
///
/// Returns materialized index pairs or `None` when no filtered candidate
/// remains.
///
/// # Errors
///
/// Returns an error for invalid lengths, aligners, starts, pass counts, or
/// result-size arithmetic.
fn n_region_indices_with_filter_core<F>(
    first_predicate: &Predicate<'_>,
    range_predicates: &[Predicate<'_>],
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    aligners: &[ArrayView1<'_, i64>],
    keep: Keep,
    candidate_pass: F,
) -> Result<Option<MaterializedIndices>, String>
where
    F: Fn(usize, usize) -> bool,
{
    let Some(paths) = build_n_region_candidate_paths(
        first_predicate,
        range_predicates,
        left_index,
        right_index,
        aligners,
    )?
    else {
        return Ok(None);
    };
    let NRegionCandidatePaths {
        left_regions: left_range_regions,
        right_regions: right_range_regions,
        starts,
        end,
        left_order,
        right_region_layout,
    } = paths;

    if keep != Keep::All {
        let mut selected_right_positions = vec![None; left_range_regions[0].len()];
        let _ = visit_n_region_matches(
            &left_range_regions,
            &right_range_regions,
            &starts,
            end,
            &right_region_layout,
            |left_position, right_position| {
                if !candidate_pass(left_order[left_position], right_position) {
                    return true;
                }
                match keep {
                    Keep::Any => {
                        selected_right_positions[left_position] = Some(right_position);
                        false
                    }
                    Keep::First => {
                        let selected = &mut selected_right_positions[left_position];
                        if selected.is_none_or(|position| {
                            right_index[right_position] < right_index[position]
                        }) {
                            *selected = Some(right_position);
                        }
                        true
                    }
                    Keep::Last => {
                        let selected = &mut selected_right_positions[left_position];
                        if selected.is_none_or(|position| {
                            right_index[right_position] > right_index[position]
                        }) {
                            *selected = Some(right_position);
                        }
                        true
                    }
                    Keep::All => unreachable!("keep='all' uses the two-pass path"),
                }
            },
        )?;

        let selected_count = selected_right_positions
            .iter()
            .filter(|position| position.is_some())
            .count();
        if selected_count == 0 {
            return Ok(None);
        }

        let mut left_output = Vec::with_capacity(selected_count);
        let mut right_output = Vec::with_capacity(selected_count);
        for (left_position, right_position) in selected_right_positions.into_iter().enumerate() {
            let Some(right_position) = right_position else {
                continue;
            };
            left_output.push(left_index[left_order[left_position]]);
            right_output.push(right_index[right_position]);
        }

        return Ok(Some(MaterializedIndices {
            left_index: left_output,
            right_index: right_output,
        }));
    }

    // The first pass counts every successful candidate per left row. These
    // counts size the exact index output and can be reused for count(*),
    // matched-row masks, and early no-match decisions by aggregate callers.
    let mut left_match_counts = vec![0usize; left_range_regions[0].len()];
    let _ = visit_n_region_matches(
        &left_range_regions,
        &right_range_regions,
        &starts,
        end,
        &right_region_layout,
        |left_position, right_position| {
            if !candidate_pass(left_order[left_position], right_position) {
                return true;
            }
            left_match_counts[left_position] += 1;
            true
        },
    )?;
    let total_matches = left_match_counts.iter().try_fold(0usize, |total, &count| {
        total
            .checked_add(count)
            .ok_or("region result size exceeds platform capacity")
    })?;
    if total_matches == 0 {
        return Ok(None);
    }

    let mut left_output = Vec::with_capacity(total_matches);
    let mut right_output = Vec::with_capacity(total_matches);
    let mut second_left_match_counts = vec![0usize; left_range_regions[0].len()];
    let _ = visit_n_region_matches(
        &left_range_regions,
        &right_range_regions,
        &starts,
        end,
        &right_region_layout,
        |left_position, right_position| {
            if !candidate_pass(left_order[left_position], right_position) {
                return true;
            }
            second_left_match_counts[left_position] += 1;
            left_output.push(left_index[left_order[left_position]]);
            right_output.push(right_index[right_position]);
            true
        },
    )?;
    let second_pass_matches = left_output.len();

    if second_left_match_counts != left_match_counts
        || second_pass_matches != total_matches
        || left_output.len() != total_matches
        || right_output.len() != total_matches
    {
        return Err("n-region second pass count disagrees with first pass".to_owned());
    }

    Ok(Some(MaterializedIndices {
        left_index: left_output,
        right_index: right_output,
    }))
}

/// Build index pairs from three or more range predicates.
///
/// Each predicate is a three-item Python tuple:
/// `(left_values, right_values, operator)`. The first predicate is the anchor
/// and its right values must already use the global order described by
/// `right_index`. Every item in `range_predicates` is paired by position with
/// one item in `aligners`; an aligner maps a global right position to the
/// corresponding position in that predicate's sorted right values.
///
/// The caller owns the preparation contract: values must be filtered for
/// nulls, each right range must be sorted in the direction required by its
/// operator, and each aligner must be a permutation of the global right
/// positions. Rust validates lengths and permutation safety, but does not sort
/// values or infer an alignment.
///
/// `keep="all"` uses a counting pass followed by an exact-size materialization
/// pass. `keep="any"`, `keep="first"`, and `keep="last"` select at most one
/// surviving right row per left row. Selection for `first` and `last` uses the
/// right-index values, not their temporary sorted positions. `keep` belongs to
/// index materialization; it must not be used to change an aggregate over all
/// matching candidates.
///
/// Performance note: this sweep is left-driven. It sorts and visits every
/// left row, and stores one left-region vector per predicate. When the join
/// direction is otherwise interchangeable, keeping the smaller input on the
/// left is generally the better heuristic. This is not a correctness
/// requirement, and broad joins may still be dominated by output size.
///
/// # Arguments
///
/// * `py` - Active Python interpreter used to create the result dictionary.
/// * `left_index` - Physical labels for the prepared left rows.
/// * `right_index` - Physical labels in the first predicate's global right
///   order.
/// * `first_predicate` - First, global-order range predicate.
/// * `range_predicates` - At least two additional range predicates.
/// * `aligners` - One right-position permutation for each additional
///   predicate. `aligners[i][global_position]` is the source position in the
///   corresponding predicate's right values.
/// * `keep` - One of `"all"`, `"any"`, `"first"`, or `"last"`.
///
/// # Returns
///
/// `None` when no left/right pair satisfies every predicate; otherwise a
/// dictionary containing equal-length `left_index` and `right_index` arrays.
///
/// # Errors
///
/// Raises `ValueError` for invalid operators, mismatched lengths, invalid
/// aligners, unsupported keep values, or an internal pass-count invariant
/// failure. Raises `TypeError` when a predicate or aligner has the wrong
/// Python shape or dtype.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn n_region_indices<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    if range_predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range_predicates must contain at least two predicates",
        ));
    }
    if aligners.len() != range_predicates.len() {
        return Err(PyValueError::new_err(
            "aligners and range_predicates must have equal lengths",
        ));
    }

    let first = parse_region_predicate(first_predicate)?;
    validate_region_predicate(&first)?;

    let mut predicates = Vec::with_capacity(range_predicates.len() + 1);
    predicates.push(first);
    for item in range_predicates.iter() {
        let predicate = item
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("range predicates must be tuples"))?;
        let predicate = parse_region_predicate(predicate)?;
        validate_region_predicate(&predicate)?;
        predicates.push(predicate);
    }

    let left_index = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let left_index_view = left_index.as_array();
    let right_index_view = right_index.as_array();
    check_predicate_lengths(&predicates, left_index_view.len(), right_index_view.len())?;
    let (first_predicate, range_predicates) = predicates
        .split_first()
        .expect("n_region_indices requires a first predicate");

    let mut aligner_arrays = Vec::with_capacity(aligners.len());
    for item in aligners.iter() {
        let aligner = item
            .extract::<PyReadonlyArray1<'py, i64>>()
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        ensure_equal_lengths_core(
            "aligner",
            aligner.as_array().len(),
            "right_index",
            right_index_view.len(),
        )
        .map_err(PyValueError::new_err)?;
        aligner_arrays.push(aligner);
    }
    let aligner_views = aligner_arrays
        .iter()
        .map(|aligner| aligner.as_array())
        .collect::<Vec<_>>();

    let indices = n_region_indices_core(
        first_predicate,
        range_predicates,
        left_index_view,
        right_index_view,
        &aligner_views,
        keep,
    )
    .map_err(PyValueError::new_err)?;

    let Some(indices) = indices else {
        return Ok(None);
    };
    Ok(Some(result_dict(
        py,
        indices.left_index,
        indices.right_index,
        None,
        None,
    )?))
}

/// Entry point for three or more range predicates followed by residual
/// predicates.
///
/// Residual predicates are evaluated only after a candidate has passed every
/// range predicate. `keep="all"` uses the same exact-size two-pass materializer
/// as [`n_region_indices`], while the other keep modes select from the
/// residual-filtered candidates. For aggregation, the candidate traversal
/// should instead update aggregate state for every residual-passing pair;
/// `keep` is not relevant to that operation.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to construct the result dictionary.
/// * `left_index` / `right_index` - Physical labels for the shared left and
///   global right layouts.
/// * `first_predicate` - First range predicate defining the global right
///   order.
/// * `range_predicates` - At least two additional range predicates.
/// * `aligners` - One global-to-predicate-right permutation per range
///   predicate, paired by position.
/// * `residual_predicates` - Additional predicates checked only after all
///   range predicates pass.
/// * `keep` - Index selection policy applied after residual filtering.
///
/// `keep` controls index selection and is not an aggregation selector.
///
/// # Returns
///
/// Returns physical index pairs or `None` when no candidate survives.
///
/// # Errors
///
/// Raises `ValueError` for invalid predicate/aligner counts, lengths,
/// operators, or keep values, and `TypeError` for malformed Python arrays.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn n_region_and_residual_indices<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    Keep::parse(keep)?;
    if range_predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range_predicates must contain at least two predicates",
        ));
    }
    if aligners.len() != range_predicates.len() {
        return Err(PyValueError::new_err(
            "aligners and range_predicates must have equal lengths",
        ));
    }

    let first = parse_region_predicate(first_predicate)?;
    validate_region_predicate(&first)?;

    let mut predicates = Vec::with_capacity(range_predicates.len() + 1);
    predicates.push(first);
    for item in range_predicates.iter() {
        let predicate = item
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("range predicates must be tuples"))?;
        let predicate = parse_region_predicate(predicate)?;
        validate_region_predicate(&predicate)?;
        predicates.push(predicate);
    }

    let left_index = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let left_index_view = left_index.as_array();
    let right_index_view = right_index.as_array();
    check_predicate_lengths(&predicates, left_index_view.len(), right_index_view.len())?;

    let aligner_arrays = aligners
        .iter()
        .map(|item| {
            item.extract::<PyReadonlyArray1<'py, i64>>()
                .map_err(|error| PyTypeError::new_err(error.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    for aligner in &aligner_arrays {
        ensure_equal_lengths_core(
            "aligner",
            aligner.as_array().len(),
            "right_index",
            right_index_view.len(),
        )
        .map_err(PyValueError::new_err)?;
    }

    let (residuals, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&residuals, left_index_view.len(), right_index_view.len())?;
    let views = residuals.iter().map(Predicate::view).collect::<Vec<_>>();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let (first_predicate, range_predicates) = predicates
        .split_first()
        .expect("n_region_and_residual_indices requires a first predicate");
    let aligner_views = aligner_arrays
        .iter()
        .map(|aligner| aligner.as_array())
        .collect::<Vec<_>>();
    let predicates_pass = |left_position, right_position| {
        predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            left_position,
            right_position,
        )
    };

    let indices = n_region_indices_with_filter_core(
        first_predicate,
        range_predicates,
        left_index_view,
        right_index_view,
        &aligner_views,
        Keep::parse(keep)?,
        predicates_pass,
    )
    .map_err(PyValueError::new_err)?;

    let Some(indices) = indices else {
        return Ok(None);
    };
    Ok(Some(result_dict(
        py,
        indices.left_index,
        indices.right_index,
        None,
        None,
    )?))
}

/// Run the dual-region sweep and update aggregate state for every match.
///
/// `reverse` changes only which side supplies aggregation values and which
/// side receives output slots. The candidate set is identical in both
/// directions. `candidate_pass` is the residual-predicate hook; the exact
/// path passes every candidate and the extended path evaluates its residuals.
///
/// # Arguments
///
/// The index, predicate, aligner, and aggregation arguments describe one
/// prepared dual join. `reverse` swaps source/output sides. `candidate_pass`
/// receives original left and global right positions and must return whether
/// residual predicates allow aggregation.
///
/// # Returns
///
/// Returns the standard aggregation tuple or `None` when no update occurs.
///
/// # Errors
///
/// Returns a Python error for invalid paths, lengths, alignments, aggregation
/// requests, or source arrays.
#[allow(clippy::too_many_arguments)]
fn aggregate_dual_region_core<'py, F>(
    py: Python<'py>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    first: Predicate<'py>,
    second: Predicate<'py>,
    aligner: ArrayView1<'_, i64>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
    allow_rank_tree: bool,
    candidate_pass: F,
) -> PyResult<Option<Bound<'py, PyTuple>>>
where
    F: Fn(usize, usize) -> bool,
{
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let Some(regions) = build_dual_region_paths(first, second, left_index, right_index, &aligner)
        .map_err(PyValueError::new_err)?
    else {
        return Ok(None);
    };
    let (output_positions, output_len, source_len) = if reverse {
        (Some(right_index), right_index.len(), left_index.len())
    } else {
        (Some(left_index), left_index.len(), right_index.len())
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    let (left_has_match, any_left_match) =
        dual_region_match_filter(&regions).map_err(PyValueError::new_err)?;
    if !any_left_match {
        return Ok(None);
    }
    let layout =
        build_ranked_region_position_layout(&regions.right_region, &regions.starts, regions.end)
            .map_err(PyValueError::new_err)?;

    // The rank tree summarizes the complete dual-region candidate set. It is
    // safe only for the forward exact path: reverse aggregation reads values
    // from the left side, and residual aggregation must inspect every pair.
    let mut estimated_width = 0_usize;
    let mut estimate_previous_end = regions.end;
    for &start in &regions.starts {
        if start <= estimate_previous_end && start < regions.end {
            estimated_width = estimated_width.saturating_add(regions.end.saturating_sub(start));
            estimate_previous_end = start;
        }
    }
    if allow_rank_tree
        && !reverse
        && set.supports_dual_rank_tree()
        && should_use_segment_tree(
            regions.left_region.len(),
            estimated_width,
            right_index.len(),
        )
    {
        set.aggregate_dual_rank_tree(
            ArrayView1::from(regions.left_region.as_slice()),
            &regions.left_order,
            ArrayView1::from(regions.right_region.as_slice()),
            ArrayView1::from(regions.starts.as_slice()),
            &left_has_match,
            &inputs,
        );
        if set.is_empty() {
            return Ok(None);
        }
        return Ok(Some(make_results_with_positions(
            py,
            set,
            output_positions,
            output_len,
            return_matched,
        )?));
    }

    let mut active_counts = BTreeMap::<usize, usize>::new();
    let mut previous_end = regions.end;
    for (left_position, (&start, &left_region)) in
        regions.starts.iter().zip(&regions.left_region).enumerate()
    {
        if start == regions.end {
            continue;
        }
        activate_dual_ranked_positions(&layout, start, previous_end, &mut active_counts)
            .map_err(PyValueError::new_err)?;
        previous_end = start;
        if !left_has_match[left_position] || left_region == 0 {
            continue;
        }
        let first_rank = layout.labels.partition_point(|&label| label < left_region);
        for (&rank, &region_count) in active_counts.range(first_rank..) {
            let region_start = layout.offsets[rank];
            for &right_position in &layout.positions[region_start..region_start + region_count] {
                if !candidate_pass(regions.left_order[left_position], right_position) {
                    continue;
                }
                if reverse {
                    set.update(regions.left_order[left_position], right_position);
                } else {
                    set.update(right_position, regions.left_order[left_position]);
                }
            }
        }
    }
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

/// Aggregate every n-region candidate without materializing index pairs.
///
/// The callback is evaluated for each complete range candidate. It must not
/// stop traversal: aggregation requires every passing pair, unlike `keep`-
/// based index selection.
///
/// # Arguments
///
/// The predicate, index, range, aligner, aggregation, and direction arguments
/// define the prepared n-region join. `candidate_pass` applies residual
/// predicates using original physical positions.
///
/// # Returns
///
/// Returns the standard aggregation tuple or `None` when no candidate passes.
///
/// # Errors
///
/// Returns a Python error for invalid path construction, lengths, aligners, or
/// aggregation inputs.
#[allow(clippy::too_many_arguments)]
fn aggregate_n_region_core<'py, F>(
    py: Python<'py>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    first: Predicate<'py>,
    ranges: &[Predicate<'py>],
    aligners: &[ArrayView1<'_, i64>],
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
    candidate_pass: F,
) -> PyResult<Option<Bound<'py, PyTuple>>>
where
    F: Fn(usize, usize) -> bool,
{
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let Some(paths) =
        build_n_region_candidate_paths(&first, ranges, left_index, right_index, aligners)
            .map_err(PyValueError::new_err)?
    else {
        return Ok(None);
    };
    let (output_positions, output_len, source_len) = if reverse {
        (Some(right_index), right_index.len(), left_index.len())
    } else {
        (Some(left_index), left_index.len(), right_index.len())
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;
    visit_n_region_matches(
        &paths.left_regions,
        &paths.right_regions,
        &paths.starts,
        paths.end,
        &paths.right_region_layout,
        |left_position, right_position| {
            if !candidate_pass(paths.left_order[left_position], right_position) {
                return true;
            }
            if reverse {
                set.update(paths.left_order[left_position], right_position);
            } else {
                set.update(right_position, paths.left_order[left_position]);
            }
            true
        },
    )
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

/// Aggregate every pair satisfying two range predicates in the forward
/// direction.
///
/// The right-side aggregation arrays are read at global right positions and
/// accumulated into one output slot per left row. Unlike index selection,
/// aggregation has no `keep` policy: every candidate satisfying both range
/// predicates contributes to the result.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to construct the result tuple.
/// * `left_index` - Physical labels for left rows and output positions.
/// * `right_index` - Physical labels in the first predicate's global right
///   order.
/// * `first_predicate` - Three-item `(left_values, right_values, operator)`
///   anchor defining the global right layout.
/// * `second_predicate` - Second three-item range predicate.
/// * `aligner` - Permutation mapping global right positions into the second
///   predicate's right-value layout.
/// * `aggregations` - Non-empty aggregation requests whose source arrays are
///   aligned to `right_index`.
/// * `return_matched` - Whether to include a boolean matched array.
///
/// # Returns
///
/// Returns `(left_positions, matched?, results)` or `None` when no pair
/// satisfies both predicates.
///
/// # Errors
///
/// Raises an error for malformed predicates, invalid lengths or alignments,
/// unsupported operators, empty aggregation requests, or misaligned source
/// arrays.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn dual_regions_aggregate<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    second_predicate: &Bound<'py, PyTuple>,
    aligner: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    aggregate_dual_region_core(
        py,
        left.as_array(),
        right.as_array(),
        parse_region_predicate(first_predicate)?,
        parse_region_predicate(second_predicate)?,
        aligner.as_array(),
        aggregations,
        return_matched,
        false,
        true,
        |_, _| true,
    )
}

/// Aggregate every pair satisfying two range predicates in reverse.
///
/// This direction reads aggregation values from the left side and writes one
/// output slot per right row. The candidate traversal and predicate semantics
/// are identical to [`dual_regions_aggregate`]; only source/output ownership
/// changes.
///
/// # Arguments
///
/// The predicate, index, and aligner arguments have the same contracts as
/// [`dual_regions_aggregate`]. `aggregations` must instead contain arrays
/// aligned to `left_index`.
///
/// # Returns
///
/// Returns `(right_positions, matched?, results)` or `None` when no pair
/// satisfies both predicates.
///
/// # Errors
///
/// Raises an error for malformed predicates, invalid lengths or alignments,
/// unsupported operators, empty aggregation requests, or left-source arrays
/// with the wrong length.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn dual_regions_aggregate_reverse<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    second_predicate: &Bound<'py, PyTuple>,
    aligner: PyReadonlyArray1<'py, i64>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    aggregate_dual_region_core(
        py,
        left.as_array(),
        right.as_array(),
        parse_region_predicate(first_predicate)?,
        parse_region_predicate(second_predicate)?,
        aligner.as_array(),
        aggregations,
        return_matched,
        true,
        false,
        |_, _| true,
    )
}

/// Aggregate dual-region candidates after applying residual predicates.
///
/// The two range predicates form a coarse candidate set. Every residual
/// predicate is then evaluated against the original left/right positions
/// before the right-side source value updates the left output slot.
///
/// # Arguments
///
/// The range, index, aligner, aggregation, and `return_matched` arguments have
/// the same contracts as [`dual_regions_aggregate`]. `residual_predicates`
/// contains additional predicate tuples evaluated in order.
///
/// # Returns
///
/// Returns forward aggregation results or `None` when residual filtering
/// removes every range candidate.
///
/// # Errors
///
/// Raises an error for malformed residual predicates, length mismatches,
/// invalid range inputs, or invalid aggregation requests.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn dual_regions_and_residual_aggregate<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    second_predicate: &Bound<'py, PyTuple>,
    aligner: PyReadonlyArray1<'py, i64>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let (residuals, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&residuals, left.as_array().len(), right.as_array().len())?;
    let views = residuals.iter().map(Predicate::view).collect::<Vec<_>>();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_dual_region_core(
        py,
        left.as_array(),
        right.as_array(),
        parse_region_predicate(first_predicate)?,
        parse_region_predicate(second_predicate)?,
        aligner.as_array(),
        aggregations,
        return_matched,
        false,
        false,
        move |left_position, right_position| {
            predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            )
        },
    )
}

/// Aggregate dual-region candidates with residual predicates in reverse.
///
/// Residual predicates are evaluated on original physical positions. Passing
/// candidates update left-source values in right-side output slots.
///
/// # Arguments
///
/// The range and residual arguments have the same contracts as
/// [`dual_regions_and_residual_aggregate`]. Aggregation arrays must be aligned
/// to `left_index`.
///
/// # Returns
///
/// Returns reverse aggregation results or `None` when no complete candidate
/// survives the range and residual stages.
///
/// # Errors
///
/// Raises an error for malformed predicates, length mismatches, invalid
/// alignments, or invalid aggregation requests.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn dual_regions_and_residual_aggregate_reverse<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    second_predicate: &Bound<'py, PyTuple>,
    aligner: PyReadonlyArray1<'py, i64>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let (residuals, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&residuals, left.as_array().len(), right.as_array().len())?;
    let views = residuals.iter().map(Predicate::view).collect::<Vec<_>>();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_dual_region_core(
        py,
        left.as_array(),
        right.as_array(),
        parse_region_predicate(first_predicate)?,
        parse_region_predicate(second_predicate)?,
        aligner.as_array(),
        aggregations,
        return_matched,
        true,
        false,
        move |left_position, right_position| {
            predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            )
        },
    )
}

/// Aggregate every match for three or more range predicates.
///
/// The first predicate defines the global right order. Each item in
/// `range_predicates` is paired by position with an aligner, and all range
/// paths must pass before a source value is aggregated.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to construct the result tuple.
/// * `left_index` / `right_index` - Physical labels for the prepared sides.
/// * `first_predicate` - Global-order range predicate.
/// * `range_predicates` - At least two additional range predicates.
/// * `aligners` - One global-to-predicate-right permutation per range
///   predicate, in the same order as `range_predicates`.
/// * `aggregations` - Requests over right-side arrays aligned to `right_index`.
/// * `return_matched` - Whether to include per-left-row match flags.
///
/// # Returns
///
/// Returns forward aggregation results or `None` when no complete match exists.
///
/// # Errors
///
/// Raises an error for fewer than two additional predicates, mismatched list
/// lengths, invalid aligners, malformed predicates, or invalid aggregations.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn n_region_aggregate<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    n_region_aggregate_entry(
        py,
        left_index,
        right_index,
        first_predicate,
        range_predicates,
        aligners,
        aggregations,
        return_matched,
        false,
        None,
    )
}

/// Aggregate three or more range predicates in reverse.
///
/// This uses the same n-region candidate set as [`n_region_aggregate`], but
/// reads left-side source arrays and writes right-side output slots.
///
/// # Arguments
///
/// The predicate, index, range, and aligner arguments have the same contracts
/// as [`n_region_aggregate`]. Aggregation arrays must be aligned to
/// `left_index`.
///
/// # Returns
///
/// Returns reverse aggregation results or `None` when no complete match exists.
///
/// # Errors
///
/// Raises an error for invalid predicate/aligner counts, lengths, operators,
/// source arrays, or aggregation requests.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn n_region_aggregate_reverse<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    n_region_aggregate_entry(
        py,
        left_index,
        right_index,
        first_predicate,
        range_predicates,
        aligners,
        aggregations,
        return_matched,
        true,
        None,
    )
}

/// Aggregate n-region candidates after applying residual predicates.
///
/// Range predicates are checked by the n-region sweep first. Residual
/// predicates run only for candidates that pass every range path, and only
/// those candidates update the right-source/left-output aggregation state.
///
/// # Arguments
///
/// The range, index, aligner, and aggregation arguments have the same
/// contracts as [`n_region_aggregate`]. `residual_predicates` contains the
/// additional filters evaluated on original physical positions.
///
/// # Returns
///
/// Returns forward aggregation results or `None` if no candidate survives.
///
/// # Errors
///
/// Raises an error for malformed residuals, invalid lengths or aligners,
/// unsupported operators, or invalid aggregation requests.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn n_region_and_residual_aggregate<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    n_region_aggregate_entry(
        py,
        left_index,
        right_index,
        first_predicate,
        range_predicates,
        aligners,
        aggregations,
        return_matched,
        false,
        Some(residual_predicates),
    )
}

/// Aggregate n-region candidates with residual predicates in reverse.
///
/// This is the reverse counterpart of
/// [`n_region_and_residual_aggregate`]. Passing left-source values update
/// right-side output slots after all range and residual predicates pass.
///
/// # Arguments
///
/// The predicate, range, aligner, and residual arguments have the same
/// contracts as [`n_region_and_residual_aggregate`]. Aggregation arrays must
/// be aligned to `left_index`.
///
/// # Returns
///
/// Returns reverse aggregation results or `None` if no candidate survives.
///
/// # Errors
///
/// Raises an error for malformed predicates, residuals, aligners, lengths, or
/// aggregation requests.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn n_region_and_residual_aggregate_reverse<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    n_region_aggregate_entry(
        py,
        left_index,
        right_index,
        first_predicate,
        range_predicates,
        aligners,
        aggregations,
        return_matched,
        true,
        Some(residual_predicates),
    )
}

/// Parse the n-region Python lists and run the shared aggregation sweep.
#[allow(clippy::too_many_arguments)]
fn n_region_aggregate_entry<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    first_predicate: &Bound<'py, PyTuple>,
    range_predicates: &Bound<'py, PyList>,
    aligners: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
    residual_predicates: Option<&Bound<'py, PyList>>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if range_predicates.len() < 2 || range_predicates.len() != aligners.len() {
        return Err(PyValueError::new_err(
            "range predicates and aligners must contain at least two equal-length items",
        ));
    }
    let first = parse_region_predicate(first_predicate)?;
    validate_region_predicate(&first)?;
    let mut ranges = Vec::with_capacity(range_predicates.len());
    for item in range_predicates.iter() {
        let predicate = item
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("range predicates must be tuples"))?;
        let predicate = parse_region_predicate(predicate)?;
        validate_region_predicate(&predicate)?;
        ranges.push(predicate);
    }
    let left = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let mut aligner_arrays = Vec::with_capacity(aligners.len());
    for item in aligners.iter() {
        let aligner = item
            .extract::<PyReadonlyArray1<'py, i64>>()
            .map_err(|error| PyTypeError::new_err(error.to_string()))?;
        ensure_equal_lengths_core(
            "aligner",
            aligner.as_array().len(),
            "right_index",
            right.as_array().len(),
        )
        .map_err(PyValueError::new_err)?;
        aligner_arrays.push(aligner);
    }
    let aligner_views = aligner_arrays
        .iter()
        .map(|aligner| aligner.as_array())
        .collect::<Vec<_>>();
    let residual_storage = if let Some(residual_predicates) = residual_predicates {
        let parsed = parse_predicates_with_nulls_strings(py, residual_predicates)?;
        check_predicate_lengths(&parsed.0, left.as_array().len(), right.as_array().len())?;
        Some(parsed)
    } else {
        None
    };
    let residual_views = residual_storage.as_ref().map(|(residuals, metadata)| {
        (
            residuals.iter().map(Predicate::view).collect::<Vec<_>>(),
            metadata.as_deref().map(null_metadata_views),
        )
    });
    aggregate_n_region_core(
        py,
        left.as_array(),
        right.as_array(),
        first,
        &ranges,
        &aligner_views,
        aggregations,
        return_matched,
        reverse,
        move |left_position, right_position| match &residual_views {
            Some((views, metadata)) => {
                predicates_match_dispatch(views, metadata.as_deref(), left_position, right_position)
            }
            None => true,
        },
    )
}

/// Register the standalone dual-region entry point.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dual_regions_indices, m)?)?;
    m.add_function(wrap_pyfunction!(dual_regions_and_residual_indices, m)?)?;
    m.add_function(wrap_pyfunction!(n_region_indices, m)?)?;
    m.add_function(wrap_pyfunction!(n_region_and_residual_indices, m)?)?;
    m.add_function(wrap_pyfunction!(dual_regions_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(dual_regions_aggregate_reverse, m)?)?;
    m.add_function(wrap_pyfunction!(dual_regions_and_residual_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(
        dual_regions_and_residual_aggregate_reverse,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(n_region_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(n_region_aggregate_reverse, m)?)?;
    m.add_function(wrap_pyfunction!(n_region_and_residual_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(
        n_region_and_residual_aggregate_reverse,
        m
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyDict, PyList};

    fn predicate<'py>(
        py: Python<'py>,
        left: &[i64],
        right: &[i64],
        operator: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, left.to_vec()).into_any(),
                PyArray1::from_vec(py, right.to_vec()).into_any(),
                operator.into_pyobject(py)?.into_any(),
            ],
        )
    }

    fn pairs(result: &Bound<'_, PyDict>) -> PyResult<Vec<(i64, i64)>> {
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

    fn aggregation_input<'py>(
        py: Python<'py>,
        values: &[i64],
        operation: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, values.to_vec()).into_any(),
                PyArray1::from_vec(py, vec![false; values.len()]).into_any(),
                operation.into_pyobject(py)?.into_any(),
            ],
        )
    }

    fn aggregate_output<'py>(result: &Bound<'py, PyTuple>) -> PyResult<Vec<i64>> {
        result
            .get_item(1)?
            .cast::<PyList>()?
            .get_item(0)?
            .extract::<Vec<i64>>()
    }

    fn aggregate_outputs<'py>(result: &Bound<'py, PyTuple>) -> PyResult<Vec<Vec<i64>>> {
        result
            .get_item(1)?
            .cast::<PyList>()?
            .iter()
            .map(|output| output.extract::<Vec<i64>>())
            .collect()
    }

    fn matches(left: i64, right: i64, operator: &str) -> bool {
        match operator {
            "<" => left < right,
            "<=" => left <= right,
            ">" => left > right,
            ">=" => left >= right,
            _ => panic!("unsupported test operator: {operator}"),
        }
    }

    #[test]
    fn region_aggregation_updates_all_dual_and_n_region_matches() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let first = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let second = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let aligner = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly();
            let aggregations = PyList::new(py, [aggregation_input(py, &[1, 2, 3, 4], "sum")?])?;

            let dual = dual_regions_aggregate(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                aligner.clone(),
                &aggregations,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("dual aggregation returned None"))?;
            assert_eq!(aggregate_output(&dual)?, vec![10, 7]);

            let ranges = PyList::new(py, [second.clone(), second.clone()])?;
            let aligners = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                ],
            )?;
            let n_region = n_region_aggregate(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &ranges,
                &aligners,
                &aggregations,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("n-region aggregation returned None"))?;
            assert_eq!(aggregate_output(&n_region)?, vec![10, 7]);

            let residual = PyList::new(py, [predicate(py, &[2, 4], &[1, 2, 3, 4], "==")?])?;
            let filtered = dual_regions_and_residual_aggregate(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly(),
                &residual,
                &aggregations,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("residual aggregation returned None"))?;
            assert_eq!(aggregate_output(&filtered)?, vec![2, 4]);

            let reverse_inputs = PyList::new(py, [aggregation_input(py, &[100, 200], "sum")?])?;
            let reverse = dual_regions_aggregate_reverse(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                aligner.clone(),
                &reverse_inputs,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("reverse aggregation returned None"))?;
            assert_eq!(aggregate_output(&reverse)?, vec![100, 100, 300, 300]);

            let filtered_reverse = dual_regions_and_residual_aggregate_reverse(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                aligner.clone(),
                &residual,
                &reverse_inputs,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("reverse residual aggregation returned None"))?;
            assert_eq!(aggregate_output(&filtered_reverse)?, vec![0, 100, 0, 200]);

            let n_reverse = n_region_aggregate_reverse(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &ranges,
                &aligners,
                &reverse_inputs,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("n reverse aggregation returned None"))?;
            assert_eq!(aggregate_output(&n_reverse)?, vec![100, 100, 300, 300]);

            let n_filtered = n_region_and_residual_aggregate(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &ranges,
                &aligners,
                &residual,
                &aggregations,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("n residual aggregation returned None"))?;
            assert_eq!(aggregate_output(&n_filtered)?, vec![2, 4]);

            let n_filtered_reverse = n_region_and_residual_aggregate_reverse(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &ranges,
                &aligners,
                &residual,
                &reverse_inputs,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("n reverse residual aggregation returned None"))?;
            assert_eq!(aggregate_output(&n_filtered_reverse)?, vec![0, 100, 0, 200]);

            let multi = PyList::new(
                py,
                [
                    aggregation_input(py, &[1, 2, 3, 4], "sum")?,
                    aggregation_input(py, &[1, 2, 3, 4], "min")?,
                    aggregation_input(py, &[1, 2, 3, 4], "max")?,
                    aggregation_input(py, &[1, 2, 3, 4], "prod")?,
                ],
            )?;
            let multi_result = dual_regions_aggregate(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                aligner,
                &multi,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("multi aggregation returned None"))?;
            assert_eq!(
                aggregate_outputs(&multi_result)?,
                // `min`/`max` return source positions, matching the existing
                // aggregation contract; sum/product return values.
                vec![vec![10, 7], vec![0, 2], vec![3, 3], vec![24, 12]]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn broad_dual_aggregation_uses_rank_tree_results() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let rows = 100;
            let left_index = PyArray1::from_vec(py, (0..rows as i64).collect());
            let right_index = PyArray1::from_vec(py, (0..rows as i64).collect());
            let left_values: Vec<i64> = (1..=rows as i64).collect();
            let right_values: Vec<i64> = (1..=rows as i64).collect();
            let first = predicate(py, &left_values, &right_values, "<=")?;
            let second = predicate(py, &left_values, &right_values, "<=")?;
            let aligner = PyArray1::from_vec(py, (0..rows as i64).collect()).readonly();
            let aggregations = PyList::new(
                py,
                [
                    aggregation_input(py, &right_values, "sum")?,
                    aggregation_input(py, &right_values, "min")?,
                    aggregation_input(py, &right_values, "max")?,
                ],
            )?;

            let result = dual_regions_aggregate(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                aligner,
                &aggregations,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("broad aggregation returned None"))?;

            assert_eq!(
                aggregate_outputs(&result)?,
                vec![
                    (0..rows)
                        .map(|row| ((row + 1 + rows) * (rows - row) / 2) as i64)
                        .collect(),
                    (0..rows as i64).collect(),
                    vec![99; rows],
                ]
            );
            Ok(())
        })
        .unwrap();
    }

    fn expected_pairs(
        left_index: &[i64],
        left_values: &[Vec<i64>],
        right_index: &[i64],
        right_values: &[Vec<i64>],
        operators: &[&str],
    ) -> Vec<(i64, i64)> {
        let mut expected = Vec::new();
        for left_position in 0..left_index.len() {
            for (right_position, &right_index_value) in right_index.iter().enumerate() {
                if operators.iter().enumerate().all(|(predicate, &operator)| {
                    matches(
                        left_values[predicate][left_position],
                        right_values[predicate][right_position],
                        operator,
                    )
                }) {
                    expected.push((left_index[left_position], right_index_value));
                }
            }
        }
        expected.sort_unstable();
        expected
    }

    #[test]
    fn dual_indices_materialize_all_keep_modes_using_index_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            let first = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let second = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let alignment = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly();

            let run = |keep| {
                dual_regions_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &second,
                    alignment.clone(),
                    keep,
                )
            };

            let all = run("all")?.ok_or_else(|| PyValueError::new_err("all returned None"))?;
            assert_eq!(pairs(&all)?.len(), 6);

            let any = run("any")?.ok_or_else(|| PyValueError::new_err("any returned None"))?;
            assert_eq!(pairs(&any)?.len(), 2);

            let first_result =
                run("first")?.ok_or_else(|| PyValueError::new_err("first returned None"))?;
            assert_eq!(pairs(&first_result)?, vec![(200, 20), (100, 10)]);

            let last_result =
                run("last")?.ok_or_else(|| PyValueError::new_err("last returned None"))?;
            assert_eq!(pairs(&last_result)?, vec![(200, 30), (100, 40)]);

            assert!(run("invalid").is_err());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_materializes_three_range_predicates() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            let first = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let second = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let third = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let range_predicates = PyList::new(py, [second.clone(), third.clone()])?;
            let first_aligner = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]);
            let second_aligner = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]);
            let aligners = PyList::new(py, [first_aligner, second_aligner])?;
            let residuals = PyList::new(py, [third])?;

            for keep in ["all", "any", "first", "last"] {
                let n_result = n_region_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &range_predicates,
                    &aligners,
                    keep,
                )?
                .ok_or_else(|| PyValueError::new_err("n-region indices returned None"))?;
                let dual_result = dual_regions_and_residual_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &second,
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly(),
                    &residuals,
                    keep,
                )?
                .ok_or_else(|| PyValueError::new_err("dual residual indices returned None"))?;

                let mut n_pairs = pairs(&n_result)?;
                let mut dual_pairs = pairs(&dual_result)?;
                n_pairs.sort_unstable();
                dual_pairs.sort_unstable();
                if keep == "any" {
                    let mut n_left = n_pairs.iter().map(|&(left, _)| left).collect::<Vec<_>>();
                    let mut dual_left =
                        dual_pairs.iter().map(|&(left, _)| left).collect::<Vec<_>>();
                    n_left.sort_unstable();
                    dual_left.sort_unstable();
                    assert_eq!(n_left, dual_left, "keep={keep}");
                } else {
                    assert_eq!(n_pairs, dual_pairs, "keep={keep}");
                }
            }

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_and_residual_indices_filters_before_keep() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            let first = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let second = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let third = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let residual = predicate(py, &[0, 1], &[0, 1, 2, 1], "!=")?;
            let range_predicates = PyList::new(py, [second, third])?;
            let residual_predicates = PyList::new(py, [residual])?;
            let aligners = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                ],
            )?;

            let run = |keep| {
                n_region_and_residual_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &range_predicates,
                    &aligners,
                    &residual_predicates,
                    keep,
                )
            };

            let all = run("all")?.ok_or_else(|| PyValueError::new_err("all returned None"))?;
            let mut all_pairs = pairs(&all)?;
            all_pairs.sort_unstable();
            assert_eq!(all_pairs, vec![(100, 10), (100, 20), (100, 30), (200, 30)]);

            let any = run("any")?.ok_or_else(|| PyValueError::new_err("any returned None"))?;
            assert_eq!(pairs(&any)?, vec![(200, 30), (100, 10)]);

            let first_result =
                run("first")?.ok_or_else(|| PyValueError::new_err("first returned None"))?;
            assert_eq!(pairs(&first_result)?, vec![(200, 30), (100, 10)]);

            let last_result =
                run("last")?.ok_or_else(|| PyValueError::new_err("last returned None"))?;
            assert_eq!(pairs(&last_result)?, vec![(200, 30), (100, 30)]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_supports_every_range_operator() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200, 300]);
            let left_values = vec![2_i64, 4, 6];
            let right_values = vec![1_i64, 2, 3, 4, 5, 6, 7];
            let right_index_values = vec![10_i64, 20, 30, 40, 50, 60, 70];

            for operator in ["<", "<="] {
                let (prepared_right, aligner) = if matches!(operator, "<" | "<=") {
                    (right_values.clone(), vec![0_i64, 1, 2, 3, 4, 5, 6])
                } else {
                    (
                        right_values.iter().rev().copied().collect(),
                        vec![6_i64, 5, 4, 3, 2, 1, 0],
                    )
                };
                // Keep the first anchor ascending; the loop exercises the
                // additional prefix and suffix builders.
                let first = predicate(py, &left_values, &right_values, "<=")?;
                let second = predicate(py, &left_values, &prepared_right, operator)?;
                let third = predicate(py, &left_values, &prepared_right, operator)?;
                let range_predicates = PyList::new(py, [second, third])?;
                let aligner_one = PyArray1::from_vec(py, aligner.clone());
                let aligner_two = PyArray1::from_vec(py, aligner);
                let aligners = PyList::new(py, [aligner_one, aligner_two])?;

                let result = n_region_indices(
                    py,
                    left_index.as_any(),
                    PyArray1::from_vec(py, right_index_values.clone()).as_any(),
                    &first,
                    &range_predicates,
                    &aligners,
                    "all",
                )?
                .ok_or_else(|| PyValueError::new_err("operator test returned None"))?;
                let mut actual = pairs(&result)?;
                actual.sort_unstable();
                let expected = expected_pairs(
                    &[100, 200, 300],
                    &[
                        left_values.clone(),
                        left_values.clone(),
                        left_values.clone(),
                    ],
                    &right_index_values,
                    &[
                        right_values.clone(),
                        right_values.clone(),
                        right_values.clone(),
                    ],
                    &["<=", operator, operator],
                );
                assert_eq!(actual, expected, "operator={operator}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_applies_independent_aligners_to_four_extra_predicates() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let left_values = vec![15_i64, 35];
            let predicate_left_values = vec![left_values.clone(); 5];
            let right_index_values = vec![10_i64, 20, 30, 40, 50];
            let global_right = vec![10_i64, 20, 30, 40, 50];
            let operators = ["<=", "<=", "<=", "<=", "<="];
            let prepared_rights = [
                global_right.clone(),
                vec![5_i64, 15, 25, 35, 45],
                global_right.clone(),
                vec![0_i64, 10, 20, 30, 40],
                global_right.clone(),
            ];
            let aligner = vec![0_i64, 1, 2, 3, 4];
            let reverse_aligner = vec![4_i64, 3, 2, 1, 0];
            let first = predicate(py, &left_values, &prepared_rights[0], operators[0])?;
            let mut range_predicates = Vec::new();
            let mut aligners = Vec::new();
            for (position, operator) in operators.iter().enumerate().skip(1) {
                range_predicates.push(predicate(
                    py,
                    &predicate_left_values[position],
                    &prepared_rights[position],
                    operator,
                )?);
                aligners.push(PyArray1::from_vec(
                    py,
                    if position == 1 || position == 3 {
                        reverse_aligner.clone()
                    } else {
                        aligner.clone()
                    },
                ));
            }
            let range_predicates = PyList::new(py, range_predicates)?;
            let aligners = PyList::new(py, aligners)?;

            let result = n_region_indices(
                py,
                left_index.as_any(),
                PyArray1::from_vec(py, right_index_values.clone()).as_any(),
                &first,
                &range_predicates,
                &aligners,
                "all",
            )?
            .ok_or_else(|| PyValueError::new_err("alignment test returned None"))?;
            let mut actual = pairs(&result)?;
            actual.sort_unstable();
            let expected_rights = vec![
                global_right.clone(),
                vec![45_i64, 35, 25, 15, 5],
                global_right.clone(),
                vec![40_i64, 30, 20, 10, 0],
                global_right,
            ];
            let expected = expected_pairs(
                &[100, 200],
                &predicate_left_values,
                &right_index_values,
                &expected_rights,
                &operators,
            );
            assert_eq!(actual, expected);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_handles_duplicates_boundaries_and_no_matches() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let right_index = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4]);
            let left_values = [1_i64, 2, 9];
            let right_values = [1_i64, 1, 2, 2];
            let first = predicate(py, &left_values, &right_values, "<=")?;
            let second = predicate(py, &left_values, &right_values, "<=")?;
            let third = predicate(py, &left_values, &right_values, "<=")?;
            let predicates = PyList::new(py, [second, third])?;
            let identity = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]);
            let aligners = PyList::new(py, [identity.clone(), identity])?;

            let result = n_region_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &predicates,
                &aligners,
                "all",
            )?
            .ok_or_else(|| PyValueError::new_err("duplicate test returned None"))?;
            let mut actual = pairs(&result)?;
            actual.sort_unstable();
            assert_eq!(
                actual,
                vec![(10, 1), (10, 2), (10, 3), (10, 4), (20, 3), (20, 4)]
            );

            let no_match_left = predicate(py, &[9_i64, 10, 11], &right_values, "<=")?;
            let no_match_second = predicate(py, &[9_i64, 10, 11], &right_values, "<=")?;
            let no_match_third = predicate(py, &[9_i64, 10, 11], &right_values, "<=")?;
            let no_match_predicates = PyList::new(py, [no_match_second, no_match_third])?;
            let no_match_aligners = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                ],
            )?;
            assert!(n_region_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &no_match_left,
                &no_match_predicates,
                &no_match_aligners,
                "all",
            )?
            .is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_keep_modes_select_from_complete_matches() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            let first = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let second = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let third = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let predicates = PyList::new(py, [second, third])?;
            let aligners = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]),
                ],
            )?;

            let mut outputs = Vec::new();
            for keep in ["any", "first", "last"] {
                let result = n_region_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &predicates,
                    &aligners,
                    keep,
                )?
                .ok_or_else(|| PyValueError::new_err("selection returned None"))?;
                let result_pairs = pairs(&result)?;
                assert_eq!(result_pairs.len(), 2, "keep={keep}");
                outputs.push((keep, result_pairs));
            }
            let mut any = outputs[0].1.clone();
            let mut first_result = outputs[1].1.clone();
            let mut last_result = outputs[2].1.clone();
            any.sort_unstable();
            first_result.sort_unstable();
            last_result.sort_unstable();
            assert_eq!(any, vec![(100, 40), (200, 30)]);
            assert_eq!(first_result, vec![(100, 10), (200, 20)]);
            assert_eq!(last_result, vec![(100, 40), (200, 30)]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_supports_descending_first_predicates() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200, 300]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40, 50, 60, 70]);
            let right_values = [7_i64, 6, 5, 4, 3, 2, 1];
            let alignment = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3, 4, 5, 6]);

            for operator in [">", ">="] {
                let first = predicate(py, &[2, 4, 6], &right_values, operator)?;
                let second = predicate(py, &[2, 4, 6], &right_values, operator)?;
                let third = predicate(py, &[2, 4, 6], &right_values, operator)?;
                let predicates = PyList::new(py, [second, third])?;
                let aligners = PyList::new(py, [alignment.clone(), alignment.clone()])?;
                let result = n_region_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &predicates,
                    &aligners,
                    "all",
                )?
                .ok_or_else(|| PyValueError::new_err("descending test returned None"))?;
                let mut actual = pairs(&result)?;
                actual.sort_unstable();
                let expected = expected_pairs(
                    &[100, 200, 300],
                    &[vec![2, 4, 6], vec![2, 4, 6], vec![2, 4, 6]],
                    &[10, 20, 30, 40, 50, 60, 70],
                    &[
                        right_values.to_vec(),
                        right_values.to_vec(),
                        right_values.to_vec(),
                    ],
                    &[operator, operator, operator],
                );
                assert_eq!(actual, expected, "operator={operator}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_returns_none_when_a_later_predicate_has_no_matches() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let first = predicate(py, &[1, 2], &[1, 2, 3], "<=")?;
            let second = predicate(py, &[100, 200], &[1, 2, 3], "<=")?;
            let third = predicate(py, &[1, 2], &[1, 2, 3], "<=")?;
            let predicates = PyList::new(py, [second, third])?;
            let aligners = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1, 2]),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2]),
                ],
            )?;
            assert!(n_region_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &predicates,
                &aligners,
                "all",
            )?
            .is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_rejects_invalid_alignment_permutations() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let first = predicate(py, &[1], &[1, 2, 3], "<=")?;
            let second = predicate(py, &[1], &[1, 2, 3], "<=")?;
            let third = predicate(py, &[1], &[1, 2, 3], "<=")?;
            let predicates = PyList::new(py, [second, third])?;
            for invalid in [vec![0_i64, 0, 2], vec![-1_i64, 1, 2], vec![0_i64, 1, 3]] {
                let aligners = PyList::new(
                    py,
                    [
                        PyArray1::from_vec(py, invalid.clone()),
                        PyArray1::from_vec(py, vec![0_i64, 1, 2]),
                    ],
                )?;
                assert!(n_region_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &predicates,
                    &aligners,
                    "all",
                )
                .is_err());
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_indices_rejects_invalid_shapes_and_operators() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![1_i64, 2]);
            let right_index = PyArray1::from_vec(py, vec![3_i64, 4]);
            let first = predicate(py, &[1, 2], &[1, 2], "<=")?;
            let second = predicate(py, &[1, 2], &[1, 2], "<=")?;
            let predicates = PyList::new(py, [second.clone()])?;
            let one_aligner = PyList::new(py, [PyArray1::from_vec(py, vec![0_i64, 1])])?;
            assert!(n_region_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &predicates,
                &one_aligner,
                "all",
            )
            .is_err());

            let third = predicate(py, &[1, 2], &[1, 2], "==")?;
            let predicates = PyList::new(py, [second, third])?;
            let wrong_length_aligner = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1]),
                    PyArray1::from_vec(py, vec![0_i64]),
                ],
            )?;
            assert!(n_region_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &predicates,
                &wrong_length_aligner,
                "all",
            )
            .is_err());

            let invalid_first = predicate(py, &[1, 2], &[1, 2], "==")?;
            let valid_aligners = PyList::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1]),
                    PyArray1::from_vec(py, vec![0_i64, 1]),
                ],
            )?;
            assert!(n_region_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &invalid_first,
                &predicates,
                &valid_aligners,
                "all",
            )
            .is_err());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn n_region_starts_reports_empty_and_boundary_searches() -> Result<(), String> {
        assert_eq!(
            n_region_starts(&[0, 1, 5], &[1, 2, 3])?,
            Some(vec![3, 0, 3])
        );
        assert_eq!(n_region_starts(&[0, 5], &[1, 2, 3])?, None);
        Ok(())
    }

    #[test]
    fn range_boundary_covers_all_inequality_edges() {
        let ascending = ArrayView1::from(&[1_i64, 2, 3, 4]);
        let descending = ArrayView1::from(&[4_i64, 3, 2, 1]);
        assert_eq!(range_boundary(3, ascending, CompareOp::Lt), 3);
        assert_eq!(range_boundary(3, ascending, CompareOp::Le), 2);
        assert_eq!(range_boundary(3, descending, CompareOp::Gt), 2);
        assert_eq!(range_boundary(3, descending, CompareOp::Ge), 1);
    }

    #[test]
    fn dual_indices_apply_residuals_before_keep() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 101, 102, 103]);
            let first = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let second = predicate(py, &[1, 3], &[1, 2, 3, 4], "<=")?;
            let residual = predicate(py, &[1, 3], &[0, 0, 2, 2], ">")?;
            let residuals = PyList::new(py, [residual])?;
            let alignment = PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly();

            let result = dual_regions_and_residual_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                alignment,
                &residuals,
                "all",
            )?
            .ok_or_else(|| PyValueError::new_err("residual join returned None"))?;

            let mut result_pairs = pairs(&result)?;
            result_pairs.sort_unstable();
            assert_eq!(
                result_pairs,
                vec![(10, 100), (10, 101), (20, 102), (20, 103)]
            );

            let run_selected = |keep| {
                dual_regions_and_residual_indices(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &second,
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly(),
                    &residuals,
                    keep,
                )
            };

            let any = run_selected("any")?
                .ok_or_else(|| PyValueError::new_err("residual any join returned None"))?;
            let mut any_pairs = pairs(&any)?;
            any_pairs.sort_unstable();
            assert_eq!(any_pairs, vec![(10, 100), (20, 102)]);

            let first_result = run_selected("first")?
                .ok_or_else(|| PyValueError::new_err("residual first join returned None"))?;
            let mut first_pairs = pairs(&first_result)?;
            first_pairs.sort_unstable();
            assert_eq!(first_pairs, vec![(10, 100), (20, 102)]);

            let last_result = run_selected("last")?
                .ok_or_else(|| PyValueError::new_err("residual last join returned None"))?;
            let mut last_pairs = pairs(&last_result)?;
            last_pairs.sort_unstable();
            assert_eq!(last_pairs, vec![(10, 101), (20, 103)]);

            let partial_residual = predicate(py, &[1, 3], &[2, 2, 2, 2], ">")?;
            let partial_residuals = PyList::new(py, [partial_residual])?;
            let partial = dual_regions_and_residual_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly(),
                &partial_residuals,
                "all",
            )?
            .ok_or_else(|| PyValueError::new_err("partial residual join returned None"))?;
            assert_eq!(pairs(&partial)?, vec![(20, 102), (20, 103)]);

            let no_match_residual = predicate(py, &[1, 3], &[10, 10, 10, 10], ">")?;
            let no_match_residuals = PyList::new(py, [no_match_residual])?;
            assert!(dual_regions_and_residual_indices(
                py,
                left_index.as_any(),
                right_index.as_any(),
                &first,
                &second,
                PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly(),
                &no_match_residuals,
                "all",
            )?
            .is_none());
            Ok(())
        })
        .unwrap();
    }
}
