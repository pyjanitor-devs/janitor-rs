use std::collections::{BTreeMap, HashMap};

use crate::aggregation_common::ensure_equal_lengths_core;
use crate::compare_op::CompareOp;
use crate::join_search::partition_point;
use crate::join_types::{result_dict, Keep};
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicate_tuple,
    parse_predicates_with_nulls_strings, predicates_match_dispatch, Predicate, PredicateView,
};
use numpy::{ndarray::ArrayView1, PyReadonlyArray1};
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

/// Counts produced by the first pass of `keep="all"` materialization.
pub(crate) struct KeepAllFirstPass {
    pub(crate) left_match_counts: Vec<usize>,
    pub(crate) total_size: usize,
    /// Final count of every nonzero right-region label.
    pub(crate) right_region_counts: BTreeMap<i64, usize>,
}

/// Counts produced by the first pass of residual-filtered materialization.
pub(crate) struct DualResidualFirstPass {
    pub(crate) left_match_counts: Vec<usize>,
    pub(crate) total_matches: usize,
    /// Total count and start position for each right-region label.
    pub(crate) right_region_layout: HashMap<i64, (usize, usize)>,
    /// Right positions grouped into the flat region layout.
    pub(crate) right_positions: Vec<usize>,
}

/// First-pass results for `keep="any"`, `keep="first"`, and `keep="last"`.
struct DualResidualSelectionFirstPass {
    pub(crate) left_has_match: Vec<bool>,
    pub(crate) matched_left_count: usize,
    pub(crate) right_region_layout: HashMap<i64, (usize, usize)>,
    pub(crate) right_positions: Vec<usize>,
}

/// Shared candidate/layout data produced while counting a residual pass.
struct DualResidualCandidatePass {
    left_match_counts: Vec<usize>,
    total_matches: usize,
    right_region_layout: HashMap<i64, (usize, usize)>,
    right_positions: Vec<usize>,
}

/// Materialized index pairs produced by the second `keep="all"` pass.
pub(crate) struct MaterializedIndices {
    pub(crate) left_index: Vec<i64>,
    pub(crate) right_index: Vec<i64>,
}

/// Count the matching left rows for `keep="any"`.
pub(crate) struct MatchCountPass {
    pub(crate) match_count: usize,
}

/// Parse a three-element region predicate using shared left/right indices.
///
/// The Python-facing predicate is `(left_values, right_values, operator)`.
/// Index arrays are shared by both primary predicates and are supplied as
/// separate arguments to the outer entry point.
fn parse_region_predicate<'py>(predicate: &Bound<'py, PyTuple>) -> PyResult<Predicate<'py>> {
    parse_predicate_tuple(predicate)
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

/// Find the first right position whose first-region label reaches each left
/// first-region label.
///
/// ELI5: the right path is sorted by region number, so binary search finds the
/// first right row that belongs to the same-or-later region. A zero label from
/// either predicate means that left row has no final match, so its start is
/// placed at `end` and its suffix is empty. This avoids searching for a start
/// that will never be used.
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

/// Return left positions ordered from the largest first-region label to the
/// smallest.
fn descending_left_order(first_left_region: &[i64]) -> Vec<usize> {
    let mut left_order = (0..first_left_region.len()).collect::<Vec<_>>();
    left_order
        .sort_unstable_by(|&left, &right| first_left_region[right].cmp(&first_left_region[left]));
    left_order
}

/// Reorder a left-region path while preserving the original-position map.
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
    let mut max_right_region = 0_i64;
    let mut previous_end = regions.end;
    let mut match_count = 0usize;

    for (&start, &left_region) in regions.starts.iter().zip(&regions.left_region) {
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
            match_count += 1;
        }
    }

    Ok(MatchCountPass { match_count })
}

/// Materialize one matching pair for each left row that has an `any` match.
pub(crate) fn keep_any_second_pass(
    regions: &DualRegionPaths,
    first_pass: &MatchCountPass,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
) -> Result<Option<MaterializedIndices>, String> {
    if first_pass.match_count == 0 {
        return Ok(None);
    }

    let mut left_output = Vec::with_capacity(first_pass.match_count);
    let mut right_output = Vec::with_capacity(first_pass.match_count);
    let mut max_right_region = 0_i64;
    let mut max_right_position = None;
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
            let right_region = regions.right_region[right_position];
            if right_region > max_right_region {
                max_right_region = right_region;
                max_right_position = Some(right_position);
            }
        }
        previous_end = start;

        if left_region == 0 || left_region > max_right_region {
            continue;
        }

        let right_position =
            max_right_position.ok_or_else(|| "matching right region had no position".to_owned())?;
        let original_left_position = regions.left_order[left_position];
        left_output.push(left_index[original_left_position]);
        right_output.push(right_index[right_position]);
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

    let mut right_regions = BTreeMap::<i64, i64>::new();
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
            let right_region = regions.right_region[right_position];
            if right_region == 0 {
                continue;
            }

            let right_value = right_index[right_position];
            let selected = right_regions.entry(right_region).or_insert(right_value);
            if keep == Keep::Last {
                *selected = (*selected).max(right_value);
            } else {
                *selected = (*selected).min(right_value);
            }
        }
        previous_end = start;

        if left_region == 0 {
            continue;
        }

        let selected = if keep == Keep::Last {
            right_regions
                .range(left_region..)
                .map(|(_, &right_value)| right_value)
                .max()
        } else {
            right_regions
                .range(left_region..)
                .map(|(_, &right_value)| right_value)
                .min()
        };
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
/// ELI5: starts move from right to left, so each right position enters the
/// `BTreeMap` once. The map groups those positions by second-region label;
/// summing labels from the left-region threshold gives this left row's match
/// count without scanning the active right positions again.
pub(crate) fn keep_all_first_pass(regions: &DualRegionPaths) -> Result<KeepAllFirstPass, String> {
    let mut right_regions = BTreeMap::<i64, usize>::new();
    let mut left_match_counts = Vec::with_capacity(regions.starts.len());
    let mut total_matches = 0usize;
    let mut previous_end = regions.end;

    for (&start, &left_region) in regions.starts.iter().zip(&regions.left_region) {
        if start == regions.end {
            left_match_counts.push(0);
            continue;
        }
        if start > previous_end {
            return Err("starts must be monotonically non-increasing".to_owned());
        }

        for right_position in start..previous_end {
            let right_region = regions.right_region[right_position];
            if right_region != 0 {
                *right_regions.entry(right_region).or_default() += 1;
            }
        }
        previous_end = start;

        let matches = right_regions
            .range(left_region..)
            .map(|(_, &count)| count)
            .sum::<usize>();
        left_match_counts.push(matches);
        total_matches += matches;
    }

    Ok(KeepAllFirstPass {
        left_match_counts,
        total_size: total_matches,
        right_region_counts: right_regions,
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

    // ELI5: this buffer stores every right position seen by the sweep. A
    // right position can later be reused for several left rows, so this is a
    // different buffer from `positions`, which stores emitted matching pairs.
    let mut right_region_offsets =
        HashMap::<i64, usize>::with_capacity(first_pass.right_region_counts.len());
    let mut right_position_offset = 0usize;
    for (&right_region, &count) in &first_pass.right_region_counts {
        right_region_offsets.insert(right_region, right_position_offset);
        right_position_offset += count;
    }

    let mut right_regions = BTreeMap::<i64, usize>::new();
    let mut right_positions = vec![0usize; right_position_offset];
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

        for right_position in start..previous_end {
            let right_region = regions.right_region[right_position];
            if right_region != 0 {
                let region_offset = *right_region_offsets
                    .get(&right_region)
                    .ok_or_else(|| "right region was missing from the first pass".to_owned())?;
                let region_count = right_regions.entry(right_region).or_default();
                let position = region_offset + *region_count;
                right_positions[position] = right_position;
                *region_count += 1;
            }
        }
        previous_end = start;

        let match_count = first_pass.left_match_counts[left_position];
        if match_count == 0 {
            continue;
        }

        let output_start = offsets[left_position];
        let output_end = output_start + match_count;
        let mut output_position = output_start;
        for (&right_region, &count) in right_regions.range(left_region..) {
            let region_offset = *right_region_offsets
                .get(&right_region)
                .ok_or_else(|| "right region was missing from the first pass".to_owned())?;
            for &right_position in right_positions[region_offset..region_offset + count].iter() {
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

/// Count residual-passing candidates for each left row.
///
/// ELI5: the first pass uses one flat position buffer. The BTreeMap tracks how
/// many positions from each region are currently visible, and the hashmap
/// tells each region where its positions begin in the flat buffer.
fn collect_dual_residual_candidates<F>(
    regions: &DualRegionPaths,
    predicates_pass: F,
    stop_after_first: bool,
) -> Result<DualResidualCandidatePass, String>
where
    F: Fn(usize, usize) -> bool,
{
    let mut right_region_layout = HashMap::<i64, (usize, usize)>::new();
    for &right_region in &regions.right_region {
        if right_region != 0 {
            let count = &mut right_region_layout.entry(right_region).or_default().0;
            *count = count
                .checked_add(1)
                .ok_or("region result size exceeds platform capacity")?;
        }
    }

    let mut right_position_total = 0usize;
    for (count, start) in right_region_layout.values_mut() {
        *start = right_position_total;
        right_position_total = right_position_total
            .checked_add(*count)
            .ok_or("region result size exceeds platform capacity")?;
    }

    let mut match_counts = vec![0usize; regions.left_region.len()];
    let mut total_matches = 0usize;
    let mut right_region_counts = BTreeMap::<i64, usize>::new();
    let mut right_positions = vec![0usize; right_position_total];
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
            let right_region = regions.right_region[right_position];
            if right_region != 0 {
                let region_start = right_region_layout[&right_region].1;
                let region_count = right_region_counts.entry(right_region).or_default();
                let position = region_start
                    .checked_add(*region_count)
                    .ok_or("region result size exceeds platform capacity")?;
                right_positions[position] = right_position;
                *region_count = region_count
                    .checked_add(1)
                    .ok_or("region result size exceeds platform capacity")?;
            }
        }
        previous_end = start;

        if left_region == 0 {
            continue;
        }

        'candidates: for (&right_region, &region_count) in right_region_counts.range(left_region..)
        {
            let region_start = right_region_layout[&right_region].1;
            for &right_position in &right_positions[region_start..region_start + region_count] {
                if predicates_pass(left_position, right_position) {
                    match_counts[left_position] = match_counts[left_position]
                        .checked_add(1)
                        .ok_or("region result size exceeds platform capacity")?;
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
        right_region_layout,
        right_positions,
    })
}

/// Count every residual-passing candidate for `keep="all"`.
fn count_dual_residual_matches<F>(
    regions: &DualRegionPaths,
    predicates_pass: F,
) -> Result<DualResidualFirstPass, String>
where
    F: Fn(usize, usize) -> bool,
{
    let pass = collect_dual_residual_candidates(regions, predicates_pass, false)?;
    Ok(DualResidualFirstPass {
        left_match_counts: pass.left_match_counts,
        total_matches: pass.total_matches,
        right_region_layout: pass.right_region_layout,
        right_positions: pass.right_positions,
    })
}

/// Count whether each left row has at least one residual-passing candidate.
fn count_dual_residual_left_matches<F>(
    regions: &DualRegionPaths,
    predicates_pass: F,
) -> Result<DualResidualSelectionFirstPass, String>
where
    F: Fn(usize, usize) -> bool,
{
    let pass = collect_dual_residual_candidates(regions, predicates_pass, true)?;
    let left_has_match = pass
        .left_match_counts
        .into_iter()
        .map(|count| count != 0)
        .collect();
    Ok(DualResidualSelectionFirstPass {
        left_has_match,
        matched_left_count: pass.total_matches,
        right_region_layout: pass.right_region_layout,
        right_positions: pass.right_positions,
    })
}

/// Materialize dual-region candidates after residual predicates pass.
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

    let mut right_region_counts = BTreeMap::<i64, usize>::new();
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

        for right_position in start..previous_end {
            let right_region = regions.right_region[right_position];
            if right_region != 0 {
                *right_region_counts.entry(right_region).or_default() += 1;
            }
        }
        previous_end = start;

        if left_region == 0 || first_pass.left_match_counts[left_position] == 0 {
            continue;
        }

        for (&right_region, &region_count) in right_region_counts.range(left_region..) {
            let region_start = first_pass.right_region_layout[&right_region].1;
            for &right_position in
                &first_pass.right_positions[region_start..region_start + region_count]
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

    if first_pass.matched_left_count == 0 {
        return Ok(None);
    }

    let mut right_region_counts = BTreeMap::<i64, usize>::new();
    let mut previous_end = regions.end;
    let mut left_output = vec![0_i64; first_pass.matched_left_count];
    let mut right_output = vec![0_i64; first_pass.matched_left_count];
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

        for right_position in start..previous_end {
            let right_region = regions.right_region[right_position];
            if right_region != 0 {
                *right_region_counts.entry(right_region).or_default() += 1;
            }
        }
        previous_end = start;

        if left_region == 0 || !first_pass.left_has_match[left_position] {
            continue;
        }

        let mut selected: Option<(i64, usize)> = None;
        'candidates: for (&right_region, &region_count) in right_region_counts.range(left_region..)
        {
            let region_start = first_pass.right_region_layout[&right_region].1;
            for &right_position in
                &first_pass.right_positions[region_start..region_start + region_count]
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

    if output_position != first_pass.matched_left_count {
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

/// Build both primary region paths and derive the first-predicate starts.
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
/// The shared `left_index` and `right_index` arrays carry the output labels;
/// each predicate supplies only its left values, right values, and operator.
/// `aligner` maps the second predicate's right layout into the first
/// predicate's global right layout.
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
            let first_pass = count_matching_left_rows(&regions)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
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

/// Register the standalone dual-region entry point.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dual_regions_indices, m)?)?;
    m.add_function(wrap_pyfunction!(dual_regions_and_residual_indices, m)?)?;
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
