//! Shared residual filtering and materialization for join candidates.
//!
//! The single-anchor and two-range join wrappers construct candidates
//! differently, but once candidates exist they use the same rules:
//!
//! - evaluate every residual predicate before applying `keep`;
//! - preserve left input order;
//! - emit every physical right row for `all`;
//! - return no pairs when no candidate survives.

use numpy::ndarray::ArrayView1;

use super::join_types::{Keep, SingleJoinResult};
use super::predicate::{
    null_metadata_views, predicates_match_dispatch, predicates_match_window, NullMetadata,
    Predicate,
};
use super::range_skip_scan::WindowResidualScan;
use crate::aggregation::common::ensure_equal_lengths_core;

/// Filter candidates represented by one half-open right-side window per left row.
///
/// `windows` may come from either `build_range_core` or `build_windows`; this
/// helper deliberately does not know how the initial window was constructed.
/// It only evaluates the residual predicates and applies `keep` afterward.
///
/// # Arguments
///
/// * `windows` - Candidate windows and their left/right physical layout.
/// * `predicates` - Residual predicates aligned to those physical positions.
/// * `metadata` - Optional authoritative null metadata for residual `!=`.
/// * `keep` - Selection applied after every residual predicate passes.
///
/// # Returns
///
/// Materialized original left/right labels. `Keep::All` emits every surviving
/// physical pair in supplied right-array order; other modes emit at most one
/// right label per left row.
///
/// # Errors
///
/// Returns an error for misaligned positions, invalid candidate bounds, or an
/// output allocation that exceeds platform capacity.
pub(crate) fn materialize_range_candidates(
    windows: &SingleJoinResult,
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if keep == Keep::Any {
        return materialize_range_any(windows, predicates, metadata);
    }
    if let Some(scan) = WindowResidualScan::plan(predicates, metadata, windows) {
        return materialize_range_candidates_skipping(windows, &scan, keep);
    }
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);
    let labels = windows.right_index.as_slice();
    let mut mask = Vec::new();

    if keep == Keep::All {
        // Count survivors first so residual predicates do not force us to
        // reserve the full range-window upper bound.
        let mut output_len = 0_usize;
        for row in 0..windows.left_index.len() {
            let left_position = windows.left_positions[row];
            let start = windows.starts[row];
            let end = windows.ends[row];
            mask.resize(end - start, 0);
            predicates_match_window(
                &views,
                metadata_views.as_deref(),
                left_position,
                start,
                end,
                &mut mask,
            );
            for &matches in &mask {
                if matches != 0 {
                    output_len = output_len
                        .checked_add(1)
                        .ok_or("single extended join result size exceeds platform capacity")?;
                }
            }
        }
        if output_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        for row in 0..windows.left_index.len() {
            let left_position = windows.left_positions[row];
            let start = windows.starts[row];
            let end = windows.ends[row];
            mask.resize(end - start, 0);
            predicates_match_window(
                &views,
                metadata_views.as_deref(),
                left_position,
                start,
                end,
                &mut mask,
            );
            for (&label, &matches) in labels[start..end].iter().zip(mask.iter()) {
                if matches != 0 {
                    output_left.push(windows.left_index[row]);
                    output_right.push(label);
                }
            }
        }
        return Ok((output_left, output_right));
    }

    // First/last/any emit at most one result per retained left row.
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(windows.left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(windows.left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;

    for row in 0..windows.left_index.len() {
        let left_position = windows.left_positions[row];
        let start = windows.starts[row];
        let end = windows.ends[row];

        if keep == Keep::Any {
            let mut selected = None;
            for right_position in start..end {
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
                    selected = Some(right_position);
                    break;
                }
            }
            if let Some(right_position) = selected {
                output_left.push(windows.left_index[row]);
                output_right.push(labels[right_position]);
            }
            continue;
        }

        mask.resize(end - start, 0);
        predicates_match_window(
            &views,
            metadata_views.as_deref(),
            left_position,
            start,
            end,
            &mut mask,
        );
        let mut selected = None;
        for (offset, (&label, &matches)) in labels[start..end].iter().zip(mask.iter()).enumerate() {
            if matches == 0 {
                continue;
            }
            let right_position = start + offset;
            match keep {
                Keep::Any => {
                    selected = Some(right_position);
                    break;
                }
                Keep::First => {
                    if selected.is_none_or(|current| label < labels[current]) {
                        selected = Some(right_position);
                    }
                }
                Keep::Last => {
                    if selected.is_none_or(|current| label > labels[current]) {
                        selected = Some(right_position);
                    }
                }
                Keep::All => unreachable!(),
            }
        }
        if let Some(right_position) = selected {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[right_position]);
        }
    }
    Ok((output_left, output_right))
}

/// Candidates read directly from the front of each window before `keep="any"`
/// considers a skip scan.
const ANY_DIRECT_PROBE: usize = 32;

/// `keep="any"`: report the first matching position of every window.
///
/// Most `any` joins find a match within a few candidates, and building skip
/// summaries would cost a full pass over the right values for nothing. So
/// every window is first read directly for up to [`ANY_DIRECT_PROBE`]
/// candidates. Only the rows still unmatched go to
/// [`WindowResidualScan::plan_windows`], over the rest of their windows; its
/// width and density gates then see just the work that remains.
///
/// ELI5: glance at the first few cards in each aisle. Only for the aisles
/// where nothing turned up do we bother labelling the boxes.
///
/// Positions are still read in ascending order, so each row reports the same
/// first match as the plain scan.
#[inline(never)]
fn materialize_range_any(
    windows: &SingleJoinResult,
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);
    let first_match = |left_position: usize, start: usize, end: usize| {
        (start..end).find(|&right_position| {
            predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            )
        })
    };
    let rows = windows.left_index.len();
    let mut selected: Vec<Option<usize>> = Vec::new();
    selected
        .try_reserve_exact(rows)
        .map_err(|_| "single extended join result allocation failed")?;
    let (mut pending_rows, mut pending_left, mut pending_starts, mut pending_ends) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for row in 0..rows {
        let (left_position, start, end) = (
            windows.left_positions[row],
            windows.starts[row],
            windows.ends[row],
        );
        let probe_end = end.min(start.saturating_add(ANY_DIRECT_PROBE));
        let found = first_match(left_position, start, probe_end);
        #[cfg(test)]
        if found.is_some() {
            super::range_skip_scan::ANY_PROBE_RESOLVED.with(|rows| rows.set(rows.get() + 1));
        }
        if found.is_none() && probe_end < end {
            #[cfg(test)]
            super::range_skip_scan::ANY_PROBE_PENDING.with(|rows| rows.set(rows.get() + 1));
            pending_rows.push(row);
            pending_left.push(left_position);
            pending_starts.push(probe_end);
            pending_ends.push(end);
        }
        selected.push(found);
    }
    if !pending_rows.is_empty() {
        let scan = WindowResidualScan::plan_windows(
            predicates,
            metadata,
            &pending_left,
            &pending_starts,
            &pending_ends,
        );
        for (index, &row) in pending_rows.iter().enumerate() {
            let (left_position, start, end) = (
                pending_left[index],
                pending_starts[index],
                pending_ends[index],
            );
            selected[row] = match &scan {
                Some(scan) => {
                    let mut found = None;
                    scan.for_each_match(left_position, start, end, |right_position| {
                        found = Some(right_position);
                        false
                    });
                    found
                }
                None => first_match(left_position, start, end),
            };
        }
    }
    let labels = windows.right_index.as_slice();
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(rows)
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(rows)
        .map_err(|_| "single extended join result allocation failed")?;
    for (row, right_position) in selected.into_iter().enumerate() {
        if let Some(right_position) = right_position {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[right_position]);
        }
    }
    Ok((output_left, output_right))
}

/// [`materialize_range_candidates`] when one range residual drives a skip
/// scan.
///
/// The scan reports the same surviving positions, in the same ascending
/// order, as the residual checklist, so every `keep` mode selects the same
/// pairs; it just never reads the blocks that cannot hold a match.
// Kept out of line: inlined, this loop changed the code generated for the
// residual-mask loop beside it, and dense reverse aggregations that never
// take the skip path ran about 7% slower (Arun 200K reverse size).
#[inline(never)]
fn materialize_range_candidates_skipping(
    windows: &SingleJoinResult,
    scan: &WindowResidualScan<'_>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let labels = windows.right_index.as_slice();
    let rows = windows.left_index.len();
    let window = |row: usize| {
        (
            windows.left_positions[row],
            windows.starts[row],
            windows.ends[row],
        )
    };

    if keep == Keep::All {
        let mut output_len = 0_usize;
        for row in 0..rows {
            let (left_position, start, end) = window(row);
            output_len = output_len
                .checked_add(scan.count_matches(left_position, start, end))
                .ok_or("single extended join result size exceeds platform capacity")?;
        }
        if output_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        for row in 0..rows {
            let (left_position, start, end) = window(row);
            scan.for_each_match(left_position, start, end, |right_position| {
                output_left.push(windows.left_index[row]);
                output_right.push(labels[right_position]);
                true
            });
        }
        return Ok((output_left, output_right));
    }

    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(rows)
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(rows)
        .map_err(|_| "single extended join result allocation failed")?;
    for row in 0..rows {
        let (left_position, start, end) = window(row);
        let mut selected: Option<usize> = None;
        scan.for_each_match(left_position, start, end, |right_position| {
            let replace = match keep {
                Keep::Any => {
                    selected = Some(right_position);
                    return false;
                }
                Keep::First => {
                    selected.is_none_or(|current| labels[right_position] < labels[current])
                }
                Keep::Last => {
                    selected.is_none_or(|current| labels[right_position] > labels[current])
                }
                Keep::All => unreachable!(),
            };
            if replace {
                selected = Some(right_position);
            }
            true
        });
        if let Some(right_position) = selected {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[right_position]);
        }
    }
    Ok((output_left, output_right))
}

/// Filter and materialize flat physical pairs produced by a first `!=` predicate.
///
/// Unlike a range window, `!=` produces a union of a strict prefix, a strict
/// suffix, and null-generated candidates. This helper receives that flat
/// candidate stream, applies residual predicates, and then applies `keep`.
///
/// # Arguments
///
/// * `left_index` / `right_index` - Original labels indexed by physical row.
/// * `left_positions` / `right_positions` - Parallel physical candidate pairs.
/// * `predicates` - Residual predicates evaluated for each candidate.
/// * `metadata` - Optional authoritative null metadata.
/// * `keep` - Selection mode applied after filtering.
///
/// # Returns
///
/// Flat materialized label pairs, or empty vectors when no candidate survives.
#[allow(dead_code)]
pub(crate) fn materialize_not_equal_candidates(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    left_positions: &[usize],
    right_positions: &[usize],
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_equal_lengths_core(
        "not-equal left positions",
        left_positions.len(),
        "not-equal right positions",
        right_positions.len(),
    )?;
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);

    if keep == Keep::All {
        // Count survivors, then write directly into exact-size output buffers.
        let mut counts_by_left = Vec::new();
        counts_by_left
            .try_reserve_exact(left_index.len())
            .map_err(|_| "single extended join result allocation failed")?;
        counts_by_left.resize(left_index.len(), 0_usize);
        for (&left_position, &right_position) in left_positions.iter().zip(right_positions) {
            if predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                let count = counts_by_left
                    .get_mut(left_position)
                    .ok_or("not-equal left position is out of bounds")?;
                *count = count
                    .checked_add(1)
                    .ok_or("single extended join result size exceeds platform capacity")?;
            }
        }
        let mut output_len = 0_usize;
        for count in &mut counts_by_left {
            let row_count = *count;
            *count = output_len;
            output_len = output_len
                .checked_add(row_count)
                .ok_or("single extended join result size exceeds platform capacity")?;
        }
        if output_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // Reserve and initialize explicitly so allocation failure is returned
        // as a normal kernel error instead of becoming an unrecoverable
        // `Vec` allocation panic for a large all-`!=` result.
        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        output_left.resize(output_len, 0_i64);
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        output_right.resize(output_len, 0_i64);
        let mut write_positions = Vec::new();
        write_positions
            .try_reserve_exact(counts_by_left.len())
            .map_err(|_| "single extended join result allocation failed")?;
        write_positions.extend_from_slice(&counts_by_left);
        for (&left_position, &right_position) in left_positions.iter().zip(right_positions) {
            if predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                let cursor = write_positions
                    .get_mut(left_position)
                    .ok_or("not-equal left position is out of bounds")?;
                let slot = *cursor;
                *cursor += 1;
                output_left[slot] = left_index[left_position];
                output_right[slot] = *right_index
                    .get(right_position)
                    .ok_or("not-equal right position is out of bounds")?;
            }
        }
        return Ok((output_left, output_right));
    }

    let mut selected = vec![None; left_index.len()];
    for (&left_position, &right_position) in left_positions.iter().zip(right_positions) {
        if !predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            left_position,
            right_position,
        ) {
            continue;
        }
        let selected_right = selected
            .get_mut(left_position)
            .ok_or("not-equal left position is out of bounds")?;
        match keep {
            Keep::Any => {
                if selected_right.is_none() {
                    *selected_right = Some(right_position);
                }
            }
            Keep::First | Keep::Last => {
                let replace = match *selected_right {
                    None => true,
                    Some(current) => {
                        let value = right_index[right_position];
                        let current_value = right_index[current];
                        if keep == Keep::First {
                            value < current_value
                        } else {
                            value > current_value
                        }
                    }
                };
                if replace {
                    *selected_right = Some(right_position);
                }
            }
            Keep::All => unreachable!(),
        }
    }

    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    for (left_position, right_position) in selected.into_iter().enumerate() {
        if let Some(right_position) = right_position {
            output_left.push(left_index[left_position]);
            output_right.push(
                *right_index
                    .get(right_position)
                    .ok_or("not-equal right position is out of bounds")?,
            );
        }
    }
    Ok((output_left, output_right))
}
