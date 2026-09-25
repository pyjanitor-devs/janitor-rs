//! Shared residual filtering and materialization for extended joins.
//!
//! The single-predicate and two-range extended wrappers construct candidates
//! differently, but once candidates exist they use the same rules:
//!
//! - evaluate every residual predicate before applying `keep`;
//! - preserve left input order;
//! - emit every physical right row for `all`;
//! - return no pairs when no candidate survives.

use numpy::ndarray::ArrayView1;

use crate::aggs::ensure_equal_lengths_core;
use crate::join_common::{Keep, SingleJoinResult};
use crate::predicate::{null_metadata_views, predicates_match_dispatch, NullMetadata, Predicate};

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
pub(crate) fn materialize_windows_for_non_ne(
    windows: &SingleJoinResult,
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);
    let labels = windows.right_index.as_slice();

    if keep == Keep::All {
        // Count survivors first so residual predicates do not force us to
        // reserve the full range-window upper bound.
        let mut output_len = 0_usize;
        for row in 0..windows.left_index.len() {
            let left_position = windows.left_positions[row];
            for right_position in windows.starts[row]..windows.ends[row] {
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
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
            for (offset, &label) in labels[start..end].iter().enumerate() {
                let right_position = start + offset;
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
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
        let mut selected = None;
        for (offset, &label) in labels[start..end].iter().enumerate() {
            let right_position = start + offset;
            if !predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                continue;
            }
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
pub(crate) fn materialize_pairs_for_ne(
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
        let mut counts_by_left = vec![0_usize; left_index.len()];
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

        let mut output_left = vec![0_i64; output_len];
        let mut output_right = vec![0_i64; output_len];
        let mut write_positions = counts_by_left.clone();
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
