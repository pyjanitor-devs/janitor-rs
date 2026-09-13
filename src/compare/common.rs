//! Shared helpers for comparison kernels.

use numpy::ndarray::ArrayView1;
use std::collections::BTreeMap;

use crate::aggs::checked_end;

/// Selection policies shared by comparison kernels that return index pairs.
///
/// `First` and `Last` refer to the smallest and largest right labels, not
/// necessarily the smallest and largest right ordinals. `Any` may return the
/// first candidate encountered by the kernel.
#[derive(Clone, Copy)]
pub(crate) enum Selection {
    /// Select the smallest matching right label.
    First,
    /// Select the largest matching right label.
    Last,
    /// Select any matching right label.
    Any,
}

/// State for one right-region value's flat duplicate-position chain.
///
/// Region values may repeat, while the original left and right index labels
/// remain unique. `head` and `tail` are valid positions whenever they are not
/// `-1`; `next` stores the links between duplicate positions.
pub(crate) struct GroupState {
    pub(crate) head: i64,
    pub(crate) tail: i64,
}

impl Default for GroupState {
    fn default() -> Self {
        Self { head: -1, tail: -1 }
    }
}

/// Add newly exposed right positions to their duplicate-value chains.
///
/// ELI5: each right position gets an arrow to the next position carrying the
/// same region value. The map stores the first and last position for each
/// value, while `next` is one flat collection of arrows instead of a separate
/// allocation per value.
pub(crate) fn add_right_region(
    right_region: ArrayView1<'_, i64>,
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
            // `tail` is a valid position whenever `head` is not -1. The
            // debug check documents and verifies that invariant before the
            // release-build cast used by this hot path.
            debug_assert!(state.tail >= 0 && (state.tail as usize) < next.len());
            next[state.tail as usize] = right_position as i64;
        }
        state.tail = right_position as i64;
    }
}

/// Validate one half-open candidate range and convert it to slice indices.
///
/// ELI5: a malformed or empty candidate row has no candidates to scan, so it
/// contributes no output while other rows continue normally.
pub(crate) fn checked_bounds(start: i64, end: i64, right_len: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(start).ok()?;
    let end = checked_end(end, right_len)?;
    if start >= end {
        return None;
    }
    Some((start, end))
}
