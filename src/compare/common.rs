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

/// Validate one region-chain row's start against both `right_len` and the
/// monotonically non-increasing `previous_end` invariant `add_right_region`
/// depends on.
///
/// ELI5: this is `checked_bounds` plus one more guard specific to the
/// region-chain kernels (`non_equi_dual_regions`/`non_equi_multi_regions`).
/// `checked_bounds` alone keeps `start` within `[0, right_len)`, which is
/// necessary but not sufficient here: `add_right_region` also assumes `start`
/// never moves past `previous_end`, the low-water mark of what has already
/// been folded into the chains. A `start` that is in-bounds for `right_len`
/// but greater than `previous_end` would still corrupt the chains (or, once
/// a later row's `start` retreats back onto an already-linked position,
/// create a self-referencing node -- see the crate's own regression test for
/// that failure mode). `checked_bounds`'s own callers (e.g. `comp_batch.rs`)
/// have no such `previous_end` state, so that check does not belong there.
pub(crate) fn checked_region_start(
    start: i64,
    right_len: usize,
    previous_end: usize,
) -> Result<Option<(usize, usize)>, String> {
    if start > previous_end as i64 {
        return Err("starts must be monotonically non-increasing".to_string());
    }
    Ok(checked_bounds(start, right_len as i64, right_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;

    #[test]
    fn unbounded_start_corrupts_previous_end_and_panics_on_the_next_row() {
        // Proves `checked_bounds`'s `start >= end` guard is load-bearing on
        // its own, independent of the monotonicity check: this drives the
        // real, unmodified `add_right_region` with the exact `previous_end`
        // sequence an UNGUARDED checked_region_start would have allowed
        // through for `right_len = 5`, `starts = [100, 50]`.
        //
        // Row 0 (`start = 100`): with no `start <= right_len` guard,
        // `previous_end` is incorrectly set to 100 (`(100..5).rev()` is
        // empty, so nothing panics yet -- the corruption is silent).
        // Row 1 (`start = 50`): `(50..100).rev()` is NOT empty, and walks
        // positions 99 down to 50 against a length-5 array.
        let right = array![0_i64, 1, 2, 3, 4]; // right_len == 5
        let mut next = vec![-1_i64; right.len()];
        let mut groups = BTreeMap::<i64, GroupState>::new();

        // Row 0: simulates an unguarded `start = 100` slipping through.
        add_right_region(right.view(), 100, 5, &mut next, &mut groups);
        let corrupted_previous_end = 100; // what `previous_end = start` would leave behind

        // Row 1: `start = 50` is `<= previous_end` (100), so an unguarded
        // implementation would treat it as monotonically fine -- but 50 is
        // still nonsense relative to `right`'s real length of 5.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            add_right_region(
                right.view(),
                50,
                corrupted_previous_end,
                &mut next,
                &mut groups,
            );
        }));
        assert!(
            result.is_err(),
            "expected an out-of-bounds panic indexing `right_region[99..50]` against a length-5 array"
        );
    }

    #[test]
    fn checked_region_start_rejects_the_same_sequence_safely() {
        // The same `starts = [100, 50]` / `right_len = 5` sequence, this
        // time through the real guarded entry point. It must be rejected
        // before `previous_end` is ever corrupted -- no panic, just an Err.
        let right_len = 5;
        let previous_end = right_len; // initial value, as every caller uses

        let error = checked_region_start(100, right_len, previous_end)
            .expect_err("start=100 exceeds previous_end=5 and must be rejected");
        assert_eq!(error, "starts must be monotonically non-increasing");

        // Even taken in isolation (as if row 0 had been skipped rather than
        // rejected), a lone `start=100` must not be accepted as in-bounds:
        // `previous_end` is still 5, so this is caught by the same check.
        let error = checked_region_start(50, right_len, previous_end)
            .expect_err("start=50 exceeds right_len=5 and must be rejected");
        assert_eq!(error, "starts must be monotonically non-increasing");
    }
}
