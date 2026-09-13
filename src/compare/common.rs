//! Shared helpers for comparison kernels.

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
