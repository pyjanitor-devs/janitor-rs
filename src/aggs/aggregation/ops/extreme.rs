//! Min/max position update rules for fused forward aggregations.

use std::cmp::Ordering;

/// Return whether a candidate strictly improves an extreme.
///
/// Equal values are not replacements, so the first encountered position is
/// retained. Callers use `-1` separately for the first non-null candidate.
/// Decide whether a comparison replaces the current extreme.
///
/// Equality is never an improvement, which preserves the first encountered
/// position when multiple candidates have the same value.
///
/// # Arguments
/// * `ordering` - Ordering of the candidate relative to the current winner.
/// * `minimum` - `true` for `min`, `false` for `max`.
///
/// # Returns
/// `true` only when the candidate is strictly better than the current winner.
pub(crate) fn improves(ordering: Ordering, minimum: bool) -> bool {
    if minimum {
        ordering == Ordering::Less
    } else {
        ordering == Ordering::Greater
    }
}
