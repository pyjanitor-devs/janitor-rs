//! Count update rules for fused forward aggregations.

/// Count every successful comparison. The source value and its null mask are
/// deliberately irrelevant, matching pandas `size` semantics.
/// Count one successful comparison event.
///
/// The source value and null mask are intentionally absent from this helper:
/// `count` implements pandas-style `size`, not non-null `count`.
///
/// # Arguments
/// * `count` - Mutable `i64` counter to increment.
pub(crate) fn increment(count: &mut i64) {
    *count += 1;
}
