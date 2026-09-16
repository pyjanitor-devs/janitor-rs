//! Product update rules for fused forward aggregations.

/// Multiply an integer using the established wrapping contract.
/// Multiply one signed value using wrapping arithmetic.
///
/// # Arguments
/// * `total` - Mutable accumulator to update.
/// * `value` - Candidate value to multiply.
pub(crate) fn multiply_i64(total: &mut i64, value: i64) {
    *total = total.wrapping_mul(value);
}

/// Multiply an unsigned integer without narrowing it through `i64`.
/// Multiply one unsigned value using wrapping `u64` arithmetic.
///
/// # Arguments
/// * `total` - Mutable `u64` accumulator to update.
/// * `value` - Candidate value to multiply.
pub(crate) fn multiply_u64(total: &mut u64, value: u64) {
    *total = total.wrapping_mul(value);
}

/// Floating products intentionally use ordinary multiplication.
/// Multiply one floating-point value using ordinary IEEE-754 multiplication.
///
/// # Arguments
/// * `total` - Mutable floating-point product.
/// * `value` - Candidate value to multiply.
pub(crate) fn multiply_f64(total: &mut f64, value: f64) {
    *total *= value;
}
