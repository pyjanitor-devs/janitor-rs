//! Sum update rules for fused forward aggregations.

/// Add an integer using the established wrapping contract.
/// Add one signed value using wrapping arithmetic.
///
/// Wrapping is explicit because fused aggregation must have deterministic
/// release and debug behavior when an integer sum exceeds `i64` bounds.
///
/// # Arguments
/// * `total` - Mutable accumulator to update.
/// * `value` - Candidate value to add.
pub(crate) fn add_i64(total: &mut i64, value: i64) {
    *total = total.wrapping_add(value);
}

/// Add an unsigned integer without narrowing it through `i64`.
/// Add one unsigned value using wrapping `u64` arithmetic.
///
/// # Arguments
/// * `total` - Mutable `u64` accumulator to update.
/// * `value` - Candidate value to add.
pub(crate) fn add_u64(total: &mut u64, value: u64) {
    *total = total.wrapping_add(value);
}

/// Add a float using the Kahan-style compensation used by existing kernels.
///
/// The compensation is retained per output row by the caller. The public
/// result is the running `total`, matching the established forward sum API.
/// Add one floating-point value with Kahan-style compensation.
///
/// # Arguments
/// * `total` - Running floating-point sum.
/// * `compensation` - Error estimate carried between additions.
/// * `value` - Candidate value to add.
pub(crate) fn add_f64(total: &mut f64, compensation: &mut f64, value: f64) {
    let difference = value - *compensation;
    let increment = *total + difference;
    *compensation = (increment - *total) - difference;
    *total = increment;
}
