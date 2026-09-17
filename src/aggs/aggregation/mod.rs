//! Forward aggregation building blocks for fused comparison kernels.
//!
//! This module separates the Python input boundary from the accumulator
//! implementation:
//!
//! 1. [`input`] parses Python tuples, operation names, and supported dtypes.
//! 2. [`state`] owns the complete accumulator state and all update rules.
//!
//! Keeping the numerical behavior in one state module makes the operation
//! contracts easy to review together. A contributor adding an operation can
//! update the enum and parser in `input.rs`, then add its state and update
//! behavior in `state.rs`.

mod input;
mod state;

// The parent module exposes only the two entry points needed by comparison
// kernels. The concrete input and operation types stay private to this
// directory, so adding a dtype does not enlarge the crate's public API.
pub(crate) use input::parse_inputs;
pub(crate) use state::AggregationSet;
