//! Forward aggregation building blocks for fused comparison kernels.
//!
//! This module separates three concerns:
//!
//! 1. [`input`] parses Python tuples and dispatches supported NumPy dtypes.
//! 2. [`op`] turns operation names into a small Rust enum.
//! 3. [`state`] owns accumulator buffers, while [`ops`] contains the numerical
//!    rule for each operation.
//!
//! Keeping those concerns separate gives contributors a clear extension point:
//! a new operation name is added to `op.rs`, its update policy to the matching
//! file in `ops/`, and its state/output handling to `state.rs`.

mod input;
mod op;
mod ops;
mod state;

// The parent module exposes only the two entry points needed by comparison
// kernels. The concrete input and operation types stay private to this
// directory, so adding a dtype does not enlarge the crate's public API.
pub(crate) use input::parse_inputs;
pub(crate) use state::AggregationSet;
