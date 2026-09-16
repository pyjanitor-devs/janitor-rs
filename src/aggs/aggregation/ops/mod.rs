//! Operation-specific accumulator updates.
//!
//! These small modules intentionally contain the numerical policy for one
//! operation. `state.rs` remains responsible for selecting the typed source
//! and output buffer, while these files make it easy to find and review the
//! behavior of an individual aggregation.

pub(super) mod count;
pub(super) mod extreme;
pub(super) mod product;
pub(super) mod sum;
