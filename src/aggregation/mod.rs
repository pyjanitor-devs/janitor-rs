//! Shared aggregation state, input parsing, and result-shape helpers.
//!
//! The conditional-join kernels own candidate traversal; this family owns the
//! aggregation contracts those kernels share.

pub(crate) mod common;
pub(crate) mod input;
pub(crate) mod state;
