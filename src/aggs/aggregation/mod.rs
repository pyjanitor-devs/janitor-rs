//! Forward aggregation building blocks for fused comparison kernels.

mod input;
mod op;
mod state;

// The parent module exposes only the two entry points needed by comparison
// kernels. The concrete input and operation types stay private to this
// directory, so adding a dtype does not enlarge the crate's public API.
pub(crate) use input::parse_inputs;
pub(crate) use state::AggregationSet;
