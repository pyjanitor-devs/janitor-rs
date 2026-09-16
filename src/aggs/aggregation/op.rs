//! Operation names accepted by fused aggregation inputs.
//!
//! Keeping operation parsing here gives contributors one obvious place to
//! add a new spelling and one explicit enum variant. The hot loop receives
//! the enum, not a Python string, so Python object inspection happens once
//! at the boundary rather than once per successful comparison.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AggregationOp {
    Sum,
    Count,
    Product,
    Min,
    Max,
}

impl AggregationOp {
    /// Parse one Python operation name into the internal operation enum.
    ///
    /// # Arguments
    ///
    /// * `value` - Python object expected to contain a string such as `"sum"`,
    ///   `"count"`, `"prod"`, `"min"`, or `"max"`. `"size"` is accepted as
    ///   an alias for `"count"`.
    ///
    /// # Errors
    ///
    /// Returns `TypeError` for a non-string object and `ValueError` for an
    /// unsupported operation name.
    pub(crate) fn parse(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name = value.extract::<String>().map_err(|_| {
            PyTypeError::new_err("aggregation must be one of sum, count, prod, min, or max")
        })?;
        match name.as_str() {
            "sum" => Ok(Self::Sum),
            "count" | "size" => Ok(Self::Count),
            "prod" | "product" => Ok(Self::Product),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            _ => Err(PyValueError::new_err(format!(
                "unsupported aggregation: {name}"
            ))),
        }
    }
}
