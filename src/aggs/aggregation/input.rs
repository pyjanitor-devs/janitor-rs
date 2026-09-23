//! Typed aggregation input parsing and dtype dispatch.
//!
//! `PyReadonlyArray1` owns the Python/NumPy borrow for the duration of the
//! call. `AggregationSet` later turns those handles into cheap ndarray views.
//! The handles must remain alive while the views are used; this is why parsing
//! and state construction happen in the same Python call.
//!
//! The null mask is an explicit part of the low-level contract. The caller is
//! responsible for supplying a null-free value array: nulls must not be
//! encoded inside the numeric array. A `true` mask entry means that the
//! corresponding value is null and must be skipped by value-based
//! aggregations; a `false` entry means that the value is valid. This layer
//! does not inspect values to infer nullness. Null tracking belongs entirely
//! in the boolean mask, and the caller must keep the array and mask aligned.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// Operation requested for one aggregation input.
///
/// The parser converts Python strings into this enum once at the API
/// boundary. `state.rs` then consumes the enum while constructing the typed
/// accumulator, so the comparison hot loop never examines Python strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AggregationOp {
    Sum,
    CountAll,
    CountNonNull,
    Product,
    Min,
    Max,
}

impl AggregationOp {
    /// Parse a Python operation name.
    ///
    /// `size` requests count-all, while `count` requests a non-null count for
    /// a specific value column. `prod` is accepted as the short alias for
    /// `product`.
    ///
    /// # Errors
    ///
    /// Returns `TypeError` for a non-string value and `ValueError` for an
    /// unsupported operation name.
    fn parse(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let name = value.extract::<String>().map_err(|_| {
            PyTypeError::new_err("aggregation must be one of sum, count, size, prod, min, or max")
        })?;
        match name.as_str() {
            "sum" => Ok(Self::Sum),
            "count" => Ok(Self::CountNonNull),
            "size" => Ok(Self::CountAll),
            "prod" | "product" => Ok(Self::Product),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            _ => Err(PyValueError::new_err(format!(
                "unsupported aggregation: {name}"
            ))),
        }
    }
}

pub(crate) enum AggregationInput<'py> {
    /// Count every successful comparison without reading a value column.
    CountAll,
    /// Count successful comparisons whose source value is valid, using only
    /// the authoritative boolean null mask and no source value array.
    CountNonNull(PyReadonlyArray1<'py, bool>),
    // Each variant preserves the original NumPy dtype. The output dtype is
    // an operation contract, not a consequence of whichever dtype happened
    // to be supplied by the caller.
    I64(
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    I32(
        PyReadonlyArray1<'py, i32>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    I16(
        PyReadonlyArray1<'py, i16>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    I8(
        PyReadonlyArray1<'py, i8>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    U64(
        PyReadonlyArray1<'py, u64>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    U32(
        PyReadonlyArray1<'py, u32>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    U16(
        PyReadonlyArray1<'py, u16>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    U8(
        PyReadonlyArray1<'py, u8>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    F64(
        PyReadonlyArray1<'py, f64>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
    F32(
        PyReadonlyArray1<'py, f32>,
        PyReadonlyArray1<'py, bool>,
        AggregationOp,
    ),
}

/// Parse Python aggregation requests and preserve their concrete NumPy dtypes.
///
/// Each list item is either a three-element tuple
/// `(values, null_mask, operation)`, the dtype-independent count form
/// `("*", null_mask, "count")`, or the two-element count-all shorthand
/// `("*", "count")`; `("*", "size")` is also accepted. The values and mask
/// are borrowed rather than copied, so their Python owners must remain alive
/// while the returned inputs are used by [`super::state::AggregationSet`].
///
/// # Arguments
///
/// * `inputs` - Python list of aggregation tuples. Value arrays must be
///   one-dimensional, null-free, and use one of the supported signed,
///   unsigned, or float NumPy dtypes. Null masks must be one-dimensional
///   boolean arrays aligned with their value arrays. The caller owns null
///   tracking: a mask entry of `true` is the only null marker recognized here.
///   A wildcard count-all request does not need a value array or mask. The
///   wildcard three-element `count` request needs only its boolean mask and
///   counts non-null values; `size` requests count-all.
///
/// # Returns
///
/// Parsed, typed requests in the same order as the Python list.
///
/// # Errors
///
/// Returns a Python exception if an item is not a tuple, has the wrong number
/// of fields, uses an unsupported operation or dtype, or has an invalid mask.
pub(crate) fn parse_inputs<'py>(
    inputs: &Bound<'py, PyList>,
) -> PyResult<Vec<AggregationInput<'py>>> {
    let mut result = Vec::with_capacity(inputs.len());
    for item in inputs.iter() {
        let tuple = item.cast::<PyTuple>().map_err(|_| {
            PyTypeError::new_err(
                "each aggregation must be (array, null_mask, aggregation) or ('*', aggregation)",
            )
        })?;
        if tuple.len() == 2 {
            let wildcard = tuple.get_item(0)?.extract::<String>().map_err(|_| {
                PyTypeError::new_err("two-element aggregations must be ('*', 'count')")
            })?;
            if wildcard != "*" {
                return Err(PyValueError::new_err(
                    "two-element aggregations must use '*' as the first value",
                ));
            }
            let operation = tuple.get_item(1)?.extract::<String>().map_err(|_| {
                PyTypeError::new_err("two-element aggregations must be ('*', 'count')")
            })?;
            if operation != "count" && operation != "size" {
                return Err(PyValueError::new_err(
                    "'*' may only be used with count or size",
                ));
            }
            result.push(AggregationInput::CountAll);
            continue;
        }
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "each aggregation must contain array, null_mask, and aggregation, or be ('*', aggregation)",
            ));
        }
        let array = tuple.get_item(0)?;
        let mask = tuple
            .get_item(1)?
            .extract::<PyReadonlyArray1<'py, bool>>()?;
        let op = AggregationOp::parse(&tuple.get_item(2)?)?;
        if array.extract::<String>().ok().as_deref() == Some("*") {
            if op != AggregationOp::CountNonNull {
                return Err(PyValueError::new_err("wildcard aggregation must use count"));
            }
            result.push(AggregationInput::CountNonNull(mask));
            continue;
        }
        let dtype = array
            .getattr("dtype")?
            .getattr("name")?
            .extract::<String>()?;
        // The macro keeps all dtype branches structurally identical. Rust
        // still monomorphizes each branch, so the eventual candidate update
        // does not need a boxed numeric value or a conversion through Python.
        macro_rules! typed {
            ($variant:ident, $ty:ty) => {
                result.push(AggregationInput::$variant(
                    array.extract::<PyReadonlyArray1<'py, $ty>>()?,
                    mask,
                    op,
                ))
            };
        }
        match dtype.as_str() {
            "int64" => typed!(I64, i64),
            "int32" => typed!(I32, i32),
            "int16" => typed!(I16, i16),
            "int8" => typed!(I8, i8),
            "uint64" => typed!(U64, u64),
            "uint32" => typed!(U32, u32),
            "uint16" => typed!(U16, u16),
            "uint8" => typed!(U8, u8),
            "float64" => typed!(F64, f64),
            "float32" => typed!(F32, f32),
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported aggregation dtype: {other}"
                )))
            }
        }
    }
    Ok(result)
}
