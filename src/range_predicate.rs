//! Typed range-anchor predicates shared by range joins and regions.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::anchor_non_equi_join::build_range_core_with_labels;
use crate::join_common::SingleJoinResult;
use crate::op::CompareOp;

/// One typed range anchor after parsing.
///
/// The five-element extended tuple is
/// `(left_values, left_index, right_values, right_index, operator)`. The
/// six-element basic tuple inserts a boolean `right_index_is_ordered` before
/// `operator`; that flag is validated by the parser but sorting is owned by
/// PyJanitor.
pub(crate) struct ParsedRangePredicate<'py, T: numpy::Element> {
    pub(crate) left: PyReadonlyArray1<'py, T>,
    pub(crate) left_index: PyReadonlyArray1<'py, i64>,
    pub(crate) right: PyReadonlyArray1<'py, T>,
    pub(crate) right_index: PyReadonlyArray1<'py, i64>,
    pub(crate) op: CompareOp,
}

/// Runtime dtype dispatch for one range anchor.
///
/// The two anchors may use different dtypes, but the left and right value
/// arrays within one anchor must share a dtype.
pub(crate) enum AnyParsedRangePredicate<'py> {
    I64(ParsedRangePredicate<'py, i64>),
    I32(ParsedRangePredicate<'py, i32>),
    I16(ParsedRangePredicate<'py, i16>),
    I8(ParsedRangePredicate<'py, i8>),
    U64(ParsedRangePredicate<'py, u64>),
    U32(ParsedRangePredicate<'py, u32>),
    U16(ParsedRangePredicate<'py, u16>),
    U8(ParsedRangePredicate<'py, u8>),
    F64(ParsedRangePredicate<'py, f64>),
    F32(ParsedRangePredicate<'py, f32>),
}

impl AnyParsedRangePredicate<'_> {
    /// Validate that each value array has a matching index-label array.
    ///
    /// Region construction uses values and labels independently while it
    /// builds paths, so malformed tuples must be rejected before that code
    /// performs positional indexing.
    pub(crate) fn validate_lengths(&self) -> Result<(), String> {
        macro_rules! validate {
            ($predicate:expr) => {{
                let predicate = $predicate;
                let left_len = predicate.left.as_array().len();
                let left_index_len = predicate.left_index.as_array().len();
                if left_len != left_index_len {
                    return Err(format!(
                        "left values and left index must have equal lengths ({} != {})",
                        left_len, left_index_len
                    ));
                }
                let right_len = predicate.right.as_array().len();
                let right_index_len = predicate.right_index.as_array().len();
                if right_len != right_index_len {
                    return Err(format!(
                        "right values and right index must have equal lengths ({} != {})",
                        right_len, right_index_len
                    ));
                }
                Ok(())
            }};
        }
        match self {
            Self::I64(value) => validate!(value),
            Self::I32(value) => validate!(value),
            Self::I16(value) => validate!(value),
            Self::I8(value) => validate!(value),
            Self::U64(value) => validate!(value),
            Self::U32(value) => validate!(value),
            Self::U16(value) => validate!(value),
            Self::U8(value) => validate!(value),
            Self::F64(value) => validate!(value),
            Self::F32(value) => validate!(value),
        }
    }

    /// Build ordinary half-open windows for this anchor.
    ///
    /// `include_right_index` controls whether the right labels are retained in
    /// the returned building block. Empty windows are retained for alignment.
    ///
    /// # Arguments
    ///
    /// * `include_right_index` - Retain the original right identifiers when
    ///   `true`; omit them when another aligned anchor owns that layout.
    ///
    /// # Returns
    ///
    /// A [`SingleJoinResult`] containing left positions, original left IDs,
    /// optional right IDs, and half-open right-position windows.
    ///
    /// # Errors
    ///
    /// Returns an error when the value/index arrays are not length-aligned or
    /// the comparator is not a supported range comparator.
    pub(crate) fn windows(&self, include_right_index: bool) -> Result<SingleJoinResult, String> {
        macro_rules! build {
            ($predicate:expr) => {{
                let predicate = $predicate;
                build_range_core_with_labels(
                    predicate.left.as_array(),
                    predicate.left_index.as_array(),
                    predicate.right.as_array(),
                    predicate.right_index.as_array(),
                    true,
                    predicate.op,
                    include_right_index,
                    true,
                )
            }};
        }
        match self {
            Self::I64(value) => build!(value),
            Self::I32(value) => build!(value),
            Self::I16(value) => build!(value),
            Self::I8(value) => build!(value),
            Self::U64(value) => build!(value),
            Self::U32(value) => build!(value),
            Self::U16(value) => build!(value),
            Self::U8(value) => build!(value),
            Self::F64(value) => build!(value),
            Self::F32(value) => build!(value),
        }
    }

    /// Return the number of logical left rows in the anchor.
    ///
    /// This is the length before empty windows are removed.
    pub(crate) fn left_len(&self) -> usize {
        match self {
            Self::I64(value) => value.left.as_array().len(),
            Self::I32(value) => value.left.as_array().len(),
            Self::I16(value) => value.left.as_array().len(),
            Self::I8(value) => value.left.as_array().len(),
            Self::U64(value) => value.left.as_array().len(),
            Self::U32(value) => value.left.as_array().len(),
            Self::U16(value) => value.left.as_array().len(),
            Self::U8(value) => value.left.as_array().len(),
            Self::F64(value) => value.left.as_array().len(),
            Self::F32(value) => value.left.as_array().len(),
        }
    }

    /// Return the number of physical right positions in the anchor.
    ///
    /// This is the length of the sorted right layout, not the number of
    /// matching candidates.
    pub(crate) fn right_len(&self) -> usize {
        match self {
            Self::I64(value) => value.right.as_array().len(),
            Self::I32(value) => value.right.as_array().len(),
            Self::I16(value) => value.right.as_array().len(),
            Self::I8(value) => value.right.as_array().len(),
            Self::U64(value) => value.right.as_array().len(),
            Self::U32(value) => value.right.as_array().len(),
            Self::U16(value) => value.right.as_array().len(),
            Self::U8(value) => value.right.as_array().len(),
            Self::F64(value) => value.right.as_array().len(),
            Self::F32(value) => value.right.as_array().len(),
        }
    }
}

/// Parse one range anchor and dispatch on its NumPy dtype.
///
/// # Arguments
///
/// * `tuple` - A five-element extended anchor or six-element basic anchor.
/// * `extended` - Selects the five-element extended form when `true`.
///
/// # Errors
///
/// Returns a Python `ValueError` for an invalid tuple, comparator, or dtype.
pub(crate) fn parse_any_range_predicate<'py>(
    tuple: &Bound<'py, PyTuple>,
    extended: bool,
) -> PyResult<AnyParsedRangePredicate<'py>> {
    let dtype = tuple
        .get_item(0)?
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;

    macro_rules! parse {
        ($ty:ty, $variant:ident) => {
            if extended {
                Ok(AnyParsedRangePredicate::$variant(
                    parse_extended_range_predicate::<$ty>(tuple)?,
                ))
            } else {
                Ok(AnyParsedRangePredicate::$variant(parse_range_predicate::<
                    $ty,
                >(tuple)?))
            }
        };
    }

    match dtype.as_str() {
        "int64" => parse!(i64, I64),
        "int32" => parse!(i32, I32),
        "int16" => parse!(i16, I16),
        "int8" => parse!(i8, I8),
        "uint64" => parse!(u64, U64),
        "uint32" => parse!(u32, U32),
        "uint16" => parse!(u16, U16),
        "uint8" => parse!(u8, U8),
        "float64" => parse!(f64, F64),
        "float32" => parse!(f32, F32),
        other => Err(PyValueError::new_err(format!(
            "unsupported range predicate dtype: {other}"
        ))),
    }
}

/// Parse the six-element basic range-anchor tuple.
///
/// # Arguments
///
/// * `tuple` - `(left, left_index, right, right_index,
///   right_index_is_ordered, operator)` with matching value dtypes.
///
/// # Returns
///
/// A typed borrowed predicate retaining the NumPy arrays for the caller's
/// search operation.
///
/// # Errors
///
/// Returns a Python error for an invalid tuple length, array dtype, ordering
/// flag, or comparator.
fn parse_range_predicate<'py, T: numpy::Element>(
    tuple: &Bound<'py, PyTuple>,
) -> PyResult<ParsedRangePredicate<'py, T>> {
    if tuple.len() != 6 {
        return Err(PyValueError::new_err(
            "range predicates must contain 6 elements",
        ));
    }
    let left = tuple.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let left_index = tuple.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = tuple.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let right_index = tuple.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    tuple.get_item(4)?.extract::<bool>()?;
    let op = CompareOp::try_from_str(tuple.get_item(5)?.extract::<&str>()?)?;
    Ok(ParsedRangePredicate {
        left,
        left_index,
        right,
        right_index,
        op,
    })
}

/// Parse the five-element range-anchor tuple used by extended joins.
///
/// # Arguments
///
/// * `tuple` - `(left, left_index, right, right_index, operator)`.
///
/// # Returns
///
/// A typed borrowed predicate retaining the NumPy arrays for the caller's
/// search operation.
///
/// # Errors
///
/// Returns a Python error for an invalid tuple length, array dtype, or
/// comparator.
fn parse_extended_range_predicate<'py, T: numpy::Element>(
    tuple: &Bound<'py, PyTuple>,
) -> PyResult<ParsedRangePredicate<'py, T>> {
    if tuple.len() != 5 {
        return Err(PyValueError::new_err(
            "extended range predicates must contain 5 elements",
        ));
    }
    let left = tuple.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let left_index = tuple.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = tuple.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let right_index = tuple.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    let op = CompareOp::try_from_str(tuple.get_item(4)?.extract::<&str>()?)?;
    Ok(ParsedRangePredicate {
        left,
        left_index,
        right,
        right_index,
        op,
    })
}
