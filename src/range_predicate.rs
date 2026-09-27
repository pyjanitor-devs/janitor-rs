//! Typed range-anchor predicates shared by range joins and regions.
//!
//! A range anchor is the Rust-facing description of one comparison between a
//! left value array and a sorted right value array. The normal six-field
//! Python representation is:
//!
//! ```text
//! (left_values, left_index, right_values, right_index,
//!  right_index_is_ordered, operator)
//! ```
//!
//! Region callers normalize this to the five-field representation:
//!
//! ```text
//! (left_values, left_index, right_values, right_index, operator)
//! ```
//!
//! The ordering flag is validated at the Python boundary but is not used by
//! the region algorithm. PyJanitor owns sorting and supplies the right values
//! in the order required by binary search. This module owns only tuple
//! parsing, dtype dispatch, shape validation, and delegation to the existing
//! typed window implementation.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::aggs::ensure_equal_lengths_core;
use crate::anchor_non_equi_join::build_range_core_with_labels;
use crate::join_common::SingleJoinResult;
use crate::op::CompareOp;

/// One typed range anchor after parsing.
///
/// The fields intentionally retain owned `PyReadonlyArray1` handles instead
/// of storing only borrowed `ArrayView1` values. The handles keep the Python
/// NumPy owners alive for the entire Rust kernel call; a view alone would
/// carry the element lifetime but would not keep the Python objects alive
/// across the dispatch and region/window construction steps. The views are
/// created only inside the short typed operations that consume this struct.
///
/// The five-element extended tuple is
/// `(left_values, left_index, right_values, right_index, operator)`. The
/// six-element basic tuple inserts a boolean `right_index_is_ordered` before
/// `operator`; that flag is validated by the parser but sorting is owned by
/// PyJanitor.
pub(crate) struct ParsedRangePredicate<'py, T: numpy::Element> {
    /// Left-side query values. There is one value for each logical left row.
    pub(crate) left: PyReadonlyArray1<'py, T>,
    /// Original labels or positional identifiers for the left rows.
    pub(crate) left_index: PyReadonlyArray1<'py, i64>,
    /// Sorted right-side values searched by the range kernel.
    pub(crate) right: PyReadonlyArray1<'py, T>,
    /// Original labels or positional identifiers paired with `right`.
    pub(crate) right_index: PyReadonlyArray1<'py, i64>,
    /// Comparison operator for this anchor.
    pub(crate) op: CompareOp,
}

/// Runtime dtype dispatch for one range anchor.
///
/// The two anchors may use different dtypes, but the left and right value
/// arrays within one anchor must share a dtype.
pub(crate) enum AnyParsedRangePredicate<'py> {
    /// A signed 64-bit anchor.
    I64(ParsedRangePredicate<'py, i64>),
    /// A signed 32-bit anchor.
    I32(ParsedRangePredicate<'py, i32>),
    /// A signed 16-bit anchor.
    I16(ParsedRangePredicate<'py, i16>),
    /// A signed 8-bit anchor.
    I8(ParsedRangePredicate<'py, i8>),
    /// An unsigned 64-bit anchor.
    U64(ParsedRangePredicate<'py, u64>),
    /// An unsigned 32-bit anchor.
    U32(ParsedRangePredicate<'py, u32>),
    /// An unsigned 16-bit anchor.
    U16(ParsedRangePredicate<'py, u16>),
    /// An unsigned 8-bit anchor.
    U8(ParsedRangePredicate<'py, u8>),
    /// A 64-bit floating-point anchor.
    F64(ParsedRangePredicate<'py, f64>),
    /// A 32-bit floating-point anchor.
    F32(ParsedRangePredicate<'py, f32>),
}

/// A range anchor after parsing an aggregation tuple.
///
/// Aggregation tuples add an ordering flag and, in the eight-field form,
/// output-position maps around the ordinary five-field range predicate. The
/// kernel should not need to remember those numeric tuple positions, so this
/// struct keeps the parsed range and metadata together.
pub(crate) struct ParsedAggregationRangeAnchor<'py> {
    /// Typed range data used by binary-search/window construction.
    pub(crate) range: AnyParsedRangePredicate<'py>,
    /// The public ordering flag, when the tuple carries one. Region and
    /// range aggregation validate it but rely on PyJanitor for sorting.
    pub(crate) ordered: Option<bool>,
    /// Optional compact output labels for forward aggregation.
    pub(crate) left_output_positions: Option<PyReadonlyArray1<'py, i64>>,
    /// Optional compact output labels for reverse aggregation.
    pub(crate) right_output_positions: Option<PyReadonlyArray1<'py, i64>>,
}

/// Parse one aggregation range anchor without exposing tuple positions to a
/// kernel.
///
/// The first anchor uses the established six- or eight-field aggregation
/// form. The second anchor uses the five-field range form because it does not
/// carry output maps. All NumPy arrays remain borrowed through the returned
/// request; no value or index column is copied.
pub(crate) fn parse_aggregation_range_anchor<'py>(
    tuple: &Bound<'py, PyTuple>,
    first: bool,
) -> PyResult<ParsedAggregationRangeAnchor<'py>> {
    let (range, ordered, left_output_positions, right_output_positions) = if first {
        let (operator_position, ordered, left_output_positions, right_output_positions) =
            match tuple.len() {
                6 => {
                    let ordered = tuple.get_item(4)?.extract::<bool>()?;
                    (5, Some(ordered), None, None)
                }
                8 => {
                    let ordered = tuple.get_item(4)?.extract::<bool>()?;
                    let left = tuple.get_item(5)?.extract::<PyReadonlyArray1<'py, i64>>()?;
                    let right = tuple.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?;
                    (7, Some(ordered), Some(left), Some(right))
                }
                _ => {
                    return Err(PyValueError::new_err(
                        "aggregation first anchor must contain 6 or 8 elements",
                    ));
                }
            };
        let left = tuple.get_item(0)?;
        let left_index = tuple.get_item(1)?;
        let right = tuple.get_item(2)?;
        let right_index = tuple.get_item(3)?;
        let operator = tuple.get_item(operator_position)?;
        (
            parse_any_range_parts(&left, &left_index, &right, &right_index, &operator)?,
            ordered,
            left_output_positions,
            right_output_positions,
        )
    } else {
        if tuple.len() != 5 {
            return Err(PyValueError::new_err(
                "aggregation second anchor must contain 5 elements",
            ));
        }
        (parse_any_range_predicate(tuple, true)?, None, None, None)
    };

    range.validate_lengths().map_err(PyValueError::new_err)?;
    if let Some(values) = left_output_positions.as_ref() {
        if values.as_array().len() != range.left_len() {
            return Err(PyValueError::new_err(
                "left output positions must match the left value length",
            ));
        }
    }
    if let Some(values) = right_output_positions.as_ref() {
        if values.as_array().len() != range.right_len() {
            return Err(PyValueError::new_err(
                "right output positions must match the right value length",
            ));
        }
    }

    Ok(ParsedAggregationRangeAnchor {
        range,
        ordered,
        left_output_positions,
        right_output_positions,
    })
}

/// Parse typed range fields that have already been extracted from a Python
/// tuple.
///
/// This is the no-wrapper form used by aggregation parsers. It keeps tuple
/// field access inside the parser boundary without allocating a normalized
/// temporary Python tuple for the lower-level dtype dispatcher.
pub(crate) fn parse_any_range_parts<'py>(
    left: &Bound<'py, PyAny>,
    left_index: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    operator: &Bound<'py, PyAny>,
) -> PyResult<AnyParsedRangePredicate<'py>> {
    let dtype = left
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    macro_rules! parse {
        ($ty:ty, $variant:ident) => {
            Ok(AnyParsedRangePredicate::$variant(ParsedRangePredicate {
                left: left.extract::<PyReadonlyArray1<'py, $ty>>()?,
                left_index: left_index.extract::<PyReadonlyArray1<'py, i64>>()?,
                right: right.extract::<PyReadonlyArray1<'py, $ty>>()?,
                right_index: right_index.extract::<PyReadonlyArray1<'py, i64>>()?,
                op: CompareOp::try_from_str(operator.extract::<&str>()?)?,
            }))
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

impl AnyParsedRangePredicate<'_> {
    /// Validate that each value array has a matching index-label array.
    ///
    /// Region construction uses values and labels independently while it
    /// builds paths, so malformed tuples must be rejected before that code
    /// performs positional indexing.
    ///
    /// # Errors
    ///
    /// Returns a descriptive string if either value/index pair has different
    /// lengths. The caller converts this into a Python `ValueError`.
    pub(crate) fn validate_lengths(&self) -> Result<(), String> {
        macro_rules! validate {
            ($predicate:expr) => {{
                let predicate = $predicate;
                let left_len = predicate.left.as_array().len();
                let left_index_len = predicate.left_index.as_array().len();
                ensure_equal_lengths_core("left values", left_len, "left index", left_index_len)?;
                let right_len = predicate.right.as_array().len();
                let right_index_len = predicate.right_index.as_array().len();
                ensure_equal_lengths_core(
                    "right values",
                    right_len,
                    "right index",
                    right_index_len,
                )?;
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
    /// The windows use positions in the supplied sorted `right` array; they
    /// are not labels from `right_index`.
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
    /// It is therefore also the expected length of every left-side residual
    /// predicate and aggregation-position map.
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
    /// It is also the expected length of every right-side residual predicate,
    /// source aggregation array, and output-position map.
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
/// * `tuple` - A Python tuple containing NumPy arrays and an operator. When
///   `extended` is `false`, it must have six fields; when `true`, five.
/// * `extended` - Selects the five-field region/range-extended form when
///   `true`, or the six-field basic form when `false`.
///
/// # Errors
///
/// Returns a Python `ValueError` for an invalid tuple shape, unsupported
/// dtype, invalid ordering flag, or unsupported comparator. Value/index length
/// validation is deliberately exposed separately through
/// [`AnyParsedRangePredicate::validate_lengths`] because ordinary range
/// windows and region construction validate at different stages.
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
/// flag, comparator, or value/index dtype conversion.
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
/// comparator. Parallel-array lengths are checked by
/// [`AnyParsedRangePredicate::validate_lengths`] before region construction.
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

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    type RangeArrays<'py> = (
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
    );

    fn range_arrays<'py>(py: Python<'py>) -> RangeArrays<'py> {
        (
            PyArray1::from_vec(py, vec![1, 2]),
            PyArray1::from_vec(py, vec![10, 11]),
            PyArray1::from_vec(py, vec![2, 3, 4]),
            PyArray1::from_vec(py, vec![20, 21, 22]),
        )
    }

    #[test]
    fn aggregation_parser_keeps_named_metadata_for_six_field_form() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let (left, left_index, right, right_index) = range_arrays(py);
            let six = PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    left_index.clone().into_any(),
                    right.clone().into_any(),
                    right_index.clone().into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?;
            let parsed = parse_aggregation_range_anchor(&six, true)?;
            assert_eq!(parsed.ordered, Some(true));
            assert!(parsed.left_output_positions.is_none());
            assert_eq!(parsed.range.left_len(), 2);
            assert_eq!(parsed.range.right_len(), 3);

            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn aggregation_parser_rejects_bad_maps_and_ordering_flags() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let (left, left_index, right, right_index) = range_arrays(py);
            let bad_map = PyArray1::from_vec(py, vec![100]);
            let tuple = PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    left_index.clone().into_any(),
                    right.clone().into_any(),
                    right_index.clone().into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    bad_map.into_any(),
                    PyArray1::from_vec(py, vec![200, 201, 202]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?;
            assert!(parse_aggregation_range_anchor(&tuple, true).is_err());

            let not_bool = PyTuple::new(
                py,
                [
                    left.into_any(),
                    left_index.into_any(),
                    right.into_any(),
                    right_index.into_any(),
                    1_i64.into_pyobject(py)?.into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?;
            assert!(parse_aggregation_range_anchor(&not_bool, true).is_err());
            Ok(())
        })
        .unwrap();
    }
}
