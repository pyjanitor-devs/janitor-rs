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
//!
//! The basic dual-range index entry point uses a shared-index representation:
//! each anchor is `(left_values, right_values, operator)`, while the single
//! `left_index`, `right_index`, and `right_index_is_ordered` values are passed
//! separately. This guarantees that both anchors address the same physical
//! left and right layouts.

use super::compare_op::CompareOp;
use super::join_search::{range_window, range_window_bounds};
#[cfg(test)]
use super::join_types::SingleJoinResult;
use crate::aggregation::common::ensure_equal_lengths_core;
use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Build one typed range window without depending on the deleted legacy
/// anchor module. The returned bounds address the supplied sorted right-value
/// layout; the right index labels are copied only when building blocks need
/// to expose them to Python.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn build_range_core_with_labels<T: PartialOrd + Copy>(
    left: numpy::ndarray::ArrayView1<'_, T>,
    left_index: numpy::ndarray::ArrayView1<'_, i64>,
    right: numpy::ndarray::ArrayView1<'_, T>,
    right_index: numpy::ndarray::ArrayView1<'_, i64>,
    _right_index_is_ordered: bool,
    op: CompareOp,
    include_right_index: bool,
    retain_empty_windows: bool,
) -> Result<SingleJoinResult, String> {
    ensure_equal_lengths_core("left", left.len(), "left_index", left_index.len())?;
    ensure_equal_lengths_core("right", right.len(), "right_index", right_index.len())?;
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return Err("single join range core requires a range comparator".to_owned());
    }
    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: if include_right_index {
            right_index.to_vec()
        } else {
            Vec::new()
        },
        starts: Vec::new(),
        ends: Vec::new(),
    };
    if left.is_empty() || right.is_empty() {
        return Ok(result);
    }
    for (left_position, &left_value) in left.iter().enumerate() {
        let (start, end) = range_window(left_value, right, op);
        if start >= end && !retain_empty_windows {
            continue;
        }
        result.left_positions.push(left_position);
        result.left_index.push(left_index[left_position]);
        result.starts.push(start);
        result.ends.push(end);
    }
    Ok(result)
}

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
    /// Left-side query values. There is one value for each logical left row;
    /// this array is not necessarily sorted because PyJanitor may sort each
    /// anchor independently before alignment.
    pub(crate) left: PyReadonlyArray1<'py, T>,
    /// Original labels or physical positions paired with `left`. Region code
    /// uses these labels to align independent anchors and uses the resulting
    /// position maps when residual predicates read source arrays.
    pub(crate) left_index: PyReadonlyArray1<'py, i64>,
    /// Sorted right-side values searched by the binary-search kernel. The
    /// parser deliberately does not sort this array; the caller owns that
    /// preparation contract.
    pub(crate) right: PyReadonlyArray1<'py, T>,
    /// Original labels or physical positions paired with the sorted right
    /// values. Sorting the values does not change these labels' identity.
    pub(crate) right_index: PyReadonlyArray1<'py, i64>,
    /// Comparison operator for this anchor.
    pub(crate) op: CompareOp,
}

/// Runtime dtype dispatch for one range anchor.
///
/// The two anchors may use different dtypes, but the left and right value
/// arrays within one anchor must share a dtype.
pub(crate) enum AnyParsedRangePredicate<'py> {
    /// A signed 64-bit anchor. Each variant keeps the concrete type so the
    /// binary search and aggregation loops remain statically typed.
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
    // Dtype dispatch is based on the left values. The extraction of the right
    // values uses that same concrete type, so a left/right dtype mismatch is
    // rejected by PyO3 instead of being silently coerced.
    let dtype = left
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    let op = CompareOp::try_from_str(operator.extract::<&str>()?)?;
    macro_rules! parse {
        ($ty:ty, $variant:ident) => {
            Ok(AnyParsedRangePredicate::$variant(ParsedRangePredicate {
                left: left.extract::<PyReadonlyArray1<'py, $ty>>()?,
                left_index: left_index.extract::<PyReadonlyArray1<'py, i64>>()?,
                right: right.extract::<PyReadonlyArray1<'py, $ty>>()?,
                right_index: right_index.extract::<PyReadonlyArray1<'py, i64>>()?,
                op,
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

    /// Compute typed half-open windows for this parsed range anchor.
    ///
    /// The returned offsets address the sorted right-value layout. The
    /// companion physical position arrays are used only to validate the
    /// range-window inputs; callers must still use `right_index[offset]` when
    /// converting a sorted offset back to an original right position.
    ///
    /// Keeping this dispatch beside [`AnyParsedRangePredicate`] avoids making
    /// every caller repeat the ten-variant dtype match. The actual binary
    /// search remains in `common::range_window_bounds`, which is independent
    /// of the parsed-predicate representation.
    pub(crate) fn bounds(&self) -> Result<(Vec<usize>, Vec<usize>), String> {
        macro_rules! bounds {
            ($predicate:expr) => {
                range_window_bounds(
                    $predicate.left.as_array(),
                    $predicate.left_index.as_array(),
                    $predicate.right.as_array(),
                    $predicate.right_index.as_array(),
                    $predicate.op,
                )
            };
        }
        match self {
            Self::I64(predicate) => bounds!(predicate),
            Self::I32(predicate) => bounds!(predicate),
            Self::I16(predicate) => bounds!(predicate),
            Self::I8(predicate) => bounds!(predicate),
            Self::U64(predicate) => bounds!(predicate),
            Self::U32(predicate) => bounds!(predicate),
            Self::U16(predicate) => bounds!(predicate),
            Self::U8(predicate) => bounds!(predicate),
            Self::F64(predicate) => bounds!(predicate),
            Self::F32(predicate) => bounds!(predicate),
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
    #[cfg(test)]
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

    /// Borrow this anchor for per-row window searches.
    ///
    /// Callers that only need one row's window at a time (streaming window
    /// intersection, the envelope sweeps) use this instead of
    /// [`Self::windows`], which stores four values for every left row.
    pub(crate) fn search(&self) -> AnchorSearch<'_> {
        macro_rules! search {
            ($variant:ident, $predicate:expr) => {{
                let predicate = $predicate;
                AnchorSearch::$variant(
                    predicate.left.as_array(),
                    predicate.right.as_array(),
                    predicate.op,
                )
            }};
        }
        match self {
            Self::I64(value) => search!(I64, value),
            Self::I32(value) => search!(I32, value),
            Self::I16(value) => search!(I16, value),
            Self::I8(value) => search!(I8, value),
            Self::U64(value) => search!(U64, value),
            Self::U32(value) => search!(U32, value),
            Self::U16(value) => search!(U16, value),
            Self::U8(value) => search!(U8, value),
            Self::F64(value) => search!(F64, value),
            Self::F32(value) => search!(F32, value),
        }
    }

    /// The anchor's left and right label arrays.
    pub(crate) fn labels(&self) -> (ArrayView1<'_, i64>, ArrayView1<'_, i64>) {
        match self {
            Self::I64(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::I32(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::I16(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::I8(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::U64(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::U32(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::U16(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::U8(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::F64(value) => (value.left_index.as_array(), value.right_index.as_array()),
            Self::F32(value) => (value.left_index.as_array(), value.right_index.as_array()),
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

/// One range anchor borrowed for per-row window searches.
///
/// ELI5: instead of writing down every row's shelf range up front, keep the
/// shelf and the ruler handy and measure a row's range when it is needed.
pub(crate) enum AnchorSearch<'a> {
    I64(ArrayView1<'a, i64>, ArrayView1<'a, i64>, CompareOp),
    I32(ArrayView1<'a, i32>, ArrayView1<'a, i32>, CompareOp),
    I16(ArrayView1<'a, i16>, ArrayView1<'a, i16>, CompareOp),
    I8(ArrayView1<'a, i8>, ArrayView1<'a, i8>, CompareOp),
    U64(ArrayView1<'a, u64>, ArrayView1<'a, u64>, CompareOp),
    U32(ArrayView1<'a, u32>, ArrayView1<'a, u32>, CompareOp),
    U16(ArrayView1<'a, u16>, ArrayView1<'a, u16>, CompareOp),
    U8(ArrayView1<'a, u8>, ArrayView1<'a, u8>, CompareOp),
    F64(ArrayView1<'a, f64>, ArrayView1<'a, f64>, CompareOp),
    F32(ArrayView1<'a, f32>, ArrayView1<'a, f32>, CompareOp),
}

impl AnchorSearch<'_> {
    /// Windows for rows `first_row..first_row + starts.len()`, written into
    /// `starts` and `ends`.
    ///
    /// The dtype is dispatched once per call, so each anchor's binary
    /// searches run in a tight typed loop the CPU can overlap across rows.
    pub(crate) fn windows_into(&self, first_row: usize, starts: &mut [usize], ends: &mut [usize]) {
        macro_rules! fill {
            ($left:expr, $right:expr, $op:expr) => {{
                let rows = first_row..first_row + starts.len();
                for ((start, end), &value) in starts
                    .iter_mut()
                    .zip(ends.iter_mut())
                    .zip($left.slice(numpy::ndarray::s![rows]).iter())
                {
                    (*start, *end) = range_window(value, *$right, *$op);
                }
            }};
        }
        match self {
            Self::I64(left, right, op) => fill!(left, right, op),
            Self::I32(left, right, op) => fill!(left, right, op),
            Self::I16(left, right, op) => fill!(left, right, op),
            Self::I8(left, right, op) => fill!(left, right, op),
            Self::U64(left, right, op) => fill!(left, right, op),
            Self::U32(left, right, op) => fill!(left, right, op),
            Self::U16(left, right, op) => fill!(left, right, op),
            Self::U8(left, right, op) => fill!(left, right, op),
            Self::F64(left, right, op) => fill!(left, right, op),
            Self::F32(left, right, op) => fill!(left, right, op),
        }
    }

    /// The anchor's comparison operator.
    pub(crate) fn op(&self) -> CompareOp {
        match self {
            Self::I64(_, _, op)
            | Self::I32(_, _, op)
            | Self::I16(_, _, op)
            | Self::I8(_, _, op)
            | Self::U64(_, _, op)
            | Self::U32(_, _, op)
            | Self::U16(_, _, op)
            | Self::U8(_, _, op)
            | Self::F64(_, _, op)
            | Self::F32(_, _, op) => *op,
        }
    }
}
