//! Shared range-join window construction.
//!
//! A basic range join has two range predicates. This module computes the
//! half-open `starts`/`ends` windows for those predicates. Extended joins
//! reuse the same primitive and perform their additional filtering elsewhere.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::aggs::max::max_starts_ends::max_start_end_core;
use crate::aggs::min::min_starts_ends::min_start_end_core;
use crate::anchor_non_equi_join::range_window;
use crate::join_candidate_materialization::materialize_range_candidates;
use crate::join_common::{result_dict, Keep, SingleJoinResult};
use crate::op::CompareOp;
use crate::predicate::parse_predicates_with_nulls_strings;

/// A typed range predicate used by the basic two-range kernel.
///
/// The value arrays describe the comparison layout, while the paired index
/// arrays carry the labels that must be returned to Python. PyJanitor owns
/// sorting and alignment before constructing this value; Rust only consumes
/// the already-aligned views.
pub(crate) struct RangePredicate<'a, T> {
    /// Left values in logical left-row order. Null rows have already been
    /// removed for range predicates.
    pub(crate) left: ArrayView1<'a, T>,
    /// Labels paired position-for-position with `left`.
    pub(crate) left_index: ArrayView1<'a, i64>,
    /// Right values in ascending value order.
    pub(crate) right: ArrayView1<'a, T>,
    /// Labels paired position-for-position with the sorted `right` values.
    pub(crate) right_index: ArrayView1<'a, i64>,
    /// Comparator applied to the paired left and right values.
    pub(crate) op: CompareOp,
}

/// Build the intersection window for exactly two aligned range predicates.
///
/// Each right-hand array must already be sorted in ascending value order by
/// PyJanitor. The two predicates are evaluated against the same logical left
/// rows and the same logical right positions; this function only intersects
/// their positional windows. It does not sort arrays or evaluate residual
/// predicates.
///
/// # Arguments
///
/// * `first` - The first range predicate, including its left/right values and
///   the corresponding original index labels.
/// * `second` - The second range predicate. Its left and right arrays must be
///   length-aligned with `first`; its right array must use the same sorted
///   physical layout as `first.right`.
///
/// # Returns
///
/// A [`SingleJoinResult`] containing one half-open `[start, end)` window for
/// each left row whose two predicates intersect. `left_index` retains input
/// order and `right_index` is copied from the first predicate's supplied
/// right-label array. A window's positions always refer to that copied right
/// layout.
///
/// # Errors
///
/// Returns an error when value/index lengths differ or either comparator is
/// not a range comparator.
#[cfg(test)]
pub(crate) fn build_windows<T: PartialOrd + Copy>(
    first: RangePredicate<'_, T>,
    second: RangePredicate<'_, T>,
) -> Result<SingleJoinResult, String> {
    let first = build_single_windows(first)?;
    let second = build_single_windows(second)?;
    intersect_windows(first, second)
}

/// Build the positional window produced by one typed range predicate.
///
/// ELI5: each predicate independently draws a highlighted interval in the
/// sorted right-hand array. This helper draws one such interval at a time;
/// [`intersect_windows`] later keeps only the overlap between two drawings.
fn build_single_windows<T: PartialOrd + Copy>(
    predicate: RangePredicate<'_, T>,
) -> Result<SingleJoinResult, String> {
    ensure_equal_lengths_core(
        "left",
        predicate.left.len(),
        "left_index",
        predicate.left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "right",
        predicate.right.len(),
        "right_index",
        predicate.right_index.len(),
    )?;
    if !predicate.op.is_range() {
        return Err("range join requires a range comparator".to_owned());
    }

    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: predicate.right_index.to_vec(),
        starts: Vec::new(),
        ends: Vec::new(),
    };
    for (left_position, &left_value) in predicate.left.iter().enumerate() {
        let (start, end) = range_window(left_value, predicate.right, predicate.op);
        // Keep one boundary pair per left row, including an empty window.
        // Two independently typed anchors must be aligned row-for-row before
        // their windows can be intersected; the final intersection removes
        // empty rows from the public result.
        result.left_positions.push(left_position);
        result.left_index.push(predicate.left_index[left_position]);
        result.starts.push(start);
        result.ends.push(end);
    }
    Ok(result)
}

/// Intersect two already-built positional windows.
///
/// The value dtypes are intentionally absent here. Each anchor has already
/// completed its own typed binary searches; this step only combines the
/// resulting positions. PyJanitor aligns the second right array to the first
/// right layout before calling Rust, so the two windows refer to the same
/// physical right positions even when their value dtypes differ.
pub(crate) fn intersect_windows(
    first: SingleJoinResult,
    second: SingleJoinResult,
) -> Result<SingleJoinResult, String> {
    ensure_equal_lengths_core(
        "first left window",
        first.left_positions.len(),
        "first left labels",
        first.left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "second left window",
        second.left_positions.len(),
        "second left labels",
        second.left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "first window starts",
        first.left_positions.len(),
        "first window ends",
        first.ends.len(),
    )?;
    ensure_equal_lengths_core(
        "second window starts",
        second.left_positions.len(),
        "second window ends",
        second.ends.len(),
    )?;
    if first.left_positions.len() != second.left_positions.len()
        || first.right_index != second.right_index
    {
        return Err(
            "dual range predicates must use aligned left rows and right positions".to_owned(),
        );
    }

    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: first.right_index,
        starts: Vec::new(),
        ends: Vec::new(),
    };
    for row in 0..first.left_positions.len() {
        let start = first.starts[row].max(second.starts[row]);
        let end = first.ends[row].min(second.ends[row]);
        if start < end {
            result.left_positions.push(first.left_positions[row]);
            result.left_index.push(first.left_index[row]);
            result.starts.push(start);
            result.ends.push(end);
        }
    }
    Ok(result)
}

/// Select one label from each arbitrary dual-range window.
///
/// Unlike a single non-equi join, a dual-range join can produce an interior
/// window such as `[2, 4)`. Prefix and suffix extrema cannot answer that
/// shape because they include positions outside the intersection. The
/// existing min/max range kernels provide the correct arbitrary-window
/// query and choose between direct scans and a segment tree using their
/// adaptive workload guard.
///
/// # Arguments
///
/// * `windows` - Intersected, non-empty half-open windows and their right
///   index labels.
/// * `keep` - Selection mode. `first` means the smallest right label,
///   `last` the largest, `any` the first physical label, and `all` every
///   label in each window.
///
/// # Returns
///
/// Materialized left and right labels in the same left-row order as
/// `windows`. For `all`, right labels remain in their supplied physical
/// right-array order within each window.
///
/// # Errors
///
/// Returns an error if an internal window boundary cannot be represented by
/// the public int64 boundary contract, or if an extrema kernel returns an
/// invalid position. The latter indicates a violated internal window
/// invariant rather than a normal no-match result; empty windows are removed
/// by [`build_windows`].
pub(crate) fn choose_range_windows(
    windows: &SingleJoinResult,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let labels = windows.right_index.as_slice();
    if windows.starts.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    if keep == Keep::All {
        // `all` does not need an extrema query: every position in every
        // half-open window is a result, and the physical right order is the
        // required output order. Because this basic range path has no
        // residual filters, the window widths give the exact final size.
        let mut output_capacity = 0_usize;
        for (&start, &end) in windows.starts.iter().zip(&windows.ends) {
            let width = end
                .checked_sub(start)
                .ok_or("range window has invalid bounds")?;
            output_capacity = output_capacity
                .checked_add(width)
                .ok_or("range join result size exceeds platform capacity")?;
        }
        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_capacity)
            .map_err(|_| "range join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_capacity)
            .map_err(|_| "range join result allocation failed")?;
        for (row, (&start, &end)) in windows.starts.iter().zip(&windows.ends).enumerate() {
            for &label in &labels[start..end] {
                output_left.push(windows.left_index[row]);
                output_right.push(label);
            }
        }
        return Ok((output_left, output_right));
    }

    if keep == Keep::Any {
        // Every stored window is non-empty, so its first physical position is
        // a valid arbitrary match.
        let mut output_left = Vec::with_capacity(windows.left_index.len());
        let mut output_right = Vec::with_capacity(windows.left_index.len());
        for (row, &start) in windows.starts.iter().enumerate() {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[start]);
        }
        return Ok((output_left, output_right));
    }

    // The reusable min/max kernels accept signed int64 boundaries because
    // that is the public NumPy representation. Convert the internal usize
    // windows with a checked cast before handing them to those kernels.
    let starts: Array1<i64> = windows
        .starts
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| "range window start exceeds int64 capacity".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let ends: Array1<i64> = windows
        .ends
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| "range window end exceeds int64 capacity".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let nulls = Array1::from_elem(labels.len(), false);
    // The RMQ kernels return offsets into `labels`, not public labels. The
    // final loop below performs that last position-to-label conversion.
    let selected_positions = match keep {
        Keep::First => min_start_end_core(
            ArrayView1::from(labels),
            starts.view(),
            ends.view(),
            nulls.view(),
        )?,
        Keep::Last => max_start_end_core(
            ArrayView1::from(labels),
            starts.view(),
            ends.view(),
            nulls.view(),
        )?,
        Keep::Any | Keep::All => unreachable!(),
    };

    let mut output_left = Vec::with_capacity(windows.left_index.len());
    let mut output_right = Vec::with_capacity(windows.left_index.len());
    for (row, &position) in selected_positions.iter().enumerate() {
        let position = usize::try_from(position)
            .map_err(|_| "range window selection returned an invalid position")?;
        let label = *labels
            .get(position)
            .ok_or("range window selection returned an out-of-bounds position")?;
        output_left.push(windows.left_index[row]);
        output_right.push(label);
    }
    Ok((output_left, output_right))
}

/// Keep the extracted NumPy owners alive while borrowed views are used.
///
/// ELI5: `ArrayView1` is only a window into an array; it does not own the
/// array. The owners therefore have to live in this struct until the window
/// calculation has finished.
pub(crate) struct ParsedRangePredicate<'py, T: numpy::Element> {
    pub(crate) left: PyReadonlyArray1<'py, T>,
    pub(crate) left_index: PyReadonlyArray1<'py, i64>,
    pub(crate) right: PyReadonlyArray1<'py, T>,
    pub(crate) right_index: PyReadonlyArray1<'py, i64>,
    pub(crate) op: CompareOp,
}

/// A range predicate whose concrete NumPy dtype is selected at runtime.
///
/// The two anchors in a dual-range join do not need the same dtype. Each one
/// is parsed into its own variant, searched with its own typed
/// `partition_point`, and converted to an untyped positional window before
/// the two windows are intersected.
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
    /// Build this anchor's positional windows without exposing its dtype.
    fn windows(&self) -> Result<SingleJoinResult, String> {
        macro_rules! build {
            ($predicate:expr) => {{
                let predicate = $predicate;
                build_single_windows(RangePredicate {
                    left: predicate.left.as_array(),
                    left_index: predicate.left_index.as_array(),
                    right: predicate.right.as_array(),
                    right_index: predicate.right_index.as_array(),
                    op: predicate.op,
                })
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

/// Parse one range anchor using the dtype of that anchor's own value arrays.
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

/// Build dual-range windows while dispatching each anchor independently.
pub(crate) fn build_any_windows(
    first: &AnyParsedRangePredicate<'_>,
    second: &AnyParsedRangePredicate<'_>,
) -> Result<SingleJoinResult, String> {
    intersect_windows(first.windows()?, second.windows()?)
}

/// Parse the six-element basic range tuple.
///
/// The tuple is `(left, left_index, right, right_index,
/// right_index_is_ordered, comparator)`. Values are expected to be non-null
/// and sorted on the right side before this function is called. Rust trusts
/// that preparation. The ordering flag is part of the shared wrapper
/// contract, not an aggregation input: its value is validated but not used
/// to choose an aggregation algorithm. Arbitrary-window `first`/`last`
/// selection uses range extrema instead of prefix/suffix tables.
///
/// # Errors
///
/// Returns `ValueError` for the wrong tuple length or an invalid comparator,
/// and propagates Python extraction errors for incompatible array dtypes.
///
/// # Arguments
///
/// * `tuple` - The six-element Python tuple at the Rust boundary. The first
///   four fields are aligned value/label arrays, field four is the shared
///   ordering flag (validated but ignored by aggregation), and the final
///   field is the string comparator.
///
/// # Returns
///
/// A parsed predicate that owns borrowed Python array handles for the duration
/// of window construction. The ordering flag is intentionally not stored:
/// arbitrary dual-range selection uses exact interval extrema.
pub(crate) fn parse_range_predicate<'py, T: numpy::Element>(
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
    // The ordering flag is part of the shared predicate tuple contract.
    // Aggregation does not use its value: PyJanitor has already sorted the
    // right-hand arrays before calling Rust. Extract it only to validate the
    // tuple shape and field type.
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

/// Parse one of the two range anchors for the range-led extended API.
///
/// Extended joins do not use `right_index_is_ordered`: residual filtering can
/// remove arbitrary candidates before `first`/`last` selection. Their anchor
/// tuples therefore contain only `(left, left_index, right, right_index, op)`.
/// The first two predicates still must have ascending right value arrays;
/// additional predicates are parsed and applied as residual filters after the
/// two windows have been intersected.
///
/// # Arguments
///
/// * `tuple` - A five-element tuple containing left values, left labels, right
///   values, right labels, and a range comparator. This internal extended
///   form is distinct from the six-element public range tuple because the
///   extended path does not use the ordering flag.
///
/// # Returns
///
/// A borrowed [`ParsedRangePredicate`] retaining the Python array owners
/// while the range windows are built.
///
/// # Errors
///
/// Returns `ValueError` for the wrong tuple length or an invalid comparator,
/// and propagates Python extraction errors for incompatible array dtypes.
///
/// # Returns
///
/// A parsed range predicate whose arrays remain borrowed from the Python tuple
/// while the dual-range extended operation constructs and filters windows.
pub(crate) fn parse_extended_range_predicate<'py, T: numpy::Element>(
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

/// Build indices for a dual-range join from two per-row windows.
///
/// Each anchor produces one positional window for every logical left row. The
/// windows are intersected row by row; index generation is therefore defined
/// by the window intersection, not by the value dtype of either anchor.
#[pyfunction]
pub fn range_join_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    return_building_blocks: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range join requires exactly two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let second_item = predicates.get_item(1)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_any_range_predicate(first_tuple, false)?;
    let second = parse_any_range_predicate(second_tuple, false)?;
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    if return_building_blocks {
        return Ok(Some(result_dict(
            py,
            windows.left_index,
            windows.right_index,
            Some(windows.starts),
            Some(windows.ends),
        )?));
    }
    let keep = Keep::parse(keep)?;
    let (left, right) = choose_range_windows(&windows, keep).map_err(PyValueError::new_err)?;
    if left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, left, right, None, None)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(range_join_indices, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices, m)?)?;
    Ok(())
}

/// Execute the range-extended join with independently typed anchors.
fn extended_join<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first: AnyParsedRangePredicate<'py>,
    second: AnyParsedRangePredicate<'py>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(2) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    ensure_equal_lengths_core(
        "first left predicate array",
        first.left_len(),
        "second left predicate array",
        second.left_len(),
    )
    .map_err(PyValueError::new_err)?;
    ensure_equal_lengths_core(
        "first right predicate array",
        first.right_len(),
        "second right predicate array",
        second.right_len(),
    )
    .map_err(PyValueError::new_err)?;
    for predicate in &parsed {
        ensure_equal_lengths_core(
            "first left predicate array",
            first.left_len(),
            "residual left predicate array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core(
            "first right predicate array",
            first.right_len(),
            "residual right predicate array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    let (out_left, out_right) =
        materialize_range_candidates(&windows, &parsed, metadata.as_deref(), keep)
            .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, out_left, out_right, None, None)?))
}

/// Build range-extended indices from two anchor windows and residual filters.
#[pyfunction]
pub fn range_join_extended_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range extended join requires at least two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let second_item = predicates.get_item(1)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    if first_tuple.len() != 5 || second_tuple.len() != 5 {
        return Err(PyValueError::new_err(
            "extended range anchors must contain 5 elements",
        ));
    }
    let first = parse_any_range_predicate(first_tuple, true)?;
    let second = parse_any_range_predicate(second_tuple, true)?;
    extended_join(py, predicates, keep, first, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array1, PyArray1, PyArrayMethods};

    fn read_pair<'py>(result: &Bound<'py, PyDict>) -> (Vec<i64>, Vec<i64>) {
        let left = result
            .get_item("left_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        let right = result
            .get_item("right_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        (left, right)
    }

    #[test]
    fn two_range_windows_intersect_in_ascending_layout() {
        let left = Array1::from_vec(vec![2_i64, 5]);
        let left_second = Array1::from_vec(vec![8_i64, 9]);
        let right = Array1::from_vec(vec![1_i64, 2, 3, 4, 5, 6, 7, 8]);
        let right_second = Array1::from_vec(vec![0_i64, 1, 2, 3, 4, 5, 6, 7]);
        let left_index = Array1::from_vec(vec![10_i64, 11]);
        let right_index = Array1::from_vec(vec![20_i64, 21, 22, 23, 24, 25, 26, 27]);

        let result = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: left_index.view(),
                right: right.view(),
                right_index: right_index.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: left_index.view(),
                right: right_second.view(),
                right_index: right_index.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        assert_eq!(result.left_index, vec![10, 11]);
        assert_eq!(result.starts, vec![2, 5]);
        assert_eq!(result.ends, vec![8, 8]);
    }

    #[test]
    fn two_range_windows_drop_empty_intersections() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_second = Array1::from_vec(vec![1_i64]);
        let right = Array1::from_vec(vec![1_i64, 2, 3]);
        let right_second = Array1::from_vec(vec![1_i64, 2, 3]);
        let labels = Array1::from_vec(vec![10_i64, 11, 12]);
        let one_left_index = Array1::from_vec(vec![0_i64]);

        let result = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: one_left_index.view(),
                right: right.view(),
                right_index: labels.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: one_left_index.view(),
                right: right_second.view(),
                right_index: labels.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        assert!(result.left_index.is_empty());
        assert!(result.starts.is_empty());
        assert!(result.ends.is_empty());
    }

    #[test]
    fn unordered_labels_are_selected_from_intersected_suffix_windows() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_second = Array1::from_vec(vec![6_i64]);
        let right = Array1::from_vec(vec![1_i64, 3, 5, 7]);
        let right_second = Array1::from_vec(vec![0_i64, 2, 4, 6]);
        let left_index = Array1::from_vec(vec![100_i64]);
        let right_index = Array1::from_vec(vec![40_i64, 10, 30, 20]);

        let windows = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: left_index.view(),
                right: right.view(),
                right_index: right_index.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: left_index.view(),
                right: right_second.view(),
                right_index: right_index.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        // P1 gives [1, 4), P2 gives [0, 3), so the intersection is [1, 3).
        // The labels in that physical window are [10, 30], even though the
        // complete right-index array [40, 10, 30, 20] is unordered.
        assert_eq!(windows.starts, vec![1]);
        assert_eq!(windows.ends, vec![3]);
        assert_eq!(
            choose_range_windows(&windows, Keep::First).unwrap(),
            (vec![100], vec![10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last).unwrap(),
            (vec![100], vec![30])
        );
    }

    #[test]
    fn arbitrary_windows_use_range_extrema_not_prefix_or_suffix_extrema() {
        let windows = SingleJoinResult {
            left_positions: vec![0, 1],
            left_index: vec![100, 101],
            right_index: vec![40, 10, 30, 20, 50],
            starts: vec![2, 1],
            ends: vec![4, 4],
        };

        // The first window is [2, 4) = [30, 20], and the second is
        // [1, 4) = [10, 30, 20]. Neither is a prefix or suffix. The
        // smallest and largest labels must be selected within each exact
        // interval, not from the surrounding right-index array.
        assert_eq!(
            choose_range_windows(&windows, Keep::First).unwrap(),
            (vec![100, 101], vec![20, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last).unwrap(),
            (vec![100, 101], vec![30, 30])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Any).unwrap(),
            (vec![100, 101], vec![30, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::All).unwrap(),
            (vec![100, 100, 101, 101, 101], vec![30, 20, 10, 30, 20])
        );
    }

    #[test]
    fn range_extended_entry_point_filters_residuals_after_two_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1, 5, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result = range_join_extended_indices(py, &predicates, "all")?
                .expect("the residual predicate should leave one candidate");
            assert_eq!(read_pair(&result), (vec![100], vec![30]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn mixed_range_anchors_search_with_their_own_dtypes() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6.0_f64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0.0_f64, 2.0, 4.0, 6.0]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result = range_join_indices(py, &predicates, "all", false)?
                .expect("the mixed-dtype windows should intersect");
            assert_eq!(read_pair(&result), (vec![100, 100], vec![10, 30]));
            Ok(())
        })
        .unwrap();
    }
}
