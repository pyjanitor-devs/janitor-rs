//! Shared range-join window construction.
//!
//! A basic range join has two range predicates. This module computes the
//! half-open `starts`/`ends` windows for those predicates. Extended joins
//! reuse the same primitive and perform their additional filtering elsewhere.
//!
//! PyJanitor performs all pandas-specific preparation before calling this
//! module: null removal, dtype selection, right-value sorting, and physical
//! position tracking. Consequently, the public PyO3 functions receive NumPy
//! arrays rather than pandas labels. A range anchor is a three-element tuple
//! `(left_values, right_values, operator)` and the shared physical maps are
//! explicit function arguments. Window offsets always address the sorted
//! right-value layout; returned right positions are obtained by indexing the
//! supplied `right_index` map.
//!
//! Aggregation functions use the same compact aligned value layout. Forward
//! aggregation writes to left output slots and consumes right source arrays;
//! reverse aggregation writes to right output slots and consumes left source
//! arrays. Residual predicates are evaluated only after the two anchor
//! windows have been intersected.

use numpy::ndarray::Array1;
#[cfg(test)]
use numpy::ndarray::ArrayView1;
use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggregation_common::ensure_equal_lengths_core;
#[cfg(test)]
use crate::compare_op::CompareOp;
use crate::join_aggregation_helpers::aggregate_range_windows;
use crate::join_candidate_materialization::materialize_range_candidates;
#[cfg(test)]
use crate::join_search::range_window;
use crate::join_types::{result_dict, Keep, SingleJoinResult};
use crate::predicate::{check_predicate_lengths, parse_predicates_with_nulls_strings};
#[cfg(test)]
use crate::range_predicate::parse_any_range_predicate;
use crate::range_predicate::{parse_any_range_parts, AnyParsedRangePredicate};

fn window_extreme_positions(
    labels: &[i64],
    starts: &[i64],
    ends: &[i64],
    minimum: bool,
) -> Result<Vec<i64>, String> {
    ensure_equal_lengths_core("range labels", starts.len(), "range starts", ends.len())?;
    let mut selected = Vec::with_capacity(starts.len());
    for (&start, &end) in starts.iter().zip(ends) {
        let start = usize::try_from(start).map_err(|_| "range start is negative".to_owned())?;
        let end = usize::try_from(end).map_err(|_| "range end is negative".to_owned())?;
        if start >= end || end > labels.len() {
            return Err("range window is empty or out of bounds".to_owned());
        }
        let mut best = start;
        for position in (start + 1)..end {
            let better = if minimum {
                labels[position] < labels[best]
            } else {
                labels[position] > labels[best]
            };
            if better {
                best = position;
            }
        }
        selected.push(i64::try_from(best).map_err(|_| "range position exceeds int64".to_owned())?);
    }
    Ok(selected)
}

/// Build the monotonic envelope used to bound an opposing range predicate.
///
/// PyJanitor sorts the right side by the primary lower-bound predicate. The
/// corresponding upper bounds are not necessarily ordered in that layout.
/// A forward cumulative maximum (or reverse cumulative minimum) is a safe
/// superset boundary; the original predicate is still evaluated by the
/// extended candidate materializer.
fn cumulative_bound<T: PartialOrd + Copy>(values: &[T], reverse_min: bool) -> Vec<T> {
    let mut output = values.to_vec();
    if reverse_min {
        // ELI5: scan from the right and carry the smallest value seen so far.
        // Every suffix therefore has one monotonic boundary, even when the
        // original endpoint values jump up and down.
        for position in (0..output.len().saturating_sub(1)).rev() {
            if output[position + 1] < output[position] {
                output[position] = output[position + 1];
            }
        }
    } else {
        // ELI5: scan from the left and carry the largest value seen so far.
        // This makes each prefix boundary safe for binary search: a real
        // match can be added to a window, but never accidentally excluded.
        for position in 1..output.len() {
            if output[position - 1] > output[position] {
                output[position] = output[position - 1];
            }
        }
    }
    output
}

/// Return a cumulative range bound for a right-hand array.
///
/// `direction` is `"max"` for a forward cumulative maximum and
/// `"reverse_min"` for a reverse cumulative minimum. This keeps the dtype
/// operation in Rust while Python retains ownership of the original array
/// for the exact residual recheck.
#[pyfunction]
pub fn range_join_cumulative_bound<'py>(
    py: Python<'py>,
    values: Bound<'py, PyAny>,
    direction: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // The direction is deliberately a small string at the Python boundary,
    // while the loop below stays generic and typed. Keeping the operation in
    // Rust avoids a dtype-changing pandas/NumPy round trip for every join.
    let reverse_min = match direction {
        "max" => false,
        "reverse_min" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown cumulative range direction: {other}"
            )))
        }
    };
    let dtype = values
        .getattr("dtype")?
        .getattr("name")?
        .extract::<String>()?;
    macro_rules! dispatch {
        ($ty:ty) => {{
            // PyO3 extraction verifies that the array really has this dtype;
            // silently coercing here could change integer overflow behavior
            // or the ordering semantics used by the later binary search.
            let values = values.extract::<PyReadonlyArray1<'py, $ty>>()?;
            let output = cumulative_bound(values.as_slice()?, reverse_min);
            Ok(PyArray1::from_vec(py, output).into_any())
        }};
    }
    match dtype.as_str() {
        "int64" => dispatch!(i64),
        "int32" => dispatch!(i32),
        "int16" => dispatch!(i16),
        "int8" => dispatch!(i8),
        "uint64" => dispatch!(u64),
        "uint32" => dispatch!(u32),
        "uint16" => dispatch!(u16),
        "uint8" => dispatch!(u8),
        "float64" => dispatch!(f64),
        "float32" => dispatch!(f32),
        other => Err(PyValueError::new_err(format!(
            "unsupported cumulative range dtype: {other}"
        ))),
    }
}

/// Parse one basic dual-range anchor using shared physical index arrays.
///
/// Each anchor carries only `(left_values, right_values, operator)`. The
/// shared `left_index` and `right_index` are supplied once by
/// [`range_join_indices`], preventing the two anchors from carrying
/// inconsistent physical layouts.
fn parse_shared_index_range_predicate<'py>(
    tuple: &Bound<'py, PyTuple>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
) -> PyResult<AnyParsedRangePredicate<'py>> {
    if tuple.len() != 3 {
        return Err(PyValueError::new_err(
            "shared-index range predicates must contain 3 elements",
        ));
    }
    parse_any_range_parts(
        &tuple.get_item(0)?,
        left_index,
        &tuple.get_item(1)?,
        right_index,
        &tuple.get_item(2)?,
    )
}

/// A typed range predicate used by the basic two-range kernel.
///
/// The value arrays describe the comparison layout, while the paired index
/// arrays carry the labels that must be returned to Python. PyJanitor owns
/// sorting and alignment before constructing this value; Rust only consumes
/// the already-aligned views.
#[cfg(test)]
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
    fn build_test_window<T: PartialOrd + Copy>(
        predicate: RangePredicate<'_, T>,
    ) -> Result<SingleJoinResult, String> {
        ensure_equal_lengths_core(
            "left test values",
            predicate.left.len(),
            "left test labels",
            predicate.left_index.len(),
        )?;
        ensure_equal_lengths_core(
            "right test values",
            predicate.right.len(),
            "right test labels",
            predicate.right_index.len(),
        )?;
        let mut result = SingleJoinResult {
            left_positions: Vec::with_capacity(predicate.left.len()),
            left_index: predicate.left_index.to_vec(),
            right_index: predicate.right_index.to_vec(),
            starts: Vec::with_capacity(predicate.left.len()),
            ends: Vec::with_capacity(predicate.left.len()),
        };
        for (position, &left_value) in predicate.left.iter().enumerate() {
            let (start, end) = range_window(left_value, predicate.right, predicate.op);
            result.left_positions.push(position);
            result.starts.push(start);
            result.ends.push(end);
        }
        Ok(result)
    }

    let first = build_test_window(first)?;
    let second = build_test_window(second)?;
    intersect_windows(first, second)
}

/// Intersect two already-built positional windows.
///
/// The value dtypes are intentionally absent here. Each anchor has already
/// completed its own typed binary searches; this step only combines the
/// resulting positions. PyJanitor aligns the second right array to the first
/// right layout before calling Rust, so the two windows refer to the same
/// physical right positions even when their value dtypes differ.
///
/// The first anchor owns the right-label vector retained for output. The
/// second anchor deliberately does not carry a second copy of those labels.
/// Consequently, this function checks positional bounds but does not compare
/// every label element. A same-length, differently ordered second right layout
/// is outside the Rust contract and must be prevented by PyJanitor's alignment
/// step. This keeps the dual-range path from paying an O(right_len) defensive
/// scan on every call.
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
    if first.left_positions.len() != second.left_positions.len() {
        return Err(
            "dual range predicates must use aligned left rows and right positions".to_owned(),
        );
    }
    // The second anchor's right labels are deliberately not copied: PyJanitor
    // guarantees that both right value arrays use the same physical layout.
    // Boundary validation still catches a second array whose windows reach
    // beyond the first anchor's right domain.
    if second.ends.iter().any(|&end| end > first.right_index.len()) {
        return Err("dual range predicates must use aligned right positions".to_owned());
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

/// Build dual-range windows while dispatching each anchor independently.
pub(crate) fn build_any_windows(
    first: &AnyParsedRangePredicate<'_>,
    second: &AnyParsedRangePredicate<'_>,
) -> Result<SingleJoinResult, String> {
    intersect_windows(first.windows(true)?, second.windows(false)?)
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
    right_index_is_ordered: bool,
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

    if right_index_is_ordered {
        let mut output_left = Vec::with_capacity(windows.left_index.len());
        let mut output_right = Vec::with_capacity(windows.left_index.len());
        for (row, (&start, &end)) in windows.starts.iter().zip(&windows.ends).enumerate() {
            let position = if keep == Keep::First { start } else { end - 1 };
            output_left.push(windows.left_index[row]);
            output_right.push(labels[position]);
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
    // Range inputs have already had null rows removed by PyJanitor. The
    // no-null RMQ entry points avoid allocating a full all-false mask solely
    // to satisfy the general nullable-array API. They return offsets into
    // `labels`, not public labels; the final loop performs that conversion.
    let selected_positions = match keep {
        Keep::First => window_extreme_positions(
            labels,
            starts.as_slice().unwrap(),
            ends.as_slice().unwrap(),
            true,
        )?,
        Keep::Last => window_extreme_positions(
            labels,
            starts.as_slice().unwrap(),
            ends.as_slice().unwrap(),
            false,
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

/// Build indices for a dual-range join from two per-row windows.
///
/// Each anchor produces one positional window for every logical left row. The
/// windows are intersected row by row; index generation is therefore defined
/// by the window intersection, not by the value dtype of either anchor.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to construct the result
///   dictionary.
/// * `predicates` - Exactly two three-element tuples of
///   `(left_values, right_values, operator)`. The value arrays in both tuples
///   must share the same left layout and the same sorted right layout.
/// * `left_index` - Physical positions paired with every left value. These
///   are original dataframe positions, not compact offsets.
/// * `right_index` - Physical positions paired with every sorted right value.
///   Window offsets are translated through this array before being returned.
/// * `right_index_is_ordered` - Whether the physical right positions are
///   ordered. This enables direct `first`/`last` selection; extrema kernels
///   are used otherwise.
/// * `keep` - Selection policy: `all`, `any`, `first`, or `last`.
/// * `return_building_blocks` - Return the physical maps plus `starts` and
///   `ends` instead of materialized pairs when true.
///
/// # Returns
///
/// Returns `None` when the intersected windows contain no candidate. Otherwise
/// returns a dictionary containing `left_index` and `right_index`, and, when
/// requested, the half-open window arrays.
#[pyfunction]
pub fn range_join_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: Bound<'py, PyAny>,
    right_index: Bound<'py, PyAny>,
    right_index_is_ordered: bool,
    keep: &str,
    return_building_blocks: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // Python has already removed nulls, sorted the primary right anchor, and
    // aligned every second anchor to that physical layout. Rust's job here is
    // only to build/intersect positional windows; it must not sort again or
    // confuse a search offset with an original dataframe position.
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range join requires exactly two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let second_item = predicates.get_item(1)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_shared_index_range_predicate(first_tuple, &left_index, &right_index)?;
    let second = parse_shared_index_range_predicate(second_tuple, &left_index, &right_index)?;
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
    let (left, right) = choose_range_windows(&windows, keep, right_index_is_ordered)
        .map_err(PyValueError::new_err)?;
    if left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, left, right, None, None)?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        range_join_cumulative_bound,
        range_join_indices,
        range_join_extended_indices,
        range_join_aggregate,
        range_join_aggregate_reverse,
        range_join_extended_aggregate,
        range_join_extended_aggregate_reverse,
    );
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
    // The first two predicates define the bounded candidate region. All later
    // predicates—including the original non-monotonic range predicate paired
    // with a cumulative envelope—are exact filters over that region.
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
    check_predicate_lengths(&parsed, first.left_len(), first.right_len())?;
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
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to construct the result.
/// * `predicates` - At least two tuples. The first two are three-element
///   `(left_values, right_values, operator)` range anchors; later tuples are
///   residual predicates parsed by the shared predicate machinery.
/// * `left_index` - Physical positions paired with every left anchor value.
/// * `right_index` - Physical positions paired with every sorted right value.
/// * `keep` - Selection policy applied after both anchors and all residuals
///   pass.
///
/// # Returns
///
/// Returns materialized physical pairs, or `None` when no candidate survives.
#[pyfunction]
pub fn range_join_extended_indices<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: Bound<'py, PyAny>,
    right_index: Bound<'py, PyAny>,
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
    if first_tuple.len() != 3 || second_tuple.len() != 3 {
        return Err(PyValueError::new_err(
            "extended range anchors must contain 3 elements",
        ));
    }
    let first = parse_shared_index_range_predicate(first_tuple, &left_index, &right_index)?;
    let second = parse_shared_index_range_predicate(second_tuple, &left_index, &right_index)?;
    extended_join(py, predicates, keep, first, second)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array1, PyArray1, PyArrayMethods};

    #[test]
    fn cumulative_bounds_are_safe_monotonic_envelopes() {
        let values = [8_i64, 2, 5, 3];

        assert_eq!(cumulative_bound(&values, false), vec![8, 8, 8, 8]);
        assert_eq!(cumulative_bound(&values, true), vec![2, 2, 3, 3]);
    }

    /// Select reference positions using the same public `keep` semantics as
    /// the production wrapper.
    ///
    /// Keeping this policy in one helper is important for test maintenance:
    /// the brute-force oracle should disagree with the implementation only
    /// when the implementation is wrong, not because one test copied a
    /// slightly different `first`/`last` tie-break rule. `first` and `last`
    /// are label-based, while `any` deliberately keeps the first physical
    /// candidate, matching the crate-wide join convention.
    fn select_reference_positions(
        passing: Vec<usize>,
        right_index: &[i64],
        keep: &str,
    ) -> Vec<usize> {
        match keep {
            "all" => passing,
            "any" => passing.into_iter().take(1).collect(),
            "first" | "last" => passing
                .into_iter()
                .min_by_key(|&position| {
                    if keep == "first" {
                        right_index[position]
                    } else {
                        -right_index[position]
                    }
                })
                .into_iter()
                .collect(),
            other => panic!("unexpected reference keep: {other}"),
        }
    }

    /// Advance the deterministic generator used by the randomized oracle.
    ///
    /// This is intentionally tiny and local to tests. It avoids a dependency
    /// on a random-number crate while giving future tests one named utility
    /// instead of each test inventing its own state transition.
    fn next_test_value(seed: &mut u64) -> i64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((*seed >> 32) % 9) as i64 - 4
    }

    struct ReferenceFixture<'a> {
        left_index: &'a [i64],
        first_left: &'a [i64],
        second_left: &'a [i64],
        right_index: &'a [i64],
        first_right: &'a [i64],
        second_right: &'a [i64],
    }

    /// The common i64 shape used by the public index tests.
    ///
    /// The production API is intentionally tuple-based for Python
    /// compatibility. Tests should not repeat the eight tuple-field
    /// positions everywhere, though: a field-order mistake in a test can make
    /// a regression look like a kernel failure. This small builder keeps the
    /// tuple layout in one documented place while malformed-tuple tests still
    /// construct their bad inputs explicitly.
    #[allow(dead_code)]
    struct IndexPredicate<'a> {
        left: &'a [i64],
        left_index: &'a [i64],
        right: &'a [i64],
        right_index: &'a [i64],
        ordered: bool,
        operator: &'a str,
    }

    fn append_index_predicate<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        predicate: IndexPredicate<'_>,
    ) -> PyResult<()> {
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, predicate.left.to_vec()).into_any(),
                PyArray1::from_vec(py, predicate.right.to_vec()).into_any(),
                predicate.operator.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        Ok(())
    }

    fn call_range_join_indices<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &[i64],
        right_index: &[i64],
        keep: &str,
        return_building_blocks: bool,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let left_index = PyArray1::from_vec(py, left_index.to_vec()).into_any();
        let right_index = PyArray1::from_vec(py, right_index.to_vec()).into_any();
        let right_index_is_ordered = right_index
            .cast::<PyArray1<i64>>()
            .map(|array| {
                array
                    .readonly()
                    .as_array()
                    .windows(2)
                    .into_iter()
                    .all(|pair| pair[0] <= pair[1])
            })
            .unwrap_or(false);
        range_join_indices(
            py,
            predicates,
            left_index,
            right_index,
            right_index_is_ordered,
            keep,
            return_building_blocks,
        )
    }

    fn call_range_join_extended_indices<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &[i64],
        right_index: &[i64],
        keep: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let left_index = PyArray1::from_vec(py, left_index.to_vec()).into_any();
        let right_index = PyArray1::from_vec(py, right_index.to_vec()).into_any();
        range_join_extended_indices(py, predicates, left_index, right_index, keep)
    }

    /// Build the public result expected from a pair-by-pair implementation.
    ///
    /// The optimized range path searches sorted values, but its public
    /// result is still defined by the original physical right labels.  This
    /// deliberately simple helper does not use windows or binary search: it
    /// is an independent oracle for the wrapper tests.
    fn reference_pairs(
        fixture: &ReferenceFixture<'_>,
        first_operator: &str,
        second_operator: &str,
        keep: &str,
    ) -> (Vec<i64>, Vec<i64>) {
        let mut output_left = Vec::new();
        let mut output_right = Vec::new();
        let first_op = CompareOp::try_from_str(first_operator).unwrap();
        let second_op = CompareOp::try_from_str(second_operator).unwrap();
        for left_position in 0..fixture.left_index.len() {
            let mut passing = Vec::new();
            for right_position in 0..fixture.right_index.len() {
                if first_op.apply(
                    &fixture.first_left[left_position],
                    &fixture.first_right[right_position],
                ) && second_op.apply(
                    &fixture.second_left[left_position],
                    &fixture.second_right[right_position],
                ) {
                    passing.push(right_position);
                }
            }
            for right_position in select_reference_positions(passing, fixture.right_index, keep) {
                output_left.push(fixture.left_index[left_position]);
                output_right.push(fixture.right_index[right_position]);
            }
        }
        (output_left, output_right)
    }

    fn assert_index_result<'py>(
        result: Option<Bound<'py, PyDict>>,
        expected: (Vec<i64>, Vec<i64>),
    ) {
        if expected.0.is_empty() {
            assert!(result.is_none(), "expected no matching pairs");
        } else {
            let result = result.expect("expected matching pairs");
            assert_eq!(read_pair(&result), expected);
        }
    }

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
    fn parsed_windows_dispatch_all_operators_and_boundary_cases() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = PyArray1::from_vec(py, vec![2_i64]);
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21, 22]);

            // `windows()` is the dtype-dispatch boundary used by the public
            // dual-range paths. These expected half-open intervals document
            // the binary-search convention for every supported operator:
            // strict/inclusive less-than creates a suffix, while
            // strict/inclusive greater-than creates a prefix.
            for (operator, expected_start, expected_end) in
                [("<", 2, 3), ("<=", 1, 3), (">", 0, 1), (">=", 0, 2)]
            {
                let predicate = PyTuple::new(
                    py,
                    [
                        left.clone().into_any(),
                        left_index.clone().into_any(),
                        right.clone().into_any(),
                        right_index.clone().into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?;
                let parsed = parse_any_range_predicate(&predicate, false)?;
                let windows = parsed.windows(true).map_err(PyValueError::new_err)?;
                assert_eq!(windows.left_positions, vec![0], "operator {operator}");
                assert_eq!(windows.starts, vec![expected_start], "operator {operator}");
                assert_eq!(windows.ends, vec![expected_end], "operator {operator}");
                assert_eq!(windows.right_index, vec![20, 21, 22]);
            }

            // `retain_empty_windows` is intentional: alignment needs one
            // window per logical left row, even when that row has no match.
            // Check both the full suffix/prefix and empty suffix/prefix
            // boundaries, which are the off-by-one cases most likely to be
            // damaged by a dispatch refactor.
            let boundary_left = PyArray1::from_vec(py, vec![0_i64, 4]);
            let boundary_left_index = PyArray1::from_vec(py, vec![30_i64, 31]);
            for (operator, expected_starts, expected_ends) in [
                ("<", vec![0, 3], vec![3, 3]),
                (">=", vec![0, 0], vec![0, 3]),
            ] {
                let predicate = PyTuple::new(
                    py,
                    [
                        boundary_left.clone().into_any(),
                        boundary_left_index.clone().into_any(),
                        right.clone().into_any(),
                        right_index.clone().into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?;
                let parsed = parse_any_range_predicate(&predicate, false)?;
                let windows = parsed.windows(true).map_err(PyValueError::new_err)?;
                assert_eq!(windows.left_positions, vec![0, 1], "operator {operator}");
                assert_eq!(windows.starts, expected_starts, "operator {operator}");
                assert_eq!(windows.ends, expected_ends, "operator {operator}");
            }
            Ok(())
        })
        .unwrap();
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
            choose_range_windows(&windows, Keep::First, false).unwrap(),
            (vec![100], vec![10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last, false).unwrap(),
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
            choose_range_windows(&windows, Keep::First, false).unwrap(),
            (vec![100, 101], vec![20, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last, false).unwrap(),
            (vec![100, 101], vec![30, 30])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Any, false).unwrap(),
            (vec![100, 101], vec![30, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::All, false).unwrap(),
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
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
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

            let result = call_range_join_extended_indices(
                py,
                &predicates,
                &[100],
                &[40, 10, 30, 20],
                "all",
            )?
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
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6.0_f64]).into_any(),
                    PyArray1::from_vec(py, vec![0.0_f64, 2.0, 4.0, 6.0]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result =
                call_range_join_indices(py, &predicates, &[100], &[40, 10, 30, 20], "all", false)?
                    .expect("the mixed-dtype windows should intersect");
            assert_eq!(read_pair(&result), (vec![100, 100], vec![10, 30]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_range_indices_dispatch_every_anchor_dtype() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            macro_rules! check_dtype {
                ($ty:ty) => {{
                    let predicates = PyList::empty(py);
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, vec![2 as $ty]).into_any(),
                            PyArray1::from_vec(py, vec![1 as $ty, 3 as $ty, 5 as $ty]).into_any(),
                            "<".into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, vec![2 as $ty]).into_any(),
                            PyArray1::from_vec(py, vec![0 as $ty, 2 as $ty, 4 as $ty]).into_any(),
                            ">=".into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                    let result = call_range_join_indices(
                        py,
                        &predicates,
                        &[100],
                        &[10, 20, 30],
                        "all",
                        false,
                    )?
                    .expect("every supported dtype should dispatch");
                    assert_eq!(read_pair(&result), (vec![100], vec![20]));
                    Ok::<(), PyErr>(())
                }};
            }

            check_dtype!(i64)?;
            check_dtype!(i32)?;
            check_dtype!(i16)?;
            check_dtype!(i8)?;
            check_dtype!(u64)?;
            check_dtype!(u32)?;
            check_dtype!(u16)?;
            check_dtype!(u8)?;
            check_dtype!(f64)?;
            check_dtype!(f32)?;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_range_indices_handles_infinite_and_nan_anchor_values() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Infinity is an ordinary ordered endpoint: every finite value
            // is below positive infinity, and negative infinity is the first
            // value in an ascending right layout.  This checks the two
            // boundary values without relying on a finite sentinel.
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![f64::INFINITY]).into_any(),
                    PyArray1::from_vec(py, vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]).into_any(),
                    ">=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![f64::INFINITY]).into_any(),
                    PyArray1::from_vec(py, vec![f64::NEG_INFINITY, 0.0, f64::INFINITY]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let result =
                call_range_join_indices(py, &predicates, &[100], &[10, 20, 30], "all", false)?
                    .expect("infinite endpoints should produce matches");
            assert_eq!(read_pair(&result), (vec![100, 100], vec![10, 20]));

            // NaN is not part of the sorted-range contract, but it can cross
            // the Python boundary.  The binary-search kernels deliberately
            // document that exact NaN parity is unspecified; this regression
            // only requires the public path to reject neither nor panic.
            let nan_predicates = PyList::empty(py);
            for operator in ["<", ">="] {
                nan_predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![f64::NAN]).into_any(),
                        PyArray1::from_vec(py, vec![-1.0_f64, 1.0]).into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let _ = call_range_join_indices(py, &nan_predicates, &[100], &[10, 20], "all", false)?;
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn public_range_indices_match_reference_for_all_orientations_and_keeps() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let first_left = vec![0_i64, 3, 6];
            let second_left = vec![5_i64, 4, 7];
            let left_index = vec![101_i64, 99, 55];
            let first_right = vec![1_i64, 3, 3, 7];
            let second_right = vec![0_i64, 2, 4, 6];
            // The labels are intentionally not sorted by value. The right
            // values are sorted, as required by the binary-search contract,
            // while the returned labels retain this physical permutation.
            let right_index = vec![40_i64, 10, 30, 20];
            // Four representative pairs cover every operator in both anchor
            // positions, including strict/inclusive and complementary
            // directions. The randomized test below supplies additional
            // pairings. Keeping the deterministic matrix at 4 x 4 instead
            // of 4 x 4 x 4 makes failures easier to attribute while still
            // testing every keep mode against every selected pair.
            let operator_pairs = [("<", "<="), ("<=", ">"), (">", "<"), (">=", ">=")];
            let keeps = ["all", "first", "last", "any"];

            for (first_operator, second_operator) in operator_pairs {
                let predicates = PyList::empty(py);
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &first_left,
                        left_index: &left_index,
                        right: &first_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: first_operator,
                    },
                )?;
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &second_left,
                        left_index: &left_index,
                        right: &second_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: second_operator,
                    },
                )?;
                for keep in keeps {
                    let expected = reference_pairs(
                        &ReferenceFixture {
                            left_index: &left_index,
                            first_left: &first_left,
                            second_left: &second_left,
                            right_index: &right_index,
                            first_right: &first_right,
                            second_right: &second_right,
                        },
                        first_operator,
                        second_operator,
                        keep,
                    );
                    let result = call_range_join_indices(
                        py,
                        &predicates,
                        &left_index,
                        &right_index,
                        keep,
                        false,
                    )?;
                    assert_index_result(result, expected);
                }
            }

            // Building blocks are the unmaterialized public form. They retain
            // one row per surviving left window and expose positional bounds
            // rather than selected labels.
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64]).into_any(),
                    PyArray1::from_vec(py, first_right).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, second_right).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let blocks =
                call_range_join_indices(py, &predicates, &[101], &right_index, "all", true)?
                    .expect("the building-block windows should be non-empty");
            assert_eq!(
                blocks.get_item("starts")?.unwrap().extract::<Vec<i64>>()?,
                vec![1]
            );
            assert_eq!(
                blocks.get_item("ends")?.unwrap().extract::<Vec<i64>>()?,
                vec![2]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn randomized_range_indices_match_bruteforce_reference() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let operators = ["<", "<=", ">", ">="];
            let keeps = ["all", "first", "last", "any"];
            let left_index = vec![100_i64, 101, 102, 103];
            let right_index = vec![30_i64, 10, 40, 20, 50];
            let mut seed = 0x9e3779b97f4a7c15_u64;

            for case in 0..32 {
                let mut first_right = (0..5)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                let mut second_right = (0..5)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                first_right.sort_unstable();
                second_right.sort_unstable();
                let first_left = (0..4)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                let second_left = (0..4)
                    .map(|_| next_test_value(&mut seed))
                    .collect::<Vec<_>>();
                let first_operator = operators[case % operators.len()];
                let second_operator = operators[(case * 3 + 1) % operators.len()];
                let keep = keeps[case % keeps.len()];
                let predicates = PyList::empty(py);
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &first_left,
                        left_index: &left_index,
                        right: &first_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: first_operator,
                    },
                )?;
                append_index_predicate(
                    py,
                    &predicates,
                    IndexPredicate {
                        left: &second_left,
                        left_index: &left_index,
                        right: &second_right,
                        right_index: &right_index,
                        ordered: false,
                        operator: second_operator,
                    },
                )?;
                let expected = reference_pairs(
                    &ReferenceFixture {
                        left_index: &left_index,
                        first_left: &first_left,
                        second_left: &second_left,
                        right_index: &right_index,
                        first_right: &first_right,
                        second_right: &second_right,
                    },
                    first_operator,
                    second_operator,
                    keep,
                );
                assert_index_result(
                    call_range_join_indices(
                        py,
                        &predicates,
                        &left_index,
                        &right_index,
                        keep,
                        false,
                    )?,
                    expected,
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_range_indices_filter_reference_candidates_before_keep() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let first_left = vec![0_i64, 3, 6];
            let second_left = vec![5_i64, 4, 7];
            let left_index = vec![101_i64, 99, 55];
            let first_right = vec![1_i64, 3, 3, 7];
            let second_right = vec![0_i64, 2, 4, 6];
            let right_index = vec![40_i64, 10, 30, 20];
            let residual_left = vec![0_i64, 3, 6];
            let residual_right = vec![0_i64, 1, 5, 8];
            let residual_operator = "<";
            let first_op = CompareOp::try_from_str("<").unwrap();
            let second_op = CompareOp::try_from_str(">").unwrap();
            let residual_op = CompareOp::try_from_str(residual_operator).unwrap();

            for keep in ["all", "first", "last", "any"] {
                let predicates = PyList::empty(py);
                for (left, right, operator) in [
                    (&first_left, &first_right, "<"),
                    (&second_left, &second_right, ">"),
                ] {
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, left.clone()).into_any(),
                            PyArray1::from_vec(py, right.clone()).into_any(),
                            operator.into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                }
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, residual_left.clone()).into_any(),
                        PyArray1::from_vec(py, residual_right.clone()).into_any(),
                        residual_operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;

                let mut expected = (Vec::new(), Vec::new());
                for left_position in 0..left_index.len() {
                    let mut passing = Vec::new();
                    for right_position in 0..right_index.len() {
                        if first_op.apply(&first_left[left_position], &first_right[right_position])
                            && second_op
                                .apply(&second_left[left_position], &second_right[right_position])
                            && residual_op.apply(
                                &residual_left[left_position],
                                &residual_right[right_position],
                            )
                        {
                            passing.push(right_position);
                        }
                    }
                    for position in select_reference_positions(passing, &right_index, keep) {
                        expected.0.push(left_index[left_position]);
                        expected.1.push(right_index[position]);
                    }
                }
                assert_index_result(
                    call_range_join_extended_indices(
                        py,
                        &predicates,
                        &left_index,
                        &right_index,
                        keep,
                    )?,
                    expected,
                );
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_index_wrappers_reject_malformed_tuples_and_parallel_lengths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let malformed = PyList::empty(py);
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                ],
            )?)?;
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error =
                call_range_join_indices(py, &malformed, &[0], &[1], "all", false).unwrap_err();
            assert!(error
                .to_string()
                .contains("shared-index range predicates must contain 3 elements"));

            let mismatched = PyList::empty(py);
            for (_left_index, operator) in [(vec![0_i64, 1], "<"), (vec![0_i64], ">")] {
                mismatched.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let error =
                call_range_join_indices(py, &mismatched, &[0, 1], &[1], "all", false).unwrap_err();
            assert!(error.to_string().contains("left and left_index"));

            let extended_malformed = PyList::empty(py);
            extended_malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                ],
            )?)?;
            extended_malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error =
                call_range_join_extended_indices(py, &extended_malformed, &[0], &[1], "all")
                    .unwrap_err();
            assert!(error
                .to_string()
                .contains("extended range anchors must contain 3 elements"));

            let empty = PyList::empty(py);
            for operator in ["<", ">"] {
                empty.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            assert!(call_range_join_indices(py, &empty, &[], &[10, 11], "all", false)?.is_none());
            Ok(())
        })
        .unwrap();
    }
}
/// Build and aggregate the per-row windows for a dual-range join.
///
/// Each anchor performs its own typed boundary search. The resulting positional
/// windows are dtype-independent: they are intersected row by row, and the
/// aggregation helper consumes only the surviving windows. With no residual
/// predicates this is the basic two-range operation; with later predicates it
/// is the extended operation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_range_extended<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range extended aggregation requires at least two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_shared_index_range_predicate(first_tuple, left_index, right_index)?;
    let second = parse_shared_index_range_predicate(second_tuple, left_index, right_index)?;
    let (parsed, metadata) =
        crate::join_aggregation_helpers::residuals(py, predicates, false, true)?;
    crate::predicate::check_predicate_lengths(&parsed, first.left_len(), first.right_len())?;
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }

    let left_index = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let output_positions = if reverse {
        Some(right_index.as_array())
    } else {
        Some(left_index.as_array())
    };
    let output_len = if reverse {
        first.right_len()
    } else {
        first.left_len()
    };
    let source_len = if reverse {
        first.left_len()
    } else {
        first.right_len()
    };
    aggregate_range_windows(
        py,
        windows,
        &parsed,
        metadata.as_deref(),
        aggregations,
        output_positions,
        output_len,
        source_len,
        return_matched,
        reverse,
        false,
    )
}

/// Aggregate exactly two range predicates in the forward direction.
///
/// The first two predicates each produce a half-open window over their own
/// sorted right-hand value layout. The Rust range kernel intersects those
/// windows row by row, then aggregates every pair in the intersection. No
/// residual predicates are accepted by this entry point; callers that have
/// additional predicates must use [`range_join_extended_aggregate`].
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two three-element tuples of
///   `(left_values, right_values, operator)` using the shared physical maps.
/// * `left_index` - Physical positions paired with the left anchor values.
/// * `right_index` - Physical positions paired with the sorted right anchor
///   values.
/// * `aggregations` - Non-empty aggregation requests over aligned source
///   arrays. Their offsets match the compact value layouts in `predicates`.
/// * `return_matched` - Include the boolean match array in the returned
///   tuple. It does not change aggregation semantics.
///
/// # Returns
///
/// Returns the standard aggregation tuple, or `None` when the two windows have
/// no intersection. The output remains aligned to the supplied trimmed output
/// layout.
#[pyfunction]
pub fn range_join_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        false,
    )
}

/// Aggregate exactly two range predicates in the reverse direction.
///
/// Reverse aggregation uses the same intersected windows as forward
/// aggregation, but writes into right-side output slots while consuming
/// left-side source values.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two three-element range-anchor tuples using the
///   shared physical maps.
/// * `left_index` - Physical positions paired with left anchor values.
/// * `right_index` - Physical positions paired with sorted right values.
/// * `aggregations` - Non-empty aggregation requests over aligned left source
///   arrays.
/// * `return_matched` - Whether to include the boolean match array.
///
/// # Returns
///
/// Returns the standard reverse aggregation tuple, or `None` when no pair
/// satisfies both range predicates.
#[pyfunction]
pub fn range_join_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        true,
    )
}

/// Aggregate two range anchors followed by residual predicates.
///
/// The first two predicates are the only predicates used for binary search.
/// Their windows are intersected row by row. Predicates three onward are
/// evaluated in their original user order for every candidate in that
/// intersection. Aggregation state is updated only after all residuals pass.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two tuples. The first two are three-element
///   range anchors; later tuples are residual filters evaluated in order.
/// * `left_index` - Physical positions paired with the left anchor values.
/// * `right_index` - Physical positions paired with sorted right values.
/// * `aggregations` - Non-empty aggregation requests over aligned source
///   arrays.
/// * `return_matched` - Whether to include one match flag for every output
///   slot.
///
/// # Returns
///
/// Returns the standard forward aggregation tuple, or `None` when no complete
/// candidate survives both range anchors and every residual predicate.
#[pyfunction]
pub fn range_join_extended_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        false,
    )
}

/// Aggregate two range anchors plus residual predicates in reverse direction.
///
/// The first two predicates build and intersect right-oriented windows. Later
/// predicates are residual filters. Surviving candidates update left-source
/// aggregation state in right output slots.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two tuples. The first two are three-element
///   range anchors; later tuples are residual filters evaluated in order.
/// * `left_index` - Physical positions paired with left anchor values.
/// * `right_index` - Physical positions paired with sorted right values.
/// * `aggregations` - Non-empty aggregation requests over aligned left-side
///   source arrays.
/// * `return_matched` - Whether to include the per-output match mask.
///
/// # Returns
///
/// Returns the standard reverse aggregation tuple, or `None` when no candidate
/// survives the anchors and residual filters.
#[pyfunction]
pub fn range_join_extended_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_extended(
        py,
        predicates,
        left_index,
        right_index,
        aggregations,
        return_matched,
        true,
    )
}

#[cfg(test)]
mod aggregation_tests {
    use super::*;
    use numpy::PyArray1;

    fn call_range_join_aggregate<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_aggregate(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    fn call_range_join_aggregate_reverse<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_aggregate_reverse(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    fn call_range_join_extended_aggregate<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_extended_aggregate(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    fn call_range_join_extended_aggregate_reverse<'py>(
        py: Python<'py>,
        predicates: &Bound<'py, PyList>,
        left_index: &Bound<'py, PyAny>,
        right_index: &Bound<'py, PyAny>,
        aggregations: &Bound<'py, PyList>,
        return_matched: bool,
    ) -> PyResult<Option<Bound<'py, PyTuple>>> {
        range_join_extended_aggregate_reverse(
            py,
            predicates,
            left_index,
            right_index,
            aggregations,
            return_matched,
        )
    }

    #[test]
    fn forward_two_range_aggregation_uses_intersected_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left = PyArray1::from_vec(py, vec![4_i64]);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.clone().into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let mask = PyArray1::from_vec(py, vec![false, false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the range intersection has matches");
            // Both anchors share the physical index arrays passed separately
            // to the entry point. The second anchor uses a different value
            // layout while referring to the same physical right positions.
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100]);
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![40]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn empty_two_range_intersection_returns_none() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11, 12]);
            for op in ["<", ">"] {
                let mut fields = vec![
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                ];
                fields.push(op.into_pyobject(py)?.into_any());
                predicates.append(PyTuple::new(py, fields)?)?;
            }
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            assert!(call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .is_none());

            let empty = PyList::empty(py);
            empty.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            empty.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let empty_left_index = PyArray1::from_vec(py, Vec::<i64>::new());
            let empty_right_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            assert!(call_range_join_aggregate(
                py,
                &empty,
                &empty_left_index,
                &empty_right_index,
                &aggregations,
                false,
            )?
            .is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn reverse_two_range_aggregation_uses_intersected_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            // P1 selects right positions [2, 4): values 5 and 7.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            // P2 also selects a suffix beginning at position two, so the
            // intersection remains right positions [2, 4).
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![4_i64]);
            let mask = PyArray1::from_vec(py, vec![false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = call_range_join_aggregate_reverse(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the reverse range intersection has matches");

            // Reverse aggregation creates one output slot per right row. The
            // left value contributes to the two right positions in the
            // intersected window, not to the source-left slot itself.
            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 4, 4]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_covers_all_operations_maps_nulls_and_return_modes() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            // The first anchor is integer-valued and the second is float-
            // valued.  Each right value array is independently sorted, but
            // both use the same physical right labels.  The intersection is
            // right positions [2, 4) for left row 0 and [3, 4) for row 1.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3.0_f64, 5.0]).into_any(),
                    PyArray1::from_vec(py, vec![0.0_f64, 2.0, 4.0, 6.0]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let nulls = PyArray1::from_vec(py, vec![false, true, false, false]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "prod".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "min".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "max".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;

            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the exact range aggregation has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100, 101]);
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70, 40]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![1200, 40]);
            // min/max return source positions, not source values.
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![2, 3]);
            assert_eq!(outputs.get_item(3)?.extract::<Vec<i64>>()?, vec![3, 3]);
            assert_eq!(outputs.get_item(4)?.extract::<Vec<i64>>()?, vec![2, 1]);
            assert_eq!(outputs.get_item(5)?.extract::<Vec<i64>>()?, vec![2, 1]);

            // Reverse aggregation consumes left values and writes one slot
            // for every right row.  The nontrivial right output map must be
            // returned unchanged, while null left value 6 is skipped by
            // value-based operations but still counts for `size`.
            let left_values = PyArray1::from_vec(py, vec![2_i64, 6]);
            let left_nulls = PyArray1::from_vec(py, vec![false, true]);
            let reverse_aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "prod".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "min".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "max".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            left_nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = call_range_join_aggregate_reverse(
                py,
                &predicates,
                &left_index,
                &right_index,
                &reverse_aggregations,
                false,
            )?
            .expect("the reverse range aggregation has matches");
            assert_eq!(result.len(), 2);
            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 2, 2]
            );
            assert_eq!(
                outputs.get_item(1)?.extract::<Vec<i64>>()?,
                vec![1, 1, 2, 2]
            );
            assert_eq!(
                outputs.get_item(2)?.extract::<Vec<i64>>()?,
                vec![-1, -1, 0, 0]
            );
            assert_eq!(
                outputs.get_item(3)?.extract::<Vec<i64>>()?,
                vec![-1, -1, 0, 0]
            );
            assert_eq!(
                outputs.get_item(4)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 1]
            );
            assert_eq!(
                outputs.get_item(5)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 2]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_range_aggregation_filters_before_forward_and_reverse_updates() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101]);
            let right_index = PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]);
            // The two range anchors produce the same candidates as the
            // exact test above. The residual `<` removes the row-1 candidate
            // and is evaluated before any aggregation update.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64, 5]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 3, 5, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30, 40]);
            let nulls = PyArray1::from_vec(py, vec![false, true, false, false]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;

            let result = call_range_join_extended_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                false,
            )?
            .expect("the residual leaves one left row with matches");
            assert_eq!(result.len(), 2);
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100, 101]);
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70, 0]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![2, 0]);
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![2, 0]);

            let left_values = PyArray1::from_vec(py, vec![2_i64, 6]);
            let left_nulls = PyArray1::from_vec(py, vec![false, true]);
            let reverse_aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            left_values.clone().into_any(),
                            left_nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            left_nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = call_range_join_extended_aggregate_reverse(
                py,
                &predicates,
                &left_index,
                &right_index,
                &reverse_aggregations,
                true,
            )?
            .expect("the residual leaves reverse matches");
            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![40, 10, 30, 20]
            );
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 2, 2]
            );
            assert_eq!(
                outputs.get_item(1)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 1]
            );
            assert_eq!(
                outputs.get_item(2)?.extract::<Vec<i64>>()?,
                vec![0, 0, 1, 1]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_supports_unsigned_and_float_sources() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let mask = PyArray1::from_vec(py, vec![false, false, false]);
            let unsigned = PyArray1::from_vec(py, vec![1_u32, 2, 3]);
            let floats = PyArray1::from_vec(py, vec![1.5_f64, 2.5, 3.5]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            unsigned.into_any(),
                            mask.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            floats.into_any(),
                            mask.into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("mixed aggregation sources should match");
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<u64>>()?, vec![3]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<f64>>()?, vec![3.5]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_rejects_malformed_tuples_and_parallel_lengths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11, 12]);
            let malformed = PyList::empty(py);
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let aggregation = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, vec![false]).into_any(),
                        "sum".into_pyobject(py)?.into_any(),
                    ],
                )?],
            )?;
            let error = call_range_join_aggregate(
                py,
                &malformed,
                &left_index,
                &right_index,
                &aggregation,
                true,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("shared-index range predicates must contain 3 elements"));

            let mismatched = PyList::empty(py);
            mismatched.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            mismatched.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = call_range_join_aggregate(
                py,
                &mismatched,
                &left_index,
                &right_index,
                &aggregation,
                true,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("right and right_index must have equal lengths"));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_preserves_integer_overflow_semantics() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            // Both anchors select both right rows.  Keeping the right layout
            // tiny makes the expected wrapping arithmetic obvious.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let values = PyArray1::from_vec(py, vec![i64::MAX, 2]);
            let nulls = PyArray1::from_vec(py, vec![false, false]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.into_any(),
                            nulls.into_any(),
                            "prod".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                false,
            )?
            .expect("the overflow fixture has matches");
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![i64::MAX.wrapping_add(2)]
            );
            assert_eq!(
                outputs.get_item(1)?.extract::<Vec<i64>>()?,
                vec![i64::MAX.wrapping_mul(2)]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn forward_aggregation_skips_a_null_inside_a_matching_window() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            // Both anchors include right position one.  Its source value is
            // null, so it must count toward `size` and matching metadata but
            // must not contribute to `sum` or `count`.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let values = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let nulls = PyArray1::from_vec(py, vec![false, true, false]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = call_range_join_aggregate(
                py,
                &predicates,
                &left_index,
                &right_index,
                &aggregations,
                true,
            )?
            .expect("the null lies inside the matching window");
            // Forward aggregation writes one slot per left row and returns
            // the source physical label rather than a compact offset.
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100]);
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![30]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![1]);
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![2]);
            Ok(())
        })
        .unwrap();
    }
}
