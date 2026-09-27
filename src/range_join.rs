//! Shared range-join window construction.
//!
//! A basic range join has two range predicates. This module computes the
//! half-open `starts`/`ends` windows for those predicates. Extended joins
//! reuse the same primitive and perform their additional filtering elsewhere.

use numpy::ndarray::{Array1, ArrayView1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::aggs::max::max_starts_ends::max_start_end_core_no_nulls;
use crate::aggs::min::min_starts_ends::min_start_end_core_no_nulls;
#[cfg(test)]
use crate::anchor_non_equi_join::build_range_core_with_labels;
use crate::join_candidate_materialization::materialize_range_candidates;
use crate::join_common::{result_dict, Keep, SingleJoinResult};
#[cfg(test)]
use crate::op::CompareOp;
use crate::predicate::{check_predicate_lengths, parse_predicates_with_nulls_strings};
use crate::range_predicate::{parse_any_range_predicate, AnyParsedRangePredicate};

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
    let first = build_range_core_with_labels(
        first.left,
        first.left_index,
        first.right,
        first.right_index,
        true,
        first.op,
        true,
        true,
    )?;
    let second = build_range_core_with_labels(
        second.left,
        second.left_index,
        second.right,
        second.right_index,
        true,
        second.op,
        false,
        true,
    )?;
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
    // Range inputs have already had null rows removed by PyJanitor. The
    // no-null RMQ entry points avoid allocating a full all-false mask solely
    // to satisfy the general nullable-array API. They return offsets into
    // `labels`, not public labels; the final loop performs that conversion.
    let selected_positions = match keep {
        Keep::First => {
            min_start_end_core_no_nulls(ArrayView1::from(labels), starts.view(), ends.view())?
        }
        Keep::Last => {
            max_start_end_core_no_nulls(ArrayView1::from(labels), starts.view(), ends.view())?
        }
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

    fn compare(left: i64, right: i64, operator: &str) -> bool {
        match operator {
            "<" => left < right,
            "<=" => left <= right,
            ">" => left > right,
            ">=" => left >= right,
            "==" => left == right,
            "!=" => left != right,
            other => panic!("unexpected reference operator: {other}"),
        }
    }

    /// Build the public result expected from a pair-by-pair implementation.
    ///
    /// The optimized range path searches sorted values, but its public
    /// result is still defined by the original physical right labels.  This
    /// deliberately simple helper does not use windows or binary search: it
    /// is an independent oracle for the wrapper tests.
    #[allow(clippy::too_many_arguments)]
    fn reference_pairs(
        left_index: &[i64],
        first_left: &[i64],
        second_left: &[i64],
        right_index: &[i64],
        first_right: &[i64],
        second_right: &[i64],
        first_operator: &str,
        second_operator: &str,
        keep: &str,
    ) -> (Vec<i64>, Vec<i64>) {
        let mut output_left = Vec::new();
        let mut output_right = Vec::new();
        for left_position in 0..left_index.len() {
            let mut passing = Vec::new();
            for right_position in 0..right_index.len() {
                if compare(
                    first_left[left_position],
                    first_right[right_position],
                    first_operator,
                ) && compare(
                    second_left[left_position],
                    second_right[right_position],
                    second_operator,
                ) {
                    passing.push(right_position);
                }
            }
            match keep {
                "all" => {
                    for right_position in passing {
                        output_left.push(left_index[left_position]);
                        output_right.push(right_index[right_position]);
                    }
                }
                "any" => {
                    if let Some(&right_position) = passing.first() {
                        output_left.push(left_index[left_position]);
                        output_right.push(right_index[right_position]);
                    }
                }
                "first" | "last" => {
                    let selected = passing.into_iter().min_by_key(|&position| {
                        if keep == "first" {
                            right_index[position]
                        } else {
                            -right_index[position]
                        }
                    });
                    if let Some(right_position) = selected {
                        output_left.push(left_index[left_position]);
                        output_right.push(right_index[right_position]);
                    }
                }
                other => panic!("unexpected reference keep: {other}"),
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
            let operators = ["<", "<=", ">", ">="];
            let keeps = ["all", "first", "last", "any"];

            for first_operator in operators {
                for second_operator in operators {
                    for keep in keeps {
                        let predicates = PyList::empty(py);
                        predicates.append(PyTuple::new(
                            py,
                            [
                                PyArray1::from_vec(py, first_left.clone()).into_any(),
                                PyArray1::from_vec(py, left_index.clone()).into_any(),
                                PyArray1::from_vec(py, first_right.clone()).into_any(),
                                PyArray1::from_vec(py, right_index.clone()).into_any(),
                                true.into_pyobject(py)?.to_owned().into_any(),
                                first_operator.into_pyobject(py)?.into_any(),
                            ],
                        )?)?;
                        predicates.append(PyTuple::new(
                            py,
                            [
                                PyArray1::from_vec(py, second_left.clone()).into_any(),
                                PyArray1::from_vec(py, left_index.clone()).into_any(),
                                PyArray1::from_vec(py, second_right.clone()).into_any(),
                                PyArray1::from_vec(py, right_index.clone()).into_any(),
                                true.into_pyobject(py)?.to_owned().into_any(),
                                second_operator.into_pyobject(py)?.into_any(),
                            ],
                        )?)?;
                        let expected = reference_pairs(
                            &left_index,
                            &first_left,
                            &second_left,
                            &right_index,
                            &first_right,
                            &second_right,
                            first_operator,
                            second_operator,
                            keep,
                        );
                        let result = range_join_indices(py, &predicates, keep, false)?;
                        assert_index_result(result, expected);
                    }
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
                    PyArray1::from_vec(py, vec![101_i64]).into_any(),
                    PyArray1::from_vec(py, first_right).into_any(),
                    PyArray1::from_vec(py, right_index).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![101_i64]).into_any(),
                    PyArray1::from_vec(py, second_right).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let blocks = range_join_indices(py, &predicates, "all", true)?
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
            let next_value = |seed: &mut u64| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*seed >> 32) % 9) as i64 - 4
            };

            for case in 0..32 {
                let mut first_right = (0..5).map(|_| next_value(&mut seed)).collect::<Vec<_>>();
                let mut second_right = (0..5).map(|_| next_value(&mut seed)).collect::<Vec<_>>();
                first_right.sort_unstable();
                second_right.sort_unstable();
                let first_left = (0..4).map(|_| next_value(&mut seed)).collect::<Vec<_>>();
                let second_left = (0..4).map(|_| next_value(&mut seed)).collect::<Vec<_>>();
                let first_operator = operators[case % operators.len()];
                let second_operator = operators[(case * 3 + 1) % operators.len()];
                let keep = keeps[case % keeps.len()];
                let predicates = PyList::empty(py);
                for (left, right, operator) in [
                    (&first_left, &first_right, first_operator),
                    (&second_left, &second_right, second_operator),
                ] {
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, left.clone()).into_any(),
                            PyArray1::from_vec(py, left_index.clone()).into_any(),
                            PyArray1::from_vec(py, right.clone()).into_any(),
                            PyArray1::from_vec(py, right_index.clone()).into_any(),
                            true.into_pyobject(py)?.to_owned().into_any(),
                            operator.into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                }
                let expected = reference_pairs(
                    &left_index,
                    &first_left,
                    &second_left,
                    &right_index,
                    &first_right,
                    &second_right,
                    first_operator,
                    second_operator,
                    keep,
                );
                assert_index_result(range_join_indices(py, &predicates, keep, false)?, expected);
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

            for keep in ["all", "first", "last", "any"] {
                let predicates = PyList::empty(py);
                for (left, left_index_values, right, operator) in [
                    (&first_left, &left_index, &first_right, "<"),
                    (&second_left, &left_index, &second_right, ">"),
                ] {
                    predicates.append(PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, left.clone()).into_any(),
                            PyArray1::from_vec(py, left_index_values.clone()).into_any(),
                            PyArray1::from_vec(py, right.clone()).into_any(),
                            PyArray1::from_vec(py, right_index.clone()).into_any(),
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
                        if compare(first_left[left_position], first_right[right_position], "<")
                            && compare(
                                second_left[left_position],
                                second_right[right_position],
                                ">",
                            )
                            && compare(
                                residual_left[left_position],
                                residual_right[right_position],
                                residual_operator,
                            )
                        {
                            passing.push(right_position);
                        }
                    }
                    match keep {
                        "all" => {
                            for position in passing {
                                expected.0.push(left_index[left_position]);
                                expected.1.push(right_index[position]);
                            }
                        }
                        "any" => {
                            if let Some(&position) = passing.first() {
                                expected.0.push(left_index[left_position]);
                                expected.1.push(right_index[position]);
                            }
                        }
                        "first" | "last" => {
                            let position = passing.into_iter().min_by_key(|&position| {
                                if keep == "first" {
                                    right_index[position]
                                } else {
                                    -right_index[position]
                                }
                            });
                            if let Some(position) = position {
                                expected.0.push(left_index[left_position]);
                                expected.1.push(right_index[position]);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                assert_index_result(
                    range_join_extended_indices(py, &predicates, keep)?,
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
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
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
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = range_join_indices(py, &malformed, "all", false).unwrap_err();
            assert!(error
                .to_string()
                .contains("range predicates must contain 6 elements"));

            let mismatched = PyList::empty(py);
            for (left_index, operator) in [(vec![0_i64, 1], "<"), (vec![0_i64], ">")] {
                mismatched.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, left_index).into_any(),
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            let error = range_join_indices(py, &mismatched, "all", false).unwrap_err();
            assert!(error.to_string().contains("left and left_index"));

            let extended_malformed = PyList::empty(py);
            extended_malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            extended_malformed.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = range_join_extended_indices(py, &extended_malformed, "all").unwrap_err();
            assert!(error
                .to_string()
                .contains("extended range anchors must contain 5 elements"));

            let empty = PyList::empty(py);
            for operator in ["<", ">"] {
                empty.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                        PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        PyArray1::from_vec(py, vec![10_i64, 11]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        operator.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
            }
            assert!(range_join_indices(py, &empty, "all", false)?.is_none());
            Ok(())
        })
        .unwrap();
    }
}
