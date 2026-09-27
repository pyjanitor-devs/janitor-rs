//! Fused aggregation for exactly two range predicates.
//!
//! The two range predicates are intersected into one half-open window per
//! left row. The windows are then passed to the existing optimized
//! `AggregationSet` range machinery; no pair indices are materialized.
//!
//! The public entry points are split by semantics:
//!
//! * `range_join_aggregate` and its reverse form handle exactly two range
//!   predicates. They intersect the two windows and aggregate every surviving
//!   pair.
//! * `range_join_extended_aggregate` and its reverse form handle the same two
//!   range anchors plus zero or more residual predicates. Residual predicates
//!   are evaluated inside the intersected windows before aggregation updates.
//!
//! The defining operation is window-based: each anchor produces one half-open
//! positional window for each logical left row, the two windows are
//! intersected, and aggregation visits the surviving right positions. The
//! value dtype of an anchor is an implementation detail of its search and is
//! not part of the API distinction.
//!
//! PyJanitor is responsible for removing null rows, aligning each predicate's
//! arrays, and sorting the right-hand range arrays in ascending order before
//! calling these functions. Rust validates the tuple shape and lengths but
//! does not sort the input.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::join_aggregation_helpers::aggregate_range_windows;
use crate::range_join::build_any_windows;
use crate::range_predicate::parse_aggregation_range_anchor;

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
    let first = parse_aggregation_range_anchor(first_tuple, true)?;
    let second = parse_aggregation_range_anchor(second_tuple, false)?;
    let (parsed, metadata) =
        crate::join_aggregation_helpers::residuals(py, predicates, false, true)?;
    crate::predicate::check_predicate_lengths(
        &parsed,
        first.range.left_len(),
        first.range.right_len(),
    )?;
    let windows = build_any_windows(&first.range, &second.range).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }

    let output_positions = if reverse {
        first
            .right_output_positions
            .as_ref()
            .map(|values| values.as_array())
    } else {
        first
            .left_output_positions
            .as_ref()
            .map(|values| values.as_array())
    };
    let output_len = output_positions
        .map(|values| values.len())
        .unwrap_or(if reverse {
            first.range.right_len()
        } else {
            first.range.left_len()
        });
    let source_len = if reverse {
        first.range.left_len()
    } else {
        first.range.right_len()
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
/// * `predicates` - Exactly two aligned range predicates. The first tuple may
///   contain six fields, or eight fields when it also carries forward and
///   reverse output-position maps. The second tuple contains five fields.
/// * `aggregations` - Non-empty aggregation requests over the source values.
/// * `return_matched` - Whether to include the boolean match array in the
///   returned tuple. It does not change aggregation semantics.
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
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    aggregate_range_extended(py, predicates, aggregations, return_matched, false)
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
/// * `predicates` - Exactly two aligned range predicates.
/// * `aggregations` - Non-empty aggregation requests over the left source
///   values.
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
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    aggregate_range_extended(py, predicates, aggregations, return_matched, true)
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
/// * `predicates` - At least two predicates. The first two are range anchors;
///   later predicates are residual filters.
/// * `aggregations` - Non-empty aggregation requests.
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
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_extended(py, predicates, aggregations, return_matched, false)
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
/// * `predicates` - At least two predicates; the first two are range anchors.
/// * `aggregations` - Non-empty aggregation requests over left-side values.
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
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_range_extended(py, predicates, aggregations, return_matched, true)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        range_join_aggregate,
        range_join_aggregate_reverse,
        range_join_extended_aggregate,
        range_join_extended_aggregate_reverse,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

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
                    left_index.clone().into_any(),
                    right.clone().into_any(),
                    right_index.clone().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    left_index.into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    right_index.into_any(),
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
            let result = range_join_aggregate(py, &predicates, &aggregations, true)?
                .expect("the range intersection has matches");
            // The first anchor uses the eight-field aggregation contract:
            // field four is the ordering bool, while fields five and six are
            // the output-position maps. Reading the old offsets would try to
            // extract this bool as an integer array.
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![0]);
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
            for (position, op) in ["<", ">"].into_iter().enumerate() {
                let mut fields = vec![
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11, 12]).into_any(),
                ];
                if position == 0 {
                    fields.push(true.into_pyobject(py)?.to_owned().into_any());
                }
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
            assert!(range_join_aggregate(py, &predicates, &aggregations, true,)?.is_none());

            let empty = PyList::empty(py);
            empty.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            empty.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, Vec::<i64>::new()).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert!(range_join_aggregate(py, &empty, &aggregations, false)?.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn reverse_two_range_aggregation_uses_intersected_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            // P1 selects right positions [2, 4): values 5 and 7.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            // P2 also selects a suffix beginning at position two, so the
            // intersection remains right positions [2, 4).
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
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
            let result = range_join_aggregate_reverse(py, &predicates, &aggregations, true)?
                .expect("the reverse range intersection has matches");

            // Reverse aggregation creates one output slot per right row. The
            // left value contributes to the two right positions in the
            // intersected window, not to the source-left slot itself.
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 1, 2, 3]);
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
            // The first anchor is integer-valued and the second is float-
            // valued.  Each right value array is independently sorted, but
            // both use the same physical right labels.  The intersection is
            // right positions [2, 4) for left row 0 and [3, 4) for row 1.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 101]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 0]).into_any(),
                    PyArray1::from_vec(py, vec![3_i64, 2, 1, 0]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3.0_f64, 5.0]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 101]).into_any(),
                    PyArray1::from_vec(py, vec![0.0_f64, 2.0, 4.0, 6.0]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
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

            let result = range_join_aggregate(py, &predicates, &aggregations, true)?
                .expect("the exact range aggregation has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![1, 0]);
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
            let result =
                range_join_aggregate_reverse(py, &predicates, &reverse_aggregations, false)?
                    .expect("the reverse range aggregation has matches");
            assert_eq!(result.len(), 2);
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![3, 2, 1, 0]);
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
            // The two range anchors produce the same candidates as the
            // exact test above. The residual `<` removes the row-1 candidate
            // and is evaluated before any aggregation update.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 101]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64, 5]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 101]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
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

            let result = range_join_extended_aggregate(py, &predicates, &aggregations, false)?
                .expect("the residual leaves one left row with matches");
            assert_eq!(result.len(), 2);
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 1]);
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
            let result = range_join_extended_aggregate_reverse(
                py,
                &predicates,
                &reverse_aggregations,
                true,
            )?
            .expect("the residual leaves reverse matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 1, 2, 3]);
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
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20, 30]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![3_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20, 30]).into_any(),
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
            let result = range_join_aggregate(py, &predicates, &aggregations, true)?
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
            let error = range_join_aggregate(py, &malformed, &aggregation, true).unwrap_err();
            assert!(error
                .to_string()
                .contains("range extended aggregation has invalid anchor tuple lengths"));

            let bad_maps = PyList::empty(py);
            bad_maps.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64, 3, 4]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11, 12]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    // One map entry for two left rows is malformed.
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            bad_maps.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11, 12]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = range_join_aggregate(py, &bad_maps, &aggregation, true).unwrap_err();
            assert!(error.to_string().contains("left output positions"));

            let mismatched = PyList::empty(py);
            mismatched.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            mismatched.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = range_join_aggregate(py, &mismatched, &aggregation, true).unwrap_err();
            assert!(error.to_string().contains("left and left_index"));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_preserves_integer_overflow_semantics() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            // Both anchors select both right rows.  Keeping the right layout
            // tiny makes the expected wrapping arithmetic obvious.
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
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
            let result = range_join_aggregate(py, &predicates, &aggregations, false)?
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
}
