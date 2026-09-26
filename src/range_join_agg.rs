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

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::join_aggregation_helpers::{aggregate_range_windows, check_residual_lengths};
use crate::range_join::{build_any_windows, parse_any_range_predicate};

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
    if !matches!(first_tuple.len(), 6 | 8) || second_tuple.len() != 5 {
        return Err(PyValueError::new_err(
            "range extended aggregation has invalid anchor tuple lengths",
        ));
    }

    // The aggregation form may carry output maps in fields four and five.
    // Normalize only the first anchor to the five-field window form so the
    // dtype dispatcher can use one parser for both index and aggregation
    // paths; the maps remain borrowed separately below.
    let first_window_tuple = if first_tuple.len() == 8 {
        PyTuple::new(
            py,
            [
                first_tuple.get_item(0)?,
                first_tuple.get_item(1)?,
                first_tuple.get_item(2)?,
                first_tuple.get_item(3)?,
                first_tuple.get_item(7)?,
            ],
        )?
    } else {
        PyTuple::new(
            py,
            [
                first_tuple.get_item(0)?,
                first_tuple.get_item(1)?,
                first_tuple.get_item(2)?,
                first_tuple.get_item(3)?,
                first_tuple.get_item(5)?,
            ],
        )?
    };
    let first = parse_any_range_predicate(&first_window_tuple, true)?;
    let second = parse_any_range_predicate(second_tuple, true)?;
    let (parsed, metadata) =
        crate::join_aggregation_helpers::residuals(py, predicates, false, true)?;
    for predicate in &parsed {
        check_residual_lengths(
            std::slice::from_ref(predicate),
            first.left_len(),
            first.right_len(),
        )?;
    }
    let windows = build_any_windows(&first, &second).map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }

    let left_output_positions = if first_tuple.len() == 8 {
        Some(
            first_tuple
                .get_item(4)?
                .extract::<PyReadonlyArray1<'py, i64>>()?,
        )
    } else {
        None
    };
    let right_output_positions = if first_tuple.len() == 8 {
        Some(
            first_tuple
                .get_item(5)?
                .extract::<PyReadonlyArray1<'py, i64>>()?,
        )
    } else {
        None
    };
    let output_positions = if reverse {
        right_output_positions
            .as_ref()
            .map(|values| values.as_array())
    } else {
        left_output_positions
            .as_ref()
            .map(|values| values.as_array())
    };
    let output_len = output_positions
        .map(|values| values.len())
        .unwrap_or(if reverse {
            first.right_len()
        } else {
            first.left_len()
        });
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
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
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
            Ok(())
        })
        .unwrap();
    }
}
