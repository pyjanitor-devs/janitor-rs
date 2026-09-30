//! Fused aggregation for equi-led conditional joins.
//!
//! PyJanitor prepares the equi indexer, optional dense right codes, physical
//! layouts, range tuples, residual tuples, and aggregation inputs. This
//! module traverses surviving equi candidates and updates AggregationSet
//! without materializing pair arrays.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::aggs::ensure_equal_lengths_core;
use crate::equi_join::{
    append_range_residuals, build_dense_right_metadata, build_equi_range_windows, decode_equi_code,
    equi_candidate_slice, parse_equi_range_predicates, validate_equi_range_predicate_count,
};
use crate::join_common::Keep;
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, Predicate,
};
/// Aggregate an equi-led join without materializing matching pairs.
///
/// PyJanitor supplies arrays in one shared physical coordinate system. The
/// left indexer contains either direct right positions (unique right keys) or
/// dense right-key codes (duplicate right keys). When duplicate codes are
/// present, this function builds `DenseRightMetadata` once and visits only the
/// physical right positions belonging to each left code. Optional range
/// windows narrow those groups before residual predicates are evaluated.
///
/// Forward aggregation uses left rows as output slots and right rows as source
/// values. Reverse aggregation uses right rows as output slots and left rows
/// as source values. `AggregationSet` preserves the existing neutral values,
/// null handling, extrema sentinels, and matched-mask semantics.
///
/// The output tuple follows the existing aggregation contract:
/// `(output_positions, matched, aggregation_arrays)` when `return_matched` is
/// true, and `(output_positions, aggregation_arrays)` otherwise. `None` means
/// that no candidate survived the equi, range, and residual predicates.
///
/// # Arguments
///
/// * `py` - Python interpreter token required to create Python return values.
/// * `left_index` - Int64 labels for the left physical layout, one per left
///   predicate row.
/// * `right_index` - Int64 labels for the right physical layout, one per
///   physical right position. These labels may represent a sorted layout.
/// * `left_indexer` - One int64 equi result per left row. `-1` means no equi
///   candidate; otherwise it is a direct right position or dense right code.
/// * `right_equi_codes` - Optional one-code-per-right-row array. `None` means
///   right equi keys are unique; `Some` means duplicate-key metadata is
///   required. Codes must be `-1` or nonnegative.
/// * `range_predicates` - Zero, one, or two three-field tuples containing
///   `(left_values, right_values, operator)`. With duplicate right keys, the
///   tuples are used as range windows. With unique right keys, they are
///   evaluated as residual predicates. Their right values must share the
///   physical layout described by `right_index`.
/// * `residual_predicates` - Remaining predicate tuples, including ordinary
///   three-field comparisons and six-field null-aware `!=` comparisons.
/// * `aggregations` - Non-empty Rust aggregation specifications prepared by
///   PyJanitor, such as `sum`, `count`, `size`, `prod`, `min`, or `max`.
/// * `return_matched` - Whether to include one boolean matched value per
///   output slot.
/// * `reverse` - Whether to aggregate left source values into right output
///   slots instead of right source values into left output slots.
///
/// # Errors
///
/// Returns a Python `ValueError` for mismatched indexer/code lengths, invalid
/// codes, malformed predicates, unsupported operators, invalid range windows,
/// or missing aggregation specifications.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn equi_join_aggregate<'py>(
    py: Python<'py>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
    left_indexer: PyReadonlyArray1<'py, i64>,
    right_equi_codes: Option<PyReadonlyArray1<'py, i64>>,
    range_predicates: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left_index_array = left_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_index_array = right_index.extract::<PyReadonlyArray1<'py, i64>>()?;
    let left_index_values = left_index_array.as_array();
    let right_index_values = right_index_array.as_array();
    let left_indexer = left_indexer.as_array();

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    validate_equi_range_predicate_count(range_predicates.len())?;
    ensure_equal_lengths_core(
        "left index",
        left_index_values.len(),
        "equi indexer",
        left_indexer.len(),
    )
    .map_err(PyValueError::new_err)?;
    let output_len = if reverse {
        right_index_values.len()
    } else {
        left_index_values.len()
    };
    let source_len = if reverse {
        left_index_values.len()
    } else {
        right_index_values.len()
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    let (parsed, metadata, windows, groups) = if let Some(right_codes) = right_equi_codes {
        ensure_equal_lengths_core(
            "right index",
            right_index_values.len(),
            "right equi codes",
            right_codes.as_array().len(),
        )
        .map_err(PyValueError::new_err)?;
        let (parsed, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
        let ranges = parse_equi_range_predicates(range_predicates, left_index, right_index)?;
        let windows = build_equi_range_windows(&ranges).map_err(PyValueError::new_err)?;
        let groups =
            build_dense_right_metadata(right_index_values, right_codes.as_array(), Keep::All)
                .map_err(PyValueError::new_err)?;
        (parsed, metadata, windows, Some(groups))
    } else {
        let combined = append_range_residuals(py, range_predicates, residual_predicates)?;
        let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &combined)?;
        (parsed, metadata, None, None)
    };

    check_predicate_lengths(&parsed, left_index_values.len(), right_index_values.len())?;
    let views = parsed.iter().map(Predicate::view).collect::<Vec<_>>();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let windows = windows
        .as_ref()
        .map(|(starts, ends)| (starts.as_slice(), ends.as_slice()));

    if let Some(groups) = groups {
        for row in 0..left_indexer.len() {
            let Some(code) =
                decode_equi_code(left_indexer[row], "left").map_err(PyValueError::new_err)?
            else {
                continue;
            };
            let candidates =
                equi_candidate_slice(code, &groups, windows, row, right_index_values.len())
                    .map_err(PyValueError::new_err)?;
            for &right_position in candidates {
                if !predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    row,
                    right_position,
                ) {
                    continue;
                }
                if reverse {
                    set.update(row, right_position);
                } else {
                    set.update(right_position, row);
                }
            }
        }
    } else {
        for row in 0..left_indexer.len() {
            let Some(code) =
                decode_equi_code(left_indexer[row], "left").map_err(PyValueError::new_err)?
            else {
                continue;
            };
            let right_position = code;
            if right_position >= right_index_values.len()
                || !predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    row,
                    right_position,
                )
            {
                continue;
            }
            if reverse {
                set.update(row, right_position);
            } else {
                set.update(right_position, row);
            }
        }
    }

    if set.is_empty() {
        return Ok(None);
    }
    let output_positions = if reverse {
        Some(right_index_values)
    } else {
        Some(left_index_values)
    };
    Ok(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equi_join_aggregate, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    fn sum_aggregation<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyList>> {
        let values = PyArray1::from_vec(py, values);
        let nulls = PyArray1::from_vec(py, vec![false; values.len()?]);
        let aggregation = PyTuple::new(
            py,
            [
                values.into_any(),
                nulls.into_any(),
                "sum".into_pyobject(py)?.into_any(),
            ],
        )?;
        PyList::new(py, [aggregation])
    }

    fn all_aggregations<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyList>> {
        let values = PyArray1::from_vec(py, values);
        let nulls = PyArray1::from_vec(py, vec![false; values.len()?]);
        PyList::new(
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
                        nulls.into_any(),
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
        )
    }

    fn empty_predicates<'py>(py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }

    #[test]
    fn unique_equi_aggregation_handles_forward_reverse_and_unmatched_rows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21]);
            let left_indexer = PyArray1::from_vec(py, vec![1_i64, -1]);
            let ranges = empty_predicates(py);
            let residuals = empty_predicates(py);
            let aggregations = sum_aggregation(py, vec![5, 7])?;

            let forward = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                None,
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .expect("the matched unique equi row should aggregate");
            assert_eq!(forward.get_item(0)?.extract::<Vec<i64>>()?, vec![10, 11]);
            assert_eq!(
                forward.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, false]
            );
            let forward_item = forward.get_item(2)?;
            let forward_values = forward_item.cast::<PyList>()?;
            assert_eq!(
                forward_values.get_item(0)?.extract::<Vec<i64>>()?,
                vec![7, 0]
            );

            let reverse = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                None,
                &ranges,
                &residuals,
                &aggregations,
                true,
                true,
            )?
            .expect("the matched unique equi row should aggregate in reverse");
            assert_eq!(reverse.get_item(0)?.extract::<Vec<i64>>()?, vec![20, 21]);
            assert_eq!(
                reverse.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, true]
            );
            let reverse_item = reverse.get_item(2)?;
            let reverse_values = reverse_item.cast::<PyList>()?;
            assert_eq!(
                reverse_values.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 5]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn duplicate_equi_aggregation_visits_each_matching_right_position() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21, 22]);
            let left_indexer = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right_codes = PyArray1::from_vec(py, vec![0_i64, 1, 0]);
            let ranges = empty_predicates(py);
            let residuals = empty_predicates(py);
            let aggregations = sum_aggregation(py, vec![5, 7, 9])?;

            let forward = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                Some(right_codes.readonly()),
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .expect("duplicate equi candidates should aggregate");
            let forward_item = forward.get_item(2)?;
            let forward_values = forward_item.cast::<PyList>()?;
            assert_eq!(
                forward_values.get_item(0)?.extract::<Vec<i64>>()?,
                vec![14, 7]
            );

            let reverse = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                Some(right_codes.readonly()),
                &ranges,
                &residuals,
                &sum_aggregation(py, vec![2, 3])?,
                true,
                true,
            )?
            .expect("duplicate equi candidates should aggregate in reverse");
            let reverse_item = reverse.get_item(2)?;
            let reverse_values = reverse_item.cast::<PyList>()?;
            assert_eq!(
                reverse_values.get_item(0)?.extract::<Vec<i64>>()?,
                vec![2, 3, 2]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn duplicate_equi_aggregation_intersects_ranges_and_filters_residuals() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21, 22]);
            let left_indexer = PyArray1::from_vec(py, vec![0_i64]);
            let right_codes = PyArray1::from_vec(py, vec![0_i64, 0, 1]);
            let ranges = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 3, 5]).into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?],
            )?;
            let residuals = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64, 2, 4]).into_any(),
                        "==".into_pyobject(py)?.into_any(),
                    ],
                )?],
            )?;
            let aggregations = sum_aggregation(py, vec![10, 20, 30])?;
            let result = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                Some(right_codes.readonly()),
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .expect("the range and residual should leave one candidate");
            let values_item = result.get_item(2)?;
            let values = values_item.cast::<PyList>()?;
            assert_eq!(values.get_item(0)?.extract::<Vec<i64>>()?, vec![20]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn duplicate_equi_aggregation_supports_all_operations_without_matched() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21, 22]);
            let left_indexer = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right_codes = PyArray1::from_vec(py, vec![0_i64, 1, 0]);
            let result = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                Some(right_codes.readonly()),
                &empty_predicates(py),
                &empty_predicates(py),
                &all_aggregations(py, vec![2, 3, 4])?,
                false,
                false,
            )?
            .expect("duplicate equi candidates should aggregate");

            assert_eq!(result.len(), 2);
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![10, 11]);
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![6, 3]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<i64>>()?, vec![8, 3]);
            assert_eq!(outputs.get_item(2)?.extract::<Vec<i64>>()?, vec![0, 1]);
            assert_eq!(outputs.get_item(3)?.extract::<Vec<i64>>()?, vec![2, 1]);
            assert_eq!(outputs.get_item(4)?.extract::<Vec<i64>>()?, vec![2, 1]);
            assert_eq!(outputs.get_item(5)?.extract::<Vec<i64>>()?, vec![2, 1]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn unique_equi_aggregation_applies_range_as_a_residual() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21]);
            let left_indexer = PyArray1::from_vec(py, vec![1_i64]);
            let ranges = PyList::new(
                py,
                [PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 3]).into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?],
            )?;
            let result = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                None,
                &ranges,
                &empty_predicates(py),
                &sum_aggregation(py, vec![9])?,
                true,
                true,
            )?
            .expect("the unique equi candidate should satisfy the range");

            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![20, 21]);
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 9]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn duplicate_equi_reverse_aggregation_intersects_two_ranges() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21, 22, 23]);
            let left_indexer = PyArray1::from_vec(py, vec![0_i64]);
            let right_codes = PyArray1::from_vec(py, vec![0_i64, 0, 0, 0]);
            let ranges = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, vec![2_i64]).into_any(),
                            PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                            "<".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            PyArray1::from_vec(py, vec![3_i64]).into_any(),
                            PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                            "<=".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                Some(right_codes.readonly()),
                &ranges,
                &empty_predicates(py),
                &sum_aggregation(py, vec![9])?,
                true,
                true,
            )?
            .expect("the two ranges should leave two reverse candidates");

            assert_eq!(
                result.get_item(0)?.extract::<Vec<i64>>()?,
                vec![20, 21, 22, 23]
            );
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, false, true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![0, 0, 9, 9]
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn equi_aggregation_rejects_malformed_duplicate_codes() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let error = equi_join_aggregate(
                py,
                &PyArray1::from_vec(py, vec![10_i64]),
                &PyArray1::from_vec(py, vec![20_i64]),
                PyArray1::from_vec(py, vec![0_i64]).readonly(),
                Some(PyArray1::from_vec(py, vec![i64::MAX]).readonly()),
                &empty_predicates(py),
                &empty_predicates(py),
                &sum_aggregation(py, vec![5])?,
                true,
                false,
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("right code exceeds the right index length"));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn equi_aggregation_checks_empty_aggregations_before_lengths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64]);
            let left_indexer = PyArray1::from_vec(py, Vec::<i64>::new());
            let aggregations = PyList::empty(py);
            let error = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                None,
                &empty_predicates(py),
                &empty_predicates(py),
                &aggregations,
                true,
                false,
            )
            .unwrap_err();
            assert_eq!(
                error.to_string(),
                "ValueError: at least one aggregation is required"
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn equi_aggregation_returns_none_when_no_candidate_survives() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64]);
            let left_indexer = PyArray1::from_vec(py, vec![-1_i64]);
            let result = equi_join_aggregate(
                py,
                &left_index,
                &right_index,
                left_indexer.readonly(),
                None,
                &empty_predicates(py),
                &empty_predicates(py),
                &sum_aggregation(py, vec![5])?,
                true,
                false,
            )?;
            assert!(result.is_none());
            Ok(())
        })
        .unwrap();
    }
}
