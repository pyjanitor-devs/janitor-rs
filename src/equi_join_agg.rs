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
    append_range_residuals, build_dense_right_metadata, build_equi_range_windows,
    DenseRightMetadata,
};
use crate::join_common::Keep;
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, Predicate,
};
use crate::range_predicate::parse_any_range_parts;

/// Return the duplicate-right physical positions for one left equi code.
///
/// Without range windows, the complete code group is returned. With a range
/// window, the group is already sorted by physical position, so two binary
/// searches restrict it to the half-open interval `[start, end)`. The returned
/// slice borrows the shared `positions` buffer and does not allocate.
///
/// # Arguments
///
/// * `code` - Dense equi-key code from one entry in `left_indexer`.
/// * `counts` - Number of physical right positions stored for each code.
/// * `offsets` - Flat-buffer boundaries for each code.
/// * `positions` - Flat physical right positions grouped by code.
/// * `windows` - Optional per-left-row `(starts, ends)` range windows.
/// * `row` - Left physical row whose range window should be used.
/// * `right_len` - Number of physical rows in the right layout.
///
/// # Errors
///
/// Returns an error when the selected range window ends beyond the physical
/// right layout. Empty or inverted windows produce an empty candidate slice.
fn candidate_slice<'a>(
    code: usize,
    counts: &[usize],
    offsets: &[usize],
    positions: &'a [usize],
    windows: Option<(&[usize], &[usize])>,
    row: usize,
    right_len: usize,
) -> Result<&'a [usize], String> {
    if counts.get(code).copied().unwrap_or(0) == 0 {
        return Ok(&[]);
    }
    let start = offsets[code];
    let end = offsets[code + 1];
    let group = &positions[start..end];
    let Some((starts, ends)) = windows else {
        return Ok(group);
    };
    if ends[row] > right_len {
        return Err("equi aggregation range window is out of bounds".to_owned());
    }
    if starts[row] >= ends[row] {
        return Ok(&[]);
    }
    let first = group.partition_point(|&position| position < starts[row]);
    let last = group.partition_point(|&position| position < ends[row]);
    Ok(&group[first..last])
}

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
///   `(left_values, right_values, operator)`. Their right values must share
///   the physical layout described by `right_index`.
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
    ensure_equal_lengths_core(
        "left index",
        left_index_values.len(),
        "equi indexer",
        left_indexer.len(),
    )
    .map_err(PyValueError::new_err)?;

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
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
        let ranges = range_predicates
            .iter()
            .map(|item| {
                let tuple = item
                    .cast::<PyTuple>()
                    .map_err(|_| PyValueError::new_err("each equi range must be a tuple"))?;
                if tuple.len() != 3 {
                    return Err(PyValueError::new_err(
                        "equi range predicates must contain 3 elements",
                    ));
                }
                parse_any_range_parts(
                    &tuple.get_item(0)?,
                    left_index,
                    &tuple.get_item(1)?,
                    right_index,
                    &tuple.get_item(2)?,
                )
            })
            .collect::<PyResult<Vec<_>>>()?;
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

    if let Some(DenseRightMetadata {
        counts,
        offsets,
        positions,
        ..
    }) = groups
    {
        for row in 0..left_indexer.len() {
            let code = left_indexer[row];
            if code < -1 {
                return Err(PyValueError::new_err(
                    "left codes must be greater than or equal to -1",
                ));
            }
            if code == -1 {
                continue;
            }
            let code =
                usize::try_from(code).map_err(|_| PyValueError::new_err("invalid left code"))?;
            let candidates = candidate_slice(
                code,
                &counts,
                &offsets,
                &positions,
                windows,
                row,
                right_index_values.len(),
            )
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
            let code = left_indexer[row];
            if code < -1 {
                return Err(PyValueError::new_err(
                    "left codes must be greater than or equal to -1",
                ));
            }
            if code == -1 {
                continue;
            }
            let right_position = usize::try_from(code)
                .map_err(|_| PyValueError::new_err("invalid unique-right position"))?;
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
