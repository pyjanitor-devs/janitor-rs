//! Index construction for one non-equi join predicate.
//!
//! Pyjanitor supplies non-null, value-sorted arrays for the search paths and
//! keeps each value array aligned with its original `int64` index array.

use numpy::ndarray::ArrayView1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::compare::op::CompareOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    First,
    Last,
    Any,
    All,
}

impl Keep {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            "any" => Ok(Self::Any),
            "all" => Ok(Self::All),
            other => Err(PyValueError::new_err(format!(
                "invalid keep value: {other} (expected one of first, last, any, all)"
            ))),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SingleJoinResult {
    pub left_index: Vec<i64>,
    pub right_index: Vec<i64>,
    pub starts: Vec<usize>,
    pub ends: Vec<usize>,
}

fn partition_point<T: PartialOrd + Copy>(
    right: ArrayView1<'_, T>,
    predicate: impl Fn(T) -> bool,
) -> usize {
    if let Some(slice) = right.as_slice() {
        return slice.partition_point(|value| predicate(*value));
    }
    let mut low = 0;
    let mut high = right.len();
    while low < high {
        let middle = low + ((high - low) >> 1);
        if predicate(right[middle]) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

fn range_bounds<T: PartialOrd + Copy>(
    left_value: T,
    right: ArrayView1<'_, T>,
    op: CompareOp,
) -> (usize, usize) {
    match op {
        // left < right: keep the suffix after the last right <= left.
        CompareOp::Lt => {
            let start = partition_point(right, |value| value <= left_value);
            (start, right.len())
        }
        // left <= right: keep the suffix from the first right >= left.
        CompareOp::Le => {
            let start = partition_point(right, |value| value < left_value);
            (start, right.len())
        }
        // left > right: keep the prefix before the first right >= left.
        CompareOp::Gt => {
            let end = partition_point(right, |value| value < left_value);
            (0, end)
        }
        // left >= right: keep the prefix through the last right <= left.
        CompareOp::Ge => {
            let end = partition_point(right, |value| value <= left_value);
            (0, end)
        }
        CompareOp::Eq | CompareOp::Ne => unreachable!("range_bounds only handles range operators"),
    }
}

/// Build one binary-search window per matching left row.
pub fn build_range_core<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    _right_index_is_ordered: bool,
    op: CompareOp,
) -> Result<SingleJoinResult, &'static str> {
    if left.is_empty() || right.is_empty() {
        return Err("single join value arrays must be non-empty");
    }
    if left.len() != left_index.len() || right.len() != right_index.len() {
        return Err("single join values and indexes must have matching lengths");
    }
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return Err("single join range core requires a range comparator");
    }

    let mut result = SingleJoinResult {
        right_index: right_index.to_vec(),
        ..SingleJoinResult::default()
    };
    for (left_position, left_value) in left.iter().enumerate() {
        let (start, end) = range_bounds(*left_value, right, op);
        if start >= end {
            continue;
        }
        result.left_index.push(left_index[left_position]);
        result.starts.push(start);
        result.ends.push(end);
    }
    Ok(result)
}

fn choose_range(
    windows: &SingleJoinResult,
    keep: Keep,
    right_index_is_ordered: bool,
) -> Result<(Vec<i64>, Vec<i64>), &'static str> {
    let labels = windows.right_index.as_slice();
    let mut output_left = Vec::new();
    let mut output_right = Vec::new();

    let prefix_min = if right_index_is_ordered {
        None
    } else {
        Some(
            labels
                .iter()
                .scan(i64::MAX, |state, &value| {
                    *state = (*state).min(value);
                    Some(*state)
                })
                .collect::<Vec<_>>(),
        )
    };
    let prefix_max = if right_index_is_ordered {
        None
    } else {
        Some(
            labels
                .iter()
                .scan(i64::MIN, |state, &value| {
                    *state = (*state).max(value);
                    Some(*state)
                })
                .collect::<Vec<_>>(),
        )
    };
    let suffix_min = if right_index_is_ordered {
        None
    } else {
        let mut values = vec![0_i64; labels.len()];
        let mut current = i64::MAX;
        for position in (0..labels.len()).rev() {
            current = current.min(labels[position]);
            values[position] = current;
        }
        Some(values)
    };
    let suffix_max = if right_index_is_ordered {
        None
    } else {
        let mut values = vec![0_i64; labels.len()];
        let mut current = i64::MIN;
        for position in (0..labels.len()).rev() {
            current = current.max(labels[position]);
            values[position] = current;
        }
        Some(values)
    };

    for (row, (&start, &end)) in windows.starts.iter().zip(windows.ends.iter()).enumerate() {
        if keep == Keep::All {
            for &label in labels.iter().take(end).skip(start) {
                output_left.push(windows.left_index[row]);
                output_right.push(label);
            }
            continue;
        }
        let selected = match keep {
            Keep::Any => Some(labels[start]),
            Keep::First => {
                if right_index_is_ordered {
                    Some(labels[start])
                } else if start == 0 {
                    Some(prefix_min.as_ref().unwrap()[end - 1])
                } else if end == labels.len() {
                    Some(suffix_min.as_ref().unwrap()[start])
                } else {
                    Some(
                        prefix_min.as_ref().unwrap()[end - 1]
                            .min(suffix_min.as_ref().unwrap()[start]),
                    )
                }
            }
            Keep::Last => {
                if right_index_is_ordered {
                    Some(labels[end - 1])
                } else if start == 0 {
                    Some(prefix_max.as_ref().unwrap()[end - 1])
                } else if end == labels.len() {
                    Some(suffix_max.as_ref().unwrap()[start])
                } else {
                    Some(
                        prefix_max.as_ref().unwrap()[end - 1]
                            .max(suffix_max.as_ref().unwrap()[start]),
                    )
                }
            }
            Keep::All => unreachable!(),
        };
        if let Some(value) = selected {
            output_left.push(windows.left_index[row]);
            output_right.push(value);
        }
    }
    Ok((output_left, output_right))
}

fn null_labels(
    mask: Option<ArrayView1<'_, bool>>,
    index: Option<ArrayView1<'_, i64>>,
) -> Result<Vec<i64>, &'static str> {
    match (mask, index) {
        (None, None) => Ok(Vec::new()),
        (Some(mask), Some(index)) => {
            if mask.len() != index.len() {
                return Err("null masks and indexes must have matching lengths");
            }
            Ok(mask
                .iter()
                .zip(index.iter())
                .filter_map(|(&is_null, &label)| is_null.then_some(label))
                .collect())
        }
        _ => Err("null masks and indexes must be provided together"),
    }
}

/// Build `!=` results from filtered non-null arrays and optional null metadata.
#[allow(clippy::too_many_arguments)]
pub fn build_not_equal_core<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    left_nulls: Option<ArrayView1<'_, bool>>,
    left_nulls_index: Option<ArrayView1<'_, i64>>,
    right_nulls: Option<ArrayView1<'_, bool>>,
    right_nulls_index: Option<ArrayView1<'_, i64>>,
    _right_index_is_ordered: bool,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), &'static str> {
    if left.is_empty() || right.is_empty() {
        return Err("single join value arrays must be non-empty");
    }
    if left.len() != left_index.len() || right.len() != right_index.len() {
        return Err("single join values and indexes must have matching lengths");
    }
    let left_null_labels = null_labels(left_nulls, left_nulls_index)?;
    let right_null_labels = null_labels(right_nulls, right_nulls_index)?;
    let mut output_left = Vec::new();
    let mut output_right = Vec::new();
    // ELI5: the value search can use a strided view, but label extrema only
    // need the labels in logical order. Copying this small index vector keeps
    // `!=` correct for a strided int64 index view without changing the
    // caller's aligned arrays.
    let labels = right_index.to_vec();

    let prefix_min = labels
        .iter()
        .scan(i64::MAX, |state, &value| {
            *state = (*state).min(value);
            Some(*state)
        })
        .collect::<Vec<_>>();
    let prefix_max = labels
        .iter()
        .scan(i64::MIN, |state, &value| {
            *state = (*state).max(value);
            Some(*state)
        })
        .collect::<Vec<_>>();
    let suffix_min = {
        let mut values = vec![0_i64; labels.len()];
        let mut current = i64::MAX;
        for position in (0..labels.len()).rev() {
            current = current.min(labels[position]);
            values[position] = current;
        }
        values
    };
    let suffix_max = {
        let mut values = vec![0_i64; labels.len()];
        let mut current = i64::MIN;
        for position in (0..labels.len()).rev() {
            current = current.max(labels[position]);
            values[position] = current;
        }
        values
    };

    fn emit(
        output_left: &mut Vec<i64>,
        output_right: &mut Vec<i64>,
        left_label: i64,
        candidate: Option<i64>,
    ) {
        if let Some(right_label) = candidate {
            output_left.push(left_label);
            output_right.push(right_label);
        }
    }
    for (left_position, left_value) in left.iter().enumerate() {
        let less_start = partition_point(right, |value| value <= *left_value);
        let greater_end = partition_point(right, |value| value < *left_value);
        if keep == Keep::All {
            for &label in labels.iter().take(greater_end) {
                output_left.push(left_index[left_position]);
                output_right.push(label);
            }
            for &label in labels.iter().skip(less_start) {
                output_left.push(left_index[left_position]);
                output_right.push(label);
            }
            for &label in &right_null_labels {
                output_left.push(left_index[left_position]);
                output_right.push(label);
            }
            continue;
        }
        if keep == Keep::Any {
            if greater_end > 0 {
                emit(
                    &mut output_left,
                    &mut output_right,
                    left_index[left_position],
                    Some(labels[0]),
                );
            } else if less_start < right.len() {
                emit(
                    &mut output_left,
                    &mut output_right,
                    left_index[left_position],
                    Some(labels[less_start]),
                );
            } else {
                emit(
                    &mut output_left,
                    &mut output_right,
                    left_index[left_position],
                    right_null_labels.first().copied(),
                );
            }
            continue;
        }
        let minimum = keep == Keep::First;
        let mut candidate = None;
        let mut combine = |value: i64| {
            candidate = Some(candidate.map_or(value, |current| {
                if (minimum && value < current) || (!minimum && value > current) {
                    value
                } else {
                    current
                }
            }));
        };
        if greater_end > 0 {
            combine(if minimum {
                prefix_min[greater_end - 1]
            } else {
                prefix_max[greater_end - 1]
            });
        }
        if less_start < right.len() {
            combine(if minimum {
                suffix_min[less_start]
            } else {
                suffix_max[less_start]
            });
        }
        if let Some(value) = if minimum {
            right_null_labels.iter().copied().min()
        } else {
            right_null_labels.iter().copied().max()
        } {
            combine(value);
        }
        emit(
            &mut output_left,
            &mut output_right,
            left_index[left_position],
            candidate,
        );
    }
    let nonnull_extreme = if keep == Keep::First {
        labels.iter().copied().min()
    } else {
        labels.iter().copied().max()
    };
    for &left_label in &left_null_labels {
        if keep == Keep::All {
            for &right_label in &labels {
                output_left.push(left_label);
                output_right.push(right_label);
            }
        } else if keep == Keep::Any {
            emit(
                &mut output_left,
                &mut output_right,
                left_label,
                labels.first().copied(),
            );
        } else {
            emit(
                &mut output_left,
                &mut output_right,
                left_label,
                nonnull_extreme,
            );
        }
    }
    Ok((output_left, output_right))
}

fn result_dict<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
    starts: Option<Vec<usize>>,
    ends: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("left_index", left.into_pyarray(py))?;
    result.set_item("right_index", right.into_pyarray(py))?;
    if let (Some(starts), Some(ends)) = (starts, ends) {
        result.set_item("starts", starts.into_pyarray(py))?;
        result.set_item("ends", ends.into_pyarray(py))?;
    }
    Ok(result)
}

macro_rules! single_join_function {
    ($name:ident, $type:ty) => {
        #[allow(clippy::too_many_arguments)]
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right: PyReadonlyArray1<'py, $type>,
            right_index: PyReadonlyArray1<'py, i64>,
            right_index_is_ordered: bool,
            comparator: &str,
            keep: &str,
            return_building_blocks: bool,
            left_nulls: Option<PyReadonlyArray1<'py, bool>>,
            left_nulls_index: Option<PyReadonlyArray1<'py, i64>>,
            right_nulls: Option<PyReadonlyArray1<'py, bool>>,
            right_nulls_index: Option<PyReadonlyArray1<'py, i64>>,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            let op = CompareOp::try_from_str(comparator)?;
            let keep = Keep::parse(keep)?;
            if op == CompareOp::Eq {
                return Err(PyValueError::new_err(
                    "single join does not compute equality; handle == upstream",
                ));
            }
            let left = left.as_array();
            let right = right.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            if op == CompareOp::Ne {
                let (out_left, out_right) = build_not_equal_core(
                    left,
                    left_index,
                    right,
                    right_index,
                    left_nulls.as_ref().map(|value| value.as_array()),
                    left_nulls_index.as_ref().map(|value| value.as_array()),
                    right_nulls.as_ref().map(|value| value.as_array()),
                    right_nulls_index.as_ref().map(|value| value.as_array()),
                    right_index_is_ordered,
                    keep,
                )
                .map_err(PyValueError::new_err)?;
                if out_left.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(result_dict(py, out_left, out_right, None, None)?));
            }
            if left_nulls.is_some()
                || left_nulls_index.is_some()
                || right_nulls.is_some()
                || right_nulls_index.is_some()
            {
                return Err(PyValueError::new_err(
                    "null metadata is only supported for !=",
                ));
            }
            let windows = build_range_core(
                left,
                left_index,
                right,
                right_index,
                right_index_is_ordered,
                op,
            )
            .map_err(PyValueError::new_err)?;
            if return_building_blocks {
                if windows.left_index.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(result_dict(
                    py,
                    windows.left_index,
                    windows.right_index,
                    Some(windows.starts),
                    Some(windows.ends),
                )?));
            }
            let (out_left, out_right) = choose_range(&windows, keep, right_index_is_ordered)
                .map_err(PyValueError::new_err)?;
            if out_left.is_empty() {
                return Ok(None);
            }
            Ok(Some(result_dict(py, out_left, out_right, None, None)?))
        }
    };
}

single_join_function!(single_join_indices_int64, i64);
single_join_function!(single_join_indices_int32, i32);
single_join_function!(single_join_indices_int16, i16);
single_join_function!(single_join_indices_int8, i8);
single_join_function!(single_join_indices_uint64, u64);
single_join_function!(single_join_indices_uint32, u32);
single_join_function!(single_join_indices_uint16, u16);
single_join_function!(single_join_indices_uint8, u8);
single_join_function!(single_join_indices_f64, f64);
single_join_function!(single_join_indices_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_join_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_f32, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s};

    #[test]
    fn contiguous_and_strided_boundaries_match() {
        let right = array![1_i64, 3, 5, 5, 9];
        let padded = array![1_i64, -1, 3, -1, 5, -1, 5, -1, 9, -1];
        let left = array![0_i64, 5, 10];
        let index = array![10_i64, 20, 30];
        for op in [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge] {
            let dense = build_range_core(
                left.view(),
                index.view(),
                right.view(),
                array![0, 1, 2, 3, 4].view(),
                true,
                op,
            )
            .unwrap();
            let strided = build_range_core(
                left.view(),
                index.view(),
                padded.slice(s![..;2]),
                array![0, 1, 2, 3, 4].view(),
                true,
                op,
            )
            .unwrap();
            assert_eq!(dense.starts, strided.starts);
            assert_eq!(dense.ends, strided.ends);
        }
    }

    #[test]
    fn unordered_labels_use_range_extrema() {
        let windows = build_range_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![1, 3, 5, 7].view(),
            array![40, 10, 30, 20].view(),
            false,
            CompareOp::Lt,
        )
        .unwrap();
        let result = choose_range(&windows, Keep::First, false).unwrap();
        assert_eq!(result, (vec![100], vec![20]));
    }

    #[test]
    fn range_windows_cover_all_comparators_and_selection_modes() {
        let left = array![4_i64];
        let left_index = array![100_i64];
        let right = array![1_i64, 3, 5, 7];
        let right_index = array![10_i64, 20, 30, 40];
        let cases = [
            (CompareOp::Lt, 2_usize, 4_usize),
            (CompareOp::Le, 2, 4),
            (CompareOp::Gt, 0, 2),
            (CompareOp::Ge, 0, 2),
        ];
        for (op, expected_start, expected_end) in cases {
            let windows = build_range_core(
                left.view(),
                left_index.view(),
                right.view(),
                right_index.view(),
                true,
                op,
            )
            .unwrap();
            assert_eq!(windows.starts, vec![expected_start]);
            assert_eq!(windows.ends, vec![expected_end]);
            assert_eq!(
                choose_range(&windows, Keep::Any, true).unwrap(),
                (vec![100], vec![right_index[expected_start]])
            );
            assert_eq!(
                choose_range(&windows, Keep::First, true).unwrap(),
                (vec![100], vec![right_index[expected_start]])
            );
            assert_eq!(
                choose_range(&windows, Keep::Last, true).unwrap(),
                (vec![100], vec![right_index[expected_end - 1]])
            );
        }
    }

    #[test]
    fn empty_and_mismatched_inputs_are_rejected() {
        let empty = array![0_i64];
        assert!(build_range_core(
            empty.view(),
            array![].view(),
            array![1_i64].view(),
            array![1_i64].view(),
            true,
            CompareOp::Lt,
        )
        .is_err());
        assert!(build_range_core(
            array![0_i64].view(),
            array![1_i64, 2].view(),
            array![1_i64].view(),
            array![1_i64].view(),
            true,
            CompareOp::Lt,
        )
        .is_err());
    }

    #[test]
    fn not_equal_excludes_null_null_and_reduces_extrema() {
        let result = build_not_equal_core(
            array![2_i64].view(),
            array![10_i64].view(),
            array![1, 2, 3].view(),
            array![30, 20, 40].view(),
            Some(array![false, true].view()),
            Some(array![10, 11].view()),
            Some(array![false, true, false].view()),
            Some(array![30, 21, 40].view()),
            false,
            Keep::First,
        )
        .unwrap();
        assert_eq!(result, (vec![10, 11], vec![21, 20]));
    }
}
