//! Multiple conditional-join predicates built on the single range kernel.
//!
//! The first predicate must be a range comparison. It creates compact
//! half-open right-side windows; every later predicate filters candidates in
//! those windows. `keep` is applied only after all residual predicates pass.
//!
//! All-`!=` joins deliberately remain in pyjanitor. Their first candidate set
//! has filtered non-null values plus full null metadata, which is a different
//! physical layout from the range path handled here.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::multi_join_indices::predicate::{
    null_metadata_views, parse_predicates_with_nulls_strings, predicates_match_dispatch,
    NullMetadata, Predicate,
};
use crate::op::CompareOp;
use crate::single_join::{build_range_core, Keep, SingleJoinResult};

fn materialize_windows(
    windows: &SingleJoinResult,
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);
    let labels = windows.right_index.as_slice();

    // Follow the established batch-indices layout: selection modes first find
    // at most one winning physical position per left row, while `all` uses a
    // count pass followed by an exact materialization pass. The latter avoids
    // reserving the full range-window upper bound when residual predicates
    // eliminate many candidates.
    if keep == Keep::All {
        let mut output_len = 0_usize;
        for row in 0..windows.left_index.len() {
            let start = windows.starts[row];
            let end = windows.ends[row];
            let left_position = windows.left_positions[row];
            for right_position in start..end {
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
                    output_len = output_len
                        .checked_add(1)
                        .ok_or("single extended join result size exceeds platform capacity")?;
                }
            }
        }
        if output_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;

        for row in 0..windows.left_index.len() {
            let start = windows.starts[row];
            let end = windows.ends[row];
            let left_position = windows.left_positions[row];
            for (offset, &label) in labels[start..end].iter().enumerate() {
                let right_position = start + offset;
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
                    output_left.push(windows.left_index[row]);
                    output_right.push(label);
                }
            }
        }
        debug_assert_eq!(output_left.len(), output_len);
        debug_assert_eq!(output_right.len(), output_len);
        return Ok((output_left, output_right));
    }

    // Each selection mode emits no more than one pair for each retained left
    // row. Reserving the number of range-window rows is therefore a tight
    // upper bound and does not require a count pass.
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(windows.left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(windows.left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    for row in 0..windows.left_index.len() {
        let start = windows.starts[row];
        let end = windows.ends[row];
        let left_position = windows.left_positions[row];
        let mut selected = None;
        for (offset, &label) in labels[start..end].iter().enumerate() {
            let right_position = start + offset;
            if !predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                continue;
            }
            match keep {
                Keep::Any => {
                    selected = Some(right_position);
                    break;
                }
                Keep::First => {
                    let replace = match selected {
                        None => true,
                        Some(current) => label < labels[current],
                    };
                    if replace {
                        selected = Some(right_position);
                    }
                }
                Keep::Last => {
                    let replace = match selected {
                        None => true,
                        Some(current) => label > labels[current],
                    };
                    if replace {
                        selected = Some(right_position);
                    }
                }
                Keep::All => unreachable!(),
            }
        }
        if let Some(right_position) = selected {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[right_position]);
        }
    }
    if output_left.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    Ok((output_left, output_right))
}

fn result_dict<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("left_index", left.into_pyarray(py))?;
    result.set_item("right_index", right.into_pyarray(py))?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn extended_join<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first_left: PyReadonlyArray1<'py, T>,
    first_left_index: PyReadonlyArray1<'py, i64>,
    first_right: PyReadonlyArray1<'py, T>,
    first_right_index: PyReadonlyArray1<'py, i64>,
    right_index_is_ordered: bool,
    first_op: CompareOp,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended join requires at least two predicates",
        ));
    }
    if !matches!(
        first_op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return Err(PyValueError::new_err(
            "single extended join requires a range predicate first",
        ));
    }
    let keep = Keep::parse(keep)?;
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    let left = first_left.as_array();
    let right = first_right.as_array();
    let left_index = first_left_index.as_array();
    let right_index = first_right_index.as_array();
    for predicate in &parsed {
        ensure_equal_lengths_core(
            "first left predicate array",
            left.len(),
            "residual left predicate array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core(
            "first right predicate array",
            right.len(),
            "residual right predicate array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    let windows = build_range_core(
        left,
        left_index,
        right,
        right_index,
        right_index_is_ordered,
        first_op,
    )
    .map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    let (out_left, out_right) = materialize_windows(&windows, &parsed, metadata.as_deref(), keep)
        .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, out_left, out_right)?))
}

macro_rules! extended_join_function {
    ($name:ident, $type:ty) => {
        /// Build flat indices for a range-led join with residual predicates.
        ///
        /// The first list item must be the six-element range form
        /// `(left, left_index, right, right_index,
        /// right_index_is_ordered, comparator)`. Later items are ordinary
        /// three-element string predicates or the six-element null-aware
        /// `!=` form. `keep` is applied only after all residual predicates
        /// pass; `keep="all"` is therefore the building-blocks mode.
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            keep: &str,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            if predicates.len() < 2 {
                return Err(PyValueError::new_err(
                    "single extended join requires at least two predicates",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            if first.len() != 6 {
                return Err(PyValueError::new_err(
                    "the first extended predicate must contain 6 elements",
                ));
            }
            let first_op = CompareOp::try_from_str(first.get_item(5)?.extract::<&str>()?)?;
            extended_join(
                py,
                predicates,
                keep,
                first
                    .get_item(0)?
                    .extract::<PyReadonlyArray1<'py, $type>>()?,
                first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                first
                    .get_item(2)?
                    .extract::<PyReadonlyArray1<'py, $type>>()?,
                first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                first.get_item(4)?.extract::<bool>()?,
                first_op,
            )
        }
    };
}

extended_join_function!(single_join_extended_indices_int64, i64);
extended_join_function!(single_join_extended_indices_int32, i32);
extended_join_function!(single_join_extended_indices_int16, i16);
extended_join_function!(single_join_extended_indices_int8, i8);
extended_join_function!(single_join_extended_indices_uint64, u64);
extended_join_function!(single_join_extended_indices_uint32, u32);
extended_join_function!(single_join_extended_indices_uint16, u16);
extended_join_function!(single_join_extended_indices_uint8, u8);
extended_join_function!(single_join_extended_indices_f64, f64);
extended_join_function!(single_join_extended_indices_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_f32, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    fn read_pair<'py>(result: &Bound<'py, PyDict>, py: Python<'py>) -> (Vec<i64>, Vec<i64>) {
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
        let _ = py;
        (left, right)
    }

    #[test]
    fn filters_range_windows_before_keep_selection() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![100_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                        PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![3_i64, 7, 9, 7]).into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();

            let result = single_join_extended_indices_int64(py, &predicates, "first")
                .unwrap()
                .unwrap();
            assert_eq!(read_pair(&result, py), (vec![100], vec![20]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn no_surviving_residual_matches_returns_none() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![100_i64]).into_any(),
                        PyArray1::from_vec(py, vec![5_i64, 7]).into_any(),
                        PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        "==".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();

            assert!(single_join_extended_indices_int64(py, &predicates, "all")
                .unwrap()
                .is_none());
            Ok(())
        })
        .unwrap();
    }
}
