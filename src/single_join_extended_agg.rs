//! Fused aggregation for multiple conditional-join predicates.
//!
//! The first predicate supplies either a range window or the complete
//! null-aware `!=` candidate stream. Later predicates filter candidates before
//! `AggregationSet` is updated. No intermediate index pairs are built.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::aggs::ensure_equal_lengths_core;
use crate::op::CompareOp;
use crate::predicate::{
    null_metadata_views, parse_predicates_with_nulls_strings, predicates_match_dispatch,
    PredicateView,
};
use crate::single_join::{range_bounds, visit_not_equal_pairs_core};

fn aggregate_range<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    op: CompareOp,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
) {
    for (left_position, &left_value) in left.iter().enumerate() {
        let (start, end) = range_bounds(left_value, right, op);
        for right_position in start..end {
            if predicates_match_dispatch(
                residuals,
                residual_metadata,
                left_position,
                right_position,
            ) {
                if reverse {
                    set.update(left_position, right_position);
                } else {
                    set.update(right_position, left_position);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn aggregate_not_equal<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_full_len: usize,
    left_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right: ArrayView1<'_, T>,
    right_full_len: usize,
    right_positions: ArrayView1<'_, i64>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
) -> Result<(), String> {
    visit_not_equal_pairs_core(
        left,
        left_full_len,
        left_positions,
        right,
        right_full_len,
        right_positions,
        left_null_positions,
        right_null_positions,
        is_extension_array,
        |left_position, right_position| {
            if predicates_match_dispatch(
                residuals,
                residual_metadata,
                left_position,
                right_position,
            ) {
                if reverse {
                    set.update(left_position, right_position);
                } else {
                    set.update(right_position, left_position);
                }
            }
        },
    )
}

fn residuals<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
) -> PyResult<(
    Vec<crate::predicate::Predicate<'py>>,
    Option<Vec<crate::predicate::NullMetadata<'py>>>,
)> {
    let values = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        values.append(item)?;
    }
    parse_predicates_with_nulls_strings(py, &values)
}

fn check_residual_lengths(
    predicates: &[crate::predicate::Predicate<'_>],
    left_len: usize,
    right_len: usize,
) -> PyResult<()> {
    for predicate in predicates {
        ensure_equal_lengths_core(
            "first left predicate array",
            left_len,
            "residual left predicate array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core(
            "first right predicate array",
            right_len,
            "residual right predicate array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_range<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left: PyReadonlyArray1<'py, T>,
    right: PyReadonlyArray1<'py, T>,
    op: CompareOp,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (parsed, metadata) = residuals(py, predicates)?;
    let left = left.as_array();
    let right = right.as_array();
    check_residual_lengths(&parsed, left.len(), right.len())?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut set = AggregationSet::new(
        if reverse { right.len() } else { left.len() },
        if reverse { left.len() } else { right.len() },
        &inputs,
    )?;
    let views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_range(
        left,
        right,
        op,
        &views,
        metadata_views.as_deref(),
        &mut set,
        reverse,
    );
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

#[allow(clippy::too_many_arguments)]
fn run_not_equal<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left: PyReadonlyArray1<'py, T>,
    left_index: PyReadonlyArray1<'py, i64>,
    left_positions: PyReadonlyArray1<'py, i64>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    right_positions: PyReadonlyArray1<'py, i64>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (parsed, metadata) = residuals(py, predicates)?;
    check_residual_lengths(&parsed, left_index.len()?, right_index.len()?)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let left_full_len = left_index.len()?;
    let right_full_len = right_index.len()?;
    let mut set = AggregationSet::new(
        if reverse {
            right_full_len
        } else {
            left_full_len
        },
        if reverse {
            left_full_len
        } else {
            right_full_len
        },
        &inputs,
    )?;
    let views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_not_equal(
        left.as_array(),
        left_full_len,
        left_positions.as_array(),
        left_null_positions.as_ref().map(|values| values.as_array()),
        right.as_array(),
        right_full_len,
        right_positions.as_array(),
        right_null_positions
            .as_ref()
            .map(|values| values.as_array()),
        is_extension_array,
        &views,
        metadata_views.as_deref(),
        &mut set,
        reverse,
    )
    .map_err(PyValueError::new_err)?;
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

#[allow(clippy::too_many_arguments)]
fn dispatch<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    first: &Bound<'py, PyTuple>,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended aggregation requires at least two predicates",
        ));
    }
    let op_position = match first.len() {
        6 => 5,
        11 => 10,
        _ => {
            return Err(PyValueError::new_err(
                "the first extended predicate must contain 6 or 11 elements",
            ))
        }
    };
    let op = CompareOp::try_from_str(first.get_item(op_position)?.extract::<&str>()?)?;
    if op != CompareOp::Ne {
        if first.len() != 6 {
            return Err(PyValueError::new_err(
                "the first range predicate must contain 6 elements",
            ));
        }
        if !matches!(
            op,
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
        ) {
            return Err(PyValueError::new_err(
                "single extended aggregation requires a range predicate first",
            ));
        }
        first.get_item(4)?.extract::<bool>()?;
        return run_range(
            py,
            predicates,
            first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
            first.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?,
            op,
            aggregations,
            reverse,
        );
    }
    if first.len() != 11 {
        return Err(PyValueError::new_err(
            "the first != predicate must contain 11 elements",
        ));
    }
    let left_null_positions = if first.get_item(3)?.is_none() {
        None
    } else {
        Some(first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    let right_null_positions = if first.get_item(7)?.is_none() {
        None
    } else {
        Some(first.get_item(7)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    run_not_equal(
        py,
        predicates,
        first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
        first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        first.get_item(2)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        left_null_positions,
        first.get_item(4)?.extract::<PyReadonlyArray1<'py, T>>()?,
        first.get_item(5)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        first.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?,
        right_null_positions,
        first.get_item(9)?.extract::<bool>()?,
        aggregations,
        reverse,
    )
}

macro_rules! extended_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch::<$ty>(py, predicates, &first, aggregations, false)
        }

        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch::<$ty>(py, predicates, &first, aggregations, true)
        }
    };
}

extended_aggregation_functions!(
    single_join_extended_aggregate_int64,
    single_join_extended_aggregate_reverse_int64,
    i64
);
extended_aggregation_functions!(
    single_join_extended_aggregate_int32,
    single_join_extended_aggregate_reverse_int32,
    i32
);
extended_aggregation_functions!(
    single_join_extended_aggregate_int16,
    single_join_extended_aggregate_reverse_int16,
    i16
);
extended_aggregation_functions!(
    single_join_extended_aggregate_int8,
    single_join_extended_aggregate_reverse_int8,
    i8
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint64,
    single_join_extended_aggregate_reverse_uint64,
    u64
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint32,
    single_join_extended_aggregate_reverse_uint32,
    u32
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint16,
    single_join_extended_aggregate_reverse_uint16,
    u16
);
extended_aggregation_functions!(
    single_join_extended_aggregate_uint8,
    single_join_extended_aggregate_reverse_uint8,
    u8
);
extended_aggregation_functions!(
    single_join_extended_aggregate_f64,
    single_join_extended_aggregate_reverse_f64,
    f64
);
extended_aggregation_functions!(
    single_join_extended_aggregate_f32,
    single_join_extended_aggregate_reverse_f32,
    f32
);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        single_join_extended_aggregate_int64,
        single_join_extended_aggregate_reverse_int64,
        single_join_extended_aggregate_int32,
        single_join_extended_aggregate_reverse_int32,
        single_join_extended_aggregate_int16,
        single_join_extended_aggregate_reverse_int16,
        single_join_extended_aggregate_int8,
        single_join_extended_aggregate_reverse_int8,
        single_join_extended_aggregate_uint64,
        single_join_extended_aggregate_reverse_uint64,
        single_join_extended_aggregate_uint32,
        single_join_extended_aggregate_reverse_uint32,
        single_join_extended_aggregate_uint16,
        single_join_extended_aggregate_reverse_uint16,
        single_join_extended_aggregate_uint8,
        single_join_extended_aggregate_reverse_uint8,
        single_join_extended_aggregate_f64,
        single_join_extended_aggregate_reverse_f64,
        single_join_extended_aggregate_f32,
        single_join_extended_aggregate_reverse_f32,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    #[test]
    fn range_residuals_filter_before_forward_aggregation() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
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
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![3_i64, 7, 9, 7]).into_any(),
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
            let result = single_join_extended_aggregate_int64(py, &predicates, &aggregations)?
                .expect("the filtered range has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_value = result.get_item(1)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn all_not_equal_residuals_aggregate_without_materializing_pairs() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64, 21]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3]).into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let values = PyArray1::from_vec(py, vec![100_i64, 200]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
            let aggregation = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;
            let result = single_join_extended_aggregate_int64(py, &predicates, &aggregations)?
                .expect("the not-equal join has matches");
            assert_eq!(
                result.get_item(0)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(1)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![200, 300]);
            Ok(())
        })
        .unwrap();
    }
}
