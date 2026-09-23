//! Fused aggregation for one conditional-join predicate.
//!
//! This module mirrors `single_join.rs`'s candidate traversal but updates
//! `AggregationSet` immediately instead of building left/right index arrays.
//! `keep` is intentionally absent: aggregation consumes every pair that
//! satisfies the comparison.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results, parse_inputs, AggregationSet};
use crate::op::CompareOp;
use crate::single_join::{range_bounds, visit_not_equal_pairs_core};

/// Execute one fused aggregation pass over a single predicate.
///
/// Range predicates use binary-search boundaries. `!=` uses the same strict
/// prefix/suffix and explicit-null traversal as the index kernel, but invokes
/// `AggregationSet::update` for each pair instead of storing that pair.
#[allow(clippy::too_many_arguments)]
fn aggregate_single<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, T>,
    right: PyReadonlyArray1<'py, T>,
    comparator: &str,
    left_positions: Option<PyReadonlyArray1<'py, i64>>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
    aggregations: &Bound<'py, PyList>,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let op = CompareOp::try_from_str(comparator)?;
    if op == CompareOp::Eq {
        return Err(PyValueError::new_err(
            "single join aggregation does not compute equality; handle == upstream",
        ));
    }

    let left = left.as_array();
    let right = right.as_array();
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    let is_not_equal = op == CompareOp::Ne;
    if !is_not_equal
        && (left_positions.is_some()
            || left_null_positions.is_some()
            || right_positions.is_some()
            || right_null_positions.is_some()
            || is_extension_array)
    {
        return Err(PyValueError::new_err(
            "position metadata is only supported for !=",
        ));
    }
    if is_not_equal && (left_positions.is_none() || right_positions.is_none()) {
        return Err(PyValueError::new_err(
            "left and right positions are required for != aggregation",
        ));
    }

    let left_full_len = if is_not_equal {
        let non_null = match left_positions.as_ref() {
            Some(values) => values.len()?,
            None => 0,
        };
        let nulls = match left_null_positions.as_ref() {
            Some(values) => values.len()?,
            None => 0,
        };
        non_null.checked_add(nulls).ok_or_else(|| {
            PyValueError::new_err("single join aggregation position count exceeds capacity")
        })?
    } else {
        left.len()
    };
    let right_full_len = if is_not_equal {
        let non_null = match right_positions.as_ref() {
            Some(values) => values.len()?,
            None => 0,
        };
        let nulls = match right_null_positions.as_ref() {
            Some(values) => values.len()?,
            None => 0,
        };
        non_null.checked_add(nulls).ok_or_else(|| {
            PyValueError::new_err("single join aggregation position count exceeds capacity")
        })?
    } else {
        right.len()
    };

    // Forward aggregation reads values from right and writes one result slot
    // per left row. Reverse aggregation swaps those two roles.
    let output_len = if reverse {
        right_full_len
    } else {
        left_full_len
    };
    let source_len = if reverse {
        left_full_len
    } else {
        right_full_len
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs)?;

    if is_not_equal {
        visit_not_equal_pairs_core(
            left,
            left_full_len,
            left_positions.as_ref().unwrap().as_array(),
            right,
            right_full_len,
            right_positions.as_ref().unwrap().as_array(),
            left_null_positions.as_ref().map(|values| values.as_array()),
            right_null_positions
                .as_ref()
                .map(|values| values.as_array()),
            is_extension_array,
            |left_position, right_position| {
                if reverse {
                    set.update(left_position, right_position);
                } else {
                    set.update(right_position, left_position);
                }
            },
        )
        .map_err(PyValueError::new_err)?;
    } else {
        // A single range join produces one contiguous right-side window per
        // left row. Reuse the optimized prefix/suffix aggregation paths
        // instead of visiting every matching position. The reverse methods
        // scatter each source row into dense right-side output slots.
        let mut boundaries = Vec::with_capacity(left.len());
        for &left_value in left.iter() {
            let (start, end) = range_bounds(left_value, right, op);
            let boundary = if matches!(op, CompareOp::Lt | CompareOp::Le) {
                start
            } else {
                end
            };
            boundaries.push(i64::try_from(boundary).map_err(|_| {
                PyValueError::new_err("single join aggregation boundary exceeds int64")
            })?);
        }
        let boundaries = ArrayView1::from(&boundaries[..]);
        if matches!(op, CompareOp::Lt | CompareOp::Le) {
            if reverse {
                set.aggregate_reverse_starts(boundaries);
            } else {
                set.aggregate_starts(boundaries);
            }
        } else {
            if reverse {
                set.aggregate_reverse_ends(boundaries);
            } else {
                set.aggregate_ends(boundaries);
            }
        }
    }

    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results(py, set)?))
}

macro_rules! single_join_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Fused forward aggregation for one range or `!=` predicate.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $forward<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $ty>,
            right: PyReadonlyArray1<'py, $ty>,
            comparator: &str,
            left_positions: Option<PyReadonlyArray1<'py, i64>>,
            left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            is_extension_array: bool,
            aggregations: &Bound<'py, PyList>,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            aggregate_single(
                py,
                left,
                right,
                comparator,
                left_positions,
                left_null_positions,
                right_positions,
                right_null_positions,
                is_extension_array,
                aggregations,
                false,
            )
        }

        /// Fused reverse aggregation for one range or `!=` predicate.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $reverse<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $ty>,
            right: PyReadonlyArray1<'py, $ty>,
            comparator: &str,
            left_positions: Option<PyReadonlyArray1<'py, i64>>,
            left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            is_extension_array: bool,
            aggregations: &Bound<'py, PyList>,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            aggregate_single(
                py,
                left,
                right,
                comparator,
                left_positions,
                left_null_positions,
                right_positions,
                right_null_positions,
                is_extension_array,
                aggregations,
                true,
            )
        }
    };
}

single_join_aggregation_functions!(
    single_join_aggregate_int64,
    single_join_aggregate_reverse_int64,
    i64
);
single_join_aggregation_functions!(
    single_join_aggregate_int32,
    single_join_aggregate_reverse_int32,
    i32
);
single_join_aggregation_functions!(
    single_join_aggregate_int16,
    single_join_aggregate_reverse_int16,
    i16
);
single_join_aggregation_functions!(
    single_join_aggregate_int8,
    single_join_aggregate_reverse_int8,
    i8
);
single_join_aggregation_functions!(
    single_join_aggregate_uint64,
    single_join_aggregate_reverse_uint64,
    u64
);
single_join_aggregation_functions!(
    single_join_aggregate_uint32,
    single_join_aggregate_reverse_uint32,
    u32
);
single_join_aggregation_functions!(
    single_join_aggregate_uint16,
    single_join_aggregate_reverse_uint16,
    u16
);
single_join_aggregation_functions!(
    single_join_aggregate_uint8,
    single_join_aggregate_reverse_uint8,
    u8
);
single_join_aggregation_functions!(
    single_join_aggregate_f64,
    single_join_aggregate_reverse_f64,
    f64
);
single_join_aggregation_functions!(
    single_join_aggregate_f32,
    single_join_aggregate_reverse_f32,
    f32
);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        single_join_aggregate_int64,
        single_join_aggregate_reverse_int64,
        single_join_aggregate_int32,
        single_join_aggregate_reverse_int32,
        single_join_aggregate_int16,
        single_join_aggregate_reverse_int16,
        single_join_aggregate_int8,
        single_join_aggregate_reverse_int8,
        single_join_aggregate_uint64,
        single_join_aggregate_reverse_uint64,
        single_join_aggregate_uint32,
        single_join_aggregate_reverse_uint32,
        single_join_aggregate_uint16,
        single_join_aggregate_reverse_uint16,
        single_join_aggregate_uint8,
        single_join_aggregate_reverse_uint8,
        single_join_aggregate_f64,
        single_join_aggregate_reverse_f64,
        single_join_aggregate_f32,
        single_join_aggregate_reverse_f32,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    #[test]
    fn range_forward_aggregation_updates_every_matching_left_row() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
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
            let result = single_join_aggregate_int64(
                py,
                left.readonly(),
                right.readonly(),
                "<",
                None,
                None,
                None,
                None,
                false,
                &aggregations,
            )?
            .expect("the range has matching candidates");
            assert_eq!(
                result.get_item(0)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(1)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![30, 30]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_reverse_aggregation_updates_every_matching_right_slot() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = PyArray1::from_vec(py, vec![5_i64, 7]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2]);
            let values = PyArray1::from_vec(py, vec![10_i64, 20]);
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
            let result = single_join_aggregate_reverse_int64(
                py,
                left.readonly(),
                right.readonly(),
                ">",
                None,
                None,
                None,
                None,
                false,
                &aggregations,
            )?
            .expect("the range has matching candidates");
            assert_eq!(
                result.get_item(0)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(1)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![30, 30]);
            Ok(())
        })
        .unwrap();
    }
}
