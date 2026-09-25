//! Fused aggregation for exactly two range predicates.
//!
//! The two range predicates are intersected into one half-open window per
//! left row. The windows are then passed to the existing optimized
//! `AggregationSet` range machinery; no pair indices are materialized.

use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::range_join::{build_windows, parse_range_predicate, RangePredicate};

/// Expand sparse surviving windows into dense aggregation boundaries.
///
/// `build_windows` omits left rows whose intersection is empty because that is
/// ideal for index generation. The optimized aggregation API instead expects
/// one range per source left row. Empty rows are therefore represented as
/// `[0, 0)`, which contributes nothing and remains unmatched.
///
/// # Arguments
///
/// * `windows` - Sparse windows produced by [`build_windows`].
/// * `left_len` - Number of rows in the complete left aggregation layout.
///
/// # Returns
///
/// Two int64 vectors with one boundary pair per left output slot. Their
/// positions match the supplied left value layout, including rows whose
/// window is empty.
///
/// # Errors
///
/// Returns an error if a positional `usize` boundary cannot be represented as
/// int64. The caller validates the sparse left positions while constructing
/// the windows.
fn dense_boundaries(
    windows: &crate::join_common::SingleJoinResult,
    left_len: usize,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let mut starts = vec![0_i64; left_len];
    let mut ends = vec![0_i64; left_len];
    for (row, &left_position) in windows.left_positions.iter().enumerate() {
        let start = i64::try_from(windows.starts[row]).map_err(|_| "range start exceeds int64")?;
        let end = i64::try_from(windows.ends[row]).map_err(|_| "range end exceeds int64")?;
        starts[left_position] = start;
        ends[left_position] = end;
    }
    Ok((starts, ends))
}

/// Run forward or reverse aggregation for exactly two range predicates.
///
/// The predicates use the ordinary six-element range contract. The local
/// aggregation slots already match the supplied left/right layouts, so this
/// basic path returns identity output positions rather than accepting separate
/// output maps. Aggregation consumes every pair in the intersection; there is
/// deliberately no `keep` parameter.
fn run_two_range<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "range aggregation requires exactly two predicates",
        ));
    }
    let first_item = predicates.get_item(0)?;
    let first_tuple = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second_tuple = second_item.cast::<PyTuple>()?;
    let first = parse_range_predicate::<T>(first_tuple)?;
    let second = parse_range_predicate::<T>(second_tuple)?;
    let left = first.left.as_array();
    let right = first.right.as_array();

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    let windows = build_windows(
        RangePredicate {
            left,
            left_index: first.left_index.as_array(),
            right,
            right_index: first.right_index.as_array(),
            op: first.op,
        },
        RangePredicate {
            left: second.left.as_array(),
            left_index: second.left_index.as_array(),
            right: second.right.as_array(),
            right_index: second.right_index.as_array(),
            op: second.op,
        },
    )
    .map_err(PyValueError::new_err)?;
    let (starts, ends) = dense_boundaries(&windows, left.len()).map_err(PyValueError::new_err)?;
    let starts = ArrayView1::from(&starts[..]);
    let ends = ArrayView1::from(&ends[..]);
    let output_len = if reverse { right.len() } else { left.len() };
    let source_len = if reverse { left.len() } else { right.len() };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    if reverse {
        set.aggregate_reverse_starts_ends(starts, ends);
    } else {
        set.aggregate_starts_ends(starts, ends);
    }
    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results_with_positions(
        py,
        set,
        None,
        output_len,
        return_matched,
    )?))
}

macro_rules! range_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Aggregate over the intersection of two ascending range predicates.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Exactly two six-element tuples of the form
        ///   `(left, left_index, right, right_index,
        ///   right_index_is_ordered, comparator)`. PyJanitor supplies
        ///   null-free arrays and sorts both right value arrays before calling
        ///   Rust. The two predicates must use the same aligned layouts.
        /// * `aggregations` - Non-empty aggregation requests. Each request is
        ///   parsed by the shared aggregation input parser and is evaluated
        ///   for every pair in the intersected range window.
        /// * `return_matched` - Include the per-left-row matched mask in the
        ///   returned tuple when true.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair is present. Otherwise returns the
        /// standard aggregation result with identity output positions and
        /// either `(output_positions, matched, arrays)` or
        /// `(output_positions, arrays)` depending on `return_matched`.
        /// There is no `keep` parameter: aggregation consumes all pairs.
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            run_two_range::<$ty>(py, predicates, aggregations, return_matched, false)
        }

        /// Aggregate the same two range predicates in reverse orientation.
        ///
        /// The predicate and aggregation arguments have the same contract as
        /// the forward function, but source values come from the left layout
        /// and output slots are indexed by right rows. Reverse aggregation
        /// uses the optimized boundary-start/end kernels already provided by
        /// `AggregationSet`.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair is present; otherwise returns the
        /// output positions, optional matched mask, and aggregation arrays.
        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            run_two_range::<$ty>(py, predicates, aggregations, return_matched, true)
        }
    };
}

range_aggregation_functions!(
    range_join_aggregate_int64,
    range_join_aggregate_reverse_int64,
    i64
);
range_aggregation_functions!(
    range_join_aggregate_int32,
    range_join_aggregate_reverse_int32,
    i32
);
range_aggregation_functions!(
    range_join_aggregate_int16,
    range_join_aggregate_reverse_int16,
    i16
);
range_aggregation_functions!(
    range_join_aggregate_int8,
    range_join_aggregate_reverse_int8,
    i8
);
range_aggregation_functions!(
    range_join_aggregate_uint64,
    range_join_aggregate_reverse_uint64,
    u64
);
range_aggregation_functions!(
    range_join_aggregate_uint32,
    range_join_aggregate_reverse_uint32,
    u32
);
range_aggregation_functions!(
    range_join_aggregate_uint16,
    range_join_aggregate_reverse_uint16,
    u16
);
range_aggregation_functions!(
    range_join_aggregate_uint8,
    range_join_aggregate_reverse_uint8,
    u8
);
range_aggregation_functions!(
    range_join_aggregate_f64,
    range_join_aggregate_reverse_f64,
    f64
);
range_aggregation_functions!(
    range_join_aggregate_f32,
    range_join_aggregate_reverse_f32,
    f32
);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
        };
    }
    add!(
        range_join_aggregate_int64,
        range_join_aggregate_reverse_int64,
        range_join_aggregate_int32,
        range_join_aggregate_reverse_int32,
        range_join_aggregate_int16,
        range_join_aggregate_reverse_int16,
        range_join_aggregate_int8,
        range_join_aggregate_reverse_int8,
        range_join_aggregate_uint64,
        range_join_aggregate_reverse_uint64,
        range_join_aggregate_uint32,
        range_join_aggregate_reverse_uint32,
        range_join_aggregate_uint16,
        range_join_aggregate_reverse_uint16,
        range_join_aggregate_uint8,
        range_join_aggregate_reverse_uint8,
        range_join_aggregate_f64,
        range_join_aggregate_reverse_f64,
        range_join_aggregate_f32,
        range_join_aggregate_reverse_f32,
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
                    true.into_pyobject(py)?.to_owned().into_any(),
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
            let result = range_join_aggregate_int64(py, &predicates, &aggregations, true)?
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
            for op in ["<", ">"] {
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                        PyArray1::from_vec(py, vec![10_i64, 11, 12]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        op.into_pyobject(py)?.into_any(),
                    ],
                )?)?;
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
            assert!(range_join_aggregate_int64(py, &predicates, &aggregations, true)?.is_none());
            Ok(())
        })
        .unwrap();
    }
}
