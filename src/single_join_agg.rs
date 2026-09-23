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
/// This function owns the traversal decision; [`AggregationSet`] only owns
/// the requested reductions. In other words, `AggregationSet` does not decide
/// whether the comparison is a range or `!=` join. This function parses the
/// comparator and chooses the matching candidate strategy before it calls
/// `set.update`, `set.aggregate_starts`, or `set.aggregate_ends`.
///
/// Range predicates use binary-search boundaries. A forward `<`/`<=` query
/// creates a suffix window for every left row, while a forward `>`/`>=` query
/// creates a prefix window. The corresponding forward and reverse boundary
/// methods on `AggregationSet` then choose their adaptive direct, sweep, or
/// prefix/suffix-table implementation.
///
/// The reverse boundary methods use the same one-boundary-per-left-row input,
/// but scatter each left source row into dense right output slots. This is why
/// reverse range aggregation can use the optimized path even though its output
/// is indexed by the right side.
///
/// `!=` uses the same strict prefix/suffix and explicit-null traversal as the
/// index kernel, but invokes `AggregationSet::update` for each valid pair
/// instead of storing that pair. The position metadata is needed because the
/// value arrays supplied for `!=` contain only non-null rows.
///
/// # Position and null metadata
///
/// For range operators, `left` and `right` are complete, aligned physical
/// arrays, so their lengths are sufficient to size the aggregation state.
///
/// For `!=`, the value arrays contain only non-null values. Their position
/// arrays map each filtered value back to the original physical layout:
///
/// ```text
/// full left values:       [10, null, 20, 30]
/// left values:            [10, 20, 30]
/// left_positions:         [0,    2,  3]
/// left_null_positions:    [1]
/// ```
///
/// The full physical length is therefore `3 + 1 = 4`, not `left.len() == 3`.
/// `None` for a null-position array means that no nulls exist. `Some(empty)`
/// means that an explicit null-position array was supplied and contains zero
/// positions. Both contribute zero to the full length and produce the same
/// comparison behavior; PyJanitor normally uses `None` when no nulls exist.
///
/// # Forward and reverse output mapping
///
/// Forward aggregation reads values from the right source and writes one
/// output slot per left row. Reverse aggregation reads values from the left
/// source and writes one output slot per right row. For example, a successful
/// pair `(left_position=2, right_position=5)` updates as follows:
///
/// ```text
/// forward: set.update(source_position=5, output_position=2)
/// reverse: set.update(source_position=2, output_position=5)
/// ```
///
/// # Arguments
///
/// * `py` - The active Python interpreter token used to borrow NumPy arrays
///   and construct the result tuple.
/// * `left` - The left predicate values. For range predicates this is the
///   complete null-filtered left layout. For `!=` it contains only non-null
///   values, in the physical order described by `left_positions`.
/// * `right` - The right predicate values. For range predicates and `!=`,
///   PyJanitor supplies the value-sorted right layout required by binary
///   search. Rust trusts that ordering and does not sort it.
/// * `comparator` - One of `"<"`, `"<="`, `">"`, `">="`, or `"!="`.
///   Equality is handled upstream by PyJanitor.
/// * `left_positions` - For `!=`, the original physical position of each
///   non-null value in `left`; otherwise `None`.
/// * `left_null_positions` - Optional original physical positions of null
///   left values for NumPy-style `!=` semantics. Extension-array `!=` joins
///   exclude null candidates instead. `None` means no nulls exist.
/// * `right_positions` - For `!=`, the original physical position of each
///   non-null value in `right`; otherwise `None`.
/// * `right_null_positions` - Optional original physical positions of null
///   right values. `None` means no nulls exist.
/// * `is_extension_array` - Whether the `!=` comparison uses pandas nullable
///   extension-array semantics. Null candidates are handled differently for
///   extension arrays and NumPy arrays.
/// * `aggregations` - Non-empty Python aggregation requests. Each request is
///   `(values, null_mask, operation)`, where `operation` is `"sum"`,
///   `"count"`, `"size"`, `"prod"`, `"min"`, or `"max"`. The arrays and
///   masks must use the complete physical layout of the side being
///   aggregated, not the filtered predicate layout.
/// * `reverse` - If `false`, aggregate right values into left output slots.
///   If `true`, aggregate left values into right output slots.
///
/// # Returns
///
/// Returns `Some((matched, results))` when at least one comparison succeeds.
/// `matched` identifies output rows with at least one successful pair, and
/// `results` contains one array per requested aggregation in request order.
/// Returns `None` when no comparison succeeds anywhere.
///
/// # Numerical contract
///
/// Integer `sum` and `prod` use fixed-width wrapping arithmetic at the source
/// dtype width. This intentionally differs from pandas, which may promote an
/// integer result when an operation overflows. Floating-point aggregation
/// uses the corresponding `f32` or `f64` arithmetic. Arithmetic used for
/// positions, lengths, and allocation sizes remains checked.
///
/// # Errors
///
/// Returns a Python `ValueError` when equality is requested, required `!=`
/// position metadata is missing, position lengths overflow, an aggregation
/// request is invalid, or a boundary cannot be represented as `i64`.
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

    // `!=` receives filtered value arrays, so their lengths alone cannot size
    // the output. Reconstruct the complete physical domain from the
    // non-null partition and the optional null partition. For example,
    // `[0, 2, 3] + [1]` describes four original rows, not three.
    let left_full_len = if is_not_equal {
        let non_null = match left_positions.as_ref() {
            Some(values) => values.len()?,
            None => 0,
        };
        // `None` means there are no null rows. `Some(empty)` is also valid and
        // contributes zero; it is an explicit empty null partition.
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
        // Keep the same physical-length reconstruction for the right side.
        // The position arrays are trusted after the shared core validates
        // that their partitions are aligned and in bounds.
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
    // per left row. Reverse aggregation swaps those two roles. The source
    // length must match the full-layout aggregation arrays, while the output
    // length determines how many matched flags and result slots are emitted.
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
        ///
        /// Range values must already be aligned and the right values must be
        /// sorted by PyJanitor. `!=` values contain only non-null entries;
        /// their position arrays map them back to the complete physical
        /// layouts. The kernel consumes every successful pair, so there is no
        /// `keep` argument.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `left` / `right` - Predicate value arrays using the dtype encoded
        ///   by this exported function name.
        /// * `comparator` - One of `<`, `<=`, `>`, `>=`, or `!=`. Equality is
        ///   handled upstream.
        /// * `left_positions` / `right_positions` - Required physical maps
        ///   for `!=`; `None` for range predicates.
        /// * `left_null_positions` / `right_null_positions` - Optional null
        ///   physical partitions for `!=`.
        /// * `is_extension_array` - Selects pandas nullable null semantics for
        ///   `!=`; it must be false for range predicates.
        /// * `aggregations` - Non-empty value/mask/operation requests. Values
        ///   use the complete physical layout of the right source.
        ///
        /// # Example
        ///
        /// With `left = [2, 3]`, sorted `right = [1, 2, 4]`, and comparator
        /// `<`, the matching suffixes are `[4]` and `[4]`. A forward sum
        /// request updates one result slot per left row from right values.
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
        ///
        /// Reverse aggregation uses the same predicate and position contract
        /// as the forward function, but writes dense output slots for right
        /// rows while reading aggregation values from left rows. Single range
        /// predicates use `aggregate_reverse_starts` or
        /// `aggregate_reverse_ends`, which may select direct, event-sweep, or
        /// boundary-table reductions internally.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `left` / `right` - Aligned predicate arrays.
        /// * `comparator` - A supported non-equality comparator.
        /// * `left_positions` / `right_positions` - Physical maps required by
        ///   `!=` and otherwise omitted.
        /// * `left_null_positions` / `right_null_positions` - Optional null
        ///   partitions for `!=`.
        /// * `is_extension_array` - pandas nullable null-semantics flag for
        ///   `!=`.
        /// * `aggregations` - Requests over the complete left source layout.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair succeeds. Otherwise returns a matched
        /// mask indexed by right rows and aggregation arrays in request order.
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
