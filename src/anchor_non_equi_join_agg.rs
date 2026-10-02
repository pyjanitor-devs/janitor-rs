//! Fused aggregation for one conditional-join predicate.
//!
//! This module mirrors `anchor_non_equi_join.rs`'s candidate traversal but updates
//! `AggregationSet` immediately instead of building left/right index arrays.
//! `keep` is intentionally absent: aggregation consumes every pair that
//! satisfies the comparison.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::aggs::ensure_equal_lengths_core;
use crate::anchor_non_equi_join::build_range_core;
use crate::common::range_window;
use crate::join_aggregation_helpers::{aggregate_range_windows, residuals};
use crate::not_equals_only::aggregate_not_equal;
use crate::op::CompareOp;
use crate::predicate::check_predicate_lengths;
use crate::range_predicate::{parse_aggregation_range_anchor, AnyParsedRangePredicate};

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
///   masks use the compact source layout supplied by PyJanitor.
/// * `left_output_positions` - For `!=`, the compact left layout positions.
///   It contains every physical position exactly once, normally with
///   non-null positions followed by null positions.
/// * `right_output_positions` - For `!=`, the corresponding compact right
///   layout positions. Range calls retain these arguments for the common
///   wrapper signature; range output alignment comes from its calculation map.
/// * `return_matched` - Whether to include the per-output boolean matched
///   array. When false, the return shape is
///   `(output_positions, aggregation_arrays)` instead of
///   `(output_positions, matched, aggregation_arrays)`.
/// * `reverse` - If `false`, aggregate right values into left output slots.
///   If `true`, aggregate left values into right output slots.
///
/// # Returns
///
/// Returns `Some((output_positions, matched, results))` when at least one
/// comparison succeeds. The first array has the same trimmed calculation
/// order as the returned aggregation arrays; `results` contains one array per
/// requested aggregation in request order.
/// Returns `None` when no comparison succeeds anywhere.
///
/// # Numerical contract
///
/// PyJanitor normalizes numeric aggregation inputs before this boundary:
/// signed integer `sum` and `prod` use `int64`, unsigned inputs use `uint64`,
/// and floating inputs retain their source dtype (`float32` or `float64`) at
/// the PyJanitor materialization boundary. `min` and `max` retain the source
/// dtype. Arithmetic used for positions, lengths, and allocation sizes remains
/// checked.
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
    left_output_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_output_positions: Option<PyReadonlyArray1<'py, i64>>,
    return_matched: bool,
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
        && (left_null_positions.is_some() || right_null_positions.is_some() || is_extension_array)
    {
        return Err(PyValueError::new_err(
            "null metadata is only supported for != aggregation",
        ));
    }
    if is_not_equal && (left_positions.is_none() || right_positions.is_none()) {
        return Err(PyValueError::new_err(
            "left and right positions are required for != aggregation",
        ));
    }

    // `!=` receives filtered value arrays, so their lengths alone cannot size
    // the physical domain. Reconstruct it from the non-null partition and the
    // optional null partition. For example, `[0, 2, 3] + [1]` describes four
    // original rows, not three. The output-position arrays are compact layouts
    // and therefore must not be used as physical-domain lengths.
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

    if let Some(values) = left_positions.as_ref() {
        ensure_equal_lengths_core(
            "left predicate values",
            left.len(),
            "left position map",
            values.len()?,
        )
        .map_err(PyValueError::new_err)?;
    }
    if let Some(values) = right_positions.as_ref() {
        ensure_equal_lengths_core(
            "right predicate values",
            right.len(),
            "right position map",
            values.len()?,
        )
        .map_err(PyValueError::new_err)?;
    }

    // Forward aggregation reads values from right and writes one trimmed
    // result slot per calculation-order left row. Reverse aggregation swaps
    // those roles. Range inputs have already been trimmed by PyJanitor, so
    // their dense optimized state must use the trimmed lengths directly.
    let output_positions = if is_not_equal {
        Some(if reverse {
            right_output_positions
                .as_ref()
                .ok_or_else(|| {
                    PyValueError::new_err("right output positions are required for != aggregation")
                })?
                .as_array()
        } else {
            left_output_positions
                .as_ref()
                .ok_or_else(|| {
                    PyValueError::new_err("left output positions are required for != aggregation")
                })?
                .as_array()
        })
    } else {
        None
    };
    let output_len = if is_not_equal {
        if reverse {
            right_output_positions.as_ref().unwrap().len()?
        } else {
            left_output_positions.as_ref().unwrap().len()?
        }
    } else if reverse {
        right.len()
    } else {
        left.len()
    };
    let source_len = if is_not_equal {
        if reverse {
            left_output_positions.as_ref().unwrap().len()?
        } else {
            right_output_positions.as_ref().unwrap().len()?
        }
    } else if reverse {
        left.len()
    } else {
        right.len()
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    if is_not_equal {
        aggregate_not_equal(
            left,
            left_full_len,
            left_positions.as_ref().unwrap().as_array(),
            left_null_positions.as_ref().map(|values| values.as_array()),
            right,
            right_full_len,
            right_positions.as_ref().unwrap().as_array(),
            right_null_positions
                .as_ref()
                .map(|values| values.as_array()),
            is_extension_array,
            &[],
            None,
            &mut set,
            reverse,
        )
        .map_err(PyValueError::new_err)?;
    } else {
        // A single range join produces one contiguous right-side window per
        // trimmed left row. Reuse the optimized prefix/suffix aggregation
        // paths instead of visiting every matching position. Reverse methods
        // write directly into the trimmed right calculation layout.
        let mut boundaries = Vec::with_capacity(left.len());
        for &left_value in left.iter() {
            let (start, end) = range_window(left_value, right, op);
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
        let output_positions = if reverse {
            right_positions.as_ref().map(|values| values.as_array())
        } else {
            left_positions.as_ref().map(|values| values.as_array())
        };
        return if set.is_empty() {
            Ok(None)
        } else {
            Ok(Some(make_results_with_positions(
                py,
                set,
                output_positions,
                output_len,
                return_matched,
            )?))
        };
    }

    if set.is_empty() {
        return Ok(None);
    }
    Ok(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?))
}

macro_rules! single_join_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Fused forward aggregation for one range or `!=` predicate.
        ///
        /// Range values must already be aligned and the right values must be
        /// sorted by PyJanitor. `!=` values contain only non-null entries;
        /// their position arrays map them back to the complete physical
        /// layouts, while aggregation sources use the compact layout. The
        /// kernel consumes every successful pair, so there is no
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
        ///   use the compact calculation layout of the source side.
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
            left_output_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_output_positions: Option<PyReadonlyArray1<'py, i64>>,
            return_matched: bool,
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
                left_output_positions,
                right_output_positions,
                return_matched,
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
            left_output_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_output_positions: Option<PyReadonlyArray1<'py, i64>>,
            return_matched: bool,
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
                left_output_positions,
                right_output_positions,
                return_matched,
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

/// Run aggregation for one range anchor followed by residual predicates.
///
/// Predicate one creates the binary-search window. Predicate two may be
/// absent; every predicate after the first is filtered inside that window.
/// This is the single-anchor extended-join contract.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Complete predicate list. Predicate one is the range
///   anchor; every later predicate is a residual filter.
/// * `left` / `right` - Null-free aligned anchor values. `right` must already
///   be sorted ascending by PyJanitor.
/// * `left_index` / `right_index` - Index labels aligned with the anchor
///   arrays.
/// * `op` - Anchor comparator: `<`, `<=`, `>`, or `>=`.
/// * `aggregations` - Non-empty aggregation requests.
/// * `output_positions` - Optional compact-slot-to-physical-row map.
/// * `output_len` - Number of output slots.
/// * `return_matched` - Include the output match mask when true.
/// * `reverse` - Aggregate left source values into right output slots when
///   true; otherwise aggregate right source values into left output slots.
///
/// # Returns
///
/// Returns the standard aggregation tuple, or `None` when no fully matching
/// candidate remains.
///
/// # Errors
///
/// Returns a Python `ValueError` for residual length mismatches, invalid
/// aggregation requests, or invalid output-position metadata.
#[allow(clippy::too_many_arguments)]
fn run_range<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left: PyReadonlyArray1<'py, T>,
    left_index: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    op: CompareOp,
    aggregations: &Bound<'py, PyList>,
    output_positions: Option<ArrayView1<'_, i64>>,
    output_len: usize,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (parsed, metadata) = residuals(py, predicates, false, false)?;
    let left_view = left.as_array();
    let right_view = right.as_array();
    check_predicate_lengths(&parsed, left_view.len(), right_view.len())?;
    let windows = build_range_core(
        left_view,
        left_index.as_array(),
        right_view,
        right_index.as_array(),
        false,
        op,
    )
    .map_err(PyValueError::new_err)?;
    aggregate_range_windows(
        py,
        windows,
        &parsed,
        metadata.as_deref(),
        aggregations,
        output_positions,
        output_len,
        if reverse {
            left_view.len()
        } else {
            right_view.len()
        },
        return_matched,
        reverse,
        false,
    )
}

/// Validate the first extended predicate and dispatch to its fused traversal.
///
/// The first tuple is the algorithm anchor. A six- or eight-element tuple is a
/// range anchor; a ten-element tuple is the null-aware all-`!=`
/// aggregation anchor. The eleven-element
/// all-`!=` tuple belongs to index generation and is rejected here. The tuple
/// shape is deliberately checked before any aggregation state or candidate
/// loop is created.
///
/// ELI5: this function is the traffic controller for the general extended
/// aggregation API. It looks only at the first tuple to decide how candidate
/// pairs should be produced. Once that choice is made, the selected helper
/// checks the remaining predicates and updates aggregation state.
///
/// The accepted first-tuple shapes are:
///
/// ```text
/// 6 fields:
/// (left, left_index, right, right_index,
///  right_index_is_ordered, comparator)
///
/// 8 fields:
/// (left, left_index, right, right_index,
///  right_index_is_ordered, left_output_positions,
///  right_output_positions, comparator)
///
/// 10 fields for `!=` aggregation:
/// (left_values, left_index, left_positions, left_null_positions,
///  right_values, right_index, right_positions, right_null_positions,
///  is_extension_array, comparator)
/// ```
///
/// In the six/eight-field forms, the first range predicate creates a window
/// and later predicates are residual filters. In the ten-field form,
/// the first predicate creates the null-aware `!=` candidate stream and every
/// later predicate must also be `!=`.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Complete extended predicate list. It must contain at
///   least the anchor predicate; it may contain no residual predicates.
/// * `first` - The first predicate tuple, already extracted from `predicates`.
/// * `aggregations` - Non-empty aggregation requests.
/// * `return_matched` - Include the per-output matched array when true.
/// * `reverse` - Selects forward or reverse output mapping.
///
/// # Returns
///
/// Returns `(output_positions, matched, aggregation_arrays)` when
/// `return_matched` is true, or `(output_positions, aggregation_arrays)` when
/// it is false. Returns `None` when no candidate survives.
#[allow(clippy::too_many_arguments)]
fn dispatch<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    first: &Bound<'py, PyTuple>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // An empty predicate list has no anchor and therefore no way to generate
    // candidates. Reject it before reading `first` so Python receives the
    // intended ValueError rather than a lower-level indexing error.
    // This aggregation dispatcher also serves the ordinary one-predicate
    // anchor API, so one predicate is valid here. The extended index API has
    // a different contract and requires an anchor plus at least one residual;
    // its `len() < 2` validation must remain in that separate wrapper.
    if predicates.is_empty() {
        return Err(PyValueError::new_err(
            "single extended aggregation requires at least one predicate",
        ));
    }
    if first.len() == 10 {
        // The ten-field shape is a distinct null-aware `!=` protocol.
        // Dispatch it before range-shape validation so its specialized maps
        // and null partitions cannot be mistaken for range output metadata.
        return crate::not_equals_only::dispatch_not_equal_aggregation::<T>(
            py,
            predicates,
            first,
            aggregations,
            return_matched,
            reverse,
        );
    }

    // Preserve the anchor aggregation API's established shape error before
    // handing valid range forms to the shared named parser. In particular,
    // an index-only eleven-field `!=` tuple must not be reported as a generic
    // range-anchor error; callers rely on this contract when diagnosing a
    // malformed extended aggregation request.
    if !matches!(first.len(), 6 | 8) {
        return Err(PyValueError::new_err(
            "the first extended aggregation predicate must contain 6, 8, or 10 elements",
        ));
    }

    // This function is instantiated once per public NumPy dtype. The shared
    // parser must inspect the tuple's runtime dtype so it can serve region
    // and range callers, but this wrapper also has a compiled `T` contract.
    // Extracting the first value array here preserves the old behavior: a
    // call through the int64 wrapper with int32 predicate values fails at the
    // Python boundary instead of silently selecting a different dispatch arm.
    // This is an intentional second boundary check: parse_any_range_parts
    // chooses a runtime variant for shared callers, while this wrapper must
    // reject a tuple whose dtype disagrees with the exported function name.
    first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let anchor = parse_aggregation_range_anchor(first, true)?;
    let left_output_len = anchor
        .left_output_positions
        .as_ref()
        .map(|values| values.len())
        .transpose()?
        .unwrap_or(anchor.range.left_len());
    let right_output_len = anchor
        .right_output_positions
        .as_ref()
        .map(|values| values.len())
        .transpose()?
        .unwrap_or(anchor.range.right_len());
    let calculation_output_positions = if reverse {
        anchor
            .right_output_positions
            .as_ref()
            .map(|values| values.as_array())
    } else {
        anchor
            .left_output_positions
            .as_ref()
            .map(|values| values.as_array())
    };
    let output_len = if reverse {
        right_output_len
    } else {
        left_output_len
    };
    // `run_range` is generic over the concrete NumPy element type, while the
    // Python boundary gives us one runtime `AnyParsedRangePredicate`. The
    // match recovers that concrete type without converting columns into a
    // common temporary dtype or adding dynamic dispatch to the hot path.
    // The small local macro removes argument-list drift between the ten dtype
    // arms while leaving the type-specialized match visible at the boundary.
    macro_rules! run_range_for {
        ($value:expr) => {
            run_range(
                py,
                predicates,
                $value.left,
                $value.left_index,
                $value.right,
                $value.right_index,
                $value.op,
                aggregations,
                calculation_output_positions,
                output_len,
                return_matched,
                reverse,
            )
        };
    }
    match anchor.range {
        AnyParsedRangePredicate::I64(value) => run_range_for!(value),
        AnyParsedRangePredicate::I32(value) => run_range_for!(value),
        AnyParsedRangePredicate::I16(value) => run_range_for!(value),
        AnyParsedRangePredicate::I8(value) => run_range_for!(value),
        AnyParsedRangePredicate::U64(value) => run_range_for!(value),
        AnyParsedRangePredicate::U32(value) => run_range_for!(value),
        AnyParsedRangePredicate::U16(value) => run_range_for!(value),
        AnyParsedRangePredicate::U8(value) => run_range_for!(value),
        AnyParsedRangePredicate::F64(value) => run_range_for!(value),
        AnyParsedRangePredicate::F32(value) => run_range_for!(value),
    }
}

macro_rules! extended_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Fused forward aggregation for multiple conditional-join
        /// predicates.
        ///
        /// The first predicate must be either a range comparator (`<`, `<=`,
        /// `>`, `>=`) or the null-aware twelve-element `!=` aggregation
        /// form. The eleven-element form belongs to index generation.
        /// Remaining predicates are aligned residual filters. Aggregation is
        /// performed as candidates pass the first predicate and all residual
        /// filters; no intermediate pair indices are returned or allocated.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Python list of aligned predicate tuples. It must
        ///   contain at least one tuple; the first tuple is the range or
        ///   `!=` anchor and later tuples are residual filters.
        /// * `aggregations` - Non-empty aggregation requests in the shared
        ///   `(values, null_mask, operation)` format, or wildcard count/size
        ///   requests. Values use the complete right-side physical layout in
        ///   forward mode.
        /// * `return_matched` - Include a boolean result slot for every output
        ///   row when true. The slot is true if at least one complete
        ///   predicate match contributed to that row.
        ///
        /// # Example
        ///
        /// Conceptually, for `left < right` followed by `left2 != right2`,
        /// Rust first finds the sorted-right range and then updates the
        /// aggregation only for candidates passing `left2 != right2`.
        ///
        /// # Returns
        ///
        /// Returns `None` when no complete pair survives. Otherwise returns
        /// output positions, optionally the matched mask, and one result array
        /// per aggregation request.
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            if predicates.is_empty() {
                return Err(PyValueError::new_err(
                    "single extended aggregation requires at least one predicate",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch::<$ty>(py, predicates, &first, aggregations, return_matched, false)
        }

        /// Fused reverse aggregation for multiple conditional-join
        /// predicates.
        ///
        /// This has the same predicate contract as the forward entry point,
        /// but aggregation values come from the left physical layout and
        /// output slots are indexed by right rows. A simple range anchor uses
        /// the optimized reverse boundary kernels; residual-filtered extended
        /// candidates are updated individually because a residual may reject
        /// arbitrary members of an anchor window.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Python list of aligned anchor and residual tuples;
        ///   it has the same six/eight/twelve-element anchor contract as the
        ///   forward function.
        /// * `aggregations` - Non-empty aggregation requests over the complete
        ///   left-side physical layout.
        /// * `return_matched` - Include the per-output matched mask when true.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair survives. Otherwise returns output
        /// positions, optionally a matched mask, and one result array per
        /// aggregation request.
        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            if predicates.is_empty() {
                return Err(PyValueError::new_err(
                    "single extended aggregation requires at least one predicate",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch::<$ty>(py, predicates, &first, aggregations, return_matched, true)
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

#[cfg(test)]
mod extended_tests {
    use super::*;
    use numpy::PyArray1;

    #[test]
    fn single_range_aggregation_accepts_one_predicate() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // With only one predicate there is no residual tuple to parse:
            // the range window itself is the complete join condition.
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

            let result =
                single_join_extended_aggregate_int64(py, &predicates, &aggregations, true)?
                    .expect("the one-predicate range has matches");
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70]);
            Ok(())
        })
        .unwrap();
    }

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
            let result =
                single_join_extended_aggregate_int64(py, &predicates, &aggregations, true)?
                    .expect("the filtered range has matches");
            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true]);
            let outputs_value = result.get_item(2)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![70]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_rejects_values_mismatched_with_compiled_wrapper_dtype() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i32]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i32, 5]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
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
            let error =
                match single_join_extended_aggregate_int64(py, &predicates, &aggregations, true) {
                    Ok(_) => panic!("the int64 wrapper must reject int32 range values"),
                    Err(error) => error,
                };
            assert_eq!(error.get_type(py).name()?, "TypeError");
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn range_aggregation_uses_eight_element_output_position_fields() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![5_i64, 7]).into_any(),
                    PyArray1::from_vec(py, vec![400_i64, 300]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 0]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64, 6]).into_any(),
                    PyArray1::from_vec(py, vec![5_i64, 7]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
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
            let result =
                single_join_extended_aggregate_int64(py, &predicates, &aggregations, true)?
                    .expect("the range join has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![1, 0]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![30, 20]);
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
            let result =
                single_join_extended_aggregate_int64(py, &predicates, &aggregations, true)?
                    .expect("the not-equal join has matches");
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(2)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![200, 300]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn not_equal_aggregation_uses_full_physical_positions() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 0]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 0]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
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
            let result =
                single_join_extended_aggregate_int64(py, &predicates, &aggregations, true)?
                    .expect("the reordered not-equal layout has matches");
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![0, 1]);
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![200, 100]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn not_equal_aggregation_rejects_non_not_equal_residuals() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let aggregations = PyList::empty(py);

            let error = single_join_extended_aggregate_int64(
                py,
                &predicates,
                &aggregations,
                true,
            )
                .expect_err("mixed operators after a != anchor must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: not_equals aggregation requires every predicate to use !="
            );

            let error =
                single_join_extended_aggregate_reverse_int64(
                    py,
                    &predicates,
                    &aggregations,
                    true,
                )
                    .expect_err("reverse mixed operators after a != anchor must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: not_equals aggregation requires every predicate to use !="
            );

            let legacy_predicate = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?;
            let legacy_predicates = PyList::new(py, [legacy_predicate])?;
            let legacy_predicates = {
                let predicates = PyList::empty(py);
                predicates.append(legacy_predicates.get_item(0)?)?;
                predicates.append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        "!=".into_pyobject(py)?.into_any(),
                    ],
                )?)?;
                predicates
            };
            let error = single_join_extended_aggregate_int64(
                py,
                &legacy_predicates,
                &aggregations,
                true,
            )
            .expect_err("the index-only eleven-element tuple must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: the first extended aggregation predicate must contain 6, 8, or 10 elements"
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn empty_extended_aggregation_predicates_raise_value_error() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let aggregations = PyList::empty(py);

            let error = single_join_extended_aggregate_int64(py, &predicates, &aggregations, true)
                .expect_err("an empty forward predicate list must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: single extended aggregation requires at least one predicate"
            );

            let error =
                single_join_extended_aggregate_reverse_int64(py, &predicates, &aggregations, true)
                    .expect_err("an empty reverse predicate list must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: single extended aggregation requires at least one predicate"
            );
            Ok(())
        })
        .unwrap();
    }
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
                None,
                None,
                true,
            )?
            .expect("the range has matching candidates");
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(2)?;
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
                None,
                None,
                true,
            )?
            .expect("the range has matching candidates");
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_value = result.get_item(2)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![30, 30]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn promoted_integer_sum_and_product_use_pandas_widths() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = PyArray1::from_vec(py, vec![1_u64]);
            let right = PyArray1::from_vec(py, vec![2_u64, 3]);
            let values = PyArray1::from_vec(py, vec![250_u64, 10]);
            let mask = PyArray1::from_vec(py, vec![false, false]);
            let sum = PyTuple::new(
                py,
                [
                    values.clone().into_any(),
                    mask.clone().into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let product = PyTuple::new(
                py,
                [
                    values.into_any(),
                    mask.into_any(),
                    "prod".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [sum, product])?;
            let result = single_join_aggregate_uint64(
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
                None,
                None,
                true,
            )?
            .expect("the range has matching candidates");
            let outputs_value = result.get_item(2)?;
            let outputs = outputs_value.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<u64>>()?, vec![260]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<u64>>()?, vec![2500]);
            Ok(())
        })
        .unwrap();
    }
}
