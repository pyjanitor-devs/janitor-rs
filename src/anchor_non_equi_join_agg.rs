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
use crate::anchor_non_equi_join::{build_range_core, range_window, visit_not_equal_pairs_core};
use crate::join_aggregation_helpers::{aggregate_range_windows, check_residual_lengths, residuals};
use crate::op::CompareOp;
use crate::predicate::{null_metadata_views, predicates_match_dispatch, PredicateView};

/// Build a lookup from original physical rows to compact aggregation slots.
///
/// A physical position is the row's original position in the full input.
/// A compact position is the row's position in the trimmed aggregation array;
/// each compact position is therefore an aggregation slot.
///
/// Before filtering or sorting, the physical input is:
///
/// ```text
/// physical row:      [0,    1,  2]
/// original values:   [null, 20, 10]
/// ```
///
/// PyJanitor supplies the aggregation values in compact sorted order:
///
/// ```text
/// compact slot:      [0,  1,  2]
/// physical row:      [2,  1,  0]
/// compact values:    [10, 20, null]
/// ```
///
/// Rust inverts that pairing to:
///
/// ```text
/// physical row:      [0,  1,  2]
/// compact slot:      [2,  1,  0]
/// ```
///
/// Thus, a candidate reported at physical row `2` is written to compact
/// aggregation slot `physical_to_slot[2]`, which is slot `0`.
///
/// This lets aggregation update the compact result directly without a later
/// scattering pass.
fn physical_to_local_positions(
    name: &str,
    full_len: usize,
    output_positions: ArrayView1<'_, i64>,
) -> Result<Vec<usize>, String> {
    if output_positions.len() != full_len {
        return Err(format!(
            "{name} output positions must cover the complete physical layout"
        ));
    }
    let mut local_positions = vec![usize::MAX; full_len];
    for (local, &physical) in output_positions.iter().enumerate() {
        let physical = usize::try_from(physical)
            .map_err(|_| format!("{name} output position must be non-negative"))?;
        if physical >= full_len {
            return Err(format!("{name} output position is out of bounds"));
        }
        if local_positions[physical] != usize::MAX {
            return Err(format!("{name} output positions contain duplicates"));
        }
        local_positions[physical] = local;
    }
    Ok(local_positions)
}

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
        if values.len()? != left.len() {
            return Err(PyValueError::new_err(
                "left position map must match left predicate values",
            ));
        }
    }
    if let Some(values) = right_positions.as_ref() {
        if values.len()? != right.len() {
            return Err(PyValueError::new_err(
                "right position map must match right predicate values",
            ));
        }
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
        let left_map = physical_to_local_positions(
            "left",
            left_full_len,
            left_output_positions.as_ref().unwrap().as_array(),
        )
        .map_err(PyValueError::new_err)?;
        let right_map = physical_to_local_positions(
            "right",
            right_full_len,
            right_output_positions.as_ref().unwrap().as_array(),
        )
        .map_err(PyValueError::new_err)?;
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
                    set.update(left_map[left_position], right_map[right_position]);
                } else {
                    set.update(right_map[right_position], left_map[left_position]);
                }
            },
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

/// Visit null-aware `!=` candidates for an extended join.
///
/// [`visit_not_equal_pairs_core`] generates the strict less-than and
/// greater-than candidates from the first `!=` predicate and adds null
/// candidates according to the NumPy or pandas extension-array contract.
/// Each candidate is passed through the residual predicates before it updates
/// the aggregation state.
///
/// The filtered value arrays contain only non-null values. `left_positions`
/// and `right_positions` map those values to the complete physical domains;
/// the optional null-position arrays complete those domains when nulls exist.
/// Separate output-position arrays describe the compact aggregation layouts.
/// Consequently, `left_full_len` and `right_full_len` describe the original
/// physical layouts, while `AggregationSet` uses compact layout lengths.
///
/// # Arguments
///
/// * `left` / `right` - Filtered, non-null first-predicate values.
/// * `left_full_len` / `right_full_len` - Full physical lengths used for
///   aggregation output and residual indexing.
/// * `left_positions` / `right_positions` - Physical positions of the
///   filtered values in the original layouts.
/// * `left_null_positions` / `right_null_positions` - Optional physical
///   positions of null rows.
/// * `left_output_positions` / `right_output_positions` - Complete physical
///   positions in the compact source/output layouts.
/// * `is_extension_array` - Selects pandas nullable versus NumPy null
///   comparison behavior.
/// * `residuals` - Parsed predicates after the first `!=` predicate.
/// * `residual_metadata` - Optional null metadata for residual predicates.
/// * `set` - Aggregation state updated for each surviving pair.
/// * `reverse` - Selects forward or reverse source/output slot mapping.
///
/// # Errors
///
/// Returns a string error when the physical position partitions are malformed.
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
    left_output_positions: ArrayView1<'_, i64>,
    right_output_positions: ArrayView1<'_, i64>,
    is_extension_array: bool,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
) -> Result<(), String> {
    let left_map = physical_to_local_positions("left", left_full_len, left_output_positions)?;
    let right_map = physical_to_local_positions("right", right_full_len, right_output_positions)?;
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
                    set.update(left_map[left_position], right_map[right_position]);
                } else {
                    set.update(right_map[right_position], left_map[left_position]);
                }
            }
        },
    )
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
    check_residual_lengths(&parsed, left_view.len(), right_view.len())?;
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
    )
}

/// Run fused aggregation for an all-`!=` extended join.
///
/// The first tuple uses the thirteen-element null-aware aggregation contract.
/// The eleven-element form belongs to index generation and is rejected here:
/// it does not contain the output-layout positions required by aggregation.
/// The thirteen-element form contains filtered value arrays and physical
/// position partitions for candidate generation, followed by complete
/// physical-to-compact output layouts. Every later tuple is a residual
/// predicate over the full physical layouts. No pair tape is materialized;
/// successful candidates update the aggregation state immediately.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - First null-aware `!=` tuple followed by residual tuples.
/// * `left` / `right` - Filtered, non-null values from the first `!=`
///   predicate. Their order and their position maps must agree.
/// * `left_index` / `right_index` - Complete physical index-label arrays;
///   their lengths define the physical domains used by residual predicates.
/// * `left_positions` / `right_positions` - Physical maps for filtered
///   non-null first-predicate values.
/// * `left_null_positions` / `right_null_positions` - Optional physical null
///   partitions.
/// * `is_extension_array` - Selects pandas extension-array null semantics.
/// * `aggregations` - Full-layout aggregation requests.
/// * `return_matched` - Include the per-output matched array when true.
/// * `reverse` - Selects right-oriented output and left-side source values.
///
/// # Returns
///
/// Returns `None` when no pair survives every predicate; otherwise returns
/// `(output_positions, matched, aggregation_arrays)` when `return_matched` is
/// true, or `(output_positions, aggregation_arrays)` when it is false.
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
    left_output_positions: PyReadonlyArray1<'py, i64>,
    right_output_positions: PyReadonlyArray1<'py, i64>,
    is_extension_array: bool,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let (parsed, metadata) = residuals(py, predicates, true, false)?;
    check_residual_lengths(&parsed, left_index.len()?, right_index.len()?)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let left_full_len = left_index.len()?;
    let right_full_len = right_index.len()?;
    let output_len = if reverse {
        right_output_positions.len()?
    } else {
        left_output_positions.len()?
    };
    let source_len = if reverse {
        left_output_positions.len()?
    } else {
        right_output_positions.len()?
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;
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
        left_output_positions.as_array(),
        right_output_positions.as_array(),
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
    // The accumulator is stored in the trimmed layout described by the
    // selected output map. Returning `None` here would make the Python side
    // fabricate an identity map and silently mislabel reordered `!=` results.
    let output_positions = if reverse {
        right_output_positions.as_array()
    } else {
        left_output_positions.as_array()
    };
    Ok(Some(make_results_with_positions(
        py,
        set,
        Some(output_positions),
        output_len,
        return_matched,
    )?))
}

/// Validate the first extended predicate and dispatch to its fused traversal.
///
/// The first tuple is the algorithm anchor. A six- or eight-element tuple is a
/// range anchor; a thirteen-element tuple is the null-aware all-`!=`
/// aggregation anchor with output-layout positions. The eleven-element
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
///  left_output_positions, right_output_positions,
///  right_index_is_ordered, comparator)
///
/// 13 fields for `!=` aggregation:
/// (left_values, left_index, left_positions, left_null_positions,
///  right_values, right_index, right_positions, right_null_positions,
///  right_index_is_ordered, is_extension_array,
///  left_output_positions, right_output_positions, comparator)
/// ```
///
/// In the six/eight-field forms, the first range predicate creates a window
/// and later predicates are residual filters. In the thirteen-field form,
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
    // Aggregation has three supported anchor layouts:
    //
    // * 6 elements: the compact range contract;
    // * 8 elements: the range contract plus trimmed output-position maps;
    // * 13 elements: the null-aware `!=` aggregation contract.
    //
    // The 11-element `!=` tuple is intentionally absent. It is the index-only
    // contract and lacks the maps needed to place aggregation results safely.
    // Rejecting it here prevents identity-position output from appearing
    // correct when the physical and compact layouts differ.
    // The length check must happen before any field extraction because the
    // field positions differ between range and null-aware `!=` tuples.
    match first.len() {
        6 | 8 | 13 => {}
        _ => {
            return Err(PyValueError::new_err(
                "the first extended aggregation predicate must contain 6, 8, or 13 elements",
            ));
        }
    }
    // Every accepted anchor puts its comparator in the final field. Reading
    // it once after structural validation keeps tuple-length handling separate
    // from operator validation and avoids one bespoke opcode branch per shape.
    let op = CompareOp::try_from_str(first.get_item(first.len() - 1)?.extract::<&str>()?)?;
    if first.len() != 13 {
        // Six- and eight-field tuples are range anchors. Equality is handled
        // upstream, and `!=` requires the separate null-aware thirteen-field
        // layout, so neither comparator is valid in this branch.
        if !matches!(
            op,
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
        ) {
            return Err(PyValueError::new_err(
                "the range aggregation predicate must use <, <=, >, or >=",
            ));
        }
        // The ordering flag is part of the shared predicate tuple contract.
        // Aggregation does not use its value: PyJanitor has already sorted the
        // right-hand arrays before calling Rust. Extract it only to validate
        // the tuple shape and field type.
        first.get_item(4)?.extract::<bool>()?;
        // The ordering flag is field four in both range forms. It is part of
        // the shared tuple contract, but aggregation does not use its value:
        // PyJanitor has already sorted the right arrays. Extracting it here
        // validates only the field type.
        let (left_output_positions, right_output_positions) = if first.len() == 8 {
            // In the eight-element range form, fields 1 and 3 identify the
            // predicate arrays' physical labels. Fields 5 and 6 identify the
            // trimmed output layout used by aggregation and returned to
            // Python; they must not be substituted for one another.
            (
                Some(first.get_item(5)?.extract::<PyReadonlyArray1<'py, i64>>()?),
                Some(first.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?),
            )
        } else {
            (None, None)
        };
        // The eight-field form describes a trimmed/reordered aggregation
        // layout. Its maps are compact-slot -> physical-row mappings, so their
        // lengths—not the label values—define the output domains. With six
        // fields, the value-array lengths already describe identity layouts.
        let (left_output_len, right_output_len) =
            if let (Some(left_output_positions), Some(right_output_positions)) =
                (&left_output_positions, &right_output_positions)
            {
                (left_output_positions.len()?, right_output_positions.len()?)
            } else {
                (
                    first
                        .get_item(0)?
                        .extract::<PyReadonlyArray1<'py, T>>()?
                        .len()?,
                    first
                        .get_item(2)?
                        .extract::<PyReadonlyArray1<'py, T>>()?
                        .len()?,
                )
            };
        // Forward aggregation writes one result per left output slot; reverse
        // aggregation writes one result per right output slot. Select only
        // the map for the requested orientation and leave the other map for a
        // separate reverse/forward call.
        let calculation_output_positions = match (
            reverse,
            left_output_positions.as_ref(),
            right_output_positions.as_ref(),
        ) {
            (true, _, Some(right_output_positions)) => Some(right_output_positions.as_array()),
            (false, Some(left_output_positions), _) => Some(left_output_positions.as_array()),
            _ => None,
        };
        // `run_range` builds the first window and applies every later
        // predicate as a residual filter before updating aggregations.
        return run_range(
            py,
            predicates,
            first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
            first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
            first.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?,
            first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?,
            op,
            aggregations,
            calculation_output_positions,
            if reverse {
                right_output_len
            } else {
                left_output_len
            },
            return_matched,
            reverse,
        );
    }
    // Reaching this point means the first tuple has thirteen fields. That
    // layout is reserved for null-aware `!=`; accepting another comparator
    // would interpret range metadata as null metadata.
    if op != CompareOp::Ne {
        return Err(PyValueError::new_err(
            "the thirteen-element aggregation predicate must use !=",
        ));
    }
    // Null-position fields are optional, but each field must be either Python
    // None or an int64 array. These positions complete the filtered
    // non-null positions to form the full physical row domain.
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
    // Output maps are mandatory for `!=` aggregation because filtered and
    // sorted compact arrays may no longer be in physical order. Returning
    // identity positions here would silently attach an aggregate to the
    // wrong original row.
    let left_output_positions = first
        .get_item(10)?
        .extract::<PyReadonlyArray1<'py, i64>>()?;
    let right_output_positions = first
        .get_item(11)?
        .extract::<PyReadonlyArray1<'py, i64>>()?;
    // `run_not_equal` generates strict less-than/greater-than and null pairs,
    // checks every residual `!=` predicate, and updates aggregation state
    // directly without materializing a pair list.
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
        left_output_positions,
        right_output_positions,
        first.get_item(9)?.extract::<bool>()?,
        aggregations,
        return_matched,
        reverse,
    )
}

macro_rules! extended_aggregation_functions {
    ($forward:ident, $reverse:ident, $ty:ty) => {
        /// Fused forward aggregation for multiple conditional-join
        /// predicates.
        ///
        /// The first predicate must be either a range comparator (`<`, `<=`,
        /// `>`, `>=`) or the null-aware thirteen-element `!=` aggregation
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
        ///   it has the same six/eight/thirteen-element anchor contract as the
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
                    true.into_pyobject(py)?.to_owned().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
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
    fn not_equal_aggregation_maps_reordered_physical_positions_to_compact_slots() {
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
                    false.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 0]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1]).into_any(),
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
            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![1, 0]);
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, true]
            );
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<i64>>()?, vec![100, 200]);
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
                    true.into_pyobject(py)?.to_owned().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
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
                "ValueError: all-!= joins require every predicate to use !="
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
                "ValueError: all-!= joins require every predicate to use !="
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
                    true.into_pyobject(py)?.to_owned().into_any(),
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
                "ValueError: the first extended aggregation predicate must contain 6, 8, or 13 elements"
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
