//! Null-aware all-`!=` candidate generation.
//!
//! The Python side prepares the value arrays and their maps back to the full
//! physical layouts. This module only validates those maps, generates the
//! disjoint candidate partitions, and applies `keep` to physical positions.
//!
//! ## Cross-language layout contract
//!
//! The Python boundary supplies compact non-null value arrays plus physical
//! position maps. A position is always an offset into the original full row
//! layout; it is never an offset into a sorted or compacted array. The anchor
//! used by the extended APIs has this ten-field ABI:
//!
//! ```text
//! (left_values, left_full_positions, left_non_null_positions,
//!  left_null_positions, right_values, right_full_positions,
//!  right_non_null_positions, right_null_positions, is_extension_array, "!=")
//! ```
//!
//! Residual predicates address the full physical layouts directly. This is
//! why the extended path does not accept compacted residual arrays and why the
//! aggregation path updates accumulators with physical positions.
//!
//! ## Null semantics
//!
//! A NumPy null is represented by the separate null-position partition and is
//! allowed to match every row on the other side. A pandas extension-array null
//! is not a valid `!=` match, which is carried by `is_extension_array`. The
//! value arrays contain only non-null rows in both cases.
//!
//! ## Execution paths
//!
//! The direct path visits candidates and applies `keep` while emitting index
//! pairs. The extended path always visits the anchor with `Keep::All`, applies
//! residual predicates, and only then applies the requested `keep`. The
//! aggregation paths use the same traversal but update `AggregationSet`
//! directly, avoiding an intermediate candidate-pair allocation.
//!
//! ## Why the boundary is split this way
//!
//! Pandas owns labels, extension dtypes, and column selection, so Python must
//! construct the physical layouts before calling Rust. Rust cannot infer the
//! generic comparison type from a Python list of heterogeneous aggregation
//! requests: each exported PyO3 function is specialized for one anchor type
//! (`i64`, `f64`, and so on). Python therefore selects one anchor kernel by
//! dtype, while Rust handles every aggregation request in that invocation.
//!
//! The right anchor is sorted only to make the two `partition_point` searches
//! possible. Sorting must not rewrite position arrays. The maps preserve the
//! original physical rows so residual predicates and aggregation columns can
//! still be indexed correctly. Reverse aggregation returns buffers in the
//! physical accumulator layout; Python applies the right-anchor permutation
//! when materializing the public result.
//!
//! Keeping null positions separate avoids manufacturing sentinel values that
//! could compare as real data. It also lets the traversal distinguish NumPy
//! null behavior from pandas extension-array null behavior without changing
//! the compact value arrays.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggregation_common::aggregation::{
    make_results_with_positions, parse_inputs, AggregationSet,
};
use crate::aggregation_common::ensure_equal_lengths_core;
use crate::compare_op::CompareOp;
use crate::join_search::partition_point;
use crate::join_types::{result_dict, Keep};
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, NullMetadataView, PredicateView,
};

/// Validate that non-null and null physical positions form a complete,
/// disjoint partition of one full input layout.
fn validate_position_partition(
    name: &str,
    full_len: usize,
    non_null: &[usize],
    null: &[usize],
) -> Result<(), String> {
    validate_partition(name, full_len, non_null, null)
}

/// Validate the optional-array presence contract for one `!=` input side.
///
/// Values and non-null positions are paired because the position map belongs
/// to the compact value array. An all-null side has neither of those, but it
/// must still provide a null-position partition.
fn validate_not_equal_partition_presence(
    context: &str,
    side: &str,
    values_present: bool,
    non_null_present: bool,
    null_present: bool,
) -> Result<(), String> {
    if values_present != non_null_present {
        return Err(format!(
            "{context} requires {side} values and {side} non-null positions together"
        ));
    }
    if !non_null_present && !null_present {
        return Err(format!(
            "{context} requires a {side} non-null or null position partition"
        ));
    }
    Ok(())
}

/// Convert an optional compact NumPy array to a view, using an empty view for
/// an all-null side. Null rows are represented by the separate null-position
/// partition and are not fabricated as values.
fn optional_array_view<'a, 'py, T: numpy::Element>(
    values: Option<&'a PyReadonlyArray1<'py, T>>,
    empty: ArrayView1<'a, T>,
) -> ArrayView1<'a, T> {
    values.map(|values| values.as_array()).unwrap_or(empty)
}

/// Validate one side of a null-aware `!=` input.
///
/// This checks value/map alignment and delegates bounds, duplicate detection,
/// and the `non-null + null == full` invariant to the partition validator.
fn validate_not_equal_side<T: Copy>(
    name: &str,
    full_len: usize,
    values: ArrayView1<'_, T>,
    non_null_positions: ArrayView1<'_, i64>,
    null_positions: Option<ArrayView1<'_, i64>>,
) -> Result<(), String> {
    if full_len == 0 {
        return Err(format!(
            "{name} full positions must contain at least one row"
        ));
    }
    ensure_equal_lengths_core(
        &format!("{name} values"),
        values.len(),
        &format!("{name} non-null positions"),
        non_null_positions.len(),
    )?;
    let non_null_positions = positions(&format!("{name} non-null"), non_null_positions)?;
    let null_positions = null_positions
        .map(|values| positions(&format!("{name} null"), values))
        .transpose()?;
    let empty = Vec::new();
    validate_position_partition(
        &format!("{name} full positions"),
        full_len,
        &non_null_positions,
        null_positions.as_deref().unwrap_or(&empty),
    )
}

/// Visit every physical pair satisfying the null-aware `!=` comparison.
///
/// `left` and `right` are compact non-null arrays; `right` must be sorted.
/// The position maps translate compact offsets into the original physical
/// layouts. Null positions are supplied separately because null values are not
/// stored in the compact arrays.
///
/// The callback receives physical `(left_position, right_position)` pairs.
/// This function never allocates a candidate-pair vector.
///
/// # Errors
///
/// Returns an error for misaligned values and maps, invalid positions,
/// duplicate positions, or incomplete full-layout partitions.
///
/// # Arguments
///
/// * `left` and `right` are compact non-null value arrays. `right` is sorted
///   in ascending order by the Python preparation layer.
/// * `left_full_len` and `right_full_len` are the lengths of the original
///   physical layouts.
/// * `left_non_null_positions` and `right_non_null_positions` map compact
///   value offsets to physical positions.
/// * `left_null_positions` and `right_null_positions` contain physical null
///   positions, when present.
/// * `is_extension_array` controls whether nulls are excluded from matching.
/// * `visit` receives physical `(left_position, right_position)` pairs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn visit_not_equal_pairs_core<T, F>(
    left: ArrayView1<'_, T>,
    left_full_len: usize,
    left_non_null_positions: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_full_len: usize,
    right_non_null_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    mut visit: F,
) -> Result<(), String>
where
    T: PartialOrd + Copy,
    F: FnMut(usize, usize),
{
    validate_not_equal_side(
        "left",
        left_full_len,
        left,
        left_non_null_positions,
        left_null_positions,
    )?;
    validate_not_equal_side(
        "right",
        right_full_len,
        right,
        right_non_null_positions,
        right_null_positions,
    )?;
    let left_positions = positions("left non-null", left_non_null_positions)?;
    let right_positions = positions("right non-null", right_non_null_positions)?;
    let left_null_positions = left_null_positions
        .map(|values| positions("left null", values))
        .transpose()?;
    let right_null_positions = right_null_positions
        .map(|values| positions("right null", values))
        .transpose()?;
    let empty = Vec::new();
    let left_null_positions = left_null_positions.as_deref().unwrap_or(&empty);
    let right_null_positions = right_null_positions.as_deref().unwrap_or(&empty);
    // Both partitions are physical positions in original row order. Walking
    // their merge keeps the public all-!= result in left-row order while the
    // compact non-null offset remains available for the binary search.
    let mut left_non_null_offset = 0;
    let mut left_null_offset = 0;
    for left_position in 0..left_full_len {
        let left_position = i64::try_from(left_position)
            .map_err(|_| "left physical position exceeds int64".to_owned())?;
        let left_position_usize = left_position as usize;
        if left_non_null_offset < left_positions.len()
            && left_positions[left_non_null_offset] == left_position_usize
        {
            let left_value = left[left_non_null_offset];
            let lt_end = partition_point(right, |value| value < left_value);
            let gt_start = partition_point(right, |value| value <= left_value);
            for &right_position in &right_positions[..lt_end] {
                visit(left_position_usize, right_position);
            }
            for &right_position in &right_positions[gt_start..] {
                visit(left_position_usize, right_position);
            }
            if !is_extension_array {
                for &right_position in right_null_positions {
                    visit(left_position_usize, right_position);
                }
            }
            left_non_null_offset += 1;
        } else if left_null_offset < left_null_positions.len()
            && left_null_positions[left_null_offset] == left_position_usize
        {
            if !is_extension_array {
                for right_position in right_positions.iter().chain(right_null_positions) {
                    visit(left_position_usize, *right_position);
                }
            }
            left_null_offset += 1;
        }
    }
    Ok(())
}

/// Update aggregation state for every physical pair satisfying an all-`!=`
/// predicate and its residuals.
///
/// Aggregation is updated directly with physical positions. No compacted
/// output-position map is introduced, so residual predicates and aggregation
/// values continue to address the original full layouts. The output position
/// passed to [`AggregationSet::update`] is also the physical output row. This
/// makes the final accumulator arrays public-order arrays for both forward
/// (left output) and reverse (right output) aggregation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_not_equal<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_full_len: usize,
    left_non_null_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right: ArrayView1<'_, T>,
    right_full_len: usize,
    right_non_null_positions: ArrayView1<'_, i64>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
) -> Result<(), String> {
    visit_not_equal_pairs_core(
        left,
        left_full_len,
        left_non_null_positions,
        right,
        right_full_len,
        right_non_null_positions,
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

fn parse_not_equal_aggregation_residuals<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
) -> PyResult<(
    Vec<crate::predicate::Predicate<'py>>,
    Option<Vec<crate::predicate::NullMetadata<'py>>>,
)> {
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        let tuple = item.cast::<PyTuple>().map_err(|_| {
            PyValueError::new_err("not_equals aggregation residual must be a tuple")
        })?;
        let op_position = match tuple.len() {
            3 => 2,
            6 => 5,
            _ => {
                return Err(PyValueError::new_err(
                    "not_equals aggregation residual must contain 3 or 6 elements",
                ));
            }
        };
        let op = CompareOp::try_from_str(tuple.get_item(op_position)?.extract::<&str>()?)?;
        if op != CompareOp::Ne {
            return Err(PyValueError::new_err(
                "not_equals aggregation requires every predicate to use !=",
            ));
        }
        residuals.append(item)?;
    }
    parse_predicates_with_nulls_strings(py, &residuals)
}

/// Parse and execute the ten-field extended `!=` aggregation anchor.
///
/// The anchor layout is:
///
/// ```text
/// (left_values, left_full_positions, left_non_null_positions,
///  left_null_positions, right_values, right_full_positions,
///  right_non_null_positions, right_null_positions, is_extension_array, "!=")
/// ```
///
/// Later predicates are residual `!=` predicates and are validated before the
/// fused aggregation traversal begins.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_not_equal_aggregation<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    first: &Bound<'py, PyTuple>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if first.len() != 10 {
        return Err(PyValueError::new_err(
            "not_equals aggregation requires a 10-element anchor predicate",
        ));
    }
    let op = CompareOp::try_from_str(first.get_item(9)?.extract::<&str>()?)?;
    if op != CompareOp::Ne {
        return Err(PyValueError::new_err(
            "not_equals aggregation anchor must use !=",
        ));
    }
    let left = if first.get_item(0)?.is_none() {
        None
    } else {
        Some(first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?)
    };
    let left_non_null_positions = if first.get_item(2)?.is_none() {
        None
    } else {
        Some(first.get_item(2)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    let left_null_positions = if first.get_item(3)?.is_none() {
        None
    } else {
        Some(first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    let right = if first.get_item(4)?.is_none() {
        None
    } else {
        Some(first.get_item(4)?.extract::<PyReadonlyArray1<'py, T>>()?)
    };
    let right_non_null_positions = if first.get_item(6)?.is_none() {
        None
    } else {
        Some(first.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    let right_null_positions = if first.get_item(7)?.is_none() {
        None
    } else {
        Some(first.get_item(7)?.extract::<PyReadonlyArray1<'py, i64>>()?)
    };
    let is_extension_array = first.get_item(8)?.extract::<bool>()?;
    run_not_equal_aggregation::<T>(
        py,
        predicates,
        left,
        first.get_item(1)?.extract()?,
        left_non_null_positions,
        left_null_positions,
        right,
        first.get_item(5)?.extract()?,
        right_non_null_positions,
        right_null_positions,
        is_extension_array,
        aggregations,
        return_matched,
        reverse,
    )
}

/// Run the complete extended all-`!=` aggregation path.
///
/// This owns Python parsing, residual parsing, partition validation,
/// accumulator allocation, fused traversal, and result assembly. It does not
/// create an intermediate candidate-pair vector.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_not_equal_aggregation<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left: Option<PyReadonlyArray1<'py, T>>,
    left_full_positions: PyReadonlyArray1<'py, i64>,
    left_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right: Option<PyReadonlyArray1<'py, T>>,
    right_full_positions: PyReadonlyArray1<'py, i64>,
    right_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if predicates.is_empty() {
        return Err(PyValueError::new_err(
            "not_equals extended aggregation requires an anchor predicate",
        ));
    }
    let (parsed, metadata) = parse_not_equal_aggregation_residuals(py, predicates)?;
    check_predicate_lengths(
        &parsed,
        left_full_positions.len()?,
        right_full_positions.len()?,
    )?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "not_equals aggregation requires at least one aggregation",
        ));
    }

    let left_full_len = left_full_positions.len()?;
    let right_full_len = right_full_positions.len()?;
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
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;
    let empty = Array1::<T>::from_vec(Vec::new());
    let left = optional_array_view(left.as_ref(), empty.view());
    let right = optional_array_view(right.as_ref(), empty.view());
    let empty_positions = Array1::<i64>::from_vec(Vec::new());
    let left_non_null_positions =
        optional_array_view(left_non_null_positions.as_ref(), empty_positions.view());
    let right_non_null_positions =
        optional_array_view(right_non_null_positions.as_ref(), empty_positions.view());
    let views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    aggregate_not_equal(
        left,
        left_full_len,
        left_non_null_positions,
        left_null_positions.as_ref().map(|values| values.as_array()),
        right,
        right_full_len,
        right_non_null_positions,
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
    // AggregationSet stores each result slot by the physical output position:
    // left positions for forward aggregation and right positions for reverse
    // aggregation. The default position array is therefore already in public
    // dataframe order. It must remain paired with `matched` and every
    // aggregation array; Python must not apply a second permutation based on
    // the sorted anchor traversal order.
    Ok(Some(make_results_with_positions(
        py,
        set,
        None,
        output_len,
        return_matched,
    )?))
}

#[allow(clippy::too_many_arguments)]
fn materialize_single_all<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_full_positions: ArrayView1<'_, i64>,
    left_non_null_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right: ArrayView1<'_, T>,
    right_full_positions: ArrayView1<'_, i64>,
    right_non_null_positions: ArrayView1<'_, i64>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let left_null_count = left_null_positions.map_or(0, |values| values.len());
    let right_null_count = right_null_positions.map_or(0, |values| values.len());
    let capacity = all_capacity(
        left,
        right,
        left_null_count,
        right_null_count,
        is_extension_array,
    )?;
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(capacity)
        .map_err(|_| "single join result allocation failed".to_owned())?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(capacity)
        .map_err(|_| "single join result allocation failed".to_owned())?;

    visit_not_equal_pairs_core(
        left,
        left_full_positions.len(),
        left_non_null_positions,
        right,
        right_full_positions.len(),
        right_non_null_positions,
        left_null_positions,
        right_null_positions,
        is_extension_array,
        |left_position, right_position| {
            output_left.push(left_full_positions[left_position]);
            output_right.push(right_full_positions[right_position]);
        },
    )?;
    Ok((output_left, output_right))
}

#[allow(clippy::too_many_arguments)]
fn visit_single_selected<T, F>(
    left: ArrayView1<'_, T>,
    left_full_len: usize,
    left_non_null_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right: ArrayView1<'_, T>,
    right_full_len: usize,
    right_non_null_positions: ArrayView1<'_, i64>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    keep: Keep,
    mut visit: F,
) -> Result<(), String>
where
    T: PartialOrd + Copy,
    F: FnMut(usize, usize),
{
    validate_not_equal_side(
        "left",
        left_full_len,
        left,
        left_non_null_positions,
        left_null_positions,
    )?;
    validate_not_equal_side(
        "right",
        right_full_len,
        right,
        right_non_null_positions,
        right_null_positions,
    )?;
    let left_positions = positions("left non-null", left_non_null_positions)?;
    let right_positions = positions("right non-null", right_non_null_positions)?;
    let left_null_positions = left_null_positions
        .map(|values| positions("left null", values))
        .transpose()?;
    let right_null_positions = right_null_positions
        .map(|values| positions("right null", values))
        .transpose()?;
    let empty = Vec::new();
    let left_null_positions = left_null_positions.as_deref().unwrap_or(&empty);
    let right_null_positions = right_null_positions.as_deref().unwrap_or(&empty);
    let prefix_extrema = match keep {
        Keep::First => Some(prefix_extreme_positions(&right_positions, true)),
        Keep::Last => Some(prefix_extreme_positions(&right_positions, false)),
        Keep::Any | Keep::All => None,
    };
    let suffix_extrema = match keep {
        Keep::First => Some(suffix_extreme_positions(&right_positions, true)),
        Keep::Last => Some(suffix_extreme_positions(&right_positions, false)),
        Keep::Any | Keep::All => None,
    };
    let null_extreme = match keep {
        Keep::First => right_null_positions.iter().copied().min(),
        Keep::Last => right_null_positions.iter().copied().max(),
        Keep::Any | Keep::All => None,
    };

    for (left_value, &left_position) in left.iter().zip(&left_positions) {
        let less_end = partition_point(right, |value| value < *left_value);
        let greater_start = partition_point(right, |value| value <= *left_value);
        let mut selected = if less_end == 0 {
            None
        } else {
            Some(match keep {
                Keep::Any => right_positions[0],
                Keep::First | Keep::Last => {
                    right_positions[prefix_extrema.as_ref().unwrap()[less_end - 1]]
                }
                Keep::All => unreachable!(),
            })
        };
        if greater_start < right.len() {
            match keep {
                Keep::Any if selected.is_none() => {
                    selected = Some(right_positions[greater_start]);
                }
                Keep::Any => {}
                Keep::First | Keep::Last => choose(
                    &mut selected,
                    right_positions[suffix_extrema.as_ref().unwrap()[greater_start]],
                    keep,
                ),
                Keep::All => unreachable!(),
            }
        }
        if !is_extension_array {
            if keep == Keep::Any && selected.is_none() {
                selected = right_null_positions.first().copied();
            } else if let Some(position) = null_extreme {
                choose(&mut selected, position, keep);
            }
        }
        if let Some(right_position) = selected {
            visit(left_position, right_position);
        }
    }

    if !is_extension_array {
        let selected = match keep {
            Keep::Any | Keep::First => (right_full_len != 0).then_some(0),
            Keep::Last => right_full_len.checked_sub(1),
            Keep::All => unreachable!(),
        };
        for &left_position in left_null_positions {
            if let Some(right_position) = selected {
                visit(left_position, right_position);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn materialize_single_selected<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_full_positions: ArrayView1<'_, i64>,
    left_non_null_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right: ArrayView1<'_, T>,
    right_full_positions: ArrayView1<'_, i64>,
    right_non_null_positions: ArrayView1<'_, i64>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let mut output_left = Vec::new();
    let mut output_right = Vec::new();
    visit_single_selected(
        left,
        left_full_positions.len(),
        left_non_null_positions,
        left_null_positions,
        right,
        right_full_positions.len(),
        right_non_null_positions,
        right_null_positions,
        is_extension_array,
        keep,
        |left_position, right_position| {
            output_left.push(left_full_positions[left_position]);
            output_right.push(right_full_positions[right_position]);
        },
    )?;
    Ok((output_left, output_right))
}

#[allow(clippy::too_many_arguments)]
fn single_join_dispatch<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    left: Option<PyReadonlyArray1<'py, T>>,
    left_full_positions: PyReadonlyArray1<'py, i64>,
    right: Option<PyReadonlyArray1<'py, T>>,
    right_full_positions: PyReadonlyArray1<'py, i64>,
    comparator: &str,
    keep: &str,
    left_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let op = CompareOp::try_from_str(comparator)?;
    let keep = Keep::parse(keep)?;
    if op != CompareOp::Ne {
        return Err(PyValueError::new_err(
            "not_equals_only requires the != comparator",
        ));
    }
    validate_not_equal_partition_presence(
        "not_equals",
        "left",
        left.is_some(),
        left_non_null_positions.is_some(),
        left_null_positions.is_some(),
    )
    .map_err(PyValueError::new_err)?;
    validate_not_equal_partition_presence(
        "not_equals",
        "right",
        right.is_some(),
        right_non_null_positions.is_some(),
        right_null_positions.is_some(),
    )
    .map_err(PyValueError::new_err)?;
    let empty = Array1::<T>::from_vec(Vec::new());
    let left = optional_array_view(left.as_ref(), empty.view());
    let right = optional_array_view(right.as_ref(), empty.view());
    let empty_positions = Array1::<i64>::from_vec(Vec::new());
    let left_non_null_positions =
        optional_array_view(left_non_null_positions.as_ref(), empty_positions.view());
    let right_non_null_positions =
        optional_array_view(right_non_null_positions.as_ref(), empty_positions.view());
    if keep == Keep::All {
        let (out_left, out_right) = materialize_single_all(
            left,
            left_full_positions.as_array(),
            left_non_null_positions,
            left_null_positions.as_ref().map(|v| v.as_array()),
            right,
            right_full_positions.as_array(),
            right_non_null_positions,
            right_null_positions.as_ref().map(|v| v.as_array()),
            is_extension_array,
        )
        .map_err(PyValueError::new_err)?;
        return if out_left.is_empty() {
            Ok(None)
        } else {
            Ok(Some(result_dict(py, out_left, out_right, None, None)?))
        };
    }
    let (out_left, out_right) = materialize_single_selected(
        left,
        left_full_positions.as_array(),
        left_non_null_positions,
        left_null_positions.as_ref().map(|v| v.as_array()),
        right,
        right_full_positions.as_array(),
        right_non_null_positions,
        right_null_positions.as_ref().map(|v| v.as_array()),
        is_extension_array,
        keep,
    )
    .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        Ok(None)
    } else {
        Ok(Some(result_dict(py, out_left, out_right, None, None)?))
    }
}

fn extended_join_dispatch<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended join requires at least two predicates",
        ));
    }
    let keep = Keep::parse(keep)?;
    let first_item = predicates.get_item(0)?;
    let first = first_item.cast::<PyTuple>()?;
    if first.len() == 10 {
        let op = CompareOp::try_from_str(first.get_item(9)?.extract::<&str>()?)?;
        if op != CompareOp::Ne {
            return Err(PyValueError::new_err(
                "the first ten-element predicate must use !=",
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
        let residual_result = materialize_with_residuals::<T>(
            py,
            predicates,
            keep.as_str(),
            first.get_item(0)?.extract()?,
            first.get_item(1)?.extract()?,
            first.get_item(2)?.extract()?,
            left_null_positions,
            first.get_item(4)?.extract()?,
            first.get_item(5)?.extract()?,
            first.get_item(6)?.extract()?,
            right_null_positions,
            first.get_item(8)?.extract()?,
        )?;
        return Ok(residual_result);
    }
    Err(PyValueError::new_err(
        "not_equals_only requires a 10-element != anchor predicate",
    ))
}

fn positions(name: &str, values: ArrayView1<'_, i64>) -> Result<Vec<usize>, String> {
    values
        .iter()
        .enumerate()
        .map(|(offset, value)| {
            usize::try_from(*value).map_err(|_| {
                format!("{name} position at offset {offset} must be a non-negative int64")
            })
        })
        .collect()
}

fn validate_partition(
    name: &str,
    full_len: usize,
    non_null: &[usize],
    null: &[usize],
) -> Result<(), String> {
    let count = non_null
        .len()
        .checked_add(null.len())
        .ok_or("position count exceeds platform capacity")?;
    if count != full_len {
        return Err(format!(
            "{name} length must equal the number of non-null values plus null positions"
        ));
    }
    let mut seen = vec![false; full_len];
    for (&position, kind) in non_null
        .iter()
        .zip(std::iter::repeat("non-null"))
        .chain(null.iter().zip(std::iter::repeat("null")))
    {
        if position >= full_len {
            return Err(format!(
                "{name} {kind} position {position} is out of bounds"
            ));
        }
        if seen[position] {
            return Err(format!("{name} position {position} appears more than once"));
        }
        seen[position] = true;
    }
    Ok(())
}

fn choose(current: &mut Option<usize>, candidate: usize, keep: Keep) {
    match current {
        None => *current = Some(candidate),
        Some(previous) if keep == Keep::First && candidate < *previous => *previous = candidate,
        Some(previous) if keep == Keep::Last && candidate > *previous => *previous = candidate,
        _ => {}
    }
}

fn prefix_extreme_positions(values: &[usize], minimum: bool) -> Vec<usize> {
    let mut result = Vec::with_capacity(values.len());
    let mut selected = None;
    for (offset, &value) in values.iter().enumerate() {
        if selected.is_none_or(|current| {
            (minimum && value < values[current]) || (!minimum && value > values[current])
        }) {
            selected = Some(offset);
        }
        result.push(selected.expect("prefix position exists after iteration"));
    }
    result
}

fn suffix_extreme_positions(values: &[usize], minimum: bool) -> Vec<usize> {
    let mut result = vec![0; values.len()];
    let mut selected = None;
    for offset in (0..values.len()).rev() {
        let value = values[offset];
        if selected.is_none_or(|current| {
            (minimum && value < values[current]) || (!minimum && value > values[current])
        }) {
            selected = Some(offset);
        }
        result[offset] = selected.expect("suffix position exists after iteration");
    }
    result
}

fn all_capacity<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    left_null_count: usize,
    right_null_count: usize,
    is_extension_array: bool,
) -> Result<usize, String> {
    let mut capacity = 0_usize;
    for value in left {
        let less_end = partition_point(right, |candidate| candidate < *value);
        let greater_start = partition_point(right, |candidate| candidate <= *value);
        let strict_count = less_end
            .checked_add(right.len().saturating_sub(greater_start))
            .ok_or("single join result size exceeds platform capacity")?;
        let row_count = if is_extension_array {
            strict_count
        } else {
            strict_count
                .checked_add(right_null_count)
                .ok_or("single join result size exceeds platform capacity")?
        };
        capacity = capacity
            .checked_add(row_count)
            .ok_or("single join result size exceeds platform capacity")?;
    }
    if !is_extension_array {
        capacity = capacity
            .checked_add(
                left_null_count
                    .checked_mul(
                        right
                            .len()
                            .checked_add(right_null_count)
                            .ok_or("single join result size exceeds platform capacity")?,
                    )
                    .ok_or("single join result size exceeds platform capacity")?,
            )
            .ok_or("single join result size exceeds platform capacity")?;
    }
    Ok(capacity)
}

/// Apply all residual predicates to an all-`!=` candidate stream.
///
/// The anchor is traversed with `Keep::All` before residual filtering. For
/// `Keep::All`, the implementation counts surviving pairs first and writes
/// them in a second pass, avoiding an intermediate candidate allocation.
///
/// The anchor is deliberately rebuilt with `Keep::All`: reducing the anchor
/// before residual evaluation could discard the pair that should win after a
/// later residual is applied. Residual predicates index the complete physical
/// layouts, so the returned position pairs remain physical positions until
/// every residual and the final `keep` selection have completed.
///
/// # Arguments
///
/// * `predicates` contains the ten-field anchor followed by residual tuples.
/// * `keep` is applied only after all residual predicates have passed.
/// * The `*_full_positions` arrays define the complete physical output and
///   residual-indexing domains.
/// * The compact value arrays and `*_non_null_positions` maps define the
///   binary-search input domains.
/// * The null-position arrays preserve null rows without inserting sentinel
///   values into the compact arrays.
#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_with_residuals<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first_left: PyReadonlyArray1<'py, T>,
    first_left_full_positions: PyReadonlyArray1<'py, i64>,
    first_left_non_null_positions: PyReadonlyArray1<'py, i64>,
    first_left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    first_right: PyReadonlyArray1<'py, T>,
    first_right_full_positions: PyReadonlyArray1<'py, i64>,
    first_right_non_null_positions: PyReadonlyArray1<'py, i64>,
    first_right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended join requires at least two predicates",
        ));
    }
    let keep = Keep::parse(keep)?;
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        let tuple = item
            .cast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("each residual comparison must be a tuple"))?;
        let op_position = match tuple.len() {
            3 => 2,
            6 => 5,
            _ => {
                return Err(PyValueError::new_err(
                    "each residual comparison must contain 3 or 6 elements",
                ));
            }
        };
        let op = CompareOp::try_from_str(tuple.get_item(op_position)?.extract::<&str>()?)?;
        if op != CompareOp::Ne {
            return Err(PyValueError::new_err(
                "all-!= joins require every predicate to use !=",
            ));
        }
        residuals.append(item)?;
    }

    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    let left_full_positions = first_left_full_positions.as_array();
    let right_full_positions = first_right_full_positions.as_array();
    check_predicate_lengths(
        &parsed,
        left_full_positions.len(),
        right_full_positions.len(),
    )?;

    let predicate_views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let matches = |left_position: usize, right_position: usize| {
        predicates_match_dispatch(
            &predicate_views,
            metadata_views.as_deref(),
            left_position,
            right_position,
        )
    };

    if keep == Keep::All {
        let mut output_len = 0_usize;
        let mut overflow = false;
        visit_not_equal_pairs_core(
            first_left.as_array(),
            left_full_positions.len(),
            first_left_non_null_positions.as_array(),
            first_right.as_array(),
            right_full_positions.len(),
            first_right_non_null_positions.as_array(),
            first_left_null_positions.as_ref().map(|v| v.as_array()),
            first_right_null_positions.as_ref().map(|v| v.as_array()),
            is_extension_array,
            |left_position, right_position| {
                if matches(left_position, right_position) {
                    if let Some(next) = output_len.checked_add(1) {
                        output_len = next;
                    } else {
                        overflow = true;
                    }
                }
            },
        )
        .map_err(PyValueError::new_err)?;
        if overflow {
            return Err(PyValueError::new_err(
                "single extended join result size exceeds platform capacity",
            ));
        }
        if output_len == 0 {
            return Ok(None);
        }

        let mut out_left = Vec::new();
        out_left
            .try_reserve_exact(output_len)
            .map_err(|_| PyValueError::new_err("single extended join result allocation failed"))?;
        let mut out_right = Vec::new();
        out_right
            .try_reserve_exact(output_len)
            .map_err(|_| PyValueError::new_err("single extended join result allocation failed"))?;
        visit_not_equal_pairs_core(
            first_left.as_array(),
            left_full_positions.len(),
            first_left_non_null_positions.as_array(),
            first_right.as_array(),
            right_full_positions.len(),
            first_right_non_null_positions.as_array(),
            first_left_null_positions.as_ref().map(|v| v.as_array()),
            first_right_null_positions.as_ref().map(|v| v.as_array()),
            is_extension_array,
            |left_position, right_position| {
                if matches(left_position, right_position) {
                    out_left.push(left_full_positions[left_position]);
                    out_right.push(right_full_positions[right_position]);
                }
            },
        )
        .map_err(PyValueError::new_err)?;
        return Ok(Some(result_dict(py, out_left, out_right, None, None)?));
    }

    let mut selected = vec![None; left_full_positions.len()];
    visit_not_equal_pairs_core(
        first_left.as_array(),
        left_full_positions.len(),
        first_left_non_null_positions.as_array(),
        first_right.as_array(),
        right_full_positions.len(),
        first_right_non_null_positions.as_array(),
        first_left_null_positions.as_ref().map(|v| v.as_array()),
        first_right_null_positions.as_ref().map(|v| v.as_array()),
        is_extension_array,
        |left_position, right_position| {
            if !matches(left_position, right_position) {
                return;
            }
            let slot = &mut selected[left_position];
            match keep {
                Keep::Any => {
                    if slot.is_none() {
                        *slot = Some(right_position);
                    }
                }
                Keep::First => {
                    if slot.is_none_or(|current| right_position < current) {
                        *slot = Some(right_position);
                    }
                }
                Keep::Last => {
                    if slot.is_none_or(|current| right_position > current) {
                        *slot = Some(right_position);
                    }
                }
                Keep::All => unreachable!(),
            }
        },
    )
    .map_err(PyValueError::new_err)?;

    let mut out_left = Vec::new();
    out_left
        .try_reserve_exact(left_full_positions.len())
        .map_err(|_| PyValueError::new_err("single extended join result allocation failed"))?;
    let mut out_right = Vec::new();
    out_right
        .try_reserve_exact(left_full_positions.len())
        .map_err(|_| PyValueError::new_err("single extended join result allocation failed"))?;
    for (left_position, right_position) in selected.into_iter().enumerate() {
        if let Some(right_position) = right_position {
            out_left.push(left_full_positions[left_position]);
            out_right.push(right_full_positions[right_position]);
        }
    }
    if out_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, out_left, out_right, None, None)?))
}

#[allow(clippy::too_many_arguments)]
fn run_single_not_equal_aggregation<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    left: Option<PyReadonlyArray1<'py, T>>,
    left_full_positions: PyReadonlyArray1<'py, i64>,
    right: Option<PyReadonlyArray1<'py, T>>,
    right_full_positions: PyReadonlyArray1<'py, i64>,
    comparator: &str,
    left_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    is_extension_array: bool,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if CompareOp::try_from_str(comparator)? != CompareOp::Ne {
        return Err(PyValueError::new_err(
            "not_equals aggregation requires the != comparator",
        ));
    }
    validate_not_equal_partition_presence(
        "not_equals aggregation",
        "left",
        left.is_some(),
        left_non_null_positions.is_some(),
        left_null_positions.is_some(),
    )
    .map_err(PyValueError::new_err)?;
    validate_not_equal_partition_presence(
        "not_equals aggregation",
        "right",
        right.is_some(),
        right_non_null_positions.is_some(),
        right_null_positions.is_some(),
    )
    .map_err(PyValueError::new_err)?;
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let empty = Array1::<T>::from_vec(Vec::new());
    let left = optional_array_view(left.as_ref(), empty.view());
    let right = optional_array_view(right.as_ref(), empty.view());
    let empty_positions = Array1::<i64>::from_vec(Vec::new());
    let left_non_null_positions =
        optional_array_view(left_non_null_positions.as_ref(), empty_positions.view());
    let right_non_null_positions =
        optional_array_view(right_non_null_positions.as_ref(), empty_positions.view());
    let left_full_len = left_full_positions.len()?;
    let right_full_len = right_full_positions.len()?;
    validate_not_equal_side(
        "left",
        left_full_len,
        left,
        left_non_null_positions,
        left_null_positions.as_ref().map(|values| values.as_array()),
    )
    .map_err(PyValueError::new_err)?;
    validate_not_equal_side(
        "right",
        right_full_len,
        right,
        right_non_null_positions,
        right_null_positions
            .as_ref()
            .map(|values| values.as_array()),
    )
    .map_err(PyValueError::new_err)?;
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
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;
    aggregate_not_equal(
        left,
        left_full_len,
        left_non_null_positions,
        left_null_positions.as_ref().map(|v| v.as_array()),
        right,
        right_full_len,
        right_non_null_positions,
        right_null_positions.as_ref().map(|v| v.as_array()),
        is_extension_array,
        &[],
        None,
        &mut set,
        reverse,
    )
    .map_err(PyValueError::new_err)?;
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

macro_rules! registered_not_equals_aggregation {
    ($forward:ident, $reverse:ident, $type:ty) => {
        /// Aggregate one null-aware `!=` predicate for the selected dtype.
        ///
        /// `left_full_positions` and `right_full_positions` describe the
        /// complete physical layouts. The optional value and non-null arrays
        /// are both `None` for an all-null side; that side must still provide
        /// its null-position partition. `aggregations` may contain multiple
        /// requests and is parsed once by Rust.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $forward<'py>(
            py: Python<'py>,
            left: Option<PyReadonlyArray1<'py, $type>>,
            left_full_positions: PyReadonlyArray1<'py, i64>,
            right: Option<PyReadonlyArray1<'py, $type>>,
            right_full_positions: PyReadonlyArray1<'py, i64>,
            comparator: &str,
            left_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            is_extension_array: bool,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            run_single_not_equal_aggregation(
                py,
                left,
                left_full_positions,
                right,
                right_full_positions,
                comparator,
                left_non_null_positions,
                left_null_positions,
                right_non_null_positions,
                right_null_positions,
                is_extension_array,
                aggregations,
                return_matched,
                false,
            )
        }
        /// Aggregate one null-aware `!=` predicate with right-side output
        /// slots and left-side source values.
        ///
        /// This has the same input contract as the forward function. Only the
        /// output/source orientation changes; the physical position maps and
        /// null rules remain identical.
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $reverse<'py>(
            py: Python<'py>,
            left: Option<PyReadonlyArray1<'py, $type>>,
            left_full_positions: PyReadonlyArray1<'py, i64>,
            right: Option<PyReadonlyArray1<'py, $type>>,
            right_full_positions: PyReadonlyArray1<'py, i64>,
            comparator: &str,
            left_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            is_extension_array: bool,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            run_single_not_equal_aggregation(
                py,
                left,
                left_full_positions,
                right,
                right_full_positions,
                comparator,
                left_non_null_positions,
                left_null_positions,
                right_non_null_positions,
                right_null_positions,
                is_extension_array,
                aggregations,
                return_matched,
                true,
            )
        }
    };
}

macro_rules! registered_not_equals_extended_aggregation {
    ($forward:ident, $reverse:ident, $type:ty) => {
        /// Aggregate an all-`!=` predicate list for the selected anchor dtype.
        ///
        /// The first predicate is the ten-field anchor ABI documented at the
        /// top of this module. Remaining tuples are residual `!=` predicates.
        /// Rust validates their lengths and operators before traversing the
        /// anchor candidates.
        #[pyfunction]
        pub fn $forward<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch_not_equal_aggregation::<$type>(
                py,
                predicates,
                &first,
                aggregations,
                return_matched,
                false,
            )
        }
        /// Reverse aggregate an all-`!=` predicate list for the selected
        /// anchor dtype.
        ///
        /// The aggregation source is the left layout and the output slots
        /// belong to the right layout.
        #[pyfunction]
        pub fn $reverse<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            aggregations: &Bound<'py, PyList>,
            return_matched: bool,
        ) -> PyResult<Option<Bound<'py, PyTuple>>> {
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            dispatch_not_equal_aggregation::<$type>(
                py,
                predicates,
                &first,
                aggregations,
                return_matched,
                true,
            )
        }
    };
}

registered_not_equals_aggregation!(
    not_equals_aggregate_int64,
    not_equals_aggregate_reverse_int64,
    i64
);
registered_not_equals_aggregation!(
    not_equals_aggregate_int32,
    not_equals_aggregate_reverse_int32,
    i32
);
registered_not_equals_aggregation!(
    not_equals_aggregate_int16,
    not_equals_aggregate_reverse_int16,
    i16
);
registered_not_equals_aggregation!(
    not_equals_aggregate_int8,
    not_equals_aggregate_reverse_int8,
    i8
);
registered_not_equals_aggregation!(
    not_equals_aggregate_uint64,
    not_equals_aggregate_reverse_uint64,
    u64
);
registered_not_equals_aggregation!(
    not_equals_aggregate_uint32,
    not_equals_aggregate_reverse_uint32,
    u32
);
registered_not_equals_aggregation!(
    not_equals_aggregate_uint16,
    not_equals_aggregate_reverse_uint16,
    u16
);
registered_not_equals_aggregation!(
    not_equals_aggregate_uint8,
    not_equals_aggregate_reverse_uint8,
    u8
);
registered_not_equals_aggregation!(
    not_equals_aggregate_f64,
    not_equals_aggregate_reverse_f64,
    f64
);
registered_not_equals_aggregation!(
    not_equals_aggregate_f32,
    not_equals_aggregate_reverse_f32,
    f32
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_int64,
    not_equals_extended_aggregate_reverse_int64,
    i64
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_int32,
    not_equals_extended_aggregate_reverse_int32,
    i32
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_int16,
    not_equals_extended_aggregate_reverse_int16,
    i16
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_int8,
    not_equals_extended_aggregate_reverse_int8,
    i8
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_uint64,
    not_equals_extended_aggregate_reverse_uint64,
    u64
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_uint32,
    not_equals_extended_aggregate_reverse_uint32,
    u32
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_uint16,
    not_equals_extended_aggregate_reverse_uint16,
    u16
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_uint8,
    not_equals_extended_aggregate_reverse_uint8,
    u8
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_f64,
    not_equals_extended_aggregate_reverse_f64,
    f64
);
registered_not_equals_extended_aggregation!(
    not_equals_extended_aggregate_f32,
    not_equals_extended_aggregate_reverse_f32,
    f32
);

// These are the stable Python entry points for the single-predicate API.
macro_rules! registered_single_join {
    ($name:ident, $type:ty) => {
        /// Generate physical join indices for one null-aware `!=` predicate.
        ///
        /// The two full-position arrays describe the original row layouts;
        /// the optional compact values and partition maps describe the
        /// non-null subsets. `keep` is applied after null-aware candidate
        /// generation.
        #[allow(clippy::too_many_arguments)]
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            left: Option<PyReadonlyArray1<'py, $type>>,
            left_full_positions: PyReadonlyArray1<'py, i64>,
            right: Option<PyReadonlyArray1<'py, $type>>,
            right_full_positions: PyReadonlyArray1<'py, i64>,
            comparator: &str,
            keep: &str,
            left_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_non_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            is_extension_array: bool,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            single_join_dispatch(
                py,
                left,
                left_full_positions,
                right,
                right_full_positions,
                comparator,
                keep,
                left_non_null_positions,
                left_null_positions,
                right_non_null_positions,
                right_null_positions,
                is_extension_array,
            )
        }
    };
}

registered_single_join!(single_join_indices_int64, i64);
registered_single_join!(single_join_indices_int32, i32);
registered_single_join!(single_join_indices_int16, i16);
registered_single_join!(single_join_indices_int8, i8);
registered_single_join!(single_join_indices_uint64, u64);
registered_single_join!(single_join_indices_uint32, u32);
registered_single_join!(single_join_indices_uint16, u16);
registered_single_join!(single_join_indices_uint8, u8);
registered_single_join!(single_join_indices_f64, f64);
registered_single_join!(single_join_indices_f32, f32);

macro_rules! registered_extended_join {
    ($name:ident, $type:ty) => {
        /// Generate physical join indices for an all-`!=` predicate list.
        ///
        /// The first list item is the ten-field anchor tuple. Every following
        /// item is a residual predicate. Residuals are evaluated against the
        /// original physical positions before `keep` is applied.
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            keep: &str,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            extended_join_dispatch::<$type>(py, predicates, keep)
        }
    };
}

registered_extended_join!(single_join_extended_indices_int64, i64);
registered_extended_join!(single_join_extended_indices_int32, i32);
registered_extended_join!(single_join_extended_indices_int16, i16);
registered_extended_join!(single_join_extended_indices_int8, i8);
registered_extended_join!(single_join_extended_indices_uint64, u64);
registered_extended_join!(single_join_extended_indices_uint32, u32);
registered_extended_join!(single_join_extended_indices_uint16, u16);
registered_extended_join!(single_join_extended_indices_uint8, u8);
registered_extended_join!(single_join_extended_indices_f64, f64);
registered_extended_join!(single_join_extended_indices_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! add {
        ($($name:ident),+ $(,)?) => { $(m.add_function(wrap_pyfunction!($name, m)?)?;)+ };
    }
    add!(
        single_join_indices_int64,
        single_join_indices_int32,
        single_join_indices_int16,
        single_join_indices_int8,
        single_join_indices_uint64,
        single_join_indices_uint32,
        single_join_indices_uint16,
        single_join_indices_uint8,
        single_join_indices_f64,
        single_join_indices_f32,
        single_join_extended_indices_int64,
        single_join_extended_indices_int32,
        single_join_extended_indices_int16,
        single_join_extended_indices_int8,
        single_join_extended_indices_uint64,
        single_join_extended_indices_uint32,
        single_join_extended_indices_uint16,
        single_join_extended_indices_uint8,
        single_join_extended_indices_f64,
        single_join_extended_indices_f32,
        not_equals_aggregate_int64,
        not_equals_aggregate_reverse_int64,
        not_equals_aggregate_int32,
        not_equals_aggregate_reverse_int32,
        not_equals_aggregate_int16,
        not_equals_aggregate_reverse_int16,
        not_equals_aggregate_int8,
        not_equals_aggregate_reverse_int8,
        not_equals_aggregate_uint64,
        not_equals_aggregate_reverse_uint64,
        not_equals_aggregate_uint32,
        not_equals_aggregate_reverse_uint32,
        not_equals_aggregate_uint16,
        not_equals_aggregate_reverse_uint16,
        not_equals_aggregate_uint8,
        not_equals_aggregate_reverse_uint8,
        not_equals_aggregate_f64,
        not_equals_aggregate_reverse_f64,
        not_equals_aggregate_f32,
        not_equals_aggregate_reverse_f32,
        not_equals_extended_aggregate_int64,
        not_equals_extended_aggregate_reverse_int64,
        not_equals_extended_aggregate_int32,
        not_equals_extended_aggregate_reverse_int32,
        not_equals_extended_aggregate_int16,
        not_equals_extended_aggregate_reverse_int16,
        not_equals_extended_aggregate_int8,
        not_equals_extended_aggregate_reverse_int8,
        not_equals_extended_aggregate_uint64,
        not_equals_extended_aggregate_reverse_uint64,
        not_equals_extended_aggregate_uint32,
        not_equals_extended_aggregate_reverse_uint32,
        not_equals_extended_aggregate_uint16,
        not_equals_extended_aggregate_reverse_uint16,
        not_equals_extended_aggregate_uint8,
        not_equals_extended_aggregate_reverse_uint8,
        not_equals_extended_aggregate_f64,
        not_equals_extended_aggregate_reverse_f64,
        not_equals_extended_aggregate_f32,
        not_equals_extended_aggregate_reverse_f32,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::IntoPyObject;

    fn collect_pairs(
        left: &[i64],
        left_non_null: &[i64],
        left_null: Option<&[i64]>,
        right: &[i64],
        right_non_null: &[i64],
        right_null: Option<&[i64]>,
        is_extension_array: bool,
    ) -> Vec<(usize, usize)> {
        let left_values = Array1::from_vec(left.to_vec());
        let left_positions = Array1::from_vec(left_non_null.to_vec());
        let right_values = Array1::from_vec(right.to_vec());
        let right_positions = Array1::from_vec(right_non_null.to_vec());
        let left_null_values = left_null.map(|values| Array1::from_vec(values.to_vec()));
        let right_null_values = right_null.map(|values| Array1::from_vec(values.to_vec()));
        let mut pairs = Vec::new();

        visit_not_equal_pairs_core(
            left_values.view(),
            left_non_null.len() + left_null.map_or(0, |values| values.len()),
            left_positions.view(),
            right_values.view(),
            right_non_null.len() + right_null.map_or(0, |values| values.len()),
            right_positions.view(),
            left_null_values.as_ref().map(|values| values.view()),
            right_null_values.as_ref().map(|values| values.view()),
            is_extension_array,
            |left_position, right_position| pairs.push((left_position, right_position)),
        )
        .unwrap();

        pairs
    }

    #[test]
    fn physical_positions_follow_sorted_value_layouts() {
        let pairs = collect_pairs(&[2], &[0], None, &[1, 3], &[1, 0], None, false);

        assert_eq!(pairs, vec![(0, 1), (0, 0)]);
    }

    #[test]
    fn numpy_nulls_match_everything_on_the_other_side() {
        let pairs = collect_pairs(&[1], &[0], Some(&[1]), &[1], &[0], Some(&[1]), false);

        assert_eq!(pairs, vec![(0, 1), (1, 0), (1, 1)]);
    }

    #[test]
    fn extension_array_nulls_do_not_match() {
        let pairs = collect_pairs(&[1], &[0], Some(&[1]), &[1], &[0], Some(&[1]), true);

        assert!(pairs.is_empty());
    }

    #[test]
    fn all_null_side_is_supported_for_numpy_nulls() {
        let pairs = collect_pairs(&[], &[], Some(&[0, 1]), &[1, 2], &[1, 0], None, false);

        assert_eq!(pairs, vec![(0, 1), (0, 0), (1, 1), (1, 0)]);
    }

    #[test]
    fn keep_first_and_last_use_physical_positions() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_positions = Array1::from_vec(vec![0_i64]);
        let right = Array1::from_vec(vec![1_i64, 2, 3]);
        let right_positions = Array1::from_vec(vec![2_i64, 0, 1]);
        let mut first = Vec::new();
        let mut last = Vec::new();

        visit_single_selected(
            left.view(),
            1,
            left_positions.view(),
            None,
            right.view(),
            3,
            right_positions.view(),
            None,
            false,
            Keep::First,
            |left_position, right_position| first.push((left_position, right_position)),
        )
        .unwrap();
        visit_single_selected(
            left.view(),
            1,
            left_positions.view(),
            None,
            right.view(),
            3,
            right_positions.view(),
            None,
            false,
            Keep::Last,
            |left_position, right_position| last.push((left_position, right_position)),
        )
        .unwrap();

        assert_eq!(first, vec![(0, 1)]);
        assert_eq!(last, vec![(0, 2)]);
    }

    #[test]
    fn keep_any_returns_the_first_candidate_from_the_traversal() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_positions = Array1::from_vec(vec![0_i64]);
        let right = Array1::from_vec(vec![1_i64, 2, 3]);
        let right_positions = Array1::from_vec(vec![2_i64, 0, 1]);
        let mut pairs = Vec::new();

        visit_single_selected(
            left.view(),
            1,
            left_positions.view(),
            None,
            right.view(),
            3,
            right_positions.view(),
            None,
            false,
            Keep::Any,
            |left_position, right_position| pairs.push((left_position, right_position)),
        )
        .unwrap();

        assert_eq!(pairs, vec![(0, 2)]);
    }

    #[test]
    fn keep_first_and_last_consider_numpy_null_positions() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_positions = Array1::from_vec(vec![0_i64]);
        let right = Array1::from_vec(vec![1_i64]);
        let right_positions = Array1::from_vec(vec![0_i64]);
        let right_null_positions = Array1::from_vec(vec![2_i64, 1]);
        let mut first = Vec::new();
        let mut last = Vec::new();

        for (keep, output) in [(Keep::First, &mut first), (Keep::Last, &mut last)] {
            visit_single_selected(
                left.view(),
                1,
                left_positions.view(),
                None,
                right.view(),
                3,
                right_positions.view(),
                Some(right_null_positions.view()),
                false,
                keep,
                |left_position, right_position| output.push((left_position, right_position)),
            )
            .unwrap();
        }

        assert_eq!(first, vec![(0, 0)]);
        assert_eq!(last, vec![(0, 2)]);
    }

    #[test]
    fn keep_all_materializes_every_physical_pair() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_full_positions = Array1::from_vec(vec![7_i64]);
        let left_positions = Array1::from_vec(vec![0_i64]);
        let right = Array1::from_vec(vec![1_i64, 3]);
        let right_full_positions = Array1::from_vec(vec![11_i64, 13]);
        let right_positions = Array1::from_vec(vec![1_i64, 0]);

        let (output_left, output_right) = materialize_single_all(
            left.view(),
            left_full_positions.view(),
            left_positions.view(),
            None,
            right.view(),
            right_full_positions.view(),
            right_positions.view(),
            None,
            false,
        )
        .unwrap();

        assert_eq!(output_left, vec![7, 7]);
        assert_eq!(output_right, vec![13, 11]);
    }

    fn aggregation_request<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyList>> {
        let values = PyArray1::from_vec(py, values);
        let mask = PyArray1::from_vec(py, vec![false; values.len()?]);
        let request = PyTuple::new(
            py,
            [
                values.into_any(),
                mask.into_any(),
                "sum".into_pyobject(py)?.into_any(),
            ],
        )?;
        PyList::new(py, [request])
    }

    fn aggregation_result<'py>(
        result: &Bound<'py, PyTuple>,
    ) -> PyResult<(Vec<i64>, Vec<bool>, Vec<i64>)> {
        let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
        let matched = result.get_item(1)?.extract::<Vec<bool>>()?;
        let values = result
            .get_item(2)?
            .cast::<PyList>()?
            .get_item(0)?
            .extract::<Vec<i64>>()?;
        Ok((positions, matched, values))
    }

    fn pair_result<'py>(result: &Bound<'py, PyDict>) -> PyResult<(Vec<i64>, Vec<i64>)> {
        let left = result
            .get_item("left_index")?
            .expect("left_index is present")
            .extract::<Vec<i64>>()?;
        let right = result
            .get_item("right_index")?
            .expect("right_index is present")
            .extract::<Vec<i64>>()?;
        Ok((left, right))
    }

    #[test]
    fn forward_and_reverse_aggregation_use_physical_output_slots() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left = PyArray1::from_vec(py, vec![1_i64, 2]);
            let left_full_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let left_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right = PyArray1::from_vec(py, vec![1_i64, 3]);
            let right_full_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right_positions = PyArray1::from_vec(py, vec![0_i64, 1]);

            let forward = run_single_not_equal_aggregation(
                py,
                Some(left.readonly()),
                left_full_positions.readonly(),
                Some(right.readonly()),
                right_full_positions.readonly(),
                "!=",
                Some(left_positions.readonly()),
                None,
                Some(right_positions.readonly()),
                None,
                false,
                &aggregation_request(py, vec![1, 3])?,
                true,
                false,
            )?
            .expect("forward aggregation should match");
            assert_eq!(
                aggregation_result(&forward)?,
                (vec![0, 1], vec![true, true], vec![3, 4])
            );

            let reverse = run_single_not_equal_aggregation(
                py,
                Some(left.readonly()),
                left_full_positions.readonly(),
                Some(right.readonly()),
                right_full_positions.readonly(),
                "!=",
                Some(left_positions.readonly()),
                None,
                Some(right_positions.readonly()),
                None,
                false,
                &aggregation_request(py, vec![1, 2])?,
                true,
                true,
            )?
            .expect("reverse aggregation should match");
            assert_eq!(
                aggregation_result(&reverse)?,
                (vec![0, 1], vec![true, true], vec![2, 3])
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn extended_residuals_are_applied_before_keep() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_values = PyArray1::from_vec(py, vec![1_i64, 2]);
            let left_full_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let left_non_null_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right_values = PyArray1::from_vec(py, vec![1_i64, 3]);
            let right_full_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right_non_null_positions = PyArray1::from_vec(py, vec![0_i64, 1]);
            let residual_left = PyArray1::from_vec(py, vec![10_i64, 20]);
            let residual_right = PyArray1::from_vec(py, vec![10_i64, 15]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left_values.clone().into_any(),
                    left_full_positions.clone().into_any(),
                    left_non_null_positions.clone().into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    right_values.clone().into_any(),
                    right_full_positions.clone().into_any(),
                    right_non_null_positions.clone().into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    residual_left.into_any(),
                    residual_right.into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let all = materialize_with_residuals(
                py,
                &predicates,
                "all",
                left_values.readonly(),
                left_full_positions.readonly(),
                left_non_null_positions.readonly(),
                None,
                right_values.readonly(),
                right_full_positions.readonly(),
                right_non_null_positions.readonly(),
                None,
                false,
            )?
            .expect("residuals should leave matching pairs");
            assert_eq!(pair_result(&all)?, (vec![0, 1, 1], vec![1, 0, 1]));

            let first = materialize_with_residuals(
                py,
                &predicates,
                "first",
                left_values.readonly(),
                left_full_positions.readonly(),
                left_non_null_positions.readonly(),
                None,
                right_values.readonly(),
                right_full_positions.readonly(),
                right_non_null_positions.readonly(),
                None,
                false,
            )?
            .expect("residuals should leave first matches");
            assert_eq!(pair_result(&first)?, (vec![0, 1], vec![1, 0]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn invalid_partitions_are_rejected() {
        let left = Array1::from_vec(vec![1_i64]);
        let right = Array1::from_vec(vec![2_i64]);
        let duplicate = Array1::from_vec(vec![0_i64]);
        let null = Array1::from_vec(vec![0_i64]);

        let error = visit_not_equal_pairs_core(
            left.view(),
            2,
            duplicate.view(),
            right.view(),
            1,
            duplicate.view(),
            Some(null.view()),
            None,
            false,
            |_, _| {},
        )
        .unwrap_err();

        assert!(error.contains("appears more than once"), "{error}");
    }

    #[test]
    fn exhaustive_small_layouts_match_brute_force_oracle() {
        for left_null_mask in 0..8_u8 {
            for right_null_mask in 0..8_u8 {
                for is_extension_array in [false, true] {
                    let left_non_null: Vec<i64> = (0..3)
                        .filter(|position| left_null_mask & (1 << position) == 0)
                        .map(|position| position as i64)
                        .collect();
                    let right_non_null: Vec<i64> = (0..3)
                        .filter(|position| right_null_mask & (1 << position) == 0)
                        .map(|position| position as i64)
                        .collect();
                    let left_positions: Vec<i64> = (0..3)
                        .filter(|position| left_null_mask & (1 << position) == 0)
                        .map(|position| position as i64)
                        .collect();
                    let right_positions: Vec<i64> = (0..3)
                        .filter(|position| right_null_mask & (1 << position) == 0)
                        .map(|position| position as i64)
                        .collect();
                    let left_null: Vec<i64> = (0..3)
                        .filter(|position| left_null_mask & (1 << position) != 0)
                        .map(|position| position as i64)
                        .collect();
                    let right_null: Vec<i64> = (0..3)
                        .filter(|position| right_null_mask & (1 << position) != 0)
                        .map(|position| position as i64)
                        .collect();

                    let actual = collect_pairs(
                        &left_non_null,
                        &left_positions,
                        Some(&left_null),
                        &right_non_null,
                        &right_positions,
                        Some(&right_null),
                        is_extension_array,
                    );
                    let mut expected = Vec::new();
                    for left_position in 0..3 {
                        for right_position in 0..3 {
                            let left_is_null = left_null_mask & (1 << left_position) != 0;
                            let right_is_null = right_null_mask & (1 << right_position) != 0;
                            let matches = if left_is_null || right_is_null {
                                !is_extension_array
                            } else {
                                left_position != right_position
                            };
                            if matches {
                                expected.push((left_position, right_position));
                            }
                        }
                    }
                    let mut actual = actual;
                    actual.sort_unstable();
                    expected.sort_unstable();
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
