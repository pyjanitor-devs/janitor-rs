//! Shared helpers for residual-filtered join aggregation.
//!
//! These helpers are used by both single-anchor and dual-range extended
//! aggregation. They parse and validate residual predicates once, then expose
//! a common range-window aggregation/materialization path.

use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::common::range_window_bounds;
use crate::join_common::SingleJoinResult;
use crate::op::CompareOp;
use crate::predicate::{
    null_metadata_views, parse_predicates_with_nulls_strings, predicates_match_dispatch,
    PredicateView,
};
use crate::range_predicate::AnyParsedRangePredicate;

/// Build dense windows for an aggregation anchor.
///
/// Empty windows are retained because aggregation output slots must remain
/// aligned with the complete left layout. The caller may later compact the
/// windows when it is materializing index pairs instead of aggregations.
pub(crate) fn aggregation_windows(
    range: &AnyParsedRangePredicate<'_>,
    include_right_index: bool,
) -> Result<SingleJoinResult, String> {
    // Keep empty windows in the dense representation. The range-first index
    // path later compacts them, while aggregation needs the left-side slot
    // relationship to remain explicit for matched-mask construction.
    macro_rules! build {
        ($predicate:expr) => {{
            let predicate = $predicate;
            let (starts, ends) = range_window_bounds(
                predicate.left.as_array(),
                predicate.left_index.as_array(),
                predicate.right.as_array(),
                predicate.right_index.as_array(),
                predicate.op,
            )?;
            Ok(SingleJoinResult {
                left_positions: (0..predicate.left.as_array().len()).collect(),
                left_index: predicate.left_index.as_array().to_vec(),
                right_index: if include_right_index {
                    predicate.right_index.as_array().to_vec()
                } else {
                    Vec::new()
                },
                starts,
                ends,
            })
        }};
    }
    match range {
        AnyParsedRangePredicate::I64(predicate) => build!(predicate),
        AnyParsedRangePredicate::I32(predicate) => build!(predicate),
        AnyParsedRangePredicate::I16(predicate) => build!(predicate),
        AnyParsedRangePredicate::I8(predicate) => build!(predicate),
        AnyParsedRangePredicate::U64(predicate) => build!(predicate),
        AnyParsedRangePredicate::U32(predicate) => build!(predicate),
        AnyParsedRangePredicate::U16(predicate) => build!(predicate),
        AnyParsedRangePredicate::U8(predicate) => build!(predicate),
        AnyParsedRangePredicate::F64(predicate) => build!(predicate),
        AnyParsedRangePredicate::F32(predicate) => build!(predicate),
    }
}

/// Parse every predicate after the first extended-join predicate.
///
/// The first predicate is consumed by the range or `!=` candidate generator.
/// This helper copies the remaining Python tuple references into a temporary
/// list because the shared predicate parser expects a list containing exactly
/// the predicates it should evaluate. The underlying NumPy arrays are still
/// borrowed; their values are not copied.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to create the temporary list.
/// * `predicates` - Full extended predicate list, including the first anchor
///   predicate at position zero.
/// * `require_not_equal` - When true, validate that every residual operator is
///   `!=`, as required when the first predicate generates the all-`!=`
///   candidate stream. Range anchors pass false and allow all residual
///   operators supported by the shared predicate parser.
/// * `skip_second_range` - When true, consume predicates two onward as
///   residual input because predicate one and predicate two were already used
///   as dual range anchors. When false, consume predicates one onward.
///
/// # Returns
///
/// Parsed residual predicates and optional null metadata in their original
/// order.
pub(crate) fn residuals<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    require_not_equal: bool,
    skip_second_range: bool,
) -> PyResult<(
    Vec<crate::predicate::Predicate<'py>>,
    Option<Vec<crate::predicate::NullMetadata<'py>>>,
)> {
    let values = PyList::empty(py);
    let start = if skip_second_range { 2 } else { 1 };
    for item in predicates.iter().skip(start) {
        if require_not_equal {
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
        }
        values.append(item)?;
    }
    parse_predicates_with_nulls_strings(py, &values)
}

/// Visit a range-led extended join and update aggregations for survivors.
///
/// The first predicate has already been separated from `residuals`. For each
/// left value, `range_window` supplies the contiguous right-side window
/// that satisfies that first predicate. Every candidate in that window is
/// then checked against the remaining predicates with
/// [`predicates_match_dispatch`]. Aggregation state is updated only after all
/// residual predicates pass.
///
/// This function intentionally does not use prefix/suffix aggregation tables.
/// A residual predicate may reject arbitrary candidates inside the first
/// predicate's window, so the surviving positions are no longer guaranteed to
/// form a complete prefix or suffix.
///
/// # Arguments
///
/// * `windows` - Candidate windows produced by the range anchor. Their
///   `starts`/`ends` offsets address the sorted right layout.
/// * `residuals` - Parsed predicates after the first predicate. Their arrays
///   must be aligned to the same physical left and right positions.
/// * `residual_metadata` - Optional authoritative null metadata for residual
///   `!=` predicates.
/// * `set` - Aggregation state updated for every fully matching pair.
/// * `reverse` - Whether source values come from the left and output slots
///   are indexed by the right side.
/// * `physical_position_maps` - Translate compact candidate offsets through
///   the anchor's physical position arrays before updating aggregation state.
///   This is true only when Python passed full-layout aggregation arrays.
///
/// # Example
///
/// With sorted `right = [2, 5, 8]`, `left = [4]`, and `left < right`, the
/// anchor produces `right[1..] = [5, 8]`. A residual predicate can remove
/// either candidate before `set.update` is called.
#[allow(clippy::too_many_arguments)]
fn aggregate_range(
    windows: &SingleJoinResult,
    residuals: &[PredicateView<'_>],
    residual_metadata: Option<&[crate::predicate::NullMetadataView<'_>]>,
    set: &mut AggregationSet<'_>,
    reverse: bool,
    physical_position_maps: bool,
) {
    for (row, (&start, &end)) in windows.starts.iter().zip(windows.ends.iter()).enumerate() {
        let left_position = windows.left_positions[row];
        for right_position in start..end {
            if predicates_match_dispatch(
                residuals,
                residual_metadata,
                left_position,
                right_position,
            ) {
                if physical_position_maps {
                    let physical_left = usize::try_from(windows.left_index[row])
                        .expect("validated physical left position");
                    let physical_right = usize::try_from(windows.right_index[right_position])
                        .expect("validated physical right position");
                    if reverse {
                        set.update(physical_left, physical_right);
                    } else {
                        set.update(physical_right, physical_left);
                    }
                } else if reverse {
                    set.update(left_position, right_position);
                } else {
                    set.update(right_position, left_position);
                }
            }
        }
    }
}

/// Finish aggregation after a range window has been constructed.
///
/// This helper contains the shared state setup and result materialization for
/// both range modes. It deliberately knows nothing about whether the windows
/// came from one range predicate or the intersection of two predicates. The
/// caller has already performed candidate filtering, so this function only
/// walks the surviving windows and updates `AggregationSet`.
///
/// `output_positions` is a compact-slot-to-physical-row map. It is passed to
/// result construction so Python can align the trimmed aggregation output
/// with the original index without inferring order from labels. When it is
/// absent, the output layout is already identity ordered.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to construct NumPy results.
/// * `windows` - Half-open candidate windows, one sparse entry per left row
///   with at least one candidate.
/// * `parsed` - Residual predicates evaluated for every candidate in a window.
/// * `metadata` - Optional authoritative null metadata for those predicates.
/// * `aggregations` - Non-empty aggregation requests parsed by the shared
///   aggregation input layer.
/// * `output_positions` - Optional compact output-slot mapping. Forward range
///   aggregation uses the left-side map; reverse aggregation uses the
///   right-side map.
/// * `output_len` - Number of compact output slots.
/// * `source_len` - Number of source rows visible to the aggregation inputs.
/// * `return_matched` - Include the per-output match mask when true.
/// * `reverse` - Select forward or reverse source/output orientation.
/// * `physical_position_maps` - When true, translate compact candidate
///   offsets through `windows.left_index` and `windows.right_index` before
///   updating aggregation state. This is used when aggregation inputs retain
///   the original full Python layouts. Existing aligned-layout callers pass
///   false because their arrays already use compact offsets.
///
/// # Returns
///
/// Returns `(output_positions, matched, aggregation_arrays)` when
/// `return_matched` is true, `(output_positions, aggregation_arrays)` when it
/// is false, or `None` when no candidate survives the residual predicates.
///
/// # Errors
///
/// Returns a Python `ValueError` when aggregation requests or output metadata
/// are invalid.
#[allow(clippy::too_many_arguments)]
pub(crate) fn aggregate_range_windows<'py>(
    py: Python<'py>,
    windows: SingleJoinResult,
    parsed: &[crate::predicate::Predicate<'_>],
    metadata: Option<&[crate::predicate::NullMetadata<'_>]>,
    aggregations: &Bound<'py, PyList>,
    output_positions: Option<ArrayView1<'_, i64>>,
    output_len: usize,
    source_len: usize,
    return_matched: bool,
    reverse: bool,
    physical_position_maps: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    if parsed.is_empty() {
        // With no residual predicates, every position in each intersected
        // window is a successful pair. Convert the sparse window list into
        // the dense boundary layout expected by the adaptive starts/ends
        // aggregators. Invalid slots remain `-1`, which those kernels treat
        // as no range. This preserves the segment-tree/boundary optimization
        // without materializing every pair in the windows.
        let boundary_len = if reverse { source_len } else { output_len };
        let mut starts = vec![-1_i64; boundary_len];
        let mut ends = vec![-1_i64; boundary_len];
        for (row, (&start, &end)) in windows.starts.iter().zip(&windows.ends).enumerate() {
            // The window rows retain the original left physical position.
            // Forward aggregation uses it as the left output slot; reverse
            // aggregation uses it as the source-left slot for the same
            // right-oriented range.
            let output_position = windows.left_positions[row];
            let start = i64::try_from(start)
                .map_err(|_| PyValueError::new_err("range aggregation start exceeds int64"))?;
            let end = i64::try_from(end)
                .map_err(|_| PyValueError::new_err("range aggregation end exceeds int64"))?;
            let start_slot = starts.get_mut(output_position).ok_or_else(|| {
                PyValueError::new_err("range aggregation output position is out of bounds")
            })?;
            let end_slot = ends.get_mut(output_position).ok_or_else(|| {
                PyValueError::new_err("range aggregation output position is out of bounds")
            })?;
            *start_slot = start;
            *end_slot = end;
        }
        let starts = ArrayView1::from(&starts[..]);
        let ends = ArrayView1::from(&ends[..]);
        if reverse {
            set.aggregate_reverse_starts_ends(starts, ends);
        } else {
            set.aggregate_starts_ends(starts, ends);
        }
        if set.is_empty() {
            return Ok(None);
        }
        return Ok(Some(make_results_with_positions(
            py,
            set,
            output_positions,
            output_len,
            return_matched,
        )?));
    }

    let views: Vec<_> = parsed.iter().map(|predicate| predicate.view()).collect();
    let metadata_views = metadata.map(null_metadata_views);
    aggregate_range(
        &windows,
        &views,
        metadata_views.as_deref(),
        &mut set,
        reverse,
        physical_position_maps,
    );
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
