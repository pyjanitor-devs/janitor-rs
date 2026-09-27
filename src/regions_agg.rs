//! Aggregation over the streaming paper sweep.
//!
//! This module deliberately does not materialize candidate pairs. The region
//! sweep visits each primary candidate once; residual predicates are applied
//! while those candidates are being visited.
//!
//! Beginner's mental model:
//!
//! 1. The first two predicates have already been converted into two integer
//!    region labels for every left and right row.
//! 2. The first region label tells us which suffix of right rows may match.
//! 3. The ordered map groups that suffix by the second region label.
//! 4. The map range keeps only groups satisfying the second predicate.
//! 5. Each physical right position in those groups is an aggregation event.
//!
//! ## Position rules
//!
//! The sweep's left and right positions are compact region coordinates. They
//! are not automatically valid positions into aggregation arrays. Region
//! construction records `left_positions` and `right_positions` mappings back
//! to the source arrays; every residual predicate and every `AggregationSet`
//! update must use those mappings.
//!
//! The first two predicates are always the region anchors and must use `<`,
//! `<=`, `>`, or `>=`. Equality and inequality operators remain valid only in
//! later residual predicates. Residual predicates are evaluated before the
//! aggregation update, so null handling and predicate alignment retain the
//! same semantics as the other kernels.
//!
//! ## Aggregation tuple forms
//!
//! The first anchor accepts either:
//!
//! ```text
//! (left, left_index, right, right_index, ordered, operator)
//! (left, left_index, right, right_index,
//!  ordered, left_output_positions, right_output_positions, operator)
//! ```
//!
//! The ordering flag is validated but not used by regions; PyJanitor has
//! already sorted the right layout. The output-position arrays are returned to
//! Python and are also shape-checked against the source arrays. The second
//! anchor always uses the five-field region form.
//!
//! The eight-field output maps have the same tuple position as the other
//! aggregation families, but their meaning is kernel-specific. Region and
//! range aggregation use the map as the compact output-position labels that
//! accompany the result. The `!=` aggregation family in
//! `anchor_non_equi_join_agg.rs` instead treats its maps as a complete
//! physical-row-to-compact-slot permutation because its source arrays have
//! been filtered and reordered. Regions must not apply that permutation logic
//! to these maps.

use crate::aggs::aggregation::{make_results_with_positions, parse_inputs, AggregationSet};
use crate::join_aggregation_helpers::residuals;
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, predicates_match_dispatch, Predicate,
};
use crate::range_predicate::{
    parse_aggregation_range_anchor, AnyParsedRangePredicate, ParsedAggregationRangeAnchor,
};
use crate::regions;
use numpy::ndarray::ArrayView1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// The two named region anchors parsed at the Python/Rust boundary.
struct PreparedPredicates<'py> {
    /// First anchor, including optional output-position maps.
    first: ParsedAggregationRangeAnchor<'py>,
    /// Second anchor, which uses the five-field region form.
    second: AnyParsedRangePredicate<'py>,
}

/// Parse the two region aggregation anchors once and expose named Rust fields.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used while borrowing the tuples.
/// * `predicates` - Full predicate list. The first tuple must use either the
///   six-field form or the established eight-field form; the second tuple
///   must use the five-field region form.
///
/// # Returns
///
/// A named anchor containing the parsed range predicate, operator, and output
/// maps. NumPy arrays are borrowed; no column values are copied.
///
/// # Errors
///
/// Returns `ValueError` for invalid tuple length, a non-boolean ordering flag,
/// invalid output maps, unsupported dtypes/operators, or mismatched
/// value/index lengths.
///
/// Parsing is intentionally separate from residual parsing. The first two
/// tuples define the paper sweep and must be parsed before any later tuple is
/// interpreted; later tuples may use operators such as `==` or `!=` that are
/// valid residual filters but cannot define a region boundary.
///
/// The returned request keeps the first anchor's output maps alongside its
/// typed range data and keeps the second typed range separately. Later
/// residual predicates remain in the original Python list and are parsed by
/// the shared residual parser only for extended aggregation.
fn parse_region_anchors<'py>(predicates: &Bound<'py, PyList>) -> PyResult<PreparedPredicates<'py>> {
    let first_item = predicates.get_item(0)?;
    let first = first_item.cast::<PyTuple>()?;
    let second_item = predicates.get_item(1)?;
    let second = second_item.cast::<PyTuple>()?;
    let parsed_first = parse_aggregation_range_anchor(first, true)?;
    let parsed_second = parse_aggregation_range_anchor(second, false)?;
    // Keep the first anchor's output metadata attached to its range. The
    // second anchor contributes only a region path; it must not overwrite the
    // canonical first-anchor physical layout during alignment.
    Ok(PreparedPredicates {
        first: parsed_first,
        second: parsed_second.range,
    })
}

/// Describe the output and source layouts for one aggregation direction.
///
/// Region positions are compact traversal coordinates. The optional map is
/// only output metadata; it does not change the physical source positions
/// used by [`AggregationSet::update`]. This remains local because returning
/// a borrowed view from a shared anchor method would borrow the entire named
/// anchor and prevent moving its independent typed range into dispatch.
fn aggregation_layout<'py>(
    first: &'py ParsedAggregationRangeAnchor<'py>,
    reverse: bool,
) -> (Option<ArrayView1<'py, i64>>, usize, usize) {
    // The ordering flag is validated by the parser and intentionally does not
    // affect region traversal: PyJanitor supplies the sorted right layout.
    let _ordering_flag_was_validated = first.ordered;
    let output_positions = if reverse {
        first
            .right_output_positions
            .as_ref()
            .map(|values| values.as_array())
    } else {
        first
            .left_output_positions
            .as_ref()
            .map(|values| values.as_array())
    };
    // Forward aggregation has one slot per left source row and reads right
    // values. Reverse aggregation swaps those roles. A supplied map changes
    // only the labels returned to Python; the accumulator remains indexed by
    // the compact output layout supplied by PyJanitor.
    let output_len = output_positions
        .map(|values| values.len())
        .unwrap_or(if reverse {
            first.range.right_len()
        } else {
            first.range.left_len()
        });
    let source_len = if reverse {
        first.range.left_len()
    } else {
        first.range.right_len()
    };
    (output_positions, output_len, source_len)
}

/// Parse and align the two primary region anchors once for an aggregation call.
///
/// Exact and extended aggregation intentionally keep separate candidate
/// consumers, because only the extended path evaluates residual predicates.
/// They should nevertheless share this setup: parsing the first anchor twice
/// or calculating output/source lengths differently would make the two public
/// entry points disagree on malformed input or mapped output layouts.
///
/// # Arguments
///
/// * `predicates` - Predicate list whose first two entries are region anchors.
/// # Returns
///
/// The parsed anchors and aligned region paths. Output/source layout is
/// calculated by each caller after this function returns so a borrowed output
/// map remains tied to the parsed anchor rather than to a temporary tuple.
fn prepare_region_aggregation<'py>(
    predicates: &Bound<'py, PyList>,
) -> PyResult<(PreparedPredicates<'py>, regions::AlignedRegions)> {
    let prepared = parse_region_anchors(predicates)?;
    // Alignment happens once, before exact and extended consumers diverge.
    // This is what guarantees that residual predicates and aggregations see
    // the same compact-to-source position maps.
    let regions = regions::align_parsed(&prepared.first.range, &prepared.second)
        .map_err(PyValueError::new_err)?;
    Ok((prepared, regions))
}

/// Execute exact dual-region aggregation and build the standard Python result.
///
/// Only the first two predicates are accepted. Both are converted into region
/// labels, and every candidate surviving both labels updates the aggregation
/// set. There is no residual predicate phase on this path.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used for parsing and result
///   construction.
/// * `predicates` - Exactly two inequality region anchors. The first anchor
///   may include compact output-position maps.
/// * `aggregations` - Non-empty aggregation requests.
/// * `return_matched` - Whether to include the matched output mask.
/// * `reverse` - Whether left values are aggregated into right output slots.
///
/// # Returns
///
/// `None` when no candidate updates the aggregation state; otherwise the
/// standard aggregation result tuple.
///
/// # Errors
///
/// Returns a Python error for invalid predicate counts, malformed anchors,
/// invalid aggregation requests, or invalid output metadata.
#[allow(clippy::too_many_arguments)]
fn aggregate_regions_exact<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // This exact path is intentionally separate from the extended path. With
    // no residual predicates, the sweep can update aggregation state directly
    // for every candidate and does not need to build predicate views.
    if predicates.len() != 2 {
        return Err(PyValueError::new_err(
            "region aggregation requires exactly two predicates",
        ));
    }

    let (prepared, regions) = prepare_region_aggregation(predicates)?;
    let (output_positions, output_len, source_len) = aggregation_layout(&prepared.first, reverse);
    if regions.left_index.is_empty() || regions.right_index.is_empty() {
        return Ok(None);
    }

    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    // The optional maps label output slots. They do not replace the source
    // mappings in `regions`, which are needed when labels compact or reverse
    // the physical traversal layout.
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    // The shared traversal supplies compact region coordinates. Translate
    // them back to source coordinates only at the accumulator boundary.
    regions::traverse_candidates(&regions, |left_position, right_position| {
        // `left_position` and `right_position` are compact region positions.
        // Convert them through the source maps exactly once, at the point
        // where an aggregation event is recorded.
        if reverse {
            set.update(
                regions.left_positions[left_position],
                regions.right_positions[right_position],
            );
        } else {
            set.update(
                regions.right_positions[right_position],
                regions.left_positions[left_position],
            );
        }
        Ok(true)
    })
    .map_err(PyValueError::new_err)?;

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

/// Execute extended region aggregation and build the standard Python result.
///
/// This is the implementation behind the two extended Python functions:
///
/// ```text
/// region_extended_aggregate           residual filters, forward
/// region_extended_aggregate_reverse   residual filters, reverse
/// ```
///
/// `regions` are built from the first two predicates. Predicates after those
/// two anchors are evaluated for each primary candidate before aggregation.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used for parsing inputs and
///   constructing the result tuple.
/// * `predicates` - At least two region anchors, followed by optional residual
///   predicates. The first anchor may contain output-position maps in its
///   eight-field form.
/// * `aggregations` - Non-empty aggregation requests accepted by
///   `AggregationSet`, such as sum, min, max, product, size, or count.
/// * `return_matched` - Whether the result includes a boolean matched array.
/// * `reverse` - When false, aggregate right-side source values into left
///   output slots. When true, aggregate left-side source values into right
///   output slots.
///
/// # Returns
///
/// Returns `None` when no pair reaches an aggregation update. Otherwise
/// returns the established aggregation tuple: output positions, optionally
/// the matched mask, and the requested aggregation arrays.
///
/// # Errors
///
/// Returns a Python error for malformed anchors, invalid residual predicates,
/// misaligned predicate lengths, empty aggregation requests, invalid
/// aggregation inputs, or invalid output maps.
#[allow(clippy::too_many_arguments)]
fn aggregate_regions_extended<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // The extended path keeps the same region sweep as the exact path, but
    // inserts a residual-filter step between candidate discovery and the
    // aggregation update. This is why it cannot use the exact path's direct
    // update loop.
    // This implementation serves the two extended Python entry points.
    // `reverse` changes which side supplies values; residual predicates are
    // always part of this path.
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "region aggregation requires at least two predicates",
        ));
    }

    // Region construction reads only the first two anchors. It aligns their
    // left and right rows by original index labels, so the two independently
    // built region paths can be traversed together safely.
    let (prepared, regions) = prepare_region_aggregation(predicates)?;
    let (output_positions, output_len, source_len) = aggregation_layout(&prepared.first, reverse);
    if regions.left_index.is_empty() || regions.right_index.is_empty() {
        // At least one anchor has no surviving aligned rows. There can be no
        // aggregation event, so return the same no-match result as other
        // aggregation kernels.
        return Ok(None);
    }

    // Only predicates after the two region anchors are residual filters.
    // `residuals` also preserves nullable `!=` metadata for the shared
    // predicate matcher.
    let (parsed, metadata) = residuals(py, predicates, false, true)?;
    // Residual arrays retain the original source layout, including rows that
    // were removed from the compact region path. Validate against the source
    // lengths, then translate each compact candidate through the mappings
    // before invoking the shared matcher.
    check_predicate_lengths(&parsed, regions.left_len, regions.right_len)?;
    // Parse aggregation requests once before entering the sweep. The set
    // owns output accumulators while borrowing the source NumPy arrays.
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }

    // An eight-field first anchor carries compact output labels. These labels
    // are not the physical-to-local permutation used by `!=` aggregation.
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    // Convert parsed residual predicates into cheap Rust-side views once.
    // The hot loop then compares physical positions without repeatedly
    // touching Python objects.
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    // The shared traversal handles primary-region candidates. This callback
    // keeps the extended-only residual filtering directly before the update.
    regions::traverse_candidates(&regions, |left_position, right_position| {
        // Residual predicates use source positions, not compact region
        // positions. Filtering here means `AggregationSet::update` sees only
        // candidates that passed every predicate, including null semantics.
        let passes = predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            regions.left_positions[left_position],
            regions.right_positions[right_position],
        );
        if passes {
            if reverse {
                set.update(
                    regions.left_positions[left_position],
                    regions.right_positions[right_position],
                );
            } else {
                set.update(
                    regions.right_positions[right_position],
                    regions.left_positions[left_position],
                );
            }
        }
        Ok(true)
    })
    .map_err(PyValueError::new_err)?;

    if set.is_empty() {
        // The region sweep may find candidates, but null masks or residual
        // filters can still reject every aggregation update.
        return Ok(None);
    }
    // Convert accumulator state into the established Python tuple shape and
    // attach the compact-to-original output map when one was supplied.
    Ok(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?))
}

/// Aggregate exactly two region predicates in the forward direction.
///
/// Every pair satisfying both inequality anchors updates the aggregation
/// state. There are no residual filter predicates on this path.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two region predicates. The first may use the
///   six-field form or the eight-field form with output-position maps.
/// * `aggregations` - Non-empty aggregation requests over right-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no pair matches; otherwise the standard aggregation result
/// tuple aligned to the left output layout.
///
/// # Errors
///
/// Returns an error when the predicate count, anchor shapes, lengths, or
/// aggregation requests are invalid.
#[pyfunction]
pub fn region_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_exact(py, predicates, aggregations, return_matched, false)
}

/// Aggregate exactly two region predicates in the reverse direction.
///
/// This uses the same two region anchors as forward aggregation, but treats
/// left-side values as the aggregation source and right-side rows as output
/// slots.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - Exactly two region predicates.
/// * `aggregations` - Non-empty aggregation requests over left-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no pair matches; otherwise the standard aggregation result
/// tuple aligned to the right output layout.
///
/// # Errors
///
/// Returns an error when the predicate count, anchor shapes, lengths, or
/// aggregation requests are invalid.
#[pyfunction]
pub fn region_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_exact(py, predicates, aggregations, return_matched, true)
}

/// Aggregate two region predicates followed by residual filter predicates.
///
/// The first two predicates perform the efficient region sweep. Every later
/// predicate is evaluated against the physical left/right candidate before
/// the right-side source value updates a left-side aggregation slot.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two predicates: two inequality region anchors
///   followed by zero or more residual filters.
/// * `aggregations` - Non-empty aggregation requests over right-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no candidate survives the region anchors and residual filters;
/// otherwise the standard result tuple aligned to the left output layout.
///
/// # Errors
///
/// Returns an error for malformed predicates, misaligned arrays, unsupported
/// residual operators, or invalid aggregation requests.
#[pyfunction]
pub fn region_extended_aggregate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_extended(py, predicates, aggregations, return_matched, false)
}

/// Aggregate two region predicates plus residual filters in reverse direction.
///
/// The first two predicates define the candidate regions. Later predicates
/// filter those candidates. A passing left-side source value updates the
/// corresponding right-side aggregation slot.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token.
/// * `predicates` - At least two predicates: two inequality region anchors
///   followed by residual filters.
/// * `aggregations` - Non-empty aggregation requests over left-side values.
/// * `return_matched` - Whether to include the output matched mask.
///
/// # Returns
///
/// `None` when no complete candidate survives; otherwise the standard result
/// tuple aligned to the right output layout.
///
/// # Errors
///
/// Returns an error for malformed predicates, misaligned arrays, unsupported
/// residual operators, or invalid aggregation requests.
#[pyfunction]
pub fn region_extended_aggregate_reverse<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    aggregate_regions_extended(py, predicates, aggregations, return_matched, true)
}

/// Register all region aggregation Python entry points on the module.
///
/// # Arguments
///
/// * `m` - The parent `janitor_rs` Python module.
///
/// # Errors
///
/// Returns any PyO3 error raised while adding a function to `m`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(region_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(region_aggregate_reverse, m)?)?;
    m.add_function(wrap_pyfunction!(region_extended_aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(region_extended_aggregate_reverse, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    fn aggregation<'py>(py: Python<'py>, values: Vec<i64>) -> PyResult<Bound<'py, PyList>> {
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

    fn dual_predicates<'py>(
        py: Python<'py>,
        first_left: Vec<i64>,
        first_right: Vec<i64>,
        first_op: &str,
        second_op: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let predicates = PyList::empty(py);
        let left_index = PyArray1::from_vec(py, (0..first_left.len() as i64).collect());
        let right_index = PyArray1::from_vec(py, (0..first_right.len() as i64).collect());
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, first_left.clone()).into_any(),
                left_index.clone().into_any(),
                PyArray1::from_vec(py, first_right.clone()).into_any(),
                right_index.clone().into_any(),
                true.into_pyobject(py)?.to_owned().into_any(),
                first_op.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        predicates.append(PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, first_left).into_any(),
                left_index.into_any(),
                PyArray1::from_vec(py, first_right).into_any(),
                right_index.into_any(),
                second_op.into_pyobject(py)?.into_any(),
            ],
        )?)?;
        Ok(predicates)
    }

    fn result_parts<'py>(
        result: &Bound<'py, PyTuple>,
    ) -> PyResult<(Vec<i64>, Vec<bool>, Vec<i64>)> {
        let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
        let matched = result.get_item(1)?.extract::<Vec<bool>>()?;
        let outputs = result
            .get_item(2)?
            .cast::<PyList>()?
            .get_item(0)?
            .extract::<Vec<i64>>()?;
        Ok((positions, matched, outputs))
    }

    fn result_without_matched<'py>(
        result: &Bound<'py, PyTuple>,
    ) -> PyResult<(Vec<i64>, Vec<Vec<i64>>)> {
        let positions = result.get_item(0)?.extract::<Vec<i64>>()?;
        let outputs_item = result.get_item(1)?;
        let outputs = outputs_item.cast::<PyList>()?;
        let outputs = outputs
            .iter()
            .map(|output| output.extract::<Vec<i64>>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok((positions, outputs))
    }

    #[test]
    fn greater_than_anchors_aggregate_forward_and_reverse() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1, 3], vec![1, 2, 3, 4], ">=", "<=")?;

            let forward_inputs = aggregation(py, vec![10, 20, 30, 40])?;
            let forward = region_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("greater-than forward aggregation should match");
            assert_eq!(
                result_parts(&forward)?,
                (vec![0, 1], vec![true, true], vec![10, 30])
            );

            let reverse_inputs = aggregation(py, vec![100, 300])?;
            let reverse = region_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                .expect("greater-than reverse aggregation should match");
            assert_eq!(
                result_parts(&reverse)?,
                (
                    vec![0, 1, 2, 3],
                    vec![true, false, true, false],
                    vec![100, 0, 300, 0]
                )
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn reverse_anchor_boundary_zero_and_full_prefix_are_handled() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // With right values [1, 2, 3], `>=` has an empty prefix for left
            // value 0 (boundary 0) and a full prefix for left value 3
            // (boundary right_len). Both rows must retain their correct
            // semantics after the reverse layout normalization.
            let predicates = dual_predicates(py, vec![0, 3], vec![1, 2, 3], ">=", "<=")?;
            let forward_inputs = aggregation(py, vec![10, 20, 30])?;
            let forward = region_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("the full-prefix row should aggregate");
            assert_eq!(
                result_parts(&forward)?,
                (vec![0, 1], vec![false, true], vec![0, 30])
            );

            let reverse_inputs = aggregation(py, vec![100, 300])?;
            let reverse = region_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                .expect("the full-prefix row should aggregate in reverse");
            assert_eq!(
                result_parts(&reverse)?,
                (vec![0, 1, 2], vec![false, false, true], vec![0, 0, 300])
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn eight_field_anchor_preserves_maps_when_a_left_row_is_dropped() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            let left = PyArray1::from_vec(py, vec![4_i64, 1]);
            let left_index = PyArray1::from_vec(py, vec![0_i64, 1]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let right_index = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    left_index.clone().into_any(),
                    right.clone().into_any(),
                    right_index.clone().into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 20, 30]).into_any(),
                    "<=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    left_index.into_any(),
                    right.into_any(),
                    right_index.into_any(),
                    ">=".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let forward_inputs = aggregation(py, vec![10, 20, 30])?;
            let forward = region_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("mapped forward aggregation should match");
            assert_eq!(
                result_parts(&forward)?,
                (vec![100, 200], vec![false, true], vec![0, 10])
            );

            let reverse_inputs = aggregation(py, vec![4, 8])?;
            let reverse = region_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                .expect("mapped reverse aggregation should match");
            assert_eq!(
                result_parts(&reverse)?,
                (vec![10, 20, 30], vec![true, false, false], vec![8, 0, 0])
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn residuals_use_source_positions_after_reversal() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // Left row zero has no `>=` candidate and is removed from the
            // compact region path. Left row one matches right source position
            // zero. The residual values distinguish source position zero from
            // its reversed traversal position three.
            let predicates = dual_predicates(py, vec![0, 1], vec![1, 2, 3, 4], ">=", "<=")?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![100_i64, 200]).into_any(),
                    PyArray1::from_vec(py, vec![200_i64, 99, 98, 97]).into_any(),
                    "==".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let forward_inputs = aggregation(py, vec![10, 20, 30, 40])?;
            let forward = region_extended_aggregate(py, &predicates, &forward_inputs, true)?
                .expect("residual forward aggregation should match");
            assert_eq!(
                result_parts(&forward)?,
                (vec![0, 1], vec![false, true], vec![0, 10])
            );

            let reverse_inputs = aggregation(py, vec![100, 200])?;
            let reverse =
                region_extended_aggregate_reverse(py, &predicates, &reverse_inputs, true)?
                    .expect("residual reverse aggregation should match");
            assert_eq!(
                result_parts(&reverse)?,
                (
                    vec![0, 1, 2, 3],
                    vec![true, false, false, false],
                    vec![200, 0, 0, 0]
                )
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn exact_aggregation_supports_multiple_operations_masks_and_no_matched_output() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1, 2], vec![1, 2], "<=", ">=")?;
            let values = PyArray1::from_vec(py, vec![2_i64, 3]);
            let nulls = PyArray1::from_vec(py, vec![false, true]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "prod".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            "*".into_pyobject(py)?.into_any(),
                            "size".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "min".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            values.clone().into_any(),
                            nulls.clone().into_any(),
                            "max".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = region_aggregate(py, &predicates, &aggregations, false)?
                .expect("the two equality pairs should aggregate");
            let (positions, outputs) = result_without_matched(&result)?;
            assert_eq!(positions, vec![0, 1]);
            assert_eq!(
                outputs,
                vec![
                    vec![2, 0],
                    vec![2, 1],
                    vec![1, 0],
                    vec![1, 1],
                    vec![0, -1],
                    vec![0, -1]
                ]
            );

            let no_match = dual_predicates(py, vec![10], vec![1, 2], "<", "<")?;
            assert!(region_aggregate(py, &no_match, &aggregations, false)?.is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn aggregation_supports_unsigned_and_float_sources() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1], vec![1], "<=", ">=")?;
            let unsigned = PyArray1::from_vec(py, vec![7_u64]);
            let floats = PyArray1::from_vec(py, vec![1.5_f64]);
            let mask = PyArray1::from_vec(py, vec![false]);
            let aggregations = PyList::new(
                py,
                [
                    PyTuple::new(
                        py,
                        [
                            unsigned.into_any(),
                            mask.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            floats.into_any(),
                            mask.into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;
            let result = region_aggregate(py, &predicates, &aggregations, false)?
                .expect("the equality pair should aggregate");
            let outputs_item = result.get_item(1)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(outputs.get_item(0)?.extract::<Vec<u64>>()?, vec![7]);
            assert_eq!(outputs.get_item(1)?.extract::<Vec<f64>>()?, vec![1.5_f64]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn aggregation_rejects_malformed_residuals_without_panicking() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = dual_predicates(py, vec![1], vec![1], "<=", ">=")?;
            predicates.append(PyTuple::new(py, [1_i64.into_pyobject(py)?.into_any()])?)?;
            let aggregations = aggregation(py, vec![1])?;
            assert!(region_extended_aggregate(py, &predicates, &aggregations, true).is_err());
            Ok(())
        })
        .unwrap();
    }
}
