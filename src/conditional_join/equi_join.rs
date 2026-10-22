//! Materialize indices for equality-led conditional joins.
//!
//! PyJanitor supplies one of two positional layouts:
//!
//! * unique equality keys: `left_index[i]` is aligned directly with
//!   `right_index[i]`; and
//! * duplicate equality keys: `starts[i]..ends[i]` is a half-open window into
//!   `right_index` for `left_index[i]`.
//!
//! The unique layout is deliberately window-free. It is used when the right
//! equality keys are unique, including when range or residual predicates are
//! evaluated on the already-aligned rows. The window layout is used when a
//! left row may match multiple right rows.

use numpy::ndarray::{s, ArrayView1, ArrayViewMut1};
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::cmp::Ordering;
use std::collections::{hash_map::Entry, HashMap};

use super::compare_op::CompareOp;
use super::join_search::partition_point_between;
use super::join_types::{result_dict, Keep};
use super::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, predicates_match_window, NullMetadataView, Predicate, PredicateView,
};
use super::range_join::adaptive_extreme_positions;
use super::range_predicate::{parse_any_range_parts, AnyParsedRangePredicate};
use super::range_skip_scan::{second_range_skip_pays_off, SecondRangeSkipScan};
use crate::aggregation::common::aggregation::{
    make_results_with_positions, parse_inputs, AggregationSet,
};
use crate::aggregation::common::ensure_equal_lengths_core;

#[cfg(test)]
thread_local! {
    static EQUI_STARTS_ENDS_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static EQUI_RESIDUAL_WINDOW_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

type IndexPairs = Option<(Vec<i64>, Vec<i64>)>;

/// Update one row's selected right position and report whether scanning may
/// stop. `first` and `last` compare physical right labels, not sorted offsets.
fn update_keep_selection(
    selected: &mut Option<usize>,
    candidate: usize,
    keep: Keep,
    right_index: ArrayView1<'_, i64>,
) -> bool {
    match keep {
        Keep::Any => {
            *selected = Some(candidate);
            false
        }
        Keep::First => {
            if selected.is_none_or(|current| right_index[candidate] < right_index[current]) {
                *selected = Some(candidate);
            }
            true
        }
        Keep::Last => {
            if selected.is_none_or(|current| right_index[candidate] > right_index[current]) {
                *selected = Some(candidate);
            }
            true
        }
        Keep::All => unreachable!("keep='all' uses the all-matches path"),
    }
}

/// Return the unique reverse-output labels and their dense aggregation slots.
///
/// The unique-equality Python preparation aligns one right label beside every
/// surviving left row. Repeated left keys therefore repeat the same right
/// label. Reverse aggregation must collapse those repeated labels back to one
/// output slot before updating the aggregation state.
fn unique_reverse_output_slots(
    right_index: ArrayView1<'_, i64>,
) -> (Vec<i64>, HashMap<i64, usize>) {
    let mut labels = Vec::with_capacity(right_index.len());
    let mut slots = HashMap::with_capacity(right_index.len());
    for &label in right_index {
        if let Entry::Vacant(entry) = slots.entry(label) {
            let slot = labels.len();
            labels.push(label);
            entry.insert(slot);
        }
    }
    (labels, slots)
}

pub fn validate_start_end(
    start: i64,
    end: i64,
    right_index_len: usize,
) -> Result<(usize, usize), String> {
    let start = usize::try_from(start).map_err(|_| "start must be non-negative")?;
    let end = usize::try_from(end).map_err(|_| "end must be non-negative")?;
    if start > end {
        return Err("start must not exceed end".to_owned());
    }
    if end > right_index_len {
        return Err("end exceeds right index length".to_owned());
    }
    Ok((start, end))
}

/// Parse the three-field range tuples used by the equi entry points.
pub(crate) fn parse_equi_range_predicates<'py>(
    range_predicates: &Bound<'py, PyList>,
    left_index: &Bound<'py, PyAny>,
    right_index: &Bound<'py, PyAny>,
) -> PyResult<Vec<AnyParsedRangePredicate<'py>>> {
    range_predicates
        .iter()
        .map(|item| {
            let tuple = item
                .cast::<PyTuple>()
                .map_err(|_| PyValueError::new_err("each equi range must be a tuple"))?;
            if tuple.len() != 3 {
                return Err(PyValueError::new_err(
                    "equi range predicates must contain 3 elements",
                ));
            }
            parse_any_range_parts(
                &tuple.get_item(0)?,
                left_index,
                &tuple.get_item(1)?,
                right_index,
                &tuple.get_item(2)?,
            )
        })
        .collect()
}

/// Validate the public equi-join range-predicate contract.
pub(crate) fn validate_equi_range_predicate_count(count: usize) -> PyResult<()> {
    if count > 2 {
        return Err(PyValueError::new_err(
            "equi range path accepts at most two range predicates",
        ));
    }
    Ok(())
}

/// Apply one or two range predicates to equality windows and materialize pairs.
///
/// The first predicate is applied with a binary search over the right values,
/// which are sorted for that predicate. If a second predicate is unordered,
/// its post-first windows are narrowed with a cumulative min/max envelope and
/// a second binary search before the exact predicate scan.
///
/// # Arguments
///
/// * `left_index` - Physical left-row labels for the compact left layout.
/// * `right_index` - Physical right-row labels for the compact right layout.
/// * `starts` - Mutable half-open right-window starts, aligned with
///   `left_index`.
/// * `ends` - Mutable half-open right-window ends, aligned with `left_index`.
/// * `first_range` - The range predicate whose right values are sorted.
/// * `second_range` - An optional second range predicate in the same layout.
/// * `keep` - Match retention policy for final materialization.
///
/// # Errors
///
/// Returns an error for mismatched lengths, invalid windows, or a non-range
/// comparator passed through the range path.
fn build_equi_range_indices_core(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    mut starts: ArrayViewMut1<'_, i64>,
    mut ends: ArrayViewMut1<'_, i64>,
    first_range: &AnyParsedRangePredicate<'_>,
    second_range: Option<&AnyParsedRangePredicate<'_>>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    ensure_equal_lengths_core("left index", left_index.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("left index", left_index.len(), "ends", ends.len())?;
    // Keep the equality windows intact: the first range narrows `starts` and
    // `ends` in place, but the unordered second-range envelope must be built
    // from the original group window rather than from overlapping narrowed
    // windows that can overwrite one another.
    let equality_windows = second_range.map(|_| (starts.to_owned(), ends.to_owned()));
    for (&start, &end) in starts.iter().zip(ends.iter()) {
        if start < 0 || end < 0 {
            continue;
        }
        validate_start_end(start, end, right_index.len())?;
    }
    let has_active_window = apply_equi_range_windows(
        first_range,
        starts.view_mut(),
        ends.view_mut(),
        right_index.len(),
    )?;
    if !has_active_window {
        return Ok(None);
    }
    if let Some(second_range) = second_range {
        validate_equi_range_inputs(second_range, starts.len(), right_index.len())?;
        if range_is_monotonic_for_windows(second_range, starts.view(), ends.view()) {
            let has_active_window = apply_equi_range_windows(
                second_range,
                starts.view_mut(),
                ends.view_mut(),
                right_index.len(),
            )?;
            if !has_active_window {
                return Ok(None);
            }
        } else {
            let (equality_starts, equality_ends) = equality_windows
                .as_ref()
                .expect("second range must retain the equality windows");
            return materialize_unordered_second_range(
                left_index,
                right_index,
                starts.view_mut(),
                ends.view_mut(),
                second_range,
                equality_starts.view(),
                equality_ends.view(),
                &[],
                None,
                keep,
            );
        }
    }

    materialize_equi_windows(left_index, right_index, starts.view(), ends.view(), keep)
}

fn apply_equi_range_windows(
    range: &AnyParsedRangePredicate<'_>,
    starts: ArrayViewMut1<'_, i64>,
    ends: ArrayViewMut1<'_, i64>,
    right_index_len: usize,
) -> Result<bool, String> {
    validate_equi_range_inputs(range, starts.len(), right_index_len)?;

    macro_rules! apply {
        ($predicate:expr) => {{
            let predicate = $predicate;
            apply_equi_range_bound(
                predicate.left.as_array(),
                predicate.right.as_array(),
                starts,
                ends,
                predicate.op,
            )
        }};
    }
    match range {
        AnyParsedRangePredicate::I64(predicate) => apply!(predicate),
        AnyParsedRangePredicate::I32(predicate) => apply!(predicate),
        AnyParsedRangePredicate::I16(predicate) => apply!(predicate),
        AnyParsedRangePredicate::I8(predicate) => apply!(predicate),
        AnyParsedRangePredicate::U64(predicate) => apply!(predicate),
        AnyParsedRangePredicate::U32(predicate) => apply!(predicate),
        AnyParsedRangePredicate::U16(predicate) => apply!(predicate),
        AnyParsedRangePredicate::U8(predicate) => apply!(predicate),
        AnyParsedRangePredicate::F64(predicate) => apply!(predicate),
        AnyParsedRangePredicate::F32(predicate) => apply!(predicate),
    }
}

fn validate_equi_range_inputs(
    range: &AnyParsedRangePredicate<'_>,
    window_len: usize,
    right_index_len: usize,
) -> Result<(), String> {
    range.validate_lengths()?;
    ensure_equal_lengths_core("left values", range.left_len(), "windows", window_len)?;
    ensure_equal_lengths_core(
        "right values",
        range.right_len(),
        "right index",
        right_index_len,
    )?;
    Ok(())
}

fn range_is_monotonic_for_windows(
    range: &AnyParsedRangePredicate<'_>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
) -> bool {
    macro_rules! check {
        ($predicate:expr) => {{
            let right = $predicate.right.as_array();
            let mut bad_prefix = vec![0usize; right.len()];
            for position in 1..right.len() {
                bad_prefix[position] = bad_prefix[position - 1]
                    + usize::from(!matches!(
                        right[position - 1].partial_cmp(&right[position]),
                        Some(Ordering::Less | Ordering::Equal)
                    ));
            }
            for (&start, &end) in starts.iter().zip(ends.iter()) {
                if start < 0 || end < 0 {
                    continue;
                }
                let start = start as usize;
                let end = end as usize;
                if end > start + 1 && bad_prefix[end - 1] != bad_prefix[start] {
                    return false;
                }
            }
            true
        }};
    }

    match range {
        AnyParsedRangePredicate::I64(predicate) => check!(predicate),
        AnyParsedRangePredicate::I32(predicate) => check!(predicate),
        AnyParsedRangePredicate::I16(predicate) => check!(predicate),
        AnyParsedRangePredicate::I8(predicate) => check!(predicate),
        AnyParsedRangePredicate::U64(predicate) => check!(predicate),
        AnyParsedRangePredicate::U32(predicate) => check!(predicate),
        AnyParsedRangePredicate::U16(predicate) => check!(predicate),
        AnyParsedRangePredicate::U8(predicate) => check!(predicate),
        AnyParsedRangePredicate::F64(predicate) => check!(predicate),
        AnyParsedRangePredicate::F32(predicate) => check!(predicate),
    }
}

fn materialize_equi_windows(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    materialize_equi_windows_and_residuals(
        left_index,
        right_index,
        starts,
        ends,
        &[],
        None,
        None,
        keep,
    )
}

/// Materialize matches for an unordered second range predicate.
///
/// `starts` and `ends` already contain the windows surviving the first range.
/// The function builds one cumulative envelope for each distinct active
/// window, binary-searches that envelope to remove a provably impossible
/// prefix or suffix, and then evaluates the original predicate exactly over
/// the remaining window. The envelope is never treated as the final truth
/// test.
///
/// PyJanitor filters nulls before entering this Rust path, so the supported
/// inputs are expected to contain ordinary ordered values rather than `NaN`.
///
/// # Arguments
///
/// * `left_index` - Physical left-row labels aligned with the predicate's
///   left values.
/// * `right_index` - Physical right-row labels aligned with the predicate's
///   right values.
/// * `starts` - Mutable post-first-range half-open windows.
/// * `ends` - Mutable post-first-range half-open windows.
/// * `range` - The unordered second range predicate.
/// * `equality_starts` - Original equality-window starts, captured before the
///   first range narrows the mutable windows.
/// * `equality_ends` - Original equality-window ends, paired with
///   `equality_starts`.
/// * `keep` - Match retention policy after exact filtering.
///
/// # Returns
///
/// `None` when no exact matches survive; otherwise the physical left/right
/// index pairs selected by `keep`.
#[allow(clippy::too_many_arguments)]
fn materialize_unordered_second_range(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    mut starts: ArrayViewMut1<'_, i64>,
    mut ends: ArrayViewMut1<'_, i64>,
    range: &AnyParsedRangePredicate<'_>,
    equality_starts: ArrayView1<'_, i64>,
    equality_ends: ArrayView1<'_, i64>,
    residual_predicates: &[PredicateView<'_>],
    null_metadata: Option<&[NullMetadataView<'_>]>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    if !narrow_unordered_second_range_for_predicate(
        range,
        equality_starts,
        equality_ends,
        starts.view_mut(),
        ends.view_mut(),
    )? {
        return Ok(None);
    }
    let use_skip = second_range_skip_pays_off(right_index.len(), starts.view(), ends.view());
    if !use_skip && !residual_predicates.is_empty() {
        // ELI5: when the shelves are short anyway, keep the residual
        // checklist that crosses off whole shelves at once; the skip scan only
        // replaces it when it can skip most of the candidates.
        return materialize_equi_windows_and_residuals(
            left_index,
            right_index,
            starts.view(),
            ends.view(),
            residual_predicates,
            null_metadata,
            Some(range),
            keep,
        );
    }
    // ELI5: the skip scan hands over only candidates that pass the second range;
    // each one must still pass every residual rule to count as a match.
    let residuals_pass = |row: usize, right_position: usize| {
        residual_predicates.is_empty()
            || predicates_match_dispatch(residual_predicates, null_metadata, row, right_position)
    };

    macro_rules! materialize {
        ($predicate:expr) => {{
            let predicate = $predicate;
            let left = predicate.left.as_array();
            let right = predicate.right.as_array();
            let scanner = if use_skip {
                SecondRangeSkipScan::new(right, predicate.op)
            } else {
                SecondRangeSkipScan::direct(right, predicate.op)
            };
            if keep == Keep::All {
                let mut output_capacity = 0_usize;
                for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                    if start < 0 || end < 0 {
                        continue;
                    }
                    let (start, end) = (start as usize, end as usize);
                    let row_matches = if residual_predicates.is_empty() {
                        scanner.count_matches(left[row], start, end)
                    } else {
                        let mut row_matches = 0_usize;
                        scanner.for_each_match(left[row], start, end, |right_position| {
                            // ELI5: one window holds at most every right
                            // row, so a single row's count cannot overflow.
                            row_matches += usize::from(residuals_pass(row, right_position));
                            true
                        });
                        row_matches
                    };
                    output_capacity = output_capacity
                        .checked_add(row_matches)
                        .ok_or("equi-join result size exceeds platform capacity")?;
                }
                if output_capacity == 0 {
                    return Ok(None);
                }
                let mut left_output = Vec::with_capacity(output_capacity);
                let mut right_output = Vec::with_capacity(output_capacity);
                for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                    if start < 0 || end < 0 {
                        continue;
                    }
                    scanner.for_each_match(
                        left[row],
                        start as usize,
                        end as usize,
                        |right_position| {
                            if residuals_pass(row, right_position) {
                                left_output.push(left_index[row]);
                                right_output.push(right_index[right_position]);
                            }
                            true
                        },
                    );
                }
                return Ok(Some((left_output, right_output)));
            }

            let mut left_output = Vec::with_capacity(left_index.len());
            let mut right_output = Vec::with_capacity(left_index.len());
            for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                if start < 0 || end < 0 {
                    continue;
                }
                let mut selected = None;
                scanner.for_each_match(left[row], start as usize, end as usize, |right_position| {
                    !residuals_pass(row, right_position)
                        || update_keep_selection(&mut selected, right_position, keep, right_index)
                });
                if let Some(right_position) = selected {
                    left_output.push(left_index[row]);
                    right_output.push(right_index[right_position]);
                }
            }
            if left_output.is_empty() {
                Ok(None)
            } else {
                Ok(Some((left_output, right_output)))
            }
        }};
    }

    match range {
        AnyParsedRangePredicate::I64(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::I32(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::I16(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::I8(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::U64(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::U32(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::U16(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::U8(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::F64(predicate) => materialize!(predicate),
        AnyParsedRangePredicate::F32(predicate) => materialize!(predicate),
    }
}

/// Build and apply the second-range envelope for already narrowed windows.
///
/// This helper is shared by direct index materialization and duplicate-key
/// aggregation. Both paths narrow candidate windows before the exact scan;
/// the residual check remains necessary because the envelope is only a safe
/// pruning proof.
///
/// # Arguments
///
/// * `range` - The unordered second range predicate and its aligned values.
/// * `equality_starts` / `equality_ends` - Original equality windows used to
///   build the cumulative envelope before the first range narrows candidates.
/// * `starts` - Mutable post-first-range window starts.
/// * `ends` - Mutable post-first-range window ends.
///
/// # Returns
///
/// `true` when at least one candidate window remains after envelope narrowing.
fn narrow_unordered_second_range_for_predicate(
    range: &AnyParsedRangePredicate<'_>,
    equality_starts: ArrayView1<'_, i64>,
    equality_ends: ArrayView1<'_, i64>,
    starts: ArrayViewMut1<'_, i64>,
    ends: ArrayViewMut1<'_, i64>,
) -> Result<bool, String> {
    macro_rules! narrow {
        ($predicate:expr) => {{
            let predicate = $predicate;
            let left = predicate.left.as_array();
            let right = predicate.right.as_array();
            let use_cummax = matches!(predicate.op, CompareOp::Lt | CompareOp::Le);
            let mut equality_windows = Vec::new();
            let mut seen_windows = std::collections::HashSet::new();
            for (&start, &end) in equality_starts.iter().zip(equality_ends.iter()) {
                if start < 0 || end < 0 {
                    continue;
                }
                let window = (start as usize, end as usize);
                if seen_windows.insert(window) {
                    equality_windows.push(window);
                }
            }
            let envelope = build_second_range_envelope(right, &equality_windows, use_cummax);
            narrow_unordered_second_range(left, &envelope, starts, ends, predicate.op)
        }};
    }

    match range {
        AnyParsedRangePredicate::I64(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::I32(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::I16(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::I8(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::U64(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::U32(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::U16(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::U8(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::F64(predicate) => narrow!(predicate),
        AnyParsedRangePredicate::F32(predicate) => narrow!(predicate),
    }
}

/// Build one cumulative envelope for each equality window.
///
/// The envelope is reset at every equality window, so values from a
/// neighboring equality group cannot affect the binary search.
///
/// When `use_cummax` is true, the envelope is a prefix maximum and is
/// non-decreasing. Otherwise it is a suffix minimum and is also
/// non-decreasing when read from left to right.
///
/// # Arguments
///
/// * `values` - Unordered second-range values in the prepared right layout.
/// * `equality_windows` - Distinct active post-first-range windows.
/// * `use_cummax` - Select the prefix-maximum or suffix-minimum envelope.
///
/// # Returns
///
/// An envelope indexed in the same coordinate system as `values`.
fn build_second_range_envelope<T: PartialOrd + Copy>(
    values: ArrayView1<'_, T>,
    equality_windows: &[(usize, usize)],
    use_cummax: bool,
) -> Vec<T> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut envelope = vec![values[0]; values.len()];
    for &(start, end) in equality_windows {
        if start >= end {
            continue;
        }
        if use_cummax {
            let mut current = values[start];
            for position in start..end {
                let value = values[position];
                if matches!(value.partial_cmp(&current), Some(Ordering::Greater)) {
                    current = value;
                }
                envelope[position] = current;
            }
        } else {
            let mut current = values[end - 1];
            for position in (start..end).rev() {
                let value = values[position];
                if matches!(value.partial_cmp(&current), Some(Ordering::Less)) {
                    current = value;
                }
                envelope[position] = current;
            }
        }
    }
    envelope
}

/// Narrow the second-range windows using a monotonic cumulative envelope.
///
/// The raw second-range values may be unordered. The envelope only proves
/// that a prefix or suffix cannot contain a match; the caller still performs
/// the exact predicate check over the narrowed window.
///
/// # Arguments
///
/// * `left` - Second-range left values, one per active window.
/// * `envelope` - Monotonic cumulative envelope in right-position space.
/// * `starts` - Mutable post-first-range window starts.
/// * `ends` - Mutable post-first-range window ends.
/// * `op` - One of `<`, `<=`, `>`, or `>=`; equality operators are invalid.
///
/// # Returns
///
/// `true` when at least one narrowed window remains active.
fn narrow_unordered_second_range<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    envelope: &[T],
    mut starts: ArrayViewMut1<'_, i64>,
    mut ends: ArrayViewMut1<'_, i64>,
    op: CompareOp,
) -> Result<bool, String> {
    let envelope = ArrayView1::from(envelope);
    let mut has_active_window = false;
    for row in 0..left.len() {
        if starts[row] < 0 || ends[row] < 0 {
            continue;
        }
        let old_start = starts[row] as usize;
        let old_end = ends[row] as usize;
        let boundary = match op {
            // For left <(=) right, the prefix cummax is non-decreasing.
            CompareOp::Lt => {
                partition_point_between(envelope, old_start, old_end, |value| value <= left[row])
            }
            CompareOp::Le => {
                partition_point_between(envelope, old_start, old_end, |value| value < left[row])
            }
            // For left >(=) right, the suffix cummin is non-decreasing.
            CompareOp::Gt => {
                partition_point_between(envelope, old_start, old_end, |value| value < left[row])
            }
            CompareOp::Ge => {
                partition_point_between(envelope, old_start, old_end, |value| value <= left[row])
            }
            CompareOp::Eq | CompareOp::Ne => {
                return Err("unordered second range requires a range comparator".to_owned())
            }
        };
        match op {
            CompareOp::Lt | CompareOp::Le => {
                starts[row] = if boundary == old_end {
                    -1
                } else {
                    has_active_window = true;
                    boundary as i64
                };
            }
            CompareOp::Gt | CompareOp::Ge => {
                ends[row] = if boundary == old_start {
                    -1
                } else {
                    has_active_window = true;
                    boundary as i64
                };
            }
            CompareOp::Eq | CompareOp::Ne => unreachable!(),
        }
    }
    Ok(has_active_window)
}

/// Borrow a range predicate as a residual-style view.
///
/// Building the NumPy views once keeps per-candidate range checks as cheap
/// as residual predicate checks.
fn range_predicate_view<'a>(range: &'a AnyParsedRangePredicate<'_>) -> PredicateView<'a> {
    macro_rules! view {
        ($variant:ident, $predicate:expr) => {
            PredicateView::$variant(
                $predicate.left.as_array(),
                $predicate.right.as_array(),
                $predicate.op,
            )
        };
    }
    match range {
        AnyParsedRangePredicate::I64(predicate) => view!(I64, predicate),
        AnyParsedRangePredicate::I32(predicate) => view!(I32, predicate),
        AnyParsedRangePredicate::I16(predicate) => view!(I16, predicate),
        AnyParsedRangePredicate::I8(predicate) => view!(I8, predicate),
        AnyParsedRangePredicate::U64(predicate) => view!(U64, predicate),
        AnyParsedRangePredicate::U32(predicate) => view!(U32, predicate),
        AnyParsedRangePredicate::U16(predicate) => view!(U16, predicate),
        AnyParsedRangePredicate::U8(predicate) => view!(U8, predicate),
        AnyParsedRangePredicate::F64(predicate) => view!(F64, predicate),
        AnyParsedRangePredicate::F32(predicate) => view!(F32, predicate),
    }
}

fn apply_equi_range_bound<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    mut starts: ArrayViewMut1<'_, i64>,
    mut ends: ArrayViewMut1<'_, i64>,
    op: CompareOp,
) -> Result<bool, String> {
    let mut has_active_window = false;
    for row in 0..left.len() {
        if starts[row] < 0 || ends[row] < 0 {
            continue;
        }
        let old_start = starts[row] as usize;
        let old_end = ends[row] as usize;
        let boundary = match op {
            CompareOp::Lt => {
                partition_point_between(right, old_start, old_end, |value| value <= left[row])
            }
            CompareOp::Le => {
                partition_point_between(right, old_start, old_end, |value| value < left[row])
            }
            CompareOp::Gt => {
                partition_point_between(right, old_start, old_end, |value| value < left[row])
            }
            CompareOp::Ge => {
                partition_point_between(right, old_start, old_end, |value| value <= left[row])
            }
            CompareOp::Eq | CompareOp::Ne => {
                return Err("range bound requires a range comparator".to_owned())
            }
        };

        match op {
            CompareOp::Lt | CompareOp::Le => {
                starts[row] = if boundary == old_end {
                    -1
                } else {
                    has_active_window = true;
                    boundary as i64
                };
            }
            CompareOp::Gt | CompareOp::Ge => {
                ends[row] = if boundary == old_start {
                    -1
                } else {
                    has_active_window = true;
                    boundary as i64
                };
            }
            CompareOp::Eq | CompareOp::Ne => unreachable!(),
        }
    }
    Ok(has_active_window)
}

#[pyfunction]
pub fn equi_range_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    mut starts: PyReadwriteArray1<'py, i64>,
    mut ends: PyReadwriteArray1<'py, i64>,
    range_predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if range_predicates.is_empty() {
        return Err(PyValueError::new_err("range_predicates must not be empty"));
    }
    validate_equi_range_predicate_count(range_predicates.len())?;
    let keep = Keep::parse(keep)?;
    let left_index_array = left_index;
    let right_index_array = right_index;
    let left_index = left_index_array.as_array();
    let right_index = right_index_array.as_array();
    let starts = starts.as_array_mut();
    let ends = ends.as_array_mut();
    let ranges = parse_equi_range_predicates(
        range_predicates,
        left_index_array.as_any(),
        right_index_array.as_any(),
    )?;
    let first_range = ranges
        .first()
        .ok_or_else(|| PyValueError::new_err("missing first range predicate"))?;
    let second_range = if ranges.len() == 2 {
        Some(
            ranges
                .get(1)
                .ok_or_else(|| PyValueError::new_err("missing second range predicate"))?,
        )
    } else {
        None
    };
    let result = build_equi_range_indices_core(
        left_index,
        right_index,
        starts,
        ends,
        first_range,
        second_range,
        keep,
    )
    .map_err(PyValueError::new_err)?;
    result
        .map(|(left, right)| result_dict(py, left, right, None, None))
        .transpose()
}

#[allow(clippy::too_many_arguments)]
/// Materialize equality-window matches after range and residual filtering.
///
/// `starts` and `ends` describe one half-open right-side candidate window for
/// each compact left row. The function has three intentionally separate
/// paths: `Keep::All` emits every surviving pair, `Keep::First`/`Keep::Last`
/// with residuals scans candidates because each candidate must be checked, and
/// the no-residual `First`/`Last` case reduces each exact window to one
/// physical extreme using the adaptive direct-scan/segment-tree helper.
///
/// ELI5: when there are extra rules, we must interview every candidate before
/// choosing one. When there are no extra rules, the first/last answer is just
/// the smallest/largest physical label on that shelf, so a shared shelf index
/// can answer many overlapping rows more cheaply.
///
/// # Arguments
///
/// * `left_index` - Physical labels for compact left rows; output left labels
///   are read from this array.
/// * `right_index` - Physical labels for the prepared right layout; output
///   right labels and extrema comparisons are read from this array.
/// * `starts` - Inclusive right-window offsets, aligned with `left_index`.
/// * `ends` - Exclusive right-window offsets, aligned with `left_index`.
/// * `residual_predicates` - Predicates that must be evaluated for each
///   candidate after the range windows have been built.
/// * `null_metadata` - Optional null masks used by residual predicate dispatch.
/// * `additional_range` - Optional extra range predicate that must be checked
///   for each candidate; its presence disables the extrema shortcut.
/// * `keep` - Emit all matches, one first/last physical label, or the first
///   passing candidate for `Any`.
///
/// # Caller contract
///
/// The calling core checks parallel-array lengths and input bounds before
/// range narrowing. Negative prepared bounds mark inactive rows and are
/// skipped, as are valid empty windows. The `All` path relies on that caller
/// validation; the other paths share one signed-window check before selecting
/// or batching candidates. The adaptive helper also checks its native unsigned
/// window contract before allocating a tree.
///
/// # Returns
///
/// `None` when no pair survives. Otherwise returns parallel physical left and
/// right labels in the order required by the selected `keep` policy.
///
/// # Errors
///
/// Returns an error for invalid signed window bounds, bounds beyond the right
/// layout, or an internal result/tree allocation overflow.
fn materialize_equi_windows_and_residuals(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    residual_predicates: &[PredicateView<'_>],
    null_metadata: Option<&[NullMetadataView<'_>]>,
    additional_range: Option<&AnyParsedRangePredicate<'_>>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    let additional_view = additional_range.map(range_predicate_view);
    let mut mask = Vec::new();

    if keep == Keep::All {
        // Keep::All cannot use an extrema query because every candidate must
        // be emitted. The first pass sizes the output exactly, avoiding
        // repeated reallocations during the second pass.
        let mut output_capacity = 0_usize;
        for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
            if start < 0 || end < 0 {
                continue;
            }
            let start = start as usize;
            let end = end as usize;
            mask.resize(end - start, 0);
            predicates_match_window(
                residual_predicates,
                null_metadata,
                row,
                start,
                end,
                &mut mask,
            );
            for (offset, &matches) in mask.iter().enumerate() {
                let right_position = start + offset;
                if matches != 0
                    && additional_view
                        .as_ref()
                        .map(|view| view.matches(row, right_position))
                        .unwrap_or(true)
                {
                    output_capacity = output_capacity
                        .checked_add(1)
                        .ok_or("equi-join result size exceeds platform capacity")?;
                }
            }
        }
        if output_capacity == 0 {
            return Ok(None);
        }
        let mut left_output = Vec::with_capacity(output_capacity);
        let mut right_output = Vec::with_capacity(output_capacity);
        for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
            if start < 0 || end < 0 {
                continue;
            }
            let start = start as usize;
            let end = end as usize;
            mask.resize(end - start, 0);
            predicates_match_window(
                residual_predicates,
                null_metadata,
                row,
                start,
                end,
                &mut mask,
            );
            for (offset, &matches) in mask.iter().enumerate() {
                let right_position = start + offset;
                if matches != 0
                    && additional_view
                        .as_ref()
                        .map(|view| view.matches(row, right_position))
                        .unwrap_or(true)
                {
                    left_output.push(left_index[row]);
                    right_output.push(right_index[right_position]);
                }
            }
        }
        return Ok(Some((left_output, right_output)));
    }

    let use_extrema = residual_predicates.is_empty()
        && additional_range.is_none()
        && matches!(keep, Keep::First | Keep::Last);
    // Reserve once from the known row bound only for the adaptive batch.
    // Residual/any paths scan immediately and do not allocate these buffers.
    let batch_capacity = if use_extrema { starts.len() } else { 0 };
    let mut rows = Vec::with_capacity(batch_capacity);
    let mut ranges = Vec::with_capacity(batch_capacity);
    let mut left_output = Vec::with_capacity(left_index.len());
    let mut right_output = Vec::with_capacity(left_index.len());
    for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
        if start < 0 || end < 0 {
            // ELI5: a negative shelf number means "no shelf for this row".
            // Skip it before converting to usize, where -1 would look huge.
            continue;
        }
        let (start, end) = validate_start_end(start, end, right_index.len())?;
        if start == end {
            continue;
        }
        if use_extrema {
            // ELI5: keep each surviving row beside its shelf coordinates.
            // Both selection paths share the sentinel/validation loop above;
            // only a batch with no extra rules can reuse extrema summaries.
            rows.push(row);
            ranges.push((start, end));
            continue;
        }

        if keep == Keep::Any {
            let mut selected = None;
            for right_position in start..end {
                if !predicates_match_dispatch(
                    residual_predicates,
                    null_metadata,
                    row,
                    right_position,
                ) {
                    continue;
                }
                if !additional_view
                    .as_ref()
                    .map(|view| view.matches(row, right_position))
                    .unwrap_or(true)
                {
                    continue;
                }
                selected = Some(right_position);
                break;
            }
            if let Some(right_position) = selected {
                left_output.push(left_index[row]);
                right_output.push(right_index[right_position]);
            }
            continue;
        }

        mask.resize(end - start, 0);
        predicates_match_window(
            residual_predicates,
            null_metadata,
            row,
            start,
            end,
            &mut mask,
        );
        let mut selected = None;
        for (offset, &matches) in mask.iter().enumerate() {
            if matches == 0 {
                continue;
            }
            let right_position = start + offset;
            if !additional_view
                .as_ref()
                .map(|view| view.matches(row, right_position))
                .unwrap_or(true)
            {
                continue;
            }
            if !update_keep_selection(&mut selected, right_position, keep, right_index) {
                break;
            }
        }
        if let Some(right_position) = selected {
            left_output.push(left_index[row]);
            right_output.push(right_index[right_position]);
        }
    }
    if use_extrema {
        let selected = adaptive_extreme_positions(right_index, &ranges, keep == Keep::First)?;
        for (row, right_position) in rows.into_iter().zip(selected) {
            left_output.push(left_index[row]);
            right_output.push(right_index[right_position]);
        }
    }
    if left_output.is_empty() {
        Ok(None)
    } else {
        Ok(Some((left_output, right_output)))
    }
}

#[allow(clippy::too_many_arguments)]
fn build_equi_range_and_residual_indices_core(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    mut starts: ArrayViewMut1<'_, i64>,
    mut ends: ArrayViewMut1<'_, i64>,
    first_range: &AnyParsedRangePredicate<'_>,
    second_range: Option<&AnyParsedRangePredicate<'_>>,
    residual_predicates: &[PredicateView<'_>],
    null_metadata: Option<&[NullMetadataView<'_>]>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    ensure_equal_lengths_core("left index", left_index.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("left index", left_index.len(), "ends", ends.len())?;
    for (&start, &end) in starts.iter().zip(ends.iter()) {
        if start < 0 || end < 0 {
            continue;
        }
        validate_start_end(start, end, right_index.len())?;
    }
    let equality_windows = second_range.map(|_| (starts.to_owned(), ends.to_owned()));

    let has_active_window = apply_equi_range_windows(
        first_range,
        starts.view_mut(),
        ends.view_mut(),
        right_index.len(),
    )?;
    if !has_active_window {
        return Ok(None);
    }
    if let Some(second_range) = second_range {
        validate_equi_range_inputs(second_range, starts.len(), right_index.len())?;
        if range_is_monotonic_for_windows(second_range, starts.view(), ends.view()) {
            let has_active_window = apply_equi_range_windows(
                second_range,
                starts.view_mut(),
                ends.view_mut(),
                right_index.len(),
            )?;
            if !has_active_window {
                return Ok(None);
            }
        } else {
            let (equality_starts, equality_ends) = equality_windows
                .as_ref()
                .expect("second range must retain the equality windows");
            return materialize_unordered_second_range(
                left_index,
                right_index,
                starts.view_mut(),
                ends.view_mut(),
                second_range,
                equality_starts.view(),
                equality_ends.view(),
                residual_predicates,
                null_metadata,
                keep,
            );
        }
    }

    materialize_equi_windows_and_residuals(
        left_index,
        right_index,
        starts.view(),
        ends.view(),
        residual_predicates,
        null_metadata,
        None,
        keep,
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn equi_range_and_residual_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    mut starts: PyReadwriteArray1<'py, i64>,
    mut ends: PyReadwriteArray1<'py, i64>,
    range_predicates: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if range_predicates.is_empty() {
        return Err(PyValueError::new_err("range_predicates must not be empty"));
    }
    validate_equi_range_predicate_count(range_predicates.len())?;
    let keep = Keep::parse(keep)?;
    let left_index_array = left_index;
    let right_index_array = right_index;
    let left_index = left_index_array.as_array();
    let right_index = right_index_array.as_array();
    let starts = starts.as_array_mut();
    let ends = ends.as_array_mut();
    let ranges = parse_equi_range_predicates(
        range_predicates,
        left_index_array.as_any(),
        right_index_array.as_any(),
    )?;
    let first_range = ranges
        .first()
        .ok_or_else(|| PyValueError::new_err("missing first range predicate"))?;
    let second_range = if ranges.len() == 2 {
        Some(
            ranges
                .get(1)
                .ok_or_else(|| PyValueError::new_err("missing second range predicate"))?,
        )
    } else {
        None
    };
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&parsed, left_index.len(), right_index.len())?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let result = build_equi_range_and_residual_indices_core(
        left_index,
        right_index,
        starts,
        ends,
        first_range,
        second_range,
        &views,
        metadata_views.as_deref(),
        keep,
    )
    .map_err(PyValueError::new_err)?;
    result
        .map(|(left, right)| result_dict(py, left, right, None, None))
        .transpose()
}

/// Aggregate candidates from the prepared equi layout.
///
/// Unique equality matches use aligned left/right rows. Duplicate equality
/// matches use `starts..ends` windows. A non-monotonic second range is first
/// narrowed with the cumulative envelope, then remains an exact per-candidate
/// residual check for correctness.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn equi_aggregate<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    starts: Option<PyReadwriteArray1<'py, i64>>,
    ends: Option<PyReadwriteArray1<'py, i64>>,
    range_predicates: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    equi_aggregate_impl(
        py,
        left_index,
        right_index,
        starts,
        ends,
        range_predicates,
        residual_predicates,
        aggregations,
        return_matched,
        reverse,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn equi_aggregate_impl<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    starts: Option<PyReadwriteArray1<'py, i64>>,
    ends: Option<PyReadwriteArray1<'py, i64>>,
    range_predicates: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
    allow_starts_ends: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    let left_index_array = left_index;
    let right_index_array = right_index;
    let left_index = left_index_array.as_array();
    let right_index = right_index_array.as_array();
    let inputs = parse_inputs(aggregations)?;
    if inputs.is_empty() {
        return Err(PyValueError::new_err(
            "at least one aggregation is required",
        ));
    }
    validate_equi_range_predicate_count(range_predicates.len())?;
    if starts.is_some() != ends.is_some() {
        return Err(PyValueError::new_err(
            "starts and ends must be provided together",
        ));
    }

    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&parsed, left_index.len(), right_index.len())?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let ranges = parse_equi_range_predicates(
        range_predicates,
        left_index_array.as_any(),
        right_index_array.as_any(),
    )?;

    let (reverse_output_positions, reverse_output_slots) = if reverse && starts.is_none() {
        let (positions, slots) = unique_reverse_output_slots(right_index);
        (Some(positions), Some(slots))
    } else {
        (None, None)
    };
    let output_len = if let Some(positions) = &reverse_output_positions {
        positions.len()
    } else if reverse {
        right_index.len()
    } else {
        left_index.len()
    };
    let source_len = if reverse {
        left_index.len()
    } else {
        right_index.len()
    };
    let mut set = AggregationSet::new(output_len, source_len, &inputs, return_matched)?;

    let mut starts_owner = starts;
    let mut ends_owner = ends;
    let mut starts = starts_owner.as_mut().map(|values| values.as_array_mut());
    let mut ends = ends_owner.as_mut().map(|values| values.as_array_mut());
    let retain_equality_windows = starts.is_some() && ranges.len() > 1;
    let equality_windows = retain_equality_windows.then(|| {
        (
            starts
                .as_ref()
                .expect("starts and ends are paired")
                .view()
                .to_owned(),
            ends.as_ref()
                .expect("starts and ends are paired")
                .view()
                .to_owned(),
        )
    });
    let mut demoted_second = None;
    if let (Some(starts), Some(ends)) = (starts.as_mut(), ends.as_mut()) {
        ensure_equal_lengths_core("left index", left_index.len(), "starts", starts.len())
            .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core("left index", left_index.len(), "ends", ends.len())
            .map_err(PyValueError::new_err)?;
        for (&start, &end) in starts.iter().zip(ends.iter()) {
            if start < 0 || end < 0 {
                continue;
            }
            validate_start_end(start, end, right_index.len()).map_err(PyValueError::new_err)?;
        }
        if let Some(first_range) = ranges.first() {
            apply_equi_range_windows(
                first_range,
                starts.view_mut(),
                ends.view_mut(),
                right_index.len(),
            )
            .map_err(PyValueError::new_err)?;
        }
        if let Some(second_range) = ranges.get(1) {
            validate_equi_range_inputs(second_range, starts.len(), right_index.len())
                .map_err(PyValueError::new_err)?;
            if range_is_monotonic_for_windows(second_range, starts.view(), ends.view()) {
                apply_equi_range_windows(
                    second_range,
                    starts.view_mut(),
                    ends.view_mut(),
                    right_index.len(),
                )
                .map_err(PyValueError::new_err)?;
            } else {
                let (equality_starts, equality_ends) = equality_windows
                    .as_ref()
                    .expect("second range must retain the equality windows");
                if !narrow_unordered_second_range_for_predicate(
                    second_range,
                    equality_starts.view(),
                    equality_ends.view(),
                    starts.view_mut(),
                    ends.view_mut(),
                )
                .map_err(PyValueError::new_err)?
                {
                    return Ok(None);
                }
                demoted_second = Some(second_range);
            }
        }
    } else {
        for range in &ranges {
            validate_equi_range_inputs(range, left_index.len(), right_index.len())
                .map_err(PyValueError::new_err)?;
        }
    }

    // ELI5: precomputed windows already applied the ranges, so only the
    // aligned no-window layout needs to re-check every range per candidate.
    let range_views: Vec<_> = if starts.is_none() {
        ranges.iter().map(range_predicate_view).collect()
    } else {
        Vec::new()
    };
    let demoted_view = demoted_second.map(range_predicate_view);

    // ELI5: equality windows give every left row one contiguous shelf of
    // right candidates. When residuals are present, cross off rejected
    // candidates once for the whole shelf instead of asking every predicate
    // about every candidate pair.
    let use_window_residuals = starts.is_some() && !views.is_empty();
    let mut residual_mask = Vec::new();

    // ELI5: after ordered ranges narrow a window, each left row's matches
    // are one contiguous block of right positions, so starts/ends describe
    // the complete match set. Reverse joins aggregate onto right rows;
    // surviving residual candidates still need pair-by-pair updates; and an
    // unordered second range filters pairs inside the window, so none of
    // those cases is safe to answer from the window alone.
    if allow_starts_ends && !reverse && residual_predicates.is_empty() && demoted_view.is_none() {
        if let (Some(starts), Some(ends)) = (starts.as_ref(), ends.as_ref()) {
            #[cfg(test)]
            EQUI_STARTS_ENDS_HITS.with(|hits| hits.set(hits.get() + 1));
            set.aggregate_starts_ends(starts.view(), ends.view());
            if set.is_empty() {
                return Ok(None);
            }
            return Ok(Some(make_results_with_positions(
                py,
                set,
                Some(left_index),
                output_len,
                return_matched,
            )?));
        }
    }

    // ELI5: with no residual rules, the unordered second range is the only
    // test left inside each shelf, so the block skip scan can hand over exactly
    // the passing positions instead of every candidate being tried in turn.
    // With residual rules on short shelves, the general loop below keeps
    // its whole-shelf residual checklist instead.
    if let (Some(second_range), Some(starts), Some(ends)) =
        (demoted_second, starts.as_ref(), ends.as_ref())
    {
        let use_skip = second_range_skip_pays_off(right_index.len(), starts.view(), ends.view());
        if use_skip || views.is_empty() {
            aggregate_unordered_second_range(
                &mut set,
                second_range,
                starts.view(),
                ends.view(),
                &views,
                metadata_views.as_deref(),
                use_skip,
                reverse,
            );
            if set.is_empty() {
                return Ok(None);
            }
            let output_positions = if reverse {
                Some(right_index)
            } else {
                Some(left_index)
            };
            return Ok(Some(make_results_with_positions(
                py,
                set,
                output_positions,
                output_len,
                return_matched,
            )?));
        }
    }
    for row in 0..left_index.len() {
        let candidate_start = starts
            .as_ref()
            .and_then(|values| values.view().get(row).copied())
            .unwrap_or(row as i64);
        let candidate_end = ends
            .as_ref()
            .and_then(|values| values.view().get(row).copied())
            .unwrap_or(candidate_start + 1);
        if candidate_start < 0 || candidate_end < 0 {
            continue;
        }
        let (candidate_start, candidate_end) = if starts.is_some() {
            validate_start_end(candidate_start, candidate_end, right_index.len())
                .map_err(PyValueError::new_err)?
        } else if candidate_start as usize >= right_index.len() {
            continue;
        } else {
            (candidate_start as usize, candidate_end as usize)
        };

        if use_window_residuals {
            #[cfg(test)]
            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| hits.set(hits.get() + 1));
            residual_mask.resize(candidate_end - candidate_start, 0);
            predicates_match_window(
                &views,
                metadata_views.as_deref(),
                row,
                candidate_start,
                candidate_end,
                &mut residual_mask,
            );
        }

        for right_position in candidate_start..candidate_end {
            let residual_matches = if use_window_residuals {
                residual_mask[right_position - candidate_start] != 0
            } else {
                predicates_match_dispatch(&views, metadata_views.as_deref(), row, right_position)
            };
            if !residual_matches {
                continue;
            }
            if range_views
                .iter()
                .any(|view| !view.matches(row, right_position))
            {
                continue;
            }
            if let Some(view) = &demoted_view {
                if !view.matches(row, right_position) {
                    continue;
                }
            }
            if reverse {
                let output_position = if let Some(slots) = &reverse_output_slots {
                    // ELI5: several aligned rows can point to the same right
                    // record; put all of them in that record's one bucket.
                    slots
                        .get(&right_index[right_position])
                        .copied()
                        .expect("every reverse candidate must have an output slot")
                } else {
                    right_position
                };
                set.update(row, output_position);
            } else {
                set.update(right_position, row);
            }
        }
    }

    if set.is_empty() {
        return Ok(None);
    }
    let output_positions = if reverse {
        reverse_output_positions
            .as_ref()
            .map_or(Some(right_index), |positions| {
                Some(ArrayView1::from(positions.as_slice()))
            })
    } else {
        Some(left_index)
    };
    Ok(Some(make_results_with_positions(
        py,
        set,
        output_positions,
        output_len,
        return_matched,
    )?))
}

/// Aggregate every pair passing an unordered second range in its window.
///
/// `starts`/`ends` are the windows left by the first range and the envelope
/// narrowing; negative windows are inactive. Residual predicates, when
/// present, are checked for each position the scanner reports; `use_skip`
/// comes from [`second_range_skip_pays_off`]. The windows come from
/// precomputed equality groups, so a reverse aggregation writes straight to
/// the right position (the unique-label slot map is only for the window-free
/// layout). Forward updates use `(right source, left output)`; reverse
/// updates use `(left source, right output)`, matching the general loop in
/// `equi_aggregate_impl`.
#[allow(clippy::too_many_arguments)]
fn aggregate_unordered_second_range(
    set: &mut AggregationSet,
    range: &AnyParsedRangePredicate<'_>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    residual_predicates: &[PredicateView<'_>],
    null_metadata: Option<&[NullMetadataView<'_>]>,
    use_skip: bool,
    reverse: bool,
) {
    macro_rules! aggregate {
        ($predicate:expr) => {{
            let predicate = $predicate;
            let left = predicate.left.as_array();
            let right = predicate.right.as_array();
            let scanner = if use_skip {
                SecondRangeSkipScan::new(right, predicate.op)
            } else {
                SecondRangeSkipScan::direct(right, predicate.op)
            };
            for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                if start < 0 || end < 0 {
                    continue;
                }
                scanner.for_each_match(left[row], start as usize, end as usize, |right_position| {
                    if !residual_predicates.is_empty()
                        && !predicates_match_dispatch(
                            residual_predicates,
                            null_metadata,
                            row,
                            right_position,
                        )
                    {
                        return true;
                    }
                    if reverse {
                        set.update(row, right_position);
                    } else {
                        set.update(right_position, row);
                    }
                    true
                });
            }
        }};
    }

    match range {
        AnyParsedRangePredicate::I64(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::I32(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::I16(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::I8(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::U64(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::U32(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::U16(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::U8(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::F64(predicate) => aggregate!(predicate),
        AnyParsedRangePredicate::F32(predicate) => aggregate!(predicate),
    }
}

/// Benchmark-only entry point that can force the legacy candidate path.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn equi_aggregate_bench<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    starts: Option<PyReadwriteArray1<'py, i64>>,
    ends: Option<PyReadwriteArray1<'py, i64>>,
    range_predicates: &Bound<'py, PyList>,
    residual_predicates: &Bound<'py, PyList>,
    aggregations: &Bound<'py, PyList>,
    return_matched: bool,
    reverse: bool,
    allow_starts_ends: bool,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    equi_aggregate_impl(
        py,
        left_index,
        right_index,
        starts,
        ends,
        range_predicates,
        residual_predicates,
        aggregations,
        return_matched,
        reverse,
        allow_starts_ends,
    )
}

/// Materialize aligned unique-equality rows after residual filtering.
///
/// The caller has already matched equality keys and projected the right side
/// into the same row order as the left side. Consequently, candidate `i` is
/// always the pair `(left_index[i], right_index[i])`; there is no right-side
/// search and no equality window to expand. The residual predicates are
/// evaluated at `(i, i)` and may be empty when the caller only needs to
/// materialize the unique equality pairs.
///
/// ELI5: each left row has its one-and-only right partner written beside it.
/// We compare each pair in the same row, keep the rows that pass, and copy
/// their labels to the output.
///
/// # Arguments
///
/// * `left_index` - Physical or public labels for the prepared left rows.
/// * `right_index` - Physical or public labels for the prepared right rows,
///   aligned one-for-one with `left_index`.
/// * `predicates` - Residual predicates whose left and right value arrays are
///   both aligned to the two index arrays. An empty slice means that every
///   aligned equality pair is accepted.
/// * `null_metadata` - Optional null-semantics metadata aligned with
///   `predicates`.
///
/// # Errors
///
/// Returns an error when the two index arrays have different lengths.
///
/// # Returns
///
/// `Ok(None)` when every aligned pair is filtered out or the inputs are
/// empty; otherwise, `Ok(Some((left_labels, right_labels)))`.
fn build_equi_uniq_residual_indices_core(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    predicates: &[PredicateView<'_>],
    null_metadata: Option<&[NullMetadataView<'_>]>,
) -> Result<IndexPairs, String> {
    ensure_equal_lengths_core(
        "left index",
        left_index.len(),
        "right index",
        right_index.len(),
    )?;
    let mut positions = Vec::with_capacity(left_index.len());
    for row in 0..left_index.len() {
        if !predicates_match_dispatch(predicates, null_metadata, row, row) {
            continue;
        }
        positions.push(row);
    }
    if positions.is_empty() {
        return Ok(None);
    }
    let mut left_output = Vec::with_capacity(positions.len());
    let mut right_output = Vec::with_capacity(positions.len());
    for &left_position in positions.iter() {
        let right_position = left_position;
        left_output.push(left_index[left_position]);
        right_output.push(right_index[right_position]);
    }
    Ok(Some((left_output, right_output)))
}

/// Build index pairs for unique equality keys and aligned residual predicates.
///
/// This is the Python-facing wrapper for
/// [`build_equi_uniq_residual_indices_core`]. It is used after PyJanitor has
/// established that every surviving right equality key is unique and has
/// projected the matching right rows into left-row order.
///
/// # Arguments
///
/// * `left_index` - One-dimensional `int64` array of prepared left labels or
///   physical positions.
/// * `right_index` - One-dimensional `int64` array of prepared right labels or
///   physical positions. It must have the same length as `left_index`; entry
///   `i` is the unique equality match for entry `i` on the left.
/// * `residual_predicates` - Python list of predicate tuples. Each tuple is
///   parsed using the shared predicate parser, and its value arrays must be
///   aligned to `left_index` and `right_index`. The list may be empty.
///
/// # Returns
///
/// A dictionary containing `left_index` and `right_index`, or `None` when no
/// aligned pair survives the residual predicates.
#[pyfunction]
pub fn equi_uniq_residual_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    residual_predicates: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let left_index = left_index.as_array();
    let right_index = right_index.as_array();
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&parsed, left_index.len(), right_index.len())?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);
    let result = build_equi_uniq_residual_indices_core(
        left_index,
        right_index,
        &views,
        metadata_views.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    result
        .map(|(left, right)| result_dict(py, left, right, None, None))
        .transpose()
}

fn build_equi_ne_indices_core(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    predicates: &[PredicateView<'_>],
    null_metadata: Option<&[NullMetadataView<'_>]>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    ensure_equal_lengths_core("left index", left_index.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("left index", left_index.len(), "ends", ends.len())?;

    let right_index_len = right_index.len();
    if keep == Keep::All {
        let mut total = 0_usize;
        for (left_position, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
            let (start, end) = validate_start_end(*start, *end, right_index_len)?;
            for right_position in start..end {
                if !predicates_match_dispatch(
                    predicates,
                    null_metadata,
                    left_position,
                    right_position,
                ) {
                    continue;
                }
                total = total
                    .checked_add(1)
                    .ok_or("equi-join result size exceeds platform capacity")?;
            }
        }
        if total == 0 {
            return Ok(None);
        }
        let mut left_output = Vec::with_capacity(total);
        let mut right_output = Vec::with_capacity(total);
        for (left_position, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
            let (start, end) = validate_start_end(*start, *end, right_index_len)?;
            for right_position in start..end {
                if !predicates_match_dispatch(
                    predicates,
                    null_metadata,
                    left_position,
                    right_position,
                ) {
                    continue;
                }
                left_output.push(left_index[left_position]);
                right_output.push(right_index[right_position]);
            }
        }
        return Ok(Some((left_output, right_output)));
    }

    let mut left_output = Vec::with_capacity(left_index.len());
    let mut right_output = Vec::with_capacity(left_index.len());
    for (left_position, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let (start, end) = validate_start_end(*start, *end, right_index_len)?;
        let mut selected = None;
        for right_position in start..end {
            if !predicates_match_dispatch(predicates, null_metadata, left_position, right_position)
            {
                continue;
            }
            if !update_keep_selection(&mut selected, right_position, keep, right_index) {
                break;
            }
        }
        if let Some(right_position) = selected {
            left_output.push(left_index[left_position]);
            right_output.push(right_index[right_position]);
        }
    }
    if left_output.is_empty() {
        Ok(None)
    } else {
        Ok(Some((left_output, right_output)))
    }
}

#[pyfunction]
pub fn equi_ne_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    residual_predicates: &Bound<'py, PyList>,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if residual_predicates.is_empty() {
        return Err(PyValueError::new_err(
            "residual_predicates must not be empty",
        ));
    }
    let left_index = left_index.as_array();
    let right_index = right_index.as_array();
    let keep = Keep::parse(keep)?;
    let starts = starts.as_array();
    let ends = ends.as_array();
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, residual_predicates)?;
    check_predicate_lengths(&parsed, left_index.len(), right_index.len())?;
    let views: Vec<_> = parsed.iter().map(Predicate::view).collect();
    let metadata_views = metadata.as_deref().map(null_metadata_views);

    let result = build_equi_ne_indices_core(
        left_index,
        right_index,
        starts,
        ends,
        &views,
        metadata_views.as_deref(),
        keep,
    )
    .map_err(PyValueError::new_err)?;
    result
        .map(|(left, right)| result_dict(py, left, right, None, None))
        .transpose()
}

fn build_equi_only_indices_core(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    right_index_is_ordered: bool,
    keep: Keep,
) -> Result<IndexPairs, String> {
    ensure_equal_lengths_core("left index", left_index.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("left index", left_index.len(), "ends", ends.len())?;

    let mut start_positions = Vec::with_capacity(starts.len());
    let mut end_positions = Vec::with_capacity(ends.len());
    let right_index_len = right_index.len();
    for (&start, &end) in starts.iter().zip(ends) {
        let (start, end) = validate_start_end(start, end, right_index_len)?;
        start_positions.push(start);
        end_positions.push(end);
    }

    if starts.is_empty() {
        return Ok(None);
    }

    if keep == Keep::All {
        let mut output_capacity = 0_usize;
        for (&start, &end) in start_positions.iter().zip(&end_positions) {
            output_capacity = output_capacity
                .checked_add(end - start)
                .ok_or("equi-join result size exceeds platform capacity")?;
        }
        if output_capacity == 0 {
            return Ok(None);
        }
        let mut left_output = Vec::with_capacity(output_capacity);
        let mut right_output = Vec::with_capacity(output_capacity);
        for (row, (&start, &end)) in start_positions.iter().zip(&end_positions).enumerate() {
            for &right_label in right_index.slice(s![start..end]).iter() {
                left_output.push(left_index[row]);
                right_output.push(right_label);
            }
        }
        return Ok(Some((left_output, right_output)));
    }

    let mut left_output = Vec::with_capacity(left_index.len());
    let mut right_output = Vec::with_capacity(left_index.len());
    for (row, (&start, &end)) in start_positions.iter().zip(&end_positions).enumerate() {
        let right_label = match keep {
            Keep::Any => right_index[start],
            Keep::First if right_index_is_ordered => right_index[start],
            Keep::Last if right_index_is_ordered => right_index[end - 1],
            Keep::First => *right_index
                .slice(s![start..end])
                .iter()
                .min()
                .ok_or("equi-join window is empty")?,
            Keep::Last => *right_index
                .slice(s![start..end])
                .iter()
                .max()
                .ok_or("equi-join window is empty")?,
            Keep::All => unreachable!(),
        };
        left_output.push(left_index[row]);
        right_output.push(right_label);
    }
    Ok(Some((left_output, right_output)))
}

/// Materialize equi-join index pairs from prepared equality windows.
///
/// `starts[row]..ends[row]` is the half-open right-side equality window for
/// `left_index[row]`. `first` and `last` select the smallest or largest right
/// label in each window when the right labels are unordered; when they are
/// monotonic increasing, the window boundaries provide those values directly.
/// `any` selects the first physical right entry, and `all` emits every entry.
///
/// # Arguments
///
/// * `left_index` - Prepared left labels or physical positions, one per
///   equality window.
/// * `right_index` - Right labels or physical positions arranged in the same
///   layout referenced by `starts` and `ends`.
/// * `starts` - Inclusive right-window starts, one per left row.
/// * `ends` - Exclusive right-window ends, one per left row.
/// * `right_index_is_ordered` - Whether `right_index` is monotonic increasing;
///   this allows constant-time `first`/`last` selection.
/// * `keep` - One of `all`, `any`, `first`, or `last`.
///
/// # Returns
///
/// A dictionary containing the materialized index pairs, or `None` when the
/// windows contain no rows.
#[pyfunction]
pub fn equi_only_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    ends: PyReadonlyArray1<'py, i64>,
    right_index_is_ordered: bool,
    keep: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let keep = Keep::parse(keep)?;
    let left_index = left_index.as_array();
    let right_index = right_index.as_array();
    let starts = starts.as_array();
    let ends = ends.as_array();
    let result = build_equi_only_indices_core(
        left_index,
        right_index,
        starts,
        ends,
        right_index_is_ordered,
        keep,
    )
    .map_err(PyValueError::new_err)?;
    result
        .map(|(left, right)| result_dict(py, left, right, None, None))
        .transpose()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(equi_only_indices, m)?)?;
    m.add_function(wrap_pyfunction!(equi_uniq_residual_indices, m)?)?;
    m.add_function(wrap_pyfunction!(equi_ne_indices, m)?)?;
    m.add_function(wrap_pyfunction!(equi_range_indices, m)?)?;
    m.add_function(wrap_pyfunction!(equi_range_and_residual_indices, m)?)?;
    m.add_function(wrap_pyfunction!(equi_aggregate, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use numpy::ndarray::Array1;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::{IntoPyObject, PyResult, Python};

    use super::{
        apply_equi_range_bound, build_equi_ne_indices_core, build_equi_only_indices_core,
        build_equi_range_and_residual_indices_core, build_equi_range_indices_core,
        build_equi_uniq_residual_indices_core, build_second_range_envelope, equi_aggregate,
        materialize_equi_windows, materialize_equi_windows_and_residuals,
        materialize_unordered_second_range, narrow_unordered_second_range,
        unique_reverse_output_slots, SecondRangeSkipScan, EQUI_RESIDUAL_WINDOW_HITS,
        EQUI_STARTS_ENDS_HITS,
    };
    use crate::conditional_join::compare_op::CompareOp;
    use crate::conditional_join::join_types::Keep;
    use crate::conditional_join::range_skip_scan::{
        SECOND_RANGE_SKIP_HITS, SECOND_RANGE_SUPER_SKIPS,
    };
    use pyo3::exceptions::PyValueError;
    use pyo3::types::{PyAnyMethods, PyBool, PyList, PyListMethods, PyTuple, PyTupleMethods};

    #[test]
    fn reverse_unique_output_slots_collapse_aligned_duplicates() {
        let right_index = Array1::from_vec(vec![7_i64, 7, 3, 7]);
        let (labels, slots) = unique_reverse_output_slots(right_index.view());

        assert_eq!(labels, vec![7, 3]);
        assert_eq!(slots.get(&7), Some(&0));
        assert_eq!(slots.get(&3), Some(&1));
    }

    #[test]
    fn equi_aggregate_skips_eliminated_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            let starts = PyArray1::from_vec(py, vec![0_i64, -1]);
            let ends = PyArray1::from_vec(py, vec![2_i64, -1]);
            let range = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64, 100]).into_any(),
                    PyArray1::from_vec(py, vec![3_i64, 4]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?;
            let ranges = PyList::new(py, [range])?;
            let residuals = PyList::empty(py);
            let aggregation = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![10_i64, 13]).into_any(),
                    PyArray1::from_vec(py, vec![false, false]).into_any(),
                    "sum".into_pyobject(py)?.into_any(),
                ],
            )?;
            let aggregations = PyList::new(py, [aggregation])?;

            let result = equi_aggregate(
                py,
                left_index.readonly(),
                right_index.readonly(),
                Some(starts.readwrite()),
                Some(ends.readwrite()),
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;

            assert_eq!(result.get_item(0)?.extract::<Vec<i64>>()?, vec![100, 101]);
            assert_eq!(
                result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![true, false]
            );
            let sums = result
                .get_item(2)?
                .cast::<PyList>()?
                .get_item(0)?
                .extract::<Vec<i64>>()?;
            assert_eq!(sums, vec![23, 0]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn equi_aggregate_evaluates_residuals_per_window() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, vec![100_i64]);
            let right_index = PyArray1::from_vec(py, (10_i64..20).collect());
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![10_i64]);
            let ranges = PyList::empty(py);
            let residual = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![false]).into_any(),
                    PyArray1::from_vec(py, (1_i64..11).collect()).into_any(),
                    PyArray1::from_vec(
                        py,
                        vec![
                            false, false, false, false, true, false, false, false, false, false,
                        ],
                    )
                    .into_any(),
                    PyBool::new(py, false).to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?;
            let residual_second = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, (1_i64..11).collect()).into_any(),
                    ">=".into_pyobject(py)?.into_any(),
                ],
            )?;
            let residuals = PyList::new(py, [residual, residual_second])?;
            let aggregations = equi_sum_aggregations(py, (1_i64..11).collect())?;

            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| hits.set(0));
            let result = equi_aggregate(
                py,
                left_index.readonly(),
                right_index.readonly(),
                Some(starts.readwrite()),
                Some(ends.readwrite()),
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;

            let (matched, sums) = equi_sum_result(&result)?;
            assert_eq!(matched, vec![true]);
            assert_eq!(sums, vec![54]);
            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| assert_eq!(hits.get(), 1));

            // Windows shorter than eight still use the helper's scalar
            // fallback, while retaining the same caller-level result.
            let small_right_index = PyArray1::from_vec(py, (10_i64..14).collect());
            let small_starts = PyArray1::from_vec(py, vec![0_i64]);
            let small_ends = PyArray1::from_vec(py, vec![4_i64]);
            let small_residual = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![15_i64]).into_any(),
                    PyArray1::from_vec(py, (10_i64..14).collect()).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?;
            let small_residuals = PyList::new(py, [small_residual])?;
            let small_aggregations = equi_sum_aggregations(py, (1_i64..5).collect())?;
            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| hits.set(0));
            let small_result = equi_aggregate(
                py,
                left_index.readonly(),
                small_right_index.readonly(),
                Some(small_starts.readwrite()),
                Some(small_ends.readwrite()),
                &ranges,
                &small_residuals,
                &small_aggregations,
                true,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("small-window aggregation returned None"))?;
            let (small_matched, small_sums) = equi_sum_result(&small_result)?;
            assert_eq!(small_matched, vec![true]);
            assert_eq!(small_sums, vec![10]);
            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| assert_eq!(hits.get(), 1));

            // The reverse layout uses the same window mask but writes into
            // right-side output slots. Extension-array nulls must reject the
            // null right value rather than treating it as a passing `!=`.
            let reverse_left_index = PyArray1::from_vec(py, vec![100_i64]);
            let reverse_right_index = PyArray1::from_vec(py, (10_i64..20).collect());
            let reverse_starts = PyArray1::from_vec(py, vec![0_i64]);
            let reverse_ends = PyArray1::from_vec(py, vec![10_i64]);
            let reverse_residual = PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![false]).into_any(),
                    PyArray1::from_vec(py, (1_i64..11).collect()).into_any(),
                    PyArray1::from_vec(
                        py,
                        vec![
                            false, false, false, false, true, false, false, false, false, false,
                        ],
                    )
                    .into_any(),
                    PyBool::new(py, true).to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?;
            let reverse_residuals = PyList::new(py, [reverse_residual])?;
            let reverse_aggregations = equi_sum_aggregations(py, vec![7])?;

            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| hits.set(0));
            let reverse_result = equi_aggregate(
                py,
                reverse_left_index.readonly(),
                reverse_right_index.readonly(),
                Some(reverse_starts.readwrite()),
                Some(reverse_ends.readwrite()),
                &ranges,
                &reverse_residuals,
                &reverse_aggregations,
                true,
                true,
            )?
            .ok_or_else(|| PyValueError::new_err("reverse equi aggregation returned None"))?;

            assert_eq!(
                reverse_result.get_item(0)?.extract::<Vec<i64>>()?,
                (10_i64..20).collect::<Vec<_>>()
            );
            assert_eq!(
                reverse_result.get_item(1)?.extract::<Vec<bool>>()?,
                vec![false, true, true, true, false, true, true, true, true, true]
            );
            let reverse_sums = reverse_result
                .get_item(2)?
                .cast::<PyList>()?
                .get_item(0)?
                .extract::<Vec<i64>>()?;
            assert_eq!(reverse_sums, vec![0, 7, 7, 7, 0, 7, 7, 7, 7, 7]);
            EQUI_RESIDUAL_WINDOW_HITS.with(|hits| assert_eq!(hits.get(), 1));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn equi_aggregate_starts_ends_handles_duplicate_groups_and_nulls() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let left_index = PyArray1::from_vec(py, (100_i64..124).collect());
            let right_index = PyArray1::from_vec(py, (10_i64..2058).collect());
            // Each equality group has repeated left rows and a broad window.
            // Two range bounds make each query an unaligned interior slice,
            // so the tree must merge both sides of its iterative query.
            let starts = PyArray1::from_vec(
                py,
                vec![0_i64; 12].into_iter().chain(vec![1024; 12]).collect(),
            );
            let ends = PyArray1::from_vec(
                py,
                vec![1024_i64; 12]
                    .into_iter()
                    .chain(vec![2048; 12])
                    .collect(),
            );
            let ranges = PyList::new(
                py,
                [
                    equi_range_tuple(
                        py,
                        vec![640_i64; 12]
                            .into_iter()
                            .chain(vec![1664; 12])
                            .collect(),
                        (0_i64..2048).collect(),
                        ">",
                    )?,
                    equi_range_tuple(
                        py,
                        vec![128_i64; 12]
                            .into_iter()
                            .chain(vec![1152; 12])
                            .collect(),
                        (0_i64..2048).collect(),
                        "<",
                    )?,
                ],
            )?;
            let residuals = PyList::empty(py);
            let values = PyArray1::from_vec(py, (0_i64..2048).collect());
            let product_values = PyArray1::from_vec(
                py,
                (0..2048)
                    .map(|position| {
                        if position == 200 || position == 1200 {
                            2
                        } else {
                            1
                        }
                    })
                    .collect(),
            );
            let float_values = PyArray1::from_vec(
                py,
                (0..2048)
                    .map(|position| match position % 3 {
                        0 => 1.0e16,
                        1 => 1.0,
                        _ => -1.0e16,
                    })
                    .collect(),
            );
            let float_product_values = PyArray1::from_vec(
                py,
                (0..2048)
                    .map(|position| {
                        if position == 200 || position == 1200 {
                            2.0
                        } else {
                            1.0
                        }
                    })
                    .collect(),
            );
            let nulls = PyArray1::from_vec(
                py,
                (0..2048)
                    .map(|position| position == 129 || position == 1153)
                    .collect::<Vec<_>>(),
            );
            let aggregations = PyList::new(
                py,
                [
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
                            "*".into_pyobject(py)?.into_any(),
                            nulls.clone().into_any(),
                            "count".into_pyobject(py)?.into_any(),
                        ],
                    )?,
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
                            product_values.clone().into_any(),
                            nulls.clone().into_any(),
                            "product".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            float_values.clone().into_any(),
                            nulls.clone().into_any(),
                            "sum".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                    PyTuple::new(
                        py,
                        [
                            float_product_values.into_any(),
                            nulls.clone().into_any(),
                            "product".into_pyobject(py)?.into_any(),
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
                            values.into_any(),
                            nulls.into_any(),
                            "max".into_pyobject(py)?.into_any(),
                        ],
                    )?,
                ],
            )?;

            let prefix_before = crate::aggregation::state::PREFIX_SUM_HITS.get();
            EQUI_STARTS_ENDS_HITS.with(|hits| hits.set(0));
            let result = equi_aggregate(
                py,
                left_index.readonly(),
                right_index.readonly(),
                Some(starts.readwrite()),
                Some(ends.readwrite()),
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;
            EQUI_STARTS_ENDS_HITS.with(|hits| assert!(hits.get() > 0));
            assert_eq!(
                crate::aggregation::state::PREFIX_SUM_HITS.get(),
                prefix_before + 1
            );

            assert_eq!(result.get_item(1)?.extract::<Vec<bool>>()?, vec![true; 24]);
            let outputs_item = result.get_item(2)?;
            let outputs = outputs_item.cast::<PyList>()?;
            assert_eq!(
                outputs.get_item(0)?.extract::<Vec<i64>>()?,
                vec![511; 12]
                    .into_iter()
                    .chain(vec![511; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                outputs.get_item(1)?.extract::<Vec<i64>>()?,
                vec![510; 12]
                    .into_iter()
                    .chain(vec![510; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                outputs.get_item(2)?.extract::<Vec<i64>>()?,
                vec![196095; 12]
                    .into_iter()
                    .chain(vec![718335; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                outputs.get_item(6)?.extract::<Vec<i64>>()?,
                vec![130; 12]
                    .into_iter()
                    .chain(vec![1154; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                outputs.get_item(7)?.extract::<Vec<i64>>()?,
                vec![639; 12]
                    .into_iter()
                    .chain(vec![1663; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                outputs.get_item(3)?.extract::<Vec<i64>>()?,
                vec![2; 12]
                    .into_iter()
                    .chain(vec![2; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                outputs.get_item(4)?.extract::<Vec<f64>>()?,
                vec![0.0; 12]
                    .into_iter()
                    .chain(vec![1.0; 12])
                    .collect::<Vec<_>>()
            );
            assert_eq!(outputs.get_item(5)?.extract::<Vec<f64>>()?, vec![2.0; 24]);
            Ok(())
        })
        .unwrap();
    }

    fn equi_range_tuple<'py>(
        py: Python<'py>,
        left: Vec<i64>,
        right: Vec<i64>,
        operator: &str,
    ) -> PyResult<pyo3::Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, left).into_any(),
                PyArray1::from_vec(py, right).into_any(),
                operator.into_pyobject(py)?.into_any(),
            ],
        )
    }

    fn equi_sum_aggregations<'py>(
        py: Python<'py>,
        values: Vec<i64>,
    ) -> PyResult<pyo3::Bound<'py, PyList>> {
        let mask = vec![false; values.len()];
        let aggregation = PyTuple::new(
            py,
            [
                PyArray1::from_vec(py, values).into_any(),
                PyArray1::from_vec(py, mask).into_any(),
                "sum".into_pyobject(py)?.into_any(),
            ],
        )?;
        PyList::new(py, [aggregation])
    }

    fn equi_sum_result(result: &pyo3::Bound<'_, PyTuple>) -> PyResult<(Vec<bool>, Vec<i64>)> {
        let matched = result.get_item(1)?.extract::<Vec<bool>>()?;
        let sums = result
            .get_item(2)?
            .cast::<PyList>()?
            .get_item(0)?
            .extract::<Vec<i64>>()?;
        Ok((matched, sums))
    }

    #[test]
    fn equi_aggregate_without_windows_checks_every_range_per_row() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // ELI5: with no windows, row `i` may only pair with right
            // position `i`, and each range can independently veto that pair.
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101, 102]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11, 12]);
            let ranges = PyList::new(
                py,
                [
                    equi_range_tuple(py, vec![1, 5, 1], vec![2, 2, 2], "<")?,
                    equi_range_tuple(py, vec![9, 9, 0], vec![3, 3, 3], ">")?,
                ],
            )?;
            let residuals = PyList::empty(py);
            let aggregations = equi_sum_aggregations(py, vec![10, 13, 17])?;

            let result = equi_aggregate(
                py,
                left_index.readonly(),
                right_index.readonly(),
                None,
                None,
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;

            let (matched, sums) = equi_sum_result(&result)?;
            assert_eq!(matched, vec![true, false, false]);
            assert_eq!(sums, vec![10, 0, 0]);
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn equi_aggregate_filters_demoted_second_range_per_candidate() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            // ELI5: the second range's right values `[5, 1, 7, 2]` are not
            // sorted inside the shared key window, so they cannot shrink the
            // window and must be checked candidate by candidate instead.
            let left_index = PyArray1::from_vec(py, vec![100_i64, 101]);
            let right_index = PyArray1::from_vec(py, vec![10_i64, 11, 12, 13]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let ends = PyArray1::from_vec(py, vec![4_i64, 4]);
            let ranges = PyList::new(
                py,
                [
                    equi_range_tuple(py, vec![0, 2], vec![1, 2, 3, 4], "<")?,
                    equi_range_tuple(py, vec![6, 1], vec![5, 1, 7, 2], ">")?,
                ],
            )?;
            let residuals = PyList::empty(py);
            let aggregations = equi_sum_aggregations(py, vec![10, 20, 30, 40])?;

            let result = equi_aggregate(
                py,
                left_index.readonly(),
                right_index.readonly(),
                Some(starts.readwrite()),
                Some(ends.readwrite()),
                &ranges,
                &residuals,
                &aggregations,
                true,
                false,
            )?
            .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;

            // Row 0: first range keeps every position; second keeps right
            // values 5, 1 and 2 (< 6), so 10 + 20 + 40.
            // Row 1: first range keeps positions 2 and 3; second needs a
            // right value below 1, which neither has.
            let (matched, sums) = equi_sum_result(&result)?;
            assert_eq!(matched, vec![true, false]);
            assert_eq!(sums, vec![70, 0]);
            Ok(())
        })
        .unwrap();
    }
    use crate::conditional_join::predicate::PredicateView;
    use crate::conditional_join::range_predicate::{AnyParsedRangePredicate, ParsedRangePredicate};

    #[test]
    fn unordered_second_range_envelope_is_directional() {
        let right = Array1::from_vec(vec![8_i64, 2, 6, 1, 7, 3]);
        let windows = [(0, 3), (3, 6)];
        assert_eq!(
            build_second_range_envelope(right.view(), &windows, true),
            vec![8, 8, 8, 1, 7, 7]
        );
        assert_eq!(
            build_second_range_envelope(right.view(), &windows, false),
            vec![2, 2, 6, 1, 3, 3]
        );
    }

    #[test]
    fn unordered_second_range_narrowing_matches_bruteforce() {
        let right = Array1::from_vec(vec![8_i64, 2, 6, 1, 7, 3, 9, 0]);
        let windows = [(0, 4), (4, 8)];
        let left = Array1::from_vec(vec![0_i64, 5, 10, 4, 8]);
        let original_starts = [0_i64, 0, 0, 4, 4];
        let original_ends = [4_i64, 4, 4, 8, 8];

        for op in [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge] {
            let use_cummax = matches!(op, CompareOp::Lt | CompareOp::Le);
            let envelope = build_second_range_envelope(right.view(), &windows, use_cummax);
            let mut starts = Array1::from_vec(original_starts.to_vec());
            let mut ends = Array1::from_vec(original_ends.to_vec());
            narrow_unordered_second_range(
                left.view(),
                &envelope,
                starts.view_mut(),
                ends.view_mut(),
                op,
            )
            .unwrap();

            let mut expected = Vec::new();
            let mut actual = Vec::new();
            for row in 0..left.len() {
                for position in original_starts[row] as usize..original_ends[row] as usize {
                    if op.apply(&left[row], &right[position]) {
                        expected.push((row, position));
                    }
                }
                if starts[row] >= 0 && ends[row] >= 0 {
                    for position in starts[row] as usize..ends[row] as usize {
                        if op.apply(&left[row], &right[position]) {
                            actual.push((row, position));
                        }
                    }
                }
            }
            assert_eq!(actual, expected, "operator: {op:?}");
        }
    }

    #[test]
    fn unordered_second_range_uses_original_equality_windows() {
        Python::initialize();
        Python::attach(|py| {
            let first_range = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                left: PyArray1::from_vec(py, vec![0_i64, 2]).readonly(),
                left_index: PyArray1::from_vec(py, vec![100_i64, 101]).readonly(),
                right: PyArray1::from_vec(py, vec![1_i64, 2, 3, 4]).readonly(),
                right_index: PyArray1::from_vec(py, vec![10_i64, 11, 12, 13]).readonly(),
                op: CompareOp::Lt,
            });
            let second_range = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                left: PyArray1::from_vec(py, vec![50_i64, 0]).readonly(),
                left_index: PyArray1::from_vec(py, vec![100_i64, 101]).readonly(),
                right: PyArray1::from_vec(py, vec![1_i64, 100, 2, 3]).readonly(),
                right_index: PyArray1::from_vec(py, vec![10_i64, 11, 12, 13]).readonly(),
                op: CompareOp::Lt,
            });
            let left_index = Array1::from_vec(vec![100_i64, 101]);
            let right_index = Array1::from_vec(vec![10_i64, 11, 12, 13]);
            let mut starts = Array1::from_vec(vec![0_i64, 0]);
            let mut ends = Array1::from_vec(vec![4_i64, 4]);

            let result = build_equi_range_indices_core(
                left_index.view(),
                right_index.view(),
                starts.view_mut(),
                ends.view_mut(),
                &first_range,
                Some(&second_range),
                Keep::All,
            )
            .unwrap()
            .unwrap();

            assert_eq!(result.0, vec![100, 101, 101]);
            assert_eq!(result.1, vec![11, 12, 13]);
        });
    }

    #[test]
    fn unordered_right_labels_select_extrema() {
        let left = Array1::from_vec(vec![10, 11]);
        let right = Array1::from_vec(vec![8, 2, 6, 1, 7, 3]);
        let starts = Array1::from_vec(vec![0, 3]);
        let ends = Array1::from_vec(vec![3, 6]);
        let result = build_equi_only_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            false,
            Keep::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.0, vec![10, 11]);
        assert_eq!(result.1, vec![2, 1]);

        let result = build_equi_only_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            false,
            Keep::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.1, vec![8, 7]);
    }

    #[test]
    fn all_uses_exact_window_width() {
        let left = Array1::from_vec(vec![10, 11]);
        let right = Array1::from_vec(vec![8, 2, 6, 1, 7, 3]);
        let starts = Array1::from_vec(vec![0, 3]);
        let ends = Array1::from_vec(vec![3, 6]);
        let result = build_equi_only_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            false,
            Keep::All,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.0, vec![10, 10, 10, 11, 11, 11]);
        assert_eq!(result.1, vec![8, 2, 6, 1, 7, 3]);
    }

    #[test]
    fn rejects_invalid_windows() {
        let left = Array1::from_vec(vec![10]);
        let right = Array1::from_vec(vec![1, 2]);
        let starts = Array1::from_vec(vec![1]);
        let ends = Array1::from_vec(vec![3]);
        let result = build_equi_only_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            true,
            Keep::Any,
        );
        assert_eq!(result, Err("end exceeds right index length".to_owned()));
    }

    #[test]
    fn ordered_windows_use_constant_time_boundaries() {
        let left = Array1::from_vec(vec![10, 11]);
        let right = Array1::from_vec(vec![2, 4, 8, 1, 5, 9]);
        let starts = Array1::from_vec(vec![0, 3]);
        let ends = Array1::from_vec(vec![3, 6]);

        let first = build_equi_only_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            true,
            Keep::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.1, vec![2, 1]);

        let last = build_equi_only_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            true,
            Keep::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(last.1, vec![8, 9]);
    }

    #[test]
    fn ne_keep_modes_apply_residual_predicates() {
        let left_index = Array1::from_vec(vec![10]);
        let right_index = Array1::from_vec(vec![1, 9, 2]);
        let starts = Array1::from_vec(vec![0]);
        let ends = Array1::from_vec(vec![3]);
        let left_values = Array1::from_vec(vec![5]);
        let right_values = Array1::from_vec(vec![1, 9, 2]);
        let predicate = PredicateView::I64(left_values.view(), right_values.view(), CompareOp::Gt);
        let predicates = [predicate];

        let all = build_equi_ne_indices_core(
            left_index.view(),
            right_index.view(),
            starts.view(),
            ends.view(),
            &predicates,
            None,
            Keep::All,
        )
        .unwrap()
        .unwrap();
        assert_eq!(all.0, vec![10, 10]);
        assert_eq!(all.1, vec![1, 2]);

        let any = build_equi_ne_indices_core(
            left_index.view(),
            right_index.view(),
            starts.view(),
            ends.view(),
            &predicates,
            None,
            Keep::Any,
        )
        .unwrap()
        .unwrap();
        assert_eq!(any.1, vec![1]);

        let first = build_equi_ne_indices_core(
            left_index.view(),
            right_index.view(),
            starts.view(),
            ends.view(),
            &predicates,
            None,
            Keep::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.1, vec![1]);

        let last = build_equi_ne_indices_core(
            left_index.view(),
            right_index.view(),
            starts.view(),
            ends.view(),
            &predicates,
            None,
            Keep::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(last.1, vec![2]);
    }

    #[test]
    fn ne_first_keeps_a_match_at_right_position_zero() {
        let left_index = Array1::from_vec(vec![10]);
        let right_index = Array1::from_vec(vec![1, 9]);
        let starts = Array1::from_vec(vec![0]);
        let ends = Array1::from_vec(vec![2]);
        let left_values = Array1::from_vec(vec![10]);
        let right_values = Array1::from_vec(vec![1, 9]);
        let predicate = PredicateView::I64(left_values.view(), right_values.view(), CompareOp::Gt);

        let result = build_equi_ne_indices_core(
            left_index.view(),
            right_index.view(),
            starts.view(),
            ends.view(),
            &[predicate],
            None,
            Keep::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.1, vec![1]);
    }

    #[test]
    fn ne_returns_none_when_residual_filters_every_candidate() {
        let left_index = Array1::from_vec(vec![10]);
        let right_index = Array1::from_vec(vec![1, 2]);
        let starts = Array1::from_vec(vec![0]);
        let ends = Array1::from_vec(vec![2]);
        let left_values = Array1::from_vec(vec![0]);
        let right_values = Array1::from_vec(vec![1, 2]);
        let predicate = PredicateView::I64(left_values.view(), right_values.view(), CompareOp::Gt);

        let result = build_equi_ne_indices_core(
            left_index.view(),
            right_index.view(),
            starts.view(),
            ends.view(),
            &[predicate],
            None,
            Keep::Any,
        )
        .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn unique_ne_without_residuals_pairs_aligned_rows() {
        let left = Array1::from_vec(vec![10, 11, 12]);
        let right = Array1::from_vec(vec![20, 21, 22]);
        let result = build_equi_uniq_residual_indices_core(left.view(), right.view(), &[], None)
            .unwrap()
            .unwrap();
        assert_eq!(result.0, vec![10, 11, 12]);
        assert_eq!(result.1, vec![20, 21, 22]);
    }

    #[test]
    fn unique_residual_filters_aligned_rows() {
        let left = Array1::from_vec(vec![10, 11, 12]);
        let right = Array1::from_vec(vec![20, 21, 22]);
        let left_values = Array1::from_vec(vec![1, 5, 3]);
        let right_values = Array1::from_vec(vec![2, 4, 3]);
        let predicate = PredicateView::I64(left_values.view(), right_values.view(), CompareOp::Gt);

        let result =
            build_equi_uniq_residual_indices_core(left.view(), right.view(), &[predicate], None)
                .unwrap()
                .unwrap();
        assert_eq!(result.0, vec![11]);
        assert_eq!(result.1, vec![21]);
    }

    #[test]
    fn unique_ne_rejects_misaligned_rows() {
        let left = Array1::from_vec(vec![10, 11]);
        let right = Array1::from_vec(vec![20]);
        let result = build_equi_uniq_residual_indices_core(left.view(), right.view(), &[], None);
        assert!(result.is_err());
    }

    #[test]
    fn range_windows_apply_keep_modes() {
        let left = Array1::from_vec(vec![10]);
        let right = Array1::from_vec(vec![8, 2, 6]);
        let starts = Array1::from_vec(vec![0]);
        let ends = Array1::from_vec(vec![3]);

        let all = materialize_equi_windows(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            Keep::All,
        )
        .unwrap()
        .unwrap();
        assert_eq!(all.1, vec![8, 2, 6]);

        let any = materialize_equi_windows(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            Keep::Any,
        )
        .unwrap()
        .unwrap();
        assert_eq!(any.1, vec![8]);

        let first = materialize_equi_windows(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            Keep::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.1, vec![2]);

        let last = materialize_equi_windows(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            Keep::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(last.1, vec![8]);
    }

    #[test]
    fn residual_windows_preserve_keep_any_short_circuit_and_selection() {
        let left_index = Array1::from_vec(vec![10]);
        let right_index = Array1::from_vec(vec![9, 8, 7, 6, 5, 4, 3, 2, 1]);
        let starts = Array1::from_vec(vec![0]);
        let ends = Array1::from_vec(vec![9]);
        let residual_left = Array1::from_vec(vec![3]);
        let residual_right = Array1::from_vec(vec![3, 0, 1, 2, 4, 5, 6, 7, 8]);
        let residuals = [PredicateView::I64(
            residual_left.view(),
            residual_right.view(),
            CompareOp::Le,
        )];

        for (keep, expected) in [(Keep::Any, 9), (Keep::First, 1), (Keep::Last, 9)] {
            let result = materialize_equi_windows_and_residuals(
                left_index.view(),
                right_index.view(),
                starts.view(),
                ends.view(),
                &residuals,
                None,
                None,
                keep,
            )
            .unwrap()
            .unwrap();
            assert_eq!(result.0, vec![10]);
            assert_eq!(result.1, vec![expected]);
        }
    }

    #[test]
    fn extrema_and_scans_preserve_inactive_window_alignment() {
        use crate::conditional_join::range_join::EXTREME_TREE_HITS;
        let left = Array1::from_iter(10..16);
        let right = Array1::from_vec(vec![4, 1, 3, 2]);
        let starts = Array1::from_vec(vec![-1, 0, 4, 1, 0, 3]);
        let ends = Array1::from_vec(vec![4, -1, 4, 3, 0, 4]);
        for (keep, expected) in [
            (Keep::First, vec![1, 2]),
            (Keep::Last, vec![3, 2]),
            (Keep::Any, vec![1, 2]),
        ] {
            let before = EXTREME_TREE_HITS.get();
            assert_eq!(
                materialize_equi_windows(
                    left.view(),
                    right.view(),
                    starts.view(),
                    ends.view(),
                    keep
                )
                .unwrap(),
                Some((vec![13, 15], expected))
            );
            assert_eq!(EXTREME_TREE_HITS.get(), before, "small workloads must scan");
        }
        for (start, end, message) in [
            (4, 5, "end exceeds right index length"),
            (5, 5, "end exceeds right index length"),
            (3, 2, "start must not exceed end"),
        ] {
            for keep in [Keep::First, Keep::Last, Keep::Any] {
                assert_eq!(
                    materialize_equi_windows(
                        left.slice(numpy::ndarray::s![..1]),
                        right.view(),
                        numpy::ndarray::ArrayView1::from(&[start]),
                        numpy::ndarray::ArrayView1::from(&[end]),
                        keep
                    ),
                    Err(message.to_owned())
                );
            }
        }
    }

    #[test]
    fn broad_equi_windows_use_adaptive_extrema() {
        use crate::conditional_join::range_join::EXTREME_TREE_HITS;
        let left = Array1::from_iter(100_i64..164);
        let right =
            Array1::from_iter((0..1_024).map(|position| ((position * 7919) % 1_024) as i64));
        let starts = Array1::from_iter((0..64).map(|row| (row * 8) as i64));
        let ends = Array1::from_iter((0..64).map(|row| (row * 8 + 512) as i64));

        let before = EXTREME_TREE_HITS.get();
        let first = materialize_equi_windows(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            Keep::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(EXTREME_TREE_HITS.get(), before + 1);
        let last = materialize_equi_windows(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            Keep::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(EXTREME_TREE_HITS.get(), before + 2);

        let expected_first = (0..64)
            .map(|row| {
                right.as_slice().unwrap()[row * 8..row * 8 + 512]
                    .iter()
                    .copied()
                    .min()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let expected_last = (0..64)
            .map(|row| {
                right.as_slice().unwrap()[row * 8..row * 8 + 512]
                    .iter()
                    .copied()
                    .max()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(first.0, left.to_vec());
        assert_eq!(first.1, expected_first);
        assert_eq!(last.0, left.to_vec());
        assert_eq!(last.1, expected_last);
    }

    #[test]
    fn range_bound_reports_when_every_window_is_eliminated() {
        let left = Array1::from_vec(vec![5]);
        let right = Array1::from_vec(vec![1, 2, 3, 4]);
        let mut starts = Array1::from_vec(vec![0]);
        let mut ends = Array1::from_vec(vec![4]);

        let has_active_window = apply_equi_range_bound(
            left.view(),
            right.view(),
            starts.view_mut(),
            ends.view_mut(),
            CompareOp::Lt,
        )
        .unwrap();

        assert!(!has_active_window);
        assert_eq!(starts.to_vec(), vec![-1]);
        assert_eq!(ends.to_vec(), vec![4]);
    }

    #[test]
    fn range_bound_preserves_partial_window_alignment() {
        let left = Array1::from_vec(vec![2, 5]);
        let right = Array1::from_vec(vec![1, 2, 3, 4, 5]);
        let mut starts = Array1::from_vec(vec![0, 0]);
        let mut ends = Array1::from_vec(vec![5, 5]);

        let has_active_window = apply_equi_range_bound(
            left.view(),
            right.view(),
            starts.view_mut(),
            ends.view_mut(),
            CompareOp::Lt,
        )
        .unwrap();

        assert!(has_active_window);
        assert_eq!(starts.to_vec(), vec![2, -1]);
        assert_eq!(ends.to_vec(), vec![5, 5]);
    }

    #[test]
    fn range_bound_uses_correct_half_open_boundaries() {
        let left = Array1::from_vec(vec![2]);
        let right = Array1::from_vec(vec![1, 2, 3]);

        let mut starts = Array1::from_vec(vec![0]);
        let mut ends = Array1::from_vec(vec![3]);
        assert!(apply_equi_range_bound(
            left.view(),
            right.view(),
            starts.view_mut(),
            ends.view_mut(),
            CompareOp::Lt,
        )
        .unwrap());
        assert_eq!(starts.to_vec(), vec![2]);

        let mut starts = Array1::from_vec(vec![0]);
        let mut ends = Array1::from_vec(vec![3]);
        assert!(apply_equi_range_bound(
            left.view(),
            right.view(),
            starts.view_mut(),
            ends.view_mut(),
            CompareOp::Le,
        )
        .unwrap());
        assert_eq!(starts.to_vec(), vec![1]);

        let mut starts = Array1::from_vec(vec![0]);
        let mut ends = Array1::from_vec(vec![3]);
        assert!(apply_equi_range_bound(
            left.view(),
            right.view(),
            starts.view_mut(),
            ends.view_mut(),
            CompareOp::Gt,
        )
        .unwrap());
        assert_eq!(ends.to_vec(), vec![1]);

        let mut starts = Array1::from_vec(vec![0]);
        let mut ends = Array1::from_vec(vec![3]);
        assert!(apply_equi_range_bound(
            left.view(),
            right.view(),
            starts.view_mut(),
            ends.view_mut(),
            CompareOp::Ge,
        )
        .unwrap());
        assert_eq!(ends.to_vec(), vec![2]);
    }

    #[test]
    fn unordered_second_range_applies_all_keep_modes() {
        Python::initialize();
        Python::attach(|py| {
            let range = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                left: PyArray1::from_vec(py, vec![3_i64]).readonly(),
                left_index: PyArray1::from_vec(py, vec![0_i64]).readonly(),
                right: PyArray1::from_vec(py, vec![4_i64, 1, 3, 2]).readonly(),
                right_index: PyArray1::from_vec(py, vec![0_i64, 1, 2, 3]).readonly(),
                op: CompareOp::Gt,
            });
            let left_index = Array1::from_vec(vec![10]);
            let right_index = Array1::from_vec(vec![10, 11, 12, 13]);
            for (keep, expected) in [
                (Keep::All, vec![11, 13]),
                (Keep::Any, vec![11]),
                (Keep::First, vec![11]),
                (Keep::Last, vec![13]),
            ] {
                let mut starts = Array1::from_vec(vec![0]);
                let mut ends = Array1::from_vec(vec![4]);
                let equality_starts = starts.clone();
                let equality_ends = ends.clone();
                let result = materialize_unordered_second_range(
                    left_index.view(),
                    right_index.view(),
                    starts.view_mut(),
                    ends.view_mut(),
                    &range,
                    equality_starts.view(),
                    equality_ends.view(),
                    &[],
                    None,
                    keep,
                )
                .unwrap()
                .unwrap();
                assert_eq!(result.1, expected);
            }
        });
    }

    /// Deterministic xorshift generator so the randomized cases are identical
    /// on every run and platform.
    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, bound: usize) -> usize {
            (self.next() % bound as u64) as usize
        }
    }

    const RANGE_OPS: [CompareOp; 4] = [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge];

    fn skip_hits() -> usize {
        SECOND_RANGE_SKIP_HITS.with(|hits| hits.get())
    }

    #[test]
    fn second_range_skip_scan_matches_bruteforce() {
        let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
        let hits_before = skip_hits();
        let super_skips_before = SECOND_RANGE_SUPER_SKIPS.with(|skips| skips.get());
        // `sparse` puts a rare extreme value (0 or 49) in a mostly-middle
        // column, so whole blocks and super-blocks fail and get skipped.
        for (len, sparse) in [1_usize, 31, 32, 33, 63, 64, 65, 100, 1_000, 4_097, 20_000]
            .into_iter()
            .flat_map(|len| [(len, false), (len, true)])
        {
            for op in RANGE_OPS {
                let ints = Array1::from_iter((0..len).map(|_| {
                    if !sparse {
                        rng.below(50) as i64
                    } else if rng.below(500) == 0 {
                        [0, 49][rng.below(2)]
                    } else {
                        25
                    }
                }));
                let floats = ints.mapv(|value| value as f64 / 4.0);
                let int_scan = SecondRangeSkipScan::new(ints.view(), op);
                let float_scan = SecondRangeSkipScan::new(floats.view(), op);
                let direct = SecondRangeSkipScan::direct(ints.view(), op);
                // Every window, including empty ones and the exact block
                // edges, for small arrays; random windows for large ones.
                let windows: Vec<(usize, usize)> = if len <= 65 {
                    (0..=len)
                        .flat_map(|start| (start..=len).map(move |end| (start, end)))
                        .collect()
                } else {
                    (0..300)
                        .map(|_| {
                            let start = rng.below(len + 1);
                            (start, start + rng.below(len - start + 1))
                        })
                        .chain([(0, len), (0, 0), (len, len)])
                        .collect()
                };
                for (start, end) in windows {
                    let threshold = if sparse {
                        [1, 2, 47, 48][rng.below(4)]
                    } else {
                        rng.below(52) as i64 - 1
                    };
                    let expected: Vec<usize> = (start..end)
                        .filter(|&position| op.apply(&threshold, &ints[position]))
                        .collect();
                    let mut got = Vec::new();
                    int_scan.for_each_match(threshold, start, end, |position| {
                        got.push(position);
                        true
                    });
                    assert_eq!(got, expected, "i64 len={len} op={op:?} [{start}, {end})");
                    got.clear();
                    float_scan.for_each_match(threshold as f64 / 4.0, start, end, |position| {
                        got.push(position);
                        true
                    });
                    assert_eq!(got, expected, "f64 len={len} op={op:?} [{start}, {end})");
                    got.clear();
                    direct.for_each_match(threshold, start, end, |position| {
                        got.push(position);
                        true
                    });
                    assert_eq!(got, expected, "direct len={len} op={op:?}");
                    // Opened runs are ordered, disjoint, inside the window,
                    // and hold every match.
                    for scanner in [&int_scan, &direct] {
                        let mut previous_end = start;
                        got.clear();
                        scanner.for_each_open_run(threshold, start, end, |run_start, run_end| {
                            assert!(previous_end <= run_start && run_start < run_end);
                            assert!(run_end <= end);
                            previous_end = run_end;
                            got.extend(
                                (run_start..run_end)
                                    .filter(|&position| op.apply(&threshold, &ints[position])),
                            );
                            true
                        });
                        assert_eq!(got, expected, "runs len={len} op={op:?} [{start}, {end})");
                    }
                    for scanner_count in [
                        int_scan.count_matches(threshold, start, end),
                        float_scan.count_matches(threshold as f64 / 4.0, start, end),
                        direct.count_matches(threshold, start, end),
                    ] {
                        assert_eq!(scanner_count, expected.len(), "count len={len} op={op:?}");
                    }
                    // Stopping early must yield the lowest matching position.
                    let mut first = None;
                    int_scan.for_each_match(threshold, start, end, |position| {
                        first = Some(position);
                        false
                    });
                    assert_eq!(first, expected.first().copied());
                }
            }
        }
        assert!(skip_hits() > hits_before, "the skip scan never ran");
        assert!(
            SECOND_RANGE_SUPER_SKIPS.with(|skips| skips.get()) > super_skips_before,
            "no super-block was ever skipped"
        );
    }

    /// Build a two-group overlap-join fixture whose second range defeats the
    /// envelope: each group starts with one very long interval, so the
    /// suffix-minimum of the right starts stays tiny across the whole group.
    ///
    /// Returns (first-range left, first-range right, second-range left,
    /// second-range right, group windows per left row, right labels).
    #[allow(clippy::type_complexity)]
    fn overscan_fixture(
        rng: &mut XorShift,
    ) -> (
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<i64>,
        Vec<(i64, i64)>,
        Vec<i64>,
    ) {
        let group_len = 600_usize;
        let mut right_start = Vec::new();
        let mut right_end = Vec::new();
        let mut group_windows = Vec::new();
        for group in 0..2 {
            let offset = right_start.len();
            let mut intervals: Vec<(i64, i64)> = (0..group_len)
                .map(|_| {
                    let start = rng.below(10_000) as i64;
                    (start, start + rng.below(40) as i64)
                })
                .collect();
            // The long interval: it starts first and ends last.
            intervals[0] = (0, 20_000);
            // Right rows are sorted by the first-range column (the end).
            intervals.sort_by_key(|&(start, end)| (end, start));
            for (start, end) in intervals {
                right_start.push(start);
                right_end.push(end);
            }
            group_windows.push((offset as i64, (offset + group_len) as i64, group));
        }
        let mut left_start = Vec::new();
        let mut left_end = Vec::new();
        let mut windows = Vec::new();
        for row in 0..400 {
            let start = rng.below(10_000) as i64;
            left_start.push(start);
            left_end.push(start + rng.below(30) as i64);
            let (group_start, group_end, _) = group_windows[row % 2];
            windows.push((group_start, group_end));
        }
        // Unique but unordered physical labels, so first/last must compare
        // labels rather than positions.
        let mut labels: Vec<i64> = (0..right_start.len() as i64)
            .map(|v| v * 7 % 1_201)
            .collect();
        labels.reverse();
        (
            left_start,
            right_end,
            left_end,
            right_start,
            windows,
            labels,
        )
    }

    #[test]
    fn overscan_second_range_uses_skip_scan_and_matches_bruteforce() {
        Python::initialize();
        Python::attach(|py| {
            let mut rng = XorShift(0xD1B5_4A32_D192_ED03);
            let (left_start, right_end, left_end, right_start, windows, labels) =
                overscan_fixture(&mut rng);
            let left_len = left_start.len();
            // Overlap: start_l <= end_r (first, sorted) and end_l >= start_r
            // (second, unordered inside each group).
            let range_pair = || {
                let first = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                    left: PyArray1::from_vec(py, left_start.clone()).readonly(),
                    left_index: PyArray1::from_vec(py, (0..left_len as i64).collect()).readonly(),
                    right: PyArray1::from_vec(py, right_end.clone()).readonly(),
                    right_index: PyArray1::from_vec(py, labels.clone()).readonly(),
                    op: CompareOp::Le,
                });
                let second = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                    left: PyArray1::from_vec(py, left_end.clone()).readonly(),
                    left_index: PyArray1::from_vec(py, (0..left_len as i64).collect()).readonly(),
                    right: PyArray1::from_vec(py, right_start.clone()).readonly(),
                    right_index: PyArray1::from_vec(py, labels.clone()).readonly(),
                    op: CompareOp::Ge,
                });
                (first, second)
            };
            let left_labels = Array1::from_iter((0..left_len as i64).map(|v| v + 1_000));
            let right_labels = Array1::from_vec(labels.clone());
            let mut expected: Vec<Vec<i64>> = vec![Vec::new(); left_len];
            for (row, &(start, end)) in windows.iter().enumerate() {
                for position in start as usize..end as usize {
                    if left_start[row] <= right_end[position]
                        && left_end[row] >= right_start[position]
                    {
                        expected[row].push(labels[position]);
                    }
                }
            }
            assert!(expected.iter().any(|matches| !matches.is_empty()));
            for keep in [Keep::All, Keep::Any, Keep::First, Keep::Last] {
                let (first, second) = range_pair();
                let mut starts = Array1::from_iter(windows.iter().map(|w| w.0));
                let mut ends = Array1::from_iter(windows.iter().map(|w| w.1));
                let hits_before = skip_hits();
                let (left_out, right_out) = build_equi_range_indices_core(
                    left_labels.view(),
                    right_labels.view(),
                    starts.view_mut(),
                    ends.view_mut(),
                    &first,
                    Some(&second),
                    keep,
                )
                .unwrap()
                .unwrap();
                assert!(
                    skip_hits() > hits_before,
                    "keep={keep:?} did not use the skip scan"
                );
                let mut got: Vec<Vec<i64>> = vec![Vec::new(); left_len];
                for (left, right) in left_out.iter().zip(&right_out) {
                    got[(*left - 1_000) as usize].push(*right);
                }
                for row in 0..left_len {
                    let want = &expected[row];
                    let have = &got[row];
                    match keep {
                        Keep::All => {
                            let mut sorted_want = want.clone();
                            let mut sorted_have = have.clone();
                            sorted_want.sort_unstable();
                            sorted_have.sort_unstable();
                            assert_eq!(sorted_have, sorted_want, "row {row}");
                        }
                        Keep::Any => {
                            assert_eq!(have.len(), usize::from(!want.is_empty()));
                            assert!(have.iter().all(|label| want.contains(label)));
                        }
                        Keep::First => {
                            assert_eq!(have.first(), want.iter().min(), "row {row}");
                        }
                        Keep::Last => {
                            assert_eq!(have.first(), want.iter().max(), "row {row}");
                        }
                    }
                }
            }
        });
    }

    #[test]
    fn overscan_second_range_aggregation_uses_skip_scan() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let mut rng = XorShift(0x2545_F491_4F6C_DD1D);
            let (left_start, right_end, left_end, right_start, windows, labels) =
                overscan_fixture(&mut rng);
            let left_len = left_start.len();
            let right_len = right_end.len();
            let values: Vec<i64> = (0..right_len as i64).map(|v| v * 3 + 1).collect();
            let left_values: Vec<i64> = (0..left_len as i64).map(|v| v * 5 + 2).collect();
            for reverse in [false, true] {
                let ranges = PyList::new(
                    py,
                    [
                        equi_range_tuple(py, left_start.clone(), right_end.clone(), "<=")?,
                        equi_range_tuple(py, left_end.clone(), right_start.clone(), ">=")?,
                    ],
                )?;
                let aggregations = if reverse {
                    equi_sum_aggregations(py, left_values.clone())?
                } else {
                    equi_sum_aggregations(py, values.clone())?
                };
                let starts = PyArray1::from_vec(py, windows.iter().map(|w| w.0).collect());
                let ends = PyArray1::from_vec(py, windows.iter().map(|w| w.1).collect());
                let hits_before = skip_hits();
                let result = equi_aggregate(
                    py,
                    PyArray1::from_vec(py, (0..left_len as i64).collect()).readonly(),
                    PyArray1::from_vec(py, labels.clone()).readonly(),
                    Some(starts.readwrite()),
                    Some(ends.readwrite()),
                    &ranges,
                    &PyList::empty(py),
                    &aggregations,
                    true,
                    reverse,
                )?
                .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;
                assert!(
                    skip_hits() > hits_before,
                    "reverse={reverse} did not use the skip scan"
                );
                let output_len = if reverse { right_len } else { left_len };
                let mut want_sums = vec![0_i64; output_len];
                let mut want_matched = vec![false; output_len];
                for (row, &(start, end)) in windows.iter().enumerate() {
                    for position in start as usize..end as usize {
                        if left_start[row] <= right_end[position]
                            && left_end[row] >= right_start[position]
                        {
                            if reverse {
                                want_sums[position] += left_values[row];
                                want_matched[position] = true;
                            } else {
                                want_sums[row] += values[position];
                                want_matched[row] = true;
                            }
                        }
                    }
                }
                let (matched, sums) = equi_sum_result(&result)?;
                assert_eq!(matched, want_matched, "reverse={reverse}");
                assert_eq!(sums, want_sums, "reverse={reverse}");
            }
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn overscan_second_range_with_residuals_uses_skip_scan_and_matches_bruteforce() {
        Python::initialize();
        Python::attach(|py| {
            let mut rng = XorShift(0x6A09_E667_F3BC_C908);
            let (left_start, right_end, left_end, right_start, windows, labels) =
                overscan_fixture(&mut rng);
            let left_len = left_start.len();
            let right_len = right_end.len();
            // Residuals: weight_l > weight_r and tag_l != tag_r.
            let left_weight = Array1::from_iter((0..left_len).map(|_| rng.below(100) as i64));
            let right_weight = Array1::from_iter((0..right_len).map(|_| rng.below(100) as i64));
            let left_tag = Array1::from_iter((0..left_len).map(|_| rng.below(3) as i64));
            let right_tag = Array1::from_iter((0..right_len).map(|_| rng.below(3) as i64));
            let residuals = [
                PredicateView::I64(left_weight.view(), right_weight.view(), CompareOp::Gt),
                PredicateView::I64(left_tag.view(), right_tag.view(), CompareOp::Ne),
            ];
            let left_labels = Array1::from_iter((0..left_len as i64).map(|v| v + 1_000));
            let right_labels = Array1::from_vec(labels.clone());
            let mut expected: Vec<Vec<i64>> = vec![Vec::new(); left_len];
            for (row, &(start, end)) in windows.iter().enumerate() {
                for position in start as usize..end as usize {
                    if left_start[row] <= right_end[position]
                        && left_end[row] >= right_start[position]
                        && left_weight[row] > right_weight[position]
                        && left_tag[row] != right_tag[position]
                    {
                        expected[row].push(labels[position]);
                    }
                }
            }
            assert!(expected.iter().any(|matches| !matches.is_empty()));
            for keep in [Keep::All, Keep::Any, Keep::First, Keep::Last] {
                let first = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                    left: PyArray1::from_vec(py, left_start.clone()).readonly(),
                    left_index: PyArray1::from_vec(py, (0..left_len as i64).collect()).readonly(),
                    right: PyArray1::from_vec(py, right_end.clone()).readonly(),
                    right_index: PyArray1::from_vec(py, labels.clone()).readonly(),
                    op: CompareOp::Le,
                });
                let second = AnyParsedRangePredicate::I64(ParsedRangePredicate {
                    left: PyArray1::from_vec(py, left_end.clone()).readonly(),
                    left_index: PyArray1::from_vec(py, (0..left_len as i64).collect()).readonly(),
                    right: PyArray1::from_vec(py, right_start.clone()).readonly(),
                    right_index: PyArray1::from_vec(py, labels.clone()).readonly(),
                    op: CompareOp::Ge,
                });
                let mut starts = Array1::from_iter(windows.iter().map(|w| w.0));
                let mut ends = Array1::from_iter(windows.iter().map(|w| w.1));
                let hits_before = skip_hits();
                let (left_out, right_out) = build_equi_range_and_residual_indices_core(
                    left_labels.view(),
                    right_labels.view(),
                    starts.view_mut(),
                    ends.view_mut(),
                    &first,
                    Some(&second),
                    &residuals,
                    None,
                    keep,
                )
                .unwrap()
                .unwrap();
                assert!(
                    skip_hits() > hits_before,
                    "keep={keep:?} did not use the skip scan"
                );
                let mut got: Vec<Vec<i64>> = vec![Vec::new(); left_len];
                for (left, right) in left_out.iter().zip(&right_out) {
                    got[(*left - 1_000) as usize].push(*right);
                }
                for row in 0..left_len {
                    let (want, have) = (&expected[row], &got[row]);
                    match keep {
                        Keep::All => {
                            let mut sorted_want = want.clone();
                            let mut sorted_have = have.clone();
                            sorted_want.sort_unstable();
                            sorted_have.sort_unstable();
                            assert_eq!(sorted_have, sorted_want, "row {row}");
                        }
                        Keep::Any => {
                            assert_eq!(have.len(), usize::from(!want.is_empty()));
                            assert!(have.iter().all(|label| want.contains(label)));
                        }
                        Keep::First => assert_eq!(have.first(), want.iter().min(), "row {row}"),
                        Keep::Last => assert_eq!(have.first(), want.iter().max(), "row {row}"),
                    }
                }
            }
        });
    }

    #[test]
    fn overscan_second_range_aggregation_with_residual_uses_skip_scan() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let mut rng = XorShift(0xBB67_AE85_84CA_A73B);
            let (left_start, right_end, left_end, right_start, windows, labels) =
                overscan_fixture(&mut rng);
            let left_len = left_start.len();
            let right_len = right_end.len();
            let left_weight: Vec<i64> = (0..left_len).map(|_| rng.below(100) as i64).collect();
            let right_weight: Vec<i64> = (0..right_len).map(|_| rng.below(100) as i64).collect();
            let values: Vec<i64> = (0..right_len as i64).map(|v| v * 3 + 1).collect();
            let left_values: Vec<i64> = (0..left_len as i64).map(|v| v * 5 + 2).collect();
            for reverse in [false, true] {
                let ranges = PyList::new(
                    py,
                    [
                        equi_range_tuple(py, left_start.clone(), right_end.clone(), "<=")?,
                        equi_range_tuple(py, left_end.clone(), right_start.clone(), ">=")?,
                    ],
                )?;
                let residuals = PyList::new(
                    py,
                    [equi_range_tuple(
                        py,
                        left_weight.clone(),
                        right_weight.clone(),
                        ">",
                    )?],
                )?;
                let aggregations = if reverse {
                    equi_sum_aggregations(py, left_values.clone())?
                } else {
                    equi_sum_aggregations(py, values.clone())?
                };
                let starts = PyArray1::from_vec(py, windows.iter().map(|w| w.0).collect());
                let ends = PyArray1::from_vec(py, windows.iter().map(|w| w.1).collect());
                let hits_before = skip_hits();
                let result = equi_aggregate(
                    py,
                    PyArray1::from_vec(py, (0..left_len as i64).collect()).readonly(),
                    PyArray1::from_vec(py, labels.clone()).readonly(),
                    Some(starts.readwrite()),
                    Some(ends.readwrite()),
                    &ranges,
                    &residuals,
                    &aggregations,
                    true,
                    reverse,
                )?
                .ok_or_else(|| PyValueError::new_err("equi aggregation returned None"))?;
                assert!(
                    skip_hits() > hits_before,
                    "reverse={reverse} did not use the skip scan"
                );
                let output_len = if reverse { right_len } else { left_len };
                let mut want_sums = vec![0_i64; output_len];
                let mut want_matched = vec![false; output_len];
                for (row, &(start, end)) in windows.iter().enumerate() {
                    for position in start as usize..end as usize {
                        if left_start[row] <= right_end[position]
                            && left_end[row] >= right_start[position]
                            && left_weight[row] > right_weight[position]
                        {
                            if reverse {
                                want_sums[position] += left_values[row];
                                want_matched[position] = true;
                            } else {
                                want_sums[row] += values[position];
                                want_matched[row] = true;
                            }
                        }
                    }
                }
                let (matched, sums) = equi_sum_result(&result)?;
                assert_eq!(matched, want_matched, "reverse={reverse}");
                assert_eq!(sums, want_sums, "reverse={reverse}");
            }
            Ok(())
        })
        .unwrap();
    }
}
