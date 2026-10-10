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

use crate::aggregation_common::aggregation::{
    make_results_with_positions, parse_inputs, AggregationSet,
};
use crate::aggregation_common::ensure_equal_lengths_core;
use crate::compare_op::CompareOp;
use crate::join_types::{result_dict, Keep};
use crate::predicate::{
    check_predicate_lengths, null_metadata_views, parse_predicates_with_nulls_strings,
    predicates_match_dispatch, NullMetadataView, Predicate, PredicateView,
};
use crate::range_predicate::{parse_any_range_parts, AnyParsedRangePredicate};

type IndexPairs = Option<(Vec<i64>, Vec<i64>)>;

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
            return materialize_unordered_second_range(
                left_index,
                right_index,
                starts.view_mut(),
                ends.view_mut(),
                second_range,
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
            for (&start, &end) in starts.iter().zip(ends.iter()) {
                if start < 0 || end < 0 {
                    continue;
                }
                let start = start as usize;
                let end = end as usize;
                if right
                    .slice(s![start..end])
                    .windows(2)
                    .into_iter()
                    .any(|window| {
                        !matches!(
                            window[0].partial_cmp(&window[1]),
                            Some(Ordering::Less | Ordering::Equal)
                        )
                    })
                {
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
    if keep == Keep::All {
        let mut output_capacity = 0_usize;
        for (&start, &end) in starts.iter().zip(ends.iter()) {
            if start < 0 || end < 0 {
                continue;
            }
            output_capacity = output_capacity
                .checked_add(end as usize - start as usize)
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
            for &right_label in right_index.slice(s![start as usize..end as usize]).iter() {
                left_output.push(left_index[row]);
                right_output.push(right_label);
            }
        }
        return Ok(Some((left_output, right_output)));
    }

    let mut left_output = Vec::with_capacity(left_index.len());
    let mut right_output = Vec::with_capacity(left_index.len());
    for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
        if start < 0 || end <= start {
            continue;
        }
        let mut selected = None;
        for right_position in start as usize..end as usize {
            selected = match (keep, selected) {
                (Keep::Any, _) => Some(right_position),
                (Keep::First, None) => Some(right_position),
                (Keep::First, Some(current))
                    if right_index[right_position] < right_index[current] =>
                {
                    Some(right_position)
                }
                (Keep::Last, None) => Some(right_position),
                (Keep::Last, Some(current))
                    if right_index[right_position] > right_index[current] =>
                {
                    Some(right_position)
                }
                (_, current) => current,
            };
            if keep == Keep::Any {
                break;
            }
        }
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
/// * `keep` - Match retention policy after exact filtering.
///
/// # Returns
///
/// `None` when no exact matches survive; otherwise the physical left/right
/// index pairs selected by `keep`.
fn materialize_unordered_second_range(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    mut starts: ArrayViewMut1<'_, i64>,
    mut ends: ArrayViewMut1<'_, i64>,
    range: &AnyParsedRangePredicate<'_>,
    keep: Keep,
) -> Result<IndexPairs, String> {
    if !narrow_unordered_second_range_for_predicate(range, starts.view_mut(), ends.view_mut())? {
        return Ok(None);
    }

    macro_rules! materialize {
        ($predicate:expr) => {{
            let predicate = $predicate;
            let left = predicate.left.as_array();
            let right = predicate.right.as_array();
            if keep == Keep::All {
                let mut output_capacity = 0_usize;
                for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                    if start < 0 || end < 0 {
                        continue;
                    }
                    let start = start as usize;
                    let end = end as usize;
                    for right_position in start..end {
                        if range_value_matches(left[row], right[right_position], predicate.op) {
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
                    for right_position in start..end {
                        if range_value_matches(left[row], right[right_position], predicate.op) {
                            left_output.push(left_index[row]);
                            right_output.push(right_index[right_position]);
                        }
                    }
                }
                return Ok(Some((left_output, right_output)));
            }

            let mut left_output = Vec::with_capacity(left_index.len());
            let mut right_output = Vec::with_capacity(left_index.len());
            for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                if start < 0 || end < 0 {
                    continue;
                }
                let start = start as usize;
                let end = end as usize;
                let mut selected = None;
                for right_position in start..end {
                    if !range_value_matches(left[row], right[right_position], predicate.op) {
                        continue;
                    }
                    match keep {
                        Keep::Any => {
                            selected = Some(right_position);
                            break;
                        }
                        Keep::First
                            if selected.is_none_or(|current| {
                                right_index[right_position] < right_index[current]
                            }) =>
                        {
                            selected = Some(right_position)
                        }
                        Keep::Last
                            if selected.is_none_or(|current| {
                                right_index[right_position] > right_index[current]
                            }) =>
                        {
                            selected = Some(right_position)
                        }
                        Keep::All | Keep::First | Keep::Last => {}
                    }
                }
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
/// This helper is shared by direct index materialization and the aggregation
/// path. Both paths must narrow the candidate windows before their exact
/// predicate/residual scan; otherwise aggregation would silently retain the
/// slower full-window behavior.
///
/// # Arguments
///
/// * `range` - The unordered second range predicate and its aligned values.
/// * `starts` - Mutable post-first-range window starts.
/// * `ends` - Mutable post-first-range window ends.
///
/// # Returns
///
/// `true` when at least one candidate window remains after envelope narrowing.
fn narrow_unordered_second_range_for_predicate(
    range: &AnyParsedRangePredicate<'_>,
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
            for (&start, &end) in starts.iter().zip(ends.iter()) {
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

fn range_value_matches<T: PartialOrd>(left: T, right: T, op: CompareOp) -> bool {
    match op {
        CompareOp::Lt => left < right,
        CompareOp::Le => left <= right,
        CompareOp::Gt => left > right,
        CompareOp::Ge => left >= right,
        CompareOp::Eq => left == right,
        CompareOp::Ne => left != right,
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

fn range_matches_at(
    range: &AnyParsedRangePredicate<'_>,
    left_position: usize,
    right_position: usize,
) -> bool {
    macro_rules! matches {
        ($predicate:expr) => {
            range_value_matches(
                $predicate.left.as_array()[left_position],
                $predicate.right.as_array()[right_position],
                $predicate.op,
            )
        };
    }
    match range {
        AnyParsedRangePredicate::I64(predicate) => matches!(predicate),
        AnyParsedRangePredicate::I32(predicate) => matches!(predicate),
        AnyParsedRangePredicate::I16(predicate) => matches!(predicate),
        AnyParsedRangePredicate::I8(predicate) => matches!(predicate),
        AnyParsedRangePredicate::U64(predicate) => matches!(predicate),
        AnyParsedRangePredicate::U32(predicate) => matches!(predicate),
        AnyParsedRangePredicate::U16(predicate) => matches!(predicate),
        AnyParsedRangePredicate::U8(predicate) => matches!(predicate),
        AnyParsedRangePredicate::F64(predicate) => matches!(predicate),
        AnyParsedRangePredicate::F32(predicate) => matches!(predicate),
    }
}

/// Return the first offset at which `predicate` is false within `[start, end)`.
///
/// The returned offset is relative to `values`, not to the supplied window.
/// This helper is kept local because bounded partitioning is specific to
/// narrowing prepared equi-join windows.
fn partition_point_between<T: PartialOrd + Copy>(
    values: ArrayView1<'_, T>,
    start: usize,
    end: usize,
    predicate: impl Fn(T) -> bool,
) -> usize {
    debug_assert!(start <= end);
    debug_assert!(end <= values.len());

    if let Some(slice) = values.as_slice() {
        return start + slice[start..end].partition_point(|value| predicate(*value));
    }
    let mut low = start;
    let mut high = end;
    while low < high {
        let middle = low + ((high - low) >> 1);
        if predicate(values[middle]) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
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
    let matches = |row: usize, right_position: usize| {
        predicates_match_dispatch(residual_predicates, null_metadata, row, right_position)
            && additional_range
                .map(|range| range_matches_at(range, row, right_position))
                .unwrap_or(true)
    };
    if keep == Keep::All {
        let mut output_capacity = 0_usize;
        for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
            if start < 0 || end < 0 {
                continue;
            }
            for right_position in start as usize..end as usize {
                if matches(row, right_position) {
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
            for right_position in start as usize..end as usize {
                if matches(row, right_position) {
                    left_output.push(left_index[row]);
                    right_output.push(right_index[right_position]);
                }
            }
        }
        return Ok(Some((left_output, right_output)));
    }

    let mut left_output = Vec::with_capacity(left_index.len());
    let mut right_output = Vec::with_capacity(left_index.len());
    for (row, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
        if start < 0 || end <= start {
            continue;
        }
        let mut selected = None;
        for right_position in start as usize..end as usize {
            if !matches(row, right_position) {
                continue;
            }
            selected = match (keep, selected) {
                (Keep::Any, _) => Some(right_position),
                (Keep::First, None) => Some(right_position),
                (Keep::First, Some(current))
                    if right_index[right_position] < right_index[current] =>
                {
                    Some(right_position)
                }
                (Keep::Last, None) => Some(right_position),
                (Keep::Last, Some(current))
                    if right_index[right_position] > right_index[current] =>
                {
                    Some(right_position)
                }
                (_, current) => current,
            };
            if keep == Keep::Any {
                break;
            }
        }
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
            if !narrow_unordered_second_range_for_predicate(
                second_range,
                starts.view_mut(),
                ends.view_mut(),
            )? {
                return Ok(None);
            }
            return materialize_equi_windows_and_residuals(
                left_index,
                right_index,
                starts.view(),
                ends.view(),
                residual_predicates,
                null_metadata,
                Some(second_range),
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
/// matches use `starts..ends` windows. Range windows are narrowed before the
/// candidate loop; a non-monotonic second range remains a residual check.
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

    let output_len = if reverse {
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
                demoted_second = Some(second_range);
            }
        }
    } else {
        for range in &ranges {
            validate_equi_range_inputs(range, left_index.len(), right_index.len())
                .map_err(PyValueError::new_err)?;
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

        for right_position in candidate_start..candidate_end {
            if !predicates_match_dispatch(&views, metadata_views.as_deref(), row, right_position) {
                continue;
            }
            if starts.is_none()
                && ranges
                    .iter()
                    .any(|range| !range_matches_at(range, row, right_position))
            {
                continue;
            }
            if let Some(range) = demoted_second {
                if !range_matches_at(range, row, right_position) {
                    continue;
                }
            }
            if reverse {
                set.update(row, right_position);
            } else {
                set.update(right_position, row);
            }
        }
    }

    if set.is_empty() {
        return Ok(None);
    }
    let output_positions = if reverse {
        Some(right_index)
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

/// Build index pairs for unique equality keys with at least one predicate.
///
/// This compatibility wrapper retains the older non-empty-predicate
/// contract. New callers that may have zero aligned predicates should use
/// [`equi_uniq_residual_indices`] instead.
///
/// # Arguments
///
/// * `left_index` - Prepared left labels or physical positions.
/// * `right_index` - Prepared right labels or physical positions aligned
///   one-for-one with `left_index`.
/// * `residual_predicates` - Non-empty list of aligned predicate tuples.
///
/// # Returns
///
/// A dictionary containing the surviving aligned index pairs, or `None` when
/// no pair satisfies the predicates.
#[pyfunction]
pub fn equi_uniq_indices<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    residual_predicates: &Bound<'py, PyList>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if residual_predicates.is_empty() {
        return Err(PyValueError::new_err(
            "residual_predicates must not be empty",
        ));
    }
    equi_uniq_residual_indices(py, left_index, right_index, residual_predicates)
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
            match keep {
                Keep::Any => {
                    selected = Some(right_position);
                    break;
                }
                Keep::First
                    if selected.is_none_or(|position| {
                        right_index[right_position] < right_index[position]
                    }) =>
                {
                    selected = Some(right_position)
                }
                Keep::Last
                    if selected.is_none_or(|position| {
                        right_index[right_position] > right_index[position]
                    }) =>
                {
                    selected = Some(right_position)
                }
                Keep::All => unreachable!(),
                Keep::First | Keep::Last => {}
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
    m.add_function(wrap_pyfunction!(equi_uniq_indices, m)?)?;
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
    use pyo3::Python;

    use super::{
        apply_equi_range_bound, build_equi_ne_indices_core, build_equi_only_indices_core,
        build_equi_uniq_residual_indices_core, build_second_range_envelope,
        materialize_equi_windows, materialize_unordered_second_range,
        narrow_unordered_second_range, range_value_matches,
    };
    use crate::compare_op::CompareOp;
    use crate::join_types::Keep;
    use crate::predicate::PredicateView;
    use crate::range_predicate::{AnyParsedRangePredicate, ParsedRangePredicate};

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
                    if range_value_matches(left[row], right[position], op) {
                        expected.push((row, position));
                    }
                }
                if starts[row] >= 0 && ends[row] >= 0 {
                    for position in starts[row] as usize..ends[row] as usize {
                        if range_value_matches(left[row], right[position], op) {
                            actual.push((row, position));
                        }
                    }
                }
            }
            assert_eq!(actual, expected, "operator: {op:?}");
        }
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
                let result = materialize_unordered_second_range(
                    left_index.view(),
                    right_index.view(),
                    starts.view_mut(),
                    ends.view_mut(),
                    &range,
                    keep,
                )
                .unwrap()
                .unwrap();
                assert_eq!(result.1, expected);
            }
        });
    }
}
