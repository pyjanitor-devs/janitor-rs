//! Index construction for single non-equi joins.
//!
//! Pyjanitor supplies non-null, value-sorted arrays for the search paths and
//! maps each filtered value back to its original physical position. Rust does
//! not sort either side or infer nullness from values such as `NaN`; those
//! responsibilities belong to the Python caller and its position metadata.
//!
//! The value arrays and their index arrays are parallel arrays. For example:
//!
//! ```text
//! right values: [1, 3, 5, 7]
//! right index:  [40, 10, 30, 20]
//! ```
//!
//! Binary search uses the sorted values. Returned results use the aligned
//! original index labels, not the physical positions in the sorted values.
//! Consequently, `right_index_is_ordered` matters when selecting `first` or
//! `last`: an unordered label array requires an extrema scan over the matched
//! prefix or suffix. For `!=`, null semantics are selected by a validated
//! pandas-extension flag: NumPy nulls participate in inequality matches,
//! while pandas extension nulls do not.
//!
//! The basic single-predicate API and the single-anchor extended API live in
//! this module because they share the same range-window and `!=` candidate
//! machinery. The basic functions apply `keep` directly to the first
//! predicate; the extended functions use the first predicate to generate
//! candidates and apply later predicates as residual filters before `keep`.
//!
//! The basic APIs return building blocks or materialized index dictionaries
//! according to their wrapper contract. The extended APIs return only a
//! materialized dictionary containing `left_index` and `right_index`; they
//! must inspect every residual predicate before applying `keep`.

#![allow(dead_code)]

use std::iter::repeat_n;

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::common::{partition_point, range_window};
use crate::join_candidate_materialization::materialize_range_candidates;
use crate::join_common::{result_dict, Keep, SingleJoinResult};
use crate::op::CompareOp;
use crate::predicate::{check_predicate_lengths, parse_predicates_with_nulls_strings};

fn legacy_positions(name: &str, values: ArrayView1<'_, i64>) -> Result<Vec<usize>, String> {
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

fn legacy_validate_partition(
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

fn legacy_choose(current: &mut Option<usize>, candidate: usize, keep: Keep) {
    match current {
        None => *current = Some(candidate),
        Some(previous) if keep == Keep::First && candidate < *previous => *previous = candidate,
        Some(previous) if keep == Keep::Last && candidate > *previous => *previous = candidate,
        _ => {}
    }
}

fn legacy_prefix_extrema(values: &[usize], minimum: bool) -> Vec<usize> {
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

fn legacy_suffix_extrema(values: &[usize], minimum: bool) -> Vec<usize> {
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

fn legacy_all_capacity<T: PartialOrd + Copy>(
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

#[allow(clippy::too_many_arguments)]
/// Build selected physical position pairs for the legacy mixed single-join
/// compatibility wrapper's `!=` branch.
///
/// This is intentionally separate from the dedicated `not_equals_only`
/// traversal. The compatibility wrapper still exposes the historical
/// materialized-pair ABI, while the dedicated path visits pairs directly.
///
/// # Arguments
///
/// * `left` / `right` - Compact non-null value arrays; `right` is sorted.
/// * `left_full_positions` / `right_full_positions` - Full physical layouts.
/// * `left_non_null_positions` / `right_non_null_positions` - Maps from
///   compact offsets to full physical positions.
/// * `left_null_positions` / `right_null_positions` - Optional null-row maps.
/// * `is_extension_array` - Whether nulls participate in `!=` matching.
/// * `keep` - Pair selection mode applied after candidate generation.
///
/// # Returns
///
/// Returns physical left/right position vectors suitable for the legacy
/// materializer. It does not return index labels.
///
/// # Errors
///
/// Returns an error if values and maps are misaligned or the position maps do
/// not form complete, disjoint partitions of their full layouts.
fn build_positions<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_full_positions: ArrayView1<'_, i64>,
    left_non_null_positions: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_full_positions: ArrayView1<'_, i64>,
    right_non_null_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    keep: Keep,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    ensure_equal_lengths_core(
        "left values",
        left.len(),
        "left non-null positions",
        left_non_null_positions.len(),
    )?;
    ensure_equal_lengths_core(
        "right values",
        right.len(),
        "right non-null positions",
        right_non_null_positions.len(),
    )?;
    let left_positions = legacy_positions("left non-null", left_non_null_positions)?;
    let right_positions = legacy_positions("right non-null", right_non_null_positions)?;
    let left_null_positions = left_null_positions
        .map(|values| legacy_positions("left null", values))
        .transpose()?;
    let right_null_positions = right_null_positions
        .map(|values| legacy_positions("right null", values))
        .transpose()?;
    let empty = Vec::new();
    let left_null_positions = left_null_positions.as_deref().unwrap_or(&empty);
    let right_null_positions = right_null_positions.as_deref().unwrap_or(&empty);
    legacy_validate_partition(
        "left full positions",
        left_full_positions.len(),
        &left_positions,
        left_null_positions,
    )?;
    legacy_validate_partition(
        "right full positions",
        right_full_positions.len(),
        &right_positions,
        right_null_positions,
    )?;

    let capacity = if keep == Keep::All {
        legacy_all_capacity(
            left,
            right,
            left_null_positions.len(),
            right_null_positions.len(),
            is_extension_array,
        )?
    } else {
        left_full_positions.len()
    };
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(capacity)
        .map_err(|_| "single join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(capacity)
        .map_err(|_| "single join result allocation failed")?;

    let prefix_extrema = match keep {
        Keep::First => Some(legacy_prefix_extrema(&right_positions, true)),
        Keep::Last => Some(legacy_prefix_extrema(&right_positions, false)),
        Keep::Any | Keep::All => None,
    };
    let suffix_extrema = match keep {
        Keep::First => Some(legacy_suffix_extrema(&right_positions, true)),
        Keep::Last => Some(legacy_suffix_extrema(&right_positions, false)),
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
        if keep == Keep::All {
            output_left.extend(repeat_n(left_position, less_end));
            output_right.extend_from_slice(&right_positions[..less_end]);
            output_left.extend(repeat_n(left_position, right.len() - greater_start));
            output_right.extend_from_slice(&right_positions[greater_start..]);
            if !is_extension_array {
                output_left.extend(repeat_n(left_position, right_null_positions.len()));
                output_right.extend_from_slice(right_null_positions);
            }
            continue;
        }
        let mut selected = None;
        if less_end > 0 {
            selected = match keep {
                Keep::Any => Some(right_positions[0]),
                Keep::First | Keep::Last => {
                    Some(right_positions[prefix_extrema.as_ref().unwrap()[less_end - 1]])
                }
                Keep::All => unreachable!(),
            };
        }
        if greater_start < right.len() {
            match keep {
                Keep::Any if selected.is_none() => {
                    selected = Some(right_positions[greater_start]);
                }
                Keep::Any => {}
                Keep::First | Keep::Last => legacy_choose(
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
                legacy_choose(&mut selected, position, keep);
            }
        }
        if let Some(right_position) = selected {
            output_left.push(left_position);
            output_right.push(right_position);
        }
    }
    if !is_extension_array {
        for &left_position in left_null_positions {
            if keep == Keep::All {
                output_left.extend(repeat_n(left_position, right_full_positions.len()));
                output_right.extend(0..right_full_positions.len());
                continue;
            }
            let selected = match keep {
                Keep::Any | Keep::First => (!right_full_positions.is_empty()).then_some(0),
                Keep::Last => right_full_positions.len().checked_sub(1),
                Keep::All => unreachable!(),
            };
            if let Some(right_position) = selected {
                output_left.push(left_position);
                output_right.push(right_position);
            }
        }
    }
    Ok((output_left, output_right))
}

/// Return the half-open physical right-array window satisfying one range op.
///
/// `right` is already sorted by the caller. The four range operators map to
/// two window shapes:
///
/// ```text
/// left <  right  -> [first right >  left, right.len())
/// left <= right  -> [first right >= left, right.len())
/// left >  right  -> [0, first right >=  left)
/// left >= right  -> [0, first right >   left)
/// ```
///
/// Equality and inequality are intentionally excluded. Equality is handled
/// upstream by pyjanitor, while inequality is the union of the strict prefix
/// and strict suffix and needs its own null-aware implementation.
///
/// Build a running label minimum or maximum for every prefix.
///
/// `minimum=true` produces prefix minima; `minimum=false` produces prefix
/// maxima. The iterator is consumed in logical right-array order. Keeping
/// this helper iterator-based lets it serve both contiguous slices and
/// strided ndarray views without copying either input first.
///
/// For example, with labels `[10, 40, 30, 20]`:
///
/// ```text
/// prefix minimum → [10, 10, 10, 10]
/// prefix maximum → [10, 40, 40, 40]
/// ```
///
/// The result at position `i` summarizes the inclusive range `0..=i`.
/// This helper returns labels, not positions, and is used by the range-join
/// selection path.
///
/// # Arguments
///
/// * `values` - Right-index labels in logical right-array order.
/// * `minimum` - `true` for a running minimum; `false` for a running maximum.
///
/// # Returns
///
/// A label vector with one inclusive-prefix result for each input label.
fn prefix_extreme<I>(values: I, minimum: bool) -> Vec<i64>
where
    I: Iterator<Item = i64>,
{
    let initial = if minimum { i64::MAX } else { i64::MIN };
    values
        .scan(initial, |state, value| {
            *state = if minimum {
                (*state).min(value)
            } else {
                (*state).max(value)
            };
            Some(*state)
        })
        .collect()
}

/// Build a running label minimum or maximum for every suffix.
///
/// `minimum=true` produces suffix minima; `minimum=false` produces suffix
/// maxima. `ExactSizeIterator` lets the helper place each result at its
/// original physical position while `DoubleEndedIterator` lets it scan from
/// right to left. This preserves the logical order for strided views.
///
/// For example, with labels `[10, 40, 30, 20]`:
///
/// ```text
/// suffix minimum → [10, 20, 20, 20]
/// suffix maximum → [40, 40, 30, 20]
/// ```
///
/// The result at position `i` summarizes the inclusive range `i..=last`.
/// This helper returns labels, not positions, and is used by the range-join
/// selection path.
///
/// # Arguments
///
/// * `values` - Right-index labels in logical right-array order.
/// * `minimum` - `true` for a running minimum; `false` for a running maximum.
///
/// # Returns
///
/// A label vector with one inclusive-suffix result for each input label.
fn suffix_extreme<I>(values: I, minimum: bool) -> Vec<i64>
where
    I: DoubleEndedIterator<Item = i64> + ExactSizeIterator,
{
    let length = values.len();
    let mut result = vec![0_i64; length];
    let initial = if minimum { i64::MAX } else { i64::MIN };
    let mut current = initial;
    for (offset, value) in values.rev().enumerate() {
        current = if minimum {
            current.min(value)
        } else {
            current.max(value)
        };
        result[length - 1 - offset] = current;
    }
    result
}

/// Return the physical position of the smallest right-index label for every
/// prefix of a `!=` candidate region.
///
/// The prefix contains right values strictly less than the current left value.
/// This table is used by `keep="first"` when `right_index_is_ordered` is
/// false, so the selected position can later be materialized through the full
/// right-index array.
///
/// For example, if the right labels in sorted-value order are
/// `[40, 10, 30, 20]`:
///
/// ```text
/// labels:       [40, 10, 30, 20]
/// prefix minima: [ 0,  1,  1,  1]
/// ```
///
/// The returned values are offsets into `labels`, not public index labels.
///
/// # Arguments
///
/// * `labels` - Right-index labels in the value-sorted right layout.
///
/// # Returns
///
/// For each prefix ending at `i`, the filtered-layout offset of its smallest
/// label. The caller maps that offset through the right position map before
/// indexing the full right-index array.
#[allow(dead_code)]
fn prefix_min_positions(labels: &[i64]) -> Vec<usize> {
    let mut result = Vec::with_capacity(labels.len());
    let mut current = None;
    for (position, &label) in labels.iter().enumerate() {
        if current.is_none_or(|selected| label < labels[selected]) {
            current = Some(position);
        }
        result.push(current.expect("a prefix position exists after iteration"));
    }
    result
}

/// Return the physical position of the largest right-index label for every
/// prefix of a `!=` candidate region.
///
/// The prefix contains right values strictly less than the current left value.
/// This table is used by `keep="last"` when `right_index_is_ordered` is false.
///
/// For example:
///
/// ```text
/// labels:       [40, 10, 30, 20]
/// prefix maxima: [ 0,  0,  0,  0]
/// ```
///
/// The returned values are offsets into `labels`, not public index labels.
///
/// # Arguments
///
/// * `labels` - Right-index labels in the value-sorted right layout.
///
/// # Returns
///
/// For each prefix ending at `i`, the filtered-layout offset of its largest
/// label. The caller maps that offset through the right position map before
/// indexing the full right-index array.
#[allow(dead_code)]
fn prefix_max_positions(labels: &[i64]) -> Vec<usize> {
    let mut result = Vec::with_capacity(labels.len());
    let mut current = None;
    for (position, &label) in labels.iter().enumerate() {
        if current.is_none_or(|selected| label > labels[selected]) {
            current = Some(position);
        }
        result.push(current.expect("a prefix position exists after iteration"));
    }
    result
}

/// Return the physical position of the smallest right-index label for every
/// suffix of a `!=` candidate region.
///
/// The suffix contains right values strictly greater than the current left
/// value. This table is used by `keep="first"` when
/// `right_index_is_ordered` is false.
///
/// For example:
///
/// ```text
/// labels:       [40, 10, 30, 20]
/// suffix minima: [ 1,  1,  3,  3]
/// ```
///
/// The returned values are offsets into `labels`, not public index labels.
///
/// # Arguments
///
/// * `labels` - Right-index labels in the value-sorted right layout.
///
/// # Returns
///
/// For each suffix beginning at `i`, the filtered-layout offset of its
/// smallest label. The caller maps that offset through the right position map
/// before indexing the full right-index array.
#[allow(dead_code)]
fn suffix_min_positions(labels: &[i64]) -> Vec<usize> {
    let mut result = vec![0_usize; labels.len()];
    let mut current = None;
    for position in (0..labels.len()).rev() {
        let label = labels[position];
        if current.is_none_or(|selected| label < labels[selected]) {
            current = Some(position);
        }
        result[position] = current.expect("a suffix position exists after iteration");
    }
    result
}

/// Return the physical position of the largest right-index label for every
/// suffix of a `!=` candidate region.
///
/// The suffix contains right values strictly greater than the current left
/// value. This table is used by `keep="last"` when
/// `right_index_is_ordered` is false.
///
/// For example:
///
/// ```text
/// labels:       [40, 10, 30, 20]
/// suffix maxima: [ 0,  2,  2,  3]
/// ```
///
/// The returned values are offsets into `labels`, not public index labels.
///
/// # Arguments
///
/// * `labels` - Right-index labels in the value-sorted right layout.
///
/// # Returns
///
/// For each suffix beginning at `i`, the filtered-layout offset of its largest
/// label. The caller maps that offset through the right position map before
/// indexing the full right-index array.
#[allow(dead_code)]
fn suffix_max_positions(labels: &[i64]) -> Vec<usize> {
    let mut result = vec![0_usize; labels.len()];
    let mut current = None;
    for position in (0..labels.len()).rev() {
        let label = labels[position];
        if current.is_none_or(|selected| label > labels[selected]) {
            current = Some(position);
        }
        result[position] = current.expect("a suffix position exists after iteration");
    }
    result
}

/// Build one binary-search window per matching left row.
///
/// # Input contract
///
/// - `left` and `right` contain non-null values.
/// - `right` is sorted in ascending order according to the comparator's
///   `PartialOrd` behavior. Rust trusts the caller and does not sort it.
/// - `left_index` and `right_index` are aligned with their value arrays and
///   contain the original `int64` labels to return.
/// - The value/index arrays may be empty, but must have equal pairwise
///   lengths. An empty side produces no windows and therefore no matches.
/// - `right_index_is_ordered` is accepted for the common wrapper contract but
///   is used later, when windows are reduced to `first` or `last` results.
///
/// # Output
///
/// `left_index` contains only left labels whose windows are non-empty.
/// `right_index` contains the complete supplied right-label array. Each
/// `starts[i]..ends[i]` is the physical range of right values matching
/// `left_index[i]`; the ranges are half-open and use `usize` positions.
///
/// # Arguments
///
/// * `left` - Non-null left values, in the caller's left-row order.
/// * `left_index` - Original `int64` labels aligned with `left`.
/// * `right` - Non-null right values, already sorted for binary search.
/// * `right_index` - Original `int64` labels aligned with `right`.
/// * `right_index_is_ordered` - Whether `right_index` is monotonically
///   increasing in the sorted-right layout. This is consumed by result
///   selection, not by boundary construction.
/// * `op` - One of the four range comparators: `<`, `<=`, `>`, or `>=`.
///
/// # Errors
///
/// Returns an error for mismatched value/index lengths or a comparator that is
/// not a range comparator.
pub fn build_range_core<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    right_index_is_ordered: bool,
    op: CompareOp,
) -> Result<SingleJoinResult, String> {
    build_range_core_with_labels(
        left,
        left_index,
        right,
        right_index,
        right_index_is_ordered,
        op,
        true,
        false,
    )
}

/// Construct one range window, optionally retaining the right labels.
///
/// The first predicate in a dual-range join owns the labels needed for output;
/// the second predicate contributes only positional boundaries. Avoiding the
/// second full label copy preserves the same alignment while reducing memory
/// traffic for large right arrays.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_range_core_with_labels<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    _right_index_is_ordered: bool,
    op: CompareOp,
    include_right_index: bool,
    retain_empty_windows: bool,
) -> Result<SingleJoinResult, String> {
    // This function only constructs the physical value windows. Whether the
    // original right labels are ordered does not affect those boundaries;
    // the flag is accepted here so the range core has the same argument
    // contract as the later selection step.
    ensure_equal_lengths_core("left", left.len(), "left_index", left_index.len())?;
    ensure_equal_lengths_core("right", right.len(), "right_index", right_index.len())?;
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return Err("single join range core requires a range comparator".to_owned());
    }

    // `SingleJoinResult` owns its building blocks so they can be returned to
    // Python after this borrowed input view goes out of scope. The copied
    // labels support both range selection (`first`, `last`, and `all`) and
    // building-block output. The copy preserves their supplied physical
    // order; it does not sort or otherwise change them.
    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: if include_right_index {
            right_index.to_vec()
        } else {
            Vec::new()
        },
        starts: Vec::new(),
        ends: Vec::new(),
    };
    if left.is_empty() || right.is_empty() {
        return Ok(result);
    }
    for (left_position, left_value) in left.iter().enumerate() {
        let (start, end) = range_window(*left_value, right, op);
        if start >= end && !retain_empty_windows {
            continue;
        }
        result.left_index.push(left_index[left_position]);
        result.left_positions.push(left_position);
        result.starts.push(start);
        result.ends.push(end);
    }
    Ok(result)
}

pub(crate) fn choose_range(
    windows: &SingleJoinResult,
    keep: Keep,
    right_index_is_ordered: bool,
    suffix_window: bool,
) -> Result<(Vec<i64>, Vec<i64>), &'static str> {
    // `windows` already contains only non-empty ranges, so every mode except
    // `all` emits at most one pair per retained left row. For `all`, reserve
    // the exact total width to avoid repeated Vec growth while materializing
    // the flattened pairs.
    let labels = windows.right_index.as_slice();
    let output_capacity = if keep == Keep::All {
        let mut capacity = 0_usize;
        for (&start, &end) in windows.starts.iter().zip(windows.ends.iter()) {
            let width = end - start;
            capacity = capacity
                .checked_add(width)
                .ok_or("single join result size exceeds platform capacity")?;
        }
        capacity
    } else {
        // Every retained window emits exactly one pair for first/last/any.
        windows.left_index.len()
    };
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(output_capacity)
        .map_err(|_| "single join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(output_capacity)
        .map_err(|_| "single join result allocation failed")?;

    // `right` is sorted for binary search, but its original labels may not be.
    // Build only the extrema table needed by this window direction and
    // selection mode; `any` and `all` never need one. A prefix window is
    // produced by `>`/`>=`; a suffix window is produced by `<`/`<=`.
    //   < or <=   → [start, right_len)   → suffix window
    //   > or >=   → [0, end)             → prefix window
    let need_extrema = !right_index_is_ordered && matches!(keep, Keep::First | Keep::Last);
    let need_min = keep == Keep::First;
    let need_max = keep == Keep::Last;
    let prefix_min = if need_extrema && !suffix_window && need_min {
        Some(prefix_extreme(labels.iter().copied(), true))
    } else {
        None
    };
    let prefix_max = if need_extrema && !suffix_window && need_max {
        Some(prefix_extreme(labels.iter().copied(), false))
    } else {
        None
    };
    let suffix_min = if need_extrema && suffix_window && need_min {
        Some(suffix_extreme(labels.iter().copied(), true))
    } else {
        None
    };
    let suffix_max = if need_extrema && suffix_window && need_max {
        Some(suffix_extreme(labels.iter().copied(), false))
    } else {
        None
    };

    for (row, (&start, &end)) in windows.starts.iter().zip(windows.ends.iter()).enumerate() {
        if keep == Keep::All {
            // Physical right-array order is the required order for `all`.
            // Emit the half-open physical range [start, end).
            for &label in &labels[start..end] {
                output_left.push(windows.left_index[row]);
                output_right.push(label);
            }
            continue;
        }
        let selected = match keep {
            // Every window is non-empty, so `start` is a valid position.
            Keep::Any => labels[start],
            Keep::First => {
                if right_index_is_ordered {
                    // When labels are monotonically increasing, the first
                    // physical label is also the smallest original label.
                    labels[start]
                } else if suffix_window {
                    // A suffix starts at `start`; its minimum is stored at
                    // that start position in the suffix table.
                    suffix_min.as_ref().unwrap()[start]
                } else {
                    // A prefix ends at the exclusive `end`; its minimum is
                    // therefore stored at `end - 1` in the prefix table.
                    prefix_min.as_ref().unwrap()[end - 1]
                }
            }
            Keep::Last => {
                if right_index_is_ordered {
                    // With ordered labels, the physical last label is the
                    // largest original label in the window.
                    labels[end - 1]
                } else if suffix_window {
                    // The suffix table answers the maximum at its start.
                    suffix_max.as_ref().unwrap()[start]
                } else {
                    // The prefix table answers the maximum at its final
                    // included position.
                    prefix_max.as_ref().unwrap()[end - 1]
                }
            }
            Keep::All => unreachable!(),
        };
        output_left.push(windows.left_index[row]);
        output_right.push(selected);
    }
    Ok((output_left, output_right))
}

/// Convert Python-facing physical positions from `int64` to Rust indexing
/// positions.
///
/// Pyjanitor supplies these positions as signed `int64` arrays because that is
/// the stable NumPy/PyO3 boundary type. Rust indexing requires `usize`, but a
/// negative value is invalid rather than a sentinel that can be cast safely.
/// Validate the value first and report its input offset in the error.
///
/// # Arguments
///
/// * `name` - Human-readable name used in validation errors.
/// * `values` - Physical positions in logical filtered-array order.
///
/// # Returns
///
/// The same positions as `usize`, preserving input order.
///
/// # Errors
///
/// Returns an error if a position is negative or cannot be represented by the
/// platform's `usize` type.
pub(crate) fn physical_positions(
    name: &str,
    values: ArrayView1<'_, i64>,
) -> Result<Vec<usize>, String> {
    values
        .iter()
        .enumerate()
        .map(|(offset, &position)| {
            usize::try_from(position).map_err(|_| {
                format!("{name} position at offset {offset} must be a non-negative int64")
            })
        })
        .collect()
}

/// Validate that non-null and null positions form a complete partition of an
/// original index array.
pub(crate) fn validate_position_partition(
    index_name: &str,
    full_len: usize,
    non_null_positions: &[usize],
    null_positions: &[usize],
) -> Result<(), String> {
    let expected = non_null_positions
        .len()
        .checked_add(null_positions.len())
        .ok_or("single join position count exceeds platform capacity")?;
    if expected != full_len {
        return Err(format!(
            "{index_name} length must equal the number of non-null values plus null positions"
        ));
    }
    let mut seen = vec![false; full_len];
    for (&position, kind) in non_null_positions
        .iter()
        .zip(std::iter::repeat("non-null"))
        .chain(null_positions.iter().zip(std::iter::repeat("null")))
    {
        if position >= full_len {
            return Err(format!(
                "{index_name} {kind} position {position} is out of bounds"
            ));
        }
        if seen[position] {
            return Err(format!(
                "{index_name} position {position} appears more than once"
            ));
        }
        seen[position] = true;
    }
    Ok(())
}

/// Visit every physical pair satisfying a null-aware `!=` comparison.
///
/// This is the aggregation counterpart to [`build_not_equal_positions_core`].
/// It follows the same strict-prefix, strict-suffix, and null semantics but
/// calls `visit` immediately instead of allocating output position vectors.
///
/// # Arguments
///
/// * `left` / `right` - Filtered non-null values; `right` must be sorted.
/// * `left_full_len` / `right_full_len` - Lengths of the original physical
///   layouts used by the aggregation source/output arrays.
/// * `left_positions` / `right_positions` - Maps from filtered offsets to
///   original physical positions.
/// * `left_null_positions` / `right_null_positions` - Original positions of
///   null rows, when present.
/// * `is_extension_array` - Whether null comparisons should be treated as
///   pandas `NA` and therefore excluded.
/// * `visit` - Callback receiving `(left_physical_position,
///   right_physical_position)` for each successful `!=` candidate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn visit_not_equal_pairs_core<T, F>(
    left: ArrayView1<'_, T>,
    left_full_len: usize,
    left_positions: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_full_len: usize,
    right_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    is_extension_array: bool,
    mut visit: F,
) -> Result<(), String>
where
    T: PartialOrd + Copy,
    F: FnMut(usize, usize),
{
    ensure_equal_lengths_core(
        "left values",
        left.len(),
        "left positions",
        left_positions.len(),
    )?;
    ensure_equal_lengths_core(
        "right values",
        right.len(),
        "right positions",
        right_positions.len(),
    )?;
    let left_positions = physical_positions("left", left_positions)?;
    let right_positions = physical_positions("right", right_positions)?;
    let left_null_positions = left_null_positions
        .map(|values| physical_positions("left null", values))
        .transpose()?;
    let right_null_positions = right_null_positions
        .map(|values| physical_positions("right null", values))
        .transpose()?;
    let empty = Vec::new();
    let left_null_positions = left_null_positions.as_deref().unwrap_or(&empty);
    let right_null_positions = right_null_positions.as_deref().unwrap_or(&empty);
    validate_position_partition(
        "left index",
        left_full_len,
        &left_positions,
        left_null_positions,
    )?;
    validate_position_partition(
        "right index",
        right_full_len,
        &right_positions,
        right_null_positions,
    )?;

    if is_extension_array {
        // Extension nulls compare as NA, which is false when used as a join
        // filter. The non-null candidate regions remain valid.
        for (left_offset, left_value) in left.iter().enumerate() {
            let left_position = left_positions[left_offset];
            let gt_start = partition_point(right, |value| value <= *left_value);
            let lt_end = partition_point(right, |value| value < *left_value);
            for &right_position in &right_positions[..lt_end] {
                visit(left_position, right_position);
            }
            for &right_position in &right_positions[gt_start..] {
                visit(left_position, right_position);
            }
        }
        return Ok(());
    }

    for (left_offset, left_value) in left.iter().enumerate() {
        let left_position = left_positions[left_offset];
        let gt_start = partition_point(right, |value| value <= *left_value);
        let lt_end = partition_point(right, |value| value < *left_value);
        for &right_position in &right_positions[..lt_end] {
            visit(left_position, right_position);
        }
        for &right_position in &right_positions[gt_start..] {
            visit(left_position, right_position);
        }
        for &right_position in right_null_positions {
            visit(left_position, right_position);
        }
    }
    for &left_position in left_null_positions {
        for &right_position in &right_positions {
            visit(left_position, right_position);
        }
        for &right_position in right_null_positions {
            visit(left_position, right_position);
        }
    }
    Ok(())
}

/// Compute a checked output-capacity bound for the `!=` kernel.
///
/// `all` can emit several pairs for one left row, so its capacity is computed
/// from the two strict binary-search regions plus the right-null labels.
#[allow(dead_code)]
fn not_equal_output_capacity<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    left_null_count: usize,
    right_null_count: usize,
    is_extension_array: bool,
) -> Result<usize, &'static str> {
    const CAPACITY_ERROR: &str = "single join result size exceeds platform capacity";

    // Null comparisons are excluded for extension arrays. Therefore, if one
    // filtered side is empty, there cannot be any `!=` output in that mode.
    if is_extension_array && (left.is_empty() || right.is_empty()) {
        return Ok(0);
    }

    // NumPy nulls compare unequal to every value, including other nulls. If
    // a filtered side is empty, all remaining pairs come from the null rows,
    // so the exact capacity can be calculated without any binary searches.
    let right_count = right
        .len()
        .checked_add(right_null_count)
        .ok_or(CAPACITY_ERROR)?;
    if left.is_empty() {
        return left_null_count
            .checked_mul(right_count)
            .ok_or(CAPACITY_ERROR);
    }
    if right.is_empty() {
        let left_count = left
            .len()
            .checked_add(left_null_count)
            .ok_or(CAPACITY_ERROR)?;
        return left_count
            .checked_mul(right_null_count)
            .ok_or(CAPACITY_ERROR);
    }

    let right_row_width = if is_extension_array {
        right.len()
    } else {
        right
            .len()
            .checked_add(right_null_count)
            .ok_or(CAPACITY_ERROR)?
    };
    let mut capacity = 0_usize;
    for left_value in left {
        // The right values are sorted. `gt_start` is the first position whose
        // value is strictly greater than the current left value, so
        // `right[gt_start..]` is the strict `>` suffix. `lt_end` is the first
        // position whose value is greater than or equal to the left value, so
        // `right[..lt_end]` is the strict `<` prefix. Equal values are outside
        // both ranges, as required by `!=`.
        //
        // Example: right = [1, 3, 5, 7], left = 5 gives
        // `lt_end = 2` (`[1, 3]`) and `gt_start = 3` (`[7]`).
        let gt_start = partition_point(right, |value| value <= *left_value);
        let lt_end = partition_point(right, |value| value < *left_value);
        let strict_width = lt_end
            .checked_add(right.len() - gt_start)
            .ok_or(CAPACITY_ERROR)?;
        let row_width = if is_extension_array {
            strict_width
        } else {
            strict_width
                .checked_add(right_null_count)
                .ok_or(CAPACITY_ERROR)?
        };
        capacity = capacity.checked_add(row_width).ok_or(CAPACITY_ERROR)?;
    }
    let null_left_capacity = if is_extension_array {
        0
    } else {
        left_null_count
            .checked_mul(right_row_width)
            .ok_or(CAPACITY_ERROR)?
    };
    capacity
        .checked_add(null_left_capacity)
        .ok_or(CAPACITY_ERROR)
}

/// Build physical-position pairs for `!=` from filtered non-null arrays and
/// optional null-position metadata.
///
/// The filtered value arrays are aligned with their position maps. A filtered
/// offset is first mapped to an original physical position, and the separate
/// materializer then maps that position through the full index array to get
/// the public label. Null positions are already original physical positions.
///
/// The right values must be sorted in ascending order. Rust does not sort or
/// infer nullness. `right_index_is_ordered` describes the labels obtained from
/// `right_index[right_positions]`; it is used only to optimize `first` and
/// `last` selection. `is_extension_array` means both operands have the same
/// pandas extension dtype, which pyjanitor validates upstream.
///
/// In NumPy mode, nulls participate in `!=` and match every opposite-side
/// value, including another null. In pandas extension mode, every comparison
/// involving a null is excluded because it produces `NA` and filters false.
/// The function returns physical position pairs; it returns empty vectors when
/// no pairs match.
///
/// Each full index length must equal the number of non-null values plus the
/// number of null positions. Non-null and null positions must be disjoint and
/// cover the full physical index range.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn build_not_equal_positions_core<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    left_positions: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    right_positions: ArrayView1<'_, i64>,
    left_null_positions: Option<ArrayView1<'_, i64>>,
    right_null_positions: Option<ArrayView1<'_, i64>>,
    right_index_is_ordered: bool,
    is_extension_array: bool,
    keep: Keep,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    // `left` and `right` contain only non-null values. Their companion
    // position arrays tell us where those filtered values came from in the
    // original, full arrays. For example, a filtered value at offset `1`
    // might have original physical position `4`.
    ensure_equal_lengths_core(
        "left values",
        left.len(),
        "left positions",
        left_positions.len(),
    )?;
    ensure_equal_lengths_core(
        "right values",
        right.len(),
        "right positions",
        right_positions.len(),
    )?;

    // Convert the Python-facing int64 positions into Rust `usize` positions
    // once. These positions are used for indexing the full index arrays and
    // are also what this core returns to the wrapper.
    let left_positions = physical_positions("left", left_positions)?;
    let right_positions = physical_positions("right", right_positions)?;

    // Null positions are optional because pyjanitor passes `None` when that
    // side contains no nulls. `map` converts each supplied array, while
    // `transpose` changes `Option<Result<...>>` into `Result<Option<...>>`,
    // allowing a malformed negative/out-of-range position to return an error.
    let left_null_positions = left_null_positions
        .map(|positions| physical_positions("left null", positions))
        .transpose()?;
    let right_null_positions = right_null_positions
        .map(|positions| physical_positions("right null", positions))
        .transpose()?;
    let empty = Vec::new();
    let left_null_positions = left_null_positions.as_deref().unwrap_or(&empty);
    let right_null_positions = right_null_positions.as_deref().unwrap_or(&empty);

    // The filtered non-null positions and the null positions must together
    // describe every physical row exactly once. This catches missing,
    // duplicated, and out-of-range positions before any output is built.
    validate_position_partition(
        "left index",
        left_index.len(),
        &left_positions,
        left_null_positions,
    )?;
    validate_position_partition(
        "right index",
        right_index.len(),
        &right_positions,
        right_null_positions,
    )?;

    // There is nothing to match when a side has no non-null values and no
    // null positions either. In particular, this avoids invoking binary
    // search for an actually empty input.
    if (left.is_empty() && left_null_positions.is_empty())
        || (right.is_empty() && right_null_positions.is_empty())
    {
        return Ok((Vec::new(), Vec::new()));
    }

    // `all` can emit many pairs per left row, so it gets an exact checked
    // capacity estimate. The selected modes emit at most one pair per left
    // row, so the full left-index length is a sufficient upper bound.
    let output_capacity = if keep == Keep::All {
        not_equal_output_capacity(
            left,
            right,
            left_null_positions.len(),
            right_null_positions.len(),
            is_extension_array,
        )
        .map_err(str::to_owned)?
    } else {
        // Every selected mode emits at most one pair per left row. The actual
        // result may be shorter when a left row has no unequal match, but the
        // full left-index length is a simple safe upper bound.
        left_index.len()
    };
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(output_capacity)
        .map_err(|_| "single join result allocation failed".to_owned())?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(output_capacity)
        .map_err(|_| "single join result allocation failed".to_owned())?;

    // If the right labels are not ordered, selecting the smallest (`first`)
    // or largest (`last`) label from a prefix/suffix would otherwise require
    // scanning that region for every left row. These tables store the best
    // physical position seen so far, so each window can select in O(1).
    let need_unordered_extrema =
        !left.is_empty() && !right_index_is_ordered && matches!(keep, Keep::First | Keep::Last);
    // Only unordered `first`/`last` selection needs labels in filtered value
    // order. Ordered labels, `any`, and `all` can use boundaries directly, so
    // avoid copying the right-label vector in those cases.
    let right_labels = need_unordered_extrema.then(|| {
        right_positions
            .iter()
            .map(|&position| right_index[position])
            .collect::<Vec<_>>()
    });
    let prefix_min = if need_unordered_extrema && keep == Keep::First {
        Some(prefix_min_positions(right_labels.as_deref().unwrap()))
    } else {
        None
    };
    let prefix_max = if need_unordered_extrema && keep == Keep::Last {
        Some(prefix_max_positions(right_labels.as_deref().unwrap()))
    } else {
        None
    };
    let suffix_min = if need_unordered_extrema && keep == Keep::First {
        Some(suffix_min_positions(right_labels.as_deref().unwrap()))
    } else {
        None
    };
    let suffix_max = if need_unordered_extrema && keep == Keep::Last {
        Some(suffix_max_positions(right_labels.as_deref().unwrap()))
    } else {
        None
    };

    // Nulls are candidates in NumPy mode, but not in extension-array mode.
    // For first/last, precompute the best null label once instead of scanning
    // every right-null position for every left row.
    let right_null_extreme = match keep {
        Keep::First if !is_extension_array => right_null_positions
            .iter()
            .copied()
            .min_by_key(|&position| right_index[position]),
        Keep::Last if !is_extension_array => right_null_positions
            .iter()
            .copied()
            .max_by_key(|&position| right_index[position]),
        _ => None,
    };

    // Process each filtered non-null left value. `left_offset` indexes the
    // filtered value/position arrays; `left_position` is the corresponding
    // physical position in the original left array.
    for (left_offset, left_value) in left.iter().enumerate() {
        let left_position = left_positions[left_offset];

        // The sorted right values are divided into three regions:
        //   right[..lt_end]   contains values strictly less than the left;
        //   right[lt_end..gt_start] contains values equal to the left;
        //   right[gt_start..] contains values strictly greater than the left.
        // `!=` emits only the first and third regions.
        let gt_start = partition_point(right, |value| value <= *left_value);
        let lt_end = partition_point(right, |value| value < *left_value);
        if keep == Keep::All {
            // Materialize every strict-less and strict-greater pair. Equal
            // values are deliberately skipped, and NumPy nulls are appended
            // because they compare unequal to this non-null value.
            for &right_position in &right_positions[..lt_end] {
                output_left.push(left_position);
                output_right.push(right_position);
            }
            for &right_position in &right_positions[gt_start..] {
                output_left.push(left_position);
                output_right.push(right_position);
            }
            if !is_extension_array {
                for &right_position in right_null_positions {
                    output_left.push(left_position);
                    output_right.push(right_position);
                }
            }
            continue;
        }
        if keep == Keep::Any {
            // Any match is sufficient. Prefer the first available strict
            // region, then a right-null candidate in NumPy mode. This avoids
            // scanning or building all matching pairs.
            let candidate = if lt_end > 0 {
                right_positions.first().copied()
            } else if gt_start < right.len() {
                Some(right_positions[gt_start])
            } else if !is_extension_array {
                right_null_positions.first().copied()
            } else {
                None
            };
            if let Some(right_position) = candidate {
                output_left.push(left_position);
                output_right.push(right_position);
            }
            continue;
        }

        // For `first` and `last`, combine at most one candidate from each
        // possible source: the strict-less prefix, the strict-greater suffix,
        // and the right-null positions. `minimum` tells the closure whether
        // the smallest or largest public right-index label wins.
        let minimum = keep == Keep::First;
        let mut candidate = None;
        let mut combine = |position: usize| {
            candidate = Some(candidate.map_or(position, |current| {
                let value = right_index[position];
                let current_value = right_index[current];
                if (minimum && value < current_value) || (!minimum && value > current_value) {
                    position
                } else {
                    current
                }
            }));
        };
        if lt_end > 0 {
            // The prefix is non-empty. If labels are ordered, its boundary
            // position is enough; otherwise use the precomputed prefix table.
            combine(if right_index_is_ordered {
                if minimum {
                    right_positions[0]
                } else {
                    right_positions[lt_end - 1]
                }
            } else if minimum {
                right_positions[prefix_min.as_ref().unwrap()[lt_end - 1]]
            } else {
                right_positions[prefix_max.as_ref().unwrap()[lt_end - 1]]
            });
        }
        if gt_start < right.len() {
            // The suffix is non-empty. As above, use its boundary directly
            // for ordered labels and its extrema table otherwise.
            combine(if right_index_is_ordered {
                if minimum {
                    right_positions[gt_start]
                } else {
                    *right_positions.last().unwrap()
                }
            } else if minimum {
                right_positions[suffix_min.as_ref().unwrap()[gt_start]]
            } else {
                right_positions[suffix_max.as_ref().unwrap()[gt_start]]
            });
        }
        if let Some(position) = right_null_extreme {
            combine(position);
        }
        if let Some(right_position) = candidate {
            output_left.push(left_position);
            output_right.push(right_position);
        }
    }

    // A null left value is not present in the filtered `left` loop above, so
    // handle it separately. In NumPy mode it compares unequal to every right
    // value, including right nulls. In extension mode it compares as `NA`,
    // which is false in a filtering operation, so it contributes no pairs.
    let nonnull_extreme = match keep {
        Keep::First => right_positions
            .iter()
            .copied()
            .min_by_key(|&position| right_index[position]),
        Keep::Last => right_positions
            .iter()
            .copied()
            .max_by_key(|&position| right_index[position]),
        _ => None,
    };
    for &left_position in left_null_positions {
        if is_extension_array {
            continue;
        }
        if keep == Keep::All {
            // Every right physical position is a valid `!=` partner for a
            // NumPy null left value.
            for &right_position in &right_positions {
                output_left.push(left_position);
                output_right.push(right_position);
            }
            for &right_position in right_null_positions {
                output_left.push(left_position);
                output_right.push(right_position);
            }
        } else if keep == Keep::Any {
            // Any right value is enough, so choose the first available right
            // non-null position, falling back to a right-null position.
            if let Some(right_position) = right_positions
                .first()
                .or_else(|| right_null_positions.first())
                .copied()
            {
                output_left.push(left_position);
                output_right.push(right_position);
            }
        } else {
            // For first/last, compare the best non-null and null candidates
            // by their public right-index labels, then emit the winner.
            let candidate = match (nonnull_extreme, right_null_extreme) {
                (Some(nonnull), Some(null)) => Some(if keep == Keep::First {
                    if right_index[nonnull] < right_index[null] {
                        nonnull
                    } else {
                        null
                    }
                } else if right_index[nonnull] > right_index[null] {
                    nonnull
                } else {
                    null
                }),
                (Some(nonnull), None) => Some(nonnull),
                (None, Some(null)) => Some(null),
                (None, None) => None,
            };
            if let Some(right_position) = candidate {
                output_left.push(left_position);
                output_right.push(right_position);
            }
        }
    }
    Ok((output_left, output_right))
}

/// Convert physical position pairs into public index-label pairs.
pub(crate) fn materialize_index_pairs(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    left_positions: Vec<usize>,
    right_positions: Vec<usize>,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_equal_lengths_core(
        "materialized left positions",
        left_positions.len(),
        "materialized right positions",
        right_positions.len(),
    )?;
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(left_positions.len())
        .map_err(|_| "single join result allocation failed".to_owned())?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(right_positions.len())
        .map_err(|_| "single join result allocation failed".to_owned())?;
    for (&left_position, &right_position) in left_positions.iter().zip(&right_positions) {
        output_left.push(
            *left_index
                .get(left_position)
                .ok_or("left position is out of bounds")?,
        );
        output_right.push(
            *right_index
                .get(right_position)
                .ok_or("right position is out of bounds")?,
        );
    }
    Ok((output_left, output_right))
}

/// Select the effective output mode for a building-block request.
///
/// Range building blocks are positional windows, while `!=` has multiple
/// disjoint regions and therefore returns fully materialized pairs. In both
/// cases a caller-provided `keep` mode has no effect; `!=` uses `All` to make
/// that materialization explicit.
pub(crate) fn effective_keep(keep: Keep, return_building_blocks: bool) -> Keep {
    if return_building_blocks {
        Keep::All
    } else {
        keep
    }
}

macro_rules! single_join_function {
    ($name:ident, $type:ty) => {
        #[allow(clippy::too_many_arguments)]
        /// Construct indices for one conditional join predicate.
        ///
        /// `left` and `right` are aligned with their original `int64` index
        /// arrays. The right values must already be sorted; Rust trusts the
        /// caller and does not sort them. `comparator` accepts `>`, `>=`, `<`,
        /// `<=`, `==`, or `!=`; equality is rejected because pyjanitor handles
        /// equi-joins upstream.
        ///
        /// `keep` accepts `first`, `last`, `any`, or `all`. `first` and `last`
        /// select by original right index label, while `all` emits every
        /// physical matching right row. `return_building_blocks` affects only
        /// range operators: it returns the retained left labels, the complete
        /// right-label array, and one half-open `starts`/`ends` window per
        /// retained left row. For `!=`, the flag is ignored because `!=`
        /// always returns materialized pairs selected by `keep`.
        ///
        /// For `!=`, the value arrays are filtered non-null arrays. Their
        /// position arrays map filtered offsets to physical positions in the
        /// full index arrays. Optional null-position arrays contain physical
        /// positions in those full arrays. `is_extension_array` means both
        /// operands use the same pandas extension dtype; pyjanitor validates
        /// that invariant before calling Rust.
        ///
        /// # Arguments
        ///
        /// * `left`, `left_index` - Left values and the full original labels.
        ///   For `!=`, `left` is filtered non-null data and `left_index` is
        ///   addressed through `left_positions`.
        /// * `right`, `right_index` - Right values and the full original
        ///   labels. For `!=`, `right` is filtered, already value-sorted data
        ///   and `right_index` is addressed through `right_positions`.
        /// * `right_index_is_ordered` - Whether right labels are monotonically
        ///   increasing in the sorted-right layout.
        /// * `comparator` - One of `>`, `>=`, `<`, `<=`, `==`, or `!=`.
        /// * `keep` - One of `first`, `last`, `any`, or `all`.
        /// * `return_building_blocks` - Applies only to range operators;
        ///   ignored for `!=`.
        /// * `left_positions`, `right_positions` - Filtered-to-original
        ///   physical position maps used only for `!=`.
        /// * `left_null_positions`, `right_null_positions` - Optional null
        ///   physical positions used only for `!=`.
        /// * `is_extension_array` - Whether both operands use the same pandas
        ///   extension dtype.
        ///
        /// # Returns
        ///
        /// Returns `None` when no pair matches. Otherwise returns a dictionary
        /// containing `left_index` and `right_index`; range building-block
        /// requests additionally contain `starts` and `ends`.
        pub(crate) fn $name<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right: PyReadonlyArray1<'py, $type>,
            right_index: PyReadonlyArray1<'py, i64>,
            right_index_is_ordered: bool,
            comparator: &str,
            keep: &str,
            return_building_blocks: bool,
            left_positions: Option<PyReadonlyArray1<'py, i64>>,
            left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_positions: Option<PyReadonlyArray1<'py, i64>>,
            right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
            is_extension_array: bool,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            let op = CompareOp::try_from_str(comparator)?;
            let requested_keep = Keep::parse(keep)?;
            let keep = effective_keep(requested_keep, return_building_blocks);
            if op == CompareOp::Eq {
                return Err(PyValueError::new_err(
                    "single join does not compute equality; handle == upstream",
                ));
            }
            let left = left.as_array();
            let right = right.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            if op == CompareOp::Ne {
                let left_positions = left_positions
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("left positions are required for !="))?;
                let right_positions = right_positions
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("right positions are required for !="))?;
                let (left_positions, right_positions) = build_positions(
                    left,
                    left_index,
                    left_positions.as_array(),
                    right,
                    right_index,
                    right_positions.as_array(),
                    left_null_positions.as_ref().map(|value| value.as_array()),
                    right_null_positions.as_ref().map(|value| value.as_array()),
                    is_extension_array,
                    keep,
                )
                .map_err(PyValueError::new_err)?;
                let (out_left, out_right) = materialize_index_pairs(
                    left_index,
                    right_index,
                    left_positions,
                    right_positions,
                )
                .map_err(PyValueError::new_err)?;
                if out_left.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(result_dict(py, out_left, out_right, None, None)?));
            }
            if left_positions.is_some()
                || left_null_positions.is_some()
                || right_positions.is_some()
                || right_null_positions.is_some()
                || is_extension_array
            {
                return Err(PyValueError::new_err(
                    "position metadata is only supported for !=",
                ));
            }
            let windows = build_range_core(
                left,
                left_index,
                right,
                right_index,
                right_index_is_ordered,
                op,
            )
            .map_err(PyValueError::new_err)?;
            if return_building_blocks {
                if windows.left_index.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(result_dict(
                    py,
                    windows.left_index,
                    windows.right_index,
                    Some(windows.starts),
                    Some(windows.ends),
                )?));
            }
            let suffix_window = matches!(op, CompareOp::Lt | CompareOp::Le);
            let (out_left, out_right) =
                choose_range(&windows, keep, right_index_is_ordered, suffix_window)
                    .map_err(PyValueError::new_err)?;
            if out_left.is_empty() {
                return Ok(None);
            }
            Ok(Some(result_dict(py, out_left, out_right, None, None)?))
        }
    };
}

single_join_function!(single_join_indices_int64, i64);
single_join_function!(single_join_indices_int32, i32);
single_join_function!(single_join_indices_int16, i16);
single_join_function!(single_join_indices_int8, i8);
single_join_function!(single_join_indices_uint64, u64);
single_join_function!(single_join_indices_uint32, u32);
single_join_function!(single_join_indices_uint16, u16);
single_join_function!(single_join_indices_uint8, u8);
single_join_function!(single_join_indices_f64, f64);
single_join_function!(single_join_indices_f32, f32);

/// Execute an all-`!=` extended join.
///
/// The first predicate supplies filtered non-null values plus physical
/// position maps and optional null positions. It is expanded with `Keep::All`
/// because later predicates must see every first-stage candidate. Residual
/// predicates use full-layout arrays and masks, so their physical positions
/// can be indexed directly. Public labels are materialized only after all
/// residual predicates pass.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to build the result dict.
/// * `predicates` - The complete predicate list. Predicate zero is the
///   null-aware `!=` anchor; every later predicate must also use `!=`.
/// * `keep` - Retention mode applied after all residual predicates pass.
/// * `first_left` / `first_right` - Filtered non-null values for the anchor.
/// * `first_left_index` / `first_right_index` - Complete physical index-label
///   arrays used for residual indexing and final output labels.
/// * `first_left_positions` / `first_right_positions` - Physical positions of
///   the filtered non-null anchor values in the complete layouts.
/// * `first_left_null_positions` / `first_right_null_positions` - Optional
///   physical positions of null rows. `None` means that side has no null rows.
/// * `right_index_is_ordered` - Shared tuple metadata describing right-label
///   order. The first stage uses `Keep::All`, so this does not affect candidate
///   generation; it is still passed through the common `!=` contract.
/// * `is_extension_array` - Selects pandas extension-array null behavior. For
///   NumPy semantics nulls can compare unequal; for pandas extension semantics
///   a null comparison is treated as missing and does not survive filtering.
///
/// # Returns
///
/// Returns a dictionary containing materialized `left_index` and
/// `right_index` arrays, or `None` when the first candidate stream or the
/// residual filters produce no pairs.
///
/// # Errors
///
/// Returns a Python `ValueError` for an invalid keep value, malformed residual
/// tuples, a residual operator other than `!=`, mismatched physical lengths,
/// invalid null metadata, or invalid physical positions.
#[allow(clippy::too_many_arguments)]
fn extended_not_equal_join<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first_left: PyReadonlyArray1<'py, T>,
    first_left_index: PyReadonlyArray1<'py, i64>,
    first_left_positions: PyReadonlyArray1<'py, i64>,
    first_left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    first_right: PyReadonlyArray1<'py, T>,
    first_right_index: PyReadonlyArray1<'py, i64>,
    first_right_positions: PyReadonlyArray1<'py, i64>,
    first_right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    _right_index_is_ordered: bool,
    is_extension_array: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    crate::not_equals_only::materialize_with_residuals(
        py,
        predicates,
        keep,
        first_left,
        first_left_index,
        first_left_positions,
        first_left_null_positions,
        first_right,
        first_right_index,
        first_right_positions,
        first_right_null_positions,
        is_extension_array,
    )
}

#[allow(clippy::too_many_arguments)]
/// Execute a single-anchor range-led extended join.
///
/// The first predicate supplies one sorted-right binary-search window for
/// each left row. The remaining predicates are parsed in their original user
/// order and evaluated as residual filters inside those windows. A later
/// range predicate is still only a residual here; intersecting two range
/// windows is the responsibility of `range_join.rs`.
///
/// # Arguments
///
/// * `py` - Active Python interpreter token used to parse residual tuples and
///   construct the returned Python dictionary.
/// * `predicates` - At least two aligned tuples. The first tuple is the range
///   anchor; every later tuple is a residual comparison.
/// * `keep` - Selection mode applied only after all residual predicates pass.
/// * `first_left` / `first_left_index` - Null-free left anchor values and
///   labels in the same logical order.
/// * `first_right` / `first_right_index` - Null-free, ascending right anchor
///   values and their aligned labels.
/// * `first_op` - The first anchor comparator: `<`, `<=`, `>`, or `>=`.
///
/// # Returns
///
/// Returns materialized public left/right labels, or `None` when no complete
/// predicate match survives.
///
/// # Errors
///
/// Returns a Python `ValueError` for an invalid predicate count, comparator,
/// residual shape, length mismatch, null metadata, or keep value.
fn extended_join<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first_left: PyReadonlyArray1<'py, T>,
    first_left_index: PyReadonlyArray1<'py, i64>,
    first_right: PyReadonlyArray1<'py, T>,
    first_right_index: PyReadonlyArray1<'py, i64>,
    first_op: CompareOp,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // A single-anchor extended join needs at least one residual predicate.
    // Dual-range joins are routed separately through `range_join.rs`.
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended join requires at least two predicates",
        ));
    }
    // `!=` requires the null-aware physical-position tuple. It must not be
    // interpreted as an ordinary range anchor with a missing null contract.
    if first_op == CompareOp::Ne {
        return Err(PyValueError::new_err(
            "all-!= joins must use the null-aware first-predicate form",
        ));
    }
    if !matches!(
        first_op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return Err(PyValueError::new_err(
            "single extended join requires a range predicate first",
        ));
    }
    let keep = Keep::parse(keep)?;

    // Copy only the Python tuple references for predicates after the anchor.
    // The shared parser then borrows their aligned arrays and authoritative
    // null metadata for the residual loop.
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    let left = first_left.as_array();
    let right = first_right.as_array();
    let left_index = first_left_index.as_array();
    let right_index = first_right_index.as_array();

    // Candidate positions from the first range are physical positions in the
    // anchor arrays. Every residual must therefore have exactly the same
    // physical left/right shape before any candidate is visited.
    check_predicate_lengths(&parsed, left.len(), right.len())?;
    // The single-extended path has exactly one binary-search anchor. Every
    // later predicate, including another range comparison, is evaluated as a
    // residual filter inside that anchor's candidate window. Dual-range
    // window intersection belongs exclusively to `range_join.rs`.
    let windows = build_range_core(left, left_index, right, right_index, false, first_op)
        .map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        // No anchor window means there are no candidates for the residual
        // predicates, so the public join result is `None`.
        return Ok(None);
    }
    // The window contains only candidates for the first predicate. The
    // materializer checks every residual in user order and applies `keep`
    // only to the survivors.
    let (out_left, out_right) =
        materialize_range_candidates(&windows, &parsed, metadata.as_deref(), keep)
            .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        // An anchor match is not enough: all candidates may have failed a
        // residual predicate.
        return Ok(None);
    }
    // The materializer has already translated surviving physical positions
    // into the original public index labels.
    Ok(Some(result_dict(py, out_left, out_right, None, None)?))
}

/// Named range anchor for the single-anchor extended index path.
struct ParsedExtendedRangeAnchor<'py, T: numpy::Element> {
    left: PyReadonlyArray1<'py, T>,
    left_index: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    op: CompareOp,
}

/// Named null-aware `!=` anchor for the single-anchor extended index path.
struct ParsedExtendedNotEqualAnchor<'py, T: numpy::Element> {
    left: PyReadonlyArray1<'py, T>,
    left_index: PyReadonlyArray1<'py, i64>,
    left_positions: PyReadonlyArray1<'py, i64>,
    left_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right: PyReadonlyArray1<'py, T>,
    right_index: PyReadonlyArray1<'py, i64>,
    right_positions: PyReadonlyArray1<'py, i64>,
    right_null_positions: Option<PyReadonlyArray1<'py, i64>>,
    right_index_is_ordered: bool,
    is_extension_array: bool,
}

enum ParsedExtendedAnchor<'py, T: numpy::Element> {
    Range(ParsedExtendedRangeAnchor<'py, T>),
    NotEqual(ParsedExtendedNotEqualAnchor<'py, T>),
}

/// Parse the first anchor for the single-anchor extended index API.
///
/// The accepted six-field range and eleven-field null-aware `!=` layouts are
/// kept compatible with the Python API, but the wrapper passes named fields to
/// the candidate builders after this boundary.
fn parse_extended_anchor<'py, T: numpy::Element + PartialOrd + Copy>(
    first: &Bound<'py, PyTuple>,
) -> PyResult<ParsedExtendedAnchor<'py, T>> {
    match first.len() {
        6 => {
            let op = CompareOp::try_from_str(first.get_item(5)?.extract::<&str>()?)?;
            first.get_item(4)?.extract::<bool>()?;
            if !matches!(
                op,
                CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
            ) {
                // This parser handles the range form only.  The separate
                // null-aware `!=` form has an eleven-field layout and is
                // parsed by the branch below; keep this message specific so
                // callers can distinguish an unsupported range operator from
                // a malformed `!=` payload.
                return Err(PyValueError::new_err(
                    "the first range predicate must use <, <=, >, or >=",
                ));
            }
            Ok(ParsedExtendedAnchor::Range(ParsedExtendedRangeAnchor {
                left: first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
                left_index: first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                right: first.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?,
                right_index: first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                op,
            }))
        }
        11 => {
            let op = CompareOp::try_from_str(first.get_item(10)?.extract::<&str>()?)?;
            if op != CompareOp::Ne {
                return Err(PyValueError::new_err(
                    "the first eleven-element predicate must use !=",
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
            Ok(ParsedExtendedAnchor::NotEqual(
                ParsedExtendedNotEqualAnchor {
                    left: first.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?,
                    left_index: first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    left_positions: first.get_item(2)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    left_null_positions,
                    right: first.get_item(4)?.extract::<PyReadonlyArray1<'py, T>>()?,
                    right_index: first.get_item(5)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    right_positions: first.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    right_null_positions,
                    right_index_is_ordered: first.get_item(8)?.extract::<bool>()?,
                    is_extension_array: first.get_item(9)?.extract::<bool>()?,
                },
            ))
        }
        _ => Err(PyValueError::new_err(
            "the first extended predicate must contain 6 or 11 elements",
        )),
    }
}

macro_rules! extended_join_function {
    ($name:ident, $type:ty, $export:literal) => {
        #[pyfunction(name = $export)]
        /// Build flat indices for a multi-predicate conditional join.
        ///
        /// Mixed joins use a six-element range first-predicate form:
        /// `(left, left_index, right, right_index,
        /// right_index_is_ordered, comparator)`. All-`!=` joins use an
        /// eleven-element first-predicate form containing filtered values,
        /// physical position maps, full indexes, null positions, ordering,
        /// extension-array semantics, and the comparator. Later items are
        /// ordinary three-element predicates or six-element null-aware `!=`
        /// predicates. `keep` is applied only after every predicate passes.
        /// PyJanitor must align all residual arrays to the same physical left
        /// and right positions before calling Rust. Rust does not sort or
        /// realign residual arrays.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - At least two aligned predicate tuples. The first
        ///   tuple establishes the candidate stream; later tuples are tested
        ///   against those candidate positions in user order.
        /// * `keep` - Retain `"first"`, `"last"`, `"any"`, or `"all"`
        ///   survivors for each left row.
        /// * Predicate one supplies the only binary-search candidate window;
        ///   predicates two onward are residual filters evaluated in user
        ///   order. Dual-range window intersection belongs to the separate
        ///   `range_join` API.
        ///
        /// # Returns
        ///
        /// Returns `None` when no complete predicate match survives;
        /// otherwise returns a dictionary with materialized `left_index` and
        /// `right_index` arrays. Building blocks are not returned by this
        /// extended API.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` for malformed tuple layouts, unsupported
        /// comparator combinations, mismatched aligned lengths, invalid null
        /// metadata, or an invalid `keep` value.
        pub fn $name<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            keep: &str,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            if predicates.len() < 2 {
                return Err(PyValueError::new_err(
                    "single extended join requires at least two predicates",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first = first_item.cast::<PyTuple>()?;
            match parse_extended_anchor::<$type>(first)? {
                ParsedExtendedAnchor::Range(anchor) => extended_join(
                    py,
                    predicates,
                    keep,
                    anchor.left,
                    anchor.left_index,
                    anchor.right,
                    anchor.right_index,
                    anchor.op,
                ),
                ParsedExtendedAnchor::NotEqual(anchor) => extended_not_equal_join(
                    py,
                    predicates,
                    keep,
                    anchor.left,
                    anchor.left_index,
                    anchor.left_positions,
                    anchor.left_null_positions,
                    anchor.right,
                    anchor.right_index,
                    anchor.right_positions,
                    anchor.right_null_positions,
                    anchor.right_index_is_ordered,
                    anchor.is_extension_array,
                ),
            }
        }
    };
}

extended_join_function!(
    single_join_extended_indices_int64,
    i64,
    "range_anchor_extended_indices_int64"
);
extended_join_function!(
    single_join_extended_indices_int32,
    i32,
    "range_anchor_extended_indices_int32"
);
extended_join_function!(
    single_join_extended_indices_int16,
    i16,
    "range_anchor_extended_indices_int16"
);
extended_join_function!(
    single_join_extended_indices_int8,
    i8,
    "range_anchor_extended_indices_int8"
);
extended_join_function!(
    single_join_extended_indices_uint64,
    u64,
    "range_anchor_extended_indices_uint64"
);
extended_join_function!(
    single_join_extended_indices_uint32,
    u32,
    "range_anchor_extended_indices_uint32"
);
extended_join_function!(
    single_join_extended_indices_uint16,
    u16,
    "range_anchor_extended_indices_uint16"
);
extended_join_function!(
    single_join_extended_indices_uint8,
    u8,
    "range_anchor_extended_indices_uint8"
);
extended_join_function!(
    single_join_extended_indices_f64,
    f64,
    "range_anchor_extended_indices_f64"
);
extended_join_function!(
    single_join_extended_indices_f32,
    f32,
    "range_anchor_extended_indices_f32"
);

/// Register the range-anchor extended index ABI used by PyJanitor's
/// multi-predicate range-first path.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_extended_indices_f32, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s};

    #[test]
    fn maps_sorted_right_values_to_original_positions() {
        let result = build_positions(
            array![4_i64].view(),
            array![0_i64].view(),
            array![0_i64].view(),
            array![1_i64, 3, 5].view(),
            array![0_i64, 1, 2].view(),
            array![2_i64, 0, 1].view(),
            None,
            None,
            false,
            Keep::All,
        )
        .unwrap();
        assert_eq!(result, (vec![0, 0, 0], vec![2, 0, 1]));
    }

    #[test]
    fn first_and_last_use_original_physical_positions() {
        let first = build_positions(
            array![4_i64].view(),
            array![0_i64].view(),
            array![0_i64].view(),
            array![1_i64, 3, 5].view(),
            array![0_i64, 1, 2].view(),
            array![2_i64, 0, 1].view(),
            None,
            None,
            false,
            Keep::First,
        )
        .unwrap();
        let last = build_positions(
            array![4_i64].view(),
            array![0_i64].view(),
            array![0_i64].view(),
            array![1_i64, 3, 5].view(),
            array![0_i64, 1, 2].view(),
            array![2_i64, 0, 1].view(),
            None,
            None,
            false,
            Keep::Last,
        )
        .unwrap();
        assert_eq!(first, (vec![0], vec![0]));
        assert_eq!(last, (vec![0], vec![2]));
    }

    #[test]
    fn null_partitions_are_disjoint_and_extension_nulls_do_not_match() {
        let result = build_positions(
            array![1_i64].view(),
            array![0_i64, 1].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![0_i64, 1].view(),
            array![0_i64].view(),
            Some(array![1_i64].view()),
            Some(array![1_i64].view()),
            true,
            Keep::All,
        )
        .unwrap();
        assert!(result.0.is_empty());
        assert!(result.1.is_empty());
    }

    #[test]
    fn contiguous_and_strided_boundaries_match() {
        let right = array![1_i64, 3, 5, 5, 9];
        let padded = array![1_i64, -1, 3, -1, 5, -1, 5, -1, 9, -1];
        let left = array![0_i64, 5, 10];
        let index = array![10_i64, 20, 30];
        for op in [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge] {
            let dense = build_range_core(
                left.view(),
                index.view(),
                right.view(),
                array![0, 1, 2, 3, 4].view(),
                true,
                op,
            )
            .unwrap();
            let strided = build_range_core(
                left.view(),
                index.view(),
                padded.slice(s![..;2]),
                array![0, 1, 2, 3, 4].view(),
                true,
                op,
            )
            .unwrap();
            assert_eq!(dense.starts, strided.starts);
            assert_eq!(dense.ends, strided.ends);
        }
    }

    #[test]
    fn unordered_labels_use_range_extrema() {
        let windows = build_range_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![1, 3, 5, 7].view(),
            array![40, 10, 30, 20].view(),
            false,
            CompareOp::Lt,
        )
        .unwrap();
        let result = choose_range(&windows, Keep::First, false, true).unwrap();
        assert_eq!(result, (vec![100], vec![20]));

        let windows = build_range_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![1, 3, 5, 7].view(),
            array![40, 10, 30, 20].view(),
            false,
            CompareOp::Gt,
        )
        .unwrap();
        assert_eq!(
            choose_range(&windows, Keep::First, false, false).unwrap(),
            (vec![100], vec![10])
        );
        assert_eq!(
            choose_range(&windows, Keep::Last, false, false).unwrap(),
            (vec![100], vec![40])
        );
    }

    #[test]
    fn range_windows_cover_all_comparators_and_selection_modes() {
        let left = array![4_i64];
        let left_index = array![100_i64];
        let right = array![1_i64, 3, 5, 7];
        let right_index = array![10_i64, 20, 30, 40];
        let cases = [
            (CompareOp::Lt, 2_usize, 4_usize),
            (CompareOp::Le, 2, 4),
            (CompareOp::Gt, 0, 2),
            (CompareOp::Ge, 0, 2),
        ];
        for (op, expected_start, expected_end) in cases {
            let windows = build_range_core(
                left.view(),
                left_index.view(),
                right.view(),
                right_index.view(),
                true,
                op,
            )
            .unwrap();
            assert_eq!(windows.starts, vec![expected_start]);
            assert_eq!(windows.ends, vec![expected_end]);
            assert_eq!(
                choose_range(
                    &windows,
                    Keep::Any,
                    true,
                    matches!(op, CompareOp::Lt | CompareOp::Le)
                )
                .unwrap(),
                (vec![100], vec![right_index[expected_start]])
            );
            assert_eq!(
                choose_range(
                    &windows,
                    Keep::First,
                    true,
                    matches!(op, CompareOp::Lt | CompareOp::Le),
                )
                .unwrap(),
                (vec![100], vec![right_index[expected_start]])
            );
            assert_eq!(
                choose_range(
                    &windows,
                    Keep::Last,
                    true,
                    matches!(op, CompareOp::Lt | CompareOp::Le),
                )
                .unwrap(),
                (vec![100], vec![right_index[expected_end - 1]])
            );
        }
    }

    #[test]
    fn empty_range_inputs_return_no_windows() {
        let empty_left = build_range_core(
            array![].view(),
            array![].view(),
            array![1_i64].view(),
            array![1_i64].view(),
            true,
            CompareOp::Lt,
        )
        .unwrap();
        assert!(empty_left.left_index.is_empty());
        assert!(empty_left.starts.is_empty());
        assert!(empty_left.ends.is_empty());

        let empty_right = build_range_core(
            array![0_i64].view(),
            array![10_i64].view(),
            array![].view(),
            array![].view(),
            true,
            CompareOp::Lt,
        )
        .unwrap();
        assert!(empty_right.left_index.is_empty());
        assert!(empty_right.starts.is_empty());
        assert!(empty_right.ends.is_empty());
    }

    #[test]
    fn mismatched_inputs_are_rejected() {
        assert!(build_range_core(
            array![0_i64].view(),
            array![1_i64, 2].view(),
            array![1_i64].view(),
            array![1_i64].view(),
            true,
            CompareOp::Lt,
        )
        .is_err());
    }

    #[test]
    fn not_equal_positions_materialize_original_labels() {
        let positions = build_not_equal_positions_core(
            array![2_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            array![1, 2, 3].view(),
            array![20, 21, 22].view(),
            array![0, 1, 2].view(),
            None,
            None,
            true,
            false,
            Keep::All,
        )
        .unwrap();
        assert_eq!(positions, (vec![0, 0], vec![0, 2]));
        assert_eq!(
            materialize_index_pairs(
                array![100_i64].view(),
                array![20, 21, 22].view(),
                positions.0,
                positions.1,
            )
            .unwrap(),
            (vec![100, 100], vec![20, 22])
        );
    }

    #[test]
    fn not_equal_first_and_last_use_labels_not_positions() {
        let first = build_not_equal_positions_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            array![1, 3, 5, 7].view(),
            array![40, 10, 30, 20].view(),
            array![0, 1, 2, 3].view(),
            None,
            None,
            false,
            false,
            Keep::First,
        )
        .unwrap();
        let last = build_not_equal_positions_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            array![1, 3, 5, 7].view(),
            array![40, 10, 30, 20].view(),
            array![0, 1, 2, 3].view(),
            None,
            None,
            false,
            false,
            Keep::Last,
        )
        .unwrap();
        assert_eq!(first, (vec![0], vec![1]));
        assert_eq!(last, (vec![0], vec![0]));
    }

    #[test]
    fn not_equal_any_selects_a_strict_or_null_candidate() {
        let strict = build_not_equal_positions_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            array![1, 3, 5, 7].view(),
            array![40, 10, 30, 20].view(),
            array![0, 1, 2, 3].view(),
            None,
            None,
            false,
            false,
            Keep::Any,
        )
        .unwrap();
        // Any valid candidate is acceptable. The current fast path chooses
        // the first physical position from the strict-less prefix.
        assert_eq!(strict, (vec![0], vec![0]));

        let null_fallback = build_not_equal_positions_core(
            array![4_i64].view(),
            array![100_i64].view(),
            array![0_i64].view(),
            array![4_i64].view(),
            array![20_i64, 21].view(),
            array![0_i64].view(),
            None,
            Some(array![1_i64].view()),
            true,
            false,
            Keep::Any,
        )
        .unwrap();
        assert_eq!(null_fallback, (vec![0], vec![1]));
    }

    #[test]
    fn not_equal_numpy_null_positions_are_candidates() {
        let positions = build_not_equal_positions_core(
            array![2_i64].view(),
            array![10_i64, 11].view(),
            array![0_i64].view(),
            array![1, 3].view(),
            array![20_i64, 21, 22].view(),
            array![0, 2].view(),
            Some(array![1_i64].view()),
            Some(array![1_i64].view()),
            true,
            false,
            Keep::All,
        )
        .unwrap();
        assert_eq!(positions, (vec![0, 0, 0, 1, 1, 1], vec![0, 2, 1, 0, 2, 1]));
    }

    #[test]
    fn not_equal_extension_null_positions_are_excluded() {
        let positions = build_not_equal_positions_core(
            array![2_i64].view(),
            array![10_i64, 11].view(),
            array![0_i64].view(),
            array![1, 3].view(),
            array![20_i64, 21, 22].view(),
            array![0, 2].view(),
            Some(array![1_i64].view()),
            Some(array![1_i64].view()),
            true,
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(positions, (vec![0, 0], vec![0, 2]));
    }

    #[test]
    fn not_equal_empty_filtered_arrays_skip_binary_search() {
        let both_null = build_not_equal_positions_core::<i64>(
            array![].view(),
            array![10_i64, 11].view(),
            array![].view(),
            array![].view(),
            array![20_i64, 21].view(),
            array![].view(),
            Some(array![0, 1].view()),
            Some(array![0, 1].view()),
            true,
            false,
            Keep::All,
        )
        .unwrap();
        assert_eq!(both_null, (vec![0, 0, 1, 1], vec![0, 1, 0, 1]));

        let extension = build_not_equal_positions_core::<i64>(
            array![].view(),
            array![10_i64].view(),
            array![].view(),
            array![].view(),
            array![20_i64].view(),
            array![].view(),
            Some(array![0].view()),
            Some(array![0].view()),
            true,
            true,
            Keep::All,
        )
        .unwrap();
        assert!(extension.0.is_empty());
        assert!(extension.1.is_empty());
    }

    #[test]
    fn not_equal_rejects_incomplete_position_partition() {
        let result = build_not_equal_positions_core(
            array![2_i64].view(),
            array![10_i64, 11].view(),
            array![0_i64].view(),
            array![1].view(),
            array![20_i64].view(),
            array![0].view(),
            None,
            None,
            true,
            false,
            Keep::Any,
        );
        assert!(result.is_err());
    }

    #[test]
    fn not_equal_rejects_out_of_bounds_physical_positions() {
        let result = build_not_equal_positions_core(
            array![2_i64].view(),
            array![10_i64, 11].view(),
            array![2_i64].view(),
            array![1_i64].view(),
            array![20_i64].view(),
            array![0_i64].view(),
            Some(array![0_i64].view()),
            None,
            true,
            false,
            Keep::Any,
        );
        assert_eq!(
            result,
            Err("left index non-null position 2 is out of bounds".to_owned())
        );
    }

    #[test]
    fn not_equal_rejects_duplicate_physical_positions() {
        let result = build_not_equal_positions_core(
            array![2_i64, 3].view(),
            array![10_i64, 11].view(),
            array![0_i64, 0].view(),
            array![1_i64].view(),
            array![20_i64].view(),
            array![0_i64].view(),
            None,
            None,
            true,
            false,
            Keep::Any,
        );
        assert_eq!(
            result,
            Err("left index position 0 appears more than once".to_owned())
        );
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};

    fn assert_value_error<'py>(
        result: PyResult<Option<Bound<'py, PyDict>>>,
        py: Python<'py>,
        expected: &str,
    ) {
        let error = result.expect_err("expected the extended join to reject its input");
        assert!(error.is_instance_of::<PyValueError>(py));
        assert_eq!(error.value(py).to_string(), expected);
    }

    fn read_pair<'py>(result: &Bound<'py, PyDict>, py: Python<'py>) -> (Vec<i64>, Vec<i64>) {
        let left = result
            .get_item("left_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        let right = result
            .get_item("right_index")
            .unwrap()
            .unwrap()
            .cast::<PyArray1<i64>>()
            .unwrap()
            .readonly()
            .as_array()
            .to_vec();
        let _ = py;
        (left, right)
    }

    #[test]
    fn filters_range_windows_before_keep_selection() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![100_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                        PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![3_i64, 7, 9, 7]).into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();

            let result = single_join_extended_indices_int64(py, &predicates, "first")
                .unwrap()
                .unwrap();
            assert_eq!(read_pair(&result, py), (vec![100], vec![20]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn filters_second_range_as_a_residual_before_keep_selection() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 30, 50, 70]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![4_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1, 2, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result = single_join_extended_indices_int64(py, &predicates, "all")?
                .expect("the intersected range has one match");
            assert_eq!(read_pair(&result, py), (vec![100], vec![70]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn no_surviving_residual_matches_returns_none() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![100_i64]).into_any(),
                        PyArray1::from_vec(py, vec![5_i64, 7]).into_any(),
                        PyArray1::from_vec(py, vec![10_i64, 20]).into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        "<".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![4_i64]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                        "==".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();

            assert!(single_join_extended_indices_int64(py, &predicates, "all")
                .unwrap()
                .is_none());
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn all_not_equal_joins_filter_flat_position_pairs() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                        PyArray1::from_vec(py, vec![10_i64, 11, 12]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64, 1, 2]).into_any(),
                        py.None().into_pyobject(py)?.into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                        PyArray1::from_vec(py, vec![20_i64, 21, 22]).into_any(),
                        PyArray1::from_vec(py, vec![0_i64, 1, 2]).into_any(),
                        py.None().into_pyobject(py)?.into_any(),
                        true.into_pyobject(py)?.to_owned().into_any(),
                        false.into_pyobject(py)?.to_owned().into_any(),
                        "!=".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64, 2, 3]).into_any(),
                        PyArray1::from_vec(py, vec![1_i64, 3, 2]).into_any(),
                        "!=".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();

            let result = single_join_extended_indices_int64(py, &predicates, "all")
                .unwrap()
                .unwrap();
            assert_eq!(
                read_pair(&result, py),
                (vec![10, 10, 11, 12], vec![21, 22, 20, 20])
            );

            let result = single_join_extended_indices_int64(py, &predicates, "first")
                .unwrap()
                .unwrap();
            assert_eq!(read_pair(&result, py), (vec![10, 11, 12], vec![21, 20, 20]));
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn validation_errors_report_their_exact_contract() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let no_predicates = PyList::empty(py);
            assert_value_error(
                single_join_extended_indices_int64(py, &no_predicates, "all"),
                py,
                "single extended join requires at least two predicates",
            );

            let bad_first_shape = PyList::empty(py);
            bad_first_shape.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            bad_first_shape.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(py, &bad_first_shape, "all"),
                py,
                "the first extended predicate must contain 6 or 11 elements",
            );

            // A six-field first predicate is the range layout.  `!=` is not
            // a range anchor, so it must be rejected at this boundary rather
            // than reaching the range builder with an unsupported operator.
            let bad_range_operator = PyList::empty(py);
            bad_range_operator.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            bad_range_operator.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(py, &bad_range_operator, "all"),
                py,
                "the first range predicate must use <, <=, >, or >=",
            );

            let invalid_keep = PyList::empty(py);
            invalid_keep.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64, 21]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            invalid_keep.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(py, &invalid_keep, "middle"),
                py,
                "invalid keep value: middle (expected one of first, last, any, all)",
            );

            let bad_residual_shape = PyList::empty(py);
            bad_residual_shape.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64, 21]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            bad_residual_shape.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![3_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(py, &bad_residual_shape, "all"),
                py,
                "each residual comparison must contain 3 or 6 elements",
            );

            let invalid_residual_operator = PyList::empty(py);
            invalid_residual_operator.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            invalid_residual_operator.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "like".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(
                    py,
                    &invalid_residual_operator,
                    "all",
                ),
                py,
                "invalid comparison operator: like (expected one of >, >=, <, <=, ==, !=)",
            );

            let mismatched_residual = PyList::empty(py);
            mismatched_residual.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64]).into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            mismatched_residual.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(py, &mismatched_residual, "all"),
                py,
                "first left predicate array and residual left predicate array must have equal lengths; got 1 and 2",
            );
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn not_equal_validation_rejects_incomplete_physical_partitions() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64]).into_any(),
                    PyArray1::from_vec(py, vec![10_i64, 11]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64, 21]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64]).into_any(),
                    py.None().into_pyobject(py)?.into_any(),
                    true.into_pyobject(py)?.to_owned().into_any(),
                    false.into_pyobject(py)?.to_owned().into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![1_i64, 2]).into_any(),
                    PyArray1::from_vec(py, vec![20_i64, 21]).into_any(),
                    "!=".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert_value_error(
                single_join_extended_indices_int64(py, &predicates, "all"),
                py,
                "left full positions length must equal the number of non-null values plus null positions",
            );
            Ok(())
        })
        .unwrap();
    }
}
