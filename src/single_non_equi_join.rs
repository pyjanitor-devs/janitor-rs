//! Index construction for one non-equi join predicate.
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

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::aggs::ensure_equal_lengths_core;
use crate::join_common::{result_dict, Keep, SingleJoinResult};
use crate::op::CompareOp;

/// Find the first position at which a monotone predicate becomes false.
///
/// The predicate must be true for an initial prefix of `right` and false for
/// the remaining suffix. This is the same boundary operation as Rust's slice
/// `partition_point`.
///
/// Contiguous views use the standard slice implementation. Strided ndarray
/// views cannot expose a contiguous slice, so they use the equivalent manual
/// binary-search loop. Both paths return a physical position in the supplied
/// right view; neither path changes or sorts the input.
pub(crate) fn partition_point<T: PartialOrd + Copy>(
    right: ArrayView1<'_, T>,
    predicate: impl Fn(T) -> bool,
) -> usize {
    if let Some(slice) = right.as_slice() {
        return slice.partition_point(|value| predicate(*value));
    }
    let mut low = 0;
    let mut high = right.len();
    while low < high {
        let middle = low + ((high - low) >> 1);
        if predicate(right[middle]) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
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
/// The returned `(start, end)` pair identifies the contiguous portion of the
/// ascending `right` array that satisfies `left_value op right_value`.
///
/// # Arguments
///
/// * `left_value` - One left-side value.
/// * `right` - An ascending right-side value view.
/// * `op` - One of `<`, `<=`, `>`, or `>=`.
///
/// # Returns
///
/// A half-open positional range into `right`. Equality and inequality are not
/// valid inputs and are unreachable after caller validation.
pub(crate) fn range_window<T: PartialOrd + Copy>(
    left_value: T,
    right: ArrayView1<'_, T>,
    op: CompareOp,
) -> (usize, usize) {
    match op {
        // left < right: keep the suffix after the last right <= left.
        CompareOp::Lt => {
            let start = partition_point(right, |value| value <= left_value);
            (start, right.len())
        }
        // left <= right: keep the suffix from the first right >= left.
        CompareOp::Le => {
            let start = partition_point(right, |value| value < left_value);
            (start, right.len())
        }
        // left > right: keep the prefix before the first right >= left.
        CompareOp::Gt => {
            let end = partition_point(right, |value| value < left_value);
            (0, end)
        }
        // left >= right: keep the prefix through the last right <= left.
        CompareOp::Ge => {
            let end = partition_point(right, |value| value <= left_value);
            (0, end)
        }
        CompareOp::Eq | CompareOp::Ne => unreachable!("range_window only handles range operators"),
    }
}

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
    _right_index_is_ordered: bool,
    op: CompareOp,
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
        right_index: right_index.to_vec(),
        starts: Vec::new(),
        ends: Vec::new(),
    };
    if left.is_empty() || right.is_empty() {
        return Ok(result);
    }
    for (left_position, left_value) in left.iter().enumerate() {
        let (start, end) = range_window(*left_value, right, op);
        if start >= end {
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
fn materialize_index_pairs(
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
fn effective_keep(keep: Keep, return_building_blocks: bool) -> Keep {
    if return_building_blocks {
        Keep::All
    } else {
        keep
    }
}

macro_rules! single_join_function {
    ($name:ident, $type:ty) => {
        #[allow(clippy::too_many_arguments)]
        #[pyfunction]
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
        /// physical matching right row. `return_building_blocks` ignores
        /// `keep` for every operator. For range operators it returns the
        /// retained left labels, the complete right-label array, and one
        /// half-open `starts`/`ends` window per retained left row. For `!=`,
        /// it returns fully materialized flat pairs, equivalent to
        /// `keep="all"`, because `!=` has no single window.
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
        /// * `return_building_blocks` - Ignore `keep` and return range windows
        ///   or, for `!=`, all fully materialized pairs.
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
        pub fn $name<'py>(
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
                let (left_positions, right_positions) = build_not_equal_positions_core(
                    left,
                    left_index,
                    left_positions.as_array(),
                    right,
                    right_index,
                    right_positions.as_array(),
                    left_null_positions.as_ref().map(|value| value.as_array()),
                    right_null_positions.as_ref().map(|value| value.as_array()),
                    right_index_is_ordered,
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_join_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(single_join_indices_f32, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, s};

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
