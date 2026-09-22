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
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::aggs::ensure_equal_lengths_core;
use crate::op::CompareOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keep {
    /// Select the smallest matching original right index label.
    First,
    /// Select the largest matching original right index label.
    Last,
    /// Select any one matching right index label without computing extrema.
    Any,
    /// Emit every matching pair in physical right-array order.
    All,
}

impl Keep {
    /// Parse the public string representation used by the PyO3 wrapper.
    ///
    /// Keeping this conversion at the Python boundary means the core
    /// functions can work with a closed enum and cannot silently accept a
    /// misspelled selection mode.
    pub(crate) fn parse(value: &str) -> PyResult<Self> {
        match value {
            "first" => Ok(Self::First),
            "last" => Ok(Self::Last),
            "any" => Ok(Self::Any),
            "all" => Ok(Self::All),
            other => Err(PyValueError::new_err(format!(
                "invalid keep value: {other} (expected one of first, last, any, all)"
            ))),
        }
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SingleJoinResult {
    /// Physical left positions corresponding to `left_index`.
    pub(crate) left_positions: Vec<usize>,
    /// Original left index labels for rows with a non-empty match window.
    pub left_index: Vec<i64>,
    /// The complete original right index-label array, in sorted-value order.
    pub right_index: Vec<i64>,
    /// Inclusive start positions of each retained row's right match window.
    pub starts: Vec<usize>,
    /// Exclusive end positions of each retained row's right match window.
    pub ends: Vec<usize>,
}

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
fn partition_point<T: PartialOrd + Copy>(
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
fn range_bounds<T: PartialOrd + Copy>(
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
        CompareOp::Eq | CompareOp::Ne => unreachable!("range_bounds only handles range operators"),
    }
}

/// Build a running minimum or maximum for every prefix of an index-label
/// sequence.
///
/// The iterator is consumed in logical right-array order. Keeping this helper
/// iterator-based lets it serve both contiguous slices and strided ndarray
/// views without copying either input first.
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

/// Build a running minimum or maximum for every suffix of an index-label
/// sequence.
///
/// `ExactSizeIterator` lets the helper place each result at its original
/// physical position while `DoubleEndedIterator` lets it scan from right to
/// left. This preserves the logical order for strided views.
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

/// Return the physical position of the running label minimum or maximum for
/// every prefix of a right-label sequence.
fn prefix_extreme_positions(labels: &[i64], minimum: bool) -> Vec<usize> {
    let mut result = Vec::with_capacity(labels.len());
    let mut current = None;
    for (position, &label) in labels.iter().enumerate() {
        if current.is_none_or(|selected| {
            (minimum && label < labels[selected]) || (!minimum && label > labels[selected])
        }) {
            current = Some(position);
        }
        result.push(current.expect("a prefix position exists after iteration"));
    }
    result
}

/// Return the physical position of the running label minimum or maximum for
/// every suffix of a right-label sequence.
fn suffix_extreme_positions(labels: &[i64], minimum: bool) -> Vec<usize> {
    let mut result = vec![0_usize; labels.len()];
    let mut current = None;
    for position in (0..labels.len()).rev() {
        let label = labels[position];
        if current.is_none_or(|selected| {
            (minimum && label < labels[selected]) || (!minimum && label > labels[selected])
        }) {
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
        right_index: right_index.to_vec(),
        ..SingleJoinResult::default()
    };
    if left.is_empty() || right.is_empty() {
        return Ok(result);
    }
    for (left_position, left_value) in left.iter().enumerate() {
        let (start, end) = range_bounds(*left_value, right, op);
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

fn choose_range(
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

fn physical_positions(name: &str, values: ArrayView1<'_, i64>) -> Result<Vec<usize>, String> {
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
fn validate_position_partition(
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

/// Compute a checked output-capacity bound for the `!=` kernel.
///
/// `all` can emit several pairs for one left row, so its capacity is computed
/// from the two strict binary-search regions plus the right-null labels. The
/// selected modes emit at most one pair for each non-null or null left row,
/// so `left.len() + left_null_count` is a sufficient bound for them.
fn not_equal_output_capacity<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    right: ArrayView1<'_, T>,
    left_null_count: usize,
    right_null_count: usize,
    is_extension_array: bool,
    keep: Keep,
) -> Result<usize, &'static str> {
    const CAPACITY_ERROR: &str = "single join result size exceeds platform capacity";

    if keep != Keep::All {
        return left
            .len()
            .checked_add(left_null_count)
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
        .map(|positions| physical_positions("left null", positions))
        .transpose()?;
    let right_null_positions = right_null_positions
        .map(|positions| physical_positions("right null", positions))
        .transpose()?;
    let empty = Vec::new();
    let left_null_positions = left_null_positions.as_deref().unwrap_or(&empty);
    let right_null_positions = right_null_positions.as_deref().unwrap_or(&empty);
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
    if (left.is_empty() && left_null_positions.is_empty())
        || (right.is_empty() && right_null_positions.is_empty())
    {
        return Ok((Vec::new(), Vec::new()));
    }

    let right_labels: Vec<i64> = right_positions
        .iter()
        .map(|&position| right_index[position])
        .collect();
    let output_capacity = not_equal_output_capacity(
        left,
        right,
        left_null_positions.len(),
        right_null_positions.len(),
        is_extension_array,
        keep,
    )
    .map_err(str::to_owned)?;
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(output_capacity)
        .map_err(|_| "single join result allocation failed".to_owned())?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(output_capacity)
        .map_err(|_| "single join result allocation failed".to_owned())?;

    let need_unordered_extrema = !left.is_empty() && !right_index_is_ordered;
    let prefix_min = (need_unordered_extrema && keep == Keep::First)
        .then(|| prefix_extreme_positions(&right_labels, true));
    let prefix_max = (need_unordered_extrema && keep == Keep::Last)
        .then(|| prefix_extreme_positions(&right_labels, false));
    let suffix_min = (need_unordered_extrema && keep == Keep::First)
        .then(|| suffix_extreme_positions(&right_labels, true));
    let suffix_max = (need_unordered_extrema && keep == Keep::Last)
        .then(|| suffix_extreme_positions(&right_labels, false));
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

    for (left_offset, left_value) in left.iter().enumerate() {
        let left_position = left_positions[left_offset];
        let gt_start = partition_point(right, |value| value <= *left_value);
        let lt_end = partition_point(right, |value| value < *left_value);
        if keep == Keep::All {
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
            for &right_position in &right_positions {
                output_left.push(left_position);
                output_right.push(right_position);
            }
            for &right_position in right_null_positions {
                output_left.push(left_position);
                output_right.push(right_position);
            }
        } else if keep == Keep::Any {
            if let Some(right_position) = right_positions
                .first()
                .or_else(|| right_null_positions.first())
                .copied()
            {
                output_left.push(left_position);
                output_right.push(right_position);
            }
        } else {
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

fn result_dict<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
    starts: Option<Vec<usize>>,
    ends: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyDict>> {
    // Building blocks are optional because ordinary selected results contain
    // only the flattened left/right label arrays. Range callers request the
    // positional starts/ends separately when they need to aggregate later.
    let result = PyDict::new(py);
    result.set_item("left_index", left.into_pyarray(py))?;
    result.set_item("right_index", right.into_pyarray(py))?;
    if let (Some(starts), Some(ends)) = (starts, ends) {
        // Keep positions as `usize` while Rust is constructing windows, then
        // normalize the Python-facing dtype to int64. This matches every
        // downstream range/aggregation kernel and avoids platform-dependent
        // NumPy `usize` output.
        let starts = positions_to_i64(starts)?;
        let ends = positions_to_i64(ends)?;
        result.set_item("starts", starts.into_pyarray(py))?;
        result.set_item("ends", ends.into_pyarray(py))?;
    }
    Ok(result)
}

/// Convert internal positional bounds to the stable Python-facing int64 type.
///
/// Rust uses `usize` for indexing because positions cannot be negative and
/// ndarray/slice APIs use `usize`. The Python API uses int64 for all positional
/// arrays, so the conversion is centralized at the PyO3 boundary and checked
/// instead of using a potentially truncating cast.
fn positions_to_i64(values: Vec<usize>) -> PyResult<Vec<i64>> {
    values
        .into_iter()
        .map(|value| {
            i64::try_from(value)
                .map_err(|_| PyValueError::new_err("single join position exceeds int64 capacity"))
        })
        .collect()
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
        /// * `left`, `left_index` - Aligned left values and original labels.
        /// * `right`, `right_index` - Aligned, already value-sorted right
        ///   values and original labels.
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
}
