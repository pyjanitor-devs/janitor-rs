//! Index construction for one non-equi join predicate.
//!
//! Pyjanitor supplies non-null, value-sorted arrays for the search paths and
//! keeps each value array aligned with its original `int64` index array. Rust
//! does not sort either side and does not infer nullness from values such as
//! `NaN`; those responsibilities belong to the Python caller and its masks.
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
//! prefix or suffix.

use numpy::ndarray::{s, ArrayView1};
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

fn null_labels(
    mask: Option<ArrayView1<'_, bool>>,
    index: Option<ArrayView1<'_, i64>>,
) -> Result<Vec<i64>, String> {
    match (mask, index) {
        (None, None) => Ok(Vec::new()),
        (Some(mask), Some(index)) => {
            // The mask is authoritative: true means null, including for
            // floating-point arrays. The value array is not inspected here.
            ensure_equal_lengths_core("null mask", mask.len(), "null index", index.len())?;
            Ok(mask
                .iter()
                .zip(index.iter())
                .filter_map(|(&is_null, &label)| is_null.then_some(label))
                .collect())
        }
        _ => Err("null masks and indexes must be provided together".to_owned()),
    }
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
    keep: Keep,
) -> Result<usize, &'static str> {
    const CAPACITY_ERROR: &str = "single join result size exceeds platform capacity";

    if keep != Keep::All {
        return left
            .len()
            .checked_add(left_null_count)
            .ok_or(CAPACITY_ERROR);
    }

    let null_row_width = right
        .len()
        .checked_add(right_null_count)
        .ok_or(CAPACITY_ERROR)?;
    let mut capacity = 0_usize;
    for left_value in left {
        let gt_start = partition_point(right, |value| value <= *left_value);
        let lt_end = partition_point(right, |value| value < *left_value);
        let strict_width = lt_end
            .checked_add(right.len() - gt_start)
            .ok_or(CAPACITY_ERROR)?;
        let row_width = strict_width
            .checked_add(right_null_count)
            .ok_or(CAPACITY_ERROR)?;
        capacity = capacity.checked_add(row_width).ok_or(CAPACITY_ERROR)?;
    }
    let null_left_capacity = left_null_count
        .checked_mul(null_row_width)
        .ok_or(CAPACITY_ERROR)?;
    capacity
        .checked_add(null_left_capacity)
        .ok_or(CAPACITY_ERROR)
}

/// Build `!=` results from filtered non-null arrays and optional null metadata.
///
/// # Input contract
///
/// `left` and `right` are the filtered non-null value arrays used for binary
/// search. Their index arrays are aligned original `int64` labels. If nulls
/// exist, `left_nulls`/`left_nulls_index` and
/// `right_nulls`/`right_nulls_index` are full-array mask/label pairs: a `true`
/// mask entry identifies the corresponding null label. Each optional pair is
/// either fully present or fully absent.
///
/// `right` must be sorted. `right_index_is_ordered` means that the supplied
/// right labels are monotonically increasing in this already value-sorted
/// layout; when true, `first`/`last` can use boundaries directly. Rust does
/// not sort values or infer nullness.
///
/// # Inequality semantics
///
/// For a non-null left row, `!=` is the union of the strict `<` prefix, the
/// strict `>` suffix, and every right-null label. Equal non-null values are
/// excluded because both non-null regions are strict. Null-null pairs are
/// included because null is treated as unequal to every value, including
/// another null. For a null left row, every right row matches, including
/// right-null rows.
///
/// `all` emits physical right-array order. The other modes emit at most one
/// original right label per left row. If no candidate exists, that left row
/// contributes no output pair.
///
/// # Arguments
///
/// * `left` - Filtered non-null left values in left-row order.
/// * `left_index` - Original labels aligned with `left`.
/// * `right` - Filtered non-null right values, already sorted.
/// * `right_index` - Original labels aligned with `right`.
/// * `left_nulls` and `left_nulls_index` - Optional full-array left mask and
///   aligned original labels.
/// * `right_nulls` and `right_nulls_index` - Optional full-array right mask and
///   aligned original labels.
/// * `right_index_is_ordered` - Whether right labels increase monotonically in
///   the sorted-right layout.
/// * `keep` - Selection mode controlling one result, any result, or all
///   results per left row.
///
/// # Errors
///
/// Returns an error for mismatched value/index lengths, mismatched mask/index
/// lengths, or incomplete optional mask/index pairs. Empty filtered arrays
/// are valid when their original rows are represented by null labels; a side
/// with no rows produces no pairs.
#[allow(clippy::too_many_arguments)]
pub fn build_not_equal_core<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    left_nulls: Option<ArrayView1<'_, bool>>,
    left_nulls_index: Option<ArrayView1<'_, i64>>,
    right_nulls: Option<ArrayView1<'_, bool>>,
    right_nulls_index: Option<ArrayView1<'_, i64>>,
    right_index_is_ordered: bool,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_equal_lengths_core("left", left.len(), "left_index", left_index.len())?;
    ensure_equal_lengths_core("right", right.len(), "right_index", right_index.len())?;
    let left_null_labels = null_labels(left_nulls, left_nulls_index)?;
    let right_null_labels = null_labels(right_nulls, right_nulls_index)?;
    if (left.is_empty() && left_null_labels.is_empty())
        || (right.is_empty() && right_null_labels.is_empty())
    {
        return Ok((Vec::new(), Vec::new()));
    }
    let output_capacity = not_equal_output_capacity(
        left,
        right,
        left_null_labels.len(),
        right_null_labels.len(),
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
    // Prefix/suffix extrema are used only for non-null left rows. Avoid
    // building them when the filtered left side is empty; null-left rows use
    // whole-array extrema below instead.
    let need_unordered_extrema = !left.is_empty() && !right_index_is_ordered;
    let prefix_min = if need_unordered_extrema && keep == Keep::First {
        Some(prefix_extreme(right_index.iter().copied(), true))
    } else {
        None
    };
    let prefix_max = if need_unordered_extrema && keep == Keep::Last {
        Some(prefix_extreme(right_index.iter().copied(), false))
    } else {
        None
    };
    let suffix_min = if need_unordered_extrema && keep == Keep::First {
        Some(suffix_extreme(right_index.iter().copied(), true))
    } else {
        None
    };
    let suffix_max = if need_unordered_extrema && keep == Keep::Last {
        Some(suffix_extreme(right_index.iter().copied(), false))
    } else {
        None
    };
    // Compute the right-null winner once. Recomputing this inside the
    // per-left-row loop would scan every right-null label for every left row.
    let right_null_extreme = match keep {
        Keep::First => right_null_labels.iter().copied().min(),
        Keep::Last => right_null_labels.iter().copied().max(),
        Keep::Any | Keep::All => None,
    };

    /// Append one selected join pair when a candidate exists.
    ///
    /// The selected modes (`first`, `last`, and `any`) represent a possible
    /// right match as `Option<i64>` because a left row may have no `<`, `>`, or
    /// null candidate. `Some(right_label)` appends one pair to the two
    /// parallel output vectors; `None` leaves both vectors unchanged.
    ///
    /// # Arguments
    ///
    /// * `output_left` - Output left-label vector, mutated in place.
    /// * `output_right` - Output right-label vector, mutated in lockstep with
    ///   `output_left`.
    /// * `left_label` - Original label for the left row being emitted.
    /// * `candidate` - Optional original right label selected for that row.
    fn emit(
        output_left: &mut Vec<i64>,
        output_right: &mut Vec<i64>,
        left_label: i64,
        candidate: Option<i64>,
    ) {
        // `None` represents a left row for which all candidate regions were
        // empty. Keeping this check in one helper prevents each selection
        // branch from duplicating the same conditional append logic.
        if let Some(right_label) = candidate {
            output_left.push(left_label);
            output_right.push(right_label);
        }
    }
    for (left_position, left_value) in left.iter().enumerate() {
        // Example: right values [1, 3, 3, 5] with left value 3 produce a
        // `<` prefix [1] ending at lt_end == 1 and a `>` suffix [5]
        // beginning at gt_start == 3. The equal values [3, 3] are excluded.
        let gt_start = partition_point(right, |value| value <= *left_value);
        let lt_end = partition_point(right, |value| value < *left_value);
        if keep == Keep::All {
            // The two strict non-null regions are disjoint. Append them in
            // physical right-array order, then append right-null labels;
            // exact parity with Python's internal ordering is not required,
            // but every valid pair must be represented.
            for &label in right_index.slice(s![..lt_end]) {
                output_left.push(left_index[left_position]);
                output_right.push(label);
            }
            for &label in right_index.slice(s![gt_start..]) {
                output_left.push(left_index[left_position]);
                output_right.push(label);
            }
            // This loop is inside the filtered non-null left path, so a
            // right-null value is never paired with a left-null value here.
            for &label in &right_null_labels {
                output_left.push(left_index[left_position]);
                output_right.push(label);
            }
            continue;
        }
        if keep == Keep::Any {
            // Prefer a strict non-null candidate because it is available from
            // a boundary without scanning. If neither strict region exists,
            // a right-null label is the only possible candidate.
            if lt_end > 0 {
                emit(
                    &mut output_left,
                    &mut output_right,
                    left_index[left_position],
                    Some(right_index[0]),
                );
            } else if gt_start < right.len() {
                emit(
                    &mut output_left,
                    &mut output_right,
                    left_index[left_position],
                    Some(right_index[gt_start]),
                );
            } else {
                emit(
                    &mut output_left,
                    &mut output_right,
                    left_index[left_position],
                    right_null_labels.first().copied(),
                );
            }
            continue;
        }
        let minimum = keep == Keep::First;
        let mut candidate = None;
        let mut combine = |value: i64| {
            candidate = Some(candidate.map_or(value, |current| {
                if (minimum && value < current) || (!minimum && value > current) {
                    value
                } else {
                    current
                }
            }));
        };
        if lt_end > 0 {
            // Select the appropriate extreme from the `<` prefix. Ordered
            // labels need no table; unordered labels use the table matching
            // the requested first/last mode.
            combine(if right_index_is_ordered && minimum {
                right_index[0]
            } else if right_index_is_ordered {
                right_index[lt_end - 1]
            } else if minimum {
                prefix_min.as_ref().unwrap()[lt_end - 1]
            } else {
                prefix_max.as_ref().unwrap()[lt_end - 1]
            });
        }
        if gt_start < right.len() {
            // Select the corresponding extreme from the `>` suffix.
            combine(if right_index_is_ordered && minimum {
                right_index[gt_start]
            } else if right_index_is_ordered {
                right_index[right_index.len() - 1]
            } else if minimum {
                suffix_min.as_ref().unwrap()[gt_start]
            } else {
                suffix_max.as_ref().unwrap()[gt_start]
            });
        }
        if let Some(value) = right_null_extreme {
            // Right-null labels are valid candidates for non-null left rows.
            // Null labels are compared by their original index only here;
            // their values are never compared to the non-null values.
            combine(value);
        }
        emit(
            &mut output_left,
            &mut output_right,
            left_index[left_position],
            candidate,
        );
    }
    let (nonnull_extreme, null_extreme) = match keep {
        Keep::First => (right_index.iter().copied().min(), right_null_extreme),
        Keep::Last => (right_index.iter().copied().max(), right_null_extreme),
        Keep::Any | Keep::All => (None, None),
    };
    for &left_label in &left_null_labels {
        // A null left value is unequal to every right value, including right
        // nulls. The filtered right values and the separately supplied right
        // null labels therefore both contribute matches here.
        if keep == Keep::All {
            for &right_label in right_index.iter() {
                output_left.push(left_label);
                output_right.push(right_label);
            }
            for &right_label in &right_null_labels {
                output_left.push(left_label);
                output_right.push(right_label);
            }
        } else if keep == Keep::Any {
            emit(
                &mut output_left,
                &mut output_right,
                left_label,
                right_index
                    .first()
                    .copied()
                    .or_else(|| right_null_labels.first().copied()),
            );
        } else {
            let candidate = match (nonnull_extreme, null_extreme) {
                (Some(nonnull), Some(null)) => Some(if keep == Keep::First {
                    nonnull.min(null)
                } else {
                    nonnull.max(null)
                }),
                (Some(nonnull), None) => Some(nonnull),
                (None, Some(null)) => Some(null),
                (None, None) => None,
            };
            emit(&mut output_left, &mut output_right, left_label, candidate);
        }
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
        /// For `!=`, the value arrays are filtered non-null arrays. Optional
        /// null masks, when present, are full-array masks paired with their
        /// full-array original index labels. A mask entry of `true` is the
        /// sole authority that the corresponding row is null.
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
        /// * `left_nulls`, `left_nulls_index`, `right_nulls`,
        ///   `right_nulls_index` - Optional full-array null metadata used only
        ///   for `!=`.
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
            left_nulls: Option<PyReadonlyArray1<'py, bool>>,
            left_nulls_index: Option<PyReadonlyArray1<'py, i64>>,
            right_nulls: Option<PyReadonlyArray1<'py, bool>>,
            right_nulls_index: Option<PyReadonlyArray1<'py, i64>>,
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
                let (out_left, out_right) = build_not_equal_core(
                    left,
                    left_index,
                    right,
                    right_index,
                    left_nulls.as_ref().map(|value| value.as_array()),
                    left_nulls_index.as_ref().map(|value| value.as_array()),
                    right_nulls.as_ref().map(|value| value.as_array()),
                    right_nulls_index.as_ref().map(|value| value.as_array()),
                    right_index_is_ordered,
                    keep,
                )
                .map_err(PyValueError::new_err)?;
                if out_left.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(result_dict(py, out_left, out_right, None, None)?));
            }
            if left_nulls.is_some()
                || left_nulls_index.is_some()
                || right_nulls.is_some()
                || right_nulls_index.is_some()
            {
                return Err(PyValueError::new_err(
                    "null metadata is only supported for !=",
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
    fn not_equal_includes_null_null_and_reduces_extrema() {
        let equal_values_are_excluded = build_not_equal_core(
            array![2_i64].view(),
            array![10_i64].view(),
            array![1, 2, 3].view(),
            array![20, 21, 22].view(),
            None,
            None,
            None,
            None,
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(equal_values_are_excluded, (vec![10, 10], vec![20, 22]));

        let result = build_not_equal_core(
            array![2_i64].view(),
            array![10_i64].view(),
            array![1, 2, 3].view(),
            array![30, 20, 40].view(),
            Some(array![false, true].view()),
            Some(array![10, 11].view()),
            Some(array![false, true, false].view()),
            Some(array![30, 21, 40].view()),
            false,
            Keep::First,
        )
        .unwrap();
        assert_eq!(result, (vec![10, 11], vec![21, 20]));

        let result = build_not_equal_core(
            array![2_i64].view(),
            array![10_i64].view(),
            array![1].view(),
            array![30].view(),
            Some(array![false, true].view()),
            Some(array![10, 11].view()),
            Some(array![false, true].view()),
            Some(array![30, 31].view()),
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(result, (vec![10, 10, 11, 11], vec![30, 31, 30, 31],));
    }

    #[test]
    fn not_equal_building_blocks_ignore_keep_and_materialize_all_pairs() {
        let with_building_blocks = build_not_equal_core(
            array![2_i64].view(),
            array![10_i64].view(),
            array![1, 3].view(),
            array![20, 30].view(),
            None,
            None,
            None,
            None,
            true,
            effective_keep(Keep::First, true),
        )
        .unwrap();
        let with_all = build_not_equal_core(
            array![2_i64].view(),
            array![10_i64].view(),
            array![1, 3].view(),
            array![20, 30].view(),
            None,
            None,
            None,
            None,
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(with_building_blocks, with_all);
    }

    #[test]
    fn not_equal_handles_empty_regions_and_ordered_labels() {
        let prefix_only = build_not_equal_core(
            array![10_i64].view(),
            array![100_i64].view(),
            array![1, 3, 5].view(),
            array![10, 20, 30].view(),
            None,
            None,
            None,
            None,
            true,
            Keep::First,
        )
        .unwrap();
        assert_eq!(prefix_only, (vec![100], vec![10]));

        let suffix_only = build_not_equal_core(
            array![0_i64].view(),
            array![101_i64].view(),
            array![1, 3, 5].view(),
            array![10, 20, 30].view(),
            None,
            None,
            None,
            None,
            true,
            Keep::Last,
        )
        .unwrap();
        assert_eq!(suffix_only, (vec![101], vec![30]));

        let null_only = build_not_equal_core(
            array![3_i64].view(),
            array![102_i64].view(),
            array![3].view(),
            array![20].view(),
            None,
            None,
            Some(array![false, true].view()),
            Some(array![20, 21].view()),
            true,
            Keep::Any,
        )
        .unwrap();
        assert_eq!(null_only, (vec![102], vec![21]));

        let no_match = build_not_equal_core(
            array![3_i64].view(),
            array![103_i64].view(),
            array![3].view(),
            array![20].view(),
            None,
            None,
            None,
            None,
            true,
            Keep::Any,
        )
        .unwrap();
        assert!(no_match.0.is_empty());
        assert!(no_match.1.is_empty());
    }

    #[test]
    fn not_equal_handles_empty_filtered_null_sides() {
        let both_null = build_not_equal_core::<i64>(
            array![].view(),
            array![].view(),
            array![].view(),
            array![].view(),
            Some(array![true, true].view()),
            Some(array![10, 11].view()),
            Some(array![true, true, true].view()),
            Some(array![20, 21, 22].view()),
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(
            both_null,
            (vec![10, 10, 10, 11, 11, 11], vec![20, 21, 22, 20, 21, 22])
        );

        let left_null = build_not_equal_core::<i64>(
            array![].view(),
            array![].view(),
            array![1, 2].view(),
            array![30, 31].view(),
            Some(array![true].view()),
            Some(array![10].view()),
            Some(array![false, true].view()),
            Some(array![30, 32].view()),
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(left_null, (vec![10, 10, 10], vec![30, 31, 32]));

        let right_null = build_not_equal_core::<i64>(
            array![1, 2].view(),
            array![40, 41].view(),
            array![].view(),
            array![].view(),
            Some(array![false, false].view()),
            Some(array![40, 41].view()),
            Some(array![true, true].view()),
            Some(array![50, 51].view()),
            true,
            Keep::All,
        )
        .unwrap();
        assert_eq!(right_null, (vec![40, 40, 41, 41], vec![50, 51, 50, 51]));
    }
}
