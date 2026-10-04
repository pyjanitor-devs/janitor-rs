//! Small, dependency-free primitives shared by join kernels.

use numpy::ndarray::ArrayView1;

use crate::compare_op::CompareOp;

/// Return the first offset at which `predicate` is false.
///
/// The predicate must be true for an initial prefix of `values` and false for
/// the remaining suffix. Contiguous arrays use the standard slice
/// implementation; strided ndarray views use the equivalent binary search.
pub(crate) fn partition_point<T: PartialOrd + Copy>(
    values: ArrayView1<'_, T>,
    predicate: impl Fn(T) -> bool,
) -> usize {
    if let Some(slice) = values.as_slice() {
        return slice.partition_point(|value| predicate(*value));
    }
    let mut low = 0;
    let mut high = values.len();
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

/// Return the half-open right-side window satisfying one range predicate.
///
/// `right` must be sorted in ascending order. The returned values are offsets
/// into that sorted view, never physical index values. Equality and
/// inequality do not define one monotone window and must be rejected by the
/// caller before invoking this helper.
pub(crate) fn range_window<T: PartialOrd + Copy>(
    left_value: T,
    right: ArrayView1<'_, T>,
    op: CompareOp,
) -> (usize, usize) {
    match op {
        CompareOp::Lt => (
            partition_point(right, |value| value <= left_value),
            right.len(),
        ),
        CompareOp::Le => (
            partition_point(right, |value| value < left_value),
            right.len(),
        ),
        CompareOp::Gt => (0, partition_point(right, |value| value < left_value)),
        CompareOp::Ge => (0, partition_point(right, |value| value <= left_value)),
        CompareOp::Eq | CompareOp::Ne => unreachable!("range_window only handles range operators"),
    }
}

/// Build dense half-open windows for every left value.
///
/// The returned `starts` and `ends` arrays have one entry per left row,
/// including empty windows. Their entries are offsets into the supplied
/// sorted right layout; they are never values from `right_index`.
pub(crate) fn range_window_bounds<T: PartialOrd + Copy>(
    left: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, T>,
    right_index: ArrayView1<'_, i64>,
    op: CompareOp,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if left.len() != left_index.len() {
        return Err("left values and left index must have equal lengths".to_owned());
    }
    if right.len() != right_index.len() {
        return Err("right values and right index must have equal lengths".to_owned());
    }
    if !matches!(
        op,
        CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge
    ) {
        return Err("range window requires <, <=, >, or >=".to_owned());
    }
    let mut starts = Vec::with_capacity(left.len());
    let mut ends = Vec::with_capacity(left.len());
    for &value in left {
        let (start, end) = range_window(value, right, op);
        starts.push(start);
        ends.push(end);
    }
    Ok((starts, ends))
}
