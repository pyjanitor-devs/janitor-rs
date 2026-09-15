//! Direct index construction for pyjanitor's dual non-equi regions.
//!
//! Pyjanitor first orders and aligns the two non-equi conditions. The
//! resulting `starts` array identifies the beginning of each left row's
//! candidate suffix in `right`. This module consumes those regions and builds
//! the final left/right index arrays without materialising the public
//! intermediate `positions` and `counts_array` objects.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

use crate::aggs::{ensure_equal_lengths_core, ensure_nonempty_core};
use crate::compare::common::{add_right_region, checked_region_start, GroupState, Selection};

type IndexResult = (Vec<i64>, Vec<i64>);
type PyIndexResult<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

/// Select one candidate according to the requested label-based policy.
///
/// The ordered map narrows the candidates to values `>= left_value`; the
/// linked chains then enumerate every ordinal carrying each qualifying value.
/// `First` and `Last` still inspect all candidates because `right_index` is
/// not required to be sorted. Pyjanitor resets the right frame to a unique
/// `RangeIndex` before building these regions, so labels alone determine the
/// result; no ordinal tie-break is needed. Region values may repeat, which is
/// why the duplicate chains are retained.
fn select_candidate(
    left_value: i64,
    groups: &BTreeMap<i64, GroupState>,
    next: &[i64],
    right_index: ArrayView1<'_, i64>,
    selection: Selection,
) -> Option<usize> {
    let mut selected = None;
    for (_, state) in groups.range(left_value..) {
        let mut position = state.head;
        while position >= 0 {
            let current_position = position as usize;
            match selection {
                Selection::Any => return Some(current_position),
                Selection::First => {
                    if selected
                        .is_none_or(|current| right_index[current_position] < right_index[current])
                    {
                        selected = Some(current_position);
                    }
                }
                Selection::Last => {
                    if selected
                        .is_none_or(|current| right_index[current_position] > right_index[current])
                    {
                        selected = Some(current_position);
                    }
                }
            }
            position = next[current_position];
        }
    }
    selected
}

fn validate_inputs(
    left: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    left_index: ArrayView1<'_, i64>,
    right: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
) -> Result<(), String> {
    ensure_nonempty_core("left", left.len())?;
    ensure_nonempty_core("right", right.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("left_index", left_index.len())?;
    ensure_nonempty_core("right_index", right_index.len())?;
    ensure_equal_lengths_core("left", left.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("left", left.len(), "left_index", left_index.len())?;
    ensure_equal_lengths_core("right", right.len(), "right_index", right_index.len())?;
    Ok(())
}

/// Build one selected pair per left row directly from dual regions.
///
/// `starts` must be monotonically non-increasing. Pyjanitor's
/// `_dual_non_equi` path sorts and aligns its regions before calling this
/// kernel, so each row only exposes the newly uncovered part of the right
/// suffix. This is an input contract, not a sorting operation in Rust.
///
/// The dual `left <= right` condition is already encoded by the candidate
/// regions. The returned pair arrays contain one pair for each left row that
/// has a candidate; `None` means no row has a successful candidate.
///
/// # Arguments
///
/// * `left` - Left-side values used to select qualifying right-value groups.
/// * `right` - Right-side values partitioned into the candidate regions.
/// * `starts` - Monotonically non-increasing candidate suffix start per left row.
/// * `left_index` - Original labels aligned with `left`.
/// * `right_index` - Original labels aligned with `right`.
/// * `selection` - Whether to select the smallest label, largest label, or any label.
///
/// # Errors
///
/// Returns an error when parallel inputs have different lengths or a start
/// cannot be represented as a valid right-side position.
///
/// # Returns
///
/// Returns aligned left/right labels, or `None` when no candidate satisfies
/// the already-resolved `left <= right` condition.
fn build_selected_indices_core(
    left: ArrayView1<'_, i64>,
    right: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    selection: Selection,
) -> Result<Option<IndexResult>, String> {
    validate_inputs(left, starts, left_index, right, right_index)?;

    let mut next = vec![-1_i64; right.len()];
    let mut groups = BTreeMap::<i64, GroupState>::new();
    let mut previous_end = right.len();
    let mut left_output = Vec::new();
    let mut right_output = Vec::new();

    for row in 0..left.len() {
        // Pyjanitor guarantees monotonically non-increasing starts.
        // `checked_region_start` rejects a violation before it can make a
        // previously added chain self-link.
        let Some(start) = checked_region_start(starts[row], right.len(), previous_end)? else {
            continue;
        };
        add_right_region(right, start, previous_end, &mut next, &mut groups);
        previous_end = start;

        if let Some(right_position) =
            select_candidate(left[row], &groups, &next, right_index, selection)
        {
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

/// Build every matching pair with a separate two-pass implementation.
///
/// ELI5: the first pass counts the tickets that will be printed. The second
/// pass rebuilds the chains and writes directly into exactly-sized output
/// vectors, so no flattened public positions result is retained.
///
/// The rebuild is intentional: after pass one, the chains contain the union
/// of all exposed suffixes. A row in the second pass must instead see the
/// chain snapshot at that row's `start`; retaining one snapshot per row would
/// use substantially more memory than rebuilding the compact chain once.
///
/// The output preserves left-row order, then the ordered right-value/ordinal
/// traversal produced by the region chains.
///
/// # Arguments
///
/// * `left` - Left-side values used to select qualifying right-value groups.
/// * `right` - Right-side values partitioned into the candidate regions.
/// * `starts` - Monotonically non-increasing candidate suffix start per left row.
/// * `left_index` - Original labels aligned with `left`.
/// * `right_index` - Original labels aligned with `right`.
///
/// # Errors
///
/// Returns an error when parallel inputs have different lengths or a start
/// cannot be represented as a valid right-side position.
///
/// # Returns
///
/// Returns all aligned left/right labels, or `None` when no candidate
/// satisfies the already-resolved `left <= right` condition.
fn build_all_indices_core(
    left: ArrayView1<'_, i64>,
    right: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
) -> Result<Option<IndexResult>, String> {
    validate_inputs(left, starts, left_index, right, right_index)?;

    let mut total = 0_usize;
    let mut next = vec![-1_i64; right.len()];
    let mut groups = BTreeMap::<i64, GroupState>::new();
    let mut previous_end = right.len();

    for row in 0..left.len() {
        let Some(start) = checked_region_start(starts[row], right.len(), previous_end)? else {
            continue;
        };
        add_right_region(right, start, previous_end, &mut next, &mut groups);
        previous_end = start;

        for (_, state) in groups.range(left[row]..) {
            let mut position = state.head;
            while position >= 0 {
                total = total
                    .checked_add(1)
                    .ok_or_else(|| "number of output pairs exceeds usize".to_string())?;
                position = next[position as usize];
            }
        }
    }

    if total == 0 {
        return Ok(None);
    }
    // second pass to build the output arrays
    let mut left_output = Vec::with_capacity(total);
    let mut right_output = Vec::with_capacity(total);
    next.fill(-1);
    groups.clear();
    previous_end = right.len();

    for row in 0..left.len() {
        let Some(start) = checked_region_start(starts[row], right.len(), previous_end)? else {
            continue;
        };
        add_right_region(right, start, previous_end, &mut next, &mut groups);
        previous_end = start;

        for (_, state) in groups.range(left[row]..) {
            let mut position = state.head;
            while position >= 0 {
                let current_position = position as usize;
                left_output.push(left_index[row]);
                right_output.push(right_index[current_position]);
                position = next[current_position];
            }
        }
    }

    if left_output.len() != total || right_output.len() != total {
        return Err("internal error: two-pass output count changed between passes".to_string());
    }
    Ok(Some((left_output, right_output)))
}

/// Select the smallest right label for each left row from dual non-equi regions.
///
/// `starts` must be monotonically non-increasing, as guaranteed by pyjanitor's
/// `_dual_non_equi` region construction. Returns `None` when no candidate
/// satisfies the already-resolved `left <= right` condition.
///
/// # Arguments
///
/// * `left` - Left-side values used by the dual-region condition.
/// * `right` - Right-side values used by the dual-region condition.
/// * `starts` - Candidate suffix start for each left row.
/// * `left_index` - Original labels aligned with `left`.
/// * `right_index` - Original labels aligned with `right`.
#[pyfunction]
pub fn build_dual_region_indices_first<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<PyIndexResult<'py>>> {
    let result = build_selected_indices_core(
        left.as_array(),
        right.as_array(),
        starts.as_array(),
        left_index.as_array(),
        right_index.as_array(),
        Selection::First,
    )
    .map_err(PyValueError::new_err)?;
    Ok(result.map(|(left, right)| {
        (
            Array1::from_vec(left).into_pyarray(py),
            Array1::from_vec(right).into_pyarray(py),
        )
    }))
}

/// Select the largest right label for each left row from dual non-equi regions.
///
/// `starts` must be monotonically non-increasing, as guaranteed by pyjanitor's
/// `_dual_non_equi` region construction. Returns `None` when no candidate
/// satisfies the already-resolved `left <= right` condition.
///
/// # Arguments
///
/// * `left` - Left-side values used by the dual-region condition.
/// * `right` - Right-side values used by the dual-region condition.
/// * `starts` - Candidate suffix start for each left row.
/// * `left_index` - Original labels aligned with `left`.
/// * `right_index` - Original labels aligned with `right`.
#[pyfunction]
pub fn build_dual_region_indices_last<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<PyIndexResult<'py>>> {
    let result = build_selected_indices_core(
        left.as_array(),
        right.as_array(),
        starts.as_array(),
        left_index.as_array(),
        right_index.as_array(),
        Selection::Last,
    )
    .map_err(PyValueError::new_err)?;
    Ok(result.map(|(left, right)| {
        (
            Array1::from_vec(left).into_pyarray(py),
            Array1::from_vec(right).into_pyarray(py),
        )
    }))
}

/// Select any matching right label for each left row from dual non-equi regions.
///
/// The scan stops at the first candidate satisfying the already-resolved
/// `left <= right` condition. `starts` must be monotonically non-increasing,
/// as guaranteed by pyjanitor's `_dual_non_equi` region construction. Returns
/// `None` when no candidate satisfies the condition.
///
/// # Arguments
///
/// * `left` - Left-side values used by the dual-region condition.
/// * `right` - Right-side values used by the dual-region condition.
/// * `starts` - Candidate suffix start for each left row.
/// * `left_index` - Original labels aligned with `left`.
/// * `right_index` - Original labels aligned with `right`.
#[pyfunction]
pub fn build_dual_region_indices_any<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<PyIndexResult<'py>>> {
    let result = build_selected_indices_core(
        left.as_array(),
        right.as_array(),
        starts.as_array(),
        left_index.as_array(),
        right_index.as_array(),
        Selection::Any,
    )
    .map_err(PyValueError::new_err)?;
    Ok(result.map(|(left, right)| {
        (
            Array1::from_vec(left).into_pyarray(py),
            Array1::from_vec(right).into_pyarray(py),
        )
    }))
}

/// Build every matching pair from dual non-equi regions.
///
/// Uses two passes so the final output vectors can be allocated exactly. The
/// first pass counts pairs; the second rebuilds the flat duplicate chains and
/// writes the final labels. Returns `None` when no pair satisfies
/// `left <= right`.
///
/// # Arguments
///
/// * `left` - Left-side values used by the dual-region condition.
/// * `right` - Right-side values used by the dual-region condition.
/// * `starts` - Candidate suffix start for each left row.
/// * `left_index` - Original labels aligned with `left`.
/// * `right_index` - Original labels aligned with `right`.
#[pyfunction]
pub fn build_dual_region_indices_all<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<PyIndexResult<'py>>> {
    let result = build_all_indices_core(
        left.as_array(),
        right.as_array(),
        starts.as_array(),
        left_index.as_array(),
        right_index.as_array(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(result.map(|(left, right)| {
        (
            Array1::from_vec(left).into_pyarray(py),
            Array1::from_vec(right).into_pyarray(py),
        )
    }))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_dual_region_indices_first, m)?)?;
    m.add_function(wrap_pyfunction!(build_dual_region_indices_last, m)?)?;
    m.add_function(wrap_pyfunction!(build_dual_region_indices_any, m)?)?;
    m.add_function(wrap_pyfunction!(build_dual_region_indices_all, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::array;

    type Inputs = (
        Array1<i64>,
        Array1<i64>,
        Array1<i64>,
        Array1<i64>,
        Array1<i64>,
    );

    fn inputs() -> Inputs {
        (
            array![2, 1],
            array![2, 2, 3],
            array![0, 0],
            array![100, 200],
            array![30, 10, 20],
        )
    }

    #[test]
    fn selected_modes_build_direct_indices_and_use_labels() {
        let (left, right, starts, left_index, right_index) = inputs();
        let first = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.0, vec![100, 200]);
        assert_eq!(first.1, vec![10, 10]);

        let last = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(last.1, vec![30, 30]);

        let any = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Any,
        )
        .unwrap()
        .unwrap();
        assert_eq!(any.1, vec![10, 10]);
    }

    #[test]
    fn selected_modes_compare_right_labels_not_candidate_order() {
        let left = array![2];
        let right = array![2, 3];
        let starts = array![0];
        let left_index = array![100];
        // Candidate traversal sees value 2 before value 3, but the labels
        // deliberately have the opposite order.
        let right_index = array![20, 10];

        let first = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::First,
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.1, vec![10]);

        let last = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Last,
        )
        .unwrap()
        .unwrap();
        assert_eq!(last.1, vec![20]);
    }

    #[test]
    fn all_builds_every_pair_without_counts_or_flattened_output() {
        let (left, right, starts, left_index, right_index) = inputs();
        let result = build_all_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.0, vec![100, 100, 100, 200, 200, 200]);
        assert_eq!(result.1, vec![10, 30, 20, 10, 30, 20]);
    }

    #[test]
    fn no_successful_regions_return_none() {
        let left = array![5, 6];
        let right = array![1, 2];
        let starts = array![2, 2];
        let left_index = array![10, 11];
        let right_index = array![20, 21];
        assert!(build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Any,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn invalid_shapes_are_rejected_and_invalid_rows_are_skipped() {
        let left = array![1, 2];
        let right = array![1, 2];
        let starts = array![0];
        let left_index = array![10, 11];
        let right_index = array![20, 21];
        assert!(build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Any,
        )
        .is_err());

        let starts = array![-1, 0];
        let result = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Any,
        )
        .unwrap()
        .unwrap();
        assert_eq!(result.0, vec![11]);
        assert_eq!(result.1, vec![21]);

        let empty = Array1::<i64>::zeros(0);
        assert!(build_selected_indices_core(
            empty.view(),
            right.view(),
            empty.view(),
            empty.view(),
            right_index.view(),
            Selection::Any,
        )
        .is_err());
    }

    #[test]
    fn validation_reports_the_nonempty_error_exactly() {
        let empty = Array1::<i64>::zeros(0);
        let right = array![1];
        let starts = Array1::<i64>::zeros(0);
        let left_index = Array1::<i64>::zeros(0);
        let right_index = array![10];
        let error = build_selected_indices_core(
            empty.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Any,
        )
        .unwrap_err();
        assert_eq!(error, "left cannot be empty");
    }

    #[test]
    fn non_monotonic_starts_are_rejected_before_self_linking() {
        let left = array![0, 0, 0];
        let right = array![0, 1, 2, 3, 4];
        // The jump from 2 to 4 must be rejected before it overwrites the
        // previous boundary. A later retreat to 3 would otherwise re-add an
        // already-linked position and create a self-loop in `next`.
        let starts = array![2, 4, 3];
        let left_index = array![100, 200, 300];
        let right_index = array![10, 20, 30, 40, 50];
        let error = build_selected_indices_core(
            left.view(),
            right.view(),
            starts.view(),
            left_index.view(),
            right_index.view(),
            Selection::Any,
        )
        .unwrap_err();
        assert_eq!(error, "starts must be monotonically non-increasing");
    }
}
