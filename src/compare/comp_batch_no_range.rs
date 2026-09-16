//! Fused comparison for already-aligned, one-candidate-per-left-row inputs.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::predicate::{
    parse_predicates_with_nulls, predicates_match_dispatch_views, NullMetadata, Predicate,
};
use crate::aggs::checked_index;
use crate::aggs::{ensure_equal_lengths, ensure_equal_lengths_core};

type BatchIndices<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);
type CoreIndices = (Vec<i64>, Vec<i64>);

/// Compare one already-selected right candidate for each left row.
///
/// The candidate positions are aligned with the left predicate arrays: row
/// `n` uses `positions[n]` as its candidate in every right predicate array.
/// Three-element predicate tuples have the form `(left, right, op)`;
/// six-element tuples additionally carry the null masks and extension-array
/// flag used by `!=` comparisons.
///
/// Invalid positions are treated as non-matches and do not abort other rows.
/// The caller-owned positions array is never modified. The function returns
/// `None` when no candidate survives, otherwise it returns the original left
/// and right labels for each surviving pair.
///
/// # Arguments
///
/// * `predicates` - Non-empty Python list of three- or six-element comparison
///   tuples. Predicate arrays must share left and right lengths.
/// * `left_index` - Original labels for the left predicate rows.
/// * `right_index` - Original labels for the right predicate rows.
/// * `positions` - One right positional candidate per left row.
///
/// # Returns
///
/// `None` when no candidates survive, otherwise `(left_index, right_index)`
/// containing only successful pairs.
///
/// # Errors
///
/// Returns a `ValueError` for malformed predicates or mismatched parallel
/// array lengths.
#[pyfunction]
pub fn compare_batch_no_range<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<BatchIndices<'py>>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err("at least one comparison is required"));
    }

    let left_len = predicates[0].left_len();
    let right_len = predicates[0].right_len();
    for predicate in predicates.iter().skip(1) {
        ensure_equal_lengths(
            "first left predicate array",
            left_len,
            "current left predicate array",
            predicate.left_len(),
        )?;
        ensure_equal_lengths(
            "first right predicate array",
            right_len,
            "current right predicate array",
            predicate.right_len(),
        )?;
    }
    ensure_equal_lengths(
        "left index",
        left_index.len()?,
        "left predicate array",
        left_len,
    )?;
    ensure_equal_lengths(
        "right index",
        right_index.len()?,
        "right predicate array",
        right_len,
    )?;
    ensure_equal_lengths(
        "positions",
        positions.len()?,
        "left predicate array",
        left_len,
    )?;

    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views: Option<Vec<_>> = metadata
        .as_ref()
        .map(|values| values.iter().map(NullMetadata::view).collect());
    let result = compare_batch_no_range_core(
        left_index.as_array(),
        right_index.as_array(),
        positions.as_array(),
        |left, right| {
            predicates_match_dispatch_views(&views, metadata_views.as_deref(), left, right)
        },
    )
    .map_err(PyValueError::new_err)?;

    Ok(result.map(|(left, right)| {
        (
            Array1::from_vec(left).into_pyarray(py),
            Array1::from_vec(right).into_pyarray(py),
        )
    }))
}

/// Compare aligned candidate positions and materialize only successful labels.
///
/// ELI5: the first pass puts a red cross on every failed candidate and counts
/// the winners. The second pass reads only the winners and copies their labels
/// into exactly-sized output vectors; it never asks the predicates the same
/// question twice.
///
/// `positions` is borrowed as read-only because the caller's NumPy array must
/// remain untouched. The local survivor buffer is the mutable copy used by
/// both passes.
fn compare_batch_no_range_core<F>(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    mut matches: F,
) -> Result<Option<CoreIndices>, String>
where
    F: FnMut(usize, usize) -> bool,
{
    ensure_equal_lengths_core("left index", left_index.len(), "positions", positions.len())?;

    let mut positions = positions.to_vec();
    let mut total = 0_usize;
    for (row, position) in positions.iter_mut().enumerate() {
        let Some(right_position) = checked_index(*position, right_index.len()) else {
            *position = -1;
            continue;
        };
        if matches(row, right_position) {
            total += 1;
        } else {
            *position = -1;
        }
    }

    if total == 0 {
        return Ok(None);
    }

    let mut output_left = Vec::with_capacity(total);
    let mut output_right = Vec::with_capacity(total);
    for (row, position) in positions.iter().enumerate() {
        let Some(right_position) = checked_index(*position, right_index.len()) else {
            continue;
        };
        output_left.push(left_index[row]);
        output_right.push(right_index[right_position]);
    }
    debug_assert_eq!(output_left.len(), total);
    debug_assert_eq!(output_right.len(), total);
    Ok(Some((output_left, output_right)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_batch_no_range, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{array, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn emits_labels_for_surviving_positions_only() {
        let left = array![10_i64, 20, 30];
        let right = array![100_i64, 200, 300];
        let positions = array![2_i64, -1, 8];
        let result = compare_batch_no_range_core(
            left.view(),
            right.view(),
            positions.view(),
            |row, position| row == 0 && position == 2,
        )
        .unwrap();
        assert_eq!(result, Some((vec![10], vec![300])));
        assert_eq!(positions, array![2_i64, -1, 8]);
    }

    #[test]
    fn returns_none_when_no_position_survives() {
        let left = array![10_i64, 20];
        let right = array![100_i64, 200];
        let positions = array![0_i64, 1];
        let result =
            compare_batch_no_range_core(left.view(), right.view(), positions.view(), |_, _| false)
                .unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn rejects_mismatched_left_and_positions_lengths() {
        let left = array![10_i64];
        let right = array![100_i64];
        let positions = array![0_i64, 0];
        let error =
            compare_batch_no_range_core(left.view(), right.view(), positions.view(), |_, _| true)
                .unwrap_err();
        assert!(error.contains("left index and positions must have equal lengths"));
    }

    #[test]
    fn wrapper_matches_multiple_heterogeneous_predicates() {
        Python::initialize();
        Python::attach(|py| {
            let left_i64 = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let right_i64 = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let left_f32 = PyArray1::from_vec(py, vec![1.0_f32, 2.0, 3.0]);
            let right_f32 = PyArray1::from_vec(py, vec![0.0_f32, 1.0, 2.0]);
            let left_mask = PyArray1::from_vec(py, vec![false; 3]);
            let right_mask = PyArray1::from_vec(py, vec![false; 3]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left_i64.into_any(),
                    right_i64.into_any(),
                    4_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    left_f32.into_any(),
                    right_f32.into_any(),
                    5_i8.into_pyobject(py)?.into_any(),
                    left_mask.into_any(),
                    right_mask.into_any(),
                    1_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 200, 300]);
            let positions = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let result = compare_batch_no_range(
                py,
                &predicates,
                left_index.readonly(),
                right_index.readonly(),
                positions.readonly(),
            )?
            .expect("all aligned candidates should match");
            assert_eq!(result.0.readonly().as_array().to_vec(), vec![10, 20, 30]);
            assert_eq!(result.1.readonly().as_array().to_vec(), vec![100, 200, 300]);
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn wrapper_preserves_null_aware_extension_semantics() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![1_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64]);
            let left_mask = PyArray1::from_vec(py, vec![true]);
            let right_mask = PyArray1::from_vec(py, vec![false]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.into_any(),
                    5_i8.into_pyobject(py)?.into_any(),
                    left_mask.into_any(),
                    right_mask.into_any(),
                    1_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![100_i64]);
            let positions = PyArray1::from_vec(py, vec![0_i64]);
            let result = compare_batch_no_range(
                py,
                &predicates,
                left_index.readonly(),
                right_index.readonly(),
                positions.readonly(),
            )?;
            assert!(result.is_none());
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}
