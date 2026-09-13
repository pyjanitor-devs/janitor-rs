//! Python-facing fused wrappers for heterogeneous residual predicates.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::common::{checked_bounds, Selection};
use crate::aggs::ensure_equal_lengths;

type BatchIndices<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

use super::predicate::{
    parse_predicates_with_nulls, predicates_match_dispatch, NullMetadata, Predicate,
};

struct ParsedBatch<'py> {
    predicates: Vec<Predicate<'py>>,
    metadata: Option<Vec<NullMetadata<'py>>>,
    left_len: usize,
    right_len: usize,
}

fn parse_and_validate<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: &Option<PyReadonlyArray1<'py, i64>>,
    ends: &Option<PyReadonlyArray1<'py, i64>>,
    left_index: &Bound<'py, PyArray1<i64>>,
    right_index: &PyReadonlyArray1<'py, i64>,
) -> PyResult<ParsedBatch<'py>> {
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
    if let Some(values) = starts {
        ensure_equal_lengths("starts", values.len()?, "left predicate array", left_len)?;
    }
    if let Some(values) = ends {
        ensure_equal_lengths("ends", values.len()?, "left predicate array", left_len)?;
    }
    Ok(ParsedBatch {
        predicates,
        metadata,
        left_len,
        right_len,
    })
}

/// Run a heterogeneous predicate batch and return one expanded pair per
/// left row that has at least one successful right candidate.
///
/// ELI5: first/last/any save one winning position per left row, then the final
/// pass copies those saved results. `All` has a separate two-pass implementation
/// because retaining every pair would duplicate the final output in memory.
fn compare_batch_indices_with_selection<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
    selection: Selection,
) -> PyResult<Option<BatchIndices<'py>>> {
    let ParsedBatch {
        predicates,
        metadata,
        left_len,
        right_len,
    } = parse_and_validate(py, predicates, &starts, &ends, &left_index, &right_index)?;

    let mut views = Vec::with_capacity(predicates.len());
    for predicate in &predicates {
        views.push(predicate.view());
    }
    let starts_view = starts.as_ref().map(|values| values.as_array());
    let ends_view = ends.as_ref().map(|values| values.as_array());
    let right_values = right_index.as_array();
    // `None` means that this left row has no selected right position. Using
    // `Option<usize>` avoids reserving a special numeric position as a
    // sentinel; valid positions remain ordinary `usize` values.
    let mut selected = vec![None; left_len];
    let mut total = 0_usize;

    // The only comparison pass uses the original boundaries. `First` and
    // `Last` deliberately scan the complete candidate range: their contracts
    // are the smallest and largest matching right *labels*, respectively;
    // right_index is not required to be sorted. We still save the positional
    // slot that owns the selected label because starts/ends and the output
    // arrays use positions internally. `Any` only needs one successful
    // position, so it can stop immediately.
    for row in 0..left_len {
        // `map_or` reads one scalar boundary: it uses this row's value when
        // a view exists, otherwise the full-range default. It does not create
        // an iterator. The next bindings intentionally shadow those scalar
        // i64 values after converting them to the indexing type, keeping the
        // hot loop concise without allocating temporaries.
        let start = starts_view.as_ref().map_or(0, |values| values[row]);
        let end = ends_view
            .as_ref()
            .map_or(right_len as i64, |values| values[row]);
        let Some((start, end)) = checked_bounds(start, end, right_len) else {
            // This row has no candidates. Skip it so valid ranges elsewhere
            // in the batch can still produce output.
            continue;
        };

        let mut selected_position = None;
        for right_position in start..end {
            if predicates_match_dispatch(&views, metadata.as_deref(), row, right_position) {
                match &selection {
                    Selection::First => {
                        if selected_position.is_none()
                            || right_values[right_position]
                                < right_values[selected_position.unwrap()]
                        {
                            selected_position = Some(right_position);
                        }
                    }
                    Selection::Last => {
                        if selected_position.is_none()
                            || right_values[right_position]
                                > right_values[selected_position.unwrap()]
                        {
                            selected_position = Some(right_position);
                        }
                    }
                    Selection::Any => {
                        selected_position = Some(right_position);
                        break;
                    }
                }
            }
        }

        if let Some(selected_position) = selected_position {
            selected[row] = Some(selected_position);
            total += 1;
        }
    }

    if total == 0 {
        return Ok(None);
    }

    let mut expanded_left = Array1::<i64>::zeros(total);
    let mut expanded_right = Array1::<i64>::zeros(total);
    // ELI5: the first binding is the read-only NumPy guard; the second
    // binding shadows it with the lightweight Rust view. The view does not
    // copy the labels, so the guard must remain alive in this scope while the
    // view borrows from the underlying Python array.
    let left_values = left_index.readonly();
    let left_values = left_values.as_array();
    let mut output_position = 0_usize;

    for row in 0..left_len {
        if let Some(right_position) = selected[row] {
            expanded_left[output_position] = left_values[row];
            expanded_right[output_position] = right_values[right_position];
            output_position += 1;
        }
    }
    debug_assert_eq!(output_position, total);
    Ok(Some((
        expanded_left.into_pyarray(py),
        expanded_right.into_pyarray(py),
    )))
}

/// Return every matching pair using a bounded two-pass scan.
///
/// The first pass records each row's first and last successful candidate and
/// counts all successes. The second pass can then allocate exact-sized output
/// arrays and revisit only the interval between those two successes. This
/// keeps `All` separate from the one-result selection modes and avoids keeping
/// a full intermediate `Vec<(row, position)>` alive beside the final arrays.
fn compare_batch_indices_all_two_pass<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<BatchIndices<'py>>> {
    let ParsedBatch {
        predicates,
        metadata,
        left_len,
        right_len,
    } = parse_and_validate(py, predicates, &starts, &ends, &left_index, &right_index)?;
    let mut views = Vec::with_capacity(predicates.len());
    for predicate in &predicates {
        views.push(predicate.view());
    }
    let starts_view = starts.as_ref().map(|values| values.as_array());
    let ends_view = ends.as_ref().map(|values| values.as_array());
    let mut first_success = vec![None; left_len];
    let mut last_success = vec![0_usize; left_len];
    let mut total = 0_usize;

    // ELI5: first find the first and last winning bookend for each row.
    // Everything between those bookends is the only part the second pass
    // needs to inspect; candidates outside them already failed.
    for row in 0..left_len {
        let start = starts_view.as_ref().map_or(0, |values| values[row]);
        let end = ends_view
            .as_ref()
            .map_or(right_len as i64, |values| values[row]);
        let Some((start, end)) = checked_bounds(start, end, right_len) else {
            continue;
        };
        for right_position in start..end {
            if predicates_match_dispatch(&views, metadata.as_deref(), row, right_position) {
                if first_success[row].is_none() {
                    first_success[row] = Some(right_position);
                }
                last_success[row] = right_position;
                total += 1;
            }
        }
    }

    if total == 0 {
        return Ok(None);
    }
    let mut expanded_left = Array1::<i64>::zeros(total);
    let mut expanded_right = Array1::<i64>::zeros(total);
    let left_values = left_index.readonly();
    let left_values = left_values.as_array();
    let right_values = right_index.as_array();
    let mut output_position = 0_usize;

    // ELI5: rescan only from the first win through the last win. This still
    // emits every successful pair, but avoids comparing the known-failing
    // prefix and suffix of a row a second time.
    for row in 0..left_len {
        let Some(start) = first_success[row] else {
            continue;
        };
        let end = last_success[row] + 1;
        for right_position in start..end {
            if predicates_match_dispatch(&views, metadata.as_deref(), row, right_position) {
                expanded_left[output_position] = left_values[row];
                expanded_right[output_position] = right_values[right_position];
                output_position += 1;
            }
        }
    }
    debug_assert_eq!(output_position, total);
    Ok(Some((
        expanded_left.into_pyarray(py),
        expanded_right.into_pyarray(py),
    )))
}

/// Select the smallest matching right label for each left row.
///
/// ELI5: all predicates must approve a candidate; keep the approved candidate
/// with the smallest right label. Returns `None` when nothing matches.
///
/// # Arguments
/// * `predicates` - Heterogeneous comparison tuples.
/// * `starts`, `ends` - Optional candidate bounds for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when there are no matches.
#[pyfunction]
pub fn compare_batch_indices_first<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<BatchIndices<'py>>> {
    compare_batch_indices_with_selection(
        py,
        predicates,
        starts,
        ends,
        left_index,
        right_index,
        Selection::First,
    )
}

/// Select the largest matching right label for each left row.
///
/// ELI5: all predicates must approve a candidate; keep the approved candidate
/// with the largest right label. Returns `None` when nothing matches.
///
/// # Arguments
/// * `predicates` - Heterogeneous comparison tuples.
/// * `starts`, `ends` - Optional candidate bounds for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when there are no matches.
#[pyfunction]
pub fn compare_batch_indices_last<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<BatchIndices<'py>>> {
    compare_batch_indices_with_selection(
        py,
        predicates,
        starts,
        ends,
        left_index,
        right_index,
        Selection::Last,
    )
}

/// Select any matching right position for each left row.
///
/// ELI5: stop scanning a row as soon as one candidate passes every predicate.
/// Returns `None` when nothing matches.
///
/// # Arguments
/// * `predicates` - Heterogeneous comparison tuples.
/// * `starts`, `ends` - Optional candidate bounds for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when there are no matches.
#[pyfunction]
pub fn compare_batch_indices_any<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<BatchIndices<'py>>> {
    compare_batch_indices_with_selection(
        py,
        predicates,
        starts,
        ends,
        left_index,
        right_index,
        Selection::Any,
    )
}

/// Return every left/right pair passing every predicate.
///
/// ELI5: keep every candidate approved by all judges instead of choosing one.
/// Returns `None` when nothing matches.
///
/// # Arguments
/// * `predicates` - Heterogeneous comparison tuples.
/// * `starts`, `ends` - Optional candidate bounds for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when there are no matches.
#[pyfunction]
pub fn compare_batch_indices_all<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<BatchIndices<'py>>> {
    compare_batch_indices_all_two_pass(py, predicates, starts, ends, left_index, right_index)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_batch_indices_first, m)?)?;
    m.add_function(wrap_pyfunction!(compare_batch_indices_last, m)?)?;
    m.add_function(wrap_pyfunction!(compare_batch_indices_any, m)?)?;
    m.add_function(wrap_pyfunction!(compare_batch_indices_all, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    #[test]
    fn first_indices_return_one_label_per_successful_left_row() {
        Python::initialize();
        Python::attach(|py| {
            let left_i64 = PyArray1::from_vec(py, vec![3_i64, 5]);
            let right_i64 = PyArray1::from_vec(py, vec![1_i64, 5, 2]);
            let left_f32 = PyArray1::from_vec(py, vec![3.0_f32, 5.0]);
            let right_f32 = PyArray1::from_vec(py, vec![2.0_f32, 4.0, 5.0]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left_i64.into_any(),
                    right_i64.into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    left_f32.into_any(),
                    right_f32.into_any(),
                    2_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let ends = PyArray1::from_vec(py, vec![3_i64, 3]);
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![102_i64, 101, 100]);

            let Some((expanded_left, expanded_right)) = compare_batch_indices_first(
                py,
                &predicates,
                Some(starts.readonly()),
                Some(ends.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )?
            else {
                panic!("expected one successful pair");
            };
            assert_eq!(expanded_left.readonly().as_array().to_vec(), vec![10]);
            assert_eq!(expanded_right.readonly().as_array().to_vec(), vec![100]);
            // Boundaries are read-only inputs; selected positions are kept in
            // the result state instead of being written back to the caller.
            assert_eq!(starts.readonly().as_array().to_vec(), vec![0, 0]);
            assert_eq!(ends.readonly().as_array().to_vec(), vec![3, 3]);

            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn last_and_any_select_the_expected_right_positions() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![3_i64, 5]);
            let right = PyArray1::from_vec(py, vec![1_i64, 5, 2]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![102_i64, 101, 100]);

            let last = compare_batch_indices_last(
                py,
                &predicates,
                None,
                None,
                left_index.clone(),
                right_index.readonly(),
            )?
            .unwrap();
            assert_eq!(last.0.readonly().as_array().to_vec(), vec![10, 20]);
            assert_eq!(last.1.readonly().as_array().to_vec(), vec![102, 102]);

            let any = compare_batch_indices_any(
                py,
                &predicates,
                None,
                None,
                left_index.clone(),
                right_index.readonly(),
            )?
            .unwrap();
            assert_eq!(any.0.readonly().as_array().to_vec(), vec![10, 20]);
            assert_eq!(any.1.readonly().as_array().to_vec(), vec![102, 102]);
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn all_indices_return_every_successful_pair() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![3_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 5, 2]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![102_i64, 100, 101]);
            let result = compare_batch_indices_all(
                py,
                &predicates,
                None,
                None,
                left_index,
                right_index.readonly(),
            )?
            .unwrap();
            assert_eq!(result.0.readonly().as_array().to_vec(), vec![10, 10]);
            assert_eq!(result.1.readonly().as_array().to_vec(), vec![102, 101]);
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn all_indices_keep_matches_between_first_and_last_success() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![3_i64]);
            let right = PyArray1::from_vec(py, vec![5_i64, 1, 4, 2]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![4_i64]);
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 101, 102, 103]);
            let result = compare_batch_indices_all(
                py,
                &predicates,
                Some(starts.readonly()),
                Some(ends.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )?
            .unwrap();
            assert_eq!(result.0.readonly().as_array().to_vec(), vec![10, 10]);
            assert_eq!(result.1.readonly().as_array().to_vec(), vec![101, 103]);
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn two_pass_indices_return_none_when_no_predicate_matches() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![1_i64, 2]);
            let right = PyArray1::from_vec(py, vec![3_i64, 4]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 101]);

            let result = compare_batch_indices_any(
                py,
                &predicates,
                None,
                None,
                left_index,
                right_index.readonly(),
            )?;
            assert!(result.is_none());
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn boundary_ranges_and_empty_inputs_follow_the_batch_contract() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![3_i64]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21]);

            // `start == right_len` is a valid empty half-open range.
            let start_at_end = PyArray1::from_vec(py, vec![2_i64]);
            let end_at_end = PyArray1::from_vec(py, vec![2_i64]);
            assert!(compare_batch_indices_any(
                py,
                &predicates,
                Some(start_at_end.readonly()),
                Some(end_at_end.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )?
            .is_none());
            // A reversed range is empty for this row.
            let reversed_start = PyArray1::from_vec(py, vec![2_i64]);
            let reversed_end = PyArray1::from_vec(py, vec![1_i64]);
            assert!(compare_batch_indices_any(
                py,
                &predicates,
                Some(reversed_start.readonly()),
                Some(reversed_end.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )?
            .is_none());
            assert!(compare_batch_indices_all(
                py,
                &predicates,
                Some(reversed_start.readonly()),
                Some(reversed_end.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )?
            .is_none());

            let oversized_end = PyArray1::from_vec(py, vec![3_i64]);
            let full_start = PyArray1::from_vec(py, vec![0_i64]);
            let error = compare_batch_indices_all(
                py,
                &predicates,
                Some(full_start.readonly()),
                Some(oversized_end.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )
            .expect("an oversized All boundary should skip only that row");
            assert!(error.is_none());
            let negative_start = PyArray1::from_vec(py, vec![-1_i64]);
            let result = compare_batch_indices_all(
                py,
                &predicates,
                Some(negative_start.readonly()),
                Some(end_at_end.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )
            .expect("a negative All boundary should skip only that row");
            assert!(result.is_none());

            // A malformed row must not discard a valid match from another
            // row in the same batch.
            let left = PyArray1::from_vec(py, vec![3_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let starts = PyArray1::from_vec(py, vec![0_i64, -1]);
            let ends = PyArray1::from_vec(py, vec![1_i64, 999]);
            let left_index = PyArray1::from_vec(py, vec![10_i64, 11]);
            let right_index = PyArray1::from_vec(py, vec![20_i64, 21]);
            let result = compare_batch_indices_any(
                py,
                &predicates,
                Some(starts.readonly()),
                Some(ends.readonly()),
                left_index.clone(),
                right_index.readonly(),
            )?
            .unwrap();
            assert_eq!(result.0.readonly().as_array().to_vec(), vec![10]);
            assert_eq!(result.1.readonly().as_array().to_vec(), vec![20]);

            // Empty left arrays have no candidate rows and therefore no
            // output, while still satisfying the parallel-length contract.
            let empty_left = PyArray1::from_vec(py, Vec::<i64>::new());
            let empty_right = PyArray1::from_vec(py, Vec::<i64>::new());
            let empty_predicates = PyList::empty(py);
            empty_predicates.append(PyTuple::new(
                py,
                [
                    empty_left.clone().into_any(),
                    empty_right.clone().into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let empty_index = PyArray1::from_vec(py, Vec::<i64>::new());
            assert!(compare_batch_indices_any(
                py,
                &empty_predicates,
                None,
                None,
                empty_index.clone(),
                empty_index.readonly(),
            )?
            .is_none());

            let no_predicates = PyList::empty(py);
            let error = compare_batch_indices_any(
                py,
                &no_predicates,
                None,
                None,
                left_index,
                right_index.readonly(),
            )
            .expect_err("an empty predicate list must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: at least one comparison is required"
            );
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn six_element_not_equal_predicates_follow_null_dispatch() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![1_i64]);
            let right = PyArray1::from_vec(py, vec![2_i64]);
            let left_booleans = PyArray1::from_vec(py, vec![true]);
            let right_booleans = PyArray1::from_vec(py, vec![false]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    5_i8.into_pyobject(py)?.into_any(),
                    left_booleans.into_any(),
                    right_booleans.into_any(),
                    1_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let left_index = PyArray1::from_vec(py, vec![10_i64]);
            let right_index = PyArray1::from_vec(py, vec![20_i64]);
            assert!(compare_batch_indices_any(
                py,
                &predicates,
                None,
                None,
                left_index.clone(),
                right_index.readonly(),
            )?
            .is_none());
            assert!(compare_batch_indices_all(
                py,
                &predicates,
                None,
                None,
                left_index.clone(),
                right_index.readonly(),
            )?
            .is_none());

            let left_booleans = PyArray1::from_vec(py, vec![true]);
            let right_booleans = PyArray1::from_vec(py, vec![false]);
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    left.into_any(),
                    right.into_any(),
                    5_i8.into_pyobject(py)?.into_any(),
                    left_booleans.into_any(),
                    right_booleans.into_any(),
                    0_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            assert!(compare_batch_indices_any(
                py,
                &predicates,
                None,
                None,
                left_index.clone(),
                right_index.readonly(),
            )?
            .is_some());
            assert!(compare_batch_indices_all(
                py,
                &predicates,
                None,
                None,
                left_index,
                right_index.readonly(),
            )?
            .is_some());
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn validation_errors_report_the_mismatched_names_and_lengths() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![1_i64, 2]);
            let right = PyArray1::from_vec(py, vec![1_i64]);
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![30_i64]);
            let make_predicates = || {
                let predicates = PyList::empty(py);
                predicates
                    .append(PyTuple::new(
                        py,
                        [
                            left.clone().into_any(),
                            right.clone().into_any(),
                            5_i8.into_pyobject(py)?.into_any(),
                        ],
                    )?)?;
                Ok::<Bound<'_, PyList>, PyErr>(predicates)
            };

            let predicates = PyList::empty(py);
            let short_left_mask = PyArray1::from_vec(py, vec![false]);
            let short_right_mask = PyArray1::from_vec(py, vec![false]);
            predicates.append(PyTuple::new(
                py,
                [
                    left.clone().into_any(),
                    right.clone().into_any(),
                    5_i8.into_pyobject(py)?.into_any(),
                    short_left_mask.into_any(),
                    short_right_mask.into_any(),
                    1_i8.into_pyobject(py)?.into_any(),
                ],
            )?)?;
            let error = compare_batch_indices_any(
                py,
                &predicates,
                None,
                None,
                left_index.clone(),
                right_index.readonly(),
            )
            .expect_err("a short left mask must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: left boolean mask and left predicate array must have equal lengths; got 1 and 2"
            );

            let predicates = make_predicates()?;
            let short_left_index = PyArray1::from_vec(py, vec![10_i64]);
            let error = compare_batch_indices_any(
                py,
                &predicates,
                None,
                None,
                short_left_index,
                right_index.readonly(),
            )
            .expect_err("a short left index must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: left index and left predicate array must have equal lengths; got 1 and 2"
            );

            let predicates = make_predicates()?;
            let short_starts = PyArray1::from_vec(py, vec![0_i64]);
            let ends = PyArray1::from_vec(py, vec![1_i64, 1]);
            let error = compare_batch_indices_any(
                py,
                &predicates,
                Some(short_starts.readonly()),
                Some(ends.readonly()),
                left_index,
                right_index.readonly(),
            )
            .expect_err("short starts must be rejected");
            assert_eq!(
                error.to_string(),
                "ValueError: starts and left predicate array must have equal lengths; got 1 and 2"
            );

            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}
