//! Direct index construction for multi-condition non-equi regions.
//!
//! The two primary non-equi conditions are supplied as candidate bounds. Any
//! additional conditions are evaluated only inside those bounds. This keeps
//! the algorithm local: region construction remains pyjanitor's job, while
//! this module owns predicate filtering and final index materialisation.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::BTreeMap;

use crate::aggs::{ensure_equal_lengths, ensure_nonempty_core};
use crate::compare::common::{checked_bounds, Selection};
use crate::compare::predicate::{
    parse_predicates_with_nulls, predicates_match_dispatch, NullMetadata, Predicate,
};

type Indices<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

struct GroupState {
    head: i64,
    tail: i64,
}

impl Default for GroupState {
    fn default() -> Self {
        Self { head: -1, tail: -1 }
    }
}

/// Add the newly exposed right positions to reusable value groups.
///
/// ELI5: `starts` moves left as pyjanitor visits its rows. Instead of scanning
/// the already-seen suffix again to discover candidates, add only the newly
/// exposed positions to one flat chain per right-region value. Extra
/// predicates are still evaluated for the current left row when the chains
/// are traversed, because their result is row-specific.
fn add_right_region(
    right_region: ArrayView1<'_, i64>,
    start: usize,
    previous_end: usize,
    next: &mut [i64],
    groups: &mut BTreeMap<i64, GroupState>,
) {
    for right_position in (start..previous_end).rev() {
        let state = groups.entry(right_region[right_position]).or_default();
        if state.head == -1 {
            state.head = right_position as i64;
        } else {
            next[state.tail as usize] = right_position as i64;
        }
        state.tail = right_position as i64;
    }
}

struct ParsedInputs<'py> {
    predicates: Vec<Predicate<'py>>,
    metadata: Option<Vec<NullMetadata<'py>>>,
    left_len: usize,
    right_len: usize,
}

fn parse_inputs<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: &PyReadonlyArray1<'py, i64>,
    right_region: &PyReadonlyArray1<'py, i64>,
    starts: &PyReadonlyArray1<'py, i64>,
    left_index: &Bound<'py, PyArray1<i64>>,
    right_index: &PyReadonlyArray1<'py, i64>,
) -> PyResult<ParsedInputs<'py>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err(
            "at least one extra predicate is required",
        ));
    }
    let left_len = left_region.len()?;
    let right_len = right_region.len()?;
    ensure_nonempty_core("left_region", left_len).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right_region", right_len).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("starts", starts.len()?).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("left_index", left_index.len()?).map_err(PyValueError::new_err)?;
    ensure_nonempty_core("right_index", right_index.len()?).map_err(PyValueError::new_err)?;
    for predicate in &predicates {
        ensure_equal_lengths(
            "left region",
            left_len,
            "predicate left array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths(
            "right region",
            right_len,
            "predicate right array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    ensure_equal_lengths("left region", left_len, "starts", starts.len()?)
        .map_err(PyValueError::new_err)?;
    ensure_equal_lengths("left region", left_len, "left index", left_index.len()?)
        .map_err(PyValueError::new_err)?;
    ensure_equal_lengths("right region", right_len, "right index", right_index.len()?)
        .map_err(PyValueError::new_err)?;
    Ok(ParsedInputs {
        predicates,
        metadata,
        left_len,
        right_len,
    })
}

fn predicate_views<'a>(
    predicates: &'a [Predicate<'_>],
) -> Vec<crate::compare::predicate::PredicateView<'a>> {
    predicates.iter().map(Predicate::view).collect()
}

// Keep the inputs explicit at this boundary: each array has a distinct role
// in the pyjanitor region contract, and bundling them would make call sites
// less readable.
#[allow(clippy::too_many_arguments)]
fn selected_core<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
    selection: Selection,
) -> PyResult<Option<Indices<'py>>> {
    let ParsedInputs {
        predicates,
        metadata,
        left_len,
        right_len,
    } = parse_inputs(
        py,
        predicates,
        &left_region,
        &right_region,
        &starts,
        &left_index,
        &right_index,
    )?;
    let left_region = left_region.as_array();
    let right_region = right_region.as_array();
    let views = predicate_views(&predicates);
    let metadata = metadata.as_deref();
    let starts = starts.as_array();
    let right_values = right_index.as_array();
    let left_values = left_index.readonly();
    let left_values = left_values.as_array();
    let mut next = vec![-1_i64; right_len];
    let mut groups = BTreeMap::<i64, GroupState>::new();
    let mut previous_end = right_len;
    let mut output_left = Vec::new();
    let mut output_right = Vec::new();

    for row in 0..left_len {
        let Some((start, _)) = checked_bounds(starts[row], right_len as i64, right_len) else {
            continue;
        };
        debug_assert!(start <= previous_end);
        add_right_region(right_region, start, previous_end, &mut next, &mut groups);
        previous_end = start;

        let mut selected_position = None;
        'candidate_groups: for (_, state) in groups.range(left_region[row]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if !predicates_match_dispatch(&views, metadata, row, right_position) {
                    position = next[right_position];
                    continue;
                }
                match selection {
                    Selection::Any => {
                        selected_position = Some(right_position);
                        break 'candidate_groups;
                    }
                    Selection::First => {
                        if selected_position.is_none_or(|current| {
                            right_values[right_position] < right_values[current]
                        }) {
                            selected_position = Some(right_position);
                        }
                    }
                    Selection::Last => {
                        if selected_position.is_none_or(|current| {
                            right_values[right_position] > right_values[current]
                        }) {
                            selected_position = Some(right_position);
                        }
                    }
                }
                position = next[right_position];
            }
        }
        if let Some(right_position) = selected_position {
            output_left.push(left_values[row]);
            output_right.push(right_values[right_position]);
        }
    }

    if output_left.is_empty() {
        Ok(None)
    } else {
        Ok(Some((
            Array1::from_vec(output_left).into_pyarray(py),
            Array1::from_vec(output_right).into_pyarray(py),
        )))
    }
}

/// Return every matching pair using two passes over each candidate region.
///
/// The first pass records each row's first and last success and counts the
/// output. The second pass allocates exact-sized arrays and rechecks only the
/// interval between those successes. Null-aware six-element predicates use the
/// same semantics as the batch comparison path.
///
/// # Arguments
///
/// * `predicates` - Three- or six-element extra-predicate tuples.
/// * `left_region` - Second-condition region value for each left row.
/// * `right_region` - Second-condition region value for each right row.
/// * `starts` - First-condition candidate suffix start for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when no candidate passes every predicate.
#[pyfunction]
pub fn compare_multi_region_indices_all<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<Indices<'py>>> {
    let ParsedInputs {
        predicates,
        metadata,
        left_len,
        right_len,
    } = parse_inputs(
        py,
        predicates,
        &left_region,
        &right_region,
        &starts,
        &left_index,
        &right_index,
    )?;
    let left_region = left_region.as_array();
    let right_region = right_region.as_array();
    let views = predicate_views(&predicates);
    let metadata = metadata.as_deref();
    let starts = starts.as_array();
    let mut next = vec![-1_i64; right_len];
    let mut groups = BTreeMap::<i64, GroupState>::new();
    let mut previous_end = right_len;
    let mut first_success = vec![None; left_len];
    let mut last_success = vec![0_usize; left_len];
    let mut total = 0_usize;

    for row in 0..left_len {
        let Some((start, _)) = checked_bounds(starts[row], right_len as i64, right_len) else {
            continue;
        };
        debug_assert!(start <= previous_end);
        add_right_region(right_region, start, previous_end, &mut next, &mut groups);
        previous_end = start;
        for (_, state) in groups.range(left_region[row]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if predicates_match_dispatch(&views, metadata, row, right_position) {
                    if first_success[row].is_none() {
                        first_success[row] = Some(right_position);
                    }
                    last_success[row] = right_position;
                    total = total.checked_add(1).ok_or_else(|| {
                        PyValueError::new_err("number of output pairs exceeds usize")
                    })?;
                }
                position = next[right_position];
            }
        }
    }
    if total == 0 {
        return Ok(None);
    }

    let left_values = left_index.readonly();
    let left_values = left_values.as_array();
    let right_values = right_index.as_array();
    let mut output_left = Array1::<i64>::zeros(total);
    let mut output_right = Array1::<i64>::zeros(total);
    let mut output_position = 0;
    next.fill(-1);
    groups.clear();
    previous_end = right_len;

    for row in 0..left_len {
        let Some((start, _)) = checked_bounds(starts[row], right_len as i64, right_len) else {
            continue;
        };
        debug_assert!(start <= previous_end);
        add_right_region(right_region, start, previous_end, &mut next, &mut groups);
        previous_end = start;
        // no point checking if there is no match
        if first_success[row].is_none() {
            continue;
        }
        let first_success = first_success[row].unwrap();
        let last_success = last_success[row];
        for (_, state) in groups.range(left_region[row]..) {
            let mut position = state.head;
            while position >= 0 {
                let right_position = position as usize;
                if right_position < first_success || right_position > last_success {
                    position = next[right_position];
                    continue;
                }
                if predicates_match_dispatch(&views, metadata, row, right_position) {
                    output_left[output_position] = left_values[row];
                    output_right[output_position] = right_values[right_position];
                    output_position += 1;
                }
                position = next[right_position];
            }
        }
    }
    debug_assert_eq!(output_position, total);
    Ok(Some((
        output_left.into_pyarray(py),
        output_right.into_pyarray(py),
    )))
}

/// Select the smallest matching right label for each left row.
///
/// # Arguments
/// * `predicates` - Three- or six-element extra-predicate tuples.
/// * `left_region` - Second-condition region value for each left row.
/// * `right_region` - Second-condition region value for each right row.
/// * `starts` - First-condition candidate suffix start for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when no candidate passes every predicate.
#[pyfunction]
pub fn compare_multi_region_indices_first<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<Indices<'py>>> {
    selected_core(
        py,
        predicates,
        left_region,
        right_region,
        starts,
        left_index,
        right_index,
        Selection::First,
    )
}

/// Select the largest matching right label for each left row.
///
/// # Arguments
/// * `predicates` - Three- or six-element extra-predicate tuples.
/// * `left_region` - Second-condition region value for each left row.
/// * `right_region` - Second-condition region value for each right row.
/// * `starts` - First-condition candidate suffix start for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when no candidate passes every predicate.
#[pyfunction]
pub fn compare_multi_region_indices_last<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<Indices<'py>>> {
    selected_core(
        py,
        predicates,
        left_region,
        right_region,
        starts,
        left_index,
        right_index,
        Selection::Last,
    )
}

/// Select any matching right label for each left row.
///
/// # Arguments
/// * `predicates` - Three- or six-element extra-predicate tuples.
/// * `left_region` - Second-condition region value for each left row.
/// * `right_region` - Second-condition region value for each right row.
/// * `starts` - First-condition candidate suffix start for each left row.
/// * `left_index`, `right_index` - Labels emitted for matching positions.
///
/// # Returns
/// Aligned output arrays, or `None` when no candidate passes every predicate.
#[pyfunction]
pub fn compare_multi_region_indices_any<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    left_region: PyReadonlyArray1<'py, i64>,
    right_region: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<Indices<'py>>> {
    selected_core(
        py,
        predicates,
        left_region,
        right_region,
        starts,
        left_index,
        right_index,
        Selection::Any,
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_multi_region_indices_first, m)?)?;
    m.add_function(wrap_pyfunction!(compare_multi_region_indices_last, m)?)?;
    m.add_function(wrap_pyfunction!(compare_multi_region_indices_any, m)?)?;
    m.add_function(wrap_pyfunction!(compare_multi_region_indices_all, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::types::PyTuple;

    fn predicate_list<'py>(
        py: Python<'py>,
        left: &Bound<'py, PyArray1<i64>>,
        right: &Bound<'py, PyArray1<i64>>,
    ) -> Bound<'py, PyList> {
        let predicates = PyList::empty(py);
        predicates
            .append(
                PyTuple::new(
                    py,
                    [
                        left.clone().into_any(),
                        right.clone().into_any(),
                        3_i8.into_pyobject(py).unwrap().into_any(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        predicates
    }

    #[test]
    fn selected_modes_build_direct_indices() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let predicates = predicate_list(py, &left, &right);
            let left_region = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right_region = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 200, 300]);

            let first = compare_multi_region_indices_first(
                py,
                &predicates,
                left_region.readonly(),
                right_region.readonly(),
                starts.readonly(),
                left_index.clone(),
                right_index.readonly(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(first.0.readonly().as_array().to_vec(), vec![10, 20]);
            assert_eq!(first.1.readonly().as_array().to_vec(), vec![200, 300]);

            let any = compare_multi_region_indices_any(
                py,
                &predicates,
                left_region.readonly(),
                right_region.readonly(),
                starts.readonly(),
                left_index.clone(),
                right_index.readonly(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(any.1.readonly().as_array().to_vec(), vec![200, 300]);

            let last = compare_multi_region_indices_last(
                py,
                &predicates,
                left_region.readonly(),
                right_region.readonly(),
                starts.readonly(),
                left_index,
                right_index.readonly(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(last.1.readonly().as_array().to_vec(), vec![300, 300]);
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn all_uses_two_pass_output_and_null_metadata() {
        Python::initialize();
        Python::attach(|py| {
            let left = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let left_mask = PyArray1::from_vec(py, vec![false, true]);
            let right_mask = PyArray1::from_vec(py, vec![false, false, false]);
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        left.clone().into_any(),
                        right.clone().into_any(),
                        5_i8.into_pyobject(py).unwrap().into_any(),
                        left_mask.into_any(),
                        right_mask.into_any(),
                        0_i8.into_pyobject(py).unwrap().into_any(),
                    ],
                )?)
                .unwrap();
            let left_region = PyArray1::from_vec(py, vec![2_i64, 3]);
            let right_region = PyArray1::from_vec(py, vec![1_i64, 2, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0]);
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 200, 300]);
            let result = compare_multi_region_indices_all(
                py,
                &predicates,
                left_region.readonly(),
                right_region.readonly(),
                starts.readonly(),
                left_index,
                right_index.readonly(),
            )
            .unwrap()
            .unwrap();
            assert_eq!(result.0.readonly().as_array().to_vec(), vec![10, 20]);
            assert_eq!(result.1.readonly().as_array().to_vec(), vec![300, 300]);
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }

    #[test]
    fn heterogeneous_predicates_match_reference_for_every_selection() {
        Python::initialize();
        Python::attach(|py| {
            let left_region = PyArray1::from_vec(py, vec![3_i64, 4, 4]);
            let right_region = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4]);
            let starts = PyArray1::from_vec(py, vec![0_i64, 0, 0]);
            let left_index = PyArray1::from_vec(py, vec![10_i64, 20, 30]);
            let right_index = PyArray1::from_vec(py, vec![100_i64, 200, 300, 400]);

            let left_i64 = PyArray1::from_vec(py, vec![2_i64, 3, 4]);
            let right_i64 = PyArray1::from_vec(py, vec![1_i64, 2, 3, 4]);
            let left_f64 = PyArray1::from_vec(py, vec![1.5_f64, 2.5, 3.5]);
            let right_f64 = PyArray1::from_vec(py, vec![1.0_f64, 2.0, 3.0, 4.0]);
            let left_i32 = PyArray1::from_vec(py, vec![0_i32, 0, 0]);
            let right_i32 = PyArray1::from_vec(py, vec![0_i32, 1, 2, 3]);
            let left_mask = PyArray1::from_vec(py, vec![false, false, false]);
            let right_mask = PyArray1::from_vec(py, vec![false, false, true, false]);
            let predicates = PyList::empty(py);

            // Three different dtypes and operators exercise the heterogeneous
            // dispatch. The final six-element predicate also exercises
            // nullable extension-array semantics: right position 2 is masked
            // and therefore cannot satisfy `!=`.
            predicates.append(PyTuple::new(
                py,
                [
                    left_i64.into_any(),
                    right_i64.into_any(),
                    3_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    left_f64.into_any(),
                    right_f64.into_any(),
                    5_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    left_i32.into_any(),
                    right_i32.into_any(),
                    5_i8.into_pyobject(py).unwrap().into_any(),
                    left_mask.into_any(),
                    right_mask.into_any(),
                    1_i8.into_pyobject(py).unwrap().into_any(),
                ],
            )?)?;

            // Reference: the region leaves positions 2 and 3 for row 0, 3
            // for row 1, and 3 for row 2. The masked != predicate removes
            // position 2, leaving one matching right label per left row.
            let expected_left = vec![10_i64, 20, 30];
            let expected_right = vec![400_i64, 400, 400];
            for call in [
                compare_multi_region_indices_first,
                compare_multi_region_indices_last,
                compare_multi_region_indices_any,
                compare_multi_region_indices_all,
            ] {
                let result = call(
                    py,
                    &predicates,
                    left_region.readonly(),
                    right_region.readonly(),
                    starts.readonly(),
                    left_index.clone(),
                    right_index.readonly(),
                )?
                .unwrap();
                assert_eq!(result.0.readonly().as_array().to_vec(), expected_left);
                assert_eq!(result.1.readonly().as_array().to_vec(), expected_right);
            }
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}
