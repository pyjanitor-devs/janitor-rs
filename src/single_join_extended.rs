//! Multiple conditional-join predicates built on the single-join kernels.
//!
//! Mixed joins must put a range comparison first. It creates compact half-open
//! right-side windows; every later predicate filters candidates in those
//! windows. When every predicate is `!=`, the first predicate may be `!=` and
//! creates flat physical position pairs with the single-join not-equal core.
//! Later predicates filter those pairs directly.
//!
//! Both paths apply `keep` only after every predicate passes. Rust trusts
//! pyjanitor to classify and order predicates, align residual arrays, sort the
//! first range/right value array when required, and provide authoritative null
//! metadata.

use numpy::{ndarray::ArrayView1, IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::op::CompareOp;
use crate::predicate::{
    null_metadata_views, parse_predicates_with_nulls_strings, predicates_match_dispatch,
    NullMetadata, Predicate,
};
use crate::single_join::{
    build_not_equal_positions_core, build_range_core, Keep, SingleJoinResult,
};

fn materialize_windows_for_non_ne(
    windows: &SingleJoinResult,
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);
    let labels = windows.right_index.as_slice();

    // Follow the established batch-indices layout: selection modes first find
    // at most one winning physical position per left row, while `all` uses a
    // count pass followed by an exact materialization pass. The latter avoids
    // reserving the full range-window upper bound when residual predicates
    // eliminate many candidates.
    if keep == Keep::All {
        let mut output_len = 0_usize;
        for row in 0..windows.left_index.len() {
            let start = windows.starts[row];
            let end = windows.ends[row];
            let left_position = windows.left_positions[row];
            for right_position in start..end {
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
                    output_len = output_len
                        .checked_add(1)
                        .ok_or("single extended join result size exceeds platform capacity")?;
                }
            }
        }
        if output_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;

        for row in 0..windows.left_index.len() {
            let start = windows.starts[row];
            let end = windows.ends[row];
            let left_position = windows.left_positions[row];
            for (offset, &label) in labels[start..end].iter().enumerate() {
                let right_position = start + offset;
                if predicates_match_dispatch(
                    &views,
                    metadata_views.as_deref(),
                    left_position,
                    right_position,
                ) {
                    output_left.push(windows.left_index[row]);
                    output_right.push(label);
                }
            }
        }
        debug_assert_eq!(output_left.len(), output_len);
        debug_assert_eq!(output_right.len(), output_len);
        return Ok((output_left, output_right));
    }

    // Each selection mode emits no more than one pair for each retained left
    // row. Reserving the number of range-window rows is therefore a tight
    // upper bound and does not require a count pass.
    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(windows.left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(windows.left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    for row in 0..windows.left_index.len() {
        let start = windows.starts[row];
        let end = windows.ends[row];
        let left_position = windows.left_positions[row];
        let mut selected = None;
        for (offset, &label) in labels[start..end].iter().enumerate() {
            let right_position = start + offset;
            if !predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                continue;
            }
            match keep {
                Keep::Any => {
                    selected = Some(right_position);
                    break;
                }
                //                   The logic is:

                //   if nothing is selected:
                //       select this candidate

                //   otherwise:
                //       replace the current candidate only if this label is smaller

                //   Example:

                //   labels = [40, 10, 30]

                //   Candidates arrive in this order:

                //   position 0, label 40 → selected = Some(0)
                //   position 1, label 10 → 10 < 40, selected = Some(1)
                //   position 2, label 30 → 30 < 10 is false, keep Some(1)
                Keep::First => {
                    let replace = match selected {
                        None => true,
                        Some(current) => label < labels[current],
                    };
                    if replace {
                        selected = Some(right_position);
                    }
                }
                //   - selected stores the currently chosen right position: Option<usize>.
                //   - right_position is the new candidate’s physical position.
                //   - label is the new candidate’s right-index value.
                //   - current is the previously selected physical position.
                //   - labels[current] is the previously selected right-index value.

                //   Example:

                //   right positions:  0    1    2
                //   right labels:    40   10   30

                //   Processing candidates:

                //   1. Position 0, label 40
                //       - Nothing selected yet.
                //       - Select position 0.

                //   2. Position 1, label 10
                //       - 10 > 40 is false.
                //       - Keep position 0.

                //   3. Position 2, label 30
                //       - 30 > 40 is false.
                //       - Keep position 0.

                //   Final selection:

                //   selected = Some(0)
                Keep::Last => {
                    let replace = match selected {
                        None => true,
                        Some(current) => label > labels[current],
                    };
                    if replace {
                        selected = Some(right_position);
                    }
                }
                Keep::All => unreachable!(),
            }
        }
        if let Some(right_position) = selected {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[right_position]);
        }
    }
    if output_left.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    Ok((output_left, output_right))
}

/// Filter and materialize flat physical pairs produced by a first `!=`
/// predicate.
///
/// The pair positions index the full physical left and right layouts supplied
/// for the residual predicates. Public index labels are not produced until
/// after every residual predicate has passed.
///
/// `all` uses two passes: the first counts surviving pairs exactly and the
/// second allocates that exact size and writes the labels. The selected modes
/// retain at most one pair per left row and therefore reserve the full left
/// index length as an upper bound.
fn materialize_pairs_for_ne(
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    left_positions: &[usize],
    right_positions: &[usize],
    predicates: &[Predicate<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    ensure_equal_lengths_core(
        "not-equal left positions",
        left_positions.len(),
        "not-equal right positions",
        right_positions.len(),
    )?;
    let views: Vec<_> = predicates.iter().map(Predicate::view).collect();
    let metadata_views = metadata.map(null_metadata_views);

    if keep == Keep::All {
        // First count the exact number of survivors. This avoids reserving a
        // potentially huge upper bound when residual predicates reject most
        // of the first `!=` candidate pairs.
        // Count survivors by physical left row so the final output can retain
        // left input order even when the first not-equal core processed
        // non-null and null left rows separately.
        let mut counts_by_left: Vec<usize> = Vec::new();
        counts_by_left
            .try_reserve_exact(left_index.len())
            .map_err(|_| "single extended join result allocation failed")?;
        counts_by_left.resize(left_index.len(), 0_usize);
        for (&left_position, &right_position) in left_positions.iter().zip(right_positions) {
            if predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                let count = counts_by_left
                    .get_mut(left_position)
                    .ok_or("not-equal left position is out of bounds")?;
                *count = (*count)
                    .checked_add(1)
                    .ok_or("single extended join result size exceeds platform capacity")?;
            }
        }

        // Turn each count into the starting slot for that left row. The
        // resulting offsets let the second pass write directly into its
        // preallocated section without storing all matched positions.
        let mut output_len = 0_usize;
        for count in &mut counts_by_left {
            let row_count = *count;
            *count = output_len;
            output_len = output_len
                .checked_add(row_count)
                .ok_or("single extended join result size exceeds platform capacity")?;
        }
        if output_len == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        output_left.resize(output_len, 0);
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_len)
            .map_err(|_| "single extended join result allocation failed")?;
        output_right.resize(output_len, 0);

        // Copy the row starts into cursors. The second pass evaluates every
        // candidate again, then advances only that left row's cursor.
        let mut write_positions = Vec::new();
        write_positions
            .try_reserve_exact(counts_by_left.len())
            .map_err(|_| "single extended join result allocation failed")?;
        write_positions.extend_from_slice(&counts_by_left);
        for (&left_position, &right_position) in left_positions.iter().zip(right_positions) {
            if predicates_match_dispatch(
                &views,
                metadata_views.as_deref(),
                left_position,
                right_position,
            ) {
                let output_position = write_positions
                    .get_mut(left_position)
                    .ok_or("not-equal left position is out of bounds")?;
                let slot = *output_position;
                *output_position += 1;
                output_left[slot] = left_index[left_position];
                output_right[slot] = *right_index
                    .get(right_position)
                    .ok_or("not-equal right position is out of bounds")?;
            }
        }
        return Ok((output_left, output_right));
    }

    // A selected mode emits no more than one pair for each left row. Store the
    // winning physical right position by left position, then walk the full
    // left layout so selected output retains left input order.
    let mut selected = vec![None; left_index.len()];
    for (&left_position, &right_position) in left_positions.iter().zip(right_positions) {
        if !predicates_match_dispatch(
            &views,
            metadata_views.as_deref(),
            left_position,
            right_position,
        ) {
            continue;
        }
        let selected_right = selected
            .get_mut(left_position)
            .ok_or("not-equal left position is out of bounds")?;
        match keep {
            Keep::Any => {
                if selected_right.is_none() {
                    *selected_right = Some(right_position);
                }
            }
            Keep::First | Keep::Last => {
                let replace = match *selected_right {
                    None => true,
                    Some(current) => {
                        let value = right_index[right_position];
                        let current_value = right_index[current];
                        if keep == Keep::First {
                            value < current_value
                        } else {
                            value > current_value
                        }
                    }
                };
                if replace {
                    *selected_right = Some(right_position);
                }
            }
            Keep::All => unreachable!(),
        }
    }

    let mut output_left = Vec::new();
    output_left
        .try_reserve_exact(left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    let mut output_right = Vec::new();
    output_right
        .try_reserve_exact(left_index.len())
        .map_err(|_| "single extended join result allocation failed")?;
    for (left_position, right_position) in selected.into_iter().enumerate() {
        if let Some(right_position) = right_position {
            output_left.push(left_index[left_position]);
            output_right.push(
                *right_index
                    .get(right_position)
                    .ok_or("not-equal right position is out of bounds")?,
            );
        }
    }
    Ok((output_left, output_right))
}

fn result_dict<'py>(
    py: Python<'py>,
    left: Vec<i64>,
    right: Vec<i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("left_index", left.into_pyarray(py))?;
    result.set_item("right_index", right.into_pyarray(py))?;
    Ok(result)
}

/// Execute an all-`!=` extended join.
///
/// The first predicate supplies filtered non-null values plus physical
/// position maps and optional null positions. It is expanded with `Keep::All`
/// because later predicates must see every first-stage candidate. Residual
/// predicates use full-layout arrays and masks, so their physical positions
/// can be indexed directly. Public labels are materialized only after all
/// residual predicates pass.
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
    right_index_is_ordered: bool,
    is_extension_array: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended join requires at least two predicates",
        ));
    }
    let keep = Keep::parse(keep)?;
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        let tuple = item
            .cast::<PyTuple>()
            .map_err(|_| PyValueError::new_err("each residual comparison must be a tuple"))?;
        let op_position = if tuple.len() == 3 {
            2
        } else if tuple.len() == 6 {
            5
        } else {
            return Err(PyValueError::new_err(
                "each residual comparison must contain 3 or 6 elements",
            ));
        };
        let op = CompareOp::try_from_str(tuple.get_item(op_position)?.extract::<&str>()?)?;
        if op != CompareOp::Ne {
            return Err(PyValueError::new_err(
                "all-!= joins require every predicate to use !=",
            ));
        }
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    let left_index = first_left_index.as_array();
    let right_index = first_right_index.as_array();

    // Residual predicates use the full physical layouts, so their lengths
    // must match the full indexes rather than the filtered first-predicate
    // value arrays.
    for predicate in &parsed {
        ensure_equal_lengths_core(
            "full left index",
            left_index.len(),
            "residual left predicate array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core(
            "full right index",
            right_index.len(),
            "residual right predicate array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }

    // The first `!=` predicate is always expanded fully. Applying `keep`
    // here would discard pairs needed by later predicates.
    let (left_positions, right_positions) = build_not_equal_positions_core(
        first_left.as_array(),
        left_index,
        first_left_positions.as_array(),
        first_right.as_array(),
        right_index,
        first_right_positions.as_array(),
        first_left_null_positions
            .as_ref()
            .map(|values| values.as_array()),
        first_right_null_positions
            .as_ref()
            .map(|values| values.as_array()),
        right_index_is_ordered,
        is_extension_array,
        Keep::All,
    )
    .map_err(PyValueError::new_err)?;
    if left_positions.is_empty() {
        return Ok(None);
    }

    let (out_left, out_right) = materialize_pairs_for_ne(
        left_index,
        right_index,
        &left_positions,
        &right_positions,
        &parsed,
        metadata.as_deref(),
        keep,
    )
    .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, out_left, out_right)?))
}

#[allow(clippy::too_many_arguments)]
fn extended_join<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first_left: PyReadonlyArray1<'py, T>,
    first_left_index: PyReadonlyArray1<'py, i64>,
    first_right: PyReadonlyArray1<'py, T>,
    first_right_index: PyReadonlyArray1<'py, i64>,
    right_index_is_ordered: bool,
    first_op: CompareOp,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "single extended join requires at least two predicates",
        ));
    }
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
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(1) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    let left = first_left.as_array();
    let right = first_right.as_array();
    let left_index = first_left_index.as_array();
    let right_index = first_right_index.as_array();
    for predicate in &parsed {
        ensure_equal_lengths_core(
            "first left predicate array",
            left.len(),
            "residual left predicate array",
            predicate.left_len(),
        )
        .map_err(PyValueError::new_err)?;
        ensure_equal_lengths_core(
            "first right predicate array",
            right.len(),
            "residual right predicate array",
            predicate.right_len(),
        )
        .map_err(PyValueError::new_err)?;
    }
    let windows = build_range_core(
        left,
        left_index,
        right,
        right_index,
        right_index_is_ordered,
        first_op,
    )
    .map_err(PyValueError::new_err)?;
    if windows.left_index.is_empty() {
        return Ok(None);
    }
    let (out_left, out_right) =
        materialize_windows_for_non_ne(&windows, &parsed, metadata.as_deref(), keep)
            .map_err(PyValueError::new_err)?;
    if out_left.is_empty() {
        return Ok(None);
    }
    Ok(Some(result_dict(py, out_left, out_right)?))
}

macro_rules! extended_join_function {
    ($name:ident, $type:ty) => {
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
        #[pyfunction]
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
            let first_op_position = if first.len() == 6 {
                5
            } else if first.len() == 11 {
                10
            } else {
                return Err(PyValueError::new_err(
                    "the first extended predicate must contain 6 or 11 elements",
                ));
            };
            let first_op =
                CompareOp::try_from_str(first.get_item(first_op_position)?.extract::<&str>()?)?;
            if first_op == CompareOp::Ne {
                if first.len() != 11 {
                    return Err(PyValueError::new_err(
                        "the first != predicate must contain 11 elements",
                    ));
                }
                let first_left_null_positions = if first.get_item(3)?.is_none() {
                    None
                } else {
                    Some(first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?)
                };
                let first_right_null_positions = if first.get_item(7)?.is_none() {
                    None
                } else {
                    Some(first.get_item(7)?.extract::<PyReadonlyArray1<'py, i64>>()?)
                };
                return extended_not_equal_join(
                    py,
                    predicates,
                    keep,
                    first
                        .get_item(0)?
                        .extract::<PyReadonlyArray1<'py, $type>>()?,
                    first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    first.get_item(2)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    first_left_null_positions,
                    first
                        .get_item(4)?
                        .extract::<PyReadonlyArray1<'py, $type>>()?,
                    first.get_item(5)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    first.get_item(6)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                    first_right_null_positions,
                    first.get_item(8)?.extract::<bool>()?,
                    first.get_item(9)?.extract::<bool>()?,
                );
            }
            if first.len() != 6 {
                return Err(PyValueError::new_err(
                    "the first range predicate must contain 6 elements",
                ));
            }
            extended_join(
                py,
                predicates,
                keep,
                first
                    .get_item(0)?
                    .extract::<PyReadonlyArray1<'py, $type>>()?,
                first.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                first
                    .get_item(2)?
                    .extract::<PyReadonlyArray1<'py, $type>>()?,
                first.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?,
                first.get_item(4)?.extract::<bool>()?,
                first_op,
            )
        }
    };
}

extended_join_function!(single_join_extended_indices_int64, i64);
extended_join_function!(single_join_extended_indices_int32, i32);
extended_join_function!(single_join_extended_indices_int16, i16);
extended_join_function!(single_join_extended_indices_int8, i8);
extended_join_function!(single_join_extended_indices_uint64, u64);
extended_join_function!(single_join_extended_indices_uint32, u32);
extended_join_function!(single_join_extended_indices_uint16, u16);
extended_join_function!(single_join_extended_indices_uint8, u8);
extended_join_function!(single_join_extended_indices_f64, f64);
extended_join_function!(single_join_extended_indices_f32, f32);

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
    use numpy::{PyArray1, PyArrayMethods};

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
}
