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

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::extended::{materialize_pairs_for_ne, materialize_windows_for_non_ne};
use crate::join_common::{result_dict, Keep};
use crate::op::CompareOp;
use crate::predicate::parse_predicates_with_nulls_strings;
use crate::single_non_equi_join::{build_not_equal_positions_core, build_range_core};

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
    Ok(Some(result_dict(py, out_left, out_right, None, None)?))
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
    // The single-extended path has exactly one binary-search anchor. Every
    // later predicate, including another range comparison, is evaluated as a
    // residual filter inside that anchor's candidate window. Dual-range
    // window intersection belongs exclusively to `range_join.rs`.
    let windows = build_range_core(left, left_index, right, right_index, false, first_op)
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
    Ok(Some(result_dict(py, out_left, out_right, None, None)?))
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
            // The wrapper contract carries this flag even though the single
            // extended kernel does not use it to choose a second window.
            // Validate its type at the boundary so malformed tuples fail
            // before candidate generation begins.
            first.get_item(4)?.extract::<bool>()?;
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
                "left index length must equal the number of non-null values plus null positions",
            );
            Ok(())
        })
        .unwrap();
    }
}
