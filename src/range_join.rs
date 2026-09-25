//! Shared range-join window construction.
//!
//! A basic range join has two range predicates. This module computes the
//! half-open `starts`/`ends` windows for those predicates. Extended joins
//! reuse the same primitive and perform their additional filtering elsewhere.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths_core;
use crate::aggs::max::max_starts_ends::max_start_end_core;
use crate::aggs::min::min_starts_ends::min_start_end_core;
use crate::extended::materialize_windows_for_non_ne;
use crate::join_common::{result_dict, Keep, SingleJoinResult};
use crate::op::CompareOp;
use crate::predicate::parse_predicates_with_nulls_strings;
use crate::single_non_equi_join::range_window;

/// A typed range predicate used by the basic two-range kernel.
///
/// The value arrays describe the comparison layout, while the paired index
/// arrays carry the labels that must be returned to Python. PyJanitor owns
/// sorting and alignment before constructing this value; Rust only consumes
/// the already-aligned views.
pub(crate) struct RangePredicate<'a, T> {
    /// Left values in logical left-row order. Null rows have already been
    /// removed for range predicates.
    pub(crate) left: ArrayView1<'a, T>,
    /// Labels paired position-for-position with `left`.
    pub(crate) left_index: ArrayView1<'a, i64>,
    /// Right values in ascending value order.
    pub(crate) right: ArrayView1<'a, T>,
    /// Labels paired position-for-position with the sorted `right` values.
    pub(crate) right_index: ArrayView1<'a, i64>,
    /// Comparator applied to the paired left and right values.
    pub(crate) op: CompareOp,
}

/// Build the intersection window for exactly two aligned range predicates.
///
/// Each right-hand array must already be sorted in ascending value order by
/// PyJanitor. The two predicates are evaluated against the same logical left
/// rows and the same logical right positions; this function only intersects
/// their positional windows. It does not sort arrays or evaluate residual
/// predicates.
///
/// # Arguments
///
/// * `first` - The first range predicate, including its left/right values and
///   the corresponding original index labels.
/// * `second` - The second range predicate. Its left and right arrays must be
///   length-aligned with `first`; its right array must use the same sorted
///   physical layout as `first.right`.
///
/// # Returns
///
/// A [`SingleJoinResult`] containing one half-open `[start, end)` window for
/// each left row whose two predicates intersect. `left_index` retains input
/// order and `right_index` is copied from the first predicate's supplied
/// right-label array. A window's positions always refer to that copied right
/// layout.
///
/// # Errors
///
/// Returns an error when value/index lengths differ or either comparator is
/// not a range comparator.
pub(crate) fn build_windows<T: PartialOrd + Copy>(
    first: RangePredicate<'_, T>,
    second: RangePredicate<'_, T>,
) -> Result<SingleJoinResult, String> {
    ensure_equal_lengths_core(
        "left",
        first.left.len(),
        "left_index",
        first.left_index.len(),
    )?;
    ensure_equal_lengths_core(
        "right",
        first.right.len(),
        "right_index",
        first.right_index.len(),
    )?;
    ensure_equal_lengths_core("second left", second.left.len(), "left", first.left.len())?;
    ensure_equal_lengths_core(
        "second right",
        second.right.len(),
        "right",
        first.right.len(),
    )?;
    if !first.op.is_range() || !second.op.is_range() {
        return Err("range join requires two range comparators".to_owned());
    }

    let mut result = SingleJoinResult {
        left_positions: Vec::new(),
        left_index: Vec::new(),
        right_index: first.right_index.to_vec(),
        starts: Vec::new(),
        ends: Vec::new(),
    };
    for (left_position, (&first_value, &second_value)) in
        first.left.iter().zip(second.left.iter()).enumerate()
    {
        // Each predicate independently produces a half-open interval in the
        // same sorted right layout. Intersecting the starts and ends keeps
        // only right positions that satisfy both predicates.
        let (first_start, first_end) = range_window(first_value, first.right, first.op);
        let (second_start, second_end) = range_window(second_value, second.right, second.op);
        let start = first_start.max(second_start);
        let end = first_end.min(second_end);
        if start < end {
            result.left_positions.push(left_position);
            result.left_index.push(first.left_index[left_position]);
            result.starts.push(start);
            result.ends.push(end);
        }
    }
    Ok(result)
}

/// Select one label from each arbitrary dual-range window.
///
/// Unlike a single non-equi join, a dual-range join can produce an interior
/// window such as `[2, 4)`. Prefix and suffix extrema cannot answer that
/// shape because they include positions outside the intersection. The
/// existing min/max range kernels provide the correct arbitrary-window
/// query and choose between direct scans and a segment tree using their
/// adaptive workload guard.
///
/// # Arguments
///
/// * `windows` - Intersected, non-empty half-open windows and their right
///   index labels.
/// * `keep` - Selection mode. `first` means the smallest right label,
///   `last` the largest, `any` the first physical label, and `all` every
///   label in each window.
///
/// # Returns
///
/// Materialized left and right labels in the same left-row order as
/// `windows`. For `all`, right labels remain in their supplied physical
/// right-array order within each window.
///
/// # Errors
///
/// Returns an error if an internal window boundary cannot be represented by
/// the public int64 boundary contract, or if an extrema kernel returns an
/// invalid position. The latter indicates a violated internal window
/// invariant rather than a normal no-match result; empty windows are removed
/// by [`build_windows`].
pub(crate) fn choose_range_windows(
    windows: &SingleJoinResult,
    keep: Keep,
) -> Result<(Vec<i64>, Vec<i64>), String> {
    let labels = windows.right_index.as_slice();
    if windows.starts.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    if keep == Keep::All {
        // `all` does not need an extrema query: every position in every
        // half-open window is a result, and the physical right order is the
        // required output order. Because this basic range path has no
        // residual filters, the window widths give the exact final size.
        let mut output_capacity = 0_usize;
        for (&start, &end) in windows.starts.iter().zip(&windows.ends) {
            let width = end
                .checked_sub(start)
                .ok_or("range window has invalid bounds")?;
            output_capacity = output_capacity
                .checked_add(width)
                .ok_or("range join result size exceeds platform capacity")?;
        }
        let mut output_left = Vec::new();
        output_left
            .try_reserve_exact(output_capacity)
            .map_err(|_| "range join result allocation failed")?;
        let mut output_right = Vec::new();
        output_right
            .try_reserve_exact(output_capacity)
            .map_err(|_| "range join result allocation failed")?;
        for (row, (&start, &end)) in windows.starts.iter().zip(&windows.ends).enumerate() {
            for &label in &labels[start..end] {
                output_left.push(windows.left_index[row]);
                output_right.push(label);
            }
        }
        return Ok((output_left, output_right));
    }

    if keep == Keep::Any {
        // Every stored window is non-empty, so its first physical position is
        // a valid arbitrary match.
        let mut output_left = Vec::with_capacity(windows.left_index.len());
        let mut output_right = Vec::with_capacity(windows.left_index.len());
        for (row, &start) in windows.starts.iter().enumerate() {
            output_left.push(windows.left_index[row]);
            output_right.push(labels[start]);
        }
        return Ok((output_left, output_right));
    }

    // The reusable min/max kernels accept signed int64 boundaries because
    // that is the public NumPy representation. Convert the internal usize
    // windows with a checked cast before handing them to those kernels.
    let starts: Array1<i64> = windows
        .starts
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| "range window start exceeds int64 capacity".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let ends: Array1<i64> = windows
        .ends
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| "range window end exceeds int64 capacity".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let nulls = Array1::from_elem(labels.len(), false);
    // The RMQ kernels return offsets into `labels`, not public labels. The
    // final loop below performs that last position-to-label conversion.
    let selected_positions = match keep {
        Keep::First => min_start_end_core(
            ArrayView1::from(labels),
            starts.view(),
            ends.view(),
            nulls.view(),
        )?,
        Keep::Last => max_start_end_core(
            ArrayView1::from(labels),
            starts.view(),
            ends.view(),
            nulls.view(),
        )?,
        Keep::Any | Keep::All => unreachable!(),
    };

    let mut output_left = Vec::with_capacity(windows.left_index.len());
    let mut output_right = Vec::with_capacity(windows.left_index.len());
    for (row, &position) in selected_positions.iter().enumerate() {
        let position = usize::try_from(position)
            .map_err(|_| "range window selection returned an invalid position")?;
        let label = *labels
            .get(position)
            .ok_or("range window selection returned an out-of-bounds position")?;
        output_left.push(windows.left_index[row]);
        output_right.push(label);
    }
    Ok((output_left, output_right))
}

/// Keep the extracted NumPy owners alive while borrowed views are used.
///
/// ELI5: `ArrayView1` is only a window into an array; it does not own the
/// array. The owners therefore have to live in this struct until the window
/// calculation has finished.
pub(crate) struct ParsedRangePredicate<'py, T: numpy::Element> {
    pub(crate) left: PyReadonlyArray1<'py, T>,
    pub(crate) left_index: PyReadonlyArray1<'py, i64>,
    pub(crate) right: PyReadonlyArray1<'py, T>,
    pub(crate) right_index: PyReadonlyArray1<'py, i64>,
    pub(crate) op: CompareOp,
}

/// Parse the six-element basic range tuple.
///
/// The tuple is `(left, left_index, right, right_index,
/// right_index_is_ordered, comparator)`. Values are expected to be non-null
/// and sorted on the right side before this function is called. Rust trusts
/// that preparation. The ordering flag is validated for the shared wrapper
/// contract, but arbitrary-window `first`/`last` selection uses range extrema
/// instead of prefix/suffix tables.
///
/// # Errors
///
/// Returns `ValueError` for the wrong tuple length or an invalid comparator,
/// and propagates Python extraction errors for incompatible array dtypes.
///
/// # Arguments
///
/// * `tuple` - The six-element Python tuple at the Rust boundary. The first
///   four fields are aligned value/label arrays, field four is the validated
///   ordering flag, and the final field is the string comparator.
///
/// # Returns
///
/// A parsed predicate that owns borrowed Python array handles for the duration
/// of window construction. The ordering flag is intentionally not stored:
/// arbitrary dual-range selection uses exact interval extrema.
pub(crate) fn parse_range_predicate<'py, T: numpy::Element>(
    tuple: &Bound<'py, PyTuple>,
) -> PyResult<ParsedRangePredicate<'py, T>> {
    if tuple.len() != 6 {
        return Err(PyValueError::new_err(
            "range predicates must contain 6 elements",
        ));
    }
    let left = tuple.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let left_index = tuple.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = tuple.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let right_index = tuple.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    tuple.get_item(4)?.extract::<bool>()?;
    let op = CompareOp::try_from_str(tuple.get_item(5)?.extract::<&str>()?)?;
    Ok(ParsedRangePredicate {
        left,
        left_index,
        right,
        right_index,
        op,
    })
}

/// Parse one of the two range anchors for the range-led extended API.
///
/// Extended joins do not use `right_index_is_ordered`: residual filtering can
/// remove arbitrary candidates before `first`/`last` selection. Their anchor
/// tuples therefore contain only `(left, left_index, right, right_index, op)`.
/// The first two predicates still must have ascending right value arrays;
/// additional predicates are parsed and applied as residual filters after the
/// two windows have been intersected.
///
/// # Arguments
///
/// * `tuple` - A five-element tuple containing left values, left labels, right
///   values, right labels, and a range comparator. This internal extended
///   form is distinct from the six-element public range tuple because the
///   extended path does not use the ordering flag.
///
/// # Returns
///
/// A borrowed [`ParsedRangePredicate`] retaining the Python array owners
/// while the range windows are built.
///
/// # Errors
///
/// Returns `ValueError` for the wrong tuple length or an invalid comparator,
/// and propagates Python extraction errors for incompatible array dtypes.
///
/// # Returns
///
/// A parsed range predicate whose arrays remain borrowed from the Python tuple
/// while the dual-range extended operation constructs and filters windows.
pub(crate) fn parse_extended_range_predicate<'py, T: numpy::Element>(
    tuple: &Bound<'py, PyTuple>,
) -> PyResult<ParsedRangePredicate<'py, T>> {
    if tuple.len() != 5 {
        return Err(PyValueError::new_err(
            "extended range predicates must contain 5 elements",
        ));
    }
    let left = tuple.get_item(0)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let left_index = tuple.get_item(1)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    let right = tuple.get_item(2)?.extract::<PyReadonlyArray1<'py, T>>()?;
    let right_index = tuple.get_item(3)?.extract::<PyReadonlyArray1<'py, i64>>()?;
    let op = CompareOp::try_from_str(tuple.get_item(4)?.extract::<&str>()?)?;
    Ok(ParsedRangePredicate {
        left,
        left_index,
        right,
        right_index,
        op,
    })
}

macro_rules! range_join_function {
    ($name:ident, $ty:ty) => {
        /// Build indices for exactly two aligned, ascending range predicates.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - Exactly two six-element tuples of the form
        ///   `(left, left_index, right, right_index,
        ///   right_index_is_ordered, comparator)`. The right values are
        ///   already sorted by PyJanitor; Rust does not sort them.
        /// * `keep` - Selects one matching right row or all matching rows.
        ///   For unordered right labels, `"first"` means the smallest label
        ///   within each exact intersection window and `"last"` means the
        ///   largest label. `"any"` selects the first physical position.
        /// * `return_building_blocks` - When true, return the intersected
        ///   positional `starts`/`ends` windows instead of applying `keep`.
        ///
        /// # Returns
        ///
        /// Returns `None` when no left row has a non-empty intersection.
        /// Otherwise returns a dictionary containing materialized `left_index`
        /// and `right_index` arrays. Building-block output additionally
        /// contains positional `starts` and `ends` arrays exported as int64.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` unless exactly two valid range predicates are
        /// supplied, or when `keep` is invalid.
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            keep: &str,
            return_building_blocks: bool,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            if predicates.len() != 2 {
                return Err(PyValueError::new_err(
                    "range join requires exactly two predicates",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let second_item = predicates.get_item(1)?;
            let first_tuple = first_item.cast::<PyTuple>()?;
            let second_tuple = second_item.cast::<PyTuple>()?;
            let first = parse_range_predicate::<$ty>(&first_tuple)?;
            let second = parse_range_predicate::<$ty>(&second_tuple)?;
            let first_predicate = RangePredicate {
                left: first.left.as_array(),
                left_index: first.left_index.as_array(),
                right: first.right.as_array(),
                right_index: first.right_index.as_array(),
                op: first.op,
            };
            let second_predicate = RangePredicate {
                left: second.left.as_array(),
                left_index: second.left_index.as_array(),
                right: second.right.as_array(),
                right_index: second.right_index.as_array(),
                op: second.op,
            };
            let windows =
                build_windows(first_predicate, second_predicate).map_err(PyValueError::new_err)?;
            if windows.left_index.is_empty() {
                return Ok(None);
            }
            if return_building_blocks {
                return Ok(Some(result_dict(
                    py,
                    windows.left_index,
                    windows.right_index,
                    Some(windows.starts),
                    Some(windows.ends),
                )?));
            }
            let keep = Keep::parse(keep)?;
            let (left, right) =
                choose_range_windows(&windows, keep).map_err(PyValueError::new_err)?;
            if left.is_empty() {
                return Ok(None);
            }
            Ok(Some(result_dict(py, left, right, None, None)?))
        }
    };
}

range_join_function!(range_join_indices_int64, i64);
range_join_function!(range_join_indices_int32, i32);
range_join_function!(range_join_indices_int16, i16);
range_join_function!(range_join_indices_int8, i8);
range_join_function!(range_join_indices_uint64, u64);
range_join_function!(range_join_indices_uint32, u32);
range_join_function!(range_join_indices_uint16, u16);
range_join_function!(range_join_indices_uint8, u8);
range_join_function!(range_join_indices_f64, f64);
range_join_function!(range_join_indices_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(range_join_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_indices_f32, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_int64, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_int32, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_int16, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_int8, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_f64, m)?)?;
    m.add_function(wrap_pyfunction!(range_join_extended_indices_f32, m)?)?;
    Ok(())
}

/// Execute a range-led extended join.
///
/// The first two predicates supply the two range windows. `build_windows`
/// intersects them before residual predicates are evaluated. The caller must
/// therefore provide two ascending range predicates; this API has no fallback
/// that treats the second predicate as an ordinary residual.
fn extended_join<'py, T: numpy::Element + PartialOrd + Copy>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    keep: &str,
    first: ParsedRangePredicate<'py, T>,
    second: ParsedRangePredicate<'py, T>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if predicates.len() < 2 {
        return Err(PyValueError::new_err(
            "range extended join requires at least two predicates",
        ));
    }
    if !first.op.is_range() || !second.op.is_range() {
        return Err(PyValueError::new_err(
            "range extended join requires two range predicates first",
        ));
    }
    let keep = Keep::parse(keep)?;
    let residuals = PyList::empty(py);
    for item in predicates.iter().skip(2) {
        residuals.append(item)?;
    }
    let (parsed, metadata) = parse_predicates_with_nulls_strings(py, &residuals)?;
    let left = first.left.as_array();
    let right = first.right.as_array();
    let left_index = first.left_index.as_array();
    let right_index = first.right_index.as_array();
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
    let first_predicate = RangePredicate {
        left,
        left_index,
        right,
        right_index,
        op: first.op,
    };
    let second_predicate = RangePredicate {
        left: second.left.as_array(),
        left_index: second.left_index.as_array(),
        right: second.right.as_array(),
        right_index: second.right_index.as_array(),
        op: second.op,
    };
    let windows =
        build_windows(first_predicate, second_predicate).map_err(PyValueError::new_err)?;
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

macro_rules! range_join_extended_function {
    ($name:ident, $ty:ty) => {
        /// Build indices for two range anchors followed by residual filters.
        ///
        /// The first two predicates are five-element tuples of the form
        /// `(left, left_index, right, right_index, comparator)`. Their right
        /// value arrays must already be ascending and aligned to one another.
        /// Predicates after the first two are evaluated in user order; they
        /// may use the ordinary residual tuple form or the null-aware `!=`
        /// form accepted by the shared predicate parser. Residual filtering
        /// happens before `keep` is applied.
        ///
        /// # Arguments
        ///
        /// * `py` - Active Python interpreter token.
        /// * `predicates` - At least two anchor/residual predicate tuples.
        /// * `keep` - `"first"`, `"last"`, `"any"`, or `"all"`.
        ///
        /// # Returns
        ///
        /// Returns `None` when no candidate survives both range anchors and
        /// all residual predicates; otherwise returns materialized left/right
        /// index arrays in left-row order.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` for invalid tuple shapes, non-range anchors,
        /// invalid residual predicates, or an invalid `keep` value.
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            predicates: &Bound<'py, PyList>,
            keep: &str,
        ) -> PyResult<Option<Bound<'py, PyDict>>> {
            if predicates.len() < 2 {
                return Err(PyValueError::new_err(
                    "range extended join requires at least two predicates",
                ));
            }
            let first_item = predicates.get_item(0)?;
            let first_tuple = first_item.cast::<PyTuple>()?;
            if first_tuple.len() != 5 {
                return Err(PyValueError::new_err(
                    "the first extended range predicate must contain 5 elements",
                ));
            }
            let second_item = predicates.get_item(1)?;
            let second_tuple = second_item.cast::<PyTuple>()?;
            let first = parse_extended_range_predicate::<$ty>(&first_tuple)?;
            let second = parse_extended_range_predicate::<$ty>(&second_tuple)?;
            extended_join(py, predicates, keep, first, second)
        }
    };
}

range_join_extended_function!(range_join_extended_indices_int64, i64);
range_join_extended_function!(range_join_extended_indices_int32, i32);
range_join_extended_function!(range_join_extended_indices_int16, i16);
range_join_extended_function!(range_join_extended_indices_int8, i8);
range_join_extended_function!(range_join_extended_indices_uint64, u64);
range_join_extended_function!(range_join_extended_indices_uint32, u32);
range_join_extended_function!(range_join_extended_indices_uint16, u16);
range_join_extended_function!(range_join_extended_indices_uint8, u8);
range_join_extended_function!(range_join_extended_indices_f64, f64);
range_join_extended_function!(range_join_extended_indices_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{ndarray::Array1, PyArray1, PyArrayMethods};

    fn read_pair<'py>(result: &Bound<'py, PyDict>) -> (Vec<i64>, Vec<i64>) {
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
        (left, right)
    }

    #[test]
    fn two_range_windows_intersect_in_ascending_layout() {
        let left = Array1::from_vec(vec![2_i64, 5]);
        let left_second = Array1::from_vec(vec![8_i64, 9]);
        let right = Array1::from_vec(vec![1_i64, 2, 3, 4, 5, 6, 7, 8]);
        let right_second = Array1::from_vec(vec![0_i64, 1, 2, 3, 4, 5, 6, 7]);
        let left_index = Array1::from_vec(vec![10_i64, 11]);
        let right_index = Array1::from_vec(vec![20_i64, 21, 22, 23, 24, 25, 26, 27]);

        let result = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: left_index.view(),
                right: right.view(),
                right_index: right_index.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: left_index.view(),
                right: right_second.view(),
                right_index: right_index.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        assert_eq!(result.left_index, vec![10, 11]);
        assert_eq!(result.starts, vec![2, 5]);
        assert_eq!(result.ends, vec![8, 8]);
    }

    #[test]
    fn two_range_windows_drop_empty_intersections() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_second = Array1::from_vec(vec![1_i64]);
        let right = Array1::from_vec(vec![1_i64, 2, 3]);
        let right_second = Array1::from_vec(vec![1_i64, 2, 3]);
        let labels = Array1::from_vec(vec![10_i64, 11, 12]);
        let one_left_index = Array1::from_vec(vec![0_i64]);

        let result = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: one_left_index.view(),
                right: right.view(),
                right_index: labels.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: one_left_index.view(),
                right: right_second.view(),
                right_index: labels.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        assert!(result.left_index.is_empty());
        assert!(result.starts.is_empty());
        assert!(result.ends.is_empty());
    }

    #[test]
    fn unordered_labels_are_selected_from_intersected_suffix_windows() {
        let left = Array1::from_vec(vec![2_i64]);
        let left_second = Array1::from_vec(vec![6_i64]);
        let right = Array1::from_vec(vec![1_i64, 3, 5, 7]);
        let right_second = Array1::from_vec(vec![0_i64, 2, 4, 6]);
        let left_index = Array1::from_vec(vec![100_i64]);
        let right_index = Array1::from_vec(vec![40_i64, 10, 30, 20]);

        let windows = build_windows(
            RangePredicate {
                left: left.view(),
                left_index: left_index.view(),
                right: right.view(),
                right_index: right_index.view(),
                op: CompareOp::Lt,
            },
            RangePredicate {
                left: left_second.view(),
                left_index: left_index.view(),
                right: right_second.view(),
                right_index: right_index.view(),
                op: CompareOp::Gt,
            },
        )
        .expect("aligned range predicates should build");

        // P1 gives [1, 4), P2 gives [0, 3), so the intersection is [1, 3).
        // The labels in that physical window are [10, 30], even though the
        // complete right-index array [40, 10, 30, 20] is unordered.
        assert_eq!(windows.starts, vec![1]);
        assert_eq!(windows.ends, vec![3]);
        assert_eq!(
            choose_range_windows(&windows, Keep::First).unwrap(),
            (vec![100], vec![10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last).unwrap(),
            (vec![100], vec![30])
        );
    }

    #[test]
    fn arbitrary_windows_use_range_extrema_not_prefix_or_suffix_extrema() {
        let windows = SingleJoinResult {
            left_positions: vec![0, 1],
            left_index: vec![100, 101],
            right_index: vec![40, 10, 30, 20, 50],
            starts: vec![2, 1],
            ends: vec![4, 4],
        };

        // The first window is [2, 4) = [30, 20], and the second is
        // [1, 4) = [10, 30, 20]. Neither is a prefix or suffix. The
        // smallest and largest labels must be selected within each exact
        // interval, not from the surrounding right-index array.
        assert_eq!(
            choose_range_windows(&windows, Keep::First).unwrap(),
            (vec![100, 101], vec![20, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Last).unwrap(),
            (vec![100, 101], vec![30, 30])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::Any).unwrap(),
            (vec![100, 101], vec![30, 10])
        );
        assert_eq!(
            choose_range_windows(&windows, Keep::All).unwrap(),
            (vec![100, 100, 101, 101, 101], vec![30, 20, 10, 30, 20])
        );
    }

    #[test]
    fn range_extended_entry_point_filters_residuals_after_two_windows() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![1_i64, 3, 5, 7]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![6_i64]).into_any(),
                    PyArray1::from_vec(py, vec![100_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 2, 4, 6]).into_any(),
                    PyArray1::from_vec(py, vec![40_i64, 10, 30, 20]).into_any(),
                    ">".into_pyobject(py)?.into_any(),
                ],
            )?)?;
            predicates.append(PyTuple::new(
                py,
                [
                    PyArray1::from_vec(py, vec![2_i64]).into_any(),
                    PyArray1::from_vec(py, vec![0_i64, 1, 5, 6]).into_any(),
                    "<".into_pyobject(py)?.into_any(),
                ],
            )?)?;

            let result = range_join_extended_indices_int64(py, &predicates, "all")?
                .expect("the residual predicate should leave one candidate");
            assert_eq!(read_pair(&result), (vec![100], vec![30]));
            Ok(())
        })
        .unwrap();
    }
}
