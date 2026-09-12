//! Python-facing fused wrappers for heterogeneous residual predicates.

use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::op::CompareOp;
use crate::aggs::checked_end;

enum Predicate<'py> {
    I64(
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        CompareOp,
    ),
    I32(
        PyReadonlyArray1<'py, i32>,
        PyReadonlyArray1<'py, i32>,
        CompareOp,
    ),
    I16(
        PyReadonlyArray1<'py, i16>,
        PyReadonlyArray1<'py, i16>,
        CompareOp,
    ),
    I8(
        PyReadonlyArray1<'py, i8>,
        PyReadonlyArray1<'py, i8>,
        CompareOp,
    ),
    U64(
        PyReadonlyArray1<'py, u64>,
        PyReadonlyArray1<'py, u64>,
        CompareOp,
    ),
    U32(
        PyReadonlyArray1<'py, u32>,
        PyReadonlyArray1<'py, u32>,
        CompareOp,
    ),
    U16(
        PyReadonlyArray1<'py, u16>,
        PyReadonlyArray1<'py, u16>,
        CompareOp,
    ),
    U8(
        PyReadonlyArray1<'py, u8>,
        PyReadonlyArray1<'py, u8>,
        CompareOp,
    ),
    F64(
        PyReadonlyArray1<'py, f64>,
        PyReadonlyArray1<'py, f64>,
        CompareOp,
    ),
    F32(
        PyReadonlyArray1<'py, f32>,
        PyReadonlyArray1<'py, f32>,
        CompareOp,
    ),
}

enum PredicateView<'a> {
    I64(ArrayView1<'a, i64>, ArrayView1<'a, i64>, CompareOp),
    I32(ArrayView1<'a, i32>, ArrayView1<'a, i32>, CompareOp),
    I16(ArrayView1<'a, i16>, ArrayView1<'a, i16>, CompareOp),
    I8(ArrayView1<'a, i8>, ArrayView1<'a, i8>, CompareOp),
    U64(ArrayView1<'a, u64>, ArrayView1<'a, u64>, CompareOp),
    U32(ArrayView1<'a, u32>, ArrayView1<'a, u32>, CompareOp),
    U16(ArrayView1<'a, u16>, ArrayView1<'a, u16>, CompareOp),
    U8(ArrayView1<'a, u8>, ArrayView1<'a, u8>, CompareOp),
    F64(ArrayView1<'a, f64>, ArrayView1<'a, f64>, CompareOp),
    F32(ArrayView1<'a, f32>, ArrayView1<'a, f32>, CompareOp),
}

type BatchIndices<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>);

struct NullMetadata<'py> {
    left: Option<PyReadonlyArray1<'py, bool>>,
    right: Option<PyReadonlyArray1<'py, bool>>,
    is_extension_array: bool,
}

impl Predicate<'_> {
    fn view<'a>(&'a self) -> PredicateView<'a> {
        match self {
            Self::I64(l, r, op) => PredicateView::I64(l.as_array(), r.as_array(), *op),
            Self::I32(l, r, op) => PredicateView::I32(l.as_array(), r.as_array(), *op),
            Self::I16(l, r, op) => PredicateView::I16(l.as_array(), r.as_array(), *op),
            Self::I8(l, r, op) => PredicateView::I8(l.as_array(), r.as_array(), *op),
            Self::U64(l, r, op) => PredicateView::U64(l.as_array(), r.as_array(), *op),
            Self::U32(l, r, op) => PredicateView::U32(l.as_array(), r.as_array(), *op),
            Self::U16(l, r, op) => PredicateView::U16(l.as_array(), r.as_array(), *op),
            Self::U8(l, r, op) => PredicateView::U8(l.as_array(), r.as_array(), *op),
            Self::F64(l, r, op) => PredicateView::F64(l.as_array(), r.as_array(), *op),
            Self::F32(l, r, op) => PredicateView::F32(l.as_array(), r.as_array(), *op),
        }
    }

    fn left_len(&self) -> usize {
        match self {
            Self::I64(left, _, _) => left.as_array().len(),
            Self::I32(left, _, _) => left.as_array().len(),
            Self::I16(left, _, _) => left.as_array().len(),
            Self::I8(left, _, _) => left.as_array().len(),
            Self::U64(left, _, _) => left.as_array().len(),
            Self::U32(left, _, _) => left.as_array().len(),
            Self::U16(left, _, _) => left.as_array().len(),
            Self::U8(left, _, _) => left.as_array().len(),
            Self::F64(left, _, _) => left.as_array().len(),
            Self::F32(left, _, _) => left.as_array().len(),
        }
    }

    fn right_len(&self) -> usize {
        match self {
            Self::I64(_, right, _) => right.as_array().len(),
            Self::I32(_, right, _) => right.as_array().len(),
            Self::I16(_, right, _) => right.as_array().len(),
            Self::I8(_, right, _) => right.as_array().len(),
            Self::U64(_, right, _) => right.as_array().len(),
            Self::U32(_, right, _) => right.as_array().len(),
            Self::U16(_, right, _) => right.as_array().len(),
            Self::U8(_, right, _) => right.as_array().len(),
            Self::F64(_, right, _) => right.as_array().len(),
            Self::F32(_, right, _) => right.as_array().len(),
        }
    }
}

impl PredicateView<'_> {
    fn matches(&self, left: usize, right: usize) -> bool {
        macro_rules! compare {
            ($l:expr, $r:expr, $op:expr) => {
                $op.apply(&$l[left], &$r[right])
            };
        }
        match self {
            Self::I64(l, r, op) => compare!(l, r, op),
            Self::I32(l, r, op) => compare!(l, r, op),
            Self::I16(l, r, op) => compare!(l, r, op),
            Self::I8(l, r, op) => compare!(l, r, op),
            Self::U64(l, r, op) => compare!(l, r, op),
            Self::U32(l, r, op) => compare!(l, r, op),
            Self::U16(l, r, op) => compare!(l, r, op),
            Self::U8(l, r, op) => compare!(l, r, op),
            Self::F64(l, r, op) => compare!(l, r, op),
            Self::F32(l, r, op) => compare!(l, r, op),
        }
    }
}

/// ELI5: all judges must approve the same candidate. Ask them in order and
/// stop at the first rejection, because later judges cannot rescue a failed
/// AND condition.
fn predicates_match(views: &[PredicateView<'_>], left: usize, right: usize) -> bool {
    // Short-circuit as early as possible.
    for predicate in views {
        if !predicate.matches(left, right) {
            return false;
        }
    }
    true
}

fn predicates_match_with_nulls(
    views: &[PredicateView<'_>],
    metadata: &[NullMetadata<'_>],
    left: usize,
    right: usize,
) -> bool {
    for position in 0..views.len() {
        let values = &metadata[position];
        if let (Some(left_values), Some(right_values)) = (&values.left, &values.right) {
            let left_boolean = left_values.as_array()[left];
            let right_boolean = right_values.as_array()[right];
            // Aligned with pandas Boolean dtype logic:
            // https://pandas.pydata.org/docs/user_guide/boolean.html#kleene-logical-operations
            if values.is_extension_array && (left_boolean || right_boolean) {
                return false;
            }
            if left_boolean || right_boolean {
                continue;
            }
        }
        if !views[position].matches(left, right) {
            return false;
        }
    }
    true
}

#[inline]
fn predicates_match_dispatch(
    views: &[PredicateView<'_>],
    metadata: Option<&[NullMetadata<'_>]>,
    left: usize,
    right: usize,
) -> bool {
    match metadata {
        Some(metadata) => predicates_match_with_nulls(views, metadata, left, right),
        None => predicates_match(views, left, right),
    }
}

/// Validate one half-open candidate range and convert it to slice indices.
///
/// ELI5: `checked_end` already knows how to reject negative or oversized
/// exclusive ends. We only add the corresponding start conversion and allow
/// `start == end`, because an empty candidate row is valid for a join.
fn checked_bounds(start: i64, end: i64, right_len: usize) -> PyResult<(usize, usize)> {
    let start = usize::try_from(start)
        .map_err(|_| {
            PyValueError::new_err(
                "candidate start and end must be non-negative and no greater than the right array length",
            )
        })?;
    let end = checked_end(end, right_len)
        .ok_or_else(|| {
            PyValueError::new_err(
                "candidate start and end must be non-negative and no greater than the right array length",
            )
        })?;
    if start > end {
        return Err(PyValueError::new_err(
            "candidate start must not exceed candidate end",
        ));
    }
    Ok((start, end))
}

fn parse_predicates<'py>(predicates: &Bound<'py, PyList>) -> PyResult<Vec<Predicate<'py>>> {
    let mut result = Vec::with_capacity(predicates.len());
    for item in predicates.iter() {
        let tuple = item
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("each comparison must be (left, right, op)"))?;
        if tuple.len() != 3 {
            return Err(PyValueError::new_err(
                "each comparison must contain left, right, and op",
            ));
        }
        let left = tuple.get_item(0)?;
        let right = tuple.get_item(1)?;
        let op = CompareOp::try_from_code(tuple.get_item(2)?.extract::<i8>()?)?;
        let dtype = left
            .getattr("dtype")?
            .getattr("name")?
            .extract::<String>()?;
        macro_rules! typed {
            ($variant:ident, $ty:ty) => {{
                result.push(Predicate::$variant(
                    left.extract::<PyReadonlyArray1<'py, $ty>>()?,
                    right.extract::<PyReadonlyArray1<'py, $ty>>()?,
                    op,
                ));
            }};
        }
        match dtype.as_str() {
            "int64" => typed!(I64, i64),
            "int32" => typed!(I32, i32),
            "int16" => typed!(I16, i16),
            "int8" => typed!(I8, i8),
            "uint64" => typed!(U64, u64),
            "uint32" => typed!(U32, u32),
            "uint16" => typed!(U16, u16),
            "uint8" => typed!(U8, u8),
            "float64" => typed!(F64, f64),
            "float32" => typed!(F32, f32),
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported comparison dtype: {other}"
                )))
            }
        }
    }
    Ok(result)
}

fn parse_predicates_with_nulls<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
) -> PyResult<(Vec<Predicate<'py>>, Option<Vec<NullMetadata<'py>>>)> {
    let base_tuples = PyList::empty(py);
    let mut metadata = Vec::with_capacity(predicates.len());
    for item in predicates.iter() {
        let tuple = item
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("each comparison must be a tuple"))?;
        if tuple.len() != 3 && tuple.len() != 6 {
            return Err(PyValueError::new_err(
                "each comparison must contain 3 or 6 elements",
            ));
        }
        let op = CompareOp::try_from_code(tuple.get_item(2)?.extract::<i8>()?)?;
        if tuple.len() == 6 && op != CompareOp::Ne {
            return Err(PyValueError::new_err(
                "the six-element predicate form is only valid for !=",
            ));
        }
        base_tuples.append(PyTuple::new(
            py,
            [tuple.get_item(0)?, tuple.get_item(1)?, tuple.get_item(2)?],
        )?)?;

        if tuple.len() == 6 {
            let left = tuple
                .get_item(3)?
                .extract::<PyReadonlyArray1<'py, bool>>()?;
            let right = tuple
                .get_item(4)?
                .extract::<PyReadonlyArray1<'py, bool>>()?;
            let extension_flag = tuple.get_item(5)?.extract::<i8>()?;
            if extension_flag != 0 && extension_flag != 1 {
                return Err(PyValueError::new_err("is_extension_array must be 0 or 1"));
            }
            metadata.push(NullMetadata {
                left: Some(left),
                right: Some(right),
                is_extension_array: extension_flag == 1,
            });
        } else {
            metadata.push(NullMetadata {
                left: None,
                right: None,
                is_extension_array: false,
            });
        }
    }
    let parsed = parse_predicates(&base_tuples)?;
    for position in 0..parsed.len() {
        let predicate = &parsed[position];
        let values = &metadata[position];
        if let Some(left) = &values.left {
            if left.len()? != predicate.left_len() {
                return Err(PyValueError::new_err(
                    "left boolean mask must match its predicate array length",
                ));
            }
        }
        if let Some(right) = &values.right {
            if right.len()? != predicate.right_len() {
                return Err(PyValueError::new_err(
                    "right boolean mask must match its predicate array length",
                ));
            }
        }
    }
    // ELI5: pyjanitor checks for actual nulls before it constructs the
    // six-element form. A six-element predicate therefore explicitly opts
    // into mask-aware comparison; do not rescan the masks here just to decide
    // whether they contain a true value.
    let metadata = if metadata.iter().any(|values| values.left.is_some()) {
        Some(metadata)
    } else {
        None
    };
    Ok((parsed, metadata))
}

enum Selection {
    First,
    Last,
    Any,
    All,
}

/// Run a heterogeneous predicate batch and return one expanded pair per
/// left row that has at least one successful right candidate, or every
/// successful pair when the `All` selection is used.
///
/// ELI5: first/last/any save one winning position per left row, while all
/// saves every winning pair. The final pass only copies those saved results.
fn compare_batch_indices_with_selection<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
    starts: Option<PyReadonlyArray1<'py, i64>>,
    ends: Option<PyReadonlyArray1<'py, i64>>,
    left_index: Bound<'py, PyArray1<i64>>,
    right_index: PyReadonlyArray1<'py, i64>,
    selection: Selection,
) -> PyResult<Option<BatchIndices<'py>>> {
    let (predicates, metadata) = parse_predicates_with_nulls(py, predicates)?;
    if predicates.is_empty() {
        return Err(PyValueError::new_err("at least one comparison is required"));
    }
    let left_len = predicates[0].left_len();
    let right_len = predicates[0].right_len();
    if predicates
        .iter()
        .any(|predicate| predicate.left_len() != left_len || predicate.right_len() != right_len)
    {
        return Err(PyValueError::new_err(
            "all comparisons must use the same left and right lengths",
        ));
    }
    if left_index.len()? != left_len || right_index.len()? != right_len {
        return Err(PyValueError::new_err(
            "index lengths must match the predicate arrays",
        ));
    }
    if let Some(values) = &starts {
        if values.len()? != left_len {
            return Err(PyValueError::new_err(
                "candidate boundaries must match the left array length",
            ));
        }
    }
    if let Some(values) = &ends {
        if values.len()? != left_len {
            return Err(PyValueError::new_err(
                "candidate boundaries must match the left array length",
            ));
        }
    }

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
    let mut selected = if matches!(&selection, Selection::All) {
        None
    } else {
        Some(vec![None; left_len])
    };
    let mut all_matches = Vec::new();
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
        let (start, end) = checked_bounds(start, end, right_len)?;

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
                    Selection::All => all_matches.push((row, right_position)),
                }
            }
        }

        if let Some(selected_position) = selected_position {
            selected.as_mut().unwrap()[row] = Some(selected_position);
            total += 1;
        }
    }

    if matches!(&selection, Selection::All) {
        total = all_matches.len();
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

    // First/last/any only copy saved positions here. All copies its saved
    // pairs; neither branch allocates a counts array.
    if matches!(selection, Selection::All) {
        for (row, right_position) in all_matches {
            expanded_left[output_position] = left_values[row];
            expanded_right[output_position] = right_values[right_position];
            output_position += 1;
        }
    } else {
        for row in 0..left_len {
            if let Some(right_position) = selected.as_ref().unwrap()[row] {
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
    compare_batch_indices_with_selection(
        py,
        predicates,
        starts,
        ends,
        left_index,
        right_index,
        Selection::All,
    )
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
                left_index,
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
                left_index,
                right_index.readonly(),
            )?
            .is_some());
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
}
