//! Typed predicate parsing and matching shared by join and aggregation paths.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyTuple};

use crate::aggs::ensure_equal_lengths;
use crate::op::CompareOp;

/// A typed comparison between one left-hand array and one right-hand array.
///
/// The enum keeps the concrete NumPy dtype with each predicate so the hot
/// matching loop can dispatch once per predicate without converting values to
/// a common representation.
pub(crate) enum Predicate<'py> {
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

/// Borrowed, Rust-only view of a [`Predicate`].
///
/// This view removes the PyO3 wrapper from the inner comparison loop while
/// retaining the original typed arrays and comparison operator.
pub(crate) enum PredicateView<'a> {
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

/// Optional null masks and extension-array semantics for one predicate.
pub(crate) struct NullMetadata<'py> {
    /// Boolean mask for the predicate's left-hand array, when null-aware
    /// comparison is enabled.
    pub(crate) left: Option<PyReadonlyArray1<'py, bool>>,
    /// Boolean mask for the predicate's right-hand array, when null-aware
    /// comparison is enabled.
    pub(crate) right: Option<PyReadonlyArray1<'py, bool>>,
    /// Whether a masked value follows pandas' nullable extension-array
    /// comparison rules.
    pub(crate) is_extension_array: bool,
}

/// Borrowed Rust-only view of one predicate's null metadata.
///
/// `NullMetadata` owns PyO3's `PyReadonlyArray1` handles because those handles
/// are needed while parsing Python arguments. The matching kernels do not need
/// the handles themselves; they only need ordinary Rust/ndarray views into the
/// boolean masks. Keeping this smaller view type separate makes that boundary
/// explicit.
///
/// A caller creates one of these views per predicate, once per function call,
/// before entering the candidate loop. That matters because a single left row
/// can inspect many right candidates, and the same mask is consulted for every
/// one of them. Repeatedly calling `as_array()` or carrying PyO3-backed values
/// through that hot loop would add unnecessary wrapper and borrow work without
/// changing the result.
pub(crate) struct NullMetadataView<'a> {
    pub(crate) left: Option<ArrayView1<'a, bool>>,
    pub(crate) right: Option<ArrayView1<'a, bool>>,
    pub(crate) is_extension_array: bool,
}

impl NullMetadata<'_> {
    /// Borrow the null masks once for reuse by a matching loop.
    ///
    /// The returned views do not copy mask data. They borrow the NumPy buffers,
    /// so this conversion costs only a small per-predicate view description and
    /// leaves the actual boolean storage in place. The owning `NullMetadata`
    /// values remain alive for the entire call, which makes the borrowed views
    /// valid while the Rust kernel evaluates candidates.
    pub(crate) fn view<'a>(&'a self) -> NullMetadataView<'a> {
        NullMetadataView {
            left: self.left.as_ref().map(|values| values.as_array()),
            right: self.right.as_ref().map(|values| values.as_array()),
            is_extension_array: self.is_extension_array,
        }
    }
}

/// Create borrowed views for all parsed null metadata once per call.
///
/// Keeping this conversion in one helper makes it harder for a caller to
/// accidentally put PyO3-backed mask access back into a candidate loop. The
/// returned vector contains only lightweight view descriptors; it does not
/// duplicate either boolean mask. `metadata` must remain alive while the
/// returned views are used.
pub(crate) fn null_metadata_views<'a>(
    metadata: &'a [NullMetadata<'a>],
) -> Vec<NullMetadataView<'a>> {
    metadata.iter().map(NullMetadata::view).collect()
}

impl Predicate<'_> {
    /// Borrow the typed NumPy arrays as ordinary ndarray views.
    ///
    /// # Returns
    ///
    /// A [`PredicateView`] borrowing the arrays and comparison operator stored
    /// in this predicate.
    pub(crate) fn view<'a>(&'a self) -> PredicateView<'a> {
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

    /// Return the number of values in the predicate's left-hand array.
    ///
    /// # Returns
    ///
    /// The left-hand array length.
    pub(crate) fn left_len(&self) -> usize {
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

    /// Return the number of values in the predicate's right-hand array.
    ///
    /// # Returns
    ///
    /// The right-hand array length.
    pub(crate) fn right_len(&self) -> usize {
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
    /// Evaluate this typed predicate for one pair of positional indices.
    ///
    /// # Arguments
    ///
    /// * `left` - Position in the left-hand array.
    /// * `right` - Position in the right-hand array.
    ///
    /// # Panics
    ///
    /// Panics if either index is outside its corresponding array. Callers are
    /// responsible for validating candidate bounds before matching.
    pub(crate) fn matches(&self, left: usize, right: usize) -> bool {
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

/// Evaluate every predicate for one candidate without null-mask handling.
///
/// ELI5: all judges must approve the same candidate. Ask them in order and
/// stop at the first rejection, because later judges cannot rescue a failed
/// AND condition.
///
/// # Arguments
///
/// * `views` - Borrowed typed predicates to evaluate.
/// * `left` - Position in each predicate's left-hand array.
/// * `right` - Position in each predicate's right-hand array.
///
/// # Returns
///
/// `true` only when every predicate accepts the candidate.
pub(crate) fn predicates_match(views: &[PredicateView<'_>], left: usize, right: usize) -> bool {
    for predicate in views {
        if !predicate.matches(left, right) {
            return false;
        }
    }
    true
}

/// Evaluate every predicate for one candidate with null-mask handling.
///
/// # Arguments
///
/// * `views` - Borrowed typed predicates to evaluate.
/// * `metadata` - Per-predicate null masks and extension-array semantics.
/// * `left` - Position in each predicate's left-hand array.
/// * `right` - Position in each predicate's right-hand array.
///
/// # Returns
///
/// `true` only when every predicate accepts the candidate.
pub(crate) fn predicates_match_with_nulls(
    views: &[PredicateView<'_>],
    metadata: &[NullMetadataView<'_>],
    left: usize,
    right: usize,
) -> bool {
    for position in 0..views.len() {
        let values = &metadata[position];
        if let (Some(left_values), Some(right_values)) = (&values.left, &values.right) {
            let left_boolean = left_values[left];
            let right_boolean = right_values[right];
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

/// Dispatch candidate matching to the masked or unmasked implementation.
///
/// # Arguments
///
/// * `views` - Borrowed typed predicates to evaluate.
/// * `metadata` - Null metadata for the predicates, or `None` when masks are
///   not active.
/// * `left` - Position in each predicate's left-hand array.
/// * `right` - Position in each predicate's right-hand array.
///
/// # Returns
///
/// `true` only when the candidate satisfies all applicable predicates and
/// null semantics.
#[inline]
pub(crate) fn predicates_match_dispatch(
    views: &[PredicateView<'_>],
    metadata: Option<&[NullMetadataView<'_>]>,
    left: usize,
    right: usize,
) -> bool {
    match metadata {
        Some(metadata) => predicates_match_with_nulls(views, metadata, left, right),
        None => predicates_match(views, left, right),
    }
}

/// Parse retained legacy numeric-opcode comparison tuples into typed
/// predicates.
///
/// Each item must be a three-element `(left, right, op)` tuple. The left and
/// right objects must be one-dimensional NumPy arrays with one of the numeric
/// dtypes supported by this module; `op` is the numeric comparison opcode
/// understood by [`CompareOp`]. New callers should use
/// [`parse_predicates_strings`], but this entry point remains for existing
/// batch wrappers until their migration is complete.
///
/// # Arguments
///
/// * `predicates` - Python list of `(left, right, op)` comparison tuples.
///
/// # Errors
///
/// Returns a Python type/value error for malformed tuples, unsupported dtypes,
/// or invalid comparison opcodes.
pub(crate) fn parse_predicates<'py>(
    predicates: &Bound<'py, PyList>,
) -> PyResult<Vec<Predicate<'py>>> {
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

/// Parse string-based comparison tuples for the extended single-join API.
///
/// The retained legacy batch wrappers use numeric opcodes. The extended
/// wrapper follows the public single-join contract and accepts readable
/// comparator strings instead.
pub(crate) fn parse_predicates_strings<'py>(
    predicates: &Bound<'py, PyList>,
) -> PyResult<Vec<Predicate<'py>>> {
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
        let op = CompareOp::try_from_str(tuple.get_item(2)?.extract::<&str>()?)?;
        let left = tuple.get_item(0)?;
        let right = tuple.get_item(1)?;
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

/// Parse the string-based residual predicate form used by
/// the extended range and single-join APIs.
///
/// Ordinary residual predicates are `(left, right, op)`. A null-aware `!=`
/// predicate is `(left, left_nulls, right, right_nulls,
/// is_extension_array, op)`, where `is_extension_array` must be a Python
/// boolean. For all-`!=` extended joins, these arrays are
/// full physical layouts: candidate position pairs index them directly, and
/// the masks are full-length authoritative null masks. The parser does not
/// align or filter these arrays. PyJanitor is responsible for alignment,
/// null filtering for ordinary range predicates, and the distinction between
/// NumPy null semantics and pandas extension-array semantics.
pub(crate) fn parse_predicates_with_nulls_strings<'py>(
    py: Python<'py>,
    predicates: &Bound<'py, PyList>,
) -> PyResult<(Vec<Predicate<'py>>, Option<Vec<NullMetadata<'py>>>)> {
    let base_tuples = PyList::empty(py);
    let mut metadata = Vec::with_capacity(predicates.len());
    for item in predicates.iter() {
        let tuple = item
            .cast::<PyTuple>()
            .map_err(|_| PyTypeError::new_err("each residual comparison must be a tuple"))?;
        if tuple.len() != 3 && tuple.len() != 6 {
            return Err(PyValueError::new_err(
                "each residual comparison must contain 3 or 6 elements",
            ));
        }
        let op_position = if tuple.len() == 3 { 2 } else { 5 };
        let op = CompareOp::try_from_str(tuple.get_item(op_position)?.extract::<&str>()?)?;
        if tuple.len() == 6 && op != CompareOp::Ne {
            return Err(PyValueError::new_err(
                "the six-element residual form is only valid for !=",
            ));
        }
        if tuple.len() == 3 {
            base_tuples.append(PyTuple::new(
                py,
                [tuple.get_item(0)?, tuple.get_item(1)?, tuple.get_item(2)?],
            )?)?;
            metadata.push(NullMetadata {
                left: None,
                right: None,
                is_extension_array: false,
            });
            continue;
        }
        let left_mask = tuple
            .get_item(1)?
            .extract::<PyReadonlyArray1<'py, bool>>()?;
        let right_mask = tuple
            .get_item(3)?
            .extract::<PyReadonlyArray1<'py, bool>>()?;
        let extension_object = tuple.get_item(4)?;
        if !extension_object.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("is_extension_array must be a bool"));
        }
        let extension_flag = extension_object.extract::<bool>()?;
        base_tuples.append(PyTuple::new(
            py,
            [tuple.get_item(0)?, tuple.get_item(2)?, tuple.get_item(5)?],
        )?)?;
        metadata.push(NullMetadata {
            left: Some(left_mask),
            right: Some(right_mask),
            is_extension_array: extension_flag,
        });
    }
    let parsed = parse_predicates_strings(&base_tuples)?;
    for position in 0..parsed.len() {
        let predicate = &parsed[position];
        let values = &metadata[position];
        if let Some(left) = &values.left {
            ensure_equal_lengths(
                "left boolean mask",
                left.len()?,
                "left predicate array",
                predicate.left_len(),
            )?;
        }
        if let Some(right) = &values.right {
            ensure_equal_lengths(
                "right boolean mask",
                right.len()?,
                "right predicate array",
                predicate.right_len(),
            )?;
        }
    }
    let metadata = if metadata.iter().any(|values| values.left.is_some()) {
        Some(metadata)
    } else {
        None
    };
    Ok((parsed, metadata))
}

/// Parse comparison tuples and their optional null-mask metadata.
///
/// A three-element tuple has the form `(left, right, op)`. A six-element tuple
/// has the form `(left, right, op, left_mask, right_mask, is_extension_array)`
/// and is allowed only for `!=` comparisons. The returned metadata is omitted
/// when no predicate supplies masks.
///
/// # Arguments
///
/// * `py` - Python interpreter token used to construct the temporary base
///   predicate tuples.
/// * `predicates` - Python list containing three- or six-element comparison
///   tuples.
///
/// # Returns
///
/// The parsed typed predicates and optional per-predicate null metadata.
///
/// # Errors
///
/// Returns a Python type/value error for malformed tuples, invalid opcodes,
/// unsupported dtypes, invalid extension flags, or mask/array length
/// mismatches. The numeric extension flag remains the legacy `0`/`1` form;
/// the string-based parser uses a real Python boolean instead.
pub(crate) fn parse_predicates_with_nulls<'py>(
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
            ensure_equal_lengths(
                "left boolean mask",
                left.len()?,
                "left predicate array",
                predicate.left_len(),
            )?;
        }
        if let Some(right) = &values.right {
            ensure_equal_lengths(
                "right boolean mask",
                right.len()?,
                "right predicate array",
                predicate.right_len(),
            )?;
        }
    }
    // ELI5: pyjanitor checks for actual nulls before it constructs the
    // six-element form. A six-element predicate therefore explicitly opts
    // into mask-aware comparison; do not rescan the masks here.
    let metadata = if metadata.iter().any(|values| values.left.is_some()) {
        Some(metadata)
    } else {
        None
    };
    Ok((parsed, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::PyArray1;

    #[test]
    fn string_null_metadata_requires_a_boolean_extension_flag() {
        Python::initialize();
        Python::attach(|py| -> PyResult<()> {
            let predicates = PyList::empty(py);
            predicates
                .append(PyTuple::new(
                    py,
                    [
                        PyArray1::from_vec(py, vec![1_i64]).into_any(),
                        PyArray1::from_vec(py, vec![false]).into_any(),
                        PyArray1::from_vec(py, vec![2_i64]).into_any(),
                        PyArray1::from_vec(py, vec![false]).into_any(),
                        1_i64.into_pyobject(py)?.into_any(),
                        "!=".into_pyobject(py)?.into_any(),
                    ],
                )?)
                .unwrap();

            let error = match parse_predicates_with_nulls_strings(py, &predicates) {
                Ok(_) => panic!("expected a non-boolean extension flag to be rejected"),
                Err(error) => error,
            };
            assert_eq!(
                error.value(py).to_string(),
                "is_extension_array must be a bool"
            );
            Ok(())
        })
        .unwrap();
    }
}
