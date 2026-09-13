//! Typed predicate parsing and matching shared by batch comparison paths.

use numpy::ndarray::ArrayView1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use super::op::CompareOp;
use crate::aggs::ensure_equal_lengths;

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

pub(crate) struct NullMetadata<'py> {
    pub(crate) left: Option<PyReadonlyArray1<'py, bool>>,
    pub(crate) right: Option<PyReadonlyArray1<'py, bool>>,
    pub(crate) is_extension_array: bool,
}

impl Predicate<'_> {
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

/// ELI5: all judges must approve the same candidate. Ask them in order and
/// stop at the first rejection, because later judges cannot rescue a failed
/// AND condition.
pub(crate) fn predicates_match(views: &[PredicateView<'_>], left: usize, right: usize) -> bool {
    for predicate in views {
        if !predicate.matches(left, right) {
            return false;
        }
    }
    true
}

pub(crate) fn predicates_match_with_nulls(
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
pub(crate) fn predicates_match_dispatch(
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
