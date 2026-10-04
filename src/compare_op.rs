use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// One clear name for each of the six ways two values can be compared.
///
/// This is shared by the multi-predicate and single-predicate join kernels.
/// Keeping the enum at the crate root avoids coupling the comparator to one
/// particular join implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

impl CompareOp {
    /// Decode the public string spelling used by the single-join API.
    ///
    /// ELI5: the Python boundary translates a readable operator once, then
    /// the hot loops carry the same small enum used by the older kernels.
    pub fn try_from_str(op: &str) -> PyResult<Self> {
        match op {
            ">" => Ok(Self::Gt),
            ">=" => Ok(Self::Ge),
            "<" => Ok(Self::Lt),
            "<=" => Ok(Self::Le),
            "==" => Ok(Self::Eq),
            "!=" => Ok(Self::Ne),
            other => Err(PyValueError::new_err(format!(
                "invalid comparison operator: {other} (expected one of >, >=, <, <=, ==, !=)"
            ))),
        }
    }

    /// Applies this comparison to one candidate pair.
    ///
    /// ELI5: an indirect function-pointer call can't be inlined, so
    /// picking a `fn(&T, &T) -> bool` once outside the loop measured
    /// *slower* than matching every iteration (25-46% slower at n=100 and
    /// n=100,000 in the scalar benchmark -- an indirect call defeats
    /// the inlining/branch-prediction the compiler gets for free from a
    /// `match` on a small `Copy` enum, which is exactly as cheap per
    /// iteration as the `i8` match every file used to carry its own copy
    /// of). Matching on the resulting `CompareOp` every iteration keeps that
    /// same per-iteration cost while sharing one definition.
    #[inline]
    pub fn apply<T: PartialOrd>(self, left: &T, right: &T) -> bool {
        match self {
            Self::Gt => left > right,
            Self::Ge => left >= right,
            Self::Lt => left < right,
            Self::Le => left <= right,
            Self::Eq => left == right,
            Self::Ne => left != right,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn each_string_decodes_to_its_operator() {
        let cases = [
            (">", CompareOp::Gt),
            (">=", CompareOp::Ge),
            ("<", CompareOp::Lt),
            ("<=", CompareOp::Le),
            ("==", CompareOp::Eq),
            ("!=", CompareOp::Ne),
        ];
        for (operator, expected) in cases {
            assert_eq!(CompareOp::try_from_str(operator).unwrap(), expected);
        }
    }

    #[test]
    fn invalid_strings_are_rejected() {
        Python::initialize();
        let error = CompareOp::try_from_str("like").unwrap_err().to_string();
        assert!(error.contains("invalid comparison operator"));
    }

    #[test]
    fn each_operator_compares_correctly() {
        let cases: [(CompareOp, i64, i64, bool); 12] = [
            (CompareOp::Gt, 5, 4, true),
            (CompareOp::Gt, 5, 5, false),
            (CompareOp::Ge, 5, 5, true),
            (CompareOp::Ge, 4, 5, false),
            (CompareOp::Lt, 4, 5, true),
            (CompareOp::Lt, 5, 5, false),
            (CompareOp::Le, 5, 5, true),
            (CompareOp::Le, 6, 5, false),
            (CompareOp::Eq, 5, 5, true),
            (CompareOp::Eq, 5, 4, false),
            (CompareOp::Ne, 5, 4, true),
            (CompareOp::Ne, 5, 5, false),
        ];
        for (op, left, right, expected) in cases {
            assert_eq!(
                op.apply(&left, &right),
                expected,
                "op={op:?} left={left} right={right}"
            );
        }
    }
}
