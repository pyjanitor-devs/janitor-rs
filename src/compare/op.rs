use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// One clear name for each of the six ways two values can be compared,
/// shared by every file under `compare/` instead of each keeping its own
/// copy of a numeric-code `match`.
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
    /// ELI5: turns the small numeric code pyjanitor's Python side passes in
    /// into one of six known comparisons, or a clear error -- an
    /// unrecognized code used to silently fall back to `!=` instead of
    /// being rejected.
    pub fn try_from_code<T: Into<i64>>(code: T) -> PyResult<Self> {
        match code.into() {
            0 => Ok(Self::Gt),
            1 => Ok(Self::Ge),
            2 => Ok(Self::Lt),
            3 => Ok(Self::Le),
            4 => Ok(Self::Eq),
            5 => Ok(Self::Ne),
            other => Err(PyValueError::new_err(format!(
                "invalid comparison operator code: {other} (expected 0..=5)"
            ))),
        }
    }

    /// ELI5: picks the comparison once, as a plain function pointer, so a
    /// per-candidate loop calls it directly instead of re-matching on the
    /// operator for every candidate pair.
    #[inline]
    pub fn comparator<T: PartialOrd>(self) -> fn(&T, &T) -> bool {
        fn gt<T: PartialOrd>(a: &T, b: &T) -> bool {
            a > b
        }
        fn ge<T: PartialOrd>(a: &T, b: &T) -> bool {
            a >= b
        }
        fn lt<T: PartialOrd>(a: &T, b: &T) -> bool {
            a < b
        }
        fn le<T: PartialOrd>(a: &T, b: &T) -> bool {
            a <= b
        }
        fn eq<T: PartialOrd>(a: &T, b: &T) -> bool {
            a == b
        }
        fn ne<T: PartialOrd>(a: &T, b: &T) -> bool {
            a != b
        }
        match self {
            Self::Gt => gt,
            Self::Ge => ge,
            Self::Lt => lt,
            Self::Le => le,
            Self::Eq => eq,
            Self::Ne => ne,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_code_decodes_to_its_operator() {
        let cases = [
            (0_i64, CompareOp::Gt),
            (1, CompareOp::Ge),
            (2, CompareOp::Lt),
            (3, CompareOp::Le),
            (4, CompareOp::Eq),
            (5, CompareOp::Ne),
        ];
        for (code, expected) in cases {
            assert_eq!(
                CompareOp::try_from_code(code).unwrap(),
                expected,
                "code={code}"
            );
        }
        // i8 codes decode the same way, without a manual cast at the call site.
        assert_eq!(CompareOp::try_from_code(2_i8).unwrap(), CompareOp::Lt);
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
            let cmp = op.comparator();
            assert_eq!(
                cmp(&left, &right),
                expected,
                "op={op:?} left={left} right={right}"
            );
        }
    }

    #[test]
    fn invalid_codes_are_rejected_not_silently_treated_as_ne() {
        for code in [-1_i64, 6, 100, i64::MIN, i64::MAX] {
            let error = CompareOp::try_from_code(code).unwrap_err().to_string();
            assert!(
                error.contains("invalid comparison operator code"),
                "code={code} produced unexpected message: {error}"
            );
        }
    }
}
