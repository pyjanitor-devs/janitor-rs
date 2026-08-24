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

    /// Applies this comparison to one candidate pair.
    ///
    /// ELI5: an indirect function-pointer call can't be inlined, so
    /// picking a `fn(&T, &T) -> bool` once outside the loop measured
    /// *slower* than matching every iteration (25-46% slower at n=100 and
    /// n=100,000 in `bench_compare_start_end` -- an indirect call defeats
    /// the inlining/branch-prediction the compiler gets for free from a
    /// `match` on a small `Copy` enum, which is exactly as cheap per
    /// iteration as the `i8` match every file used to carry its own copy
    /// of). Validating the raw code once via `try_from_code` and matching
    /// on the resulting `CompareOp` every iteration keeps that same
    /// per-iteration cost while still sharing one definition and never
    /// falling back to `!=` on an unrecognized code.
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
            assert_eq!(
                op.apply(&left, &right),
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
