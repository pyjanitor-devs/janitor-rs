//! Existence ("keep=any") join on exactly two range predicates, solved with a
//! staircase instead of a window search.
//!
//! The question per left row is "does ANY right row satisfy both
//! inequalities?". Sort the right rows by the first column. The rows that
//! satisfy the first inequality form a suffix (`left.x <(=) right.x`) or a
//! prefix (`left.x >(=) right.x`) of that order. The second inequality only
//! needs the single best `y` in that suffix/prefix: the smallest `y` when the
//! left row wants `left.y > right.y`, the largest when it wants `left.y <
//! right.y`. A running min/max over the sorted order precomputes that best
//! `y` (and which row owns it) for every possible cut point, so each left row
//! costs one binary search and one comparison.
//!
//! ELI5: shops stand along a road; a kid asks "is there a shop ahead of me
//! cheaper than my pocket money?". A helper walks the road once from the far
//! end writing "cheapest price from here to the end" on a sign at every
//! shop. Each kid then reads ONE sign instead of visiting every shop.
//!
//! Contract: values are non-null and non-NaN (the caller strips nulls, as for
//! the other kernels; an unmasked NaN is a caller-contract violation), and
//! `keep="any"` semantics apply: one genuine match per matching left row, not
//! necessarily the first. Pyjanitor routes only joins whose conditions are
//! exactly these two range predicates here; equality keys, residual
//! predicates and other `keep` modes use the window kernels.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::borrow::Cow;

use super::compare_op::CompareOp;

/// One input column, normalised to the three element types the kernel is
/// instantiated for.
///
/// Signed integers and unsigned integers narrower than 64 bits widen
/// losslessly to `i64`; `float32` widens exactly to `f64`. `u64` stays its own
/// family so values above `i64::MAX` are never cast. That gives three element
/// types per predicate (nine kernel instances) instead of ten (a hundred).
/// Native, contiguous 64-bit columns are borrowed; everything else is copied
/// once.
///
/// ELI5: sort all the coins into three jars (whole numbers, huge whole
/// numbers, decimals) so the counting machine only needs three settings.
enum Column<'py> {
    I64Borrowed(PyReadonlyArray1<'py, i64>),
    I64Owned(Vec<i64>),
    U64Borrowed(PyReadonlyArray1<'py, u64>),
    U64Owned(Vec<u64>),
    F64Borrowed(PyReadonlyArray1<'py, f64>),
    F64Owned(Vec<f64>),
}

/// Borrow a label array, copying only when it is not contiguous.
fn contiguous<'a>(labels: &'a PyReadonlyArray1<'_, i64>) -> Cow<'a, [i64]> {
    match labels.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(labels.as_array().to_vec()),
    }
}

/// A borrowed view of a [`Column`] in one of the three element families.
enum Values<'a> {
    I64(&'a [i64]),
    U64(&'a [u64]),
    F64(&'a [f64]),
}

impl<'py> Column<'py> {
    fn load(array: &Bound<'py, PyAny>) -> PyResult<Self> {
        macro_rules! native {
            ($t:ty, $borrowed:ident, $owned:ident) => {
                if let Ok(a) = array.cast::<PyArray1<$t>>() {
                    let readonly = a.readonly();
                    return Ok(if readonly.as_slice().is_ok() {
                        Column::$borrowed(readonly)
                    } else {
                        Column::$owned(readonly.as_array().to_vec())
                    });
                }
            };
        }
        macro_rules! widen {
            ($owned:ident, $to:ty, $($from:ty),+) => {
                $(
                    if let Ok(a) = array.cast::<PyArray1<$from>>() {
                        let values = a.readonly().as_array().iter().map(|&v| v as $to).collect();
                        return Ok(Column::$owned(values));
                    }
                )+
            };
        }
        native!(i64, I64Borrowed, I64Owned);
        native!(u64, U64Borrowed, U64Owned);
        native!(f64, F64Borrowed, F64Owned);
        widen!(I64Owned, i64, i32, i16, i8, u32, u16, u8);
        widen!(F64Owned, f64, f32);
        Err(PyValueError::new_err(
            "range_exists_indices supports integer and float columns",
        ))
    }

    fn values(&self) -> PyResult<Values<'_>> {
        // Borrowed variants were only created for contiguous arrays, so
        // `as_slice` cannot fail here; the error path keeps that invariant
        // from ever turning into a panic.
        Ok(match self {
            Column::I64Borrowed(a) => Values::I64(a.as_slice()?),
            Column::I64Owned(v) => Values::I64(v),
            Column::U64Borrowed(a) => Values::U64(a.as_slice()?),
            Column::U64Owned(v) => Values::U64(v),
            Column::F64Borrowed(a) => Values::F64(a.as_slice()?),
            Column::F64Owned(v) => Values::F64(v),
        })
    }
}

/// The left and right values of one predicate, required to share a family.
enum Pair<'a> {
    I64(&'a [i64], &'a [i64]),
    U64(&'a [u64], &'a [u64]),
    F64(&'a [f64], &'a [f64]),
}

fn pair<'a>(left: Values<'a>, right: Values<'a>) -> Result<Pair<'a>, String> {
    match (left, right) {
        (Values::I64(l), Values::I64(r)) => Ok(Pair::I64(l, r)),
        (Values::U64(l), Values::U64(r)) => Ok(Pair::U64(l, r)),
        (Values::F64(l), Values::F64(r)) => Ok(Pair::F64(l, r)),
        _ => Err("left and right columns of a predicate must share a dtype family".to_owned()),
    }
}

/// Existence join indices for two range predicates.
///
/// # Arguments
///
/// * `left_x`, `right_x`, `op_x` - The first range predicate
///   (`left_x op_x right_x`).
/// * `left_y`, `right_y`, `op_y` - The second range predicate.
/// * `left_index`, `right_index` - Labels reported for each left/right row.
///
/// # Returns
///
/// `{"left_index", "right_index"}` holding one genuine pair for every left row
/// that has any match, in left order, or `None` when no left row matches.
///
/// # Errors
///
/// `ValueError` for non-range operators, mismatched lengths, unsupported
/// dtypes, or left/right columns from different dtype families.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn range_exists_indices<'py>(
    py: Python<'py>,
    left_x: &Bound<'py, PyAny>,
    right_x: &Bound<'py, PyAny>,
    op_x: &str,
    left_y: &Bound<'py, PyAny>,
    right_y: &Bound<'py, PyAny>,
    op_y: &str,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let op_x = CompareOp::try_from_str(op_x)?;
    let op_y = CompareOp::try_from_str(op_y)?;
    // Labels are read-only; copy only if numpy handed over a strided view.
    let left_labels = contiguous(&left_index);
    let right_labels = contiguous(&right_index);
    let (lx, rx) = (Column::load(left_x)?, Column::load(right_x)?);
    let (ly, ry) = (Column::load(left_y)?, Column::load(right_y)?);
    let x = pair(lx.values()?, rx.values()?).map_err(PyValueError::new_err)?;
    let y = pair(ly.values()?, ry.values()?).map_err(PyValueError::new_err)?;

    macro_rules! run {
        ($xl:expr, $xr:expr, $yl:expr, $yr:expr) => {
            staircase_any($xl, $xr, op_x, $yl, $yr, op_y, &left_labels, &right_labels)
        };
    }
    let result = match (x, y) {
        (Pair::I64(a, b), Pair::I64(c, d)) => run!(a, b, c, d),
        (Pair::I64(a, b), Pair::U64(c, d)) => run!(a, b, c, d),
        (Pair::I64(a, b), Pair::F64(c, d)) => run!(a, b, c, d),
        (Pair::U64(a, b), Pair::I64(c, d)) => run!(a, b, c, d),
        (Pair::U64(a, b), Pair::U64(c, d)) => run!(a, b, c, d),
        (Pair::U64(a, b), Pair::F64(c, d)) => run!(a, b, c, d),
        (Pair::F64(a, b), Pair::I64(c, d)) => run!(a, b, c, d),
        (Pair::F64(a, b), Pair::U64(c, d)) => run!(a, b, c, d),
        (Pair::F64(a, b), Pair::F64(c, d)) => run!(a, b, c, d),
    }
    .map_err(PyValueError::new_err)?;

    let (left_out, right_out) = result;
    if left_out.is_empty() {
        return Ok(None);
    }
    let out = PyDict::new(py);
    out.set_item("left_index", PyArray1::from_vec(py, left_out))?;
    out.set_item("right_index", PyArray1::from_vec(py, right_out))?;
    Ok(Some(out))
}

/// Right values in x order, plus the original position of each sorted row
/// (`None` when the input was already sorted).
type SortedRight<'a, X, Y> = (Cow<'a, [X]>, Cow<'a, [Y]>, Option<Vec<usize>>);

/// Left rows searched together. Eight independent searches keep eight loads
/// in flight, hiding the latency of each search's dependent probe chain.
const LANES: usize = 8;

/// Number of leading elements of `values` that are `< key` (`strict`) or
/// `<= key` (not `strict`), for `LANES` keys at once. `values` must be sorted
/// ascending and non-empty.
///
/// Every key walks the same sequence of step sizes, so the loop body is
/// identical for all lanes and carries no data-dependent branch.
///
/// ELI5: eight kids ask "how many shops are cheaper than me?" at the same
/// time. One helper keeps eight fingers on the price list and slides them all
/// down together, instead of finishing one kid before starting the next.
#[inline(always)]
fn prefix_lens<X: PartialOrd + Copy>(values: &[X], keys: &[X], strict: bool) -> [usize; LANES] {
    let mut base = [0_usize; LANES];
    let mut size = values.len();
    while size > 1 {
        let half = size / 2;
        for lane in 0..LANES {
            let probe = values[base[lane] + half - 1];
            let below = if strict {
                probe < keys[lane]
            } else {
                probe <= keys[lane]
            };
            base[lane] += if below { half } else { 0 };
        }
        size -= half;
    }
    for lane in 0..LANES {
        let probe = values[base[lane]];
        let below = if strict {
            probe < keys[lane]
        } else {
            probe <= keys[lane]
        };
        base[lane] += usize::from(below);
    }
    base
}

/// One-key version of [`prefix_lens`] for the tail rows.
#[inline(always)]
fn prefix_len<X: PartialOrd + Copy>(values: &[X], key: X, strict: bool) -> usize {
    values.partition_point(|value| if strict { *value < key } else { *value <= key })
}

/// The staircase kernel itself, free of PyO3 so it is easy to test.
#[allow(clippy::too_many_arguments)]
fn staircase_any<X: PartialOrd + Copy, Y: PartialOrd + Copy>(
    lx: &[X],
    rx: &[X],
    op_x: CompareOp,
    ly: &[Y],
    ry: &[Y],
    op_y: CompareOp,
    left_labels: &[i64],
    right_labels: &[i64],
) -> Result<(Vec<i64>, Vec<i64>), String> {
    if lx.len() != ly.len() || lx.len() != left_labels.len() {
        return Err("left value and label lengths differ".to_owned());
    }
    if rx.len() != ry.len() || rx.len() != right_labels.len() {
        return Err("right value and label lengths differ".to_owned());
    }
    let suffix = match op_x {
        CompareOp::Lt | CompareOp::Le => true,
        CompareOp::Gt | CompareOp::Ge => false,
        _ => return Err("first predicate must be a range comparison".to_owned()),
    };
    let want_min = match op_y {
        CompareOp::Gt | CompareOp::Ge => true,
        CompareOp::Lt | CompareOp::Le => false,
        _ => return Err("second predicate must be a range comparison".to_owned()),
    };
    let n = rx.len();
    if n == 0 || lx.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Sort the right side by x unless it already is (the sort step allocates
    // nothing when it is; `stair` and `witness` below are always built).
    // ELI5: if the shops are already in road order, don't re-shuffle them.
    let already_sorted = rx.windows(2).all(|pair| pair[0] <= pair[1]);
    let (bx, by, order): SortedRight<X, Y> = if already_sorted {
        (Cow::Borrowed(rx), Cow::Borrowed(ry), None)
    } else {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            rx[a]
                .partial_cmp(&rx[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let bx: Vec<X> = order.iter().map(|&p| rx[p]).collect();
        let by: Vec<Y> = order.iter().map(|&p| ry[p]).collect();
        (Cow::Owned(bx), Cow::Owned(by), Some(order))
    };

    // witness[p] = sorted position of the best y over the suffix starting at p
    // (or the prefix ending at p). Storing the position, not the value, lets
    // the caller report a genuine right row for `any`.
    let better = |candidate: Y, current: Y| {
        if want_min {
            candidate < current
        } else {
            candidate > current
        }
    };
    let mut witness: Vec<usize> = vec![0; n];
    let mut stair: Vec<Y> = by.to_vec();
    if suffix {
        witness[n - 1] = n - 1;
        for p in (0..n - 1).rev() {
            let held = witness[p + 1];
            witness[p] = if better(by[p], by[held]) { p } else { held };
            stair[p] = by[witness[p]];
        }
    } else {
        for p in 1..n {
            let held = witness[p - 1];
            witness[p] = if better(by[p], by[held]) { p } else { held };
            stair[p] = by[witness[p]];
        }
    }

    // `<=` on the left x against a suffix, and `>` against a prefix, both
    // count right rows strictly below the key; the other two include equals.
    let strict = matches!(op_x, CompareOp::Le | CompareOp::Gt);
    let mut left_out = Vec::new();
    let mut right_out = Vec::new();
    // Turn one left row's cut point into a verdict and, for matches, a pair.
    let mut finish = |row: usize, cut: usize| {
        let slot = if suffix {
            if cut == n {
                return;
            }
            cut
        } else {
            if cut == 0 {
                return;
            }
            cut - 1
        };
        if op_y.apply(&ly[row], &stair[slot]) {
            let held = witness[slot];
            left_out.push(left_labels[row]);
            right_out.push(right_labels[order.as_ref().map_or(held, |o| o[held])]);
        }
    };
    let batches = lx.len() / LANES;
    for batch in 0..batches {
        let first = batch * LANES;
        let cuts = prefix_lens(&bx, &lx[first..first + LANES], strict);
        for (lane, &cut) in cuts.iter().enumerate() {
            finish(first + lane, cut);
        }
    }
    for (row, &x) in lx.iter().enumerate().skip(batches * LANES) {
        finish(row, prefix_len(&bx, x, strict));
    }
    Ok((left_out, right_out))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(range_exists_indices, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic generator so failures reproduce on every platform.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, bound: u64) -> i64 {
            (self.next() % bound) as i64
        }
    }

    /// Every operator pair, with heavy ties and unsorted right sides: the
    /// matched left rows must equal brute force exactly, each left row must
    /// appear once, and every returned pair must really satisfy both
    /// predicates.
    #[test]
    fn matches_brute_force_for_every_operator_pair() {
        let ops = [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge];
        let mut rng = XorShift(0x57A1_2CA5);
        for &op_x in &ops {
            for &op_y in &ops {
                for trial in 0..400 {
                    let (nl, nr) = (1 + rng.below(25) as usize, 1 + rng.below(30) as usize);
                    let span = [3, 6, 40][trial % 3];
                    let lx: Vec<i64> = (0..nl).map(|_| rng.below(span)).collect();
                    let ly: Vec<f64> = (0..nl).map(|_| rng.below(span) as f64 * 0.5).collect();
                    let mut rx: Vec<i64> = (0..nr).map(|_| rng.below(span)).collect();
                    if trial % 2 == 0 {
                        rx.sort_unstable();
                    }
                    let ry: Vec<f64> = (0..nr).map(|_| rng.below(span) as f64 * 0.5).collect();
                    let left_labels: Vec<i64> = (0..nl as i64).map(|l| 1000 + l).collect();
                    let right_labels: Vec<i64> = (0..nr as i64).map(|l| 7 * l + 3).collect();
                    let (got_left, got_right) =
                        staircase_any(&lx, &rx, op_x, &ly, &ry, op_y, &left_labels, &right_labels)
                            .unwrap();
                    let expected: Vec<i64> = (0..nl)
                        .filter(|&i| {
                            (0..nr)
                                .any(|j| op_x.apply(&lx[i], &rx[j]) && op_y.apply(&ly[i], &ry[j]))
                        })
                        .map(|i| left_labels[i])
                        .collect();
                    assert_eq!(got_left, expected, "{op_x:?}/{op_y:?} trial {trial}");
                    for (l, r) in got_left.iter().zip(&got_right) {
                        let i = (*l - 1000) as usize;
                        let j = ((*r - 3) / 7) as usize;
                        assert!(
                            op_x.apply(&lx[i], &rx[j]) && op_y.apply(&ly[i], &ry[j]),
                            "{op_x:?}/{op_y:?}: left {i} paired with non-match {j}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn empty_sides_and_bad_operators() {
        let ok = staircase_any::<i64, i64>(
            &[],
            &[1],
            CompareOp::Le,
            &[],
            &[1],
            CompareOp::Gt,
            &[],
            &[0],
        );
        assert_eq!(ok.unwrap(), (Vec::new(), Vec::new()));
        assert!(staircase_any::<i64, i64>(
            &[1],
            &[1],
            CompareOp::Eq,
            &[1],
            &[1],
            CompareOp::Gt,
            &[0],
            &[0],
        )
        .is_err());
        assert!(staircase_any::<i64, i64>(
            &[1],
            &[1, 2],
            CompareOp::Le,
            &[1],
            &[1],
            CompareOp::Gt,
            &[0],
            &[0, 1],
        )
        .is_err());
    }

    /// Call the wrapper with sequential labels (left from 100, right from 500).
    fn call<'py>(
        py: Python<'py>,
        lx: Bound<'py, PyAny>,
        rx: Bound<'py, PyAny>,
        ly: Bound<'py, PyAny>,
        ry: Bound<'py, PyAny>,
        op_x: &str,
        op_y: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let labels = |n: usize, base: i64| {
            PyArray1::from_vec(py, (0..n as i64).map(|l| l + base).collect()).readonly()
        };
        let (nl, nr) = (lx.len()?, rx.len()?);
        range_exists_indices(
            py,
            &lx,
            &rx,
            op_x,
            &ly,
            &ry,
            op_y,
            labels(nl, 100),
            labels(nr, 500),
        )
    }

    /// The matched left labels of a wrapper result (empty for `None`).
    fn matched(result: PyResult<Option<Bound<'_, PyDict>>>) -> Vec<i64> {
        result
            .unwrap()
            .map(|d| {
                d.get_item("left_index")
                    .unwrap()
                    .unwrap()
                    .extract::<Vec<i64>>()
                    .unwrap()
            })
            .unwrap_or_default()
    }

    /// The Python-facing wrapper: every dtype family reaches the right kernel
    /// instance, widened and borrowed columns agree, `u64` above `i64::MAX`
    /// is never cast, and bad inputs raise instead of guessing.
    #[test]
    fn wrapper_dispatches_dtype_families() {
        use numpy::PyArray1;
        Python::initialize();
        Python::attach(|py| {
            // Left rows 0..3: x = 1,5,9 against right x = 2,6; y = 3,3,3
            // against right y = 1,9. `x <= right.x` and `y > right.y`:
            // row0 (1<=2, 3>1) yes; row1 (5<=6, 3>9 no; 5<=2 no) no; row2 no.
            let want = vec![100];
            macro_rules! family {
                ($t:ty) => {{
                    let lx = PyArray1::from_vec(py, vec![1 as $t, 5 as $t, 9 as $t]).into_any();
                    let rx = PyArray1::from_vec(py, vec![2 as $t, 6 as $t]).into_any();
                    let ly = PyArray1::from_vec(py, vec![3 as $t, 3 as $t, 3 as $t]).into_any();
                    let ry = PyArray1::from_vec(py, vec![1 as $t, 9 as $t]).into_any();
                    assert_eq!(
                        matched(call(py, lx, rx, ly, ry, "<=", ">")),
                        want,
                        stringify!($t)
                    );
                }};
            }
            family!(i8);
            family!(i16);
            family!(i32);
            family!(i64);
            family!(u8);
            family!(u16);
            family!(u32);
            family!(u64);
            family!(f32);
            family!(f64);

            // u64 values above i64::MAX must compare as unsigned, not wrap.
            let big = u64::MAX - 1;
            let lx = PyArray1::from_vec(py, vec![big, 1_u64]).into_any();
            let rx = PyArray1::from_vec(py, vec![u64::MAX]).into_any();
            let ly = PyArray1::from_vec(py, vec![5_i64, 5]).into_any();
            let ry = PyArray1::from_vec(py, vec![1_i64]).into_any();
            assert_eq!(matched(call(py, lx, rx, ly, ry, "<=", ">")), vec![100, 101]);
            let lx = PyArray1::from_vec(py, vec![u64::MAX, 1_u64]).into_any();
            let rx = PyArray1::from_vec(py, vec![big]).into_any();
            let ly = PyArray1::from_vec(py, vec![5_i64, 5]).into_any();
            let ry = PyArray1::from_vec(py, vec![1_i64]).into_any();
            assert_eq!(matched(call(py, lx, rx, ly, ry, "<=", ">")), vec![101]);

            // A strided (non-contiguous) column is copied, not misread.
            let wide = PyArray1::from_vec(py, vec![1_i64, 99, 5, 99, 9, 99]);
            let strided = wide
                .call_method1("__getitem__", (pyo3::types::PySlice::new(py, 0, 6, 2),))
                .unwrap();
            let rx = PyArray1::from_vec(py, vec![2_i64, 6]).into_any();
            let ly = PyArray1::from_vec(py, vec![3_i64, 3, 3]).into_any();
            let ry = PyArray1::from_vec(py, vec![1_i64, 9]).into_any();
            assert_eq!(matched(call(py, strided, rx, ly, ry, "<=", ">")), want);

            // Int against float within one predicate, and non-range operators.
            let lx = PyArray1::from_vec(py, vec![1_i64]).into_any();
            let rx = PyArray1::from_vec(py, vec![2.0_f64]).into_any();
            let ly = PyArray1::from_vec(py, vec![3_i64]).into_any();
            let ry = PyArray1::from_vec(py, vec![1_i64]).into_any();
            assert!(call(py, lx, rx, ly.clone(), ry.clone(), "<=", ">").is_err());
            let lx = PyArray1::from_vec(py, vec![1_i64]).into_any();
            let rx = PyArray1::from_vec(py, vec![2_i64]).into_any();
            assert!(call(
                py,
                lx.clone(),
                rx.clone(),
                ly.clone(),
                ry.clone(),
                "==",
                ">"
            )
            .is_err());
            assert!(call(py, lx, rx, ly, ry, "<=", "!=").is_err());
            // The mixed-family check applies to y and to u64-against-i64 too.
            let ix = || PyArray1::from_vec(py, vec![1_i64]).into_any();
            let fy = PyArray1::from_vec(py, vec![1.0_f64]).into_any();
            assert!(call(py, ix(), ix(), ix(), fy, "<=", ">").is_err());
            let ux = PyArray1::from_vec(py, vec![1_u64]).into_any();
            assert!(call(py, ux, ix(), ix(), ix(), "<=", ">").is_err());

            // Every operator pair through the wrapper with more left rows than
            // one 8-lane batch, a ragged tail, and a strided label array.
            let lx: Vec<i64> = (0..21).map(|i| (i * 7) % 11).collect();
            let ly: Vec<i64> = (0..21).map(|i| (i * 5) % 9).collect();
            let rx = vec![3_i64, 8, 1, 6, 9, 2];
            let ry = vec![4_i64, 0, 7, 2, 5, 8];
            for op_x in ["<", "<=", ">", ">="] {
                for op_y in ["<", "<=", ">", ">="] {
                    let hit = |l: usize, r: usize| {
                        let ok = |op: &str, a: i64, b: i64| match op {
                            "<" => a < b,
                            "<=" => a <= b,
                            ">" => a > b,
                            _ => a >= b,
                        };
                        ok(op_x, lx[l], rx[r]) && ok(op_y, ly[l], ry[r])
                    };
                    let want: Vec<i64> = (0..lx.len())
                        .filter(|&l| (0..rx.len()).any(|r| hit(l, r)))
                        .map(|l| 100 + l as i64)
                        .collect();
                    let col = |v: &Vec<i64>| PyArray1::from_vec(py, v.clone()).into_any();
                    let mut got =
                        matched(call(py, col(&lx), col(&rx), col(&ly), col(&ry), op_x, op_y));
                    got.sort_unstable();
                    assert_eq!(got, want, "{op_x} {op_y}");
                }
            }
            let wide = PyArray1::from_vec(py, vec![0_i64, 9, 1, 9, 2, 9]);
            let strided = wide
                .call_method1("__getitem__", (pyo3::types::PySlice::new(py, 0, 6, 2),))
                .unwrap();
            let strided_labels = strided.cast::<PyArray1<i64>>().unwrap().readonly();
            assert_eq!(contiguous(&strided_labels).as_ref(), &[0, 1, 2]);

            // No match anywhere is None, not an empty dict.
            let lx = PyArray1::from_vec(py, vec![9_i64]).into_any();
            let rx = PyArray1::from_vec(py, vec![2_i64]).into_any();
            let ly = PyArray1::from_vec(py, vec![3_i64]).into_any();
            let ry = PyArray1::from_vec(py, vec![1_i64]).into_any();
            assert!(call(py, lx, rx, ly, ry, "<=", ">").unwrap().is_none());
        });
    }
}
