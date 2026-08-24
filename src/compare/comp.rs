use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::ensure_tape_width;
use crate::compare::op::CompareOp;

/// For every left row `i`, compare `left[i]` against `right[starts[i]..ends[i])`
/// under `op`, but only at positions the caller has already flagged as
/// live in `matches` (a *flat* array covering every row's candidate range
/// back to back, not one array per row).
///
/// ELI5: `matches` is one long tape shared by every row. Each tape entry is
/// one tiny yes/no flag for one candidate position: `1` means "this position
/// is still in the running" and `0` means "an earlier condition already
/// ruled it out". As we scan row `i`'s slice of `right`, we walk the tape
/// forward one tick (`n += 1`) for every candidate position, even when the
/// flag says to skip it. That keeps two rows lined up with the single flat
/// tape even when their candidate ranges have different widths.
///
/// Returns `(result, counts_array, total)`: `result[n]` is `1`/`0` for
/// whether position `n` on the tape satisfies `op` (left as `0` when the
/// tape said skip), `counts_array[i]` is how many positions row `i`
/// matched, and `total` is the grand total across every row.
///
/// A row with an invalid/inverted range (`start` or `end` is `-1`, the
/// crate's "no match" sentinel, or `start >= end`, checked in `i64` space
/// before either bound is cast to `usize`) contributes zero ticks to the
/// tape, matching a genuinely empty range.
///
/// ELI5 (why `-1` is checked before the cast, not after): a `usize` can't
/// represent `-1`, so casting it doesn't keep it negative -- it wraps
/// around to the *largest* possible `usize` instead. That's bigger than
/// any real `start`, so a `start_ >= end_` check done *after* casting
/// would think the row has a huge, valid range instead of no match at
/// all, and the loop would walk `right`/`matches` straight past their
/// real length.
pub fn compare_start_end_core<T: PartialOrd + Copy>(
    left: ArrayView1<T>,
    right: ArrayView1<T>,
    starts: ArrayView1<i64>,
    ends: ArrayView1<i64>,
    matches: ArrayView1<i8>,
    op: CompareOp,
) -> (Array1<i8>, Array1<i64>, i64) {
    let mut result = Array1::<i8>::zeros(matches.len());
    let mut counts_array = Array1::<i64>::zeros(left.len());
    let mut total: i64 = 0;
    let mut n: usize = 0;
    let zipped = izip!(left.into_iter(), starts.into_iter(), ends.into_iter());
    for (position, (left_val, start, end)) in zipped.enumerate() {
        if *start == -1 || *end == -1 || *start >= *end {
            // No candidates for this row: 0 ticks on the shared `matches`
            // tape, matching a genuinely empty [start, end) range. A lone
            // `-1` sentinel cast to `usize` would otherwise wrap past
            // `arr.len()`/`matches.len()` instead of contributing nothing.
            continue;
        }
        let start_ = *start as usize;
        let end_ = *end as usize;
        let mut counter: i64 = 0;
        for nn in start_..end_ {
            if matches[n] == 0 {
                n += 1;
                continue;
            }
            let right_val = right[nn];
            let compare = op.apply(left_val, &right_val);
            counter += compare as i64;
            total += compare as i64;
            result[n] = compare as i8;
            n += 1;
        }
        counts_array[position] = counter;
    }
    (result, counts_array, total)
}

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            matches: PyReadonlyArray1<'py, i8>,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64)> {
            let starts_view = starts.as_array();
            let ends_view = ends.as_array();
            // ELI5: mirrors `compare_start_end_core`'s own row-rejection
            // condition exactly (`start`/`end` sentinel or inverted range),
            // not `checked_range`'s -- the core here never bounds `end`
            // against `right.len()`, so reusing `checked_range` would
            // under-count the width a too-large `end` actually walks.
            let expected_matches_width: usize = starts_view
                .iter()
                .zip(ends_view.iter())
                .filter(|(s, e)| **s != -1 && **e != -1 && **s < **e)
                .map(|(s, e)| (*e as usize) - (*s as usize))
                .sum();
            ensure_tape_width(expected_matches_width, matches.as_array().len())?;
            let op = CompareOp::try_from_code(op)?;
            let (result, counts_array, total) = compare_start_end_core(
                left.as_array(),
                right.as_array(),
                starts_view,
                ends_view,
                matches.as_array(),
                op,
            );
            Ok((
                result.into_pyarray(py),
                counts_array.into_pyarray(py),
                total,
            ))
        }
    };
}

generic_compare!(compare_start_end_int64, i64);
generic_compare!(compare_start_end_int32, i32);
generic_compare!(compare_start_end_int16, i16);
generic_compare!(compare_start_end_int8, i8);
generic_compare!(compare_start_end_uint64, u64);
generic_compare!(compare_start_end_uint32, u32);
generic_compare!(compare_start_end_uint16, u16);
generic_compare!(compare_start_end_uint8, u8);
generic_compare!(compare_start_end_f64, f64);
generic_compare!(compare_start_end_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    use CompareOp::{Eq as EQ, Ge as GE, Gt as GT, Le as LE, Lt as LT, Ne as NE};

    #[test]
    fn each_op_code_matches_its_operator() {
        let left = array![5_i64];
        let right = array![5_i64, 4, 6];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let matches = array![1_i8, 1, 1]; // nothing pre-filtered

        let cases = [
            (GT, [0_i8, 1, 0]), // 5>5 f, 5>4 t, 5>6 f
            (GE, [1, 1, 0]),
            (LT, [0, 0, 1]),
            (LE, [1, 0, 1]),
            (EQ, [1, 0, 0]),
            (NE, [0, 1, 1]),
        ];
        for (op, expected) in cases {
            let (result, counts, total) = compare_start_end_core(
                left.view(),
                right.view(),
                starts.view(),
                ends.view(),
                matches.view(),
                op,
            );
            assert_eq!(result, Array1::from_vec(expected.to_vec()), "op={op:?}");
            let expected_count: i64 = expected.iter().map(|&v| v as i64).sum();
            assert_eq!(counts, array![expected_count], "op={op:?}");
            assert_eq!(total, expected_count, "op={op:?}");
        }
    }

    #[test]
    fn sentinel_range_contributes_zero_ticks_not_a_panic() {
        // -1 is the crate's "invalid/no match" sentinel. Cast naively to
        // usize alone it becomes usize::MAX, and the inner loop would
        // walk `right`/`matches` straight out of bounds instead of
        // contributing zero ticks to the shared tape.
        let left = array![3_i64, 5_i64];
        let right = array![1_i64, 4, 9];
        let starts = array![0_i64, 1_i64];
        let ends = array![3_i64, -1_i64];
        let matches = array![1_i8, 1, 1];

        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            LT,
        );
        // row 0: 3<1 f, 3<4 t, 3<9 t
        assert_eq!(result, array![0, 1, 1]);
        assert_eq!(counts, array![2, 0]);
        assert_eq!(total, 2);
    }

    #[test]
    fn matches_mask_skips_positions_but_still_advances_the_tape() {
        // right has 3 candidates for the single row; the middle one is
        // pre-filtered out via matches=0, so it must not be compared even
        // though 4 > 3 would otherwise be true.
        let left = array![3_i64];
        let right = array![1_i64, 4, 9];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let matches = array![1_i8, 0, 1];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, array![1_i8, 0, 0]); // middle stays 0, not evaluated
        assert_eq!(counts, array![1]);
        assert_eq!(total, 1);
    }

    #[test]
    fn empty_range_contributes_nothing() {
        let left = array![3_i64];
        let right: Array1<i64> = array![];
        let starts = array![0_i64];
        let ends = array![0_i64];
        let matches: Array1<i8> = array![];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        assert_eq!(result, Array1::<i8>::zeros(0));
        assert_eq!(counts, array![0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn zero_matches_gives_zero_total_and_zero_counts() {
        let left = array![1_i64, 2];
        let right = array![100_i64, 200];
        let starts = array![0_i64, 0];
        let ends = array![2_i64, 2];
        let matches = array![1_i8, 1, 1, 1]; // flat tape: 2 rows x 2 candidates
        let (_, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT, // left values are far smaller than right values -> never `>`
        );
        assert_eq!(counts, array![0, 0]);
        assert_eq!(total, 0);
    }

    #[test]
    fn duplicate_values_all_count_under_ge_and_le() {
        let left = array![5_i64];
        let right = array![5_i64, 5, 5];
        let starts = array![0_i64];
        let ends = array![3_i64];
        let matches = array![1_i8, 1, 1];
        let (_, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GE,
        );
        assert_eq!(counts, array![3]);
        assert_eq!(total, 3);
    }

    #[test]
    fn multiple_rows_share_one_flat_matches_tape() {
        // two rows, each with a 2-wide candidate range: the tape length is
        // the sum of both rows' widths, not per-row.
        let left = array![1_i64, 10];
        let right = array![0_i64, 2, 9, 11];
        let starts = array![0_i64, 2];
        let ends = array![2_i64, 4];
        let matches = array![1_i8, 1, 1, 1];
        let (result, counts, total) = compare_start_end_core(
            left.view(),
            right.view(),
            starts.view(),
            ends.view(),
            matches.view(),
            GT,
        );
        // row 0: 1>0 t, 1>2 f ; row 1: 10>9 t, 10>11 f
        assert_eq!(result, array![1_i8, 0, 1, 0]);
        assert_eq!(counts, array![1, 1]);
        assert_eq!(total, 2);
    }
}
