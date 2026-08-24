use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_index, ensure_equal_lengths};

/// Pure-Rust reverse-sum core for the "no range" (equi-join) shape: for
/// every `(left_index[i], right_index[i])` pair, add `arr[left_index[i]]`
/// (cast via `to_i64`) into row position `right_index[i]`'s running total,
/// skipping the addition (but still touching the slot) when `booleans`
/// flags that left row as null. Takes plain `ArrayView1`s, not PyO3 types,
/// so it can be tested and benchmarked without a Python interpreter --
/// see `benches/kernels.rs`.
///
/// ELI5: `index_left` needs a bound check against `arr.len()` because it's
/// an ordinary index; `index_right` doesn't, because it's already a dense
/// row position guaranteed `< length` by construction on the pyjanitor
/// side (that's the assumption `DenseSlots` itself is built on -- see
/// issue #23).
pub fn sum_rev_no_range_int_core<T, F>(
    arr: ArrayView1<T>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
    mut to_i64: F,
) -> (Array1<i64>, Array1<i64>)
where
    T: Copy,
    F: FnMut(T) -> i64,
{
    let mut slots: DenseSlots<i64> = DenseSlots::new(length);
    let zipped = left_index.into_iter().zip(right_index);
    for (index_left, index_right) in zipped {
        let Some(left) = checked_index(*index_left, arr.len()) else {
            continue;
        };
        let current = to_i64(arr[left]);
        let boolean = booleans[left];
        let Some(total) = slots.touch(*index_right, 0) else {
            continue;
        };
        if boolean {
            continue;
        }
        *total += current;
    }
    slots.to_arrays(|value| *value)
}

/// `i64` benchmark/test entry point that follows the same cast-on-access
/// path as the corresponding Python wrapper (an identity cast here).
pub fn sum_rev_no_range_i64_core(
    arr: ArrayView1<i64>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
) -> (Array1<i64>, Array1<i64>) {
    sum_rev_no_range_int_core(arr, left_index, right_index, booleans, length, |value| {
        value
    })
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let (indexers, result) = sum_rev_no_range_int_core(
                arr.as_array(),
                left_index.as_array(),
                right_index.as_array(),
                booleans.as_array(),
                length as usize,
                |value| value as i64,
            );
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_sum_rev_no_range_int64, i64);
compute_ints!(compute_sum_rev_no_range_int32, i32);
compute_ints!(compute_sum_rev_no_range_int16, i16);
compute_ints!(compute_sum_rev_no_range_int8, i8);
compute_ints!(compute_sum_rev_no_range_uint64, u64);
compute_ints!(compute_sum_rev_no_range_uint32, u32);
compute_ints!(compute_sum_rev_no_range_uint16, u16);
compute_ints!(compute_sum_rev_no_range_uint8, u8);

/// Pure-Rust reverse-sum core for the "no range" (equi-join) shape's float
/// path: same touch/accumulate structure as `sum_rev_no_range_int_core`,
/// but each row position's slot carries a Neumaier-compensated `(total,
/// compensation)` pair instead of a plain `i64` -- see the inline
/// comments for why. Takes plain `ArrayView1`s for the same reason: no
/// Python interpreter needed to test or benchmark it.
pub fn sum_rev_no_range_float_core<T, F>(
    arr: ArrayView1<T>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
    mut to_f64: F,
) -> (Array1<i64>, Array1<f64>)
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    let mut slots: DenseSlots<(f64, f64)> = DenseSlots::new(length);
    let zipped = left_index.into_iter().zip(right_index);
    for (index_left, index_right) in zipped {
        let Some(left) = checked_index(*index_left, arr.len()) else {
            continue;
        };
        let current = to_f64(arr[left]);
        let boolean = booleans[left];
        let Some((total, compensation)) = slots.touch(*index_right, (0., 0.)) else {
            continue;
        };
        if boolean {
            continue;
        }
        let difference = current - *compensation;
        let increment = *total + difference;
        *compensation = (increment - *total) - difference;
        // adapted from pandas' cython code
        // # GH#53606; GH#60303
        // # If val is +/- infinity compensation is NaN
        // # which would lead to results being NaN instead
        // # of +/- infinity. We cannot use util.is_nan
        // # because of no gil
        if !compensation.is_finite() {
            *compensation = 0.;
        }
        *total = increment;
    }
    slots.to_arrays(|(total, _compensation)| *total)
}

/// `f64` benchmark/test entry point that follows the same cast-on-access
/// path as the corresponding Python wrapper (an identity cast here).
pub fn sum_rev_no_range_f64_core(
    arr: ArrayView1<f64>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
) -> (Array1<i64>, Array1<f64>) {
    sum_rev_no_range_float_core(arr, left_index, right_index, booleans, length, |value| {
        value
    })
}

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
            length: i64,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            ensure_equal_lengths(
                "arr",
                arr.as_array().len(),
                "booleans",
                booleans.as_array().len(),
            )?;
            let (indexers, result) = sum_rev_no_range_float_core(
                arr.as_array(),
                left_index.as_array(),
                right_index.as_array(),
                booleans.as_array(),
                length as usize,
                |value| value as f64,
            );
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_sum_rev_no_range_f64, f64);
compute_floats!(compute_sum_rev_no_range_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_sum_rev_no_range_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod determinism_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::{compute_sum_rev_no_range_f64, compute_sum_rev_no_range_int64};

    // Regression test for issue #69: `length` is only a capacity hint
    // (mirroring the old code's `HashMap::with_capacity(length)`), never a
    // bound on `right_index` values. pyjanitor's equi-join caller passes
    // `right_index.size` (the match COUNT) as `length`, while `right_index`
    // holds real right-dataframe row positions that can run far past that
    // count on a sparse join -- e.g. one match at row position 10 out of
    // an 11-row right dataframe gives `length=1`, `right_index=[10]`. The
    // old `DenseSlots` (pre-#69) indexed `seen[10]` into a 1-element `Vec`
    // and panicked; this must instead return that one row, unpanicked.
    #[test]
    fn length_far_below_the_true_right_index_domain_does_not_panic() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let arr = PyArray1::from_vec(py, vec![7_i64]);
            let left_index = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![10_i64]);
            let booleans = PyArray1::from_vec(py, vec![false]);
            let (indexers, result) = compute_sum_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
                1,
            )
            .expect("a sparse right_index beyond length must not error or panic");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![10]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![7]);
        });
    }

    // ELI5: issue #23's bug report showed the *old* HashMap-backed version
    // returning `right_index`/value pairs in a different order on every
    // call for identical inputs -- e.g. `[1, 0, 2, 3, 5, 4]` one time,
    // `[4, 1, 2, 3, 0, 5]` the next. These tests exercise the same
    // permuted-`right_index` shape (a dense one-to-one join, so every row
    // position gets touched exactly once) and check that the dense
    // accumulator always answers with the *same*, ascending order.
    #[test]
    fn repeated_calls_agree_and_are_sorted_ascending_by_row_position() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let arr_values = vec![10_i64, 20, 30, 40, 50, 60];
            let left_index_values = vec![0_i64, 1, 2, 3, 4, 5];
            // A permutation of 0..6, matching issue #23's reproduction shape.
            let right_index_values = vec![1_i64, 0, 2, 3, 5, 4];
            let booleans_values = vec![false; 6];

            let mut previous: Option<(Vec<i64>, Vec<i64>)> = None;
            for _ in 0..5 {
                let arr = PyArray1::from_vec(py, arr_values.clone());
                let left_index = PyArray1::from_vec(py, left_index_values.clone());
                let right_index = PyArray1::from_vec(py, right_index_values.clone());
                let booleans = PyArray1::from_vec(py, booleans_values.clone());
                let (indexers, result) = compute_sum_rev_no_range_int64(
                    py,
                    arr.readonly(),
                    left_index.readonly(),
                    right_index.readonly(),
                    booleans.readonly(),
                    6,
                )
                .expect("equal-length inputs must not error");
                let indexers = indexers.readonly().to_vec().unwrap();
                let result = result.readonly().to_vec().unwrap();

                assert_eq!(indexers, vec![0, 1, 2, 3, 4, 5], "output must be ascending");
                match &previous {
                    None => previous = Some((indexers, result)),
                    Some((prev_indexers, prev_result)) => {
                        assert_eq!(&indexers, prev_indexers);
                        assert_eq!(&result, prev_result);
                    }
                }
            }
            let (_, result) = previous.unwrap();
            // right_index[i] receives arr[left_index[i]]; row position 0
            // was fed by left row 1 (arr value 20), position 1 by left row
            // 0 (arr value 10), and so on.
            assert_eq!(result, vec![20, 10, 30, 40, 60, 50]);
        });
    }

    #[test]
    fn sparse_row_positions_emit_only_touched_slots_in_ascending_order() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // `length` (8) is wider than the row positions actually
            // touched (0, 2, 5) -- untouched slots must not appear at all,
            // matching the old `HashMap`'s "only keys that were ever
            // `.entry()`-ed show up" behavior.
            let arr = PyArray1::from_vec(py, vec![1_i64, 2, 3]);
            let left_index = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let right_index = PyArray1::from_vec(py, vec![5_i64, 0, 2]);
            let booleans = PyArray1::from_vec(py, vec![false, false, false]);
            let (indexers, result) = compute_sum_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
                8,
            )
            .expect("equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![0, 2, 5]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![2, 3, 1]);
        });
    }

    #[test]
    fn a_row_filtered_out_by_the_null_mask_still_touches_its_slot() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // Mirrors the old `HashMap::entry(...).or_insert(...)` call
            // happening before the null-mask `continue`: the row still
            // "shows up" for this right index, just at the identity value
            // (0 for sum), instead of being dropped from the output.
            let arr = PyArray1::from_vec(py, vec![99_i64]);
            let left_index = PyArray1::from_vec(py, vec![0_i64]);
            let right_index = PyArray1::from_vec(py, vec![0_i64]);
            let booleans = PyArray1::from_vec(py, vec![true]);
            let (indexers, result) = compute_sum_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
                1,
            )
            .expect("equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![0]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0]);
        });
    }

    #[test]
    fn float_path_stays_deterministic_and_keeps_compensated_precision() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // Three left rows all feed the same right slot -- exercises
            // the merged (total, compensation) tuple accumulator (the
            // shape issue #48 also targets) across repeated summation.
            let arr_values = vec![0.1_f64, 0.2, 0.3];
            let left_index_values = vec![0_i64, 1, 2];
            let right_index_values = vec![0_i64, 0, 0];
            let booleans_values = vec![false, false, false];

            let mut previous: Option<f64> = None;
            for _ in 0..3 {
                let arr = PyArray1::from_vec(py, arr_values.clone());
                let left_index = PyArray1::from_vec(py, left_index_values.clone());
                let right_index = PyArray1::from_vec(py, right_index_values.clone());
                let booleans = PyArray1::from_vec(py, booleans_values.clone());
                let (indexers, result) = compute_sum_rev_no_range_f64(
                    py,
                    arr.readonly(),
                    left_index.readonly(),
                    right_index.readonly(),
                    booleans.readonly(),
                    1,
                )
                .expect("equal-length inputs must not error");
                assert_eq!(indexers.readonly().to_vec().unwrap(), vec![0]);
                let total = result.readonly().to_vec().unwrap()[0];
                assert!((total - 0.6).abs() < 1e-9);
                if let Some(prev) = previous {
                    assert_eq!(prev, total, "compensated total must be reproducible");
                }
                previous = Some(total);
            }
        });
    }
}
