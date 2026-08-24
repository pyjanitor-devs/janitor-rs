use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_index, ensure_equal_lengths};

/// Pure-Rust reverse-max core for the "no range" (equi-join) shape: for
/// every `(left_index[i], right_index[i])` pair, `arr[left_index[i]]`
/// competes to become row position `right_index[i]`'s new best, skipping
/// the comparison (but still touching the slot) when `booleans` flags
/// that left row as null. Only the *winning row's position* is returned,
/// not its value -- pyjanitor uses it as `ser.iloc[out]` downstream, so
/// there's no need to carry the value itself past this function. Takes
/// plain `ArrayView1`s, not PyO3 types, so it can be tested and
/// benchmarked without a Python interpreter -- see `benches/kernels.rs`.
///
/// ELI5: one slot per row position, holding (winning row's position in
/// `arr`, winning row's value) together -- see issue #23 for the
/// dense-slots rationale and issue #48 for why this used to be two
/// same-keyed `HashMap`s.
pub fn max_rev_no_range_core<T>(
    arr: ArrayView1<T>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
) -> (Array1<i64>, Array1<i64>)
where
    T: Copy + Default + PartialOrd,
{
    let mut slots: DenseSlots<(i64, T)> = DenseSlots::new(length);
    let zipped = left_index.into_iter().zip(right_index);
    for (index_left, index_right) in zipped {
        // ELI5: `index_left` names a position in `arr`/`booleans`, read
        // straight from the caller-supplied `left_index` array -- unlike
        // a `start..end` range, there's no natural "empty" fallback here,
        // so an out-of-bounds or negative value must be rejected before
        // it's used to index anything. `right_index` is never used to
        // index an array directly (only as a `DenseSlots` key), so it
        // doesn't need `checked_index`'s upper-bound check -- `touch`
        // itself grows to fit or rejects a negative key; see issue #69.
        let Some(left) = checked_index(*index_left, arr.len()) else {
            continue;
        };
        let current = arr[left];
        let boolean = booleans[left];
        let Some((base, base_val)) = slots.touch(*index_right, (-1, current)) else {
            continue;
        };
        if boolean {
            continue;
        }
        if (*base == -1) || (current > *base_val) {
            *base_val = current;
            *base = left as i64;
        }
    }
    slots.to_arrays(|(base, _base_val)| *base)
}

macro_rules! compute {
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
            let (indexers, result) = max_rev_no_range_core(
                arr.as_array(),
                left_index.as_array(),
                right_index.as_array(),
                booleans.as_array(),
                length as usize,
            );
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute!(compute_max_rev_no_range_int64, i64);
compute!(compute_max_rev_no_range_int32, i32);
compute!(compute_max_rev_no_range_int16, i16);
compute!(compute_max_rev_no_range_int8, i8);
compute!(compute_max_rev_no_range_uint64, u64);
compute!(compute_max_rev_no_range_uint32, u32);
compute!(compute_max_rev_no_range_uint16, u16);
compute!(compute_max_rev_no_range_uint8, u8);
compute!(compute_max_rev_no_range_f64, f64);
compute!(compute_max_rev_no_range_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_max_rev_no_range_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::compute_max_rev_no_range_int64;

    #[test]
    fn touched_row_positions_are_emitted_ascending_with_winning_row_index() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // Row 0 (val 10) and row 2 (val 20) both feed group 1; row 2
            // wins. Row 1 (val 30) is the only feeder for group 0. The
            // result is the *winning row's position*, not its value --
            // pyjanitor uses it as `ser.iloc[out]` downstream.
            let arr = PyArray1::from_vec(py, vec![10_i64, 30, 20]);
            let left_index = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let right_index = PyArray1::from_vec(py, vec![1_i64, 0, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false, false]);
            let (indexers, result) = compute_max_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
                2,
            )
            .expect("valid equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![0, 1]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![1, 2]);
        });
    }

    // Regression test for issue #69 -- see the matching test in
    // sum_rev/sum_no_range.rs for the full rationale: `length` is only a
    // capacity hint, never a bound on `right_index` values.
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
            let (indexers, result) = compute_max_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
                1,
            )
            .expect("a sparse right_index beyond length must not error or panic");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![10]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0]);
        });
    }
}
