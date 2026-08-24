use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::dense::DenseSlots;
use crate::aggs::{checked_index, ensure_equal_lengths};

/// Pure-Rust reverse-product core for the "no range" (equi-join) shape:
/// for every `(left_index[i], right_index[i])` pair, multiply
/// `arr[left_index[i]]` (cast via `to_i64`) into row position
/// `right_index[i]`'s running product, skipping the multiplication (but
/// still touching the slot, at the multiplicative identity `1`) when
/// `booleans` flags that left row as null. Takes plain `ArrayView1`s, not
/// PyO3 types, so it can be tested and benchmarked without a Python
/// interpreter -- see `benches/kernels.rs`.
pub fn prod_rev_no_range_int_core<T, F>(
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
        let total = slots.touch(*index_right as usize, 1);
        if boolean {
            continue;
        }
        *total *= current;
    }
    slots.to_arrays(|value| *value)
}

/// `i64` benchmark/test entry point that follows the same cast-on-access
/// path as the corresponding Python wrapper (an identity cast here).
pub fn prod_rev_no_range_i64_core(
    arr: ArrayView1<i64>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
) -> (Array1<i64>, Array1<i64>) {
    prod_rev_no_range_int_core(arr, left_index, right_index, booleans, length, |value| {
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
            let (indexers, result) = prod_rev_no_range_int_core(
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

compute_ints!(compute_prod_rev_no_range_int64, i64);
compute_ints!(compute_prod_rev_no_range_int32, i32);
compute_ints!(compute_prod_rev_no_range_int16, i16);
compute_ints!(compute_prod_rev_no_range_int8, i8);
compute_ints!(compute_prod_rev_no_range_uint64, u64);
compute_ints!(compute_prod_rev_no_range_uint32, u32);
compute_ints!(compute_prod_rev_no_range_uint16, u16);
compute_ints!(compute_prod_rev_no_range_uint8, u8);

/// Pure-Rust reverse-product core for the "no range" (equi-join) shape's
/// float path: same touch/accumulate structure as
/// `prod_rev_no_range_int_core`, but the running product and its result
/// are `f64` -- unlike sum, product needs no Kahan/Neumaier compensation
/// state, so this stays a plain scalar accumulator, not a tuple.
pub fn prod_rev_no_range_float_core<T, F>(
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
    let mut slots: DenseSlots<f64> = DenseSlots::new(length);
    let zipped = left_index.into_iter().zip(right_index);
    for (index_left, index_right) in zipped {
        let Some(left) = checked_index(*index_left, arr.len()) else {
            continue;
        };
        let current = to_f64(arr[left]);
        let boolean = booleans[left];
        let total = slots.touch(*index_right as usize, 1.);
        if boolean {
            continue;
        }
        *total *= current;
    }
    slots.to_arrays(|value| *value)
}

/// `f64` benchmark/test entry point that follows the same cast-on-access
/// path as the corresponding Python wrapper (an identity cast here).
pub fn prod_rev_no_range_f64_core(
    arr: ArrayView1<f64>,
    left_index: ArrayView1<i64>,
    right_index: ArrayView1<i64>,
    booleans: ArrayView1<bool>,
    length: usize,
) -> (Array1<i64>, Array1<f64>) {
    prod_rev_no_range_float_core(arr, left_index, right_index, booleans, length, |value| {
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
            let (indexers, result) = prod_rev_no_range_float_core(
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

compute_floats!(compute_prod_rev_no_range_f64, f64);
compute_floats!(compute_prod_rev_no_range_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_no_range_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod correctness_tests {
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    use super::compute_prod_rev_no_range_int64;

    #[test]
    fn touched_row_positions_are_emitted_ascending_with_products() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            // Row 0 (val 2) and row 2 (val 4) both feed group 1: 2*4=8.
            // Row 1 (val 3) is the only feeder for group 0.
            let arr = PyArray1::from_vec(py, vec![2_i64, 3, 4]);
            let left_index = PyArray1::from_vec(py, vec![0_i64, 1, 2]);
            let right_index = PyArray1::from_vec(py, vec![1_i64, 0, 1]);
            let booleans = PyArray1::from_vec(py, vec![false, false, false]);
            let (indexers, result) = compute_prod_rev_no_range_int64(
                py,
                arr.readonly(),
                left_index.readonly(),
                right_index.readonly(),
                booleans.readonly(),
                2,
            )
            .expect("valid equal-length inputs must not error");
            assert_eq!(indexers.readonly().to_vec().unwrap(), vec![0, 1]);
            assert_eq!(result.readonly().to_vec().unwrap(), vec![3, 8]);
        });
    }
}
