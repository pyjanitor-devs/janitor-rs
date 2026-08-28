use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_range, ensure_equal_lengths, WrapMul};
use std::collections::{hash_map::Entry, HashMap};

fn validate_inputs<T>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(), &'static str> {
    if starts.len() != ends.len() {
        return Err("starts and ends must have equal lengths");
    }
    if arr.len() != starts.len() {
        return Err("arr, starts, and ends must have equal lengths");
    }
    if arr.len() != booleans.len() {
        return Err("arr and booleans must have equal lengths");
    }
    if arr.is_empty() || index.is_empty() {
        return Err("arr, starts, booleans, and index cannot be empty");
    }
    Ok(())
}

fn capacity_hint(
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    right_len: usize,
) -> usize {
    // ELI5: the total number of covered positions is a safe ceiling for the
    // number of distinct labels, so it limits rehashing without trusting a
    // Python-side size or reserving beyond the right index.
    starts
        .iter()
        .zip(ends.iter())
        .filter_map(|(start, end)| checked_range(*start, *end, right_len))
        .fold(0_usize, |total, (start, end)| {
            total.saturating_add(end - start)
        })
        .min(right_len)
}

/// Multiply values into one compact state slot for each distinct label.
///
/// ELI5: every label has one numbered drawer containing its product. A null
/// row leaves the drawer at the multiplicative identity, `1`, just as the
/// old implementation did. `A` is the accumulator type: every integer dtype
/// instantiates this with `A = i64`, except `uint64`, which instantiates it
/// with `A = u64` so values `>= 2**63` don't get sign-flipped by a forced
/// `i64` cast (see `WrapMul`).
pub fn prod_rev_start_end_int_core<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Array1<i64>, Array1<A>), &'static str>
where
    T: Copy,
    A: WrapMul,
    F: FnMut(T) -> A,
{
    validate_inputs(arr, starts, ends, index, booleans)?;
    let mut slots = HashMap::with_capacity(capacity_hint(starts, ends, index.len()));
    let mut labels = Vec::new();
    let mut products = Vec::new();
    for (current, start, end, boolean) in izip!(arr, starts, ends, booleans) {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start..end {
            let label = index[item];
            let slot = match slots.entry(label) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    products.push(A::ONE);
                    slot
                }
            };
            if !*boolean {
                products[slot] = products[slot].wrap_mul(convert(*current));
            }
        }
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(products)))
}

pub fn prod_rev_start_end_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> f64,
{
    validate_inputs(arr, starts, ends, index, booleans)?;
    let mut slots = HashMap::with_capacity(capacity_hint(starts, ends, index.len()));
    let mut labels = Vec::new();
    let mut products = Vec::new();
    for (current, start, end, boolean) in izip!(arr, starts, ends, booleans) {
        let Some((start, end)) = checked_range(*start, *end, index.len()) else {
            continue;
        };
        for item in start..end {
            let label = index[item];
            let slot = match slots.entry(label) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    products.push(1.0_f64);
                    slot
                }
            };
            if !*boolean {
                products[slot] *= to_f64(*current);
            }
        }
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(products)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)> {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let result = prod_rev_start_end_int_core(arr, starts, ends, index, booleans, |value| {
                value as $acc
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
        }
    };
}

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            let result =
                prod_rev_start_end_float_core(arr, starts, ends, index, booleans, |value| {
                    value as f64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapMul`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_start_end_int64, i64, i64);
compute_ints!(compute_prod_rev_start_end_int32, i32, i64);
compute_ints!(compute_prod_rev_start_end_int16, i16, i64);
compute_ints!(compute_prod_rev_start_end_int8, i8, i64);
compute_ints!(compute_prod_rev_start_end_uint64, u64, u64);
compute_ints!(compute_prod_rev_start_end_uint32, u32, i64);
compute_ints!(compute_prod_rev_start_end_uint16, u16, i64);
compute_ints!(compute_prod_rev_start_end_uint8, u8, i64);
compute_floats!(compute_prod_rev_start_end_f64, f64);
compute_floats!(compute_prod_rev_start_end_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_end_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let got = prod_rev_start_end_int_core(
            array![value].view(),
            array![0_i64].view(),
            array![1_i64].view(),
            array![10_i64].view(),
            array![false].view(),
            |v: u64| v,
        );
        assert_eq!(got, Ok((array![10], array![value])));
    }

    #[test]
    fn compact_slots_multiply_duplicate_labels() {
        let got = prod_rev_start_end_int_core(
            array![2_i64, 3, 4].view(),
            array![0_i64, 1, 0].view(),
            array![2_i64, 3, 1].view(),
            array![10_i64, 20, 10].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10, 20], array![24, 6])));
    }

    #[test]
    fn null_rows_emit_labels_with_multiplicative_identity() {
        let got = prod_rev_start_end_int_core(
            array![5_i64].view(),
            array![0_i64].view(),
            array![2_i64].view(),
            array![10_i64, 20].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![10, 20], array![1, 1])));
    }

    #[test]
    fn invalid_or_zero_width_ranges_are_skipped() {
        let got = prod_rev_start_end_int_core(
            array![1_i64, 2, 3].view(),
            array![2_i64, -1, 1].view(),
            array![2_i64, 1, 4].view(),
            array![10_i64, 20].view(),
            array![false, false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![], array![])));
    }

    #[test]
    fn validation_rejects_shape_mismatches_and_empty_inputs() {
        assert!(prod_rev_start_end_int_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![1_i64, 1].view(),
            array![10_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
        assert!(prod_rev_start_end_int_core(
            (array![] as Array1<i64>).view(),
            array![].view(),
            array![].view(),
            array![10_i64].view(),
            array![].view(),
            |value: i64| value,
        )
        .is_err());
    }
}
