use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::checked_range;
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
    for (start, end) in starts.iter().zip(ends.iter()) {
        let start = usize::try_from(*start).map_err(|_| "range bounds must be non-negative")?;
        let end = usize::try_from(*end).map_err(|_| "range bounds must be non-negative")?;
        if start > index.len() || end > index.len() {
            return Err("range bounds must not exceed right_index length");
        }
        if start > end {
            return Err("range start must not exceed range end");
        }
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
/// old implementation did.
pub fn prod_rev_start_end_int_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_i64: F,
) -> Result<(Array1<i64>, Array1<i64>), &'static str>
where
    T: Copy,
    F: FnMut(T) -> i64,
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
                    products.push(1_i64);
                    slot
                }
            };
            if !*boolean {
                products[slot] = products[slot].wrapping_mul(to_i64(*current));
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
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
            let result = prod_rev_start_end_int_core(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as i64,
            )
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
            let result = prod_rev_start_end_float_core(
                arr.as_array(),
                starts.as_array(),
                ends.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((result.0.into_pyarray(py), result.1.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_prod_rev_start_end_int64, i64);
compute_ints!(compute_prod_rev_start_end_int32, i32);
compute_ints!(compute_prod_rev_start_end_int16, i16);
compute_ints!(compute_prod_rev_start_end_int8, i8);
compute_ints!(compute_prod_rev_start_end_uint64, u64);
compute_ints!(compute_prod_rev_start_end_uint32, u32);
compute_ints!(compute_prod_rev_start_end_uint16, u16);
compute_ints!(compute_prod_rev_start_end_uint8, u8);
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
