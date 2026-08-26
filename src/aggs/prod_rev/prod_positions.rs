use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, checked_range, ensure_equal_lengths};
use std::collections::HashMap;

/// Multiply values for each label reached through the positions tape.
///
/// ELI5: the HashMap gives each label a compact slot, while labels and
/// products live in Vecs. New labels start at the multiplicative identity 1.
pub fn prod_positions_int_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    capacity: usize,
    to_value: F,
) -> (Vec<i64>, Vec<i64>)
where
    T: Copy,
    F: Fn(T) -> i64,
{
    let capacity = capacity.min(index.len()).min(positions.len());
    let mut slots: HashMap<i64, usize> = HashMap::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut products = Vec::with_capacity(capacity);
    for (current, start, end, boolean) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    ) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let current_ = to_value(*current);
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let label = index[indexer_];
            let slot = match slots.entry(label) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    products.push(1_i64);
                    slot
                }
            };
            if !*boolean {
                // ELI5: use the same defined wraparound arithmetic as the
                // forward kernel, so debug and release builds agree when an
                // integer product exceeds its type's range.
                products[slot] = products[slot].wrapping_mul(current_);
            }
        }
    }
    (labels, products)
}

pub fn prod_positions_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    capacity: usize,
    to_value: F,
) -> (Vec<i64>, Vec<f64>)
where
    T: Copy,
    F: Fn(T) -> f64,
{
    let capacity = capacity.min(index.len()).min(positions.len());
    let mut slots: HashMap<i64, usize> = HashMap::with_capacity(capacity);
    let mut labels = Vec::with_capacity(capacity);
    let mut products = Vec::with_capacity(capacity);
    for (current, start, end, boolean) in izip!(
        arr.into_iter(),
        starts.into_iter(),
        ends.into_iter(),
        booleans.into_iter()
    ) {
        let Some((start_, end_)) = checked_range(*start, *end, positions.len()) else {
            continue;
        };
        let current_ = to_value(*current);
        for nn in start_..end_ {
            let Some(indexer_) = checked_index(positions[nn], index.len()) else {
                continue;
            };
            let label = index[indexer_];
            let slot = match slots.entry(label) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let slot = labels.len();
                    entry.insert(slot);
                    labels.push(label);
                    products.push(1.);
                    slot
                }
            };
            if !*boolean {
                products[slot] *= current_;
            }
        }
    }
    (labels, products)
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
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            if arr.is_empty() || index.is_empty() || positions.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr, starts, ends, booleans, index, and positions cannot be empty",
                ));
            }
            let (labels, products) = prod_positions_int_core(
                arr,
                starts,
                ends,
                index,
                positions,
                booleans,
                index.len().min(positions.len()),
                |value| value as i64,
            );
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(products);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_ints!(compute_prod_rev_positions_int64, i64);
compute_ints!(compute_prod_rev_positions_int32, i32);
compute_ints!(compute_prod_rev_positions_int16, i16);
compute_ints!(compute_prod_rev_positions_int8, i8);
compute_ints!(compute_prod_rev_positions_uint64, u64);
compute_ints!(compute_prod_rev_positions_uint32, u32);
compute_ints!(compute_prod_rev_positions_uint16, u16);
compute_ints!(compute_prod_rev_positions_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            ensure_equal_lengths("starts", starts.len(), "ends", ends.len())?;
            ensure_equal_lengths("arr", arr.len(), "starts", starts.len())?;
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            ensure_equal_lengths("arr", arr.len(), "booleans", booleans.len())?;
            if arr.is_empty() || index.is_empty() || positions.is_empty() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "arr, starts, ends, booleans, index, and positions cannot be empty",
                ));
            }
            let (labels, products) = prod_positions_float_core(
                arr,
                starts,
                ends,
                index,
                positions,
                booleans,
                index.len().min(positions.len()),
                |value| value as f64,
            );
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(products);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

compute_floats!(compute_prod_rev_positions_f64, f64);
compute_floats!(compute_prod_rev_positions_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_positions_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{prod_positions_float_core, prod_positions_int_core};
    use numpy::ndarray::array;

    #[test]
    fn positions_keep_first_seen_labels_and_products() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64, 2, 4];
        let ends = array![2_i64, 4, 5];
        let index = array![10_i64, 20, 10];
        let positions = array![0_i64, 1, 2, 1, -1];
        let booleans = array![false, false, false];
        let (labels, products) = prod_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            100,
            |value| value,
        );
        assert_eq!(labels, vec![10, 20]);
        assert_eq!(products, vec![6, 6]);
    }

    #[test]
    fn null_rows_do_not_change_product_from_identity() {
        let arr = array![7_i64];
        let starts = array![0_i64];
        let ends = array![1_i64];
        let index = array![4_i64];
        let positions = array![0_i64];
        let booleans = array![true];
        let (labels, products) = prod_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            1,
            |value| value,
        );
        assert_eq!(labels, vec![4]);
        assert_eq!(products, vec![1]);
    }

    #[test]
    fn floating_positions_handle_duplicate_labels_and_zero_values() {
        let arr = array![2.0_f64, 3.0, 4.0, 0.0];
        let starts = array![0_i64, 2, 4, 6];
        let ends = array![2_i64, 4, 6, 8];
        let index = array![10_i64, 20, 10];
        let positions = array![0_i64, 1, 2, 1, 0, 2, 0, 2];
        let booleans = array![false, false, false, false];

        let (labels, products) = prod_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            3,
            |value| value,
        );

        assert_eq!(labels, vec![10, 20]);
        assert_eq!(products, vec![0.0, 6.0]);
    }

    #[test]
    fn floating_positions_preserve_identity_for_null_rows() {
        let arr = array![7.0_f64];
        let starts = array![0_i64];
        let ends = array![1_i64];
        let index = array![4_i64];
        let positions = array![0_i64];
        let booleans = array![true];

        let (labels, products) = prod_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            1,
            |value| value,
        );

        assert_eq!(labels, vec![4]);
        assert_eq!(products, vec![1.0]);
    }

    #[test]
    fn integer_positions_wrap_on_overflow() {
        let arr = array![i64::MAX, 2];
        let starts = array![0_i64, 1];
        let ends = array![1_i64, 2];
        let index = array![5_i64];
        let positions = array![0_i64, 0];
        let booleans = array![false, false];

        let (labels, products) = prod_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            1,
            |value| value,
        );

        assert_eq!(labels, vec![5]);
        assert_eq!(products, vec![-2]);
    }
}
