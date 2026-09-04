use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{
    checked_index, checked_range, ensure_equal_lengths_core, ensure_nonempty_core,
    should_use_dense_match_storage, WrapMul,
};
use std::collections::{hash_map::Entry, HashMap};

/// Multiply values for each label reached through the positions tape.
/// `index` is expected to contain unique labels, as guaranteed by the
/// pyjanitor producer for this path.
///
/// ELI5: the HashMap is keyed directly by each validated ordinal. New labels
/// start at the multiplicative identity 1, and pairs are emitted in HashMap
/// iteration order.
/// Integer products use wrapping arithmetic, so overflow has the same
/// deterministic result in debug and release builds. `A` is the accumulator
/// type: every integer dtype instantiates this with `A = i64`, except
/// `uint64`, which instantiates it with `A = u64` so values `>= 2**63`
/// don't get sign-flipped by a forced `i64` cast (see `WrapMul`).
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive start of each positional range.
/// * `ends` - Exclusive end of each positional range.
/// * `index` - Right-side labels in ordinal position order.
/// * `positions` - Positional candidate tape mapping to `index`.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
#[allow(clippy::too_many_arguments)]
pub fn prod_positions_int_core<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_value: F,
) -> Result<(Vec<i64>, Vec<A>), String>
where
    T: Copy,
    A: WrapMul,
    F: Fn(T) -> A,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let dense = should_use_dense_match_storage(index.len(), positions.len());
    Ok(prod_positions_int_core_with_storage(
        arr, starts, ends, index, positions, booleans, to_value, dense,
    ))
}

/// Run integer positional product aggregation with an explicit storage mode.
/// This is a Rust-only benchmark entry point; production callers should use
/// [`prod_positions_int_core`] for automatic dispatch.
///
/// The array arguments and `to_value` have the same meanings as
/// [`prod_positions_int_core`]. `dense` selects vector storage when true and
/// HashMap storage when false.
#[allow(clippy::too_many_arguments)]
pub fn prod_positions_int_core_with_storage<T, A, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_value: F,
    dense: bool,
) -> (Vec<i64>, Vec<A>)
where
    T: Copy,
    A: WrapMul,
    F: Fn(T) -> A,
{
    // ELI5: reserve only for ordinals actually encountered; the tape length
    // is not a reliable estimate of distinct right-side state.
    if dense {
        let mut seen = vec![false; index.len()];
        let mut products = vec![A::ONE; index.len()];
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
                seen[indexer_] = true;
                // Insert state before checking the mask: null-only labels
                // must still be emitted with the multiplicative identity.
                if !*boolean {
                    products[indexer_] = products[indexer_].wrap_mul(current_);
                }
            }
        }
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for (ordinal, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[ordinal]);
                values.push(products[ordinal]);
            }
        }
        return (labels, values);
    }

    let mut products: HashMap<usize, A> = HashMap::new();
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
            let product = match products.entry(indexer_) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(A::ONE),
            };
            // Insert state before checking the mask: null-only labels must
            // still be emitted with the multiplicative identity.
            if !*boolean {
                // ELI5: use the same defined wraparound arithmetic as the
                // forward kernel, so debug and release builds agree when an
                // integer product exceeds its type's range.
                *product = product.wrap_mul(current_);
            }
        }
    }
    let mut labels = Vec::with_capacity(products.len());
    let mut values = Vec::with_capacity(products.len());
    for (ordinal, product) in products {
        labels.push(index[ordinal]);
        values.push(product);
    }
    (labels, values)
}

#[allow(clippy::too_many_arguments)]
/// Multiply floating-point values for each label reached through `positions`.
/// `index` is expected to contain unique labels, as guaranteed by the
/// pyjanitor producer for this path.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive start of each positional range.
/// * `ends` - Exclusive end of each positional range.
/// * `index` - Right-side labels in ordinal position order.
/// * `positions` - Positional candidate tape mapping to `index`.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn prod_positions_float_core<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_value: F,
) -> Result<(Vec<i64>, Vec<f64>), String>
where
    T: Copy,
    F: Fn(T) -> f64,
{
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("starts", starts.len())?;
    ensure_nonempty_core("ends", ends.len())?;
    ensure_nonempty_core("index", index.len())?;
    ensure_nonempty_core("positions", positions.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "ends", ends.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let dense = should_use_dense_match_storage(index.len(), positions.len());
    Ok(prod_positions_float_core_with_storage(
        arr, starts, ends, index, positions, booleans, to_value, dense,
    ))
}

/// Run floating-point positional product aggregation with an explicit storage mode.
/// This is a Rust-only benchmark entry point; production callers should use
/// [`prod_positions_float_core`] for automatic dispatch.
///
/// The array arguments and `to_value` have the same meanings as
/// [`prod_positions_float_core`]. `dense` selects vector storage when true and
/// HashMap storage when false.
#[allow(clippy::too_many_arguments)]
pub fn prod_positions_float_core_with_storage<T, F>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    to_value: F,
    dense: bool,
) -> (Vec<i64>, Vec<f64>)
where
    T: Copy,
    F: Fn(T) -> f64,
{
    if dense {
        let mut seen = vec![false; index.len()];
        let mut products = vec![1.; index.len()];
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
                seen[indexer_] = true;
                // Insert state before checking the mask: null-only labels
                // must still be emitted with the multiplicative identity.
                if !*boolean {
                    products[indexer_] *= current_;
                }
            }
        }
        let mut labels = Vec::new();
        let mut values = Vec::new();
        for (ordinal, was_seen) in seen.into_iter().enumerate() {
            if was_seen {
                labels.push(index[ordinal]);
                values.push(products[ordinal]);
            }
        }
        return (labels, values);
    }

    let mut products: HashMap<usize, f64> = HashMap::new();
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
            let product = match products.entry(indexer_) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => entry.insert(1.),
            };
            // Insert state before checking the mask: null-only labels must
            // still be emitted with the multiplicative identity.
            if !*boolean {
                *product *= current_;
            }
        }
    }
    let mut labels = Vec::with_capacity(products.len());
    let mut values = Vec::with_capacity(products.len());
    for (ordinal, product) in products {
        labels.push(index[ordinal]);
        values.push(product);
    }
    (labels, values)
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// Finds products for labels reached through the positional candidate
        /// tape.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive start of each positional range.
        /// * `ends` - Exclusive end of each positional range.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `positions` - Positional candidate tape mapping to `index`.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        ///
        /// `arr`, `index`, and `positions` must not be empty. `starts`, `ends`,
        /// and `booleans` must match `arr` in length. Returned label/value
        /// pairs are aligned but unordered.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let (labels, products) =
                prod_positions_int_core(arr, starts, ends, index, positions, booleans, |value| {
                    value as $acc
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            let indexers = Array1::from_vec(labels);
            let result = Array1::from_vec(products);
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapMul`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_positions_int64, i64, i64);
compute_ints!(compute_prod_rev_positions_int32, i32, i64);
compute_ints!(compute_prod_rev_positions_int16, i16, i64);
compute_ints!(compute_prod_rev_positions_int8, i8, i64);
compute_ints!(compute_prod_rev_positions_uint64, u64, u64);
compute_ints!(compute_prod_rev_positions_uint32, u32, i64);
compute_ints!(compute_prod_rev_positions_uint16, u16, i64);
compute_ints!(compute_prod_rev_positions_uint8, u8, i64);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// Finds products for labels reached through the positional candidate
        /// tape.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive start of each positional range.
        /// * `ends` - Exclusive end of each positional range.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `positions` - Positional candidate tape mapping to `index`.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        ///
        /// `arr`, `index`, and `positions` must not be empty. `starts`, `ends`,
        /// and `booleans` must match `arr` in length. Returned label/value
        /// pairs are aligned but unordered.
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
            let index = index.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let (labels, products) =
                prod_positions_float_core(arr, starts, ends, index, positions, booleans, |value| {
                    value as f64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
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
    fn positions_keep_labels_and_products() {
        let arr = array![2_i64, 3, 4];
        let starts = array![0_i64, 2, 4];
        let ends = array![2_i64, 4, 5];
        let index = array![10_i64, 20, 30];
        let positions = array![0_i64, 1, 2, 1, -1];
        let booleans = array![false, false, false];
        let (labels, products) = prod_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();
        let mut got: Vec<_> = labels.into_iter().zip(products).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(10, 2), (20, 6), (30, 3)]);
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
            |value| value,
        )
        .unwrap();
        assert_eq!(labels, vec![4]);
        assert_eq!(products, vec![1]);
    }

    #[test]
    fn floating_positions_handle_unique_labels_and_zero_values() {
        let arr = array![2.0_f64, 3.0, 4.0, 0.0];
        let starts = array![0_i64, 2, 4, 6];
        let ends = array![2_i64, 4, 6, 8];
        let index = array![10_i64, 20, 30];
        let positions = array![0_i64, 1, 2, 1, 0, 2, 0, 2];
        let booleans = array![false, false, false, false];

        let (labels, products) = prod_positions_float_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |value| value,
        )
        .unwrap();

        let mut got: Vec<_> = labels.into_iter().zip(products).collect();
        got.sort_unstable_by_key(|(label, _)| *label);
        assert_eq!(got, vec![(10, 0.0), (20, 6.0), (30, 0.0)]);
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
            |value| value,
        )
        .unwrap();

        assert_eq!(labels, vec![4]);
        assert_eq!(products, vec![1.0]);
    }

    #[test]
    fn u64_accumulator_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let arr = array![value];
        let starts = array![0_i64];
        let ends = array![1_i64];
        let index = array![5_i64];
        let positions = array![0_i64];
        let booleans = array![false];
        let (labels, products) = prod_positions_int_core(
            arr.view(),
            starts.view(),
            ends.view(),
            index.view(),
            positions.view(),
            booleans.view(),
            |v: u64| v,
        )
        .unwrap();
        assert_eq!(labels, vec![5]);
        assert_eq!(products, vec![value]);
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
            |value| value,
        )
        .unwrap();

        assert_eq!(labels, vec![5]);
        assert_eq!(products, vec![-2]);
    }
}
