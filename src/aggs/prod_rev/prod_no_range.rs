use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{checked_index, ensure_equal_lengths_core, ensure_nonempty_core};
use std::collections::HashMap;

/// Multiply values grouped by right labels without range metadata.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `left_index` - Positions into `arr`; must be valid.
/// * `right_index` - Output label for each left position.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn prod_rev_no_range_int_core<T: Copy, F: FnMut(T) -> i64>(
    arr: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_i64: F,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    // Empty inputs are rejected before shape mismatches intentionally. This
    // is the compatibility contract for the shared core validation helpers.
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("left_index", left_index.len())?;
    ensure_nonempty_core("right_index", right_index.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_equal_lengths_core(
        "left_index",
        left_index.len(),
        "right_index",
        right_index.len(),
    )?;
    // ELI5: grow one lookup table only for labels actually observed. The
    // compact layout still avoids the second map used by the old reducer.
    // Intentionally start empty: right_index is a match stream and repeated
    // labels can make pair count much larger than distinct label count.
    // This saves memory for repeated labels, at the cost of rehashing when
    // nearly every match has a unique label.
    let mut products = HashMap::<i64, i64>::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let product = products.entry(*index_right).or_insert(1_i64);
        if booleans[left] {
            continue;
        }
        *product = product.wrapping_mul(to_i64(arr[left]));
    }

    let mut labels = Vec::with_capacity(products.len());
    let mut values = Vec::with_capacity(products.len());
    for (label, product) in products {
        labels.push(label);
        values.push(product);
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(values)))
}

/// `u64`-native counterpart to `prod_rev_no_range_int_core`.
///
/// ELI5: `uint64` values `>= 2**63` don't fit in `i64`; funneling them
/// through the shared `i64` accumulator wraps them to a negative number.
/// This keeps the accumulator and the returned products in `u64`
/// end-to-end so large unsigned products come back correct instead of
/// sign-flipped.
///
/// # Arguments
///
/// * `arr` - Left-side unsigned values to aggregate; must not be empty.
/// * `left_index` - Positions into `arr`; must be valid.
/// * `right_index` - Output label for each left position.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn prod_rev_no_range_u64_core(
    arr: ArrayView1<'_, u64>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> Result<(Array1<i64>, Array1<u64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("left_index", left_index.len())?;
    ensure_nonempty_core("right_index", right_index.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_equal_lengths_core(
        "left_index",
        left_index.len(),
        "right_index",
        right_index.len(),
    )?;
    // Use the same on-demand allocation policy for the u64 product path. It
    // saves memory for repeated labels but may rehash for unique labels.
    let mut products = HashMap::<i64, u64>::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let product = products.entry(*index_right).or_insert(1_u64);
        if booleans[left] {
            continue;
        }
        *product = product.wrapping_mul(arr[left]);
    }

    let mut labels = Vec::with_capacity(products.len());
    let mut values = Vec::with_capacity(products.len());
    for (label, product) in products {
        labels.push(label);
        values.push(product);
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(values)))
}

/// Multiply floating-point values grouped by right labels without range
/// metadata.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `left_index` - Positions into `arr`; must be valid.
/// * `right_index` - Output label for each left position.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn prod_rev_no_range_float_core<T: Copy, F: FnMut(T) -> f64>(
    arr: ArrayView1<'_, T>,
    left_index: ArrayView1<'_, i64>,
    right_index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Array1<i64>, Array1<f64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_nonempty_core("left_index", left_index.len())?;
    ensure_nonempty_core("right_index", right_index.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    ensure_equal_lengths_core(
        "left_index",
        left_index.len(),
        "right_index",
        right_index.len(),
    )?;
    // Use the same on-demand allocation policy for the float product path. It
    // saves memory for repeated labels but may rehash for unique labels.
    let mut products = HashMap::<i64, f64>::new();

    for (index_left, index_right) in left_index.iter().zip(right_index.iter()) {
        let left = checked_index(*index_left, arr.len())
            .ok_or("left_index must contain valid positions in arr")?;
        let product = products.entry(*index_right).or_insert(1.0_f64);
        if booleans[left] {
            continue;
        }
        *product *= to_f64(arr[left]);
    }

    let mut labels = Vec::with_capacity(products.len());
    let mut values = Vec::with_capacity(products.len());
    for (label, product) in products {
        labels.push(label);
        values.push(product);
    }
    Ok((Array1::from_vec(labels), Array1::from_vec(values)))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty) => {
        /// Groups products by right-side labels without range metadata.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `left_index` - Positions into `arr`; must be valid.
        /// * `right_index` - Output label for each left position.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            let booleans = booleans.as_array();
            let (indexers, result) =
                prod_rev_no_range_int_core(arr, left_index, right_index, booleans, |value| {
                    value as i64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
        }
    };
}

/// `uint64` export: returns `u64` products so values `>= 2**63` survive
/// the round trip to Python instead of wrapping to a negative `i64`.
#[pyfunction]
#[allow(clippy::type_complexity)]
/// Groups `uint64` products by right-side labels without range metadata.
///
/// # Arguments
///
/// * `arr` - Left-side unsigned values to aggregate; must not be empty.
/// * `left_index` - Positions into `arr`; must be valid.
/// * `right_index` - Output label for each left position.
/// * `booleans` - Null mask for `arr`; `True` rows are skipped.
pub fn compute_prod_rev_no_range_uint64<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<'py, u64>,
    left_index: PyReadonlyArray1<'py, i64>,
    right_index: PyReadonlyArray1<'py, i64>,
    booleans: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<u64>>)> {
    let arr = arr.as_array();
    let left_index = left_index.as_array();
    let right_index = right_index.as_array();
    let booleans = booleans.as_array();
    let (indexers, result) = prod_rev_no_range_u64_core(arr, left_index, right_index, booleans)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok((indexers.into_pyarray(py), result.into_pyarray(py)))
}

compute_ints!(compute_prod_rev_no_range_int64, i64);
compute_ints!(compute_prod_rev_no_range_int32, i32);
compute_ints!(compute_prod_rev_no_range_int16, i16);
compute_ints!(compute_prod_rev_no_range_int8, i8);
compute_ints!(compute_prod_rev_no_range_uint32, u32);
compute_ints!(compute_prod_rev_no_range_uint16, u16);
compute_ints!(compute_prod_rev_no_range_uint8, u8);

macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// Groups floating-point products by right-side labels without range
        /// metadata.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `left_index` - Positions into `arr`; must be valid.
        /// * `right_index` - Output label for each left position.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
            right_index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let left_index = left_index.as_array();
            let right_index = right_index.as_array();
            let booleans = booleans.as_array();
            let (indexers, result) =
                prod_rev_no_range_float_core(arr, left_index, right_index, booleans, |value| {
                    value as f64
                })
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
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
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn integer_core_returns_correct_label_product_pairs() {
        let got = prod_rev_no_range_int_core(
            array![5_i64, 2, 7].view(),
            array![0_i64, 1, 2, 1].view(),
            array![20_i64, 40, 20, 40].view(),
            array![false, false, false].view(),
            |value| value,
        );
        let (labels, products) = got.unwrap();
        let mut got: Vec<_> = labels
            .iter()
            .zip(products.iter())
            .map(|(&label, &product)| (label, product))
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![(20, 35), (40, 4)]);
    }

    #[test]
    fn empty_input_error_precedes_shape_mismatch() {
        let error = prod_rev_no_range_int_core(
            Array1::<i64>::zeros(0).view(),
            array![].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");
    }

    #[test]
    fn empty_input_error_precedes_shape_mismatch_for_u64_and_float_cores() {
        let error = prod_rev_no_range_u64_core(
            Array1::<u64>::zeros(0).view(),
            array![].view(),
            array![20_i64].view(),
            array![false].view(),
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");

        let error = prod_rev_no_range_float_core(
            Array1::<f64>::zeros(0).view(),
            array![].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .unwrap_err();
        assert_eq!(error, "arr cannot be empty");
    }

    #[test]
    fn u64_core_preserves_values_at_and_above_i64_max() {
        let value = (i64::MAX as u64) + 5;
        let got = prod_rev_no_range_u64_core(
            array![value].view(),
            array![0_i64].view(),
            array![20_i64].view(),
            array![false].view(),
        );
        assert_eq!(got, Ok((array![20], array![value])));
    }

    #[test]
    fn integer_core_supports_every_integer_dtype_and_wraps() {
        macro_rules! assert_product {
            ($type:ty) => {
                let got = prod_rev_no_range_int_core(
                    array![5 as $type, 2 as $type, 7 as $type].view(),
                    array![0_i64, 1, 2, 1].view(),
                    array![20_i64, 40, 20, 40].view(),
                    array![false, false, false].view(),
                    |value| value as i64,
                );
                let (labels, products) = got.unwrap();
                let mut got: Vec<_> = labels
                    .iter()
                    .zip(products.iter())
                    .map(|(&label, &product)| (label, product))
                    .collect();
                got.sort_unstable();
                assert_eq!(got, vec![(20, 35), (40, 4)]);
            };
        }

        assert_product!(i64);
        assert_product!(i32);
        assert_product!(i16);
        assert_product!(i8);
        assert_product!(u64);
        assert_product!(u32);
        assert_product!(u16);
        assert_product!(u8);

        let got = prod_rev_no_range_int_core(
            array![i64::MAX, 2_i64].view(),
            array![0_i64, 1].view(),
            array![20_i64, 20].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![20], array![i64::MAX.wrapping_mul(2)])));
    }

    #[test]
    fn float_core_promotes_f32_and_preserves_identity_and_infinity() {
        let got = prod_rev_no_range_float_core(
            array![1.5_f32, 2.0].view(),
            array![0_i64, 1].view(),
            array![20_i64, 20].view(),
            array![false, false].view(),
            |value| value as f64,
        );
        assert_eq!(got, Ok((array![20], array![3.0])));

        let got = prod_rev_no_range_float_core(
            array![f64::INFINITY, 2.0].view(),
            array![0_i64, 1].view(),
            array![20_i64, 20].view(),
            array![false, true].view(),
            |value| value,
        );
        assert!(got.unwrap().1[0].is_infinite());

        let got = prod_rev_no_range_float_core(
            array![1.5_f64].view(),
            array![0_i64].view(),
            array![40_i64].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((array![40], array![1.0])));
    }

    #[test]
    fn cores_reject_mismatches_and_invalid_positions() {
        assert!(prod_rev_no_range_int_core(
            array![1_i64].view(),
            array![0_i64].view(),
            array![20_i64, 40].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
        assert!(prod_rev_no_range_int_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
        assert!(prod_rev_no_range_float_core(
            array![1.0_f64].view(),
            array![i64::from(u32::MAX) + 1].view(),
            array![20_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
    }
}
