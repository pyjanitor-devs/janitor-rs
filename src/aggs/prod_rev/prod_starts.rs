use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::aggs::{ensure_equal_lengths_core, ensure_nonempty_core, starts_domain, WrapMul};

/// Multiply values into one state slot per right-row ordinal.
///
/// `A` is the accumulator type: every integer dtype instantiates this with
/// `A = i64`, except `uint64`, which instantiates it with `A = u64` so
/// values `>= 2**63` don't get sign-flipped by a forced `i64` cast (see
/// `WrapMul`).
///
/// `starts` describes suffix ranges `[start, index.len())`. Invalid starts
/// and empty `starts`/`index` inputs are rejected by `starts_domain`.
/// Null rows leave each touched slot at the multiplicative identity, `1`.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive ordinal start of each suffix range.
/// * `index` - Right-side labels in ordinal position order.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
#[allow(private_bounds)]
pub fn prod_rev_starts_int_core<T: Copy, A: WrapMul, F: FnMut(T) -> A>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut convert: F,
) -> Result<(Vec<i64>, Vec<A>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;
    // ELI5: a suffix row joins the running product at its start bucket.
    // Combine rows sharing a start into one product event, then sweep
    // left-to-right so each event applies to that position and every later
    // position. Wrapping multiplication is associative, so this preserves
    // the integer result while avoiding a full suffix walk per row.
    let mut values = vec![A::ONE; width];
    for (current, start, boolean) in izip!(arr, starts, booleans) {
        if *boolean || *start == index.len() as i64 {
            continue;
        }
        let slot = *start as usize - min_start;
        values[slot] = values[slot].wrap_mul(convert(*current));
    }
    let mut running = A::ONE;
    for value in &mut values {
        running = running.wrap_mul(*value);
        *value = running;
    }
    let labels = index.iter().skip(min_start).copied().collect();
    Ok((labels, values))
}

/// Multiply floating-point values into each compact suffix slot.
///
/// # Arguments
///
/// * `arr` - Left-side values to aggregate; must not be empty.
/// * `starts` - Inclusive ordinal start of each suffix range.
/// * `index` - Right-side labels in ordinal position order.
/// * `booleans` - Null mask for `arr`; `true` rows are skipped.
pub fn prod_rev_starts_float_core<T: Copy, F: FnMut(T) -> f64>(
    arr: ArrayView1<'_, T>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    mut to_f64: F,
) -> Result<(Vec<i64>, Vec<f64>), String> {
    ensure_nonempty_core("arr", arr.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "starts", starts.len())?;
    ensure_equal_lengths_core("arr", arr.len(), "booleans", booleans.len())?;
    let (min_start, width) = starts_domain(starts, index.len())?;
    // Keep floats on the direct nested loop. Unlike integer wrapping
    // multiplication, IEEE-754 multiplication is not safely regroupable:
    // changing the order can change rounding and the propagation of NaN or
    // infinity. ELI5: integer products are exact on a wrapping number wheel,
    // but decimal-like floats can give a different answer when we rearrange
    // the multiplication, so this path preserves the old row/position order.
    let mut values = vec![1_f64; width];
    for (current, start, boolean) in izip!(arr, starts, booleans) {
        if *boolean {
            continue;
        }
        let current = to_f64(*current);
        // Example: with `min_start = 2` and `start = 5`, `skip(3)` maps the
        // compact slot 3 back to original position 5 and multiplies this row
        // through the rest of its suffix.
        for value in values.iter_mut().skip(*start as usize - min_start) {
            *value *= current;
        }
    }
    let labels = index.iter().skip(min_start).copied().collect();
    Ok((labels, values))
}

macro_rules! compute_ints {
    ($fname:ident, $type:ty, $acc:ty) => {
        /// Finds the product for each right-side label covered by the reverse
        /// suffix ranges.
        ///
        /// Null rows do not participate; untouched products remain `1`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive ordinal start of each suffix range.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<$acc>>)> {
            let (labels, values) = prod_rev_starts_int_core(
                arr.as_array(),
                starts.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as $acc,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(labels).into_pyarray(py),
                Array1::from_vec(values).into_pyarray(py),
            ))
        }
    };
}
macro_rules! compute_floats {
    ($fname:ident, $type:ty) => {
        /// Finds the product for each right-side label covered by the reverse
        /// suffix ranges.
        ///
        /// Null rows do not participate; untouched products remain `1.0`.
        ///
        /// # Arguments
        ///
        /// * `arr` - Left-side values to aggregate; must not be empty.
        /// * `starts` - Inclusive ordinal start of each suffix range.
        /// * `index` - Right-side labels in ordinal position order.
        /// * `booleans` - Null mask for `arr`; `True` rows are skipped.
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            index: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
            let (labels, values) = prod_rev_starts_float_core(
                arr.as_array(),
                starts.as_array(),
                index.as_array(),
                booleans.as_array(),
                |value| value as f64,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            Ok((
                Array1::from_vec(labels).into_pyarray(py),
                Array1::from_vec(values).into_pyarray(py),
            ))
        }
    };
}

// `uint64` is the one dtype whose accumulator is `u64` instead of `i64` --
// see `WrapMul`'s doc comment. Every other dtype fits inside `i64` losslessly.
compute_ints!(compute_prod_rev_start_int64, i64, i64);
compute_ints!(compute_prod_rev_start_int32, i32, i64);
compute_ints!(compute_prod_rev_start_int16, i16, i64);
compute_ints!(compute_prod_rev_start_int8, i8, i64);
compute_ints!(compute_prod_rev_start_uint64, u64, u64);
compute_ints!(compute_prod_rev_start_uint32, u32, i64);
compute_ints!(compute_prod_rev_start_uint16, u16, i64);
compute_ints!(compute_prod_rev_start_uint8, u8, i64);
compute_floats!(compute_prod_rev_start_f64, f64);
compute_floats!(compute_prod_rev_start_f32, f32);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compute_prod_rev_start_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn integer_core_groups_suffixes_and_preserves_identity() {
        let arr = array![2_i64, 3];
        let starts = array![0_i64, 1];
        let index = array![20_i64, 10, 90];
        let booleans = array![false, false];
        let got = prod_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got, Ok((vec![20, 10, 90], vec![2, 6, 6])));
    }

    #[test]
    fn integer_core_wraps_and_null_rows_leave_identity() {
        let got = prod_rev_starts_int_core(
            array![i64::MAX, 2].view(),
            array![0_i64, 0].view(),
            array![10_i64].view(),
            array![false, false].view(),
            |value| value,
        );
        assert_eq!(got, Ok((vec![10], vec![-2])));

        let got = prod_rev_starts_int_core(
            array![2_i64].view(),
            array![0_i64].view(),
            array![10_i64].view(),
            array![true].view(),
            |value| value,
        );
        assert_eq!(got, Ok((vec![10], vec![1])));
    }

    #[test]
    fn integer_sweep_preserves_products_and_zero_width_rows() {
        let arr = Array1::from_elem(100, 2_i64);
        let mut starts = Array1::from_elem(100, 0_i64);
        starts[99] = 1000;
        let index = Array1::from_iter(0_i64..1000);
        let booleans = Array1::from_elem(100, false);
        let got = prod_rev_starts_int_core(
            arr.view(),
            starts.view(),
            index.view(),
            booleans.view(),
            |value| value,
        );
        assert_eq!(got.unwrap().1, vec![0_i64; 1000]);
    }

    #[test]
    fn uint64_core_preserves_values_above_i64_max() {
        let value = (i64::MAX as u64) + 1;
        let got = prod_rev_starts_int_core(
            array![value].view(),
            array![0_i64].view(),
            array![10_i64].view(),
            array![false].view(),
            |current| current,
        );
        assert_eq!(got, Ok((vec![10], vec![value])));
    }

    #[test]
    fn float32_core_promotes_output_and_rejects_invalid_bounds() {
        let got = prod_rev_starts_float_core(
            array![2.5_f32, 4.0].view(),
            array![0_i64, 1].view(),
            array![20_i64, 10].view(),
            array![false, false].view(),
            |value| value as f64,
        );
        assert_eq!(got, Ok((vec![20, 10], vec![2.5, 10.0])));

        assert!(prod_rev_starts_int_core(
            array![1_i64].view(),
            array![-1_i64].view(),
            array![10_i64].view(),
            array![false].view(),
            |value| value,
        )
        .is_err());
    }
}
