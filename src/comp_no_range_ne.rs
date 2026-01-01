use itertools::izip;
use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn binary_compare<T: std::cmp::PartialOrd>(left: &T, right: &T, op: i64) -> bool {
    match op {
        0 => left > right,
        1 => left >= right,
        2 => left < right,
        3 => left <= right,
        4 => left == right,
        _ => left != right,
    }
}

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        fn $fname(
            left: ArrayView1<'_, $type>,
            right: ArrayView1<'_, $type>,
            positions: ArrayView1<'_, i64>,
            left_booleans: ArrayView1<'_, bool>,
            right_booleans: ArrayView1<'_, bool>,
            is_extension_array: bool,
            op: i64,
        ) -> (Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = izip!(
                left.into_iter(),
                left_booleans.into_iter(),
                positions.into_iter(),
            );
            for (left_val, left_bool, right_pos) in zipped {
                //pd.NA != pd.NA returns pd.NA, which defaults to False
                // pd.NA != anything returns pd.NA, which defaults to False
                // whereas np.nan != np.nan returns True
                // np.nan != anything returns True
                if (*right_pos == -1) || (*left_bool && is_extension_array) {
                    result[n] = -1;
                    n += 1;
                    continue;
                }
                if *left_bool && !is_extension_array {
                    result[n] = *right_pos;
                    n += 1;
                    total += 1;
                    continue;
                }
                let right_bool = right_booleans[*right_pos as usize];
                if right_bool && is_extension_array {
                    result[n] = -1;
                    n += 1;
                    continue;
                }
                if right_bool && !is_extension_array {
                    result[n] = *right_pos;
                    n += 1;
                    total += 1;
                    continue;
                }
                let right_val = right[*right_pos as usize];
                let compare = binary_compare(left_val, &right_val, op);
                total += compare as i64;
                result[n] = if compare { *right_pos } else { -1 };
                n += 1;
            }

            (result, total)
        }
    };
}

generic_compare!(array_compare_int64, i64);
generic_compare!(array_compare_int32, i32);
generic_compare!(array_compare_int16, i16);
generic_compare!(array_compare_int8, i64);
generic_compare!(array_compare_uint64, u64);
generic_compare!(array_compare_uint32, u32);
generic_compare!(array_compare_uint16, u16);
generic_compare!(array_compare_uint8, u8);
generic_compare!(array_compare_float64, f64);
generic_compare!(array_compare_float32, f32);

#[pyfunction(name = "compare_no_range_ne_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_int64(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_int32(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_int16(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_int8(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_float32(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_float64(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_uint64(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_uint32(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_uint16(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_ne_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    positions: PyReadonlyArray1<'py, i64>,
    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,
    is_extension_array: bool,
    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();

    let (result, total) = array_compare_uint8(
        left,
        right,
        positions,
        left_booleans,
        right_booleans,
        is_extension_array,
        op,
    );
    (result.into_pyarray(py), total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    fn test_binary_compare_greater_than() {
        assert!(binary_compare(&5i64, &3i64, 0));
        assert!(!binary_compare(&3i64, &5i64, 0));
        assert!(!binary_compare(&5i64, &5i64, 0));
    }

    #[test]
    fn test_binary_compare_greater_equal() {
        assert!(binary_compare(&5i64, &3i64, 1));
        assert!(!binary_compare(&3i64, &5i64, 1));
        assert!(binary_compare(&5i64, &5i64, 1));
    }

    #[test]
    fn test_binary_compare_less_than() {
        assert!(binary_compare(&3i64, &5i64, 2));
        assert!(!binary_compare(&5i64, &3i64, 2));
        assert!(!binary_compare(&5i64, &5i64, 2));
    }

    #[test]
    fn test_binary_compare_less_equal() {
        assert!(binary_compare(&3i64, &5i64, 3));
        assert!(!binary_compare(&5i64, &3i64, 3));
        assert!(binary_compare(&5i64, &5i64, 3));
    }

    #[test]
    fn test_binary_compare_equal() {
        assert!(binary_compare(&5i64, &5i64, 4));
        assert!(!binary_compare(&5i64, &3i64, 4));
    }

    #[test]
    fn test_binary_compare_not_equal() {
        assert!(binary_compare(&5i64, &3i64, 5));
        assert!(!binary_compare(&5i64, &5i64, 5));
    }

    #[test]
    fn test_array_compare_int64_equal() {
        let left = array![1i64, 2i64, 3i64];
        let right = array![1i64, 2i64, 3i64];
        let positions = array![0i64, 1i64, 2i64];
        let left_booleans = array![false, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_int64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            4, // equal operator
        );

        assert_eq!(result[0], 0i64);
        assert_eq!(result[1], 1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_array_compare_int64_not_equal() {
        let left = array![1i64, 2i64, 3i64];
        let right = array![1i64, 2i64, 4i64];
        let positions = array![0i64, 1i64, 2i64];
        let left_booleans = array![false, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_int64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            5, // not equal operator
        );

        assert_eq!(result[0], -1i64);
        assert_eq!(result[1], -1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_array_compare_int64_greater_than() {
        let left = array![1i64, 2i64, 3i64];
        let right = array![2i64, 2i64, 2i64];
        let positions = array![0i64, 1i64, 2i64];
        let left_booleans = array![false, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_int64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            0, // greater than operator
        );

        assert_eq!(result[0], -1i64);
        assert_eq!(result[1], -1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_array_compare_with_null_position() {
        let left = array![1i64, 2i64, 3i64];
        let right = array![1i64, 2i64, 3i64];
        let positions = array![-1i64, 1i64, 2i64];
        let left_booleans = array![false, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_int64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            4, // equal operator
        );

        assert_eq!(result[0], -1i64);
        assert_eq!(result[1], 1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_with_left_boolean() {
        let left = array![1i64, 2i64, 3i64];
        let right = array![1i64, 2i64, 3i64];
        let positions = array![0i64, 1i64, 2i64];
        let left_booleans = array![true, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_int64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            4, // equal operator
        );

        assert_eq!(result[0], 0i64);
        assert_eq!(result[1], 1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_array_compare_with_extension_array() {
        let left = array![1i64, 2i64, 3i64];
        let right = array![1i64, 2i64, 3i64];
        let positions = array![0i64, 1i64, 2i64];
        let left_booleans = array![true, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_int64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            true, // is_extension_array = true
            4,    // equal operator
        );

        assert_eq!(result[0], -1i64);
        assert_eq!(result[1], 1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_float64() {
        let left = array![1.5f64, 2.5f64, 3.5f64];
        let right = array![1.5f64, 2.5f64, 3.5f64];
        let positions = array![0i64, 1i64, 2i64];
        let left_booleans = array![false, false, false];
        let right_booleans = array![false, false, false];

        let (result, total) = array_compare_float64(
            left.view(),
            right.view(),
            positions.view(),
            left_booleans.view(),
            right_booleans.view(),
            false,
            4, // equal operator
        );

        assert_eq!(result[0], 0i64);
        assert_eq!(result[1], 1i64);
        assert_eq!(result[2], 2i64);
        assert_eq!(total, 3);
    }
}

