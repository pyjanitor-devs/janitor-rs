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
            op: i64,
        ) -> (Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(positions.into_iter());
            for (left_val, right_pos) in zipped {
                if *right_pos == -1 {
                    result[n] = -1;
                    n += 1;
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

#[pyfunction(name = "compare_no_range_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_int64(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_int32(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_int16(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_int8(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_float32(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_float64(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_uint64(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_uint32(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_uint16(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[pyfunction(name = "compare_no_range_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    positions: PyReadonlyArray1<'py, i64>,

    op: i64,
) -> (Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let positions = positions.as_array();

    let (result, total) = array_compare_uint8(left, right, positions, op);
    (result.into_pyarray(py), total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_compare_greater_than() {
        assert!(binary_compare(&5, &3, 0));
        assert!(!binary_compare(&3, &5, 0));
        assert!(!binary_compare(&5, &5, 0));
    }

    #[test]
    fn test_binary_compare_greater_equal() {
        assert!(binary_compare(&5, &3, 1));
        assert!(!binary_compare(&3, &5, 1));
        assert!(binary_compare(&5, &5, 1));
    }

    #[test]
    fn test_binary_compare_less_than() {
        assert!(!binary_compare(&5, &3, 2));
        assert!(binary_compare(&3, &5, 2));
        assert!(!binary_compare(&5, &5, 2));
    }

    #[test]
    fn test_binary_compare_less_equal() {
        assert!(!binary_compare(&5, &3, 3));
        assert!(binary_compare(&3, &5, 3));
        assert!(binary_compare(&5, &5, 3));
    }

    #[test]
    fn test_binary_compare_equal() {
        assert!(!binary_compare(&5, &3, 4));
        assert!(!binary_compare(&3, &5, 4));
        assert!(binary_compare(&5, &5, 4));
    }

    #[test]
    fn test_binary_compare_not_equal() {
        assert!(binary_compare(&5, &3, 5));
        assert!(binary_compare(&3, &5, 5));
        assert!(!binary_compare(&5, &5, 5));
    }

    #[test]
    fn test_array_compare_int64_greater() {
        let left = Array1::from(vec![5i64, 3i64, 7i64]);
        let right = Array1::from(vec![2i64, 4i64, 6i64]);
        let positions = Array1::from(vec![0i64, 1i64, 2i64]);

        let (result, total) = array_compare_int64(left.view(), right.view(), positions.view(), 0);
        assert_eq!(result.to_vec(), vec![0i64, -1i64, 2i64]);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_int64_with_invalid_position() {
        let left = Array1::from(vec![5i64, 3i64, 7i64]);
        let right = Array1::from(vec![2i64, 4i64, 6i64]);
        let positions = Array1::from(vec![0i64, -1i64, 2i64]);

        let (result, total) = array_compare_int64(left.view(), right.view(), positions.view(), 0);
        assert_eq!(result.to_vec(), vec![0i64, -1i64, 2i64]);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_float64_equal() {
        let left = Array1::from(vec![1.5f64, 2.5f64, 3.5f64]);
        let right = Array1::from(vec![1.5f64, 2.0f64, 3.5f64]);
        let positions = Array1::from(vec![0i64, 1i64, 2i64]);

        let (result, total) = array_compare_float64(left.view(), right.view(), positions.view(), 4);
        assert_eq!(result.to_vec(), vec![0i64, -1i64, 2i64]);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_uint32_less_equal() {
        let left = Array1::from(vec![1u32, 2u32, 3u32]);
        let right = Array1::from(vec![5u32, 2u32, 1u32]);
        let positions = Array1::from(vec![0i64, 1i64, 2i64]);

        let (result, total) = array_compare_uint32(left.view(), right.view(), positions.view(), 3);
        assert_eq!(result.to_vec(), vec![0i64, 1i64, -1i64]);
        assert_eq!(total, 2);
    }

    #[test]
    fn test_array_compare_int32_not_equal() {
        let left = Array1::from(vec![1i32, 2i32, 3i32]);
        let right = Array1::from(vec![1i32, 2i32, 3i32]);
        let positions = Array1::from(vec![0i64, 1i64, 2i64]);

        let (result, total) = array_compare_int32(left.view(), right.view(), positions.view(), 5);
        assert_eq!(result.to_vec(), vec![-1i64, -1i64, -1i64]);
        assert_eq!(total, 0);
    }
}

