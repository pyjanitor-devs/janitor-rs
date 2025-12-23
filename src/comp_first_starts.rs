use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

fn binary_compare<T: std::cmp::PartialOrd>(left: &T, right: &T, op: i8) -> bool {
    match op {
        0 => left > right,
        1 => left >= right,
        2 => left < right,
        3 => left <= right,
        4 => left == right,
        _ => left != right,
    }
}

/// keep original - for debugging
// fn array_compare_int64(
//     left: ArrayView1<'_, i64>,
//     right: ArrayView1<'_, i64>,
//     starts: ArrayView1<'_, i64>,
//
//     op: i8,
// ) -> (Array1<i8>, Array1<i64>, i64) {
//     let mut result = Array1::<i8>::zeros(length as usize);
//     let mut counts_array = Array1::<i64>::zeros(left.len());
//     let mut total: i64 = 0;
//     let end: usize = right.len();
//     let mut n: usize = 0;
//     let zipped = left.into_iter().zip(starts.into_iter());
//     for (position, (left_val, end)) in zipped.enumerate() {
//         let start_ = *start as usize;
//         let mut counter: i64 = 0;
//         for nn in start_..end {
//             let right_val = right[nn];
//             let compare = binary_compare(left_val, &right_val, op);
//             counter += compare as i64;
//             total += compare as i64;
//             result[n] = compare as i8;
//             n += 1;
//         }
//         counts_array[position] = counter;
//     }
//     (result, counts_array, total)
// }

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        fn $fname(
            left: ArrayView1<'_, $type>,
            right: ArrayView1<'_, $type>,
            starts: ArrayView1<'_, i64>,
            op: i8,
        ) -> (Array1<i8>, Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let end: usize = right.len();
            let length: usize = starts.iter().map(|x| end - (*x as usize)).sum();
            let mut result = Array1::<i8>::zeros(length);
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let zipped = left.into_iter().zip(starts.into_iter());
            for (position, (left_val, start)) in zipped.enumerate() {
                let start_ = *start as usize;
                let mut counter: i64 = 0;
                for nn in start_..end {
                    let right_val = right[nn];
                    let compare = binary_compare(left_val, &right_val, op);
                    counter += compare as i64;
                    total += compare as i64;
                    result[n] = compare as i8;
                    n += 1;
                }
                counts_array[position] = counter;
            }
            (result, counts_array, total)
        }
    };
}

generic_compare!(array_compare_int64, i64);
generic_compare!(array_compare_int32, i32);
generic_compare!(array_compare_int16, i16);
generic_compare!(array_compare_int8, i8);
generic_compare!(array_compare_uint64, u64);
generic_compare!(array_compare_uint32, u32);
generic_compare!(array_compare_uint16, u16);
generic_compare!(array_compare_uint8, u8);
generic_compare!(array_compare_float64, f64);
generic_compare!(array_compare_float32, f32);

#[pyfunction(name = "compare_first_start_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_int64(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_int32(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_int16(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_int8(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_float32(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_float64(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_uint64(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_uint32(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_uint16(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_first_start_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,

    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let (result, counts_array, total) = array_compare_uint8(left, right, starts, op);
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_compare_greater_than() {
        assert!(binary_compare(&5, &3, 0));
        assert!(!binary_compare(&3, &5, 0));
        assert!(!binary_compare(&3, &3, 0));
    }

    #[test]
    fn test_binary_compare_greater_equal() {
        assert!(binary_compare(&5, &3, 1));
        assert!(!binary_compare(&3, &5, 1));
        assert!(binary_compare(&3, &3, 1));
    }

    #[test]
    fn test_binary_compare_less_than() {
        assert!(!binary_compare(&5, &3, 2));
        assert!(binary_compare(&3, &5, 2));
        assert!(!binary_compare(&3, &3, 2));
    }

    #[test]
    fn test_binary_compare_less_equal() {
        assert!(!binary_compare(&5, &3, 3));
        assert!(binary_compare(&3, &5, 3));
        assert!(binary_compare(&3, &3, 3));
    }

    #[test]
    fn test_binary_compare_equal() {
        assert!(!binary_compare(&5, &3, 4));
        assert!(binary_compare(&3, &3, 4));
    }

    #[test]
    fn test_binary_compare_not_equal() {
        assert!(binary_compare(&5, &3, 5));
        assert!(!binary_compare(&3, &3, 5));
    }

    #[test]
    fn test_array_compare_int64_basic() {
        let left = Array1::from(vec![5i64, 3, 7]);
        let right = Array1::from(vec![1i64, 2, 3, 4, 5]);
        let starts = Array1::from(vec![0i64, 1, 2]);

        let (result, counts, total) =
            array_compare_int64(left.view(), right.view(), starts.view(), 0);

        assert_eq!(result.len(), 12); // 5 + 4 + 3
        assert_eq!(counts.len(), 3);
        assert!(total >= 0);
    }

    #[test]
    fn test_array_compare_int32_equal() {
        let left = Array1::from(vec![3i32, 3]);
        let right = Array1::from(vec![1i32, 2, 3]);
        let starts = Array1::from(vec![0i64, 0]);

        let (result, counts, total) =
            array_compare_int32(left.view(), right.view(), starts.view(), 4);

        assert_eq!(counts[0], 1); // 3 == 3
        assert_eq!(counts[1], 1);
        assert_eq!(total, 2);
        assert_eq!(result.len(), 6); // 3 + 3
    }

    #[test]
    fn test_array_compare_float64() {
        let left = Array1::from(vec![3.5f64, 2.5]);
        let right = Array1::from(vec![1.0f64, 2.0, 3.0]);
        let starts = Array1::from(vec![0i64, 1]);

        let (result, counts, total) =
            array_compare_float64(left.view(), right.view(), starts.view(), 0);

        assert_eq!(counts.len(), 2);
        assert_eq!(result.len(), 5); // 3 + 2
        assert_eq!(total, 4);
    }

    #[test]
    fn test_array_compare_uint8() {
        let left = Array1::from(vec![5u8, 10]);
        let right = Array1::from(vec![3u8, 7, 12]);
        let starts = Array1::from(vec![0i64, 0]);

        let (result, counts, total) =
            array_compare_uint8(left.view(), right.view(), starts.view(), 2);

        assert_eq!(result.len(), 6);
        assert_eq!(counts.len(), 2);
        assert_eq!(total, 3);
    }

    #[test]
    fn test_array_compare_empty_starts() {
        let left = Array1::from(vec![1i64]);
        let right = Array1::from(vec![1i64, 2, 3]);
        let starts = Array1::from(vec![3i64]); // Start at end

        let (result, counts, total) =
            array_compare_int64(left.view(), right.view(), starts.view(), 0);

        assert_eq!(result.len(), 0);
        assert_eq!(counts[0], 0);
        assert_eq!(total, 0);
    }
}
