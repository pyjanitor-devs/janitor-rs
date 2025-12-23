/// compare rows where only ends exist - for !=
/// and matches exist
use itertools::izip;
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

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        fn $fname(
            left: ArrayView1<'_, $type>,
            right: ArrayView1<'_, $type>,
            ends: ArrayView1<'_, i64>,
            left_booleans: ArrayView1<'_, bool>,
            right_booleans: ArrayView1<'_, bool>,
            matches: ArrayView1<'_, i8>,
            is_extension_array: bool,
            op: i8,
        ) -> (Array1<i8>, Array1<i64>, i64)
        // The macro will expand into the contents of this block.
        {
            let mut result = Array1::<i8>::zeros(matches.len());
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let start_: usize = 0;
            let zipped = izip!(
                left.into_iter(),
                left_booleans.into_iter(),
                ends.into_iter(),
            );
            for (position, (left_val, left_bool, end)) in zipped.enumerate() {
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    if matches[n] == 0 {
                        n += 1;
                        continue;
                    }
                    let right_bool_ = right_booleans[nn];
                    //pd.NA != pd.NA returns pd.NA, which defaults to False
                    // pd.NA != anything returns pd.NA, which defaults to False
                    // whereas np.nan != np.nan returns True
                    // np.nan != anything returns True
                    if (*left_bool || right_bool_) && is_extension_array {
                        n += 1;
                        continue;
                    }
                    if (*left_bool || right_bool_) && !is_extension_array {
                        result[n] = 1;
                        counter += 1;
                        total += 1;
                        n += 1;
                        continue;
                    }

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

#[pyfunction(name = "compare_end_ne_int64")]
pub fn compare_int64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i64>,
    right: PyReadonlyArray1<'py, i64>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int64(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_int32")]
pub fn compare_int32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i32>,
    right: PyReadonlyArray1<'py, i32>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int32(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_int16")]
pub fn compare_int16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i16>,
    right: PyReadonlyArray1<'py, i16>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int16(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_int8")]
pub fn compare_int8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, i8>,
    right: PyReadonlyArray1<'py, i8>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_int8(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_float32")]
pub fn compare_float32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f32>,
    right: PyReadonlyArray1<'py, f32>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_float32(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_float64")]
pub fn compare_float64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, f64>,
    right: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_float64(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_uint64")]
pub fn compare_uint64<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u64>,
    right: PyReadonlyArray1<'py, u64>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint64(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_uint32")]
pub fn compare_uint32<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u32>,
    right: PyReadonlyArray1<'py, u32>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint32(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_uint16")]
pub fn compare_uint16<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u16>,
    right: PyReadonlyArray1<'py, u16>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint16(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[pyfunction(name = "compare_end_ne_uint8")]
pub fn compare_uint8<'py>(
    py: Python<'py>,
    left: PyReadonlyArray1<'py, u8>,
    right: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,

    left_booleans: PyReadonlyArray1<'py, bool>,
    right_booleans: PyReadonlyArray1<'py, bool>,

    matches: PyReadonlyArray1<'py, i8>,
    is_extension_array: bool,
    op: i8,
) -> (Bound<'py, PyArray1<i8>>, Bound<'py, PyArray1<i64>>, i64) {
    let left = left.as_array();
    let right = right.as_array();
    let starts = starts.as_array();

    let matches = matches.as_array();
    let left_booleans = left_booleans.as_array();
    let right_booleans = right_booleans.as_array();
    let (result, counts_array, total) = array_compare_uint8(
        left,
        right,
        starts,
        left_booleans,
        right_booleans,
        matches,
        is_extension_array,
        op,
    );
    (
        result.into_pyarray(py),
        counts_array.into_pyarray(py),
        total,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::Array1;

    #[test]
    fn test_array_compare_int64_ne_ends() {
        let left = Array1::from_vec(vec![10, 20]);
        let right = Array1::from_vec(vec![5, 10, 25, 20]);
        let ends = Array1::from_vec(vec![2, 4]);
        let left_booleans = Array1::from_vec(vec![false, false]);
        let right_booleans = Array1::from_vec(vec![false, false, false, false]);
        let matches = Array1::from_vec(vec![1, 1, 1, 1, 1, 1]);

        let (result, counts_array, total) = array_compare_int64(
            left.view(),
            right.view(),
            ends.view(),
            left_booleans.view(),
            right_booleans.view(),
            matches.view(),
            false,
            5, // not equal
        );

        assert_eq!(result.len(), 6);
        assert_eq!(counts_array.len(), 2);
        assert_eq!(total, 4);
    }
}
