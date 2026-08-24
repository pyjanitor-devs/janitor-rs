use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

macro_rules! bin_search {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            left_index: PyReadonlyArray1<'py, i64>,
        ) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64) {
            let left = left.as_array();
            let right = right.as_array();
            let left_index = left_index.as_array();
            let right_slice = right.as_slice();
            let mut result = Array1::<i64>::zeros(left.len());
            let mut total: usize = left.len();
            for (pos, left_value) in left.into_iter().enumerate() {
                let min_idx = if let Some(slice) = right_slice {
                    slice.partition_point(|v| *v < *left_value)
                } else {
                    let mut min_idx = 0;
                    let mut max_idx = right.len();
                    while min_idx < max_idx {
                        // to avoid overflow
                        // adapted from numba's implementation
                        let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
                        let current_value = right[mid_idx as usize];
                        if current_value >= *left_value {
                            max_idx = mid_idx;
                        } else {
                            min_idx = mid_idx + 1;
                        }
                    }
                    min_idx
                };
                if min_idx == 0 {
                    total -= 1;
                    continue;
                }
                let mid_idx = min_idx - 1;
                let current_value = right[mid_idx as usize];
                if current_value == *left_value {
                    result[pos as usize] = 0 as i64;
                    total -= 1;
                    continue;
                }
                result[pos as usize] = min_idx as i64;
            }
            let mut index_left = Array1::<i64>::zeros(total as usize);
            let mut search_indices = Array1::<i64>::zeros(total as usize);
            let len_right = right.len() as i64;
            let mut n = 0;
            for (pos, item) in result.into_iter().enumerate() {
                if item == 0 {
                    continue;
                }
                search_indices[n] = len_right - item;
                let ind = left_index[pos];
                index_left[n] = ind;
                n += 1;
            }
            (
                search_indices.into_pyarray(py),
                index_left.into_pyarray(py),
                total as i64,
            )
        }
    };
}

bin_search!(binary_search_gt_regions_int64, i64);
bin_search!(binary_search_gt_regions_int32, i32);
bin_search!(binary_search_gt_regions_int16, i16);
bin_search!(binary_search_gt_regions_int8, i8);
bin_search!(binary_search_gt_regions_uint64, u64);
bin_search!(binary_search_gt_regions_uint32, u32);
bin_search!(binary_search_gt_regions_uint16, u16);
bin_search!(binary_search_gt_regions_uint8, u8);
bin_search!(binary_search_gt_regions_f64, f64);
bin_search!(binary_search_gt_regions_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_int64, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_int32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_int16, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_int8, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_f32, m)?)?;
    m.add_function(wrap_pyfunction!(binary_search_gt_regions_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArrayMethods, PyReadonlyArray1};

    /// Builds a non-contiguous `PyReadonlyArray1<i64>` by slicing a
    /// double-width Python-side array with a step of 2, so the wrapper's
    /// fallback branch (`.as_slice()` returns `None`) runs instead of the
    /// fast path -- proving the two branches agree without needing to
    /// hand-derive an expected value independently of either.
    fn strided_i64<'py>(py: Python<'py>, values: &[i64]) -> PyReadonlyArray1<'py, i64> {
        let padded: Vec<i64> = values.iter().flat_map(|v| [*v, -1]).collect();
        let numpy = py.import("numpy").unwrap();
        let full = numpy.call_method1("array", (padded,)).unwrap();
        let strided = full
            .get_item(pyo3::types::PySlice::new(
                py,
                0,
                (values.len() * 2) as isize,
                2,
            ))
            .unwrap();
        strided.extract().unwrap()
    }

    #[test]
    fn contiguous_and_strided_paths_agree() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let right_values = [10_i64, 20, 20, 30];
            let right_dense = PyArray1::from_vec(py, right_values.to_vec());
            assert!(right_dense.readonly().as_array().as_slice().is_some());
            let right_strided = strided_i64(py, &right_values);
            assert!(right_strided.as_array().as_slice().is_none());

            let left = PyArray1::from_vec(py, vec![5_i64, 20, 35]);
            let left_index = PyArray1::from_vec(py, vec![100_i64, 200, 300]);

            let (fast_search, fast_left, fast_total) = binary_search_gt_regions_int64(
                py,
                left.readonly(),
                right_dense.readonly(),
                left_index.readonly(),
            );
            let (fallback_search, fallback_left, fallback_total) = binary_search_gt_regions_int64(
                py,
                left.readonly(),
                right_strided,
                left_index.readonly(),
            );
            assert_eq!(
                fast_search.readonly().as_array().to_vec(),
                fallback_search.readonly().as_array().to_vec()
            );
            assert_eq!(
                fast_left.readonly().as_array().to_vec(),
                fallback_left.readonly().as_array().to_vec()
            );
            assert_eq!(fast_total, fallback_total);
        });
    }
}
