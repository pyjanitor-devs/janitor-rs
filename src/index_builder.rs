use numpy::ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;


fn build_left_index_single(
    left_index: ArrayView1<'_, i64>,
    search_indices: ArrayView1<'_, i64>,
    length: i64,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(length as usize);
    let mut n: usize = 0;
    let mut val: i64;
    for (i, number) in search_indices.indexed_iter() {
        val = left_index[i];
        let num: usize = *number as usize;
        for _ in 0..num {
            result[n] = val;
            n += 1;
        }
    }
    result
}

#[pyfunction(name = "build_left_index_single")]
pub fn left_index_single<'py>(
    py: Python<'py>,
    left_index: PyReadonlyArray1<'py, i64>,
    search_indices: PyReadonlyArray1<'py, i64>,
    length: i64,
) -> Bound<'py, PyArray1<i64>> {
    let left_index = left_index.as_array();
    let search_indices = search_indices.as_array();
    let result = build_left_index_single(left_index, search_indices, length);
    result.into_pyarray(py)
}
