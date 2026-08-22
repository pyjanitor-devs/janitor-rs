use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

// ELI5: `$type` below only picks the dtype of the *input* array (`arr`) --
// the result is always `i64` because this function returns the *position*
// of the min element (`base = indexer`), not its value, and positions are
// always `i64` regardless of what dtype the values themselves are. That's
// unrelated to promotion (contrast with `sum`/`prod`, which really do widen
// an accumulated value); `$type` here must still match the numpy dtype the
// function's name promises, or pyo3 rejects the array at the Python
// boundary. `compute_min_positions_int8` once had `i64` here (a leftover
// copy-paste from a wider sibling) even though its name promises `i8`
// input -- see issue #30.
macro_rules! generic_compute {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            arr: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            booleans: PyReadonlyArray1<'py, bool>,
        ) -> Bound<'py, PyArray1<i64>>
        // The macro will expand into the contents of this block.
        {
            let arr = arr.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let positions = positions.as_array();
            let booleans = booleans.as_array();
            let mut result = Array1::<i64>::zeros(starts.len());
            let zipped = izip!(starts.into_iter(), ends.into_iter());
            for (pos, (start, end)) in zipped.enumerate() {
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut base: i64 = -1;
                let mut base_val = arr[0];
                for nn in start_..end_ {
                    let indexer = positions[nn];
                    if indexer == -1 {
                        continue;
                    }
                    let indexer_: usize = indexer as usize;
                    if booleans[indexer_] {
                        continue;
                    }
                    let current = arr[indexer_];
                    if (base == -1) || (current < base_val) {
                        base_val = current;
                        base = indexer;
                    }
                }
                result[pos] = base;
            }
            result.into_pyarray(py)
        }
    };
}

generic_compute!(compute_min_positions_int64, i64);
generic_compute!(compute_min_positions_int32, i32);
generic_compute!(compute_min_positions_int16, i16);
generic_compute!(compute_min_positions_int8, i8); // fixed: was `i64`, see issue #30
generic_compute!(compute_min_positions_uint64, u64);
generic_compute!(compute_min_positions_uint32, u32);
generic_compute!(compute_min_positions_uint16, u16);
generic_compute!(compute_min_positions_uint8, u8);
generic_compute!(compute_min_positions_f64, f64);
generic_compute!(compute_min_positions_f32, f32);

#[cfg(test)]
mod tests {
    use super::*;

    type Int8PositionsFn = for<'py> fn(
        Python<'py>,
        PyReadonlyArray1<'py, i8>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, i64>,
        PyReadonlyArray1<'py, bool>,
    ) -> Bound<'py, PyArray1<i64>>;

    #[test]
    fn int8_wrapper_accepts_an_int8_array() {
        // ELI5: the typed slot only accepts a wrapper whose `arr` is really
        // `i8`; changing the macro argument back to `i64` breaks compilation.
        let _wrapper: Int8PositionsFn = compute_min_positions_int8;
    }
}
