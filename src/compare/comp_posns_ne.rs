/// comparison on !=, based on positions
/// handles comparisions where nulls exist
use itertools::izip;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::op::CompareOp;

macro_rules! generic_compare {
    ($fname:ident, $type:ty) => {
        #[pyfunction]
        pub fn $fname<'py>(
            py: Python<'py>,
            left: PyReadonlyArray1<'py, $type>,
            right: PyReadonlyArray1<'py, $type>,
            starts: PyReadonlyArray1<'py, i64>,
            ends: PyReadonlyArray1<'py, i64>,
            positions: PyReadonlyArray1<'py, i64>,
            left_booleans: PyReadonlyArray1<'py, bool>,
            right_booleans: PyReadonlyArray1<'py, bool>,
            is_extension_array: bool,
            op: i8,
        ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64)> {
            let left = left.as_array();
            let right = right.as_array();
            let starts = starts.as_array();
            let ends = ends.as_array();
            let positions = positions.as_array();
            let left_booleans = left_booleans.as_array();
            let right_booleans = right_booleans.as_array();
            let op = CompareOp::try_from_code(op)?;
            let mut result = Array1::<i64>::zeros(positions.len());
            let mut counts_array = Array1::<i64>::zeros(left.len());
            let mut total: i64 = 0;
            let mut n: usize = 0;
            let positions_len = positions.len();
            let right_len = right.len();
            let zipped = izip!(
                left.into_iter(),
                left_booleans.into_iter(),
                starts.into_iter(),
                ends.into_iter(),
            );
            for (position, (left_val, left_bool, start, end)) in zipped.enumerate() {
                // See comp_posns.rs: result is presized to positions.len(),
                // so an invalid row (indexing into positions, not right)
                // is silently skipped rather than rejected.
                if *start < 0 || *start >= *end || (*end as usize) > positions_len {
                    continue;
                }
                let start_ = *start as usize;
                let end_ = *end as usize;
                let mut counter: i64 = 0;
                for nn in start_..end_ {
                    let mut indexer = positions[nn];
                    //pd.NA != pd.NA returns pd.NA, which defaults to False
                    // pd.NA != anything returns pd.NA, which defaults to False
                    // whereas np.nan != np.nan returns True
                    // np.nan != anything returns True
                    //
                    // Broadened (issue #56) from `indexer == -1` to any
                    // out-of-bounds indexer, same reasoning as
                    // comp_posns.rs -- a `positions` entry that is some
                    // other negative or oversized value used to survive
                    // into `right_booleans[indexer as usize]`/
                    // `right[indexer as usize]` below, out of bounds.
                    if indexer < 0
                        || (indexer as usize) >= right_len
                        || (*left_bool && is_extension_array)
                    {
                        result[n] = -1;
                        n += 1;
                        continue;
                    }
                    if *left_bool && !is_extension_array {
                        result[n] = indexer;
                        n += 1;
                        counter += 1;
                        total += 1;
                        continue;
                    }
                    let right_bool = right_booleans[indexer as usize];
                    if right_bool && is_extension_array {
                        result[n] = -1;
                        n += 1;
                        continue;
                    }
                    if right_bool && !is_extension_array {
                        result[n] = indexer;
                        n += 1;
                        counter += 1;
                        total += 1;
                        continue;
                    }
                    let right_val = right[indexer as usize];
                    let compare = op.apply(left_val, &right_val);
                    counter += compare as i64;
                    total += compare as i64;
                    indexer = if compare { indexer } else { -1 };
                    result[n] = indexer;
                    n += 1;
                }
                counts_array[position] = counter;
            }
            Ok((
                result.into_pyarray(py),
                counts_array.into_pyarray(py),
                total,
            ))
        }
    };
}

generic_compare!(compare_posns_ne_int64, i64);
generic_compare!(compare_posns_ne_int32, i32);
generic_compare!(compare_posns_ne_int16, i16);
generic_compare!(compare_posns_ne_int8, i8);
generic_compare!(compare_posns_ne_uint64, u64);
generic_compare!(compare_posns_ne_uint32, u32);
generic_compare!(compare_posns_ne_uint16, u16);
generic_compare!(compare_posns_ne_uint8, u8);
generic_compare!(compare_posns_ne_f64, f64);
generic_compare!(compare_posns_ne_f32, f32);

/// Registers this file's dtype-specialized Python exports.
///
/// ELI5: this file owns a short guest list for just its own exported
/// functions, instead of a central file trying to track every
/// department's exports itself.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compare_posns_ne_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_f32, m)?)?;
    m.add_function(wrap_pyfunction!(compare_posns_ne_f64, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::Python;

    type CompareResult<'py> = PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, i64)>;

    fn run(py: Python<'_>, start: i64, end: i64, indexer: i64) -> CompareResult<'_> {
        let left = PyArray1::from_vec(py, vec![2_i64]);
        let right = PyArray1::from_vec(py, vec![1_i64]);
        let starts = PyArray1::from_vec(py, vec![start]);
        let ends = PyArray1::from_vec(py, vec![end]);
        let positions = PyArray1::from_vec(py, vec![indexer]);
        let left_booleans = PyArray1::from_vec(py, vec![false]);
        let right_booleans = PyArray1::from_vec(py, vec![false]);
        compare_posns_ne_int64(
            py,
            left.readonly(),
            right.readonly(),
            starts.readonly(),
            ends.readonly(),
            positions.readonly(),
            left_booleans.readonly(),
            right_booleans.readonly(),
            false,
            5, // CompareOp::Ne
        )
    }

    #[test]
    fn out_of_bounds_indexer_is_treated_as_no_match_not_a_panic() {
        // right (and right_booleans) have length 1; positions=[5] used
        // to index right_booleans[5]/right[5].
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 1, 5).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), vec![-1_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn end_beyond_positions_len_contributes_nothing_not_a_panic() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 2, 0).expect("must not panic");
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(total, 0);
        });
    }

    #[test]
    fn valid_indexer_compares_normally() {
        Python::initialize();
        Python::attach(|py| {
            if py.import("numpy").is_err() {
                eprintln!("skipping Python-wrapper test: NumPy is unavailable");
                return;
            }
            let (result, counts, total) = run(py, 0, 1, 0).expect("must not panic");
            // left=[2], right=[1], op=Ne: 2 != 1 is true, indexer stays 0.
            assert_eq!(result.readonly().to_vec().unwrap(), vec![0_i64]);
            assert_eq!(counts.readonly().to_vec().unwrap(), vec![1_i64]);
            assert_eq!(total, 1);
        });
    }
}
