use pyo3::prelude::*;

mod comp;
mod comp_ends;
mod comp_first;
mod comp_first_ends;
mod comp_first_starts;
mod comp_ne;
mod comp_ne_1st;
mod comp_ne_ends;
mod comp_ne_ends_1st;
mod comp_ne_starts;
mod comp_ne_starts_1st;
mod comp_posns;
mod comp_posns_ne;
mod comp_starts;
mod index_builder;
mod left_le_right;
/// Helper functions for PyJanitor implemented in Rust.
#[pymodule]
fn janitor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index_builder::index_repeat, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_trim, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_positions, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_positions_first, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_positions_last, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_starts, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_starts_1st, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_starts_last, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_ends, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_ends_1st, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_ends_last, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_starts_ends, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_starts_ends_1st, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_starts_ends_last, m)?)?;

    m.add_function(wrap_pyfunction!(comp::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ne::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_starts_1st::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_1st::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ne_ends_1st::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_first::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_first_starts::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_starts::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_first_ends::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_first_ends::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_starts::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_starts::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_ends::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_ends::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_posns::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_posns_ne::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(left_le_right::region_positions, m)?)?;

    Ok(())
}
