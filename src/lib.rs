use pyo3::prelude::*;

mod bin_search_ge;
mod bin_search_gt;
mod bin_search_le;
mod bin_search_lt;
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
mod comp_no_range;
mod comp_no_range_ne;
mod comp_posns;
mod comp_posns_ne;
mod comp_starts;
mod index_builder;
mod left_le_right;
mod max_ends;
mod max_ends_matches;
mod max_positions;
mod max_starts;
mod max_starts_ends;
mod max_starts_ends_matches;
mod max_starts_matches;
mod min_ends;
mod min_ends_matches;
mod min_positions;
mod min_starts;
mod min_starts_ends;
mod min_starts_ends_matches;
mod min_starts_matches;
mod prod_ends;
mod prod_ends_matches;
mod prod_positions;
mod prod_starts;
mod prod_starts_ends;
mod prod_starts_ends_matches;
mod prod_starts_matches;
mod sum_ends;
mod sum_ends_matches;
mod sum_positions;
mod sum_starts;
mod sum_starts_ends;
mod sum_starts_ends_matches;
mod sum_starts_matches;

/// Helper functions for PyJanitor implemented in Rust.
#[pymodule]
fn janitor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index_builder::index_repeat, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_trim, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::reorder_indices, m)?)?;
    m.add_function(wrap_pyfunction!(index_builder::index_trim_pos, m)?)?;
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

    m.add_function(wrap_pyfunction!(min_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(max_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        sum_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sum_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        sum_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        prod_starts_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        max_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        max_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        min_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        min_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        min_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(min_starts_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(max_starts_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(max_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(prod_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(prod_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(min_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(sum_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(sum_positions::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(min_positions::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(max_positions::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(prod_positions::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_starts_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(max_starts_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_starts_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(max_ends_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(max_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(min_starts_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_starts_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(min_ends_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(min_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_starts_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(prod_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends_matches::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(sum_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(sum_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(bin_search_lt::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_lt::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(bin_search_le::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_le::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(bin_search_ge::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_ge::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(bin_search_gt::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(bin_search_gt::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range_ne::compare_float64, m)?)?;

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

    m.add_function(wrap_pyfunction!(comp_no_range::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(comp_no_range::compare_float64, m)?)?;

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
