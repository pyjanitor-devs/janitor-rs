use pyo3::prelude::*;
mod aggs;
mod bin_search;
mod compare;
mod index_builder;
mod left_le_right;

/// Helper functions for PyJanitor implemented in Rust.
#[pymodule]
fn janitor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {


    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_no_range::compute_prod_rev_no_range_f64,
        m
    )?)?;


























    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_no_range::compute_sum_rev_no_range_f64,
        m
    )?)?;















    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_no_range::compute_min_rev_no_range_f64,
        m
    )?)?;

















    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_no_range::compute_max_rev_no_range_f64,
        m
    )?)?;










    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts::compute_max_rev_start_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends::compute_max_rev_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_matches::compute_max_rev_start_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_ends_matches::compute_max_rev_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_positions::compute_max_rev_positions_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends::compute_max_rev_start_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max_rev::max_starts_ends_matches::compute_max_rev_start_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts::compute_min_rev_start_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends::compute_min_rev_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_matches::compute_min_rev_start_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_ends_matches::compute_min_rev_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_positions::compute_min_rev_positions_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends::compute_min_rev_start_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min_rev::min_starts_ends_matches::compute_min_rev_start_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts::compute_prod_rev_start_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends::compute_prod_rev_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_matches::compute_prod_rev_start_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_ends_matches::compute_prod_rev_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_positions::compute_prod_rev_positions_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends::compute_prod_rev_start_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod_rev::prod_starts_ends_matches::compute_prod_rev_start_end_match_f64,
        m
    )?)?;

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

    m.add_function(wrap_pyfunction!(compare::comp::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends::compute_sum_rev_start_end_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_ends::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_ends::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::prod::prod_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_starts::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts::compute_sum_rev_start_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_start,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_no_range,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_end,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_positions,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_start_matches,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_end_matches,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_start_end,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::size_rev::computes::compute_size_rev_start_end_matches,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_positions::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_positions::compute_sum_rev_positions_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_positions::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::min::min_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_positions::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_positions::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_positions::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::max::max_positions::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_positions::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_starts_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_matches::compute_sum_rev_start_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_starts_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::max::max_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_starts_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::min::min_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_starts_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::prod::prod_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum::sum_ends_matches::compute_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_starts_ends_matches::compute_sum_rev_start_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_int64, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_int32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_int16, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_int8, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_f32, m)?)?;
    m.add_function(wrap_pyfunction!(aggs::sum::sum_ends::compute_f64, m)?)?;

    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends::compute_sum_rev_end_f64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_f32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        aggs::sum_rev::sum_ends_matches::compute_sum_rev_end_match_f64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_lt::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_le::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_ge::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        bin_search::bin_search_gt::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range_ne::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_no_range::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_no_range::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_starts::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_starts::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_starts::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_starts::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_starts::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_starts_1st::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_ends::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ne_1st::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_ne_ends_1st::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(compare::comp_first::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_int8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_starts::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_uint64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_uint32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_uint16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_uint8,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_int64,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_int32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_int16,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(compare::comp_first_ends::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_first_ends::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_starts::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_ends::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_float32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns::compare_float64, m)?)?;

    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_uint64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_uint32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_uint16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_uint8, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_int64, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_int32, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_int16, m)?)?;
    m.add_function(wrap_pyfunction!(compare::comp_posns_ne::compare_int8, m)?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_posns_ne::compare_float32,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        compare::comp_posns_ne::compare_float64,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(left_le_right::region_positions, m)?)?;

    Ok(())
}
