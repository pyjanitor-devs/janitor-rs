use pyo3::prelude::*;

mod index_builder;

/// A Python module implemented in Rust.
#[pymodule]
fn janitor_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(index_builder::left_index_single, m)?)?;
    Ok(())
}
