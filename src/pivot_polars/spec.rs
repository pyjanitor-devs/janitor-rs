use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::{PyLazyFrame, PyDataFrame};
use polars_lazy::frame::IntoLazy;
use polars_lazy::prelude::LazyFrame;

// import ideas from https://github.com/pola-rs/pyo3-polars/pull/14/changes
// relevant changes here and in Cargo.toml
#[pyfunction]
pub fn spec_reshape(pydf: PyDataFrame, spec: PyDataFrame) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();
    let _:DataFrame = spec.into();
    // let df = {
    //     // some work on the dataframe here
    //     todo!()
    // };

    // wrap the dataframe and it will be automatically converted to a python polars dataframe
    Ok(PyDataFrame(df))
}