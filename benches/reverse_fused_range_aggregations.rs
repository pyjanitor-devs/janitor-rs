//! Benchmarks for the issue #203 reverse fused range-only APIs.
//!
//! The fixtures deliberately exercise the public Python-facing wrappers. This
//! includes input parsing, dtype dispatch, dense output allocation, null-mask
//! handling, and the shared state traversal. The benchmark dimensions cover
//! small/large source batches, narrow/wide right domains, and sparse/dense
//! coverage. A mixed-null scenario also verifies that mask checks remain part
//! of the measured path.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::registered_module;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::hint::black_box;
use std::time::Duration;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    source_len: usize,
    right_len: usize,
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    mixed_nulls: bool,
    dense: bool,
}

fn request<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyArray1<i64>>,
    mask: &Bound<'py, PyArray1<bool>>,
    operation: &str,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            values.clone().into_any(),
            mask.clone().into_any(),
            operation.into_pyobject(py)?.into_any(),
        ],
    )
}

fn requests<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyArray1<i64>>,
    mask: &Bound<'py, PyArray1<bool>>,
) -> PyResult<Bound<'py, PyList>> {
    let sum = request(py, values, mask, "sum")?;
    let product = request(py, values, mask, "product")?;
    let min = request(py, values, mask, "min")?;
    let max = request(py, values, mask, "max")?;
    let count_nonnull = request(py, values, mask, "count")?;
    let count_all = PyTuple::new(
        py,
        [
            "*".into_pyobject(py)?.into_any(),
            "count".into_pyobject(py)?.into_any(),
        ],
    )?;
    PyList::new(py, [sum, product, min, max, count_nonnull, count_all])
}

fn make_call<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyModule>,
    name: &str,
    shape: Shape,
    scenario: Scenario,
) -> PyResult<(Py<PyAny>, Py<PyTuple>)> {
    let values = PyArray1::from_vec(
        py,
        (0..shape.source_len)
            .map(|row| (row % 17 + 1) as i64)
            .collect(),
    );
    let mask = PyArray1::from_vec(
        py,
        (0..shape.source_len)
            .map(|row| scenario.mixed_nulls && row % 5 == 0)
            .collect(),
    );
    let right_index = PyArray1::from_vec(
        py,
        (0..shape.right_len as i64)
            .map(|position| position * 2 + 1)
            .collect(),
    );
    let aggregations = requests(py, &values, &mask)?;
    let function = module.getattr(name)?.unbind();
    let arguments = match name {
        "aggregate_starts_reverse" => {
            let starts = if scenario.dense {
                vec![0_i64; shape.source_len]
            } else {
                (0..shape.source_len)
                    .map(|row| (row % shape.right_len.max(1)) as i64)
                    .collect()
            };
            let starts = PyArray1::from_vec(py, starts);
            PyTuple::new(
                py,
                [
                    starts.into_any(),
                    right_index.clone().into_any(),
                    aggregations.clone().into_any(),
                ],
            )?
        }
        "aggregate_ends_reverse" => {
            let ends = if scenario.dense {
                vec![shape.right_len as i64; shape.source_len]
            } else {
                (0..shape.source_len)
                    .map(|row| (row % shape.right_len.max(1) + 1) as i64)
                    .collect()
            };
            let ends = PyArray1::from_vec(py, ends);
            PyTuple::new(
                py,
                [
                    ends.into_any(),
                    right_index.clone().into_any(),
                    aggregations.clone().into_any(),
                ],
            )?
        }
        "aggregate_starts_ends_reverse" => {
            let (starts, ends): (Vec<_>, Vec<_>) = if scenario.dense {
                (
                    vec![0_i64; shape.source_len],
                    vec![shape.right_len as i64; shape.source_len],
                )
            } else {
                (
                    (0..shape.source_len)
                        .map(|row| (row % shape.right_len.max(1)) as i64)
                        .collect(),
                    (0..shape.source_len)
                        .map(|row| (row % shape.right_len.max(1) + 1) as i64)
                        .collect(),
                )
            };
            let starts = PyArray1::from_vec(py, starts);
            let ends = PyArray1::from_vec(py, ends);
            PyTuple::new(
                py,
                [
                    starts.into_any(),
                    ends.into_any(),
                    right_index.into_any(),
                    aggregations.into_any(),
                ],
            )?
        }
        _ => unreachable!("benchmark only constructs issue #203 paths"),
    }
    .unbind();
    Ok((function, arguments))
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("reverse_fused_range_aggregations");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(500));

    for shape in [
        Shape {
            name: "small_narrow",
            source_len: 16,
            right_len: 32,
        },
        Shape {
            name: "small_wide",
            source_len: 16,
            right_len: 512,
        },
        Shape {
            name: "large_narrow",
            source_len: 2_048,
            right_len: 64,
        },
        Shape {
            name: "large_wide",
            source_len: 2_048,
            right_len: 4_096,
        },
    ] {
        Python::attach(|py| {
            let module = registered_module(py).expect("module registration must succeed");
            for scenario in [
                Scenario {
                    name: "sparse",
                    mixed_nulls: false,
                    dense: false,
                },
                Scenario {
                    name: "dense",
                    mixed_nulls: false,
                    dense: true,
                },
                Scenario {
                    name: "mixed_nulls",
                    mixed_nulls: true,
                    dense: true,
                },
            ] {
                for name in [
                    "aggregate_starts_reverse",
                    "aggregate_ends_reverse",
                    "aggregate_starts_ends_reverse",
                ] {
                    let call = make_call(py, &module, name, shape, scenario)
                        .expect("benchmark fixture must satisfy wrapper preconditions");
                    let label = format!("{}/{}/{}", name, shape.name, scenario.name);
                    group.bench_with_input(
                        BenchmarkId::new("public_wrapper", label),
                        &call,
                        |b, call| {
                            b.iter(|| {
                                call.0
                                    .bind(py)
                                    .call1(black_box(call.1.bind(py)))
                                    .expect("benchmark fixture must remain valid");
                            });
                        },
                    );
                }
            }
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
