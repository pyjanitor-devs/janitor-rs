//! Production-wrapper benchmark for reverse no-range aggregations.
//!
//! Unlike the algorithm experiments, this benchmark invokes the actual
//! Python-facing functions registered by the extension module. It therefore
//! includes PyO3 argument dispatch and the reducer's real allocations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::registered_module;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use std::hint::black_box;

mod support;
use support::count_allocations;

struct Fixture {
    arr: Py<PyAny>,
    left_index: Py<PyAny>,
    right_index: Py<PyAny>,
    booleans: Py<PyAny>,
}

fn array<T: numpy::Element>(py: Python<'_>, values: Vec<T>) -> Py<PyAny> {
    PyArray1::from_vec(py, values).unbind().into_any()
}

fn fixture(py: Python<'_>, pairs: usize, labels: usize) -> Fixture {
    Fixture {
        arr: array(py, (0..pairs).map(|value| (value % 97) as i64).collect()),
        left_index: array(py, (0..pairs).map(|value| (value % pairs) as i64).collect()),
        right_index: array(
            py,
            (0..pairs).map(|value| (value % labels) as i64).collect(),
        ),
        booleans: array(py, vec![false; pairs]),
    }
}

fn call_args<'py>(py: Python<'py>, fixture: &Fixture) -> Bound<'py, PyTuple> {
    PyTuple::new(
        py,
        [
            fixture.arr.clone_ref(py),
            fixture.left_index.clone_ref(py),
            fixture.right_index.clone_ref(py),
            fixture.booleans.clone_ref(py),
        ],
    )
    .expect("benchmark arguments must be constructible")
}

fn invoke(py: Python<'_>, module: &Bound<'_, PyModule>, name: &str, fixture: &Fixture) {
    let function = module
        .getattr(name)
        .unwrap_or_else(|error| panic!("missing registered function {name}: {error}"));
    function
        .call1(call_args(py, fixture))
        .unwrap_or_else(|error| panic!("production wrapper {name} failed: {error}"));
}

fn bench_no_range_production(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("reverse_no_range_production_wrappers");
    group.sample_size(10);

    for &(name, labels) in &[("repeated_labels", 100), ("near_unique_labels", 100_000)] {
        Python::attach(|py| {
            let module = registered_module(py).expect("module registration must succeed");
            let fixture = fixture(py, 100_000, labels);

            eprintln!("{name} allocation report (bytes / allocations / peak-live):");
            for family in ["sum", "prod", "min", "max"] {
                let function = format!("compute_{family}_rev_no_range_int64");
                let report = count_allocations(|| invoke(py, &module, &function, &fixture));
                eprintln!("  {family}: {report:?}");
                group.bench_with_input(BenchmarkId::new(family, name), &fixture, |b, fixture| {
                    b.iter(|| invoke(py, &module, &function, black_box(fixture)))
                });
            }
        });
    }

    group.finish();
}

criterion_group!(benches, bench_no_range_production);
criterion_main!(benches);
