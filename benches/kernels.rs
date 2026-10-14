use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{dual_regions_aggregate, dual_regions_and_residual_aggregate};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::hint::black_box;

fn make_predicate<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
    operator: &str,
) -> Bound<'py, PyTuple> {
    PyTuple::new(
        py,
        [
            left.clone(),
            right.clone(),
            operator.into_pyobject(py).unwrap().into_any(),
        ],
    )
    .unwrap()
}

fn make_size_aggregation<'py>(py: Python<'py>) -> Bound<'py, PyList> {
    PyList::new(
        py,
        [PyTuple::new(
            py,
            [
                "*".into_pyobject(py).unwrap().into_any(),
                "size".into_pyobject(py).unwrap().into_any(),
            ],
        )
        .unwrap()],
    )
    .unwrap()
}

fn selective_second_predicate(c: &mut Criterion) {
    Python::initialize();
    Python::attach(|py| {
        let rows = 100_000_usize;
        let left_index = PyArray1::from_vec(py, (0..rows as i64).collect());
        let right_index = PyArray1::from_vec(py, (0..rows as i64).collect());
        let first_left = PyArray1::from_vec(py, vec![0_i64; rows]);
        let second_left = PyArray1::from_vec(py, vec![0_i64; rows]);
        let right_ascending = PyArray1::from_vec(py, (0..rows as i64).collect());
        let right_descending = PyArray1::from_vec(py, (0..rows as i64).rev().collect());
        let first = make_predicate(py, first_left.as_any(), right_ascending.as_any(), "<=");
        let second = make_predicate(py, second_left.as_any(), right_descending.as_any(), ">=");
        let aligner = PyArray1::from_vec(py, (0..rows as i64).rev().collect()).readonly();
        let aggregations = make_size_aggregation(py);

        // An always-true residual forces the pre-tree traversal while keeping
        // the same selective second predicate and aggregation contract.
        let residual_left = PyArray1::from_vec(py, vec![1_i64; rows]);
        let residual_right = PyArray1::from_vec(py, vec![1_i64; rows]);
        let residual = PyList::new(
            py,
            [make_predicate(
                py,
                residual_left.as_any(),
                residual_right.as_any(),
                "==",
            )],
        )
        .unwrap();

        let mut group = c.benchmark_group("dual_regions_selective_second_predicate");
        group.bench_function("adaptive", |b| {
            b.iter(|| {
                let result = dual_regions_aggregate(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &second,
                    aligner.clone(),
                    &aggregations,
                    false,
                )
                .unwrap();
                black_box(result);
            });
        });
        group.bench_function("forced_legacy", |b| {
            b.iter(|| {
                let result = dual_regions_and_residual_aggregate(
                    py,
                    left_index.as_any(),
                    right_index.as_any(),
                    &first,
                    &second,
                    aligner.clone(),
                    &residual,
                    &aggregations,
                    false,
                )
                .unwrap();
                black_box(result);
            });
        });
        group.finish();
    });
}

criterion_group!(benches, selective_second_predicate);
criterion_main!(benches);
