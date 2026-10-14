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

const BROAD_ROWS: usize = 4_096;

/// Benchmark adaptive vs. forced-legacy dual-region aggregation on a broad
/// candidate set, where the second predicate keeps most first-predicate pairs.
///
/// The right side is `0..BROAD_ROWS` for both predicates; `first_left` and
/// `second_left` decide how each left row's window opens.
fn bench_broad_dual(
    c: &mut Criterion,
    group_name: &str,
    first_left: Vec<i64>,
    second_left: Vec<i64>,
) {
    Python::initialize();
    Python::attach(|py| {
        let rows = BROAD_ROWS;
        let left_index = PyArray1::from_vec(py, (0..rows as i64).collect());
        let right_index = PyArray1::from_vec(py, (0..rows as i64).collect());
        let first_left = PyArray1::from_vec(py, first_left);
        let second_left = PyArray1::from_vec(py, second_left);
        let right_ascending = PyArray1::from_vec(py, (0..rows as i64).collect());
        let right_descending = PyArray1::from_vec(py, (0..rows as i64).rev().collect());
        let first = make_predicate(py, first_left.as_any(), right_ascending.as_any(), "<=");
        let second = make_predicate(py, second_left.as_any(), right_descending.as_any(), ">=");
        let aligner = PyArray1::from_vec(py, (0..rows as i64).rev().collect()).readonly();
        let aggregations = make_size_aggregation(py);

        // Keep the comparison on the same broad candidate set while forcing
        // the legacy implementation to inspect each matching pair.
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

        let mut group = c.benchmark_group(group_name);
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

/// Every left row matches every right row, and every right row becomes
/// visible at the first left row: the rank tree's best case.
fn broad_dual_predicate(c: &mut Criterion) {
    bench_broad_dual(
        c,
        "dual_regions_broad",
        vec![0; BROAD_ROWS],
        vec![BROAD_ROWS as i64; BROAD_ROWS],
    );
}

/// Scattered left keys give each left row a different first-predicate
/// window and second-predicate threshold.
///
/// ELI5: unlike the all-at-once case, the tree grows a little between left
/// rows and each query covers a different label suffix, which is closer to
/// a real join. Multiplying by a constant coprime to `BROAD_ROWS` shuffles
/// `0..BROAD_ROWS` without a random-number dependency.
fn staggered_dual_predicate(c: &mut Criterion) {
    let rows = BROAD_ROWS as i64;
    bench_broad_dual(
        c,
        "dual_regions_staggered",
        (0..rows).map(|v| (v * 2_654_435_761) % rows).collect(),
        (0..rows).map(|v| (v * 40_503) % rows).collect(),
    );
}

criterion_group!(
    benches,
    selective_second_predicate,
    broad_dual_predicate,
    staggered_dual_predicate
);
criterion_main!(benches);
