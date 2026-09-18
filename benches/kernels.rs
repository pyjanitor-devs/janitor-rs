//! Benchmarks for retained binary-search, index-building, and aggregation kernels.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{
    binary_search_lt_core, binary_search_lt_first_core, min_positions_core, repeat_index_core,
    sum_start_core, trim_index_core,
};
use numpy::ndarray::Array1;
use std::hint::black_box;

fn bench_binary_search(c: &mut Criterion) {
    let n = 100_000;
    let left = Array1::from_iter((0..n as i64).map(|value| value * 2 + 1));
    let right = Array1::from_iter((0..n as i64).map(|value| value * 2));
    let starts = Array1::zeros(n);
    let ends = Array1::from_elem(n, n as i64);

    c.bench_function("binary_search_lt", |b| {
        b.iter(|| {
            binary_search_lt_core(
                black_box(left.view()),
                black_box(right.view()),
                black_box(starts.view()),
                black_box(ends.view()),
            )
        })
    });
}

fn bench_binary_search_first(c: &mut Criterion) {
    let n = 100_000;
    let left = Array1::from_iter((0..n as i64).map(|value| value * 2 + 1));
    let right = Array1::from_iter((0..n as i64).map(|value| value * 2));
    let left_index = Array1::from_iter(0..n as i64);

    c.bench_function("binary_search_lt_first", |b| {
        b.iter(|| {
            binary_search_lt_first_core(
                black_box(left.view()),
                black_box(right.view()),
                black_box(left_index.view()),
            )
        })
    });
}

fn bench_sum(c: &mut Criterion) {
    let n = 100_000;
    let values = Array1::from_iter(0..n as i64);
    let starts = Array1::from_elem(n, 0_i64);
    let mask = Array1::from_elem(n, false);

    c.bench_function("sum_start", |b| {
        b.iter(|| {
            sum_start_core(
                black_box(values.view()),
                black_box(starts.view()),
                black_box(mask.view()),
            )
        })
    });
}

fn bench_index_building(c: &mut Criterion) {
    let n = 100_000;
    let index = Array1::from_iter(0..n as i64);
    let counts = Array1::from_elem(n, 1_i64);

    c.bench_function("repeat_index", |b| {
        b.iter(|| repeat_index_core(black_box(index.view()), black_box(counts.view()), n as i64))
    });

    c.bench_function("trim_index", |b| {
        b.iter(|| trim_index_core(black_box(index.view()), black_box(counts.view()), n as i64))
    });
}

fn bench_min_positions(c: &mut Criterion) {
    let n = 100_000;
    let values = Array1::from_iter(0..n as i64);
    let starts = Array1::zeros(n);
    let ends = Array1::from_elem(n, n as i64);
    let index = Array1::from_iter(0..n as i64);
    let positions = Array1::from_iter(0..n as i64);
    let mask = Array1::from_elem(n, false);

    c.bench_function("min_positions", |b| {
        b.iter(|| {
            min_positions_core(
                black_box(values.view()),
                black_box(starts.view()),
                black_box(ends.view()),
                black_box(index.view()),
                black_box(positions.view()),
                black_box(mask.view()),
            )
        })
    });
}

criterion_group!(
    benches,
    bench_binary_search,
    bench_binary_search_first,
    bench_sum,
    bench_index_building,
    bench_min_positions
);
criterion_main!(benches);
