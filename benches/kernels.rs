//! Benchmarks for retained binary-search, index-building, and aggregation kernels.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{
    binary_search_ge_first_core, binary_search_gt_first_core, binary_search_le_first_core,
    binary_search_lt_core, binary_search_lt_first_core, min_positions_core, repeat_index_core,
    sum_end_core, sum_start_core, sum_start_end_core, sum_start_u32_core, trim_index_core,
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

    c.bench_function("binary_search_gt_first", |b| {
        b.iter(|| {
            binary_search_gt_first_core(
                black_box(left.view()),
                black_box(right.view()),
                black_box(left_index.view()),
            )
        })
    });

    c.bench_function("binary_search_ge_first", |b| {
        b.iter(|| {
            binary_search_ge_first_core(
                black_box(left.view()),
                black_box(right.view()),
                black_box(left_index.view()),
            )
        })
    });

    c.bench_function("binary_search_le_first", |b| {
        b.iter(|| {
            binary_search_le_first_core(
                black_box(left.view()),
                black_box(right.view()),
                black_box(left_index.view()),
            )
        })
    });
}

/// `sum_*_core` are O(sum of interval widths) on their direct-scan path, not
/// O(n) -- a width that scales with `n` (e.g. a start near the front of the
/// array for every row) would make this benchmark O(n^2). Bound every row's
/// width to a small constant regardless of `n` (see AGENTS.md's "Aggregation
/// benchmarks: bound the interval width" gotcha).
const SUM_WIDTH: i64 = 8;

fn bench_sum(c: &mut Criterion) {
    let n = 100_000;
    let n64 = n as i64;
    let values = Array1::from_iter(0..n64);
    let mask = Array1::from_elem(n, false);

    // sum_start sums arr[start..] to the end, so every start must sit near
    // the end for the width to stay bounded.
    let starts = Array1::from_elem(n, (n64 - SUM_WIDTH).max(0));
    c.bench_function("sum_start", |b| {
        b.iter(|| {
            sum_start_core(
                black_box(values.view()),
                black_box(starts.view()),
                black_box(mask.view()),
            )
        })
    });

    // sum_end sums arr[..end] from the very start, so the mirror image:
    // every end sits near the beginning.
    let ends = Array1::from_elem(n, SUM_WIDTH.min(n64));
    c.bench_function("sum_end", |b| {
        b.iter(|| {
            sum_end_core(
                black_box(values.view()),
                black_box(ends.view()),
                black_box(mask.view()),
            )
        })
    });

    // sum_start_end takes an explicit [start, end) per row, so a sliding
    // bounded window covering the whole array is realistic and stays
    // O(n * SUM_WIDTH).
    let sliding_starts = Array1::from_iter(0..n64);
    let sliding_ends = Array1::from_iter((0..n64).map(|value| (value + SUM_WIDTH).min(n64)));
    c.bench_function("sum_start_end", |b| {
        b.iter(|| {
            sum_start_end_core(
                black_box(values.view()),
                black_box(sliding_starts.view()),
                black_box(sliding_ends.view()),
                black_box(mask.view()),
            )
        })
    });

    // A large u32 column with a single tiny suffix query, protecting the
    // cast-on-access optimization: reading only the queried width should
    // stay cheap even though the underlying column is large.
    let u32_values = Array1::from_iter(0..n as u32);
    let u32_mask = Array1::from_elem(n, false);
    let u32_starts = Array1::from_elem(1, (n64 - SUM_WIDTH).max(0));
    c.bench_function("sum_start_u32", |b| {
        b.iter(|| {
            sum_start_u32_core(
                black_box(u32_values.view()),
                black_box(u32_starts.view()),
                black_box(u32_mask.view()),
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
