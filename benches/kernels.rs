//! Benchmarks for the representative kernels extracted for issue #21:
//! one from each of binary search, comparison, index building, and
//! aggregation. Each is run at a "small" (100 rows) and "large"
//! (100,000 rows) size so a regression that only shows up at scale isn't
//! hidden by a small-N benchmark, and vice versa.
//!
//! Run with `cargo bench` -- no Python interpreter or pyjanitor checkout
//! needed, since these are the plain-Rust `*_core` functions, not the
//! `#[pyfunction]` wrappers.
//!
//! ELI5: this file just calls each kernel many times on made-up input of
//! two different sizes and reports how long the calls took -- the same
//! kind of check as `benchmarks/bench_join_agg_sum.py` on the pyjanitor
//! side (PR pyjanitor-devs/pyjanitor#1673), but for the Rust kernel in
//! isolation.

use criterion::{criterion_group, criterion_main, Criterion};
use numpy::ndarray::Array1;
use std::hint::black_box;

use janitor_rs::aggs::sum::sum_ends::sum_end_core;
use janitor_rs::aggs::sum::sum_starts::sum_start_core;
use janitor_rs::aggs::sum::sum_starts_ends::sum_start_end_core;
use janitor_rs::bin_search::bin_search_lt::binary_search_lt_core;
use janitor_rs::compare::comp::compare_start_end_core;
use janitor_rs::index_builder::{repeat_index_core, trim_index_core};

/// Build the inputs `bench_bin_search_lt` needs at size `n`: an ascending
/// "right" array to search, a "left" array of query values, and starts
/// spread across the whole array (binary search is O(log width), so
/// unlike the sum kernels below, a width that scales with `n` is fine).
struct Fixture {
    right: Array1<i64>,
    left: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
}

impl Fixture {
    fn new(n: usize) -> Self {
        let right = Array1::from_iter((0..n as i64).map(|i| i * 2));
        let left = Array1::from_iter((0..n as i64).map(|i| i * 2 + 1));
        let starts = Array1::from_iter((0..n as i64).map(|i| i % (n as i64 + 1)));
        let ends = Array1::from_elem(n, n as i64);
        Fixture {
            right,
            left,
            starts,
            ends,
        }
    }
}

fn bench_bin_search_lt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bin_search_lt");
    for n in [100, 100_000] {
        let f = Fixture::new(n);
        group.bench_function(format!("n={n}"), |b| {
            b.iter(|| {
                binary_search_lt_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.starts.view()),
                    black_box(f.ends.view()),
                )
            })
        });
    }
    group.finish();
}

fn bench_compare_start_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_start_end");
    for n in [100, 100_000] {
        // one row per query, each with a single-position [i, i+1) range,
        // so the flat matches tape is exactly length n (not n^2)
        let right = Array1::from_iter((0..n as i64).map(|i| i * 2));
        let left = Array1::from_iter((0..n as i64).map(|i| i * 2 + 1));
        let starts = Array1::from_iter(0..n as i64);
        let ends = Array1::from_iter((0..n as i64).map(|i| i + 1));
        let matches = Array1::from_elem(n, 1_i8);
        group.bench_function(format!("n={n}"), |b| {
            b.iter(|| {
                compare_start_end_core(
                    black_box(left.view()),
                    black_box(right.view()),
                    black_box(starts.view()),
                    black_box(ends.view()),
                    black_box(matches.view()),
                    black_box(0), // op=0 -> `>`
                )
            })
        });
    }
    group.finish();
}

fn bench_index_builders(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_builder");
    for n in [100, 100_000] {
        let index = Array1::from_iter(0..n as i64);
        let counts = Array1::from_elem(n, 1_i64);
        group.bench_function(format!("repeat_index n={n}"), |b| {
            b.iter(|| {
                repeat_index_core(black_box(index.view()), black_box(counts.view()), n as i64)
            })
        });
        group.bench_function(format!("trim_index n={n}"), |b| {
            b.iter(|| trim_index_core(black_box(index.view()), black_box(counts.view()), n as i64))
        });
    }
    group.finish();
}

/// `sum_*_core` are O(sum of interval widths), not O(n) -- unlike binary
/// search or the index builders, a width that scales with `n` (like
/// `Fixture` above uses, averaging n/2 per row) would make the "large"
/// case take O(n^2) total work and never finish in a reasonable time.
/// This caps every row's width at `WIDTH` regardless of `n`, so total
/// work stays O(n * WIDTH) -- a bounded-width workload, which is the
/// realistic shape for this kernel (see pyjanitor-devs/pyjanitor#1673 and
/// janitor-rs#26 for the O(sum of widths) vs. O(n + m) discussion this
/// bound is meant to stay clear of).
const SUM_BENCH_WIDTH: i64 = 8;

fn bench_sum_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_prefix_kernels");
    for n in [100, 100_000] {
        let n64 = n as i64;
        let arr = Array1::from_iter((0..n64).map(|i| i * 2));
        let booleans = Array1::from_elem(n, false);

        // sum_start_core sums arr[start..] (to the very end), so every
        // row's start must sit near the end for the width to stay
        // bounded -- one shared start does that for every row.
        let starts_for_sum_start = Array1::from_elem(n, (n64 - SUM_BENCH_WIDTH).max(0));
        // sum_end_core sums arr[..end] (from the very start), so the
        // mirror image: every end sits near the beginning.
        let ends_for_sum_end = Array1::from_elem(n, SUM_BENCH_WIDTH.min(n64));
        // sum_start_end_core takes an explicit [start, end) per row, so a
        // sliding bounded window covering the whole array is realistic
        // and still stays O(n * WIDTH).
        let sliding_starts = Array1::from_iter(0..n64);
        let sliding_ends = Array1::from_iter((0..n64).map(|i| (i + SUM_BENCH_WIDTH).min(n64)));

        group.bench_function(format!("sum_start n={n}"), |b| {
            b.iter(|| {
                sum_start_core(
                    black_box(arr.view()),
                    black_box(starts_for_sum_start.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("sum_end n={n}"), |b| {
            b.iter(|| {
                sum_end_core(
                    black_box(arr.view()),
                    black_box(ends_for_sum_end.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("sum_start_end n={n}"), |b| {
            b.iter(|| {
                sum_start_end_core(
                    black_box(arr.view()),
                    black_box(sliding_starts.view()),
                    black_box(sliding_ends.view()),
                    black_box(booleans.view()),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bin_search_lt,
    bench_compare_start_end,
    bench_index_builders,
    bench_sum_kernels
);
criterion_main!(benches);
