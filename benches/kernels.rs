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

/// Every benchmark below builds its inputs via a small, purpose-built
/// `<Kernel>Fixture::new(n)` -- one convention across the file, rather
/// than each benchmark choosing its own way to assemble arrays. A shared
/// fixture isn't used across kernels: each needs a genuinely different
/// shape (see each struct's own doc comment), and forcing them into one
/// struct would just bloat it with fields only one kernel uses.
///
/// Inputs for `bench_bin_search_lt`: an ascending "right" array to
/// search, a "left" array of query values, and starts spread across the
/// whole array (binary search is O(log width), so unlike the sum kernels
/// below, a width that scales with `n` is fine).
struct BinarySearchFixture {
    right: Array1<i64>,
    left: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
}

impl BinarySearchFixture {
    fn new(n: usize) -> Self {
        let right = Array1::from_iter((0..n as i64).map(|i| i * 2));
        let left = Array1::from_iter((0..n as i64).map(|i| i * 2 + 1));
        // starts[i] = i, spread evenly across the whole array
        let starts = Array1::from_iter(0..n as i64);
        let ends = Array1::from_elem(n, n as i64);
        BinarySearchFixture {
            right,
            left,
            starts,
            ends,
        }
    }
}

/// ELI5: for every value in `left`, find where it would slot into the
/// sorted `right` array -- the building block behind a `<` join.
fn bench_bin_search_lt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bin_search_lt");
    for n in [100, 100_000] {
        let f = BinarySearchFixture::new(n);
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

/// Inputs for `bench_compare_start_end`: one row per query, each with a
/// single-position `[i, i+1)` range, so the flat `matches` tape is
/// exactly length `n` (not `n^2`).
struct CompareFixture {
    right: Array1<i64>,
    left: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
    matches: Array1<i8>,
}

impl CompareFixture {
    fn new(n: usize) -> Self {
        let right = Array1::from_iter((0..n as i64).map(|i| i * 2));
        let left = Array1::from_iter((0..n as i64).map(|i| i * 2 + 1));
        let starts = Array1::from_iter(0..n as i64);
        let ends = Array1::from_iter((0..n as i64).map(|i| i + 1));
        let matches = Array1::from_elem(n, 1_i8);
        CompareFixture {
            right,
            left,
            starts,
            ends,
            matches,
        }
    }
}

/// ELI5: for every row's slice of `right`, mark which positions satisfy
/// `left OP right` -- the building block behind a range-join predicate
/// (e.g. `A < B and C != D`).
fn bench_compare_start_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_start_end");
    for n in [100, 100_000] {
        let f = CompareFixture::new(n);
        group.bench_function(format!("n={n}"), |b| {
            b.iter(|| {
                compare_start_end_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.starts.view()),
                    black_box(f.ends.view()),
                    black_box(f.matches.view()),
                    black_box(0), // op=0 -> `>`
                )
            })
        });
    }
    group.finish();
}

/// Inputs for `bench_index_builders`: an `index` array and a `counts`
/// array of matching length, one entry per row.
struct IndexBuilderFixture {
    index: Array1<i64>,
    counts: Array1<i64>,
}

impl IndexBuilderFixture {
    fn new(n: usize) -> Self {
        let index = Array1::from_iter(0..n as i64);
        let counts = Array1::from_elem(n, 1_i64);
        IndexBuilderFixture { index, counts }
    }
}

/// ELI5: `repeat_index` turns "3 apples, 2 plums" into "apple, apple,
/// apple, plum, plum" (numpy.repeat); `trim_index` drops the entries
/// whose count was zero. Both build the final left/right index arrays a
/// join returns.
fn bench_index_builders(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_builder");
    for n in [100, 100_000] {
        let f = IndexBuilderFixture::new(n);
        group.bench_function(format!("repeat_index n={n}"), |b| {
            b.iter(|| {
                repeat_index_core(
                    black_box(f.index.view()),
                    black_box(f.counts.view()),
                    n as i64,
                )
            })
        });
        group.bench_function(format!("trim_index n={n}"), |b| {
            b.iter(|| {
                trim_index_core(
                    black_box(f.index.view()),
                    black_box(f.counts.view()),
                    n as i64,
                )
            })
        });
    }
    group.finish();
}

/// `sum_*_core` are O(sum of interval widths), not O(n) -- unlike binary
/// search or the index builders, a width that scales with `n` (like
/// `BinarySearchFixture` uses, averaging n/2 per row) would make the
/// "large" case take O(n^2) total work and never finish in a reasonable
/// time. This caps every row's width at `WIDTH` regardless of `n`, so
/// total work stays O(n * WIDTH) -- a bounded-width workload, which is
/// the realistic shape for this kernel (see pyjanitor-devs/pyjanitor#1673
/// and janitor-rs#26 for the O(sum of widths) vs. O(n + m) discussion
/// this bound is meant to stay clear of).
const SUM_BENCH_WIDTH: i64 = 8;

/// Inputs for `bench_sum_kernels`: a shared `arr`/`booleans`, plus three
/// bounded-width start/end variants -- one per kernel, since each takes a
/// different shape of range (suffix, prefix, or an explicit `[start,
/// end)` window).
struct SumFixture {
    arr: Array1<i64>,
    booleans: Array1<bool>,
    starts_for_sum_start: Array1<i64>,
    ends_for_sum_end: Array1<i64>,
    sliding_starts: Array1<i64>,
    sliding_ends: Array1<i64>,
}

impl SumFixture {
    fn new(n: usize) -> Self {
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

        SumFixture {
            arr,
            booleans,
            starts_for_sum_start,
            ends_for_sum_end,
            sliding_starts,
            sliding_ends,
        }
    }
}

fn bench_sum_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_prefix_kernels");
    for n in [100, 100_000] {
        let f = SumFixture::new(n);

        group.bench_function(format!("sum_start n={n}"), |b| {
            b.iter(|| {
                sum_start_core(
                    black_box(f.arr.view()),
                    black_box(f.starts_for_sum_start.view()),
                    black_box(f.booleans.view()),
                )
            })
        });
        group.bench_function(format!("sum_end n={n}"), |b| {
            b.iter(|| {
                sum_end_core(
                    black_box(f.arr.view()),
                    black_box(f.ends_for_sum_end.view()),
                    black_box(f.booleans.view()),
                )
            })
        });
        group.bench_function(format!("sum_start_end n={n}"), |b| {
            b.iter(|| {
                sum_start_end_core(
                    black_box(f.arr.view()),
                    black_box(f.sliding_starts.view()),
                    black_box(f.sliding_ends.view()),
                    black_box(f.booleans.view()),
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
