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
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use janitor_rs::bench_support::{
    binary_search_ge_first_core, binary_search_gt_first_core, binary_search_le_first_core,
    binary_search_lt_core, binary_search_lt_first_core, compare_start_end_core, repeat_index_core,
    sum_end_core, sum_start_core, sum_start_end_core, sum_start_u32_core, trim_index_core,
};

/// Counts bytes and calls allocated through the global allocator, so
/// `bench_bin_search_first` can report an allocation delta for a single
/// call alongside criterion's timing -- criterion itself only measures
/// wall time, and the whole point of the one-pass conversion (issue #24)
/// is fewer allocations, not just less time.
struct CountingAllocator;

static BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Runs `f` once and returns `(bytes allocated, allocation calls)` charged
/// to it specifically, isolated from whatever the harness itself has
/// already allocated by taking a before/after delta rather than an
/// absolute count.
fn count_allocations<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    let bytes_before = BYTES_ALLOCATED.load(Ordering::Relaxed);
    let calls_before = ALLOC_CALLS.load(Ordering::Relaxed);
    black_box(f());
    (
        BYTES_ALLOCATED.load(Ordering::Relaxed) - bytes_before,
        ALLOC_CALLS.load(Ordering::Relaxed) - calls_before,
    )
}

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

/// Inputs for `bench_bin_search_first`: every `left[i]` is guaranteed to
/// find a match somewhere in `right`, exercising the high-survival case
/// for the grow-on-demand output vectors.
struct BinarySearchFirstFixture {
    right: Array1<i64>,
    left: Array1<i64>,
    left_index: Array1<i64>,
}

impl BinarySearchFirstFixture {
    fn new(n: usize) -> Self {
        let right = Array1::from_iter((0..n as i64).map(|i| i * 2));
        let left = Array1::from_iter((0..n as i64).map(|i| i * 2 + 1));
        let left_index = Array1::from_iter(0..n as i64);
        BinarySearchFirstFixture {
            right,
            left,
            left_index,
        }
    }
}

/// ELI5: for every value in `left`, find its match position in `right`
/// and keep only the rows that actually matched -- the one-pass version
/// (issue #24) does this by pushing straight into two `Vec`s instead of
/// writing a full-length array with an internal marker and then filtering
/// it in a second pass. Covers all four operator directions (`<`, `>`,
/// `>=`, `<=`), since each uses a different internal marker convention.
fn bench_bin_search_first(c: &mut Criterion) {
    // One-time allocation report (not per criterion iteration -- criterion
    // runs each closure many times, and printing on every run would be
    // noise, not a report). See `count_allocations`'s doc comment for why
    // this needs its own instrumentation rather than criterion's built-in
    // timing.
    eprintln!("\nbin_search_first allocation report (single call, bytes / alloc count):");
    for n in [100, 100_000] {
        let f = BinarySearchFirstFixture::new(n);
        let (bytes, calls) = count_allocations(|| {
            binary_search_lt_first_core(f.left.view(), f.right.view(), f.left_index.view())
        });
        eprintln!("  lt_first n={n:>7}: {bytes:>9} bytes / {calls:>3} allocs");

        // A deliberately sparse case: every query is above `right`, so no
        // row survives the strict-less-than-first search. This guards the
        // memory claim that dropped rows do not trigger two full buffers.
        let sparse = BinarySearchFirstFixture {
            right: Array1::from_iter((0..n as i64).map(|i| i * 2)),
            left: Array1::from_elem(n, n as i64 * 2 + 1),
            left_index: Array1::from_iter(0..n as i64),
        };
        let (bytes, calls) = count_allocations(|| {
            binary_search_lt_first_core(
                sparse.left.view(),
                sparse.right.view(),
                sparse.left_index.view(),
            )
        });
        eprintln!("  lt_first sparse n={n:>7}: {bytes:>9} bytes / {calls:>3} allocs");
    }

    let mut group = c.benchmark_group("bin_search_first");
    for n in [100, 100_000] {
        let f = BinarySearchFirstFixture::new(n);
        group.bench_function(format!("lt n={n}"), |b| {
            b.iter(|| {
                binary_search_lt_first_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.left_index.view()),
                )
            })
        });
        group.bench_function(format!("gt n={n}"), |b| {
            b.iter(|| {
                binary_search_gt_first_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.left_index.view()),
                )
            })
        });
        group.bench_function(format!("ge n={n}"), |b| {
            b.iter(|| {
                binary_search_ge_first_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.left_index.view()),
                )
            })
        });
        group.bench_function(format!("le n={n}"), |b| {
            b.iter(|| {
                binary_search_le_first_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.left_index.view()),
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

/// A large column with one tiny suffix query, used to protect the
/// cast-on-access optimization in non-i64 integer wrappers.
///
/// ELI5: the regular throughput fixture asks for eight values `n` times.
/// This one asks once, so copying all `n` values before reading the final
/// eight becomes glaringly expensive instead of hiding among useful work.
struct SparseCastSumFixture {
    arr: Array1<u32>,
    booleans: Array1<bool>,
    starts: Array1<i64>,
}

impl SparseCastSumFixture {
    fn new(n: usize) -> Self {
        Self {
            arr: Array1::from_iter(0..n as u32),
            booleans: Array1::from_elem(n, false),
            starts: Array1::from_elem(1, (n as i64 - SUM_BENCH_WIDTH).max(0)),
        }
    }
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

    for n in [100, 100_000] {
        let f = SparseCastSumFixture::new(n);
        group.bench_function(format!("sum_start_u32 sparse arr_n={n} queries=1"), |b| {
            b.iter(|| {
                sum_start_u32_core(
                    black_box(f.arr.view()),
                    black_box(f.starts.view()),
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
    bench_bin_search_first,
    bench_compare_start_end,
    bench_index_builders,
    bench_sum_kernels
);
criterion_main!(benches);
