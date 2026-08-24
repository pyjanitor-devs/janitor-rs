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
use numpy::ndarray::{Array1, ArrayView1};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

use janitor_rs::bench_support::{
    binary_search_ge_first_core, binary_search_gt_first_core, binary_search_le_first_core,
    binary_search_lt_core, binary_search_lt_first_core, compare_start_end_core, max_end_core,
    max_start_core, max_start_end_core, min_end_core, min_start_core, min_start_end_core,
    repeat_index_core, sum_end_core, sum_start_core, sum_start_end_core, sum_start_u32_core,
    trim_index_core, CompareOp,
};

/// Counts bytes, calls, and outstanding (live) bytes allocated through the
/// global allocator, so `bench_bin_search_first` can report an allocation
/// delta -- and `bench_bin_search_first_old_vs_new` a peak-memory delta --
/// for a single call alongside criterion's timing. Criterion itself only
/// measures wall time, and the whole point of the one-pass conversion
/// (issue #24) and the follow-up grow-on-demand fix is fewer allocations
/// and less peak memory, not just less time.
struct CountingAllocator;

static BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        let current = CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK_BYTES.fetch_max(current, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Runs `f` once and returns `(bytes allocated, allocation calls, peak
/// live bytes)` charged to it specifically, isolated from whatever the
/// harness itself has already allocated/holds by taking before/after
/// deltas rather than absolute counts. `PEAK_BYTES` is reset to the
/// pre-call live-byte baseline immediately before `f` runs (safe because
/// these benches are single-threaded and sequential) so the returned peak
/// is the high-water mark reached *during this call*, not since process
/// start.
fn count_allocations<T>(f: impl FnOnce() -> T) -> (usize, usize, usize) {
    let bytes_before = BYTES_ALLOCATED.load(Ordering::Relaxed);
    let calls_before = ALLOC_CALLS.load(Ordering::Relaxed);
    let current_before = CURRENT_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(current_before, Ordering::Relaxed);
    black_box(f());
    let peak_during = PEAK_BYTES.load(Ordering::Relaxed);
    (
        BYTES_ALLOCATED.load(Ordering::Relaxed) - bytes_before,
        ALLOC_CALLS.load(Ordering::Relaxed) - calls_before,
        peak_during.saturating_sub(current_before),
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
    eprintln!("\nbin_search_first allocation report (single call, bytes / alloc count / peak):");
    for n in [100, 100_000] {
        let f = BinarySearchFirstFixture::new(n);
        let (bytes, calls, peak) = count_allocations(|| {
            binary_search_lt_first_core(f.left.view(), f.right.view(), f.left_index.view())
        });
        eprintln!("  lt_first n={n:>7}: {bytes:>9} bytes / {calls:>3} allocs / {peak:>9} peak");

        // A deliberately sparse case: every query is above `right`, so no
        // row survives the strict-less-than-first search. This guards the
        // memory claim that dropped rows do not trigger two full buffers.
        let sparse = BinarySearchFirstFixture {
            right: Array1::from_iter((0..n as i64).map(|i| i * 2)),
            left: Array1::from_elem(n, n as i64 * 2 + 1),
            left_index: Array1::from_iter(0..n as i64),
        };
        let (bytes, calls, peak) = count_allocations(|| {
            binary_search_lt_first_core(
                sparse.left.view(),
                sparse.right.view(),
                sparse.left_index.view(),
            )
        });
        eprintln!(
            "  lt_first sparse n={n:>7}: {bytes:>9} bytes / {calls:>3} allocs / {peak:>9} peak"
        );
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

/// A bench-only copy of `binary_search_lt_first_core` as it existed before
/// commit `68f5130` ("fix: avoid eager allocations for sparse first
/// searches"), i.e. with `Vec::with_capacity(left.len())` for both output
/// `Vec`s instead of the current `Vec::new()` grow-on-demand. Kept here
/// rather than in `src/` -- it exists purely so
/// `bench_bin_search_first_old_vs_new` can run both allocation strategies
/// back-to-back in the same process, on the same fixtures, in the same
/// run, which is a fairer comparison than timing two separate `cargo
/// bench` runs on two different commits. Every line other than the two
/// `Vec::with_capacity` calls is identical to the current
/// `binary_search_lt_first_core`, so allocation strategy is the only
/// variable being measured.
fn binary_search_lt_first_core_with_capacity(
    left: ArrayView1<i64>,
    right: ArrayView1<i64>,
    left_index: ArrayView1<i64>,
) -> (Vec<i64>, Vec<i64>) {
    let len_right = right.len();
    let mut search_indices = Vec::with_capacity(left.len());
    let mut index_left = Vec::with_capacity(left.len());
    for (pos, left_value) in left.into_iter().enumerate() {
        let mut min_idx = 0;
        let mut max_idx = len_right;
        while min_idx < max_idx {
            let mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            let current_value = right[mid_idx];
            if current_value <= *left_value {
                min_idx = mid_idx + 1;
            } else {
                max_idx = mid_idx;
            }
        }
        if min_idx == len_right {
            continue;
        }
        let current_value = right[min_idx];
        if current_value == *left_value {
            continue;
        }
        search_indices.push(min_idx as i64);
        index_left.push(left_index[pos]);
    }
    (search_indices, index_left)
}

/// Inputs for `bench_bin_search_first_old_vs_new`: like
/// `BinarySearchFirstFixture`, but only `survival_pct` percent of rows are
/// constructed to match (row `i` matches iff `i % 100 < survival_pct`,
/// interleaved rather than clustered so the survival rate is
/// representative of a mixed real column, not an artifact of row order).
/// A non-matching row is set to a value at or above `right`'s max element,
/// which `binary_search_lt_first_core` can never find anything greater
/// than.
struct SurvivalFixture {
    right: Array1<i64>,
    left: Array1<i64>,
    left_index: Array1<i64>,
}

impl SurvivalFixture {
    fn new(n: usize, survival_pct: usize) -> Self {
        let right = Array1::from_iter((0..n as i64).map(|i| i * 2));
        let no_match_value = n as i64 * 2;
        let left = Array1::from_iter((0..n as i64).map(|i| {
            if (i as usize) % 100 < survival_pct {
                i * 2 + 1
            } else {
                no_match_value
            }
        }));
        let left_index = Array1::from_iter(0..n as i64);
        SurvivalFixture {
            right,
            left,
            left_index,
        }
    }
}

/// Direct old-(`Vec::with_capacity`)-vs-new-(`Vec::new`) comparison for
/// `binary_search_lt_first_core`, across a spread of survival rates. The
/// PR #46 description's allocation numbers were measured before the
/// grow-on-demand fix and describe the 100%-survival case only; this
/// fills the gap flagged in review -- no evidence existed comparing the
/// two allocation strategies at partial survival, where `Vec::new()`
/// pays for reallocations that `Vec::with_capacity` avoided, in exchange
/// for not over-allocating on the sparse end.
fn bench_bin_search_first_old_vs_new(c: &mut Criterion) {
    eprintln!("\nbin_search_first old (Vec::with_capacity) vs new (Vec::new) allocation report:");
    let mut group = c.benchmark_group("bin_search_first_old_vs_new");
    // Reduced from criterion's defaults (100 samples / 5s measurement) --
    // this group times 16 combinations (2 impls x 4 survival rates x 2
    // sizes), and the eprintln allocation/peak report above is the
    // primary signal here, not a tight confidence interval on wall time.
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_millis(500));
    for n in [100, 100_000] {
        for survival_pct in [0, 10, 50, 100] {
            let f = SurvivalFixture::new(n, survival_pct);

            let (bytes, calls, peak) = count_allocations(|| {
                binary_search_lt_first_core_with_capacity(
                    f.left.view(),
                    f.right.view(),
                    f.left_index.view(),
                )
            });
            eprintln!(
                "  old  n={n:>7} survival={survival_pct:>3}%: {bytes:>9} bytes / {calls:>3} allocs / {peak:>9} peak"
            );
            group.bench_function(format!("old n={n} survival={survival_pct}%"), |b| {
                b.iter(|| {
                    binary_search_lt_first_core_with_capacity(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.left_index.view()),
                    )
                })
            });

            let (bytes, calls, peak) = count_allocations(|| {
                binary_search_lt_first_core(f.left.view(), f.right.view(), f.left_index.view())
            });
            eprintln!(
                "  new  n={n:>7} survival={survival_pct:>3}%: {bytes:>9} bytes / {calls:>3} allocs / {peak:>9} peak"
            );
            group.bench_function(format!("new n={n} survival={survival_pct}%"), |b| {
                b.iter(|| {
                    binary_search_lt_first_core(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.left_index.view()),
                    )
                })
            });
        }
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
                    black_box(CompareOp::Gt),
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

/// Inputs shared by both `min_*_core` and `max_*_core`: identical shape to
/// `SumFixture`'s start/end/start_end fixtures (bounded interval width per
/// AGENTS.md's aggregation-benchmark guidance), just reused across both
/// families since, unlike `sum`, neither needs a per-dtype cast variant --
/// see issue #49's "why cheap" note.
struct MinMaxFixture {
    arr: Array1<i64>,
    booleans: Array1<bool>,
    starts_for_start: Array1<i64>,
    ends_for_end: Array1<i64>,
    sliding_starts: Array1<i64>,
    sliding_ends: Array1<i64>,
}

impl MinMaxFixture {
    fn new(n: usize) -> Self {
        let n64 = n as i64;
        let arr = Array1::from_iter((0..n64).map(|i| i * 2));
        let booleans = Array1::from_elem(n, false);

        // Same reasoning as SumFixture: min_start/max_start scan
        // arr[start..] (to the very end), so every row's start must sit
        // near the end for the width to stay bounded.
        let starts_for_start = Array1::from_elem(n, (n64 - SUM_BENCH_WIDTH).max(0));
        // min_end/max_end scan arr[..end] (from the very start), the
        // mirror image.
        let ends_for_end = Array1::from_elem(n, SUM_BENCH_WIDTH.min(n64));
        // min_start_end/max_start_end take an explicit [start, end) per
        // row, so a sliding bounded window covering the whole array is
        // realistic and still stays O(n * WIDTH).
        let sliding_starts = Array1::from_iter(0..n64);
        let sliding_ends = Array1::from_iter((0..n64).map(|i| (i + SUM_BENCH_WIDTH).min(n64)));

        MinMaxFixture {
            arr,
            booleans,
            starts_for_start,
            ends_for_end,
            sliding_starts,
            sliding_ends,
        }
    }
}

/// ELI5: see `bench_sum_kernels`'s doc comment -- this is the same shape
/// of benchmark, just for the "find the biggest/smallest" door instead of
/// the "add these up" door (issue #49).
fn bench_min_max_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_max_kernels");
    for n in [100, 100_000] {
        let f = MinMaxFixture::new(n);

        group.bench_function(format!("min_start n={n}"), |b| {
            b.iter(|| {
                min_start_core(
                    black_box(f.arr.view()),
                    black_box(f.starts_for_start.view()),
                    black_box(f.booleans.view()),
                )
            })
        });
        group.bench_function(format!("min_end n={n}"), |b| {
            b.iter(|| {
                min_end_core(
                    black_box(f.arr.view()),
                    black_box(f.ends_for_end.view()),
                    black_box(f.booleans.view()),
                )
            })
        });
        group.bench_function(format!("min_start_end n={n}"), |b| {
            b.iter(|| {
                min_start_end_core(
                    black_box(f.arr.view()),
                    black_box(f.sliding_starts.view()),
                    black_box(f.sliding_ends.view()),
                    black_box(f.booleans.view()),
                )
            })
        });

        group.bench_function(format!("max_start n={n}"), |b| {
            b.iter(|| {
                max_start_core(
                    black_box(f.arr.view()),
                    black_box(f.starts_for_start.view()),
                    black_box(f.booleans.view()),
                )
            })
        });
        group.bench_function(format!("max_end n={n}"), |b| {
            b.iter(|| {
                max_end_core(
                    black_box(f.arr.view()),
                    black_box(f.ends_for_end.view()),
                    black_box(f.booleans.view()),
                )
            })
        });
        group.bench_function(format!("max_start_end n={n}"), |b| {
            b.iter(|| {
                max_start_end_core(
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
    bench_bin_search_first,
    bench_bin_search_first_old_vs_new,
    bench_compare_start_end,
    bench_index_builders,
    bench_sum_kernels,
    bench_min_max_kernels
);
criterion_main!(benches);
