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
use std::hint::black_box;

mod support;
// ELI5: every benchmark executable uses the same scorekeeper for allocations,
// so memory numbers mean the same thing across old and optimized kernels.
use support::count_allocations;

use janitor_rs::bench_support::{
    binary_search_ge_first_core, binary_search_gt_first_core, binary_search_le_first_core,
    binary_search_lt_core, binary_search_lt_first_core, compare_end_allocating_core,
    compare_end_in_place_core, compare_ne_end_allocating_core, compare_ne_end_in_place_core,
    compare_ne_start_allocating_core, compare_ne_start_end_allocating_core,
    compare_ne_start_end_in_place_core, compare_ne_start_in_place_core,
    compare_start_allocating_core, compare_start_end_core, compare_start_end_in_place_core,
    compare_start_in_place_core, min_positions_core, repeat_index_core, sum_end_core,
    sum_start_core, sum_start_end_core, sum_start_u32_core, trim_index_core, CompareOp,
};
use std::collections::HashMap;

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

/// Same logical content as `BinarySearchFixture`, but `right` is stored at
/// double width with a junk value interleaved after each real one, so a
/// stride-2 view over it is non-contiguous (`.as_slice()` returns `None`)
/// while still reading the identical `0, 2, 4, ...` sequence
/// `BinarySearchFixture` does -- letting `bench_bin_search_lt` compare the
/// `slice::partition_point` fast path against the `ArrayView1` manual-loop
/// fallback on otherwise-identical data.
struct StridedBinarySearchFixture {
    right_padded: Array1<i64>,
    left: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
}

impl StridedBinarySearchFixture {
    fn new(n: usize) -> Self {
        let right_padded = Array1::from_iter((0..n as i64).flat_map(|i| [i * 2, -1]));
        let left = Array1::from_iter((0..n as i64).map(|i| i * 2 + 1));
        let starts = Array1::from_iter(0..n as i64);
        let ends = Array1::from_elem(n, n as i64);
        StridedBinarySearchFixture {
            right_padded,
            left,
            starts,
            ends,
        }
    }

    fn right_view(&self) -> ArrayView1<'_, i64> {
        self.right_padded.slice(numpy::ndarray::s![..;2])
    }
}

/// ELI5: for every value in `left`, find where it would slot into the
/// sorted `right` array -- the building block behind a `<` join. Run
/// against both a contiguous `right` (the fast path) and a strided one
/// (the fallback loop), so a regression in either isn't hidden by only
/// ever benchmarking the other.
fn bench_bin_search_lt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bin_search_lt");
    for n in [100, 100_000] {
        let f = BinarySearchFixture::new(n);
        group.bench_function(format!("n={n} (contiguous)"), |b| {
            b.iter(|| {
                binary_search_lt_core(
                    black_box(f.left.view()),
                    black_box(f.right.view()),
                    black_box(f.starts.view()),
                    black_box(f.ends.view()),
                )
            })
        });

        let sf = StridedBinarySearchFixture::new(n);
        group.bench_function(format!("n={n} (strided)"), |b| {
            b.iter(|| {
                binary_search_lt_core(
                    black_box(sf.left.view()),
                    black_box(sf.right_view()),
                    black_box(sf.starts.view()),
                    black_box(sf.ends.view()),
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

/// A realistic flat-tape fixture: every left row examines the complete right
/// side, so the tape contains `left_len * right_len` candidate slots.
struct PairedCompareFixture {
    right: Array1<i64>,
    left: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
    matches: Array1<i8>,
}

impl PairedCompareFixture {
    fn new(left_len: usize, right_len: usize, survivor_stride: Option<usize>) -> Self {
        let tape_width = left_len * right_len;
        PairedCompareFixture {
            right: Array1::from_iter(0..right_len as i64),
            left: Array1::from_iter((0..left_len as i64).map(|i| i * 2)),
            starts: Array1::zeros(left_len),
            ends: Array1::from_elem(left_len, right_len as i64),
            matches: Array1::from_iter((0..tape_width).map(|position| {
                if survivor_stride.is_some_and(|stride| position % stride == 0) {
                    0
                } else {
                    1
                }
            })),
        }
    }
}

/// Paired comparison of one additional predicate. Both kernels receive the
/// same live mask and calculate the same counts; the allocating version
/// creates a replacement full-width tape, while the in-place version reuses
/// the supplied tape. `iter_batched` keeps mask setup outside the measured
/// closure so the comparison is about the kernel, not fixture preparation.
fn bench_compare_start_end_allocating_vs_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_start_end_allocating_vs_in_place");
    for (left_len, right_len) in [(500, 5_000), (1_000, 10_000)] {
        for (mask_name, survivor_stride) in [("dense", None), ("25pct_dead", Some(4))] {
            let f = PairedCompareFixture::new(left_len, right_len, survivor_stride);
            let label = format!("candidates={}/mask={mask_name}", f.matches.len());

            group.bench_function(format!("{label}/allocating"), |b| {
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

            group.bench_function(format!("{label}/in_place"), |b| {
                b.iter_batched(
                    || f.matches.clone(),
                    |mut matches| {
                        compare_start_end_in_place_core(
                            black_box(f.left.view()),
                            black_box(f.right.view()),
                            black_box(f.starts.view()),
                            black_box(f.ends.view()),
                            black_box(matches.view_mut()),
                            black_box(CompareOp::Gt),
                        )
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            let mut reusable_mask = f.matches.clone();
            let (alloc_bytes, alloc_calls, alloc_peak) = count_allocations(|| {
                compare_start_end_core(
                    f.left.view(),
                    f.right.view(),
                    f.starts.view(),
                    f.ends.view(),
                    f.matches.view(),
                    CompareOp::Gt,
                )
            });
            let (in_place_bytes, in_place_calls, in_place_peak) = count_allocations(|| {
                compare_start_end_in_place_core(
                    f.left.view(),
                    f.right.view(),
                    f.starts.view(),
                    f.ends.view(),
                    reusable_mask.view_mut(),
                    CompareOp::Gt,
                )
            });
            eprintln!(
                "\n{label}: allocating {alloc_bytes} bytes/{alloc_calls} allocs/{alloc_peak} peak; in-place {in_place_bytes} bytes/{in_place_calls} allocs/{in_place_peak} peak"
            );
        }
    }
    group.finish();
}

fn bench_compare_side_shapes_allocating_vs_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_side_shapes_allocating_vs_in_place");
    for (left_len, right_len) in [(500, 5_000), (1_000, 10_000)] {
        for (mask_name, survivor_stride) in [("dense", None), ("25pct_dead", Some(4))] {
            let f = PairedCompareFixture::new(left_len, right_len, survivor_stride);
            let counts = Array1::from_elem(left_len, right_len as i64);
            let label = format!("candidates={}/mask={mask_name}", f.matches.len());

            group.bench_function(format!("{label}/starts_only/allocating"), |b| {
                b.iter(|| {
                    compare_start_allocating_core(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.starts.view()),
                        black_box(counts.view()),
                        black_box(f.matches.view()),
                        black_box(CompareOp::Gt),
                    )
                })
            });
            group.bench_function(format!("{label}/starts_only/in_place"), |b| {
                b.iter_batched(
                    || f.matches.clone(),
                    |mut matches| {
                        compare_start_in_place_core(
                            black_box(f.left.view()),
                            black_box(f.right.view()),
                            black_box(f.starts.view()),
                            black_box(counts.view()),
                            black_box(matches.view_mut()),
                            black_box(CompareOp::Gt),
                        )
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            group.bench_function(format!("{label}/ends_only/allocating"), |b| {
                b.iter(|| {
                    compare_end_allocating_core(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.ends.view()),
                        black_box(counts.view()),
                        black_box(f.matches.view()),
                        black_box(CompareOp::Gt),
                    )
                })
            });
            group.bench_function(format!("{label}/ends_only/in_place"), |b| {
                b.iter_batched(
                    || f.matches.clone(),
                    |mut matches| {
                        compare_end_in_place_core(
                            black_box(f.left.view()),
                            black_box(f.right.view()),
                            black_box(f.ends.view()),
                            black_box(counts.view()),
                            black_box(matches.view_mut()),
                            black_box(CompareOp::Gt),
                        )
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            let (start_alloc_bytes, start_alloc_calls, _) = count_allocations(|| {
                compare_start_allocating_core(
                    f.left.view(),
                    f.right.view(),
                    f.starts.view(),
                    counts.view(),
                    f.matches.view(),
                    CompareOp::Gt,
                )
            });
            let mut start_mask = f.matches.clone();
            let (start_in_place_bytes, start_in_place_calls, _) = count_allocations(|| {
                compare_start_in_place_core(
                    f.left.view(),
                    f.right.view(),
                    f.starts.view(),
                    counts.view(),
                    start_mask.view_mut(),
                    CompareOp::Gt,
                )
            });
            let (end_alloc_bytes, end_alloc_calls, _) = count_allocations(|| {
                compare_end_allocating_core(
                    f.left.view(),
                    f.right.view(),
                    f.ends.view(),
                    counts.view(),
                    f.matches.view(),
                    CompareOp::Gt,
                )
            });
            let mut end_mask = f.matches.clone();
            let (end_in_place_bytes, end_in_place_calls, _) = count_allocations(|| {
                compare_end_in_place_core(
                    f.left.view(),
                    f.right.view(),
                    f.ends.view(),
                    counts.view(),
                    end_mask.view_mut(),
                    CompareOp::Gt,
                )
            });
            eprintln!(
                "\n{label}: starts allocating {start_alloc_bytes} bytes/{start_alloc_calls} allocs; starts in-place {start_in_place_bytes} bytes/{start_in_place_calls} allocs; ends allocating {end_alloc_bytes} bytes/{end_alloc_calls} allocs; ends in-place {end_in_place_bytes} bytes/{end_in_place_calls} allocs"
            );
        }
    }
    group.finish();
}

fn bench_compare_nullable_shapes_allocating_vs_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("compare_nullable_shapes_allocating_vs_in_place");
    for (left_len, right_len) in [(500, 5_000), (1_000, 10_000)] {
        for (mask_name, survivor_stride) in [("dense", None), ("25pct_dead", Some(4))] {
            let f = PairedCompareFixture::new(left_len, right_len, survivor_stride);
            let counts = Array1::from_elem(left_len, right_len as i64);
            let left_booleans = Array1::from_elem(left_len, false);
            let right_booleans =
                Array1::from_iter((0..right_len).map(|position| position % 17 == 0));
            let label = format!("candidates={}/mask={mask_name}", f.matches.len());

            group.bench_function(format!("{label}/starts_only/allocating"), |b| {
                b.iter(|| {
                    compare_ne_start_allocating_core(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.starts.view()),
                        black_box(counts.view()),
                        black_box(left_booleans.view()),
                        black_box(right_booleans.view()),
                        black_box(f.matches.view()),
                        true,
                        black_box(CompareOp::Ne),
                    )
                })
            });
            group.bench_function(format!("{label}/starts_only/in_place"), |b| {
                b.iter_batched(
                    || f.matches.clone(),
                    |mut matches| {
                        compare_ne_start_in_place_core(
                            black_box(f.left.view()),
                            black_box(f.right.view()),
                            black_box(f.starts.view()),
                            black_box(counts.view()),
                            black_box(left_booleans.view()),
                            black_box(right_booleans.view()),
                            black_box(matches.view_mut()),
                            true,
                            black_box(CompareOp::Ne),
                        )
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            group.bench_function(format!("{label}/ends_only/allocating"), |b| {
                b.iter(|| {
                    compare_ne_end_allocating_core(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.ends.view()),
                        black_box(counts.view()),
                        black_box(left_booleans.view()),
                        black_box(right_booleans.view()),
                        black_box(f.matches.view()),
                        true,
                        black_box(CompareOp::Ne),
                    )
                })
            });
            group.bench_function(format!("{label}/ends_only/in_place"), |b| {
                b.iter_batched(
                    || f.matches.clone(),
                    |mut matches| {
                        compare_ne_end_in_place_core(
                            black_box(f.left.view()),
                            black_box(f.right.view()),
                            black_box(f.ends.view()),
                            black_box(counts.view()),
                            black_box(left_booleans.view()),
                            black_box(right_booleans.view()),
                            black_box(matches.view_mut()),
                            true,
                            black_box(CompareOp::Ne),
                        )
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            group.bench_function(format!("{label}/starts_ends/allocating"), |b| {
                b.iter(|| {
                    compare_ne_start_end_allocating_core(
                        black_box(f.left.view()),
                        black_box(f.right.view()),
                        black_box(f.starts.view()),
                        black_box(f.ends.view()),
                        black_box(left_booleans.view()),
                        black_box(right_booleans.view()),
                        black_box(f.matches.view()),
                        true,
                        black_box(CompareOp::Ne),
                    )
                })
            });
            group.bench_function(format!("{label}/starts_ends/in_place"), |b| {
                b.iter_batched(
                    || f.matches.clone(),
                    |mut matches| {
                        compare_ne_start_end_in_place_core(
                            black_box(f.left.view()),
                            black_box(f.right.view()),
                            black_box(f.starts.view()),
                            black_box(f.ends.view()),
                            black_box(left_booleans.view()),
                            black_box(right_booleans.view()),
                            black_box(matches.view_mut()),
                            true,
                            black_box(CompareOp::Ne),
                        )
                    },
                    criterion::BatchSize::SmallInput,
                )
            });

            let (alloc_bytes, alloc_calls, _) = count_allocations(|| {
                compare_ne_start_end_allocating_core(
                    f.left.view(),
                    f.right.view(),
                    f.starts.view(),
                    f.ends.view(),
                    left_booleans.view(),
                    right_booleans.view(),
                    f.matches.view(),
                    true,
                    CompareOp::Ne,
                )
            });
            let mut mask = f.matches.clone();
            let (in_place_bytes, in_place_calls, _) = count_allocations(|| {
                compare_ne_start_end_in_place_core(
                    f.left.view(),
                    f.right.view(),
                    f.starts.view(),
                    f.ends.view(),
                    left_booleans.view(),
                    right_booleans.view(),
                    mask.view_mut(),
                    true,
                    CompareOp::Ne,
                )
            });
            eprintln!(
                "\n{label}: nullable starts+ends allocating {alloc_bytes} bytes/{alloc_calls} allocs; in-place {in_place_bytes} bytes/{in_place_calls} allocs"
            );
        }
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

struct MinPositionsFixture {
    arr: Array1<i64>,
    starts: Array1<i64>,
    ends: Array1<i64>,
    index: Array1<i64>,
    positions: Array1<i64>,
    booleans: Array1<bool>,
}

impl MinPositionsFixture {
    fn new(n: usize, duplicate: bool) -> Self {
        const WIDTH: usize = 8;
        let arr = Array1::from_iter((0..n).map(|i| (n - i) as i64));
        let starts = Array1::from_iter((0..n).map(|i| (i * WIDTH) as i64));
        let ends = Array1::from_iter((0..n).map(|i| ((i + 1) * WIDTH) as i64));
        let index = Array1::from_iter(0..n as i64);
        let positions = Array1::from_iter((0..n).flat_map(|i| {
            (0..WIDTH).map(move |offset| {
                if duplicate {
                    0
                } else {
                    ((i + offset) % n) as i64
                }
            })
        }));
        let booleans = Array1::from_elem(n, false);
        Self {
            arr,
            starts,
            ends,
            index,
            positions,
            booleans,
        }
    }
}

fn min_positions_old(
    arr: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    positions: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
    capacity: usize,
) -> (Vec<i64>, Vec<i64>) {
    let mut dictionary: HashMap<i64, (i64, i64)> =
        HashMap::with_capacity(capacity.min(index.len()).min(positions.len()));
    for (posn, (((current, start), end), boolean)) in arr
        .iter()
        .zip(starts.iter())
        .zip(ends.iter())
        .zip(booleans.iter())
        .enumerate()
    {
        let Some(start_) = usize::try_from(*start).ok() else {
            continue;
        };
        let Some(end_) = usize::try_from(*end)
            .ok()
            .filter(|&end| end <= positions.len())
        else {
            continue;
        };
        if start_ >= end_ {
            continue;
        }
        for nn in start_..end_ {
            let Some(indexer_) = usize::try_from(positions[nn])
                .ok()
                .filter(|&i| i < index.len())
            else {
                continue;
            };
            let entry = dictionary.entry(index[indexer_]).or_insert((-1, *current));
            if !*boolean && (entry.0 == -1 || *current < entry.1) {
                *entry = (posn as i64, *current);
            }
        }
    }
    dictionary
        .into_iter()
        .map(|(label, (row, _))| (label, row))
        .unzip()
}

fn bench_min_positions(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_positions_old_vs_compact");
    for n in [32, 10_000, 100_000, 1_000_000] {
        for duplicate in [true, false] {
            let f = MinPositionsFixture::new(n, duplicate);
            let kind = if duplicate { "duplicate" } else { "unique" };
            let label = format!("n={n} {kind}");
            let old = count_allocations(|| {
                min_positions_old(
                    f.arr.view(),
                    f.starts.view(),
                    f.ends.view(),
                    f.index.view(),
                    f.positions.view(),
                    f.booleans.view(),
                    f.index.len(),
                )
            });
            let compact = count_allocations(|| {
                min_positions_core(
                    f.arr.view(),
                    f.starts.view(),
                    f.ends.view(),
                    f.index.view(),
                    f.positions.view(),
                    f.booleans.view(),
                )
                .expect("benchmark inputs satisfy positions validation")
            });
            eprintln!("min_positions {label}: old {old:?}; compact {compact:?}");
            group.bench_function(format!("old {label}"), |b| {
                b.iter(|| {
                    min_positions_old(
                        black_box(f.arr.view()),
                        black_box(f.starts.view()),
                        black_box(f.ends.view()),
                        black_box(f.index.view()),
                        black_box(f.positions.view()),
                        black_box(f.booleans.view()),
                        f.index.len(),
                    )
                })
            });
            group.bench_function(format!("compact {label}"), |b| {
                b.iter(|| {
                    min_positions_core(
                        black_box(f.arr.view()),
                        black_box(f.starts.view()),
                        black_box(f.ends.view()),
                        black_box(f.index.view()),
                        black_box(f.positions.view()),
                        black_box(f.booleans.view()),
                    )
                    .expect("benchmark inputs satisfy positions validation")
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_bin_search_lt,
    bench_bin_search_first,
    bench_bin_search_first_old_vs_new,
    bench_compare_start_end,
    bench_compare_start_end_allocating_vs_in_place,
    bench_compare_side_shapes_allocating_vs_in_place,
    bench_compare_nullable_shapes_allocating_vs_in_place,
    bench_index_builders,
    bench_sum_kernels,
    bench_min_positions
);
criterion_main!(benches);
