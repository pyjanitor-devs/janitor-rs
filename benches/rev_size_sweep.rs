//! Old-vs-new benchmark for the reverse size starts/ends sweeps.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{size_rev_ends_core, size_rev_starts_core};
use numpy::ndarray::{Array1, ArrayView1};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        let live = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK.fetch_max(live, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocation_report<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    let before_allocated = ALLOCATED.load(Ordering::Relaxed);
    let before_current = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before_current, Ordering::Relaxed);
    black_box(f());
    (
        ALLOCATED.load(Ordering::Relaxed) - before_allocated,
        PEAK.load(Ordering::Relaxed).saturating_sub(before_current),
    )
}

fn old_starts(starts: ArrayView1<'_, i64>, right_len: usize) -> (Array1<i64>, Array1<i64>) {
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let mut result = vec![0_i64; right_len - min_start];
    for start in starts {
        for value in result.iter_mut().skip(*start as usize - min_start) {
            *value += 1;
        }
    }
    let indexers = (min_start..right_len).map(|item| item as i64).collect();
    (Array1::from_vec(indexers), Array1::from_vec(result))
}

fn old_ends(ends: ArrayView1<'_, i64>) -> (Array1<i64>, Array1<i64>) {
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut result = vec![0_i64; max_end];
    for end in ends {
        for value in result.iter_mut().take(*end as usize) {
            *value += 1;
        }
    }
    let indexers = (0..max_end).map(|item| item as i64).collect();
    (Array1::from_vec(indexers), Array1::from_vec(result))
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_size_sweep_old_vs_new");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));

    for (name, rows, right_len) in [
        ("tiny_dense", 32_usize, 32_usize),
        ("large_dense", 1_000, 10_000),
        ("very_large_dense", 10_000, 100_000),
        ("super_large_narrow", 1_000_000, 1_000_000),
    ] {
        let starts = if name.ends_with("narrow") {
            Array1::from_elem(rows, (right_len - 8) as i64)
        } else {
            Array1::zeros(rows)
        };
        let ends = if name.ends_with("narrow") {
            Array1::from_elem(rows, 8_i64)
        } else {
            Array1::from_elem(rows, right_len as i64)
        };
        let index = Array1::from_iter(0..right_len as i64);

        let old_start = old_starts(starts.view(), right_len);
        let new_start = size_rev_starts_core(starts.view(), index.view()).unwrap();
        assert_eq!(old_start, new_start, "starts mismatch for {name}");
        let old_end = old_ends(ends.view());
        let new_end = size_rev_ends_core(ends.view(), index.view()).unwrap();
        assert_eq!(old_end, new_end, "ends mismatch for {name}");

        let old_start_memory = allocation_report(|| old_starts(starts.view(), right_len));
        let new_start_memory =
            allocation_report(|| size_rev_starts_core(starts.view(), index.view()));
        let old_end_memory = allocation_report(|| old_ends(ends.view()));
        let new_end_memory = allocation_report(|| size_rev_ends_core(ends.view(), index.view()));
        eprintln!(
            "{name}: starts old {}B/{}B new {}B/{}B; ends old {}B/{}B new {}B/{}B",
            old_start_memory.0,
            old_start_memory.1,
            new_start_memory.0,
            new_start_memory.1,
            old_end_memory.0,
            old_end_memory.1,
            new_end_memory.0,
            new_end_memory.1
        );

        group.bench_function(format!("starts/old/{name}"), |b| {
            b.iter(|| old_starts(black_box(starts.view()), right_len))
        });
        group.bench_function(format!("starts/sweep/{name}"), |b| {
            b.iter(|| size_rev_starts_core(black_box(starts.view()), black_box(index.view())))
        });
        group.bench_function(format!("ends/old/{name}"), |b| {
            b.iter(|| old_ends(black_box(ends.view())))
        });
        group.bench_function(format!("ends/sweep/{name}"), |b| {
            b.iter(|| size_rev_ends_core(black_box(ends.view()), black_box(index.view())))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
