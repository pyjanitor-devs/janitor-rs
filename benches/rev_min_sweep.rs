//! Old-vs-new benchmark for reverse min starts/ends sweeps.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{min_rev_ends_core, min_rev_starts_core};
use numpy::ndarray::{Array1, ArrayView1};
use std::hint::black_box;

mod support;
use support::count_allocations;

fn old_starts(
    arr: ArrayView1<'_, i64>,
    starts: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> (Vec<i64>, Vec<i64>) {
    assert_eq!(arr.len(), starts.len());
    assert_eq!(arr.len(), booleans.len());
    assert!(!arr.is_empty() && !index.is_empty());
    let min_start = starts.iter().copied().min().unwrap() as usize;
    let width = index.len() - min_start;
    let mut values = vec![arr[0]; width];
    let mut positions = vec![-1_i64; width];
    for (row, ((current, start), boolean)) in arr
        .iter()
        .zip(starts.iter())
        .zip(booleans.iter())
        .enumerate()
    {
        for (position, value) in positions
            .iter_mut()
            .zip(values.iter_mut())
            .skip(*start as usize - min_start)
        {
            if !*boolean && (*position == -1 || *current < *value) {
                *position = row as i64;
                *value = *current;
            }
        }
    }
    let indexers = (min_start..index.len()).map(|item| index[item]).collect();
    (indexers, positions)
}

fn old_ends(
    arr: ArrayView1<'_, i64>,
    ends: ArrayView1<'_, i64>,
    index: ArrayView1<'_, i64>,
    booleans: ArrayView1<'_, bool>,
) -> (Vec<i64>, Vec<i64>) {
    assert_eq!(arr.len(), ends.len());
    assert_eq!(arr.len(), booleans.len());
    assert!(!arr.is_empty() && !index.is_empty());
    let max_end = ends.iter().copied().max().unwrap() as usize;
    let mut values = vec![arr[0]; max_end];
    let mut positions = vec![-1_i64; max_end];
    for (row, ((current, end), boolean)) in
        arr.iter().zip(ends.iter()).zip(booleans.iter()).enumerate()
    {
        for (position, value) in positions
            .iter_mut()
            .zip(values.iter_mut())
            .take(*end as usize)
        {
            if !*boolean && (*position == -1 || *current < *value) {
                *position = row as i64;
                *value = *current;
            }
        }
    }
    let indexers = (0..max_end).map(|item| index[item]).collect();
    (indexers, positions)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_min_sweep_old_vs_new");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(2));
    for (name, rows, right_len) in [
        ("tiny_dense", 32_usize, 32_usize),
        ("large_dense", 1_000, 10_000),
        ("very_large_dense", 10_000, 100_000),
        ("super_large_narrow", 1_000_000, 1_000_000),
    ] {
        let arr = Array1::from_iter((0..rows).map(|row| (row % 97) as i64));
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
        let booleans = Array1::from_elem(rows, false);
        assert_eq!(
            old_starts(arr.view(), starts.view(), index.view(), booleans.view()),
            min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view()).unwrap()
        );
        assert_eq!(
            old_ends(arr.view(), ends.view(), index.view(), booleans.view()),
            min_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view()).unwrap()
        );
        let old_start_memory = count_allocations(|| {
            old_starts(arr.view(), starts.view(), index.view(), booleans.view())
        });
        let new_start_memory = count_allocations(|| {
            min_rev_starts_core(arr.view(), starts.view(), index.view(), booleans.view())
        });
        let old_end_memory =
            count_allocations(|| old_ends(arr.view(), ends.view(), index.view(), booleans.view()));
        let new_end_memory = count_allocations(|| {
            min_rev_ends_core(arr.view(), ends.view(), index.view(), booleans.view())
        });
        eprintln!(
            "{name}: starts old {}B/{}B new {}B/{}B; ends old {}B/{}B new {}B/{}B",
            old_start_memory.0,
            old_start_memory.2,
            new_start_memory.0,
            new_start_memory.2,
            old_end_memory.0,
            old_end_memory.2,
            new_end_memory.0,
            new_end_memory.2
        );
        group.bench_function(format!("starts/old/{name}"), |b| {
            b.iter(|| {
                old_starts(
                    black_box(arr.view()),
                    black_box(starts.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("starts/adaptive/{name}"), |b| {
            b.iter(|| {
                min_rev_starts_core(
                    black_box(arr.view()),
                    black_box(starts.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("ends/old/{name}"), |b| {
            b.iter(|| {
                old_ends(
                    black_box(arr.view()),
                    black_box(ends.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("ends/adaptive/{name}"), |b| {
            b.iter(|| {
                min_rev_ends_core(
                    black_box(arr.view()),
                    black_box(ends.view()),
                    black_box(index.view()),
                    black_box(booleans.view()),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
