//! Old HashMap versus dense suffix-domain state for reverse starts+matches min.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

fn old_min(arr: &[i64], starts: &[usize], index: &[i64], matches: &[i8]) -> Vec<(i64, i64)> {
    let mut values = HashMap::with_capacity(index.len());
    let mut n = 0;
    for (row, (current, &start)) in arr.iter().zip(starts).enumerate() {
        for item in start..index.len() {
            if matches[n] != 0 {
                let entry = values.entry(index[item]).or_insert((-1_i64, *current));
                if entry.0 == -1 || *current < entry.1 {
                    *entry = (row as i64, *current);
                }
            }
            n += 1;
        }
    }
    values
        .into_iter()
        .map(|(label, (row, _))| (label, row))
        .collect()
}

fn dense_min(arr: &[i64], starts: &[usize], index: &[i64], matches: &[i8]) -> Vec<(i64, i64)> {
    let min_start = starts.iter().copied().min().unwrap_or(index.len());
    let width = index.len() - min_start;
    let mut seen = vec![false; width];
    let mut touched = Vec::new();
    let mut rows = vec![-1_i64; width];
    let mut values = vec![0_i64; width];
    let mut n = 0;
    for (row, (current, &start)) in arr.iter().zip(starts).enumerate() {
        for item in start..index.len() {
            if matches[n] != 0 {
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                if rows[slot] == -1 || *current < values[slot] {
                    values[slot] = *current;
                    rows[slot] = row as i64;
                }
            }
            n += 1;
        }
    }
    touched
        .into_iter()
        .map(|slot| (index[slot + min_start], rows[slot]))
        .collect()
}

struct Fixture {
    arr: Vec<i64>,
    starts: Vec<usize>,
    index: Vec<i64>,
    matches: Vec<i8>,
}

impl Fixture {
    fn new(rows: usize, right_len: usize, width: usize) -> Self {
        Self {
            arr: (0..rows).map(|i| (rows - i) as i64).collect(),
            starts: vec![right_len - width; rows],
            index: (0..right_len).map(|i| 10_000 + i as i64 * 7).collect(),
            matches: (0..rows * width)
                .map(|i| if i % 4 == 0 { 1 } else { 0 })
                .collect(),
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_rev_starts_matches_old_vs_dense");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_secs(1));
    for (name, rows, right_len, width) in [
        ("tiny", 32, 1_024, 8),
        ("large", 100_000, 1_000_000, 8),
        ("very_large", 1_000_000, 2_000_000, 8),
        ("super_large", 2_000_000, 4_000_000, 8),
        ("wide_suffix", 1_000, 10_000, 10_000),
    ] {
        let fixture = Fixture::new(rows, right_len, width);
        group.bench_with_input(BenchmarkId::new("old_hashmap", name), &fixture, |b, f| {
            b.iter(|| black_box(old_min(&f.arr, &f.starts, &f.index, &f.matches)))
        });
        group.bench_with_input(BenchmarkId::new("dense_ordinal", name), &fixture, |b, f| {
            b.iter(|| black_box(dense_min(&f.arr, &f.starts, &f.index, &f.matches)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
