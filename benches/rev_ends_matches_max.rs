//! Old HashMap versus dense ordinal state for reverse ends+matches maximum.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

fn old_max(arr: &[i64], index: &[i64], ends: &[usize], matches: &[i8]) -> Vec<(i64, i64)> {
    let mut values = HashMap::with_capacity(index.len());
    let mut n = 0;
    for (row, (current, &end)) in arr.iter().zip(ends).enumerate() {
        for item in 0..end {
            if matches[n] != 0 {
                let entry = values.entry(index[item]).or_insert((-1_i64, *current));
                if entry.0 == -1 || *current > entry.1 {
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

fn dense_max(arr: &[i64], index: &[i64], ends: &[usize], matches: &[i8]) -> Vec<(i64, i64)> {
    let max_end = ends.iter().copied().max().unwrap_or(0);
    let mut seen = vec![false; max_end];
    let mut touched = Vec::new();
    let mut rows = vec![-1_i64; max_end];
    let mut values = vec![0_i64; max_end];
    let mut n = 0;
    for (row, (current, &end)) in arr.iter().zip(ends).enumerate() {
        for item in 0..end {
            if matches[n] != 0 {
                if !seen[item] {
                    seen[item] = true;
                    touched.push(item);
                }
                if rows[item] == -1 || *current > values[item] {
                    values[item] = *current;
                    rows[item] = row as i64;
                }
            }
            n += 1;
        }
    }
    touched
        .into_iter()
        .map(|item| (index[item], rows[item]))
        .collect()
}

struct Fixture {
    arr: Vec<i64>,
    index: Vec<i64>,
    ends: Vec<usize>,
    matches: Vec<i8>,
}

impl Fixture {
    fn new(rows: usize, right_len: usize, width: usize) -> Self {
        Self {
            arr: (0..rows).map(|i| (i % 97) as i64).collect(),
            index: (0..right_len).map(|i| 10_000 + i as i64 * 7).collect(),
            ends: vec![width; rows],
            matches: (0..rows * width)
                .map(|i| if i % 4 == 0 { 1 } else { 0 })
                .collect(),
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_rev_ends_matches_old_vs_dense");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_secs(1));
    for (name, rows, right_len, width) in [
        ("tiny", 32, 1_024, 8),
        ("large", 100_000, 1_000_000, 8),
        ("very_large", 1_000_000, 2_000_000, 8),
        ("super_large", 2_000_000, 4_000_000, 8),
        ("wide_prefix", 1_000, 10_000, 10_000),
    ] {
        let fixture = Fixture::new(rows, right_len, width);
        group.bench_with_input(BenchmarkId::new("old_hashmap", name), &fixture, |b, f| {
            b.iter(|| black_box(old_max(&f.arr, &f.index, &f.ends, &f.matches)))
        });
        group.bench_with_input(BenchmarkId::new("dense_ordinal", name), &fixture, |b, f| {
            b.iter(|| black_box(dense_max(&f.arr, &f.index, &f.ends, &f.matches)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
