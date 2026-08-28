//! Old label HashMap versus dense position slots for reverse positions min.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

fn old_min(arr: &[i64], index: &[i64], positions: &[i64]) -> Vec<(i64, i64)> {
    let mut values: HashMap<i64, (i64, i64)> =
        HashMap::with_capacity(index.len().min(positions.len()));
    for (row, &position) in positions.iter().enumerate() {
        let Ok(position) = usize::try_from(position) else {
            continue;
        };
        if position >= index.len() {
            continue;
        }
        let label = index[position];
        let entry = values.entry(label).or_insert((i64::MAX, -1));
        if arr[row] < entry.0 {
            *entry = (arr[row], row as i64);
        }
    }
    values
        .into_iter()
        .map(|(label, (_, row))| (label, row))
        .collect()
}

fn dense_min(arr: &[i64], index: &[i64], positions: &[i64]) -> Vec<(i64, i64)> {
    let width = positions
        .iter()
        .filter_map(|&position| usize::try_from(position).ok().filter(|&p| p < index.len()))
        .max()
        .map_or(0, |position| position + 1);
    let mut seen = vec![false; width];
    let mut touched = Vec::new();
    let mut best_values = vec![i64::MAX; width];
    let mut best_rows = vec![-1_i64; width];
    for (row, &position) in positions.iter().enumerate() {
        let Ok(position) = usize::try_from(position) else {
            continue;
        };
        if position >= index.len() {
            continue;
        }
        if !seen[position] {
            seen[position] = true;
            touched.push(position);
        }
        if arr[row] < best_values[position] {
            best_values[position] = arr[row];
            best_rows[position] = row as i64;
        }
    }
    touched
        .into_iter()
        .map(|position| (index[position], best_rows[position]))
        .collect()
}

struct Fixture {
    arr: Vec<i64>,
    index: Vec<i64>,
    positions: Vec<i64>,
}

impl Fixture {
    fn new(rows: usize, right_len: usize, width: usize, sparse: bool) -> Self {
        let limit = if sparse { 8 } else { right_len };
        let positions = (0..rows * width).map(|i| (i % limit) as i64).collect();
        Self {
            arr: (0..rows * width).map(|i| (i % 10_000) as i64).collect(),
            index: (0..right_len).map(|i| 10_000 + i as i64 * 7).collect(),
            positions,
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_rev_positions_old_vs_dense");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_secs(1));
    for (name, rows, right_len, width) in [
        ("tiny", 32, 1_024, 8),
        ("large", 100_000, 1_000_000, 8),
        ("very_large", 1_000_000, 2_000_000, 8),
        ("super_large", 2_000_000, 4_000_000, 8),
    ] {
        for sparse in [true, false] {
            let kind = if sparse { "sparse" } else { "broad" };
            let fixture = Fixture::new(rows, right_len, width, sparse);
            group.bench_with_input(
                BenchmarkId::new(format!("old_hashmap_{kind}"), name),
                &fixture,
                |b, x| b.iter(|| black_box(old_min(&x.arr, &x.index, &x.positions))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("dense_ordinal_{kind}"), name),
                &fixture,
                |b, x| b.iter(|| black_box(dense_min(&x.arr, &x.index, &x.positions))),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
