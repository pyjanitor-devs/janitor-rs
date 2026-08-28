//! Old label HashMap versus bounded dense slots for reverse starts/ends/matches min.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

struct Fixture {
    arr: Vec<i64>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    index: Vec<i64>,
    matches: Vec<i8>,
}

impl Fixture {
    fn new(rows: usize, right_len: usize, range_width: usize, broad: bool) -> Self {
        let mut starts = Vec::with_capacity(rows);
        let mut ends = Vec::with_capacity(rows);
        let mut matches = Vec::with_capacity(rows * range_width);
        for row in 0..rows {
            let start = if broad {
                (row * 97_003) % (right_len - range_width + 1)
            } else {
                row % (right_len - range_width + 1)
            };
            starts.push(start);
            ends.push(start + range_width);
            matches.extend(std::iter::repeat_n(1_i8, range_width));
        }
        Self {
            arr: (0..rows)
                .map(|row| (10_000 - (row % 10_000)) as i64)
                .collect(),
            starts,
            ends,
            index: (0..right_len).map(|i| 10_000 + i as i64 * 7).collect(),
            matches,
        }
    }
}

fn old_min(f: &Fixture) -> Vec<(i64, i64)> {
    let mut values: HashMap<i64, (i64, i64)> = HashMap::with_capacity(f.index.len());
    let mut tape = 0;
    for row in 0..f.arr.len() {
        for item in f.starts[row]..f.ends[row] {
            if f.matches[tape] != 0 {
                let entry = values.entry(f.index[item]).or_insert((i64::MAX, -1));
                if f.arr[row] < entry.0 {
                    *entry = (f.arr[row], row as i64);
                }
            }
            tape += 1;
        }
    }
    values
        .into_iter()
        .map(|(label, (_, row))| (label, row))
        .collect()
}

fn dense_min(f: &Fixture) -> Vec<(i64, i64)> {
    let min_start = f.starts.iter().copied().min().unwrap();
    let max_end = f.ends.iter().copied().max().unwrap();
    let width = max_end - min_start;
    let mut seen = vec![false; width];
    let mut touched = Vec::new();
    let mut best_values = vec![i64::MAX; width];
    let mut best_rows = vec![-1_i64; width];
    let mut tape = 0;
    for row in 0..f.arr.len() {
        for item in f.starts[row]..f.ends[row] {
            if f.matches[tape] != 0 {
                let slot = item - min_start;
                if !seen[slot] {
                    seen[slot] = true;
                    touched.push(slot);
                }
                if f.arr[row] < best_values[slot] {
                    best_values[slot] = f.arr[row];
                    best_rows[slot] = row as i64;
                }
            }
            tape += 1;
        }
    }
    touched
        .into_iter()
        .map(|slot| (f.index[min_start + slot], best_rows[slot]))
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_rev_starts_ends_matches_old_vs_dense");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_secs(1));
    for (name, rows, right_len) in [
        ("tiny", 32, 1_024),
        ("large", 100_000, 1_000_000),
        ("very_large", 1_000_000, 2_000_000),
        ("super_large", 2_000_000, 4_000_000),
    ] {
        for (kind, width, broad) in [("narrow", 8, false), ("broad", 64, true)] {
            let fixture = Fixture::new(rows, right_len, width, broad);
            group.bench_with_input(
                BenchmarkId::new(format!("old_hashmap_{kind}"), name),
                &fixture,
                |b, x| b.iter(|| black_box(old_min(x))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("dense_slice_{kind}"), name),
                &fixture,
                |b, x| b.iter(|| black_box(dense_min(x))),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
