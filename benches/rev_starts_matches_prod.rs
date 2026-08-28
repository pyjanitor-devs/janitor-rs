use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

fn old(a: &[i64], s: &[usize], ix: &[i64], m: &[i8]) -> Vec<(i64, i64)> {
    let mut h = HashMap::with_capacity(ix.len());
    let mut n = 0;
    for (v, &st) in a.iter().zip(s) {
        for i in st..ix.len() {
            if m[n] != 0 {
                let e = h.entry(ix[i]).or_insert(1_i64);
                *e = e.wrapping_mul(*v);
            }
            n += 1;
        }
    }
    h.into_iter().collect()
}

fn dense(a: &[i64], s: &[usize], ix: &[i64], m: &[i8]) -> Vec<(i64, i64)> {
    let min = s.iter().copied().min().unwrap_or(ix.len());
    let w = ix.len() - min;
    let mut seen = vec![false; w];
    let mut touched = Vec::new();
    let mut totals = vec![1_i64; w];
    let mut n = 0;
    for (v, &st) in a.iter().zip(s) {
        for i in st..ix.len() {
            if m[n] != 0 {
                let q = i - min;
                if !seen[q] {
                    seen[q] = true;
                    touched.push(q);
                }
                totals[q] = totals[q].wrapping_mul(*v);
            }
            n += 1;
        }
    }
    touched
        .into_iter()
        .map(|q| (ix[q + min], totals[q]))
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
            arr: (0..rows).map(|i| (i % 97 + 1) as i64).collect(),
            starts: vec![right_len - width; rows],
            index: (0..right_len).map(|i| 10_000 + i as i64 * 7).collect(),
            matches: (0..rows * width)
                .map(|i| if i % 4 == 0 { 1 } else { 0 })
                .collect(),
        }
    }
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("prod_rev_starts_matches_old_vs_dense");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(1));
    g.warm_up_time(Duration::from_secs(1));
    for (name, rows, right_len, width) in [
        ("tiny", 32, 1_024, 8),
        ("large", 100_000, 1_000_000, 8),
        ("very_large", 1_000_000, 2_000_000, 8),
        ("super_large", 2_000_000, 4_000_000, 8),
        ("wide_suffix", 1_000, 10_000, 10_000),
    ] {
        let f = Fixture::new(rows, right_len, width);
        g.bench_with_input(BenchmarkId::new("old_hashmap", name), &f, |b, x| {
            b.iter(|| black_box(old(&x.arr, &x.starts, &x.index, &x.matches)))
        });
        g.bench_with_input(BenchmarkId::new("dense_ordinal", name), &f, |b, x| {
            b.iter(|| black_box(dense(&x.arr, &x.starts, &x.index, &x.matches)))
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
