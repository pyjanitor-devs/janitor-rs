use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

fn old(a: &[i64], s: &[usize], ix: &[i64], m: &[i8]) -> Vec<(i64, i64)> {
    let mut h = HashMap::with_capacity(ix.len());
    let mut n = 0;
    for (r, (v, &st)) in a.iter().zip(s).enumerate() {
        for i in st..ix.len() {
            if m[n] != 0 {
                let e = h.entry(ix[i]).or_insert((-1, *v));
                if e.0 == -1 || *v > e.1 {
                    *e = (r as i64, *v);
                }
            }
            n += 1;
        }
    }
    h.into_iter().map(|(k, (r, _))| (k, r)).collect()
}
fn dense(a: &[i64], s: &[usize], ix: &[i64], m: &[i8]) -> Vec<(i64, i64)> {
    let ms = s.iter().copied().min().unwrap_or(ix.len());
    let w = ix.len() - ms;
    let mut seen = vec![false; w];
    let mut t = Vec::new();
    let mut rows = vec![-1; w];
    let mut vals = vec![0; w];
    let mut n = 0;
    for (r, (v, &st)) in a.iter().zip(s).enumerate() {
        for i in st..ix.len() {
            if m[n] != 0 {
                let q = i - ms;
                if !seen[q] {
                    seen[q] = true;
                    t.push(q);
                }
                if rows[q] == -1 || *v > vals[q] {
                    vals[q] = *v;
                    rows[q] = r as i64;
                }
            }
            n += 1;
        }
    }
    t.into_iter().map(|q| (ix[q + ms], rows[q])).collect()
}
struct F {
    a: Vec<i64>,
    s: Vec<usize>,
    ix: Vec<i64>,
    m: Vec<i8>,
}
impl F {
    fn new(r: usize, l: usize, w: usize) -> Self {
        Self {
            a: (0..r).map(|i| (i % 97) as i64).collect(),
            s: vec![l - w; r],
            ix: (0..l).map(|i| 10000 + i as i64 * 7).collect(),
            m: (0..r * w).map(|i| if i % 4 == 0 { 1 } else { 0 }).collect(),
        }
    }
}
fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("max_rev_starts_matches_old_vs_dense");
    g.sample_size(10);
    g.measurement_time(Duration::from_secs(1));
    g.warm_up_time(Duration::from_secs(1));
    for (n, r, l, w) in [
        ("tiny", 32, 1024, 8),
        ("large", 100000, 1000000, 8),
        ("very_large", 1000000, 2000000, 8),
        ("super_large", 2000000, 4000000, 8),
        ("wide_suffix", 1000, 10000, 10000),
    ] {
        let f = F::new(r, l, w);
        g.bench_with_input(BenchmarkId::new("old_hashmap", n), &f, |b, x| {
            b.iter(|| black_box(old(&x.a, &x.s, &x.ix, &x.m)))
        });
        g.bench_with_input(BenchmarkId::new("dense_ordinal", n), &f, |b, x| {
            b.iter(|| black_box(dense(&x.a, &x.s, &x.ix, &x.m)))
        });
    }
    g.finish()
}
criterion_group!(benches, bench);
criterion_main!(benches);
