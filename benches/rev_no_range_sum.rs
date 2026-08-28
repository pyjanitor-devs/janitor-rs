//! Pre-sized HashMap versus growth-from-empty for reverse no-range sum.
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

struct Fixture {
    values: Vec<i64>,
    labels: Vec<i64>,
}
impl Fixture {
    fn new(n: usize, unique: bool) -> Self {
        Self {
            values: (0..n).map(|i| (i % 17 + 1) as i64).collect(),
            labels: (0..n)
                .map(|i| if unique { i as i64 } else { (i % 64) as i64 })
                .collect(),
        }
    }
}
fn sized(f: &Fixture) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(f.labels.len());
    for (&value, &label) in f.values.iter().zip(&f.labels) {
        let total = map.entry(label).or_insert(0_i64);
        *total = total.wrapping_add(value);
    }
    map.into_iter().collect()
}
fn growing(f: &Fixture) -> Vec<(i64, i64)> {
    let mut map = HashMap::new();
    for (&value, &label) in f.values.iter().zip(&f.labels) {
        let total = map.entry(label).or_insert(0_i64);
        *total = total.wrapping_add(value);
    }
    map.into_iter().collect()
}
fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_rev_no_range_capacity");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_secs(1));
    for (name, n) in [
        ("tiny", 32),
        ("large", 100_000),
        ("very_large", 1_000_000),
        ("super_large", 2_000_000),
    ] {
        for (unique, kind) in [(false, "duplicate_labels"), (true, "unique_labels")] {
            let f = Fixture::new(n, unique);
            group.bench_with_input(
                BenchmarkId::new("pre_sized", format!("{kind}/{name}")),
                &f,
                |b, x| b.iter(|| black_box(sized(x))),
            );
            group.bench_with_input(
                BenchmarkId::new("growing", format!("{kind}/{name}")),
                &f,
                |b, x| b.iter(|| black_box(growing(x))),
            );
        }
    }
    group.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
