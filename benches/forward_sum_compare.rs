use criterion::{criterion_group, criterion_main, Criterion};
use numpy::ndarray::Array1;
use std::hint::black_box;

use janitor_rs::bench_support::{sum_end_core, sum_start_core, sum_start_end_core};

fn old_sum_start(arr: &Array1<i64>, starts: &Array1<i64>, booleans: &Array1<bool>) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(starts.len());
    for (pos, start) in starts.iter().enumerate() {
        let mut total = 0_i64;
        for nn in (*start as usize)..arr.len() {
            if !booleans[nn] {
                total = total.wrapping_add(arr[nn]);
            }
        }
        result[pos] = total;
    }
    result
}

fn old_sum_end(arr: &Array1<i64>, ends: &Array1<i64>, booleans: &Array1<bool>) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(ends.len());
    for (pos, end) in ends.iter().enumerate() {
        let mut total = 0_i64;
        for nn in 0..*end as usize {
            if !booleans[nn] {
                total = total.wrapping_add(arr[nn]);
            }
        }
        result[pos] = total;
    }
    result
}

fn old_sum_start_end(
    arr: &Array1<i64>,
    starts: &Array1<i64>,
    ends: &Array1<i64>,
    booleans: &Array1<bool>,
) -> Array1<i64> {
    let mut result = Array1::<i64>::zeros(starts.len());
    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let mut total = 0_i64;
        for nn in *start as usize..*end as usize {
            if !booleans[nn] {
                total = total.wrapping_add(arr[nn]);
            }
        }
        result[pos] = total;
    }
    result
}

fn bench_forward_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_sum_origin_main_vs_adaptive");
    let n = 1_000_000;
    let queries = 1_000;
    let arr = Array1::from_iter(0..n as i64);
    let booleans = Array1::from_elem(n, false);

    for width in [1_i64, 1_000, 3_000, 4_000, 10_000] {
        let starts = Array1::from_elem(queries, n as i64 - width);
        let ends = Array1::from_elem(queries, width);
        let range_starts = Array1::from_elem(queries, n as i64 - width);
        let range_ends = Array1::from_elem(queries, n as i64);
        group.bench_function(format!("old_direct width={width}"), |b| {
            b.iter(|| old_sum_start(black_box(&arr), black_box(&starts), black_box(&booleans)))
        });
        group.bench_function(format!("adaptive_rust width={width}"), |b| {
            b.iter(|| {
                sum_start_core(
                    black_box(arr.view()),
                    black_box(starts.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("old_direct_end width={width}"), |b| {
            b.iter(|| old_sum_end(black_box(&arr), black_box(&ends), black_box(&booleans)))
        });
        group.bench_function(format!("adaptive_rust_end width={width}"), |b| {
            b.iter(|| {
                sum_end_core(
                    black_box(arr.view()),
                    black_box(ends.view()),
                    black_box(booleans.view()),
                )
            })
        });
        group.bench_function(format!("old_direct_start_end width={width}"), |b| {
            b.iter(|| {
                old_sum_start_end(
                    black_box(&arr),
                    black_box(&range_starts),
                    black_box(&range_ends),
                    black_box(&booleans),
                )
            })
        });
        group.bench_function(format!("adaptive_rust_start_end width={width}"), |b| {
            b.iter(|| {
                sum_start_end_core(
                    black_box(arr.view()),
                    black_box(range_starts.view()),
                    black_box(range_ends.view()),
                    black_box(booleans.view()),
                )
            })
        });
    }

    let n = 20_000;
    let queries = 20_000;
    let arr = Array1::from_elem(n, 1_i64);
    let starts = Array1::from_elem(queries, 0_i64);
    let booleans = Array1::from_elem(n, false);
    group.bench_function("dense_20k_x_20k/old_direct", |b| {
        b.iter(|| old_sum_start(black_box(&arr), black_box(&starts), black_box(&booleans)))
    });
    group.bench_function("dense_20k_x_20k/adaptive_rust", |b| {
        b.iter(|| {
            sum_start_core(
                black_box(arr.view()),
                black_box(starts.view()),
                black_box(booleans.view()),
            )
        })
    });
    group.finish();
}

criterion_group!(benches, bench_forward_sum);
criterion_main!(benches);
