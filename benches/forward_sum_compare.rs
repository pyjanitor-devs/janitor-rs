use criterion::{criterion_group, criterion_main, Criterion};
use numpy::ndarray::Array1;
use std::hint::black_box;

use janitor_rs::bench_support::{
    max_end_core, max_start_core, max_start_end_core, min_end_core, min_start_core,
    min_start_end_core, prod_end_core, prod_end_float_core, prod_start_core, prod_start_end_core,
    prod_start_float_core, sum_end_core, sum_end_float_core_with_cast, sum_start_core,
    sum_start_end_core, sum_start_float_core_with_cast,
};

fn old_min_start(arr: &Array1<i64>, starts: &Array1<i64>, mask: &Array1<bool>) -> Array1<i64> {
    let mut out = Array1::from_elem(starts.len(), -1);
    for (pos, start) in starts.iter().enumerate() {
        let start = *start as usize;
        let mut winner = -1;
        for nn in start..arr.len() {
            if !mask[nn] && (winner < 0 || arr[nn] < arr[winner as usize]) {
                winner = nn as i64;
            }
        }
        out[pos] = winner;
    }
    out
}

fn old_max_start(arr: &Array1<i64>, starts: &Array1<i64>, mask: &Array1<bool>) -> Array1<i64> {
    let mut out = Array1::from_elem(starts.len(), -1);
    for (pos, start) in starts.iter().enumerate() {
        let start = *start as usize;
        let mut winner = -1;
        for nn in start..arr.len() {
            if !mask[nn] && (winner < 0 || arr[nn] > arr[winner as usize]) {
                winner = nn as i64;
            }
        }
        out[pos] = winner;
    }
    out
}

fn old_min_end(arr: &Array1<i64>, ends: &Array1<i64>, mask: &Array1<bool>) -> Array1<i64> {
    let mut out = Array1::from_elem(ends.len(), -1);
    for (pos, end) in ends.iter().enumerate() {
        let mut winner = -1;
        for nn in 0..*end as usize {
            if !mask[nn] && (winner < 0 || arr[nn] < arr[winner as usize]) {
                winner = nn as i64;
            }
        }
        out[pos] = winner;
    }
    out
}

fn old_max_end(arr: &Array1<i64>, ends: &Array1<i64>, mask: &Array1<bool>) -> Array1<i64> {
    let mut out = Array1::from_elem(ends.len(), -1);
    for (pos, end) in ends.iter().enumerate() {
        let mut winner = -1;
        for nn in 0..*end as usize {
            if !mask[nn] && (winner < 0 || arr[nn] > arr[winner as usize]) {
                winner = nn as i64;
            }
        }
        out[pos] = winner;
    }
    out
}

fn old_min_start_end(
    arr: &Array1<i64>,
    starts: &Array1<i64>,
    ends: &Array1<i64>,
    mask: &Array1<bool>,
) -> Array1<i64> {
    let mut out = Array1::from_elem(starts.len(), -1);
    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let mut winner = -1;
        for nn in *start as usize..*end as usize {
            if !mask[nn] && (winner < 0 || arr[nn] < arr[winner as usize]) {
                winner = nn as i64;
            }
        }
        out[pos] = winner;
    }
    out
}

fn old_max_start_end(
    arr: &Array1<i64>,
    starts: &Array1<i64>,
    ends: &Array1<i64>,
    mask: &Array1<bool>,
) -> Array1<i64> {
    let mut out = Array1::from_elem(starts.len(), -1);
    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        let mut winner = -1;
        for nn in *start as usize..*end as usize {
            if !mask[nn] && (winner < 0 || arr[nn] > arr[winner as usize]) {
                winner = nn as i64;
            }
        }
        out[pos] = winner;
    }
    out
}

fn old_prod_start_end(
    arr: &Array1<i64>,
    starts: &Array1<i64>,
    ends: &Array1<i64>,
    mask: &Array1<bool>,
) -> Array1<i64> {
    let mut out = Array1::from_elem(starts.len(), 1_i64);
    for (pos, (start, end)) in starts.iter().zip(ends.iter()).enumerate() {
        for nn in *start as usize..*end as usize {
            if !mask[nn] {
                out[pos] = out[pos].wrapping_mul(arr[nn]);
            }
        }
    }
    out
}

fn old_prod_start(arr: &Array1<i64>, starts: &Array1<i64>, mask: &Array1<bool>) -> Array1<i64> {
    let mut out = Array1::zeros(starts.len());
    for (pos, start) in starts.iter().enumerate() {
        let mut total = 1_i64;
        for nn in *start as usize..arr.len() {
            if !mask[nn] {
                total = total.wrapping_mul(arr[nn]);
            }
        }
        out[pos] = total;
    }
    out
}

fn old_prod_end(arr: &Array1<i64>, ends: &Array1<i64>, mask: &Array1<bool>) -> Array1<i64> {
    let mut out = Array1::zeros(ends.len());
    for (pos, end) in ends.iter().enumerate() {
        let mut total = 1_i64;
        for nn in 0..*end as usize {
            if !mask[nn] {
                total = total.wrapping_mul(arr[nn]);
            }
        }
        out[pos] = total;
    }
    out
}

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
                .unwrap()
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
                .unwrap()
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
                .unwrap()
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
            .unwrap()
        })
    });
    group.finish();
}

fn bench_forward_all_aggregations(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_all_aggregations_old_vs_adaptive");
    group.sample_size(10);

    for (size_name, n) in [
        ("small", 1_000_usize),
        ("large", 100_000),
        ("very_large", 1_000_000),
    ] {
        let arr = Array1::from_iter((0..n).map(|value| (value % 97) as i64 + 1));
        let float_arr = arr.mapv(|value| value as f64);
        let mask = Array1::from_elem(n, false);
        for (shape, width) in [("narrow", 1_usize), ("broad", n / 2)] {
            let queries = 1_000.min(n.max(1));
            let start = (n - width) as i64;
            let starts = Array1::from_elem(queries, start);
            let ends = Array1::from_elem(queries, width as i64);
            let label = format!("{size_name}/{shape}");

            assert_eq!(
                old_sum_start(&arr, &starts, &mask),
                sum_start_core(arr.view(), starts.view(), mask.view()).unwrap()
            );
            group.bench_function(format!("sum/start/old/{label}"), |b| {
                b.iter(|| old_sum_start(black_box(&arr), black_box(&starts), black_box(&mask)))
            });
            group.bench_function(format!("sum/start/adaptive/{label}"), |b| {
                b.iter(|| {
                    sum_start_core(
                        black_box(arr.view()),
                        black_box(starts.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });
            assert_eq!(
                old_sum_end(&arr, &ends, &mask),
                sum_end_core(arr.view(), ends.view(), mask.view()).unwrap()
            );
            group.bench_function(format!("sum/end/old/{label}"), |b| {
                b.iter(|| old_sum_end(black_box(&arr), black_box(&ends), black_box(&mask)))
            });
            group.bench_function(format!("sum/end/adaptive/{label}"), |b| {
                b.iter(|| {
                    sum_end_core(
                        black_box(arr.view()),
                        black_box(ends.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });

            let range_starts = Array1::from_elem(queries, 0_i64);
            let range_ends = Array1::from_elem(queries, width as i64);
            assert_eq!(
                old_min_start_end(&arr, &range_starts, &range_ends, &mask),
                min_start_end_core(
                    arr.view(),
                    range_starts.view(),
                    range_ends.view(),
                    mask.view(),
                )
                .unwrap()
            );
            group.bench_function(format!("min/start_end/old/{label}"), |b| {
                b.iter(|| {
                    old_min_start_end(
                        black_box(&arr),
                        black_box(&range_starts),
                        black_box(&range_ends),
                        black_box(&mask),
                    )
                })
            });
            group.bench_function(format!("min/start_end/adaptive/{label}"), |b| {
                b.iter(|| {
                    min_start_end_core(
                        black_box(arr.view()),
                        black_box(range_starts.view()),
                        black_box(range_ends.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });

            assert_eq!(
                old_max_start_end(&arr, &range_starts, &range_ends, &mask),
                max_start_end_core(
                    arr.view(),
                    range_starts.view(),
                    range_ends.view(),
                    mask.view(),
                )
                .unwrap()
            );
            group.bench_function(format!("max/start_end/old/{label}"), |b| {
                b.iter(|| {
                    old_max_start_end(
                        black_box(&arr),
                        black_box(&range_starts),
                        black_box(&range_ends),
                        black_box(&mask),
                    )
                })
            });
            group.bench_function(format!("max/start_end/adaptive/{label}"), |b| {
                b.iter(|| {
                    max_start_end_core(
                        black_box(arr.view()),
                        black_box(range_starts.view()),
                        black_box(range_ends.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });

            assert_eq!(
                old_prod_start_end(&arr, &range_starts, &range_ends, &mask),
                prod_start_end_core(
                    arr.view(),
                    range_starts.view(),
                    range_ends.view(),
                    mask.view(),
                    |value| value,
                )
                .unwrap()
            );
            group.bench_function(format!("prod/start_end/old/{label}"), |b| {
                b.iter(|| {
                    old_prod_start_end(
                        black_box(&arr),
                        black_box(&range_starts),
                        black_box(&range_ends),
                        black_box(&mask),
                    )
                })
            });
            group.bench_function(format!("prod/start_end/adaptive/{label}"), |b| {
                b.iter(|| {
                    prod_start_end_core(
                        black_box(arr.view()),
                        black_box(range_starts.view()),
                        black_box(range_ends.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });

            assert_eq!(
                old_min_start(&arr, &starts, &mask),
                min_start_core(arr.view(), starts.view(), mask.view()).unwrap()
            );
            group.bench_function(format!("min/start/old/{label}"), |b| {
                b.iter(|| old_min_start(black_box(&arr), black_box(&starts), black_box(&mask)))
            });
            group.bench_function(format!("min/start/adaptive/{label}"), |b| {
                b.iter(|| {
                    min_start_core(
                        black_box(arr.view()),
                        black_box(starts.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });
            assert_eq!(
                old_min_end(&arr, &ends, &mask),
                min_end_core(arr.view(), ends.view(), mask.view()).unwrap()
            );
            group.bench_function(format!("min/end/old/{label}"), |b| {
                b.iter(|| old_min_end(black_box(&arr), black_box(&ends), black_box(&mask)))
            });
            group.bench_function(format!("min/end/adaptive/{label}"), |b| {
                b.iter(|| {
                    min_end_core(
                        black_box(arr.view()),
                        black_box(ends.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });

            assert_eq!(
                old_max_start(&arr, &starts, &mask),
                max_start_core(arr.view(), starts.view(), mask.view()).unwrap()
            );
            group.bench_function(format!("max/start/old/{label}"), |b| {
                b.iter(|| old_max_start(black_box(&arr), black_box(&starts), black_box(&mask)))
            });
            group.bench_function(format!("max/start/adaptive/{label}"), |b| {
                b.iter(|| {
                    max_start_core(
                        black_box(arr.view()),
                        black_box(starts.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });
            assert_eq!(
                old_max_end(&arr, &ends, &mask),
                max_end_core(arr.view(), ends.view(), mask.view()).unwrap()
            );
            group.bench_function(format!("max/end/old/{label}"), |b| {
                b.iter(|| old_max_end(black_box(&arr), black_box(&ends), black_box(&mask)))
            });
            group.bench_function(format!("max/end/adaptive/{label}"), |b| {
                b.iter(|| {
                    max_end_core(
                        black_box(arr.view()),
                        black_box(ends.view()),
                        black_box(mask.view()),
                    )
                    .unwrap()
                })
            });

            assert_eq!(
                old_prod_start(&arr, &starts, &mask),
                prod_start_core(arr.view(), starts.view(), mask.view(), |value| value).unwrap()
            );
            group.bench_function(format!("prod/start/old/{label}"), |b| {
                b.iter(|| old_prod_start(black_box(&arr), black_box(&starts), black_box(&mask)))
            });
            group.bench_function(format!("prod/start/adaptive/{label}"), |b| {
                b.iter(|| {
                    prod_start_core(
                        black_box(arr.view()),
                        black_box(starts.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });
            assert_eq!(
                old_prod_end(&arr, &ends, &mask),
                prod_end_core(arr.view(), ends.view(), mask.view(), |value| value).unwrap()
            );
            group.bench_function(format!("prod/end/old/{label}"), |b| {
                b.iter(|| old_prod_end(black_box(&arr), black_box(&ends), black_box(&mask)))
            });
            group.bench_function(format!("prod/end/adaptive/{label}"), |b| {
                b.iter(|| {
                    prod_end_core(
                        black_box(arr.view()),
                        black_box(ends.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });

            group.bench_function(format!("sum/start/float/{label}"), |b| {
                b.iter(|| {
                    sum_start_float_core_with_cast(
                        black_box(float_arr.view()),
                        black_box(starts.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });
            group.bench_function(format!("sum/end/float/{label}"), |b| {
                b.iter(|| {
                    sum_end_float_core_with_cast(
                        black_box(float_arr.view()),
                        black_box(ends.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });
            group.bench_function(format!("prod/start/float/{label}"), |b| {
                b.iter(|| {
                    prod_start_float_core(
                        black_box(float_arr.view()),
                        black_box(starts.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });
            group.bench_function(format!("prod/end/float/{label}"), |b| {
                b.iter(|| {
                    prod_end_float_core(
                        black_box(float_arr.view()),
                        black_box(ends.view()),
                        black_box(mask.view()),
                        |value| value,
                    )
                    .unwrap()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_forward_sum, bench_forward_all_aggregations);
criterion_main!(benches);
