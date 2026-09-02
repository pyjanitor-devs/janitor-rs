//! Algorithm-level experiments for reverse `_starts_ends`.
//!
//! These deliberately use the same `[start, end)` semantics as the production
//! kernels, but keep the alternatives local until their runtime/memory tradeoffs
//! justify a production rewrite.

use criterion::{criterion_group, criterion_main, Criterion};
use janitor_rs::bench_support::{
    max_rev_start_end_core, min_rev_start_end_core, prod_rev_start_end_f64, prod_rev_start_end_i64,
    size_rev_start_end_core, sum_rev_start_end_f64, sum_rev_start_end_i64,
};
use numpy::ndarray::ArrayView1;
use std::collections::HashMap;
use std::hint::black_box;

mod support;
use support::count_allocations;

struct Input<'a> {
    values: &'a [i64],
    starts: &'a [usize],
    ends: &'a [usize],
    index: &'a [i64],
    starts_i64: &'a [i64],
    ends_i64: &'a [i64],
    booleans: &'a [bool],
}

struct FloatInput<'a> {
    values: &'a [f64],
    index: &'a [i64],
    starts_i64: &'a [i64],
    ends_i64: &'a [i64],
    booleans: &'a [bool],
}

fn sort_pairs<T>(mut pairs: Vec<(i64, T)>) -> Vec<(i64, T)> {
    pairs.sort_by_key(|(label, _)| *label);
    pairs
}

fn production_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let (labels, values) = sum_rev_start_end_i64(
        ArrayView1::from(input.values),
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
        ArrayView1::from(input.booleans),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(values).collect())
}

fn production_float_sum(input: &FloatInput<'_>) -> Vec<(i64, f64)> {
    let (labels, values) = sum_rev_start_end_f64(
        ArrayView1::from(input.values),
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
        ArrayView1::from(input.booleans),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(values).collect())
}

fn production_product(input: &Input<'_>) -> Vec<(i64, i64)> {
    let (labels, values) = prod_rev_start_end_i64(
        ArrayView1::from(input.values),
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
        ArrayView1::from(input.booleans),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(values).collect())
}

fn production_float_product(input: &FloatInput<'_>) -> Vec<(i64, f64)> {
    let (labels, values) = prod_rev_start_end_f64(
        ArrayView1::from(input.values),
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
        ArrayView1::from(input.booleans),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(values).collect())
}

fn production_min(input: &Input<'_>) -> Vec<(i64, i64)> {
    let (labels, rows) = min_rev_start_end_core(
        ArrayView1::from(input.values),
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
        ArrayView1::from(input.booleans),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(rows).collect())
}

fn production_max(input: &Input<'_>) -> Vec<(i64, i64)> {
    let (labels, rows) = max_rev_start_end_core(
        ArrayView1::from(input.values),
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
        ArrayView1::from(input.booleans),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(rows).collect())
}

fn production_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let (labels, counts) = size_rev_start_end_core(
        ArrayView1::from(input.starts_i64),
        ArrayView1::from(input.ends_i64),
        ArrayView1::from(input.index),
    )
    .expect("benchmark input must satisfy production preconditions");
    sort_pairs(labels.into_iter().zip(counts).collect())
}

enum OldSizeStorage {
    Sparse(HashMap<usize, i64>),
    Dense { seen: Vec<bool>, counts: Vec<i64> },
}

fn old_checked_range(start: i64, end: i64, len: usize) -> Option<(usize, usize)> {
    let start = usize::try_from(start).ok()?;
    let end = usize::try_from(end).ok()?;
    (end <= len && start < end).then_some((start, end))
}

/// The pre-refactor generic range reducer specialized to counting.
fn old_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut storage = OldSizeStorage::Sparse(HashMap::new());
    let mut touched = Vec::new();
    for row in 0..input.starts.len() {
        let Some((start, end)) = old_checked_range(
            input.starts_i64[row],
            input.ends_i64[row],
            input.index.len(),
        ) else {
            continue;
        };
        for item in start..end {
            match &mut storage {
                OldSizeStorage::Dense { seen, counts } => {
                    if !seen[item] {
                        seen[item] = true;
                        touched.push(item);
                    }
                    counts[item] += 1;
                }
                OldSizeStorage::Sparse(counts) => {
                    let count = counts.entry(item).or_insert_with(|| {
                        touched.push(item);
                        0
                    });
                    *count += 1;
                    if counts.len().saturating_mul(2) >= input.index.len() {
                        let old = std::mem::replace(
                            &mut storage,
                            OldSizeStorage::Dense {
                                seen: Vec::new(),
                                counts: Vec::new(),
                            },
                        );
                        let OldSizeStorage::Sparse(counts) = old else {
                            unreachable!("storage changed while promoting");
                        };
                        let mut seen = vec![false; input.index.len()];
                        let mut dense_counts = vec![0_i64; input.index.len()];
                        for (item, count) in counts {
                            seen[item] = true;
                            dense_counts[item] = count;
                        }
                        storage = OldSizeStorage::Dense {
                            seen,
                            counts: dense_counts,
                        };
                    }
                }
            }
        }
    }
    let mut result = Vec::with_capacity(touched.len());
    match storage {
        OldSizeStorage::Dense { counts, .. } => {
            for item in touched {
                result.push((input.index[item], counts[item]));
            }
        }
        OldSizeStorage::Sparse(counts) => {
            for item in touched {
                result.push((input.index[item], counts[&item]));
            }
        }
    }
    sort_pairs(result)
}

fn hash_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            *map.entry(input.index[item]).or_insert(0) += input.values[row];
        }
    }
    let mut out = map.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut totals = vec![0_i64; input.index.len()];
    let mut seen = vec![false; input.index.len()];
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            seen[item] = true;
            totals[item] += input.values[row];
        }
    }
    let mut out = input
        .index
        .iter()
        .enumerate()
        .filter(|(item, _)| seen[*item])
        .map(|(item, &label)| (label, totals[item]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_product(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            let product = map.entry(input.index[item]).or_insert(1_i64);
            *product = product.wrapping_mul(input.values[row]);
        }
    }
    let mut out = map.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_float_product(input: &FloatInput<'_>) -> Vec<(i64, f64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts_i64[row] as usize..input.ends_i64[row] as usize {
            let product = map.entry(input.index[item]).or_insert(1.0_f64);
            *product *= input.values[row];
        }
    }
    let mut out = map.into_iter().collect::<Vec<_>>();
    out.sort_unstable_by_key(|(label, _)| *label);
    out
}

fn dense_product(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut products = vec![1_i64; input.index.len()];
    let mut seen = vec![false; input.index.len()];
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            seen[item] = true;
            products[item] = products[item].wrapping_mul(input.values[row]);
        }
    }
    let mut out = input
        .index
        .iter()
        .enumerate()
        .filter(|(item, _)| seen[*item])
        .map(|(item, &label)| (label, products[item]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_winner(input: &Input<'_>, find_max: bool) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            let label = input.index[item];
            let winner = map.entry(label).or_insert((input.values[row], row as i64));
            let replaces = if find_max {
                input.values[row] > winner.0
            } else {
                input.values[row] < winner.0
            };
            if replaces {
                *winner = (input.values[row], row as i64);
            }
        }
    }
    let mut out = map
        .into_iter()
        .map(|(label, (_, row))| (label, row))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_winner(input: &Input<'_>, find_max: bool) -> Vec<(i64, i64)> {
    let mut values = vec![None; input.index.len()];
    let mut rows = vec![-1_i64; input.index.len()];
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            let replaces = match values[item] {
                None => true,
                Some(value) if find_max => input.values[row] > value,
                Some(value) => input.values[row] < value,
            };
            if replaces {
                values[item] = Some(input.values[row]);
                rows[item] = row as i64;
            }
        }
    }
    let mut out = input
        .index
        .iter()
        .enumerate()
        .filter(|(item, _)| rows[*item] >= 0)
        .map(|(item, &label)| (label, rows[item]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn ordinal_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut slots = HashMap::with_capacity(input.index.len());
    let mut ordinal_to_slot = Vec::with_capacity(input.index.len());
    let mut labels = Vec::new();
    for &label in input.index {
        let slot = match slots.get(&label) {
            Some(&slot) => slot,
            None => {
                let slot = labels.len();
                slots.insert(label, slot);
                labels.push(label);
                slot
            }
        };
        ordinal_to_slot.push(slot);
    }
    let mut totals = vec![0_i64; labels.len()];
    let mut seen = vec![false; labels.len()];
    for (row, (&start, &end)) in input.starts.iter().zip(input.ends).enumerate() {
        for &slot in ordinal_to_slot.iter().take(end).skip(start) {
            seen[slot] = true;
            totals[slot] += input.values[row];
        }
    }
    let mut out = labels
        .into_iter()
        .enumerate()
        .filter(|(slot, _)| seen[*slot])
        .map(|(slot, label)| (label, totals[slot]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn hash_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut map = HashMap::with_capacity(input.index.len());
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            *map.entry(input.index[item]).or_insert(0) += 1;
        }
    }
    let mut out = map.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn dense_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut counts = vec![0_i64; input.index.len()];
    let mut seen = vec![false; input.index.len()];
    for row in 0..input.values.len() {
        for item in input.starts[row]..input.ends[row] {
            seen[item] = true;
            counts[item] += 1;
        }
    }
    let mut out = input
        .index
        .iter()
        .enumerate()
        .filter(|(item, _)| seen[*item])
        .map(|(item, &label)| (label, counts[item]))
        .collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn sweep_sum(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    let mut active_events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += input.values[row];
        events[input.ends[row]] -= input.values[row];
        active_events[input.starts[row]] += 1;
        active_events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut active = 0_i64;
    let mut out = Vec::new();
    for item in 0..input.index.len() {
        running += events[item];
        active += active_events[item];
        if active > 0 {
            out.push((input.index[item], running));
        }
    }
    out.sort_unstable();
    out
}

fn sweep_sum_compact(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    let mut active_events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += input.values[row];
        events[input.ends[row]] -= input.values[row];
        active_events[input.starts[row]] += 1;
        active_events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut active = 0_i64;
    let mut compacted = HashMap::with_capacity(input.index.len());
    for item in 0..input.index.len() {
        running += events[item];
        active += active_events[item];
        if active > 0 {
            *compacted.entry(input.index[item]).or_insert(0) += running;
        }
    }
    let mut out = compacted.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn sweep_size(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += 1;
        events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut out = Vec::new();
    for (item, event) in events.iter().take(input.index.len()).enumerate() {
        running += *event;
        if running != 0 {
            out.push((input.index[item], running));
        }
    }
    out.sort_unstable();
    out
}

fn sweep_size_compact(input: &Input<'_>) -> Vec<(i64, i64)> {
    let mut events = vec![0_i64; input.index.len() + 1];
    for row in 0..input.values.len() {
        events[input.starts[row]] += 1;
        events[input.ends[row]] -= 1;
    }
    let mut running = 0_i64;
    let mut compacted = HashMap::with_capacity(input.index.len());
    for (item, event) in events.iter().take(input.index.len()).enumerate() {
        running += *event;
        if running != 0 {
            *compacted.entry(input.index[item]).or_insert(0) += running;
        }
    }
    let mut out = compacted.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out
}

fn allocations<T>(f: impl FnOnce() -> T) -> (usize, usize) {
    let (bytes, _calls, peak) = count_allocations(|| black_box(f()));
    (bytes, peak)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("reverse_starts_ends_experiments");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(1));
    // The contiguous singleton case exercises dense storage because the union
    // of the ranges spans the whole domain. The scattered case deliberately
    // keeps only 200 one-position ranges far apart, exposing span-based dense
    // classification and its allocation cost. The validation-heavy case has
    // no aggregation work, isolating the repeated range checks.
    for (name, rows, right_len, width, duplicate) in [
        ("tiny_narrow_unique", 32, 32, 4, false),
        ("tiny_narrow_unique_reversed", 32, 32, 4, false),
        ("large_broad_unique", 1_000, 10_000, 8_000, false),
        ("large_broad_unique_reversed", 1_000, 10_000, 8_000, false),
        ("large_narrow_unique", 1_000, 10_000, 8, false),
        ("large_narrow_unique_reversed", 1_000, 10_000, 8, false),
        ("very_large_broad_unique", 1_000, 100_000, 80_000, false),
        (
            "very_large_broad_unique_reversed",
            1_000,
            100_000,
            80_000,
            false,
        ),
        (
            "super_large_narrow_duplicate",
            1_000_000,
            1_000_000,
            8,
            true,
        ),
        ("scattered_sparse_unique", 200, 1_000_000, 1, false),
        (
            "validation_heavy_zero_width",
            1_000_000,
            1_000_000,
            1,
            false,
        ),
        (
            "super_large_singleton_unique",
            1_000_000,
            1_000_000,
            1,
            false,
        ),
        ("super_large_narrow_unique", 1_000_000, 1_000_000, 8, false),
    ] {
        let values = (0..rows)
            .map(|row| (row % 7 + 1) as i64)
            .collect::<Vec<_>>();
        let starts = if name == "validation_heavy_zero_width" {
            vec![0; rows]
        } else if name == "scattered_sparse_unique" {
            (0..rows)
                .map(|row| row * (right_len - 1) / (rows - 1))
                .collect::<Vec<_>>()
        } else {
            (0..rows)
                .map(|row| row % (right_len - width + 1))
                .collect::<Vec<_>>()
        };
        let ends = if name == "validation_heavy_zero_width" {
            vec![0; rows]
        } else {
            starts
                .iter()
                .map(|&start| start + width)
                .collect::<Vec<_>>()
        };
        let starts_i64 = starts.iter().map(|&value| value as i64).collect::<Vec<_>>();
        let ends_i64 = ends.iter().map(|&value| value as i64).collect::<Vec<_>>();
        let booleans = vec![false; rows];
        let reversed = name.ends_with("reversed");
        let index = (0..right_len)
            .map(|item| {
                if duplicate {
                    (item % 32) as i64
                } else if reversed {
                    (right_len - 1 - item) as i64
                } else {
                    item as i64
                }
            })
            .collect::<Vec<_>>();
        let input = Input {
            values: &values,
            starts: &starts,
            ends: &ends,
            index: &index,
            starts_i64: &starts_i64,
            ends_i64: &ends_i64,
            booleans: &booleans,
        };
        let float_values = values.iter().map(|&value| value as f64).collect::<Vec<_>>();
        let float_input = FloatInput {
            values: &float_values,
            index: &index,
            starts_i64: &starts_i64,
            ends_i64: &ends_i64,
            booleans: &booleans,
        };

        let hash = hash_sum(&input);
        if !duplicate {
            assert_eq!(hash, dense_sum(&input));
        }
        assert_eq!(hash, sweep_sum_compact(&input));
        assert_eq!(hash_size(&input), sweep_size_compact(&input));
        if !duplicate {
            assert_eq!(hash_product(&input), dense_product(&input));
            assert_eq!(hash_winner(&input, true), dense_winner(&input, true));
            assert_eq!(hash_winner(&input, false), dense_winner(&input, false));
            assert_eq!(hash_size(&input), dense_size(&input));
            assert_eq!(production_sum(&input), hash_sum(&input));
            assert_eq!(production_product(&input), hash_product(&input));
            assert_eq!(production_min(&input), hash_winner(&input, false));
            assert_eq!(production_max(&input), hash_winner(&input, true));
            assert_eq!(production_size(&input), hash_size(&input));
            assert_eq!(
                production_float_sum(&float_input),
                hash_sum(&input)
                    .into_iter()
                    .map(|(label, value)| (label, value as f64))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                production_float_product(&float_input),
                hash_float_product(&float_input)
            );
        }
        eprintln!(
            "{name}: hash_sum {:?}, dense_sum {:?}, ordinal_sum {:?}, sweep_sum {:?}, sweep_sum_compact {:?}, hash_size {:?}, sweep_size {:?}, sweep_size_compact {:?}",
            allocations(|| hash_sum(&input)),
            if duplicate { (0, 0) } else { allocations(|| dense_sum(&input)) },
            allocations(|| ordinal_sum(&input)),
            allocations(|| sweep_sum(&input)),
            allocations(|| sweep_sum_compact(&input)),
            allocations(|| hash_size(&input)),
            allocations(|| sweep_size(&input)),
            allocations(|| sweep_size_compact(&input)),
        );
        if !duplicate {
            eprintln!(
                "{name}: production allocations sum {:?}, prod {:?}, float_sum {:?}, float_prod {:?}, min {:?}, max {:?}, old_size {:?}, size {:?}",
                allocations(|| production_sum(&input)),
                allocations(|| production_product(&input)),
                allocations(|| production_float_sum(&float_input)),
                allocations(|| production_float_product(&float_input)),
                allocations(|| production_min(&input)),
                allocations(|| production_max(&input)),
                allocations(|| old_size(&input)),
                allocations(|| production_size(&input)),
            );
        }
        group.bench_function(format!("sum/hash/{name}"), |b| {
            b.iter(|| hash_sum(black_box(&input)))
        });
        if !duplicate {
            group.bench_function(format!("sum/production/{name}"), |b| {
                b.iter(|| production_sum(black_box(&input)))
            });
            group.bench_function(format!("prod/production/{name}"), |b| {
                b.iter(|| production_product(black_box(&input)))
            });
            group.bench_function(format!("float_sum/production/{name}"), |b| {
                b.iter(|| production_float_sum(black_box(&float_input)))
            });
            group.bench_function(format!("float_prod/production/{name}"), |b| {
                b.iter(|| production_float_product(black_box(&float_input)))
            });
            group.bench_function(format!("float_prod/hash/{name}"), |b| {
                b.iter(|| hash_float_product(black_box(&float_input)))
            });
            group.bench_function(format!("min/production/{name}"), |b| {
                b.iter(|| production_min(black_box(&input)))
            });
            group.bench_function(format!("max/production/{name}"), |b| {
                b.iter(|| production_max(black_box(&input)))
            });
            group.bench_function(format!("size/old/{name}"), |b| {
                b.iter(|| old_size(black_box(&input)))
            });
            group.bench_function(format!("size/production/{name}"), |b| {
                b.iter(|| production_size(black_box(&input)))
            });
        }
        group.bench_function(format!("sum/ordinal/{name}"), |b| {
            b.iter(|| ordinal_sum(black_box(&input)))
        });
        group.bench_function(format!("sum/sweep/{name}"), |b| {
            b.iter(|| sweep_sum(black_box(&input)))
        });
        group.bench_function(format!("sum/sweep_compact/{name}"), |b| {
            b.iter(|| sweep_sum_compact(black_box(&input)))
        });
        group.bench_function(format!("size/hash/{name}"), |b| {
            b.iter(|| hash_size(black_box(&input)))
        });
        if !duplicate {
            group.bench_function(format!("size/dense/{name}"), |b| {
                b.iter(|| dense_size(black_box(&input)))
            });
        }
        group.bench_function(format!("size/sweep/{name}"), |b| {
            b.iter(|| sweep_size(black_box(&input)))
        });
        group.bench_function(format!("size/sweep_compact/{name}"), |b| {
            b.iter(|| sweep_size_compact(black_box(&input)))
        });
        group.bench_function(format!("prod/hash/{name}"), |b| {
            b.iter(|| hash_product(black_box(&input)))
        });
        group.bench_function(format!("min/hash/{name}"), |b| {
            b.iter(|| hash_winner(black_box(&input), false))
        });
        group.bench_function(format!("max/hash/{name}"), |b| {
            b.iter(|| hash_winner(black_box(&input), true))
        });
        if !duplicate {
            group.bench_function(format!("prod/dense/{name}"), |b| {
                b.iter(|| dense_product(black_box(&input)))
            });
            group.bench_function(format!("min/dense/{name}"), |b| {
                b.iter(|| dense_winner(black_box(&input), false))
            });
            group.bench_function(format!("max/dense/{name}"), |b| {
                b.iter(|| dense_winner(black_box(&input), true))
            });
        }
        if !duplicate {
            group.bench_function(format!("sum/dense/{name}"), |b| {
                b.iter(|| dense_sum(black_box(&input)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
