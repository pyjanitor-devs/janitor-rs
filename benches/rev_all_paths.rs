//! Coverage benchmark for every listed reverse Python-facing aggregation path.
//!
//! Each listed numeric dtype is measured across range, positional, no-range,
//! and match-tape paths. Shapes cover small/wide, small/narrow, large/wide,
//! and large/narrow inputs. Inputs are prepared once per benchmark so timings
//! describe the aggregation path rather than NumPy-array construction.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use janitor_rs::bench_support::registered_module;
use numpy::{Element, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};
use std::hint::black_box;
use std::time::Duration;

const DTYPES: &[&str] = &[
    "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "f32", "f64",
];
const FAMILIES: &[&str] = &["max", "min", "prod", "sum"];
const NUMERIC_PATHS: &[&str] = &[
    "start",
    "end",
    "start_end",
    "positions",
    "no_range",
    "start_match",
    "end_match",
    "start_end_match",
];
const SIZE_PATHS: &[&str] = &[
    "start",
    "end",
    "positions",
    "start_matches",
    "end_matches",
    "start_end",
    "start_end_matches",
];

type AnyArgs = Py<PyTuple>;
type Call = (Py<PyAny>, AnyArgs);

struct Scenario {
    name: &'static str,
    mixed_nulls: bool,
    dead_matches: bool,
}

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    rows: usize,
    domain: usize,
    width: usize,
}

fn array<T: Element>(py: Python<'_>, values: Vec<T>) -> Py<PyAny> {
    PyArray1::from_vec(py, values).unbind().into_any()
}

fn args(py: Python<'_>, values: Vec<Py<PyAny>>) -> AnyArgs {
    PyTuple::new(py, values)
        .expect("benchmark arguments must be constructible")
        .unbind()
}

fn call(module: &Bound<'_, PyModule>, name: &str, arguments: AnyArgs) -> Call {
    (
        module
            .getattr(name)
            .unwrap_or_else(|error| panic!("missing registered benchmark function {name}: {error}"))
            .unbind(),
        arguments,
    )
}

fn typed_values<T>(rows: usize, convert: impl Fn(u8) -> T) -> Vec<T> {
    (0..rows).map(|row| convert((row % 7 + 1) as u8)).collect()
}

fn numeric_calls<T: Element>(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    suffix: &str,
    shape: Shape,
    scenario: &Scenario,
    convert: impl Fn(u8) -> T,
) -> Vec<Call> {
    let Shape {
        rows,
        domain,
        width,
        ..
    } = shape;
    let arr = array(py, typed_values(rows, convert));
    let starts = array(py, vec![0_i64; rows]);
    let ends = array(py, vec![width as i64; rows]);
    let index = array(py, (0..domain as i64).collect::<Vec<_>>());
    let booleans = array(
        py,
        (0..rows)
            .map(|row| scenario.mixed_nulls && row % 3 == 0)
            .collect::<Vec<_>>(),
    );
    let positions = array(py, (0..width as i64).collect::<Vec<_>>());
    let left_index = array(
        py,
        (0..rows.saturating_mul(width))
            .map(|row| (row % rows) as i64)
            .collect::<Vec<_>>(),
    );
    let right_index = array(
        py,
        (0..rows.saturating_mul(width))
            .map(|item| (item % width) as i64)
            .collect::<Vec<_>>(),
    );
    let counts = array(py, vec![width as i64; rows]);
    let match_value = if scenario.dead_matches { 0_i8 } else { 1_i8 };
    let matches = array(py, vec![match_value; rows.saturating_mul(width)]);
    let start_counts = array(py, vec![domain as i64; rows]);
    let start_matches = array(py, vec![match_value; rows.saturating_mul(domain)]);

    let mut calls = Vec::with_capacity(FAMILIES.len() * NUMERIC_PATHS.len());
    for family in FAMILIES {
        for path in NUMERIC_PATHS {
            let name = format!("compute_{family}_rev_{path}_{suffix}");
            let arguments = match *path {
                "start" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        starts.clone_ref(py),
                        index.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "end" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        ends.clone_ref(py),
                        index.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "start_end" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        starts.clone_ref(py),
                        ends.clone_ref(py),
                        index.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "positions" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        starts.clone_ref(py),
                        ends.clone_ref(py),
                        index.clone_ref(py),
                        positions.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "no_range" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        left_index.clone_ref(py),
                        right_index.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "start_match" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        starts.clone_ref(py),
                        start_counts.clone_ref(py),
                        index.clone_ref(py),
                        start_matches.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "end_match" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        index.clone_ref(py),
                        ends.clone_ref(py),
                        counts.clone_ref(py),
                        matches.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                "start_end_match" => args(
                    py,
                    vec![
                        arr.clone_ref(py),
                        starts.clone_ref(py),
                        ends.clone_ref(py),
                        index.clone_ref(py),
                        counts.clone_ref(py),
                        matches.clone_ref(py),
                        booleans.clone_ref(py),
                    ],
                ),
                _ => unreachable!(),
            };
            calls.push(call(module, &name, arguments));
        }
    }
    calls
}

fn size_calls(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
    shape: Shape,
    dead_matches: bool,
) -> Vec<Call> {
    let Shape {
        rows,
        domain,
        width,
        ..
    } = shape;
    let starts = array(py, vec![0_i64; rows]);
    let ends = array(py, vec![width as i64; rows]);
    let index = array(py, (0..domain as i64).collect::<Vec<_>>());
    let positions = array(py, (0..width as i64).collect::<Vec<_>>());
    let match_value = if dead_matches { 0_i8 } else { 1_i8 };
    let matches = array(py, vec![match_value; rows.saturating_mul(width)]);
    let start_matches = array(py, vec![match_value; rows.saturating_mul(domain)]);
    let mut calls = Vec::with_capacity(SIZE_PATHS.len());
    for path in SIZE_PATHS {
        let name = format!("compute_size_rev_{path}");
        let arguments = match *path {
            "start" => args(py, vec![starts.clone_ref(py), index.clone_ref(py)]),
            "end" => args(py, vec![ends.clone_ref(py), index.clone_ref(py)]),
            "positions" => args(
                py,
                vec![
                    starts.clone_ref(py),
                    ends.clone_ref(py),
                    index.clone_ref(py),
                    positions.clone_ref(py),
                ],
            ),
            "start_matches" => args(
                py,
                vec![
                    starts.clone_ref(py),
                    index.clone_ref(py),
                    start_matches.clone_ref(py),
                ],
            ),
            "end_matches" => args(
                py,
                vec![
                    ends.clone_ref(py),
                    index.clone_ref(py),
                    matches.clone_ref(py),
                ],
            ),
            "start_end" => args(
                py,
                vec![
                    starts.clone_ref(py),
                    ends.clone_ref(py),
                    index.clone_ref(py),
                ],
            ),
            "start_end_matches" => args(
                py,
                vec![
                    starts.clone_ref(py),
                    ends.clone_ref(py),
                    index.clone_ref(py),
                    matches.clone_ref(py),
                ],
            ),
            _ => unreachable!(),
        };
        calls.push(call(module, &name, arguments));
    }
    calls
}

fn run_calls(py: Python<'_>, calls: &[Call]) {
    for (function, arguments) in calls {
        function
            .bind(py)
            .call1(arguments.bind(py))
            .expect("benchmark fixture must satisfy wrapper preconditions");
    }
}

fn bench(c: &mut Criterion) {
    Python::initialize();
    let mut group = c.benchmark_group("reverse_all_paths_all_dtypes");
    group.sample_size(10);
    group.measurement_time(Duration::from_millis(500));

    for &shape in &[
        Shape {
            name: "small_narrow",
            rows: 8,
            domain: 16,
            width: 2,
        },
        Shape {
            name: "small_wide",
            rows: 8,
            domain: 16,
            width: 16,
        },
        Shape {
            name: "large_narrow",
            rows: 1_024,
            domain: 65_536,
            width: 8,
        },
        Shape {
            name: "large_wide",
            rows: 1_024,
            domain: 4_096,
            width: 4_096,
        },
    ] {
        Python::attach(|py| {
            let module = registered_module(py).expect("module registration must succeed");
            for scenario in &[
                Scenario {
                    name: "baseline",
                    mixed_nulls: false,
                    dead_matches: false,
                },
                Scenario {
                    name: "mixed_nulls",
                    mixed_nulls: true,
                    dead_matches: false,
                },
                Scenario {
                    name: "dead_matches",
                    mixed_nulls: false,
                    dead_matches: true,
                },
                Scenario {
                    name: "mixed_nulls_dead_matches",
                    mixed_nulls: true,
                    dead_matches: true,
                },
            ] {
                for suffix in DTYPES {
                    let calls = match *suffix {
                        "int8" => {
                            numeric_calls::<i8>(py, &module, suffix, shape, scenario, |v| v as i8)
                        }
                        "int16" => {
                            numeric_calls::<i16>(py, &module, suffix, shape, scenario, |v| v as i16)
                        }
                        "int32" => {
                            numeric_calls::<i32>(py, &module, suffix, shape, scenario, |v| v as i32)
                        }
                        "int64" => {
                            numeric_calls::<i64>(py, &module, suffix, shape, scenario, |v| v as i64)
                        }
                        "uint8" => numeric_calls::<u8>(py, &module, suffix, shape, scenario, |v| v),
                        "uint16" => {
                            numeric_calls::<u16>(py, &module, suffix, shape, scenario, |v| v as u16)
                        }
                        "uint32" => {
                            numeric_calls::<u32>(py, &module, suffix, shape, scenario, |v| v as u32)
                        }
                        "uint64" => {
                            numeric_calls::<u64>(py, &module, suffix, shape, scenario, |v| v as u64)
                        }
                        "f32" => {
                            numeric_calls::<f32>(py, &module, suffix, shape, scenario, |v| v as f32)
                        }
                        "f64" => {
                            numeric_calls::<f64>(py, &module, suffix, shape, scenario, |v| v as f64)
                        }
                        _ => unreachable!(),
                    };
                    let label = format!("{}/{}/{suffix}", shape.name, scenario.name);
                    group.bench_with_input(
                        BenchmarkId::new("numeric", &label),
                        &calls,
                        |b, calls| b.iter(|| run_calls(py, black_box(calls))),
                    );
                }
                let calls = size_calls(py, &module, shape, scenario.dead_matches);
                let label = format!("{}/{}", shape.name, scenario.name);
                group.bench_with_input(BenchmarkId::new("size", label), &calls, |b, calls| {
                    b.iter(|| run_calls(py, black_box(calls)))
                });
            }
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
