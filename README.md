# janitor-rs

`janitor-rs` provides the Rust/PyO3 extension kernels used by
[pyjanitor](https://github.com/pyjanitor-devs/pyjanitor). It is not a
standalone command-line tool.

## Development

Install Rust and [uv](https://docs.astral.sh/uv/), then run the Rust tests:

```sh
cargo test --no-default-features
```

On macOS, the repository wrapper provides a consistent Python framework for
PyO3 tests:

```sh
./scripts/test-rust.sh
```

Pass additional Cargo test arguments after the wrapper command:

```sh
./scripts/test-rust.sh single_range_predicate
```

Run formatting and lint checks with:

```sh
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Building

Build a wheel locally with [maturin](https://www.maturin.rs/):

```sh
maturin build --release
```

The package version is read from `Cargo.toml`. Release preparation and PyPI
publishing are documented in [`RELEASING.md`](RELEASING.md).

## Repository layout

- `src/` contains the typed join and aggregation kernels.
- `benches/` contains Criterion benchmarks.
- `.github/workflows/` contains CI and on-demand release workflows.

The Python-facing integration tests live in the pyjanitor repository. Changes
to the Rust/Python boundary should be tested in both repositories.
