# Changelog

All notable changes to janitor-rs are documented here.

## [Unreleased]

## [0.6.4](https://github.com/pyjanitor-devs/janitor-rs/compare/v0.6.3...v0.6.4) - 2026-10-14

### Other

- add dual-region rank segment tree ([#239](https://github.com/pyjanitor-devs/janitor-rs/pull/239))
- build range predicate views once per equi call ([#240](https://github.com/pyjanitor-devs/janitor-rs/pull/240))
- Implement Rust kernels for equality-led interval-overlap joins ([#235](https://github.com/pyjanitor-devs/janitor-rs/pull/235))

## [0.6.3](https://github.com/pyjanitor-devs/janitor-rs/compare/v0.6.2...v0.6.3) - 2026-10-04

### Other

- short-circuit extended any joins ([#229](https://github.com/pyjanitor-devs/janitor-rs/pull/229))
- make macOS Rust tests reproducible ([#228](https://github.com/pyjanitor-devs/janitor-rs/pull/228))
- publish package metadata ([#227](https://github.com/pyjanitor-devs/janitor-rs/pull/227))
- Document pyjanitor compatibility gate ([#226](https://github.com/pyjanitor-devs/janitor-rs/pull/226))
- add on-demand release automation ([#224](https://github.com/pyjanitor-devs/janitor-rs/pull/224))

Release entries are prepared automatically by release-plz from conventional
commit messages.  The release workflow publishes the matching Maturin wheel
and source distribution after the generated version tag is created.
