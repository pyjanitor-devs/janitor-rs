# Changelog

All notable changes to janitor-rs are documented here.

## [Unreleased]

## [0.6.6](https://github.com/pyjanitor-devs/janitor-rs/compare/v0.6.5...v0.6.6) - 2026-10-24

### Fixed

- avoid rebuilding proven empty joins
- preserve sweep gate for clustered matches
- gate sweeps before ranking and validate layouts
- retain sweep gates and validate labels
- preserve rank sweep density and memory gates
- resolve main branch conflicts
- preserve equi unique index export
- rebase envelope keep sweep onto main

### Other

- *(equi)* sample residual selectivity before mask fallback
- Merge pull request #281 from pyjanitor-devs/issue-273-reverse-aggregation
- optimize reverse envelope range aggregation
- gate range skip scans on passing rows
- *(equi)* rewrap an over-long skip_scan_is_dense doc line
- *(equi)* gate the second-range skip scan on row density with residuals
- stop sweep probes with geometric batches
- batch sweep gate window searches
- remove redundant sweep gate
- remove unused equi unique index export
- state trusted no-NaN range precondition
- simplify trusted range monotonicity check
- remove redundant equi uniq wrapper
- stream dual-range window intersection and compact envelope sweep state
- calibrate envelope sweep gate
- gate rank sweep by residual density
- answer keep=first/last on envelope-backed range joins with a rank sweep
- document rank sweep gate
- record range join rebase guidance
- restore rank sweep gate after rebase
- answer envelope-backed range aggregations with a rank sweep
- mask other residuals over opened runs when opened blocks are dense
- drop keep=any branches made unreachable by materialize_range_any
- keep skip-scan setup off keep="any" early matches
- skip unprunable blocks of a range residual in default window joins
- skip unprunable blocks when scanning an unordered second range
- require maturin 1.9.4 for PyO3 extension builds
- retain PyPy wheels on 32-bit targets
- remove deprecated PyO3 feature configuration
- remove unused itertools dependency
- skip free-threaded Python on 32-bit wheels
- bump PyO3 and rust-numpy

### Changed

- update PyO3 and rust-numpy; Python argument and array-mismatch error text may differ, with some argument names now exposed through exception notes
- 32-bit Linux wheels no longer include free-threaded CPython 3.14t because rust-numpy 0.30 rejects that unsupported target; regular CPython and PyPy wheels remain available

## [0.6.5](https://github.com/pyjanitor-devs/janitor-rs/compare/v0.6.4...v0.6.5) - 2026-10-18

### Fixed

- make random null benchmark deterministic
- align mask count prefix dispatch and retain direct overflow coverage
- tighten equality rank-tree fast path

### Other

- keep packed spans with aggregation
- remove obsolete aggregation migration matrix
- move packed spans into conditional joins
- move comparison operators into conditional joins
- group aggregation modules
- group conditional join kernels
- broaden equi residual coverage
- evaluate equi aggregate residuals per window
- pack queried spans for adaptive extrema
- Use positive region labels in probe test
- Name keep-any probe limits
- Improve keep-any probe sampling
- Merge origin/main into issue-234
- make right null coverage discriminating
- reset equi aggregation windows per iteration
- measure equi residual materialization
- Remove obsolete sweep dispatch flag
- exercise unordered residual range path
- Document n-region dispatch tradeoffs
- Tune n-region dispatch cost
- cover residual range paths
- Fix n-region dispatch regressions
- exercise window predicate kernel
- Optimize n-region tree construction
- Improve n-region dispatch and tree reuse
- make u64 wrapping assertion nonzero
- cover dual rank tree wrapping sums
- preserve residual join fast paths
- explain multi-region kd-tree contracts
- Merge remote-tracking branch 'origin/main' into issue-234
- Clarify k-d tree aggregation documentation
- evaluate residual predicates per window
- merge dual rank tree inserts along ancestor paths
- Use k-d tree for multi-region joins
- densify region rank bookkeeping
- check prefix offsets across many disjoint spans
- trim AGENTS.md overlap with #254 count-cutoff entry
- describe million-query prefix oracle
- record per-fixture random null results
- make random null recheck reproducible
- verify prefix sums with random nulls
- keep prefix totals in registers
- explain accepted u64 regression
- record focused unsigned sum recheck
- clarify integer prefix dispatch
- remove count instrumentation from prefix sum PR
- measure count prefixes and retain conservative dispatch
- remove agent workflow documents from prefix sum PR
- answer integer range sums with wrapping prefixes
- require independent benchmarks in every PR review
- describe extrema tree coverage as a hull
- record range extrema runtime and memory evidence
- isolate extrema branch evidence and benchmark queries
- explain adaptive extrema materialization
- restore adaptive range extrema queries
- fix packed tree wording
- use overlapping queries per group
- refresh packed tree comments
- cover many queried equality groups
- pack disjoint spans into one tree
- assign queried spans with binary search
- build trees per queried span
- cover compact aggregation variants
- limit starts-ends aggregation tree span
- explain equality aggregation fast path guard
- strengthen equality aggregation coverage
- cover starts-ends aggregation variants
- measure equality aggregation reductions
- reuse starts-ends aggregation for equality joins
- measure large equality rank trees
- optimize equality-led aggregation with rank trees

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
