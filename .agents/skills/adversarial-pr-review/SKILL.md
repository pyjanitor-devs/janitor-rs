---
name: adversarial-pr-review
description: Stress-test a pull request or proposed code diff when the user asks for an adversarial, red-team, or pre-merge review. Find evidence-backed correctness, compatibility, safety, performance, and test-coverage failures. Do not use for ordinary implementation work or a routine style-focused review.
---

# Adversarial PR Review

Try to falsify the change's claim of correctness. Treat passing tests and the PR
description as evidence to challenge, not proof.

## Establish the Contract

Identify the review target and comparison base from the request, repository, and
PR metadata. Read applicable repository instructions, the complete diff, linked
issue or specification, nearby implementation, and tests. Do not review only the
changed lines when callers or invariants determine their behavior.

State material assumptions when the intended behavior or base is ambiguous. A
review is read-only unless the user separately asks for fixes; temporary local
experiments are allowed when they do not modify tracked source.

Before forming conclusions, define a review matrix covering every changed mode,
input shape, dtype/type variant, error path, and implementation branch. For
optimized code, write down the exact dispatch predicate and its boundary
conditions. A test name or comment claiming branch coverage is not evidence
that the branch executed.

## Attack the Change

Build a concise list of claims and invariants introduced or relied on by the
diff, then look for counterexamples. Prioritize attack surfaces based on the
change rather than applying a fixed checklist. Consider, where relevant:

- boundary, empty, null, duplicate, malformed, and extreme-size inputs;
- alternate modes, feature flags, configuration combinations, and error paths;
- ordering, identity, type, schema, and backward-compatibility guarantees;
- partial failure, retries, concurrency, lifecycle transitions, and cleanup;
- authorization boundaries, injection, data exposure, data loss, and unsafe
  defaults;
- algorithmic blowups, unnecessary materialization, and hot-path regressions;
- whether tests assert the behavior that matters or merely reproduce the
  implementation.

For every heuristic, threshold, cache, or dispatch decision:

- calculate representative inputs just below, exactly at, and just above the
  cutoff;
- verify the actual branch with temporary instrumentation, counters, tracing,
  or a minimal reproduction when static inspection is inconclusive;
- test both sides, including malformed and empty inputs in the decision;
- check that the cost model includes build, query, allocation, and cleanup
  costs.

For every changed algorithmic path, compare it with a simple reference
implementation. Prefer property-based or table-driven comparisons when
practical. Include `-1` sentinels, `0`, exact `len`, `len + 1`, inverted and
empty ranges, all-null ranges, duplicate/tied values, NaNs, infinities,
overflow, non-power-of-two lengths, and extreme sizes as allowed by the
contract.

For aggregation and indexing code, explicitly cover the full shape matrix
(`starts`, `ends`, and `starts_ends`), forward/reverse or direct/adaptive
variants, integer and floating-point semantics, null masks, identity values,
tie-breaking, ordering guarantees, and invalid boundaries. Record which tests
exercise each branch; do not count a test whose inputs never satisfy the
branch predicate.

Trace changed values through callers and downstream consumers. Compare against
the base implementation when a regression could be hidden by refactoring. Use
focused tests, small reproductions, static checks, and existing test suites to
confirm or reject concrete hypotheses. Do not manufacture findings unsupported
by the code or observable behavior.

## Empirical Verification

Use this staged loop for non-trivial reviews:

1. Read repository instructions and tests first.
2. Review the complete diff and derive the contract and branch matrix.
3. Run focused tests for each matrix cell; add temporary branch instrumentation
   when execution is uncertain.
4. Compare optimized and baseline/reference implementations on normal and
   adversarial inputs.
5. Run the full suite, lint/type checks, formatting, and relevant builds.
6. Benchmark both sides of performance dispatch boundaries, including narrow
   workloads where setup overhead may dominate.
7. Inspect the final diff and working tree; remove temporary instrumentation and
   distinguish confirmed findings from residual risks.

Do not claim all paths are covered from a green suite alone. State branch
execution evidence, reference comparisons, performance cases, and unverified
cells explicitly.

For a broad or high-risk change, use independent subagents when available and
authorized to attack distinct risk areas without sharing tentative conclusions;
reconcile their evidence before reporting.

## Report Findings First

Return actionable findings ordered by severity. For each finding include:

- the concrete failure and affected scenario;
- why it matters and how likely it is;
- a precise file and line reference;
- the evidence or minimal reproduction;
- the smallest credible remediation direction, without implementing it unless
  requested.

Use questions only for genuine contract ambiguity, not as substitutes for
findings. Separate confirmed defects from residual risks or unverified concerns.
If no findings survive scrutiny, say so explicitly and report what was tested,
what could not be verified, and the remaining risk.
