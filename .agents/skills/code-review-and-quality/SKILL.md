---
name: code-review-and-quality
description: Multi-axis review of correctness, readability, architecture, security, performance, and verification before merging.
---

# Code Review and Quality

Use this skill as the second pass after `adversarial-pr-review` for every PR.
It is a quality gate, not a rubber stamp: approve changes that improve the
codebase, but require concrete fixes for regressions or broken contracts.

## Review axes

Check correctness against the specification and callers, including empty,
null, boundary, malformed, overflow, identity, ordering, and error cases.
Check readability and simplicity: clear names, straightforward control flow,
useful ELI5 comments for non-obvious invariants, and no dead code or needless
indirection. Check architecture: established module boundaries, canonical
helpers, explicit type boundaries, appropriate abstraction, and focused change
size. Check security and input validation at boundaries. Check performance for
unbounded work, unnecessary allocations, hot-loop regressions, and missing
benchmark cases.

## Verification

Review tests before implementation and ensure they assert behavior rather than
only reproducing the implementation. For optimized code, confirm the tests
actually execute each dispatch branch. Run the relevant test suite, formatter,
lint, build, and benchmarks; record limitations and residual risks. Inspect the
final diff and working tree for accidental scope or temporary instrumentation.

## Findings

Report actionable findings first, ordered by severity, with file/line,
observable impact, evidence, likelihood, and the smallest remediation. Mark
required changes clearly; label optional improvements as such. Separate
confirmed defects from risks and unverified concerns. Do not block a change
over personal style preferences when it follows project conventions.

## Change quality

Every commit or PR description must explain what changed, why, relevant issue
or benchmark evidence, and known tradeoffs. Prefer standard-library or
existing dependencies. Do not silently delete uncertain dead code; identify it
and ask unless it is clearly temporary review instrumentation.
