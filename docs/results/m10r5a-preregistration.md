# M10-R5A — pre-registration

**Written and staged before any router was fitted.** Recorded so the operating point, the
selection rule and the verdict thresholds cannot be chosen after seeing results — the discipline
that let Phase 40 be correctly skipped rather than built to justify itself.

**Timestamp:** see `git log` for this file's first commit.

## Cohort

All IAM **validation** writers with ≥12 lines: **32 writers** (R2 used 20). Per writer, support =
first 5 lines by `sample_id`, query = next 7. Character slots are **base-emitted** (R2's
`evaluation/emitted_spans.py`), never forced-aligned to the reference. IAM test split untouched.

## Eligibility for intervention

An emitted occurrence is eligible iff:

```text
memory prediction != base prediction
AND memory prediction ∈ base top-2 candidates at that slot
```

Memory is the unchanged mean profile (R1 measured exemplars worse; R2 measured residual/whitening/
shrinkage no better).

## Router ladder — escalate only on measured shortfall

1. **A — empirically shrunk pair-conditioned probability** (primary, simplest):
   `p̂(HELP | base→mem) = (h_pair + m·p₀) / (h_pair + d_pair + m)`, pseudo-count `m = 5`,
   `p₀` = global HELP rate among eligible disagreements in the fold's training writers.
   Override iff `p̂ ≥ τ`.
2. **B — L2-regularized logistic regression** on base confidence, margin, entropy, memory
   similarity, memory runner-up margin, candidate rank, support count, and the pair's shrunk prior.
   Fitted only if A fails or is coverage-limited.
3. **C — ≤3-layer MLP under 50K params.** Only if B shows a shortfall B could plausibly close.

## Fitting protocol — leave-one-writer-out, everything inside the fold

For each held-out writer, **the pair statistics `p₀`, the shrinkage and the threshold `τ` are all
fitted on the other 31 writers only**, then applied unchanged to the held-out writer. `τ` is chosen
per fold to maximize net corrections on that fold's training writers. No quantity is selected using
the writer it is evaluated on, and no threshold is chosen after seeing pooled results.

## Primary metric

**CER delta**, measured at writer level. Not HELP:DAMAGE, not router classification accuracy.

## Verdict thresholds — fixed now

**PASS**

```text
mean writer-level CER delta < 0 (an improvement)
AND its writer-level bootstrap 95% CI excludes zero
AND net corrections > 0
AND correction precision = HELP/(HELP+DAMAGE) >= 0.65   (i.e. HELP:DAMAGE >= 1.86)
```

**COVERAGE-LIMITED** — precise but too narrow to matter:

```text
precision >= 0.65 and net corrections > 0
BUT the CI includes zero, or |CER delta| < 0.001 (0.1pp), or coverage < 1% of characters
```

This outcome is **the justification for a GlyphVerifier**: it would mean routing works and the
bottleneck is how few slots carry usable evidence.

**FAIL**

```text
precision < 0.65, or mean CER delta >= 0
```

FAIL does not automatically justify a GlyphVerifier; per `docs/M10R-PLAN.md` it would first require
checking whether the failure is in the eligibility construction itself.

## Reported regardless of outcome

writers improved / harmed / unchanged · median and mean writer CER delta · writer-level bootstrap
95% CI · coverage (share of characters intervened on) · HELP · DAMAGE · NEITHER · net corrections ·
precision · the fitted `τ` distribution across folds · which ladder rung was reached and why.
