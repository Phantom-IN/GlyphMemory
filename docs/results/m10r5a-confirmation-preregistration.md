# M10-R5A confirmation — pre-registration

**Written and staged before the confirmation cohort was evaluated.** The threshold below was
derived from development writers only. It is frozen.

## Cohorts (disjoint, verified in code)

```text
development (already used, R5A discovery):  32 IAM validation writers, >=12 lines
confirmation (never used for anything):     34 IAM validation writers, 6-11 lines
overlap:                                    none (asserted at run time)
IAM test split:                             untouched
```

**A stated asymmetry:** the confirmation writers are the *remainder* of the validation split and
have fewer lines each (6–11 vs ≥12), so 115 query lines in total against the development cohort's
224. Support is 5 lines for both. This is not a matched cohort — it is the only untouched one, and
the difference is reported rather than hidden.

## Frozen mechanism — no part of this may change

```text
GM-Base            gm-base-v0, frozen
profile            existing mean profile, support n=5
slots              base-emitted spans (evaluation/emitted_spans.py); never forced-aligned
eligibility        memory != base  AND  memory in base top-2
router             Router B, L2 logistic regression, the exact discovery feature set:
                   base confidence, top1-top2 margin, entropy, memory similarity,
                   memory runner-up margin, log observation count, candidate rank,
                   log span length, in-fold shrunk pair prior (pseudo-count m=5)
```

## Threshold rule and the frozen value

Rule, applied to cross-writer (leave-one-writer-out) predictions on the **32 development writers
only**: the smallest `τ` satisfying `precision >= 0.70`, `net corrections > 0`, `coverage >= 1%`.
The 0.70 target deliberately leaves margin above the 0.65 the confirmation gate requires, because
R5A measured a ~5-point precision drop from in-fold to held-out.

```text
SELECTED tau = 0.60
```

Development operating point at that `τ` (leave-one-writer-out, 32 writers):

```text
HELP 78   DAMAGE 33   net +45
precision 0.703   coverage 1.43%
writer mean CER delta -0.00524   95% CI [-0.00834, -0.00251]
```

Neighbouring thresholds, recorded so the choice is auditable: `0.50` precision 0.693, `0.55` 0.697
(both below the 0.70 rule), `0.65` 0.724, `0.70` 0.733.

## Fitting discipline for the confirmation run

The final router is fitted **once, on all 32 development writers** — pair priors, global prior,
feature standardization and regression weights — then applied unchanged to the 34 confirmation
writers at `τ = 0.60`. No label, outcome or statistic from a confirmation writer enters any fitted
quantity. Asserted in code, not assumed.

## Gate — identical to the original R5A pre-registration

**PASS** requires all of:

```text
mean writer-level CER delta < 0
writer-level bootstrap 95% CI excludes zero
net corrections > 0
precision = HELP / (HELP + DAMAGE) >= 0.65
```

**FAIL** if any clause fails. The gate is not reinterpreted after results are visible, and no
alternate `τ`, feature set, pair prior or confidence gate may replace this result. Diagnostics run
afterwards are labelled post-hoc exploratory and cannot change the verdict.
