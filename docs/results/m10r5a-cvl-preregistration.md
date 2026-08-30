# M10-R5A — external CVL confirmation: pre-registration

**Written and staged before any CVL result is computed.** Nothing below may change after results
are visible. This is a confirmation study, not an optimization wave.

## 1. Why CVL, and why now

The IAM confirmation (`m10r5a-confirmation-001`) returned FAIL with a writer-level interval of
`[-0.00593, +0.00083]` — an outcome **consistent with limited power**, which cannot distinguish a
small real effect from no effect. The power analysis there required ≈70 writers; IAM validation
holds 66 in total and all have now been used. CVL is the project's designated **external**
personalization benchmark (`docs/00-RESEARCH.md` §12a, `docs/03-DATA.md` §2) and is the only
remaining instrument with adequate power that does not spend the IAM test split.

**This is a genuinely external test.** CVL is a different corpus, different writers, different
scanning, and `gm-base-v0` never saw any of it. A domain shift is expected and is part of what is
being tested.

## 2. Cohort — measured from the repository, not assumed

```text
CVL manifest              runs/cvl/manifest.jsonl   10,980 lines / 310 writers
German passage (p6)       excluded by the adapter default (docs/03-DATA.md §2)
passages present          p1 3283 · p2 3014* · p3 2234* · p4 1979 · p7 224 · p8 246
eligible writers          308   (>=5 passage-disjoint support lines AND >=3 query lines)
query lines per writer    capped at 10; all 308 eligible writers reach the cap
total query lines         3,080        total forward passes ~4,620
```

*counts as reported by the manifest scan; exact per-passage totals are recorded in the result
artifact.

**Inclusion rule (fixed):** a writer is included iff, after passage-disjoint partitioning, the
support pool holds ≥5 lines and the query pool holds ≥3. Excluded writers are **counted and
reported**, never silently dropped (`CLAUDE.md` invariant 7).

## 3. Support / query construction — passage-disjoint, and why it is mandatory here

Every CVL writer copies the **same** passages, so without passage-disjointness a writer would be
enrolled and queried on the same text and the result would be meaningless (`docs/03-DATA.md` §4c).
Fixed rule:

```text
group each writer's lines by passage_id
sort groups ascending by size
fill the QUERY pool from whole groups, smallest first, until >= 10 lines (cap 10)
the remaining whole groups form the SUPPORT pool
draw n = 5 support lines from that pool, deterministically, seed 1337
```

Support and query therefore never share a passage. Determinism: seed `1337`, lines ordered by
`sample_id` before any draw.

## 4. Frozen mechanism — no component may be refitted on CVL

```text
recognizer      gm-base-v0, frozen (fea77c9aaafd52d4), verified byte-identical before and after
profile         existing mean WriterProfile, support n = 5
slots           base-emitted spans (evaluation/emitted_spans.py); never forced-aligned
eligibility     memory != base  AND  memory in base top-2
router          Router B, L2 logistic regression — the exact artifact applied in the IAM
                confirmation, fitted on the 32 IAM development writers only
features        base confidence, top1-top2 margin, entropy, memory similarity, memory runner-up
                margin, log observation count, candidate rank, log span length, shrunk pair prior
pair prior      shrinkage pseudo-count m = 5, priors estimated on the 32 IAM development writers
threshold       tau = 0.60, frozen
```

**No CVL data enters any fitted quantity** — not the pair priors, not the feature standardization,
not the regression weights, not `τ`. Asserted at run time, not assumed. The router is *not* refitted
on the larger 66-writer IAM pool either: the frozen artifact is the one already applied to the IAM
confirmation, so this tests that exact object.

## 5. The CVL ground-truth caveat — decisive for how results may be read

`docs/00-RESEARCH.md` §12a, measured in M1 (`data-cvl-characterisation-001`): CVL ships **no
line-level transcription**. The text is reconstructed from word-image filenames, which omits **all
sentence punctuation** and, in a small tail of lines, whole trailing words.

```text
CVL absolute CER      NOT a quotable HTR figure; inflated by that omission; not comparable
                      with IAM CER or with published CVL work using a different reconstruction
CVL adaptation DELTA  valid — the same fixed query pool is scored in both conditions, so the
                      omission is constant across conditions and largely cancels
```

**The primary endpoint is therefore a delta, by necessity as well as by design.** Absolute CER is
reported as context only, always with this caveat attached.

## 6. Endpoints

**Primary:** writer-level CER delta (personalized − base), with a **writer-level** bootstrap 95% CI
(4,000 resamples, resampling writers, never characters — characters within a writer are correlated).

**Also reported, regardless of outcome:** precision `HELP/(HELP+DAMAGE)` · coverage (share of
characters intervened on) · HELP · DAMAGE · NEITHER · net corrections · writers improved / harmed /
unchanged · mean, median and worst writer delta · hard-writer vs easy-writer halves split by each
writer's own base error rate · base and personalized CER (with the §5 caveat) · excluded-writer
count and reasons · `gm-base-v0` fingerprint before and after.

## 7. Gate — identical to R5A's, not renegotiated

**PASS** requires all four:

```text
mean writer-level CER delta < 0
writer-level bootstrap 95% CI excludes zero
net corrections > 0
precision >= 0.65
```

**FAIL** if any clause fails. No alternate `τ`, feature set, pair prior, eligibility rule or
cohort filter may substitute for this result. Diagnostic curves may be computed **after** the
primary result is frozen and must be labelled post-hoc exploratory.

## 8. What each outcome would mean

```text
PASS   the frozen router transfers to an external corpus with adequate power. Recommend freezing
       the full stack and preparing the one-time IAM test protocol. Still no GlyphVerifier.
FAIL, precision >= 0.65 but CI includes zero
       effect too small to matter even at n=308; the mechanism is precise but not useful.
FAIL, precision < 0.65
       separation does not transfer across corpora. This is the first result that would justify a
       GlyphVerifier specifically as a separation mechanism, since power would no longer be the
       confound.
FAIL, mean delta >= 0
       the IAM point estimate did not reflect a real effect. Record and stop this branch.
```

## 9. Interpretation caveats fixed in advance

- This is the **fourth** evaluation of this router (IAM discovery, post-hoc `τ=0.50`, IAM
  confirmation, CVL). It is the first that is external and adequately powered, but the sequence
  should be stated whenever the result is quoted.
- Domain shift is expected to raise base CER substantially. A *smaller relative* gain on CVL than on
  IAM would not by itself indicate mechanism failure; the pre-registered gate is what decides.
- The pair priors encode **IAM** confusion structure. Whether they transfer is part of the test, not
  an assumption.

## 10. Integrity checks required before the result is reported

```text
gm-base-v0 byte-identical before and after
IAM test split not read at any point
no CVL writer present in any fitted quantity (asserted)
excluded writers counted and reported
ruff clean; targeted tests and the full suite run
```
