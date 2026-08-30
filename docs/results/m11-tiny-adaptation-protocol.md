# M11 — Tiny Adaptation: is a microscopic decision-boundary change *necessary* for reliable few-shot personalization?

> **STATUS 2026-08-25 — EXECUTED AT T0, AND THE LADDER IS CLOSED.**
> T0 returned **TRANSFER FAIL** (`docs/EXPERIMENTS.md` → `m11t0-class-bias-001`,
> `docs/phases/phase-46-m11-tiny-adaptation.md`). §4's escalation rule selects the
> **no-escalation** branch — `support_loss_ratio 0.8081` against an UNDERFIT threshold of `> 0.90`,
> with a positive query delta — so **T1, T2 and T3 are not run and not proposed**, and no second LR,
> step count, seed scheme, CVL-T0 or post-hoc adaptation variant is permitted. The text below is
> preserved exactly as pre-registered; nothing in it was edited after results existed.

**Written as a protocol for review before execution.** Sections 1–12 were the pre-registration,
frozen before the first run in the discipline `docs/results/m10r5a-cvl-preregistration.md` and
`docs/results/m10r3-glyph-verifier-preregistration.md` established. §13 listed what needed an
explicit decision before anything started. Addendum A1 was approved and staged before any
development result existed.

---

## 1. The research question

> **Does reliable few-shot writer personalization require modifying even a microscopic part of
> GM-Base's decision boundary, rather than relying entirely on external writer-memory similarity?**

Phase 45 closed a specific claim: *the character-conditioned similarity formulations tested in this
project do not reliably identify when GM-Base should be overridden.* It explicitly did **not** close
H1's general form — that individual writer knowledge can live outside the weights — and it is not a
proof over all possible similarity functions. This branch tests that general form against its
minimal alternative.

**"Necessary" is the operative word, and it makes the null hypothesis the project-friendly one.** If
external memory matches a tiny gradient arm, the founding principle survives a real attack at
adequate power. If ~80 learned parameters reliably beat it, the principle needs revising, and the
evidence will say by how much and at what cost.

**Why now rather than earlier.** M8 asked a version of this at 4 writers and 2 seeds
(`m8-rival-baseline-table-001`): `batchnorm_ft +0.0095`, `head_ft +0.0089`, `full_ft +0.0066`
against GlyphMemory V0's `+0.0004`. Suggestive; settles nothing. The lesson of M10-R is precisely
that `n=32` could not resolve an effect `n=308` could.

## 2. The comparison, and what is held fixed

The design isolates the **adaptation mechanism** and nothing else:

```text
same gm-base-v0 (fea77c9aaafd52d4)  +  same 5 support lines  +  same query pool
                                    |
              ONLY THIS CHANGES ────┴────────────────────────────────
                                                                     |
   external WriterProfile, no gradients   vs   ~80 learned writer-specific bias parameters
```

Same preprocessing, same tokenizer, same writer-disjoint splits, same CER computation, same support
draws, same seeds. No arm sees a line another arm does not.

### Arms

| | Arm | Adaptation source | Params/writer | Storage/writer | Gradient steps |
|---|---|---|---:|---:|---:|
| **A** | Frozen GM-Base | none | 0 | 0 B | 0 |
| **B** | GlyphMemory, best gradient-free | V0 profile + Router B @ `τ=0.60`, **both frozen** | 0 | ~59 KB | 0 |
| **C** | Tiny adaptation | ladder §3, one rung at a time | see §3 | see §3 | ≤ 20 |
| **R** | Existing rival baselines | `evaluation/rival_baselines.py`, recorded reference | see §7 | — | 20 |

**Arm B is frozen, not re-tuned.** It is the incumbent exactly as the CVL run left it. Re-fitting it
here would make this a new optimization of the mechanism Phase 45 closed, which §M10-R's closure
forbids.

**Arm R is a recorded reference, not a re-run.** `m8-rival-baseline-table-001`'s numbers come from
4 writers × 2 seeds on the IAM **test** split and are quoted with that limitation attached every
time. They are context for interpreting arm C, never a substitute for it.

## 3. The escalation ladder — measured parameter counts, not estimates

Counted by loading `gm-base-v0` (1,544,560 params, vocab **80**) and summing the named parameter
groups. GM-Base is **frozen except for the listed group** at every rung.

| Rung | Trainable parameters | Frozen | Count | % of GM-Base | fp32 / writer | fp16 / writer |
|---|---|---|---:|---:|---:|---:|
| **T0** | `head.projection.bias` | everything else | **80** | **0.0052%** | **320 B** | 160 B |
| **T1** | `head.norm.weight`, `head.norm.bias`, `head.projection.bias` | everything else, incl. `head.projection.weight` (30,720) | **848** | **0.0549%** | 3.31 KB | 1.66 KB |
| **T2** | all BatchNorm affine γ, β (52 tensors, 26 BN layers) | everything else | **8,128** | **0.5262%** | 31.75 KB | 15.88 KB |
| **T3** | low-rank residual adapter on the 384-d sequence feature, rank 4 | **all of GM-Base** | **3,072** | **0.1989%** | 12.00 KB | 6.00 KB |

**T3's adapter, specified exactly** so it cannot grow during implementation:

```text
h' = h + U(V h)      h in R^384 (the BiGRU output entering the head)
V : 384 -> 4         U : 4 -> 384        no bias, no activation, U initialized to ZERO
parameters 384*4 + 4*384 = 3,072    identity at initialization
```

**The ladder is ordered by expressive *kind*, not by parameter count** — T3 (3,072) is smaller than
T2 (8,128). Each rung can express something the previous cannot: T0 shifts class priors; T1
recalibrates the feature normalisation feeding the head; T2 recalibrates per-channel statistics
throughout the encoder; T3 is the first rung that can **mix** feature dimensions rather than scale
them diagonally.

**T0 is the scientific centre of this design, not a warm-up.** V0 fusion adds
`alpha·(1−P_blank)·score` to the logits — a per-class additive correction *derived from memory*,
bounded at 0.41 logits (`m10-readout-headroom-audit-001`). T0 adds a per-class additive constant
*learned from the support lines*. **Same functional form; the only difference is how the correction
is obtained.** If T0 works where arm B does not, the failure localises precisely to the retrieval
key rather than to the readout form.

**Explicitly excluded from this ladder, at any point:** LoRA, full fine-tuning, Transformers,
prompt tuning, cross-attention, any adapter above rank 4, any added module beyond T3, and any rung
touching `head.projection.weight` (30,720 params) or the BiGRU weights. If T0–T3 all fail, the
answer to §1 is "no", and a larger method would answer a different question.

## 4. Escalation rule — measurable, and it can forbid escalation

**A rung's failure permits escalation only if the failure has a measurable cause the next rung can
plausibly address.** The discriminating measurement, fixed here, is the same underfit-vs-transfer
distinction that decided M10-R4's stop rule:

```text
after adaptation, per writer, measure BOTH:
   support_loss_ratio = CTC loss on the 5 SUPPORT lines, adapted / frozen
   query_delta        = CER on the QUERY pool, adapted - frozen

UNDERFIT      support_loss_ratio > 0.90  (the rung barely fits even the data it saw)
              -> the rung lacks capacity. ESCALATE to the next rung.

TRANSFER FAIL support_loss_ratio <= 0.90 AND query_delta >= 0
              (the rung fits its support and does not generalise)
              -> more capacity makes this WORSE, not better. DO NOT ESCALATE. STOP.
```

The `0.90` threshold is fixed now and may not be adjusted after a result is visible. **A transfer
failure at any rung terminates the ladder**, because every later rung has strictly more capacity and
5 support lines is already very little data. Escalation is therefore not the default; it must be
earned by a specific, recorded number.

## 5. Cohorts, splits, and disjointness assertions

```text
DEVELOPMENT   IAM validation, 32 writers (>= 12 lines)   recipe, LR, step count fixed HERE, then frozen
CONFIRMATION  IAM validation, 34 writers (6-11 lines)    frozen recipe, applied once
EXTERNAL      CVL, 308 writers, passage-disjoint         frozen recipe, applied once
IAM TEST      NOT SPENT
```

**The IAM test split is not available.** `docs/00-RESEARCH.md` Protocol 2 allows one spend and M8
used it. Every number in this branch comes from IAM validation or CVL. A future test spend is a
separate, explicitly argued decision and is not assumed here.

**Neither IAM half is "fresh" or "untouched"** — the 32 are the R5A development cohort and the 34
were spent once as the R5A confirmation cohort. **CVL is not a pristine confirmation set either**: a
CVL result motivated the branch this one succeeds, so any CVL run here is an **external transfer
evaluation**, reported with that history attached.

**Asserted at run time, not assumed** (failure raises, never warns):

```text
1. no DEVELOPMENT writer id appears in CONFIRMATION            (train ∩ val = Ø by writer)
2. no IAM record with split == "test" is read, at any point
3. support ∩ query = Ø by sample_id, per writer
4. CVL support ∩ query = Ø by passage_id, per writer          (docs/03-DATA.md §4c)
5. gm-base-v0 fingerprint == fea77c9aaafd52d4 before AND after every rung
6. arm C reloads gm-base-v0 from disk before EVERY writer; a parameter leaking across writers
   would manufacture a gain, so this is asserted by re-checking the fingerprint of the loaded
   state, not by trusting the loop
7. no LR, step count, threshold or stopping rule is read from CONFIRMATION or CVL data
```

## 6. Adaptation recipe — fixed on DEVELOPMENT, then frozen

```text
loss             CTC on the 5 support lines, identical to training (model/loss.py)
optimizer        AdamW, weight_decay 0.0   (a 80-parameter bias vector has nothing to regularize
                 toward zero that the frozen model does not already encode)
learning rate    selected ONCE on the DEVELOPMENT cohort from {1e-3, 3e-3, 1e-2, 3e-2} by mean
                 writer CER delta, then FROZEN for CONFIRMATION and CVL. The grid is fixed here
steps            <= 20, fixed; the exact count chosen on DEVELOPMENT alongside the LR, then frozen
stopping rule    fixed step count. NO early stopping, NO checkpoint selection, NO per-writer
                 tuning of any kind. Phase 36 measured probe-based checkpoint selection at
                 -0.93pp on held-out writers; a fixed count cannot launder evaluation
                 information into the parameters
selection ban    no quantity is ever selected on CONFIRMATION or CVL writers (assertion 7 above)
seeds            3 deterministic support draws per writer, seeds 1337 / 1338 / 1339; every arm
                 receives the IDENTICAL draws; results reported as the mean over seeds with the
                 per-seed spread shown
device           resolved through runtime/device.py and logged (docs/10-RUNTIME.md)
```

**Gradients in arm C do not violate `CLAUDE.md` invariant 4.** The invariant governs *GlyphMemory's
own enrollment path*. `evaluation/rival_baselines.py` already runs 20 gradient steps per writer for
exactly this purpose and `docs/00-RESEARCH.md` §7 names head-only fine-tune "the real rival".
**Adopting** arm C as the system is a different matter entirely — §11.

## 7. Metrics — every one reported for every arm, whatever the outcome

```text
PRIMARY     writer-level mean CER delta (adapted - frozen), with a writer-level bootstrap 95% CI
            4,000 resamples over WRITERS, never over characters

ALSO, ALWAYS
  CER@0 / @1 / @3 / @5 / @10        via evaluation/few_shot.py, the unmodified harness.
                                     @1 and @3 are reported with the caveat that M8 measured every
                                     gradient arm NEGATIVE at n=1
  median writer CER delta            (the mean is sensitive to a single catastrophic writer)
  worst writer CER delta
  writers improved / harmed / unchanged, as counts and percentages
  WER, and the substitution / insertion / deletion decomposition (evaluation/taxonomy.py)
  hard-writer vs easy-writer halves, split by each writer's OWN base error rate
  generic-recognition check: arm A's CER on the full 1,122-line val split, which arm C must
                             not alter -- it is a per-writer change and the no-profile path
                             must be byte-identical

COST, per writer
  trainable parameter count          exact, from the model
  storage of writer-specific state   bytes fp32 and fp16
  gradient steps                     exact
  enrollment wall-clock              seconds, device logged
  CPU p50 line latency at inference  4 threads, batch 1, W512 -- against arm A's 35.5 ms
                                     (Phase 15) and arm B's measured cost
```

## 8. Pre-registered PASS / FAIL gate — frozen before the first fit

Phrased so the **project-friendly outcome requires no action**.

**A rung is judged NECESSARY only if all four clauses hold on CONFIRMATION, and then again on CVL:**

```text
1. mean writer-level CER delta < 0, and the writer-level bootstrap 95% CI excludes zero
2. that delta beats arm B's on the SAME writers, PAIRED, and the CI of the DIFFERENCE excludes zero
3. writers harmed <= 25%          (arm B's CVL figure is 37.3% -- 115 of 308)
4. no generic-recognition regression: arm A's CER on the full val split is unchanged to five
   decimals, since arm C must not alter the no-profile path
```

**Any clause failing is a FAIL for that rung.** No alternate LR, step count, seed, eligibility rule
or cohort filter may substitute for the result. Diagnostics may be computed **after** the primary
result is recorded and must be labelled post-hoc exploratory.

**The ladder stops at the first rung that PASSES.** A failed rung escalates only under §4.

## 9. What each outcome would support — written before any of them exists

```text
T0 (80 params) PASSES
    Evidence that changing the decision boundary matters even when the writer-specific state is
    microscopic -- 0.0052% of GM-Base, 320 bytes. Sharpest possible result against memory:
    IDENTICAL readout form (a per-class additive logit constant), different source of the
    correction. Localises the failure to the retrieval key, independently confirming Phase 45.
    It does NOT mean the project should adopt gradient-based enrollment -- see below.

Only T2 / T3 PASS
    Capacity matters, not merely a boundary shift. Weakens the "microscopic" framing and raises
    storage toward arm B's ~59 KB, so the cost argument for memory strengthens even as the
    accuracy argument weakens. Report both halves.

NOTHING PASSES
    External memory is NOT beaten by its minimal gradient alternative at adequate power. The
    founding principle survives a genuine attempt to break it. This is a publishable positive
    result for the project's thesis and requires no change to anything.

ARM B BEATS ARM C
    Would reverse M8's n=4 indication. Report prominently: M8 used 4 writers and 2 seeds on the
    test split, and this would supersede it at n=308.

CLAUSES 1-2 PASS, CLAUSE 3 FAILS
    A tiny adaptation is better on average and still unreliable -- the same disease as arm B.
    The honest report is that NEITHER method is deployable, and that is the finding.
```

**Adoption is not implied by a PASS, and this is fixed in advance.** If ~80 adapted parameters
reliably outperform gradient-free memory, that is evidence about *what the mechanism needs*. It is
**not** a decision to change what GlyphMemory is. Adoption would require **a separate ADR** amending
`CLAUDE.md` invariant 4 ("No deployment-time gradients … No optimizer, no backprop, no LoRA, no
per-writer checkpoint"), restating `CLAUDE.md` §1's founding principle, and restating the research
claim — `docs/M10R-PLAN.md` §T already says a parameter-efficient adaptation "would abandon the
gradient-free constraint and is therefore a different research claim, not a continuation." That ADR
does not exist, is not drafted here, and belongs to the project owner after a result exists.

**If tiny adaptation does not outperform the gradient-free baseline, that result is recorded before
any next rung is considered.** Recording is not optional and not deferred to the end of the ladder.

## 10. Compute estimate

**Estimate, not a measurement** (`CLAUDE.md` invariant 6). Derived from recorded figures: GM-Base
trained 100 epochs × 11,025 lines in ~9h14m on MPS ⇒ ~30 ms per line forward+backward; inference is
35.5 ms/line CPU batch-1 (Phase 15) and considerably faster on MPS.

```text
per writer, one rung, one seed
  adaptation   20 steps x 5 lines x ~30 ms          ~3 s
  query scoring  ~7-10 lines forward                 ~1 s
                                                    ~4 s

DEVELOPMENT   32 writers x 3 seeds x 4 LR x <=2 step counts   ~2.7 h   (recipe search, T0 only)
CONFIRMATION  34 writers x 3 seeds                            ~7 min
CVL           308 writers x 3 seeds                           ~1.0 h
generic-recognition check   1,122 lines, once per rung        ~2 min

one rung, end to end, after the recipe is fixed                ~1.2 h
T0 including the one-time recipe search                        ~4 h
full ladder T0-T3 if every rung escalates (worst case)         ~8 h
```

No CUDA-specific code; MPS has no CTC kernel so the loss falls back to CPU, which is already handled
(`docs/10-RUNTIME.md`).

## 11. Comparison table to be filled — the shape of the final report

Every cell measured in this branch except the arm R row, which is quoted from `m8-rival-baseline-table-001`.

| Arm | Params/writer | Storage | Steps | CER@5 delta | CI | Median | Harmed % | Enroll s | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A · frozen GM-Base | 0 | 0 B | 0 | 0 (reference) | — | — | — | 0 | 35.5 ms |
| B · GlyphMemory (frozen) | 0 | ~59 KB | 0 | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* |
| T0 · 80 class biases | 80 | 320 B | ≤20 | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* | *to measure* |
| T1 · minimal head subset | 848 | 3.31 KB | ≤20 | *only if T0 escalates* | | | | | |
| T2 · BatchNorm affine | 8,128 | 31.75 KB | ≤20 | *only if T1 escalates* | | | | | |
| T3 · rank-4 adapter | 3,072 | 12.00 KB | ≤20 | *only if T2 escalates* | | | | | |
| R · M8 rivals (reference) | 31,568 / 8,128 / 1.54M | — | 20 | `+0.0089` / `+0.0095` / `+0.0066` | 4 writers, 2 seeds, **test split** | — | — | — | — |

## 12. Threats this design does not remove

| Threat | Status |
|---|---|
| 5 support lines is very little for any gradient method; M8 measured all three gradient arms **negative** at `n=1` | Real. `n=5` is the project's standard and is kept for comparability. `n=10` is a pre-registered secondary, reported separately |
| CVL absolute CER is not quotable — reconstructed ground truth omits punctuation | Unchanged from `m10r5a-cvl-preregistration.md` §5. Deltas valid; absolutes are context only |
| The LR/step search happens on DEVELOPMENT writers, who have been observed many times | Real, and why CONFIRMATION and CVL exist and are scored once with a frozen recipe |
| Arm C's enrollment cost is not comparable to arm B's | That is the point, and it is measured rather than argued: params, bytes, steps, wall-clock, latency |
| A gradient arm winning contradicts the project's stated principle | It would, which is why the experiment is worth running. §9 fixes what a win does and does not license |
| Deletions are 25.8% of errors and arm B cannot address them | Arm C can *in principle*, being a decoder-side change rather than a reranker. Worth watching; not a gate |
| Three seeds may understate variance | The per-seed spread is reported, never only the mean |

## 13. Decisions needed before anything starts

1. **Go / no-go on the branch at all.** The alternative is writing up what exists, which is a
   genuine negative result plus a fully supported few-shot benchmarking contribution.
2. **Is the ladder right?** In particular, is T0 at 80 parameters worth a rung, or should this start
   at T2, which M8's `n=4` data already favours? The argument for T0 is §3's functional-equivalence
   point; the argument against is that it may simply be too small to move anything.
3. **Should CVL be spent again?** It is the last external instrument, it is no longer independent of
   this line of work, and there is no third corpus behind it.
4. **The `n=1` question.** M8 measured every gradient arm negative at `n=1`. If reliability at small
   `n` is the real product requirement, a failure at `n=1` may matter more than a pass at `n=5`.

**No code will be written and no parameter fitted until these are answered and this protocol is
approved.**

---

## Addendum A1 — protocol/reality reconciliation, approved and recorded 2026-08-25

**Staged before any T0 development result exists.** No LR has been selected, no writer adapted, no
CER computed. Every item below was found by verifying §§3–7 against the repository, reported to the
project owner, and approved by them before execution.

**None of these amendments changes a gate clause, the model, the LR grid, the optimizer, the step
count, a cohort definition, or any evaluation outcome.** §8's four PASS clauses and §4's escalation
rule are untouched.

### A1.1 · CER@10 is structurally unmeasurable on CONFIRMATION

Measured from the manifest. With the query pool reserved first — required by
`evaluation/few_shot.py` so the CER denominator does not drift with `n` — the support pool is:

```text
DEVELOPMENT   32 writers, >=12 lines   support pool 5-12   n=10 eligible: 21/32
CONFIRMATION  34 writers, 6-11 lines   support pool == 5 for ALL 34   n=10 eligible: 0/34
```

§7 listed CER@0/1/3/5/10 under "ALSO, ALWAYS". Amended:

- **CONFIRMATION: `CER@10 = not measurable (0/34 writers eligible)`**, reported with that exact
  wording and the count, never silently omitted (`CLAUDE.md` invariant 7).
- **DEVELOPMENT: CER@10 reported for the 21/32 eligible writers only, labelled explicitly a
  subset metric.** It must **not** be compared directly against any CONFIRMATION figure, and must
  not be compared against DEVELOPMENT CER@1/3/5, which are computed over all 32.
- CER@0/1/3/5 are unaffected: 32/32 and 34/34 eligible respectively.

### A1.2 · The three seeds are degenerate at n=5 on CONFIRMATION

Two facts compose. Every CONFIRMATION writer's support pool is exactly 5, so at `n=5` there is
exactly **one** possible support subset. And adaptation is **deterministic**: `fine_tuned_model`
holds the model in `.eval()` (no dropout), the support lines are fixed, and AdamW is deterministic —
asserted by `tests/test_rival_class_bias.py::TestDeterminism`.

Consequence: seeds 1337/1338/1339 produce **bit-identical** results at `n=5` on CONFIRMATION.

- **The three registered seeds are preserved and all three are run and reported**, exactly as §6/§9
  specify. Nothing is dropped.
- **Zero between-seed variance at `n=5` on CONFIRMATION reflects a degenerate support draw, not
  method stability**, and every report of that number must carry this sentence. Reporting
  `between-seed variance 0.000` unqualified would read as a stability claim the design cannot
  support.
- Seeds remain genuinely informative at `n=1` and `n=3` on both cohorts, and at `n=5` on
  DEVELOPMENT, where only 3/32 writers have a support pool of exactly 5.

### A1.3 · Weight decay, fixed without disturbing M8

§6 specifies `weight_decay 0.0`. `evaluation/rival_baselines.fine_tuned_model` called
`torch.optim.AdamW(params, lr=lr)`, whose default is **0.01**. AdamW's decoupled decay pulls
parameters toward **zero**, and T0's bias starts at `gm-base-v0`'s *trained* bias — decay would
erode the pretrained value rather than regularize the adaptation.

- An explicit `weight_decay` parameter was added, **defaulting to `0.01`** so
  `m8-rival-baseline-table-001` — the one sanctioned IAM test-split spend — stays reproducible.
- **T0 passes `0.0` explicitly.** Both are pinned by tests.

### A1.4 · Arm B's DEVELOPMENT comparison is in-fold

Router B's pair priors, feature standardization, regression weights and `τ=0.60` were fitted on the
32 DEVELOPMENT writers (`m10r-selective-router-001`). Applying the frozen router to those same
writers is therefore **in-fold and optimistic**.

- **DEVELOPMENT arm-B numbers are descriptive only** and are labelled in-fold wherever they appear.
- **The pre-registered paired T0-vs-B comparison (§8 clause 2) is decided on CONFIRMATION**, where
  Router B is genuinely held out. This is unchanged from §8 — recorded here so the development
  figures cannot later be mistaken for the gate.

### A1.5 · Execution sequence, fixed here

```text
1. stage this addendum                         <- done before any development result exists
2. DEVELOPMENT LR selection over {1e-3, 3e-3, 1e-2, 3e-2}, 32 writers
3. freeze the selected LR and step count; record them
4. CONFIRMATION, run ONCE with the frozen recipe
5. apply §8's gate and §4's escalation rule exactly
6. report PASS / UNDERFIT / TRANSFER FAIL / FAIL, then STOP
```

**T1 is not executed under any outcome.** An UNDERFIT classification results in a *proposal* for T1
and nothing more. The IAM test split stays untouched and `gm-base-v0` stays byte-identical
(`fea77c9aaafd52d4`, asserted before and after).
