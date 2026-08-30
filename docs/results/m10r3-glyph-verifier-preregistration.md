# M10-R3/R4 — writer-conditioned GlyphVerifier: design and pre-registration

**Written and staged before any verifier weight exists.** Nothing below may change after a training
run has produced a number. Supporting measurements: `docs/results/m10r3-design-audit-001.json`,
`docs/EXPERIMENTS.md` → `m10r3-design-audit-001`.

**Why this is being built.** The CVL external confirmation (`m10r5a-cvl-confirmation-001`) resolved
the effect statistically at n=308 — the writer-level CI excludes zero — and simultaneously showed
that HELP/DAMAGE **separation** is what fails to transfer: precision `0.703` (IAM development) →
`0.649` (the 34-writer R5A confirmation cohort) → `0.5425` (CVL). Power is no longer the confound.
Separation is the
measured bottleneck, and a dedicated writer-conditioned representation is the mechanism that
addresses separation specifically.

---

## 1. Four design assumptions, measured before committing to them

`docs/M10R-PLAN.md` §D/§E/§F-G sketched this verifier. Three of its four load-bearing assumptions
are now measured and **wrong**. They are corrected here rather than by editing that plan
(`CLAUDE.md` invariant 8).

| # | What the plan assumed | What is measured | Consequence |
|---|---|---|---|
| 1 | "forced alignment gives frame spans and the encoder downsamples width by exactly 4, so a span maps to a known pixel column range" — i.e. **the span is the glyph** | A character occupies **20.5 px** (p50 20.0). A forced-aligned span is **6.0 px** (p50 4.0); a base-emitted span in the eligible pool is **1 frame = 4 px** for **92%** of slots | Rung 2 as written would have trained on 4-pixel slivers. Glyph regions must be **reconstructed** (§3) |
| 2 | A verifier's natural score is "looks like memory's candidate vs looks like the base's character" | The **differential is worse than either arm**: AUROC `0.510` vs `0.653` for `mem_sim` alone. `base_sim` negated scores `0.402` — *below* chance, i.e. high similarity to the base's own character **positively** predicts HELP | Both similarities are dominated by a common "how clean is this span" factor. Subtracting cancels the signal. The verifier must be **trained** with a normalised multi-candidate objective (§5), never differenced post-hoc |
| 3 | "Mine negatives from R1's measured confusion pairs" while training on train writers only | `gm-base-v0` has memorised IAM train writers: slot accuracy **0.9976** vs **0.9268** on val; the train eligible pool is **6 HELP / 347 DAMAGE** (HELP-rate `0.0018`). Per-pair HELP-rate transfers at Spearman **ρ=0.362**. Train-time augmentation only moves accuracy to `0.9906` | "Empirically damaging" pairs **cannot** come from train writers, and R1's pairs are measured on validation writers so using them leaks the evaluation cohort. §4 substitutes the structure that *does* transfer |
| 4 | Gate B is "AUPRC > cosine baseline by a stated margin" | The margin was never stated and the baseline was never measured | Both are fixed in §6, from measurement |

**The one assumption that survived:** the base head's **top-2 competitor graph** transfers from train
to validation writers at Spearman **ρ=0.882** over 273 shared pairs, covering **97.2%** of validation
competitor mass. Which characters *compete* is a stable property of the recognizer. Which competitor
is *right* is not. §4 mines the former and refuses the latter.

## 2. The ceiling, fixed in advance so the result cannot be over- or under-read

On the 32-writer development cohort, 9,429 reference characters:

```text
top-2 eligible slots                396  (4.26% of all emitted slots)
  HELP 156   DAMAGE 189   NEITHER 51
perfect verifier, top-2          +156 corrections = 1.65pp absolute CER
accept-everything, top-2          -33 corrections = -0.35pp
Router B today                                     -0.57pp
```

A **flawless** verifier is worth at most **1.65pp** of absolute CER at this eligibility rule, and
Router B already realises 0.57pp of it. This branch is competing for roughly **1.1pp**. Relaxing to
top-3 raises the ceiling to 2.05pp but raises the accept-all cost from −0.35pp to −0.93pp; top-2 is
retained.

## 3. Glyph region extraction — `memory/glyph_regions.py`

The measured fix for assumption 1. **Midpoint tiling between adjacent glyph centers** recovers the
true per-character width: 20.05 px measured against a 20.53 px reference.

```text
centers      support side : forced-aligned span center (start_t + end_t - 1) / 2
             query   side : EmittedOccurrence.peak
cell (frames) lo = (c_prev + c)/2   or  c - 2.5  at a line start
              hi = (c + c_next)/2   or  c + 2.5  at a line end
window        fixed 40 px wide, centered on 4*c + 2, clipped to the line, background-padded
height        full 64 px. No resize, no stretch, no aspect change (docs/03-DATA.md §9)
channels      [0] the normalized line pixels
              [1] a cell mask, 1.0 inside [4*lo, 4*hi], 0.0 outside
output        [2, 64, 40] float
```

**Why a fixed 40 px window and not the cell width.** A CNN needs a fixed canvas. The measured cell is
p50 19 px / p90 30 px / p99 43 px, so 40 px holds well past p90 with genuine neighbouring context.

**Why the mask channel is not optional.** At 40 px the window always contains parts of the
neighbouring glyphs. Without the mask the network cannot know *which* glyph it is being asked about.
It costs 216 parameters.

**Why support uses forced alignment and query uses emitted peaks — at training and at inference
alike.** Enrollment has the support transcription by protocol (`docs/07-WRITER-MEMORY.md`), so forced
alignment is available on the support side in deployment. Query text is not available in deployment,
so query centers come from the base's own emission. Training uses **exactly this asymmetry**, so
there is no train/inference segmentation mismatch. This is the R2/R5A construction, unchanged.

## 4. Negative-mining policy

Mined **only from IAM train writers**, from the base head's top-2 competitor graph — the structure
measured to transfer (ρ=0.882), not the structure measured not to (ρ=0.362).

```text
scan       every emitted slot of every IAM train writer, correct or not
record     (emitted_char, runner_up_char) from EmittedOccurrence.candidates[:2]
exclude    runner-ups equal to <blank> -- blank has no glyph crop, and such slots are
           ineligible at inference anyway (memory's candidate is always a real character)
keep       pairs with >= 20 observations
hardNeg(c) = the 8 most frequent competitors of c; padded from the global competitor
           frequency if c has fewer than 8
```

Explicitly **not** done, each for a stated measured reason:

- **not** weighted by train-writer HELP/DAMAGE outcomes — ρ=0.362, and degenerate at a 0.0018 HELP rate
- **not** taken from R1's confusion pairs — measured on validation writers; that is leakage
- **not** rescued by augmentation — measured to move slot accuracy only 0.9976 → 0.9906

The three negative types the design brief requires, and where each comes from:

```text
TYPE A  same character, different writer    -> other writers in the batch (§5, L_writer)
TYPE B  same writer, confusable character   -> hardNeg(c) ∩ this writer's profile (§5, L_char)
TYPE C  empirically damaging pairs          -> NOT AVAILABLE from train writers.
                                               Substituted by empirically CONFUSABLE pairs,
                                               which is what hardNeg(c) is.
```

**Type C is a deviation from the brief and is named as one.** The brief asked for negatives drawn
from pairs that empirically damage. That quantity is only observable where the recognizer makes
errors, and on the permitted training population it makes almost none. Confusability is the closest
transferable substitute, and its transfer coefficient is measured rather than assumed.

## 5. Verifier architecture and objective

### 5a. The network

An **embedding** network, not a pairwise relation network. Two constraints force this and both are
measured, not stylistic:

- **Invariant 4 (no deployment-time gradients) plus Objective 4 (profile size).** Enrollment must be
  forward passes only. Storing raw crops for a relation comparator costs 70 chars × 3 exemplars ×
  2,560 B ≈ **537 KB**, far past the current ~59 KB mean profile. Storing one 112-d fp16 embedding
  per character costs **15.3 KB**.
- **Inference cost.** A relation network re-runs a comparator per candidate; an embedding network
  encodes the query crop once and takes cosines against stored vectors.

```text
GlyphEncoder
  input                       [B, 2, 64, 40]
  stem    Conv3x3 2->24 s2 + BN + ReLU              [B,  24, 32, 20]
  stage1  InvertedResidual2D  24-> 40 s(2,2)        [B,  40, 16, 10]
          InvertedResidual2D  40-> 40 s(1,1)
  stage2  InvertedResidual2D  40-> 80 s(2,2)        [B,  80,  8,  5]
          InvertedResidual2D  80-> 80 s(1,1)
  stage3  InvertedResidual2D  80->112 s(2,2)        [B, 112,  4,  3]
          InvertedResidual2D 112->112 s(1,1)
  head    global average pool -> Linear 112->112 -> L2 normalize
  output                      [B, 112]

parameters  284,752   (brief's envelope: 150,000-350,000)
system      1,544,560 (GM-Base) + 284,752 + ~50 (Router) = 1,829,362  vs Objective 1's 3,000,000
profile     +15.3 KB fp16 per writer
```

`InvertedResidual2D` is GM-Base's own block, reused unmodified — no new primitives.

**Enrollment (gradient-free, unchanged in kind from V0):** forced-align the 5 support lines → extract
glyph crops → forward pass → per-character mean of L2-normalized embeddings → renormalize → store.

**Inference:** extract the query glyph crop at an eligible slot → one forward pass → cosine against
the writer's stored per-character embeddings. This replaces `mem_sim`; nothing else in the path moves.

### 5b. The loss — determined by assumption 2's failure

Measurement B says a post-hoc differential destroys the signal because both similarities share a
dominant common-mode factor. The fix is to make the objective **normalised across candidates**, so
the common mode cancels in the gradient rather than in a hand-made subtraction:

```text
L_char   = -log  exp(cos(q, m_{w,c})/T) / SUM_{c' in {c} u hardNeg(c) n profile(w)} exp(cos(q, m_{w,c'})/T)
L_writer = -log  exp(cos(q, m_{w,c})/T) / SUM_{w' in batch, c in profile(w')}        exp(cos(q, m_{w',c})/T)

L = L_char + 1.0 * L_writer        T = 0.07, fixed
```

`L_char` is exactly the decision made at inference: *which character in this writer's profile does
this crop match?* `L_writer` is `docs/M10R-PLAN.md` §Q's mandatory Type-A term — without it the
network learns "what an `r` looks like" instead of "how writer 42 makes an `r`".

**Two terms, against §J's "one, not several".** The deviation is deliberate: Type-A negatives cannot
be expressed inside `L_char`, which is within-writer by construction. `λ_w ∈ {0, 1}` is the single
pre-registered ablation (§8).

## 6. Episode construction

```text
one episode      = one IAM train writer with >= 12 lines (249 such writers)
support          5 lines, forced-aligned            -> per-character mean embeddings
query            5 further, disjoint lines, base-emitted peaks -> crops labelled via align_operations
                 crops whose aligned outcome is an insertion, or whose reference character is
                 outside the profile, are dropped and COUNTED (invariant 7)
batch            8 writers per step; the other 7 supply L_writer's negatives
augmentation     the gm-base-v0 training augmentation, applied to the line before extraction
optimizer        AdamW, lr 3e-4, weight decay 1e-4, WarmupCosine, warmup 5%
steps            4,000, batch 8 writers   (~32,000 writer-episodes)
seed             1337; device resolved through runtime/device.py and logged
```

**No checkpoint selection.** The final step's weights are used. Phase 36 measured checkpoint
selection on a small probe to be *actively harmful* (−0.93pp on held-out writers); a fixed step count
cannot launder validation information into the weights. Training loss and a **train-writer** held-out
probe are logged for diagnosis only and may not select anything.

**Writer disjointness is asserted, not assumed:** no IAM validation or test writer may appear in any
episode, in the competitor graph, or in any normalisation statistic. Checked at run time.

## 7. Gate B — pre-registered, and not renegotiable after results

**Cohort: all 66 IAM validation writers.** They are unseen by the *verifier*, which trains only on
train writers. They are **not** unobserved by this project: the 32 are the R5A development cohort and
the 34 were spent once as the R5A confirmation cohort. Neither half is "fresh" or "untouched" and
neither is described that way anywhere below. All 66 are used rather than a subset because the IAM
confirmation's failure mode was insufficient power, and n≈700 HELP/DAMAGE pairs roughly doubles
n=345. The **32-development / 34-confirmation split is reported separately** as a pre-registered
secondary, so that the differing observation history of the two halves can be seen rather than
assumed away.

**Control**, computed in the same run on the same slots, defined as the best of the cheap
alternatives so the gate cannot be won against a strawman:

```text
control = argmax AUPRC over { mem_sim , mem_margin , mem_sim - base_sim }
```
On the development cohort that maximum is `mem_sim` at AUPRC `0.6006` / AUROC `0.6531`; it is
recomputed on the full 66 rather than carried over.

**PASS requires all three:**

```text
1.  AUPRC(verifier) - AUPRC(control)  >=  +0.05   AND its writer-level bootstrap 95% CI excludes 0
2.  AUROC(verifier) - AUROC(control)  >=  +0.05   AND its writer-level bootstrap 95% CI excludes 0
3.  precision @ 2% coverage           >=   0.65
```

Clause 3 is the exact bar CVL failed (`0.5425`). A verifier that cannot itself reach it has not
addressed the measured bottleneck. Bootstrap: 4,000 resamples **over writers**, never over
characters. Coverage is the share of all emitted slots intervened on, matching how every prior
precision figure in this branch was computed.

**Reported but NOT gating** — context for how integration would work, not a second chance to pass:

```text
B'  verifier vs Router B out-of-fold (AUROC 0.7275 / AUPRC 0.6671 on the development cohort)
B'' verifier CPU cost per line at the measured ~8.0 ms/crop
```

**FAIL means:** no verifier score is added to Router B, and no CER evaluation is run. Router B, `τ`,
`gm-base-v0` and the IAM test split are untouched either way at this stage.

## 8. Stop rule and the single permitted escalation

Fixed now, because `docs/M10R-PLAN.md` §E's open-ended 0→3 ladder is how a branch turns into a fourth
evaluation of the same object.

```text
Gate B PASS  -> R4b: add the verifier score to Router B as a 10th feature, refit by
                leave-one-writer-out, evaluate net CER. That refit is the FIRST permitted
                modification of Router B, and only then.
Gate B FAIL  -> diagnose ONCE, against a criterion fixed here:
                  train-writer discrimination ALSO fails  -> underfitting. One escalation
                    permitted: widen to the 350K ceiling, or add exemplars per character. One.
                  train-writer discrimination SUCCEEDS but validation fails -> transfer failure.
                    NO escalation. This is the same collapse Router B showed, reproduced in a
                    dedicated representation, and it is evidence for docs/M10R-PLAN.md §T's
                    falsification statement rather than a tuning problem.
```

No architecture search, no learning-rate sweep, no threshold search on validation writers, and no
re-run with a different eligibility rule may substitute for a failed Gate B.

## 9. Constraints held throughout

```text
Router B, tau = 0.60          frozen; touched only after Gate B PASSES, at R4b
gm-base-v0                    frozen, fingerprint fea77c9aaafd52d4 asserted before and after
IAM test split                not read
CVL                           not read at R3/R4. It is no longer a pristine confirmation set --
                              the CVL result is what motivated this branch -- so any later CVL run
                              is an EXTERNAL TRANSFER EVALUATION, reported with that history
writer disjointness           asserted at run time across episodes, competitor graph, statistics
deployment-time gradients     none: enrollment is forward passes + alignment + mean pooling
language models               none: hardNeg is a per-character visual competitor set with no
                              context, no neighbouring characters, no lexicon and no n-gram
dropped samples               counted and logged with sample_id, path and reason
```

## 10. Known risks, stated before the run

| Risk | Why it is not mitigated away |
|---|---|
| The training population is one where the recognizer is 99.76% accurate; the deployment population is one where it is 92.68% accurate | Unavoidable — every writer at a realistic error rate is an evaluation writer. §4 mines only the structure measured to transfer across that gap, and the gap itself is now measured rather than unnoticed |
| Prior observation of the 32 development writers could inflate Gate B | The 32/34 split is reported separately and pre-registered as a secondary endpoint |
| ~8.0 ms/crop adds 45-65% to a 35.5 ms line latency | Recorded as Gate B″, measured properly only at integration. A gain of ~1pp CER bought with a 50% latency increase fails the three-pillar Pareto test and would be reported as such |
| 20% of eligible slots have no profile entry for the base-emitted character | Affects any two-way formulation. The `L_char` softmax is over the profile's characters, so an absent character simply does not appear — no sentinel value is invented |
| Gate B passes and net CER still does not move | That is the R4b question and is exactly why Gate B is a discrimination gate rather than a CER gate |

---

## Addendum A — operational definitions, fixed 2026-08-24 before R4 training completed

Written while episode preparation was still running and **before any verifier weight existed or
any Gate B number was computed**. Two clauses above were under-specified in a way that would have
let the result choose its own definition afterwards. **Section 7's gate clauses are untouched.**

**A1 — what "train-side discrimination" in §8's stop rule means.** §8 distinguishes underfitting
from transfer failure by asking whether train-side discrimination also fails. HELP/DAMAGE AUPRC
cannot serve: §1 measured that IAM train writers yield 6 HELP against 347 DAMAGE, so the statistic
does not exist there. The measurable stand-in, fixed here:

```text
candidate-set character accuracy = share of query crops whose highest-cosine character, among
    ({target} u hardNeg(target)) n profile(writer), is the target

measured on  (a) 20 held-out IAM TRAIN writers, excluded from every training batch
             (b) the 66 IAM validation writers, for comparison
chance       ~= 1/9 with the registered 8 hard negatives
```

This is the verifier's own training objective evaluated on unseen writers, so it answers "did the
network learn anything at all" independently of the base recognizer's error rate. §8's branch reads:
train-side accuracy near chance -> underfitting; train-side accuracy high while Gate B fails ->
transfer or objective failure, and **no escalation**.

**A2 — which slots Gate B scores, and what the verifier's score is.** §7 requires control and
verifier "computed in the same run on the same slots". Made explicit:

```text
slot population   fixed by the FROZEN eligibility rule -- cosine memory's argmax != base AND
                  that candidate in base top-2. The verifier does not get to define its own
                  eligible set, or the comparison would not be paired
verifier score    cosine(query embedding, writer's embedding of the MEMORY CANDIDATE) -- the
                  direct analogue of mem_sim, which is the feature it would replace at R4b
```

Both arms therefore rank the identical slot list, and the writer-level bootstrap of their
difference is a paired test.
