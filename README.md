# GlyphMemory

Lightweight neural handwritten text recognition with persistent, gradient-free few-shot writer
adaptation — and a measurement of when that adaptation can be trusted.

> General handwriting knowledge lives in model weights.
> Individual writer knowledge lives in external memory.

This repository accompanies the manuscript **"The Glyph Is Not the Error: Writer-Conditioned
Identity Versus Corrective Utility in Handwriting Recognition"** (under review). It contains the
source code, the pre-registered experimental protocols, the committed result artifacts, the
training configurations, and the analysis code used to produce the reported intervals.

---

## The question

A generic handwriting recognizer must transcribe writers it has never seen. A few labelled lines
from a new writer plainly contain writer-specific information — but exploiting it requires more than
showing that the information exists. The recognizer has already formed a decision at every character
it emits, and most of those decisions are correct. The system must determine *when* writer evidence
should overturn a decision already made, and being wrong about that costs a correct character.

The work separates four quantities that a single error rate conflates:

```text
PRESENCE  ->  COMPLEMENTARITY  ->  ACCESSIBILITY  ->  RELIABILITY
is writer      is it right       can an inference    are the interventions
structure      where the         signal find it?     reliable enough
there?         recognizer                            to act on?
               is wrong?
```

These are used as distinct questions organising this investigation, not as a proposed universal
theory of personalization.

## What was found

- Writer-specific structure survives in a recognizer trained for writer-independent transcription,
  and it is **complementary** rather than redundant — writer memory holds the correct answer for
  41.7% of the recognizer's character errors on the development cohort.
- Making that complementarity actionable does not make it reliable. In a pre-registered evaluation
  on **308 writers of an external corpus**, selective use of writer memory produced a small,
  statistically resolved mean improvement — while its intervention precision was 0.5425 and 115 of
  the 308 writers ended worse than they began.
- The natural repair fails. Two independently constructed representations — a learned projection over
  frozen features, and a 284,752-parameter pixel verifier trained from scratch — each improved on the
  **writer-identity objective they were given** while their usefulness for deciding *when to correct*
  declined.

Negative results are kept, not discarded. Every failed gate, and the clause that decided it, is
recorded in the pre-registration and result artifacts under `docs/results/`.

## Design constraints

These were fixed at the start and enforced throughout. Breaking one is a scientific error, not a bug.

1. **No language models anywhere in the recognition path.** No LLM, LM-fused decoder, lexicon,
   dictionary or spell correction. The system transcribes what is visually written, not what is
   linguistically likely.
2. **No pretrained OCR/HTR models.** No TrOCR, PaddleOCR, EasyOCR, cloud OCR, or pretrained
   backbones. Weights start randomly initialised.
3. **Parameter ceiling of 3.0M**, enforced by a test rather than by good intentions. The released
   recognizer is 1,544,560 parameters.
4. **No deployment-time gradients.** Enrolling a writer is forward passes, alignment and prototype
   compilation — no optimizer, no backpropagation, no per-writer checkpoint.
5. **Writer identifiers never leak across splits.** Disjointness is asserted in code, not assumed.
6. **No invented numbers.** No error rate, latency, size or gain is stated that was not produced by a
   command in this repository.
7. **No silent data handling.** Nothing is dropped without a logged reason and a counter.
8. **Negative results are preserved.**

## The recognizer

```text
line image -> CNN encoder -> 2-layer BiGRU -> linear head -> CTC -> text
                    + optional -> writer profile retrieval -> selective correction
```

| | |
|---|---|
| Parameters | 1,544,560 (5.89 MB FP32) |
| IAM test | 7.24% character error rate, 26.32% word error rate |
| Decoding | greedy; no language model, no lexicon |
| CPU latency | 35.5 ms median, batch 1 |
| Writer state | per-character mean prototypes, 49,920 bytes/writer at five support lines |
| Enrollment | five transcribed lines, gradient-free |

The IAM test split was read **once**, for the row above. No later experiment reads it.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) for environment management
- Apple silicon (MPS), NVIDIA (CUDA) or CPU — all three supported through one abstraction

## Installation

```bash
git clone https://github.com/Phantom-IN/GlyphMemory.git
cd GlyphMemory
uv sync
```

Development tooling (pytest, ruff) is a `dev` dependency group, installed by default. Use
`uv sync --no-dev` for a runtime-only environment.

## Usage

```bash
uv run glyphmemory --help
```

| Command | What it does |
|---|---|
| `version` | print the version |
| `info` | show the resolved device and captured environment |
| `config show` | validate and print a configuration file |
| `data make-synthetic` | generate a synthetic corpus |
| `train` | train the recognizer, validating by character error rate |
| `evaluate` | evaluate a checkpoint: error rates, per-writer breakdown, error taxonomy |
| `few-shot` | few-shot adaptation curves and gain statistics |
| `rival-baselines` | head-only, BatchNorm-only and full fine-tuning against writer memory |
| `benchmark` | latency, throughput and memory |

`train`, `evaluate`, `few-shot` and `rival-baselines` need a dataset you have obtained yourself; the
rest run on a clean checkout.

```bash
uv run glyphmemory info                 # resolved device and environment
uv run glyphmemory info --device cpu
uv run glyphmemory config show configs/default.yaml
```

`--device` accepts `auto`, `cpu`, `mps` or `cuda`. An explicitly requested but unavailable
accelerator is an error rather than a silent fall back to CPU.

Generate a synthetic corpus, which needs no dataset download and no system fonts:

```bash
uv run glyphmemory data make-synthetic --out /tmp/synth --writers 3 --lines 4 --seed 1337
```

**Synthetic data is a correctness harness, never a benchmark.** Every record is labelled
`dataset="synthetic"` and never appears in a reported result.

## Reproducing the reported intervals

`scripts/reproduce/` contains the writer-level bootstrap and the selective policy, **recovered
verbatim from the scripts that produced the published numbers** rather than reimplemented. Two
details there are load-bearing and deliberately not tidied: the resampling unit is always the writer,
never the character; and two different upper-percentile index conventions were used historically and
are preserved as separate functions.

```bash
uv run pytest tests/test_reproduce_bootstrap.py tests/test_reproduce_router_b.py
```

These tests recompute published statistics from artifacts committed in `docs/results/` and check them
against the recorded values. They read no dataset and load no model.

## Testing

```bash
uv run pytest            # 77 test modules
uv run ruff check .
```

## Layout

```text
src/glyphmemory/     library: model, data, ctc, alignment, memory, evaluation, probes, training
tests/               77 test modules
scripts/reproduce/   recovered analysis code for the published intervals
configs/             training configurations, including the one behind the released recognizer
docs/results/        pre-registered protocols and committed result artifacts
artifacts/           charset definition (trained weights are not redistributed)
```

## Datasets

The IAM and CVL handwriting databases are **third-party corpora and are not redistributed here**.
Both must be obtained from their maintainers under the access conditions those maintainers specify.
No new human-subject data was collected for this work.

## Model weights

The exact trained checkpoint used for the reported experiments is **not redistributed**. The base
training configuration is in `configs/` so the recognizer can be retrained — note that a retrained
model will not carry the fingerprint (`fea77c9aaafd52d4`) that every committed result records.

## Pre-registration

Five protocol documents were written and staged before the runs they govern, two of them carrying
later addenda, each fixing the cohort, the endpoint, the decision threshold and the stop rule in
advance. They sit in `docs/results/` alongside the result artifacts they gate. Three of those gates
*forbade further work* rather than licensing it.

Pre-registration constrains researcher degrees of freedom. It does not make a result correct.

## Citing this work

The manuscript is under review. Until it appears, please cite the repository:

```bibtex
@software{vanage2026glyphmemory,
  author  = {Vanage, Vaibhav},
  title   = {{GlyphMemory}: writer-conditioned identity versus corrective utility
             in handwriting recognition},
  year    = {2026},
  url     = {https://github.com/Phantom-IN/GlyphMemory}
}
```

## Contact

Vaibhav Vanage — Independent Researcher, India.
[vaibhav.vanage@gmail.com](mailto:vaibhav.vanage@gmail.com) ·
[ORCID 0009-0001-6740-2480](https://orcid.org/0009-0001-6740-2480)

Questions about the protocols or the reproducibility code are welcome as GitHub issues.

## License

[Apache-2.0](LICENSE) for the source code. Trained model weights are derived from third-party datasets and
would carry a separate licensing review before any release.
