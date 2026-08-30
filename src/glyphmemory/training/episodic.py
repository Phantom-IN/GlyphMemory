"""Episodic training: -- V0 and V1.

    SUPPORT -> encoder -> forced alignment -> pooled prototypes  (DETACHED in V0, ATTACHED
                                                                    in V1 -- the one thing
                                                                    that differs)
    QUERY   -> encoder (gradient-attached) -> generic logits
                        + memory retrieval -> gated fusion -> personalized logits
                        -> CTC loss against query text

V0 (`episodic_step`) reuses `memory/compiler.py::compile_profile` unchanged -- its own internal
``torch.no_grad()`` already is V0's detachment boundary. V1 (`episodic_step_v1`) cannot reuse
`compile_profile` the same way (it always detaches), so it is a separate support-side loop built
from the same already-tested pieces (`POOLING_STRATEGIES`, `GlyphAccumulator`, `forced_align`), with
gradient enabled and no `.detach()` call on the pooled vectors.

**Naming collision, stated once (see the phase docs for the full reasoning).** "V0"/"V1" here means
*episodic gradient strategy* (support detached vs. not), a different axis from "V0"/"V1" (feature
representation).

**`gm-base-v0.pt` is never modified.** This module always writes to a caller-specified path, never
the frozen artifact (ADR-0008) -- the caller is responsible for initializing `model` from
`gm-base-v0`'s weights and choosing a different save path.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from glyphmemory.alignment import AlignmentInfeasibleError, forced_align
from glyphmemory.config.schema import MemoryConfig
from glyphmemory.ctc.normalization import NFC_V1, normalize
from glyphmemory.ctc.tokenizer import Tokenizer
from glyphmemory.data.collate import VariableWidthCollator
from glyphmemory.data.dataset import LineDataset
from glyphmemory.data.episodes import Episode, EpisodeSampler, iter_writer_cycle
from glyphmemory.data.manifest import ManifestRecord
from glyphmemory.data.preprocessing import preprocess_path
from glyphmemory.memory.compiler import (
    FEATURE_ATTRIBUTES,
    PROJECTED_FEATURE_LAYERS,
    compile_profile,
)
from glyphmemory.memory.fusion import personalize
from glyphmemory.memory.pooling import POOLING_STRATEGIES
from glyphmemory.memory.profile import PROFILE_SCHEMA_VERSION, Glyph, WriterProfile
from glyphmemory.memory.prototypes import GlyphAccumulator
from glyphmemory.model.htr import GMBase
from glyphmemory.model.loss import CTCDiagnostics, ctc_loss
from glyphmemory.runtime.logging import get_logger
from glyphmemory.training.checkpoint import is_better
from glyphmemory.training.episodic_validation import (
    DEFAULT_PROBE_EVERY,
    ProbeCheck,
    ValidationProbe,
)
from glyphmemory.training.schedule import WarmupCosine

logger = get_logger("training.episodic")

DEFAULT_GRAD_CLIP_NORM = 5.0
DEFAULT_LEARNING_RATE = 1e-4

#: Consecutive unusable draws before the loop gives up rather than spinning.
MAX_CONSECUTIVE_SKIPS = 50


def episodic_step(
    model: GMBase,
    tokenizer: Tokenizer,
    episode: Episode,
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    memory_config: MemoryConfig,
    device: torch.device,
) -> tuple[Tensor, CTCDiagnostics, WriterProfile]:
    """One V0 episodic forward/loss step for a single episode.

    Args:
        model: Trained in place by the caller's own ``loss.backward()`` / optimizer step — this
            function only computes the forward pass and loss, matching
            `model/loss.py::ctc_loss_for`'s own "compute, do not step" shape.
        memory_config: Must have ``enabled=True`` — a disabled config would make
            `memory.fusion.personalize` return the base logits unchanged, silently training a step
            that never touches the memory mechanism at all.

    Returns:
        ``(loss, diagnostics, profile)``. ``profile`` is returned so a caller (or a test) can
        inspect the detached support-side prototypes directly, not just the loss.

    Raises:
        ValueError: ``memory_config.enabled`` is ``False``, or the episode's query lines are all
            CTC-infeasible (an empty batch — should not happen on real manifest data already
            filtered by pipeline, but never silently trains on nothing).
    """
    if not memory_config.enabled:
        raise ValueError(
            "episodic_step requires memory_config.enabled=True -- otherwise personalize() "
            "returns the base logits unchanged and this step would silently never train the "
            "memory mechanism at all."
        )

    charset = tokenizer.charset
    support_lines = [
        (preprocess_path(records_by_id[sid].image).tensor, records_by_id[sid].text)
        for sid in episode.support_ids
    ]
    # compile_profile's own internal `torch.no_grad()` is V0's detachment boundary -- nothing
    # additional is needed here to keep the support path out of the surrounding training graph.
    profile = compile_profile(
        model,
        charset,
        support_lines,
        model_fingerprint=model_fingerprint,
        config=memory_config,
        device=device,
    )

    query_records = tuple(records_by_id[sid] for sid in episode.query_ids)
    query_dataset = LineDataset(records=query_records, tokenizer=tokenizer)
    samples = [query_dataset[i] for i in range(len(query_dataset))]
    batch = VariableWidthCollator(training=True, pad_value=query_dataset.pad_value)(samples)
    batch = batch.to(device)
    if batch.is_empty:
        raise ValueError(
            f"episode for writer {episode.writer_id!r} has no CTC-feasible query line "
            f"(requested {len(episode.query_ids)}); cannot compute a loss from nothing."
        )

    output = model(batch.images, batch.input_lengths)  # gradient-attached: no no_grad here
    corrected = personalize(output, profile, charset, memory_config)
    loss, diagnostics = ctc_loss(
        corrected, batch.targets, output.input_lengths, batch.target_lengths
    )
    return loss, diagnostics, profile


def episodic_step_v1(
    model: GMBase,
    tokenizer: Tokenizer,
    episode: Episode,
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    memory_config: MemoryConfig,
    device: torch.device,
) -> tuple[Tensor, CTCDiagnostics, WriterProfile]:
    """One V1 episodic forward/loss step: gradient flows through the support side's feature and
    pooling path too, not just query/fusion ("allow gradients through support feature projections
    and prototype creation").

    Cannot reuse `compile_profile` the way `episodic_step` (V0) does -- `compile_profile` always
    runs its support loop under `torch.no_grad()` and detaches every pooled vector before
    accumulating it, which is exactly the property V1 needs to *not* have. This function is
    therefore a genuinely separate support-side loop, not merely a call to `compile_profile` with a
    flag flipped -- but it reuses every already-tested piece `compile_profile` itself is built from
    (`memory/pooling.py::POOLING_STRATEGIES`, `memory/prototypes.py::GlyphAccumulator`,
    `alignment/forced_align`), so the only genuinely new code is the orchestration: which tensor
    stays detached and which does not.

    **Forced alignment stays fixed/non-differentiable, in both V0 and V1** -- `docs/
    07-WRITER-MEMORY.md` is explicit that attempting differentiable CTC alignment is out of scope
    for this phase. `forced_align` is called on `log_probs.detach()`, a plain detached copy, so the
    Viterbi search itself never enters the autograd graph; the *pooling* that follows
    (`POOLING_STRATEGIES`, e.g. `posterior_weighted`'s confidence-weighted sum) is called with the
    original, gradient-attached `log_probs` and `features`, so gradient flows back through the
    pooling weights and the raw features into the encoder -- V1's defining difference from V0.

    **The support pass runs in `model.eval()`, not `model.train()`, even though gradient is
    enabled.** `.eval()`/`.train()` controls BatchNorm/dropout behavior, which is independent of
    autograd tracking (`torch.no_grad()`/`.detach()` controls that).

    Args:
        memory_config: Must have ``enabled=True`` (see `episodic_step`) and
            ``prototype_strategy="mean"`` -- V1 tests episodic *gradient strategy*, not
            prototype-variant axis (naming-collision note applies here too), so this function only
            supports the plain running-sum accumulator (`GlyphAccumulator`), which is
            autograd-transparent by construction (no `.detach` anywhere in its own
            `observe`/`finalize`).

    Returns:
        ``(loss, diagnostics, profile)`` -- unlike V0's, ``profile``'s prototype tensors here
        carry live gradient tracking (`requires_grad=True`, a non-`None` `grad_fn`) back to
        `model`'s parameters, proven directly in this module's own tests.

    Raises:
        ValueError: ``memory_config.enabled`` is ``False``, ``memory_config.prototype_strategy`` is
            not ``"mean"``, ``memory_config.feature_layer`` is not ``"sequence"``/``"visual"`` (the
            learned projection is axis, out of scope here), or the episode's query lines are all
            CTC-infeasible.
    """
    if not memory_config.enabled:
        raise ValueError(
            "episodic_step_v1 requires memory_config.enabled=True -- otherwise personalize() "
            "returns the base logits unchanged and this step would silently never train the "
            "memory mechanism at all."
        )
    if memory_config.prototype_strategy != "mean":
        raise ValueError(
            f"episodic_step_v1 only supports prototype_strategy='mean' (GlyphAccumulator's "
            f"autograd-transparent running-sum path), got {memory_config.prototype_strategy!r} "
            "prototype-variant axis is out of scope for (naming-collision note)."
        )
    if memory_config.feature_layer not in FEATURE_ATTRIBUTES:
        raise ValueError(
            f"Unknown feature_layer {memory_config.feature_layer!r}; "
            f"expected one of {sorted(FEATURE_ATTRIBUTES)}."
        )
    if memory_config.feature_layer in PROJECTED_FEATURE_LAYERS:
        raise ValueError(
            f"episodic_step_v1 does not support feature_layer "
            f"{memory_config.feature_layer!r} -- the learned projection is axis, out of "
            "scope for episodic gradient-strategy ablation (naming-collision note)."
        )

    charset = tokenizer.charset
    pool = POOLING_STRATEGIES[memory_config.pooling]
    feature_attribute = FEATURE_ATTRIBUTES[memory_config.feature_layer]

    was_training = model.training
    model.eval()  # matches V0's support-pass convention; gradient tracking is independent of this
    accumulator = GlyphAccumulator()
    feature_dim: int | None = None
    try:
        for sample_id in episode.support_ids:
            record = records_by_id[sample_id]
            reference = normalize(record.text, NFC_V1)
            image = preprocess_path(record.image).tensor
            batch = image.unsqueeze(0).to(device)
            output = model(batch)  # gradient-attached: no no_grad here, V1's defining property
            length = int(output.input_lengths[0])

            features = getattr(output, feature_attribute)[0, :length]
            if feature_dim is None:
                feature_dim = int(features.shape[-1])
            logits = output.logits[0, :length]
            log_probs = torch.log_softmax(logits, dim=-1)

            try:
                alignment = forced_align(log_probs.detach(), reference, charset)
            except AlignmentInfeasibleError as error:
                raise AlignmentInfeasibleError(
                    f"support line sample_id={sample_id!r} path={record.image!r}: {error}"
                ) from error
            for span in alignment.spans:
                class_index = charset.index_of(span.token)
                vector = pool(span, log_probs, features, class_index)  # gradient-attached
                accumulator.observe(span.token, vector, span.score)  # NOT detached: V1 vs. V0
    finally:
        model.train(was_training)

    if feature_dim is None:
        raise ValueError(
            f"episode for writer {episode.writer_id!r} has no support line at all "
            f"(requested {len(episode.support_ids)})."
        )

    glyphs = {
        character: Glyph(
            character=character,
            prototype=prototype,
            number_of_observations=count,
            mean_alignment_confidence=confidence,
            feature_layer=memory_config.feature_layer,
        )
        for character, (prototype, count, confidence) in accumulator.finalize().items()
    }
    profile = WriterProfile(
        schema_version=PROFILE_SCHEMA_VERSION,
        model_fingerprint=model_fingerprint,
        feature_layer=memory_config.feature_layer,
        feature_dim=feature_dim,
        glyphs=glyphs,
    )

    query_records = tuple(records_by_id[sid] for sid in episode.query_ids)
    query_dataset = LineDataset(records=query_records, tokenizer=tokenizer)
    samples = [query_dataset[i] for i in range(len(query_dataset))]
    batch = VariableWidthCollator(training=True, pad_value=query_dataset.pad_value)(samples)
    batch = batch.to(device)
    if batch.is_empty:
        raise ValueError(
            f"episode for writer {episode.writer_id!r} has no CTC-feasible query line "
            f"(requested {len(episode.query_ids)}); cannot compute a loss from nothing."
        )

    output = model(batch.images, batch.input_lengths)
    corrected = personalize(output, profile, charset, memory_config)
    loss, diagnostics = ctc_loss(
        corrected, batch.targets, output.input_lengths, batch.target_lengths
    )
    return loss, diagnostics, profile


@dataclass(frozen=True, slots=True)
class EpisodicStepLog:
    """One completed (non-skipped) **optimizer** step's outcome.

    Without accumulation that is also one episode, and every field means what it says. The gradient
    this step applied came from all of them, so attributing ``grad_norm`` to a single writer or
    support size is not meaningful when ``episodes > 1`` — stated here rather than silently invited
    by the field names. ``query_lines`` is the total across the group and ``loss`` their mean.
    """

    step: int
    writer_id: str
    support_size: int
    query_lines: int
    loss: float
    grad_norm: float
    clipped: bool
    #: The rate this step actually ran at.
    learning_rate: float = DEFAULT_LEARNING_RATE
    #: Episodes accumulated into this one optimizer step. 1 unless accumulation is in use.
    episodes: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "writer_id": self.writer_id,
            "support_size": self.support_size,
            "query_lines": self.query_lines,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "clipped": self.clipped,
            "learning_rate": self.learning_rate,
            "episodes": self.episodes,
        }


@dataclass(frozen=True, slots=True)
class EpisodicTrainingLog:
    """A completed `train_episodic_v0` run, in the same "state it, don't imply it" shape
    `training/trainer.py::EpochStats` uses for the corpus-shuffle loop.
    """

    steps: tuple[EpisodicStepLog, ...]
    skipped_steps: int
    skipped_writers: tuple[str, ...]
    seconds: float
    #: Of `skipped_steps`, how many were skipped because a **support** line could not be
    #: force-aligned (its transcript does not fit its frame count).
    skipped_alignment: int = 0
    #: Periodic generic-recognition checks, if a `ValidationProbe` was supplied.
    probe_checks: tuple[ProbeCheck, ...] = ()
    #: With `select_best`, the step whose weights this run ended on.
    selected_step: int | None = None
    #: The probe CER at `selected_step`.
    selected_cer: float | None = None
    #: The learning-rate schedule this run used, if any.
    schedule: WarmupCosine | None = None

    @property
    def n_steps(self) -> int:
        """**Optimizer** steps. Under accumulation this is not the number of episodes trained on —
        see `n_episodes`, which is the honest "how much data did this run touch" number.
        """
        return len(self.steps)

    @property
    def n_episodes(self) -> int:
        """Episodes actually trained on. Equals `n_steps` without accumulation."""
        return sum(step.episodes for step in self.steps)

    @property
    def probe_seconds(self) -> float:
        """Wall-clock spent inside the diagnostic probe, **included** in ``seconds``.

        Reported separately so the probe's own cost is visible rather than silently inflating this
        run's throughput — the phase doc's own risk about a diagnostic distorting the run it
        measures is answered with a number, not an assurance.
        """
        return sum(check.seconds for check in self.probe_checks)

    @property
    def training_seconds(self) -> float:
        """``seconds`` minus the probe's own cost — comparable to an uninstrumented run's."""
        return self.seconds - self.probe_seconds

    @property
    def clip_rate(self) -> float:
        return sum(1 for s in self.steps if s.clipped) / self.n_steps if self.steps else 0.0

    @property
    def mean_loss(self) -> float | None:
        return sum(s.loss for s in self.steps) / self.n_steps if self.steps else None

    @property
    def final_loss(self) -> float | None:
        return self.steps[-1].loss if self.steps else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "n_episodes": self.n_episodes,
            "skipped_steps": self.skipped_steps,
            "skipped_alignment": self.skipped_alignment,
            "skipped_writers": list(self.skipped_writers),
            "clip_rate": round(self.clip_rate, 6),
            "mean_loss": self.mean_loss,
            "final_loss": self.final_loss,
            "seconds": round(self.seconds, 3),
            "probe_seconds": round(self.probe_seconds, 3),
            "training_seconds": round(self.training_seconds, 3),
            "probe_checks": [check.as_dict() for check in self.probe_checks],
            "selected_step": self.selected_step,
            "selected_cer": self.selected_cer,
            "schedule": None if self.schedule is None else self.schedule.describe(),
        }


def _run_episodic_training(
    step_fn: Callable[..., tuple[Tensor, CTCDiagnostics, WriterProfile]],
    model: GMBase,
    tokenizer: Tokenizer,
    sampler: EpisodeSampler,
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    n_steps: int,
    memory_config: MemoryConfig | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    seed: int = 1337,
    device: torch.device | str = "cpu",
    probe: ValidationProbe | None = None,
    probe_every: int = DEFAULT_PROBE_EVERY,
    schedule: WarmupCosine | None = None,
    accumulation_steps: int = 1,
    select_best: bool = False,
) -> EpisodicTrainingLog:
    """The training loop shared by V0 (`train_episodic_v0`) and V1 (`train_episodic_v1`) --
    identical in every respect except which per-episode ``step_fn`` it calls (`episodic_step` vs.
    `episodic_step_v1`), so the loop itself (optimizer, gradient clipping, non-finite-loss handling,
    episode sampling) is written once, not duplicated.

    Reuses `training/trainer.py`'s own conventions (gradient clipping counted not just applied, a
    non-finite loss skipped and counted rather than absorbed) for an episodic, not corpus-shuffle,
    batch shape: one sampled `Episode` per step, drawn from `data/episodes.py::iter_writer_cycle`
    rather than a `DataLoader` epoch.

    Args:
        model: Mutated in place. Caller is responsible for having initialized it from `gm-base-v0`'s
            weights (or wherever else) -- this function never loads or saves
            `artifacts/gm-base-v0.pt` itself.
        n_steps: A stated, bounded step count -- this project's own "no invented numbers, no
            overclaiming" discipline means the caller states this plainly rather than this function
            pretending to run "enough.".
        probe: Optional periodic generic-recognition diagnostic. **Read-only with respect to
            training**: it consumes no RNG (its batches are preprocessed once, unaugmented, at
            construction), takes no optimizer step, and restores the model's training mode, so an
            instrumented run's weights are bit-identical to an uninstrumented one's -- asserted in
            this module's own tests, not merely intended.
        probe_every: Steps between probe checks. A check also runs at step 0 (before any update, the
            run's own zero-step reference) and after the final step, so the curve always has both
            endpoints regardless of whether ``n_steps`` divides evenly.
        schedule: Optional step-based learning-rate schedule, applied through a plain ``LambdaLR``
            as a *multiplier* on ``learning_rate``. `training/schedule.py::WarmupCosine` reused
            unchanged rather than reimplemented: it is already step-driven (not epoch-driven), so it
            fits this loop's shape directly.
        accumulation_steps: Episodes whose gradients accumulate before one optimizer step
            (Configuration B). Each episode contributes ``loss / accumulation_steps``, so the
            accumulated gradient is the group's **mean**, not its sum -- summing would also scale
            the gradient by ``k``, confounding "larger effective batch" with "``k`` times the
            learning rate". **``n_steps`` counts optimizer steps, so a ``k``-way run trains on
            ``n_steps * k`` episodes**: read `EpisodicTrainingLog.n_episodes`, not `n_steps`, for
            how much data a run touched. 1 (the default) is one-episode-per-step loop exactly.
        select_best: End the run on its best-by-probe-CER checkpoint rather than its final one,
            extending `training/trainer.py`'s own "selection is by validation CER, never by loss"
            rule to the episodic loop, which has never had it. Requires ``probe``. The best-so-far
            state is kept **in memory**, not written to disk on each improvement: the model is ~6 MB
            and episodic probe checks are far more frequent than the standard trainer's per-epoch
            validations, so the ``best.pt``/``last.pt`` file-pair pattern would buy nothing here.
            **The step-0 check is not a candidate** -- see the comment at the selection site for why
            admitting it would turn this into "do not train".

    Raises:
        ValueError: ``schedule.total_steps`` disagrees with ``n_steps`` -- a cosine computed against
            the wrong horizon decays to the wrong place without ever erroring
            (`training/schedule.py`'s own module docstring), so the mismatch is refused here rather
            than silently mistrained -- or ``accumulation_steps`` is below 1.
    """
    resolved_config = memory_config or MemoryConfig(
        enabled=True, feature_layer="sequence", prototype_strategy="mean"
    )
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    model.to(resolved_device)
    model.train()

    if accumulation_steps < 1:
        raise ValueError(f"accumulation_steps must be at least 1, got {accumulation_steps}")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = None
    if schedule is not None:
        if schedule.total_steps != n_steps:
            raise ValueError(
                f"schedule.total_steps ({schedule.total_steps}) must equal n_steps ({n_steps}); "
                "a schedule computed against the wrong horizon mistrains silently."
            )
        scheduler = LambdaLR(optimizer, lr_lambda=schedule)
    cycle = iter_writer_cycle(sorted(sampler.writers), seed=seed)

    if probe is not None and probe_every < 1:
        raise ValueError(f"probe_every must be positive when a probe is given, got {probe_every}")
    if select_best and probe is None:
        raise ValueError(
            "select_best requires a probe -- selection is by validation CER, and without a probe "
            "there is no validation CER to select on. Selecting on training loss instead is "
            "exactly what forbids."
        )

    steps: list[EpisodicStepLog] = []
    skipped_steps = 0
    skipped_alignment = 0
    skipped_writers: list[str] = []
    probe_checks: list[ProbeCheck] = []
    draw_index = 0
    consecutive_skips = 0
    best_cer: float | None = None
    best_step: int | None = None
    best_state: dict[str, Tensor] | None = None
    started = time.perf_counter()

    if probe is not None:
        probe_checks.append(probe.evaluate(model, step=0))

    while len(steps) < n_steps:
        optimizer.zero_grad(set_to_none=True)
        group_writer: str | None = None
        group_support = 0
        group_query_lines = 0
        group_losses: list[float] = []

        while len(group_losses) < accumulation_steps:
            if consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
                raise RuntimeError(
                    f"{consecutive_skips} consecutive unusable draws at step {len(steps)}; "
                    "refusing to keep spinning. This is a systematically infeasible "
                    "configuration (see the logged per-skip reasons), not bad luck."
                )
            writer_id = next(cycle)
            try:
                episode = sampler.sample(writer_id, draw_index)
            except ValueError:
                skipped_steps += 1
                skipped_writers.append(writer_id)
                consecutive_skips += 1
                draw_index += 1
                continue
            draw_index += 1

            try:
                loss, diagnostics, _profile = step_fn(
                    model,
                    tokenizer,
                    episode,
                    records_by_id,
                    model_fingerprint=model_fingerprint,
                    memory_config=resolved_config,
                    device=resolved_device,
                )
            except AlignmentInfeasibleError as error:
                # A support line whose transcript cannot fit its frame count. The *query* side
                # already tolerates this (the training collator drops-with-counter), but the support
                # side calls `forced_align` directly and would otherwise kill the run.
                skipped_steps += 1
                skipped_alignment += 1
                skipped_writers.append(writer_id)
                consecutive_skips += 1
                logger.warning(
                    "[alignment_infeasible] step=%d writer=%s support_ids=%s: %s",
                    len(steps),
                    writer_id,
                    list(episode.support_ids),
                    error,
                )
                continue
            consecutive_skips = 0

            if not torch.isfinite(loss):
                skipped_steps += 1
                consecutive_skips += 1
                logger.warning(
                    "Non-finite loss at episodic step %d (writer %s); skipping.",
                    len(steps),
                    writer_id,
                )
                continue

            (loss / accumulation_steps).backward()
            if group_writer is None:
                group_writer = writer_id
                group_support = episode.support_size
            group_query_lines += diagnostics.batch_size
            group_losses.append(float(loss.detach()))

        norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        clipped = float(norm) > grad_clip_norm
        # Read the rate *before* stepping the scheduler: this is the rate this update ran at.
        step_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        assert group_writer is not None  # the inner loop only exits with a full group
        steps.append(
            EpisodicStepLog(
                step=len(steps),
                writer_id=group_writer,
                support_size=group_support,
                query_lines=group_query_lines,
                loss=sum(group_losses) / len(group_losses),
                grad_norm=float(norm),
                clipped=clipped,
                learning_rate=step_rate,
                episodes=len(group_losses),
            )
        )

        if probe is not None and (len(steps) % probe_every == 0 or len(steps) == n_steps):
            check = probe.evaluate(model, step=len(steps))
            probe_checks.append(check)
            # Candidacy begins after the first optimizer step, deliberately. The step-0 check is the
            # *untrained* starting weights, and on a generic-recognition probe those are the best
            # point of any episodic run measured so far -- so admitting step 0 would turn
            # "checkpoint selection" into "do not train", discarding the personalization the run
            # exists to produce along with the regression. Step 0 stays in `probe_checks` as the
            # reference it is; it is simply not a thing this can return.
            if select_best and is_better(check.cer, best_cer):
                best_cer = check.cer
                best_step = check.step
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

    if select_best and best_state is not None:
        model.load_state_dict(best_state)
        # The final check's CER may be None (a validation that produced no value); `best_cer` cannot
        # be, since `is_better` never lets None win. Formatting the two the same way would raise
        # inside logging on that path.
        final_cer = probe_checks[-1].cer
        logger.info(
            "Selected the step-%d checkpoint (probe CER %.5f) over the final step-%d state "
            "(probe CER %s).",
            best_step,
            best_cer,
            len(steps),
            "n/a" if final_cer is None else f"{final_cer:.5f}",
        )

    elapsed = time.perf_counter() - started
    log = EpisodicTrainingLog(
        steps=tuple(steps),
        skipped_steps=skipped_steps,
        skipped_alignment=skipped_alignment,
        skipped_writers=tuple(skipped_writers),
        seconds=elapsed,
        probe_checks=tuple(probe_checks),
        selected_step=best_step,
        selected_cer=best_cer,
        schedule=schedule,
    )
    if log.clip_rate > 0.5:
        logger.warning(
            "Gradient clipping fired on %.0f%% of steps. That is instability, not a threshold "
            "problem.",
            100 * log.clip_rate,
        )
    return log


def train_episodic_v0(
    model: GMBase,
    tokenizer: Tokenizer,
    sampler: EpisodeSampler,
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    n_steps: int,
    memory_config: MemoryConfig | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    seed: int = 1337,
    device: torch.device | str = "cpu",
    probe: ValidationProbe | None = None,
    probe_every: int = DEFAULT_PROBE_EVERY,
    schedule: WarmupCosine | None = None,
    accumulation_steps: int = 1,
    select_best: bool = False,
) -> EpisodicTrainingLog:
    """V0 episodic training (support detached) for ``n_steps``, mutating ``model`` in place. See
    `_run_episodic_training` for the shared loop and `episodic_step` for the per-step boundary this
    uses.
    """
    return _run_episodic_training(
        episodic_step,
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=model_fingerprint,
        n_steps=n_steps,
        memory_config=memory_config,
        learning_rate=learning_rate,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
        device=device,
        probe=probe,
        probe_every=probe_every,
        schedule=schedule,
        accumulation_steps=accumulation_steps,
        select_best=select_best,
    )


def train_episodic_v1(
    model: GMBase,
    tokenizer: Tokenizer,
    sampler: EpisodeSampler,
    records_by_id: dict[str, ManifestRecord],
    *,
    model_fingerprint: str,
    n_steps: int,
    memory_config: MemoryConfig | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
    seed: int = 1337,
    device: torch.device | str = "cpu",
    probe: ValidationProbe | None = None,
    probe_every: int = DEFAULT_PROBE_EVERY,
    schedule: WarmupCosine | None = None,
    accumulation_steps: int = 1,
    select_best: bool = False,
) -> EpisodicTrainingLog:
    """V1 episodic training (gradient through support's feature/pooling path too) for ``n_steps``,
    mutating ``model`` in place. See `_run_episodic_training` for the shared loop and
    `episodic_step_v1` for the per-step boundary this uses.
    """
    return _run_episodic_training(
        episodic_step_v1,
        model,
        tokenizer,
        sampler,
        records_by_id,
        model_fingerprint=model_fingerprint,
        n_steps=n_steps,
        memory_config=memory_config,
        learning_rate=learning_rate,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
        device=device,
        probe=probe,
        probe_every=probe_every,
        schedule=schedule,
        accumulation_steps=accumulation_steps,
        select_best=select_best,
    )
