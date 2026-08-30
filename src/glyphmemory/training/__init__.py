"""Training loop, schedule, checkpointing and run records.

One optimizer, one schedule, one device.
"""

from glyphmemory.training.checkpoint import (
    BEST_FILENAME,
    CHECKPOINT_SCHEMA_VERSION,
    LAST_FILENAME,
    SELECTION_METRIC,
    CheckpointCompatibilityError,
    CheckpointMeta,
    LoadedCheckpoint,
    is_better,
    load_checkpoint,
    restore_model,
    save_checkpoint,
)
from glyphmemory.training.episodic import (
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LEARNING_RATE,
    EpisodicStepLog,
    EpisodicTrainingLog,
    episodic_step,
    episodic_step_v1,
    train_episodic_v0,
    train_episodic_v1,
)
from glyphmemory.training.episodic_validation import (
    DEFAULT_LINES_PER_WRITER,
    DEFAULT_PROBE_BATCH_SIZE,
    DEFAULT_PROBE_EVERY,
    DEFAULT_PROBE_WRITERS,
    HARNESS_WRITERS,
    ProbeCheck,
    ValidationProbe,
    select_probe_records,
)
from glyphmemory.training.run_record import (
    REQUIRED_FIELDS,
    build_run_record,
    manifest_fingerprints,
    missing_fields,
)
from glyphmemory.training.schedule import (
    WarmupCosine,
    build_scheduler,
    warmup_steps_for,
)
from glyphmemory.training.trainer import (
    DEFAULT_PREVIEW_SAMPLES,
    EpochStats,
    Trainer,
    ValidationStats,
)

__all__ = [
    "BEST_FILENAME",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_GRAD_CLIP_NORM",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_LINES_PER_WRITER",
    "DEFAULT_PREVIEW_SAMPLES",
    "DEFAULT_PROBE_BATCH_SIZE",
    "DEFAULT_PROBE_EVERY",
    "DEFAULT_PROBE_WRITERS",
    "HARNESS_WRITERS",
    "LAST_FILENAME",
    "REQUIRED_FIELDS",
    "SELECTION_METRIC",
    "CheckpointCompatibilityError",
    "CheckpointMeta",
    "EpisodicStepLog",
    "EpisodicTrainingLog",
    "EpochStats",
    "LoadedCheckpoint",
    "ProbeCheck",
    "Trainer",
    "ValidationProbe",
    "ValidationStats",
    "WarmupCosine",
    "build_run_record",
    "build_scheduler",
    "episodic_step",
    "episodic_step_v1",
    "is_better",
    "load_checkpoint",
    "manifest_fingerprints",
    "missing_fields",
    "restore_model",
    "save_checkpoint",
    "select_probe_records",
    "train_episodic_v0",
    "train_episodic_v1",
    "warmup_steps_for",
]
