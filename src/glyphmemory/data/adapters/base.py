"""Dataset adapter interface.

An adapter converts one source corpus into the internal manifest and **does nothing else**. No
preprocessing, no tokenization, no splitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Separates the dataset prefix from the corpus-local identifier in a sample_id.
SAMPLE_ID_SEPARATOR = "/"


@runtime_checkable
class DatasetAdapter(Protocol):
    """Converts a source corpus into a GlyphMemory manifest."""

    #: Short dataset identifier written to every record's ``dataset`` field.
    name: str

    def prepare(self, source_dir: Path, output_dir: Path) -> Path:
        """Convert the corpus at ``source_dir`` into a manifest under ``output_dir``.

        Returns the path to the written ``manifest.jsonl``. Implementations must route every
        rejected sample through :class:`~glyphmemory.data.validation.IntegrityCounters` rather than
        dropping it.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Provenance for the run record: source layout, version, options, exclusions."""
        ...


def make_sample_id(dataset: str, local_id: str) -> str:
    """Build a globally unique sample ID.

    **Decision:** sample IDs are unique *across datasets*, not merely within one, and are therefore
    prefixed with the dataset name:

        make_sample_id("cvl", "0001-1-1")  ->  "cvl/0001-1-1"

    Per-dataset uniqueness would let two records collide there, and a collision in a support/query
    split is exactly the kind of silent error the integrity machinery cannot detect after the fact.

    The same rule applies to ``writer_id``, for the same reason.
    """
    if not dataset:
        raise ValueError("dataset name must not be empty")
    if not local_id:
        raise ValueError("local_id must not be empty")
    if SAMPLE_ID_SEPARATOR in dataset:
        raise ValueError(f"dataset name must not contain {SAMPLE_ID_SEPARATOR!r}: {dataset!r}")
    return f"{dataset}{SAMPLE_ID_SEPARATOR}{local_id}"


def make_writer_id(dataset: str, local_id: str) -> str:
    """Build a globally unique writer ID. See :func:`make_sample_id`."""
    return make_sample_id(dataset, local_id)


def split_sample_id(sample_id: str) -> tuple[str, str]:
    """Inverse of :func:`make_sample_id`. Returns ``(dataset, local_id)``."""
    dataset, separator, local_id = sample_id.partition(SAMPLE_ID_SEPARATOR)
    if not separator:
        raise ValueError(
            f"sample_id {sample_id!r} is not dataset-prefixed; expected "
            f"'<dataset>{SAMPLE_ID_SEPARATOR}<local_id>'"
        )
    return dataset, local_id
