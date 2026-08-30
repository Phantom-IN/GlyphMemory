"""Dataset adapters — the only code permitted to know a corpus's on-disk layout."""

from glyphmemory.data.adapters.base import (
    SAMPLE_ID_SEPARATOR,
    DatasetAdapter,
    make_sample_id,
    make_writer_id,
    split_sample_id,
)
from glyphmemory.data.adapters.cvl import (
    CVL_PASSAGES,
    TRANSCRIPT_LIMITATION,
    CVLAdapter,
    CVLPassage,
)
from glyphmemory.data.adapters.iam import (
    PASSAGE_NOTE,
    SEGMENTATION_NOTE,
    STRUCK_OUT_MARKER,
    TOKENISATION_NOTE,
    IAMAdapter,
    IAMForm,
    IAMLine,
    passage_id_for,
)
from glyphmemory.data.adapters.synthetic import SyntheticAdapter

__all__ = [
    "CVL_PASSAGES",
    "PASSAGE_NOTE",
    "SAMPLE_ID_SEPARATOR",
    "SEGMENTATION_NOTE",
    "STRUCK_OUT_MARKER",
    "TOKENISATION_NOTE",
    "TRANSCRIPT_LIMITATION",
    "CVLAdapter",
    "CVLPassage",
    "DatasetAdapter",
    "IAMAdapter",
    "IAMForm",
    "IAMLine",
    "SyntheticAdapter",
    "make_sample_id",
    "make_writer_id",
    "passage_id_for",
    "split_sample_id",
]
