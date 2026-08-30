"""Model definitions, loss and parameter accounting.

GM-Base is the assembled recognizer: visual encoder -> BiGRU -> character head -> CTC logits.
"""

from glyphmemory.model.blocks import DEFAULT_EXPANSION, InvertedResidual2D
from glyphmemory.model.encoder import (
    BLOCKS_PER_STAGE,
    STAGE_CHANNELS,
    STEM_CHANNELS,
    VisualEncoder,
    stage_output_shapes,
)
from glyphmemory.model.head import CharacterHead
from glyphmemory.model.htr import GMBase, HTROutput
from glyphmemory.model.loss import (
    CTCDiagnostics,
    count_infeasible,
    ctc_loss,
    ctc_loss_for,
    required_alignment_lengths,
)
from glyphmemory.model.model_info import (
    HARD_MAX_PARAMETERS,
    PREFERRED_MAX_PARAMETERS,
    ParameterReport,
    assert_within_budget,
    parameter_count,
    parameter_count_by_module,
    parameter_report,
)
from glyphmemory.model.sequence import DIRECTIONS, SequenceEncoder

__all__ = [
    "BLOCKS_PER_STAGE",
    "DEFAULT_EXPANSION",
    "DIRECTIONS",
    "HARD_MAX_PARAMETERS",
    "PREFERRED_MAX_PARAMETERS",
    "STAGE_CHANNELS",
    "STEM_CHANNELS",
    "CTCDiagnostics",
    "CharacterHead",
    "GMBase",
    "HTROutput",
    "InvertedResidual2D",
    "ParameterReport",
    "SequenceEncoder",
    "VisualEncoder",
    "assert_within_budget",
    "count_infeasible",
    "ctc_loss",
    "ctc_loss_for",
    "parameter_count",
    "parameter_count_by_module",
    "parameter_report",
    "required_alignment_lengths",
    "stage_output_shapes",
]
