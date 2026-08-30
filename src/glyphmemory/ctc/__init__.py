"""CTC-side text handling: normalization, vocabulary, tokenization and coverage.

Deliberately free of any dependency on :mod:`glyphmemory.data`, so validation can consume a
tokenizer without creating an import cycle.
"""

from glyphmemory.ctc.coverage import CoverageReport, charset_coverage
from glyphmemory.ctc.decode import (
    DEFAULT_DECODER,
    DecoderConfig,
    collapse_repeats,
    ctc_collapse,
    decode_output,
    greedy_decode,
    greedy_decode_batch,
    greedy_decode_ids,
    one_hot_logits,
    strip_blanks,
)
from glyphmemory.ctc.normalization import (
    DEFAULT_POLICY,
    IDENTITY,
    NFC_V1,
    POLICIES,
    NormalizationPolicy,
    get_policy,
    normalize,
    normalizes_to_empty,
)
from glyphmemory.ctc.tokenizer import (
    BLANK_INDEX,
    BLANK_TOKEN,
    DEFAULT_CHARSET_NAME,
    DEFAULT_CHARSET_PATH,
    PUNCTUATION_V1,
    Charset,
    Tokenizer,
    UnsupportedCharacterError,
    load_tokenizer,
)

__all__ = [
    "BLANK_INDEX",
    "BLANK_TOKEN",
    "DEFAULT_CHARSET_NAME",
    "DEFAULT_CHARSET_PATH",
    "DEFAULT_DECODER",
    "DEFAULT_POLICY",
    "IDENTITY",
    "NFC_V1",
    "POLICIES",
    "PUNCTUATION_V1",
    "Charset",
    "CoverageReport",
    "DecoderConfig",
    "NormalizationPolicy",
    "Tokenizer",
    "UnsupportedCharacterError",
    "charset_coverage",
    "collapse_repeats",
    "ctc_collapse",
    "decode_output",
    "get_policy",
    "greedy_decode",
    "greedy_decode_batch",
    "greedy_decode_ids",
    "load_tokenizer",
    "normalize",
    "normalizes_to_empty",
    "one_hot_logits",
    "strip_blanks",
]
