"""Character-level vocabulary and tokenizer.

V1 is character-level.

**CTC blank is index 0**, everywhere. Loss, decoder, forced alignment and memory fusion all assume
it; an off-by-one here produces a model that trains without complaint and decodes garbage.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glyphmemory.ctc.normalization import (
    DEFAULT_POLICY,
    NormalizationPolicy,
    get_policy,
    normalize,
)

#: Sentinel occupying index 0. Never appears in text; never emitted by ``decode``.
BLANK_TOKEN = "<blank>"
BLANK_INDEX = 0

CHARSET_SCHEMA_VERSION = "1"

#: The punctuation frozen for v1.
#:
#: **Decision.** This is the punctuation set widely reported for IAM's
#: line-level ground truth, which together with space, digits and both letter cases gives
#: the familiar 79-symbol IAM inventory. It deliberately excludes ``$ % @ = _ < > [ ] { }
#: | \ ~ ^ `` and back-quote, which that inventory does not contain.
#:
#: **This is provisional and unverified against real data.** No IAM or CVL corpus has been
#: parsed yet (IAM access is pending). Use
#: :meth:`Charset.from_texts` to derive a measured charset once real transcripts exist, and
#: publish any change as ``charset_en_v2`` rather than editing v1 in place — the fingerprint
#: must change loudly, because a checkpoint is meaningless under a different vocabulary.
PUNCTUATION_V1 = "!\"#&'()*+,-./:;?"

UPPERCASE = "".join(chr(c) for c in range(ord("A"), ord("Z") + 1))
LOWERCASE = "".join(chr(c) for c in range(ord("a"), ord("z") + 1))
DIGITS = "".join(str(d) for d in range(10))
SPACE = " "

DEFAULT_CHARSET_NAME = "charset_en_v1"
DEFAULT_CHARSET_PATH = Path("artifacts") / f"{DEFAULT_CHARSET_NAME}.json"


class UnsupportedCharacterError(ValueError):
    """Text contains a character the charset does not define.

    Raised rather than silently mapped. **There is no ``<unk>`` token in v1** — see the module note
    in :class:`Charset`.
    """

    def __init__(self, character: str, position: int, text: str) -> None:
        self.character = character
        self.position = position
        self.text = text
        name = _describe_character(character)
        super().__init__(
            f"Character {character!r} (U+{ord(character):04X}, {name}) at position {position} "
            f"is not in the charset. GlyphMemory has no <unk> token: an unsupported character "
            f"is a counted, logged data problem, not something to map away silently."
        )


def _describe_character(character: str) -> str:
    import unicodedata

    try:
        return unicodedata.name(character)
    except ValueError:
        return "unnamed codepoint"


@dataclass(frozen=True, slots=True)
class Charset:
    """An ordered symbol table with CTC blank at index 0.

    **Decision: no ``<unk>`` token in v1.** An unsupported character is a data problem, and
    ``<unk>`` turns a data problem into a token — making it invisible in exactly the way forbids.
    Instead :meth:`encode` raises, :class:`~glyphmemory.data.validation.IntegrityCounters` records
    it under ``unsupported_character``, and the rate is measured before any decision to widen the
    charset. If measurement later shows the rejection rate is material, the response is a measured
    ``charset_en_v2``, recorded as an ADR — not a quietly added catch-all class.
    """

    symbols: tuple[str, ...]
    name: str = DEFAULT_CHARSET_NAME

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("Charset must contain at least the blank symbol.")
        if self.symbols[0] != BLANK_TOKEN:
            raise ValueError(
                f"CTC blank must occupy index 0; found {self.symbols[0]!r}. "
                "Loss, decoder, alignment and fusion all assume blank == 0."
            )
        duplicates = [item for item, count in Counter(self.symbols).items() if count > 1]
        if duplicates:
            raise ValueError(f"Charset contains duplicate symbol(s): {sorted(duplicates)}")
        for symbol in self.symbols[1:]:
            if len(symbol) != 1:
                raise ValueError(
                    f"Charset symbols must be single characters; found {symbol!r}. "
                    "V1 is character-level."
                )

    # ------------------------------------------------------------------ construction

    @classmethod
    def english_v1(cls) -> Charset:
        """The frozen v1 English charset: blank, space, A-Z, a-z, 0-9, punctuation."""
        characters = SPACE + UPPERCASE + LOWERCASE + DIGITS + PUNCTUATION_V1
        return cls(symbols=(BLANK_TOKEN, *characters), name=DEFAULT_CHARSET_NAME)

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        *,
        name: str,
        policy: NormalizationPolicy = DEFAULT_POLICY,
    ) -> Charset:
        """Derive a charset from real transcripts.

        Symbols are sorted by codepoint for a deterministic, auditable order.
        """
        observed: set[str] = set()
        for text in texts:
            observed.update(normalize(text, policy))
        return cls(symbols=(BLANK_TOKEN, *sorted(observed)), name=name)

    # ------------------------------------------------------------------ lookup

    @property
    def size(self) -> int:
        """Number of classes including blank — the width of the CTC output layer."""
        return len(self.symbols)

    @property
    def blank(self) -> int:
        return BLANK_INDEX

    @property
    def characters(self) -> tuple[str, ...]:
        """Every symbol except blank."""
        return self.symbols[1:]

    def __contains__(self, character: str) -> bool:
        return character in self._lookup

    @property
    def _lookup(self) -> dict[str, int]:
        # Rebuilt per access on a frozen slots dataclass; charsets are tiny (~80 entries) and this
        # keeps the type hashable and trivially serializable.
        return {symbol: index for index, symbol in enumerate(self.symbols)}

    def index_of(self, character: str) -> int:
        try:
            return self._lookup[character]
        except KeyError:
            raise UnsupportedCharacterError(character, -1, character) from None

    def char_at(self, index: int) -> str:
        if not 0 <= index < self.size:
            raise IndexError(f"Class index {index} out of range for charset of size {self.size}.")
        return self.symbols[index]

    # ------------------------------------------------------------------ identity

    def fingerprint(self) -> str:
        """Stable hash of the symbol sequence.

        Answers exactly one question: *will this charset produce the same integer sequences?* The
        name is metadata and is deliberately excluded, so renaming a charset does not invalidate
        compatible checkpoints while changing, adding, removing or reordering any symbol does.
        """
        digest = hashlib.sha256()
        digest.update(f"glyphmemory-charset-v{CHARSET_SCHEMA_VERSION}\n".encode())
        for symbol in self.symbols:
            digest.update(symbol.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    # ------------------------------------------------------------------ persistence

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHARSET_SCHEMA_VERSION,
            "name": self.name,
            "blank_token": BLANK_TOKEN,
            "blank_index": BLANK_INDEX,
            "size": self.size,
            "fingerprint": self.fingerprint(),
            "characters": "".join(self.characters),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Charset:
        if payload.get("blank_index", BLANK_INDEX) != BLANK_INDEX:
            raise ValueError(f"Charset declares blank_index {payload['blank_index']}, expected 0.")
        charset = cls(
            symbols=(BLANK_TOKEN, *payload["characters"]),
            name=payload.get("name", DEFAULT_CHARSET_NAME),
        )
        recorded = payload.get("fingerprint")
        if recorded and recorded != charset.fingerprint():
            raise ValueError(
                f"Charset fingerprint mismatch for {charset.name!r}: file records {recorded} "
                f"but its characters hash to {charset.fingerprint()}. The file was edited "
                "without regenerating it."
            )
        return charset

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Charset:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class Tokenizer:
    """Encodes normalized text to class indices and back.

    Holds the normalization policy as well as the charset, because both determine what an integer
    sequence means. :meth:`fingerprint` covers the pair and is recorded per run.
    """

    charset: Charset
    policy: NormalizationPolicy = DEFAULT_POLICY

    @classmethod
    def english_v1(cls) -> Tokenizer:
        return cls(charset=Charset.english_v1())

    @property
    def vocab_size(self) -> int:
        return self.charset.size

    @property
    def blank_index(self) -> int:
        return self.charset.blank

    def encode(self, text: str, *, apply_normalization: bool = True) -> list[int]:
        """Encode text to class indices.

        Args:
            text: Transcript to encode.
            apply_normalization: Apply the tokenizer's policy first. Leave ``True`` so that
                ``decode(encode(t)) == normalize(t)``.

        Raises:
            UnsupportedCharacterError: A character is not in the charset. Never mapped away.
        """
        prepared = normalize(text, self.policy) if apply_normalization else text
        lookup = self.charset._lookup
        indices: list[int] = []
        for position, character in enumerate(prepared):
            try:
                indices.append(lookup[character])
            except KeyError:
                raise UnsupportedCharacterError(character, position, prepared) from None
        return indices

    def decode(self, indices: Iterable[int]) -> str:
        """Map class indices back to text.

        Raises:
            ValueError: A blank index was passed.
            IndexError: An index is outside the charset.
        """
        characters: list[str] = []
        for index in indices:
            if index == BLANK_INDEX:
                raise ValueError(
                    "decode() received the blank class. It is the inverse of encode(), not a "
                    "CTC decoder — collapse repeats and strip blanks first."
                )
            characters.append(self.charset.char_at(index))
        return "".join(characters)

    def supports(self, text: str, *, apply_normalization: bool = True) -> bool:
        """Whether every character in ``text`` is encodable."""
        prepared = normalize(text, self.policy) if apply_normalization else text
        return all(character in self.charset for character in prepared)

    def unsupported_characters(self, text: str, *, apply_normalization: bool = True) -> set[str]:
        """Characters in ``text`` the charset does not define."""
        prepared = normalize(text, self.policy) if apply_normalization else text
        return {character for character in prepared if character not in self.charset}

    def fingerprint(self) -> str:
        """Identity of the charset *and* policy pair, recorded with every run."""
        digest = hashlib.sha256()
        digest.update(self.charset.fingerprint().encode("ascii"))
        digest.update(b"\x00")
        digest.update(self.policy.name.encode("utf-8"))
        return digest.hexdigest()

    def describe(self) -> dict[str, Any]:
        return {
            "charset": self.charset.name,
            "charset_fingerprint": self.charset.fingerprint(),
            "vocab_size": self.vocab_size,
            "blank_index": self.blank_index,
            "normalization": self.policy.describe(),
            "tokenizer_fingerprint": self.fingerprint(),
        }


def load_tokenizer(
    charset_path: str | Path = DEFAULT_CHARSET_PATH, *, policy_name: str = DEFAULT_POLICY.name
) -> Tokenizer:
    """Load a tokenizer from a committed charset artifact."""
    return Tokenizer(charset=Charset.load(charset_path), policy=get_policy(policy_name))
