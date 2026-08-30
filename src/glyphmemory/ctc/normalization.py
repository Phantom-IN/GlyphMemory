"""Text normalization policy.

Normalization is where a transcription system quietly becomes a *correction* system.

``nfc_v1`` performs exactly four transforms, in order:

1. Unicode NFC composition
2. CRLF / CR -> LF
3. collapse runs of whitespace to a single ASCII space
4. strip leading and trailing whitespace

Everything else is forbidden.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizationPolicy:
    """A named, versioned normalization rule.

    The name is recorded with every experiment and every reported metric. Two results computed under
    different policies are not comparable, so the policy travels with the number rather than being
    assumed.
    """

    name: str
    transforms: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        """Reportable description, embedded in metrics output and run records."""
        return {
            "name": self.name,
            "transforms": list(self.transforms),
            "notes": list(self.notes),
        }


NFC_V1 = NormalizationPolicy(
    name="nfc_v1",
    transforms=(
        "unicode NFC composition",
        "CRLF and CR converted to LF",
        "runs of whitespace collapsed to a single ASCII space",
        "leading and trailing whitespace stripped",
    ),
    notes=(
        "Case, punctuation and spelling are never modified.",
        "Whitespace collapsing is a reportable behaviour: CER computed under this policy is "
        "not directly comparable with work that preserves multiple spaces.",
        "Non-ASCII whitespace (e.g. U+00A0 no-break space) collapses to an ASCII space, "
        "because it is visually indistinguishable in a handwritten line.",
    ),
)

IDENTITY = NormalizationPolicy(
    name="identity",
    transforms=(),
    notes=("No transformation. For tests and for inspecting raw corpus text.",),
)

POLICIES: dict[str, NormalizationPolicy] = {
    NFC_V1.name: NFC_V1,
    IDENTITY.name: IDENTITY,
}

DEFAULT_POLICY = NFC_V1


def get_policy(name: str) -> NormalizationPolicy:
    """Look up a policy by name, as configs reference it."""
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown normalization policy {name!r}. Available: {sorted(POLICIES)}."
        ) from None


def normalize(text: str, policy: NormalizationPolicy = DEFAULT_POLICY) -> str:
    """Apply a normalization policy.

    Args:
        text: Raw transcript.
        policy: Which rule to apply. Defaults to :data:`NFC_V1`.

    Returns:
        The normalized transcript. Case, punctuation and spelling are untouched.
    """
    # Compared by NAME, never by identity. A policy that has crossed a process boundary — as it does
    # when a DataLoader worker unpickles the dataset under `spawn` — is an equal but distinct
    # object, so `is` would reject the very policy it was given.
    if not policy.transforms:
        return text

    if policy.name != NFC_V1.name:
        raise ValueError(f"Policy {policy.name!r} has no implementation.")

    # 1. NFC. Composes decomposed sequences so "e" + combining acute becomes a single
    #    codepoint, which keeps character counts (and therefore CER) meaningful.
    result = unicodedata.normalize("NFC", text)

    # 2. Line endings. Done before collapsing so a CRLF cannot survive as two separators.
    result = result.replace("\r\n", "\n").replace("\r", "\n")

    # 3 + 4. str.split() with no argument splits on runs of Unicode whitespace and discards
    #        leading/trailing runs, so collapsing and stripping happen together.
    return " ".join(result.split())


def normalizes_to_empty(text: str, policy: NormalizationPolicy = DEFAULT_POLICY) -> bool:
    """Whether a transcript is empty once normalized.

    Such a line carries no transcription and must be counted as a missing transcript rather than
    trained on.

    Under ``nfc_v1`` this is currently **equivalent to** ``not text.strip()``: both reduce to "is
    this only Unicode whitespace?", since ``str.split()`` and ``str.strip()`` share a whitespace
    definition and NFC never maps a non-space character to a space. The function exists because the
    check is *policy-aware* — the reported reason names the policy that decided, and a future or
    NFKC-based policy could diverge — not because it catches more cases today.
    """
    return not normalize(text, policy)
