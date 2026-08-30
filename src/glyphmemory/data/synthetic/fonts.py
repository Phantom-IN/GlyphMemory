"""Typeface discovery for the synthetic generator.

**Decision: no font binaries are vendored into the repository.**.

The phase plan recommended bundling three to five OFL fonts. That was motivated by cross-platform
determinism — system font paths differ between macOS and the Linux CI runner, and a fixture that
renders differently per platform is useless. Pillow satisfies that goal outright: it ships a
scalable font reachable through :func:`PIL.ImageFont.load_default`, identical on every platform,
requiring no download and adding no licensing surface. Vendoring binaries to solve a problem the
dependency already solves is not worth the cost.

Consequently **writer identity is ``(typeface, style vector)``, not typeface alone**. With a single
guaranteed typeface, writers are still distinct — see
:class:`~glyphmemory.data.synthetic.render.WriterStyle`.

**Known limitation this creates.** One typeface means every writer shares glyph *topology*: no
writer has a closed ``a`` where another has an open one. Real writers differ that way, so this
oracle is easier than reality in that specific respect.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import ImageFont

from glyphmemory.runtime.logging import get_logger

logger = get_logger("data.synthetic.fonts")

#: Optional directory for repository-vendored typefaces. Absent by default.
REPO_FONT_DIR = Path("assets") / "fonts"

FONT_SUFFIXES = (".ttf", ".otf")

#: Identifier for Pillow's built-in scalable font — the one guaranteed everywhere.
BUNDLED_FONT_NAME = "pillow-default"

#: Consulted only when explicitly requested. Never used by tests or CI, because these paths do not
#: exist on the Linux runner.
SYSTEM_FONT_DIRS: tuple[Path, ...] = (
    Path("/System/Library/Fonts/Supplemental"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype"),
    Path("/usr/share/fonts"),
)


@dataclass(frozen=True, slots=True)
class FontSource:
    """A typeface the generator can render with."""

    name: str
    origin: str
    path: Path | None = None

    def load(self, size: float) -> Any:
        """Load this typeface at ``size`` points."""
        if self.path is None:
            return ImageFont.load_default(size=size)
        return ImageFont.truetype(str(self.path), size=size)

    def describe(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "origin": self.origin,
            "path": str(self.path) if self.path else None,
        }


def bundled_font() -> FontSource:
    """Pillow's built-in scalable font. Always available, identical on every platform."""
    return FontSource(name=BUNDLED_FONT_NAME, origin="pillow", path=None)


def _fonts_in(directory: Path, origin: str) -> list[FontSource]:
    if not directory.is_dir():
        return []
    found = [
        FontSource(name=path.stem, origin=origin, path=path)
        for path in sorted(directory.rglob("*"))
        if path.suffix.lower() in FONT_SUFFIXES
    ]
    return found


def discover_fonts(
    *,
    extra_dir: Path | None = None,
    allow_system: bool = False,
    max_fonts: int | None = None,
) -> list[FontSource]:
    """Return the typefaces available, in deterministic order.

    The bundled font is always first, so results never depend on what happens to be
    installed. Discovery order is bundled -> repository ``assets/fonts`` -> ``extra_dir`` ->
    system directories (only when ``allow_system`` is set).

    Args:
        extra_dir: Additional directory to search.
        allow_system: Search platform font directories too. **Leave this off for anything
            reproducible** — the results differ between macOS and the CI runner.
        max_fonts: Truncate the list, keeping the bundled font first.
    """
    sources: list[FontSource] = [bundled_font()]
    sources.extend(_fonts_in(REPO_FONT_DIR, "repo"))
    if extra_dir is not None:
        sources.extend(_fonts_in(Path(extra_dir), "extra"))
    if allow_system:
        for directory in SYSTEM_FONT_DIRS:
            sources.extend(_fonts_in(directory, "system"))

    seen: set[str] = set()
    unique: list[FontSource] = []
    for source in sources:
        key = str(source.path) if source.path else source.name
        if key not in seen:
            seen.add(key)
            unique.append(source)

    if max_fonts is not None:
        unique = unique[:max_fonts]

    logger.debug("Discovered %d typeface(s): %s", len(unique), [s.name for s in unique])
    return unique


def resolve_fonts(fonts: Sequence[FontSource] | None) -> list[FontSource]:
    """Normalise a caller-supplied font list, defaulting to the guaranteed bundled font."""
    if fonts:
        return list(fonts)
    return [bundled_font()]
