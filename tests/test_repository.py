"""Repository hygiene.

These guard a failure mode that is **invisible locally**: the working tree is complete and every
test passes, while the committed repository is missing files. It only surfaces in CI, usually as a
confusing unrelated error.

It has already happened once. An unanchored ``data/`` line in ``.gitignore`` — intended to keep
datasets out of the repository — matched every directory of that name at any depth and silently
excluded the whole ``src/glyphmemory/data/`` package.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directories whose contents must always be committable.
SOURCE_DIRS = ("src", "tests", "configs", "docs/results", ".github")

#: Deliberately withheld from the repository. An exclusion here is a decision, so it is named
#: rather than left to be rediscovered as a mysterious missing file.
DELIBERATELY_EXCLUDED = frozenset({"docs/results/publication-handoff.md"})

#: Individual files that must be committable despite broad ignore rules.
REQUIRED_FILES = (
    "artifacts/charset_en_v1.json",
    "pyproject.toml",
    "uv.lock",
    "README.md",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _is_git_repo() -> bool:
    return (REPO_ROOT / ".git").exists() and _git("rev-parse", "--git-dir").returncode == 0


requires_git = pytest.mark.skipif(
    not _is_git_repo(), reason="not a git repository (e.g. installed from a source archive)"
)


def _ignored(paths: list[str]) -> list[str]:
    """Subset of ``paths`` that git would exclude."""
    if not paths:
        return []
    # check-ignore exits 1 when nothing matches, which is the success case here.
    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=REPO_ROOT,
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@requires_git
@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_no_source_file_is_gitignored(directory: str):
    """A source file being excluded from the repository is never legitimate."""
    root = REPO_ROOT / directory
    if not root.is_dir():
        pytest.skip(f"{directory}/ does not exist")

    candidates = [
        str(path.relative_to(REPO_ROOT))
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".cfg"}
        and "__pycache__" not in path.parts
    ]
    ignored = [p for p in _ignored(candidates) if p not in DELIBERATELY_EXCLUDED]
    assert not ignored, (
        f"{len(ignored)} file(s) under {directory}/ are excluded by .gitignore and would be "
        f"missing from a fresh clone: {sorted(ignored)[:10]}"
    )


@requires_git
@pytest.mark.parametrize("relative_path", REQUIRED_FILES)
def test_required_file_is_not_gitignored(relative_path: str):
    """Files that broad ignore rules could plausibly swallow."""
    if not (REPO_ROOT / relative_path).is_file():
        pytest.skip(f"{relative_path} does not exist yet")
    assert not _ignored([relative_path]), f"{relative_path} is excluded by .gitignore"


@requires_git
def test_python_package_is_fully_tracked():
    """Every committed-worthy module under src/ is known to git.

    Catches both an over-broad ignore rule and a forgotten ``git add``. Untracked *and* unignored
    files are reported separately so the message says which problem it is.
    """
    on_disk = {
        str(path.relative_to(REPO_ROOT))
        for path in (REPO_ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    tracked = set(_git("ls-files", "src").stdout.splitlines())
    missing = sorted(on_disk - tracked)
    assert not missing, (
        f"{len(missing)} module(s) under src/ are not tracked by git — a fresh clone would "
        f"not contain them: {missing[:10]}"
    )


@requires_git
def test_no_dataset_images_are_tracked():
    """The other direction: restricted handwriting data must never enter history."""
    tracked = _git("ls-files").stdout.splitlines()
    images = [
        path
        for path in tracked
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        and not path.startswith("docs/")
    ]
    assert not images, f"image file(s) tracked outside docs/: {images}"


@requires_git
def test_dataset_directories_are_still_ignored():
    """Anchoring the patterns must not have stopped them doing their job."""
    for directory in ("data", "datasets"):
        assert _ignored([f"{directory}/some_corpus/line.png"]), (
            f"root-level {directory}/ is no longer ignored — restricted corpora could be committed"
        )


def test_torchaudio_never_enters_the_shipped_recognition_path():
    """``torchaudio`` is the CTC forced aligner's reference oracle, added to the ``dev`` dependency
    group only. The whole justification for hand-rolling the aligner is that the shipped path never
    depends on it — a stray import would quietly turn a dev/test tool into a runtime dependency.
    """
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "import torchaudio" in text or "from torchaudio" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"torchaudio imported outside tests/: {offenders}"
