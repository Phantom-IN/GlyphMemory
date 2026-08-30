"""Peak RSS: platform-aware units, never a guess."""

from __future__ import annotations

import sys

from glyphmemory.benchmark.memory import peak_rss_bytes


def test_peak_rss_is_positive_on_a_platform_with_resource():
    value = peak_rss_bytes()
    if sys.platform in ("darwin", "linux"):
        assert value is not None
        assert value > 0
    else:
        assert value is None
