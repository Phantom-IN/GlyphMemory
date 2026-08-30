"""Peak resident set size — best effort, platform-dependent, never invented.

``resource.getrusage`` reports ``ru_maxrss`` in different units on different platforms: bytes on
macOS/BSD, kibibytes on Linux.
"""

from __future__ import annotations

import sys


def peak_rss_bytes() -> int | None:
    """Peak RSS of this process so far, in bytes. ``None`` where the platform does not report it
    (e.g. Windows, which has no ``resource`` module).
    """
    try:
        import resource
    except ImportError:
        return None

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(raw)  # bytes on macOS/BSD
    return int(raw) * 1024  # kibibytes on Linux
