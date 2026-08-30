"""Logging setup.

Console logging by default; an optional file sink for runs. Kept deliberately small — structured
metrics belong in ``metrics.jsonl`` (see ``experiment.py``), not in log lines.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ROOT_LOGGER_NAME = "glyphmemory"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | None = None,
    force: bool = True,
) -> logging.Logger:
    """Configure the ``glyphmemory`` logger and return it.

    Args:
        level: Logging level, as a level number or name.
        log_file: Optional path receiving the same records as the console.
        force: Replace existing handlers. Prevents duplicate output when the CLI is invoked more
            than once in a single process, as it is under pytest.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Records are emitted by this logger's own handlers only.
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the project logger."""
    if name is None or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
