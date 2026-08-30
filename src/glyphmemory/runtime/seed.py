"""Deterministic seeding.

Seeds Python, NumPy and PyTorch (all backends).
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

MAX_SEED = 2**32 - 1


def seed_everything(seed: int, *, log: bool = True) -> int:
    """Seed all relevant RNGs and return the seed for recording."""
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Seed must be in [0, {MAX_SEED}], got {seed}.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Covers CUDA and MPS generators where present; safe no-ops otherwise.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if log:
        logger.info("Seeded RNGs with %d", seed)
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a distinct, derived seed."""
    worker_seed = torch.initial_seed() % (MAX_SEED + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (MAX_SEED + 1))
