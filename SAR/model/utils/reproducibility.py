"""Reproducibility helpers shared across training/evaluation scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Default seed used throughout the project.
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Force every relevant library into deterministic mode."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int, base_seed: int = SEED) -> None:
    """Re-seed each dataloader worker deterministically."""
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for DataLoader shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
