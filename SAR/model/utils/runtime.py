"""Small runtime helpers shared across scripts: device, logging, parameter counts."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logging(
    name: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """Configure root logging and return a named logger."""
    logging.basicConfig(level=level, format=fmt)
    return logging.getLogger(name) if name else logging.getLogger()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters (optionally only trainable ones)."""
    params = (
        p for p in model.parameters() if (p.requires_grad or not trainable_only)
    )
    return sum(p.numel() for p in params)


def log_param_count(model: nn.Module, logger: Optional[logging.Logger] = None) -> None:
    """Log total and trainable parameter counts in millions."""
    log = logger or logging.getLogger()
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    log.info(
        "Params: %.2fM (trainable: %.2fM)", total / 1e6, trainable / 1e6
    )
