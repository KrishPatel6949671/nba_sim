"""RNG seeding and determinism helpers.

Call :func:`set_seed` at the top of any entry point that needs
reproducibility (training, evaluation, simulation). GPU determinism
(``torch.use_deterministic_algorithms(True)``) costs ~5-10% throughput —
fine for single-game inference but left off by default for training
unless ``deterministic: true`` is set in the train config.
"""

from __future__ import annotations

import torch


def set_seed(seed: int, *, deterministic: bool = False) -> torch.Generator:
    """Seed Python / NumPy / PyTorch. Returns a CUDA-aware generator."""
    raise NotImplementedError


def set_deterministic() -> None:
    """Enable deterministic CUDA kernels. Must be called before any forward pass."""
    raise NotImplementedError
