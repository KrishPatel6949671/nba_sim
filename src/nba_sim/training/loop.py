"""Training loop: curriculum, loss composition, checkpointing, early stopping.

Two-stage curriculum (see PLAN.md §5.3):

1. **Team-level pretrain** — freeze player heads; train pace + off_rtg.
2. **Joint training** — unfreeze everything; optimize the full composite loss.

Orchestration:
    * AdamW + cosine-with-warmup, grad clip 1.0, bf16 mixed precision on CUDA.
    * Best-checkpoint tracking on val composite NLL.
    * Early stopping (patience=5).
    * W&B or TensorBoard logging (configurable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train(config: dict[str, Any]) -> Path:
    """Run training end-to-end. Returns path to best checkpoint.

    The config is the union of ``configs/train.yaml`` and any overrides
    from the CLI. See PLAN.md §5 for the full spec.
    """
    raise NotImplementedError


def evaluate(checkpoint: str | Path, split: str = "val") -> dict[str, float]:
    """Load a checkpoint, run the full eval pass on a split, return metrics."""
    raise NotImplementedError


def pretrain_team_head(config: dict[str, Any]) -> None:
    """Curriculum stage 1 — train only pace + off_rtg."""
    raise NotImplementedError


def joint_train(config: dict[str, Any]) -> None:
    """Curriculum stage 2 — train full composite loss."""
    raise NotImplementedError
