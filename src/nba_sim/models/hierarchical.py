"""Full hierarchical team -> player model.

Composition (see PLAN.md §4.1):

    HierarchicalBoxScoreModel
        ├── PlayerEncoder         (for each player on each team)
        ├── RosterAttentionPool   (home, away)
        ├── GameContextEncoder
        ├── TeamHead              -> pace, off_rtg
        └── PlayerAllocHead       (home, away) -> per-player distributions

Forward returns a :class:`BoxScoreDistribution` so that both the loss
(:mod:`nba_sim.models.losses`) and the sampler
(:mod:`nba_sim.simulate.sampler`) share the same output contract.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nba_sim.models.heads import BoxScoreDistribution


class HierarchicalBoxScoreModel(nn.Module):
    """The primary v1 model."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Build from a parsed ``configs/model.yaml`` dict."""
        super().__init__()
        raise NotImplementedError

    def forward(self, batch: dict[str, torch.Tensor]) -> BoxScoreDistribution:
        """Consume a padded batch and emit all distributions."""
        raise NotImplementedError

    @classmethod
    def from_checkpoint(cls, path: str) -> "HierarchicalBoxScoreModel":
        """Load weights + config from a checkpoint file."""
        raise NotImplementedError
