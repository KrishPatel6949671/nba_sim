"""Composite negative-log-likelihood loss across all heads.

Loss (see PLAN.md §5.1)::

    L = λ_min * NLL_minutes_Dirichlet
      + λ_pace * NLL_pace
      + λ_rtg  * NLL_off_rtg
      + Σ_stat λ_stat * NLL_stat_over_players
      + λ_gate * BCE_plays_in_this_game
      + λ_pool * ||δ_player_embedding||²

The make-head NLLs use the sampled / observed attempts as the Binomial
``total_count`` so make <= attempted is enforced in the likelihood too.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nba_sim.models.heads import BoxScoreDistribution


@dataclass
class LossWeights:
    """Per-head coefficients. Loaded from ``configs/train.yaml:loss_weights``."""

    minutes: float
    pace: float
    off_rtg: float
    fga: float
    tpa: float
    fta: float
    fgm: float
    tpm: float
    ftm: float
    oreb: float
    dreb: float
    ast: float
    stl: float
    blk: float
    tov: float
    pf: float
    gate: float
    embedding_pool: float


def composite_nll(
    preds: BoxScoreDistribution,
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    embedding_deltas: torch.Tensor | None = None,
    embedding_shrinkage_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the weighted composite loss and its per-head decomposition.

    Returns:
        ``{"loss": total, "minutes": ..., "pace": ..., ...}`` — useful for logging.
    """
    raise NotImplementedError
