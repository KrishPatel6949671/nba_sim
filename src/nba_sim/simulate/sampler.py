"""Low-level sampling from a :class:`BoxScoreDistribution`.

This module is the only place that enforces hard constraints at sample
time (see PLAN.md §4.4). The flow is:

    1. Sample per-team pace and off_rtg from Normals.
    2. Sample per-team minute allocations (Dirichlet × 240) then round
       with residual redistribution so the team total is exactly 240.
    3. Sample per-player attempts (FGA, 3PA, FTA) from NegativeBinomial.
    4. Sample per-player makes as Binomial(total=attempts) — guarantees
       makes <= attempted by construction. Enforce 3PM <= FGM post-hoc by
       clamping (rare; logged if triggered).
    5. Sample other counting stats (OREB, DREB, AST, STL, BLK, TOV, PF)
       independently per player.
    6. Compute PTS = 2*FGM + 3PM + FTM. Compute REB = OREB + DREB.
    7. Allocate +/- from team margin × minute share.
"""

from __future__ import annotations

import torch

from nba_sim.data.schema import BoxScore
from nba_sim.models.heads import BoxScoreDistribution


def sample_box_score(
    dist: BoxScoreDistribution,
    *,
    home_players: list[tuple[int, str]],
    away_players: list[tuple[int, str]],
    home_team: str,
    away_team: str,
    date_iso: str,
    generator: torch.Generator | None = None,
) -> BoxScore:
    """Sample one complete, constraint-satisfying box score."""
    raise NotImplementedError


def sample_ensemble(
    dist: BoxScoreDistribution,
    n: int,
    *,
    home_players: list[tuple[int, str]],
    away_players: list[tuple[int, str]],
    home_team: str,
    away_team: str,
    date_iso: str,
    generator: torch.Generator | None = None,
) -> list[BoxScore]:
    """Sample ``n`` box scores sharing one forward pass."""
    raise NotImplementedError


def _round_minutes_to_240(shares: torch.Tensor) -> torch.Tensor:
    """Round Dirichlet-sampled minute shares to 0.1-minute increments summing to 240."""
    raise NotImplementedError


def _resolve_player_embedding(player_id: int, model) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Three-tier cold-start fallback for unseen players (see PLAN.md §7.3)."""
    raise NotImplementedError
