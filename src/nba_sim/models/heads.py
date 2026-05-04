"""Distribution-parameterizing output heads.

Every head returns an object from ``torch.distributions`` so that
:func:`torch.distributions.Distribution.log_prob` can be used directly in
the composite NLL loss and :func:`Distribution.sample` in the simulator.

See PLAN.md §4.3 for the choice of distribution per stat and §4.4 for how
each hard constraint is enforced by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import (
    Bernoulli,
    Binomial,
    Dirichlet,
    Distribution,
    NegativeBinomial,
    Normal,
)


@dataclass
class BoxScoreDistribution:
    """Container for every distribution produced by a forward pass.

    ``per_team`` entries are keyed by ``"home"`` / ``"away"``.
    """

    # Team level
    pace: Normal
    off_rtg: Normal                           # batched over (home, away)

    # Per-team, per-player minute allocation.
    minutes_home: Dirichlet
    minutes_away: Dirichlet

    # Per-player count heads — each is a batched distribution of shape [B, P].
    plays_gate_home: Bernoulli
    plays_gate_away: Bernoulli
    fga_home: NegativeBinomial
    fga_away: NegativeBinomial
    tpa_home: NegativeBinomial
    tpa_away: NegativeBinomial
    fta_home: NegativeBinomial
    fta_away: NegativeBinomial
    # Makes conditional on attempts are not instantiated here; they are
    # created at sample/log_prob time because ``total_count`` depends on
    # the observed/sampled attempt count. See :meth:`conditional_make_dist`.
    fgm_probs_home: torch.Tensor              # logits for Binomial(total=FGA)
    fgm_probs_away: torch.Tensor
    tpm_probs_home: torch.Tensor
    tpm_probs_away: torch.Tensor
    ftm_probs_home: torch.Tensor
    ftm_probs_away: torch.Tensor

    oreb_home: NegativeBinomial
    oreb_away: NegativeBinomial
    dreb_home: NegativeBinomial
    dreb_away: NegativeBinomial
    ast_home: NegativeBinomial
    ast_away: NegativeBinomial
    stl_home: NegativeBinomial
    stl_away: NegativeBinomial
    blk_home: NegativeBinomial
    blk_away: NegativeBinomial
    tov_home: NegativeBinomial
    tov_away: NegativeBinomial
    pf_home: NegativeBinomial
    pf_away: NegativeBinomial

    def conditional_make_dist(self, attempts: torch.Tensor, probs_logits: torch.Tensor) -> Binomial:
        """Build ``Binomial(total_count=attempts, logits=probs_logits)``."""
        raise NotImplementedError


class TeamHead(nn.Module):
    """Predict ``pace`` and ``off_rtg`` (home+away) from pooled reps + context."""

    def __init__(self, d_in: int, hidden: tuple[int, ...] = (128, 64)) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        home_rep: torch.Tensor,    # [B, D_t]
        away_rep: torch.Tensor,    # [B, D_t]
        context: torch.Tensor,     # [B, D_ctx]
        matchup: torch.Tensor,     # [B, D_m]
    ) -> tuple[Normal, Normal]:
        """Return ``(pace_dist, off_rtg_dist)`` where off_rtg has event shape [2]."""
        raise NotImplementedError


class PlayerAllocHead(nn.Module):
    """Produce all per-player distributions for a single team.

    Conditions on per-player encodings and the predicted team-level
    pace/efficiency (teacher-forced during training; sampled at inference).
    """

    def __init__(
        self,
        d_player: int,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        player_reps: torch.Tensor,     # [B, P, D_player]
        team_preds: torch.Tensor,      # [B, D_team_pred] — pace + off_rtg encoded
        mask: torch.Tensor,            # [B, P]
    ) -> dict[str, Distribution | torch.Tensor]:
        """Return a dict with keys ``minutes, plays_gate, fga, tpa, fta,
        fgm_logits, tpm_logits, ftm_logits, oreb, dreb, ast, stl, blk, tov, pf``."""
        raise NotImplementedError
