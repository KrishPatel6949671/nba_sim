"""Distribution-parameterizing output heads.

Every head returns an object from ``torch.distributions`` so that
:func:`torch.distributions.Distribution.log_prob` can be used directly in
the composite NLL loss and :func:`Distribution.sample` in the simulator.

See PLAN.md §4.3 for the choice of distribution per stat and §4.4 for how
each hard constraint is enforced by construction.

Masking convention: heads output distributions covering all P padded
roster slots. The composite loss is responsible for masking NLL
contributions from padded slots; the simulator zeros padded-slot samples
post-hoc. Heads themselves only mask where the math requires it (the
Dirichlet α for padded slots, and the Bernoulli gate logit).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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


def _inv_softplus(y: float) -> float:
    """Inverse of ``F.softplus`` — for initializing a bias so softplus(bias) ≈ y."""
    return math.log(math.expm1(y))


def _make_trunk(d_in: int, hidden: Sequence[int], dropout: float) -> nn.Sequential:
    """Stack of ``Linear → LayerNorm → GELU → Dropout`` blocks.

    Unlike encoders._build_mlp this has no terminal Linear — output dim is
    ``hidden[-1]`` and the trunk ends in an activation, ready to feed
    multiple parallel projection heads.
    """
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    return nn.Sequential(*layers)


@dataclass
class BoxScoreDistribution:
    """Container for every distribution produced by a forward pass.

    ``per_team`` entries are keyed by ``"home"`` / ``"away"``.
    """

    pace: Normal
    off_rtg: Normal                           # batched over (home, away) — event_shape [2]

    minutes_home: Dirichlet
    minutes_away: Dirichlet

    plays_gate_home: Bernoulli
    plays_gate_away: Bernoulli
    fga_home: NegativeBinomial
    fga_away: NegativeBinomial
    tpa_home: NegativeBinomial
    tpa_away: NegativeBinomial
    fta_home: NegativeBinomial
    fta_away: NegativeBinomial
    # Makes are conditional on attempts; we only store the logits here.
    # See conditional_make_dist below.
    fgm_probs_home: torch.Tensor
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

    @staticmethod
    def conditional_make_dist(
        attempts: torch.Tensor, probs_logits: torch.Tensor
    ) -> Binomial:
        """``Binomial(total_count=attempts, logits=probs_logits)``.

        ``total_count`` must be a non-negative float tensor (PyTorch will
        accept integer counts too but float keeps the API uniform whether
        attempts came from observation or from sampling). Each slot gets
        its own ``total_count`` — no broadcasting tricks required.
        """
        return Binomial(total_count=attempts.float(), logits=probs_logits)


class TeamHead(nn.Module):
    """Predict ``pace`` and ``off_rtg`` (home+away) from pooled reps + context."""

    def __init__(
        self,
        d_in: int,
        hidden: tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        pace_init: float = 100.0,
        pace_noise_init: float = 3.0,
        rtg_init: float = 110.0,
        rtg_noise_init: float = 4.0,
        min_sigma: float = 1e-2,
    ) -> None:
        super().__init__()
        self.min_sigma = min_sigma
        self.trunk = _make_trunk(d_in, hidden, dropout)
        d_h = hidden[-1]
        # (pace_mu, off_home_mu, off_away_mu) and matching log_sigmas.
        self.mu_proj = nn.Linear(d_h, 3)
        self.log_sigma_proj = nn.Linear(d_h, 3)

        with torch.no_grad():
            self.mu_proj.bias.copy_(torch.tensor([pace_init, rtg_init, rtg_init]))
            # Initialize log_sigma bias so exp(bias) ≈ target σ at init.
            self.log_sigma_proj.bias.copy_(
                torch.tensor([
                    math.log(pace_noise_init),
                    math.log(rtg_noise_init),
                    math.log(rtg_noise_init),
                ])
            )
            # Zero the weight matrices for the σ head so init outputs are
            # purely the bias — keeps init σ exactly at the configured value.
            self.log_sigma_proj.weight.zero_()

    def forward(
        self,
        home_rep: torch.Tensor,    # [B, D_t]
        away_rep: torch.Tensor,    # [B, D_t]
        context: torch.Tensor,     # [B, D_ctx]
        matchup: torch.Tensor,     # [B, D_m]
    ) -> tuple[Normal, Normal]:
        """Return ``(pace_dist, off_rtg_dist)``.

        - ``pace_dist`` has batch_shape ``[B]``.
        - ``off_rtg_dist`` has batch_shape ``[B, 2]`` with index 0 = home, 1 = away.
        """
        x = torch.cat([home_rep, away_rep, context, matchup], dim=-1)
        h = self.trunk(x)
        mu = self.mu_proj(h)                                   # [B, 3]
        sigma = self.log_sigma_proj(h).exp().clamp(min=self.min_sigma)
        pace_dist = Normal(mu[:, 0], sigma[:, 0])
        off_rtg_dist = Normal(mu[:, 1:], sigma[:, 1:])
        return pace_dist, off_rtg_dist


class DirichletMinutes(nn.Module):
    """Per-player Dirichlet concentration head — produces a per-team distribution
    over the P roster slots that sums to 1 (then × 240 at sample time)."""

    def __init__(self, d_in: int, alpha_init_scale: float = 2.0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, 1)
        with torch.no_grad():
            self.proj.bias.fill_(_inv_softplus(alpha_init_scale))
            self.proj.weight.zero_()

    def forward(
        self,
        hidden: torch.Tensor,    # [B, P, D]
        mask: torch.Tensor,      # [B, P] bool
    ) -> Dirichlet:
        raw = self.proj(hidden).squeeze(-1)                    # [B, P]
        alpha = nn.functional.softplus(raw) + 1e-4
        # DNP / padded slots get a vanishing α so they receive ~0 minute mass.
        # Using torch.where avoids in-place ops that would break autograd
        # if `hidden` came from a frozen module during the curriculum phase.
        alpha = torch.where(mask, alpha, torch.full_like(alpha, 1e-4))
        return Dirichlet(alpha)


class _CountHead(nn.Module):
    """NegativeBinomial(total_count, logits) head, batched over [B, P]."""

    def __init__(self, d_in: int, total_count_init: float = 5.0) -> None:
        super().__init__()
        self.total_count_proj = nn.Linear(d_in, 1)
        self.logits_proj = nn.Linear(d_in, 1)
        with torch.no_grad():
            self.total_count_proj.bias.fill_(_inv_softplus(total_count_init))

    def forward(self, hidden: torch.Tensor) -> NegativeBinomial:
        tc_raw = self.total_count_proj(hidden).squeeze(-1)
        logits = self.logits_proj(hidden).squeeze(-1)
        total_count = nn.functional.softplus(tc_raw) + 1e-4
        return NegativeBinomial(total_count=total_count, logits=logits)


class _PercentHead(nn.Module):
    """Returns the per-player Binomial logit for a percent stat.

    The actual ``Binomial`` is materialized later by
    :meth:`BoxScoreDistribution.conditional_make_dist` once attempts are
    known (sampled or observed), so the Binomial(total=FGA) construction
    needs the FGA tensor that lives at the call site.
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)


class PlayerAllocHead(nn.Module):
    """Produce all per-player distributions for a single team.

    Conditions on per-player encodings and the predicted team-level
    pace/efficiency (teacher-forced during training; sampled at inference
    — the caller decides what to pass in ``team_preds``).
    """

    # Stats produced as NegativeBinomial counts. Three of these (fga/tpa/fta)
    # also feed the Binomial percent heads below.
    _COUNT_STATS: tuple[str, ...] = (
        "fga", "tpa", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
    )
    _PERCENT_STATS: tuple[str, ...] = ("fgm", "tpm", "ftm")

    def __init__(
        self,
        d_player: int,
        d_team_pred: int = 3,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
        dirichlet_alpha_init_scale: float = 2.0,
        gate_bias_init: float = 1.5,
        nb_total_count_init: float = 5.0,
    ) -> None:
        super().__init__()
        self.d_player = d_player
        self.d_team_pred = d_team_pred

        self.trunk = _make_trunk(d_player + d_team_pred, hidden, dropout)
        d_h = hidden[-1]

        self.minutes_head = DirichletMinutes(d_h, alpha_init_scale=dirichlet_alpha_init_scale)

        self.gate_proj = nn.Linear(d_h, 1)
        with torch.no_grad():
            self.gate_proj.bias.fill_(gate_bias_init)

        self.count_heads = nn.ModuleDict(
            {stat: _CountHead(d_h, total_count_init=nb_total_count_init)
             for stat in self._COUNT_STATS}
        )
        self.percent_heads = nn.ModuleDict(
            {stat: _PercentHead(d_h) for stat in self._PERCENT_STATS}
        )

    def forward(
        self,
        player_reps: torch.Tensor,     # [B, P, D_player]
        team_preds: torch.Tensor,      # [B, D_team_pred]
        mask: torch.Tensor,            # [B, P]
    ) -> dict[str, Distribution | torch.Tensor]:
        b, p, d = player_reps.shape
        if d != self.d_player:
            raise ValueError(
                f"PlayerAllocHead expected d_player={self.d_player}, got {d}"
            )
        if team_preds.shape[-1] != self.d_team_pred:
            raise ValueError(
                f"PlayerAllocHead expected d_team_pred={self.d_team_pred}, "
                f"got {team_preds.shape[-1]}"
            )

        tp = team_preds.unsqueeze(1).expand(b, p, -1)              # [B, P, D_team_pred]
        h = self.trunk(torch.cat([player_reps, tp], dim=-1))       # [B, P, d_h]

        gate_logits = self.gate_proj(h).squeeze(-1)                # [B, P]
        # Force padded slots to ~0 prob of playing.
        gate_logits = torch.where(
            mask, gate_logits, torch.full_like(gate_logits, -1e4)
        )

        out: dict[str, Distribution | torch.Tensor] = {
            "minutes": self.minutes_head(h, mask),
            "plays_gate": Bernoulli(logits=gate_logits),
        }
        for stat in self._COUNT_STATS:
            out[stat] = self.count_heads[stat](h)
        for stat in self._PERCENT_STATS:
            out[f"{stat}_logits"] = self.percent_heads[stat](h)
        return out
