"""Composite negative-log-likelihood loss across all heads.

Loss (see PLAN.md §5.1)::

    L = λ_min   * NLL_minutes_Dirichlet
      + λ_pace  * NLL_pace
      + λ_rtg   * NLL_off_rtg
      + Σ_stat  λ_stat * NLL_stat_over_active_players
      + λ_gate  * BCE_plays_in_this_game
      + λ_pool  * ||δ_player_embedding||²   (only if deltas are passed in)

The make-head NLLs use the *observed* attempts as the Binomial
``total_count`` (teacher-forcing the conditional factorisation
``P(make | attempt) * P(attempt)``).

Reductions:

- Team heads (pace, off_rtg): mean over the batch dim B.
- Minutes Dirichlet: mean over B (one Dirichlet per game).
- All per-player heads (gate, counts, percent-conditional-on-attempts):
  masked mean over active player-slots — sum of log_prob over active
  slots, divided by the count of active slots — so a head's loss has
  units of "nats per active player-game" and the λ=1.0 default in
  ``configs/train.yaml`` is a reasonable starting balance across heads.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Binomial

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

    @classmethod
    def default(cls) -> "LossWeights":
        return cls(
            minutes=1.0, pace=1.0, off_rtg=1.0,
            fga=1.0, tpa=1.0, fta=1.0,
            fgm=1.0, tpm=1.0, ftm=1.0,
            oreb=1.0, dreb=1.0, ast=1.0, stl=1.0, blk=1.0, tov=1.0, pf=1.0,
            gate=0.5,
            embedding_pool=1e-3,
        )


# Stats that share the same per-side schema (count head, NLL on observed target).
_COUNT_STATS: tuple[str, ...] = (
    "fga", "tpa", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
)
# Conditional makes: (make_stat, attempt_stat). The Binomial total_count is
# the *observed* attempt count, the logits are pred.{make}_probs_{side}.
_PERCENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("fgm", "fga"), ("tpm", "tpa"), ("ftm", "fta"),
)


def _masked_mean(log_prob: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``-log_prob`` over True entries of ``mask``."""
    nll = -log_prob * mask
    return nll.sum() / mask.sum().clamp(min=1.0)


def composite_nll(
    preds: BoxScoreDistribution,
    targets: dict[str, torch.Tensor],
    weights: LossWeights,
    embedding_deltas: torch.Tensor | None = None,
    embedding_shrinkage_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the weighted composite loss and its per-head decomposition.

    ``targets`` contract — see :mod:`nba_sim.training.dataset` for producer:

        pace                   : [B]
        off_rtg                : [B, 2]   (home, away)
        home_mask, away_mask   : [B, P] bool
        home_minutes_share     : [B, P]   proportions, padded slots ε-filled so each row sums to 1
        away_minutes_share     : [B, P]
        home_plays_gate        : [B, P]   float in {0, 1}
        away_plays_gate        : [B, P]
        home_{stat}, away_{stat} for stat in
            {fga, tpa, fta, fgm, tpm, ftm, oreb, dreb, ast, stl, blk, tov, pf} : [B, P] non-neg integer counts (as float).

    ``embedding_deltas``: optional ``[N_active, D]`` tensor of player-
    embedding deviations from their role centroid (PLAN §5.4). If
    provided, an L2 shrinkage term is added — optionally weighted per
    player by ``embedding_shrinkage_weight`` so rookies/low-game players
    get stronger shrinkage.

    Returns ``{"loss": total_weighted, "<head>": unweighted_per_head_nll, ...}``.
    """
    per: dict[str, torch.Tensor] = {}

    # --- Team-level ---------------------------------------------------------
    per["pace"] = -preds.pace.log_prob(targets["pace"]).mean()
    per["off_rtg"] = -preds.off_rtg.log_prob(targets["off_rtg"]).mean()

    # --- Minutes (Dirichlet, one per team per game) -------------------------
    per["minutes"] = (
        -preds.minutes_home.log_prob(targets["home_minutes_share"]).mean()
        - preds.minutes_away.log_prob(targets["away_minutes_share"]).mean()
    ) / 2.0

    # --- Plays-in-game gate (Bernoulli per player) --------------------------
    gate_home_lp = preds.plays_gate_home.log_prob(targets["home_plays_gate"])
    gate_away_lp = preds.plays_gate_away.log_prob(targets["away_plays_gate"])
    per["gate"] = (
        _masked_mean(gate_home_lp, targets["home_mask"])
        + _masked_mean(gate_away_lp, targets["away_mask"])
    ) / 2.0

    # --- NB count heads -----------------------------------------------------
    for stat in _COUNT_STATS:
        lp_home = getattr(preds, f"{stat}_home").log_prob(targets[f"home_{stat}"])
        lp_away = getattr(preds, f"{stat}_away").log_prob(targets[f"away_{stat}"])
        per[stat] = (
            _masked_mean(lp_home, targets["home_mask"])
            + _masked_mean(lp_away, targets["away_mask"])
        ) / 2.0

    # --- Conditional Binomial makes -----------------------------------------
    for make, attempts in _PERCENT_PAIRS:
        for side in ("home", "away"):
            logits = getattr(preds, f"{make}_probs_{side}")
            attempt_tgt = targets[f"{side}_{attempts}"]
            make_tgt = targets[f"{side}_{make}"]
            dist = Binomial(total_count=attempt_tgt.float(), logits=logits)
            lp = dist.log_prob(make_tgt)
            contrib = _masked_mean(lp, targets[f"{side}_mask"])
            # Average home+away contributions under one head name.
            per[make] = per.get(make, torch.zeros((), device=lp.device)) + contrib / 2.0

    # --- Optional embedding-pool shrinkage (PLAN §5.4) ---------------------
    if embedding_deltas is not None:
        sq = embedding_deltas.pow(2).sum(dim=-1)             # [N_active]
        if embedding_shrinkage_weight is not None:
            sq = sq * embedding_shrinkage_weight
        per["embedding_pool"] = sq.mean()
    else:
        per["embedding_pool"] = torch.zeros((), device=per["pace"].device)

    # --- Weighted sum -------------------------------------------------------
    total = (
        weights.pace * per["pace"]
        + weights.off_rtg * per["off_rtg"]
        + weights.minutes * per["minutes"]
        + weights.gate * per["gate"]
        + weights.embedding_pool * per["embedding_pool"]
    )
    for stat in _COUNT_STATS:
        total = total + getattr(weights, stat) * per[stat]
    for make, _ in _PERCENT_PAIRS:
        total = total + getattr(weights, make) * per[make]

    return {"loss": total, **per}
