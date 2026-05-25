"""Low-level sampling from a :class:`BoxScoreDistribution`.

This module is the only place that enforces the hard constraints listed
in PLAN.md §4.4 at sample time. The contract:

1. **Sample team-level scalars** (pace, off_rtg) from the team-head Normals.
2. **Sample minute allocations** per team — Dirichlet draw × 240, then
   round to 0.1-min ticks and redistribute the residual onto the largest
   active slot so the per-team sum is exactly 240.0.
3. **Sample attempts** (FGA, 3PA, FTA, etc.) from per-player
   NegativeBinomials.
4. **Sample makes conditional on attempts** — ``Binomial(total_count=
   attempts_sampled, logits=fgm_probs)`` guarantees ``FGM ≤ FGA`` by
   construction (and likewise for 3PM, FTM).
5. **Enforce ``3PM ≤ FGM``** by clamp (the only constraint not satisfied
   by construction, since FGM and 3PM are sampled from independent heads).
6. **Zero padded slots** across every per-player tensor.

The high-level :func:`sample_box_score` that wraps this into the
:class:`BoxScore` schema with cold-start handling is a Phase 4 deliverable
(PLAN §10). :func:`sample_raw_box_score` is the building block — every
downstream caller (constraint test, evaluator, Phase 4 sampler) goes
through it.
"""

from __future__ import annotations

import torch

from nba_sim.data.schema import BoxScore
from nba_sim.models.heads import BoxScoreDistribution


# ---------------------------------------------------------------------------
# Stat name lists. Kept here (not imported from elsewhere) so the sampler
# remains self-contained.
# ---------------------------------------------------------------------------

# Stats with an unconditional NegativeBinomial head per player.
_NB_STATS: tuple[str, ...] = (
    "fga", "tpa", "fta", "oreb", "dreb",
    "ast", "stl", "blk", "tov", "pf",
)
# Conditional Binomials: (make_stat, attempt_stat, logits_field_template).
_PERCENT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("fgm", "fga", "fgm_probs_{side}"),
    ("tpm", "tpa", "tpm_probs_{side}"),
    ("ftm", "fta", "ftm_probs_{side}"),
)


# ---------------------------------------------------------------------------
# Minute rounding
# ---------------------------------------------------------------------------


def _round_minutes_to_240(
    shares: torch.Tensor,
    mask: torch.Tensor,
    *,
    tick: float = 0.1,
    total: float = 240.0,
) -> torch.Tensor:
    """Convert per-team minute shares to absolute minutes summing to ``total``.

    Steps, in order:

    1. Zero the padded slots so they cannot receive any minute mass.
    2. Re-normalize the remaining (active) mass to sum to ``total`` per row.
    3. Round to the nearest ``tick``.
    4. Compute the residual ``total - sum(rounded)`` per row and add it to
       the active slot that currently has the largest rounded value. The
       residual is bounded by ``P · tick / 2 ≈ 0.75`` so this never makes
       the top slot exceed plausible NBA minutes.

    Parameters
    ----------
    shares : ``[B, P]`` non-negative tensor — typically a Dirichlet sample.
    mask   : ``[B, P]`` bool, ``True`` for active player slots.

    Returns
    -------
    ``[B, P]`` minutes tensor with every row summing to ``total`` (within
    float precision) and every padded slot exactly 0.
    """
    if shares.shape != mask.shape:
        raise ValueError(
            f"shares {tuple(shares.shape)} and mask {tuple(mask.shape)} must match"
        )
    mask_f = mask.to(shares.dtype)
    # 1: Zero padded entries.
    active = shares * mask_f
    # 2: Re-normalize active mass to sum to total.
    row_sum = active.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    scaled = active * (total / row_sum)
    # 3: Round to ticks.
    rounded = torch.round(scaled / tick) * tick
    # 4: Place residual on top-active slot.
    residual = total - rounded.sum(dim=-1, keepdim=True)
    # Top-active argmax: padded slots get -inf so they're never chosen.
    masked_for_argmax = torch.where(
        mask, rounded, torch.full_like(rounded, float("-inf"))
    )
    top_idx = masked_for_argmax.argmax(dim=-1, keepdim=True)
    rounded = rounded.scatter_add(-1, top_idx, residual)
    # Defensive: re-zero padded slots (scatter_add never touches them, but
    # makes the invariant explicit at the boundary).
    return rounded * mask_f


# ---------------------------------------------------------------------------
# Raw-counts sampler (Phase 2 ship-gate building block)
# ---------------------------------------------------------------------------


def sample_raw_box_score(
    dist: BoxScoreDistribution,
    home_mask: torch.Tensor,
    away_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Sample one constraint-satisfying box score per row of the batch.

    Parameters
    ----------
    dist
        Output of :meth:`HierarchicalBoxScoreModel.forward`.
    home_mask, away_mask
        ``[B, P]`` bool masks; ``True`` = active player slot.

    Returns
    -------
    dict[str, Tensor]
        Keys::

            pace                            [B]
            off_rtg                         [B, 2]   (home, away)
            {home,away}_minutes             [B, P]   sums to 240 per row
            {home,away}_plays_gate          [B, P]   {0, 1}
            {home,away}_{stat}              [B, P]   for stat in 13 stats
            {home,away}_pts                 [B, P]   derived
            {home,away}_reb                 [B, P]   derived

        Where the 13 stats are ``{fga, tpa, fta, fgm, tpm, ftm, oreb, dreb,
        ast, stl, blk, tov, pf}``. Every padded slot is 0 across every
        per-player tensor.

    Notes
    -----
    Uses the global torch RNG. Wrap the call in ``torch.manual_seed(...)``
    or a ``torch.random.fork_rng`` context for reproducibility.
    """
    out: dict[str, torch.Tensor] = {}

    # ---- team-level scalars ---------------------------------------------
    out["pace"] = dist.pace.sample()                  # [B]
    out["off_rtg"] = dist.off_rtg.sample()            # [B, 2]

    for side, mask in (("home", home_mask), ("away", away_mask)):
        m_f = mask.to(torch.float32)

        # ---- minutes -----------------------------------------------------
        shares = getattr(dist, f"minutes_{side}").sample()
        out[f"{side}_minutes"] = _round_minutes_to_240(shares, mask)

        # ---- plays-in-game gate ------------------------------------------
        gate = getattr(dist, f"plays_gate_{side}").sample().to(torch.float32) * m_f
        out[f"{side}_plays_gate"] = gate

        # ---- NB count stats ----------------------------------------------
        for stat in _NB_STATS:
            s = getattr(dist, f"{stat}_{side}").sample().to(torch.float32) * m_f
            out[f"{side}_{stat}"] = s

        # ---- conditional Binomial makes ----------------------------------
        # FGM | FGA, 3PM | 3PA, FTM | FTA. Sampled independently → ``3PM ≤ FGM``
        # is enforced by post-hoc clamp (PLAN §4.4 explicitly allows this).
        makes: dict[str, torch.Tensor] = {}
        for make, attempt, logits_template in _PERCENT_PAIRS:
            logits = getattr(dist, logits_template.format(side=side))
            d = BoxScoreDistribution.conditional_make_dist(
                out[f"{side}_{attempt}"], logits
            )
            makes[make] = d.sample().to(torch.float32) * m_f
        makes["tpm"] = torch.minimum(makes["tpm"], makes["fgm"])
        for make, samp in makes.items():
            out[f"{side}_{make}"] = samp

        # ---- derived per-player totals -----------------------------------
        # PTS = 2·FGM + TPM + FTM (extra-point-for-3 convention).
        # REB = OREB + DREB.
        out[f"{side}_pts"] = (
            2.0 * out[f"{side}_fgm"] + out[f"{side}_tpm"] + out[f"{side}_ftm"]
        )
        out[f"{side}_reb"] = out[f"{side}_oreb"] + out[f"{side}_dreb"]

    return out


# ---------------------------------------------------------------------------
# Phase 4 stubs (full schema wrapping + cold start)
# ---------------------------------------------------------------------------


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
    """Sample one complete, schema-validated :class:`BoxScore` (Phase 4).

    Wraps :func:`sample_raw_box_score` and turns the raw tensors into the
    :class:`BoxScore` Pydantic schema (cold-start handling per PLAN §7.3
    happens upstream when the embedding is resolved).
    """
    raise NotImplementedError("Phase 4 deliverable — see simulate/api.py")


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
    """Sample ``n`` box scores sharing one forward pass (Phase 4)."""
    raise NotImplementedError("Phase 4 deliverable — see simulate/api.py")


def _resolve_player_embedding(player_id: int, model) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Three-tier cold-start fallback for unseen players (Phase 4, PLAN §7.3)."""
    raise NotImplementedError("Phase 4 deliverable")
