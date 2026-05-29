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

import datetime as _dt

import torch

from nba_sim.data.schema import (
    BoxScore,
    BoxScoreEnsemble,
    SimPlayerBoxLine,
    SimTeamBoxLine,
)
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
# Schema wrapping — turn raw tensors into BoxScore / BoxScoreEnsemble
# ---------------------------------------------------------------------------


# Per-player counting stats in the order SimPlayerBoxLine expects them.
_PLAYER_STAT_FIELDS: tuple[str, ...] = (
    "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
    "oreb", "dreb", "reb",
    "ast", "stl", "blk", "tov", "pf",
)


def _build_sim_player_box(
    raw: dict[str, torch.Tensor],
    side: str,
    players: list[tuple[int, str]],
    team_margin: float,
) -> list[SimPlayerBoxLine]:
    """Convert one side's row-0 sample tensors into a list of
    :class:`SimPlayerBoxLine` (one per provided player).

    Plus-minus per player ≈ ``team_margin × (player_min / 48)`` per PLAN
    §4.3 — a deterministic allocation by on-court time. DNP players
    (``minutes == 0``) get ``plus_minus = 0``. The 1/48 factor comes from
    "one floor position is worth 48 minutes"; summing across the ~10-player
    rotation gives ≈ 5 × team_margin, the expected total since 5 positions
    are on the floor at all times.
    """
    out: list[SimPlayerBoxLine] = []
    for i, (pid, name) in enumerate(players):
        mins = float(raw[f"{side}_minutes"][0, i].item())
        pm = (team_margin * mins / 48.0) if mins > 0.0 else 0.0
        kwargs: dict[str, int | float | str] = {
            "player_id": pid,
            "player_name": name,
            "minutes": mins,
            "plus_minus": pm,
        }
        for stat in _PLAYER_STAT_FIELDS:
            kwargs[stat] = int(raw[f"{side}_{stat}"][0, i].item())
        out.append(SimPlayerBoxLine(**kwargs))
    return out


def _build_sim_team_box(
    team_name: str,
    players: list[SimPlayerBoxLine],
    pace: float,
    off_rtg: float,
    def_rtg: float,
) -> SimTeamBoxLine:
    """Assemble :class:`SimTeamBoxLine`. team_pts is the player sum per
    PLAN §4.4 — the only team-level *predicted* quantities are pace and
    efficiency; team_pts is mechanically `sum(player_pts)` so the constraint
    `sum(player_pts) == team_pts` is satisfied by construction."""
    return SimTeamBoxLine(
        team=team_name,
        players=players,
        pts=sum(p.pts for p in players),
        pace=pace,
        off_rtg=off_rtg,
        def_rtg=def_rtg,
    )


def _seed_rng(seed: int, device: torch.device) -> None:
    """Seed every RNG ``sample_raw_box_score`` may touch."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def sample_box_score(
    dist: BoxScoreDistribution,
    *,
    home_players: list[tuple[int, str]],
    away_players: list[tuple[int, str]],
    home_team: str,
    away_team: str,
    date_iso: str,
    seed: int | None = None,
) -> BoxScore:
    """Sample one schema-validated :class:`BoxScore`.

    Expects ``dist`` to have batch dim ``B = 1`` (one game per forward
    pass). Cold-start handling for unseen players happens upstream when
    the input batch is assembled — by the time we get here, the model
    has already produced a distribution conditioned on whatever embedding
    the dataset resolved (PLAN §7.3 path lives in ``simulate/api.py``).

    Parameters
    ----------
    dist
        Output of :meth:`HierarchicalBoxScoreModel.forward` for a single game.
    home_players, away_players
        ``(player_id, display_name)`` pairs **in the same order** as the
        rows in the batch's ``{side}_player_feats`` tensor.
    home_team, away_team
        Team abbreviations for the output schema.
    date_iso
        ``"YYYY-MM-DD"`` game date.
    seed
        Optional deterministic seed. Uses :func:`torch.random.fork_rng` so
        the call doesn't perturb the caller's RNG state.

    Returns
    -------
    :class:`BoxScore` with one :class:`SimTeamBoxLine` per side, each
    holding a list of :class:`SimPlayerBoxLine`. Constraints (sum-to-240,
    fgm≤fga, etc.) are guaranteed by :func:`sample_raw_box_score`.
    """
    P = 15
    n_home, n_away = len(home_players), len(away_players)
    if n_home == 0 or n_away == 0:
        raise ValueError(
            f"need at least one player per side, got home={n_home}, away={n_away}"
        )
    if n_home > P or n_away > P:
        raise ValueError(
            f"too many players (max {P}): home={n_home}, away={n_away}"
        )

    device = dist.pace.mean.device
    home_mask = torch.zeros(1, P, dtype=torch.bool, device=device)
    away_mask = torch.zeros(1, P, dtype=torch.bool, device=device)
    home_mask[0, :n_home] = True
    away_mask[0, :n_away] = True

    if seed is not None:
        with torch.random.fork_rng():
            _seed_rng(seed, device)
            raw = sample_raw_box_score(dist, home_mask, away_mask)
    else:
        raw = sample_raw_box_score(dist, home_mask, away_mask)

    pace_s = float(raw["pace"][0].item())
    off_home = float(raw["off_rtg"][0, 0].item())
    off_away = float(raw["off_rtg"][0, 1].item())

    # Tally team_margin from sampled player pts so plus_minus stays
    # consistent with the actual integer-rounded points each player got.
    home_pts_total = sum(int(raw["home_pts"][0, i].item()) for i in range(n_home))
    away_pts_total = sum(int(raw["away_pts"][0, i].item()) for i in range(n_away))
    team_margin = float(home_pts_total - away_pts_total)

    home_lines = _build_sim_player_box(raw, "home", home_players, +team_margin)
    away_lines = _build_sim_player_box(raw, "away", away_players, -team_margin)

    return BoxScore(
        home=_build_sim_team_box(
            home_team, home_lines,
            pace=pace_s, off_rtg=off_home, def_rtg=off_away,
        ),
        away=_build_sim_team_box(
            away_team, away_lines,
            pace=pace_s, off_rtg=off_away, def_rtg=off_home,
        ),
        date=_dt.date.fromisoformat(date_iso),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Ensemble + aggregation
# ---------------------------------------------------------------------------


def _derive_seeds(master: int, n: int) -> list[int]:
    """``n`` distinct deterministic seeds derived from one ``master`` seed.

    Uses a dedicated ``torch.Generator`` so the derivation doesn't touch
    the global RNG. ``randint`` on a 31-bit range avoids the platform-
    dependent overflow of 64-bit seeds in some torch builds.
    """
    gen = torch.Generator()
    gen.manual_seed(int(master))
    return torch.randint(0, 2**31 - 1, (n,), generator=gen).tolist()


def _aggregate_box_scores(samples: list[BoxScore], how: str) -> BoxScore:
    """Build a synthetic :class:`BoxScore` representing per-cell aggregate.

    ``how`` ∈ ``{"mean", "p10", "p90"}``. Counting stats are rounded to
    int (Pydantic requires ``int`` for them); minutes / plus_minus / pace /
    off_rtg / def_rtg stay float. The result is a *summary* — it's not
    drawn from any single posterior path, so consumers should treat it
    as a cell-wise descriptor, not a coherent draw.
    """
    if not samples:
        raise ValueError("no samples to aggregate")

    def _agg(values: list[float], *, to_int: bool) -> float | int:
        if not values:
            raise ValueError("empty values for aggregation")
        if how == "mean":
            v = sum(values) / len(values)
        elif how == "p10":
            v = _quantile(values, 0.10)
        elif how == "p90":
            v = _quantile(values, 0.90)
        else:
            raise ValueError(f"unknown how={how!r}")
        return int(round(v)) if to_int else float(v)

    def _gather_player(side: str, idx: int, stat: str) -> list[float]:
        return [float(getattr(getattr(s, side).players[idx], stat))
                for s in samples]

    def _gather_team(side: str, stat: str) -> list[float]:
        return [float(getattr(getattr(s, side), stat)) for s in samples]

    def _build_team(side: str) -> SimTeamBoxLine:
        first_team = getattr(samples[0], side)
        n_players = len(first_team.players)
        agg_players: list[SimPlayerBoxLine] = []
        for i in range(n_players):
            player0 = first_team.players[i]
            kwargs: dict[str, int | float | str] = {
                "player_id": player0.player_id,
                "player_name": player0.player_name,
                "minutes": _agg(_gather_player(side, i, "minutes"), to_int=False),
                "plus_minus": _agg(_gather_player(side, i, "plus_minus"), to_int=False),
            }
            for stat in _PLAYER_STAT_FIELDS:
                kwargs[stat] = _agg(_gather_player(side, i, stat), to_int=True)
            agg_players.append(SimPlayerBoxLine(**kwargs))
        return SimTeamBoxLine(
            team=first_team.team,
            players=agg_players,
            pts=_agg(_gather_team(side, "pts"), to_int=True),
            pace=_agg(_gather_team(side, "pace"), to_int=False),
            off_rtg=_agg(_gather_team(side, "off_rtg"), to_int=False),
            def_rtg=_agg(_gather_team(side, "def_rtg"), to_int=False),
        )

    return BoxScore(
        home=_build_team("home"),
        away=_build_team("away"),
        date=samples[0].date,
        seed=None,
    )


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile, identical to numpy.quantile default."""
    if not values:
        raise ValueError("empty values for quantile")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def sample_ensemble(
    dist: BoxScoreDistribution,
    n: int,
    *,
    home_players: list[tuple[int, str]],
    away_players: list[tuple[int, str]],
    home_team: str,
    away_team: str,
    date_iso: str,
    seed: int | None = None,
) -> BoxScoreEnsemble:
    """Sample ``n`` schema-validated box scores sharing one forward pass.

    PLAN §7.4: "vectorize sampling by expanding the batch dim; all
    n_samples share one forward pass through the encoder." We keep the
    forward result as-is (a single B=1 dist) and loop the sampling step
    — for the per-player heads, that's already cheap compared to the
    encoder pass the caller already paid for.

    When ``seed`` is given, ``n`` distinct deterministic per-sample seeds
    are derived from it so the ensemble is reproducible but not all
    identical.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    per_sample_seeds: list[int | None]
    if seed is not None:
        per_sample_seeds = list(_derive_seeds(seed, n))
    else:
        per_sample_seeds = [None] * n

    samples = [
        sample_box_score(
            dist,
            home_players=home_players,
            away_players=away_players,
            home_team=home_team,
            away_team=away_team,
            date_iso=date_iso,
            seed=s,
        )
        for s in per_sample_seeds
    ]

    return BoxScoreEnsemble(
        samples=samples,
        mean=_aggregate_box_scores(samples, "mean"),
        interval_low=_aggregate_box_scores(samples, "p10"),
        interval_high=_aggregate_box_scores(samples, "p90"),
    )


# ---------------------------------------------------------------------------
# Cold-start stub
# ---------------------------------------------------------------------------


def _resolve_player_embedding(player_id: int, model) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """Three-tier cold-start fallback for unseen players (PLAN §7.3).

    Lives at the boundary between the dataset and the model — the right
    home is ``simulate/api.py`` where rosters first get resolved to model
    inputs. Left as a stub here so the public surface of sampler.py
    stays purely about sampling, not about identity resolution.
    """
    raise NotImplementedError("cold-start lives in simulate/api.py")
