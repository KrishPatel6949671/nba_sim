"""Snapshot → model-batch assembly (v2PLAN.md §15).

Turns a refreshed ``data/snapshot/`` plus a chosen ``(home, away)`` matchup
into the single-game batch the model's forward pass consumes. Assembly steps:

  1. Read rosters / player_features / team_features / team_lastgame + as_of.json.
  2. Resolve team ids; filter each side's players; sort by p_min_avg_10 desc to
     mirror the training-time roster ordering (head(max_players)).
  3. Assemble per-side feature blocks **by reusing** ``BoxScoreDataset._build_side``
     — filling the schedule-derived player numerics (rest_days,
     travel_miles_prev) and the four bool flags (is_starter, dnp, is_home,
     b2b). The two opponent-vs-position numerics (opp_def_rtg_vs_pos,
     opp_blk_allowed_vs_pos) stay null and standardize to the league mean
     (cold-start tier 1+3, §15.4).
  4. Assemble the 24-d context vector (``_build_context``) and 16-d matchup
     vector (``_build_matchup``) from as_of + team_lastgame + team_features.
     ``h2h_last_meeting_margin`` is computed from the as-of season's interim
     games (the pair's last meeting before as_of, home-anchored) so it is not
     cold-started.
  5. Adopt the train split's player_id_map + feature_stats (same train-vs-eval
     standardization discipline as v1).

Cold-start (§15.4): an unseen ``player_id`` maps to embedding id 0; the two
opp-vs-position player numerics and the density flags (is_3in4 / is_4in6)
degrade to mean / False because the dataset builders read them via ``.get``
with a safe fallback.

**Implementation status:** Step 1 (snapshot load, team resolution, h2h,
schedule context, per-side player frame) is implemented as the helpers below;
the public ``build_synthetic_game_batch`` is wired in Step 3.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from nba_sim.data.etl import GAMES_FILENAME
from nba_sim.features.context import (
    _PHASE_EARLY_MONTHS,
    _PHASE_LATE_MONTHS,
    _PHASE_MID_MONTHS,
    arena_altitude,
    travel_distance_miles,
)
from nba_sim.snapshot import (
    AS_OF_FILENAME,
    PLAYER_FEATURES_FILENAME,
    ROSTERS_FILENAME,
    TEAM_FEATURES_FILENAME,
    TEAM_LASTGAME_FILENAME,
)
from nba_sim.snapshot.refresh import season_for_date
from nba_sim.training.dataset import (
    _COUNT_STAT_COLS,
    _MATCHUP_TEAM_DIFF_BASES,
    BoxScoreDataset,
    collate_games,
)

# A projected lineup names its top-5-by-recent-minutes as starters; everyone
# else is bench. The model only ever saw is_starter as a 0/1 flag.
_N_STARTERS = 5
# Projected rotation. A roster player is marked active (``dnp = False``) only if
# they are among the top ``_ROTATION_SIZE`` by recent minutes AND clear the
# minutes floor (projected starters are always active). The team-point total in
# the sampler is a *sum* of per-player count heads, which fire for every active
# slot — so feeding the whole 15-man roster as ``dnp = False`` overshoots the
# total by ~30-45 pts. The model trained on ~10.6 active players/side, so we cap
# at the typical rotation. ``p_min_avg_10`` is per-appearance minutes (it does
# not encode play frequency), so a rank cap — not a minutes threshold alone — is
# what bounds the count. The floor (P(play|p_min_avg_10) crosses ~0.5 near 2-4
# min) lets genuinely shallow rotations fall below the cap. (v2PLAN §15.4 / §20:
# no injury feed, so the rotation is *projected* from recent minutes; the
# residual player-sum-vs-team-head gap is the model's own, present on real
# games too — closing it fully needs the team-head allocation, a training-time
# change.)
_ROTATION_SIZE = 10
_ROTATION_MIN_AVG_MINUTES = 5.0
# rest_days is clipped to 0..5 nights (v1 context.py convention).
_REST_DAYS_MAX = 5


@dataclasses.dataclass(frozen=True)
class _SnapshotData:
    """The four snapshot parquets + parsed provenance for one as_of date."""

    rosters: pl.DataFrame
    player_features: pl.DataFrame
    team_features: pl.DataFrame
    team_lastgame: pl.DataFrame
    as_of_date: _dt.date
    provenance: dict[str, Any]


def _load_snapshot(snapshot_dir: Path) -> _SnapshotData:
    """Read the snapshot directory written by ``nba-sim refresh``.

    Raises ``FileNotFoundError`` (with the documented refresh hint) if the
    directory or any of the five files is missing.
    """
    paths = {
        "rosters": snapshot_dir / ROSTERS_FILENAME,
        "player_features": snapshot_dir / PLAYER_FEATURES_FILENAME,
        "team_features": snapshot_dir / TEAM_FEATURES_FILENAME,
        "team_lastgame": snapshot_dir / TEAM_LASTGAME_FILENAME,
        "as_of": snapshot_dir / AS_OF_FILENAME,
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"snapshot at {snapshot_dir} is missing {missing} — run 'nba-sim refresh' first"
        )
    provenance: dict[str, Any] = json.loads(paths["as_of"].read_text())
    return _SnapshotData(
        rosters=pl.read_parquet(paths["rosters"]),
        player_features=pl.read_parquet(paths["player_features"]),
        team_features=pl.read_parquet(paths["team_features"]),
        team_lastgame=pl.read_parquet(paths["team_lastgame"]),
        as_of_date=_dt.date.fromisoformat(provenance["as_of_date"]),
        provenance=provenance,
    )


def _resolve_team_id(rosters: pl.DataFrame, abbr: str) -> int:
    """Map a team abbreviation to its id via the roster snapshot."""
    match = rosters.filter(pl.col("team_abbr") == abbr)
    if match.is_empty():
        raise LookupError(f"team {abbr} not in snapshot — try nba-sim refresh")
    return int(match["team_id"][0])


def _team_id_to_abbr(team_features: pl.DataFrame) -> dict[int, str]:
    """All teams' id -> abbr (team_features covers every team, so it resolves
    a ``last_arena_team_id`` that may be outside the chosen matchup)."""
    return dict(
        zip(
            (int(t) for t in team_features["team_id"].to_list()),
            team_features["team_abbr"].to_list(),
            strict=True,
        )
    )


def _load_season_games(interim_dir: Path, season: int) -> pl.DataFrame:
    """The interim ``games.parquet`` for ``season`` (empty frame if absent —
    h2h then degrades to 0, a no-prior-meeting cold start)."""
    path = interim_dir / str(season) / GAMES_FILENAME
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def _h2h_last_meeting_margin(
    season_games: pl.DataFrame,
    *,
    home_id: int,
    away_id: int,
    as_of_date: _dt.date,
) -> float:
    """Point margin of the pair's most recent meeting before ``as_of_date``,
    re-anchored to the **current home team** (positive = current home team won
    that meeting); 0.0 if they have not met this season.

    Mirrors :func:`nba_sim.features.matchup.head_to_head_last_margin` restricted
    to the as-of season (v1 builds h2h per-season), with the snapshot leakage
    rule (``date < as_of_date``) and dropped games excluded.
    """
    if season_games.is_empty():
        return 0.0
    pair = season_games.filter(
        (~pl.col("dropped"))
        & (pl.col("date") < as_of_date)
        & (
            ((pl.col("home_team_id") == home_id) & (pl.col("away_team_id") == away_id))
            | ((pl.col("home_team_id") == away_id) & (pl.col("away_team_id") == home_id))
        )
    )
    if pair.is_empty():
        return 0.0
    last = pair.sort(["date", "game_id"]).tail(1).row(0, named=True)
    if last["home_team_id"] == home_id:
        return float(last["home_pts"] - last["away_pts"])
    return float(last["away_pts"] - last["home_pts"])


def _schedule_context(
    team_lastgame: pl.DataFrame,
    *,
    team_id: int,
    as_of_date: _dt.date,
    venue_abbr: str,
    id_to_abbr: dict[int, str],
) -> tuple[float | None, bool, float | None]:
    """Return ``(rest_days, b2b, travel_miles_prev)`` for one team as-of.

    - ``rest_days`` = nights of rest = ``(as_of - last_game_date).days - 1``,
      clipped 0..5 (v1 ``context.py`` convention).
    - ``b2b`` = the gap is exactly one calendar day.
    - ``travel_miles_prev`` = great-circle from the team's last venue to the
      game venue (the home team's arena).

    All ``None`` / ``False`` when the team has no prior game this season — the
    snapshot's ``team_lastgame`` omits such teams (clean cold-start).
    """
    row = team_lastgame.filter(pl.col("team_id") == team_id)
    if row.is_empty():
        return None, False, None
    last = row.row(0, named=True)
    gap = (as_of_date - last["last_game_date"]).days
    rest_days = float(max(0, min(_REST_DAYS_MAX, gap - 1)))
    b2b = gap == 1
    last_arena_abbr = id_to_abbr.get(int(last["last_arena_team_id"]))
    travel = (
        travel_distance_miles(last_arena_abbr, venue_abbr)
        if last_arena_abbr is not None
        else None
    )
    return rest_days, b2b, travel


def _build_side_frame(
    player_features: pl.DataFrame,
    *,
    team_id: int,
    is_home: bool,
    rest_days: float | None,
    b2b: bool,
    travel: float | None,
    max_players: int,
) -> pl.DataFrame:
    """Per-side player frame matching the columns ``_build_side`` consumes.

    Filters ``player_features`` to ``team_id``, orders by ``p_min_avg_10`` desc
    and keeps the top ``max_players`` (mirrors the dataset's minutes-descending
    truncation). Overwrites the two schedule-derived null numerics
    (``rest_days`` / ``travel_miles_prev``) with the computed values, leaving
    the two opp-vs-position numerics null (→ standardized mean). Adds the four
    bool flags and the dummy ``minutes`` + count-stat columns ``_build_side``
    reads but the simulator discards.

    ``dnp`` is **projected from recent minutes**, not hardcoded ``False``: a
    player is active only if a projected starter (top-5) or among the top
    ``_ROTATION_SIZE`` by recent minutes while clearing
    ``_ROTATION_MIN_AVG_MINUTES`` (null → bench). This caps the active set near
    the ~10.6-players/side the model trained on; feeding the whole 15-man
    roster as ``dnp = False`` is what made the summed team points overshoot
    (the count heads fire for every "active" slot). ``is_starter`` = top-5 by
    recent minutes.
    """
    side = (
        player_features.filter(pl.col("team_id") == team_id)
        .sort("p_min_avg_10", descending=True, nulls_last=True)
        .head(max_players)
    )
    rank = pl.int_range(0, pl.len())
    is_starter = rank < _N_STARTERS
    in_rotation = (rank < _ROTATION_SIZE) & (pl.col("p_min_avg_10") >= _ROTATION_MIN_AVG_MINUTES)
    active = (is_starter | in_rotation).fill_null(False)
    return side.with_columns(
        is_starter.alias("is_starter"),
        (~active).alias("dnp"),
        pl.lit(is_home).alias("is_home"),
        pl.lit(b2b).alias("b2b"),
        pl.lit(rest_days, dtype=pl.Float64).alias("rest_days"),
        pl.lit(travel, dtype=pl.Float64).alias("travel_miles_prev"),
        pl.lit(0.0, dtype=pl.Float64).alias("minutes"),
        *[pl.lit(0.0, dtype=pl.Float64).alias(stat) for stat in _COUNT_STAT_COLS],
    )


# ---------------------------------------------------------------------------
# Step 2 — game-level context + matchup columns
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _GameContext:
    """Game-level (both-sides-shared) context derived from as_of + venue."""

    season_phase: str
    day_of_week: int
    month: int
    altitude_ft: float


def _season_phase(d: _dt.date) -> str:
    """Month-bucketed season phase, matching v1 ``context.py`` exactly for
    regular-season dates (the playoffs bucket needs an ``is_playoffs`` flag we
    don't have for a hypothetical future game, so May+ falls through to
    ``"late"`` — same as v1's non-playoff path)."""
    month = d.month
    if month in _PHASE_EARLY_MONTHS:
        return "early"
    if month in _PHASE_MID_MONTHS:
        return "mid"
    if month in _PHASE_LATE_MONTHS:
        return "late"
    return "late"


def _game_context(as_of_date: _dt.date, venue_abbr: str) -> _GameContext:
    """Compute the shared context block. ``day_of_week`` mirrors v1's
    ``polars weekday() - 1`` (Mon=0), which equals ``date.weekday()``."""
    return _GameContext(
        season_phase=_season_phase(as_of_date),
        day_of_week=as_of_date.weekday(),
        month=as_of_date.month,
        altitude_ft=arena_altitude(venue_abbr),
    )


def _team_stat_row(team_features: pl.DataFrame, team_id: int) -> dict[str, float | None]:
    """The 9 ``t_*`` rolling stats for one team (all None if the team is
    absent — ``_build_matchup`` then falls back to league means)."""
    row = team_features.filter(pl.col("team_id") == team_id)
    if row.is_empty():
        return dict.fromkeys(_MATCHUP_TEAM_DIFF_BASES, None)
    r = row.row(0, named=True)
    return {base: r[base] for base in _MATCHUP_TEAM_DIFF_BASES}


def _attach_context_matchup(
    side: pl.DataFrame,
    *,
    own_stats: dict[str, float | None],
    opp_stats: dict[str, float | None],
    game_ctx: _GameContext,
    h2h: float,
) -> pl.DataFrame:
    """Broadcast the game-level context + matchup columns onto every row of one
    side, so ``_build_context`` / ``_build_matchup`` can read them off the
    side's first row.

    Each side carries **its own** 9 ``t_*`` (the matchup diff reads home's minus
    away's) plus the **opponent's** ``t_def_rtg_10`` / ``t_pace_10`` as
    ``opp_def_rtg_10`` / ``opp_pace_10``. ``h2h`` is the home-anchored margin for
    the home side (negated for the away side; only the home row's value is read).
    """
    return side.with_columns(
        *[pl.lit(own_stats[b], dtype=pl.Float64).alias(b) for b in _MATCHUP_TEAM_DIFF_BASES],
        pl.lit(opp_stats["t_def_rtg_10"], dtype=pl.Float64).alias("opp_def_rtg_10"),
        pl.lit(opp_stats["t_pace_10"], dtype=pl.Float64).alias("opp_pace_10"),
        pl.lit(h2h, dtype=pl.Float64).alias("h2h_last_meeting_margin"),
        pl.lit(game_ctx.season_phase).alias("season_phase"),
        pl.lit(game_ctx.day_of_week, dtype=pl.Int64).alias("day_of_week"),
        pl.lit(game_ctx.month, dtype=pl.Int64).alias("month"),
        pl.lit(game_ctx.altitude_ft, dtype=pl.Float64).alias("altitude_ft"),
    )


def build_synthetic_game_batch(
    *,
    home_team: str,
    away_team: str,
    snapshot_dir: Path = Path("data/snapshot"),
    train_parquet: Path = Path("data/processed/train.parquet"),
    interim_dir: Path = Path("data/interim"),
    max_players: int = 15,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, str]], list[tuple[int, str]], str]:
    """Build a ``(model_batch, home_players, away_players, as_of_iso)`` tuple.

    ``model_batch`` is a ``dict[str, torch.Tensor]`` with ``B == 1`` that goes
    straight into ``model.forward(batch)`` — no teacher-forcing keys. See the
    module docstring for the assembly steps. ``interim_dir`` is read only to
    compute ``h2h_last_meeting_margin`` for the chosen pair.

    ``home_players`` / ``away_players`` are ``(player_id, player_name)`` in the
    same order the batch's active slots use (minutes-descending), so the
    sampler can map predicted slots back to names.
    """
    snap = _load_snapshot(snapshot_dir)
    # The train split pins the player_id_map + feature_stats (standardization)
    # and the roster slot count P, exactly as the v1 eval path does.
    train_ds = BoxScoreDataset(train_parquet)
    cap = min(max_players, train_ds.max_players)

    home_id = _resolve_team_id(snap.rosters, home_team)
    away_id = _resolve_team_id(snap.rosters, away_team)

    id_to_abbr = _team_id_to_abbr(snap.team_features)
    season_games = _load_season_games(interim_dir, season_for_date(snap.as_of_date))
    home_h2h = _h2h_last_meeting_margin(
        season_games, home_id=home_id, away_id=away_id, as_of_date=snap.as_of_date
    )

    # The game is played at the home team's arena, so both teams' travel /
    # altitude / season-phase share that venue.
    game_ctx = _game_context(snap.as_of_date, home_team)
    home_sched = _schedule_context(
        snap.team_lastgame, team_id=home_id, as_of_date=snap.as_of_date,
        venue_abbr=home_team, id_to_abbr=id_to_abbr,
    )
    away_sched = _schedule_context(
        snap.team_lastgame, team_id=away_id, as_of_date=snap.as_of_date,
        venue_abbr=home_team, id_to_abbr=id_to_abbr,
    )
    home_stats = _team_stat_row(snap.team_features, home_id)
    away_stats = _team_stat_row(snap.team_features, away_id)

    home_side = _attach_context_matchup(
        _build_side_frame(
            snap.player_features, team_id=home_id, is_home=True,
            rest_days=home_sched[0], b2b=home_sched[1], travel=home_sched[2],
            max_players=cap,
        ),
        own_stats=home_stats, opp_stats=away_stats, game_ctx=game_ctx, h2h=home_h2h,
    )
    away_side = _attach_context_matchup(
        _build_side_frame(
            snap.player_features, team_id=away_id, is_home=False,
            rest_days=away_sched[0], b2b=away_sched[1], travel=away_sched[2],
            max_players=cap,
        ),
        own_stats=away_stats, opp_stats=home_stats, game_ctx=game_ctx, h2h=-home_h2h,
    )
    if home_side.is_empty() or away_side.is_empty():
        raise LookupError(
            "home or away team has no players in the snapshot — try nba-sim refresh"
        )

    # Reuse the v1 dataset builders so feature assembly + standardization are
    # byte-identical to training.
    home_data = train_ds._build_side(home_side)
    away_data = train_ds._build_side(away_side)
    context = train_ds._build_context(home_side.row(0, named=True), away_side.row(0, named=True))
    matchup = train_ds._build_matchup(home_side.row(0, named=True), away_side.row(0, named=True))

    item = {
        "home_player_feats": home_data["feats"],
        "away_player_feats": away_data["feats"],
        "home_player_ids": home_data["player_ids"],
        "away_player_ids": away_data["player_ids"],
        "home_role_ids": home_data["role_ids"],
        "away_role_ids": away_data["role_ids"],
        "home_mask": home_data["mask"],
        "away_mask": away_data["mask"],
        "context": context,
        "matchup": matchup,
    }
    batch = collate_games([item])

    home_players = [
        (int(pid), str(name))
        for pid, name in zip(
            home_side["player_id"].to_list(), home_side["player_name"].to_list(), strict=True
        )
    ]
    away_players = [
        (int(pid), str(name))
        for pid, name in zip(
            away_side["player_id"].to_list(), away_side["player_name"].to_list(), strict=True
        )
    ]
    return batch, home_players, away_players, snap.as_of_date.isoformat()
