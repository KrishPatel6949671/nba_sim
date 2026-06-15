"""Snapshot refresh orchestrator — builds ``data/snapshot/`` for one as_of date.

Four stages (v2PLAN.md §14):

    A. Resolve as_of_date.  Default = latest interim game date + 1 day
       ("the day after the most recent data we have"); ``--as-of`` overrides.
    B. Resolve rosters.     Offline (Phase 6) = derive from interim appearances
       (:mod:`nba_sim.snapshot.rosters`); live CommonTeamRoster = Phase 8.
    C. Build features as-of. The heart of v2: append a **synthetic target row**
       per (team, player) dated as_of, run the existing rolling kernels over
       ``interim + synthetic``, then filter back to the synthetic rows.
    D. Atomic write.        Stage everything under ``snapshot/.tmp/`` then swap
       it into place; ``as_of.json`` is written last so a partial refresh
       never advertises as complete.

Leakage discipline (the one correctness rule the §14.3 pseudo-code glosses
over): features are built from interim games with ``date < as_of_date``
(strict). This is both the v1 rule (PLAN.md §3.2 — strictly prior games) and
what makes the §12 / §14.6 "feature equivalence at a known date" test pass:
with ``--as-of <a real game date D>`` the real games played on D must be
excluded from the synthetic row's window, exactly as the processed pipeline
excludes a row's own game. With the default (latest + 1) it's a no-op.

Synthetic-target-row strategy (Appendix B, refined for the all-players refresh)
-------------------------------------------------------------------------------
We append **zero-filled** synthetic target rows dated ``as_of_date`` and run
the existing kernels over ``interim + synthetic``. Because every aggregate is
``shift(1)`` / ``cum_sum - current`` and the synthetic rows sort last (their
date is strictly greater than every interim row, which is filtered to
``date < as_of``), the zero-filled targets never enter their own windows —
each synthetic row's features are precisely what the model would have seen had
the game been played on as_of.

Two refinements vs. the per-matchup worked example in Appendix B (which builds
one ``SNAPSHOT_<abbr>`` game per team for a *single* matchup):

1. **Players are deduplicated by ``player_id``, not ``(team, player)``.**
   Rolling form is a property of the *player*, not the team; a player traded
   mid-season appears on two teams' rosters but has one true last-10. If we
   emitted two same-date synthetic rows for one ``player_id``, ``shift(1)``
   over the ``player_id`` partition would let the first synthetic (0-minute)
   row leak into the second's window. So we compute features on **one
   synthetic row per distinct ``player_id``** (sharing a single sentinel game
   ``SNAPSHOT_<as_of>``), then **join those player-level features back onto
   the per-``(team, player)`` rosters frame by ``player_id``**. Both roster
   entries of a traded player get identical (correct) rolling form; their
   team-dependent identity (team_abbr, most-recent position) comes from
   rosters.
2. Teams use one synthetic team-box row per distinct ``team_id`` under a
   single sentinel game ``SNAPSHOT_TEAM_<as_of>`` — ``team_id`` is already
   unique per team so no dedup is needed.

Filter-back is by sentinel prefix: player features keep ``game_id`` starting
``SNAPSHOT_`` (the only synthetic rows in the player concat) and team features
keep ``SNAPSHOT_TEAM_``.

``player_features.parquet`` column contract (resolved decision: "include as
null")
---------------------------------------------------------------------------
It carries the **full** ``_PLAYER_NUMERIC_COLS`` set (47), so it literally
mirrors what ``BoxScoreDataset._build_side`` consumes, but four of those
columns are **opponent / as-of-dependent** and cannot be produced by a
matchup-agnostic refresh, so they are written **null** and filled by
``build.py`` at simulate time (FeatureStats maps null → standardized league
mean as a clean cold-start):

    rest_days, travel_miles_prev          (depend on as_of vs each team's
                                           last game + the simulated venue)
    opp_def_rtg_vs_pos, opp_blk_allowed_vs_pos  (depend on the chosen opponent)

The remaining 43 are produced here by ``player_rolling`` + ``season_to_date``.
The four per-game bool flags (``is_starter`` / ``dnp`` / ``is_home`` / ``b2b``)
are likewise lineup / matchup dependent and are assembled in ``build.py`` at
simulate time, not stored here.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from nba_sim.data.etl import (
    GAMES_FILENAME,
    PLAYER_BOX_FILENAME,
    TEAM_BOX_FILENAME,
    interim_dir,
)
from nba_sim.features.rolling import player_rolling, season_to_date, team_rolling
from nba_sim.snapshot import (
    AS_OF_FILENAME,
    CODE_VERSION,
    PLAYER_FEATURES_FILENAME,
    QA_REPORT_FILENAME,
    ROSTERS_FILENAME,
    TEAM_FEATURES_FILENAME,
    TEAM_LASTGAME_FILENAME,
    TMP_DIRNAME,
    snapshot_dir,
)
from nba_sim.snapshot.rosters import build_rosters
from nba_sim.training.dataset import _MATCHUP_TEAM_DIFF_BASES, _PLAYER_NUMERIC_COLS

logger = logging.getLogger(__name__)

# Sentinel game-id prefixes for the synthetic target rows (Appendix B). The
# player and team paths use distinct prefixes so each rolling kernel sees one
# synthetic row per partition and the filter-back step is unambiguous.
SNAPSHOT_GAME_PREFIX = "SNAPSHOT_"
SNAPSHOT_TEAM_GAME_PREFIX = "SNAPSHOT_TEAM_"

# Of the 47 _PLAYER_NUMERIC_COLS, these are not producible by a matchup-
# agnostic refresh and are written null (see module docstring). build.py fills
# them at simulate time.
NULL_AT_REFRESH_NUMERIC: tuple[str, ...] = (
    "rest_days",
    "travel_miles_prev",
    "opp_def_rtg_vs_pos",
    "opp_blk_allowed_vs_pos",
)


# ---------------------------------------------------------------------------
# Stage A — as_of date resolution
# ---------------------------------------------------------------------------

def season_for_date(d: _dt.date) -> int:
    """Season start-year that calendar date ``d`` falls in.

    NBA season N (the "N-(N+1)" season) runs Oct of year N → Jun of year
    N+1, so months Jul-Dec map to year N and Jan-Jun map to year N-1.
    """
    return d.year if d.month >= 7 else d.year - 1


def _coerce_date(value: str | _dt.date) -> _dt.date:
    """Normalize an ISO ``"YYYY-MM-DD"`` string / date / datetime to a date.

    ``datetime`` is a subclass of ``date`` so the order of the isinstance
    checks matters — narrow to its calendar date first.
    """
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value)
    raise TypeError(f"as_of must be an ISO date string or datetime.date, got {type(value)!r}")


def _discover_seasons(interim_root: Path) -> list[int]:
    """Sorted interim season start-years present on disk.

    A season is "present" iff ``<interim_root>/<int>/games.parquet`` exists
    (mirrors :func:`nba_sim.data.etl._outputs_for_season`). Non-numeric dirs
    (e.g. a stray ``.tmp``) are ignored.
    """
    if not interim_root.exists():
        return []
    seasons: list[int] = []
    for child in interim_root.iterdir():
        if child.is_dir() and child.name.isdigit() and (child / GAMES_FILENAME).exists():
            seasons.append(int(child.name))
    return sorted(seasons)


def _interim_latest_game_date(seasons: list[int], *, interim_root: Path) -> _dt.date:
    """Max ``date`` across every season's ``games.parquet``.

    Scans only the ``date`` column (cheap) and ignores ``dropped`` games is
    *not* done here on purpose: the latest calendar date we have data for is
    what defines "the day after the most recent data," regardless of whether
    that particular game survived QA.
    """
    latest: _dt.date | None = None
    for s in seasons:
        path = interim_root / str(s) / GAMES_FILENAME
        if not path.exists():
            continue
        val = (
            pl.scan_parquet(path)
            .select(pl.col("date").max().alias("max_date"))
            .collect()["max_date"][0]
        )
        if val is None:
            continue
        if latest is None or val > latest:
            latest = val
    if latest is None:
        raise ValueError(
            "no dated interim games found under "
            f"{interim_root} (seasons={seasons}) — run the v1 ETL first"
        )
    return latest


def resolve_as_of_date(
    as_of: str | _dt.date | None = None,
    *,
    interim_root: Path | None = None,
) -> tuple[_dt.date, _dt.date]:
    """Resolve the snapshot's reference date (Stage A, §14.1).

    Returns ``(as_of_date, interim_latest_game_date)``.

    - ``as_of`` given (ISO string or date) → used verbatim as ``as_of_date``.
    - ``as_of is None`` → ``as_of_date = interim_latest_game_date + 1 day``.

    ``interim_latest_game_date`` is the max ``date`` across
    ``<interim>/<season>/games.parquet`` and is surfaced to the user (and
    recorded in provenance) before any network call so an obviously-wrong
    resolution can be aborted.
    """
    root = interim_root if interim_root is not None else interim_dir()
    seasons = _discover_seasons(root)
    if not seasons:
        raise FileNotFoundError(
            f"no interim seasons found under {root} — run the v1 ETL "
            "(`nba-sim build`) before refreshing a snapshot"
        )
    interim_latest = _interim_latest_game_date(seasons, interim_root=root)
    as_of_date = (
        interim_latest + _dt.timedelta(days=1) if as_of is None else _coerce_date(as_of)
    )
    return as_of_date, interim_latest


def _seasons_up_to(as_of_date: _dt.date, *, interim_root: Path) -> list[int]:
    """Interim season start-years that can contribute games before ``as_of_date``.

    Any season whose start-year is ``> season_for_date(as_of_date)`` begins
    after ``as_of_date``, so it can hold no row with ``date < as_of_date``.
    The strict date filter in :func:`_load_interim` does the exact cut; this
    just avoids opening parquet files that can't contribute.
    """
    target = season_for_date(as_of_date)
    return [s for s in _discover_seasons(interim_root) if s <= target]


def _load_interim(
    seasons: list[int],
    *,
    as_of_date: _dt.date,
    interim_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Concat ``games`` / ``player_box`` / ``team_box`` across ``seasons`` and
    filter to interim games with ``date < as_of_date`` (the leakage rule).

    Returns ``(games, player_box, team_box)``. The box frames are restricted
    to the surviving ``game_id`` set so no row from an on/after-as_of game
    sneaks into a rolling window. ``vertical_relaxed`` tolerates mild
    cross-season dtype drift (same discipline as ``etl._build_split_frame``).
    """
    games_parts: list[pl.DataFrame] = []
    player_parts: list[pl.DataFrame] = []
    team_parts: list[pl.DataFrame] = []
    for s in seasons:
        d = interim_root / str(s)
        games_parts.append(pl.read_parquet(d / GAMES_FILENAME))
        player_parts.append(pl.read_parquet(d / PLAYER_BOX_FILENAME))
        team_parts.append(pl.read_parquet(d / TEAM_BOX_FILENAME))

    games = pl.concat(games_parts, how="vertical_relaxed").filter(
        pl.col("date") < as_of_date
    )
    player_box = pl.concat(player_parts, how="vertical_relaxed")
    team_box = pl.concat(team_parts, how="vertical_relaxed")

    # Keep only box rows whose game survived the leakage cut.
    keep_ids = games.select("game_id")
    player_box = player_box.join(keep_ids, on="game_id", how="inner")
    team_box = team_box.join(keep_ids, on="game_id", how="inner")
    return games, player_box, team_box


# ---------------------------------------------------------------------------
# Stage C — synthetic target rows + features
# ---------------------------------------------------------------------------

# Identity columns carried on player_features (everything else is a numeric
# feature). These come from the rosters frame, not the synthetic box row.
_PLAYER_IDENTITY_COLS: tuple[str, ...] = (
    "team_id",
    "team_abbr",
    "player_id",
    "player_name",
    "position",
)


def _zero_for(dtype: pl.DataType) -> object:
    """The zero-filled default for a synthetic-row column of ``dtype``:
    ``0`` for ints, ``0.0`` for floats, ``False`` for bools, else null."""
    if dtype.is_integer():
        return 0
    if dtype.is_float():
        return 0.0
    if dtype == pl.Boolean:
        return False
    return None


def _coerce_to_schema(base: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Return ``base`` reshaped to exactly ``schema`` (columns, order, dtypes).

    Columns already on ``base`` (the identity / computed ones) are cast to the
    target dtype; absent columns are filled with :func:`_zero_for` so the
    synthetic frame concats cleanly onto the interim frame. Broadcasts over
    ``base``'s height, so a 0-row ``base`` yields a 0-row correctly-typed frame.
    """
    exprs: list[pl.Expr] = []
    for name, dtype in schema.items():
        if name in base.columns:
            exprs.append(pl.col(name).cast(dtype).alias(name))
        else:
            exprs.append(pl.lit(_zero_for(dtype), dtype=dtype).alias(name))
    return base.select(exprs)


def _build_synthetic_target(
    rosters: pl.DataFrame,
    player_box: pl.DataFrame,
    team_box: pl.DataFrame,
    games: pl.DataFrame,
    *,
    as_of_date: _dt.date,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Construct the zero-filled synthetic rows (Appendix B / module docstring).

    Returns ``(syn_player_box, syn_team_box, syn_player_games,
    syn_team_games)``, each schema-matched (via :func:`_coerce_to_schema`) to
    the corresponding interim frame so the concat in the feature builders is a
    clean vertical stack:

    - ``syn_player_box``  — one zero-filled row per **distinct ``player_id``**
      in ``rosters`` (dedup, see module docstring §1), all sharing
      ``game_id = SNAPSHOT_<as_of>``. ``team_id`` is a placeholder (0): the
      only consumer of a synthetic row's ``team_id`` is the USG join, whose
      result is shifted out of the synthetic row's own window.
    - ``syn_team_box``    — one zero-filled row per **distinct ``team_id``** in
      ``rosters``, all sharing ``game_id = SNAPSHOT_TEAM_<as_of>``.
    - ``syn_player_games`` / ``syn_team_games`` — a single Game-shaped row each
      (``date = as_of_date``, ``season``, ``dropped=False``) for the two
      sentinel games so ``_attach_dates`` can inner-join date/season on.
    """
    as_of_iso = as_of_date.isoformat()
    player_game_id = f"{SNAPSHOT_GAME_PREFIX}{as_of_iso}"
    team_game_id = f"{SNAPSHOT_TEAM_GAME_PREFIX}{as_of_iso}"

    # One synthetic player row per distinct player_id (dedup — module docstring).
    player_base = rosters.select("player_id").unique().with_columns(
        game_id=pl.lit(player_game_id),
        team_id=pl.lit(0),
    )
    syn_player_box = _coerce_to_schema(player_base, player_box.schema)

    # One synthetic team row per distinct team_id.
    team_base = rosters.select("team_id", "team_abbr").unique().with_columns(
        game_id=pl.lit(team_game_id),
    )
    syn_team_box = _coerce_to_schema(team_base, team_box.schema)

    # Matching single-row Game headers. Only game_id / date / season / dropped
    # are read downstream (_attach_dates); the rest are filled by schema.
    def _game_row(game_id: str) -> pl.DataFrame:
        base = pl.DataFrame(
            {"game_id": [game_id]},
        ).with_columns(
            season=pl.lit(season),
            date=pl.lit(as_of_date),
            dropped=pl.lit(False),  # explicit so the row survives _attach_dates
        )
        return _coerce_to_schema(base, games.schema)

    return (
        syn_player_box,
        syn_team_box,
        _game_row(player_game_id),
        _game_row(team_game_id),
    )


def _build_player_features(
    rosters: pl.DataFrame,
    games: pl.DataFrame,
    player_box: pl.DataFrame,
    team_box: pl.DataFrame,
    syn_player_box: pl.DataFrame,
    syn_player_games: pl.DataFrame,
) -> pl.DataFrame:
    """Build ``player_features.parquet`` (§13.4) via the synthetic-row trick.

    Concats the synthetic player rows onto ``player_box`` / ``games``, runs the
    existing :func:`~nba_sim.features.rolling.player_rolling` (with ``team_box``
    for USG%) and :func:`~nba_sim.features.rolling.season_to_date`, filters back
    to the synthetic (``SNAPSHOT_``) rows to get **player-level** features, then
    joins those onto the per-``(team, player)`` ``rosters`` frame by
    ``player_id`` (module docstring §1).

    Output columns: ``_PLAYER_IDENTITY_COLS`` + the full ``_PLAYER_NUMERIC_COLS``
    block, with :data:`NULL_AT_REFRESH_NUMERIC` written null (filled by
    ``build.py`` at simulate time).
    """
    combined_box = pl.concat([player_box, syn_player_box], how="vertical_relaxed")
    combined_games = pl.concat([games, syn_player_games], how="vertical_relaxed")

    p_roll = player_rolling(combined_box, combined_games, team_box=team_box)
    p_std = season_to_date(combined_box, combined_games)

    null_set = set(NULL_AT_REFRESH_NUMERIC)
    std_cols = [c for c in _PLAYER_NUMERIC_COLS if c not in null_set and c.startswith("std_")]

    joined = p_roll.join(
        p_std.select(["game_id", "player_id", *std_cols]),
        on=["game_id", "player_id"],
        how="left",
    ).filter(pl.col("game_id").str.starts_with(SNAPSHOT_GAME_PREFIX))

    # The 43 non-null numerics must all have been produced; fail loudly if the
    # upstream feature layout drifts out from under us.
    produced = [c for c in _PLAYER_NUMERIC_COLS if c not in null_set]
    missing = [c for c in produced if c not in joined.columns]
    if missing:
        raise RuntimeError(f"player feature columns not produced by kernels: {missing}")

    feats_by_player = joined.select(["player_id", *produced]).unique(
        subset=["player_id"], keep="first"
    )

    # Per-(team, player) identity from rosters; player-level features broadcast.
    numeric_exprs = [
        pl.lit(None, dtype=pl.Float64).alias(c) if c in null_set else pl.col(c)
        for c in _PLAYER_NUMERIC_COLS
    ]
    return (
        rosters.select(_PLAYER_IDENTITY_COLS)
        .join(feats_by_player, on="player_id", how="left")
        .select([*(pl.col(c) for c in _PLAYER_IDENTITY_COLS), *numeric_exprs])
        .sort(["team_id", "player_id"])
    )


def _build_team_features(
    games: pl.DataFrame,
    team_box: pl.DataFrame,
    syn_team_box: pl.DataFrame,
    syn_team_games: pl.DataFrame,
) -> pl.DataFrame:
    """Build ``team_features.parquet`` (§13.5): one row per team with the 9
    standardized team-rolling cols (:data:`_MATCHUP_TEAM_DIFF_BASES` —
    ``t_pace_5/10``, ``t_off_rtg_5/10``, ``t_def_rtg_5/10``, ``t_win_pct_10``,
    ``t_pts_avg_10``, ``t_pts_allowed_10``) plus ``team_id`` / ``team_abbr``.

    Same synthetic-row trick: append the ``SNAPSHOT_TEAM_`` rows, run
    :func:`~nba_sim.features.rolling.team_rolling`, filter back.
    """
    combined_box = pl.concat([team_box, syn_team_box], how="vertical_relaxed")
    combined_games = pl.concat([games, syn_team_games], how="vertical_relaxed")

    t_roll = team_rolling(combined_box, combined_games)
    return (
        t_roll.filter(pl.col("game_id").str.starts_with(SNAPSHOT_TEAM_GAME_PREFIX))
        .select(["team_id", "team_abbr", *_MATCHUP_TEAM_DIFF_BASES])
        .sort("team_id")
    )


# ---------------------------------------------------------------------------
# Stage C (cont.) — team_lastgame
# ---------------------------------------------------------------------------

# §13.6 dtype contract.
_TEAM_LASTGAME_SCHEMA: dict[str, pl.DataType] = {
    "team_id": pl.Int64(),
    "last_game_date": pl.Date(),
    "last_arena_team_id": pl.Int64(),
    "last_was_home": pl.Boolean(),
}


def _build_team_lastgame(
    games: pl.DataFrame,
    *,
    as_of_date: _dt.date,
) -> pl.DataFrame:
    """Build ``team_lastgame.parquet`` (§13.6) inline from interim games.

    Per team_id: the most recent game (``date < as_of_date``, **in the as-of
    season**) → ``last_game_date``, ``last_arena_team_id`` (the home team of
    that game — the physical venue), ``last_was_home`` (whether the team
    hosted). Drives rest_days / b2b / travel at simulate time.

    Two rules mirror :func:`nba_sim.features.context.add_context_features` so
    simulate-time schedule features match what the model trained on:

    - **Dropped games are excluded** — the prior game for a rest interval must
      be one that actually happened (context.py filters ``~dropped``).
    - **Restricted to the as-of season** — context.py shifts within a
      ``(team_id, season)`` partition, so the first game of a season has null
      rest. A team with no game yet this season gets **no row here**, and
      ``build.py`` then emits null rest/travel (clean cold-start), rather than
      a spurious multi-month gap clipped to 5.
    """
    season = season_for_date(as_of_date)
    base = games.filter(
        (~pl.col("dropped"))
        & (pl.col("date") < as_of_date)
        & (pl.col("season") == season)
    )
    if base.is_empty():
        return pl.DataFrame(schema=_TEAM_LASTGAME_SCHEMA)

    # Fan out to one row per (team, game); the arena is the home team's in
    # both views (the game is physically at the home building).
    home = base.select(
        pl.col("home_team_id").alias("team_id"),
        pl.col("date").alias("last_game_date"),
        pl.col("home_team_id").alias("last_arena_team_id"),
        pl.lit(True).alias("last_was_home"),
        "game_id",
    )
    away = base.select(
        pl.col("away_team_id").alias("team_id"),
        pl.col("date").alias("last_game_date"),
        pl.col("home_team_id").alias("last_arena_team_id"),
        pl.lit(False).alias("last_was_home"),
        "game_id",
    )
    # Sort ascending so `.last()` within each team group is the most recent
    # game (game_id breaks any same-day tie deterministically).
    long = pl.concat([home, away]).sort(["team_id", "last_game_date", "game_id"])
    out = long.group_by("team_id").agg(
        last_game_date=pl.col("last_game_date").last(),
        last_arena_team_id=pl.col("last_arena_team_id").last(),
        last_was_home=pl.col("last_was_home").last(),
    )
    return out.select(
        [pl.col(c).cast(dt) for c, dt in _TEAM_LASTGAME_SCHEMA.items()]
    ).sort("team_id")


# ---------------------------------------------------------------------------
# Stage D — provenance + atomic write
# ---------------------------------------------------------------------------

def _build_provenance(
    *,
    as_of_date: _dt.date,
    interim_latest_game_date: _dt.date,
    n_teams: int,
    n_players: int,
    offline: bool,
) -> dict[str, Any]:
    """Assemble the ``as_of.json`` payload (§13.2): as_of_date, refreshed_at,
    source ('interim-only' offline / 'nba_api+interim' live), n_teams,
    n_players, interim_latest_game_date, code_version,
    model_checkpoint_used_for_validation (None unless a sanity forward pass
    ran).
    """
    return {
        "as_of_date": as_of_date.isoformat(),
        # UTC, second precision, trailing "Z" — matches the §13.2 example.
        "refreshed_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "interim-only" if offline else "nba_api+interim",
        "n_teams": n_teams,
        "n_players": n_players,
        "interim_latest_game_date": interim_latest_game_date.isoformat(),
        "code_version": CODE_VERSION,
        "model_checkpoint_used_for_validation": None,
    }


def _snapshot_is_current(dest: Path, interim_latest_game_date: _dt.date) -> bool:
    """True iff an existing snapshot was already built from interim data this
    fresh (its recorded ``interim_latest_game_date`` matches the current one).

    Only consulted on the default (no explicit ``--as-of``) path; a corrupt or
    absent ``as_of.json`` reads as "not current" so refresh always proceeds.
    """
    p = dest / AS_OF_FILENAME
    if not p.exists():
        return False
    try:
        prov = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return bool(prov.get("interim_latest_game_date") == interim_latest_game_date.isoformat())


def _write_snapshot_atomic(
    frames: dict[str, pl.DataFrame],
    provenance: dict[str, Any],
    *,
    dest: Path,
    qa_report: dict[str, Any] | None = None,
) -> None:
    """Stage all parquets under ``dest/.tmp/`` then promote into place (§14.4).

    Build phase writes every parquet into ``dest/.tmp/``, then ``qa_report.json``
    (live only — the rejected roster rows), and ``as_of.json`` last; only once
    the build fully succeeds do we promote the files into ``dest`` with
    ``os.replace`` (atomic per file on the same filesystem), ``as_of.json``
    **last**. So a crash during the build never touches the live ``dest`` (no
    new ``as_of.json`` appears), and the completeness marker becoming visible
    implies every file beside it is already the new one.
    """
    staging = dest / TMP_DIRNAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    # Build phase — parquets, then qa_report (if any), then as_of.json last.
    for filename, df in frames.items():
        df.write_parquet(staging / filename)
    if qa_report is not None:
        (staging / QA_REPORT_FILENAME).write_text(json.dumps(qa_report, indent=2))
    (staging / AS_OF_FILENAME).write_text(json.dumps(provenance, indent=2))

    # Promote phase — move staged files into dest, qa_report just before the
    # as_of.json completeness marker (which goes last).
    for filename in frames:
        os.replace(staging / filename, dest / filename)
    if qa_report is not None:
        os.replace(staging / QA_REPORT_FILENAME, dest / QA_REPORT_FILENAME)
    os.replace(staging / AS_OF_FILENAME, dest / AS_OF_FILENAME)

    shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refresh(
    *,
    as_of: str | _dt.date | None = None,
    force: bool = False,
    offline: bool = True,
    refresh_cache: bool = False,
    snapshot_root: Path | None = None,
    interim_root: Path | None = None,
) -> Path:
    """Rebuild ``data/snapshot/`` for one as_of date. Returns the snapshot dir.

    Flow:
        A. ``as_of_date, interim_latest = resolve_as_of_date(as_of)``; surface
           the resolved date; honor the "snapshot newer than interim" skip
           unless ``force``.
        B. ``rosters = build_rosters(..., offline=offline)``.
        C. ``player_features`` / ``team_features`` / ``team_lastgame``.
        D. ``_write_snapshot_atomic({...}, provenance)``.

    Parameters
    ----------
    as_of
        ISO ``"YYYY-MM-DD"`` / date / None (None → latest interim + 1 day).
    force
        Bypass the "snapshot already newer than interim" skip.
    offline
        Default ``True`` — derive rosters from interim appearances (no
        nba_api). ``False`` fetches live ``CommonTeamRoster`` per team and
        also writes ``qa_report.json`` with any rejected rows.
    refresh_cache
        Live path only: bypass the on-disk fetch cache for the roster /
        player-info endpoints (the ``nba-sim refresh --refresh`` flag,
        consistent with v1 ``nba-sim fetch --refresh``).
    snapshot_root, interim_root
        Override roots (tests redirect via ``NBA_SIM_SNAPSHOT_DIR`` /
        ``NBA_SIM_INTERIM_DIR``); default to the env-aware helpers.
    """
    iroot = interim_root if interim_root is not None else interim_dir()
    sroot = snapshot_root if snapshot_root is not None else snapshot_dir()

    # Stage A — resolve the reference date and surface it before any work.
    as_of_date, interim_latest = resolve_as_of_date(as_of, interim_root=iroot)
    season = season_for_date(as_of_date)
    logger.info(
        "refresh: as_of=%s (season %d), interim latest=%s, offline=%s",
        as_of_date, season, interim_latest, offline,
    )

    # Idempotency skip (default path only): if the interim data hasn't advanced
    # since the last refresh, rebuilding would reproduce the same snapshot. An
    # explicit --as-of or --force always rebuilds.
    if as_of is None and not force and _snapshot_is_current(sroot, interim_latest):
        logger.info(
            "snapshot at %s already built from interim latest %s — skipping "
            "(pass force=True to override)",
            sroot, interim_latest,
        )
        return sroot

    seasons = _seasons_up_to(as_of_date, interim_root=iroot)
    games, player_box, team_box = _load_interim(
        seasons, as_of_date=as_of_date, interim_root=iroot
    )

    # Stage B — rosters (offline = derive from interim appearances; live =
    # CommonTeamRoster per team, with rejected rows collected for QA).
    rosters, roster_qa = build_rosters(
        games=games,
        player_box=player_box,
        as_of_date=as_of_date,
        season=season,
        offline=offline,
        refresh=refresh_cache,
    )

    # Stage C — synthetic-row features + team_lastgame.
    syn_player_box, syn_team_box, syn_player_games, syn_team_games = _build_synthetic_target(
        rosters, player_box, team_box, games, as_of_date=as_of_date, season=season
    )
    player_features = _build_player_features(
        rosters, games, player_box, team_box, syn_player_box, syn_player_games
    )
    team_features = _build_team_features(
        games, team_box, syn_team_box, syn_team_games
    )
    team_lastgame = _build_team_lastgame(games, as_of_date=as_of_date)

    n_teams = int(rosters["team_id"].n_unique()) if not rosters.is_empty() else 0
    n_players = int(rosters["player_id"].n_unique()) if not rosters.is_empty() else 0
    provenance = _build_provenance(
        as_of_date=as_of_date,
        interim_latest_game_date=interim_latest,
        n_teams=n_teams,
        n_players=n_players,
        offline=offline,
    )

    # Stage D — atomic write. Live refreshes also emit qa_report.json with the
    # rejected roster rows / failed team fetches; offline has none, so it keeps
    # writing exactly the four parquets + as_of.json (Phase 6/7 behavior).
    qa_report: dict[str, Any] | None = None
    if not offline:
        qa_report = {
            "as_of_date": as_of_date.isoformat(),
            "generated_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_malformed": len(roster_qa),
            "malformed": roster_qa,
        }
    _write_snapshot_atomic(
        {
            ROSTERS_FILENAME: rosters,
            PLAYER_FEATURES_FILENAME: player_features,
            TEAM_FEATURES_FILENAME: team_features,
            TEAM_LASTGAME_FILENAME: team_lastgame,
        },
        provenance,
        dest=sroot,
        qa_report=qa_report,
    )
    logger.info(
        "refresh: wrote snapshot to %s (%d teams, %d players, as_of %s)",
        sroot, n_teams, n_players, as_of_date,
    )
    return sroot
