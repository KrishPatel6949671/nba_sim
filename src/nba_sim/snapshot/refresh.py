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

Synthetic-target-row strategy (Appendix B)
------------------------------------------
For each rostered ``(team, player)`` we append a **zero-filled** synthetic
player-box row with ``game_id = SNAPSHOT_<abbr>_<as_of>`` and one synthetic
per-team team-box row (``game_id = SNAPSHOT_TEAM_<abbr>_<as_of>``), plus the
matching synthetic ``games`` rows (``dropped=False``, ``season`` = the season
as_of falls in). Two sentinel games per matchup (not one) so each team owns
exactly one row per rolling partition. Because every aggregate is
``shift(1)`` / ``cum_sum - current``, the zero-filled targets never enter
their own windows — the synthetic row's features are precisely what the model
would have seen had the game been played on as_of.

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
import logging
from pathlib import Path
from typing import Any

import polars as pl

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
    raise NotImplementedError("TODO(task 2): map date → season start-year")


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
    raise NotImplementedError("TODO(task 2): resolve as_of + latest interim date")


def _seasons_up_to(as_of_date: _dt.date, *, interim_root: Path) -> list[int]:
    """Interim season start-years with data on/before ``as_of_date``."""
    raise NotImplementedError("TODO(task 2)")


def _load_interim(
    seasons: list[int],
    *,
    as_of_date: _dt.date,
    interim_root: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Concat ``games`` / ``player_box`` / ``team_box`` across ``seasons`` and
    filter to interim games with ``date < as_of_date`` (the leakage rule).

    Returns ``(games, player_box, team_box)``.
    """
    raise NotImplementedError("TODO(task 4): load + date-filter interim frames")


# ---------------------------------------------------------------------------
# Stage C — synthetic target rows + features
# ---------------------------------------------------------------------------

def _build_synthetic_target(
    rosters: pl.DataFrame,
    *,
    as_of_date: _dt.date,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Construct the zero-filled synthetic rows (Appendix B).

    Returns ``(syn_player_box, syn_team_box, syn_player_games,
    syn_team_games)``:

    - ``syn_player_box``  — one zero-filled PlayerBoxLine-shaped row per
      rostered (team, player), ``game_id = SNAPSHOT_<abbr>_<as_of>``.
    - ``syn_team_box``    — one zero-filled TeamBoxLine-shaped row per team,
      ``game_id = SNAPSHOT_TEAM_<abbr>_<as_of>``.
    - ``syn_player_games`` / ``syn_team_games`` — matching Game-shaped rows
      (``date = as_of_date``, ``season``, ``dropped=False``) for each sentinel
      game so ``_attach_dates`` can join date/season on.

    Schemas match the interim frames column-for-column so they concat cleanly.
    """
    raise NotImplementedError("TODO(task 4): build synthetic target rows")


def _build_player_features(
    rosters: pl.DataFrame,
    games: pl.DataFrame,
    player_box: pl.DataFrame,
    team_box: pl.DataFrame,
    *,
    as_of_date: _dt.date,
    season: int,
) -> pl.DataFrame:
    """Build ``player_features.parquet`` (§13.4) via the synthetic-row trick.

    Concats the synthetic player rows onto ``player_box`` / ``games``, runs
    the existing :func:`nba_sim.features.rolling.player_rolling` (with
    ``team_box`` for USG%) and :func:`~nba_sim.features.rolling.season_to_date`,
    then filters back to the ``SNAPSHOT_`` rows. Selects the full
    ``_PLAYER_NUMERIC_COLS`` block (with :data:`NULL_AT_REFRESH_NUMERIC`
    written null), plus identity cols (team_id, team_abbr, player_id,
    player_name, position). See the module docstring for the column contract.
    """
    raise NotImplementedError("TODO(task 4): build player_features")


def _build_team_features(
    games: pl.DataFrame,
    team_box: pl.DataFrame,
    *,
    as_of_date: _dt.date,
    season: int,
) -> pl.DataFrame:
    """Build ``team_features.parquet`` (§13.5): one row per team with the 9
    standardized team-rolling cols (``t_pace_5/10``, ``t_off_rtg_5/10``,
    ``t_def_rtg_5/10``, ``t_win_pct_10``, ``t_pts_avg_10``,
    ``t_pts_allowed_10``).

    Same synthetic-row trick on the team side: append ``SNAPSHOT_TEAM_`` rows,
    run :func:`nba_sim.features.rolling.team_rolling`, filter back.
    """
    raise NotImplementedError("TODO(task 4): build team_features")


# ---------------------------------------------------------------------------
# Stage C (cont.) — team_lastgame
# ---------------------------------------------------------------------------

def _build_team_lastgame(
    games: pl.DataFrame,
    *,
    as_of_date: _dt.date,
) -> pl.DataFrame:
    """Build ``team_lastgame.parquet`` (§13.6) inline from interim games.

    Per team_id: the row with ``date < as_of_date`` and max ``date`` →
    ``last_game_date``, ``last_arena_team_id`` (the home team of that game),
    ``last_was_home`` (team_id == that game's home_team_id). Drives rest_days
    / b2b / travel at simulate time.
    """
    raise NotImplementedError("TODO(task 5): build team_lastgame")


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
    raise NotImplementedError("TODO(task 6): build provenance dict")


def _write_snapshot_atomic(
    frames: dict[str, pl.DataFrame],
    provenance: dict[str, Any],
    *,
    dest: Path,
) -> None:
    """Stage all parquets under ``dest/.tmp/`` then swap into place (§14.4).

    Parquets first, then ``as_of.json`` **last** so an interrupted refresh
    never leaves a complete-looking snapshot. The dir swap removes any
    existing snapshot only after the new ``.tmp`` is fully written.
    """
    raise NotImplementedError("TODO(task 6): atomic dir swap + provenance last")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def refresh(
    *,
    as_of: str | _dt.date | None = None,
    force: bool = False,
    offline: bool = True,
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
        Phase 6 default ``True`` — derive rosters from interim, no nba_api.
        The live fetch path lands in Phase 8.
    snapshot_root, interim_root
        Override roots (tests redirect via ``NBA_SIM_SNAPSHOT_DIR`` /
        ``NBA_SIM_INTERIM_DIR``); default to the env-aware helpers.
    """
    raise NotImplementedError("TODO(task 6): orchestrate Stages A-D")
