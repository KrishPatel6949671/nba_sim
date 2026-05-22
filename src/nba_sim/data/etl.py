"""Raw -> interim -> processed transforms.

All functions here are pure transforms with deterministic output paths:

* ``raw_to_interim(season)`` reads ``data/raw/`` (via the typed ``fetch.*``
  helpers — see the module docstring of :mod:`nba_sim.data.fetch`) and
  writes per-season typed parquet under ``data/interim/<season>/``.
* ``interim_to_processed(splits)`` joins interim tables with rolling
  features (see :mod:`nba_sim.features`) and writes
  ``data/processed/{train,val,test}.parquet``.
* ``build_feature_tables(season)`` precomputes rolling aggregates shared
  by multiple features.

All ETL is **idempotent** (tmp-file-then-rename) and **incremental**
(skip seasons whose outputs are newer than their inputs).
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import ValidationError

from nba_sim.data.fetch import (
    cache_dir,
    fetch_games_for_season,
    fetch_player_box,
    fetch_team_advanced_box,
    fetch_team_box,
)
from nba_sim.data.schema import (
    Game,
    PlayerBoxLine,
    RawGame,
    RawPlayerBoxLine,
    RawTeamBoxLine,
    Roster,
    TeamBoxLine,
)
from nba_sim.features.context import add_context_features
from nba_sim.features.matchup import (
    add_matchup_features,
    opponent_defrtg_by_position,
)
from nba_sim.features.rolling import (
    player_rolling,
    season_to_date,
    team_rolling,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitSpec:
    """Which seasons go to which split. Loaded from ``configs/data.yaml``."""

    train: list[int]
    val: list[int]
    test: list[int]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Mirrors fetch.cache_dir(): tests redirect via env var so they never touch
# the user's real interim directory.
def interim_dir() -> Path:
    p = os.environ.get("NBA_SIM_INTERIM_DIR", "data/interim")
    return Path(p).expanduser().resolve()


def processed_dir() -> Path:
    """Same pattern as interim_dir — tests redirect via env var."""
    p = os.environ.get("NBA_SIM_PROCESSED_DIR", "data/processed")
    return Path(p).expanduser().resolve()


def _season_dir(season: int) -> Path:
    return interim_dir() / str(season)


# Filenames are constants so tests can introspect them and we have one place
# to rename if the layout changes.
GAMES_FILENAME = "games.parquet"
PLAYER_BOX_FILENAME = "player_box.parquet"
TEAM_BOX_FILENAME = "team_box.parquet"
ROSTERS_FILENAME = "rosters.parquet"
QA_REPORT_FILENAME = "qa_report.json"
# Feature-table outputs (produced by build_feature_tables).
PLAYER_FEATURES_FILENAME = "player_features.parquet"
TEAM_FEATURES_FILENAME = "team_features.parquet"
SEASON_TO_DATE_FILENAME = "season_to_date.parquet"
# Processed-layer outputs (produced by interim_to_processed).
PROCESSED_TRAIN_FILENAME = "train.parquet"
PROCESSED_VAL_FILENAME = "val.parquet"
PROCESSED_TEST_FILENAME = "test.parquet"

# Per PLAN.md §2.5: 5 players × 48 minutes = 240 per team in regulation,
# +25 per OT period (5 players × 5 min). Anything outside this set means
# the game was called early or had a data anomaly — drop it from training.
_REGULATION_TEAM_MINUTES = 240.0
_OT_PERIOD_MINUTES = 25.0
_MAX_OT_PERIODS = 6  # NBA history record is 6 OTs; very tolerant upper bound.
_TEAM_MIN_TOLERANCE = 0.5  # mm:ss rounding artifacts vs. authoritative seconds.


def _valid_team_minute_totals() -> set[float]:
    return {
        _REGULATION_TEAM_MINUTES + i * _OT_PERIOD_MINUTES
        for i in range(_MAX_OT_PERIODS + 1)
    }


def _team_minutes_ok(total: float) -> bool:
    return any(
        abs(total - allowed) <= _TEAM_MIN_TOLERANCE
        for allowed in _valid_team_minute_totals()
    )


# ---------------------------------------------------------------------------
# Atomic writes (same pattern as fetch._write_atomic).
# ---------------------------------------------------------------------------

def _write_parquet_atomic(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def _models_to_df(models: list[Any]) -> pl.DataFrame:
    """Pydantic models -> polars DataFrame, preserving field order."""
    if not models:
        # An empty DataFrame still needs a schema for downstream readers.
        # Returning an empty no-column frame is fine for v1; the consumers
        # will know how to handle "season had 0 valid X".
        return pl.DataFrame()
    return pl.DataFrame([m.model_dump() for m in models])


# ---------------------------------------------------------------------------
# Incremental-skip check.
# ---------------------------------------------------------------------------

def _newest_mtime(paths: list[Path]) -> float:
    """Latest mtime across the given paths (recursive for directories).
    Returns 0.0 if nothing exists — i.e. "no inputs yet, never skip"."""
    latest = 0.0
    for p in paths:
        if not p.exists():
            continue
        if p.is_file():
            latest = max(latest, p.stat().st_mtime)
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    latest = max(latest, f.stat().st_mtime)
    return latest


def _outputs_for_season(season: int) -> list[Path]:
    d = _season_dir(season)
    return [
        d / GAMES_FILENAME,
        d / PLAYER_BOX_FILENAME,
        d / TEAM_BOX_FILENAME,
        d / ROSTERS_FILENAME,
        d / QA_REPORT_FILENAME,
    ]


def _is_up_to_date(season: int) -> bool:
    outputs = _outputs_for_season(season)
    if not all(p.exists() for p in outputs):
        return False
    oldest_output = min(p.stat().st_mtime for p in outputs)
    # Coarse but cheap: compare against the upstream cache directories.
    # Re-fetching any game's box score (rare; cache is immutable for
    # historical data) invalidates the whole season's interim. Fine.
    cache_inputs = [
        cache_dir() / "leaguegamefinder",
        cache_dir() / "boxscoretraditionalv3",
        cache_dir() / "boxscoreadvancedv3",
    ]
    newest_input = _newest_mtime(cache_inputs)
    return oldest_output >= newest_input


# ---------------------------------------------------------------------------
# Pure helpers (no I/O) — easy to unit-test.
# ---------------------------------------------------------------------------

def _group_raw_games(games_df: pl.DataFrame) -> dict[str, list[RawGame]]:
    """Each game appears twice in ``LeagueGameFinder`` (once per team).
    Group by ``GAME_ID``. Skips rows that fail RawGame validation."""
    groups: dict[str, list[RawGame]] = {}
    for row in games_df.iter_rows(named=True):
        try:
            raw = RawGame.model_validate(row)
        except ValidationError as e:
            logger.warning("RawGame validation failed: %s | row=%s", e, row)
            continue
        groups.setdefault(raw.GAME_ID, []).append(raw)
    return groups


def _build_games(
    grouped: dict[str, list[RawGame]],
) -> tuple[list[Game], list[dict[str, str]]]:
    """Build :class:`Game` records from grouped raw rows.

    Returns ``(games, dropped)``. ``dropped`` is a list of
    ``{"game_id": ..., "reason": ...}`` for games we couldn't form.
    """
    games: list[Game] = []
    dropped: list[dict[str, str]] = []
    for game_id, rows in grouped.items():
        try:
            games.append(Game.from_raw_pair(rows))
        except (ValueError, ValidationError) as e:
            dropped.append({"game_id": game_id, "reason": str(e)})
    return games, dropped


# ---------------------------------------------------------------------------
# Per-game box-score build.
# ---------------------------------------------------------------------------

# Advanced V3 column names — best-guess; the schema docstrings on the
# nba_api package list these. If they're wrong on real data, the extractor
# below returns None for that field and we log the missing column in QA.
_ADVANCED_PACE = "pace"
_ADVANCED_OFF_RTG = "offensiveRating"
_ADVANCED_DEF_RTG = "defensiveRating"


def _extract_advanced(
    advanced_df: pl.DataFrame, team_id: int
) -> tuple[float | None, float | None, float | None, list[str]]:
    """Return ``(pace, off_rtg, def_rtg, missing_cols)`` for a team.

    Tolerant of column-name drift: any expected column not present yields
    ``None`` and is reported back so QA can flag it once per season.
    """
    missing: list[str] = []
    if advanced_df.is_empty():
        return None, None, None, [_ADVANCED_PACE, _ADVANCED_OFF_RTG, _ADVANCED_DEF_RTG]

    row = advanced_df.filter(pl.col("teamId") == team_id)
    if row.is_empty():
        return None, None, None, []

    def _get(col: str) -> float | None:
        if col not in row.columns:
            missing.append(col)
            return None
        v = row[col][0]
        return float(v) if v is not None else None

    return _get(_ADVANCED_PACE), _get(_ADVANCED_OFF_RTG), _get(_ADVANCED_DEF_RTG), missing


def _build_player_lines(
    player_box_df: pl.DataFrame,
) -> tuple[list[PlayerBoxLine], list[dict[str, str]]]:
    """Validate + transform per-player rows. Row-level errors are dropped
    and reported — one bad row shouldn't tank the whole game."""
    lines: list[PlayerBoxLine] = []
    dropped: list[dict[str, str]] = []
    for row in player_box_df.iter_rows(named=True):
        try:
            raw = RawPlayerBoxLine.model_validate(row)
            lines.append(PlayerBoxLine.from_raw(raw))
        except (ValueError, ValidationError) as e:
            dropped.append(
                {
                    "game_id": str(row.get("gameId", "?")),
                    "person_id": str(row.get("personId", "?")),
                    "reason": str(e),
                }
            )
    return lines, dropped


def _build_team_lines(
    team_box_df: pl.DataFrame,
    advanced_df: pl.DataFrame,
    *,
    home_team_id: int,
) -> tuple[list[TeamBoxLine], list[dict[str, str]], list[str]]:
    """Build :class:`TeamBoxLine` rows and merge in advanced metrics.

    Returns ``(lines, dropped, missing_advanced_cols)``.
    """
    lines: list[TeamBoxLine] = []
    dropped: list[dict[str, str]] = []
    missing_cols_seen: set[str] = set()
    for row in team_box_df.iter_rows(named=True):
        try:
            raw = RawTeamBoxLine.model_validate(row)
            is_home = raw.teamId == home_team_id
            line = TeamBoxLine.from_raw(raw, is_home=is_home)
            pace, off_rtg, def_rtg, missing = _extract_advanced(advanced_df, raw.teamId)
            missing_cols_seen.update(missing)
            # Re-construct with the merged advanced fields. We can't mutate
            # a pydantic model in place if there are validators we'd re-trigger;
            # model_copy(update=...) is the documented path and re-runs
            # validators, which is what we want.
            line = line.model_copy(
                update={"pace": pace, "off_rtg": off_rtg, "def_rtg": def_rtg}
            )
            lines.append(line)
        except (ValueError, ValidationError) as e:
            dropped.append(
                {
                    "game_id": str(row.get("gameId", "?")),
                    "team_id": str(row.get("teamId", "?")),
                    "reason": str(e),
                }
            )
    return lines, dropped, sorted(missing_cols_seen)


def _derive_rosters(player_lines: list[PlayerBoxLine], season: int) -> list[Roster]:
    """Roster := union of every player_id that appeared in a game for that team.

    Cheaper than calling ``commonteamroster`` 30 times per season and
    sufficient for v1 — features in §3 don't read rosters directly.
    """
    by_team: dict[int, set[int]] = {}
    for line in player_lines:
        by_team.setdefault(line.team_id, set()).add(line.player_id)
    return [
        Roster(team_id=tid, season=season, player_ids=sorted(pids))
        for tid, pids in sorted(by_team.items())
    ]


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def raw_to_interim(season: int, *, refresh: bool = False) -> Path:
    """Transform raw cached payloads for one season into typed parquet.

    Writes:
        data/interim/<season>/games.parquet
        data/interim/<season>/player_box.parquet
        data/interim/<season>/team_box.parquet
        data/interim/<season>/rosters.parquet
        data/interim/<season>/qa_report.json

    Returns the season directory.

    Args:
        season: start year of the season (e.g. ``2023`` for 2023-24).
        refresh: if False (default), skip when outputs are newer than the
            raw cache mtimes. If True, always re-run.
    """
    out_dir = _season_dir(season)

    if not refresh and _is_up_to_date(season):
        logger.info("interim/%d up-to-date — skipping (pass refresh=True to override)", season)
        return out_dir

    # 1. Build the per-game header table.
    games_df = fetch_games_for_season(season)
    grouped = _group_raw_games(games_df)
    games, dropped_games = _build_games(grouped)
    games_by_id = {g.game_id: g for g in games}
    logger.info(
        "season %d: %d games from LeagueGameFinder, %d formed, %d dropped at header-build",
        season, len(grouped), len(games), len(dropped_games),
    )

    # 2. Fan out to box scores. One sequential pass through games — the
    #    fetch helpers are rate-limited internally, so threading wouldn't
    #    help (and would risk hitting NBA's per-IP throttle).
    all_player_lines: list[PlayerBoxLine] = []
    all_team_lines: list[TeamBoxLine] = []
    dropped_player_lines: list[dict[str, str]] = []
    dropped_team_lines: list[dict[str, str]] = []
    missing_advanced_cols: set[str] = set()

    for game in games:
        try:
            player_box = fetch_player_box(game.game_id)
            team_box = fetch_team_box(game.game_id)
            advanced = fetch_team_advanced_box(game.game_id)
        except Exception as e:
            # Network or 404 — flag the game and continue. We don't retry
            # at this layer; fetch.py already retries with backoff.
            game.dropped = True
            game.dropped_reason = f"fetch_failed: {e}"
            dropped_games.append({"game_id": game.game_id, "reason": str(e)})
            continue

        p_lines, p_dropped = _build_player_lines(player_box)
        t_lines, t_dropped, missing = _build_team_lines(
            team_box, advanced, home_team_id=game.home_team_id
        )

        # Per PLAN.md §2.5: drop games whose team-minutes total is anomalous.
        # Use the team-box totals (authoritative; LeagueGameFinder.MIN matches
        # but we already have the team-box parsed here).
        if t_lines and not all(_team_minutes_ok(t.minutes) for t in t_lines):
            mins = {t.team_abbr: t.minutes for t in t_lines}
            game.dropped = True
            game.dropped_reason = f"team_minutes_anomaly: {mins}"
            dropped_games.append({"game_id": game.game_id, "reason": game.dropped_reason})
            # Don't add this game's rows to the training tables.
            continue

        all_player_lines.extend(p_lines)
        all_team_lines.extend(t_lines)
        dropped_player_lines.extend(p_dropped)
        dropped_team_lines.extend(t_dropped)
        missing_advanced_cols.update(missing)

    # 3. Derive rosters from player appearances.
    rosters = _derive_rosters(all_player_lines, season)

    # 4. QA report.
    qa = {
        "season": season,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "n_games_input": len(grouped),
        "n_games_kept": sum(1 for g in games if not g.dropped),
        "n_games_dropped": sum(1 for g in games if g.dropped) + len(
            [d for d in dropped_games if d["game_id"] not in games_by_id]
        ),
        "n_player_lines": len(all_player_lines),
        "n_player_lines_dropped": len(dropped_player_lines),
        "n_team_lines": len(all_team_lines),
        "n_team_lines_dropped": len(dropped_team_lines),
        "n_rosters": len(rosters),
        "missing_advanced_cols": sorted(missing_advanced_cols),
        "dropped_games": dropped_games,
        # Cap row-level lists to avoid runaway report sizes; QA-by-eyeball
        # only needs the head + the counts.
        "dropped_player_lines": dropped_player_lines[:200],
        "dropped_team_lines": dropped_team_lines[:200],
    }

    # 5. Write everything atomically.
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(_models_to_df(games), out_dir / GAMES_FILENAME)
    _write_parquet_atomic(_models_to_df(all_player_lines), out_dir / PLAYER_BOX_FILENAME)
    _write_parquet_atomic(_models_to_df(all_team_lines), out_dir / TEAM_BOX_FILENAME)
    _write_parquet_atomic(_models_to_df(rosters), out_dir / ROSTERS_FILENAME)
    _write_json_atomic(qa, out_dir / QA_REPORT_FILENAME)

    logger.info(
        "season %d: wrote %d games, %d player lines, %d team lines, %d rosters",
        season, qa["n_games_kept"], qa["n_player_lines"], qa["n_team_lines"], qa["n_rosters"],
    )
    return out_dir


def _feature_outputs_for_season(season: int) -> list[Path]:
    d = _season_dir(season)
    return [
        d / PLAYER_FEATURES_FILENAME,
        d / TEAM_FEATURES_FILENAME,
        d / SEASON_TO_DATE_FILENAME,
    ]


def _features_up_to_date(season: int) -> bool:
    """True iff feature outputs exist and are newer than the interim inputs.

    Inputs are the per-season ``games.parquet``, ``player_box.parquet``,
    ``team_box.parquet`` — re-running raw_to_interim invalidates the
    feature tables for that season.
    """
    outputs = _feature_outputs_for_season(season)
    if not all(p.exists() for p in outputs):
        return False
    oldest_output = min(p.stat().st_mtime for p in outputs)
    d = _season_dir(season)
    inputs = [d / GAMES_FILENAME, d / PLAYER_BOX_FILENAME, d / TEAM_BOX_FILENAME]
    newest_input = _newest_mtime(inputs)
    return oldest_output >= newest_input


def build_feature_tables(season: int, *, refresh: bool = False) -> Path:
    """Precompute rolling aggregates that multiple feature groups share.

    Reads the per-season interim parquets and writes three feature tables
    into the same season directory:

        data/interim/<season>/player_features.parquet
        data/interim/<season>/team_features.parquet
        data/interim/<season>/season_to_date.parquet

    The split layer (``interim_to_processed``) reads these and joins
    matchup features (PLAN.md §3.1) on top. Materializing here means a
    train/val/test re-split doesn't redo the per-season rolling work.

    Idempotent: skips when outputs are newer than the interim inputs.
    """
    out_dir = _season_dir(season)
    if not refresh and _features_up_to_date(season):
        logger.info("features/%d up-to-date — skipping", season)
        return out_dir

    games_path = out_dir / GAMES_FILENAME
    player_path = out_dir / PLAYER_BOX_FILENAME
    team_path = out_dir / TEAM_BOX_FILENAME
    # raw_to_interim must have produced these. If they're missing the user
    # has the steps out of order; a clear FileNotFoundError beats a polars
    # SchemaError later.
    for p in (games_path, player_path, team_path):
        if not p.exists():
            raise FileNotFoundError(
                f"build_feature_tables({season}): {p} not found — run raw_to_interim first"
            )

    games = pl.read_parquet(games_path)
    player_box = pl.read_parquet(player_path)
    team_box = pl.read_parquet(team_path)

    # Order matters: matchup features (opp rolling) read from team_features,
    # so team_rolling must run first. player_rolling and team_rolling are
    # independent; their order is just convention.
    team_features = team_rolling(team_box, games)
    player_rolling_out = player_rolling(player_box, games, team_box=team_box)

    # DefRtg-vs-position is a per-(opp_team, game, position) lookup table.
    # We compute it once per season and pass it into add_matchup_features
    # for the join — see opponent_defrtg_by_position docstring for why it's
    # separate from add_matchup_features (different input dependencies).
    defrtg_vs_pos = opponent_defrtg_by_position(player_box, games, team_box)

    # Stitch matchup features onto the player rolling frame. The combined
    # frame is what we persist as player_features.parquet — same grain
    # (one row per player-game), now with opp_*/h2h_/opp_def_rtg_vs_pos
    # columns added in.
    player_features = add_matchup_features(
        player_rolling_out, team_features, games, defrtg_vs_pos=defrtg_vs_pos
    )

    # Context features (rest, b2b, density, altitude, travel, calendar)
    # are per-(team, game). Join onto player_features at (game_id, team_id)
    # — every player on the team in that game shares the same context.
    # We `.select` down to just the context-specific columns to avoid
    # re-introducing identity columns (date, season, is_playoffs,
    # team_abbr) that the player frame already carries.
    context_features = add_context_features(games).select([
        "game_id", "team_id",
        "is_home", "rest_days", "b2b", "is_3in4", "is_4in6",
        "season_phase", "day_of_week", "month",
        "altitude_ft", "travel_miles_prev",
    ])
    player_features = player_features.join(
        context_features, on=["game_id", "team_id"], how="left"
    )

    # Season-to-date averages are the strictly-prior cumulative-mean
    # features that the season-average baseline depends on (PLAN §6.3).
    # Without joining these onto player_features, any model trained on the
    # processed parquet is competing with SAB on lossier signals (rolling
    # 5/10/20 windows rather than the full season-to-date mean), which
    # makes the Phase 1 ship gate effectively unwinnable. Keep std_games
    # plus every std_{stat}_avg column; drop the per-row identity cols
    # already on player_features (date, season).
    std = season_to_date(player_box, games)
    std_join = std.select(
        ["game_id", "player_id"]
        + [c for c in std.columns if c == "std_games" or (c.startswith("std_") and c.endswith("_avg"))]
    )
    player_features = player_features.join(std_join, on=["game_id", "player_id"], how="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(player_features, out_dir / PLAYER_FEATURES_FILENAME)
    _write_parquet_atomic(team_features, out_dir / TEAM_FEATURES_FILENAME)
    _write_parquet_atomic(std, out_dir / SEASON_TO_DATE_FILENAME)

    logger.info(
        "features %d: wrote %d player rows, %d team rows, %d std rows",
        season, player_features.height, team_features.height, std.height,
    )
    return out_dir


# Columns we *don't* want to carry over from team_features into the joined
# player frame — they either duplicate columns already on player_features
# (date, season) or aren't useful at the player grain (team totals).
_TEAM_FEATURE_KEEP_PREFIXES = ("t_",)
_TEAM_FEATURE_KEEP_KEYS = ("game_id", "team_id")


def _team_features_for_join(team_features: pl.DataFrame) -> pl.DataFrame:
    """Subset team_features to (game_id, team_id, t_*) so the join doesn't
    pull duplicate date/season/totals columns into the player frame."""
    keep = [
        c for c in team_features.columns
        if c in _TEAM_FEATURE_KEEP_KEYS or c.startswith(_TEAM_FEATURE_KEEP_PREFIXES)
    ]
    return team_features.select(keep)


def _build_split_frame(seasons: list[int]) -> pl.DataFrame:
    """Read each season's feature tables, join team→player, concat.

    Returns an empty DataFrame if ``seasons`` is empty (legal — the
    config layer allows empty val/test splits)."""
    if not seasons:
        return pl.DataFrame()

    parts: list[pl.DataFrame] = []
    for season in sorted(seasons):
        d = _season_dir(season)
        pf_path = d / PLAYER_FEATURES_FILENAME
        tf_path = d / TEAM_FEATURES_FILENAME
        if not pf_path.exists() or not tf_path.exists():
            raise FileNotFoundError(
                f"season {season}: feature tables missing — run build_feature_tables({season}) first"
            )
        pf = pl.read_parquet(pf_path)
        tf = pl.read_parquet(tf_path)
        # Skip seasons that produced no rows (would happen only if every
        # game in the season was dropped — pathological but defensible).
        if pf.is_empty():
            continue
        joined = pf.join(
            _team_features_for_join(tf),
            on=["game_id", "team_id"],
            how="left",
        )
        parts.append(joined)

    if not parts:
        return pl.DataFrame()
    # vertical_relaxed accepts mild schema differences across seasons (e.g.
    # an older season missing an advanced metric); strict vertical would
    # blow up on a single null-dtype mismatch.
    return pl.concat(parts, how="vertical_relaxed")


def interim_to_processed(
    splits: SplitSpec, *, refresh: bool = False
) -> dict[str, Path]:
    """Join interim + features per split and emit modeling-ready parquets.

    For each split:
        - Ensure ``build_feature_tables`` has run for every season in it.
        - Read player_features and team_features, join team-rolling onto
          each player row by (game_id, team_id).
        - Concat across seasons, write ``data/processed/<split>.parquet``.

    Returns a dict ``{"train": Path, "val": Path, "test": Path}`` (always
    all three keys, even if a split is empty — the file is still written
    so downstream code can rely on the path existing).

    Note: matchup and context features (PLAN.md §3.1, ``features/matchup.py``
    and ``features/context.py``) are not joined yet — they'll be wired in
    once those modules are implemented. The processed parquet shape may
    grow but never shrink as features land.
    """
    out_dir = processed_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure every season's feature tables are present (idempotent — the
    # per-season function skips when up-to-date).
    all_seasons = sorted(set(splits.train) | set(splits.val) | set(splits.test))
    for season in all_seasons:
        build_feature_tables(season, refresh=refresh)

    paths = {
        "train": out_dir / PROCESSED_TRAIN_FILENAME,
        "val": out_dir / PROCESSED_VAL_FILENAME,
        "test": out_dir / PROCESSED_TEST_FILENAME,
    }
    split_seasons = {"train": splits.train, "val": splits.val, "test": splits.test}

    for split_name, season_list in split_seasons.items():
        frame = _build_split_frame(season_list)
        _write_parquet_atomic(frame, paths[split_name])
        logger.info(
            "processed/%s: %d rows from seasons %s",
            split_name, frame.height, season_list,
        )

    return paths
