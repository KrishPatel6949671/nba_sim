"""Snapshot-layer tests (v2PLAN.md §18.1).

Phase 6 covers the **non-network** subset: as_of resolution, the offline
refresh writing all four parquets + ``as_of.json`` against a tiny fixture
interim directory, schema validity, the synthetic-row sentinel ids, the
``p_min_avg_10`` hand-check / known-date equivalence, provenance round-trip,
and the ``snapshot-status`` block format.

All tests redirect the data roots via ``monkeypatch.setenv`` (same discipline
as ``tests/test_data_etl.py``) so they never touch the user's real
``data/interim`` or ``data/snapshot``. Network / slow tests (full 30-team
refresh) are Phase 8 and marked accordingly.

Phase-7 tests (context vector, build_synthetic_game_batch, simulate v2) live
with the simulate API and are intentionally not here yet.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nba_sim.data.etl import (
    GAMES_FILENAME,
    PLAYER_BOX_FILENAME,
    TEAM_BOX_FILENAME,
    _models_to_df,
)
from nba_sim.data.schema import Game, PlayerBoxLine, TeamBoxLine
from nba_sim.features.rolling import player_rolling, season_to_date
from nba_sim.snapshot import (
    AS_OF_FILENAME,
    CODE_VERSION,
    PLAYER_FEATURES_FILENAME,
    ROSTERS_FILENAME,
    TEAM_FEATURES_FILENAME,
    TEAM_LASTGAME_FILENAME,
)
from nba_sim.snapshot.refresh import (
    NULL_AT_REFRESH_NUMERIC,
    SNAPSHOT_GAME_PREFIX,
    SNAPSHOT_TEAM_GAME_PREFIX,
    _build_synthetic_target,
    _load_interim,
    refresh,
    resolve_as_of_date,
    season_for_date,
)
from nba_sim.snapshot.rosters import NULLABLE_BIO_COLUMNS, ROSTER_COLUMNS, build_rosters
from nba_sim.snapshot.status import format_status, read_provenance
from nba_sim.training.dataset import _MATCHUP_TEAM_DIFF_BASES, _PLAYER_NUMERIC_COLS

# Season used by the fixture interim builder (start-year).
_FIXTURE_SEASON = 2023

# The four parquets refresh must always write.
_SNAPSHOT_PARQUETS = (
    ROSTERS_FILENAME,
    PLAYER_FEATURES_FILENAME,
    TEAM_FEATURES_FILENAME,
    TEAM_LASTGAME_FILENAME,
)
# Everything a complete snapshot directory contains (provenance written last).
_SNAPSHOT_FILES = (*_SNAPSHOT_PARQUETS, AS_OF_FILENAME)


# ---------------------------------------------------------------------------
# Fixture interim season
# ---------------------------------------------------------------------------
#
# A deterministic 12-game BOS-vs-LAL season (2023). Per-game minutes increase
# monotonically so rolling windows are hand-computable; player 20 is traded
# BOS -> LAL after the 6th game to exercise the dedup-by-player path. Built via
# the same pydantic -> _models_to_df route the real ETL uses, so the parquet
# schemas are byte-faithful to interim.

_BOS, _LAL = 1, 2
_N_GAMES = 12


@dataclasses.dataclass
class _Fixture:
    """Everything a snapshot test needs: redirected roots, the in-memory
    interim frames (for v1 cross-checks), and the game dates / ids."""

    interim_root: Path
    snapshot_root: Path
    season: int
    dates: list[_dt.date]
    gids: list[str]
    games: pl.DataFrame
    player_box: pl.DataFrame
    team_box: pl.DataFrame


def _player_line(
    gid: str, pid: int, name: str, tid: int, abbr: str, minutes: int, position: str
) -> PlayerBoxLine:
    m = float(minutes)
    return PlayerBoxLine(
        game_id=gid, player_id=pid, player_name=name, team_id=tid, team_abbr=abbr,
        position=position, minutes=m, pts=int(m), fgm=3, fga=6, tpm=1, tpa=2, ftm=2,
        fta=2, oreb=1, dreb=3, reb=4, ast=2, stl=1, blk=1, tov=1, pf=2, plus_minus=1.0,
        is_starter=bool(position), is_active=True, dnp=(m == 0.0),
    )


def _team_line(gid: str, tid: int, abbr: str, is_home: bool, pace: float) -> TeamBoxLine:
    return TeamBoxLine(
        game_id=gid, team_id=tid, team_abbr=abbr, is_home=is_home, minutes=240.0, pts=110,
        fgm=40, fga=88, tpm=10, tpa=30, ftm=15, fta=20, oreb=10, dreb=34, reb=44, ast=25,
        stl=7, blk=5, tov=12, pf=18, plus_minus=2.0 if is_home else -2.0, pace=pace,
        off_rtg=112.0, def_rtg=108.0,
    )


def _build_fixture_frames(
    season: int,
) -> tuple[list[_dt.date], list[str], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    dates = [_dt.date(season, 11, 1) + _dt.timedelta(days=2 * i) for i in range(_N_GAMES)]
    gids = [f"00{season}{i:04d}" for i in range(1, _N_GAMES + 1)]
    games: list[Game] = []
    players: list[PlayerBoxLine] = []
    teams: list[TeamBoxLine] = []
    for i, (gid, d) in enumerate(zip(gids, dates, strict=False)):
        games.append(Game(
            game_id=gid, season=season, date=d, home_team_id=_BOS, away_team_id=_LAL,
            home_team_abbr="BOS", away_team_abbr="LAL", home_pts=110, away_pts=108,
            dropped=False,
        ))
        players.append(_player_line(gid, 10, "Star A", _BOS, "BOS", 20 + i, "G"))
        players.append(_player_line(gid, 11, "Role B", _BOS, "BOS", 10 + i, "F"))
        # Player 20 is traded BOS -> LAL after the 6th game.
        traded_team, traded_abbr = (_BOS, "BOS") if i < 6 else (_LAL, "LAL")
        players.append(_player_line(gid, 20, "Traded C", traded_team, traded_abbr, 15 + i, "F"))
        players.append(_player_line(gid, 30, "Star D", _LAL, "LAL", 25 + i, "C"))
        teams.append(_team_line(gid, _BOS, "BOS", True, 100.0 + i))
        teams.append(_team_line(gid, _LAL, "LAL", False, 98.0 + i))
    return dates, gids, _models_to_df(games), _models_to_df(players), _models_to_df(teams)


def _write_fixture_interim(
    interim_root: Path, season: int = _FIXTURE_SEASON
) -> tuple[list[_dt.date], list[str], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Write a tiny but schema-faithful interim season under ``interim_root``
    (``<season>/{games,player_box,team_box}.parquet``) and return the in-memory
    frames + game dates / ids for cross-checks."""
    dates, gids, games, player_box, team_box = _build_fixture_frames(season)
    season_dir = interim_root / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    games.write_parquet(season_dir / GAMES_FILENAME)
    player_box.write_parquet(season_dir / PLAYER_BOX_FILENAME)
    team_box.write_parquet(season_dir / TEAM_BOX_FILENAME)
    return dates, gids, games, player_box, team_box


@pytest.fixture()
def snapshot_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Fixture:
    """Redirect interim + snapshot roots into ``tmp_path`` and seed the fixture
    interim season. Returns a :class:`_Fixture`."""
    interim_root = tmp_path / "interim"
    snapshot_root = tmp_path / "snapshot"
    monkeypatch.setenv("NBA_SIM_INTERIM_DIR", str(interim_root))
    monkeypatch.setenv("NBA_SIM_SNAPSHOT_DIR", str(snapshot_root))
    dates, gids, games, player_box, team_box = _write_fixture_interim(interim_root)
    return _Fixture(
        interim_root=interim_root, snapshot_root=snapshot_root, season=_FIXTURE_SEASON,
        dates=dates, gids=gids, games=games, player_box=player_box, team_box=team_box,
    )


# ---------------------------------------------------------------------------
# Stage A — as_of resolution
# ---------------------------------------------------------------------------

def test_resolve_as_of_date_default(snapshot_env: _Fixture) -> None:
    """No --as-of -> latest interim game date + 1 day."""
    as_of_date, interim_latest = resolve_as_of_date(
        None, interim_root=snapshot_env.interim_root
    )
    assert interim_latest == snapshot_env.dates[-1]
    assert as_of_date == snapshot_env.dates[-1] + _dt.timedelta(days=1)


def test_resolve_as_of_date_explicit(snapshot_env: _Fixture) -> None:
    """--as-of YYYY-MM-DD is respected verbatim."""
    as_of_date, interim_latest = resolve_as_of_date(
        "2024-01-15", interim_root=snapshot_env.interim_root
    )
    assert as_of_date == _dt.date(2024, 1, 15)
    assert interim_latest == snapshot_env.dates[-1]


def test_season_for_date() -> None:
    """Jul-Dec -> year N; Jan-Jun -> year N-1 (e.g. 2024-03-15 -> 2023)."""
    assert season_for_date(_dt.date(2023, 10, 1)) == 2023
    assert season_for_date(_dt.date(2023, 12, 31)) == 2023
    assert season_for_date(_dt.date(2024, 1, 1)) == 2023
    assert season_for_date(_dt.date(2024, 3, 15)) == 2023
    assert season_for_date(_dt.date(2024, 6, 30)) == 2023
    assert season_for_date(_dt.date(2024, 7, 1)) == 2024


# ---------------------------------------------------------------------------
# Stage B-D - offline refresh
# ---------------------------------------------------------------------------

def test_refresh_writes_all_four_parquets(snapshot_env: _Fixture) -> None:
    """refresh(offline=True) writes rosters/player_features/team_features/
    team_lastgame + as_of.json, and cleans up the staging dir."""
    dest = refresh(offline=True)
    for name in _SNAPSHOT_FILES:
        assert (dest / name).exists(), name
    assert not (dest / ".tmp").exists()


def test_refresh_provenance_json_complete(snapshot_env: _Fixture) -> None:
    """as_of.json has every documented key (§13.2) with the expected values."""
    dest = refresh(offline=True)
    prov = json.loads((dest / AS_OF_FILENAME).read_text())
    assert set(prov) == {
        "as_of_date", "refreshed_at", "source", "n_teams", "n_players",
        "interim_latest_game_date", "code_version",
        "model_checkpoint_used_for_validation",
    }
    assert prov["as_of_date"] == (snapshot_env.dates[-1] + _dt.timedelta(days=1)).isoformat()
    assert prov["interim_latest_game_date"] == snapshot_env.dates[-1].isoformat()
    assert prov["source"] == "interim-only"
    assert prov["n_teams"] == 2
    assert prov["n_players"] == 4
    assert prov["code_version"] == CODE_VERSION
    assert prov["model_checkpoint_used_for_validation"] is None
    assert prov["refreshed_at"].endswith("Z")


def test_refresh_atomic_partial_write_safety(
    snapshot_env: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write leaves the live snapshot untouched: parquets stage in
    ``.tmp/`` and as_of.json is promoted last, so a failed rebuild can't
    replace a good as_of.json."""
    dest = refresh(offline=True)
    good = (dest / AS_OF_FILENAME).read_text()

    calls = {"n": 0}
    original: Any = pl.DataFrame.write_parquet

    def _boom(self: pl.DataFrame, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated disk failure")
        original(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _boom)
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        refresh(offline=True, force=True)

    # The live provenance is byte-identical — the crash never reached promote.
    assert (dest / AS_OF_FILENAME).read_text() == good


def test_snapshot_provenance_round_trip(snapshot_env: _Fixture) -> None:
    """refresh writes provenance; status.read_provenance reads it back
    unchanged (success-criteria table, §12)."""
    dest = refresh(offline=True)
    prov = read_provenance(dest)
    assert prov == json.loads((dest / AS_OF_FILENAME).read_text())
    assert prov["as_of_date"] == (snapshot_env.dates[-1] + _dt.timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# Stage C — synthetic rows + schema validity
# ---------------------------------------------------------------------------

def test_synthetic_target_rows_use_sentinel_ids(snapshot_env: _Fixture) -> None:
    """Synthetic player game ids start with SNAPSHOT_; team ids with
    SNAPSHOT_TEAM_; one sentinel game per side; one player row per distinct id
    (the dedup-by-player invariant)."""
    as_of_date = snapshot_env.dates[-1] + _dt.timedelta(days=1)
    season = season_for_date(as_of_date)
    games, player_box, team_box = _load_interim(
        [snapshot_env.season], as_of_date=as_of_date, interim_root=snapshot_env.interim_root
    )
    rosters = build_rosters(
        games=games, player_box=player_box, as_of_date=as_of_date, season=season
    )
    syn_player_box, syn_team_box, syn_player_games, syn_team_games = _build_synthetic_target(
        rosters, player_box, team_box, games, as_of_date=as_of_date, season=season
    )
    player_gid = f"{SNAPSHOT_GAME_PREFIX}{as_of_date.isoformat()}"
    team_gid = f"{SNAPSHOT_TEAM_GAME_PREFIX}{as_of_date.isoformat()}"
    assert syn_player_box["game_id"].unique().to_list() == [player_gid]
    assert syn_team_box["game_id"].unique().to_list() == [team_gid]
    assert syn_player_games["game_id"].to_list() == [player_gid]
    assert syn_team_games["game_id"].to_list() == [team_gid]
    assert syn_player_box.height == rosters["player_id"].n_unique()


def test_rosters_schema_matches_contract(snapshot_env: _Fixture) -> None:
    """rosters.parquet has exactly ROSTER_COLUMNS (two_way dropped), position
    non-null, bio nullable, correct dtypes."""
    dest = refresh(offline=True)
    rosters = pl.read_parquet(dest / ROSTERS_FILENAME)
    assert rosters.columns == list(ROSTER_COLUMNS)
    assert "two_way" not in rosters.columns
    assert rosters["position"].null_count() == 0
    for col in NULLABLE_BIO_COLUMNS:
        assert rosters[col].null_count() == rosters.height
    assert rosters.schema["team_id"] == pl.Int64
    assert rosters.schema["birth_date"] == pl.Date
    # p10, p11, p20(BOS), p20(LAL), p30 — the traded player is on both teams.
    assert rosters.height == 5


def test_snapshot_schema_validity(snapshot_env: _Fixture) -> None:
    """All four parquets match the documented schemas (§13.7). player_features
    carries the full 47 _PLAYER_NUMERIC_COLS with the four
    NULL_AT_REFRESH_NUMERIC present-but-null (Float64)."""
    dest = refresh(offline=True)
    rosters = pl.read_parquet(dest / ROSTERS_FILENAME)
    player_features = pl.read_parquet(dest / PLAYER_FEATURES_FILENAME)
    team_features = pl.read_parquet(dest / TEAM_FEATURES_FILENAME)
    team_lastgame = pl.read_parquet(dest / TEAM_LASTGAME_FILENAME)

    assert rosters.columns == list(ROSTER_COLUMNS)
    assert player_features.columns == [
        "team_id", "team_abbr", "player_id", "player_name", "position",
        *_PLAYER_NUMERIC_COLS,
    ]
    for col in NULL_AT_REFRESH_NUMERIC:
        assert player_features.schema[col] == pl.Float64
        assert player_features[col].null_count() == player_features.height
    assert team_features.columns == ["team_id", "team_abbr", *_MATCHUP_TEAM_DIFF_BASES]
    assert team_lastgame.columns == [
        "team_id", "last_game_date", "last_arena_team_id", "last_was_home",
    ]
    assert team_lastgame.schema["last_game_date"] == pl.Date
    assert team_lastgame.schema["last_was_home"] == pl.Boolean
    assert player_features.height == 5
    assert team_features.height == 2
    assert team_lastgame.height == 2


def test_player_feature_matches_processed_at_known_date(snapshot_env: _Fixture) -> None:
    """At a known game date, the synthetic feature row equals what v1's rolling
    kernels compute for the real game on that date, within 1e-6 (§14.6) — the
    date<as_of leakage discipline holds. Covers the traded player (identical
    player-level form on both teams) and a hand-computed p_min_avg_10."""
    known = snapshot_env.dates[-1]
    last_gid = snapshot_env.gids[-1]
    v1 = player_rolling(
        snapshot_env.player_box, snapshot_env.games, team_box=snapshot_env.team_box
    )
    v1_std = season_to_date(snapshot_env.player_box, snapshot_env.games)

    def v1_val(pid: int, col: str) -> float:
        row = v1.filter((pl.col("player_id") == pid) & (pl.col("game_id") == last_gid))
        return float(row[col][0])

    def v1_std_val(pid: int, col: str) -> float:
        row = v1_std.filter((pl.col("player_id") == pid) & (pl.col("game_id") == last_gid))
        return float(row[col][0])

    dest = refresh(offline=True, as_of=known.isoformat())
    player_features = pl.read_parquet(dest / PLAYER_FEATURES_FILENAME)

    def snap(pid: int, abbr: str, col: str) -> float:
        row = player_features.filter(
            (pl.col("player_id") == pid) & (pl.col("team_abbr") == abbr)
        )
        return float(row[col][0])

    rolling_cols = (
        "p_min_avg_10", "p_min_avg_5", "p_ts_10", "p_pts_per_min_10",
        "p_usage_avg_10", "p_games_played_season",
    )
    for pid, abbr in ((10, "BOS"), (30, "LAL")):
        for col in rolling_cols:
            assert abs(snap(pid, abbr, col) - v1_val(pid, col)) <= 1e-6, (pid, col)
        for col in ("std_minutes_avg", "std_pts_avg", "std_games"):
            assert abs(snap(pid, abbr, col) - v1_std_val(pid, col)) <= 1e-6, (pid, col)

    # p10's minutes over the 10 games before the last are 21..30 -> mean 25.5.
    assert abs(snap(10, "BOS", "p_min_avg_10") - 25.5) <= 1e-6

    # Traded player: identical (leak-free) form on both teams, equal to v1.
    p20 = v1_val(20, "p_min_avg_10")
    assert abs(snap(20, "BOS", "p_min_avg_10") - p20) <= 1e-6
    assert abs(snap(20, "LAL", "p_min_avg_10") - p20) <= 1e-6


# ---------------------------------------------------------------------------
# snapshot-status (§17.2/§17.3)
# ---------------------------------------------------------------------------

def test_snapshot_status_block_format() -> None:
    """status.format_status renders the documented block with a deterministic
    freshness verdict (inject ``now``); past STALE_AFTER_DAYS it flips to
    STALE with a refresh hint."""
    provenance = {
        "as_of_date": "2024-03-15",
        "refreshed_at": "2024-03-16T08:42:11Z",
        "source": "nba_api+interim",
        "n_teams": 30,
        "n_players": 451,
        "interim_latest_game_date": "2024-03-14",
        "code_version": "v0.2.0-dev",
        "model_checkpoint_used_for_validation": None,
    }
    fresh = format_status(provenance, now=_dt.datetime(2024, 3, 15, 12, tzinfo=_dt.UTC))
    lines = fresh.splitlines()
    assert lines[0].startswith("as_of_date:") and lines[0].rstrip().endswith("2024-03-15")
    assert lines[1].startswith("refreshed_at:")
    assert lines[2].rstrip().endswith("nba_api+interim")
    assert lines[3].rstrip().endswith("30")
    assert lines[4].rstrip().endswith("451")
    assert lines[5].startswith("interim_latest_game_date:")
    assert lines[6].startswith("freshness:") and "fresh (1 day old)" in lines[6]

    stale = format_status(provenance, now=_dt.datetime(2024, 4, 1, tzinfo=_dt.UTC))
    assert "STALE" in stale and "run 'nba-sim refresh'" in stale


# ---------------------------------------------------------------------------
# Phase 8 — network / slow (full live refresh)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
def test_refresh_under_30s() -> None:
    """Full 30-team live refresh ≤ 30 s, rate-limited at 0.6 s/req (§12).
    Phase 8."""
    pytest.skip("Phase 8: live nba_api refresh")
