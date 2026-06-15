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
import torch

from nba_sim.data.etl import (
    GAMES_FILENAME,
    PLAYER_BOX_FILENAME,
    TEAM_BOX_FILENAME,
    SplitSpec,
    _models_to_df,
    build_feature_tables,
    interim_to_processed,
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
from nba_sim.snapshot.build import build_synthetic_game_batch
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
from nba_sim.training.dataset import (
    _MATCHUP_TEAM_DIFF_BASES,
    _PLAYER_NUMERIC_COLS,
    BoxScoreDataset,
)

# Model-input batch keys build_synthetic_game_batch must emit (B==1).
_MODEL_INPUT_KEYS = {
    "home_player_feats", "away_player_feats",
    "home_player_ids", "away_player_ids",
    "home_role_ids", "away_role_ids",
    "home_mask", "away_mask",
    "context", "matchup",
}
# Indices in the 47-dim player numeric block that build.py cold-starts to the
# league mean (opponent-vs-position), so they won't match v1 exactly.
_OPP_COLS = ("opp_def_rtg_vs_pos", "opp_blk_allowed_vs_pos")

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
    rosters, _qa = build_rosters(
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
# Step 4 — build_synthetic_game_batch (§15.5)
# ---------------------------------------------------------------------------

@pytest.fixture()
def processed_env(
    snapshot_env: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> tuple[_Fixture, Path, Path]:
    """Extend ``snapshot_env`` with the v1 processed parquets (train + val both
    built from the fixture season) so ``build.py`` can adopt train's
    feature_stats / id_map and the equivalence test can read v1's per-game
    vectors."""
    processed_root = snapshot_env.interim_root.parent / "processed"
    monkeypatch.setenv("NBA_SIM_PROCESSED_DIR", str(processed_root))
    build_feature_tables(snapshot_env.season)
    interim_to_processed(
        SplitSpec(train=[snapshot_env.season], val=[snapshot_env.season], test=[])
    )
    return snapshot_env, processed_root / "train.parquet", processed_root / "val.parquet"


def test_build_synthetic_game_batch_shapes(
    processed_env: tuple[_Fixture, Path, Path],
) -> None:
    """The batch carries exactly the model-input keys with B==1 and the
    documented tensor shapes/dtypes; the player lists match the active masks."""
    fix, train_pq, _ = processed_env
    dest = refresh(offline=True)
    batch, home_players, away_players, as_of_iso = build_synthetic_game_batch(
        home_team="BOS", away_team="LAL",
        snapshot_dir=dest, train_parquet=train_pq, interim_dir=fix.interim_root,
    )
    p = 15
    assert set(batch) == _MODEL_INPUT_KEYS
    assert batch["home_player_feats"].shape == (1, p, 55)
    assert batch["away_player_feats"].shape == (1, p, 55)
    assert batch["home_player_ids"].shape == (1, p)
    assert batch["home_player_ids"].dtype == torch.int64
    assert batch["home_mask"].shape == (1, p)
    assert batch["home_mask"].dtype == torch.bool
    assert batch["context"].shape == (1, 24)
    assert batch["matchup"].shape == (1, 16)
    assert as_of_iso == (fix.dates[-1] + _dt.timedelta(days=1)).isoformat()
    assert int(batch["home_mask"].sum()) == len(home_players)
    assert int(batch["away_mask"].sum()) == len(away_players)


def test_build_synthetic_game_batch_no_nans(
    processed_env: tuple[_Fixture, Path, Path],
) -> None:
    """No non-finite values reach the model: the null opp-vs-position numerics
    standardize to the league mean (cold-start, §15.4)."""
    fix, train_pq, _ = processed_env
    dest = refresh(offline=True)
    batch, *_ = build_synthetic_game_batch(
        home_team="BOS", away_team="LAL",
        snapshot_dir=dest, train_parquet=train_pq, interim_dir=fix.interim_root,
    )
    for key in ("home_player_feats", "away_player_feats", "context", "matchup"):
        assert torch.isfinite(batch[key]).all(), key


def test_context_vector_matches_v1_at_known_date(
    processed_env: tuple[_Fixture, Path, Path],
) -> None:
    """At a real game date the snapshot-built context + matchup vectors match
    v1's vectors for that game within float noise (§15.5). The fixture's 2-day
    spacing keeps is_3in4 / is_4in6 False in v1 too, so the full 24- and 16-dim
    vectors align. p10's player feats match v1 except the two cold-started
    opp-vs-position numerics."""
    fix, train_pq, val_pq = processed_env
    known = fix.dates[-1]
    last_gid = fix.gids[-1]

    train_ds = BoxScoreDataset(train_pq)
    val_ds = BoxScoreDataset(
        val_pq, player_id_map=train_ds.player_id_map, feature_stats=train_ds.feature_stats
    )
    matches = [i for i, g in enumerate(val_ds._games) if g["game_id"][0] == last_gid]
    assert matches, f"game {last_gid} not in val parquet"
    v1 = val_ds[matches[0]]

    dest = refresh(offline=True, as_of=known.isoformat())
    batch, _hp, _ap, _iso = build_synthetic_game_batch(
        home_team="BOS", away_team="LAL",
        snapshot_dir=dest, train_parquet=train_pq, interim_dir=fix.interim_root,
    )

    assert torch.allclose(batch["context"][0], v1["context"], atol=1e-4)
    assert torch.allclose(batch["matchup"][0], v1["matchup"], atol=1e-4)

    # p10 has the top minutes under both orderings -> home slot 0 on each side.
    opp_idx = {_PLAYER_NUMERIC_COLS.index(c) for c in _OPP_COLS}
    keep = [i for i in range(55) if i not in opp_idx]
    snap_p10 = batch["home_player_feats"][0, 0, keep]
    v1_p10 = v1["home_player_feats"][0, keep]
    assert torch.allclose(snap_p10, v1_p10, atol=1e-4)


# ---------------------------------------------------------------------------
# Phase 8 — live roster resolution (non-network: fetch layer is mocked)
# ---------------------------------------------------------------------------
#
# build_rosters(offline=False) fetches CommonTeamRoster per team. These tests
# inject canned frames by monkeypatching the fetch seam (rosters._get_team_list
# / rosters.fetch_roster / rosters.fetch_player_info, mirroring the v1 ETL's
# _stub_fetchers pattern), so they never touch the network.


def _roster_df(team_id: int, players: list[tuple[int, str, str]]) -> pl.DataFrame:
    """A canned ``CommonTeamRoster`` dataset-0 frame for one team.

    ``players`` is ``[(player_id, player_name, position), ...]``; bio columns
    are filled with valid values so the CommonPlayerInfo backfill never fires.
    Column names mirror the real endpoint (see RawRosterEntry).
    """
    rows = [
        {
            "TeamID": team_id, "SEASON": "2023", "LeagueID": "00",
            "PLAYER": name, "PLAYER_SLUG": name.lower().replace(" ", "-"),
            "NUM": str(pid), "POSITION": pos, "HEIGHT": "6-6", "WEIGHT": "215",
            "BIRTH_DATE": "1995-05-05T00:00:00", "AGE": 29.0, "EXP": "4",
            "SCHOOL": "Test U", "PLAYER_ID": pid,
        }
        for pid, name, pos in players
    ]
    return pl.DataFrame(rows)


def _patch_live_fetch(
    monkeypatch: pytest.MonkeyPatch,
    teams: list[tuple[int, str]],
    roster_by_team: dict[int, pl.DataFrame],
) -> None:
    """Patch the live fetch seam to canned frames (no network)."""
    from nba_sim.snapshot import rosters as rosters_mod

    monkeypatch.setattr(
        rosters_mod, "_get_team_list",
        lambda: [{"id": tid, "abbreviation": abbr} for tid, abbr in teams],
    )
    monkeypatch.setattr(
        rosters_mod, "fetch_roster",
        lambda team_id, season, *, refresh=False: roster_by_team[team_id],
    )


def test_roster_bio_parsers() -> None:
    """The Phase-8 bio parsers handle nba_api's quirky formats (§13.3)."""
    from nba_sim.snapshot.rosters import (
        parse_birth_date,
        parse_experience,
        parse_height_to_inches,
        parse_weight,
    )

    assert parse_height_to_inches("6-9") == 81.0
    assert parse_height_to_inches("") is None and parse_height_to_inches(None) is None
    assert parse_weight("215") == 215.0 and parse_weight("") is None
    assert parse_experience("R") == 0 and parse_experience("12") == 12
    assert parse_experience("") is None and parse_experience(None) is None
    assert parse_birth_date("1984-12-30T00:00:00") == _dt.date(1984, 12, 30)
    assert parse_birth_date("DEC 30, 1984") == _dt.date(1984, 12, 30)
    assert parse_birth_date("") is None and parse_birth_date("garbage") is None


def test_live_rosters_route_malformed_to_qa(monkeypatch: pytest.MonkeyPatch) -> None:
    """A roster row that fails RawRosterEntry validation is routed to the QA
    list (not raised) and the good rows survive — the v1 ETL's QA discipline
    so one bad entry never poisons the refresh (§14.2)."""
    good = _roster_df(_BOS, [(10, "Star A", "G"), (11, "Role B", "F")])
    bad = pl.DataFrame([{  # PLAYER_ID null -> validation failure
        "TeamID": _BOS, "SEASON": "2023", "LeagueID": "00", "PLAYER": "Broken",
        "PLAYER_SLUG": "broken", "NUM": "", "POSITION": "", "HEIGHT": "6-1",
        "WEIGHT": "180", "BIRTH_DATE": None, "AGE": None, "EXP": None,
        "SCHOOL": None, "PLAYER_ID": None,
    }])
    roster = pl.concat([good, bad], how="vertical_relaxed")
    _patch_live_fetch(monkeypatch, [(_BOS, "BOS")], {_BOS: roster})

    df, malformed = build_rosters(
        games=pl.DataFrame(), player_box=pl.DataFrame(),
        as_of_date=_dt.date(2024, 3, 15), season=2023, offline=False,
    )
    assert df.columns == list(ROSTER_COLUMNS)
    assert df.height == 2
    assert sorted(df["player_id"].to_list()) == [10, 11]
    assert len(malformed) == 1
    assert malformed[0]["team_id"] == _BOS and malformed[0]["player_id"] is None
    assert "validation" in malformed[0]["error"]


def test_live_rosters_cache_hit_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_rosters(offline=False) goes through the real on-disk fetch cache:
    a second build makes no new network call and returns byte-identical rosters
    (§19 cache-discipline property). The network boundary (fetch._call_endpoint)
    is mocked + counted; the cache dir is redirected to tmp_path."""
    monkeypatch.setenv("NBA_SIM_CACHE_DIR", str(tmp_path / "cache"))
    from nba_sim.data import fetch
    from nba_sim.snapshot import rosters as rosters_mod

    monkeypatch.setattr(
        rosters_mod, "_get_team_list",
        lambda: [{"id": _BOS, "abbreviation": "BOS"}, {"id": _LAL, "abbreviation": "LAL"}],
    )
    roster_for = {
        _BOS: _roster_df(_BOS, [(10, "Star A", "G"), (11, "Role B", "F")]),
        _LAL: _roster_df(_LAL, [(30, "Star D", "C")]),
    }
    calls = {"n": 0}

    def _fake_endpoint(endpoint: str, params: dict[str, Any]) -> list[pl.DataFrame]:
        calls["n"] += 1
        tid = int(params["team_id"])
        # CommonTeamRoster emits [roster, coaches]; fetch_roster reads dataset 0.
        return [roster_for[tid], pl.DataFrame({"TEAM_ID": [tid]})]

    monkeypatch.setattr(fetch, "_call_endpoint", _fake_endpoint)

    empty = pl.DataFrame()
    as_of = _dt.date(2024, 3, 15)
    df1, qa1 = build_rosters(
        games=empty, player_box=empty, as_of_date=as_of, season=2023, offline=False
    )
    n_cold = calls["n"]
    assert n_cold == 2  # one network call per team (cold cache)
    assert qa1 == []
    assert df1.columns == list(ROSTER_COLUMNS)
    assert df1.height == 3

    df2, qa2 = build_rosters(
        games=empty, player_box=empty, as_of_date=as_of, season=2023, offline=False
    )
    assert calls["n"] == n_cold  # cache hit: no new network calls
    assert qa2 == []
    assert df2.equals(df1)  # deterministic + identical


def test_offline_live_feature_equivalence(
    snapshot_env: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§19 ship gate: for the same as-of date the live refresh's feature outputs
    match the offline-mode equivalent. The mocked live rosters reproduce the
    offline membership + identity (names/positions, traded player 20 on both
    teams) with non-null bio, so player_features / team_features / team_lastgame
    (all interim-derived) are byte-identical; only the rosters' bio differs."""
    as_of = (snapshot_env.dates[-1] + _dt.timedelta(days=1)).isoformat()

    dest = refresh(offline=True, as_of=as_of)
    off_players = pl.read_parquet(dest / PLAYER_FEATURES_FILENAME)
    off_team = pl.read_parquet(dest / TEAM_FEATURES_FILENAME)
    off_last = pl.read_parquet(dest / TEAM_LASTGAME_FILENAME)
    off_rosters = pl.read_parquet(dest / ROSTERS_FILENAME)

    _patch_live_fetch(
        monkeypatch, [(_BOS, "BOS"), (_LAL, "LAL")],
        {
            _BOS: _roster_df(_BOS, [(10, "Star A", "G"), (11, "Role B", "F"),
                                    (20, "Traded C", "F")]),
            _LAL: _roster_df(_LAL, [(30, "Star D", "C"), (20, "Traded C", "F")]),
        },
    )
    dest = refresh(offline=False, as_of=as_of)
    live_players = pl.read_parquet(dest / PLAYER_FEATURES_FILENAME)
    live_team = pl.read_parquet(dest / TEAM_FEATURES_FILENAME)
    live_last = pl.read_parquet(dest / TEAM_LASTGAME_FILENAME)
    live_rosters = pl.read_parquet(dest / ROSTERS_FILENAME)

    # Feature tables are interim-derived -> identical regardless of roster source.
    assert live_team.equals(off_team)
    assert live_last.equals(off_last)
    assert live_players.equals(off_players)

    # Rosters share membership but differ in bio (the documented difference).
    def _membership(df: pl.DataFrame) -> set[tuple[int, int]]:
        return set(zip(df["team_id"].to_list(), df["player_id"].to_list(), strict=True))

    assert _membership(live_rosters) == _membership(off_rosters)
    assert off_rosters["height_in"].null_count() == off_rosters.height  # offline: null bio
    assert live_rosters["height_in"].null_count() == 0  # live: CommonTeamRoster bio


# ---------------------------------------------------------------------------
# Phase 8 — network / slow (full live refresh)
# ---------------------------------------------------------------------------


def _require_network(host: str = "stats.nba.com", port: int = 443) -> None:
    """Skip unless ``host:port`` accepts a TCP connection (a short probe)."""
    import socket

    try:
        socket.create_connection((host, port), timeout=3).close()
    except OSError:
        pytest.skip(f"no network access to {host}:{port}")


@pytest.mark.slow
@pytest.mark.network
def test_refresh_under_30s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Full 30-team live refresh ≤ 30 s, rate-limited at 0.6 s/req (§12 / §19).

    Reads the real ``data/interim`` (skips if absent) and hits the live
    nba_api (skips if unreachable). The snapshot is written into ``tmp_path``
    so the user's real ``data/snapshot`` is never touched. Without network /
    artifacts (CI) it skips cleanly; the ``network`` marker also keeps it out
    of a ``-m "not network"`` run.
    """
    import time

    from nba_sim.data.etl import interim_dir
    from nba_sim.snapshot.refresh import _discover_seasons
    from nba_sim.snapshot.refresh import refresh as run_refresh

    if not _discover_seasons(interim_dir()):
        pytest.skip("no interim seasons on disk — run the v1 ETL first")
    _require_network()

    # Redirect only the snapshot output; read the real interim + fetch cache.
    monkeypatch.setenv("NBA_SIM_SNAPSHOT_DIR", str(tmp_path / "snapshot"))

    start = time.monotonic()
    dest = run_refresh(offline=False, force=True)
    elapsed = time.monotonic() - start

    prov = json.loads((dest / AS_OF_FILENAME).read_text())
    assert prov["source"] == "nba_api+interim"
    assert prov["n_teams"] == 30, f"expected 30 teams, got {prov['n_teams']}"
    assert elapsed <= 30.0, f"live refresh took {elapsed:.1f}s (> 30s budget)"
