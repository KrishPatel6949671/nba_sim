"""ETL-layer tests.

Pure tests — no network, no real fetcher. We monkey-patch the four
``fetch.*`` helpers imported into :mod:`nba_sim.data.etl` to return
synthetic V3-shaped dataframes, then exercise the orchestration end-to-end.

Covers:
    - Minute-validation helpers and the regulation/OT minute set.
    - ``_group_raw_games`` / ``_build_games`` / ``_build_player_lines`` /
      ``_build_team_lines`` / ``_extract_advanced`` / ``_derive_rosters``.
    - ``_models_to_df`` (empty + non-empty).
    - ``_is_up_to_date`` (no outputs, outputs newer than inputs).
    - ``raw_to_interim`` happy path, short-game drop, fetch-failure drop,
      pair-mismatch drop, and incremental skip.
    - ``build_feature_tables`` happy path, missing-interim, idempotent skip,
      ``refresh=True`` rewrite, and matchup columns wired in.
    - ``interim_to_processed`` writes all three split files, joins p_*, t_*,
      and opp_*/h2h_* columns, auto-builds feature tables, handles empty
      splits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nba_sim.data import etl
from nba_sim.data.etl import (
    _build_games,
    _build_player_lines,
    _build_team_lines,
    _derive_rosters,
    _extract_advanced,
    _group_raw_games,
    _is_up_to_date,
    _models_to_df,
    _team_minutes_ok,
    _valid_team_minute_totals,
    raw_to_interim,
)
from nba_sim.data.schema import (
    Game,
    PlayerBoxLine,
    RawGame,
    RawPlayerBoxLine,
)


# ---------------------------------------------------------------------------
# Synthetic V3 row builders (overlap with test_data_schema deliberately —
# importing across test modules is brittle, duplication is cheap).
# ---------------------------------------------------------------------------

def _v3_game_row(
    team_abbr: str,
    opp_abbr: str,
    team_id: int,
    gid: str,
    *,
    is_home: bool,
    season_id: str = "22023",
    date: str = "2024-04-14",
    pts: int = 110,
    mins: int = 240,
) -> dict[str, Any]:
    matchup = f"{team_abbr} vs. {opp_abbr}" if is_home else f"{team_abbr} @ {opp_abbr}"
    return {
        "SEASON_ID": season_id,
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": team_abbr,
        "GAME_ID": gid,
        "GAME_DATE": date,
        "MATCHUP": matchup,
        "WL": "W" if pts > 100 else "L",
        "PTS": pts,
        "MIN": mins,
    }


def _v3_player_row(
    gid: str,
    team_id: int,
    team_abbr: str,
    person_id: int,
    *,
    position: str = "F",
    mins: str = "34:12",
    pts: int = 20,
) -> dict[str, Any]:
    return {
        "gameId": gid,
        "teamId": team_id,
        "teamTricode": team_abbr,
        "personId": person_id,
        "firstName": "A",
        "familyName": "B",
        "position": position,
        "comment": "" if position else "DNP - Coach's Decision",
        "minutes": mins,
        "fieldGoalsMade": 8,
        "fieldGoalsAttempted": 15,
        "threePointersMade": 2,
        "threePointersAttempted": 5,
        "freeThrowsMade": 2,
        "freeThrowsAttempted": 2,
        "reboundsOffensive": 1,
        "reboundsDefensive": 4,
        "reboundsTotal": 5,
        "assists": 3,
        "steals": 1,
        "blocks": 0,
        "turnovers": 2,
        "foulsPersonal": 3,
        "points": pts,
        "plusMinusPoints": 5.0,
    }


def _v3_team_row(
    gid: str,
    team_id: int,
    team_abbr: str,
    *,
    mins: str = "240:00",
    pts: int = 110,
) -> dict[str, Any]:
    return {
        "gameId": gid,
        "teamId": team_id,
        "teamTricode": team_abbr,
        "minutes": mins,
        "fieldGoalsMade": 40,
        "fieldGoalsAttempted": 88,
        "threePointersMade": 12,
        "threePointersAttempted": 35,
        "freeThrowsMade": 18,
        "freeThrowsAttempted": 22,
        "reboundsOffensive": 10,
        "reboundsDefensive": 32,
        "reboundsTotal": 42,
        "assists": 25,
        "steals": 7,
        "blocks": 4,
        "turnovers": 13,
        "foulsPersonal": 19,
        "points": pts,
        "plusMinusPoints": 4.0,
    }


def _v3_advanced_row(
    gid: str,
    team_id: int,
    *,
    pace: float = 100.0,
    off_rtg: float = 110.0,
    def_rtg: float = 105.0,
) -> dict[str, Any]:
    return {
        "gameId": gid,
        "teamId": team_id,
        "pace": pace,
        "offensiveRating": off_rtg,
        "defensiveRating": def_rtg,
    }


def _game_pair_df(
    gid: str = "0022300001",
    *,
    home_tid: int = 1,
    away_tid: int = 2,
    home_pts: int = 120,
    away_pts: int = 100,
    mins: int = 240,
    season_id: str = "22023",
) -> pl.DataFrame:
    return pl.DataFrame([
        _v3_game_row("BOS", "LAL", home_tid, gid, is_home=True,
                     pts=home_pts, mins=mins, season_id=season_id),
        _v3_game_row("LAL", "BOS", away_tid, gid, is_home=False,
                     pts=away_pts, mins=mins, season_id=season_id),
    ])


# ---------------------------------------------------------------------------
# Minute-validation helpers
# ---------------------------------------------------------------------------

def test_valid_team_minute_totals_covers_regulation_and_OTs() -> None:
    totals = _valid_team_minute_totals()
    # regulation + 6 OTs (NBA all-time record).
    assert totals == {240.0, 265.0, 290.0, 315.0, 340.0, 365.0, 390.0}


@pytest.mark.parametrize(
    ("minutes", "ok"),
    [
        (240.0, True),     # regulation
        (240.3, True),     # within tolerance
        (239.6, True),     # within tolerance
        (265.0, True),     # 1 OT
        (390.0, True),     # 6 OT (max we model)
        (235.0, False),    # short game
        (250.0, False),    # between regulation and OT
        (391.0, False),    # past 6 OT
    ],
)
def test_team_minutes_ok(minutes: float, ok: bool) -> None:
    assert _team_minutes_ok(minutes) is ok


# ---------------------------------------------------------------------------
# _group_raw_games / _build_games
# ---------------------------------------------------------------------------

def test_group_raw_games_pairs_by_game_id() -> None:
    df = pl.concat([
        _game_pair_df("0022300001"),
        _game_pair_df("0022300002", home_tid=3, away_tid=4),
    ])
    groups = _group_raw_games(df)
    assert set(groups) == {"0022300001", "0022300002"}
    assert all(len(rows) == 2 for rows in groups.values())


def test_group_raw_games_skips_invalid_rows() -> None:
    """A row missing a required RawGame field is dropped silently."""
    rows = [
        _v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True),
        _v3_game_row("LAL", "BOS", 2, "0022300001", is_home=False),
    ]
    rows[0]["MATCHUP"] = None  # MATCHUP is required (str, not Optional)
    df = pl.DataFrame(rows)
    groups = _group_raw_games(df)
    # The first row was rejected; only the away row survives.
    assert len(groups.get("0022300001", [])) == 1


def test_build_games_happy_path() -> None:
    groups = _group_raw_games(_game_pair_df())
    games, dropped = _build_games(groups)
    assert len(games) == 1
    assert dropped == []
    assert games[0].home_team_abbr == "BOS"
    assert games[0].away_team_abbr == "LAL"


def test_build_games_drops_single_row_games() -> None:
    """LGF should always have 2 rows per game; a 1-row group is data corruption."""
    df = pl.DataFrame([_v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True)])
    groups = _group_raw_games(df)
    games, dropped = _build_games(groups)
    assert games == []
    assert len(dropped) == 1
    assert dropped[0]["game_id"] == "0022300001"
    assert "Expected 2" in dropped[0]["reason"]


# ---------------------------------------------------------------------------
# _build_player_lines
# ---------------------------------------------------------------------------

def test_build_player_lines_validates_and_keeps_dnps() -> None:
    df = pl.DataFrame([
        _v3_player_row("0022300001", 1, "BOS", 100),
        _v3_player_row("0022300001", 1, "BOS", 101, position="", mins=""),
    ])
    lines, dropped = _build_player_lines(df)
    assert len(lines) == 2
    assert dropped == []
    by_pid = {p.player_id: p for p in lines}
    assert by_pid[100].is_starter is True and by_pid[100].dnp is False
    assert by_pid[101].is_starter is False and by_pid[101].dnp is True


def test_build_player_lines_drops_invariant_violators() -> None:
    """One bad row shouldn't tank the whole game."""
    bad = _v3_player_row("0022300001", 1, "BOS", 100)
    bad["fieldGoalsMade"] = 99
    bad["fieldGoalsAttempted"] = 10
    good = _v3_player_row("0022300001", 1, "BOS", 101)
    df = pl.DataFrame([bad, good])
    lines, dropped = _build_player_lines(df)
    assert len(lines) == 1
    assert lines[0].player_id == 101
    assert len(dropped) == 1
    assert dropped[0]["person_id"] == "100"
    assert "fgm" in dropped[0]["reason"]


# ---------------------------------------------------------------------------
# _extract_advanced
# ---------------------------------------------------------------------------

def test_extract_advanced_happy() -> None:
    df = pl.DataFrame([
        _v3_advanced_row("g", 1, pace=100.0, off_rtg=115.0, def_rtg=105.0),
        _v3_advanced_row("g", 2, pace=98.0, off_rtg=105.0, def_rtg=115.0),
    ])
    pace, off_rtg, def_rtg, missing = _extract_advanced(df, team_id=1)
    assert (pace, off_rtg, def_rtg) == (100.0, 115.0, 105.0)
    assert missing == []


def test_extract_advanced_team_not_present() -> None:
    """Filter yields nothing — return Nones but don't flag columns missing."""
    df = pl.DataFrame([_v3_advanced_row("g", 1)])
    pace, off_rtg, def_rtg, missing = _extract_advanced(df, team_id=999)
    assert (pace, off_rtg, def_rtg) == (None, None, None)
    assert missing == []


def test_extract_advanced_empty_df_flags_all_cols_missing() -> None:
    """Empty df from upstream is treated as "endpoint returned nothing"."""
    pace, off_rtg, def_rtg, missing = _extract_advanced(pl.DataFrame(), team_id=1)
    assert (pace, off_rtg, def_rtg) == (None, None, None)
    assert set(missing) == {"pace", "offensiveRating", "defensiveRating"}


def test_extract_advanced_missing_one_column() -> None:
    """A column-name drift should surface in `missing`, not crash."""
    df = pl.DataFrame([{
        "gameId": "g", "teamId": 1,
        "offensiveRating": 110.0, "defensiveRating": 105.0,
    }])
    pace, off_rtg, def_rtg, missing = _extract_advanced(df, team_id=1)
    assert pace is None
    assert off_rtg == 110.0
    assert def_rtg == 105.0
    assert missing == ["pace"]


# ---------------------------------------------------------------------------
# _build_team_lines
# ---------------------------------------------------------------------------

def test_build_team_lines_merges_advanced() -> None:
    team_box = pl.DataFrame([
        _v3_team_row("g", 1, "BOS", pts=120),
        _v3_team_row("g", 2, "LAL", pts=100),
    ])
    adv = pl.DataFrame([
        _v3_advanced_row("g", 1, pace=101.0, off_rtg=120.0, def_rtg=100.0),
        _v3_advanced_row("g", 2, pace=101.0, off_rtg=100.0, def_rtg=120.0),
    ])
    lines, dropped, missing = _build_team_lines(team_box, adv, home_team_id=1)
    assert len(lines) == 2 and dropped == [] and missing == []
    home = next(t for t in lines if t.is_home)
    away = next(t for t in lines if not t.is_home)
    assert home.team_abbr == "BOS" and home.pts == 120
    assert (home.pace, home.off_rtg, home.def_rtg) == (101.0, 120.0, 100.0)
    # off/def rating swap symmetrically — sanity-check the join didn't crosswire.
    assert (away.off_rtg, away.def_rtg) == (100.0, 120.0)


def test_build_team_lines_reports_missing_advanced_cols() -> None:
    team_box = pl.DataFrame([_v3_team_row("g", 1, "BOS")])
    adv = pl.DataFrame([{
        "gameId": "g", "teamId": 1,
        "offensiveRating": 110.0, "defensiveRating": 105.0,
    }])
    lines, _dropped, missing = _build_team_lines(team_box, adv, home_team_id=1)
    assert len(lines) == 1
    assert lines[0].pace is None
    assert lines[0].off_rtg == 110.0
    assert "pace" in missing


# ---------------------------------------------------------------------------
# _derive_rosters
# ---------------------------------------------------------------------------

def test_derive_rosters_dedupes_and_groups_by_team() -> None:
    lines = []
    for tid, abbr, pid in [(1, "BOS", 100), (1, "BOS", 101), (2, "LAL", 200), (1, "BOS", 100)]:
        raw = RawPlayerBoxLine.model_validate(_v3_player_row("g", tid, abbr, pid))
        lines.append(PlayerBoxLine.from_raw(raw))
    rosters = _derive_rosters(lines, season=2023)
    assert len(rosters) == 2
    by_team = {r.team_id: r for r in rosters}
    assert by_team[1].player_ids == [100, 101]  # 100 dedup'd
    assert by_team[2].player_ids == [200]
    assert all(r.season == 2023 for r in rosters)


# ---------------------------------------------------------------------------
# _models_to_df
# ---------------------------------------------------------------------------

def test_models_to_df_empty_input_returns_empty_df() -> None:
    df = _models_to_df([])
    assert df.shape == (0, 0)


def test_models_to_df_preserves_model_fields() -> None:
    raw_home = RawGame.model_validate(_v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True))
    raw_away = RawGame.model_validate(_v3_game_row("LAL", "BOS", 2, "0022300001", is_home=False))
    g = Game.from_raw_pair([raw_home, raw_away])
    df = _models_to_df([g])
    assert df.shape[0] == 1
    assert set(df.columns) >= {
        "game_id", "season", "date", "home_team_id", "away_team_id",
        "home_pts", "away_pts", "is_overtime", "is_playoffs", "dropped",
    }
    assert df["game_id"][0] == "0022300001"


# ---------------------------------------------------------------------------
# _is_up_to_date
# ---------------------------------------------------------------------------

def _redirect_dirs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Point both interim and cache dirs at tmp_path so the test never touches
    real data. Returns (interim_root, cache_root)."""
    interim = tmp_path / "interim"
    cache = tmp_path / "cache"
    monkeypatch.setenv("NBA_SIM_INTERIM_DIR", str(interim))
    monkeypatch.setenv("NBA_SIM_CACHE_DIR", str(cache))
    return interim, cache


def test_is_up_to_date_false_when_outputs_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_dirs(monkeypatch, tmp_path)
    assert _is_up_to_date(2023) is False


def test_is_up_to_date_true_when_outputs_newer_than_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interim, cache = _redirect_dirs(monkeypatch, tmp_path)

    # Old cache file
    (cache / "leaguegamefinder").mkdir(parents=True)
    inp = cache / "leaguegamefinder" / "x.parquet"
    inp.write_bytes(b"")
    os.utime(inp, (1_700_000_000, 1_700_000_000))

    # Fresh interim outputs (mtime defaults to now)
    out_dir = interim / "2023"
    out_dir.mkdir(parents=True)
    for name in (
        etl.GAMES_FILENAME, etl.PLAYER_BOX_FILENAME, etl.TEAM_BOX_FILENAME,
        etl.ROSTERS_FILENAME, etl.QA_REPORT_FILENAME,
    ):
        (out_dir / name).write_bytes(b"")

    assert _is_up_to_date(2023) is True


# ---------------------------------------------------------------------------
# raw_to_interim orchestration
# ---------------------------------------------------------------------------

def _stub_fetchers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    games_df: pl.DataFrame,
    player_box_df: pl.DataFrame,
    team_box_df: pl.DataFrame,
    advanced_df: pl.DataFrame,
) -> dict[str, int]:
    """Patch the fetch.* names *as imported into etl* and count calls."""
    counts = {"games": 0, "player": 0, "team": 0, "adv": 0}

    def _games(season: int, **_kw: Any) -> pl.DataFrame:
        counts["games"] += 1
        return games_df

    def _player(gid: str, **_kw: Any) -> pl.DataFrame:
        counts["player"] += 1
        return player_box_df

    def _team(gid: str, **_kw: Any) -> pl.DataFrame:
        counts["team"] += 1
        return team_box_df

    def _adv(gid: str, **_kw: Any) -> pl.DataFrame:
        counts["adv"] += 1
        return advanced_df

    monkeypatch.setattr(etl, "fetch_games_for_season", _games)
    monkeypatch.setattr(etl, "fetch_player_box", _player)
    monkeypatch.setattr(etl, "fetch_team_box", _team)
    monkeypatch.setattr(etl, "fetch_team_advanced_box", _adv)
    return counts


def test_raw_to_interim_writes_all_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_dirs(monkeypatch, tmp_path)
    gid = "0022300001"
    _stub_fetchers(
        monkeypatch,
        games_df=_game_pair_df(gid, home_tid=1, away_tid=2, home_pts=120, away_pts=100),
        player_box_df=pl.DataFrame([
            _v3_player_row(gid, 1, "BOS", 100),
            _v3_player_row(gid, 1, "BOS", 101),
            _v3_player_row(gid, 2, "LAL", 200),
        ]),
        team_box_df=pl.DataFrame([
            _v3_team_row(gid, 1, "BOS", pts=120),
            _v3_team_row(gid, 2, "LAL", pts=100),
        ]),
        advanced_df=pl.DataFrame([
            _v3_advanced_row(gid, 1, pace=100.0, off_rtg=115.0, def_rtg=105.0),
            _v3_advanced_row(gid, 2, pace=100.0, off_rtg=105.0, def_rtg=115.0),
        ]),
    )

    out = raw_to_interim(2023)
    for fname in (
        etl.GAMES_FILENAME, etl.PLAYER_BOX_FILENAME, etl.TEAM_BOX_FILENAME,
        etl.ROSTERS_FILENAME, etl.QA_REPORT_FILENAME,
    ):
        assert (out / fname).exists(), f"missing {fname}"

    games = pl.read_parquet(out / etl.GAMES_FILENAME)
    assert games.shape[0] == 1
    assert games["game_id"][0] == gid
    assert games["home_team_abbr"][0] == "BOS"
    assert games["dropped"][0] is False

    pb = pl.read_parquet(out / etl.PLAYER_BOX_FILENAME)
    assert pb.shape[0] == 3

    tb = pl.read_parquet(out / etl.TEAM_BOX_FILENAME)
    assert tb.shape[0] == 2
    assert sorted(tb["pace"].to_list()) == [100.0, 100.0]

    rosters = pl.read_parquet(out / etl.ROSTERS_FILENAME)
    assert rosters.shape[0] == 2

    qa = json.loads((out / etl.QA_REPORT_FILENAME).read_text())
    assert qa["season"] == 2023
    assert qa["n_games_kept"] == 1
    assert qa["n_games_dropped"] == 0
    assert qa["n_player_lines"] == 3
    assert qa["n_team_lines"] == 2
    assert qa["missing_advanced_cols"] == []
    assert qa["dropped_games"] == []


def test_raw_to_interim_drops_short_game(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Team-minutes total ≠ regulation/OT set => game marked dropped, rows excluded."""
    _redirect_dirs(monkeypatch, tmp_path)
    gid = "0022300001"
    _stub_fetchers(
        monkeypatch,
        games_df=_game_pair_df(gid, mins=235),
        player_box_df=pl.DataFrame([_v3_player_row(gid, 1, "BOS", 100)]),
        team_box_df=pl.DataFrame([
            _v3_team_row(gid, 1, "BOS", mins="235:00"),
            _v3_team_row(gid, 2, "LAL", mins="235:00"),
        ]),
        advanced_df=pl.DataFrame([
            _v3_advanced_row(gid, 1), _v3_advanced_row(gid, 2),
        ]),
    )

    out = raw_to_interim(2023)
    games = pl.read_parquet(out / etl.GAMES_FILENAME)
    assert games["dropped"][0] is True
    assert "team_minutes_anomaly" in games["dropped_reason"][0]

    # Row tables exclude the dropped game.
    pb = pl.read_parquet(out / etl.PLAYER_BOX_FILENAME)
    assert pb.shape == (0, 0)  # _models_to_df([]) -> empty

    qa = json.loads((out / etl.QA_REPORT_FILENAME).read_text())
    assert qa["n_games_kept"] == 0
    assert qa["n_games_dropped"] == 1
    assert qa["dropped_games"][0]["game_id"] == gid


def test_raw_to_interim_drops_game_on_fetch_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _redirect_dirs(monkeypatch, tmp_path)
    gid = "0022300001"
    monkeypatch.setattr(etl, "fetch_games_for_season", lambda s, **_kw: _game_pair_df(gid))

    def _boom(*_a: Any, **_kw: Any) -> pl.DataFrame:
        raise RuntimeError("simulated 404")

    monkeypatch.setattr(etl, "fetch_player_box", _boom)
    monkeypatch.setattr(etl, "fetch_team_box", lambda *_a, **_kw: pl.DataFrame())
    monkeypatch.setattr(etl, "fetch_team_advanced_box", lambda *_a, **_kw: pl.DataFrame())

    out = raw_to_interim(2023)
    games = pl.read_parquet(out / etl.GAMES_FILENAME)
    assert games["dropped"][0] is True
    assert "fetch_failed" in games["dropped_reason"][0]

    qa = json.loads((out / etl.QA_REPORT_FILENAME).read_text())
    assert qa["n_games_kept"] == 0
    assert any("simulated 404" in d["reason"] for d in qa["dropped_games"])


def test_raw_to_interim_reports_pair_mismatch_in_qa(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A LGF group with only 1 row never makes it to the fetch step;
    it appears in dropped_games with the build-time reason."""
    _redirect_dirs(monkeypatch, tmp_path)
    gid = "0022300001"
    single_row_df = pl.DataFrame([
        _v3_game_row("BOS", "LAL", 1, gid, is_home=True),
    ])
    _stub_fetchers(
        monkeypatch,
        games_df=single_row_df,
        player_box_df=pl.DataFrame(),
        team_box_df=pl.DataFrame(),
        advanced_df=pl.DataFrame(),
    )

    out = raw_to_interim(2023)
    qa = json.loads((out / etl.QA_REPORT_FILENAME).read_text())
    assert qa["n_games_kept"] == 0
    assert qa["n_games_dropped"] == 1
    assert qa["dropped_games"][0]["game_id"] == gid
    assert "Expected 2" in qa["dropped_games"][0]["reason"]


def test_raw_to_interim_incremental_skip_and_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Second call hits skip path (no extra fetcher invocations);
    refresh=True forces a re-run."""
    _redirect_dirs(monkeypatch, tmp_path)
    gid = "0022300001"
    counts = _stub_fetchers(
        monkeypatch,
        games_df=_game_pair_df(gid, home_tid=1, away_tid=2),
        player_box_df=pl.DataFrame([_v3_player_row(gid, 1, "BOS", 100)]),
        team_box_df=pl.DataFrame([
            _v3_team_row(gid, 1, "BOS"),
            _v3_team_row(gid, 2, "LAL"),
        ]),
        advanced_df=pl.DataFrame([
            _v3_advanced_row(gid, 1), _v3_advanced_row(gid, 2),
        ]),
    )

    raw_to_interim(2023)
    first = dict(counts)
    assert first["games"] == 1 and first["player"] == 1

    raw_to_interim(2023)
    assert counts == first, "second call should hit skip path"

    raw_to_interim(2023, refresh=True)
    assert counts["games"] == first["games"] + 1
    assert counts["player"] == first["player"] + 1


# ---------------------------------------------------------------------------
# build_feature_tables / interim_to_processed
#
# These exercise the feature-derivation half of the pipeline. We bypass
# raw_to_interim and seed the interim/<season>/ directory with synthetic
# typed parquets directly — this isolates the feature stage from the
# fetch/transform stage and keeps the tests fast.
# ---------------------------------------------------------------------------

import time
import datetime as _dt

from nba_sim.data.etl import (
    SplitSpec,
    build_feature_tables,
    interim_to_processed,
    processed_dir,
)


def _redirect_all_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    """interim + processed redirection. Tests that touch interim_to_processed
    need both env vars set so the output never lands in the real repo."""
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    monkeypatch.setenv("NBA_SIM_INTERIM_DIR", str(interim))
    monkeypatch.setenv("NBA_SIM_PROCESSED_DIR", str(processed))
    return interim, processed


def _seed_season_interim(season_dir: Path, season: int, *, n_games: int = 3) -> None:
    """Write a small typed interim layout for one season. Two teams, two
    players per team per game, regulation minutes, win/loss alternating
    so plus_minus has both signs. Sufficient to drive rolling features
    and surface any join breakage."""
    season_dir.mkdir(parents=True, exist_ok=True)

    dates = [_dt.date(season, 11, 1 + 2 * i) for i in range(n_games)]
    games_rows: list[dict[str, Any]] = []
    pb_rows: list[dict[str, Any]] = []
    tb_rows: list[dict[str, Any]] = []
    for i, d in enumerate(dates):
        gid = f"{season}{i:04d}"
        home_pts = 110 + i
        away_pts = 105 + i
        games_rows.append({
            "game_id": gid, "season": season, "date": d,
            "home_team_id": 1, "away_team_id": 2,
            "home_team_abbr": "BOS", "away_team_abbr": "LAL",
            "home_pts": home_pts, "away_pts": away_pts,
            "is_overtime": False, "is_playoffs": False,
            "dropped": False, "dropped_reason": None,
        })
        for team_id, abbr, pts, pm in [
            (1, "BOS", home_pts, float(home_pts - away_pts)),
            (2, "LAL", away_pts, float(away_pts - home_pts)),
        ]:
            tb_rows.append({
                "game_id": gid, "team_id": team_id, "team_abbr": abbr,
                "is_home": team_id == 1, "minutes": 240.0,
                "pts": pts, "fgm": 40, "fga": 85, "tpm": 12, "tpa": 30,
                "ftm": 18, "fta": 22, "oreb": 8, "dreb": 32, "reb": 40,
                "ast": 22, "stl": 7, "blk": 4, "tov": 14, "pf": 18,
                "plus_minus": pm,
                "pace": 100.0 + i, "off_rtg": 110.0 + i, "def_rtg": 105.0 - i,
            })
            # Two players per team per game.
            for slot, pid in enumerate(
                (100, 101) if team_id == 1 else (200, 201)
            ):
                pb_rows.append({
                    "game_id": gid, "player_id": pid, "player_name": f"P{pid}",
                    "team_id": team_id, "team_abbr": abbr,
                    "position": "G" if slot == 0 else "F",
                    "minutes": 30.0 + i, "pts": 20 + i,
                    "fgm": 8, "fga": 16, "tpm": 2, "tpa": 5,
                    "ftm": 2, "fta": 3,
                    "oreb": 1, "dreb": 5, "reb": 6,
                    "ast": 4, "stl": 1, "blk": 0, "tov": 2, "pf": 3,
                    "plus_minus": pm,
                    "is_starter": slot == 0, "is_active": True, "dnp": False,
                })

    pl.DataFrame(games_rows).write_parquet(season_dir / etl.GAMES_FILENAME)
    pl.DataFrame(pb_rows).write_parquet(season_dir / etl.PLAYER_BOX_FILENAME)
    pl.DataFrame(tb_rows).write_parquet(season_dir / etl.TEAM_BOX_FILENAME)


def test_build_feature_tables_writes_all_three_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    out = build_feature_tables(2022)
    for fname in (
        etl.PLAYER_FEATURES_FILENAME,
        etl.TEAM_FEATURES_FILENAME,
        etl.SEASON_TO_DATE_FILENAME,
    ):
        assert (out / fname).exists(), f"missing {fname}"

    pf = pl.read_parquet(out / etl.PLAYER_FEATURES_FILENAME)
    tf = pl.read_parquet(out / etl.TEAM_FEATURES_FILENAME)
    std = pl.read_parquet(out / etl.SEASON_TO_DATE_FILENAME)

    # Row counts: 3 games × 2 teams × 2 players = 12 player rows; 6 team rows.
    assert pf.height == 12
    assert tf.height == 6
    assert std.height == 12

    # Each output carries the expected feature prefix.
    assert any(c.startswith("p_") for c in pf.columns)
    assert any(c.startswith("t_") for c in tf.columns)
    assert any(c.startswith("std_") for c in std.columns)


def test_build_feature_tables_raises_when_interim_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A user who calls build_feature_tables before raw_to_interim should
    get a clear FileNotFoundError naming the missing path."""
    _redirect_all_dirs(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="run raw_to_interim first"):
        build_feature_tables(2022)


def test_build_feature_tables_skips_when_up_to_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Second call must not rewrite — mtime should be unchanged."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    out = build_feature_tables(2022)
    pf_path = out / etl.PLAYER_FEATURES_FILENAME
    mtime_before = pf_path.stat().st_mtime

    # Wait long enough that any rewrite would visibly change mtime.
    time.sleep(0.05)
    build_feature_tables(2022)
    assert pf_path.stat().st_mtime == mtime_before


def test_build_feature_tables_refresh_forces_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    out = build_feature_tables(2022)
    pf_path = out / etl.PLAYER_FEATURES_FILENAME
    mtime_before = pf_path.stat().st_mtime

    time.sleep(0.05)
    build_feature_tables(2022, refresh=True)
    assert pf_path.stat().st_mtime > mtime_before


def test_interim_to_processed_writes_all_three_split_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interim, processed = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)
    _seed_season_interim(interim / "2023", 2023)

    spec = SplitSpec(train=[2022], val=[2023], test=[])
    paths = interim_to_processed(spec)

    assert set(paths) == {"train", "val", "test"}
    for p in paths.values():
        assert p.exists(), f"{p} should be written even if empty"

    train = pl.read_parquet(paths["train"])
    val = pl.read_parquet(paths["val"])
    assert train.height == 12  # one season × 12 rows
    assert val.height == 12


def test_interim_to_processed_join_includes_both_player_and_team_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Joined frame must carry both p_* (player rolling) and t_* (team
    rolling) columns — that's the whole point of the join."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    spec = SplitSpec(train=[2022], val=[], test=[])
    paths = interim_to_processed(spec)
    train = pl.read_parquet(paths["train"])

    p_cols = [c for c in train.columns if c.startswith("p_")]
    t_cols = [c for c in train.columns if c.startswith("t_")]
    assert p_cols, "expected at least one p_* column in joined frame"
    assert t_cols, "expected at least one t_* column in joined frame"


def test_interim_to_processed_auto_builds_missing_feature_tables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If feature tables don't exist yet, interim_to_processed must call
    build_feature_tables itself rather than crash."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)
    # Sanity: features don't exist yet.
    assert not (interim / "2022" / etl.PLAYER_FEATURES_FILENAME).exists()

    spec = SplitSpec(train=[2022], val=[], test=[])
    paths = interim_to_processed(spec)
    # Features must now exist, AND the processed frame must be non-empty.
    assert (interim / "2022" / etl.PLAYER_FEATURES_FILENAME).exists()
    assert pl.read_parquet(paths["train"]).height > 0


def test_interim_to_processed_empty_split_still_writes_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty val or test split must still produce a file on disk so
    downstream code (DataLoader, evaluation) can rely on the path."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    spec = SplitSpec(train=[2022], val=[], test=[])
    paths = interim_to_processed(spec)
    assert paths["val"].exists()
    assert paths["test"].exists()
    assert pl.read_parquet(paths["val"]).is_empty()
    assert pl.read_parquet(paths["test"]).is_empty()


def test_processed_dir_respects_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Same env-var-redirection pattern as interim_dir."""
    custom = tmp_path / "custom_processed"
    monkeypatch.setenv("NBA_SIM_PROCESSED_DIR", str(custom))
    assert processed_dir() == custom.resolve()


# ---------------------------------------------------------------------------
# Matchup feature wiring — these guard the pipeline contract that
# build_feature_tables fans matchup columns into player_features.parquet
# and that interim_to_processed carries them through. Hand-computed
# correctness lives in test_features.py; here we check only flow-through.
# ---------------------------------------------------------------------------

# Columns the matchup module contributes to player_features. Pulled from
# the matchup function's public surface so the test moves in lockstep if
# the column set changes.
_MATCHUP_COLS = {
    "opp_team_id",
    "opp_def_rtg_10",
    "opp_pace_10",
    "h2h_last_meeting_margin",
    "opp_def_rtg_vs_pos",
}


def test_build_feature_tables_writes_matchup_columns_into_player_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """player_features.parquet must carry opp_*, h2h_*, opp_def_rtg_vs_pos
    after build_feature_tables runs. This is the wiring contract — if the
    matchup call gets dropped from build_feature_tables, this fails."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    build_feature_tables(2022)
    pf = pl.read_parquet(interim / "2022" / etl.PLAYER_FEATURES_FILENAME)
    missing = _MATCHUP_COLS - set(pf.columns)
    assert not missing, f"player_features missing matchup cols: {sorted(missing)}"


def test_build_feature_tables_matchup_values_present_after_first_game(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sanity check that the matchup values are *populated* (non-null) for
    games past the first one. We don't pin specific numbers here — the
    arithmetic is locked down in test_features.py — but we do verify the
    null/non-null pattern that signals the join actually happened."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    # 3 games so we have a non-trivial prior-history window.
    _seed_season_interim(interim / "2022", 2022, n_games=3)

    build_feature_tables(2022)
    pf = pl.read_parquet(interim / "2022" / etl.PLAYER_FEATURES_FILENAME)

    games = pl.read_parquet(interim / "2022" / etl.GAMES_FILENAME).sort("date")
    first_gid = games["game_id"][0]
    last_gid = games["game_id"][-1]

    # First game: opp rolling and h2h are both null/0 (no prior history).
    first = pf.filter(pl.col("game_id") == first_gid)
    assert first["opp_def_rtg_10"].is_null().all()
    assert (first["h2h_last_meeting_margin"] == 0.0).all()
    # opp_team_id is still populated — it's identity, not history.
    assert first["opp_team_id"].is_not_null().all()

    # Last game: opp_def_rtg_10 must be populated (opp has prior games now).
    last = pf.filter(pl.col("game_id") == last_gid)
    assert last["opp_def_rtg_10"].is_not_null().all()


def test_interim_to_processed_carries_matchup_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If build_feature_tables put matchup columns into player_features,
    interim_to_processed must preserve them through the join + concat +
    write to ``data/processed/<split>.parquet``."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    spec = SplitSpec(train=[2022], val=[], test=[])
    paths = interim_to_processed(spec)
    train = pl.read_parquet(paths["train"])
    missing = _MATCHUP_COLS - set(train.columns)
    assert not missing, f"processed train missing matchup cols: {sorted(missing)}"


# ---------------------------------------------------------------------------
# Context feature wiring — mirrors the matchup-wiring tests. Hand-computed
# values are in test_features.py; here we check only flow-through.
# ---------------------------------------------------------------------------

_CONTEXT_COLS = {
    "is_home", "rest_days", "b2b", "is_3in4", "is_4in6",
    "season_phase", "day_of_week", "month",
    "altitude_ft", "travel_miles_prev",
}


def test_build_feature_tables_writes_context_columns_into_player_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """player_features.parquet must carry the 10 context columns after
    build_feature_tables runs. Guards the wiring: if the context call
    gets dropped, this fails."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    build_feature_tables(2022)
    pf = pl.read_parquet(interim / "2022" / etl.PLAYER_FEATURES_FILENAME)
    missing = _CONTEXT_COLS - set(pf.columns)
    assert not missing, f"player_features missing context cols: {sorted(missing)}"


def test_build_feature_tables_context_values_populated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sanity that the join actually attached values, not just column
    headers. Pure per-row features (altitude, day_of_week, month,
    is_home, season_phase) must be populated on every row including
    the first game; shift-based features (rest_days, b2b, travel) are
    null only for the very first game of the season."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022, n_games=3)

    build_feature_tables(2022)
    pf = pl.read_parquet(interim / "2022" / etl.PLAYER_FEATURES_FILENAME)

    # Pure per-row features: no nulls anywhere.
    for col in ("altitude_ft", "day_of_week", "month", "is_home", "season_phase"):
        assert pf[col].is_not_null().all(), f"{col} has nulls — context join missed rows"

    # Shift-based features: not all null. We don't pin specific values
    # here (test_features.py does that); just confirm the join populated
    # at least the later games.
    assert pf["rest_days"].is_not_null().any()
    assert pf["travel_miles_prev"].is_not_null().any()


def test_interim_to_processed_carries_context_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """interim_to_processed must preserve context columns through the
    join + concat + write to data/processed/<split>.parquet."""
    interim, _ = _redirect_all_dirs(monkeypatch, tmp_path)
    _seed_season_interim(interim / "2022", 2022)

    spec = SplitSpec(train=[2022], val=[], test=[])
    paths = interim_to_processed(spec)
    train = pl.read_parquet(paths["train"])
    missing = _CONTEXT_COLS - set(train.columns)
    assert not missing, f"processed train missing context cols: {sorted(missing)}"
