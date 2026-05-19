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
