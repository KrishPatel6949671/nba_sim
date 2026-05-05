"""Schema-layer tests.

Pure tests — no network, no fetcher. The point of the schema layer being
typed-and-validated is that we can prove its correctness from synthetic
rows that mirror V3's shape.

Covers:
    - ``parse_minutes_mmss`` edge cases (DNPs, normal, OT-length).
    - Pydantic validators on :class:`PlayerBoxLine` and :class:`TeamBoxLine`.
    - Raw layer tolerance for missing/unknown V3 fields.
    - ``from_raw`` conversions for starters, DNPs, and the pair-merge for Games.
    - Round-trip write/read of a :class:`PlayerBoxLine` through parquet.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from nba_sim.data.schema import (
    Game,
    PlayerBoxLine,
    RawGame,
    RawPlayerBoxLine,
    RawTeamBoxLine,
    TeamBoxLine,
    parse_minutes_mmss,
)


# ---------------------------------------------------------------------------
# Fixtures / synthetic V3 rows
# ---------------------------------------------------------------------------

def _valid_player_kwargs(**overrides: Any) -> dict[str, Any]:
    """Kwargs for a valid PlayerBoxLine. Override one field at a time to
    construct invariant-violating rows for negative tests."""
    base: dict[str, Any] = dict(
        game_id="0022300001",
        player_id=2544,
        player_name="LeBron James",
        team_id=1610612747,
        team_abbr="LAL",
        position="F",
        minutes=34.2,
        pts=27, fgm=10, fga=18,
        tpm=2, tpa=5,
        ftm=5, fta=6,
        oreb=1, dreb=7, reb=8,
        ast=9, stl=1, blk=0, tov=3, pf=2,
        plus_minus=5.0,
        is_starter=True, is_active=True, dnp=False,
    )
    base.update(overrides)
    return base


def _v3_starter_row() -> dict[str, Any]:
    return {
        "gameId": "0022300001",
        "teamId": 1610612747,
        "teamTricode": "LAL",
        "personId": 2544,
        "firstName": "LeBron",
        "familyName": "James",
        "position": "F",
        "comment": "",
        "minutes": "34:12",
        "fieldGoalsMade": 10,
        "fieldGoalsAttempted": 18,
        "threePointersMade": 2,
        "threePointersAttempted": 5,
        "freeThrowsMade": 5,
        "freeThrowsAttempted": 6,
        "reboundsOffensive": 1,
        "reboundsDefensive": 7,
        "reboundsTotal": 8,
        "assists": 9,
        "steals": 1,
        "blocks": 0,
        "turnovers": 3,
        "foulsPersonal": 2,
        "points": 27,
        "plusMinusPoints": 5.0,
    }


def _v3_dnp_row() -> dict[str, Any]:
    """A DNP row from V3 — empty position, empty minutes, all stats omitted."""
    return {
        "gameId": "0022300001",
        "teamId": 1610612747,
        "teamTricode": "LAL",
        "personId": 9999,
        "firstName": "Bench",
        "familyName": "Player",
        "position": "",
        "comment": "DNP - Coach's Decision",
        "minutes": "",
        # No counting-stat keys — Pydantic fills None for missing optional fields.
    }


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


# ---------------------------------------------------------------------------
# parse_minutes_mmss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("inp", "expected"),
    [
        ("", 0.0),
        (None, 0.0),
        ("0:00", 0.0),
        ("34:12", 34.2),
        ("0:30", 0.5),
        ("48:00", 48.0),
        ("53:00", 53.0),  # OT minutes
    ],
)
def test_minutes_mm_ss_parser(inp: str | None, expected: float) -> None:
    assert parse_minutes_mmss(inp) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Raw layer tolerance
# ---------------------------------------------------------------------------

def test_raw_tolerates_missing_fields() -> None:
    """RawPlayerBoxLine accepts a DNP row with all counting stats omitted."""
    raw = RawPlayerBoxLine.model_validate(_v3_dnp_row())
    assert raw.minutes == ""
    assert raw.fieldGoalsMade is None
    assert raw.points is None
    assert raw.position == ""
    assert raw.comment.startswith("DNP")


def test_raw_ignores_unknown_fields() -> None:
    """V3 may add columns over time — extra='ignore' must silently drop them."""
    row = _v3_starter_row()
    row["someFutureV3Field"] = 42
    raw = RawPlayerBoxLine.model_validate(row)
    assert not hasattr(raw, "someFutureV3Field")


# ---------------------------------------------------------------------------
# PlayerBoxLine validators
# ---------------------------------------------------------------------------

def test_player_box_rejects_fgm_exceeding_fga() -> None:
    with pytest.raises(ValidationError, match="fgm"):
        PlayerBoxLine(**_valid_player_kwargs(fgm=15, fga=10))


def test_player_box_rejects_tpm_exceeding_fgm() -> None:
    """3-pointers are a strict subset of FGs."""
    with pytest.raises(ValidationError, match="tpm"):
        PlayerBoxLine(**_valid_player_kwargs(tpm=5, fgm=3))


def test_player_box_rejects_inconsistent_rebound_split() -> None:
    with pytest.raises(ValidationError, match="reb"):
        PlayerBoxLine(**_valid_player_kwargs(oreb=2, dreb=3, reb=10))


def test_dnp_implies_zero_minutes() -> None:
    """The dnp flag must agree with minutes; mismatches in either direction reject."""
    with pytest.raises(ValidationError, match="dnp"):
        PlayerBoxLine(**_valid_player_kwargs(minutes=0.0, fgm=0, fga=0,
                                              tpm=0, tpa=0, ftm=0, fta=0,
                                              oreb=0, dreb=0, reb=0,
                                              ast=0, stl=0, blk=0, tov=0, pf=0,
                                              pts=0, dnp=False))
    with pytest.raises(ValidationError, match="dnp"):
        PlayerBoxLine(**_valid_player_kwargs(minutes=10.0, dnp=True))


# ---------------------------------------------------------------------------
# PlayerBoxLine.from_raw
# ---------------------------------------------------------------------------

def test_player_from_raw_starter() -> None:
    raw = RawPlayerBoxLine.model_validate(_v3_starter_row())
    pbl = PlayerBoxLine.from_raw(raw)

    assert pbl.game_id == "0022300001"  # leading zeros preserved
    assert pbl.player_id == 2544
    assert pbl.player_name == "LeBron James"
    assert pbl.team_abbr == "LAL"
    assert pbl.position == "F"
    assert pbl.minutes == pytest.approx(34.2)
    assert pbl.is_starter is True
    assert pbl.dnp is False
    assert pbl.pts == 27
    assert pbl.reb == 8


def test_player_from_raw_dnp() -> None:
    """A DNP must come out with zeros across the board and the right flags."""
    raw = RawPlayerBoxLine.model_validate(_v3_dnp_row())
    pbl = PlayerBoxLine.from_raw(raw)

    assert pbl.minutes == 0.0
    assert pbl.dnp is True
    assert pbl.is_starter is False  # empty position string
    # Every counting stat coerced from None to 0.
    assert pbl.pts == 0
    assert pbl.fgm == 0
    assert pbl.fga == 0
    assert pbl.reb == 0
    assert pbl.plus_minus == 0.0


# ---------------------------------------------------------------------------
# TeamBoxLine
# ---------------------------------------------------------------------------

def test_team_box_from_raw_validates() -> None:
    raw = RawTeamBoxLine.model_validate({
        "gameId": "0022300001",
        "teamId": 1610612747,
        "teamTricode": "LAL",
        "minutes": "240:00",
        "fieldGoalsMade": 40, "fieldGoalsAttempted": 88,
        "threePointersMade": 12, "threePointersAttempted": 35,
        "freeThrowsMade": 18, "freeThrowsAttempted": 22,
        "reboundsOffensive": 10, "reboundsDefensive": 32, "reboundsTotal": 42,
        "assists": 25, "steals": 7, "blocks": 4, "turnovers": 13,
        "foulsPersonal": 19, "points": 110, "plusMinusPoints": 4.0,
    })
    tbl = TeamBoxLine.from_raw(raw, is_home=True)
    assert tbl.is_home is True
    assert tbl.minutes == 240.0
    assert tbl.reb == tbl.oreb + tbl.dreb
    # Advanced fields are filled later by an ETL merge.
    assert tbl.pace is None
    assert tbl.off_rtg is None


def test_team_box_rejects_inconsistent_rebound_split() -> None:
    raw = RawTeamBoxLine.model_validate({
        "gameId": "0022300001",
        "teamId": 1, "teamTricode": "LAL", "minutes": "240:00",
        "fieldGoalsMade": 40, "fieldGoalsAttempted": 88,
        "threePointersMade": 12, "threePointersAttempted": 35,
        "freeThrowsMade": 18, "freeThrowsAttempted": 22,
        "reboundsOffensive": 10, "reboundsDefensive": 32, "reboundsTotal": 99,  # bad
        "assists": 25, "steals": 7, "blocks": 4, "turnovers": 13,
        "foulsPersonal": 19, "points": 110, "plusMinusPoints": 4.0,
    })
    with pytest.raises(ValidationError, match="reb"):
        TeamBoxLine.from_raw(raw, is_home=True)


# ---------------------------------------------------------------------------
# Game.from_raw_pair
# ---------------------------------------------------------------------------

def test_game_from_raw_pair_home_away_split() -> None:
    home = RawGame.model_validate(_v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True))
    away = RawGame.model_validate(_v3_game_row("LAL", "BOS", 2, "0022300001", is_home=False))
    g = Game.from_raw_pair([home, away])

    assert g.game_id == "0022300001"
    assert g.home_team_abbr == "BOS"
    assert g.away_team_abbr == "LAL"
    assert g.home_team_id == 1
    assert g.away_team_id == 2
    assert g.season == 2023
    assert g.is_playoffs is False
    assert g.is_overtime is False


def test_game_from_raw_pair_handles_arg_order() -> None:
    """Result should be invariant to which row is listed first."""
    home = RawGame.model_validate(_v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True))
    away = RawGame.model_validate(_v3_game_row("LAL", "BOS", 2, "0022300001", is_home=False))
    g_swap = Game.from_raw_pair([away, home])
    assert g_swap.home_team_abbr == "BOS"
    assert g_swap.away_team_abbr == "LAL"


def test_game_from_raw_pair_detects_playoffs() -> None:
    home = RawGame.model_validate(_v3_game_row(
        "BOS", "LAL", 1, "0042300001", is_home=True, season_id="42023",
    ))
    away = RawGame.model_validate(_v3_game_row(
        "LAL", "BOS", 2, "0042300001", is_home=False, season_id="42023",
    ))
    g = Game.from_raw_pair([home, away])
    assert g.is_playoffs is True


def test_game_from_raw_pair_detects_overtime() -> None:
    home = RawGame.model_validate(_v3_game_row(
        "BOS", "LAL", 1, "0022300001", is_home=True, mins=265,  # 240 + 1 OT
    ))
    away = RawGame.model_validate(_v3_game_row(
        "LAL", "BOS", 2, "0022300001", is_home=False, mins=265,
    ))
    g = Game.from_raw_pair([home, away])
    assert g.is_overtime is True


def test_game_from_raw_pair_rejects_mismatched_ids() -> None:
    home = RawGame.model_validate(_v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True))
    away = RawGame.model_validate(_v3_game_row("LAL", "MIA", 2, "0022300002", is_home=False))
    with pytest.raises(ValueError, match="Mismatched"):
        Game.from_raw_pair([home, away])


def test_game_from_raw_pair_rejects_wrong_count() -> None:
    home = RawGame.model_validate(_v3_game_row("BOS", "LAL", 1, "0022300001", is_home=True))
    with pytest.raises(ValueError, match="Expected 2"):
        Game.from_raw_pair([home])


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------

def test_interim_parquet_round_trip(tmp_path) -> None:
    """Writing a PlayerBoxLine to parquet and reading it back must preserve
    every field — including string IDs (no leading-zero loss), bool flags,
    and floats."""
    raw = RawPlayerBoxLine.model_validate(_v3_starter_row())
    original = PlayerBoxLine.from_raw(raw)

    df = pl.DataFrame([original.model_dump()])
    path = tmp_path / "row.parquet"
    df.write_parquet(path)

    df_back = pl.read_parquet(path)
    rt = PlayerBoxLine.model_validate(df_back.row(0, named=True))
    assert rt == original
