"""Feature-engineering tests.

Covers the implemented surface of ``features/rolling.py``:

    - ``player_rolling``: shape, correctness against hand-computed values,
      leakage safety (first row null, cross-player independence, dropped
      games excluded), optional USG% via ``team_box``, season-reset of
      game-count counter while still letting form cross seasons.
    - ``team_rolling``: rolling means, win-pct derivation from plus-minus,
      points-allowed derivation.
    - ``season_to_date``: prior-only cumulative averages, season reset.

The context/matchup stubs remain ``pytest.skip`` until those modules land
(PLAN.md §3.1 Phase 1).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl
import pytest

from nba_sim.features.rolling import (
    PLAYER_WINDOWS,
    TEAM_WINDOWS,
    player_rolling,
    season_to_date,
    team_rolling,
)


# ---------------------------------------------------------------------------
# Synthetic builders. Interim-grain rows, not V3 raw — these mirror the
# Pydantic interim schemas in :mod:`nba_sim.data.schema`. Kept terse and
# parameterized so each test can override just the field(s) it cares about.
# ---------------------------------------------------------------------------

def _game(gid: str, date: dt.date, season: int, *, dropped: bool = False) -> dict[str, Any]:
    return {"game_id": gid, "date": date, "season": season, "dropped": dropped}


def _games_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _pbox_row(
    gid: str,
    pid: int,
    tid: int,
    *,
    mn: float = 30.0,
    pts: int = 20,
    fgm: int = 8,
    fga: int = 16,
    tpm: int = 2,
    tpa: int = 5,
    ftm: int = 2,
    fta: int = 4,
    oreb: int = 1,
    dreb: int = 5,
    reb: int = 6,
    ast: int = 4,
    stl: int = 1,
    blk: int = 0,
    tov: int = 2,
    pf: int = 3,
) -> dict[str, Any]:
    return {
        "game_id": gid, "player_id": pid, "team_id": tid,
        "minutes": float(mn), "pts": pts,
        "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
        "oreb": oreb, "dreb": dreb, "reb": reb,
        "ast": ast, "stl": stl, "blk": blk, "tov": tov, "pf": pf,
    }


def _tbox_row(
    gid: str,
    tid: int,
    *,
    mn: float = 240.0,
    pts: int = 110,
    plus_minus: float = 5.0,
    pace: float = 100.0,
    off_rtg: float = 110.0,
    def_rtg: float = 105.0,
    fga: int = 85,
    fta: int = 22,
    tov: int = 14,
) -> dict[str, Any]:
    return {
        "game_id": gid, "team_id": tid, "minutes": float(mn), "pts": pts,
        "plus_minus": plus_minus, "pace": pace, "off_rtg": off_rtg, "def_rtg": def_rtg,
        "fga": fga, "fta": fta, "tov": tov,
    }


def _three_game_player_setup() -> tuple[pl.DataFrame, pl.DataFrame]:
    """One player, one team, three games in one season.

    Used by several correctness tests so their hand-computed expectations
    share a fixture.
    """
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
        _game("G3", dt.date(2023, 1, 5), 2022),
    ])
    # Minutes 30/24/18, PTS 20/16/10, FGA 15/12/8, etc.
    player_box = pl.DataFrame([
        _pbox_row("G1", 100, 1, mn=30, pts=20, fga=15, fta=4, tov=3, reb=5, ast=5, stl=2, blk=1, pf=2,
                  fgm=8, tpm=2, tpa=5, ftm=2, oreb=1, dreb=4),
        _pbox_row("G2", 100, 1, mn=24, pts=16, fga=12, fta=2, tov=2, reb=3, ast=3, stl=1, blk=0, pf=1,
                  fgm=7, tpm=2, tpa=4, ftm=0, oreb=0, dreb=3),
        _pbox_row("G3", 100, 1, mn=18, pts=10, fga=8,  fta=3, tov=1, reb=3, ast=2, stl=0, blk=0, pf=2,
                  fgm=5, tpm=0, tpa=2, ftm=0, oreb=1, dreb=2),
    ])
    return games, player_box


# ===========================================================================
# player_rolling
# ===========================================================================

def test_player_rolling_returns_empty_for_empty_input() -> None:
    games = _games_df([])
    out = player_rolling(pl.DataFrame(), games)
    assert out.is_empty()


def test_player_rolling_shape_without_team_box() -> None:
    """Without team_box, USG% columns must be absent."""
    games, pb = _three_game_player_setup()
    out = player_rolling(pb, games)

    expected_min_avg = {f"p_min_avg_{w}" for w in PLAYER_WINDOWS}
    expected_ts = {f"p_ts_{w}" for w in PLAYER_WINDOWS}
    expected_pts_pm = {f"p_pts_per_min_{w}" for w in PLAYER_WINDOWS}
    per_min_stats = ("fga", "tpa", "fta", "reb", "ast", "stl", "blk", "tov", "pf")
    expected_per_min = {f"p_{s}_per_min_10" for s in per_min_stats}

    cols = set(out.columns)
    assert expected_min_avg <= cols
    assert expected_ts <= cols
    assert expected_pts_pm <= cols
    assert expected_per_min <= cols
    assert "p_games_played_season" in cols
    # USG% columns should NOT be present without team_box.
    assert not any(c.startswith("p_usage_avg_") for c in cols)


def test_player_rolling_shape_with_team_box() -> None:
    """With team_box, USG% columns appear for each player window."""
    games, pb = _three_game_player_setup()
    tb = pl.DataFrame([
        _tbox_row("G1", 1), _tbox_row("G2", 1), _tbox_row("G3", 1),
    ])
    out = player_rolling(pb, games, team_box=tb)
    for w in PLAYER_WINDOWS:
        assert f"p_usage_avg_{w}" in out.columns


def test_player_rolling_first_game_is_null() -> None:
    """A player's first game has no prior history — every rolling output
    must be null. This is the primary leakage guarantee."""
    games, pb = _three_game_player_setup()
    out = player_rolling(pb, games).sort("date")
    g1 = out.filter(pl.col("game_id") == "G1")
    # Every rolling column should be null on the first row.
    rolling_cols = [
        c for c in out.columns
        if c.startswith(("p_min_avg_", "p_ts_", "p_pts_per_min_", "p_games_played_season"))
        or (c.startswith("p_") and c.endswith("_per_min_10"))
    ]
    rolling_cols.remove("p_games_played_season")  # this is 0, not null
    for c in rolling_cols:
        assert g1[c][0] is None, f"{c} should be null on first game, got {g1[c][0]}"
    assert g1["p_games_played_season"][0] == 0


def test_player_rolling_second_game_uses_only_first() -> None:
    """G2's rolling values must match G1's stats exactly — single prior game."""
    games, pb = _three_game_player_setup()
    out = player_rolling(pb, games).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    # G1: minutes=30, pts=20, fga=15, fta=4.
    assert g2["p_min_avg_5"] == 30.0
    assert g2["p_pts_per_min_5"] == pytest.approx(20.0 / 30.0)
    # TS% over G1: 20 / (2 * (15 + 0.44*4)) = 20 / (2 * 16.76) = 0.5966...
    assert g2["p_ts_5"] == pytest.approx(20.0 / (2 * (15 + 0.44 * 4)))
    # FGA/min over window=10: 15 / 30 = 0.5.
    assert g2["p_fga_per_min_10"] == pytest.approx(0.5)
    assert g2["p_games_played_season"] == 1


def test_player_rolling_third_game_pooled_correctly() -> None:
    """G3 rolling uses G1+G2 pooled (sum/sum), not mean-of-ratios."""
    games, pb = _three_game_player_setup()
    out = player_rolling(pb, games).sort("date")
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)
    # Mean minutes = (30 + 24) / 2 = 27.
    assert g3["p_min_avg_5"] == pytest.approx(27.0)
    # Pooled pts/min: (20 + 16) / (30 + 24) = 36 / 54 = 0.6667.
    assert g3["p_pts_per_min_5"] == pytest.approx(36.0 / 54.0)
    # Pooled TS%: (20 + 16) / (2 * ((15 + 0.44*4) + (12 + 0.44*2))).
    denom = 2 * ((15 + 0.44 * 4) + (12 + 0.44 * 2))
    assert g3["p_ts_5"] == pytest.approx(36.0 / denom)
    assert g3["p_games_played_season"] == 2


def test_player_rolling_pooled_not_mean_of_ratios() -> None:
    """Stress the distinction with one big and one tiny game.

    Mean-of-ratios over (20/30, 0/1) = (0.667 + 0) / 2 = 0.333.
    Pooled                          = (20 + 0) / (30 + 1) = 0.645.
    """
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
        _game("G3", dt.date(2023, 1, 5), 2022),
    ])
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1, mn=30, pts=20),
        _pbox_row("G2", 100, 1, mn=1, pts=0, fga=0, fgm=0, tpm=0, tpa=0,
                  ftm=0, fta=0, oreb=0, dreb=0, reb=0, ast=0, stl=0,
                  blk=0, tov=0, pf=0),
        _pbox_row("G3", 100, 1),
    ])
    out = player_rolling(pb, games).sort("date")
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)
    # Pooled, not 0.333.
    assert g3["p_pts_per_min_5"] == pytest.approx(20.0 / 31.0)


def test_player_rolling_dnp_keeps_window_slot_with_zero_minutes() -> None:
    """A DNP (mn=0, all counts=0) should still occupy a window slot — the
    "minutes played in last N games" feature must reflect the benching."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
        _game("G3", dt.date(2023, 1, 5), 2022),
    ])
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1, mn=30, pts=20),
        _pbox_row("G2", 100, 1, mn=0, pts=0, fga=0, fgm=0, tpm=0, tpa=0,
                  ftm=0, fta=0, oreb=0, dreb=0, reb=0, ast=0, stl=0,
                  blk=0, tov=0, pf=0),
        _pbox_row("G3", 100, 1, mn=30, pts=20),
    ])
    out = player_rolling(pb, games).sort("date")
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)
    # Two prior games: 30 and 0 minutes => avg 15.
    assert g3["p_min_avg_5"] == pytest.approx(15.0)
    # Pooled pts/min: 20 / 30. The DNP contributes 0 to both sums.
    assert g3["p_pts_per_min_5"] == pytest.approx(20.0 / 30.0)


def test_player_rolling_per_minute_returns_none_when_window_has_zero_minutes() -> None:
    """If every prior game in the window was a DNP, sum(minutes) == 0 and
    we emit None rather than NaN (explicit Polars when/then guard)."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
    ])
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1, mn=0, pts=0, fga=0, fgm=0, tpm=0, tpa=0,
                  ftm=0, fta=0, oreb=0, dreb=0, reb=0, ast=0, stl=0,
                  blk=0, tov=0, pf=0),
        _pbox_row("G2", 100, 1, mn=20, pts=10),
    ])
    out = player_rolling(pb, games).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    assert g2["p_pts_per_min_5"] is None
    assert g2["p_ts_5"] is None  # 0 attempts in prior game


def test_player_rolling_isolates_players() -> None:
    """Player A's history must not leak into Player B's rolling window."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
        _game("G3", dt.date(2023, 1, 5), 2022),
    ])
    pb = pl.DataFrame([
        # Player A: high-minute games G1, G2, G3.
        _pbox_row("G1", 100, 1, mn=40, pts=30),
        _pbox_row("G2", 100, 1, mn=40, pts=30),
        _pbox_row("G3", 100, 1, mn=40, pts=30),
        # Player B: first appearance is G3.
        _pbox_row("G3", 200, 1, mn=10, pts=4),
    ])
    out = player_rolling(pb, games).sort("date")
    b_g3 = out.filter((pl.col("game_id") == "G3") & (pl.col("player_id") == 200)).row(0, named=True)
    # Player B's first game must be null — A's history cannot leak.
    assert b_g3["p_min_avg_5"] is None
    assert b_g3["p_games_played_season"] == 0


def test_player_rolling_excludes_dropped_games() -> None:
    """A game flagged ``dropped=True`` must not appear in the output frame,
    and must not contribute to any other game's rolling window."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G_BAD", dt.date(2023, 1, 3), 2022, dropped=True),
        _game("G3", dt.date(2023, 1, 5), 2022),
    ])
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1, mn=30, pts=20),
        _pbox_row("G_BAD", 100, 1, mn=200, pts=999),  # outlier; must be ignored
        _pbox_row("G3", 100, 1, mn=20, pts=10),
    ])
    out = player_rolling(pb, games).sort("date")
    # Dropped row must be absent.
    assert "G_BAD" not in out["game_id"].to_list()
    # G3's rolling must reflect only G1, not the bogus G_BAD value.
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)
    assert g3["p_min_avg_5"] == pytest.approx(30.0)
    assert g3["p_pts_per_min_5"] == pytest.approx(20.0 / 30.0)


def test_player_rolling_games_played_resets_each_season() -> None:
    """``p_games_played_season`` must reset across season boundaries even
    though the rolling window may legitimately cross them."""
    games = _games_df([
        _game("S1G1", dt.date(2022, 12, 1), 2022),
        _game("S1G2", dt.date(2022, 12, 3), 2022),
        _game("S2G1", dt.date(2023, 10, 1), 2023),
    ])
    pb = pl.DataFrame([
        _pbox_row("S1G1", 100, 1, mn=30, pts=20),
        _pbox_row("S1G2", 100, 1, mn=24, pts=18),
        _pbox_row("S2G1", 100, 1, mn=28, pts=22),
    ])
    out = player_rolling(pb, games).sort("date")
    s2 = out.filter(pl.col("game_id") == "S2G1").row(0, named=True)
    # Counter reset.
    assert s2["p_games_played_season"] == 0
    # But the rolling window crosses seasons (intentional, per the design).
    assert s2["p_min_avg_5"] == pytest.approx((30 + 24) / 2)


def test_player_rolling_usage_uses_canonical_bref_scale() -> None:
    """USG% with one prior game and known team totals should equal the
    Basketball-Reference formula to within float precision. Catches the
    "missing /5" bug from the initial smoke test."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
    ])
    pb = pl.DataFrame([
        # Player: mn=30, FGA=15, FTA=4, TOV=3.
        _pbox_row("G1", 100, 1, mn=30, fga=15, fta=4, tov=3),
        _pbox_row("G2", 100, 1, mn=30, fga=15, fta=4, tov=3),
    ])
    tb = pl.DataFrame([
        _tbox_row("G1", 1, mn=240.0, fga=85, fta=22, tov=14),
        _tbox_row("G2", 1, mn=240.0, fga=85, fta=22, tov=14),
    ])
    out = player_rolling(pb, games, team_box=tb).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    # USG% G1 = 100 * (15 + 0.44*4 + 3) * (240/5) / (30 * (85 + 0.44*22 + 14))
    expected = 100.0 * (15 + 0.44 * 4 + 3) * (240 / 5) / (30 * (85 + 0.44 * 22 + 14))
    assert g2["p_usage_avg_5"] == pytest.approx(expected)
    # Sanity: in canonical range (0–100).
    assert 0 < g2["p_usage_avg_5"] < 100


# ===========================================================================
# team_rolling
# ===========================================================================

def test_team_rolling_returns_empty_for_empty_input() -> None:
    out = team_rolling(pl.DataFrame(), _games_df([]))
    assert out.is_empty()


def test_team_rolling_first_game_is_null() -> None:
    games = _games_df([_game("G1", dt.date(2023, 1, 1), 2022)])
    tb = pl.DataFrame([_tbox_row("G1", 1)])
    out = team_rolling(tb, games)
    row = out.row(0, named=True)
    for c in ("t_pace_5", "t_off_rtg_5", "t_def_rtg_5",
              "t_win_pct_10", "t_pts_avg_10", "t_pts_allowed_10"):
        assert row[c] is None, f"{c} should be null on first game"


def test_team_rolling_second_game_uses_only_first() -> None:
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
    ])
    tb = pl.DataFrame([
        _tbox_row("G1", 1, pace=100.0, off_rtg=110.0, def_rtg=105.0,
                  pts=110, plus_minus=5.0),
        _tbox_row("G2", 1, pace=102.0, off_rtg=108.0, def_rtg=111.0,
                  pts=105, plus_minus=-3.0),
    ])
    out = team_rolling(tb, games).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    assert g2["t_pace_5"] == pytest.approx(100.0)
    assert g2["t_off_rtg_5"] == pytest.approx(110.0)
    assert g2["t_def_rtg_5"] == pytest.approx(105.0)
    # Win pct: G1 was a win (plus_minus > 0).
    assert g2["t_win_pct_10"] == pytest.approx(1.0)
    # pts_avg = 110, pts_allowed = 110 - 5 = 105.
    assert g2["t_pts_avg_10"] == pytest.approx(110.0)
    assert g2["t_pts_allowed_10"] == pytest.approx(105.0)


def test_team_rolling_win_pct_pooled_from_plus_minus() -> None:
    """t_win_pct_10 must equal (wins so far) / (games so far) — pooled from
    the boolean (plus_minus > 0) signal."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
        _game("G3", dt.date(2023, 1, 5), 2022),
        _game("G4", dt.date(2023, 1, 7), 2022),
    ])
    tb = pl.DataFrame([
        _tbox_row("G1", 1, plus_minus=5.0),    # win
        _tbox_row("G2", 1, plus_minus=-3.0),   # loss
        _tbox_row("G3", 1, plus_minus=10.0),   # win
        _tbox_row("G4", 1, plus_minus=-1.0),   # loss (irrelevant; G4 is the target)
    ])
    out = team_rolling(tb, games).sort("date")
    g4 = out.filter(pl.col("game_id") == "G4").row(0, named=True)
    # Prior to G4: 2 wins / 3 games.
    assert g4["t_win_pct_10"] == pytest.approx(2 / 3)


def test_team_rolling_pts_allowed_derived_from_plus_minus() -> None:
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
    ])
    tb = pl.DataFrame([
        _tbox_row("G1", 1, pts=120, plus_minus=15.0),  # opp scored 105
        _tbox_row("G2", 1),
    ])
    out = team_rolling(tb, games).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    assert g2["t_pts_allowed_10"] == pytest.approx(105.0)


def test_team_rolling_isolates_teams() -> None:
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
    ])
    tb = pl.DataFrame([
        _tbox_row("G1", 1, pace=100.0),
        _tbox_row("G1", 2, pace=200.0),   # extreme outlier on team 2
        _tbox_row("G2", 1, pace=100.0),
    ])
    out = team_rolling(tb, games).sort(["team_id", "date"])
    t1_g2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G2")).row(0, named=True)
    # Team 1's G2 rolling must NOT pull team 2's outlier.
    assert t1_g2["t_pace_5"] == pytest.approx(100.0)


# ===========================================================================
# season_to_date
# ===========================================================================

def test_season_to_date_returns_empty_for_empty_input() -> None:
    out = season_to_date(pl.DataFrame(), _games_df([]))
    assert out.is_empty()


def test_season_to_date_first_game_has_zero_games_and_null_avgs() -> None:
    games, pb = _three_game_player_setup()
    out = season_to_date(pb, games).sort("date")
    g1 = out.filter(pl.col("game_id") == "G1").row(0, named=True)
    assert g1["std_games"] == 0
    # Every avg column should be null on the first game.
    avg_cols = [c for c in out.columns if c.startswith("std_") and c.endswith("_avg")]
    assert avg_cols, "expected at least one std_*_avg column"
    for c in avg_cols:
        assert g1[c] is None, f"{c} should be null on first game"


def test_season_to_date_second_game_equals_first() -> None:
    games, pb = _three_game_player_setup()
    out = season_to_date(pb, games).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    assert g2["std_games"] == 1
    assert g2["std_minutes_avg"] == pytest.approx(30.0)
    assert g2["std_pts_avg"] == pytest.approx(20.0)
    assert g2["std_fga_avg"] == pytest.approx(15.0)


def test_season_to_date_third_game_averages_first_two() -> None:
    games, pb = _three_game_player_setup()
    out = season_to_date(pb, games).sort("date")
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)
    assert g3["std_games"] == 2
    assert g3["std_minutes_avg"] == pytest.approx((30 + 24) / 2)
    assert g3["std_pts_avg"] == pytest.approx((20 + 16) / 2)


def test_season_to_date_resets_across_seasons() -> None:
    games = _games_df([
        _game("S1G1", dt.date(2022, 12, 1), 2022),
        _game("S1G2", dt.date(2022, 12, 3), 2022),
        _game("S2G1", dt.date(2023, 10, 1), 2023),
    ])
    pb = pl.DataFrame([
        _pbox_row("S1G1", 100, 1, mn=30, pts=20),
        _pbox_row("S1G2", 100, 1, mn=24, pts=18),
        _pbox_row("S2G1", 100, 1, mn=28, pts=22),
    ])
    out = season_to_date(pb, games).sort("date")
    s2 = out.filter(pl.col("game_id") == "S2G1").row(0, named=True)
    # std_* aggregates partition on (player_id, season): season 2023 starts fresh.
    assert s2["std_games"] == 0
    assert s2["std_pts_avg"] is None


# ===========================================================================
# Leakage property (small, exhaustive)
# ===========================================================================

def test_no_future_date_contributes_to_any_rolling_value() -> None:
    """For every row R and every prior game P that contributes to R's
    rolling stats, P.date < R.date. We verify by reconstructing the
    rolling means from scratch and showing they match the implementation
    when computed over strict prior-date windows."""
    games = _games_df([
        _game("G1", dt.date(2023, 1, 1), 2022),
        _game("G2", dt.date(2023, 1, 3), 2022),
        _game("G3", dt.date(2023, 1, 5), 2022),
        _game("G4", dt.date(2023, 1, 7), 2022),
    ])
    minutes = [30, 24, 18, 22]
    pb = pl.DataFrame([
        _pbox_row(f"G{i+1}", 100, 1, mn=m) for i, m in enumerate(minutes)
    ])
    out = player_rolling(pb, games, windows=(5,)).sort("date")
    actual = out["p_min_avg_5"].to_list()

    # Independent reference: at row i, the rolling-mean-of-N=5 over priors
    # is mean(minutes[0:i]) if i > 0 else None.
    expected = [None] + [sum(minutes[:i]) / i for i in range(1, len(minutes))]
    for a, e in zip(actual, expected, strict=True):
        if e is None:
            assert a is None
        else:
            assert a == pytest.approx(e)


# ===========================================================================
# Not-yet-implemented features (context / matchup).
# ===========================================================================

def test_context_rest_days_computed_correctly() -> None:
    pytest.skip("features/context.py not implemented")


def test_b2b_flag_correct_on_consecutive_dates() -> None:
    pytest.skip("features/context.py not implemented")


def test_opp_defrtg_uses_opponent_prior_games_only() -> None:
    pytest.skip("features/matchup.py not implemented")


def test_cold_start_player_feature_is_nan_filled() -> None:
    pytest.skip("cold-start fill handled in §7 simulator, not in rolling")
