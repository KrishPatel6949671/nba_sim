"""Feature-engineering tests.

Covers ``features/rolling.py``, ``features/matchup.py``, and
``features/context.py``:

    - ``player_rolling`` / ``team_rolling`` / ``season_to_date``:
      shape, correctness against hand-computed values, leakage safety
      (first row null, cross-player and cross-team independence, dropped
      games excluded), optional USG% via ``team_box``, season-reset of
      counters while still letting rolling form cross seasons.
    - ``head_to_head_last_margin``: first-meeting 0; sign-anchoring when
      venues swap; uses *most recent* prior meeting not first; isolated
      across team pairs; dropped games excluded.
    - ``opponent_defrtg_by_position``: first-game null; cumulative-prior
      values; position="" excluded; season reset.
    - ``add_matchup_features``: join shape, opp_team_id correctness,
      h2h null-fill to 0, defrtg_vs_pos optional, dropped-game rows
      filtered.
    - ``arena_altitude`` / ``travel_distance_miles`` / scalar haversine.
    - ``add_context_features``: per-(team, game) fan-out, rest/b2b/density,
      season_phase (playoffs override), day-of-week zero-indexing,
      altitude/travel joins, cross-season reset, dropped-game filter.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import polars as pl
import pytest

from nba_sim.features.context import (
    _haversine_miles,
    add_context_features,
    arena_altitude,
    travel_distance_miles,
)
from nba_sim.features.matchup import (
    add_matchup_features,
    head_to_head_last_margin,
    opponent_blocks_allowed_by_position,
    opponent_defrtg_by_position,
)
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
    expected_career = {
        "p_fg_pct_career", "p_tp_pct_career", "p_ft_pct_career",
        "p_blk_per36_career",
    }

    cols = set(out.columns)
    assert expected_min_avg <= cols
    assert expected_ts <= cols
    assert expected_pts_pm <= cols
    assert expected_per_min <= cols
    assert expected_career <= cols
    assert "p_ft_rate_10" in cols
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
        or c.endswith("_career")
        or c == "p_ft_rate_10"
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


def test_player_rolling_career_priors_and_ft_rate() -> None:
    """Career-pooled shooting priors and FT-rate-10 use the same
    cum_sum() - current trick as season_to_date but without season reset.

    G1: fgm=8/fga=15, tpm=2/tpa=5, ftm=2/fta=4, blk=1, min=30.
    G2: fgm=7/fga=12, tpm=2/tpa=4, ftm=0/fta=2, blk=0, min=24.

    At G2 (1 prior game): all priors reflect G1 alone.
    At G3 (2 prior games): all priors reflect G1+G2 pooled.
    """
    games, pb = _three_game_player_setup()
    out = player_rolling(pb, games).sort("date")
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)

    # G2: priors == G1 stats.
    assert g2["p_fg_pct_career"] == pytest.approx(8 / 15)
    assert g2["p_tp_pct_career"] == pytest.approx(2 / 5)
    assert g2["p_ft_pct_career"] == pytest.approx(2 / 4)
    assert g2["p_blk_per36_career"] == pytest.approx(36 * 1 / 30)
    assert g2["p_ft_rate_10"] == pytest.approx(4 / 15)

    # G3: priors == G1+G2 pooled (sum / sum, not mean of ratios).
    assert g3["p_fg_pct_career"] == pytest.approx((8 + 7) / (15 + 12))
    assert g3["p_tp_pct_career"] == pytest.approx((2 + 2) / (5 + 4))
    assert g3["p_ft_pct_career"] == pytest.approx((2 + 0) / (4 + 2))
    assert g3["p_blk_per36_career"] == pytest.approx(36 * (1 + 0) / (30 + 24))
    assert g3["p_ft_rate_10"] == pytest.approx((4 + 2) / (15 + 12))


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
# Matchup helpers and shared fixtures
# ===========================================================================

def _full_game(
    gid: str,
    date: dt.date,
    season: int,
    *,
    home_team_id: int,
    away_team_id: int,
    home_pts: int,
    away_pts: int,
    home_team_abbr: str = "BOS",
    away_team_abbr: str = "LAL",
    is_playoffs: bool = False,
    dropped: bool = False,
) -> dict[str, Any]:
    """Game row carrying the home/away team IDs, abbreviations, points,
    and playoff flag that matchup *and* context features read. ``_game()``
    (used by rolling tests) omits these — kept separate so the rolling-only
    tests stay terse.

    Default abbreviations (BOS/LAL) and ``is_playoffs=False`` mean the
    pre-existing matchup tests that don't pass these kwargs continue to
    work — the extra columns are simply ignored by matchup functions.
    """
    return {
        "game_id": gid, "date": date, "season": season,
        "home_team_id": home_team_id, "away_team_id": away_team_id,
        "home_team_abbr": home_team_abbr, "away_team_abbr": away_team_abbr,
        "home_pts": home_pts, "away_pts": away_pts,
        "is_playoffs": is_playoffs,
        "dropped": dropped,
    }


# ===========================================================================
# head_to_head_last_margin
# ===========================================================================

def test_h2h_returns_empty_for_empty_input() -> None:
    out = head_to_head_last_margin(pl.DataFrame())
    assert out.is_empty()


def test_h2h_first_meeting_is_zero() -> None:
    """No prior meeting → margin must be 0 (PLAN §3.1 contract)."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
    ])
    out = head_to_head_last_margin(games)
    assert out["h2h_last_meeting_margin"][0] == 0.0


def test_h2h_second_meeting_same_venue_preserves_sign() -> None:
    """BOS hosted both meetings. Prior margin from BOS POV = +10. Current
    home is still BOS, so the value should remain +10."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 2, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=112, away_pts=108),
    ])
    out = head_to_head_last_margin(games)
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    assert g2["h2h_last_meeting_margin"] == 10.0


def test_h2h_second_meeting_swapped_venue_flips_sign() -> None:
    """Venue swap is the key correctness test: the prior margin's sign
    must flip to reflect the *current* home team's POV.

    G1: BOS home, BOS won by 10 (BOS's margin = +10).
    G2: LAL home — now we're asking from LAL's perspective, so margin = -10.
    Naive "prev_home_pts - prev_away_pts" would emit +10 (wrong sign).
    """
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 2, 1), 2022,
                   home_team_id=2, away_team_id=1, home_pts=115, away_pts=108),
    ])
    out = head_to_head_last_margin(games)
    g2 = out.filter(pl.col("game_id") == "G2").row(0, named=True)
    assert g2["h2h_last_meeting_margin"] == -10.0


def test_h2h_uses_most_recent_meeting_not_first() -> None:
    """Three meetings; G3 should see G2's margin, not G1's."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 2, 1), 2022,
                   home_team_id=2, away_team_id=1, home_pts=115, away_pts=108),
        _full_game("G3", dt.date(2023, 3, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=120, away_pts=110),
    ])
    out = head_to_head_last_margin(games)
    g3 = out.filter(pl.col("game_id") == "G3").row(0, named=True)
    # G2: LAL won by 7. Current home is BOS, so flip → -7.
    assert g3["h2h_last_meeting_margin"] == -7.0


def test_h2h_isolates_team_pairs() -> None:
    """A BOS-CHI game cannot pollute the BOS-LAL h2h chain."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G_OTHER", dt.date(2023, 1, 10), 2022,
                   home_team_id=1, away_team_id=3, home_pts=130, away_pts=80),
        _full_game("G2", dt.date(2023, 2, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=112, away_pts=108),
    ])
    out = head_to_head_last_margin(games)
    # BOS-CHI is a first meeting too → 0.
    assert out.filter(pl.col("game_id") == "G_OTHER")["h2h_last_meeting_margin"][0] == 0.0
    # BOS-LAL G2 sees G1, not G_OTHER's blowout.
    assert out.filter(pl.col("game_id") == "G2")["h2h_last_meeting_margin"][0] == 10.0


def test_h2h_excludes_dropped_games() -> None:
    """A dropped prior meeting must not contribute to the chain."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_pts=999, away_pts=0, dropped=True),
        _full_game("G2", dt.date(2023, 2, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=112, away_pts=108),
    ])
    out = head_to_head_last_margin(games)
    # G2 should see no prior meeting because G1 was dropped.
    assert "G1" not in out["game_id"].to_list()  # dropped row absent
    assert out.filter(pl.col("game_id") == "G2")["h2h_last_meeting_margin"][0] == 0.0


# ===========================================================================
# opponent_defrtg_by_position
# ===========================================================================

def _matchup_setup_two_games() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Two BOS-LAL games with two starters per team per game. Used by
    multiple position-matchup tests so the hand-computed expectations stay
    consistent."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 10), 2022,
                   home_team_id=1, away_team_id=2, home_pts=115, away_pts=105),
    ])
    # Both starters at G and F per team. LAL's G scores 20, LAL's F scores 15.
    player_box = pl.DataFrame([
        # G1
        _pbox_row("G1", 100, 1, pts=25),  # BOS G
        _pbox_row("G1", 101, 1, pts=18),  # BOS F (we'll override position)
        _pbox_row("G1", 200, 2, pts=20),  # LAL G
        _pbox_row("G1", 201, 2, pts=15),  # LAL F
        # G2
        _pbox_row("G2", 100, 1, pts=22),
        _pbox_row("G2", 101, 1, pts=20),
        _pbox_row("G2", 200, 2, pts=24),
        _pbox_row("G2", 201, 2, pts=17),
    ]).with_columns(
        pl.when(pl.col("player_id").is_in([100, 200])).then(pl.lit("G"))
        .otherwise(pl.lit("F"))
        .alias("position")
    )
    team_box = pl.DataFrame([
        _tbox_row("G1", 1, pace=100.0), _tbox_row("G1", 2, pace=100.0),
        _tbox_row("G2", 1, pace=100.0), _tbox_row("G2", 2, pace=100.0),
    ])
    return games, player_box, team_box


def test_opp_defrtg_by_pos_returns_empty_for_empty_input() -> None:
    out = opponent_defrtg_by_position(pl.DataFrame(), pl.DataFrame(), pl.DataFrame())
    assert out.is_empty()


def test_opp_defrtg_by_pos_first_game_is_null() -> None:
    """First game in season → no prior possessions → cum_poss_prior=0 → None."""
    games, pb, tb = _matchup_setup_two_games()
    out = opponent_defrtg_by_position(pb, games, tb)
    g1 = out.filter(pl.col("game_id") == "G1")
    assert g1.height > 0
    for v in g1["opp_def_rtg_vs_pos"]:
        assert v is None


def test_opp_defrtg_by_pos_second_game_is_hand_computed() -> None:
    """G2: opp=BOS (team 1). LAL's G scored 20 at G1; LAL's F scored 15.
    BOS's pace at G1 = 100. So:
        opp_def_rtg_vs_pos(opp=BOS, G2, G) = 100 * 20 / 100 = 20.0
        opp_def_rtg_vs_pos(opp=BOS, G2, F) = 100 * 15 / 100 = 15.0

    And opp=LAL at G2: BOS's G scored 25, BOS's F scored 18; LAL's pace=100.
        opp_def_rtg_vs_pos(opp=LAL, G2, G) = 25.0
        opp_def_rtg_vs_pos(opp=LAL, G2, F) = 18.0
    """
    games, pb, tb = _matchup_setup_two_games()
    out = opponent_defrtg_by_position(pb, games, tb)
    g2 = out.filter(pl.col("game_id") == "G2")
    by_key = {(r["opp_team_id"], r["position"]): r["opp_def_rtg_vs_pos"]
              for r in g2.iter_rows(named=True)}
    assert by_key[(1, "G")] == pytest.approx(20.0)
    assert by_key[(1, "F")] == pytest.approx(15.0)
    assert by_key[(2, "G")] == pytest.approx(25.0)
    assert by_key[(2, "F")] == pytest.approx(18.0)


def test_opp_defrtg_by_pos_excludes_empty_position() -> None:
    """A player with position="" (non-starter) must not appear in the
    output and must not contribute to any (opp_team, position) bucket."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 10), 2022,
                   home_team_id=1, away_team_id=2, home_pts=115, away_pts=105),
    ])
    # LAL's "G" player scores 20, LAL's "" player (off bench) scores 50.
    # If filtering works, only the G=20 contribution should appear.
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1, pts=10),
        _pbox_row("G1", 200, 2, pts=20),
        _pbox_row("G1", 201, 2, pts=50),
        _pbox_row("G2", 100, 1, pts=10),
        _pbox_row("G2", 200, 2, pts=20),
    ]).with_columns(
        pl.when(pl.col("player_id") == 100).then(pl.lit("G"))
        .when(pl.col("player_id") == 200).then(pl.lit("G"))
        .otherwise(pl.lit(""))
        .alias("position")
    )
    tb = pl.DataFrame([
        _tbox_row("G1", 1), _tbox_row("G1", 2),
        _tbox_row("G2", 1), _tbox_row("G2", 2),
    ])
    out = opponent_defrtg_by_position(pb, games, tb)
    # No "" rows ever appear.
    assert "" not in out["position"].to_list()
    # And the G2 value for opp=BOS, pos=G uses only the position=G data
    # from G1 (pts=20), not the 50 from position="".
    g2_bos_g = out.filter(
        (pl.col("game_id") == "G2") & (pl.col("opp_team_id") == 1) & (pl.col("position") == "G")
    ).row(0, named=True)
    assert g2_bos_g["opp_def_rtg_vs_pos"] == pytest.approx(20.0)


def test_opp_defrtg_by_pos_resets_across_seasons() -> None:
    """A new season starts with cum_poss_prior=0 again. The carried-over
    point totals from last season must not leak into this season's rating."""
    games = pl.DataFrame([
        _full_game("S1G1", dt.date(2022, 12, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("S2G1", dt.date(2023, 10, 1), 2023,
                   home_team_id=1, away_team_id=2, home_pts=112, away_pts=108),
    ])
    pb = pl.DataFrame([
        _pbox_row("S1G1", 100, 1, pts=25),
        _pbox_row("S1G1", 200, 2, pts=30),
        _pbox_row("S2G1", 100, 1, pts=22),
        _pbox_row("S2G1", 200, 2, pts=20),
    ]).with_columns(pl.lit("G").alias("position"))
    tb = pl.DataFrame([
        _tbox_row("S1G1", 1), _tbox_row("S1G1", 2),
        _tbox_row("S2G1", 1), _tbox_row("S2G1", 2),
    ])
    out = opponent_defrtg_by_position(pb, games, tb)
    # S2G1 is the FIRST game of the new season → null.
    s2 = out.filter(pl.col("game_id") == "S2G1")
    for v in s2["opp_def_rtg_vs_pos"]:
        assert v is None


# ===========================================================================
# opponent_blocks_allowed_by_position
# ===========================================================================

def test_opp_blk_allowed_by_pos_returns_empty_for_empty_input() -> None:
    out = opponent_blocks_allowed_by_position(pl.DataFrame(), pl.DataFrame())
    assert out.is_empty()


def test_opp_blk_allowed_by_pos_first_game_is_null() -> None:
    """First game in season → cum_games_prior=0 → None (cold-start)."""
    games, pb, _tb = _matchup_setup_two_games()
    out = opponent_blocks_allowed_by_position(pb, games)
    g1 = out.filter(pl.col("game_id") == "G1")
    assert g1.height > 0
    for v in g1["opp_blk_allowed_vs_pos"]:
        assert v is None


def test_opp_blk_allowed_by_pos_second_game_is_hand_computed() -> None:
    """G2: opp=BOS (team 1). At G1, LAL's G recorded 3 blocks, LAL's F got 1.
    BOS has played 1 prior game at each position.
        opp_blk_allowed_vs_pos(opp=BOS, G2, G) = 3 / 1 = 3.0
        opp_blk_allowed_vs_pos(opp=BOS, G2, F) = 1 / 1 = 1.0

    Symmetrically, opp=LAL at G2:
        opp_blk_allowed_vs_pos(opp=LAL, G2, G) = (BOS G's G1 blk) / 1
        opp_blk_allowed_vs_pos(opp=LAL, G2, F) = (BOS F's G1 blk) / 1
    """
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 10), 2022,
                   home_team_id=1, away_team_id=2, home_pts=115, away_pts=105),
    ])
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1, blk=2),   # BOS G — 2 blocks vs LAL
        _pbox_row("G1", 101, 1, blk=4),   # BOS F — 4 blocks vs LAL
        _pbox_row("G1", 200, 2, blk=3),   # LAL G — 3 blocks vs BOS
        _pbox_row("G1", 201, 2, blk=1),   # LAL F — 1 block  vs BOS
        _pbox_row("G2", 100, 1, blk=0),
        _pbox_row("G2", 101, 1, blk=0),
        _pbox_row("G2", 200, 2, blk=0),
        _pbox_row("G2", 201, 2, blk=0),
    ]).with_columns(
        pl.when(pl.col("player_id").is_in([100, 200])).then(pl.lit("G"))
        .otherwise(pl.lit("F"))
        .alias("position")
    )
    out = opponent_blocks_allowed_by_position(pb, games)
    g2 = out.filter(pl.col("game_id") == "G2")
    by_key = {(r["opp_team_id"], r["position"]): r["opp_blk_allowed_vs_pos"]
              for r in g2.iter_rows(named=True)}
    assert by_key[(1, "G")] == pytest.approx(3.0)
    assert by_key[(1, "F")] == pytest.approx(1.0)
    assert by_key[(2, "G")] == pytest.approx(2.0)
    assert by_key[(2, "F")] == pytest.approx(4.0)


# ===========================================================================
# add_matchup_features
# ===========================================================================

def test_add_matchup_features_returns_empty_for_empty_input() -> None:
    out = add_matchup_features(
        pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), defrtg_vs_pos=None,
    )
    assert out.is_empty()


def test_add_matchup_features_attaches_opp_team_id() -> None:
    """The most fundamental contract: every player row must learn who its
    opponent was. BOS players → opp_team_id=2, LAL players → opp_team_id=1."""
    games, pb, tb = _matchup_setup_two_games()
    tr = team_rolling(tb, games)
    pr = player_rolling(pb, games, team_box=tb)
    out = add_matchup_features(pr, tr, games)
    for r in out.iter_rows(named=True):
        expected_opp = 2 if r["team_id"] == 1 else 1
        assert r["opp_team_id"] == expected_opp


def test_add_matchup_features_joins_opp_rolling() -> None:
    """opp_def_rtg_10 for a BOS player at G2 = LAL's t_def_rtg_10 at G2
    (i.e. LAL's def_rtg from their games before G2 — just G1)."""
    games, pb, tb = _matchup_setup_two_games()
    tr = team_rolling(tb, games)
    pr = player_rolling(pb, games, team_box=tb)
    out = add_matchup_features(pr, tr, games)
    bos_at_g2 = out.filter(
        (pl.col("game_id") == "G2") & (pl.col("team_id") == 1)
    ).row(0, named=True)
    lal_def_rtg_g1 = (
        tb.filter((pl.col("team_id") == 2) & (pl.col("game_id") == "G1"))["def_rtg"][0]
    )
    assert bos_at_g2["opp_def_rtg_10"] == pytest.approx(lal_def_rtg_g1)


def test_add_matchup_features_h2h_fills_null_with_zero() -> None:
    """First meetings produce null pre-join (no row in h2h table for that
    game_id). The function must fill with 0 per PLAN.md §3.1."""
    games, pb, tb = _matchup_setup_two_games()
    tr = team_rolling(tb, games)
    pr = player_rolling(pb, games, team_box=tb)
    out = add_matchup_features(pr, tr, games)
    g1_h2h = out.filter(pl.col("game_id") == "G1")["h2h_last_meeting_margin"].unique().to_list()
    assert g1_h2h == [0.0]


def test_add_matchup_features_position_join_optional() -> None:
    """When defrtg_vs_pos is None, the column simply isn't there."""
    games, pb, tb = _matchup_setup_two_games()
    tr = team_rolling(tb, games)
    pr = player_rolling(pb, games, team_box=tb)
    out_without = add_matchup_features(pr, tr, games, defrtg_vs_pos=None)
    assert "opp_def_rtg_vs_pos" not in out_without.columns

    dvp = opponent_defrtg_by_position(pb, games, tb)
    out_with = add_matchup_features(pr, tr, games, defrtg_vs_pos=dvp)
    assert "opp_def_rtg_vs_pos" in out_with.columns


def test_add_matchup_features_drops_player_rows_for_dropped_games() -> None:
    """A player row whose game is flagged dropped has no entry in the
    opponent map (the map filters on ~dropped). The inner join drops it."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2, home_pts=110, away_pts=100),
        _full_game("G_BAD", dt.date(2023, 1, 10), 2022,
                   home_team_id=1, away_team_id=2, home_pts=999, away_pts=0,
                   dropped=True),
    ])
    pb = pl.DataFrame([
        _pbox_row("G1", 100, 1),
        _pbox_row("G_BAD", 100, 1),
    ])
    tb = pl.DataFrame([
        _tbox_row("G1", 1), _tbox_row("G1", 2),
        _tbox_row("G_BAD", 1), _tbox_row("G_BAD", 2),
    ])
    # rolling itself doesn't see the dropped row (it filters too), so feed
    # it the raw player_box and check that the matchup join drops the bad row.
    pr = pb  # use raw, bypass rolling for this test
    tr = team_rolling(tb, games)
    out = add_matchup_features(pr, tr, games)
    assert "G_BAD" not in out["game_id"].to_list()


def test_add_matchup_features_row_count_preserved() -> None:
    """No row explosion: input row count == output row count when all
    player rows belong to non-dropped games."""
    games, pb, tb = _matchup_setup_two_games()
    tr = team_rolling(tb, games)
    pr = player_rolling(pb, games, team_box=tb)
    out = add_matchup_features(pr, tr, games)
    assert out.height == pr.height


# ===========================================================================
# context.py — scalar arena lookups and haversine
# ===========================================================================

def test_arena_altitude_known_teams() -> None:
    """Spot-check the major altitude outliers and a sea-level team."""
    assert arena_altitude("DEN") == pytest.approx(5280.0)  # Mile High
    assert arena_altitude("UTA") == pytest.approx(4226.0)
    assert arena_altitude("BOS") == pytest.approx(20.0)


def test_arena_altitude_unknown_returns_default() -> None:
    """Quiet fallback to ~sea level — minimizes the altitude signal
    on a typo rather than crashing mid-pipeline."""
    assert arena_altitude("XYZ") == 50.0
    assert arena_altitude("") == 50.0


def test_travel_distance_same_arena_is_zero() -> None:
    """LAL and LAC share Crypto.com Arena — distance must be exactly 0."""
    assert travel_distance_miles("LAL", "LAC") == 0.0
    assert travel_distance_miles("LAC", "LAL") == 0.0


def test_travel_distance_symmetric() -> None:
    """Great-circle distance is symmetric. Polars 1.17 uses arcsin form
    while the Python scalar uses atan2 — symmetry catches any drift."""
    a = travel_distance_miles("BOS", "DEN")
    b = travel_distance_miles("DEN", "BOS")
    assert a == pytest.approx(b)
    # Sanity bound: BOS<->DEN great-circle is ~1750 mi.
    assert 1500 < a < 2000


def test_travel_distance_known_pair_approx() -> None:
    """Catch obvious lat/lng typos: NYK-BOS is famously ~190 mi."""
    d = travel_distance_miles("NYK", "BOS")
    assert 180 < d < 200, f"NYK->BOS = {d}, expected ~190"


def test_travel_distance_unknown_returns_nan() -> None:
    """No sensible default for distance — surface unknowns loudly."""
    assert math.isnan(travel_distance_miles("ZZZ", "BOS"))
    assert math.isnan(travel_distance_miles("BOS", "ZZZ"))


def test_haversine_to_self_is_zero() -> None:
    assert _haversine_miles(42.0, -71.0, 42.0, -71.0) == 0.0


# ===========================================================================
# add_context_features
# ===========================================================================

def test_add_context_features_returns_empty_for_empty_input() -> None:
    out = add_context_features(pl.DataFrame())
    assert out.is_empty()


def test_add_context_fans_out_one_row_per_team_per_game() -> None:
    """Each game produces two rows, one per team, with the correct
    ``is_home`` flag on each side."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
    ])
    out = add_context_features(games)
    assert out.height == 2
    assert set(out["team_id"].to_list()) == {1, 2}
    by_tid = {r["team_id"]: r for r in out.iter_rows(named=True)}
    assert by_tid[1]["is_home"] is True
    assert by_tid[2]["is_home"] is False


def test_add_context_schedule_features_null_on_first_game() -> None:
    """First game of a (team, season) has no prior → all shift-based
    features are null. ``altitude_ft`` and ``is_home`` are not history-
    dependent, so they're populated."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
    ])
    out = add_context_features(games)
    for r in out.iter_rows(named=True):
        assert r["rest_days"] is None
        assert r["b2b"] is None
        assert r["is_3in4"] is None
        assert r["is_4in6"] is None
        assert r["travel_miles_prev"] is None
        # But identity / pure-row features are still populated.
        assert r["altitude_ft"] is not None
        assert r["day_of_week"] is not None


def test_add_context_b2b_true_on_consecutive_days() -> None:
    """Played yesterday → gap=1 → rest_days=0, b2b=True. Verifies the
    "nights of rest" convention (rest_days = gap - 1)."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 2), 2022,
                   home_team_id=1, away_team_id=3,
                   home_team_abbr="BOS", away_team_abbr="CHI",
                   home_pts=105, away_pts=100),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    bos_g2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G2")).row(0, named=True)
    assert bos_g2["b2b"] is True
    assert bos_g2["rest_days"] == 0


def test_add_context_rest_days_clipped_at_5() -> None:
    """Long gap (18 days) → rest_days clipped to 5, not 17. b2b False."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 20), 2022,
                   home_team_id=1, away_team_id=3,
                   home_team_abbr="BOS", away_team_abbr="CHI",
                   home_pts=105, away_pts=100),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    bos_g2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G2")).row(0, named=True)
    assert bos_g2["rest_days"] == 5
    assert bos_g2["b2b"] is False


def test_add_context_is_3in4_true_for_three_games_in_four_days() -> None:
    """Jan 1 → Jan 3 → Jan 4 is 3 games spanning 4 days; the third game
    has gap_2back = 3 ≤ 3 → is_3in4 True."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 3), 2022,
                   home_team_id=1, away_team_id=3,
                   home_team_abbr="BOS", away_team_abbr="CHI",
                   home_pts=105, away_pts=100),
        _full_game("G3", dt.date(2023, 1, 4), 2022,
                   home_team_id=1, away_team_id=4,
                   home_team_abbr="BOS", away_team_abbr="MIA",
                   home_pts=120, away_pts=110),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    bos_g3 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G3")).row(0, named=True)
    assert bos_g3["is_3in4"] is True
    # And the boundary case the test could miss: 4-day spans of 5+ shouldn't
    # trigger. Build that explicitly:
    games2 = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 3), 2022,
                   home_team_id=1, away_team_id=3,
                   home_team_abbr="BOS", away_team_abbr="CHI",
                   home_pts=105, away_pts=100),
        _full_game("G3", dt.date(2023, 1, 5), 2022,
                   home_team_id=1, away_team_id=4,
                   home_team_abbr="BOS", away_team_abbr="MIA",
                   home_pts=120, away_pts=110),
    ])
    # 3 games across 5 calendar days (Jan 1-5) — NOT 3in4.
    out2 = add_context_features(games2).sort(["team_id", "date"])
    bos2 = out2.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G3")).row(0, named=True)
    assert bos2["is_3in4"] is False


def test_add_context_is_4in6_true_for_four_games_in_six_days() -> None:
    """Jan 1, 2, 4, 6: gap_3back = 5 ≤ 5 → is_4in6 True."""
    games = pl.DataFrame([
        _full_game(f"G{i+1}", date, 2022,
                   home_team_id=1, away_team_id=10 + i,
                   home_team_abbr="BOS", away_team_abbr=opp,
                   home_pts=100, away_pts=95)
        for i, (date, opp) in enumerate([
            (dt.date(2023, 1, 1), "LAL"),
            (dt.date(2023, 1, 2), "CHI"),
            (dt.date(2023, 1, 4), "MIA"),
            (dt.date(2023, 1, 6), "PHI"),
        ])
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    bos_g4 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G4")).row(0, named=True)
    assert bos_g4["is_4in6"] is True


def test_add_context_season_phase_playoffs_takes_precedence() -> None:
    """is_playoffs=True overrides the month-bucket rule. A May game with
    is_playoffs=False would be 'late'; with True it must be 'playoffs'."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 5, 15), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100,
                   is_playoffs=True),
    ])
    out = add_context_features(games)
    for r in out.iter_rows(named=True):
        assert r["season_phase"] == "playoffs"


@pytest.mark.parametrize(
    ("month", "phase"),
    [
        (10, "early"), (11, "early"),
        (12, "mid"),   (1, "mid"),   (2, "mid"),
        (3, "late"),   (4, "late"),
    ],
)
def test_add_context_season_phase_by_month(month: int, phase: str) -> None:
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, month, 15), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100,
                   is_playoffs=False),
    ])
    out = add_context_features(games)
    for r in out.iter_rows(named=True):
        assert r["season_phase"] == phase


def test_add_context_day_of_week_zero_indexed_monday() -> None:
    """PLAN says day_of_week is 0..6. Polars' dt.weekday() returns 1..7
    with Mon=1, so the function subtracts 1. Jan 1 2023 was Sunday."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,  # Sun
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 2), 2022,  # Mon
                   home_team_id=1, away_team_id=3,
                   home_team_abbr="BOS", away_team_abbr="CHI",
                   home_pts=105, away_pts=100),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    g1 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G1")).row(0, named=True)
    g2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G2")).row(0, named=True)
    assert g1["day_of_week"] == 6  # Sunday
    assert g2["day_of_week"] == 0  # Monday


def test_add_context_month_matches_date() -> None:
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 3, 15), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
    ])
    out = add_context_features(games)
    for r in out.iter_rows(named=True):
        assert r["month"] == 3


def test_add_context_altitude_known_and_unknown() -> None:
    """Known team gets its real altitude; unknown abbr falls back to 50."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=99,
                   home_team_abbr="DEN", away_team_abbr="???",
                   home_pts=110, away_pts=100),
    ])
    out = add_context_features(games)
    den = out.filter(pl.col("team_abbr") == "DEN").row(0, named=True)
    unk = out.filter(pl.col("team_abbr") == "???").row(0, named=True)
    assert den["altitude_ft"] == pytest.approx(5280.0)
    assert unk["altitude_ft"] == 50.0


def test_add_context_travel_zero_when_staying_at_same_arena() -> None:
    """Two home games in a row → previous arena == current arena → 0 miles."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 3), 2022,
                   home_team_id=1, away_team_id=3,
                   home_team_abbr="BOS", away_team_abbr="CHI",
                   home_pts=105, away_pts=100),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    bos_g2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G2")).row(0, named=True)
    assert bos_g2["travel_miles_prev"] == 0.0


def test_add_context_travel_road_trip_distance() -> None:
    """BOS home, then BOS at NYK → travel ~190 mi (the canonical NE-corridor
    great-circle). Pinned only to a window to allow lat/lng refinement."""
    games = pl.DataFrame([
        _full_game("G1", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("G2", dt.date(2023, 1, 3), 2022,
                   home_team_id=20, away_team_id=1,
                   home_team_abbr="NYK", away_team_abbr="BOS",
                   home_pts=105, away_pts=100),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    bos_g2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "G2")).row(0, named=True)
    assert 180 < bos_g2["travel_miles_prev"] < 200


def test_add_context_cross_season_reset() -> None:
    """First game of a new season — even with a prior-season game in the
    same frame — gets null rest/b2b/density/travel. The summer break
    must not masquerade as a multi-month rest streak."""
    games = pl.DataFrame([
        _full_game("S1G1", dt.date(2022, 12, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
        _full_game("S2G1", dt.date(2023, 10, 15), 2023,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
    ])
    out = add_context_features(games).sort(["team_id", "date"])
    s2 = out.filter((pl.col("team_id") == 1) & (pl.col("game_id") == "S2G1")).row(0, named=True)
    assert s2["rest_days"] is None
    assert s2["b2b"] is None
    assert s2["is_3in4"] is None
    assert s2["is_4in6"] is None
    assert s2["travel_miles_prev"] is None


def test_add_context_excludes_dropped_games() -> None:
    """A dropped row must not appear in the output AND must not contribute
    to the next game's shift-based features. Otherwise a fake game's
    date would corrupt rest_days for the next real game."""
    games = pl.DataFrame([
        _full_game("G_BAD", dt.date(2023, 1, 1), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=999, away_pts=0,
                   dropped=True),
        _full_game("G1", dt.date(2023, 1, 5), 2022,
                   home_team_id=1, away_team_id=2,
                   home_team_abbr="BOS", away_team_abbr="LAL",
                   home_pts=110, away_pts=100),
    ])
    out = add_context_features(games)
    assert "G_BAD" not in out["game_id"].to_list()
    # G1 is now effectively the team's first game → null shift features.
    g1 = out.filter(pl.col("game_id") == "G1").row(0, named=True)
    assert g1["rest_days"] is None


# ===========================================================================
# Not-yet-implemented features
# ===========================================================================

def test_cold_start_player_feature_is_nan_filled() -> None:
    pytest.skip("cold-start fill handled in §7 simulator, not in rolling")
