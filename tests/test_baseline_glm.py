"""Tests for ``nba_sim.models.baseline_glm``.

Covers both ``SeasonAverageBaseline`` and ``PoissonGLMBaseline``:

    - ``fit`` / ``predict`` lifecycle and output shape.
    - Predict-before-fit guard.
    - SeasonAverage strictly-prior cumulative mean: first row falls back,
      second equals first, third equals mean of first two; rookie falls
      back to position mean; unknown position falls back to global mean;
      players are isolated; cross-season prior-season lookup works.
    - GLM: predictions non-negative, feature subset auto-resolved,
      missing-feature error, null/NaN imputation uses fit-time mean,
      sanity check that features drive predictions.
    - ``project_to_constraints``: makes ≤ attempts, 3P ⊆ FG, REB / PTS
      identities, minutes sum to 240 per team, zero-sum equal-fallback.
    - ``save`` / ``load`` round-trips for both baselines.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from nba_sim.models.baseline_glm import (
    PoissonGLMBaseline,
    SeasonAverageBaseline,
)


# ---------------------------------------------------------------------------
# Synthetic processed-row builder. Covers the columns both baselines expect:
# identifiers, targets, plus a small subset of the GLM's default features.
# Tests that don't care about a column simply leave it at its default.
# ---------------------------------------------------------------------------


def _row(
    *,
    game_id: str,
    player_id: int,
    season: int,
    date: dt.date,
    team_id: int = 100,
    position: str = "G",
    minutes: float = 30.0,
    pts: int = 20,
    fgm: int = 8,
    fga: int = 16,
    tpm: int = 2,
    tpa: int = 6,
    ftm: int = 4,
    fta: int = 5,
    oreb: int = 1,
    dreb: int = 4,
    reb: int = 5,
    ast: int = 5,
    stl: int = 1,
    blk: int = 1,
    tov: int = 2,
    pf: int = 2,
    p_min_avg_10: float = 30.0,
    t_pace_10: float = 100.0,
    opp_def_rtg_10: float = 110.0,
    is_home: bool = True,
    rest_days: int = 1,
    b2b: bool = False,
    is_3in4: bool = False,
    is_4in6: bool = False,
    day_of_week: int = 3,
    month: int = 11,
    altitude_ft: float = 50.0,
    travel_miles_prev: float = 0.0,
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "player_id": player_id,
        "team_id": team_id,
        "season": season,
        "date": date,
        "position": position,
        "minutes": minutes,
        "pts": pts,
        "fgm": fgm, "fga": fga,
        "tpm": tpm, "tpa": tpa,
        "ftm": ftm, "fta": fta,
        "oreb": oreb, "dreb": dreb, "reb": reb,
        "ast": ast, "stl": stl, "blk": blk, "tov": tov, "pf": pf,
        "p_min_avg_10": p_min_avg_10,
        "t_pace_10": t_pace_10,
        "opp_def_rtg_10": opp_def_rtg_10,
        "is_home": is_home,
        "rest_days": rest_days,
        "b2b": b2b,
        "is_3in4": is_3in4,
        "is_4in6": is_4in6,
        "day_of_week": day_of_week,
        "month": month,
        "altitude_ft": altitude_ft,
        "travel_miles_prev": travel_miles_prev,
    }


def _df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# A reusable small dataset: 3 players × 2 seasons × 4 games each. Player 1
# averages 20 pts, player 2 averages 25, player 3 averages 30 — gives the
# GLM something to discriminate on.
def _baseline_setup() -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for player_id, base_pts in [(1, 20), (2, 25), (3, 30)]:
        for season in (2022, 2023):
            for g in range(4):
                rows.append(
                    _row(
                        game_id=f"{season}_{player_id}_{g}",
                        player_id=player_id,
                        season=season,
                        date=dt.date(season, 11, 1) + dt.timedelta(days=g),
                        position=["F", "G", "C"][player_id - 1],
                        minutes=30.0 + player_id,
                        pts=base_pts + g,
                        p_min_avg_10=30.0 + player_id,
                    )
                )
    return _df(rows)


# ---------------------------------------------------------------------------
# SeasonAverageBaseline — lifecycle + contract.
# ---------------------------------------------------------------------------


def test_sab_predict_before_fit_raises() -> None:
    df = _baseline_setup()
    with pytest.raises(RuntimeError, match="before fit"):
        SeasonAverageBaseline().predict(df)


def test_sab_fit_requires_player_id_column() -> None:
    df = _baseline_setup().drop("player_id")
    with pytest.raises(ValueError, match="player_id"):
        SeasonAverageBaseline().fit(df)


def test_sab_fit_returns_self_and_marks_fitted() -> None:
    sab = SeasonAverageBaseline()
    assert not sab.is_fitted
    out = sab.fit(_baseline_setup())
    assert out is sab
    assert sab.is_fitted


def test_sab_predict_shape_is_three_ids_plus_one_pred_per_stat() -> None:
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    pred = sab.predict(df)
    # 3 ID cols + 16 stats = 19 columns; rows preserved one-for-one.
    assert pred.height == df.height
    assert set(pred.columns) >= {"game_id", "player_id", "team_id"}
    assert sum(c.startswith("pred_") for c in pred.columns) == 16


def test_sab_predict_preserves_identifier_columns() -> None:
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    pred = sab.predict(df)
    # Same (game_id, player_id, team_id) set comes back.
    src = df.select("game_id", "player_id", "team_id").sort("game_id")
    out = pred.select("game_id", "player_id", "team_id").sort("game_id")
    assert src.equals(out)


# ---------------------------------------------------------------------------
# SeasonAverageBaseline — cumulative-prior correctness.
# ---------------------------------------------------------------------------


def test_sab_first_game_of_season_falls_back_to_prior_season_avg() -> None:
    # Player 1's 2022 pts: [20, 21, 22, 23] → mean 21.5.
    # First 2023 game has 0 priors-in-current-season, should hit prior-season avg.
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    pred = sab.predict(df)
    first_2023 = pred.filter(
        (pl.col("player_id") == 1) & (pl.col("game_id") == "2023_1_0")
    )
    assert first_2023.height == 1
    assert first_2023["pred_pts"][0] == pytest.approx(21.5)


def test_sab_second_game_of_season_equals_first_games_value() -> None:
    # The second game's strictly-prior mean is just the first game's value.
    # Player 1 2023: game 0 has pts=20, so game 1's std avg should be exactly 20.
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    pred = sab.predict(df)
    game1 = pred.filter(
        (pl.col("player_id") == 1) & (pl.col("game_id") == "2023_1_1")
    )
    assert game1["pred_pts"][0] == pytest.approx(20.0)


def test_sab_third_game_of_season_equals_mean_of_first_two() -> None:
    # Player 2 2023: pts = [25, 26, 27, 28]. Game 2's std avg = (25 + 26) / 2 = 25.5.
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    pred = sab.predict(df)
    game2 = pred.filter(
        (pl.col("player_id") == 2) & (pl.col("game_id") == "2023_2_2")
    )
    assert game2["pred_pts"][0] == pytest.approx(25.5)


def test_sab_rookie_falls_back_to_position_mean() -> None:
    # Train: only player 1 (position F), avg pts 21.5.
    # Predict: a brand-new player 99, position F, never in training → uses
    # position F's mean.
    train = _df(
        [
            _row(
                game_id=f"2022_1_{g}",
                player_id=1,
                season=2022,
                date=dt.date(2022, 11, 1) + dt.timedelta(days=g),
                position="F",
                pts=20 + g,
            )
            for g in range(4)
        ]
    )
    novel = _df(
        [
            _row(
                game_id="2023_99_0",
                player_id=99,
                season=2023,
                date=dt.date(2023, 11, 1),
                position="F",
                pts=0,  # irrelevant — std count is 0 for this row
            )
        ]
    )
    sab = SeasonAverageBaseline().fit(train)
    pred = sab.predict(novel)
    # Position F mean is 21.5 (player 1's 2022 mean = mean of 20..23).
    assert pred["pred_pts"][0] == pytest.approx(21.5)


def test_sab_unknown_position_falls_back_to_global_mean() -> None:
    # Train has only position "G". Predict a player with position "PG" (not
    # in training). Player has no prior season either. Should fall through to
    # global mean.
    train = _df(
        [
            _row(
                game_id=f"2022_1_{g}",
                player_id=1,
                season=2022,
                date=dt.date(2022, 11, 1) + dt.timedelta(days=g),
                position="G",
                pts=20 + g,
            )
            for g in range(4)
        ]
    )
    global_mean_pts = float(train["pts"].mean())
    novel = _df(
        [
            _row(
                game_id="2023_99_0",
                player_id=99,
                season=2023,
                date=dt.date(2023, 11, 1),
                position="PG",  # unseen position
                pts=0,
            )
        ]
    )
    sab = SeasonAverageBaseline().fit(train)
    pred = sab.predict(novel)
    assert pred["pred_pts"][0] == pytest.approx(global_mean_pts)


def test_sab_isolates_players_within_same_season() -> None:
    # Each player's strictly-prior std should ignore other players.
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    pred = sab.predict(df)
    # Player 1 game 1's pred_pts must be 20 (player 1's game 0), NOT pulled
    # toward player 2 or 3.
    p1_g1 = pred.filter(
        (pl.col("player_id") == 1) & (pl.col("game_id") == "2023_1_1")
    )
    assert p1_g1["pred_pts"][0] == pytest.approx(20.0)
    # Likewise player 3 game 1 should be 30.
    p3_g1 = pred.filter(
        (pl.col("player_id") == 3) & (pl.col("game_id") == "2023_3_1")
    )
    assert p3_g1["pred_pts"][0] == pytest.approx(30.0)


def test_sab_prior_season_asof_uses_strictly_less_than_target() -> None:
    # The asof-join uses lookup_season = season + 1 specifically so we never
    # match the target season itself. Verify: a player with games in both
    # 2022 and 2023 in the training set — when predicting their first game
    # of 2023, the prior-season lookup must hit 2022, not 2023.
    train = _df(
        [
            _row(
                game_id=f"2022_1_{g}",
                player_id=1,
                season=2022,
                date=dt.date(2022, 11, 1) + dt.timedelta(days=g),
                pts=20 + g,  # 2022 mean = 21.5
            )
            for g in range(4)
        ]
        + [
            _row(
                game_id=f"2023_1_{g}",
                player_id=1,
                season=2023,
                date=dt.date(2023, 11, 1) + dt.timedelta(days=g),
                pts=100 + g,  # 2023 mean would be ~101.5; must NOT be used as fallback
            )
            for g in range(4)
        ]
    )
    sab = SeasonAverageBaseline().fit(train)
    # Predict the first 2023 game — std_count is 0, must use 2022 avg.
    pred = sab.predict(train).filter(pl.col("game_id") == "2023_1_0")
    assert pred["pred_pts"][0] == pytest.approx(21.5)


def test_sab_save_and_load_round_trip(tmp_path: Path) -> None:
    df = _baseline_setup()
    sab = SeasonAverageBaseline().fit(df)
    out_path = tmp_path / "sab.joblib"
    sab.save(out_path)

    loaded = SeasonAverageBaseline.load(out_path)
    assert loaded.is_fitted
    p1 = sab.predict(df).sort("game_id")
    p2 = loaded.predict(df).sort("game_id")
    assert p1.equals(p2)


def test_sab_save_before_fit_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        SeasonAverageBaseline().save(tmp_path / "x.joblib")


# ---------------------------------------------------------------------------
# PoissonGLMBaseline — lifecycle + contract.
# ---------------------------------------------------------------------------


def test_glm_predict_before_fit_raises() -> None:
    df = _baseline_setup()
    with pytest.raises(RuntimeError, match="before fit"):
        PoissonGLMBaseline().predict(df)


def test_glm_save_before_fit_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        PoissonGLMBaseline().save(tmp_path / "x.joblib")


def test_glm_fit_with_no_usable_features_raises() -> None:
    # Build a frame that has no overlap with _DEFAULT_FEATURES — none of the
    # p_*, t_*, opp_*, h2h_*, context cols exist.
    rows = [
        {
            "game_id": "g1",
            "player_id": 1,
            "team_id": 100,
            "season": 2022,
            "date": dt.date(2022, 11, 1),
            "position": "G",
            "minutes": 30.0,
            "pts": 20,
            "fgm": 8, "fga": 16, "tpm": 2, "tpa": 6,
            "ftm": 4, "fta": 5,
            "oreb": 1, "dreb": 4, "reb": 5,
            "ast": 5, "stl": 1, "blk": 1, "tov": 2, "pf": 2,
        }
    ]
    df = pl.DataFrame(rows)
    with pytest.raises(ValueError, match="no usable feature columns"):
        PoissonGLMBaseline().fit(df)


def test_glm_fit_resolves_only_columns_present() -> None:
    # Build a frame with just two of the default features — fit should pick
    # those two and ignore the rest of the default list.
    df = _baseline_setup().drop(
        "opp_def_rtg_10", "is_home", "rest_days", "b2b", "is_3in4",
        "is_4in6", "day_of_week", "month", "altitude_ft", "travel_miles_prev",
    )
    glm = PoissonGLMBaseline().fit(df)
    assert set(glm._feature_cols_resolved) == {"p_min_avg_10", "t_pace_10"}


def test_glm_predict_shape() -> None:
    df = _baseline_setup()
    glm = PoissonGLMBaseline().fit(df)
    pred = glm.predict(df)
    # 3 ID cols + 1 minutes + 15 counting stats = 19 columns.
    assert pred.height == df.height
    assert sum(c.startswith("pred_") for c in pred.columns) == 16
    assert "pred_minutes" in pred.columns
    assert "pred_pts" in pred.columns


def test_glm_predictions_are_non_negative() -> None:
    df = _baseline_setup()
    glm = PoissonGLMBaseline().fit(df)
    pred = glm.predict(df)
    for col in pred.columns:
        if not col.startswith("pred_"):
            continue
        assert (pred[col] >= 0.0).all(), f"{col} has negative predictions"


def test_glm_predictions_differentiate_by_feature_value() -> None:
    # Sanity check that the model actually learns *something* from features.
    # Train with player_id-encoded minutes baseline; predict for two players
    # with very different p_min_avg_10 values and assert minutes are different.
    rows: list[dict[str, Any]] = []
    for season in (2022, 2023):
        for g in range(6):
            rows.append(
                _row(
                    game_id=f"low_{season}_{g}",
                    player_id=1,
                    season=season,
                    date=dt.date(season, 11, 1) + dt.timedelta(days=g),
                    minutes=12.0,
                    p_min_avg_10=12.0,
                )
            )
            rows.append(
                _row(
                    game_id=f"hi_{season}_{g}",
                    player_id=2,
                    season=season,
                    date=dt.date(season, 11, 1) + dt.timedelta(days=g),
                    minutes=36.0,
                    p_min_avg_10=36.0,
                )
            )
    train = _df(rows)
    glm = PoissonGLMBaseline().fit(train)

    # Probe: two synthetic rows with the same scaffold but differing
    # p_min_avg_10. The minutes prediction should track p_min_avg_10.
    probe_lo = _row(
        game_id="probe_lo", player_id=999, season=2024,
        date=dt.date(2024, 11, 1), p_min_avg_10=10.0,
    )
    probe_hi = _row(
        game_id="probe_hi", player_id=999, season=2024,
        date=dt.date(2024, 11, 1), p_min_avg_10=38.0,
    )
    pred = glm.predict(_df([probe_lo, probe_hi]))
    lo = pred.filter(pl.col("game_id") == "probe_lo")["pred_minutes"][0]
    hi = pred.filter(pl.col("game_id") == "probe_hi")["pred_minutes"][0]
    assert hi > lo


def test_glm_nulls_in_features_filled_with_fit_mean() -> None:
    df = _baseline_setup()
    glm = PoissonGLMBaseline().fit(df)
    fit_mean = glm._feature_means["p_min_avg_10"]

    # Replace p_min_avg_10 with null and predict. The prediction must match
    # what we'd get if we'd passed the fit-time mean instead.
    null_row = df.head(1).with_columns(pl.lit(None, dtype=pl.Float64).alias("p_min_avg_10"))
    mean_row = df.head(1).with_columns(pl.lit(fit_mean, dtype=pl.Float64).alias("p_min_avg_10"))
    p_null = glm.predict(null_row)
    p_mean = glm.predict(mean_row)
    for col in p_null.columns:
        if col.startswith("pred_"):
            assert p_null[col][0] == pytest.approx(p_mean[col][0])


def test_glm_boolean_features_cast_without_error() -> None:
    # is_home, b2b, is_3in4, is_4in6 are bool. Just verify fit + predict run
    # cleanly when these are present (they're in the default feature list).
    df = _baseline_setup()
    glm = PoissonGLMBaseline().fit(df)
    pred = glm.predict(df)
    assert pred.height == df.height


def test_glm_save_and_load_round_trip(tmp_path: Path) -> None:
    df = _baseline_setup()
    glm = PoissonGLMBaseline().fit(df)
    out_path = tmp_path / "glm.joblib"
    glm.save(out_path)

    loaded = PoissonGLMBaseline.load(out_path)
    assert loaded.is_fitted
    p1 = glm.predict(df).sort("game_id")
    p2 = loaded.predict(df).sort("game_id")
    for col in p1.columns:
        if col.startswith("pred_"):
            for a, b in zip(p1[col], p2[col]):
                assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# PoissonGLMBaseline — project_to_constraints.
# ---------------------------------------------------------------------------


def _projected_frame() -> pl.DataFrame:
    """Hand-built predicted frame for projection tests. Two players on the
    same team in one game — minutes pre-projection sum to 60 (not 240) so
    the renormalization is exercised."""
    return pl.DataFrame(
        [
            {
                "game_id": "g1", "player_id": 1, "team_id": 100,
                "pred_minutes": 20.0,
                "pred_pts": 12.0,
                "pred_fgm": 10.0, "pred_fga": 8.0,    # FGM > FGA → must cap
                "pred_tpm": 4.0, "pred_tpa": 3.0,     # TPM > TPA → cap
                "pred_ftm": 6.0, "pred_fta": 4.0,     # FTM > FTA → cap
                "pred_oreb": 2.0, "pred_dreb": 5.0, "pred_reb": 99.0,  # reb gets rebuilt
                "pred_ast": 3.0, "pred_stl": 1.0, "pred_blk": 1.0,
                "pred_tov": 2.0, "pred_pf": 3.0,
            },
            {
                "game_id": "g1", "player_id": 2, "team_id": 100,
                "pred_minutes": 40.0,
                "pred_pts": 18.0,
                "pred_fgm": 6.0, "pred_fga": 12.0,
                "pred_tpm": 1.0, "pred_tpa": 4.0,
                "pred_ftm": 3.0, "pred_fta": 4.0,
                "pred_oreb": 1.0, "pred_dreb": 4.0, "pred_reb": 99.0,
                "pred_ast": 2.0, "pred_stl": 1.0, "pred_blk": 0.0,
                "pred_tov": 1.0, "pred_pf": 2.0,
            },
        ]
    )


def test_project_caps_makes_at_attempts() -> None:
    df = _projected_frame()
    out = PoissonGLMBaseline().project_to_constraints(df)
    p1 = out.filter(pl.col("player_id") == 1)
    # FGM was 10, FGA 8 — must be capped to 8 (and then 3PM ≤ FGM = 8 also).
    assert p1["pred_fgm"][0] == pytest.approx(8.0)
    # TPM was 4, TPA 3 → capped to 3, then 3PM ≤ FGM (=8) so stays at 3.
    assert p1["pred_tpm"][0] == pytest.approx(3.0)
    # FTM 6 > FTA 4 → cap to 4.
    assert p1["pred_ftm"][0] == pytest.approx(4.0)


def test_project_caps_threes_within_field_goals() -> None:
    # TPM ≤ FGM and TPA ≤ FGA.
    df = pl.DataFrame(
        [
            {
                "game_id": "g1", "player_id": 1, "team_id": 100,
                "pred_minutes": 30.0, "pred_pts": 0.0,
                "pred_fgm": 5.0, "pred_fga": 10.0,
                "pred_tpm": 9.0, "pred_tpa": 20.0,
                "pred_ftm": 0.0, "pred_fta": 0.0,
                "pred_oreb": 0.0, "pred_dreb": 0.0, "pred_reb": 0.0,
                "pred_ast": 0.0, "pred_stl": 0.0, "pred_blk": 0.0,
                "pred_tov": 0.0, "pred_pf": 0.0,
            }
        ]
    )
    out = PoissonGLMBaseline().project_to_constraints(df)
    # 3PM (9) was > FGM (5) → capped to 5.
    assert out["pred_tpm"][0] == pytest.approx(5.0)
    # 3PA (20) was > FGA (10) → capped to 10.
    assert out["pred_tpa"][0] == pytest.approx(10.0)


def test_project_rebuilds_reb_identity() -> None:
    df = _projected_frame()
    out = PoissonGLMBaseline().project_to_constraints(df)
    for row in out.iter_rows(named=True):
        assert row["pred_reb"] == pytest.approx(row["pred_oreb"] + row["pred_dreb"])


def test_project_rebuilds_pts_identity_from_capped_components() -> None:
    df = _projected_frame()
    out = PoissonGLMBaseline().project_to_constraints(df)
    for row in out.iter_rows(named=True):
        expected = 2.0 * row["pred_fgm"] + row["pred_tpm"] + row["pred_ftm"]
        assert row["pred_pts"] == pytest.approx(expected)


def test_project_normalizes_team_minutes_to_240() -> None:
    df = _projected_frame()
    out = PoissonGLMBaseline().project_to_constraints(df)
    team_sum = out.group_by(["game_id", "team_id"]).agg(
        pl.col("pred_minutes").sum().alias("s")
    )
    assert team_sum["s"][0] == pytest.approx(240.0)


def test_project_zero_sum_minutes_falls_back_to_equal_split() -> None:
    # Two players on the same team, both predicted 0 minutes. Equal split
    # would put 120 each.
    df = pl.DataFrame(
        [
            {
                "game_id": "g1", "player_id": 1, "team_id": 100,
                "pred_minutes": 0.0, "pred_pts": 0.0,
                "pred_fgm": 0.0, "pred_fga": 0.0,
                "pred_tpm": 0.0, "pred_tpa": 0.0,
                "pred_ftm": 0.0, "pred_fta": 0.0,
                "pred_oreb": 0.0, "pred_dreb": 0.0, "pred_reb": 0.0,
                "pred_ast": 0.0, "pred_stl": 0.0, "pred_blk": 0.0,
                "pred_tov": 0.0, "pred_pf": 0.0,
            },
            {
                "game_id": "g1", "player_id": 2, "team_id": 100,
                "pred_minutes": 0.0, "pred_pts": 0.0,
                "pred_fgm": 0.0, "pred_fga": 0.0,
                "pred_tpm": 0.0, "pred_tpa": 0.0,
                "pred_ftm": 0.0, "pred_fta": 0.0,
                "pred_oreb": 0.0, "pred_dreb": 0.0, "pred_reb": 0.0,
                "pred_ast": 0.0, "pred_stl": 0.0, "pred_blk": 0.0,
                "pred_tov": 0.0, "pred_pf": 0.0,
            },
        ]
    )
    out = PoissonGLMBaseline().project_to_constraints(df)
    assert out["pred_minutes"][0] == pytest.approx(120.0)
    assert out["pred_minutes"][1] == pytest.approx(120.0)


def test_project_isolates_minutes_normalization_across_teams() -> None:
    # Two different teams in one game — each independently sums to 240.
    df = pl.DataFrame(
        [
            {
                "game_id": "g1", "player_id": 1, "team_id": 100,
                "pred_minutes": 30.0,
                "pred_pts": 0.0, "pred_fgm": 0.0, "pred_fga": 0.0,
                "pred_tpm": 0.0, "pred_tpa": 0.0, "pred_ftm": 0.0, "pred_fta": 0.0,
                "pred_oreb": 0.0, "pred_dreb": 0.0, "pred_reb": 0.0,
                "pred_ast": 0.0, "pred_stl": 0.0, "pred_blk": 0.0,
                "pred_tov": 0.0, "pred_pf": 0.0,
            },
            {
                "game_id": "g1", "player_id": 2, "team_id": 200,
                "pred_minutes": 90.0,
                "pred_pts": 0.0, "pred_fgm": 0.0, "pred_fga": 0.0,
                "pred_tpm": 0.0, "pred_tpa": 0.0, "pred_ftm": 0.0, "pred_fta": 0.0,
                "pred_oreb": 0.0, "pred_dreb": 0.0, "pred_reb": 0.0,
                "pred_ast": 0.0, "pred_stl": 0.0, "pred_blk": 0.0,
                "pred_tov": 0.0, "pred_pf": 0.0,
            },
        ]
    )
    out = PoissonGLMBaseline().project_to_constraints(df)
    # Each team has one player; renormalization sends that player to 240.
    assert out["pred_minutes"][0] == pytest.approx(240.0)
    assert out["pred_minutes"][1] == pytest.approx(240.0)
