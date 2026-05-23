"""Tests for ``nba_sim.training.metrics``.

Toy DataFrames with hand-checked answers — these metrics are pure data
manipulation, so the assertions are exact numerical comparisons rather
than tolerance-based.
"""

from __future__ import annotations

import polars as pl
import pytest

from nba_sim.training.metrics import (
    constraint_violation_rate,
    interval_coverage,
    per_stat_mae,
    per_stat_rmse,
    reliability_bins,
    team_pts_mae,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _actual_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "game_id":  [1, 1, 2, 2],
            "player_id":[1, 2, 1, 2],
            "team_id":  [10, 10, 20, 20],
            "pts":      [20.0, 10.0, 30.0, 5.0],
            "reb":      [5.0, 8.0, 3.0, 6.0],
        }
    )


def _predicted_df() -> pl.DataFrame:
    # pts errors: 2, 2, 2, 2   → MAE = 2; RMSE = 2
    # reb errors: 0, 4, 0, 4   → MAE = 2; RMSE = sqrt(8) ≈ 2.828
    return pl.DataFrame(
        {
            "game_id":  [1, 1, 2, 2],
            "player_id":[1, 2, 1, 2],
            "team_id":  [10, 10, 20, 20],
            "pred_pts": [22.0, 12.0, 28.0, 7.0],
            "pred_reb": [5.0, 4.0, 3.0, 2.0],
        }
    )


# ----------------------------------------------------------------------------
# per_stat_mae / per_stat_rmse
# ----------------------------------------------------------------------------


def test_per_stat_mae_hand_checked() -> None:
    out = per_stat_mae(_actual_df(), _predicted_df())
    assert out["pts"] == pytest.approx(2.0)
    assert out["reb"] == pytest.approx(2.0)


def test_per_stat_mae_auto_infers_stats() -> None:
    out = per_stat_mae(_actual_df(), _predicted_df())
    # Should have inferred both pts and reb from pred_* columns.
    assert set(out.keys()) == {"pts", "reb"}


def test_per_stat_mae_respects_explicit_stats_subset() -> None:
    out = per_stat_mae(_actual_df(), _predicted_df(), stats=("pts",))
    assert set(out.keys()) == {"pts"}


def test_per_stat_rmse_hand_checked() -> None:
    out = per_stat_rmse(_actual_df(), _predicted_df())
    assert out["pts"] == pytest.approx(2.0)            # all errors = 2
    assert out["reb"] == pytest.approx((32 / 4) ** 0.5)  # sqrt((0+16+0+16)/4)


def test_per_stat_mae_uses_inner_join() -> None:
    """Rows with no matching prediction must be silently dropped, not
    silently treated as missing → keeps the contract honest."""
    actual = _actual_df()
    predicted = _predicted_df().head(2)   # only first 2 rows have predictions
    out = per_stat_mae(actual, predicted)
    # Only 2 rows joined; errors are 2 and 2 → MAE 2.0
    assert out["pts"] == pytest.approx(2.0)


# ----------------------------------------------------------------------------
# interval_coverage
# ----------------------------------------------------------------------------


def test_interval_coverage_all_inside_returns_1() -> None:
    actual = _actual_df()
    low = _predicted_df().with_columns(pl.col("pred_pts") - 50, pl.col("pred_reb") - 50)
    high = _predicted_df().with_columns(pl.col("pred_pts") + 50, pl.col("pred_reb") + 50)
    out = interval_coverage(actual, low, high)
    assert out["pts"] == 1.0
    assert out["reb"] == 1.0


def test_interval_coverage_all_outside_returns_0() -> None:
    actual = _actual_df()
    # All bounds positioned so actual is outside.
    low = _predicted_df().with_columns(
        pl.lit(100.0).alias("pred_pts"), pl.lit(100.0).alias("pred_reb")
    )
    high = _predicted_df().with_columns(
        pl.lit(200.0).alias("pred_pts"), pl.lit(200.0).alias("pred_reb")
    )
    out = interval_coverage(actual, low, high)
    assert out["pts"] == 0.0
    assert out["reb"] == 0.0


def test_interval_coverage_partial() -> None:
    actual = _actual_df()
    # Build bounds so exactly 3/4 rows fall inside for pts.
    # actuals: 20, 10, 30, 5
    # low:     15, 15,  0, 0
    # high:    25, 25, 40, 40
    # Row 1: 15<=20<=25 → in. Row 2: 10<15 → OUT. Rows 3,4: in. → 3/4.
    low = pl.DataFrame({
        "game_id":  [1, 1, 2, 2],
        "player_id":[1, 2, 1, 2],
        "team_id":  [10, 10, 20, 20],
        "pred_pts": [15.0, 15.0, 0.0, 0.0],
    })
    high = pl.DataFrame({
        "game_id":  [1, 1, 2, 2],
        "player_id":[1, 2, 1, 2],
        "team_id":  [10, 10, 20, 20],
        "pred_pts": [25.0, 25.0, 40.0, 40.0],
    })
    out = interval_coverage(actual, low, high, stats=("pts",))
    assert out["pts"] == pytest.approx(0.75)


# ----------------------------------------------------------------------------
# reliability_bins
# ----------------------------------------------------------------------------


def test_reliability_bins_shape_and_columns() -> None:
    out = reliability_bins(_actual_df(), _predicted_df(), n_bins=2)
    assert set(out.columns) == {"stat", "bin", "pred_mean", "observed_mean", "count"}
    # 2 bins × 2 stats = 4 rows
    assert out.height == 4


def test_reliability_bins_count_sums_match_input() -> None:
    actual = _actual_df()
    pred = _predicted_df()
    out = reliability_bins(actual, pred, n_bins=2)
    # All rows should be accounted for in each stat.
    for s in ("pts", "reb"):
        total = out.filter(pl.col("stat") == s)["count"].sum()
        assert total == actual.height


def test_reliability_bins_empty_returns_empty() -> None:
    empty_actual = _actual_df().head(0)
    empty_pred = _predicted_df().head(0)
    out = reliability_bins(empty_actual, empty_pred, n_bins=10)
    assert out.height == 0
    assert set(out.columns) == {"stat", "bin", "pred_mean", "observed_mean", "count"}


# ----------------------------------------------------------------------------
# constraint_violation_rate
# ----------------------------------------------------------------------------


def _clean_samples() -> pl.DataFrame:
    """Two games, two teams each, two players per team. Everything legal."""
    return pl.DataFrame(
        {
            "game_id":  [1, 1, 1, 1, 2, 2, 2, 2],
            "team_id":  [10, 10, 11, 11, 20, 20, 21, 21],
            "player_id":[1, 2, 3, 4, 5, 6, 7, 8],
            "minutes":  [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0],
            "fga":      [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "fgm":      [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "tpa":      [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "tpm":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "fta":      [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "ftm":      [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "oreb":     [1.0] * 8, "dreb": [2.0] * 8, "ast": [3.0] * 8,
            "stl":      [1.0] * 8, "blk": [0.0] * 8, "tov": [2.0] * 8, "pf": [3.0] * 8,
        }
    )


def test_constraint_violation_rate_clean_returns_all_zero() -> None:
    out = constraint_violation_rate(_clean_samples())
    for k, v in out.items():
        assert v == 0.0, f"{k} = {v} (expected 0)"


def test_constraint_violation_minutes_sum_violation() -> None:
    """Set team-10's minutes to sum to 200 instead of 240 → game 1 violates."""
    samples = _clean_samples().with_columns(
        pl.when((pl.col("team_id") == 10) & (pl.col("player_id") == 1))
          .then(pl.lit(80.0))
          .otherwise(pl.col("minutes"))
          .alias("minutes")
    )
    out = constraint_violation_rate(samples)
    # 1 of 2 games has a minutes-sum violation.
    assert out["minutes_sum_240"] == pytest.approx(0.5)


def test_constraint_violation_fgm_greater_than_fga() -> None:
    samples = _clean_samples().with_columns(
        pl.when(pl.col("player_id") == 1).then(pl.lit(15.0)).otherwise(pl.col("fgm")).alias("fgm")
    )
    out = constraint_violation_rate(samples)
    # Only game 1 (player 1 is in it).
    assert out["fgm_le_fga"] == pytest.approx(0.5)


def test_constraint_violation_3pm_greater_than_fgm() -> None:
    samples = _clean_samples().with_columns(
        pl.when(pl.col("player_id") == 5).then(pl.lit(10.0)).otherwise(pl.col("tpm")).alias("tpm")
    )
    out = constraint_violation_rate(samples)
    # Player 5 is in game 2.
    assert out["tpm_le_fgm"] == pytest.approx(0.5)


def test_constraint_violation_negative_count() -> None:
    samples = _clean_samples().with_columns(
        pl.when(pl.col("player_id") == 1).then(pl.lit(-1.0)).otherwise(pl.col("ast")).alias("ast")
    )
    out = constraint_violation_rate(samples)
    assert out["non_negative"] == pytest.approx(0.5)


def test_constraint_violation_non_integer_count() -> None:
    samples = _clean_samples().with_columns(
        pl.when(pl.col("player_id") == 1).then(pl.lit(2.7)).otherwise(pl.col("ast")).alias("ast")
    )
    out = constraint_violation_rate(samples)
    assert out["integer_counts"] == pytest.approx(0.5)


def test_constraint_violation_empty_returns_empty_dict() -> None:
    empty = _clean_samples().head(0)
    out = constraint_violation_rate(empty)
    assert out == {}


# ----------------------------------------------------------------------------
# team_pts_mae
# ----------------------------------------------------------------------------


def test_team_pts_mae_hand_checked() -> None:
    actual = _actual_df()
    pred = _predicted_df()
    # Team 10: actual pts 20+10=30, pred 22+12=34, |err|=4
    # Team 20: actual pts 30+5=35,  pred 28+7=35,  |err|=0
    # MAE = (4 + 0) / 2 = 2.0
    out = team_pts_mae(actual, pred)
    assert out == pytest.approx(2.0)
