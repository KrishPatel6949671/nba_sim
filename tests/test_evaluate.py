"""Tests for ``nba_sim.training.evaluate``.

Strategy: train a tiny model on synthetic data via the test_train_loop
fixture, then exercise every public function in ``evaluate`` against the
resulting checkpoint. Assertions cover schema, ordering, value bounds,
and the end-to-end orchestration (metrics JSON + PNGs land on disk).
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
import torch

from nba_sim.training.evaluate import (
    _PLAYER_STATS,
    eval_dataset_n_games,
    evaluate,
    predict_constraint_samples,
    predict_intervals,
    predict_means,
    predict_team_aggregates,
)
from nba_sim.training.loop import train
from nba_sim.training.metrics import per_stat_mae

# Reuse train-loop fixtures.
from tests.test_model_shapes import _model_config
from tests.test_train_loop import _base_train_config, _write_synthetic


# ---------------------------------------------------------------------------
# Shared fixture: a tiny trained checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_setup(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Train for 2 epochs on synthetic data; share the checkpoint across
    tests in this module so we don't pay the training cost per-test."""
    tmp = tmp_path_factory.mktemp("eval_fixture")
    _write_synthetic(tmp / "train.parquet", n_games=20)
    _write_synthetic(tmp / "val.parquet", n_games=8, seed_offset=10_000)
    cfg = _base_train_config(max_epochs=2, pretrain=0)
    train(
        train_parquet=tmp / "train.parquet",
        val_parquet=tmp / "val.parquet",
        model_config=_model_config(),
        train_config=cfg,
        out_dir=tmp / "ckpts",
        verbose=False,
    )
    return {
        "train_parquet": tmp / "train.parquet",
        "val_parquet": tmp / "val.parquet",
        "checkpoint": tmp / "ckpts" / "best.pt",
        "report_dir": tmp / "reports",
    }


# ---------------------------------------------------------------------------
# predict_means
# ---------------------------------------------------------------------------


def test_predict_means_schema_and_id_cols(trained_setup: dict[str, Path]) -> None:
    df = predict_means(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    assert df.height > 0
    for col in ("game_id", "player_id", "team_id"):
        assert col in df.columns, col
    # pred_<stat> for every player-stat is present.
    for s in _PLAYER_STATS:
        assert f"pred_{s}" in df.columns, f"missing pred_{s}"


def test_predict_means_row_count_matches_active_players(trained_setup: dict[str, Path]) -> None:
    df = predict_means(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    actuals = pl.read_parquet(trained_setup["val_parquet"])
    # Prediction rows == active player-rows in the parquet (synthetic data
    # is is_active=True everywhere).
    assert df.height == actuals.height


def test_predict_means_values_are_non_negative_for_counts(trained_setup: dict[str, Path]) -> None:
    """All count predictions should be ≥ 0 (NB.mean is non-negative)."""
    df = predict_means(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    for s in ("fga", "tpa", "fta", "fgm", "tpm", "ftm",
              "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "minutes", "pts"):
        col = df[f"pred_{s}"]
        assert col.min() >= 0.0, f"pred_{s} has negative values: min={col.min()}"


def test_predict_means_conditional_makes_le_attempts(trained_setup: dict[str, Path]) -> None:
    """E[FGM] = σ(logits)·E[FGA] ≤ E[FGA]. Same for 3PM/FTM."""
    df = predict_means(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    assert (df["pred_fgm"] <= df["pred_fga"] + 1e-6).all()
    assert (df["pred_tpm"] <= df["pred_tpa"] + 1e-6).all()
    assert (df["pred_ftm"] <= df["pred_fta"] + 1e-6).all()


def test_predict_means_derived_pts_matches_definition(trained_setup: dict[str, Path]) -> None:
    df = predict_means(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    recomputed = 2.0 * df["pred_fgm"] + df["pred_tpm"] + df["pred_ftm"]
    diff = (df["pred_pts"] - recomputed).abs()
    assert diff.max() < 1e-5


# ---------------------------------------------------------------------------
# predict_intervals
# ---------------------------------------------------------------------------


def test_predict_intervals_low_le_high(trained_setup: dict[str, Path]) -> None:
    low, high = predict_intervals(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        n_samples=20,
        batch_size=4,
        device="cpu",
        seed=0,
    )
    assert low.height == high.height > 0
    for s in _PLAYER_STATS:
        lo_col = low[f"pred_{s}"]
        hi_col = high[f"pred_{s}"]
        # Allow exact equality (rare on a tiny K=20 sample) but never lo > hi.
        diff = hi_col - lo_col
        assert diff.min() >= -1e-6, f"pred_{s}: low > high somewhere"


def test_predict_intervals_deterministic_under_seed(trained_setup: dict[str, Path]) -> None:
    """Same seed → identical quantile DataFrames."""
    a_low, a_high = predict_intervals(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        n_samples=15, batch_size=4, device="cpu", seed=42,
    )
    b_low, b_high = predict_intervals(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        n_samples=15, batch_size=4, device="cpu", seed=42,
    )
    for s in ("fga", "fgm", "pts", "minutes"):
        assert (a_low[f"pred_{s}"] - b_low[f"pred_{s}"]).abs().max() < 1e-6
        assert (a_high[f"pred_{s}"] - b_high[f"pred_{s}"]).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# predict_team_aggregates
# ---------------------------------------------------------------------------


def test_predict_team_aggregates_two_rows_per_game(trained_setup: dict[str, Path]) -> None:
    df = predict_team_aggregates(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    n_games = eval_dataset_n_games(trained_setup["val_parquet"])
    assert df.height == 2 * n_games
    assert set(df["side"].unique().to_list()) == {"home", "away"}
    # pace and off_rtg present + finite
    for col in ("pred_pace", "pace", "pred_off_rtg", "off_rtg"):
        v = df[col]
        assert v.is_finite().all(), f"{col} has non-finite values"


# ---------------------------------------------------------------------------
# predict_constraint_samples
# ---------------------------------------------------------------------------


def test_predict_constraint_samples_are_clean(trained_setup: dict[str, Path]) -> None:
    """Even an under-trained model must produce constraint-clean samples
    (by construction, per the sampler)."""
    from nba_sim.training.metrics import constraint_violation_rate

    df = predict_constraint_samples(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
        seed=0,
    )
    rates = constraint_violation_rate(df)
    for k, v in rates.items():
        assert v == 0.0, f"{k} violated on {v*100:.2f}% of games"


# ---------------------------------------------------------------------------
# evaluate (orchestration)
# ---------------------------------------------------------------------------


def test_evaluate_writes_metrics_json_and_pngs(trained_setup: dict[str, Path]) -> None:
    report_dir = trained_setup["report_dir"]
    summary = evaluate(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        report_dir=report_dir,
        n_interval_samples=15,
        batch_size=4,
        device="cpu",
        seed=0,
    )
    # Top-level metric keys per PLAN §6.2.
    for k in ("per_stat_mae", "per_stat_rmse", "interval_coverage",
              "team_pts_mae", "team_pts_mae_team_head",
              "pace_mae", "off_rtg_mae",
              "constraint_violation_rate", "reliability_bins"):
        assert k in summary, k

    # Files landed on disk.
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "team_pts_scatter.png").exists()
    rel_pngs = sorted(report_dir.glob("reliability_*.png"))
    assert len(rel_pngs) >= 10   # one per scored stat

    # Round-trip via JSON: load it back and confirm scalar metrics match.
    with open(report_dir / "metrics.json") as f:
        loaded = json.load(f)
    assert loaded["team_pts_mae"] == pytest.approx(summary["team_pts_mae"])
    assert loaded["pace_mae"] == pytest.approx(summary["pace_mae"])
    assert set(loaded["per_stat_mae"]) == set(summary["per_stat_mae"])


def test_evaluate_per_stat_mae_matches_independent_call(trained_setup: dict[str, Path]) -> None:
    """``evaluate``'s per-stat MAE must equal a fresh
    ``per_stat_mae(actuals, predict_means(...))`` call — catches any
    drift between the orchestration path and the underlying metric."""
    means_df = predict_means(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        batch_size=4,
        device="cpu",
    )
    actuals = pl.read_parquet(trained_setup["val_parquet"])
    stats_to_score = tuple(s for s in _PLAYER_STATS if s in actuals.columns)
    direct = per_stat_mae(actuals, means_df, stats=stats_to_score)

    summary = evaluate(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        report_dir=trained_setup["report_dir"] / "drift_check",
        n_interval_samples=10,
        batch_size=4,
        device="cpu",
        seed=0,
        include_plots=False,
    )
    for stat, mae in direct.items():
        assert summary["per_stat_mae"][stat] == pytest.approx(mae, abs=1e-6)


def test_evaluate_no_plots_skips_pngs(trained_setup: dict[str, Path]) -> None:
    """``include_plots=False`` produces metrics.json but no PNGs."""
    out = trained_setup["report_dir"] / "no_plots"
    summary = evaluate(
        checkpoint=trained_setup["checkpoint"],
        parquet=trained_setup["val_parquet"],
        train_parquet=trained_setup["train_parquet"],
        report_dir=out,
        n_interval_samples=10,
        batch_size=4,
        device="cpu",
        seed=0,
        include_plots=False,
    )
    assert (out / "metrics.json").exists()
    assert not (out / "team_pts_scatter.png").exists()
    assert not list(out.glob("reliability_*.png"))
    # And the artifacts dict reflects the choice.
    assert "reliability_pngs" not in summary["_artifacts"]
    assert "team_pts_scatter_png" not in summary["_artifacts"]
