"""Evaluation metrics — per-stat MAE, calibration, constraint-violation counters.

See PLAN.md §6.2 for the full list. Every metric here takes polars DataFrames
of (actual, predicted) rows and returns a scalar or a dict of scalars; no
torch dependencies so these can run outside the training process.
"""

from __future__ import annotations

import polars as pl


def per_stat_mae(actual: pl.DataFrame, predicted: pl.DataFrame) -> dict[str, float]:
    """Per-stat mean absolute error."""
    raise NotImplementedError


def per_stat_rmse(actual: pl.DataFrame, predicted: pl.DataFrame) -> dict[str, float]:
    """Per-stat root mean squared error."""
    raise NotImplementedError


def interval_coverage(
    actual: pl.DataFrame,
    low: pl.DataFrame,
    high: pl.DataFrame,
    target: float = 0.80,
) -> dict[str, float]:
    """Fraction of actual values inside the [low, high] predictive interval."""
    raise NotImplementedError


def reliability_bins(
    actual: pl.DataFrame,
    predicted_mean: pl.DataFrame,
    n_bins: int = 10,
) -> pl.DataFrame:
    """Buckets of predicted mean vs. observed mean — for reliability plots."""
    raise NotImplementedError


def constraint_violation_rate(samples: pl.DataFrame) -> dict[str, float]:
    """Count constraint violations per-sample. Should be 0 for the NN sampler."""
    raise NotImplementedError


def team_pts_mae(actual: pl.DataFrame, predicted: pl.DataFrame) -> float:
    """Team PTS MAE rolled up from per-player predictions."""
    raise NotImplementedError
