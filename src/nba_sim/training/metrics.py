"""Evaluation metrics — per-stat MAE, calibration, constraint-violation counters.

See PLAN.md §6.2 for the full list. Every metric takes polars DataFrames
of (actual, predicted) rows and returns a scalar or a dict of scalars —
no torch dependencies so these can run outside the training process and
the GLM baseline + the NN can be evaluated with literally the same code.

Conventions:

- ``actual`` has columns named after stats (``fga``, ``fgm``, ``pts`` ...).
- ``predicted`` has matching ``pred_<stat>`` columns.
- Joins are by ``(game_id, player_id, team_id)`` for per-player metrics,
  ``(game_id, team_id)`` for team-rollup metrics. Always inner-join —
  silent row drops are bugs, not features.
"""

from __future__ import annotations

import polars as pl

DEFAULT_ID_COLS: tuple[str, ...] = ("game_id", "player_id", "team_id")

# Count columns we routinely check constraints on.
_COUNT_COLS: tuple[str, ...] = (
    "fga", "fgm", "tpa", "tpm", "fta", "ftm",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
)


def _infer_stats(predicted: pl.DataFrame, stats: tuple[str, ...] | None) -> tuple[str, ...]:
    if stats is not None:
        return tuple(stats)
    return tuple(c[len("pred_") :] for c in predicted.columns if c.startswith("pred_"))


def per_stat_mae(
    actual: pl.DataFrame,
    predicted: pl.DataFrame,
    stats: tuple[str, ...] | None = None,
    on: tuple[str, ...] = DEFAULT_ID_COLS,
) -> dict[str, float]:
    """Per-stat mean absolute error on the inner-join of (actual, predicted).

    If ``stats`` is None, infers stats from ``pred_<stat>`` columns in
    ``predicted``.
    """
    stats = _infer_stats(predicted, stats)
    joined = actual.join(predicted, on=list(on), how="inner")
    return {
        s: float((joined[s].cast(pl.Float64) - joined[f"pred_{s}"]).abs().mean())
        for s in stats
    }


def per_stat_rmse(
    actual: pl.DataFrame,
    predicted: pl.DataFrame,
    stats: tuple[str, ...] | None = None,
    on: tuple[str, ...] = DEFAULT_ID_COLS,
) -> dict[str, float]:
    """Per-stat root mean squared error."""
    stats = _infer_stats(predicted, stats)
    joined = actual.join(predicted, on=list(on), how="inner")
    return {
        s: float(((joined[s].cast(pl.Float64) - joined[f"pred_{s}"]) ** 2).mean() ** 0.5)
        for s in stats
    }


def interval_coverage(
    actual: pl.DataFrame,
    low: pl.DataFrame,
    high: pl.DataFrame,
    target: float = 0.80,   # noqa: ARG001 — documentation-only, declared intent
    stats: tuple[str, ...] | None = None,
    on: tuple[str, ...] = DEFAULT_ID_COLS,
) -> dict[str, float]:
    """Fraction of actual values inside [low, high] per stat.

    ``low`` and ``high`` use the same ``pred_<stat>`` schema as point-
    prediction DataFrames; ``target`` documents the intended PI width
    (e.g. 0.80 for an 80% interval) but does not gate behavior.
    """
    stats = _infer_stats(low, stats)
    low_renamed = low.rename({f"pred_{s}": f"low_{s}" for s in stats})
    high_renamed = high.rename({f"pred_{s}": f"high_{s}" for s in stats})
    joined = actual.join(low_renamed, on=list(on), how="inner").join(
        high_renamed, on=list(on), how="inner"
    )
    out: dict[str, float] = {}
    for s in stats:
        a = joined[s].cast(pl.Float64)
        in_interval = (a >= joined[f"low_{s}"]) & (a <= joined[f"high_{s}"])
        out[s] = float(in_interval.mean())
    return out


def reliability_bins(
    actual: pl.DataFrame,
    predicted_mean: pl.DataFrame,
    n_bins: int = 10,
    stats: tuple[str, ...] | None = None,
    on: tuple[str, ...] = DEFAULT_ID_COLS,
) -> pl.DataFrame:
    """Per-stat reliability buckets of predicted mean vs. observed mean.

    For each stat, sort rows by ``pred_<stat>`` and partition into
    ``n_bins`` equal-count buckets (rank-based to avoid relying on the
    polars ``qcut`` API). Per bucket: mean(pred), mean(actual), count.
    Returns a long-format DataFrame with columns
    ``[stat, bin, pred_mean, observed_mean, count]``.
    """
    stats = _infer_stats(predicted_mean, stats)
    joined = actual.join(predicted_mean, on=list(on), how="inner")
    n = joined.height
    if n == 0:
        return pl.DataFrame(
            schema={
                "stat": pl.String, "bin": pl.Int64,
                "pred_mean": pl.Float64, "observed_mean": pl.Float64,
                "count": pl.UInt32,
            }
        )

    rows: list[pl.DataFrame] = []
    for s in stats:
        sorted_df = joined.sort(f"pred_{s}").with_row_index("_idx")
        binned = sorted_df.with_columns(
            (pl.col("_idx") * n_bins // n).cast(pl.Int64).clip(0, n_bins - 1).alias("bin")
        )
        agg = (
            binned.group_by("bin")
            .agg(
                pl.col(f"pred_{s}").mean().alias("pred_mean"),
                pl.col(s).cast(pl.Float64).mean().alias("observed_mean"),
                pl.len().alias("count"),
            )
            .with_columns(pl.lit(s).alias("stat"))
            .sort("bin")
        )
        rows.append(agg.select(["stat", "bin", "pred_mean", "observed_mean", "count"]))
    return pl.concat(rows)


def constraint_violation_rate(
    samples: pl.DataFrame,
    minutes_tol: float = 0.5,
    integer_tol: float = 1e-6,
) -> dict[str, float]:
    """Per-game constraint-violation rates over ``samples``.

    A game is counted as violating a constraint if *any* row/team within
    it breaks the rule. Reports the fraction of distinct ``game_id``\\ s
    in violation. Should be 0.00 for the NN sampler — see PLAN.md §1 and
    §4.4.

    Constraints checked (see PLAN §9):
    - ``minutes_sum_240``: per-team minutes sum within ``minutes_tol`` of 240
    - ``fgm_le_fga``, ``tpm_le_tpa``, ``ftm_le_fta``: makes ≤ attempts
    - ``tpm_le_fgm``: 3-pointers are field goals
    - ``non_negative``: all count columns ≥ 0
    - ``integer_counts``: every count is within ``integer_tol`` of an integer
    """
    n_games = samples["game_id"].n_unique()
    if n_games == 0:
        return {}

    out: dict[str, float] = {}

    team_min = samples.group_by(["game_id", "team_id"]).agg(
        pl.col("minutes").sum().alias("ts")
    )
    bad = team_min.filter((pl.col("ts") - 240.0).abs() > minutes_tol)["game_id"].n_unique()
    out["minutes_sum_240"] = bad / n_games

    row_checks = {
        "fgm_le_fga": pl.col("fgm") > pl.col("fga"),
        "tpm_le_tpa": pl.col("tpm") > pl.col("tpa"),
        "ftm_le_fta": pl.col("ftm") > pl.col("fta"),
        "tpm_le_fgm": pl.col("tpm") > pl.col("fgm"),
    }
    for name, expr in row_checks.items():
        bad = samples.filter(expr)["game_id"].n_unique()
        out[name] = bad / n_games

    present_counts = [c for c in _COUNT_COLS if c in samples.columns]
    if present_counts:
        any_neg = samples.with_columns(
            pl.any_horizontal([pl.col(c) < 0 for c in present_counts]).alias("_neg")
        )
        bad = any_neg.filter(pl.col("_neg"))["game_id"].n_unique()
        out["non_negative"] = bad / n_games

        non_int = samples.with_columns(
            pl.any_horizontal(
                [(pl.col(c) - pl.col(c).round()).abs() > integer_tol for c in present_counts]
            ).alias("_noint")
        )
        bad = non_int.filter(pl.col("_noint"))["game_id"].n_unique()
        out["integer_counts"] = bad / n_games

    return out


def team_pts_mae(
    actual: pl.DataFrame,
    predicted: pl.DataFrame,
    on: tuple[str, ...] = ("game_id", "team_id"),
) -> float:
    """Team PTS MAE rolled up by summing per-player predictions.

    Expects player-level rows in both inputs with ``pts`` / ``pred_pts``
    columns. Groups by ``on`` and sums, then MAEs the team totals.
    """
    team_actual = actual.group_by(list(on)).agg(
        pl.col("pts").cast(pl.Float64).sum().alias("team_pts")
    )
    team_pred = predicted.group_by(list(on)).agg(
        pl.col("pred_pts").sum().alias("team_pred_pts")
    )
    joined = team_actual.join(team_pred, on=list(on), how="inner")
    return float((joined["team_pts"] - joined["team_pred_pts"]).abs().mean())


def team_pts_mae_team_head(
    actual: pl.DataFrame,
    team_predicted: pl.DataFrame,
    on: tuple[str, ...] = ("game_id", "team_id"),
) -> float:
    """Team PTS MAE using the team head's direct estimate.

    ``team_predicted`` is the per-(game,team) DataFrame returned by
    :func:`nba_sim.training.evaluate.predict_team_aggregates` — it carries
    ``pred_pace`` and ``pred_off_rtg`` per team-game. Predicted team PTS
    is ``pred_pace * pred_off_rtg / 100``.

    Empirically (see scripts/team_pts_diagnostic.py) this beats the
    sum-of-players path by ~3-4 MAE units because summing 8-10 noisy
    per-player count draws compounds variance — the team head emits a
    single direct estimate of an inherently team-level quantity.
    """
    team_actual = actual.group_by(list(on)).agg(
        pl.col("pts").cast(pl.Float64).sum().alias("team_pts")
    )
    team_pred = team_predicted.with_columns(
        (pl.col("pred_pace") * pl.col("pred_off_rtg") / 100.0).alias("team_pred_pts")
    ).select([*on, "team_pred_pts"])
    joined = team_actual.join(team_pred, on=list(on), how="inner")
    return float((joined["team_pts"] - joined["team_pred_pts"]).abs().mean())
