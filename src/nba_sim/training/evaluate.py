"""Full evaluation pass — PLAN §6.

Materialize per-player and per-team predictions from a trained checkpoint,
score against actuals, and produce the metrics JSON + diagnostic plots
the Phase 2 ship gate (PLAN §10) and the v1 success criteria (PLAN §1)
both read.

Public surface
--------------

- :func:`predict_means` — DataFrame of per-player expected values
  (``pred_<stat>`` columns).
- :func:`predict_intervals` — pair of DataFrames at the 10th/90th
  percentile per stat, computed by sampling K box scores via
  :func:`nba_sim.simulate.sampler.sample_raw_box_score` and taking
  empirical quantiles per slot. Uses the *simulator's* distribution so
  calibration is measured against the same thing inference produces.
- :func:`predict_team_aggregates` — per-game-team DataFrame with
  ``pred_pace``, ``pred_off_rtg``, ``pred_team_pts`` and matching actuals.
- :func:`evaluate` — orchestrates everything, writes
  ``reports/metrics.json`` plus reliability and team-PTS scatter PNGs.

Conventions
-----------

- Splits are passed by parquet path; ``train_parquet`` is required so the
  eval dataset reuses the train-fitted ``player_id_map`` + ``feature_stats``
  (no test-set leakage into normalization stats).
- All predictions are deterministic w.r.t. the seed (``torch.manual_seed``
  before any sampling).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import torch

from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.simulate.sampler import sample_raw_box_score
from nba_sim.training.dataset import BoxScoreDataset, collate_games
from nba_sim.training.metrics import (
    constraint_violation_rate,
    interval_coverage,
    per_stat_mae,
    per_stat_rmse,
    reliability_bins,
    team_pts_mae,
    team_pts_mae_team_head,
)


# ---------------------------------------------------------------------------
# Stat lists (kept here so this module is self-contained)
# ---------------------------------------------------------------------------

# NB-headed counting stats produced by the model per player.
_NB_STATS: tuple[str, ...] = (
    "fga", "tpa", "fta", "oreb", "dreb",
    "ast", "stl", "blk", "tov", "pf",
)
# Conditional-Binomial makes; each is (make_stat, attempt_stat).
_MAKE_PAIRS: tuple[tuple[str, str], ...] = (("fgm", "fga"), ("tpm", "tpa"), ("ftm", "fta"))
# Every per-player stat we score (includes derived pts + reb).
_PLAYER_STATS: tuple[str, ...] = (
    "minutes",
    "fga", "fgm", "tpa", "tpm", "fta", "ftm",
    "oreb", "dreb", "reb",
    "ast", "stl", "blk", "tov", "pf",
    "pts",
)
_ID_COLS: tuple[str, ...] = ("game_id", "player_id", "team_id")


# ---------------------------------------------------------------------------
# Device + iteration helpers
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _load_for_eval(
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    device: torch.device,
) -> tuple[HierarchicalBoxScoreModel, BoxScoreDataset]:
    """Reconstruct model + dataset with train-fitted normalization."""
    train_ds = BoxScoreDataset(train_parquet)
    eval_ds = BoxScoreDataset(
        parquet,
        player_id_map=train_ds.player_id_map,
        feature_stats=train_ds.feature_stats,
    )
    model = HierarchicalBoxScoreModel.from_checkpoint(checkpoint).to(device).eval()
    return model, eval_ds


def _iterate_in_batches(
    dataset: BoxScoreDataset, batch_size: int
):
    """Yield ``(batch_dict, start_idx)`` tuples in dataset order."""
    n = len(dataset)
    for start in range(0, n, batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, n))]
        yield collate_games(items), start


# ---------------------------------------------------------------------------
# Per-player predicted means
# ---------------------------------------------------------------------------


def _expected_makes(
    logits: torch.Tensor, attempts_mean: torch.Tensor
) -> torch.Tensor:
    """E[Binomial(n=E[attempts], p=σ(logits))] = σ(logits) · E[attempts]."""
    return torch.sigmoid(logits) * attempts_mean


def _dist_means_to_player_rows(
    *,
    dataset: BoxScoreDataset,
    start_idx: int,
    batch: dict[str, torch.Tensor],
    dist,
) -> list[dict[str, Any]]:
    """Materialize per-active-slot prediction rows for one mini-batch.

    Uses ``dataset._games[idx]`` to pull the (game_id, player_id, team_id)
    metadata in the same slot order the tensors use (sorted minutes desc).
    """
    rows: list[dict[str, Any]] = []
    B, P = batch["home_mask"].shape

    # Cache tensor handles per side once per batch.
    side_handles: dict[str, dict[str, torch.Tensor]] = {}
    for side in ("home", "away"):
        side_handles[side] = {
            "mask":     batch[f"{side}_mask"].cpu(),
            "minutes":  (getattr(dist, f"minutes_{side}").mean * 240.0).cpu(),
            **{s: getattr(dist, f"{s}_{side}").mean.cpu() for s in _NB_STATS},
        }
        # Conditional makes via E[FGM] = σ(logits) · E[FGA] etc.
        for make, attempt in _MAKE_PAIRS:
            side_handles[side][make] = _expected_makes(
                getattr(dist, f"{make}_probs_{side}").cpu(),
                side_handles[side][attempt],
            )
        side_handles[side]["pts"] = (
            2.0 * side_handles[side]["fgm"]
            + side_handles[side]["tpm"]
            + side_handles[side]["ftm"]
        )
        side_handles[side]["reb"] = (
            side_handles[side]["oreb"] + side_handles[side]["dreb"]
        )

    for b in range(B):
        game_idx = start_idx + b
        game_df = dataset._games[game_idx]
        home_df = game_df.filter(pl.col("is_home")).head(dataset.max_players)
        away_df = game_df.filter(~pl.col("is_home")).head(dataset.max_players)
        for side, sdf in (("home", home_df), ("away", away_df)):
            handle = side_handles[side]
            mask_b = handle["mask"][b]
            n_active = sdf.height
            for i in range(n_active):
                if not bool(mask_b[i]):
                    # Defensive: dataset truncates to max_players → should
                    # never trip, but skip rather than emit a padded row.
                    continue
                row = {
                    "game_id":   sdf["game_id"][i],
                    "player_id": int(sdf["player_id"][i]),
                    "team_id":   int(sdf["team_id"][i]),
                }
                for s in _PLAYER_STATS:
                    row[f"pred_{s}"] = float(handle[s][b, i])
                rows.append(row)
    return rows


def predict_means(
    *,
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    batch_size: int = 64,
    device: str = "auto",
) -> pl.DataFrame:
    """Materialize per-active-player expected-value predictions.

    Returns a DataFrame with columns ``(game_id, player_id, team_id,
    pred_<stat> ...)`` ready to join with the actuals parquet via
    :func:`nba_sim.training.metrics.per_stat_mae` and friends.
    """
    dev = _resolve_device(device)
    model, eval_ds = _load_for_eval(checkpoint, parquet, train_parquet, dev)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch, start in _iterate_in_batches(eval_ds, batch_size):
            batch = _move(batch, dev)
            dist = model(batch)
            rows.extend(
                _dist_means_to_player_rows(
                    dataset=eval_ds, start_idx=start, batch=batch, dist=dist
                )
            )
    return pl.DataFrame(rows) if rows else _empty_pred_df()


def _empty_pred_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "game_id": pl.String, "player_id": pl.Int64, "team_id": pl.Int64,
            **{f"pred_{s}": pl.Float64 for s in _PLAYER_STATS},
        }
    )


# ---------------------------------------------------------------------------
# Per-player sampled intervals (10th / 90th percentiles)
# ---------------------------------------------------------------------------


def _sample_quantile_arrays(
    dist,
    home_mask: torch.Tensor,
    away_mask: torch.Tensor,
    *,
    n_samples: int,
    low_q: float,
    high_q: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Stack K samples per stat per side, return ``{key: (low, high)}``.

    ``key`` is e.g. ``home_fga`` or ``away_pts``; tensors are ``[B, P]``.
    Reuses :func:`sample_raw_box_score` so calibration is measured against
    the same distribution the simulator emits.
    """
    accum: dict[str, list[torch.Tensor]] = defaultdict(list)
    for _ in range(n_samples):
        out = sample_raw_box_score(dist, home_mask, away_mask)
        for k, t in out.items():
            if k in ("pace", "off_rtg"):
                continue  # Team-level intervals handled separately.
            accum[k].append(t.float().cpu())
    qs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k, ts in accum.items():
        stacked = torch.stack(ts, dim=0)  # [K, B, P]
        lo = torch.quantile(stacked, low_q, dim=0)
        hi = torch.quantile(stacked, high_q, dim=0)
        qs[k] = (lo, hi)
    return qs


def predict_intervals(
    *,
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    n_samples: int = 200,
    low_q: float = 0.10,
    high_q: float = 0.90,
    batch_size: int = 32,
    device: str = "auto",
    seed: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Materialize per-player 10th/90th percentile predictions.

    Returns ``(low_df, high_df)`` each with the same ``(game_id, player_id,
    team_id, pred_<stat>)`` schema as :func:`predict_means`. Use directly
    with :func:`nba_sim.training.metrics.interval_coverage`.

    The quantiles are *empirical* over ``n_samples`` draws from
    :func:`sample_raw_box_score` per game, so calibration is measured
    against the simulator's true output distribution (including the
    conditional-Binomial structure and the 240-minute rounding).
    """
    torch.manual_seed(seed)
    dev = _resolve_device(device)
    model, eval_ds = _load_for_eval(checkpoint, parquet, train_parquet, dev)
    low_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch, start in _iterate_in_batches(eval_ds, batch_size):
            batch = _move(batch, dev)
            dist = model(batch)
            qs = _sample_quantile_arrays(
                dist,
                batch["home_mask"],
                batch["away_mask"],
                n_samples=n_samples,
                low_q=low_q,
                high_q=high_q,
            )
            B = batch["home_mask"].shape[0]
            for b in range(B):
                game_idx = start + b
                game_df = eval_ds._games[game_idx]
                home_df = game_df.filter(pl.col("is_home")).head(eval_ds.max_players)
                away_df = game_df.filter(~pl.col("is_home")).head(eval_ds.max_players)
                for side, sdf in (("home", home_df), ("away", away_df)):
                    mask_b = batch[f"{side}_mask"][b].cpu()
                    for i in range(sdf.height):
                        if not bool(mask_b[i]):
                            continue
                        ids = {
                            "game_id":   sdf["game_id"][i],
                            "player_id": int(sdf["player_id"][i]),
                            "team_id":   int(sdf["team_id"][i]),
                        }
                        low_row = dict(ids)
                        high_row = dict(ids)
                        for stat in _PLAYER_STATS:
                            key = f"{side}_{stat}"
                            if key in qs:
                                low_row[f"pred_{stat}"] = float(qs[key][0][b, i])
                                high_row[f"pred_{stat}"] = float(qs[key][1][b, i])
                            else:
                                # Stat wasn't directly sampled — derive
                                # missing entries from constituent samples.
                                # Already handled by sample_raw_box_score
                                # (it emits pts and reb), so we shouldn't
                                # reach here. Fall back to NaN.
                                low_row[f"pred_{stat}"] = float("nan")
                                high_row[f"pred_{stat}"] = float("nan")
                        low_rows.append(low_row)
                        high_rows.append(high_row)
    low_df = pl.DataFrame(low_rows) if low_rows else _empty_pred_df()
    high_df = pl.DataFrame(high_rows) if high_rows else _empty_pred_df()
    return low_df, high_df


# ---------------------------------------------------------------------------
# Team-level predictions
# ---------------------------------------------------------------------------


def predict_team_aggregates(
    *,
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    batch_size: int = 64,
    device: str = "auto",
) -> pl.DataFrame:
    """One row per (game_id, team_id) with predicted + actual pace / off_rtg.

    Columns: ``game_id, team_id, side, pred_pace, pace, pred_off_rtg,
    off_rtg``. ``side`` is ``"home"`` or ``"away"``.
    """
    dev = _resolve_device(device)
    model, eval_ds = _load_for_eval(checkpoint, parquet, train_parquet, dev)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch, start in _iterate_in_batches(eval_ds, batch_size):
            batch = _move(batch, dev)
            dist = model(batch)
            pace_pred = dist.pace.mean.cpu()           # [B]
            off_pred = dist.off_rtg.mean.cpu()         # [B, 2]
            pace_true = batch["pace"].cpu()
            off_true = batch["off_rtg"].cpu()
            B = pace_pred.shape[0]
            for b in range(B):
                game_idx = start + b
                game_df = eval_ds._games[game_idx]
                home_team = int(
                    game_df.filter(pl.col("is_home"))["team_id"][0]
                )
                away_team = int(
                    game_df.filter(~pl.col("is_home"))["team_id"][0]
                )
                gid = game_df["game_id"][0]
                rows.append({
                    "game_id": gid, "team_id": home_team, "side": "home",
                    "pred_pace": float(pace_pred[b]), "pace": float(pace_true[b]),
                    "pred_off_rtg": float(off_pred[b, 0]),
                    "off_rtg": float(off_true[b, 0]),
                })
                rows.append({
                    "game_id": gid, "team_id": away_team, "side": "away",
                    "pred_pace": float(pace_pred[b]), "pace": float(pace_true[b]),
                    "pred_off_rtg": float(off_pred[b, 1]),
                    "off_rtg": float(off_true[b, 1]),
                })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Constraint sampling (one sample per game for the violation-rate metric)
# ---------------------------------------------------------------------------


def predict_constraint_samples(
    *,
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 0,
) -> pl.DataFrame:
    """Sample one box score per game and return the long-format DataFrame
    that :func:`metrics.constraint_violation_rate` consumes."""
    torch.manual_seed(seed)
    dev = _resolve_device(device)
    model, eval_ds = _load_for_eval(checkpoint, parquet, train_parquet, dev)
    rows: list[dict[str, Any]] = []
    stat_keys = ("minutes", "fga", "fgm", "tpa", "tpm", "fta", "ftm",
                 "oreb", "dreb", "ast", "stl", "blk", "tov", "pf")

    with torch.no_grad():
        for batch, start in _iterate_in_batches(eval_ds, batch_size):
            batch = _move(batch, dev)
            dist = model(batch)
            sampled = sample_raw_box_score(
                dist, batch["home_mask"], batch["away_mask"]
            )
            B = batch["home_mask"].shape[0]
            for b in range(B):
                game_idx = start + b
                game_df = eval_ds._games[game_idx]
                home_df = game_df.filter(pl.col("is_home")).head(eval_ds.max_players)
                away_df = game_df.filter(~pl.col("is_home")).head(eval_ds.max_players)
                for side, sdf in (("home", home_df), ("away", away_df)):
                    mask_b = batch[f"{side}_mask"][b].cpu()
                    for i in range(sdf.height):
                        if not bool(mask_b[i]):
                            continue
                        row = {
                            "game_id":   sdf["game_id"][i],
                            "player_id": int(sdf["player_id"][i]),
                            "team_id":   int(sdf["team_id"][i]),
                        }
                        for s in stat_keys:
                            row[s] = float(sampled[f"{side}_{s}"][b, i].cpu())
                        rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _plot_reliability(rel_df: pl.DataFrame, report_dir: Path) -> list[str]:
    """One PNG per stat: predicted-mean bucket vs. observed-mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for stat in rel_df["stat"].unique().sort().to_list():
        sub = rel_df.filter(pl.col("stat") == stat).sort("bin")
        if sub.is_empty():
            continue
        fig, ax = plt.subplots(figsize=(4, 4))
        x = sub["pred_mean"].to_numpy()
        y = sub["observed_mean"].to_numpy()
        ax.scatter(x, y, s=40)
        # y = x reference line spans the data range.
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", linewidth=1)
        ax.set_xlabel(f"predicted {stat}")
        ax.set_ylabel(f"observed {stat}")
        ax.set_title(f"reliability: {stat}")
        fig.tight_layout()
        path = report_dir / f"reliability_{stat}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_team_pts_scatter(
    actuals: pl.DataFrame, predictions: pl.DataFrame, report_dir: Path
) -> str:
    """Per-game-team team-PTS predicted vs. actual scatter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    team_actual = actuals.group_by(["game_id", "team_id"]).agg(
        pl.col("pts").cast(pl.Float64).sum().alias("team_pts")
    )
    team_pred = predictions.group_by(["game_id", "team_id"]).agg(
        pl.col("pred_pts").sum().alias("team_pred_pts")
    )
    joined = team_actual.join(team_pred, on=["game_id", "team_id"], how="inner")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    x = joined["team_pred_pts"].to_numpy()
    y = joined["team_pts"].to_numpy()
    ax.scatter(x, y, s=8, alpha=0.5)
    if len(x):
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("predicted team PTS")
    ax.set_ylabel("actual team PTS")
    ax.set_title("team PTS calibration")
    fig.tight_layout()
    path = report_dir / "team_pts_scatter.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate(
    *,
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    report_dir: str | Path = "reports",
    n_interval_samples: int = 200,
    n_reliability_bins: int = 10,
    batch_size: int = 64,
    device: str = "auto",
    seed: int = 0,
    include_plots: bool = True,
) -> dict[str, Any]:
    """Full eval pass: writes ``report_dir/metrics.json`` and PNGs; returns
    the metrics dict.

    Metrics produced (PLAN §6.2):

    - ``per_stat_mae``       — per-stat (incl. minutes + pts)
    - ``per_stat_rmse``
    - ``team_pts_mae``       — per-game-team
    - ``pace_mae``, ``off_rtg_mae``
    - ``interval_coverage``  — empirical 80% coverage per stat
    - ``constraint_violation_rate``  — 1 sampled box per game
    - ``reliability_bins``   — long-format JSON, plotted to PNG
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---- predictions ------------------------------------------------------
    means_df = predict_means(
        checkpoint=checkpoint, parquet=parquet,
        train_parquet=train_parquet, batch_size=batch_size, device=device,
    )
    low_df, high_df = predict_intervals(
        checkpoint=checkpoint, parquet=parquet,
        train_parquet=train_parquet,
        n_samples=n_interval_samples, batch_size=max(8, batch_size // 4),
        device=device, seed=seed,
    )
    team_df = predict_team_aggregates(
        checkpoint=checkpoint, parquet=parquet,
        train_parquet=train_parquet, batch_size=batch_size, device=device,
    )
    sampled_df = predict_constraint_samples(
        checkpoint=checkpoint, parquet=parquet,
        train_parquet=train_parquet, batch_size=batch_size, device=device,
        seed=seed,
    )

    # ---- actuals ----------------------------------------------------------
    actuals = pl.read_parquet(parquet)

    # ---- metrics ----------------------------------------------------------
    metrics: dict[str, Any] = {}
    stats_to_score = tuple(s for s in _PLAYER_STATS if s in actuals.columns)
    metrics["per_stat_mae"] = per_stat_mae(actuals, means_df, stats=stats_to_score)
    metrics["per_stat_rmse"] = per_stat_rmse(actuals, means_df, stats=stats_to_score)
    metrics["interval_coverage"] = interval_coverage(
        actuals, low_df, high_df, target=0.80, stats=stats_to_score
    )
    # team_pts_mae kept under the legacy name (player-sum path) for
    # backwards-compat with prior reports. team_pts_mae_team_head is the
    # direct estimate from pace * off_rtg / 100 — see metrics.py for why
    # it's strictly better as a team-level predictor.
    metrics["team_pts_mae"] = team_pts_mae(actuals, means_df)
    if not team_df.is_empty():
        metrics["team_pts_mae_team_head"] = team_pts_mae_team_head(actuals, team_df)
    else:
        metrics["team_pts_mae_team_head"] = float("nan")

    # Pace + off_rtg MAE (team-level scalars).
    if not team_df.is_empty():
        pace_err = (team_df["pred_pace"] - team_df["pace"]).abs()
        off_err = (team_df["pred_off_rtg"] - team_df["off_rtg"]).abs()
        metrics["pace_mae"] = float(pace_err.mean())
        metrics["off_rtg_mae"] = float(off_err.mean())
    else:
        metrics["pace_mae"] = float("nan")
        metrics["off_rtg_mae"] = float("nan")

    metrics["constraint_violation_rate"] = constraint_violation_rate(sampled_df)

    # Reliability bins — write JSON-friendly long records.
    rel_df = reliability_bins(
        actuals, means_df, n_bins=n_reliability_bins, stats=stats_to_score
    )
    metrics["reliability_bins"] = rel_df.to_dicts()

    # ---- plots ------------------------------------------------------------
    artifact_paths: dict[str, Any] = {"metrics_json": str(report_dir / "metrics.json")}
    if include_plots:
        artifact_paths["reliability_pngs"] = _plot_reliability(rel_df, report_dir)
        artifact_paths["team_pts_scatter_png"] = _plot_team_pts_scatter(
            actuals, means_df, report_dir
        )

    # ---- write ------------------------------------------------------------
    out_json = report_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    return {**metrics, "_artifacts": artifact_paths,
            "_n_games": eval_dataset_n_games(parquet)}


def eval_dataset_n_games(parquet: str | Path) -> int:
    return pl.read_parquet(parquet)["game_id"].n_unique()


def _json_default(o: Any) -> Any:
    """Serialize tensors / NaN to JSON-safe values."""
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, float) and math.isnan(o):
        return None
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate a hierarchical NBA NN.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parquet", required=True, type=Path,
                        help="Eval split parquet (val or test).")
    parser.add_argument("--train-parquet", required=True, type=Path,
                        help="Train split, needed to load the fitted id-map + stats.")
    parser.add_argument("--report-dir", default="reports", type=Path)
    parser.add_argument("--n-interval-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    summary = evaluate(
        checkpoint=args.checkpoint,
        parquet=args.parquet,
        train_parquet=args.train_parquet,
        report_dir=args.report_dir,
        n_interval_samples=args.n_interval_samples,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        include_plots=not args.no_plots,
    )
    # Print the top-line metrics for quick inspection.
    print(f"per-stat MAE: {summary['per_stat_mae']}")
    print(f"team PTS MAE (player-sum): {summary['team_pts_mae']:.3f}")
    print(f"team PTS MAE (team-head):  {summary['team_pts_mae_team_head']:.3f}")
    print(f"pace MAE:     {summary['pace_mae']:.3f}")
    print(f"off_rtg MAE:  {summary['off_rtg_mae']:.3f}")
    print(f"80% PI coverage: {summary['interval_coverage']}")
    print(f"constraint violations: {summary['constraint_violation_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
