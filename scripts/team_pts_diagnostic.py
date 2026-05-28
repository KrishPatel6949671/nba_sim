"""Team-PTS diagnostic: is the team head a better team-PTS predictor than
summing player pts?

Background
----------

`team_pts_mae` in evaluate.py rolls up per-player `pred_pts` (each from a
Binomial/NegBin draw) into a team total. The team head predicts pace and
off_rtg but those are only reported as side metrics — the team head's
implicit team-PTS estimate (`pace * off_rtg / 100`) is never compared
against the player-sum.

This script loads every `models/tune5/*/best.pt` checkpoint and computes,
per (game_id, team_id) on the val split:

  - actual_team_pts                 — sum of player pts (truth)
  - pred_team_pts_player_sum        — sum of per-player pred_pts (current path)
  - pred_team_pts_team_head         — pred_pace * pred_off_rtg / 100
  - pred_team_pts_blend50           — mean of the two predictions
  - pred_team_pts_oracle            — true_pace * true_off_rtg / 100
                                      (floor on what the team-head path
                                      could achieve at MAE=0 on pace/rtg)

Prints a comparison table. If team_head MAE materially undershoots
player_sum MAE, the next move is to add a coupling loss between the two
signals and retrain.

Usage
-----
    python scripts/team_pts_diagnostic.py
    python scripts/team_pts_diagnostic.py --runs 00_baseline,02_bigger_player_embed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from nba_sim.training.evaluate import (
    predict_means,
    predict_team_aggregates,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _team_pts_from_player_sum(
    actuals: pl.DataFrame, means_df: pl.DataFrame
) -> pl.DataFrame:
    """One row per (game_id, team_id) with actual + player-sum predicted pts."""
    actual_team = actuals.group_by(["game_id", "team_id"]).agg(
        pl.col("pts").cast(pl.Float64).sum().alias("actual_team_pts")
    )
    pred_team = means_df.group_by(["game_id", "team_id"]).agg(
        pl.col("pred_pts").sum().alias("pred_team_pts_player_sum")
    )
    return actual_team.join(pred_team, on=["game_id", "team_id"], how="inner")


def _team_pts_from_team_head(team_df: pl.DataFrame) -> pl.DataFrame:
    """Compute pred / oracle team_pts from pace * off_rtg / 100.

    `predict_team_aggregates` returns per-team rows with both predicted and
    true (pace, off_rtg). We use both to produce the predicted team-head
    estimate and the oracle floor.
    """
    return team_df.with_columns([
        (pl.col("pred_pace") * pl.col("pred_off_rtg") / 100.0)
            .alias("pred_team_pts_team_head"),
        (pl.col("pace") * pl.col("off_rtg") / 100.0)
            .alias("pred_team_pts_oracle"),
    ]).select([
        "game_id", "team_id",
        "pred_team_pts_team_head", "pred_team_pts_oracle",
        "pred_pace", "pace", "pred_off_rtg", "off_rtg",
    ])


def _diagnose_one(
    run_name: str,
    ckpt: Path,
    val_parquet: Path,
    train_parquet: Path,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    actuals = pl.read_parquet(val_parquet)
    means_df = predict_means(
        checkpoint=ckpt, parquet=val_parquet,
        train_parquet=train_parquet, batch_size=batch_size, device=device,
    )
    team_df = predict_team_aggregates(
        checkpoint=ckpt, parquet=val_parquet,
        train_parquet=train_parquet, batch_size=batch_size, device=device,
    )

    player_sum = _team_pts_from_player_sum(actuals, means_df)
    team_head = _team_pts_from_team_head(team_df)
    joined = player_sum.join(team_head, on=["game_id", "team_id"], how="inner")

    joined = joined.with_columns(
        ((pl.col("pred_team_pts_player_sum") + pl.col("pred_team_pts_team_head")) / 2.0)
            .alias("pred_team_pts_blend50")
    )

    def _mae(col: str) -> float:
        return float((joined[col] - joined["actual_team_pts"]).abs().mean())

    return {
        "run": run_name,
        "n_rows": int(joined.height),
        "mae_player_sum": _mae("pred_team_pts_player_sum"),
        "mae_team_head":  _mae("pred_team_pts_team_head"),
        "mae_blend50":    _mae("pred_team_pts_blend50"),
        "mae_oracle":     _mae("pred_team_pts_oracle"),
        "per_team_df":    joined,
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    cols = ["run", "n", "player_sum", "team_head", "blend50", "oracle"]
    rows: list[list[str]] = []
    for r in results:
        rows.append([
            r["run"],
            str(r["n_rows"]),
            f"{r['mae_player_sum']:.3f}",
            f"{r['mae_team_head']:.3f}",
            f"{r['mae_blend50']:.3f}",
            f"{r['mae_oracle']:.3f}",
        ])
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))
    print("\n" + "=" * 78)
    print("TEAM-PTS DIAGNOSTIC: MAE by predictor (lower is better)")
    print("=" * 78)
    print(line(cols))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))
    print("=" * 78)
    print(
        "player_sum = sum(pred_player_pts) — current path\n"
        "team_head  = pred_pace * pred_off_rtg / 100 — proposed path\n"
        "blend50    = simple 50/50 mean of the two\n"
        "oracle     = true_pace * true_off_rtg / 100 (floor for team_head path)"
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Team-PTS predictor diagnostic across tune5 checkpoints.",
    )
    parser.add_argument(
        "--runs", type=str, default=None,
        help="Comma-separated run names (e.g. '00_baseline,02_bigger_player_embed'). "
             "Defaults to every dir under models/tune5/ with a best.pt.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--per-team-csv", type=Path, default=None,
        help="If set, write per-(game,team) predictions for the FIRST run to this path.",
    )
    parser.add_argument(
        "--summary-json", type=Path,
        default=_repo_root() / "reports" / "tune5" / "team_pts_diagnostic.json",
        help="Where to write the summary MAE numbers.",
    )
    args = parser.parse_args(argv)

    repo = _repo_root()
    tune_root = repo / "models" / "tune5"
    val_parquet = repo / "data" / "processed" / "val.parquet"
    train_parquet = repo / "data" / "processed" / "train.parquet"

    for p in (val_parquet, train_parquet):
        if not p.exists():
            print(f"missing parquet: {p}", file=sys.stderr)
            return 2

    if args.runs:
        run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        run_names = sorted(
            d.name for d in tune_root.iterdir()
            if d.is_dir() and (d / "best.pt").exists()
        )
    if not run_names:
        print(f"no checkpoints under {tune_root}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    for name in run_names:
        ckpt = tune_root / name / "best.pt"
        if not ckpt.exists():
            print(f"  [{name}] no best.pt, skipping", file=sys.stderr)
            continue
        print(f"[{name}] running diagnostic...", flush=True)
        r = _diagnose_one(
            name, ckpt, val_parquet, train_parquet,
            batch_size=args.batch_size, device=args.device,
        )
        results.append(r)

    _print_table(results)

    if args.per_team_csv and results:
        args.per_team_csv.parent.mkdir(parents=True, exist_ok=True)
        results[0]["per_team_df"].write_csv(args.per_team_csv)
        print(f"\nper-team predictions for {results[0]['run']} → {args.per_team_csv}")

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w") as f:
        json.dump(
            [{k: v for k, v in r.items() if k != "per_team_df"} for r in results],
            f, indent=2,
        )
    print(f"summary → {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
