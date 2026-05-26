"""Phase 3 hyperparameter tuning sweep — round 2.

Round-1 sweep (scripts/tune.py → reports/tune/) taught us:

  • Skipping the 2-epoch team-only pretrain (Run 05) is mildly better than
    keeping it: team_pts 13.38 → 13.02, pts 3.846 → 3.823, ast 1.111 → 1.089.
    That's our new baseline.
  • Down-weighting lambda_minutes to 0.3 BROKE training (Runs 01, 03):
    val_nll went +6.95 (vs -27 baseline), every per-stat MAE 2-10x worse.
    The minutes Dirichlet anchors the model -- without it the count heads
    predict for the wrong slots.
  • Doubling team weights (Run 02) only moved pace MAE (3.96 → 3.86); did
    not move team_pts_mae. Team-side supervision is not the bottleneck.
  • Halving LR + doubling epochs (Run 04) was marginal: val_nll slightly
    better, headline metrics nearly identical. Not convergence-starved.
  • team_pts_mae ≈ 13.3 ≈ sqrt(12) · per_player_pts_mae(3.85), i.e. the
    sum-of-independent-errors floor. To budge it we need either sharper
    per-player predictions OR error correlation that helps team totals.

NEW RUNS (each stacks ONE lever on top of pretrain_epochs=0)
------------------------------------------------------------

  0  pretrain=0 only                  -- new baseline (fresh comparison point)
  1  + lr=5e-4, epochs=60, patience=10 -- stack last sweep's neutral-positive
  2  + batch_size 64 → 128             -- smoother grads, more team signal/step
  3  + weight_decay 1e-4 → 1e-3        -- tighten PI coverage (90-96% → 80%)
  4  + embedding_pool 1e-3 → 1e-2      -- stronger partial pooling on players
  5  + lambda_minutes 0.7, lambda_gate 1.0
                                       -- mild minutes downweight + strong gate

Why each magnitude:
  batch_size 64 → 128
      Doubling is the standard rung. Team loss has 1-2 observations per
      game vs ~25 per-player slots; doubling the batch doubles team-side
      gradient signal per step. 256 risks OOM with the current model on
      full data.

  weight_decay 1e-4 → 1e-3
      Log-scaled regularizer; 10x is one standard rung. PI coverage at
      90-96% (target 80%) means predictive variance is too wide --
      stronger L2 should tighten weights and hence dispersion params.

  embedding_pool 1e-3 → 1e-2
      Same log-rung as weight_decay. Pulls per-player embedding deltas
      toward 0 (i.e. position/role prior), which mostly helps rare/
      low-minute players where there's least data to learn the delta.

  lambda_minutes 1.0 → 0.7, lambda_gate 0.5 → 1.0
      Round-1 Run 01 showed lambda_minutes=0.3 breaks training because
      the model loses the "who plays" anchor. The gate term IS that
      anchor (Bernoulli "did this player appear"). So: gentle 30% cut
      to minutes while doubling the gate -- preserve the anchor while
      letting count heads compete for gradient. If THIS collapses too,
      we know minutes weight is rigid and won't move.

  lr 5e-4, epochs 60, patience 10
      Same magnitudes as Round-1 Run 04 (which was neutral-positive on
      its own). Testing whether finer convergence compounds with the
      curriculum drop.

All on top of pretrain_epochs=0. Same seed (42) across all runs.

Usage
-----
    python scripts/tune2.py                   # all 6 runs
    python scripts/tune2.py --runs 0,2        # subset by index
    python scripts/tune2.py --max-epochs 5    # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from nba_sim.training.evaluate import evaluate as eval_full
from nba_sim.training.loop import _deep_merge, train


# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------


# Shared base override applied to every run: drop the curriculum pretrain.
# This is the established improvement from the round-1 sweep (Run 05).
_BASE: dict[str, Any] = {"curriculum": {"team_level_pretrain_epochs": 0}}


def _stack(extra: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``extra`` on top of the shared _BASE override."""
    out = copy.deepcopy(_BASE)
    return _deep_merge(out, extra)


RUNS: list[dict[str, Any]] = [
    {
        "name": "00_no_curriculum_baseline",
        "summary": "pretrain=0 only (new baseline; replicates round-1 Run 05)",
        "overrides": _stack({}),
    },
    {
        "name": "01_finer_converge",
        "summary": "+ lr=5e-4, epochs=60, patience=10",
        "overrides": _stack({
            "optim": {"lr": 5.0e-4},
            "schedule": {"max_epochs": 60},
            "early_stopping": {"patience": 10},
        }),
    },
    {
        "name": "02_batch_128",
        "summary": "+ batch_size 64 -> 128 (more team signal/step)",
        "overrides": _stack({"loop": {"batch_size": 128}}),
    },
    {
        "name": "03_weight_decay_up",
        "summary": "+ weight_decay 1e-4 -> 1e-3 (tighten PI coverage)",
        "overrides": _stack({"optim": {"weight_decay": 1.0e-3}}),
    },
    {
        "name": "04_embedding_pool_up",
        "summary": "+ embedding_pool 1e-3 -> 1e-2 (stronger partial pooling)",
        "overrides": _stack({"loss_weights": {"embedding_pool": 1.0e-2}}),
    },
    {
        "name": "05_minutes_soft_gate_strong",
        "summary": "+ lambda_minutes 1.0->0.7, lambda_gate 0.5->1.0",
        "overrides": _stack({
            "loss_weights": {"minutes": 0.7, "gate": 1.0},
        }),
    },
]


# Stats surfaced in the printed comparison; full set is in metrics.json.
KEY_STATS = ("minutes", "pts", "fga", "fgm", "tpa", "tpm", "reb", "ast")


# ---------------------------------------------------------------------------
# Path + config helpers
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


# ---------------------------------------------------------------------------
# Per-run driver
# ---------------------------------------------------------------------------


def _run_one(
    run: dict[str, Any],
    base_train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_parquet: Path,
    val_parquet: Path,
    repo_root: Path,
    epochs_override: int | None,
) -> dict[str, Any]:
    train_cfg = _deep_merge(copy.deepcopy(base_train_cfg), run["overrides"])
    if epochs_override is not None:
        train_cfg = _deep_merge(
            train_cfg, {"schedule": {"max_epochs": epochs_override}}
        )

    ckpt_dir = repo_root / "models" / "tune2" / run["name"]
    report_dir = repo_root / "reports" / "tune2" / run["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}\n[{run['name']}] {run['summary']}\n{'=' * 68}",
          flush=True)
    print(f"  overrides: {json.dumps(run['overrides'])}", flush=True)
    t0 = time.time()

    train_summary = train(
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        model_config=model_cfg,
        train_config=train_cfg,
        out_dir=ckpt_dir,
        verbose=True,
    )

    eval_summary = eval_full(
        checkpoint=ckpt_dir / "best.pt",
        parquet=val_parquet,
        train_parquet=train_parquet,
        report_dir=report_dir,
        n_interval_samples=50,    # smaller K for tuning; full eval uses 200
        batch_size=int(train_cfg["loop"]["batch_size"]),
        device=str(train_cfg["run"].get("device", "auto")),
        seed=int(train_cfg["run"]["seed"]),
        include_plots=False,
    )

    return {
        "name": run["name"],
        "summary": run["summary"],
        "overrides": run["overrides"],
        "best_val_nll": train_summary["best_val_nll"],
        "best_epoch": train_summary["best_epoch"],
        "epochs_trained": train_summary["epochs_trained"],
        "per_stat_mae": eval_summary["per_stat_mae"],
        "interval_coverage": eval_summary.get("interval_coverage", {}),
        "team_pts_mae": eval_summary["team_pts_mae"],
        "pace_mae": eval_summary["pace_mae"],
        "off_rtg_mae": eval_summary["off_rtg_mae"],
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Pretty-print the comparison
# ---------------------------------------------------------------------------


def _print_comparison(results: list[dict[str, Any]]) -> None:
    if not results:
        return

    header = ["run", "val_nll", "team_pts", "pace"] + list(KEY_STATS)
    rows: list[list[str]] = []
    numeric: dict[str, list[float]] = {c: [] for c in header if c != "run"}
    for r in results:
        cells: dict[str, Any] = {
            "run":       r["name"],
            "val_nll":   r["best_val_nll"],
            "team_pts":  r["team_pts_mae"],
            "pace":      r["pace_mae"],
        }
        for s in KEY_STATS:
            cells[s] = r["per_stat_mae"].get(s, float("nan"))
        for c in numeric:
            numeric[c].append(float(cells[c]))
        rows.append([cells["run"]] + [f"{cells[c]:.3f}" for c in header[1:]])

    best_row = {c: int(min(range(len(v)), key=lambda i: v[i]))
                for c, v in numeric.items()}
    for col_i, c in enumerate(header):
        if c == "run":
            continue
        rows[best_row[c]][col_i] += "*"

    widths = [max(len(h), max(len(r[i]) for r in rows))
              for i, h in enumerate(header)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * 96)
    print("TUNE2 COMPARISON (lower is better; * = best in column)")
    print("=" * 96)
    print(line(header))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))
    print("=" * 96)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_runs_arg(spec: str, n: int) -> list[int]:
    out: list[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"non-numeric run index: {tok!r}")
        i = int(tok)
        if not 0 <= i < n:
            raise ValueError(f"run index out of range [0, {n}): {i}")
        out.append(i)
    return out


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Round-2 hyperparameter sweep on top of pretrain=0.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"),
        help="Base train.yaml (defaults to configs/train.yaml).",
    )
    parser.add_argument(
        "--runs", type=str, default=None,
        help="Comma-separated run indices to execute (e.g. '0,2'). "
             "Defaults to all 6.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override every run's schedule.max_epochs (smoke-test mode).",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root()
    train_yaml = _resolve(args.config, repo_root).resolve()
    base_train_cfg = _load_yaml(train_yaml)
    model_cfg = _load_yaml(_resolve(base_train_cfg["model_config"], repo_root))
    data_cfg = _load_yaml(_resolve(base_train_cfg["data_config"], repo_root))
    processed = _resolve(data_cfg["paths"]["processed"], repo_root)
    train_parquet = processed / "train.parquet"
    val_parquet = processed / "val.parquet"

    if args.runs is None:
        selected = list(range(len(RUNS)))
    else:
        try:
            selected = _parse_runs_arg(args.runs, len(RUNS))
        except ValueError as e:
            print(f"--runs error: {e}", file=sys.stderr)
            return 2

    out_root = repo_root / "reports" / "tune2"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"

    results: list[dict[str, Any]] = []
    for idx in selected:
        try:
            r = _run_one(
                RUNS[idx], base_train_cfg, model_cfg,
                train_parquet, val_parquet, repo_root,
                epochs_override=args.max_epochs,
            )
        except Exception as e:
            print(f"\n[{RUNS[idx]['name']}] FAILED: {e!r}", file=sys.stderr)
            continue
        results.append(r)
        with open(summary_path, "w") as f:
            json.dump({"runs": results}, f, indent=2)

    _print_comparison(results)
    print(f"\nresults saved to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
