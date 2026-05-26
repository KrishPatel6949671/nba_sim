"""Phase 3 hyperparameter tuning sweep.

Runs 6 training configurations on the same data, evaluates each on val,
and prints/saves a per-stat MAE comparison.

WHICH HYPERPARAMETERS AND WHY
-----------------------------

Phase-2 eval left these gaps vs PLAN §1 success criteria:

    team_pts_mae       = 13.38   (target ≤ 8.5)    -- biggest gap
    pts reduction vs GLM = 12%    (target ≥ 15%)
    reb reduction vs GLM = 12%    (target ≥ 15%)
    ast reduction vs GLM =  6%    (target ≥ 15%)
    80% PI coverage      = 90-96% (target ≈ 80%, over-conservative)
    pace MAE             = 3.96   (target ≤ 3.5)

The composite NLL drove from +8.85 to -27.10 during training; nearly all
of that magnitude is the minutes-Dirichlet log_prob term, which dominates
the gradient and starves the count heads.

The 6 runs each attack one lever, then combine:

    0  baseline               -- fresh comparison point
    1  λ_minutes 1.0 → 0.3    -- demote the dominant term; let counts compete
    2  λ_pace, λ_off_rtg ×2   -- sharper team signal → tighter team totals
    3  combo of 1 + 2         -- do they compound?
    4  lr ÷2, epochs ×2, patience ×2  -- finer convergence (PI calibration)
    5  pretrain_epochs 2 → 0  -- PLAN §10 "no curriculum" ablation

Why those numbers:
  λ_minutes 1.0 → 0.3
      At convergence, minutes contributes ≈ -25 to the loss; ten count
      heads together contribute ≈ -15. Multiplying minutes by 0.3 brings
      it to ≈ -7.5, so counts now dominate -- the inversion we want.
      0.5 would leave minutes still on top.

  λ_pace, λ_off_rtg 1.0 → 2.0
      Clean decisive doubling. 1.5 is too timid against per-game noise;
      3.0+ risks team loss swamping player heads.

  lr 1e-3 → 5e-4, epochs 30 → 60, patience 5 → 10
      Run 0 early-stopped at epoch 23 -- cosine had barely decayed.
      Halving lr lets it take finer steps; doubling the budget gives
      cosine room to anneal; patience 10 stops a noisy 1-2 epoch dip
      from killing the run early.

  pretrain_epochs 2 → 0
      Only meaningful magnitude for an ablation. Either the curriculum
      helps or it doesn't.

All runs share seed=42 and the same data split so any delta is
attributable to the override, not init noise.

Usage
-----
    python scripts/tune.py                   # all 6 runs
    python scripts/tune.py --runs 0,3        # subset by index
    python scripts/tune.py --max-epochs 5    # quick smoke test
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


RUNS: list[dict[str, Any]] = [
    {
        "name": "00_baseline",
        "summary": "current config (locks in a fresh comparison point)",
        "overrides": {},
    },
    {
        "name": "01_minutes_down",
        "summary": "lambda_minutes 1.0 -> 0.3 (let count heads dominate)",
        "overrides": {"loss_weights": {"minutes": 0.3}},
    },
    {
        "name": "02_team_up",
        "summary": "lambda_pace, lambda_off_rtg 1.0 -> 2.0 (double team signal)",
        "overrides": {"loss_weights": {"pace": 2.0, "off_rtg": 2.0}},
    },
    {
        "name": "03_combo",
        "summary": "01 + 02 combined",
        "overrides": {
            "loss_weights": {"minutes": 0.3, "pace": 2.0, "off_rtg": 2.0},
        },
    },
    {
        "name": "04_finer_converge",
        "summary": "lr 1e-3 -> 5e-4, epochs 30 -> 60, patience 5 -> 10",
        "overrides": {
            "optim": {"lr": 5.0e-4},
            "schedule": {"max_epochs": 60},
            "early_stopping": {"patience": 10},
        },
    },
    {
        "name": "05_no_curriculum",
        "summary": "team_level_pretrain_epochs 2 -> 0 (PLAN §10 ablation)",
        "overrides": {"curriculum": {"team_level_pretrain_epochs": 0}},
    },
]


# Per-stat MAEs we surface in the printed comparison (full set is in JSON).
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

    ckpt_dir = repo_root / "models" / "tune" / run["name"]
    report_dir = repo_root / "reports" / "tune" / run["name"]
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
        include_plots=False,      # plots add ~30s; not needed for tuning
    )

    return {
        "name": run["name"],
        "summary": run["summary"],
        "overrides": run["overrides"],
        "best_val_nll": train_summary["best_val_nll"],
        "best_epoch": train_summary["best_epoch"],
        "epochs_trained": train_summary["epochs_trained"],
        "per_stat_mae": eval_summary["per_stat_mae"],
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
        cells = {
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

    # Best (min) cell per numeric column; marked with a trailing '*'.
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
    print("TUNING COMPARISON (lower is better; * = best in column)")
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
        description="Phase 3 hyperparameter sweep over train.yaml overrides.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"),
        help="Base train.yaml (defaults to configs/train.yaml).",
    )
    parser.add_argument(
        "--runs", type=str, default=None,
        help="Comma-separated run indices to execute (e.g. '0,3'). "
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

    out_root = repo_root / "reports" / "tune"
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
        # Persist incrementally so a crash mid-sweep doesn't lose prior runs.
        with open(summary_path, "w") as f:
            json.dump({"runs": results}, f, indent=2)

    _print_comparison(results)
    print(f"\nresults saved to {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
