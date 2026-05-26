"""Phase 3 hyperparameter tuning sweep — round 3 (architecture + regime).

WHAT TUNE2 TAUGHT US
--------------------

Every non-collapsed tune2 run hit best_val_nll at epoch 18-20. AdamW +
linear-warmup-cosine + LR ∈ {5e-4, 1e-3} all settle into the *same*
local minimum:

  • Halving LR + doubling epoch budget: best epoch still 19.
  • Weight_decay x10: val_nll identical to baseline at the 4th decimal.
  • Batch 64 -> 128: uniformly slightly worse.
  • embedding_pool weight x10: bit-identical to baseline -- the loss term
    isn't wired into training (embedding_deltas never passed in).
  • Mild minutes downweight + strong gate (R5): the ONLY run that moved
    per-stat MAEs meaningfully (pts 3.804 vs 3.823, ast 1.084 vs 1.089).
  • PI coverage stuck at 90-95% on most stats (target ≈ 80%).

Conclusion: optimization regime is exhausted. Same minimum every time.

To break the plateau we need ONE of:
  (a) Change WHICH minima are reachable -> bigger model (different fn class)
  (b) Change HOW the optimizer explores -> kicked LR schedule / different betas
  (c) Reverse REGULARIZATION direction -> if we're underfit, less is more.

NEW RUNS (all stack on pretrain_epochs=0; uses train + model config overrides)
------------------------------------------------------------------------------

  0  baseline                      -- pretrain=0; clean reference
  1  bigger player rep             -- d_team 128->192 + player_mlp [128,128]->[192,192]
  2  bigger team head              -- team_head.hidden [128,64]->[256,128]
  3  bigger alloc head             -- player_alloc_head.hidden [128,128]->[256,256]
  4  regularization reversed       -- all dropouts -> 0, weight_decay 1e-4 -> 1e-5
  5  kicked optimization regime    -- lr 5e-3, warmup 500, betas (0.9, 0.999),
                                     max_epochs 60, patience 10

Why each magnitude:

  d_team 128 -> 192 (Run 1)
      192 = 1.5 x 128 -- 50% capacity bump in the universal "player feature"
      dim. It propagates to team head input (2*d_team) AND alloc head input
      (d_player := d_team). Also keeps 192/n_heads(4) = 48 cleanly divisible.

  team_head and alloc_head hidden x2 (Runs 2, 3)
      Doubling the hidden width is the standard "more capacity" rung. Each
      isolates ONE head so we can attribute any improvement to that piece.

  Dropouts 0.1 -> 0.0 + weight_decay 1e-4 -> 1e-5 (Run 4)
      Total regularization stripped. If val_loss improves we were underfit
      (the 4-season train set was sufficient and reg was the ceiling). If
      val_loss worsens we were tuned right and need more data instead.

  lr 1e-3 -> 5e-3, warmup 100 -> 500 (Run 5)
      5x peak LR is a deliberate "kick" past the epoch-19 minimum -- a
      single high-LR exploration phase. Long warmup (500 steps) prevents
      blow-up at the start; cosine then anneals normally to 0. Betas
      (0.9, 0.999) are standard Adam (vs current (0.9, 0.95) which damps
      second-moment estimation); pairs naturally with higher peak LR.
      Epochs->60, patience->10 because the trajectory is different and we
      want to let it actually finish.

All runs use seed=42. We override BOTH train_config AND model_config
per-run (tune2 only touched train_config).

Usage
-----
    python scripts/tune3.py                   # all 6 (~60-90 min)
    python scripts/tune3.py --runs 0,5        # subset
    python scripts/tune3.py --max-epochs 3    # smoke test
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


# Shared train-config override applied to every run: drop curriculum.
# Established improvement from round 1 (Run 05) and round 2 (Run 00).
_BASE_TRAIN: dict[str, Any] = {"curriculum": {"team_level_pretrain_epochs": 0}}


def _stack_train(extra: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(copy.deepcopy(_BASE_TRAIN), extra)


RUNS: list[dict[str, Any]] = [
    {
        "name": "00_baseline",
        "summary": "pretrain=0, default model (replicates tune2 Run 00)",
        "train_overrides": _stack_train({}),
        "model_overrides": {},
    },
    {
        "name": "01_bigger_player_rep",
        "summary": "d_team 128->192, player_mlp [128,128]->[192,192]",
        "train_overrides": _stack_train({}),
        "model_overrides": {
            "dims": {"d_team": 192},
            "encoder": {"player_mlp_hidden": [192, 192]},
        },
    },
    {
        "name": "02_bigger_team_head",
        "summary": "team_head.hidden [128,64]->[256,128] (2x capacity)",
        "train_overrides": _stack_train({}),
        "model_overrides": {
            "team_head": {"hidden": [256, 128]},
        },
    },
    {
        "name": "03_bigger_alloc_head",
        "summary": "player_alloc_head.hidden [128,128]->[256,256] (2x capacity)",
        "train_overrides": _stack_train({}),
        "model_overrides": {
            "player_alloc_head": {"hidden": [256, 256]},
        },
    },
    {
        "name": "04_reg_reversed",
        "summary": "all dropouts -> 0, weight_decay 1e-4 -> 1e-5",
        "train_overrides": _stack_train({"optim": {"weight_decay": 1.0e-5}}),
        "model_overrides": {
            "encoder": {
                "player_mlp_dropout": 0.0,
                "roster_attention_dropout": 0.0,
            },
            "player_alloc_head": {"dropout": 0.0},
        },
    },
    {
        "name": "05_kicked_regime",
        "summary": "lr 5e-3, warmup 500, betas (0.9, 0.999), 60 epochs, patience 10",
        "train_overrides": _stack_train({
            "optim": {
                "lr": 5.0e-3,
                "betas": [0.9, 0.999],
            },
            "schedule": {
                "warmup_steps": 500,
                "max_epochs": 60,
            },
            "early_stopping": {"patience": 10},
        }),
        "model_overrides": {},
    },
]


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
    base_model_cfg: dict[str, Any],
    train_parquet: Path,
    val_parquet: Path,
    repo_root: Path,
    epochs_override: int | None,
) -> dict[str, Any]:
    train_cfg = _deep_merge(
        copy.deepcopy(base_train_cfg), run.get("train_overrides", {})
    )
    model_cfg = _deep_merge(
        copy.deepcopy(base_model_cfg), run.get("model_overrides", {})
    )
    if epochs_override is not None:
        train_cfg = _deep_merge(
            train_cfg, {"schedule": {"max_epochs": epochs_override}}
        )

    ckpt_dir = repo_root / "models" / "tune3" / run["name"]
    report_dir = repo_root / "reports" / "tune3" / run["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}\n[{run['name']}] {run['summary']}\n{'=' * 68}",
          flush=True)
    print(f"  train_overrides: {json.dumps(run.get('train_overrides', {}))}",
          flush=True)
    print(f"  model_overrides: {json.dumps(run.get('model_overrides', {}))}",
          flush=True)
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
        n_interval_samples=50,
        batch_size=int(train_cfg["loop"]["batch_size"]),
        device=str(train_cfg["run"].get("device", "auto")),
        seed=int(train_cfg["run"]["seed"]),
        include_plots=False,
    )

    return {
        "name": run["name"],
        "summary": run["summary"],
        "train_overrides": run.get("train_overrides", {}),
        "model_overrides": run.get("model_overrides", {}),
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

    header = ["run", "val_nll", "team_pts", "pace", "best_ep"] + list(KEY_STATS)
    rows: list[list[str]] = []
    numeric: dict[str, list[float]] = {c: [] for c in header
                                       if c not in ("run", "best_ep")}
    for r in results:
        cells: dict[str, Any] = {
            "run":       r["name"],
            "val_nll":   r["best_val_nll"],
            "team_pts":  r["team_pts_mae"],
            "pace":      r["pace_mae"],
            "best_ep":   r["best_epoch"],
        }
        for s in KEY_STATS:
            cells[s] = r["per_stat_mae"].get(s, float("nan"))
        for c in numeric:
            numeric[c].append(float(cells[c]))
        rows.append([
            cells["run"],
            f"{cells['val_nll']:.3f}",
            f"{cells['team_pts']:.3f}",
            f"{cells['pace']:.3f}",
            str(int(cells["best_ep"])),
        ] + [f"{cells[s]:.3f}" for s in KEY_STATS])

    best_row = {c: int(min(range(len(v)), key=lambda i: v[i]))
                for c, v in numeric.items()}
    for col_i, c in enumerate(header):
        if c in ("run", "best_ep"):
            continue
        rows[best_row[c]][col_i] += "*"

    widths = [max(len(h), max(len(r[i]) for r in rows))
              for i, h in enumerate(header)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * 110)
    print("TUNE3 COMPARISON (lower is better; * = best in column; best_ep is informational)")
    print("=" * 110)
    print(line(header))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))
    print("=" * 110)


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
        description="Round-3 sweep: architecture + regime (break the epoch-19 plateau).",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"),
        help="Base train.yaml (defaults to configs/train.yaml).",
    )
    parser.add_argument(
        "--runs", type=str, default=None,
        help="Comma-separated run indices to execute (e.g. '0,5'). "
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
    base_model_cfg = _load_yaml(_resolve(base_train_cfg["model_config"], repo_root))
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

    out_root = repo_root / "reports" / "tune3"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "summary.json"

    results: list[dict[str, Any]] = []
    for idx in selected:
        try:
            r = _run_one(
                RUNS[idx], base_train_cfg, base_model_cfg,
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
