"""Phase 3 hyperparameter tuning sweep — round 4 (DIAGNOSTIC).

WHAT TUNE3 + FULL-DATA RUN TAUGHT US
------------------------------------

Tune3 on 4 seasons monotonically improved val_nll through epoch ~14,
reaching -27.16. When scaled to the full 22-season train set, the *same*
config exhibits a wholly different pattern:

    epoch  0: train=-20.48  val=-26.16  best=-26.16
    epoch  1: train=-28.98  val=-26.93  best=-26.93  <-- best
    epoch  2: train=-29.29  val=-26.90
    ...
    epoch 11: train=-29.85  val=-26.33
    early stop at epoch 11 (best @ epoch 1)

Train continues to drop monotonically (-20 -> -29.85). Val plateaus
in [-26.27, -26.93] after epoch 1, oscillating in a 0.66-unit band.
Generalization gap of ~3.3 NLL units. This is overfitting, not a
capacity-limited minimum -- and tune3's optimizer/architecture knobs
won't fix it because they were tuned on data that didn't overfit.

A first-order suspect: data split is train=[2000..2021] / val=[2022].
Twenty-two years of NBA evolution (3-pt revolution, pace shift,
positional fluidity) in train, ONE modern season in val. The model
averages over eras; val has only one era.

TUNE4 IS A DIAGNOSTIC, NOT AN OPTIMIZATION
------------------------------------------

Six runs, each ISOLATING one hypothesis for the train/val decoupling.
The goal is not to find the best config -- it's to identify WHICH
LEVER moves val_nll. Whatever moves it tells us what's wrong.

  0  baseline                     -- replicates the overfitting pattern
  1  recent_only_2018             -- filter train to seasons 2018-2021 only.
                                    If val_nll improves, era shift is real.
  2  heavy_reg                    -- dropout 0.1->0.3 everywhere, wd 1e-4->1e-3.
                                    If val improves, classical overfit.
  3  strong_embedding_pool        -- embedding_pool weight 1e-3->1.0 (1000x).
                                    If val improves, per-player IDs were
                                    the memorization vector.
  4  slow_lr_long_warmup          -- lr 1e-3->3e-4, warmup 100->2000,
                                    max_epochs 30->60, patience 5->15.
                                    If val improves, optimizer was stepping
                                    past a sharper minimum.
  5  big_batch                    -- batch 64->256, lr 1e-3->2e-3 (linear
                                    scaling rule), warmup 100->400.
                                    If val improves, gradient noise was
                                    dominating the late-epoch trajectory.

Interpretation guide:
  * Multiple runs improve val by similar amounts -> redundant levers,
    pick the cheapest.
  * Only Run 1 improves -> commit to recent-era training. The other knobs
    are tuning noise.
  * Run 1 doesn't help but Run 2/3 does -> overfitting is generic, not
    era-driven. Lean on regularization for the full data.
  * Nothing helps -> the val_nll floor is structural (val set too small
    or too narrow). Re-examine the split.

Usage
-----
    python scripts/tune4.py                   # all 6 (~3-4h on 3050)
    python scripts/tune4.py --runs 0,1        # subset
    python scripts/tune4.py --max-epochs 3    # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from nba_sim.training.evaluate import evaluate as eval_full
from nba_sim.training.loop import _deep_merge, train


# ---------------------------------------------------------------------------
# Run definitions
# ---------------------------------------------------------------------------


_BASE_TRAIN: dict[str, Any] = {"curriculum": {"team_level_pretrain_epochs": 0}}


def _stack_train(extra: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(copy.deepcopy(_BASE_TRAIN), extra)


RUNS: list[dict[str, Any]] = [
    {
        "name": "00_baseline",
        "summary": "default configs (replicates the overfit pattern)",
        "train_overrides": _stack_train({}),
        "model_overrides": {},
        "train_seasons": None,
    },
    {
        "name": "01_recent_only_2018",
        "summary": "train on 2018-2021 only (4 modern seasons, val=2022)",
        "train_overrides": _stack_train({}),
        "model_overrides": {},
        "train_seasons": [2018, 2019, 2020, 2021],
    },
    {
        "name": "02_heavy_reg",
        "summary": "dropout 0.1->0.3, weight_decay 1e-4->1e-3",
        "train_overrides": _stack_train({"optim": {"weight_decay": 1.0e-3}}),
        "model_overrides": {
            "encoder": {
                "player_mlp_dropout": 0.3,
                "roster_attention_dropout": 0.3,
            },
            "player_alloc_head": {"dropout": 0.3},
        },
        "train_seasons": None,
    },
    {
        "name": "03_strong_embedding_pool",
        "summary": "embedding_pool weight 1e-3->1.0 (1000x)",
        "train_overrides": _stack_train({"loss_weights": {"embedding_pool": 1.0}}),
        "model_overrides": {},
        "train_seasons": None,
    },
    {
        "name": "04_slow_lr_long_warmup",
        "summary": "lr 3e-4, warmup 2000, 60 epochs, patience 15",
        "train_overrides": _stack_train({
            "optim": {"lr": 3.0e-4},
            "schedule": {"warmup_steps": 2000, "max_epochs": 60},
            "early_stopping": {"patience": 15},
        }),
        "model_overrides": {},
        "train_seasons": None,
    },
    {
        "name": "05_big_batch",
        "summary": "batch 64->256, lr 2e-3 (linear scaling), warmup 400",
        "train_overrides": _stack_train({
            "optim": {"lr": 2.0e-3},
            "loop": {"batch_size": 256},
            "schedule": {"warmup_steps": 400},
        }),
        "model_overrides": {},
        "train_seasons": None,
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


def _maybe_filter_train_seasons(
    train_parquet: Path,
    seasons: list[int] | None,
    out_dir: Path,
) -> Path:
    """If ``seasons`` is set, write a filtered copy and return its path.

    Else return ``train_parquet`` unchanged. Filtering keeps only rows
    whose ``season`` column is in the allowed set. Player_id_map is
    rebuilt from this filtered set by BoxScoreDataset, so omitted-era
    players get padding_idx — fine, the val era doesn't reference them.
    """
    if seasons is None:
        return train_parquet
    out = out_dir / "train_filtered.parquet"
    if out.exists() and out.stat().st_mtime > train_parquet.stat().st_mtime:
        return out
    df = pl.read_parquet(train_parquet)
    n_before = df.height
    df = df.filter(pl.col("season").is_in(seasons))
    n_after = df.height
    print(f"  filtered train: {n_before} -> {n_after} rows "
          f"(seasons={seasons})", flush=True)
    df.write_parquet(out)
    return out


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

    ckpt_dir = repo_root / "models" / "tune4" / run["name"]
    report_dir = repo_root / "reports" / "tune4" / run["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}\n[{run['name']}] {run['summary']}\n{'=' * 68}",
          flush=True)
    print(f"  train_overrides: {json.dumps(run.get('train_overrides', {}))}",
          flush=True)
    print(f"  model_overrides: {json.dumps(run.get('model_overrides', {}))}",
          flush=True)

    effective_train_parquet = _maybe_filter_train_seasons(
        train_parquet, run.get("train_seasons"), ckpt_dir,
    )

    t0 = time.time()
    train_summary = train(
        train_parquet=effective_train_parquet,
        val_parquet=val_parquet,
        model_config=model_cfg,
        train_config=train_cfg,
        out_dir=ckpt_dir,
        verbose=True,
    )

    eval_summary = eval_full(
        checkpoint=ckpt_dir / "best.pt",
        parquet=val_parquet,
        train_parquet=effective_train_parquet,
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
        "train_seasons": run.get("train_seasons"),
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
    print("TUNE4 COMPARISON (lower is better; * = best in column; best_ep is informational)")
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
        description="Round-4 diagnostic sweep: isolate the train/val decoupling cause.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"),
        help="Base train.yaml (defaults to configs/train.yaml).",
    )
    parser.add_argument(
        "--runs", type=str, default=None,
        help="Comma-separated run indices to execute (e.g. '0,1'). "
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

    out_root = repo_root / "reports" / "tune4"
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
