"""Phase 3 hyperparameter tuning sweep — round 5 (on modern-era data).

WHAT TUNE4 TAUGHT US
--------------------

Tune4 isolated the cause of the train/val decoupling we hit on the
22-season dataset. Five hypotheses were tested in isolation; only one
moved val_nll meaningfully below the overfit-baseline (-26.93):

  01_recent_only_2018          best_val_nll = -27.151  (DELTA -0.22, the only win)
  02_heavy_reg                 best_val_nll = -26.608  (worse, model was not classically overfit)
  03_strong_embedding_pool     best_val_nll = -26.872  (essentially no change)
  04_slow_lr_long_warmup       best_val_nll = -26.989  (essentially no change)
  05_big_batch                 best_val_nll = -26.975  (essentially no change)

Restricting train to 2018-2021 (4 modern seasons) also dramatically
improved team_pts MAE (14.91 -> 12.61) and pace MAE (4.05 -> 3.87).
Era distribution shift was the real bottleneck -- adding pre-Curry-era
seasons was actively hurting the modern val prediction.

TUNE5: hyperparam search on a coherent modern-era split
-------------------------------------------------------

Now training on 2014-2022 (9 modern seasons, post-3pt-revolution) with
val=2023. This is ~2x the data of tune4 R1 with the same era coherence,
so we'd expect baseline val_nll somewhere in [-27.4, -27.7] range.

The six runs each test ONE hyperparameter the prior rounds left
ambiguous on a clean dataset:

  0  baseline                  -- defaults (sanity / reference)
  1  bigger_player_rep         -- d_team 128->192, player_mlp [128,128]->[192,192]
                                 Tune3 R1 was a wash on 4 seasons. With 2.25x
                                 the data, capacity at the encoder may bind.
  2  bigger_player_embed       -- d_player_embed 32->64.
                                 More distinct modern-era players = potentially
                                 underpowered embedding. Doubles the table size.
  3  minutes_gate_rebalance    -- λ_minutes 1.0->0.7, λ_gate 0.5->1.0.
                                 Tune2 R5 was the only run with meaningful
                                 per-stat MAE shifts; revisit on clean data.
  4  longer_patience           -- max_epochs 30->60, patience 5->15.
                                 Tune4 R1 hit best_epoch=11 with patience=5,
                                 close to the limit; was it actually converged?
  5  lower_lr                  -- lr 1e-3->5e-4, warmup 100->500.
                                 Smaller dataset + same model = potentially
                                 needs a gentler optimizer.

PREREQ
------

Before running, update ``configs/data.yaml`` to:
    splits:
      train: [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
      val:   [2023]
      test:  [2024]

Then ``nba-sim build-features`` to regenerate the processed parquets
with the new split. tune5 reads from the standard paths.

Usage
-----
    python scripts/tune5.py                   # all 6 (~70-90 min on 3050)
    python scripts/tune5.py --runs 0,1        # subset
    python scripts/tune5.py --max-epochs 3    # smoke test
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
        "summary": "default configs on 2014-2022 train, 2023 val",
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
        "name": "02_bigger_player_embed",
        "summary": "d_player_embed 32->64 (player embedding table doubled)",
        "train_overrides": _stack_train({}),
        "model_overrides": {
            "embeddings": {"d_player_embed": 64},
        },
    },
    {
        "name": "03_minutes_gate_rebalance",
        "summary": "lambda_minutes 1.0->0.7, lambda_gate 0.5->1.0 (revisit tune2 R5)",
        "train_overrides": _stack_train({
            "loss_weights": {"minutes": 0.7, "gate": 1.0},
        }),
        "model_overrides": {},
    },
    {
        "name": "04_longer_patience",
        "summary": "max_epochs 30->60, patience 5->15 (let it converge)",
        "train_overrides": _stack_train({
            "schedule": {"max_epochs": 60},
            "early_stopping": {"patience": 15},
        }),
        "model_overrides": {},
    },
    {
        "name": "05_lower_lr",
        "summary": "lr 1e-3->5e-4, warmup 100->500 (gentler optimizer)",
        "train_overrides": _stack_train({
            "optim": {"lr": 5.0e-4},
            "schedule": {"warmup_steps": 500},
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


def _summarize_parquet(path: Path, label: str) -> None:
    """Print row count + season range; sanity-check before running."""
    df = pl.scan_parquet(path).select(pl.col("season")).collect()
    seasons = sorted(df.unique().to_series().to_list())
    print(f"  {label:5s} {path.name}: {df.height} rows, seasons={seasons}",
          flush=True)


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

    ckpt_dir = repo_root / "models" / "tune5" / run["name"]
    report_dir = repo_root / "reports" / "tune5" / run["name"]
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
    print("TUNE5 COMPARISON (lower is better; * = best in column; best_ep is informational)")
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
        description="Round-5 sweep: hyperparam search on modern-era data (2014-2022).",
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

    # Sanity-check the split — print what's in train/val before running so
    # the user can confirm data.yaml + rebuild were actually done.
    print("data split (sanity check):", flush=True)
    _summarize_parquet(train_parquet, "TRAIN")
    _summarize_parquet(val_parquet, "VAL")

    if args.runs is None:
        selected = list(range(len(RUNS)))
    else:
        try:
            selected = _parse_runs_arg(args.runs, len(RUNS))
        except ValueError as e:
            print(f"--runs error: {e}", file=sys.stderr)
            return 2

    out_root = repo_root / "reports" / "tune5"
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
