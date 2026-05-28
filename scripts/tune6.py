"""Phase 3 hyperparameter tuning sweep — round 6: coupling weight.

WHAT TUNE5 + COUPLING-FIX TAUGHT US
-----------------------------------

tune5 confirmed the model hit a hyperparameter ceiling on per-stat NLL,
so further per-stat MAE gains needed structural changes. The team-PTS
diagnostic (scripts/team_pts_diagnostic.py) then showed the team head's
pace × off_rtg / 100 estimate was ~3-4 MAE units better than summing
per-player pts. A coupling loss was added to nudge the player sum
toward that better signal.

First attempt: MSE coupling at λ=0.1. Catastrophically destroyed
the shooting heads (fga MAE 2.24→4.26, pts MAE 3.66→6.69) because
squared-residual scales as point_diff² and dominated the composite NLL.

Second attempt: Smooth-L1 (Huber) coupling at λ=0.01. Worked:
  - team_pts_mae (player-sum):  13.73 → 12.61   (−8%)
  - team_pts_mae (team-head):    9.42 →  9.48   (~unchanged, detached)
  - per-stat MAEs:               ~flat vs baseline
  - constraint violations:       0% (still by construction)

This sweep tests whether the Huber loss can take a heavier weight to
close more of the gap. The v1 §1 ship gate wants:
  - team_pts MAE  ≤ 8.5  (currently 9.48 team-head, 12.61 player-sum)
  - pts MAE       ≥ 15% better than GLM  (currently −13%, need −15%)
  - reb / ast     same (currently −14% / −11%)

If λ ∈ {0.02, 0.03, 0.05} all land in the same neighborhood as λ=0.01
on per-stat MAE while pulling player-sum team_pts further down, the
highest weight that doesn't degrade per-stat is the new default. If any
weight breaks per-stat MAE (e.g. fga > 2.6), back off.

PREREQ
------

Assumes configs/data.yaml has train ⊇ [2014..2022] (2022 added per
the previous step) and val=[2023]. Assumes configs/model.yaml is at
the Run 01 defaults (d_team=192, player_mlp [192, 192]).

Usage
-----
    python scripts/tune6.py                   # all 3 (~50-70 min on 3050)
    python scripts/tune6.py --runs 0,2        # subset
    python scripts/tune6.py --max-epochs 3    # smoke test
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
# Run definitions — only the coupling weight varies.
# ---------------------------------------------------------------------------


RUNS: list[dict[str, Any]] = [
    {
        "name": "00_coupling_0p02",
        "summary": "Huber coupling, λ=0.02",
        "train_overrides": {"loss_weights": {"coupling": 0.02}},
        "model_overrides": {},
    },
    {
        "name": "01_coupling_0p03",
        "summary": "Huber coupling, λ=0.03",
        "train_overrides": {"loss_weights": {"coupling": 0.03}},
        "model_overrides": {},
    },
    {
        "name": "02_coupling_0p05",
        "summary": "Huber coupling, λ=0.05",
        "train_overrides": {"loss_weights": {"coupling": 0.05}},
        "model_overrides": {},
    },
]


KEY_STATS = ("minutes", "pts", "fga", "fgm", "tpa", "tpm", "reb", "ast")


# ---------------------------------------------------------------------------
# Path + config helpers (identical conventions to tune5.py)
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

    ckpt_dir = repo_root / "models" / "tune6" / run["name"]
    report_dir = repo_root / "reports" / "tune6" / run["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 68}\n[{run['name']}] {run['summary']}\n{'=' * 68}",
          flush=True)
    print(f"  train_overrides: {json.dumps(run.get('train_overrides', {}))}",
          flush=True)
    print(f"  model_overrides: {json.dumps(run.get('model_overrides', {}))}",
          flush=True)
    print(f"  effective coupling weight: "
          f"{train_cfg['loss_weights']['coupling']}", flush=True)
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
        "coupling_weight": float(train_cfg["loss_weights"]["coupling"]),
        "train_overrides": run.get("train_overrides", {}),
        "model_overrides": run.get("model_overrides", {}),
        "best_val_nll": train_summary["best_val_nll"],
        "best_epoch": train_summary["best_epoch"],
        "epochs_trained": train_summary["epochs_trained"],
        "per_stat_mae": eval_summary["per_stat_mae"],
        "interval_coverage": eval_summary.get("interval_coverage", {}),
        "team_pts_mae": eval_summary["team_pts_mae"],
        "team_pts_mae_team_head": eval_summary.get(
            "team_pts_mae_team_head", float("nan")
        ),
        "pace_mae": eval_summary["pace_mae"],
        "off_rtg_mae": eval_summary["off_rtg_mae"],
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _print_comparison(results: list[dict[str, Any]]) -> None:
    if not results:
        return

    header = (
        ["run", "λ", "val_nll",
         "team_pts_psum", "team_pts_thead",
         "pace", "best_ep"]
        + list(KEY_STATS)
    )
    rows: list[list[str]] = []
    numeric: dict[str, list[float]] = {
        c: [] for c in header if c not in ("run", "best_ep", "λ")
    }

    for r in results:
        cells: dict[str, Any] = {
            "run":            r["name"],
            "λ":              r.get("coupling_weight", float("nan")),
            "val_nll":        r["best_val_nll"],
            "team_pts_psum":  r["team_pts_mae"],
            "team_pts_thead": r.get("team_pts_mae_team_head", float("nan")),
            "pace":           r["pace_mae"],
            "best_ep":        r["best_epoch"],
        }
        for s in KEY_STATS:
            cells[s] = r["per_stat_mae"].get(s, float("nan"))
        for c in numeric:
            numeric[c].append(float(cells[c]))
        rows.append([
            cells["run"],
            f"{cells['λ']:.3f}",
            f"{cells['val_nll']:.3f}",
            f"{cells['team_pts_psum']:.3f}",
            f"{cells['team_pts_thead']:.3f}",
            f"{cells['pace']:.3f}",
            str(int(cells["best_ep"])),
        ] + [f"{cells[s]:.3f}" for s in KEY_STATS])

    # Mark the best value in each numeric column with a trailing *.
    best_row = {c: int(min(range(len(v)), key=lambda i: v[i]))
                for c, v in numeric.items()}
    for col_i, c in enumerate(header):
        if c in ("run", "best_ep", "λ"):
            continue
        rows[best_row[c]][col_i] += "*"

    widths = [max(len(h), max(len(r[i]) for r in rows))
              for i, h in enumerate(header)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * 130)
    print("TUNE6 COMPARISON (lower is better; * = best in column; "
          "team_pts_psum=player-sum, team_pts_thead=team-head)")
    print("=" * 130)
    print(line(header))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))
    print("=" * 130)
    print("\nReference (from prior runs, for context — not part of this sweep):")
    print("  Run 01 (no coupling):        team_pts_psum=13.73, "
          "team_pts_thead=9.42, pts MAE=3.66")
    print("  λ=0.01 (current default):    team_pts_psum=12.61, "
          "team_pts_thead=9.48, pts MAE=3.74")


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
        description="Round-6 sweep: coupling weight λ ∈ {0.02, 0.03, 0.05}.",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"),
        help="Base train.yaml (defaults to configs/train.yaml).",
    )
    parser.add_argument(
        "--runs", type=str, default=None,
        help="Comma-separated run indices to execute (e.g. '0,2'). "
             "Defaults to all 3.",
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

    for p in (train_parquet, val_parquet):
        if not p.exists():
            print(f"missing parquet: {p}", file=sys.stderr)
            return 2

    print("data split (sanity check):", flush=True)
    _summarize_parquet(train_parquet, "TRAIN")
    _summarize_parquet(val_parquet, "VAL")

    print(f"\nbase model: d_team={base_model_cfg['dims']['d_team']}, "
          f"player_mlp={base_model_cfg['encoder']['player_mlp_hidden']}",
          flush=True)
    print(f"base coupling weight (will be overridden per run): "
          f"{base_train_cfg['loss_weights'].get('coupling', 'unset')}",
          flush=True)

    if args.runs is None:
        selected = list(range(len(RUNS)))
    else:
        try:
            selected = _parse_runs_arg(args.runs, len(RUNS))
        except ValueError as e:
            print(f"--runs error: {e}", file=sys.stderr)
            return 2

    out_root = repo_root / "reports" / "tune6"
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
