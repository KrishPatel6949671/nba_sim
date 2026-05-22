"""Batch feature build for SLURM/cron. Thin wrapper around ``nba-sim build-features``.

Usage:
    python scripts/build_features.py                          # full pipeline per configs/data.yaml
    python scripts/build_features.py --seasons 2022,2023      # per-season feature tables only
    python scripts/build_features.py --refresh -v             # force re-build with verbose logs

Builds per-season feature tables under ``data/interim/<season>/`` and (when
``--seasons`` is omitted) emits ``data/processed/{train,val,test}.parquet``
using the splits in ``configs/data.yaml``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_sim.cli import build_features as cli_build_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated season list (e.g. '2022,2023'). Default = all from config.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Splits config (used only when --seasons is omitted).",
    )
    p.add_argument("--refresh", action="store_true", help="Force re-build")
    p.add_argument(
        "--skip-processed",
        action="store_true",
        help="Build per-season feature tables only; skip the train/val/test write.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    cli_build_features(
        seasons=args.seasons,
        config=args.config,
        refresh=args.refresh,
        skip_processed=args.skip_processed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
