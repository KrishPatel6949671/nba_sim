"""Batch fetch for SLURM/cron. Thin wrapper around ``nba-sim fetch``.

Usage:
    python scripts/fetch_all_seasons.py --start 2000 --end 2024
    python scripts/fetch_all_seasons.py --start 2022 --end 2023 --refresh -v

Reads the cache + interim layout defined by :mod:`nba_sim.data.etl`. Safe to
re-run; ``raw_to_interim`` skips seasons whose outputs are already newer
than the cached fetches (use ``--refresh`` to force a rebuild).
"""

from __future__ import annotations

import argparse

from nba_sim.cli import fetch as cli_fetch


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, required=True, help="First season start year")
    p.add_argument("--end", type=int, required=True, help="Last season start year (inclusive)")
    p.add_argument("--refresh", action="store_true", help="Bypass cache and refetch")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    cli_fetch(
        start_season=args.start,
        end_season=args.end,
        refresh=args.refresh,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
