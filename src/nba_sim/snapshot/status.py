"""``nba-sim snapshot-status`` helper — read ``as_of.json`` and render the
human-readable status block (v2PLAN.md §17.2).

Deliberately dependency-light (stdlib + json only): importing this module
never pulls in polars / torch, so the status command stays fast.

Contract: pure read-side. It never mutates the snapshot; it only reports what
``refresh`` last wrote, plus a derived freshness verdict.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

# Snapshots older than this are flagged stale — a *warning*, not an error
# (§16.4); the simulate path lets the user override with --stale-ok.
STALE_AFTER_DAYS = 7


def read_provenance(snapshot_root: Path | None = None) -> dict[str, Any]:
    """Load + parse ``as_of.json`` from the snapshot dir.

    Raises ``FileNotFoundError`` with the documented "run 'nba-sim refresh'
    first" hint when the snapshot (or its provenance file) is absent.
    """
    raise NotImplementedError("TODO(task 7): read + validate as_of.json")


def format_status(provenance: dict[str, Any], *, now: _dt.datetime | None = None) -> str:
    """Render the §17.2 status block from a parsed ``as_of.json``.

    Lines: as_of_date, refreshed_at (+ relative annotation), source, teams,
    players, interim_latest_game_date, and a ``freshness`` verdict
    (fresh / N days old / STALE past :data:`STALE_AFTER_DAYS`). ``now`` is
    injectable so tests are deterministic.
    """
    raise NotImplementedError("TODO(task 7): format status block")


def _relative_age(ts: _dt.datetime, now: _dt.datetime) -> str:
    """Human relative age ('today' / 'yesterday' / 'N days ago') for the
    ``refreshed_at`` annotation."""
    raise NotImplementedError("TODO(task 7)")
