"""``nba-sim snapshot-status`` helper — read ``as_of.json`` and render the
human-readable status block (v2PLAN.md §17.2).

Deliberately dependency-light (stdlib + json only): importing this module
never pulls in polars / torch, so the status command stays fast.

Contract: pure read-side. It never mutates the snapshot; it only reports what
``refresh`` last wrote, plus a derived freshness verdict.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

from nba_sim.snapshot import AS_OF_FILENAME, snapshot_dir

# Snapshots older than this are flagged stale — a *warning*, not an error
# (§16.4); the simulate path lets the user override with --stale-ok.
STALE_AFTER_DAYS = 7

# Column at which values begin — wide enough for the longest label
# ("interim_latest_game_date:") plus a gap, so every value left-aligns (§17.2).
_LABEL_WIDTH = 31


def read_provenance(snapshot_root: Path | None = None) -> dict[str, Any]:
    """Load + parse ``as_of.json`` from the snapshot dir.

    Raises ``FileNotFoundError`` with the documented "run 'nba-sim refresh'
    first" hint when the snapshot (or its provenance file) is absent.
    """
    root = snapshot_root if snapshot_root is not None else snapshot_dir()
    path = root / AS_OF_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no snapshot provenance at {path} — run 'nba-sim refresh' first"
        )
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def format_status(provenance: dict[str, Any], *, now: _dt.datetime | None = None) -> str:
    """Render the §17.2 status block from a parsed ``as_of.json``.

    Lines: as_of_date, refreshed_at (+ relative annotation), source, teams,
    players, interim_latest_game_date, and a ``freshness`` verdict. Freshness
    is measured against ``interim_latest_game_date`` — i.e. how many days of
    new games the snapshot is missing — flagged STALE past
    :data:`STALE_AFTER_DAYS`. ``now`` is injectable so tests are deterministic.
    """
    if now is None:
        now = _dt.datetime.now(_dt.UTC)

    # refreshed_at: echo the stored stamp + a coarse "when it ran" annotation.
    refreshed_raw = str(provenance.get("refreshed_at", ""))
    try:
        rel = _relative_age(_dt.datetime.fromisoformat(refreshed_raw), now)
        refreshed_line = f"{refreshed_raw} ({rel})"
    except ValueError:
        refreshed_line = refreshed_raw  # unparseable stamp — show it verbatim

    # freshness: age of the underlying interim data (always >= 0).
    interim_latest = _dt.date.fromisoformat(provenance["interim_latest_game_date"])
    age = max((now.date() - interim_latest).days, 0)
    unit = "day" if age == 1 else "days"
    if age > STALE_AFTER_DAYS:
        freshness = f"STALE ({age} {unit} old — run 'nba-sim refresh')"
    else:
        freshness = f"fresh ({age} {unit} old)"

    rows: tuple[tuple[str, Any], ...] = (
        ("as_of_date", provenance["as_of_date"]),
        ("refreshed_at", refreshed_line),
        ("source", provenance["source"]),
        ("teams", provenance["n_teams"]),
        ("players", provenance["n_players"]),
        ("interim_latest_game_date", provenance["interim_latest_game_date"]),
        ("freshness", freshness),
    )
    return "\n".join(f"{label + ':':<{_LABEL_WIDTH}}{value}" for label, value in rows)


def _relative_age(ts: _dt.datetime, now: _dt.datetime) -> str:
    """Human relative age ('today' / 'yesterday' / 'N days ago') for the
    ``refreshed_at`` annotation, by calendar-day difference."""
    days = (now.date() - ts.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"
