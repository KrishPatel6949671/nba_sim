"""Snapshot layer — a locally-cached, as-of-dated view of league state that
lets the simulator run for games that have **not happened yet** (v2PLAN.md §13).

It mirrors ``data/interim/`` / ``data/processed/`` with a fifth
``data/snapshot/`` directory holding, for one ``as_of`` date:

    rosters.parquet          one row per (team_id, player_id) active as-of
    player_features.parquet  one row per (team_id, player_id): the model's
                             per-player INPUT feature block for a synthetic
                             game dated as_of
    team_features.parquet    one row per team_id: the 9 team-rolling cols
    team_lastgame.parquet    per-team last-game date / arena (rest + travel)
    as_of.json               provenance

Contract
--------
- :func:`nba_sim.snapshot.refresh.refresh` (Stages A-D) atomically rebuilds
  the whole directory for one ``as_of`` date.
- ``build_synthetic_game_batch`` (Phase 7, :mod:`nba_sim.snapshot.build`)
  reads the directory + the train split's id-map / feature-stats and emits
  a model-ready batch with ``B == 1``.

Leakage discipline
------------------
Every feature in the snapshot is derived from interim games with
``date < as_of_date`` — strictly prior, exactly the v1 rule (PLAN.md §3.2).
The synthetic target row carries ``date == as_of_date`` so the existing
``shift(1)`` / ``cum_sum - current`` rolling kernels treat it as "today's
unplayed game" and never let it leak into its own window.

Public exports are resolved lazily (PEP 562 ``__getattr__``) so
``import nba_sim.snapshot`` stays free of polars / torch — only the path
constants below are eager.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

__all__ = [
    "AS_OF_FILENAME",
    "CODE_VERSION",
    "PLAYER_FEATURES_FILENAME",
    "QA_REPORT_FILENAME",
    "ROSTERS_FILENAME",
    "TEAM_FEATURES_FILENAME",
    "TEAM_LASTGAME_FILENAME",
    "TMP_DIRNAME",
    "build_synthetic_game_batch",
    "refresh",
    "snapshot_dir",
]

# Filenames — constants so tests can introspect them and there's one place to
# rename if the layout changes (mirrors the etl.*_FILENAME convention).
ROSTERS_FILENAME = "rosters.parquet"
PLAYER_FEATURES_FILENAME = "player_features.parquet"
TEAM_FEATURES_FILENAME = "team_features.parquet"
TEAM_LASTGAME_FILENAME = "team_lastgame.parquet"
AS_OF_FILENAME = "as_of.json"
QA_REPORT_FILENAME = "qa_report.json"
# Atomic-write staging dir: refresh writes here, then swaps it into place.
TMP_DIRNAME = ".tmp"

# Stamped into as_of.json. Bump when the snapshot on-disk format changes.
CODE_VERSION = "v0.2.0-dev"


def snapshot_dir() -> Path:
    """Snapshot root, honoring ``NBA_SIM_SNAPSHOT_DIR`` (tests redirect it).

    Mirrors :func:`nba_sim.data.etl.interim_dir` / ``processed_dir`` so the
    whole pipeline shares one env-redirect discipline.
    """
    p = os.environ.get("NBA_SIM_SNAPSHOT_DIR", "data/snapshot")
    return Path(p).expanduser().resolve()


def __getattr__(name: str) -> Any:  # lazy — keep package import free of polars/torch
    if name == "refresh":
        from nba_sim.snapshot.refresh import refresh

        return refresh
    if name == "build_synthetic_game_batch":
        from nba_sim.snapshot.build import build_synthetic_game_batch

        return build_synthetic_game_batch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
