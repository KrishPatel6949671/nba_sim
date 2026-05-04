"""Raw -> interim -> processed transforms.

All functions here are pure transforms with deterministic output paths:

* ``raw_to_interim(season)`` reads ``data/raw/`` and writes per-season
  typed parquet under ``data/interim/<season>/``.
* ``interim_to_processed(splits)`` joins interim tables with rolling
  features (see :mod:`nba_sim.features`) and writes
  ``data/processed/{train,val,test}.parquet``.
* ``build_feature_tables(season)`` precomputes rolling aggregates shared
  by multiple features.

All ETL is **idempotent** (tmp-file-then-rename) and **incremental**
(skip seasons whose outputs are newer than their inputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitSpec:
    """Which seasons go to which split. Loaded from ``configs/data.yaml``."""

    train: list[int]
    val: list[int]
    test: list[int]


def raw_to_interim(season: int) -> Path:
    """Transform raw cached payloads for one season into typed parquet.

    Writes:
        data/interim/<season>/games.parquet
        data/interim/<season>/player_box.parquet
        data/interim/<season>/team_box.parquet
        data/interim/<season>/rosters.parquet
        data/interim/<season>/qa_report.json
    """
    raise NotImplementedError


def build_feature_tables(season: int) -> None:
    """Precompute rolling aggregates that multiple feature groups share."""
    raise NotImplementedError


def interim_to_processed(splits: SplitSpec) -> dict[str, Path]:
    """Join interim + features and emit the modeling-ready splits.

    Returns a dict with keys ``{"train","val","test"}`` mapping to parquet paths.
    """
    raise NotImplementedError
