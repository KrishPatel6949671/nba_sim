"""Season-level train / val / test splits.

We hold out **full seasons**, not random games, so that form-driven features
for a validation-season player never leak through games in the training set
(see PLAN.md §3.2 and §6.1).
"""

from __future__ import annotations

from pathlib import Path

from nba_sim.data.etl import SplitSpec


def load_split_spec(config_path: str | Path) -> SplitSpec:
    """Read ``configs/data.yaml`` and return the split configuration."""
    raise NotImplementedError


def assert_no_season_overlap(spec: SplitSpec) -> None:
    """Fail fast if any season appears in more than one split."""
    raise NotImplementedError
