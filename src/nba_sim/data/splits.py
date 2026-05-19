"""Season-level train / val / test splits.

We hold out **full seasons**, not random games, so that form-driven features
for a validation-season player never leak through games in the training set
(see PLAN.md §3.2 and §6.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from nba_sim.data.etl import SplitSpec

_SPLIT_KEYS = ("train", "val", "test")


def load_split_spec(config_path: str | Path) -> SplitSpec:
    """Read a YAML config and return the split configuration.

    The file must contain a top-level ``splits:`` mapping with ``train``,
    ``val``, and ``test`` keys, each a list of season start years (ints).

    Calls :func:`assert_no_season_overlap` before returning — silent overlap
    is leakage by definition, and we never want to load a leaky spec.
    """
    text = Path(config_path).read_text()
    # safe_load — never yaml.load on user config (arbitrary-code-execution risk).
    payload: Any = yaml.safe_load(text)
    if not isinstance(payload, dict) or "splits" not in payload:
        raise ValueError(f"{config_path}: missing top-level 'splits:' key")

    splits = payload["splits"]
    if not isinstance(splits, dict):
        raise ValueError(f"{config_path}: 'splits' must be a mapping, got {type(splits).__name__}")

    parsed: dict[str, list[int]] = {}
    for key in _SPLIT_KEYS:
        if key not in splits:
            raise ValueError(f"{config_path}: 'splits.{key}' is missing")
        value = splits[key]
        if not isinstance(value, list):
            raise ValueError(
                f"{config_path}: 'splits.{key}' must be a list, got {type(value).__name__}"
            )
        for i, season in enumerate(value):
            # Reject bool too — bool is an int subclass in Python and would
            # silently slip through `isinstance(season, int)`.
            if not isinstance(season, int) or isinstance(season, bool):
                raise ValueError(
                    f"{config_path}: 'splits.{key}[{i}]' must be int, got {season!r}"
                )
        parsed[key] = value

    spec = SplitSpec(train=parsed["train"], val=parsed["val"], test=parsed["test"])
    assert_no_season_overlap(spec)
    return spec


def assert_no_season_overlap(spec: SplitSpec) -> None:
    """Raise if any season appears in more than one of train/val/test.

    Reports *all* offenders, not just the first, so a single failed load
    surfaces every conflict in one shot.
    """
    # Dedupe within each split first — this validator checks cross-split
    # leakage only. A season repeated inside one split is a different config
    # bug (and one we don't currently police).
    appearances: dict[int, list[str]] = {}
    for split_name in _SPLIT_KEYS:
        for season in set(getattr(spec, split_name)):
            appearances.setdefault(season, []).append(split_name)

    duplicates = {s: where for s, where in appearances.items() if len(where) > 1}
    if duplicates:
        details = ", ".join(
            f"{season} in {sorted(where)}" for season, where in sorted(duplicates.items())
        )
        raise ValueError(f"Season(s) appear in multiple splits: {details}")
