"""Splits-layer tests.

Pure tests — no network, no filesystem outside ``tmp_path``.

Covers:
    - ``load_split_spec`` happy path against a real-shaped YAML, and every
      validation branch (missing keys, wrong types, non-int entries, bools,
      and the bundled overlap check).
    - ``assert_no_season_overlap`` for clean specs, pairwise overlaps in
      every direction, three-way overlaps, and the all-offenders error
      message format.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nba_sim.data.etl import SplitSpec
from nba_sim.data.splits import assert_no_season_overlap, load_split_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, body: str) -> Path:
    """Write ``body`` to ``path`` and return it. ``body`` is dedented by the
    caller — we keep it as-is to make the YAML in the tests readable."""
    path.write_text(body)
    return path


def _valid_yaml() -> str:
    return (
        "splits:\n"
        "  train: [2020, 2021]\n"
        "  val: [2022]\n"
        "  test: [2023]\n"
    )


# ---------------------------------------------------------------------------
# load_split_spec — happy path
# ---------------------------------------------------------------------------

def test_load_split_spec_happy_path(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "data.yaml", _valid_yaml())
    spec = load_split_spec(path)
    assert isinstance(spec, SplitSpec)
    assert spec.train == [2020, 2021]
    assert spec.val == [2022]
    assert spec.test == [2023]


def test_load_split_spec_preserves_order(tmp_path: Path) -> None:
    """Config order should survive; sorting in the loader would mask a
    deliberate reordering bug in the source config."""
    path = _write_yaml(
        tmp_path / "data.yaml",
        "splits:\n  train: [2021, 2020, 2019]\n  val: [2022]\n  test: [2023]\n",
    )
    spec = load_split_spec(path)
    assert spec.train == [2021, 2020, 2019]


def test_load_split_spec_accepts_empty_lists(tmp_path: Path) -> None:
    """Empty splits are fine for the loader — the modeling layer may reject
    them later, but this is a config-load concern only."""
    path = _write_yaml(
        tmp_path / "data.yaml",
        "splits:\n  train: []\n  val: []\n  test: []\n",
    )
    spec = load_split_spec(path)
    assert spec.train == [] and spec.val == [] and spec.test == []


def test_load_split_spec_works_against_real_config() -> None:
    """The repo's own configs/data.yaml must load. If this regresses, the
    pipeline can't bootstrap."""
    spec = load_split_spec("configs/data.yaml")
    # Don't pin exact contents — that would couple the test to one config
    # snapshot. Pin invariants instead.
    assert len(spec.train) >= 1
    assert len(spec.val) >= 1
    # Train should be earlier (or equal) than val/test in absolute year span.
    assert max(spec.train) <= min(spec.val + spec.test)


# ---------------------------------------------------------------------------
# load_split_spec — validation branches
# ---------------------------------------------------------------------------

def test_load_split_spec_rejects_missing_splits_key(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "data.yaml", "other_key: 1\n")
    with pytest.raises(ValueError, match="missing top-level 'splits:'"):
        load_split_spec(path)


def test_load_split_spec_rejects_non_dict_splits(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path / "data.yaml", "splits: 42\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_split_spec(path)


@pytest.mark.parametrize("missing_key", ["train", "val", "test"])
def test_load_split_spec_rejects_missing_split_key(
    tmp_path: Path, missing_key: str
) -> None:
    body_lines = ["splits:"]
    for key in ("train", "val", "test"):
        if key != missing_key:
            body_lines.append(f"  {key}: [2020]")
    path = _write_yaml(tmp_path / "data.yaml", "\n".join(body_lines) + "\n")
    with pytest.raises(ValueError, match=f"'splits.{missing_key}' is missing"):
        load_split_spec(path)


def test_load_split_spec_rejects_non_list_value(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "data.yaml",
        "splits:\n  train: 2020\n  val: [2022]\n  test: [2023]\n",
    )
    with pytest.raises(ValueError, match="'splits.train' must be a list"):
        load_split_spec(path)


def test_load_split_spec_rejects_non_int_entry(tmp_path: Path) -> None:
    """A stray string like '2022' is the most common config typo."""
    path = _write_yaml(
        tmp_path / "data.yaml",
        "splits:\n  train: [2020, '2021']\n  val: [2022]\n  test: [2023]\n",
    )
    with pytest.raises(ValueError, match=r"'splits.train\[1\]' must be int"):
        load_split_spec(path)


def test_load_split_spec_rejects_bool_entry(tmp_path: Path) -> None:
    """Python's ``bool`` is an ``int`` subclass — ``isinstance(True, int)``
    is True. Without an explicit bool check, a literal ``true`` in YAML
    would silently parse as season 1."""
    path = _write_yaml(
        tmp_path / "data.yaml",
        "splits:\n  train: [2020, true]\n  val: [2022]\n  test: [2023]\n",
    )
    with pytest.raises(ValueError, match=r"'splits.train\[1\]' must be int"):
        load_split_spec(path)


def test_load_split_spec_rejects_overlap(tmp_path: Path) -> None:
    """The bundled overlap check must fire — silent leakage is the whole
    reason this function exists."""
    path = _write_yaml(
        tmp_path / "data.yaml",
        "splits:\n  train: [2022]\n  val: [2022]\n  test: [2023]\n",
    )
    with pytest.raises(ValueError, match="multiple splits"):
        load_split_spec(path)


def test_load_split_spec_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_split_spec(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# assert_no_season_overlap
# ---------------------------------------------------------------------------

def test_assert_no_overlap_passes_on_clean_spec() -> None:
    spec = SplitSpec(train=[2020, 2021], val=[2022], test=[2023])
    assert_no_season_overlap(spec)  # no exception


def test_assert_no_overlap_passes_on_all_empty() -> None:
    """An empty spec trivially has no overlap; the empties check belongs
    to the caller (training, evaluation), not this validator."""
    assert_no_season_overlap(SplitSpec(train=[], val=[], test=[]))


@pytest.mark.parametrize(
    ("kwargs", "expected_splits"),
    [
        ({"train": [2022], "val": [2022], "test": [2023]}, ["train", "val"]),
        ({"train": [2023], "val": [2022], "test": [2023]}, ["test", "train"]),
        ({"train": [2021], "val": [2022], "test": [2022]}, ["test", "val"]),
    ],
)
def test_assert_no_overlap_catches_pairwise_overlap(
    kwargs: dict[str, list[int]], expected_splits: list[str]
) -> None:
    spec = SplitSpec(**kwargs)
    with pytest.raises(ValueError) as exc:
        assert_no_season_overlap(spec)
    msg = str(exc.value)
    for s in expected_splits:
        assert s in msg, f"split {s!r} missing from error: {msg!r}"


def test_assert_no_overlap_catches_three_way_overlap() -> None:
    spec = SplitSpec(train=[2022], val=[2022], test=[2022])
    with pytest.raises(ValueError) as exc:
        assert_no_season_overlap(spec)
    msg = str(exc.value)
    # All three split names should appear.
    assert "train" in msg and "val" in msg and "test" in msg


def test_assert_no_overlap_reports_all_offenders() -> None:
    """When multiple seasons overlap, the error lists every one — fix-it-once
    cadence rather than first-failure whack-a-mole."""
    spec = SplitSpec(train=[2020, 2021, 2022], val=[2021, 2022], test=[2023])
    with pytest.raises(ValueError) as exc:
        assert_no_season_overlap(spec)
    msg = str(exc.value)
    assert "2021" in msg
    assert "2022" in msg


def test_assert_no_overlap_silent_when_only_within_split_dupes() -> None:
    """Repeating a season inside one split isn't an overlap *across* splits
    — this function specifically checks cross-split leakage. If we ever
    want intra-split-uniqueness, that's a separate validator."""
    spec = SplitSpec(train=[2020, 2020, 2021], val=[2022], test=[2023])
    assert_no_season_overlap(spec)  # no exception
