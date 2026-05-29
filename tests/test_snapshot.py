"""Snapshot-layer tests (v2PLAN.md §18.1).

Phase 6 covers the **non-network** subset: as_of resolution, the offline
refresh writing all four parquets + ``as_of.json`` against a tiny fixture
interim directory, schema validity, the synthetic-row sentinel ids, the
``p_min_avg_10`` hand-check / known-date equivalence, provenance round-trip,
and the ``snapshot-status`` block format.

All tests redirect the data roots via ``monkeypatch.setenv`` (same discipline
as ``tests/test_data_etl.py``) so they never touch the user's real
``data/interim`` or ``data/snapshot``. Network / slow tests (full 30-team
refresh) are Phase 8 and marked accordingly.

Phase-7 tests (context vector, build_synthetic_game_batch, simulate v2) live
with the simulate API and are intentionally not here yet.
"""

from __future__ import annotations

import pytest

from nba_sim.snapshot import (
    AS_OF_FILENAME,
    PLAYER_FEATURES_FILENAME,
    ROSTERS_FILENAME,
    TEAM_FEATURES_FILENAME,
    TEAM_LASTGAME_FILENAME,
)

# Season used by the fixture interim builder (start-year).
_FIXTURE_SEASON = 2023

# The four parquets refresh must always write.
_SNAPSHOT_PARQUETS = (
    ROSTERS_FILENAME,
    PLAYER_FEATURES_FILENAME,
    TEAM_FEATURES_FILENAME,
    TEAM_LASTGAME_FILENAME,
)
# Everything a complete snapshot directory contains (provenance written last).
_SNAPSHOT_FILES = (*_SNAPSHOT_PARQUETS, AS_OF_FILENAME)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_fixture_interim(interim_root, season: int = _FIXTURE_SEASON):
    """Write a tiny but schema-faithful interim season under ``interim_root``.

    Must produce ``<season>/{games,player_box,team_box}.parquet`` with the
    real interim columns (see data/etl.py) and enough games per team that the
    rolling windows have history — a handful of teams, a couple of players
    each, several dated games so ``p_min_avg_10`` is hand-computable.
    """
    raise NotImplementedError("TODO(task 8): build fixture interim season")


@pytest.fixture()
def snapshot_env(tmp_path, monkeypatch):
    """Redirect interim + snapshot roots into ``tmp_path`` and seed a fixture
    interim season. Yields the (interim_root, snapshot_root) pair."""
    interim_root = tmp_path / "interim"
    snapshot_root = tmp_path / "snapshot"
    monkeypatch.setenv("NBA_SIM_INTERIM_DIR", str(interim_root))
    monkeypatch.setenv("NBA_SIM_SNAPSHOT_DIR", str(snapshot_root))
    # TODO(task 8): call _write_fixture_interim(interim_root), then
    # `return interim_root, snapshot_root` so the dependent tests can run.
    pytest.skip("TODO(task 8): fixture interim builder not implemented")


# ---------------------------------------------------------------------------
# Stage A — as_of resolution
# ---------------------------------------------------------------------------

def test_resolve_as_of_date_default(snapshot_env) -> None:
    """No --as-of → latest interim game date + 1 day."""
    pytest.skip("TODO(task 8): assert default = latest interim + 1")


def test_resolve_as_of_date_explicit(snapshot_env) -> None:
    """--as-of YYYY-MM-DD is respected verbatim."""
    pytest.skip("TODO(task 8): assert explicit override")


def test_season_for_date() -> None:
    """Jul-Dec -> year N; Jan-Jun -> year N-1 (e.g. 2024-03-15 -> 2023)."""
    pytest.skip("TODO(task 8): season_for_date boundaries")


# ---------------------------------------------------------------------------
# Stage B-D - offline refresh
# ---------------------------------------------------------------------------

def test_refresh_writes_all_four_parquets(snapshot_env) -> None:
    """refresh(offline=True) writes rosters/player_features/team_features/
    team_lastgame + as_of.json."""
    pytest.skip("TODO(task 8): run refresh, assert all five files exist")


def test_refresh_provenance_json_complete(snapshot_env) -> None:
    """as_of.json has every documented key (§13.2)."""
    pytest.skip("TODO(task 8): assert provenance keys")


def test_refresh_atomic_partial_write_safety(snapshot_env) -> None:
    """A crash mid-write (forced via a patched writer) leaves no
    complete-looking snapshot visible; as_of.json is written last."""
    pytest.skip("TODO(task 8): simulate mid-write failure")


def test_snapshot_provenance_round_trip(snapshot_env) -> None:
    """refresh writes provenance; status.read_provenance reads + validates it
    (success-criteria table, §12)."""
    pytest.skip("TODO(task 8): write → read round-trip")


# ---------------------------------------------------------------------------
# Stage C — synthetic rows + schema validity
# ---------------------------------------------------------------------------

def test_synthetic_target_rows_use_sentinel_ids(snapshot_env) -> None:
    """Synthetic game ids start with SNAPSHOT_ (player) / SNAPSHOT_TEAM_
    (team)."""
    pytest.skip("TODO(task 8): assert sentinel id prefixes")


def test_rosters_schema_matches_contract(snapshot_env) -> None:
    """rosters.parquet has exactly ROSTER_COLUMNS (two_way dropped), correct
    dtypes, position non-null, bio nullable."""
    pytest.skip("TODO(task 8): assert rosters schema")


def test_snapshot_schema_validity(snapshot_env) -> None:
    """All four parquets match the documented schemas — column names, dtypes,
    non-null constraints — and round-trip read cleanly (§13.7).

    player_features carries the full 47 _PLAYER_NUMERIC_COLS, with the four
    NULL_AT_REFRESH_NUMERIC columns present-but-null (Float64).
    """
    pytest.skip("TODO(task 8): assert all four schemas")


def test_player_feature_matches_processed_at_known_date(snapshot_env) -> None:
    """For a (team, player) with a real game in the fixture, the synthetic
    feature row's p_min_avg_10 matches a hand-computed value within 1e-6
    (§14.6) — i.e. date<as_of leakage discipline holds and equals the
    processed-pipeline value at a known date (§12)."""
    pytest.skip("TODO(task 8): hand-check p_min_avg_10")


# ---------------------------------------------------------------------------
# snapshot-status (§17.2/§17.3)
# ---------------------------------------------------------------------------

def test_snapshot_status_block_format(snapshot_env) -> None:
    """status.format_status renders the documented block with a deterministic
    freshness verdict (inject ``now``)."""
    pytest.skip("TODO(task 8): assert status block lines")


# ---------------------------------------------------------------------------
# Phase 8 — network / slow (full live refresh)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.network
def test_refresh_under_30s() -> None:
    """Full 30-team live refresh ≤ 30 s, rate-limited at 0.6 s/req (§12).
    Phase 8."""
    pytest.skip("Phase 8: live nba_api refresh")
