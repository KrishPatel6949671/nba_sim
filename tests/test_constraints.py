"""Hard-constraint tests — the key property of this simulator.

Samples N box scores from a (random-init or trained) model and asserts
every constraint from PLAN.md §4.4 holds on every sample. Uses
:mod:`hypothesis` to vary roster size, active roster count, and seed.
"""

from __future__ import annotations

import pytest


def test_minutes_sum_to_240_per_team() -> None:
    """After round-with-residual redistribution, team MIN total == 240.0 exactly."""
    pytest.skip("not implemented")


def test_fgm_le_fga_per_player() -> None:
    pytest.skip("not implemented")


def test_tpm_le_tpa_per_player() -> None:
    pytest.skip("not implemented")


def test_tpm_le_fgm_per_player() -> None:
    pytest.skip("not implemented")


def test_ftm_le_fta_per_player() -> None:
    pytest.skip("not implemented")


def test_all_counts_non_negative_integers() -> None:
    pytest.skip("not implemented")


def test_team_pts_equals_sum_of_player_pts() -> None:
    pytest.skip("not implemented")


def test_reb_equals_oreb_plus_dreb_per_player() -> None:
    pytest.skip("not implemented")


def test_dnp_rostered_players_get_zero_minutes() -> None:
    pytest.skip("not implemented")


def test_constraint_violation_rate_is_zero_over_1000_samples() -> None:
    """Scale test: 1000 samples, every constraint holds on every one."""
    pytest.skip("not implemented")
