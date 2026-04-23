"""Public ``simulate_game`` API tests.

Covers the contract in PLAN.md §7:
    - Schema validity of the returned :class:`BoxScore`.
    - Determinism: same seed on same device -> identical box score.
    - ``n_samples=1`` vs. ``n_samples>1`` return-type switch.
    - Cold-start path for unseen players does not crash and produces
      non-degenerate output.
    - Mean-mode returns expected values.
"""

from __future__ import annotations

import pytest


def test_simulate_game_returns_valid_boxscore() -> None:
    pytest.skip("not implemented")


def test_simulate_game_deterministic_with_seed() -> None:
    pytest.skip("not implemented")


def test_simulate_game_n_samples_returns_ensemble() -> None:
    pytest.skip("not implemented")


def test_simulate_game_unseen_player_cold_start() -> None:
    pytest.skip("not implemented")


def test_simulate_game_minutes_sum_to_240_per_team() -> None:
    """Integration-level constraint check on the public API."""
    pytest.skip("not implemented")


def test_simulate_game_mean_mode() -> None:
    pytest.skip("not implemented")
