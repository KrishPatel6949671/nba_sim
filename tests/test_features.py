"""Feature-engineering tests.

Property tests use :mod:`hypothesis` to stress-check that no rolling
feature incorporates future data (the primary leakage risk — see
PLAN.md §3.2).
"""

from __future__ import annotations

import pytest


def test_rolling_mean_on_fixed_fixture() -> None:
    """Hand-computed rolling mean for one player matches the implementation."""
    pytest.skip("not implemented")


def test_no_future_game_leakage_property() -> None:
    """Hypothesis property: for any (player, game) row, no contributing prior
    game has a date >= the target date."""
    pytest.skip("not implemented")


def test_context_rest_days_computed_correctly() -> None:
    pytest.skip("not implemented")


def test_b2b_flag_correct_on_consecutive_dates() -> None:
    pytest.skip("not implemented")


def test_opp_defrtg_uses_opponent_prior_games_only() -> None:
    pytest.skip("not implemented")


def test_cold_start_player_feature_is_nan_filled() -> None:
    pytest.skip("not implemented")
