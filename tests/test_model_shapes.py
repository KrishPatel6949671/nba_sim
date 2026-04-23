"""Forward-pass shape tests for every module in :mod:`nba_sim.models`.

These tests exercise a random-init model on a small synthetic batch
and assert tensor shapes match PLAN.md §4.2. A separate test ensures
gradients flow to every parameter.
"""

from __future__ import annotations

import pytest


def test_player_encoder_output_shape() -> None:
    pytest.skip("not implemented")


def test_roster_attention_pool_output_shape() -> None:
    pytest.skip("not implemented")


def test_game_context_encoder_output_shape() -> None:
    pytest.skip("not implemented")


def test_team_head_output_shapes() -> None:
    pytest.skip("not implemented")


def test_player_alloc_head_returns_all_distributions() -> None:
    pytest.skip("not implemented")


def test_hierarchical_forward_pass_end_to_end() -> None:
    pytest.skip("not implemented")


def test_gradients_flow_to_every_parameter() -> None:
    """After one backward pass, ``param.grad`` is not None for every parameter."""
    pytest.skip("not implemented")
