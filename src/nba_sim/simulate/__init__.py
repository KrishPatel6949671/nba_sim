"""Simulation — sampling box scores from a trained model.

Modules:
    sampler : low-level sampling from distribution heads, constraint enforcement.
    api     : simulate_game(...) public entry point.

Public re-exports for convenience.
"""

from __future__ import annotations

__all__ = ["simulate_game"]


def __getattr__(name: str):  # lazy to avoid importing torch at package import time
    if name == "simulate_game":
        from nba_sim.simulate.api import simulate_game

        return simulate_game
    raise AttributeError(name)
