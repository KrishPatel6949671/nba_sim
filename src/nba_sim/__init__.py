"""nba_sim — hierarchical NBA box score simulator.

Top-level package. The public API is deliberately small: users import
``simulate_game`` from ``nba_sim.simulate.api`` and the schema types from
``nba_sim.data.schema``. Everything else is an implementation detail.

See PLAN.md in the repo root for the full design.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
