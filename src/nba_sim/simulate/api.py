"""Public simulation API — the function users actually call.

The signature is stable; v2 may add kwargs but will not remove them.
See PLAN.md §7 for the full contract.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from nba_sim.data.schema import BoxScore, BoxScoreEnsemble


def simulate_game(
    home_team: str,
    away_team: str,
    date: str | _dt.date,
    *,
    home_roster: list[str] | None = None,
    away_roster: list[str] | None = None,
    n_samples: int = 1,
    seed: int | None = None,
    device: str = "cuda",
    checkpoint: str | Path = "models/best.pt",
    return_distributions: bool = False,
) -> BoxScore | BoxScoreEnsemble:
    """Simulate one game (or an ensemble) between ``home_team`` and ``away_team``.

    Args:
        home_team: 3-letter NBA abbreviation (``"BOS"``, ``"LAL"``, ...).
        away_team: same.
        date: ISO ``"YYYY-MM-DD"`` string or a ``date`` object.
        home_roster, away_roster: optional list of player names or ids.
            If ``None``, uses the team's most recent known active roster.
        n_samples: 1 returns a :class:`BoxScore`; >1 returns a
            :class:`BoxScoreEnsemble` with mean + 80% interval + all samples.
        seed: RNG seed for full determinism. If ``None``, non-deterministic.
        device: ``"cuda"``, ``"cpu"``, or ``"auto"``.
        checkpoint: path to a trained model checkpoint.
        return_distributions: if True, attach raw distribution params to the
            return value for advanced users who want posterior access.

    Returns:
        :class:`BoxScore` or :class:`BoxScoreEnsemble`.
    """
    raise NotImplementedError


def _resolve_rosters(
    home_team: str,
    away_team: str,
    date: _dt.date,
    explicit_home: list[str] | None,
    explicit_away: list[str] | None,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Resolve player strings to (id, display_name) pairs."""
    raise NotImplementedError
