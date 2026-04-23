"""Rolling N-game features for players and teams.

Every aggregate strictly uses rows with ``date < target_date`` — no same-game
or future leakage. This is tested as a hypothesis property in
``tests/test_features.py`` (see PLAN.md §3.2).

Windows computed:
    Players: 5, 10, 20.
    Teams:   5, 10.

Feature list in PLAN.md §3.1.
"""

from __future__ import annotations

import polars as pl


def player_rolling(player_box: pl.DataFrame, windows: tuple[int, ...] = (5, 10, 20)) -> pl.DataFrame:
    """Return per-(player,game) rolling features over prior games only.

    Args:
        player_box: long-format per-player per-game lines, one row per
            (player_id, game_id). Must include ``date`` and all counting stats.
        windows: window sizes in games-played (not calendar days).
    """
    raise NotImplementedError


def team_rolling(team_box: pl.DataFrame, windows: tuple[int, ...] = (5, 10)) -> pl.DataFrame:
    """Return per-(team,game) rolling features over prior games only."""
    raise NotImplementedError


def season_to_date(player_box: pl.DataFrame) -> pl.DataFrame:
    """Cumulative season-to-date averages (used by the 'season average' baseline)."""
    raise NotImplementedError
