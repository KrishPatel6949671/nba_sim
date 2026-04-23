"""Game-context features.

Features (see PLAN.md §3.1):
    is_home, rest_days, b2b, 3-in-4, 4-in-6, season_phase, day_of_week,
    month, altitude_ft, travel_miles_prev.

Altitude and arena geolocation come from a small static table embedded in
this module (Denver and Utah are the notable outliers).
"""

from __future__ import annotations

import polars as pl


def add_context_features(games: pl.DataFrame) -> pl.DataFrame:
    """Compute context features for every (team, game) pair.

    ``games`` must already be sorted by ``(team_id, date)``; this function
    relies on that ordering to derive rest / b2b.
    """
    raise NotImplementedError


def arena_altitude(team_abbr: str) -> float:
    """Elevation in feet for the team's home arena."""
    raise NotImplementedError


def travel_distance_miles(prev_team_abbr: str, next_team_abbr: str) -> float:
    """Great-circle distance between the two arenas."""
    raise NotImplementedError
