"""Matchup-level features.

Features (see PLAN.md §3.1):
    opp_def_rtg_10     — opponent's rolling DefRtg over last 10 games
    opp_pace_10        — opponent's rolling pace
    opp_def_rtg_vs_pos — opponent's season DefRtg allowed to player's position
    h2h_last_meeting_margin — point margin in the two teams' last meeting, or 0
"""

from __future__ import annotations

import polars as pl


def add_matchup_features(
    per_player: pl.DataFrame,
    team_rolling: pl.DataFrame,
) -> pl.DataFrame:
    """Join opponent-aware features onto the per-player per-game rows."""
    raise NotImplementedError


def opponent_defrtg_by_position(team_box: pl.DataFrame) -> pl.DataFrame:
    """Season-level DefRtg allowed to each position, per team."""
    raise NotImplementedError


def head_to_head_last_margin(games: pl.DataFrame) -> pl.DataFrame:
    """Last meeting's margin for each (home, away, date) triple, or 0."""
    raise NotImplementedError
