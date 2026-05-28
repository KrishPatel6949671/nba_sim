"""Rolling N-game features for players and teams.

Every aggregate strictly uses rows with ``date < target_date`` — no same-game
or future leakage (PLAN.md §3.2). The Polars pattern is::

    col.shift(1).rolling_*(N, min_periods=1).over(partition)

The ``shift(1)`` slides the target row out of its own window. The ``.over(...)``
keeps one entity's history from leaking into another's. The frame must be
sorted by ``(partition_key, date, game_id)`` first — Polars rolling reads
positional order, not the date column.

Windows (PLAN.md §3.1):
    Players: 5, 10, 20.
    Teams:   5, 10.

Per-minute rate stats (FGA, 3PA, FTA, REB, AST, STL, BLK, TOV, PF) are
computed only at window=10 because that's all PLAN.md §3.1 asks for; adding
5/20 would 3x the column count for marginal value.
"""

from __future__ import annotations

import polars as pl

# Public constants — tests introspect these so the windows can't silently drift.
PLAYER_WINDOWS: tuple[int, ...] = (5, 10, 20)
TEAM_WINDOWS: tuple[int, ...] = (5, 10)

# Per-minute rate stats live at window=10 only (PLAN.md §3.1).
_PER_MIN_STATS: tuple[str, ...] = (
    "fga", "tpa", "fta", "reb", "ast", "stl", "blk", "tov", "pf",
)
_PER_MIN_WINDOW: int = 10

# Counting stats that the season-to-date baseline averages.
_SEASON_TO_DATE_STATS: tuple[str, ...] = (
    "minutes", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
    "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ratio_of_sums(
    num_col: str,
    den_col: str,
    window: int,
    *,
    partition: str,
    alias: str,
) -> pl.Expr:
    """Pooled ratio over a rolling window: ``sum(num)_w / sum(den)_w``.

    ``shift(1)`` keeps the target row out of both sums. ``min_periods=1`` lets
    a player with even one prior game produce a value rather than NaN. The
    explicit ``when(den > 0)`` guards 0/0 (a whole window of DNPs) which
    Polars would otherwise emit as NaN — explicit ``None`` is easier to
    detect in downstream sanity checks than a propagated NaN.
    """
    num = pl.col(num_col).shift(1).rolling_sum(window, min_periods=1).over(partition)
    den = pl.col(den_col).shift(1).rolling_sum(window, min_periods=1).over(partition)
    return pl.when(den > 0).then(num / den).otherwise(None).alias(alias)


def _rolling_mean(
    col: str, window: int, *, partition: str, alias: str
) -> pl.Expr:
    """Plain shifted rolling mean. NaN for the very first row in a partition
    (no prior data); a value from row 2 onward."""
    return (
        pl.col(col)
        .shift(1)
        .rolling_mean(window, min_periods=1)
        .over(partition)
        .alias(alias)
    )


def _attach_dates(
    box: pl.DataFrame, games: pl.DataFrame, *, key: str = "game_id"
) -> pl.DataFrame:
    """Join ``date`` and ``season`` from games onto a box-score frame.

    PlayerBoxLine / TeamBoxLine carry ``game_id`` but neither carries the
    date — the date lives on Game. Doing the join inside this module keeps
    the public function signatures self-sufficient (callers don't have to
    remember to pre-join), and keeps the test harness simple.

    Dropped games are filtered out: a game flagged ``dropped=True`` is not
    valid training data, and including its box rows in the rolling window
    would smear bad data forward into the next clean game.
    """
    keep = games.filter(~pl.col("dropped")).select([key, "date", "season"])
    return box.join(keep, on=key, how="inner")


# ---------------------------------------------------------------------------
# Player rolling features
# ---------------------------------------------------------------------------

def player_rolling(
    player_box: pl.DataFrame,
    games: pl.DataFrame,
    *,
    team_box: pl.DataFrame | None = None,
    windows: tuple[int, ...] = PLAYER_WINDOWS,
) -> pl.DataFrame:
    """Per-(player, game) rolling features over prior games only.

    Args:
        player_box: long-format PlayerBoxLine rows (one per player-game).
            Must include columns from :class:`PlayerBoxLine` — at minimum
            ``game_id, player_id, team_id, minutes, pts, fgm, fga, tpm,
            tpa, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf``.
        games: per-game header table (Game schema). Used to attach
            ``date`` and ``season`` and to filter out dropped games.
        team_box: optional TeamBoxLine frame; when present, per-game USG%
            is derived and rolled. When absent, ``p_usage_avg_*`` columns
            are simply omitted (useful for unit tests that don't construct
            team rows).
        windows: tuple of game-count windows. Defaults to (5, 10, 20).

    Returns:
        The player_box augmented with: ``date``, ``season``,
        ``p_min_avg_{w}``, ``p_ts_{w}``, ``p_pts_per_min_{w}`` for each w
        in ``windows``, ``p_{stat}_per_min_10`` for each per-minute stat,
        ``p_games_played_season``, and ``p_usage_avg_{w}`` when team_box
        is provided.
    """
    if player_box.is_empty():
        return player_box

    df = _attach_dates(player_box, games)
    df = df.sort(["player_id", "date", "game_id"])

    # Pre-compute per-row TS denominator (FGA + 0.44 * FTA). Pulling this
    # out as a column lets us roll a sum over it cheaply instead of recomputing
    # the polynomial inside every rolling expression.
    derived: list[pl.Expr] = [
        (pl.col("fga") + 0.44 * pl.col("fta")).alias("_ts_denom"),
    ]

    # Optional USG% per row. Basketball-Reference canonical formula:
    #   USG% = 100 * (FGA + 0.44 * FTA + TOV) * (team_min / 5)
    #          / ( player_min * (team_FGA + 0.44 * team_FTA + team_TOV) )
    # The ``team_min / 5`` factor normalizes against one player-slot's worth
    # of minutes (5 players on the floor at once), not the team-wide total.
    # Dropping the /5 would inflate everyone's USG% by 5x — caught in smoke
    # test where a 30-min player came out at ~143% instead of ~29%.
    # We join the team totals once and treat USG% as a regular column from
    # then on, so rolling it is no different from rolling any other stat.
    if team_box is not None:
        team_subset = team_box.select([
            "game_id",
            "team_id",
            pl.col("minutes").alias("_team_min"),
            pl.col("fga").alias("_team_fga"),
            pl.col("fta").alias("_team_fta"),
            pl.col("tov").alias("_team_tov"),
        ])
        df = df.join(team_subset, on=["game_id", "team_id"], how="left")
        derived.extend([
            (pl.col("_team_fga") + 0.44 * pl.col("_team_fta") + pl.col("_team_tov"))
            .alias("_team_poss"),
        ])

    df = df.with_columns(derived)

    if team_box is not None:
        # USG% is undefined when player_min == 0 (DNP) or team_poss == 0
        # (an empty team box, which shouldn't happen but we guard anyway).
        df = df.with_columns([
            pl.when((pl.col("minutes") > 0) & (pl.col("_team_poss") > 0))
            .then(
                100.0
                * (pl.col("fga") + 0.44 * pl.col("fta") + pl.col("tov"))
                * (pl.col("_team_min") / 5.0)
                / (pl.col("minutes") * pl.col("_team_poss"))
            )
            .otherwise(None)
            .alias("_usage"),
        ])

    # Build all rolling expressions in one with_columns so Polars can fuse
    # the rolling kernels into a single pass.
    rolls: list[pl.Expr] = []
    for w in windows:
        rolls.append(_rolling_mean("minutes", w, partition="player_id", alias=f"p_min_avg_{w}"))
        # TS% pooled — sum(pts) / (2 * sum(_ts_denom)).
        num = pl.col("pts").shift(1).rolling_sum(w, min_periods=1).over("player_id")
        den = pl.col("_ts_denom").shift(1).rolling_sum(w, min_periods=1).over("player_id")
        rolls.append(
            pl.when(den > 0).then(num / (2.0 * den)).otherwise(None).alias(f"p_ts_{w}")
        )
        rolls.append(
            _ratio_of_sums("pts", "minutes", w, partition="player_id", alias=f"p_pts_per_min_{w}")
        )
        if team_box is not None:
            rolls.append(
                _rolling_mean("_usage", w, partition="player_id", alias=f"p_usage_avg_{w}")
            )

    # Per-minute rate stats at window=10 only.
    for stat in _PER_MIN_STATS:
        rolls.append(
            _ratio_of_sums(
                stat, "minutes", _PER_MIN_WINDOW,
                partition="player_id",
                alias=f"p_{stat}_per_min_{_PER_MIN_WINDOW}",
            )
        )

    # Career-pooled shooting priors. Partitions on player_id only (no
    # season) so the prior spans the player's whole career — useful for
    # shooting-skill stats that rolling-N windows don't stabilize (a 75%
    # career FT shooter stays near 75% almost regardless of last-10 noise).
    # Same ``cum_sum() - current`` trick as season_to_date, just without the
    # season reset.
    for make, attempt, alias in (
        ("fgm", "fga", "p_fg_pct_career"),
        ("tpm", "tpa", "p_tp_pct_career"),
        ("ftm", "fta", "p_ft_pct_career"),
    ):
        num = pl.col(make).cum_sum().over("player_id") - pl.col(make)
        den = pl.col(attempt).cum_sum().over("player_id") - pl.col(attempt)
        rolls.append(
            pl.when(den > 0).then(num / den).otherwise(None).alias(alias)
        )

    # Career blocks per 36 minutes — same partition, normalized to per-36
    # so the scale is human-readable and matches the per-min convention.
    blk_prior = pl.col("blk").cum_sum().over("player_id") - pl.col("blk")
    min_prior = pl.col("minutes").cum_sum().over("player_id") - pl.col("minutes")
    rolls.append(
        pl.when(min_prior > 0)
        .then(36.0 * blk_prior / min_prior)
        .otherwise(None)
        .alias("p_blk_per36_career")
    )

    # Free-throw rate over last 10 games (pooled FTA / FGA). Decouples
    # "gets to the line" from "shoots a lot" — bigs who draw fouls need a
    # different FTA prediction than perimeter shooters with the same FGA
    # volume. Belongs in rolling rather than career because foul-drawing
    # is more game-flow / scheme dependent than pure shooting skill.
    rolls.append(
        _ratio_of_sums(
            "fta", "fga", 10,
            partition="player_id", alias="p_ft_rate_10",
        )
    )

    # Cumulative season game count, prior-only:
    #   cum_count includes the current row, so subtract 1 to get "games
    #   played before this one". Partitioning on (player_id, season) makes
    #   the counter reset each new season.
    rolls.append(
        (pl.col("date").cum_count().over(["player_id", "season"]) - 1)
        .cast(pl.Int64)
        .alias("p_games_played_season")
    )

    df = df.with_columns(rolls)

    # Drop scratch columns — they were means, not features.
    drop_cols = ["_ts_denom"]
    if team_box is not None:
        drop_cols += ["_team_min", "_team_fga", "_team_fta", "_team_tov", "_team_poss", "_usage"]
    df = df.drop(drop_cols)
    return df


# ---------------------------------------------------------------------------
# Team rolling features
# ---------------------------------------------------------------------------

def team_rolling(
    team_box: pl.DataFrame,
    games: pl.DataFrame,
    *,
    windows: tuple[int, ...] = TEAM_WINDOWS,
) -> pl.DataFrame:
    """Per-(team, game) rolling features over prior games only.

    Per PLAN.md §3.1: pace, off_rtg, def_rtg over windows (5, 10); win pct,
    pts avg, pts allowed at window=10. ``plus_minus`` is the canonical signal
    for both wins (>0) and points-allowed (pts - plus_minus); no opponent
    self-join needed.
    """
    if team_box.is_empty():
        return team_box

    df = _attach_dates(team_box, games)
    df = df.sort(["team_id", "date", "game_id"])

    # Derive win and points-allowed inline. ``plus_minus`` is team_pts -
    # opp_pts by definition, so opp_pts == pts - plus_minus. A tie isn't
    # possible in the NBA, so plus_minus > 0 is a clean win indicator.
    df = df.with_columns([
        (pl.col("plus_minus") > 0).cast(pl.Float64).alias("_won"),
        (pl.col("pts") - pl.col("plus_minus")).alias("_pts_allowed"),
    ])

    rolls: list[pl.Expr] = []
    for w in windows:
        rolls.append(_rolling_mean("pace", w, partition="team_id", alias=f"t_pace_{w}"))
        rolls.append(_rolling_mean("off_rtg", w, partition="team_id", alias=f"t_off_rtg_{w}"))
        rolls.append(_rolling_mean("def_rtg", w, partition="team_id", alias=f"t_def_rtg_{w}"))

    # Window=10 only — these are scheduling/strength signals, less useful at w=5.
    rolls.append(_rolling_mean("_won", 10, partition="team_id", alias="t_win_pct_10"))
    rolls.append(_rolling_mean("pts", 10, partition="team_id", alias="t_pts_avg_10"))
    rolls.append(
        _rolling_mean("_pts_allowed", 10, partition="team_id", alias="t_pts_allowed_10")
    )

    df = df.with_columns(rolls).drop(["_won", "_pts_allowed"])
    return df


# ---------------------------------------------------------------------------
# Season-to-date averages (baseline)
# ---------------------------------------------------------------------------

def season_to_date(
    player_box: pl.DataFrame,
    games: pl.DataFrame,
) -> pl.DataFrame:
    """Cumulative season-to-date averages per player.

    Used by the "season-average" baseline in PLAN.md §6.3, which the
    hierarchical model must beat on every counting stat. Strictly excludes
    the target row by the ``cum_sum - current`` trick:

        sum_prior  = cum_sum().over(part) - col(stat)
        count_prior = cum_count().over(part) - 1

    so the first game in a season has ``sum_prior=0, count_prior=0`` and the
    average is ``None`` (no prior history), and every later game has its
    own row excluded from both numerator and denominator.

    Returns a frame with ``std_{stat}_avg`` columns for each counting stat
    plus ``std_games`` (count of prior games this season).
    """
    if player_box.is_empty():
        return player_box

    df = _attach_dates(player_box, games)
    df = df.sort(["player_id", "season", "date", "game_id"])

    part = ["player_id", "season"]

    # Build cum_sums and prior-counts in one pass.
    new_cols: list[pl.Expr] = []
    for stat in _SEASON_TO_DATE_STATS:
        new_cols.append(
            (pl.col(stat).cum_sum().over(part) - pl.col(stat))
            .alias(f"_std_{stat}_sum_prior")
        )
    new_cols.append(
        (pl.col("date").cum_count().over(part) - 1)
        .cast(pl.Int64)
        .alias("std_games")
    )
    df = df.with_columns(new_cols)

    # Then divide. We can't fold this into the previous with_columns because
    # the divisor (std_games) is itself a freshly-computed column.
    avg_cols: list[pl.Expr] = []
    for stat in _SEASON_TO_DATE_STATS:
        avg_cols.append(
            pl.when(pl.col("std_games") > 0)
            .then(pl.col(f"_std_{stat}_sum_prior") / pl.col("std_games"))
            .otherwise(None)
            .alias(f"std_{stat}_avg")
        )
    df = df.with_columns(avg_cols)

    # Drop the scratch prior-sums; keep std_games (it's a useful feature on its own).
    df = df.drop([f"_std_{stat}_sum_prior" for stat in _SEASON_TO_DATE_STATS])
    return df
