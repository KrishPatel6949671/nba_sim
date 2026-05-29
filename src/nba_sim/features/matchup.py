"""Matchup-level features.

Features (PLAN.md §3.1):
    opp_def_rtg_10            — opponent's rolling DefRtg over their last 10 games
    opp_pace_10               — opponent's rolling pace over their last 10 games
    opp_def_rtg_vs_pos        — opp's season DefRtg allowed to player's position
    opp_blk_allowed_vs_pos    — opp's season blocks-per-game allowed to
                                player's position. Anchors the BLK prediction
                                to the situational signal (some teams give up
                                lots of blocks at C; others get blocked at G).
    h2h_last_meeting_margin   — point margin in the two teams' most recent
                                prior meeting, viewed from the current home
                                team's perspective. 0 if no prior meeting.

Leakage discipline (PLAN.md §3.2):
    Every aggregate uses ``date < target_date``:
      * Opponent rolling stats come from ``team_rolling``, which already
        ``shift(1)``s internally — opp's value at game G covers opp's games
        strictly before G.
      * Position-conditioned features (``opp_def_rtg_vs_pos``,
        ``opp_blk_allowed_vs_pos``) use the ``cum_sum - current`` idiom
        inside ``(opp_team, season, position)``, so the target game is
        excluded.
      * ``h2h_last_meeting_margin`` shifts back one row per unordered
        ``(team_a, team_b)`` pair — it can never see the target game.

Signature notes:
    The original stubs in this file took narrower inputs than the
    computations actually need. We widened them rather than smuggle the
    extra tables in through some other mechanism. See the module-level
    docstrings of each function for current signatures.
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _opponent_map(games: pl.DataFrame) -> pl.DataFrame:
    """Long-format ``(game_id, team_id, opp_team_id, season, date)`` map.

    The ``games`` frame has one wide row per game with ``home_team_id`` and
    ``away_team_id`` as separate columns. Many matchup operations want to
    look up "the OTHER team" for an arbitrary ``(game_id, team_id)`` pair;
    that's a join, and a join wants a long-format key table.

    We build the long view by concatenating two slices of ``games``: a
    home-view (team_id = home, opp = away) and an away-view (team_id =
    away, opp = home). Each game produces exactly two rows in the result.

    Dropped games are filtered out so they never participate in matchup
    derivations.
    """
    base = games.filter(~pl.col("dropped"))
    home_view = base.select([
        "game_id", "season", "date",
        pl.col("home_team_id").alias("team_id"),
        pl.col("away_team_id").alias("opp_team_id"),
    ])
    away_view = base.select([
        "game_id", "season", "date",
        pl.col("away_team_id").alias("team_id"),
        pl.col("home_team_id").alias("opp_team_id"),
    ])
    return pl.concat([home_view, away_view])


# ---------------------------------------------------------------------------
# head_to_head_last_margin
# ---------------------------------------------------------------------------

def head_to_head_last_margin(games: pl.DataFrame) -> pl.DataFrame:
    """For each game, the margin of the two teams' most recent prior meeting,
    anchored to the current home team. 0 when there's no prior meeting.

    Why home-team perspective: teams swap home/away across meetings, so a
    naive "prev_home_pts - prev_away_pts" would have its sign flip whenever
    the venue flips, which is meaningless as a feature. We re-anchor to the
    *current* home team — positive value means "current home team won the
    last time these two played." Sign now carries information.

    Output columns: ``game_id``, ``h2h_last_meeting_margin``.
    """
    if games.is_empty():
        return games

    base = games.filter(~pl.col("dropped"))

    # Unordered team-pair key. min_horizontal/max_horizontal compute this
    # entirely in-engine — no python callback, no map_elements overhead.
    base = base.with_columns([
        pl.min_horizontal("home_team_id", "away_team_id").alias("_t1"),
        pl.max_horizontal("home_team_id", "away_team_id").alias("_t2"),
    ])

    # Sort chronologically within each pair, then shift one row back to
    # pick up the previous meeting. Including game_id in the sort breaks
    # rare same-day ties (regular season has no same-day repeats but the
    # data is messy enough to warrant the determinism).
    base = base.sort(["_t1", "_t2", "date", "game_id"])
    base = base.with_columns([
        pl.col("home_team_id").shift(1).over(["_t1", "_t2"]).alias("_prev_home_team"),
        pl.col("home_pts").shift(1).over(["_t1", "_t2"]).alias("_prev_home_pts"),
        pl.col("away_pts").shift(1).over(["_t1", "_t2"]).alias("_prev_away_pts"),
    ])

    # Re-anchor the prior margin to *current* home team's perspective.
    base = base.with_columns(
        pl.when(pl.col("_prev_home_team").is_null())
        .then(pl.lit(0.0))
        .when(pl.col("_prev_home_team") == pl.col("home_team_id"))
        .then((pl.col("_prev_home_pts") - pl.col("_prev_away_pts")).cast(pl.Float64))
        .otherwise((pl.col("_prev_away_pts") - pl.col("_prev_home_pts")).cast(pl.Float64))
        .alias("h2h_last_meeting_margin")
    )
    return base.select(["game_id", "h2h_last_meeting_margin"])


# ---------------------------------------------------------------------------
# opponent_defrtg_by_position
# ---------------------------------------------------------------------------

def opponent_defrtg_by_position(
    player_box: pl.DataFrame,
    games: pl.DataFrame,
    team_box: pl.DataFrame,
) -> pl.DataFrame:
    """Per-(opp_team, game, position) cumulative-prior DefRtg allowed.

    Computed as::

        opp_def_rtg_vs_pos
          = 100 * cum_pts_to_pos_prior(opp_team, season, position, date)
                / cum_opp_possessions_prior(opp_team, season, date)

    where both cum-sums strictly exclude the target game.

    Numerator: sum of points scored by opposing players of the given
    position against ``opp_team``, accumulated within the season.
    Denominator: sum of ``opp_team``'s pace over prior games. Pace is
    offensive possessions per game; team-level offensive and defensive
    possessions match to within ~1 per game, so this is a tight proxy.

    Position ``""`` (non-starter in V3 box scores — a missing-data signal,
    not a real position group) is excluded from the aggregation. v1
    accepts the loss; v2 will fill positions from ``commonplayerinfo``.

    Returns a frame with one row per
    ``(opp_team_id, game_id, position)`` and a single feature column
    ``opp_def_rtg_vs_pos``. Use :func:`add_matchup_features` to join it
    onto a per-player frame.
    """
    if player_box.is_empty() or games.is_empty() or team_box.is_empty():
        return pl.DataFrame()

    opp_map = _opponent_map(games)

    # Attach (opp_team_id, season, date) to each player row.
    pb = player_box.join(opp_map, on=["game_id", "team_id"], how="inner")
    pb = pb.filter(pl.col("position") != "")
    if pb.is_empty():
        return pl.DataFrame()

    # Per-(opp_team, game, position) points scored. We aggregate at the
    # game level first (instead of running the cum_sum on raw player rows)
    # so the partition is over distinct games, not over individual
    # contributions — the shift logic below assumes one row per game per
    # (opp_team, position).
    pts = pb.group_by(
        ["opp_team_id", "season", "date", "game_id", "position"]
    ).agg(pl.col("pts").sum().alias("pts_scored"))

    pts = pts.sort(["opp_team_id", "season", "position", "date", "game_id"])
    pts = pts.with_columns(
        (
            pl.col("pts_scored").cum_sum().over(["opp_team_id", "season", "position"])
            - pl.col("pts_scored")
        ).alias("cum_pts_prior")
    )

    # Cumulative-prior possessions for each (opp_team, season). We pull
    # opp's pace from team_box (their offensive possessions). Joining via
    # opp_map yields the season + date needed for the cum_sum sort.
    # opp_map has 2 rows per game (one per team); restricting on
    # (game_id, opp_team_id) returns the single right row.
    season_meta = opp_map.select([
        "game_id",
        pl.col("team_id").alias("opp_team_id"),
        "season", "date",
    ]).unique(subset=["game_id", "opp_team_id"])

    poss = team_box.select(
        pl.col("game_id"),
        pl.col("team_id").alias("opp_team_id"),
        pl.col("pace").alias("opp_pace"),
    ).join(season_meta, on=["game_id", "opp_team_id"], how="inner")

    poss = poss.sort(["opp_team_id", "season", "date", "game_id"])
    poss = poss.with_columns(
        (
            pl.col("opp_pace").cum_sum().over(["opp_team_id", "season"])
            - pl.col("opp_pace")
        ).alias("cum_poss_prior")
    ).select(["opp_team_id", "game_id", "cum_poss_prior"])

    result = pts.join(poss, on=["opp_team_id", "game_id"], how="left")

    # First-game guard: cum_poss_prior == 0 means the opponent had no prior
    # games this season, so we can't compute a per-100-poss rate. Emit None
    # (cold-start) rather than +inf or 0.
    result = result.with_columns(
        pl.when(pl.col("cum_poss_prior") > 0)
        .then(100.0 * pl.col("cum_pts_prior") / pl.col("cum_poss_prior"))
        .otherwise(None)
        .alias("opp_def_rtg_vs_pos")
    )
    return result.select(
        ["opp_team_id", "game_id", "position", "opp_def_rtg_vs_pos"]
    )


# ---------------------------------------------------------------------------
# opponent_blocks_allowed_by_position
# ---------------------------------------------------------------------------

def opponent_blocks_allowed_by_position(
    player_box: pl.DataFrame,
    games: pl.DataFrame,
) -> pl.DataFrame:
    """Per-(opp_team, game, position) cumulative-prior blocks-per-game allowed.

    Computed as::

        opp_blk_allowed_vs_pos
          = cum_blk_by_position_vs_opp_prior(opp_team, season, position)
            / cum_games_at_position_vs_opp_prior(opp_team, season, position)

    Both numerator and denominator strictly exclude the target game.

    Numerator: sum of blocks recorded by opposing players of the given
    position against ``opp_team``, accumulated within the season. (When a
    center on team A plays against team B and records 3 blocks, that
    contributes +3 to ``opp_blk_allowed_vs_pos[B, C]``.)
    Denominator: count of prior games in which a player at the given
    position played against ``opp_team``.

    Mean blocks per game is more interpretable than a per-100-poss rate
    here because BLK is a low-mean rare event and the per-100-poss scale
    would make tiny differences look large. The model z-scores the
    column either way, so the choice of physical unit is a presentation
    detail, not a modeling one.

    Position ``""`` (missing-data signal in V3 box scores) is excluded,
    same as :func:`opponent_defrtg_by_position`.

    Returns a frame with one row per ``(opp_team_id, game_id, position)``
    and a single feature column ``opp_blk_allowed_vs_pos``. Joined onto
    per-player frames by :func:`add_matchup_features`.
    """
    if player_box.is_empty() or games.is_empty():
        return pl.DataFrame()

    opp_map = _opponent_map(games)
    pb = player_box.join(opp_map, on=["game_id", "team_id"], how="inner")
    pb = pb.filter(pl.col("position") != "")
    if pb.is_empty():
        return pl.DataFrame()

    # Per-(opp_team, game, position) blocks. Same aggregation-then-cumsum
    # idiom as opponent_defrtg_by_position — the cum_sum partition wants
    # one row per game per (opp_team, position) so the shift logic is
    # well-defined.
    blk = pb.group_by(
        ["opp_team_id", "season", "date", "game_id", "position"]
    ).agg(pl.col("blk").sum().alias("blk_allowed"))

    blk = blk.sort(["opp_team_id", "season", "position", "date", "game_id"])
    blk = blk.with_columns([
        (
            pl.col("blk_allowed").cum_sum().over(["opp_team_id", "season", "position"])
            - pl.col("blk_allowed")
        ).alias("cum_blk_prior"),
        (
            pl.col("date").cum_count().over(["opp_team_id", "season", "position"]) - 1
        ).cast(pl.Int64).alias("cum_games_prior"),
    ])

    blk = blk.with_columns(
        pl.when(pl.col("cum_games_prior") > 0)
        .then(pl.col("cum_blk_prior") / pl.col("cum_games_prior"))
        .otherwise(None)
        .alias("opp_blk_allowed_vs_pos")
    )
    return blk.select(
        ["opp_team_id", "game_id", "position", "opp_blk_allowed_vs_pos"]
    )


# ---------------------------------------------------------------------------
# add_matchup_features
# ---------------------------------------------------------------------------

def add_matchup_features(
    per_player: pl.DataFrame,
    team_rolling: pl.DataFrame,
    games: pl.DataFrame,
    *,
    defrtg_vs_pos: pl.DataFrame | None = None,
    blk_allowed_vs_pos: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join opponent-aware features onto per-player rows.

    Joins (each guarded — see notes below):
        - ``opp_team_id``                  from ``games`` via the long opp map
        - ``opp_def_rtg_10`` / ``opp_pace_10``
                                           from ``team_rolling`` renamed
        - ``h2h_last_meeting_margin``      per-game, derived inline
        - ``opp_def_rtg_vs_pos``           from ``defrtg_vs_pos`` if provided
        - ``opp_blk_allowed_vs_pos``       from ``blk_allowed_vs_pos`` if provided

    The two position-matchup tables are passed in (not derived inline)
    because they require ``player_box`` which isn't in this function's
    signature. The orchestrator (``build_feature_tables``) calls
    :func:`opponent_defrtg_by_position` and
    :func:`opponent_blocks_allowed_by_position` and passes results here.

    Args:
        per_player: per-(player, game) frame. Must include
            ``game_id``, ``team_id``, ``position``.
        team_rolling: per-(team, game) frame with at least
            ``t_def_rtg_10`` and ``t_pace_10``. Output of
            :func:`nba_sim.features.rolling.team_rolling`.
        games: per-game header (Game schema).
        defrtg_vs_pos: optional output of
            :func:`opponent_defrtg_by_position`. When ``None`` the
            ``opp_def_rtg_vs_pos`` column is simply not added.
        blk_allowed_vs_pos: optional output of
            :func:`opponent_blocks_allowed_by_position`. When ``None`` the
            ``opp_blk_allowed_vs_pos`` column is simply not added.

    Returns:
        ``per_player`` augmented with ``opp_team_id``, ``opp_def_rtg_10``,
        ``opp_pace_10``, ``h2h_last_meeting_margin``, and the position-
        matchup columns when their inputs are provided. Player rows with
        no opponent in the map (i.e. game was filtered out as dropped)
        are themselves dropped.
    """
    if per_player.is_empty():
        return per_player

    # 1. Identify opponent for each player row.
    opp_map = _opponent_map(games).select(["game_id", "team_id", "opp_team_id"])
    result = per_player.join(opp_map, on=["game_id", "team_id"], how="inner")

    # 2. Opp rolling: rename and join on (game_id, opp_team_id).
    #    `available` guards against team_rolling versions that don't carry
    #    a column we want (the rolling module's window set is configurable).
    wanted = {"t_def_rtg_10": "opp_def_rtg_10", "t_pace_10": "opp_pace_10"}
    available = {src: dst for src, dst in wanted.items() if src in team_rolling.columns}
    if available:
        opp_rolling = team_rolling.select([
            pl.col("game_id"),
            pl.col("team_id").alias("opp_team_id"),
            *[pl.col(src).alias(dst) for src, dst in available.items()],
        ])
        result = result.join(opp_rolling, on=["game_id", "opp_team_id"], how="left")

    # 3. H2H margin. PLAN.md §3.1: "0 if no prior meeting".
    h2h = head_to_head_last_margin(games)
    result = result.join(h2h, on="game_id", how="left")
    result = result.with_columns(
        pl.col("h2h_last_meeting_margin").fill_null(0.0)
    )

    # 4. Position matchup tables, if provided.
    if defrtg_vs_pos is not None and not defrtg_vs_pos.is_empty():
        result = result.join(
            defrtg_vs_pos,
            on=["game_id", "opp_team_id", "position"],
            how="left",
        )
    if blk_allowed_vs_pos is not None and not blk_allowed_vs_pos.is_empty():
        result = result.join(
            blk_allowed_vs_pos,
            on=["game_id", "opp_team_id", "position"],
            how="left",
        )

    return result
