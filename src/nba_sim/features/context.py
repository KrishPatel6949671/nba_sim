"""Game-context features.

Features (PLAN.md §3.1):
    is_home, rest_days, b2b, is_3in4, is_4in6,
    season_phase, day_of_week, month,
    altitude_ft, travel_miles_prev.

All features are produced per (team, game) pair — we fan out the wide
``games`` header to a long format with one row per team-game, then derive
schedule features (rest, b2b, density, travel) via chronological shifts
within each ``(team_id, season)`` partition.

Convention: ``rest_days`` is **nights of rest** = ``(date - prev_date).days
- 1``. Played yesterday → 0 rest → ``b2b = True``. This matches PLAN's
"clipped 0..5" range (the lower bound only makes sense under this
convention; raw day-diff has a minimum of 1).

Leakage discipline (PLAN.md §3.2): every feature is derivable from the
target game's own attributes (date, location, ``is_playoffs``) and the
team's strictly prior games (via ``shift``). The cross-season partition
ensures a June playoff game doesn't influence the next October's opener.

Arena data lives in a single dict ``_ARENAS: {team_abbr: (lat, lng, alt)}``,
read by both ``arena_altitude`` and ``travel_distance_miles``. Coverage:
30 current franchises + historicals active in 2000-2024 (SEA pre-2008,
VAN pre-2001, NJN pre-2012, plus Charlotte/New Orleans abbreviation
transitions). Coordinates are the team's *current* arena — relocations
within the active window (Sacramento's Golden 1 in 2016, etc.) are not
modeled. v2 can add a date-aware lookup.
"""

from __future__ import annotations

import math

import polars as pl


# NIST mean Earth radius, miles. Used by both the Python and Polars
# haversine surfaces — kept in one constant so they stay in lockstep.
_EARTH_RADIUS_MILES: float = 3959.0

# `_ARENAS` schema: team_abbr -> (lat_deg, lng_deg, altitude_ft).
# Single source of truth for both lookups. Lat/lng to ~4 decimal places
# (~10 m precision — plenty for cross-country haversines).
_ARENAS: dict[str, tuple[float, float, float]] = {
    "ATL": (33.7573,  -84.3963, 1050.0),
    "BOS": (42.3662,  -71.0621,   20.0),
    "BKN": (40.6826,  -73.9754,   30.0),
    "CHA": (35.2251,  -80.8392,  751.0),
    "CHI": (41.8807,  -87.6742,  600.0),
    "CLE": (41.4965,  -81.6882,  653.0),
    "DAL": (32.7905,  -96.8104,  430.0),
    "DEN": (39.7487, -105.0077, 5280.0),  # Mile High, the obvious outlier.
    "DET": (42.3411,  -83.0553,  600.0),
    "GSW": (37.7680, -122.3877,   13.0),
    "HOU": (29.7508,  -95.3621,   80.0),
    "IND": (39.7640,  -86.1555,  715.0),
    "LAC": (34.0430, -118.2673,  269.0),  # Same building as LAL.
    "LAL": (34.0430, -118.2673,  269.0),
    "MEM": (35.1382,  -90.0505,  337.0),
    "MIA": (25.7814,  -80.1870,    7.0),
    "MIL": (43.0451,  -87.9170,  605.0),
    "MIN": (44.9795,  -93.2761,  840.0),
    "NOP": (29.9490,  -90.0820,    7.0),
    "NYK": (40.7505,  -73.9934,   33.0),
    "OKC": (35.4634,  -97.5151, 1201.0),
    "ORL": (28.5392,  -81.3839,  100.0),
    "PHI": (39.9012,  -75.1720,   39.0),
    "PHX": (33.4458, -112.0712, 1086.0),
    "POR": (45.5316, -122.6668,   50.0),
    "SAC": (38.5806, -121.4995,   30.0),
    "SAS": (29.4271,  -98.4375,  650.0),
    "TOR": (43.6435,  -79.3791,  249.0),
    "UTA": (40.7683, -111.9011, 4226.0),  # Second-highest after Denver.
    "WAS": (38.8981,  -77.0209,   25.0),
    # Historical franchises active during the 2000-2024 modeling window.
    "SEA": (47.6221, -122.3540,  175.0),  # Pre-2008 SuperSonics.
    "VAN": (49.2778, -123.1089,  223.0),  # Pre-2001 Grizzlies.
    "NJN": (40.8128,  -74.0691,   33.0),  # Pre-2012 Nets.
    # Abbreviation aliases for the same arena over the team's history.
    "CHH": (35.2251,  -80.8392,  751.0),  # Charlotte Hornets (original).
    "CHO": (35.2251,  -80.8392,  751.0),  # Bobcats transition.
    "NOH": (29.9490,  -90.0820,    7.0),  # New Orleans Hornets era.
    "NOK": (29.9490,  -90.0820,    7.0),  # Briefly Oklahoma City Hornets.
}


# Default for unknown abbreviations. 50 ft ≈ "low elevation, basically sea
# level" — minimizes the altitude signal when we don't know it. Loud
# alternatives (raise, NaN) would noise up an otherwise-clean feature
# column on every ETL run.
_DEFAULT_ALTITUDE_FT: float = 50.0


# Season phase month buckets (PLAN.md §3.1). NBA regular season runs
# Oct-April; preseason in Sep-Oct lumped into "early" since v1 doesn't
# distinguish it. Playoffs are detected via the is_playoffs flag, not
# the month.
_PHASE_EARLY_MONTHS: tuple[int, ...] = (9, 10, 11)
_PHASE_MID_MONTHS:   tuple[int, ...] = (12, 1, 2)
_PHASE_LATE_MONTHS:  tuple[int, ...] = (3, 4)


# ---------------------------------------------------------------------------
# Public scalar lookups
# ---------------------------------------------------------------------------

def arena_altitude(team_abbr: str) -> float:
    """Elevation in feet for the team's home arena.

    Returns 50.0 for unknown abbreviations rather than raising — we'd
    rather emit a "low" value than crash mid-pipeline on a data oddity
    (a typo or a team abbr we forgot to add).
    """
    if team_abbr in _ARENAS:
        return _ARENAS[team_abbr][2]
    return _DEFAULT_ALTITUDE_FT


def travel_distance_miles(prev_team_abbr: str, next_team_abbr: str) -> float:
    """Great-circle distance between two arenas, miles.

    Returns NaN for unknown abbreviations — distance has no sensible
    fallback value, so we surface the gap loudly. Returns 0.0 when both
    abbreviations point to the same building (LAL <-> LAC at
    Crypto.com Arena).
    """
    if prev_team_abbr not in _ARENAS or next_team_abbr not in _ARENAS:
        return float("nan")
    lat1, lng1, _ = _ARENAS[prev_team_abbr]
    lat2, lng2, _ = _ARENAS[next_team_abbr]
    return _haversine_miles(lat1, lng1, lat2, lng2)


# ---------------------------------------------------------------------------
# Haversine — two surfaces, one formula.
# ---------------------------------------------------------------------------

def _haversine_miles(
    lat1: float, lng1: float, lat2: float, lng2: float
) -> float:
    """Python-scalar haversine for ``travel_distance_miles``."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_MILES * c


def _haversine_expr(
    lat1: pl.Expr, lng1: pl.Expr, lat2: pl.Expr, lng2: pl.Expr,
) -> pl.Expr:
    """Polars-expression haversine over columns. Equivalent to the Python
    version above; runs in-engine across the whole frame at once.

    Polars 1.17 expressions don't expose ``atan2``, so we use the
    arcsin-form of haversine which is mathematically identical over the
    haversine's domain (0 ≤ a ≤ 1):

        c = 2 * atan2(sqrt(a), sqrt(1-a))   ⟺   2 * arcsin(sqrt(a))

    Null in any input → null output (Polars propagates nulls through
    arithmetic). First game of a season has no prior arena, so
    travel_miles_prev correctly emits null there.
    """
    phi1 = lat1.radians()
    phi2 = lat2.radians()
    dphi = phi2 - phi1
    dlam = (lng2 - lng1).radians()
    a = (dphi / 2).sin().pow(2) + phi1.cos() * phi2.cos() * (dlam / 2).sin().pow(2)
    return _EARTH_RADIUS_MILES * 2 * a.sqrt().arcsin()


def _arenas_frame() -> pl.DataFrame:
    """Project ``_ARENAS`` into a Polars frame for joins. Built on demand
    so module import stays cheap."""
    return pl.DataFrame(
        [
            {"team_abbr": abbr, "_lat": lat, "_lng": lng, "_alt": alt}
            for abbr, (lat, lng, alt) in _ARENAS.items()
        ]
    )


# ---------------------------------------------------------------------------
# Frame-level: add_context_features
# ---------------------------------------------------------------------------

def add_context_features(games: pl.DataFrame) -> pl.DataFrame:
    """Compute per-(team, game) context features from a games header table.

    Fans out each game to two rows (one per team) and emits:
        is_home, rest_days, b2b, is_3in4, is_4in6,
        season_phase, day_of_week, month,
        altitude_ft, travel_miles_prev.

    Schedule features (rest, b2b, density, travel) are computed within
    ``(team_id, season)`` partitions — the first game of each new season
    has null rest/b2b/density and null travel.

    Dropped games (``dropped=True``) are excluded; including them would
    corrupt the shift-based features (a "game" that didn't actually
    happen would invent a fake rest interval).

    Returns a long frame keyed on ``(game_id, team_id)`` — same grain as
    :func:`nba_sim.features.matchup._opponent_map`.
    """
    if games.is_empty():
        return games

    base = games.filter(~pl.col("dropped"))

    # Fan-out to long format. The arena's team_abbr is always the home
    # team's, regardless of which row of the pair we're on — the game is
    # physically at the home team's building.
    home_view = base.select(
        "game_id", "season", "date", "is_playoffs",
        pl.col("home_team_id").alias("team_id"),
        pl.col("home_team_abbr").alias("team_abbr"),
        pl.lit(True).alias("is_home"),
        pl.col("home_team_abbr").alias("_arena_team_abbr"),
    )
    away_view = base.select(
        "game_id", "season", "date", "is_playoffs",
        pl.col("away_team_id").alias("team_id"),
        pl.col("away_team_abbr").alias("team_abbr"),
        pl.lit(False).alias("is_home"),
        pl.col("home_team_abbr").alias("_arena_team_abbr"),
    )
    long_df = pl.concat([home_view, away_view])

    # Sort within (team_id, season) by date so shift(N) picks up the
    # game N positions back chronologically. game_id breaks any same-day
    # ties deterministically (modern NBA has no same-day doubleheaders
    # but the historical data is messy enough to warrant the safety).
    long_df = long_df.sort(["team_id", "season", "date", "game_id"])

    # Shift-based features. We compute the three needed "previous game"
    # date columns in one pass, then derive day-gap deltas, then derive
    # the actual features. Splitting into steps lets each later step
    # reference the earlier columns by name without recomputing the shift.
    long_df = long_df.with_columns([
        pl.col("date").shift(1).over(["team_id", "season"]).alias("_prev_date"),
        pl.col("date").shift(2).over(["team_id", "season"]).alias("_date_2_back"),
        pl.col("date").shift(3).over(["team_id", "season"]).alias("_date_3_back"),
        pl.col("_arena_team_abbr").shift(1).over(["team_id", "season"]).alias("_prev_arena_team_abbr"),
    ])
    long_df = long_df.with_columns([
        (pl.col("date") - pl.col("_prev_date")).dt.total_days().alias("_gap_days"),
        (pl.col("date") - pl.col("_date_2_back")).dt.total_days().alias("_gap_2back"),
        (pl.col("date") - pl.col("_date_3_back")).dt.total_days().alias("_gap_3back"),
    ])

    long_df = long_df.with_columns([
        # rest_days = nights of rest = gap - 1, clipped 0..5.
        # Null when there's no prior game in this (team, season).
        pl.when(pl.col("_gap_days").is_null())
        .then(None)
        .otherwise((pl.col("_gap_days") - 1).clip(0, 5))
        .alias("rest_days"),

        # b2b: exactly 1 calendar day between games.
        pl.when(pl.col("_gap_days").is_null())
        .then(None)
        .otherwise(pl.col("_gap_days") == 1)
        .alias("b2b"),

        # 3 games in 4 days: today + 2 prior games span ≤ 3 days.
        pl.when(pl.col("_gap_2back").is_null())
        .then(None)
        .otherwise(pl.col("_gap_2back") <= 3)
        .alias("is_3in4"),

        # 4 games in 6 days: today + 3 prior games span ≤ 5 days.
        pl.when(pl.col("_gap_3back").is_null())
        .then(None)
        .otherwise(pl.col("_gap_3back") <= 5)
        .alias("is_4in6"),

        # day_of_week: 0=Mon..6=Sun. Polars' dt.weekday() is 1..7 (Mon=1).
        (pl.col("date").dt.weekday() - 1).alias("day_of_week"),

        pl.col("date").dt.month().alias("month"),

        # season_phase: is_playoffs takes precedence, then month bucket.
        pl.when(pl.col("is_playoffs"))
        .then(pl.lit("playoffs"))
        .when(pl.col("date").dt.month().is_in(list(_PHASE_EARLY_MONTHS)))
        .then(pl.lit("early"))
        .when(pl.col("date").dt.month().is_in(list(_PHASE_MID_MONTHS)))
        .then(pl.lit("mid"))
        .when(pl.col("date").dt.month().is_in(list(_PHASE_LATE_MONTHS)))
        .then(pl.lit("late"))
        # Edge case: a non-playoff game in May-August (shouldn't happen
        # with NBA data but defensive). Bucket into "late" — closest to
        # the natural extrapolation.
        .otherwise(pl.lit("late"))
        .alias("season_phase"),
    ])

    # Altitude lookup: left join the arenas frame on team_abbr.
    arenas = _arenas_frame()
    long_df = long_df.join(
        arenas.select(["team_abbr", pl.col("_alt").alias("altitude_ft")]),
        on="team_abbr", how="left",
    ).with_columns(
        # Backfill unknowns with the default. Same fall-back as the
        # scalar arena_altitude() — the two surfaces agree by construction.
        pl.col("altitude_ft").fill_null(_DEFAULT_ALTITUDE_FT)
    )

    # Travel: join current and previous arena coords, compute haversine
    # entirely in Polars expressions (no map_elements / Python callbacks).
    long_df = long_df.join(
        arenas.select([
            pl.col("team_abbr").alias("_arena_team_abbr"),
            pl.col("_lat").alias("_cur_lat"),
            pl.col("_lng").alias("_cur_lng"),
        ]),
        on="_arena_team_abbr", how="left",
    ).join(
        arenas.select([
            pl.col("team_abbr").alias("_prev_arena_team_abbr"),
            pl.col("_lat").alias("_prev_lat"),
            pl.col("_lng").alias("_prev_lng"),
        ]),
        on="_prev_arena_team_abbr", how="left",
    )

    long_df = long_df.with_columns(
        _haversine_expr(
            pl.col("_prev_lat"), pl.col("_prev_lng"),
            pl.col("_cur_lat"),  pl.col("_cur_lng"),
        ).alias("travel_miles_prev")
    )

    # Drop scratch.
    scratch_cols = [
        "_prev_date", "_date_2_back", "_date_3_back",
        "_arena_team_abbr", "_prev_arena_team_abbr",
        "_gap_days", "_gap_2back", "_gap_3back",
        "_cur_lat", "_cur_lng", "_prev_lat", "_prev_lng",
    ]
    return long_df.drop(scratch_cols)
