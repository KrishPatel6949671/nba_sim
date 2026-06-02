"""Roster resolution for the snapshot layer.

Produces ``rosters.parquet`` — one row per ``(team_id, player_id)`` active
in the as-of season (v2PLAN.md §13.3).

Two sources, selected by the refresh's ``offline`` flag:

- **offline (Phase 6 — the only path wired in this module today):** derive
  the roster from interim ``player_box`` appearances in the as-of season,
  restricted to ``date < as_of`` — i.e. "every player who has actually
  suited up for the team this season, given the data we have." Bio fields
  (height / weight / experience / birth date / jersey) aren't in box scores
  and are written null.
- **live (Phase 8):** ``CommonTeamRoster`` per team via
  :func:`nba_sim.data.fetch.fetch_roster`, validated through
  :class:`RawRosterEntry` and augmented with ``CommonPlayerInfo`` for bio.

Resolved decisions (against v2PLAN.md ambiguities — see refresh.py header):
- ``two_way`` is **dropped** from the schema for v2: no nba_api endpoint
  exposes a reliable two-way-contract flag. Revisit with a contract /
  injury source in v3.
- bio fields are **nullable**; ``position`` is non-null, defaulting to ``""``
  when unknown, and per ``(team, player)`` is the player's most recent
  non-empty box-score position on/before as_of.

Leakage discipline: offline rosters only include appearances with
``date < as_of`` — a player who debuts on/after as_of is not yet "known."
Rosters themselves carry no rolling features, so there is no same-game
leakage surface here; the discipline is purely about *which players exist*
as of the reference date.
"""

from __future__ import annotations

import datetime as _dt

import polars as pl
from pydantic import BaseModel, ConfigDict

# Column order of rosters.parquet (§13.3, two_way dropped). Tests introspect
# this so the schema can't silently drift.
ROSTER_COLUMNS: tuple[str, ...] = (
    "team_id",
    "team_abbr",
    "player_id",
    "player_name",
    "position",
    "height_in",
    "weight_lbs",
    "jersey",
    "experience_years",
    "birth_date",
)

# Bio columns that are nullable (unavailable offline / source-dependent live).
NULLABLE_BIO_COLUMNS: tuple[str, ...] = (
    "height_in",
    "weight_lbs",
    "jersey",
    "experience_years",
    "birth_date",
)

# The exact dtype contract for rosters.parquet (§13.3). Used to (a) build a
# correctly-typed empty frame when there are no appearances and (b) cast the
# assembled frame so the parquet's dtypes never drift from the documented
# schema. Order matches ROSTER_COLUMNS.
_ROSTER_SCHEMA: dict[str, pl.DataType] = {
    "team_id": pl.Int64(),
    "team_abbr": pl.Utf8(),
    "player_id": pl.Int64(),
    "player_name": pl.Utf8(),
    "position": pl.Utf8(),
    "height_in": pl.Float64(),
    "weight_lbs": pl.Float64(),
    "jersey": pl.Utf8(),
    "experience_years": pl.Int64(),
    "birth_date": pl.Date(),
}


class RawRosterEntry(BaseModel):
    """Mirror of a ``CommonTeamRoster`` row (dataset 0). Loose / API-shaped,
    like the ``Raw*`` models in :mod:`nba_sim.data.schema`.

    Used by the **live** (Phase 8) path to validate fetched rosters before
    they're typed into ``rosters.parquet``; malformed rows are routed to
    ``qa_report.json`` so one bad entry doesn't poison the refresh.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    PLAYER_ID: int
    PLAYER: str = ""
    TeamID: int | None = None
    NUM: str | None = None          # jersey number (nullable; "" in source)
    POSITION: str = ""              # "G" / "F" / "C" / "G-F" / "F-C" / ...
    HEIGHT: str | None = None       # "6-9"
    WEIGHT: str | None = None       # "215"
    BIRTH_DATE: str | None = None   # "DEC 30, 1984" or ISO (source-dependent)
    AGE: float | None = None
    EXP: str | None = None          # "R" (rookie) or an integer-as-string
    SCHOOL: str | None = None


def normalize_position(pos: str | None) -> str:
    """Map a raw POSITION onto the ``{"G", "F", "C", ""}`` token space.

    ``CommonTeamRoster`` returns hyphenated combos ("G-F", "F-C", "C-F"); we
    keep the **primary** group (text before the first "-"), upper-cased;
    anything unrecognized (or empty) maps to "".

    Used by the **live** (Phase 8) path only. The offline path stores the raw
    box-score position verbatim, because that is exactly what the v1
    training pipeline fed the encoder: ``_position_one_hot`` buckets only
    ``{G, F, C, ""}`` and treats combos like "F-C" as all-zeros, and the
    position-matchup features group on the raw string. Normalizing offline
    would change the very encoding the model was trained on.
    """
    if not pos:
        return ""
    primary = pos.split("-", 1)[0].strip().upper()
    return primary if primary in {"G", "F", "C"} else ""


def parse_height_to_inches(height: str | None) -> float | None:
    """Parse a ``"feet-inches"`` string ("6-9") into total inches (81.0).

    Returns ``None`` on empty / malformed input. Only the live (Phase 8)
    path populates height; offline rows leave it null.
    """
    if not height:
        return None
    parts = height.split("-", 1)
    if len(parts) != 2:
        return None
    try:
        feet, inches = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return float(feet * 12 + inches)


def build_rosters(
    *,
    games: pl.DataFrame,
    player_box: pl.DataFrame,
    as_of_date: _dt.date,
    season: int,
    offline: bool = True,
) -> pl.DataFrame:
    """Return the rosters frame (``ROSTER_COLUMNS`` schema) for the snapshot.

    Parameters
    ----------
    games, player_box
        Interim frames concatenated across the seasons up to ``as_of`` and
        already filtered to ``date < as_of_date``. ``games`` is needed to
        attach ``date`` / ``season`` to box rows (box scores carry only
        ``game_id``).
    as_of_date
        Snapshot reference date.
    season
        Season start-year ``as_of_date`` falls in
        (:func:`nba_sim.snapshot.refresh.season_for_date`); restricts the
        roster to that season's appearances.
    offline
        Phase 6: always ``True`` → derive from interim appearances. The
        ``False`` (live ``CommonTeamRoster``) path is wired in Phase 8.

    Returns
    -------
    A frame with exactly ``ROSTER_COLUMNS``. Offline rows have null bio.
    ``position`` is the most recent non-empty box-score position per
    ``(team_id, player_id)`` on/before as_of, else "".
    """
    if not offline:
        raise NotImplementedError(
            "live CommonTeamRoster fetch is Phase 8; pass offline=True"
        )

    # Box scores carry only game_id; pull date/season off the game header so we
    # can apply the leakage rule (date < as_of) and restrict to the as-of
    # season. inner join also drops any appearance whose game we don't have a
    # header for. games has season/date and player_box has neither, so there's
    # no column clash.
    appearances = player_box.join(
        games.select(["game_id", "date", "season"]), on="game_id", how="inner"
    ).filter((pl.col("date") < as_of_date) & (pl.col("season") == season))

    if appearances.is_empty():
        return pl.DataFrame(schema=_ROSTER_SCHEMA)

    # Most-recent value wins for trade / name-change churn: sort ascending so
    # `.last()` within each (team, player) group is the latest appearance.
    # Polars preserves input row order within groups, so this is well-defined.
    appearances = appearances.sort(["date", "game_id"])

    identity = appearances.group_by(["team_id", "player_id"]).agg(
        team_abbr=pl.col("team_abbr").last(),
        player_name=pl.col("player_name").last(),
    )

    # Position: most recent *non-empty* box position (a player can have ""
    # in a game where they came off the bench). Left-join so a player who is
    # always "" still appears, with position "".
    positions = (
        appearances.filter(pl.col("position") != "")
        .group_by(["team_id", "player_id"])
        .agg(position=pl.col("position").last())
    )

    out = (
        identity.join(positions, on=["team_id", "player_id"], how="left")
        .with_columns(
            pl.col("position").fill_null(""),
            # Bio is unavailable from box scores — null at the documented dtype.
            pl.lit(None, dtype=pl.Float64).alias("height_in"),
            pl.lit(None, dtype=pl.Float64).alias("weight_lbs"),
            pl.lit(None, dtype=pl.Utf8).alias("jersey"),
            pl.lit(None, dtype=pl.Int64).alias("experience_years"),
            pl.lit(None, dtype=pl.Date).alias("birth_date"),
        )
        # Select in ROSTER_COLUMNS order and pin every dtype in one pass
        # (_ROSTER_SCHEMA is keyed in that order) — guarantees the §13.3
        # column + dtype contract.
        .select([pl.col(c).cast(dt) for c, dt in _ROSTER_SCHEMA.items()])
        .sort(["team_id", "player_id"])
    )
    return out
