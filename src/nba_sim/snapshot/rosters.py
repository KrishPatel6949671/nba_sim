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
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

from nba_sim.data.fetch import fetch_player_info, fetch_roster

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


def parse_weight(weight: str | None) -> float | None:
    """Parse a ``WEIGHT`` string ("215") into pounds (215.0).

    Returns ``None`` on empty / malformed input. Live (Phase 8) path only.
    """
    if not weight:
        return None
    try:
        return float(str(weight).strip())
    except ValueError:
        return None


def parse_experience(exp: str | None) -> int | None:
    """Parse ``CommonTeamRoster.EXP`` into seasons of experience.

    ``"R"`` (rookie) → 0; an integer-as-string ("12") → 12; empty / ``None``
    / unparseable → ``None``. Live (Phase 8) path only.
    """
    if exp is None:
        return None
    s = str(exp).strip()
    if not s:
        return None
    if s.upper() == "R":
        return 0
    try:
        return int(s)
    except ValueError:
        return None


def parse_birth_date(value: str | None) -> _dt.date | None:
    """Parse a roster ``BIRTH_DATE`` into a :class:`datetime.date`.

    Handles nba_api's ISO timestamp form ("1984-12-30T00:00:00", also a bare
    "1984-12-30") and the display form ("DEC 30, 1984"). Empty / ``None`` /
    unparseable → ``None``. Live (Phase 8) path only.
    """
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    # ISO first — CommonTeamRoster usually returns "YYYY-MM-DDT00:00:00";
    # datetime.fromisoformat also accepts a bare "YYYY-MM-DD" (py3.11+).
    try:
        return _dt.datetime.fromisoformat(s).date()
    except ValueError:
        pass
    # Display form, e.g. "DEC 30, 1984".
    try:
        return _dt.datetime.strptime(s, "%b %d, %Y").date()
    except ValueError:
        return None


def build_rosters(
    *,
    games: pl.DataFrame,
    player_box: pl.DataFrame,
    as_of_date: _dt.date,
    season: int,
    offline: bool = True,
    refresh: bool = False,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Return ``(rosters, malformed)`` for the snapshot.

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
        ``True`` → derive from interim appearances (no nba_api; ``games`` /
        ``player_box`` are consumed). ``False`` → live ``CommonTeamRoster``
        per team (:func:`_build_live_rosters`); ``games`` / ``player_box`` /
        ``as_of_date`` are unused — only ``season`` selects the roster season.
    refresh
        Live path only: bypass the on-disk fetch cache for the roster /
        player-info endpoints (forwarded to :func:`fetch_roster`).

    Returns
    -------
    ``(rosters, malformed)`` where ``rosters`` is a frame with exactly
    ``ROSTER_COLUMNS`` and ``malformed`` is the (possibly empty) list of
    rejected live rows / failed team fetches for ``qa_report.json``. Offline
    always returns ``malformed == []`` and null bio; live carries bio from
    ``CommonTeamRoster`` (backfilled from ``CommonPlayerInfo`` only where
    missing). ``position`` offline is the most recent non-empty box-score
    position per ``(team_id, player_id)`` on/before as_of, else "".
    """
    if not offline:
        return _build_live_rosters(season, refresh=refresh)

    # Box scores carry only game_id; pull date/season off the game header so we
    # can apply the leakage rule (date < as_of) and restrict to the as-of
    # season. inner join also drops any appearance whose game we don't have a
    # header for. games has season/date and player_box has neither, so there's
    # no column clash.
    appearances = player_box.join(
        games.select(["game_id", "date", "season"]), on="game_id", how="inner"
    ).filter((pl.col("date") < as_of_date) & (pl.col("season") == season))

    if appearances.is_empty():
        return pl.DataFrame(schema=_ROSTER_SCHEMA), []

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
    return out, []


# ---------------------------------------------------------------------------
# Live (Phase 8) — CommonTeamRoster per team, validated + bio-backfilled.
# ---------------------------------------------------------------------------


def _get_team_list() -> list[dict[str, Any]]:
    """Canonical ``{id, abbreviation, ...}`` for the 30 NBA teams.

    Lazy import keeps ``import nba_sim.snapshot.rosters`` free of nba_api (the
    offline path never needs it). Tests monkeypatch this to inject a fake team
    list, so the live builder is exercisable without the static dataset.
    """
    from nba_api.stats.static import teams

    return list(teams.get_teams())


def _player_info_bio(player_id: int, *, refresh: bool = False) -> dict[str, Any] | None:
    """Pull ``(height_in, weight_lbs, experience_years)`` from
    ``CommonPlayerInfo`` to backfill bio the roster row left empty.

    Best-effort: returns ``None`` on any fetch / empty-frame failure (bio is
    nullable, §13.3). Only invoked for the rare player whose ``CommonTeamRoster``
    row is missing a bio field, so the refresh stays roster-bound (§14.2).
    """
    try:
        df = fetch_player_info(player_id, refresh=refresh)
    except Exception:  # network/HTTP after tenacity retries — bio is optional
        return None
    if df.is_empty():
        return None
    rec = df.row(0, named=True)
    exp_raw = rec.get("SEASON_EXP")
    experience_years = (
        int(exp_raw)
        if isinstance(exp_raw, int | float)
        else parse_experience(None if exp_raw is None else str(exp_raw))
    )
    height = rec.get("HEIGHT")
    weight = rec.get("WEIGHT")
    return {
        "height_in": parse_height_to_inches(None if height is None else str(height)),
        "weight_lbs": parse_weight(None if weight is None else str(weight)),
        "experience_years": experience_years,
    }


def _roster_row_from_entry(
    entry: RawRosterEntry, team_id: int, team_abbr: str, *, refresh: bool = False
) -> dict[str, Any]:
    """Map a validated ``RawRosterEntry`` onto one ``ROSTER_COLUMNS`` dict.

    ``team_id`` / ``team_abbr`` come from the canonical team list (authoritative;
    ``CommonTeamRoster.TeamID`` is unreliable). Bio is parsed from the roster
    row and, only where a field is missing, backfilled from ``CommonPlayerInfo``.
    """
    height_in = parse_height_to_inches(entry.HEIGHT)
    weight_lbs = parse_weight(entry.WEIGHT)
    experience_years = parse_experience(entry.EXP)
    birth_date = parse_birth_date(entry.BIRTH_DATE)

    if height_in is None or weight_lbs is None or experience_years is None:
        info = _player_info_bio(entry.PLAYER_ID, refresh=refresh)
        if info is not None:
            if height_in is None:
                height_in = info["height_in"]
            if weight_lbs is None:
                weight_lbs = info["weight_lbs"]
            if experience_years is None:
                experience_years = info["experience_years"]

    return {
        "team_id": team_id,
        "team_abbr": team_abbr,
        "player_id": int(entry.PLAYER_ID),
        "player_name": entry.PLAYER,
        "position": normalize_position(entry.POSITION),
        "height_in": height_in,
        "weight_lbs": weight_lbs,
        "jersey": entry.NUM if entry.NUM else None,
        "experience_years": experience_years,
        "birth_date": birth_date,
    }


def _build_live_rosters(
    season: int, *, refresh: bool = False
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fetch + validate live ``CommonTeamRoster`` for all 30 teams (§14.2).

    For each canonical team, fetch the roster (rate-limited + cached via
    :func:`fetch_roster`), validate each row through :class:`RawRosterEntry`,
    and assemble a ``ROSTER_COLUMNS`` row (bio from the roster, backfilled from
    ``CommonPlayerInfo`` only where missing). A failed team fetch or a row that
    fails validation is appended to ``malformed`` (→ ``qa_report.json``) so one
    bad entry never poisons the whole refresh, mirroring the v1 ETL's QA
    discipline.
    """
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []

    for team in _get_team_list():
        team_id = int(team["id"])
        team_abbr = str(team["abbreviation"])
        try:
            raw = fetch_roster(team_id, season, refresh=refresh)
        except Exception as e:  # network/HTTP after tenacity retries
            malformed.append({"team_id": team_id, "error": f"fetch_failed: {e}"})
            continue

        for rec in raw.iter_rows(named=True):
            try:
                entry = RawRosterEntry.model_validate(rec)
            except ValidationError as e:
                malformed.append({
                    "team_id": team_id,
                    "player_id": rec.get("PLAYER_ID"),
                    "error": f"validation: {e.errors(include_url=False)}",
                })
                continue
            rows.append(_roster_row_from_entry(entry, team_id, team_abbr, refresh=refresh))

    if not rows:
        return pl.DataFrame(schema=_ROSTER_SCHEMA), malformed

    frame = (
        pl.DataFrame(rows)
        .select([pl.col(c).cast(dt) for c, dt in _ROSTER_SCHEMA.items()])
        .unique(subset=["team_id", "player_id"], keep="first")
        .sort(["team_id", "player_id"])
    )
    return frame, malformed
