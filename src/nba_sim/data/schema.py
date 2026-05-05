"""Typed schemas for raw / interim / processed records and output box scores.

Three layers (see PLAN.md §2.3):

1. **Raw** — mirrors nba_api V3 fields 1:1, optional where the API may omit
   data (e.g. counting stats are ``None`` for DNPs). ``extra="ignore"`` so
   unknown V3 fields don't crash; we keep what we use.
2. **Interim** — typed, validated, ``minutes`` parsed to float, DNP flags
   set, all shooting/rebound invariants enforced via ``model_validator``.
3. **Processed** — denormalized, feature-engineered, ready for the DataLoader.

Plus the public box-score output types returned by :func:`simulate_game`.

Conversion: each interim model exposes a ``from_raw(...)`` classmethod that
takes the corresponding raw model (or, for ``Game``, the *pair* of raw rows
that describe one game from each team's perspective).
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, ConfigDict, Field, model_validator


def parse_minutes_mmss(s: str | None) -> float:
    """Parse V3 traditional-box ``minutes`` strings into float minutes.

    V3 returns ``"mm:ss"`` (e.g. ``"34:12"``) for players who played, and an
    empty string ``""`` for DNPs / inactives. Returns 0.0 for empty/None.
    """
    if not s:
        return 0.0
    m_str, s_str = s.split(":", 1)
    return int(m_str) + int(s_str) / 60.0


# ---------------------------------------------------------------------------
# Raw layer — loose, API-shaped.
# ---------------------------------------------------------------------------


class RawGame(BaseModel):
    """Raw row from ``LeagueGameFinder``. One row per *team-game* — every
    game appears twice in the source dataframe (once per team)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    SEASON_ID: str          # e.g. "22023" (regular) or "42023" (playoffs); last 4 chars = start year
    TEAM_ID: int
    TEAM_ABBREVIATION: str
    GAME_ID: str            # zero-padded; never cast to int
    GAME_DATE: str          # ISO date "YYYY-MM-DD"
    MATCHUP: str            # "BOS vs. LAL" (home) or "BOS @ LAL" (away)
    WL: str | None = None
    PTS: int | None = None
    MIN: int | None = None  # team total minutes (~240 in regulation, more for OT)


class RawPlayerBoxLine(BaseModel):
    """Raw row from ``BoxScoreTraditionalV3`` player frame (dataset 0).

    Counting stats are ``None`` for DNPs (where ``minutes == ""``). Don't
    coerce to int at this layer — the absence is meaningful.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    gameId: str
    teamId: int
    teamTricode: str
    personId: int
    firstName: str = ""
    familyName: str = ""
    position: str = ""      # "F" / "G" / "C" / "F-C" / "" — empty for non-starters
    comment: str = ""       # "DNP - Coach's Decision", "Inactive - ...", etc.
    minutes: str = ""       # "mm:ss" or ""
    fieldGoalsMade: int | None = None
    fieldGoalsAttempted: int | None = None
    threePointersMade: int | None = None
    threePointersAttempted: int | None = None
    freeThrowsMade: int | None = None
    freeThrowsAttempted: int | None = None
    reboundsOffensive: int | None = None
    reboundsDefensive: int | None = None
    reboundsTotal: int | None = None
    assists: int | None = None
    steals: int | None = None
    blocks: int | None = None
    turnovers: int | None = None
    foulsPersonal: int | None = None
    points: int | None = None
    plusMinusPoints: float | None = None


class RawTeamBoxLine(BaseModel):
    """Raw row from ``BoxScoreTraditionalV3`` team-totals frame (dataset 2).

    Team-level totals are always present (a team always has stats even if
    individual players DNP'd), so the counting stats are non-optional.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    gameId: str
    teamId: int
    teamTricode: str
    minutes: str = ""       # "mm:ss" — team total, ~"240:00" in regulation
    fieldGoalsMade: int = 0
    fieldGoalsAttempted: int = 0
    threePointersMade: int = 0
    threePointersAttempted: int = 0
    freeThrowsMade: int = 0
    freeThrowsAttempted: int = 0
    reboundsOffensive: int = 0
    reboundsDefensive: int = 0
    reboundsTotal: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    foulsPersonal: int = 0
    points: int = 0
    plusMinusPoints: float = 0.0


class RawRoster(BaseModel):
    model_config = {"extra": "allow"}


class RawPlayerInfo(BaseModel):
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Interim layer — strictly typed. Validators enforce make <= attempt etc.
# ---------------------------------------------------------------------------


class Game(BaseModel):
    """Per-game header. Built from the *pair* of ``RawGame`` rows for a
    given ``GAME_ID`` (one per team in ``LeagueGameFinder``)."""

    game_id: str               # zero-padded; matches V3's `gameId`
    season: int                # start year (2023-24 -> 2023)
    date: _dt.date
    home_team_id: int
    away_team_id: int
    home_team_abbr: str
    away_team_abbr: str
    home_pts: int
    away_pts: int
    is_overtime: bool = False
    is_playoffs: bool = False
    dropped: bool = False      # flagged malformed rows
    dropped_reason: str | None = None

    @classmethod
    def from_raw_pair(cls, rows: list[RawGame]) -> "Game":
        """Combine the two ``RawGame`` rows for one game into a single header.

        ``LeagueGameFinder`` emits one row per team. The home/away split is
        encoded in the ``MATCHUP`` string: ``"BOS vs. LAL"`` is the home
        team's row; ``"BOS @ LAL"`` is the away team's row.
        """
        if len(rows) != 2:
            raise ValueError(f"Expected 2 RawGame rows per game, got {len(rows)}")
        if rows[0].GAME_ID != rows[1].GAME_ID:
            raise ValueError(
                f"Mismatched GAME_IDs: {rows[0].GAME_ID!r} vs {rows[1].GAME_ID!r}"
            )
        home_idx = 0 if "vs." in rows[0].MATCHUP else 1
        home, away = rows[home_idx], rows[1 - home_idx]
        # SEASON_ID prefix: '1'=preseason, '2'=regular, '4'=playoffs, '5'=play-in
        season = int(home.SEASON_ID[-4:])
        is_playoffs = home.SEASON_ID.startswith("4")
        # Team total minutes scale linearly with OT: 240 reg, +25 per OT.
        # Using >240 is a tolerant heuristic; a stricter check belongs in QA, not the schema.
        team_min_total = home.MIN or 0
        return cls(
            game_id=home.GAME_ID,
            season=season,
            date=_dt.date.fromisoformat(home.GAME_DATE),
            home_team_id=home.TEAM_ID,
            away_team_id=away.TEAM_ID,
            home_team_abbr=home.TEAM_ABBREVIATION,
            away_team_abbr=away.TEAM_ABBREVIATION,
            home_pts=home.PTS or 0,
            away_pts=away.PTS or 0,
            is_overtime=team_min_total > 240,
            is_playoffs=is_playoffs,
        )


class PlayerBoxLine(BaseModel):
    """Per-player per-game line. Use as a training target — NOT the output
    format. Built from a single :class:`RawPlayerBoxLine` via :meth:`from_raw`.
    """

    game_id: str
    player_id: int
    player_name: str
    team_id: int
    team_abbr: str
    position: str              # "" for non-starters, otherwise "F"/"G"/"C"/"F-C"/etc.
    minutes: float = Field(ge=0.0)
    pts: int = Field(ge=0)
    fgm: int = Field(ge=0)
    fga: int = Field(ge=0)
    tpm: int = Field(ge=0)
    tpa: int = Field(ge=0)
    ftm: int = Field(ge=0)
    fta: int = Field(ge=0)
    oreb: int = Field(ge=0)
    dreb: int = Field(ge=0)
    reb: int = Field(ge=0)
    ast: int = Field(ge=0)
    stl: int = Field(ge=0)
    blk: int = Field(ge=0)
    tov: int = Field(ge=0)
    pf: int = Field(ge=0)
    plus_minus: float
    is_starter: bool = False
    is_active: bool = True     # on the active roster for this game
    dnp: bool = False          # rostered but did not play (minutes==0)

    @model_validator(mode="after")
    def _shooting_invariants(self) -> "PlayerBoxLine":
        # Makes <= attempts (true by definition; flags upstream data corruption).
        if self.fgm > self.fga:
            raise ValueError(f"fgm ({self.fgm}) > fga ({self.fga})")
        if self.tpm > self.tpa:
            raise ValueError(f"tpm ({self.tpm}) > tpa ({self.tpa})")
        if self.ftm > self.fta:
            raise ValueError(f"ftm ({self.ftm}) > fta ({self.fta})")
        # 3pt are a strict subset of FGs.
        if self.tpm > self.fgm:
            raise ValueError(f"tpm ({self.tpm}) > fgm ({self.fgm})")
        if self.tpa > self.fga:
            raise ValueError(f"tpa ({self.tpa}) > fga ({self.fga})")
        # Rebound split must match total.
        if self.oreb + self.dreb != self.reb:
            raise ValueError(
                f"oreb+dreb ({self.oreb + self.dreb}) != reb ({self.reb})"
            )
        # DNP flag must agree with minutes.
        if (self.minutes == 0.0) != self.dnp:
            raise ValueError(f"dnp ({self.dnp}) inconsistent with minutes ({self.minutes})")
        return self

    @classmethod
    def from_raw(cls, raw: RawPlayerBoxLine) -> "PlayerBoxLine":
        m = parse_minutes_mmss(raw.minutes)
        # First+last → display name; ``nameI`` ("J. Tatum") is dropped — names
        # are for human-facing output, not joins (we use personId for joins).
        full_name = f"{raw.firstName} {raw.familyName}".strip()
        return cls(
            game_id=raw.gameId,
            player_id=raw.personId,
            player_name=full_name,
            team_id=raw.teamId,
            team_abbr=raw.teamTricode,
            position=raw.position,
            minutes=m,
            pts=raw.points or 0,
            fgm=raw.fieldGoalsMade or 0,
            fga=raw.fieldGoalsAttempted or 0,
            tpm=raw.threePointersMade or 0,
            tpa=raw.threePointersAttempted or 0,
            ftm=raw.freeThrowsMade or 0,
            fta=raw.freeThrowsAttempted or 0,
            oreb=raw.reboundsOffensive or 0,
            dreb=raw.reboundsDefensive or 0,
            reb=raw.reboundsTotal or 0,
            ast=raw.assists or 0,
            stl=raw.steals or 0,
            blk=raw.blocks or 0,
            tov=raw.turnovers or 0,
            pf=raw.foulsPersonal or 0,
            plus_minus=raw.plusMinusPoints or 0.0,
            # V3 convention: a non-empty `position` string means starter.
            is_starter=bool(raw.position),
            is_active=True,
            dnp=(m == 0.0),
        )


class TeamBoxLine(BaseModel):
    """Per-team per-game line. Carries the V3 traditional totals plus
    optional advanced metrics (``pace``, ``off_rtg``, ``def_rtg``) which
    are merged in from ``BoxScoreAdvancedV3`` during ETL.
    """

    game_id: str
    team_id: int
    team_abbr: str
    is_home: bool
    minutes: float = Field(ge=0.0)
    pts: int = Field(ge=0)
    fgm: int = Field(ge=0)
    fga: int = Field(ge=0)
    tpm: int = Field(ge=0)
    tpa: int = Field(ge=0)
    ftm: int = Field(ge=0)
    fta: int = Field(ge=0)
    oreb: int = Field(ge=0)
    dreb: int = Field(ge=0)
    reb: int = Field(ge=0)
    ast: int = Field(ge=0)
    stl: int = Field(ge=0)
    blk: int = Field(ge=0)
    tov: int = Field(ge=0)
    pf: int = Field(ge=0)
    plus_minus: float
    pace: float | None = None
    off_rtg: float | None = None
    def_rtg: float | None = None

    @model_validator(mode="after")
    def _shooting_invariants(self) -> "TeamBoxLine":
        if self.fgm > self.fga:
            raise ValueError(f"fgm ({self.fgm}) > fga ({self.fga})")
        if self.tpm > self.tpa:
            raise ValueError(f"tpm ({self.tpm}) > tpa ({self.tpa})")
        if self.ftm > self.fta:
            raise ValueError(f"ftm ({self.ftm}) > fta ({self.fta})")
        if self.tpm > self.fgm:
            raise ValueError(f"tpm ({self.tpm}) > fgm ({self.fgm})")
        if self.tpa > self.fga:
            raise ValueError(f"tpa ({self.tpa}) > fga ({self.fga})")
        if self.oreb + self.dreb != self.reb:
            raise ValueError(
                f"oreb+dreb ({self.oreb + self.dreb}) != reb ({self.reb})"
            )
        return self

    @classmethod
    def from_raw(cls, raw: RawTeamBoxLine, *, is_home: bool) -> "TeamBoxLine":
        return cls(
            game_id=raw.gameId,
            team_id=raw.teamId,
            team_abbr=raw.teamTricode,
            is_home=is_home,
            minutes=parse_minutes_mmss(raw.minutes),
            pts=raw.points,
            fgm=raw.fieldGoalsMade,
            fga=raw.fieldGoalsAttempted,
            tpm=raw.threePointersMade,
            tpa=raw.threePointersAttempted,
            ftm=raw.freeThrowsMade,
            fta=raw.freeThrowsAttempted,
            oreb=raw.reboundsOffensive,
            dreb=raw.reboundsDefensive,
            reb=raw.reboundsTotal,
            ast=raw.assists,
            stl=raw.steals,
            blk=raw.blocks,
            tov=raw.turnovers,
            pf=raw.foulsPersonal,
            plus_minus=raw.plusMinusPoints,
        )


class Roster(BaseModel):
    team_id: int
    season: int
    player_ids: list[int]


class PlayerInfo(BaseModel):
    player_id: int
    name: str
    birthdate: _dt.date | None
    height_in: float | None
    weight_lbs: float | None
    position: str | None     # {PG, SG, SF, PF, C} after cleaning
    draft_year: int | None
    rookie_season: int | None


# ---------------------------------------------------------------------------
# Processed layer — ready for the DataLoader.
# ---------------------------------------------------------------------------


class TrainingRow(BaseModel):
    """One player-game row with features joined in. Wide and flat."""

    model_config = {"extra": "allow"}  # actual field set fixed in implementation


class TeamTrainingRow(BaseModel):
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Public simulator output types — returned by simulate_game(...).
# ---------------------------------------------------------------------------


class SimPlayerBoxLine(BaseModel):
    """Player box line produced by the simulator.

    Distinct from :class:`PlayerBoxLine` (which is a training-target record)
    so that downstream consumers can tell "sampled" apart from "observed".
    """

    player_id: int
    player_name: str
    minutes: float = Field(ge=0.0)
    pts: int = Field(ge=0)
    fgm: int = Field(ge=0)
    fga: int = Field(ge=0)
    tpm: int = Field(ge=0)
    tpa: int = Field(ge=0)
    ftm: int = Field(ge=0)
    fta: int = Field(ge=0)
    oreb: int = Field(ge=0)
    dreb: int = Field(ge=0)
    reb: int = Field(ge=0)         # derived: oreb + dreb
    ast: int = Field(ge=0)
    stl: int = Field(ge=0)
    blk: int = Field(ge=0)
    tov: int = Field(ge=0)
    pf: int = Field(ge=0)
    plus_minus: float


class SimTeamBoxLine(BaseModel):
    team: str
    players: list[SimPlayerBoxLine]
    pts: int = Field(ge=0)
    pace: float
    off_rtg: float
    def_rtg: float


class BoxScore(BaseModel):
    """Single simulated box score."""

    home: SimTeamBoxLine
    away: SimTeamBoxLine
    date: _dt.date
    seed: int | None = None


class BoxScoreEnsemble(BaseModel):
    """N simulated samples with mean + 80% interval summaries per cell.

    ``samples`` is the raw list; ``mean`` and ``interval`` are derived and
    provided for convenience when callers don't want to aggregate themselves.
    """

    samples: list[BoxScore]
    mean: BoxScore
    interval_low: BoxScore     # 10th percentile per cell
    interval_high: BoxScore    # 90th percentile per cell
