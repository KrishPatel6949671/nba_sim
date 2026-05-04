"""Typed schemas for raw / interim / processed records and output box scores.

Three layers (see PLAN.md §2.3):

1. **Raw** — mirrors API fields 1:1, all optional, string-tolerant.
2. **Interim** — typed, validated, MIN parsed to float, IDs int, DNP flags set.
3. **Processed** — denormalized, feature-engineered, ready for the DataLoader.

Plus the public box-score output types returned by :func:`simulate_game`.
"""

from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Raw layer — loose, API-shaped.
# ---------------------------------------------------------------------------


class RawGame(BaseModel):
    """Raw row from ``LeagueGameFinder``. Mostly strings — validated downstream."""

    model_config = {"extra": "allow"}


class RawPlayerBoxLine(BaseModel):
    """Raw row from ``BoxScoreTraditionalV2`` player-level output."""

    model_config = {"extra": "allow"}


class RawTeamBoxLine(BaseModel):
    """Raw row from ``BoxScoreTraditionalV2`` / ``BoxScoreAdvancedV2`` team-level output."""

    model_config = {"extra": "allow"}


class RawRoster(BaseModel):
    model_config = {"extra": "allow"}


class RawPlayerInfo(BaseModel):
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Interim layer — strictly typed. Validators enforce make <= attempt etc.
# ---------------------------------------------------------------------------


class Game(BaseModel):
    game_id: int
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


class PlayerBoxLine(BaseModel):
    """Per-player per-game line. Use as input target, NOT output format."""

    game_id: int
    player_id: int
    player_name: str
    team_id: int
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
    ast: int = Field(ge=0)
    stl: int = Field(ge=0)
    blk: int = Field(ge=0)
    tov: int = Field(ge=0)
    pf: int = Field(ge=0)
    plus_minus: float
    is_starter: bool = False
    is_active: bool = True     # on the active roster for this game
    dnp: bool = False          # rostered but did not play (MIN=0)

    # TODO validators in implementation:
    #   - fgm <= fga; tpm <= tpa; ftm <= fta; tpm <= fgm
    #   - if minutes == 0 -> dnp should be True


class TeamBoxLine(BaseModel):
    game_id: int
    team_id: int
    team_abbr: str
    is_home: bool
    pts: int
    pace: float
    off_rtg: float
    def_rtg: float


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
