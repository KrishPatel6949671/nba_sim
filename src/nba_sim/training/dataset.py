"""PyTorch ``Dataset`` for the processed parquet splits.

This is the producer side of the contract that
:class:`nba_sim.models.hierarchical.HierarchicalBoxScoreModel` and
:func:`nba_sim.models.losses.composite_nll` consume.

Each item is one **game**, not one player. The dict returned by
``__getitem__`` carries the union of:

- the inputs the model needs (player features, ids, roles, masks,
  context, matchup, teacher-forcing values for pace + off_rtg);
- the targets the loss needs (pace, off_rtg, minutes shares, plays
  gates, and per-player count/percent targets).

Producing them together lets the training loop be literally::

    preds = model(batch)
    loss = composite_nll(preds, batch, weights)["loss"]

without any splitting plumbing.

Conventions:

- Tensor shapes are per-game: ``[P]`` for per-player vectors, ``[P, D_p]``
  for per-player features, scalars for game-level targets. Adding the
  ``B`` dim happens in :func:`collate_games`.
- ``P = 15`` is the padding target. Games with more active players
  (~0.1% of games) keep their top-``P`` by minutes.
- ``player_id = 0`` and ``role_id = 0`` are reserved for padding /
  unseen — both embedding tables use ``padding_idx=0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Column lists. Order matters — each tuple defines the channel layout of the
# corresponding output tensor.
# ---------------------------------------------------------------------------

# 45 numeric player features (z-score standardized).
_PLAYER_NUMERIC_COLS: tuple[str, ...] = (
    # Rolling form (12).
    "p_min_avg_5", "p_min_avg_10", "p_min_avg_20",
    "p_usage_avg_5", "p_usage_avg_10", "p_usage_avg_20",
    "p_ts_5", "p_ts_10", "p_ts_20",
    "p_pts_per_min_5", "p_pts_per_min_10", "p_pts_per_min_20",
    # Per-minute counting rates (9).
    "p_fga_per_min_10", "p_tpa_per_min_10", "p_fta_per_min_10",
    "p_reb_per_min_10", "p_ast_per_min_10", "p_stl_per_min_10",
    "p_blk_per_min_10", "p_tov_per_min_10", "p_pf_per_min_10",
    # Season-to-date (17 incl. games count).
    "p_games_played_season",
    "std_games", "std_minutes_avg",
    "std_pts_avg", "std_fgm_avg", "std_fga_avg",
    "std_tpm_avg", "std_tpa_avg", "std_ftm_avg", "std_fta_avg",
    "std_oreb_avg", "std_dreb_avg", "std_ast_avg",
    "std_stl_avg", "std_blk_avg", "std_tov_avg", "std_pf_avg",
    # Schedule-flavor numerics (2).
    "rest_days", "travel_miles_prev",
    # Career shooting-skill priors + FT-rate (5). Address the chronic
    # ftm/fta/blk gap where rolling-N averages don't stabilize the signal.
    "p_fg_pct_career", "p_tp_pct_career", "p_ft_pct_career",
    "p_blk_per36_career", "p_ft_rate_10",
)

# 4 position one-hot buckets. ``""`` catches missing / unrecognized.
_POSITION_TOKENS: tuple[str, ...] = ("G", "F", "C", "")

# 4 passthrough booleans (cast to 0/1 float).
_PLAYER_BOOL_COLS: tuple[str, ...] = ("is_starter", "dnp", "is_home", "b2b")

# Sanity: 45 + 4 + 4 = 53 = d_player_raw in configs/model.yaml.
_D_PLAYER_RAW: int = (
    len(_PLAYER_NUMERIC_COLS) + len(_POSITION_TOKENS) + len(_PLAYER_BOOL_COLS)
)
assert _D_PLAYER_RAW == 53, _D_PLAYER_RAW

# Context numerics (5) standardized; rest of d_context_raw=24 is one-hots +
# cyclic encodings + bools assembled inline in ``_build_context``.
_CONTEXT_NUMERIC_COLS: tuple[str, ...] = (
    "altitude_ft",
    "rest_days",         # standardized once; used for both home and away
    "travel_miles_prev",
)
_SEASON_PHASES: tuple[str, ...] = ("early", "mid", "late", "playoffs")
_N_DAYS_OF_WEEK: int = 7

# Matchup numerics — diffs of team rolling stats + opp_* + h2h. Standardized.
_MATCHUP_TEAM_DIFF_BASES: tuple[str, ...] = (
    "t_pace_5", "t_pace_10",
    "t_off_rtg_5", "t_off_rtg_10",
    "t_def_rtg_5", "t_def_rtg_10",
    "t_win_pct_10", "t_pts_avg_10", "t_pts_allowed_10",
)
_MATCHUP_OPP_BASES: tuple[str, ...] = ("opp_def_rtg_10", "opp_pace_10")
# 9 diffs + 2 home-opp + 2 away-opp + h2h + 2 (rest, travel) diffs = 16.
_D_MATCHUP: int = (
    len(_MATCHUP_TEAM_DIFF_BASES) + 2 * len(_MATCHUP_OPP_BASES) + 1 + 2
)
assert _D_MATCHUP == 16, _D_MATCHUP

# Stat columns that the model produces per-player heads for.
_COUNT_STAT_COLS: tuple[str, ...] = (
    "fga", "tpa", "fta", "fgm", "tpm", "ftm",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
)

# Minimum minute mass given to padded / DNP slots before Dirichlet
# renormalization. Dirichlet.log_prob(0) blows up; ε > 0 is required.
_MIN_SLOT_MASS: float = 1e-3


# ---------------------------------------------------------------------------
# Fitted statistics
# ---------------------------------------------------------------------------


@dataclass
class FeatureStats:
    """(mean, std) per column. Built once from the train split and re-used
    verbatim on val/test to avoid leaking statistics across splits."""

    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)

    def standardize(self, values: np.ndarray, col: str) -> np.ndarray:
        """Z-score ``values`` using the fitted stats for ``col``.

        Missing-from-stats columns are passed through unchanged. Nulls /
        NaNs become 0 (= the column mean post-scaling).
        """
        if col not in self.means:
            return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        m = self.means[col]
        s = max(self.stds.get(col, 1.0), 1e-6)
        out = (values - m) / s
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _possessions(fga: float, fta: float, oreb: float, tov: float) -> float:
    """Standard pace approximation: ``FGA + 0.44·FTA − OREB + TOV``."""
    return float(fga) + 0.44 * float(fta) - float(oreb) + float(tov)


def _safe_first(series: pl.Series, default: float = 0.0) -> float:
    if series.is_empty():
        return default
    v = series[0]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


def _role_id(is_starter: bool | None, p_min_avg_10: float | None) -> int:
    """Bucket a player into one of 3 roles. 0 is reserved for padding."""
    if is_starter:
        return 1
    if p_min_avg_10 is None or (isinstance(p_min_avg_10, float) and math.isnan(p_min_avg_10)):
        return 3
    if p_min_avg_10 >= 12.0:
        return 2
    return 3


def _position_one_hot(pos: str | None) -> np.ndarray:
    """1-hot encode ``pos`` into ``_POSITION_TOKENS`` order; unknown → last."""
    out = np.zeros(len(_POSITION_TOKENS), dtype=np.float32)
    if pos in _POSITION_TOKENS:
        out[_POSITION_TOKENS.index(pos)] = 1.0
    else:
        out[-1] = 1.0
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BoxScoreDataset(Dataset):
    """One item per game; produces the dict the model + loss both consume.

    Parameters
    ----------
    processed_parquet
        Path to ``data/processed/{train,val,test}.parquet`` (the output of
        :func:`nba_sim.data.etl.interim_to_processed`).
    player_id_map
        Pre-fitted mapping ``raw_player_id -> embedding_id`` in
        ``[1, max_player_id-1]``. When ``None``, the dataset fits one from
        this parquet (the train-split workflow). Pass the train dataset's
        ``.player_id_map`` to val/test to avoid leakage.
    feature_stats
        Pre-fitted z-score statistics. Same train-vs-val/test discipline
        as ``player_id_map``.
    max_players
        Padding / truncation target per team. Default 15 (matches
        ``configs/model.yaml:roster.max_active_players``).
    max_player_id
        Embedding vocabulary size. Must match
        ``configs/model.yaml:embeddings.n_players``.
    """

    def __init__(
        self,
        processed_parquet: str | Path,
        player_id_map: dict[int, int] | None = None,
        feature_stats: FeatureStats | None = None,
        max_players: int = 15,
        max_player_id: int = 5000,
    ) -> None:
        self.max_players = int(max_players)
        self.max_player_id = int(max_player_id)

        df = pl.read_parquet(processed_parquet)
        self._check_required_columns(df)

        # Drop the rare inactive row defensively (the etl already filters
        # to active=True, but better safe — the model contract says
        # "active roster only").
        if "is_active" in df.columns:
            df = df.filter(pl.col("is_active"))

        # Fit OR adopt the player-id map.
        if player_id_map is None:
            self.player_id_map = self._fit_player_id_map(df)
        else:
            self.player_id_map = dict(player_id_map)

        # Fit OR adopt feature stats.
        if feature_stats is None:
            self.feature_stats = self._fit_feature_stats(df)
        else:
            self.feature_stats = feature_stats

        # Sort + partition so __getitem__ is O(1) frame lookup.
        # Within a game: home first, then by minutes desc (so truncation
        # to top-15 by minutes is a simple head-slice).
        df = df.sort(
            ["game_id", "is_home", "minutes"],
            descending=[False, True, True],
        )
        self._game_ids: list[str] = df["game_id"].unique(maintain_order=True).to_list()
        self._games: list[pl.DataFrame] = df.partition_by(
            "game_id", maintain_order=True
        )
        if len(self._games) != len(self._game_ids):
            raise RuntimeError(
                f"partition mismatch: {len(self._games)} frames vs "
                f"{len(self._game_ids)} unique game_ids"
            )

    # ---- introspection ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._games)

    @property
    def game_ids(self) -> list[str]:
        return list(self._game_ids)

    # ---- fitters ---------------------------------------------------------

    @staticmethod
    def _check_required_columns(df: pl.DataFrame) -> None:
        required = [
            "game_id", "player_id", "team_id", "is_home",
            "minutes", "position",
            *_COUNT_STAT_COLS, "pts",
            "season_phase", "day_of_week", "month",
            "altitude_ft", "travel_miles_prev",
            "rest_days", "b2b", "is_3in4", "is_4in6",
            "h2h_last_meeting_margin",
            *_MATCHUP_OPP_BASES, *_MATCHUP_TEAM_DIFF_BASES,
            "is_starter", "dnp",
            *_PLAYER_NUMERIC_COLS,
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"processed parquet is missing required columns: {missing}"
            )

    def _fit_player_id_map(self, df: pl.DataFrame) -> dict[int, int]:
        """Assign contiguous embedding ids in [1, max_player_id-1].

        Players are ranked by total appearances (descending) so the most-
        common players get the lowest, longest-lived ids. Ties broken by
        ``player_id`` ascending for determinism. Overflow → 0 (= pad /
        unseen), handled by ``nn.Embedding(padding_idx=0)``.
        """
        counts = (
            df.group_by("player_id")
            .agg(pl.len().alias("_n"))
            .sort(["_n", "player_id"], descending=[True, False])
        )
        cap = self.max_player_id - 1   # leave 0 for pad
        ids = counts["player_id"].to_list()[:cap]
        return {int(pid): i + 1 for i, pid in enumerate(ids)}

    def _fit_feature_stats(self, df: pl.DataFrame) -> FeatureStats:
        """Fit z-score stats on raw columns we standardize.

        Targets, ids, booleans, and one-hot bases are NOT included — they
        either go through the heads or are already on a unit-ish scale.
        """
        fit_cols = (
            *_PLAYER_NUMERIC_COLS,
            *_CONTEXT_NUMERIC_COLS,
            *_MATCHUP_TEAM_DIFF_BASES,
            *_MATCHUP_OPP_BASES,
            "h2h_last_meeting_margin",
        )
        stats = FeatureStats()
        for c in fit_cols:
            if c not in df.columns:
                continue
            col = df[c].cast(pl.Float64)
            m = col.mean()
            s = col.std()
            stats.means[c] = float(m) if m is not None else 0.0
            stats.stds[c] = float(s) if s is not None and s > 0 else 1.0
        return stats

    # ---- item construction ----------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = self._games[idx]
        home = g.filter(pl.col("is_home"))
        away = g.filter(~pl.col("is_home"))
        if home.is_empty() or away.is_empty():
            raise RuntimeError(
                f"game {self._game_ids[idx]} is missing one side "
                f"(home={home.height}, away={away.height})"
            )

        home_data = self._build_side(home)
        away_data = self._build_side(away)

        # ---- team-level targets, derived from observed totals -----------
        h_tot = self._team_totals(home)
        a_tot = self._team_totals(away)
        h_poss = _possessions(h_tot["fga"], h_tot["fta"], h_tot["oreb"], h_tot["tov"])
        a_poss = _possessions(a_tot["fga"], a_tot["fta"], a_tot["oreb"], a_tot["tov"])
        # Pace is the shared "possessions per 48 min" — average the two
        # teams so a single scalar represents the game tempo.
        pace = 0.5 * (h_poss + a_poss)
        h_off = h_tot["pts"] / max(h_poss, 1.0) * 100.0
        a_off = a_tot["pts"] / max(a_poss, 1.0) * 100.0

        # ---- context & matchup (one shared vector per game) -------------
        home_row = home.row(0, named=True)
        away_row = away.row(0, named=True)
        context = self._build_context(home_row, away_row)
        matchup = self._build_matchup(home_row, away_row)

        pace_t = torch.tensor(pace, dtype=torch.float32)
        off_t = torch.tensor([h_off, a_off], dtype=torch.float32)

        item: dict[str, torch.Tensor] = {
            # ---- model inputs --------------------------------------------
            "home_player_feats": home_data["feats"],
            "away_player_feats": away_data["feats"],
            "home_player_ids":   home_data["player_ids"],
            "away_player_ids":   away_data["player_ids"],
            "home_role_ids":     home_data["role_ids"],
            "away_role_ids":     away_data["role_ids"],
            "home_mask":         home_data["mask"],
            "away_mask":         away_data["mask"],
            "context":           context,
            "matchup":           matchup,
            # Teacher-forcing — same tensors as the targets.
            "pace_true":         pace_t,
            "off_rtg_true":      off_t,
            # ---- loss targets --------------------------------------------
            "pace":              pace_t,
            "off_rtg":           off_t,
            "home_minutes_share": home_data["minutes_share"],
            "away_minutes_share": away_data["minutes_share"],
            "home_plays_gate":    home_data["plays_gate"],
            "away_plays_gate":    away_data["plays_gate"],
        }
        for stat in _COUNT_STAT_COLS:
            item[f"home_{stat}"] = home_data[f"stat_{stat}"]
            item[f"away_{stat}"] = away_data[f"stat_{stat}"]
        return item

    # ---- per-team builders ----------------------------------------------

    def _build_side(self, side: pl.DataFrame) -> dict[str, torch.Tensor]:
        """Build padded per-player tensors for one team.

        The frame is already sorted minutes-descending → ``head(P)``
        keeps the most-impactful players when active > P.
        """
        P = self.max_players
        side = side.head(P)
        n_active = side.height

        feats = np.zeros((P, _D_PLAYER_RAW), dtype=np.float32)
        player_ids = np.zeros(P, dtype=np.int64)
        role_ids = np.zeros(P, dtype=np.int64)
        mask = np.zeros(P, dtype=bool)
        minutes_raw = np.zeros(P, dtype=np.float32)
        plays_gate = np.zeros(P, dtype=np.float32)
        stats: dict[str, np.ndarray] = {
            s: np.zeros(P, dtype=np.float32) for s in _COUNT_STAT_COLS
        }

        # Vectorized standardization over the active slice for each
        # numeric column, then write into the [0:n_active] block of the
        # padded tensor.
        numeric_block = np.zeros((n_active, len(_PLAYER_NUMERIC_COLS)), dtype=np.float32)
        for j, col in enumerate(_PLAYER_NUMERIC_COLS):
            raw = side[col].cast(pl.Float64).to_numpy()
            numeric_block[:, j] = self.feature_stats.standardize(raw, col)
        feats[:n_active, : len(_PLAYER_NUMERIC_COLS)] = numeric_block

        # Position one-hots + booleans + per-row metadata in a single pass.
        positions = side["position"].to_list()
        is_starter_l = side["is_starter"].to_list()
        dnp_l = side["dnp"].to_list()
        is_home_l = side["is_home"].to_list()
        b2b_l = side["b2b"].to_list()
        p_min_avg_10_l = side["p_min_avg_10"].cast(pl.Float64).to_list()
        player_id_l = side["player_id"].to_list()
        minutes_l = side["minutes"].cast(pl.Float64).to_list()
        stat_lists = {
            s: side[s].cast(pl.Float64).to_list() for s in _COUNT_STAT_COLS
        }

        pos_off = len(_PLAYER_NUMERIC_COLS)
        bool_off = pos_off + len(_POSITION_TOKENS)

        for i in range(n_active):
            feats[i, pos_off : pos_off + len(_POSITION_TOKENS)] = _position_one_hot(positions[i])
            feats[i, bool_off + 0] = 1.0 if is_starter_l[i] else 0.0
            feats[i, bool_off + 1] = 1.0 if dnp_l[i] else 0.0
            feats[i, bool_off + 2] = 1.0 if is_home_l[i] else 0.0
            feats[i, bool_off + 3] = 1.0 if b2b_l[i] else 0.0

            mask[i] = True
            player_ids[i] = self.player_id_map.get(int(player_id_l[i]), 0)
            role_ids[i] = _role_id(is_starter_l[i], p_min_avg_10_l[i])

            m = float(minutes_l[i]) if minutes_l[i] is not None else 0.0
            minutes_raw[i] = max(m, 0.0)
            plays_gate[i] = 1.0 if minutes_raw[i] > 0.0 else 0.0

            for s in _COUNT_STAT_COLS:
                v = stat_lists[s][i]
                stats[s][i] = float(v) if v is not None else 0.0

        # Dirichlet minutes share: padded / DNP slots get ε so the row
        # remains strictly positive and sums to 1 (Dirichlet.log_prob
        # requirement). See losses._make_batch_with_targets fixture.
        adjusted = np.where(minutes_raw > 0.0, minutes_raw, _MIN_SLOT_MASS)
        minutes_share = adjusted / adjusted.sum()

        return {
            "feats":         torch.from_numpy(feats),
            "player_ids":    torch.from_numpy(player_ids),
            "role_ids":      torch.from_numpy(role_ids),
            "mask":          torch.from_numpy(mask),
            "minutes_share": torch.from_numpy(minutes_share.astype(np.float32)),
            "plays_gate":    torch.from_numpy(plays_gate),
            **{f"stat_{s}": torch.from_numpy(stats[s]) for s in _COUNT_STAT_COLS},
        }

    @staticmethod
    def _team_totals(side: pl.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for c in ("pts", "fga", "fta", "oreb", "tov"):
            out[c] = float(side[c].cast(pl.Float64).sum() or 0.0)
        return out

    # ---- game-level builders --------------------------------------------

    def _build_context(self, home_row: dict, away_row: dict) -> torch.Tensor:
        """Assemble the 24-dim per-game context vector.

        Layout (in order):
        - season_phase one-hot (4)
        - day_of_week one-hot (7)
        - month sin/cos (2)
        - altitude_ft standardized (1)
        - rest_days standardized: home, away (2)
        - b2b: home, away (2)
        - is_3in4: home, away (2)
        - is_4in6: home, away (2)
        - travel_miles_prev standardized: home, away (2)
        """
        vec = np.zeros(24, dtype=np.float32)
        off = 0

        # Season phase (4).
        phase = home_row.get("season_phase") or ""
        if phase in _SEASON_PHASES:
            vec[off + _SEASON_PHASES.index(phase)] = 1.0
        off += len(_SEASON_PHASES)

        # Day of week (7). Polars stores it as 0..6.
        dow = home_row.get("day_of_week")
        if dow is not None and 0 <= int(dow) < _N_DAYS_OF_WEEK:
            vec[off + int(dow)] = 1.0
        off += _N_DAYS_OF_WEEK

        # Month cyclic encoding (2).
        month = home_row.get("month")
        if month is not None:
            theta = 2.0 * math.pi * (int(month) - 1) / 12.0
            vec[off + 0] = math.sin(theta)
            vec[off + 1] = math.cos(theta)
        off += 2

        # Altitude (1) — game venue is the home arena, so always read home.
        vec[off] = self.feature_stats.standardize(
            np.array([_safe_first(pl.Series([home_row.get("altitude_ft")]))], dtype=np.float32),
            "altitude_ft",
        )[0]
        off += 1

        # Rest days, b2b, is_3in4, is_4in6, travel — one slot each side.
        for key in ("rest_days",):
            for side_row in (home_row, away_row):
                v = side_row.get(key)
                v_arr = np.array([float(v) if v is not None else 0.0], dtype=np.float32)
                vec[off] = self.feature_stats.standardize(v_arr, key)[0]
                off += 1
        for key in ("b2b", "is_3in4", "is_4in6"):
            for side_row in (home_row, away_row):
                vec[off] = 1.0 if side_row.get(key) else 0.0
                off += 1
        for key in ("travel_miles_prev",):
            for side_row in (home_row, away_row):
                v = side_row.get(key)
                v_arr = np.array([float(v) if v is not None else 0.0], dtype=np.float32)
                vec[off] = self.feature_stats.standardize(v_arr, key)[0]
                off += 1

        assert off == 24, off
        return torch.from_numpy(vec)

    def _build_matchup(self, home_row: dict, away_row: dict) -> torch.Tensor:
        """Assemble the 16-dim home-vs-away matchup vector.

        Layout:
        - 9 standardized diffs (home − away) of team rolling stats
        - 2 standardized home-team ``opp_*`` (= away team's def_rtg / pace)
        - 2 standardized away-team ``opp_*``
        - 1 standardized h2h_last_meeting_margin (from home row)
        - 2 diffs of (rest_days, travel_miles_prev), already standardized
        """
        vec = np.zeros(16, dtype=np.float32)
        off = 0

        def _z(col: str, raw: float) -> float:
            return float(
                self.feature_stats.standardize(np.array([raw], dtype=np.float32), col)[0]
            )

        # 9 team-stat diffs (home minus away).
        for base in _MATCHUP_TEAM_DIFF_BASES:
            h = home_row.get(base)
            a = away_row.get(base)
            h = float(h) if h is not None else self.feature_stats.means.get(base, 0.0)
            a = float(a) if a is not None else self.feature_stats.means.get(base, 0.0)
            vec[off] = _z(base, h) - _z(base, a)
            off += 1

        # 4 opp_* (raw home then raw away).
        for base in _MATCHUP_OPP_BASES:
            v = home_row.get(base)
            vec[off] = _z(base, float(v) if v is not None else self.feature_stats.means.get(base, 0.0))
            off += 1
        for base in _MATCHUP_OPP_BASES:
            v = away_row.get(base)
            vec[off] = _z(base, float(v) if v is not None else self.feature_stats.means.get(base, 0.0))
            off += 1

        # h2h margin — signed, "home perspective" by convention in the etl.
        v = home_row.get("h2h_last_meeting_margin")
        v = float(v) if v is not None else 0.0
        vec[off] = _z("h2h_last_meeting_margin", v)
        off += 1

        # 2 diffs: rest_days, travel_miles_prev.
        for key in ("rest_days", "travel_miles_prev"):
            h = home_row.get(key)
            a = away_row.get(key)
            h = float(h) if h is not None else 0.0
            a = float(a) if a is not None else 0.0
            vec[off] = _z(key, h) - _z(key, a)
            off += 1

        assert off == 16, off
        return torch.from_numpy(vec)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_games(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-game dicts into a ``[B, ...]`` batch dict.

    All values in ``items[i]`` are tensors of identical shape across ``i``;
    we just stack along the new leading dim. Used as the DataLoader's
    ``collate_fn``.
    """
    if not items:
        return {}
    out: dict[str, torch.Tensor] = {}
    for k in items[0]:
        out[k] = torch.stack([it[k] for it in items], dim=0)
    return out
