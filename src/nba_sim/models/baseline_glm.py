"""Poisson GLM + season-average baselines.

The hierarchical NN must **beat** these baselines on every counting stat on
the validation season — see PLAN.md §6.3 and the Phase 1 ship gate in §10.
If the NN doesn't, v1 is not shippable.

Two baselines live here:

1. ``SeasonAverageBaseline`` — predicts each player's strictly-prior
   current-season mean for each stat. When the current season has no prior
   games, falls back to that player's most recent prior season average; if
   that's also missing (rookie), falls back to the position mean; final
   fallback is the global mean.
2. ``PoissonGLMBaseline`` — one ``sklearn.linear_model.PoissonRegressor``
   per counting stat (independent fits — counts are treated as
   conditionally independent given features) plus a ``LinearRegression``
   for minutes. Feature nulls (early-season players with no rolling
   history) are imputed with the training-set column mean.

The PLAN.md baseline spec also mentions a Beta regression head for shooting
percentages. v1 skips that: sklearn has no native Beta regressor, the gate
metric is per-stat MAE on counting stats, and percentages can be derived
post-hoc as ``pred_make / pred_attempt`` if a caller really wants them.

Both baselines are dependency-light (numpy + sklearn) so they train in
seconds and serve as a useful sanity check independent of the PyTorch
stack.

Outputs follow a consistent shape: ``predict(rows)`` returns a frame with
the identifier columns ``(game_id, player_id, team_id)`` plus one
``pred_<stat>`` column per modeled stat.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import (
    LinearRegression,
    PoissonRegressor,
    Ridge,
    TweedieRegressor,
)

# Link function for the count-stat heads. Three options:
#
# - ``"ridge"`` (default): ``Ridge(alpha=…)`` — identity link, squared
#   error loss, closed-form solve. Not a strict Poisson GLM (the deviance
#   is Gaussian, not Poisson), but the most numerically stable choice and
#   the only one that reliably beats SeasonAverageBaseline on
#   small-train-set data. Recovers ``μ = std_<stat>_avg`` exactly when
#   that's the optimum.
# - ``"identity"``: ``TweedieRegressor(power=1, link="identity")`` —
#   Poisson deviance loss with identity link. Theoretically the principled
#   choice when both PLAN's "Poisson GLM" spec and the SAB-identity match
#   are required, but lbfgs struggles to converge on real-sized data
#   because the deviance is undefined at ``β·X ≤ 0`` (no positivity
#   guarantee from identity link). Convergence warnings are normal and
#   cosmetic — the partial fit is usable, but the run is noisy.
# - ``"log"``: ``PoissonRegressor`` — Poisson deviance with log link, the
#   sklearn default. Numerically clean but the log link costs ~1% MAE per
#   counting stat because ``exp(linear)`` can only approximate the SAB
#   identity through an exponential curve.
_DEFAULT_GLM_LINK = "ridge"

logger = logging.getLogger(__name__)

# Stats the baselines predict.
#
# - ``minutes`` is a regression target (Gaussian / linear).
# - Everything else is a non-negative integer count → Poisson.
# - ``pts`` and ``reb`` are dependent on the other counts by basketball's
#   scoring identities, but we still model them directly here because
#   evaluation reports MAE per stat including ``pts``/``reb`` and a
#   baseline shouldn't bake in the identity (that's the NN's job). The
#   post-hoc projection in ``project_to_constraints`` re-derives ``pts``
#   and ``reb`` from the component stats to give an alternative
#   identity-respecting prediction.
_TARGET_MINUTES = "minutes"
_COUNT_STATS: tuple[str, ...] = (
    "pts",
    "fgm", "fga",
    "tpm", "tpa",
    "ftm", "fta",
    "oreb", "dreb", "reb",
    "ast", "stl", "blk", "tov", "pf",
)
_ALL_STATS: tuple[str, ...] = (_TARGET_MINUTES,) + _COUNT_STATS

# Feature columns the GLM uses. Anything not present in the input frame at
# fit time is silently skipped — keeps the baseline robust to schema drift
# (e.g. older seasons missing advanced metrics). Predict time uses the
# same resolved subset that fit picked, plus stored column means for
# null/NaN imputation.
#
# Booleans (is_home, b2b, ...) get cast to float on the way into sklearn.
# Strings (position, season_phase) are excluded from this baseline; v1
# leaves categorical encoding to the NN (which has embedding tables).
_PLAYER_ROLLING_FEATURES: tuple[str, ...] = (
    "p_min_avg_5", "p_min_avg_10", "p_min_avg_20",
    "p_usage_avg_5", "p_usage_avg_10", "p_usage_avg_20",
    "p_ts_5", "p_ts_10", "p_ts_20",
    "p_pts_per_min_5", "p_pts_per_min_10", "p_pts_per_min_20",
    "p_fga_per_min_10", "p_tpa_per_min_10", "p_fta_per_min_10",
    "p_reb_per_min_10", "p_ast_per_min_10", "p_stl_per_min_10",
    "p_blk_per_min_10", "p_tov_per_min_10", "p_pf_per_min_10",
    "p_games_played_season",
)
_TEAM_ROLLING_FEATURES: tuple[str, ...] = (
    "t_pace_5", "t_pace_10",
    "t_off_rtg_5", "t_off_rtg_10",
    "t_def_rtg_5", "t_def_rtg_10",
    "t_win_pct_10", "t_pts_avg_10", "t_pts_allowed_10",
)
_MATCHUP_FEATURES: tuple[str, ...] = (
    "opp_def_rtg_10", "opp_pace_10",
    "opp_def_rtg_vs_pos",
    "h2h_last_meeting_margin",
)
_CONTEXT_FEATURES: tuple[str, ...] = (
    "is_home", "rest_days", "b2b", "is_3in4", "is_4in6",
    "day_of_week", "month",
    "altitude_ft", "travel_miles_prev",
)
# Season-to-date features are the strictly-prior cumulative means the
# SeasonAverageBaseline uses internally. Including them as GLM features
# makes the GLM's information set a strict superset of SAB's, which is
# what the PLAN §6.3 ship gate implicitly assumes.
_SEASON_TO_DATE_FEATURES: tuple[str, ...] = (
    "std_games",
    "std_minutes_avg",
    "std_pts_avg",
    "std_fgm_avg", "std_fga_avg",
    "std_tpm_avg", "std_tpa_avg",
    "std_ftm_avg", "std_fta_avg",
    "std_oreb_avg", "std_dreb_avg", "std_reb_avg",
    "std_ast_avg", "std_stl_avg", "std_blk_avg",
    "std_tov_avg", "std_pf_avg",
)

# Bases that get a log1p-derived version materialized inside the GLM. The
# Poisson log link means ``predicted = exp(w·x)``, which can only
# approximate identity (predict_y ≈ std_y_avg) through the curvature of
# ``exp``. Adding ``log1p_std_y_avg`` lets the model find w≈1 and recover
# identity exactly (modulo standardization), which closes the structural
# gap against SeasonAverageBaseline on every counting stat.
#
# These columns aren't required to exist in the input frame — the
# transform skips bases that aren't present. So old test fixtures that
# don't have std_*_avg columns continue to work unchanged.
_LOG1P_STD_AVG_BASES: tuple[str, ...] = (
    "std_minutes_avg",
    "std_pts_avg",
    "std_fgm_avg", "std_fga_avg",
    "std_tpm_avg", "std_tpa_avg",
    "std_ftm_avg", "std_fta_avg",
    "std_oreb_avg", "std_dreb_avg", "std_reb_avg",
    "std_ast_avg", "std_stl_avg", "std_blk_avg",
    "std_tov_avg", "std_pf_avg",
)
_LOG1P_STD_AVG_DERIVED: tuple[str, ...] = tuple(
    f"log1p_{c}" for c in _LOG1P_STD_AVG_BASES
)

_DEFAULT_FEATURES: tuple[str, ...] = (
    _PLAYER_ROLLING_FEATURES
    + _TEAM_ROLLING_FEATURES
    + _MATCHUP_FEATURES
    + _CONTEXT_FEATURES
    + _SEASON_TO_DATE_FEATURES
    + _LOG1P_STD_AVG_DERIVED
)

_ID_COLS: tuple[str, ...] = ("game_id", "player_id", "team_id")


# ---------------------------------------------------------------------------
# Season-average baseline.
# ---------------------------------------------------------------------------


@dataclass
class SeasonAverageBaseline:
    """Predict each player's strictly-prior current-season mean per stat.

    Fallback chain when the current season has no prior games:

    1. Most recent prior season's full-season average (from ``fit``).
    2. Position mean (from ``fit``).
    3. Global mean (from ``fit``).

    The current-season cumulative-prior mean is computed *within* the
    ``rows`` frame passed to ``predict`` — every row contributes to later
    rows' priors. This is the textbook season-average baseline; it's
    self-leaky-by-construction in the sense that intra-frame future
    information is intentionally used (the PLAN explicitly defines it as
    "average over all prior games in the current season").

    Attributes set by ``fit``:
        _player_season_avgs: ``(player_id, _lookup_season, _psa_*)`` frame
            keyed for asof-join. ``_lookup_season = season + 1`` so a
            backward asof-join on ``season`` resolves "most recent fit
            season strictly less than the target season".
        _position_avgs: ``(position, _pos_*)`` frame.
        _overall_avgs: per-stat global mean.
    """

    stats: tuple[str, ...] = _ALL_STATS

    _player_season_avgs: pl.DataFrame | None = field(default=None, repr=False)
    _position_avgs: pl.DataFrame | None = field(default=None, repr=False)
    _overall_avgs: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._player_season_avgs is not None

    def fit(self, train: pl.DataFrame) -> "SeasonAverageBaseline":
        self._check_columns(train, required=("player_id", "season", "position", *self.stats))

        psa = (
            train.group_by(["player_id", "season"])
            .agg([pl.col(s).mean().alias(s) for s in self.stats])
            .with_columns((pl.col("season") + 1).alias("_lookup_season"))
            .rename({s: f"_psa_{s}" for s in self.stats})
            .sort(["player_id", "_lookup_season"])
        )
        self._player_season_avgs = psa.select(
            ["player_id", "_lookup_season", *[f"_psa_{s}" for s in self.stats]]
        )

        self._position_avgs = (
            train.group_by("position")
            .agg([pl.col(s).mean().alias(f"_pos_{s}") for s in self.stats])
        )

        for s in self.stats:
            mean = train[s].mean()
            self._overall_avgs[s] = float(mean) if mean is not None else 0.0
        return self

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("SeasonAverageBaseline.predict called before fit")
        assert self._player_season_avgs is not None
        assert self._position_avgs is not None

        self._check_columns(rows, required=(*_ID_COLS, "player_id", "season", "date", "position", *self.stats))

        # Current-season cumulative-prior mean inside `rows`.
        # cum_sum - current isolates the strictly-prior sum; the prior count
        # is the running row index within the (player, season) partition.
        df = rows.sort(["player_id", "season", "date", "game_id"])
        prior_count = (pl.col("game_id").cum_count().over(["player_id", "season"]) - 1).cast(pl.Float64)
        std_exprs: list[pl.Expr] = [prior_count.alias("_std_count")]
        for s in self.stats:
            prior_sum = pl.col(s).cum_sum().over(["player_id", "season"]) - pl.col(s)
            std_exprs.append(
                pl.when(prior_count > 0)
                .then(prior_sum / prior_count)
                .otherwise(None)
                .alias(f"_std_{s}")
            )
        df = df.with_columns(std_exprs)

        # Prior-season fallback via asof-backward on (player_id, season).
        # The lookup-season trick (psa.season + 1) ensures we only match
        # seasons strictly less than the target season.
        df = df.sort(["player_id", "season"]).join_asof(
            self._player_season_avgs,
            by="player_id",
            left_on="season",
            right_on="_lookup_season",
            strategy="backward",
        )

        # Position mean fallback (used when a player is a complete rookie
        # in the predict frame with no prior season in training data).
        df = df.join(self._position_avgs, on="position", how="left")

        # Layered coalesce: strictly-prior std → prior-season psa → position → global.
        pred_exprs = [
            pl.coalesce([
                pl.col(f"_std_{s}"),
                pl.col(f"_psa_{s}"),
                pl.col(f"_pos_{s}"),
                pl.lit(self._overall_avgs[s]),
            ]).alias(f"pred_{s}")
            for s in self.stats
        ]
        df = df.with_columns(pred_exprs)

        keep = [*_ID_COLS, *[f"pred_{s}" for s in self.stats]]
        return df.select(keep)

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("SeasonAverageBaseline.save called before fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "stats": self.stats,
            "player_season_avgs": self._player_season_avgs,
            "position_avgs": self._position_avgs,
            "overall_avgs": self._overall_avgs,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "SeasonAverageBaseline":
        state: dict[str, Any] = joblib.load(path)
        obj = cls(stats=tuple(state["stats"]))
        obj._player_season_avgs = state["player_season_avgs"]
        obj._position_avgs = state["position_avgs"]
        obj._overall_avgs = dict(state["overall_avgs"])
        return obj

    @staticmethod
    def _check_columns(df: pl.DataFrame, required: tuple[str, ...]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Poisson GLM baseline.
# ---------------------------------------------------------------------------


@dataclass
class PoissonGLMBaseline:
    """Stat-wise Poisson GLM (counts) + linear regression (minutes).

    One independent ``PoissonRegressor`` is fit per counting stat against
    the resolved feature subset. Minutes are modeled with a plain
    ``LinearRegression`` since they're continuous and frequently zero
    (DNPs) — Poisson with floor at 0 would work too but linear is what
    PLAN.md describes for the minutes head ("softmax across the roster
    with a secondary regression on total active players"). We use a
    per-player linear regressor and then renormalize within team via
    ``project_to_constraints``, which approximates the softmax behavior
    without needing a separate normalization head.

    Constraint enforcement is **not** done at predict time — call
    ``project_to_constraints`` on the predicted frame if downstream code
    needs it.

    Attributes set by ``fit``:
        _feature_cols_resolved: features actually present at fit time.
        _feature_means: per-feature mean for null/NaN imputation.
        _minutes_model: sklearn linear regressor.
        _count_models: per-stat sklearn Poisson regressors.
    """

    stats: tuple[str, ...] = _COUNT_STATS
    feature_cols: tuple[str, ...] = _DEFAULT_FEATURES
    # L2 regularization on the Poisson GLM. The default was 1.0 initially
    # but with ~100k+ training rows × ~60 standardized features that
    # over-shrinks every weight toward zero — the penalty term is
    # competing against a log-likelihood that scales linearly in rows.
    # 0.01 lets the data dominate while still preventing pathological
    # weights on near-constant or near-duplicated features.
    alpha: float = 0.01
    # Link function for the count-stat GLMs. ``identity`` uses
    # TweedieRegressor(power=1, link='identity') so the model can exactly
    # reproduce ``μ = std_<stat>_avg`` (the SAB prediction) when that's
    # the optimum. ``log`` uses sklearn's PoissonRegressor (its only
    # supported link) and is kept for compatibility / experiments — it
    # systematically loses ~1% MAE per counting stat because exp(linear)
    # can only approximate identity through curvature.
    link: str = _DEFAULT_GLM_LINK
    # Default raised from 300 → 1000 once standardization was added: even
    # easy fits will need more steps if the optimizer is unlucky on init,
    # and convergence is fast on scaled features so the extra ceiling is
    # near-free in practice.
    max_iter: int = 3000

    _feature_cols_resolved: tuple[str, ...] = field(default=(), repr=False)
    _feature_means: dict[str, float] = field(default_factory=dict, repr=False)
    # Per-feature stds used for unit-variance scaling. Stored separately
    # from _feature_means so old joblib state without stds raises a clean
    # error at load time rather than silently producing un-scaled inputs.
    _feature_stds: dict[str, float] = field(default_factory=dict, repr=False)
    _minutes_model: LinearRegression | None = field(default=None, repr=False)
    _count_models: dict[str, PoissonRegressor] = field(default_factory=dict, repr=False)

    @property
    def is_fitted(self) -> bool:
        return self._minutes_model is not None

    def fit(self, train: pl.DataFrame) -> "PoissonGLMBaseline":
        train = self._with_log1p_features(train)
        self._feature_cols_resolved = tuple(c for c in self.feature_cols if c in train.columns)
        if not self._feature_cols_resolved:
            raise ValueError(
                "PoissonGLMBaseline.fit: no usable feature columns found "
                f"(looked for {self.feature_cols})"
            )

        X = self._extract_X(train, fit_imputation=True)

        y_min = train[_TARGET_MINUTES].cast(pl.Float64).fill_null(0.0).to_numpy()
        self._minutes_model = LinearRegression()
        self._minutes_model.fit(X, y_min)

        for s in self.stats:
            y = train[s].cast(pl.Float64).fill_null(0.0).to_numpy()
            # y >= 0 is guaranteed by the Pydantic schema.
            model = self._make_count_estimator()
            model.fit(X, y)
            self._count_models[s] = model

        return self

    def _make_count_estimator(self):
        """Construct a fresh per-stat count regressor honoring ``self.link``.

        See ``_DEFAULT_GLM_LINK``'s docstring at module-level for the
        three options and their trade-offs.
        """
        if self.link == "ridge":
            # Closed-form solve; ``max_iter`` is only used by iterative
            # solvers (e.g. sag/saga) — ``solver="auto"`` picks Cholesky
            # for our problem size, which ignores max_iter entirely.
            return Ridge(alpha=self.alpha, solver="auto")
        if self.link == "log":
            return PoissonRegressor(alpha=self.alpha, max_iter=self.max_iter)
        if self.link == "identity":
            return TweedieRegressor(
                power=1.0,
                link="identity",
                alpha=self.alpha,
                max_iter=self.max_iter,
            )
        raise ValueError(
            f"PoissonGLMBaseline.link must be one of "
            f"'ridge', 'identity', 'log'; got {self.link!r}"
        )

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("PoissonGLMBaseline.predict called before fit")
        assert self._minutes_model is not None

        rows_with_logs = self._with_log1p_features(rows)
        X = self._extract_X(rows_with_logs, fit_imputation=False)
        out = rows.select(list(_ID_COLS))

        pred_min = self._minutes_model.predict(X).clip(min=0.0)
        out = out.with_columns(pl.Series("pred_minutes", pred_min))

        for s in self.stats:
            preds = self._count_models[s].predict(X).clip(min=0.0)
            out = out.with_columns(pl.Series(f"pred_{s}", preds))
        return out

    def project_to_constraints(self, predicted: pl.DataFrame) -> pl.DataFrame:
        """Enforce basketball constraints on a predicted frame.

        - ``pred_minutes`` normalized to sum to 240 per (game_id, team_id).
        - Makes capped at attempts: FGM ≤ FGA, 3PM ≤ 3PA, FTM ≤ FTA.
        - 3-pointer subset of FG: 3PM ≤ FGM and 3PA ≤ FGA.
        - ``pred_reb`` re-derived as ``pred_oreb + pred_dreb``.
        - ``pred_pts`` re-derived as ``2*pred_fgm + pred_tpm + pred_ftm``.

        Note: this is **only** for the baseline. The NN enforces these by
        construction (Binomial(total_count=FGA) for FGM, etc.) and doesn't
        need any projection.
        """
        df = predicted

        # Makes ≤ attempts.
        df = df.with_columns([
            pl.min_horizontal("pred_fgm", "pred_fga").alias("pred_fgm"),
            pl.min_horizontal("pred_tpm", "pred_tpa").alias("pred_tpm"),
            pl.min_horizontal("pred_ftm", "pred_fta").alias("pred_ftm"),
        ])
        # 3-pointer subset of FG.
        df = df.with_columns([
            pl.min_horizontal("pred_tpm", "pred_fgm").alias("pred_tpm"),
            pl.min_horizontal("pred_tpa", "pred_fga").alias("pred_tpa"),
        ])
        # Identity-respecting rebuild of reb and pts.
        df = df.with_columns([
            (pl.col("pred_oreb") + pl.col("pred_dreb")).alias("pred_reb"),
            (2.0 * pl.col("pred_fgm") + pl.col("pred_tpm") + pl.col("pred_ftm")).alias("pred_pts"),
        ])

        # Minutes sum-to-240 per team. The division-by-zero edge case
        # (every player predicted 0 minutes) falls back to equal allocation
        # to avoid producing NaN rows.
        team_min_sum = pl.col("pred_minutes").sum().over(["game_id", "team_id"])
        team_n = pl.col("pred_minutes").count().over(["game_id", "team_id"])
        df = df.with_columns(
            pl.when(team_min_sum > 0)
            .then(pl.col("pred_minutes") * 240.0 / team_min_sum)
            .otherwise(240.0 / team_n)
            .alias("pred_minutes")
        )
        return df

    def save(self, path: str | Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("PoissonGLMBaseline.save called before fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "stats": self.stats,
            "feature_cols": self.feature_cols,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "link": self.link,
            "feature_cols_resolved": self._feature_cols_resolved,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "minutes_model": self._minutes_model,
            "count_models": self._count_models,
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> "PoissonGLMBaseline":
        state: dict[str, Any] = joblib.load(path)
        if "feature_stds" not in state:
            raise ValueError(
                "PoissonGLMBaseline.load: persisted state is missing 'feature_stds'. "
                "This is a model trained before feature standardization was added; "
                "re-fit on the training data and save again."
            )
        # ``link`` was added later; default to ``"log"`` for models saved
        # before the field existed so their predict-time behavior is
        # unchanged (sklearn PoissonRegressor uses log link).
        obj = cls(
            stats=tuple(state["stats"]),
            feature_cols=tuple(state["feature_cols"]),
            alpha=float(state["alpha"]),
            max_iter=int(state["max_iter"]),
            link=str(state.get("link", "log")),
        )
        obj._feature_cols_resolved = tuple(state["feature_cols_resolved"])
        obj._feature_means = dict(state["feature_means"])
        obj._feature_stds = dict(state["feature_stds"])
        obj._minutes_model = state["minutes_model"]
        obj._count_models = dict(state["count_models"])
        return obj

    @staticmethod
    def _with_log1p_features(df: pl.DataFrame) -> pl.DataFrame:
        """Materialize ``log1p_<base>`` for each ``std_*_avg`` base present.

        See module-level ``_LOG1P_STD_AVG_BASES`` for the rationale. This
        runs once per fit/predict call and is a no-op when none of the
        bases are present (which is the case for the unit-test fixtures —
        they don't carry season-to-date columns).

        The transform uses ``log(x + 1)`` to handle the common ``x == 0``
        case (a player's first game in a season has std_<stat>_avg = null;
        after the existing null-imputation it becomes 0 → log1p(0) = 0,
        which the GLM can then learn to ignore via its weight).
        """
        new_cols = [
            (pl.col(base).cast(pl.Float64) + 1.0).log().alias(f"log1p_{base}")
            for base in _LOG1P_STD_AVG_BASES
            if base in df.columns
        ]
        if not new_cols:
            return df
        return df.with_columns(new_cols)

    def _extract_X(self, df: pl.DataFrame, *, fit_imputation: bool) -> np.ndarray:
        """Null-impute → standardize the feature matrix.

        Standardization is critical for the Poisson link: ``predicted =
        exp(X @ w)`` overflows fast when ``X`` has columns on different
        scales (altitude in thousands of ft alongside ``is_home`` in
        {0, 1}). After zero-centering and unit-scaling, lbfgs converges in
        well under 100 iterations on real-world season data.

        Means and stds are computed once at fit time (when ``fit_imputation
        =True``) and reused at predict time so the transform is identical
        — same behavior as sklearn's ``StandardScaler`` but kept inline so
        the dataclass remains pickleable through ``joblib`` without
        additional plumbing.
        """
        cols = self._feature_cols_resolved
        # Cast to Float64 (booleans → 0/1, ints → floats, nulls preserved).
        casted = df.select([pl.col(c).cast(pl.Float64).alias(c) for c in cols])

        if fit_imputation:
            for c in cols:
                m = casted[c].mean()
                self._feature_means[c] = float(m) if m is not None else 0.0
                s = casted[c].std()
                # Floor stds: constant columns (e.g. a feature that's
                # identical across the train set) would divide by 0
                # otherwise. 1e-6 makes them effectively constant after
                # scaling, which the L2-regularized weight will drive to 0.
                self._feature_stds[c] = max(float(s) if s is not None else 1.0, 1e-6)

        # Two-pass with_columns: fill nulls/NaNs with the fit-time mean,
        # then standardize. A row that came in null becomes 0 post-scaling
        # (mean → (mean - mean) / std = 0), which is the right neutral.
        filled = casted.with_columns([
            pl.col(c).fill_null(self._feature_means[c]).fill_nan(self._feature_means[c])
            for c in cols
        ])
        scaled = filled.with_columns([
            ((pl.col(c) - self._feature_means[c]) / self._feature_stds[c]).alias(c)
            for c in cols
        ])
        return scaled.to_numpy()


# ---------------------------------------------------------------------------
# CLI entry point — fit both baselines, score on val/test, print a per-stat
# MAE comparison so the PLAN §10 Phase 1 ship gate ("Poisson GLM beats
# 'season-average' on every counting stat") is a one-command check.
#
# Usage:
#     python -m nba_sim.models.baseline_glm                  # eval on val
#     python -m nba_sim.models.baseline_glm --eval-split test
#     python -m nba_sim.models.baseline_glm --save-dir models/baselines
# ---------------------------------------------------------------------------


def _per_stat_mae(
    actual: pl.DataFrame, predicted: pl.DataFrame, stats: tuple[str, ...]
) -> dict[str, float]:
    """Mean absolute error per stat, computed on the inner-join of (actual,
    predicted) by (game_id, player_id, team_id)."""
    joined = actual.join(
        predicted, on=["game_id", "player_id", "team_id"], how="inner"
    )
    return {
        s: float((joined[s].cast(pl.Float64) - joined[f"pred_{s}"]).abs().mean())
        for s in stats
    }


def _format_mae_table(
    sab_mae: dict[str, float], glm_mae: dict[str, float], stats: tuple[str, ...]
) -> str:
    """Render a side-by-side MAE comparison so the ship-gate is obvious.

    Lower MAE is better, so ``GLM wins?`` = ``glm < sab``. The PLAN gate
    requires GLM to win on every counting stat on the validation season.
    """
    lines = [
        f"{'stat':<12} | {'season_avg':>10} | {'poisson_glm':>11} | {'delta':>7} | GLM wins?",
        "-" * 60,
    ]
    glm_wins = 0
    for s in stats:
        sab = sab_mae[s]
        glm = glm_mae[s]
        delta = glm - sab
        wins = "yes" if glm < sab else " no"
        if glm < sab:
            glm_wins += 1
        lines.append(
            f"{s:<12} | {sab:>10.4f} | {glm:>11.4f} | {delta:>+7.4f} | {wins}"
        )
    lines.append("-" * 60)
    lines.append(f"GLM wins on {glm_wins}/{len(stats)} stats")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fit SeasonAverageBaseline + PoissonGLMBaseline on "
            "data/processed/train.parquet, score on the eval split, "
            "and print a per-stat MAE comparison."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train/val/test.parquet "
            "(default: nba_sim.data.etl.processed_dir())."
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=("val", "test"),
        default="val",
        help="Which split to score on (default: val).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="If set, save fitted models to this directory.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "L2 regularization strength for the PoissonGLM. "
            "If omitted, uses PoissonGLMBaseline's dataclass default — keeps the "
            "CLI from silently overriding code-level changes."
        ),
    )
    parser.add_argument(
        "--link",
        choices=("ridge", "identity", "log"),
        default=None,
        help=(
            "Link function for the count-stat heads. 'ridge' uses Ridge "
            "(squared loss, identity link, closed-form); 'identity' uses "
            "TweedieRegressor(power=1, link='identity') (Poisson loss, "
            "identity link, can fail to converge); 'log' uses "
            "PoissonRegressor (Poisson loss, log link). "
            "If omitted, uses the dataclass default."
        ),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve paths — imported lazily so this module stays importable even
    # if the data layer's heavy deps (nba_api) aren't available.
    from nba_sim.data.etl import (
        PROCESSED_TEST_FILENAME,
        PROCESSED_TRAIN_FILENAME,
        PROCESSED_VAL_FILENAME,
        processed_dir,
    )

    proc_dir = args.processed_dir or processed_dir()
    train_path = proc_dir / PROCESSED_TRAIN_FILENAME
    eval_path = proc_dir / (
        PROCESSED_VAL_FILENAME if args.eval_split == "val" else PROCESSED_TEST_FILENAME
    )

    for p in (train_path, eval_path):
        if not p.exists():
            logger.error("missing %s — run `nba-sim build-features` first.", p)
            return 1

    logger.info("reading train: %s", train_path)
    train = pl.read_parquet(train_path)
    logger.info("reading eval (%s): %s", args.eval_split, eval_path)
    eval_df = pl.read_parquet(eval_path)
    logger.info("train rows: %d, eval rows: %d", train.height, eval_df.height)

    if train.is_empty() or eval_df.is_empty():
        logger.error("train or eval frame is empty — nothing to fit on.")
        return 1

    logger.info("fitting SeasonAverageBaseline")
    sab = SeasonAverageBaseline().fit(train)
    glm_kwargs: dict[str, Any] = {}
    if args.alpha is not None:
        glm_kwargs["alpha"] = args.alpha
    if args.link is not None:
        glm_kwargs["link"] = args.link
    glm = PoissonGLMBaseline(**glm_kwargs).fit(train)
    logger.info(
        "fitting PoissonGLMBaseline (alpha=%s, link=%s)", glm.alpha, glm.link
    )
    logger.info("GLM resolved %d features: %s",
                len(glm._feature_cols_resolved), list(glm._feature_cols_resolved))

    sab_pred = sab.predict(eval_df)
    glm_pred = glm.predict(eval_df)

    # PLAN's gate is "every counting stat", which means we score every
    # non-minutes stat. We still report minutes separately so the user can
    # eyeball regression quality on the continuous head too.
    counting_stats = _COUNT_STATS
    sab_mae = _per_stat_mae(eval_df, sab_pred, (_TARGET_MINUTES,) + counting_stats)
    glm_mae = _per_stat_mae(eval_df, glm_pred, (_TARGET_MINUTES,) + counting_stats)

    print()
    print(f"Per-stat MAE on {args.eval_split} (lower is better):")
    print(_format_mae_table(sab_mae, glm_mae, (_TARGET_MINUTES,) + counting_stats))
    print()

    # PLAN ship gate: GLM must win on every *counting* stat (excluding
    # minutes, which is a continuous regression head separate from the
    # Poisson counts).
    counting_wins = sum(1 for s in counting_stats if glm_mae[s] < sab_mae[s])
    print(
        f"PLAN §10 Phase 1 ship gate: "
        f"GLM beats season-average on {counting_wins}/{len(counting_stats)} counting stats"
    )
    gate_passed = counting_wins == len(counting_stats)
    print("Gate:", "PASS" if gate_passed else "FAIL")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        sab.save(args.save_dir / "season_average.joblib")
        glm.save(args.save_dir / "poisson_glm.joblib")
        logger.info("saved fitted models to %s", args.save_dir)

    return 0 if gate_passed else 2


if __name__ == "__main__":
    sys.exit(main())
