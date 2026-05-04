"""Poisson GLM + Beta regression baseline.

The hierarchical NN must **beat** this model or v1 is not shippable
(see PLAN.md §6.3). Implemented with scikit-learn / statsmodels so it
trains in minutes and is independent of the PyTorch stack.

Two baselines live here:

1. ``SeasonAverageBaseline`` — predict player's season-to-date average.
2. ``PoissonGLMBaseline`` — one Poisson GLM per counting stat, one Beta
   regression per percentage stat, minutes via softmax over the roster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class SeasonAverageBaseline:
    """Predict each player's prior-games season mean for each stat."""

    def fit(self, train: pl.DataFrame) -> "SeasonAverageBaseline":
        raise NotImplementedError

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError


@dataclass
class PoissonGLMBaseline:
    """Stat-wise Poisson / Beta regression baseline."""

    def fit(self, train: pl.DataFrame) -> "PoissonGLMBaseline":
        raise NotImplementedError

    def predict(self, rows: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def project_to_constraints(self, predicted: pl.DataFrame) -> pl.DataFrame:
        """Apply post-hoc projection so MIN sums to 240 and made <= attempted.

        Only the baseline needs this; the NN enforces constraints by construction.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "PoissonGLMBaseline":
        raise NotImplementedError
