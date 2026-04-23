"""IO helpers: atomic parquet writes, config loaders, path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file into a dict. Fails loudly on unknown keys in configs."""
    raise NotImplementedError


def write_parquet_atomic(df: pl.DataFrame, path: str | Path) -> None:
    """Write to ``path.tmp`` then rename to ``path`` — prevents half-written files."""
    raise NotImplementedError


def project_root() -> Path:
    """Return the repo root (the directory containing ``pyproject.toml``)."""
    raise NotImplementedError


def data_dir() -> Path:
    """Return the data root, respecting ``NBA_SIM_DATA_DIR`` env var."""
    raise NotImplementedError


def model_dir() -> Path:
    """Return the model checkpoint directory, respecting ``NBA_SIM_MODEL_DIR``."""
    raise NotImplementedError
