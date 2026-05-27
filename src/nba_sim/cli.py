"""Command-line interface — the primary entry point for end users.

Subcommands (see PLAN.md §8):

    fetch           — download/refresh seasons via nba_api
    build-features  — raw -> interim -> processed
    train           — train baseline or hierarchical model
    evaluate        — evaluate a checkpoint on val/test
    simulate        — simulate a single game (or an ensemble)
    cache-stats     — print cache size and per-endpoint counts
    cache-clear     — delete cached entries for a given endpoint

Built with ``typer``. The ``app`` Typer instance is exposed as the
``nba-sim`` console script via ``pyproject.toml`` [project.scripts].
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
import yaml

app = typer.Typer(
    name="nba-sim",
    help="NBA box score simulator — data, training, simulation.",
    no_args_is_help=True,
)

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_splits(config_path: Path):
    """Load configs/data.yaml → ``SplitSpec``.

    Imported here (rather than at module top) because the SplitSpec class
    lives in ``nba_sim.data.etl`` which in turn imports ``nba_api`` machinery
    transitively. Keeping the import lazy means ``nba-sim --help`` stays fast.
    """
    from nba_sim.data.etl import SplitSpec

    if not config_path.exists():
        raise typer.BadParameter(f"config file not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())
    if "splits" not in cfg:
        raise typer.BadParameter(f"config {config_path} missing 'splits' section")
    s = cfg["splits"]
    return SplitSpec(train=list(s["train"]), val=list(s["val"]), test=list(s["test"]))


@app.command()
def fetch(
    start_season: int = typer.Option(..., help="First season start year, e.g. 2000"),
    end_season: int = typer.Option(..., help="Last season start year, inclusive"),
    refresh: bool = typer.Option(False, help="Bypass cache and refetch"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download / refresh seasons via nba_api and write interim parquets.

    Iterates ``[start_season, end_season]`` inclusive and runs
    :func:`raw_to_interim` for each. That helper drives the per-game fetch
    of player/team/advanced box scores (each call cached on disk), then
    validates and writes ``data/interim/<season>/{games,player_box,
    team_box,rosters}.parquet`` plus a ``qa_report.json``.

    Idempotent: seasons whose interim outputs are newer than the upstream
    cache mtimes are skipped (unless ``--refresh`` is passed).
    """
    _setup_logging(verbose)
    from nba_sim.data.etl import raw_to_interim

    if end_season < start_season:
        raise typer.BadParameter(
            f"end_season ({end_season}) must be >= start_season ({start_season})"
        )

    for season in range(start_season, end_season + 1):
        logger.info("fetching + interim for season %d", season)
        raw_to_interim(season, refresh=refresh)


@app.command("build-features")
def build_features(
    seasons: str | None = typer.Option(
        None, help="Comma-separated season list, default = all from configs/data.yaml"
    ),
    config: Path = typer.Option(
        Path("configs/data.yaml"),
        help="Splits config (only used when --seasons is omitted).",
    ),
    refresh: bool = typer.Option(False, help="Force re-build, ignoring up-to-date outputs"),
    skip_processed: bool = typer.Option(
        False, help="Build per-season feature tables only; skip the train/val/test write."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build feature tables (and the processed train/val/test parquets).

    For each season in the resolved list, calls
    :func:`build_feature_tables` (writes ``player_features.parquet``,
    ``team_features.parquet``, ``season_to_date.parquet`` under
    ``data/interim/<season>/``).

    Unless ``--skip-processed`` is passed, then calls
    :func:`interim_to_processed` with the splits from the config to emit
    ``data/processed/{train,val,test}.parquet``. ``interim_to_processed``
    auto-builds any feature tables not already present, so you can run
    just this command after ``nba-sim fetch`` finishes.

    If ``--seasons`` is given, only those seasons get feature tables and
    the processed write is implicitly skipped (the splits config seasons
    aren't built so a write would be incomplete).
    """
    _setup_logging(verbose)
    from nba_sim.data.etl import build_feature_tables, interim_to_processed

    if seasons is not None:
        season_list = [int(s.strip()) for s in seasons.split(",") if s.strip()]
        for season in season_list:
            logger.info("building features for season %d", season)
            build_feature_tables(season, refresh=refresh)
        logger.info(
            "explicit --seasons given; skipping processed write to avoid partial output. "
            "Re-run without --seasons to write data/processed/{train,val,test}.parquet."
        )
        return

    splits = _load_splits(config)
    all_seasons = sorted(set(splits.train) | set(splits.val) | set(splits.test))
    for season in all_seasons:
        logger.info("building features for season %d", season)
        build_feature_tables(season, refresh=refresh)

    if skip_processed:
        logger.info("--skip-processed given; not writing train/val/test parquets")
        return

    logger.info("writing processed train/val/test parquets")
    paths = interim_to_processed(splits, refresh=refresh)
    for split_name, path in paths.items():
        logger.info("  %s -> %s", split_name, path)


@app.command()
def train(
    config: Path = typer.Option(
        Path("configs/train.yaml"),
        help="Path to train.yaml.",
    ),
    max_epochs: int | None = typer.Option(
        None, help="Override schedule.max_epochs."
    ),
    seed: int | None = typer.Option(None, help="Override run.seed."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train the hierarchical NN end-to-end.

    Resolves ``model_config`` / ``data_config`` from ``train.yaml`` relative
    to the repo root, runs the training loop, and writes checkpoints +
    ``training_summary.json`` under ``paths.checkpoint_dir``.
    """
    _setup_logging(verbose)
    from nba_sim.training.loop import train_from_yaml

    if not config.exists():
        raise typer.BadParameter(f"config file not found: {config}")

    overrides: dict = {}
    if max_epochs is not None:
        overrides["schedule"] = {"max_epochs": max_epochs}
    if seed is not None:
        overrides["run"] = {"seed": seed}

    summary = train_from_yaml(config, overrides=overrides or None)
    typer.echo(
        f"best_val_nll={summary['best_val_nll']:.6f} "
        f"@ epoch {summary['best_epoch']} "
        f"({summary['epochs_trained']} trained)"
    )


@app.command()
def evaluate(
    split: str = typer.Option("val", help="{val|test}"),
    checkpoint: str = typer.Option("models/best.pt"),
) -> None:
    """Evaluate a checkpoint on the named split."""
    raise NotImplementedError


@app.command()
def simulate(
    home: str = typer.Option(..., help="3-letter home team abbr"),
    away: str = typer.Option(..., help="3-letter away team abbr"),
    date: str = typer.Option(..., help="ISO YYYY-MM-DD"),
    n_samples: int = typer.Option(1),
    seed: int | None = typer.Option(None),
    checkpoint: str = typer.Option("models/best.pt"),
) -> None:
    """Simulate a single game (or an ensemble). Prints to stdout."""
    raise NotImplementedError


@app.command("cache-stats")
def cache_stats() -> None:
    """Print size and per-endpoint counts for the nba_api cache."""
    raise NotImplementedError


@app.command("cache-clear")
def cache_clear(
    endpoint: str | None = typer.Option(None, help="Only clear this endpoint; default = all"),
) -> None:
    """Delete cached entries."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
