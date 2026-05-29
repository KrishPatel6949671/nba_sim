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
    checkpoint: Path = typer.Option(
        Path("models/best.pt"), help="Path to .pt checkpoint."
    ),
    config: Path = typer.Option(
        Path("configs/train.yaml"),
        help="train.yaml; its data_config resolves the processed parquets.",
    ),
    report_dir: Path = typer.Option(
        Path("reports"), help="Where to write metrics.json + plots."
    ),
    n_interval_samples: int = typer.Option(
        200, help="Samples per game for empirical 10th/90th-percentile intervals."
    ),
    batch_size: int = typer.Option(64),
    device: str = typer.Option("auto", help="{auto|cpu|cuda}"),
    seed: int = typer.Option(0),
    no_plots: bool = typer.Option(False, help="Skip reliability + scatter PNGs."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Evaluate a checkpoint on the named split.

    Resolves ``<split>.parquet`` and ``train.parquet`` from
    ``train.yaml → data_config → paths.processed`` (same path discipline
    as ``nba-sim train``). Runs the full eval pipeline (per-stat MAE,
    interval coverage, constraint violations, team-PTS MAE under both
    the player-sum and team-head paths) and writes
    ``<report_dir>/metrics.json`` plus diagnostic PNGs.
    """
    _setup_logging(verbose)

    if split not in ("val", "test"):
        raise typer.BadParameter(f"split must be 'val' or 'test', got {split!r}")
    if not checkpoint.exists():
        raise typer.BadParameter(f"checkpoint not found: {checkpoint}")
    if not config.exists():
        raise typer.BadParameter(f"config file not found: {config}")

    from nba_sim.training.evaluate import evaluate as run_eval
    from nba_sim.training.loop import _load_yaml, _resolve_path

    train_cfg = _load_yaml(config)
    repo_root = config.resolve().parent.parent
    data_cfg = _load_yaml(_resolve_path(train_cfg["data_config"], repo_root))
    processed_dir = _resolve_path(data_cfg["paths"]["processed"], repo_root)

    train_parquet = processed_dir / "train.parquet"
    eval_parquet = processed_dir / f"{split}.parquet"
    for label, p in (("train", train_parquet), (split, eval_parquet)):
        if not p.exists():
            raise typer.BadParameter(f"{label} parquet not found: {p}")

    summary = run_eval(
        checkpoint=checkpoint,
        parquet=eval_parquet,
        train_parquet=train_parquet,
        report_dir=report_dir,
        n_interval_samples=n_interval_samples,
        batch_size=batch_size,
        device=device,
        seed=seed,
        include_plots=not no_plots,
    )

    typer.echo(f"per-stat MAE: {summary['per_stat_mae']}")
    typer.echo(
        f"team PTS MAE (player-sum): {summary['team_pts_mae']:.3f}"
    )
    typer.echo(
        f"team PTS MAE (team-head):  {summary['team_pts_mae_team_head']:.3f}"
    )
    typer.echo(f"pace MAE:     {summary['pace_mae']:.3f}")
    typer.echo(f"off_rtg MAE:  {summary['off_rtg_mae']:.3f}")
    typer.echo(f"80% PI coverage: {summary['interval_coverage']}")
    typer.echo(f"constraint violations: {summary['constraint_violation_rate']}")


@app.command()
def simulate(
    home: str = typer.Option(..., help="3-letter home team abbr"),
    away: str = typer.Option(..., help="3-letter away team abbr"),
    date: str = typer.Option(..., help="ISO YYYY-MM-DD"),
    n_samples: int = typer.Option(1, help="1=single sample; >1=ensemble"),
    seed: int | None = typer.Option(None, help="RNG seed for determinism"),
    checkpoint: Path = typer.Option(Path("models/best.pt")),
    train_parquet: Path = typer.Option(Path("data/processed/train.parquet")),
    val_parquet: Path = typer.Option(Path("data/processed/val.parquet")),
    test_parquet: Path = typer.Option(Path("data/processed/test.parquet")),
    device: str = typer.Option("auto", help="{auto|cpu|cuda}"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Simulate a single game (or an ensemble). Prints box score to stdout.

    The (home, away, date) tuple must currently correspond to a game
    present in the val or test split. Custom rosters / arbitrary dates
    are a v2 feature (see simulate/api.py docstring).
    """
    _setup_logging(verbose)
    from nba_sim.simulate.api import simulate_game

    for label, p in (
        ("checkpoint", checkpoint),
        ("train", train_parquet),
        ("val", val_parquet),
    ):
        if not p.exists():
            raise typer.BadParameter(f"{label} not found: {p}")

    test_arg = test_parquet if test_parquet.exists() else None

    result = simulate_game(
        home_team=home,
        away_team=away,
        date=date,
        n_samples=n_samples,
        seed=seed,
        device=device,
        checkpoint=checkpoint,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_arg,
    )

    if n_samples == 1:
        _print_box_score(result)
    else:
        typer.echo(f"== ensemble of {n_samples} samples ==")
        typer.echo("\nMEAN box score:")
        _print_box_score(result.mean)
        typer.echo("\n10th/90th percentile team totals:")
        for side, lo, hi in (
            ("home", result.interval_low.home, result.interval_high.home),
            ("away", result.interval_low.away, result.interval_high.away),
        ):
            typer.echo(
                f"  {lo.team:>3}: pts [{lo.pts}, {hi.pts}], "
                f"pace [{lo.pace:.1f}, {hi.pace:.1f}], "
                f"off_rtg [{lo.off_rtg:.1f}, {hi.off_rtg:.1f}]"
            )


def _print_box_score(bs) -> None:  # type: ignore[no-untyped-def]
    """One-game stdout rendering. Tight columns, top-of-rotation focus."""
    for side_name, side in (("HOME", bs.home), ("AWAY", bs.away)):
        typer.echo(
            f"\n[{side_name}] {side.team}: "
            f"PTS={side.pts}  pace={side.pace:.1f}  "
            f"off_rtg={side.off_rtg:.1f}  def_rtg={side.def_rtg:.1f}"
        )
        typer.echo(
            f"  {'Player':<22} {'MIN':>5} {'PTS':>4} "
            f"{'FG':>7} {'3P':>7} {'FT':>7} "
            f"{'REB':>4} {'AST':>4} {'STL':>4} {'BLK':>4} {'TOV':>4}"
        )
        # Sort by minutes desc — top of rotation first.
        for p in sorted(side.players, key=lambda x: -x.minutes):
            if p.minutes <= 0.0:
                continue
            typer.echo(
                f"  {p.player_name[:22]:<22} "
                f"{p.minutes:>5.1f} {p.pts:>4} "
                f"{p.fgm:>2}/{p.fga:<4} {p.tpm:>2}/{p.tpa:<4} {p.ftm:>2}/{p.fta:<4} "
                f"{p.reb:>4} {p.ast:>4} {p.stl:>4} {p.blk:>4} {p.tov:>4}"
            )


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
