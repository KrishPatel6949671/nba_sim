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

import typer

app = typer.Typer(
    name="nba-sim",
    help="NBA box score simulator — data, training, simulation.",
    no_args_is_help=True,
)


@app.command()
def fetch(
    start_season: int = typer.Option(..., help="First season start year, e.g. 2000"),
    end_season: int = typer.Option(..., help="Last season start year, inclusive"),
    refresh: bool = typer.Option(False, help="Bypass cache and refetch"),
) -> None:
    """Download / refresh seasons via nba_api."""
    raise NotImplementedError


@app.command("build-features")
def build_features(
    seasons: str | None = typer.Option(None, help="Comma-separated season list, default = all from configs/data.yaml"),
) -> None:
    """Run raw -> interim -> processed for the given seasons."""
    raise NotImplementedError


@app.command()
def train(
    config: str = typer.Option("configs/train.yaml"),
    model: str = typer.Option("hierarchical", help="{baseline|hierarchical}"),
    max_epochs: int | None = typer.Option(None, help="Override config max_epochs"),
) -> None:
    """Train the baseline or the hierarchical NN."""
    raise NotImplementedError


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
