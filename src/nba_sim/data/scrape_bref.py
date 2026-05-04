"""Basketball Reference scraper — fallback when nba_api is incomplete.

Use sparingly and politely:
    - >= 3 s between requests (configurable via ``BREF_MIN_DELAY_SECONDS``).
    - Respect robots.txt.
    - Always cache to ``data/raw/.cache/bref/``.

Only used to fill specific gaps (e.g. pre-2000 seasons, flaky height/weight
lookups from ``commonplayerinfo``). v1 only covers 2000+ so this is mostly
a safety net; keep it off in ``configs/data.yaml`` by default.
"""

from __future__ import annotations

import polars as pl


def scrape_player_bio(bref_id: str) -> pl.DataFrame:
    """Scrape the Basketball Reference player page for bio fields."""
    raise NotImplementedError


def scrape_season_boxes(season_year: int) -> pl.DataFrame:
    """Scrape Basketball Reference regular-season box scores for one season."""
    raise NotImplementedError


def map_bref_id_to_nba_id(bref_id: str) -> int | None:
    """Cross-reference Basketball Reference slug to the nba.com player id."""
    raise NotImplementedError
