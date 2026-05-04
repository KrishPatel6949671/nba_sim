"""nba_api wrappers with caching + rate limiting.

Every outbound call goes through :func:`cached_call`, which:

1. Hashes ``(endpoint, params)`` + dataset index into a cache key and
   returns the cached result from
   ``data/raw/.cache/<endpoint>/<hash>.<NN>.parquet`` on hit.
2. On miss, calls the endpoint with :mod:`tenacity` retries (exponential
   backoff, jittered, max 5 attempts) behind a module-level token bucket
   that enforces ``NBA_API_MIN_DELAY_SECONDS`` between requests.
3. Writes *every* dataset returned by the endpoint to the cache, so a
   later call for a different dataset index hits the cache without
   re-querying the network.

Endpoints wrapped (see PLAN.md §2.1):
    - leaguegamefinder
    - boxscoretraditionalv3
    - boxscoreadvancedv3
    - commonteamroster
    - commonplayerinfo
    - playergamelog
    - leaguedashteamstats
    - scoreboardv2
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import polars as pl
from requests.exceptions import RequestException
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Endpoint registry. Imported lazily so that importing this module does not
# pull nba_api in until a fetch is actually performed — nba_api imports
# pandas and a pile of HTTP machinery we don't need for ETL-only workflows.
# ---------------------------------------------------------------------------

_ENDPOINT_REGISTRY: dict[str, tuple[str, str]] = {
    "leaguegamefinder":      ("nba_api.stats.endpoints.leaguegamefinder",      "LeagueGameFinder"),
    "boxscoretraditionalv3": ("nba_api.stats.endpoints.boxscoretraditionalv3", "BoxScoreTraditionalV3"),
    "boxscoreadvancedv3":    ("nba_api.stats.endpoints.boxscoreadvancedv3",    "BoxScoreAdvancedV3"),
    "commonteamroster":      ("nba_api.stats.endpoints.commonteamroster",      "CommonTeamRoster"),
    "commonplayerinfo":      ("nba_api.stats.endpoints.commonplayerinfo",      "CommonPlayerInfo"),
    "playergamelog":         ("nba_api.stats.endpoints.playergamelog",         "PlayerGameLog"),
    "leaguedashteamstats":   ("nba_api.stats.endpoints.leaguedashteamstats",   "LeagueDashTeamStats"),
    "scoreboardv2":          ("nba_api.stats.endpoints.scoreboardv2",          "ScoreboardV2"),
}


def _get_endpoint_class(name: str) -> type:
    if name not in _ENDPOINT_REGISTRY:
        known = ", ".join(sorted(_ENDPOINT_REGISTRY))
        raise ValueError(f"Unknown endpoint {name!r}. Known: {known}")
    module_path, class_name = _ENDPOINT_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Cache-path helpers
# ---------------------------------------------------------------------------

def cache_dir() -> Path:
    """Return the cache root, respecting ``NBA_SIM_CACHE_DIR`` env var."""
    p = os.environ.get("NBA_SIM_CACHE_DIR", "data/raw/.cache")
    return Path(p).expanduser().resolve()


def _jsonable(v: Any) -> Any:
    """Coerce a param value to something ``json.dumps`` can serialize."""
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (_dt.date, _dt.datetime)):
        return v.isoformat()
    return str(v)


def _cache_key(params: dict[str, Any]) -> str:
    """Stable SHA-256 prefix over ``params`` — 16 hex chars is plenty
    (collisions vanishingly unlikely at our cardinality)."""
    normalized = {k: _jsonable(v) for k, v in sorted(params.items())}
    payload = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cache_path(endpoint: str, key: str, dataset: int) -> Path:
    # Two-digit dataset index keeps filenames sortable and avoids a clash
    # with someone's future ``<hash>.0`` vs ``<hash>.10``.
    return cache_dir() / endpoint / f"{key}.{dataset:02d}.parquet"


def _write_atomic(df: pl.DataFrame, path: Path) -> None:
    """Write to ``path.tmp`` then rename — prevents half-written files
    if the process is killed mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Rate limiting — module-level token bucket, 1 token, delay in seconds.
# Shared across threads; multiple processes will each have their own
# (fine for our use — we never parallelize fetch across processes).
# ---------------------------------------------------------------------------

_last_request_time = 0.0
_request_lock = threading.Lock()


def _rate_limit() -> None:
    global _last_request_time
    min_delay = float(os.environ.get("NBA_API_MIN_DELAY_SECONDS", "0.6"))
    with _request_lock:
        now = time.monotonic()
        elapsed = now - _last_request_time
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        _last_request_time = time.monotonic()


# ---------------------------------------------------------------------------
# Core: rate-limited, retried endpoint invocation.
# ---------------------------------------------------------------------------

_retry_policy = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1, max=30),
    retry=retry_if_exception_type(RequestException),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@_retry_policy
def _call_endpoint(endpoint: str, params: dict[str, Any]) -> list[pl.DataFrame]:
    """Invoke nba_api and return every dataset as polars DataFrames."""
    _rate_limit()
    cls = _get_endpoint_class(endpoint)
    timeout = float(os.environ.get("NBA_API_REQUEST_TIMEOUT", "30"))
    logger.info("nba_api call: %s %s", endpoint, params)
    instance = cls(timeout=timeout, **params)
    pandas_frames = instance.get_data_frames()
    return [pl.from_pandas(df) for df in pandas_frames]


# ---------------------------------------------------------------------------
# Public primitive
# ---------------------------------------------------------------------------

def cached_call(
    endpoint: str,
    params: dict[str, Any],
    *,
    dataset: int = 0,
    refresh: bool = False,
) -> pl.DataFrame:
    """Fetch ``endpoint`` with ``params``, using an on-disk cache.

    Args:
        endpoint: nba_api endpoint name (e.g. ``"boxscoretraditionalv3"``).
            Must be one of the keys in :data:`_ENDPOINT_REGISTRY`.
        params: kwargs forwarded to the endpoint constructor.
        dataset: which of the endpoint's returned dataframes to return.
            Endpoints like ``boxscoretraditionalv3`` emit several; see the
            typed helpers below for the right index per endpoint.
        refresh: if True, bypass the cache and refetch.
    """
    key = _cache_key(params)
    target = _cache_path(endpoint, key, dataset)

    if target.exists() and not refresh:
        logger.debug("cache hit: %s/%s.%02d", endpoint, key, dataset)
        return pl.read_parquet(target)

    frames = _call_endpoint(endpoint, params)

    # Persist every dataset — later calls for other indices will hit the cache.
    for i, df in enumerate(frames):
        _write_atomic(df, _cache_path(endpoint, key, i))

    if dataset >= len(frames):
        raise IndexError(
            f"{endpoint} returned {len(frames)} dataset(s); requested index {dataset}"
        )
    return frames[dataset]


# ---------------------------------------------------------------------------
# Typed helpers — one per endpoint from PLAN.md §2.1.
# ---------------------------------------------------------------------------

def _season_label(start_year: int) -> str:
    """Convert a start year (e.g. ``2023``) to nba_api's season format (``"2023-24"``)."""
    end = (start_year + 1) % 100
    return f"{start_year}-{end:02d}"


def fetch_games_for_season(
    season: int,
    *,
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pl.DataFrame:
    """Return the ``LeagueGameFinder`` game list for one season start year."""
    return cached_call(
        "leaguegamefinder",
        {
            "season_nullable": _season_label(season),
            "season_type_nullable": season_type,
            "league_id_nullable": "00",
        },
        dataset=0,
        refresh=refresh,
    )


def fetch_player_box(game_id: str, *, refresh: bool = False) -> pl.DataFrame:
    """Per-player traditional box — dataset 0 of ``BoxScoreTraditionalV3``."""
    return cached_call(
        "boxscoretraditionalv3", {"game_id": game_id}, dataset=0, refresh=refresh
    )


def fetch_team_box(game_id: str, *, refresh: bool = False) -> pl.DataFrame:
    """Per-team traditional box — dataset 2 of ``BoxScoreTraditionalV3``."""
    return cached_call(
        "boxscoretraditionalv3", {"game_id": game_id}, dataset=2, refresh=refresh
    )


def fetch_player_advanced_box(game_id: str, *, refresh: bool = False) -> pl.DataFrame:
    """Per-player advanced box (usage, TS%) — dataset 0 of ``BoxScoreAdvancedV3``."""
    return cached_call(
        "boxscoreadvancedv3", {"game_id": game_id}, dataset=0, refresh=refresh
    )


def fetch_team_advanced_box(game_id: str, *, refresh: bool = False) -> pl.DataFrame:
    """Per-team advanced box (pace, off/def rating) — dataset 1 of ``BoxScoreAdvancedV3``."""
    return cached_call(
        "boxscoreadvancedv3", {"game_id": game_id}, dataset=1, refresh=refresh
    )


def fetch_roster(team_id: int, season: int, *, refresh: bool = False) -> pl.DataFrame:
    """Roster for a team in a given season start year."""
    return cached_call(
        "commonteamroster",
        {"team_id": team_id, "season": _season_label(season)},
        dataset=0,
        refresh=refresh,
    )


def fetch_player_info(player_id: int, *, refresh: bool = False) -> pl.DataFrame:
    """Static player info (height, weight, position, draft year, experience)."""
    return cached_call(
        "commonplayerinfo", {"player_id": player_id}, dataset=0, refresh=refresh
    )


def fetch_player_game_log(
    player_id: int,
    season: int,
    *,
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pl.DataFrame:
    """Per-game log for one player in one season."""
    return cached_call(
        "playergamelog",
        {
            "player_id": player_id,
            "season": _season_label(season),
            "season_type_all_star": season_type,
        },
        dataset=0,
        refresh=refresh,
    )


def fetch_league_team_stats(
    season: int,
    *,
    season_type: str = "Regular Season",
    refresh: bool = False,
) -> pl.DataFrame:
    """Season-level team stats (priors for cold-start and calibration)."""
    return cached_call(
        "leaguedashteamstats",
        {
            "season": _season_label(season),
            "season_type_all_star": season_type,
        },
        dataset=0,
        refresh=refresh,
    )


def fetch_scoreboard(
    game_date: str | _dt.date,
    *,
    refresh: bool = False,
) -> pl.DataFrame:
    """Game headers for a single calendar date — dataset 0 of ``ScoreboardV2``."""
    iso = game_date.isoformat() if isinstance(game_date, _dt.date) else game_date
    return cached_call(
        "scoreboardv2",
        {"game_date": iso, "league_id": "00", "day_offset": 0},
        dataset=0,
        refresh=refresh,
    )


# ---------------------------------------------------------------------------
# Cache maintenance
# ---------------------------------------------------------------------------

def cache_clear(endpoint: str | None = None) -> int:
    """Delete cached entries. Returns number of files removed."""
    root = cache_dir()
    target = root / endpoint if endpoint else root
    if not target.exists():
        return 0
    count = 0
    for p in target.rglob("*.parquet"):
        p.unlink()
        count += 1
    return count


def cache_stats() -> dict[str, int]:
    """Return ``{endpoint: file_count}`` under the cache root."""
    root = cache_dir()
    if not root.exists():
        return {}
    return {
        d.name: sum(1 for _ in d.rglob("*.parquet"))
        for d in sorted(p for p in root.iterdir() if p.is_dir())
    }

if __name__ == "__main__":
    t = time.time()
    df = fetch_games_for_season(2023)
    print(f"{time.time()-t:.2f}s, {df.shape}, cols={df.columns[:6]}")
    t = time.time()
    df2 = fetch_games_for_season(2023)
    print(f"{time.time()-t:.2f}s  # should be ~0s, cache hit")

    gid = df["GAME_ID"][0]
    box = fetch_player_box(gid)
    print(box.shape, box.columns[:6])
    print(cache_stats())