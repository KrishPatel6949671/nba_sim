"""Public simulation API — the function users actually call.

The signature is stable; v2 may add kwargs but will not remove them.
See PLAN.md §7 for the full contract.

v1 scope (this file)
--------------------

- ``(home_team, away_team, date)`` must correspond to a game present in
  one of the processed parquets (val by default, then test). Synthetic
  rosters and arbitrary-date prediction are deferred to v2 — they need a
  fresh feature-assembly path that doesn't reuse the training ETL.
- Cold-start handling for unseen players: deferred to v2. The dataset
  maps any out-of-vocab player_id to embedding id 0 (the pad/unknown
  bucket), so unseen players don't crash — they just get the unknown-
  player embedding. The three-tier projection from PLAN §7.3 is a v2
  feature.
- ``return_distributions=True`` is deferred to v2.

Determinism
-----------

When ``seed`` is provided, sampling is bit-deterministic on the same
device. CUDA determinism flags are NOT toggled here — the caller can
set ``torch.use_deterministic_algorithms(True)`` upstream if cross-run
determinism across CUDA driver / cuDNN versions matters.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import polars as pl
import torch

from nba_sim.data.schema import BoxScore, BoxScoreEnsemble
from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.simulate.sampler import sample_box_score, sample_ensemble
from nba_sim.training.dataset import BoxScoreDataset, collate_games


def simulate_game(
    home_team: str,
    away_team: str,
    date: str | _dt.date,
    *,
    home_roster: list[str] | None = None,
    away_roster: list[str] | None = None,
    n_samples: int = 1,
    seed: int | None = None,
    device: str = "auto",
    checkpoint: str | Path = "models/best.pt",
    train_parquet: str | Path = "data/processed/train.parquet",
    val_parquet: str | Path = "data/processed/val.parquet",
    test_parquet: str | Path | None = "data/processed/test.parquet",
    return_distributions: bool = False,
) -> BoxScore | BoxScoreEnsemble:
    """Simulate one game (or an ensemble) between ``home_team`` and ``away_team``.

    Args:
        home_team: 3-letter NBA team abbreviation (e.g. ``"BOS"``).
        away_team: same.
        date: ISO ``"YYYY-MM-DD"`` string or :class:`datetime.date`.
        home_roster, away_roster: **v2** — not yet supported. Pass
            ``None`` to use the rosters in the processed data.
        n_samples: ``1`` returns a :class:`BoxScore`; >1 returns a
            :class:`BoxScoreEnsemble` with mean / 80% PI / all samples.
        seed: RNG seed for determinism. When None, sampling is
            non-deterministic.
        device: ``"auto"`` / ``"cuda"`` / ``"cpu"``.
        checkpoint: path to a trained model checkpoint
            (``models/best.pt`` is the default produced by training).
        train_parquet: train split — required to pin the player_id_map
            and z-score statistics so val/test rows are featurized
            identically to how the model saw training data.
        val_parquet, test_parquet: searched in order for the
            ``(home_team, away_team, date)`` row.
        return_distributions: **v2** — not yet supported.

    Returns:
        :class:`BoxScore` when ``n_samples == 1``, otherwise
        :class:`BoxScoreEnsemble`.

    Raises:
        NotImplementedError: when v2-only kwargs are set
            (``home_roster``, ``away_roster``, ``return_distributions``).
        LookupError: when the requested game isn't in any of the
            provided parquets.
        ValueError: on bad ``n_samples``.
    """
    if home_roster is not None or away_roster is not None:
        raise NotImplementedError(
            "Custom rosters are a v2 feature; pass None to use the "
            "rosters from the processed data."
        )
    if return_distributions:
        raise NotImplementedError("return_distributions is a v2 feature.")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    date_obj = (
        _dt.date.fromisoformat(date) if isinstance(date, str) else date
    )
    device_obj = _resolve_device(device)

    train_ds = BoxScoreDataset(train_parquet)
    eval_ds, idx = _locate_game(
        home_team, away_team, date_obj,
        train_ds=train_ds,
        candidate_parquets=[p for p in (val_parquet, test_parquet) if p is not None],
    )

    model = HierarchicalBoxScoreModel.from_checkpoint(checkpoint).to(device_obj).eval()

    item = eval_ds[idx]
    batch = collate_games([item])
    batch = {
        k: (v.to(device_obj) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }
    # Drop teacher-forcing keys so the player-alloc head conditions on the
    # team head's predicted pace/off_rtg — that's what the simulator must
    # use at inference (the truth isn't available for unseen games).
    for k in ("pace_true", "off_rtg_true"):
        batch.pop(k, None)

    with torch.no_grad():
        dist = model(batch)

    home_players, away_players, home_abbr, away_abbr = _extract_rosters_and_teams(
        eval_ds, idx
    )
    date_iso = date_obj.isoformat()

    if n_samples == 1:
        return sample_box_score(
            dist,
            home_players=home_players, away_players=away_players,
            home_team=home_abbr, away_team=away_abbr,
            date_iso=date_iso, seed=seed,
        )
    return sample_ensemble(
        dist, n_samples,
        home_players=home_players, away_players=away_players,
        home_team=home_abbr, away_team=away_abbr,
        date_iso=date_iso, seed=seed,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def _locate_game(
    home_team: str,
    away_team: str,
    date_obj: _dt.date,
    *,
    train_ds: BoxScoreDataset,
    candidate_parquets: list[str | Path],
) -> tuple[BoxScoreDataset, int]:
    """Find the game across candidate parquets in order; return (ds, idx).

    Train-fitted ``player_id_map`` and ``feature_stats`` are reused on
    each candidate so val/test rows get the same normalization the
    model was trained against.
    """
    last_error: str | None = None
    for pq in candidate_parquets:
        pq_path = Path(pq)
        if not pq_path.exists():
            last_error = f"{pq_path} missing"
            continue
        ds = BoxScoreDataset(
            pq_path,
            player_id_map=train_ds.player_id_map,
            feature_stats=train_ds.feature_stats,
        )
        idx = _find_game_index(ds, home_team, away_team, date_obj)
        if idx is not None:
            return ds, idx
        last_error = (
            f"no game with home={home_team}, away={away_team}, "
            f"date={date_obj.isoformat()} in {pq_path.name}"
        )
    raise LookupError(
        last_error or f"no candidate parquets to search for "
        f"{home_team} vs {away_team} on {date_obj.isoformat()}"
    )


def _find_game_index(
    ds: BoxScoreDataset, home_team: str, away_team: str, date_obj: _dt.date
) -> int | None:
    """Linear scan for the matching game. O(N) over games per dataset —
    fine for a few thousand games and one inference call."""
    for i, game_df in enumerate(ds._games):
        if game_df["date"][0] != date_obj:
            continue
        home_rows = game_df.filter(pl.col("is_home"))
        away_rows = game_df.filter(~pl.col("is_home"))
        if home_rows.is_empty() or away_rows.is_empty():
            continue
        if (
            home_rows["team_abbr"][0] == home_team
            and away_rows["team_abbr"][0] == away_team
        ):
            return i
    return None


def _extract_rosters_and_teams(
    ds: BoxScoreDataset, idx: int
) -> tuple[list[tuple[int, str]], list[tuple[int, str]], str, str]:
    """Pull (player_id, player_name) lists in dataset order plus team
    abbreviations. Order matches the ordering the model's forward pass
    saw — :class:`BoxScoreDataset` sorts ``(is_home desc, minutes desc)``
    within each game so per-side ``head(max_players)`` matches the
    truncation the dataset applied to the model input."""
    game_df = ds._games[idx]
    home_df = game_df.filter(pl.col("is_home")).head(ds.max_players)
    away_df = game_df.filter(~pl.col("is_home")).head(ds.max_players)

    home_players = list(zip(
        home_df["player_id"].cast(pl.Int64).to_list(),
        home_df["player_name"].to_list(),
    ))
    away_players = list(zip(
        away_df["player_id"].cast(pl.Int64).to_list(),
        away_df["player_name"].to_list(),
    ))
    return home_players, away_players, home_df["team_abbr"][0], away_df["team_abbr"][0]


def _resolve_rosters(
    home_team: str,
    away_team: str,
    date: _dt.date,
    explicit_home: list[str] | None,
    explicit_away: list[str] | None,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """v2: resolve player strings to (id, display_name) pairs.

    The v1 path uses the rosters embedded in the processed parquet — see
    ``_extract_rosters_and_teams`` — so this resolution doesn't run yet.
    """
    raise NotImplementedError("v2 feature — see module docstring")
