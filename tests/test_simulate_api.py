"""Public ``simulate_game`` API tests.

Covers the contract in PLAN.md §7:
    - Schema validity of the returned :class:`BoxScore`.
    - Determinism: same seed on same device -> identical box score.
    - ``n_samples=1`` vs. ``n_samples>1`` return-type switch.
    - Cold-start path for unseen players does not crash and produces
      non-degenerate output.
    - Mean-mode returns expected values.

These tests are *integration* — they need a trained checkpoint
(``models/best.pt``) and the processed parquets on disk. When those
artifacts are missing (e.g. a fresh clone in CI), the tests skip rather
than fail.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nba_sim.data.schema import BoxScore, BoxScoreEnsemble


_REPO = Path(__file__).resolve().parent.parent
_CKPT = _REPO / "models" / "best.pt"
_TRAIN = _REPO / "data" / "processed" / "train.parquet"
_VAL = _REPO / "data" / "processed" / "val.parquet"


def _require_artifacts() -> None:
    """Skip when the trained model or processed parquets aren't on disk."""
    missing = [p for p in (_CKPT, _TRAIN, _VAL) if not p.exists()]
    if missing:
        names = ", ".join(p.relative_to(_REPO).as_posix() for p in missing)
        pytest.skip(f"missing artifacts ({names}) — run train + build-features first")


def _find_one_real_game() -> tuple[str, str, str]:
    """Pick the first (home_abbr, away_abbr, date_iso) tuple from val.parquet.

    Used to give every test a known-present game without hard-coding a
    specific matchup that might be missing from any given val split.
    """
    import polars as pl
    rows = (
        pl.scan_parquet(_VAL)
        .select(["game_id", "team_abbr", "is_home", "date"])
        .unique()
        .collect()
        .sort(["game_id", "is_home"], descending=[False, True])
    )
    if rows.is_empty():
        pytest.skip("val.parquet has no rows")
    first_gid = rows["game_id"][0]
    game = rows.filter(pl.col("game_id") == first_gid)
    home = game.filter(pl.col("is_home"))
    away = game.filter(~pl.col("is_home"))
    if home.is_empty() or away.is_empty():
        pytest.skip(f"game {first_gid} missing a side in val.parquet")
    return home["team_abbr"][0], away["team_abbr"][0], home["date"][0].isoformat()


def _simulate(**kwargs):
    """Wrap simulate_game with the v1 path settings: cpu, our parquets."""
    from nba_sim.simulate.api import simulate_game
    return simulate_game(
        device="cpu",
        checkpoint=str(_CKPT),
        train_parquet=str(_TRAIN),
        val_parquet=str(_VAL),
        test_parquet=None,
        **kwargs,
    )


def test_simulate_game_returns_valid_boxscore() -> None:
    _require_artifacts()
    home, away, date = _find_one_real_game()
    bs = _simulate(home_team=home, away_team=away, date=date, n_samples=1, seed=42)
    assert isinstance(bs, BoxScore)
    assert bs.home.team == home
    assert bs.away.team == away
    # Schema invariants the sampler guarantees.
    assert bs.home.pts == sum(p.pts for p in bs.home.players)
    assert bs.away.pts == sum(p.pts for p in bs.away.players)
    assert bs.home.def_rtg == bs.away.off_rtg
    assert bs.away.def_rtg == bs.home.off_rtg


def test_simulate_game_deterministic_with_seed() -> None:
    _require_artifacts()
    home, away, date = _find_one_real_game()
    a = _simulate(home_team=home, away_team=away, date=date, n_samples=1, seed=99)
    b = _simulate(home_team=home, away_team=away, date=date, n_samples=1, seed=99)
    assert a.home.pts == b.home.pts
    assert a.away.pts == b.away.pts
    for pa, pb in zip(a.home.players, b.home.players):
        assert pa.pts == pb.pts and pa.minutes == pb.minutes


def test_simulate_game_n_samples_returns_ensemble() -> None:
    _require_artifacts()
    home, away, date = _find_one_real_game()
    out_single = _simulate(home_team=home, away_team=away, date=date, n_samples=1, seed=1)
    out_many = _simulate(home_team=home, away_team=away, date=date, n_samples=6, seed=1)
    assert isinstance(out_single, BoxScore)
    assert isinstance(out_many, BoxScoreEnsemble)
    assert len(out_many.samples) == 6
    # Mean + intervals are also BoxScores with the same roster sizes.
    assert isinstance(out_many.mean, BoxScore)
    assert len(out_many.mean.home.players) == len(out_many.samples[0].home.players)


def test_simulate_game_minutes_sum_to_240_per_team() -> None:
    """Integration-level constraint check on the public API."""
    _require_artifacts()
    home, away, date = _find_one_real_game()
    bs = _simulate(home_team=home, away_team=away, date=date, n_samples=1, seed=7)
    # Allow tiny float32 noise from the round-with-residual step.
    assert abs(sum(p.minutes for p in bs.home.players) - 240.0) < 1e-2
    assert abs(sum(p.minutes for p in bs.away.players) - 240.0) < 1e-2


def test_simulate_game_unknown_game_raises_lookup() -> None:
    """Asking for a game not in val.parquet should fail with LookupError —
    NOT a silent wrong-game return."""
    _require_artifacts()
    with pytest.raises(LookupError):
        _simulate(home_team="XXX", away_team="YYY", date="2099-01-01", n_samples=1)


def test_simulate_game_unseen_player_cold_start() -> None:
    pytest.skip("v2: custom rosters not yet implemented")


def test_simulate_game_mean_mode() -> None:
    pytest.skip("v2: mean-mode return not yet implemented")
