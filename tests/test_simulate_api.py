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

import dataclasses
import datetime as _dt
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
    for pa, pb in zip(a.home.players, b.home.players, strict=False):
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


# ---------------------------------------------------------------------------
# v2 snapshot path (§16.5 / §18.2). Self-contained — builds a tiny interim,
# refreshes a snapshot, builds the processed parquets, and mints an (untrained)
# checkpoint from configs/model.yaml, so these run without real artifacts.
# ---------------------------------------------------------------------------

_SIM_SEASON = 2023


@dataclasses.dataclass
class _SimEnv:
    snapshot_dir: Path
    interim_dir: Path
    train_parquet: Path
    val_parquet: Path
    checkpoint: Path
    home: str
    away: str
    as_of: _dt.date
    real_date: _dt.date


def _sim_player(gid: str, pid: int, name: str, tid: int, abbr: str, minutes: int, pos: str):  # type: ignore[no-untyped-def]
    from nba_sim.data.schema import PlayerBoxLine

    m = max(float(minutes), 0.0)
    return PlayerBoxLine(
        game_id=gid, player_id=pid, player_name=name, team_id=tid, team_abbr=abbr, position=pos,
        minutes=m, pts=int(m), fgm=3, fga=6, tpm=1, tpa=2, ftm=2, fta=2, oreb=1, dreb=3, reb=4,
        ast=2, stl=1, blk=1, tov=1, pf=2, plus_minus=1.0, is_starter=bool(pos), is_active=True,
        dnp=(m == 0.0),
    )


def _sim_team(gid: str, tid: int, abbr: str, is_home: bool, pace: float):  # type: ignore[no-untyped-def]
    from nba_sim.data.schema import TeamBoxLine

    return TeamBoxLine(
        game_id=gid, team_id=tid, team_abbr=abbr, is_home=is_home, minutes=240.0,
        pts=110 if is_home else 108, fgm=40, fga=88, tpm=10, tpa=30, ftm=15, fta=20, oreb=10,
        dreb=34, reb=44, ast=25, stl=7, blk=5, tov=12, pf=18,
        plus_minus=2.0 if is_home else -2.0, pace=pace, off_rtg=112.0, def_rtg=108.0,
    )


def _build_sim_artifacts(root: Path) -> _SimEnv:
    import torch
    import yaml

    from nba_sim.data.etl import (
        GAMES_FILENAME,
        PLAYER_BOX_FILENAME,
        TEAM_BOX_FILENAME,
        SplitSpec,
        _models_to_df,
        build_feature_tables,
        interim_to_processed,
        processed_dir,
    )
    from nba_sim.data.schema import Game
    from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
    from nba_sim.snapshot import snapshot_dir
    from nba_sim.snapshot.refresh import refresh

    interim = root / "interim"
    season_dir = interim / str(_SIM_SEASON)
    season_dir.mkdir(parents=True, exist_ok=True)
    dates = [_dt.date(2023, 11, 1) + _dt.timedelta(days=2 * i) for i in range(12)]
    gids = [f"00{_SIM_SEASON}{i:04d}" for i in range(1, 13)]
    bos = [(10, "G"), (11, "F"), (12, "C")]
    lal = [(20, "G"), (21, "F"), (22, "C")]
    games, players, teams = [], [], []
    for i, (gid, d) in enumerate(zip(gids, dates, strict=True)):
        games.append(Game(
            game_id=gid, season=_SIM_SEASON, date=d, home_team_id=1, away_team_id=2,
            home_team_abbr="BOS", away_team_abbr="LAL", home_pts=110, away_pts=108, dropped=False,
        ))
        for j, (pid, pos) in enumerate(bos):
            players.append(_sim_player(gid, pid, f"BOS {pid}", 1, "BOS", 20 + i - 3 * j, pos))
        for j, (pid, pos) in enumerate(lal):
            players.append(_sim_player(gid, pid, f"LAL {pid}", 2, "LAL", 22 + i - 3 * j, pos))
        teams += [_sim_team(gid, 1, "BOS", True, 100.0 + i), _sim_team(gid, 2, "LAL", False, 98.0 + i)]
    _models_to_df(games).write_parquet(season_dir / GAMES_FILENAME)
    _models_to_df(players).write_parquet(season_dir / PLAYER_BOX_FILENAME)
    _models_to_df(teams).write_parquet(season_dir / TEAM_BOX_FILENAME)

    refresh(offline=True)
    build_feature_tables(_SIM_SEASON)
    interim_to_processed(SplitSpec(train=[_SIM_SEASON], val=[_SIM_SEASON], test=[]))

    config = yaml.safe_load((_REPO / "configs" / "model.yaml").read_text())
    checkpoint = root / "model.pt"
    model = HierarchicalBoxScoreModel(config)
    torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)

    return _SimEnv(
        snapshot_dir=snapshot_dir(),
        interim_dir=interim,
        train_parquet=processed_dir() / "train.parquet",
        val_parquet=processed_dir() / "val.parquet",
        checkpoint=checkpoint,
        home="BOS", away="LAL",
        as_of=dates[-1] + _dt.timedelta(days=1),
        real_date=dates[8],
    )


@pytest.fixture()
def sim_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _SimEnv:
    monkeypatch.setenv("NBA_SIM_INTERIM_DIR", str(tmp_path / "interim"))
    monkeypatch.setenv("NBA_SIM_SNAPSHOT_DIR", str(tmp_path / "snapshot"))
    monkeypatch.setenv("NBA_SIM_PROCESSED_DIR", str(tmp_path / "processed"))
    return _build_sim_artifacts(tmp_path)


def _sim(env: _SimEnv, **kwargs: object) -> BoxScore | BoxScoreEnsemble:
    from nba_sim.simulate.api import simulate_game

    base: dict[str, object] = {
        "home_team": env.home, "away_team": env.away, "device": "cpu",
        "checkpoint": env.checkpoint, "train_parquet": env.train_parquet,
        "val_parquet": env.val_parquet, "test_parquet": None,
        "snapshot_dir": env.snapshot_dir, "interim_dir": env.interim_dir, "stale_ok": True,
    }
    base.update(kwargs)
    return simulate_game(**base)


def test_simulate_snapshot_path_returns_valid_boxscore(sim_env: _SimEnv) -> None:
    bs = _sim(sim_env, date=None, n_samples=1, seed=42)
    assert isinstance(bs, BoxScore)
    assert bs.home.team == "BOS" and bs.away.team == "LAL"
    assert bs.date == sim_env.as_of  # snapshot path dates the box score as_of
    assert bs.home.pts == sum(p.pts for p in bs.home.players)
    assert bs.away.pts == sum(p.pts for p in bs.away.players)


def test_simulate_snapshot_deterministic_with_seed(sim_env: _SimEnv) -> None:
    a = _sim(sim_env, date=None, seed=99)
    b = _sim(sim_env, date=None, seed=99)
    assert isinstance(a, BoxScore) and isinstance(b, BoxScore)
    assert a.model_dump() == b.model_dump()


def test_simulate_snapshot_minutes_sum_to_240(sim_env: _SimEnv) -> None:
    bs = _sim(sim_env, date=None, seed=7)
    assert isinstance(bs, BoxScore)
    assert abs(sum(p.minutes for p in bs.home.players) - 240.0) < 1e-2
    assert abs(sum(p.minutes for p in bs.away.players) - 240.0) < 1e-2


def test_simulate_no_snapshot_raises_clear_error(sim_env: _SimEnv, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nba-sim refresh"):
        _sim(sim_env, date=None, snapshot_dir=tmp_path / "does_not_exist")


def test_simulate_v1_path_still_works_with_date(sim_env: _SimEnv) -> None:
    bs = _sim(sim_env, date=sim_env.real_date.isoformat(), seed=1)
    assert isinstance(bs, BoxScore)
    assert bs.date == sim_env.real_date


def test_simulate_force_snapshot_overrides_v1_lookup(sim_env: _SimEnv) -> None:
    # A real date is supplied, but force_snapshot uses the snapshot path, so the
    # box score is dated as_of rather than the requested date.
    bs = _sim(sim_env, date=sim_env.real_date.isoformat(), force_snapshot=True, seed=1)
    assert isinstance(bs, BoxScore)
    assert bs.date == sim_env.as_of
