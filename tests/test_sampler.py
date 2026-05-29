"""Tests for the schema-wrapping side of ``nba_sim.simulate.sampler``.

The hard-constraint contract on the raw tensor sampler lives in
``test_constraints.py``. This file covers the Pydantic-wrapping layer:

- ``sample_box_score`` returns a schema-valid :class:`BoxScore`.
- Same seed on the same device → identical output (determinism).
- Different seeds produce non-identical output (non-degenerate randomness).
- ``sample_ensemble`` returns N coherent samples plus
  mean / p10 / p90 summaries; the summaries pass the schema.
- Aggregates obey ``p10 ≤ mean ≤ p90`` per per-player stat (sanity).
- Calling ``sample_box_score`` does not perturb the caller's global RNG
  state (we use ``torch.random.fork_rng`` internally).
"""

from __future__ import annotations

import torch

from nba_sim.data.schema import BoxScore, BoxScoreEnsemble
from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.simulate.sampler import sample_box_score, sample_ensemble
from tests.test_constraints import _make_full_batch
from tests.test_model_shapes import _model_config


def _make_dist_for_one_game(n_home: int, n_away: int):
    """Forward a single-game batch through a random-init model and
    return the resulting BoxScoreDistribution. Cheap — features are
    noise; the returned dist is fine for sampling tests."""
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch = _make_full_batch(b=1, n_active_home=n_home, n_active_away=n_away)
    with torch.no_grad():
        dist = model(batch)
    return dist


def _players(n: int, *, id_start: int) -> list[tuple[int, str]]:
    return [(id_start + i, f"Player {id_start + i}") for i in range(n)]


# ---------------------------------------------------------------------------
# sample_box_score
# ---------------------------------------------------------------------------


def test_sample_box_score_returns_valid_boxscore() -> None:
    """Returned object is a BoxScore with the right roster sizes; Pydantic
    validation passes (counts ≥ 0, makes ≤ attempts via the raw sampler)."""
    n_home, n_away = 12, 11
    dist = _make_dist_for_one_game(n_home, n_away)
    bs = sample_box_score(
        dist,
        home_players=_players(n_home, id_start=100),
        away_players=_players(n_away, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
        seed=42,
    )
    assert isinstance(bs, BoxScore)
    assert bs.home.team == "BOS"
    assert bs.away.team == "LAL"
    assert len(bs.home.players) == n_home
    assert len(bs.away.players) == n_away
    # team_pts = sum(player pts) by construction (PLAN §4.4).
    assert bs.home.pts == sum(p.pts for p in bs.home.players)
    assert bs.away.pts == sum(p.pts for p in bs.away.players)
    # def_rtg is opp's off_rtg.
    assert bs.home.def_rtg == bs.away.off_rtg
    assert bs.away.def_rtg == bs.home.off_rtg
    # Minutes sum to 240 per side (sampler invariant, within float32 noise).
    assert abs(sum(p.minutes for p in bs.home.players) - 240.0) < 1e-2
    assert abs(sum(p.minutes for p in bs.away.players) - 240.0) < 1e-2


def test_sample_box_score_determinism_with_seed() -> None:
    """Same seed on the same dist → bit-identical sample. Twice in a row
    must match cell-for-cell across home + away."""
    n_home, n_away = 10, 12
    dist = _make_dist_for_one_game(n_home, n_away)
    kwargs = dict(
        dist=dist,
        home_players=_players(n_home, id_start=100),
        away_players=_players(n_away, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
    )
    a = sample_box_score(**kwargs, seed=7)
    b = sample_box_score(**kwargs, seed=7)
    for side in ("home", "away"):
        ta, tb = getattr(a, side), getattr(b, side)
        assert ta.pts == tb.pts
        assert ta.pace == tb.pace
        for i, (pa, pb) in enumerate(zip(ta.players, tb.players)):
            for stat in ("pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
                         "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf",
                         "minutes"):
                assert getattr(pa, stat) == getattr(pb, stat), \
                    f"{side} player[{i}].{stat}: {getattr(pa, stat)} != {getattr(pb, stat)}"


def test_sample_box_score_different_seeds_diverge() -> None:
    """Two different seeds should produce visibly different samples on at
    least one stat — the model output is far from degenerate."""
    n_home, n_away = 12, 12
    dist = _make_dist_for_one_game(n_home, n_away)
    kwargs = dict(
        dist=dist,
        home_players=_players(n_home, id_start=100),
        away_players=_players(n_away, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
    )
    a = sample_box_score(**kwargs, seed=1)
    b = sample_box_score(**kwargs, seed=2)
    assert (a.home.pts, a.away.pts) != (b.home.pts, b.away.pts), \
        "two seeds happened to produce identical team totals — vanishingly unlikely"


def test_sample_box_score_does_not_disturb_global_rng() -> None:
    """``fork_rng`` must isolate the sample call from the caller's RNG."""
    dist = _make_dist_for_one_game(n_home=5, n_away=5)
    torch.manual_seed(123)
    pre = torch.rand(4)
    torch.manual_seed(123)
    _ = sample_box_score(
        dist,
        home_players=_players(5, id_start=100),
        away_players=_players(5, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
        seed=999,
    )
    post = torch.rand(4)
    assert torch.equal(pre, post), \
        "global RNG state moved after seeded sample_box_score call"


def test_sample_box_score_rejects_oversized_roster() -> None:
    dist = _make_dist_for_one_game(n_home=15, n_away=15)
    too_many = _players(16, id_start=100)
    okay = _players(15, id_start=200)
    try:
        sample_box_score(
            dist,
            home_players=too_many, away_players=okay,
            home_team="BOS", away_team="LAL", date_iso="2025-02-14",
        )
    except ValueError as e:
        assert "16" in str(e) or "too many" in str(e)
    else:
        raise AssertionError("expected ValueError for oversized roster")


# ---------------------------------------------------------------------------
# sample_ensemble
# ---------------------------------------------------------------------------


def test_sample_ensemble_returns_n_samples_and_summaries() -> None:
    n_home, n_away = 10, 11
    dist = _make_dist_for_one_game(n_home, n_away)
    ens = sample_ensemble(
        dist, n=8,
        home_players=_players(n_home, id_start=100),
        away_players=_players(n_away, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
        seed=42,
    )
    assert isinstance(ens, BoxScoreEnsemble)
    assert len(ens.samples) == 8
    assert all(isinstance(s, BoxScore) for s in ens.samples)
    # Summary BoxScores have the same roster sizes as the underlying samples.
    assert len(ens.mean.home.players) == n_home
    assert len(ens.interval_low.home.players) == n_home
    assert len(ens.interval_high.home.players) == n_home


def test_sample_ensemble_determinism_with_master_seed() -> None:
    """Same master seed → ensemble's individual samples are identical
    one-for-one across two calls."""
    n = 5
    dist = _make_dist_for_one_game(n_home=10, n_away=10)
    kwargs = dict(
        dist=dist,
        home_players=_players(10, id_start=100),
        away_players=_players(10, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
    )
    a = sample_ensemble(n=n, **kwargs, seed=7).samples
    b = sample_ensemble(n=n, **kwargs, seed=7).samples
    for i, (sa, sb) in enumerate(zip(a, b)):
        assert sa.home.pts == sb.home.pts, f"sample {i} home pts diverged"
        assert sa.away.pts == sb.away.pts, f"sample {i} away pts diverged"


def test_sample_ensemble_samples_are_not_all_identical() -> None:
    """A master seed must produce DIFFERENT per-sample seeds; the samples
    should not all collapse to the same draw."""
    n_home, n_away = 10, 10
    dist = _make_dist_for_one_game(n_home, n_away)
    ens = sample_ensemble(
        dist, n=10,
        home_players=_players(n_home, id_start=100),
        away_players=_players(n_away, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
        seed=42,
    )
    pts_totals = {s.home.pts for s in ens.samples}
    assert len(pts_totals) > 1, "ensemble collapsed — all samples gave the same home pts"


def test_sample_ensemble_p10_le_mean_le_p90_per_player_stat() -> None:
    """Per-player counting stats: aggregate must satisfy p10 ≤ mean ≤ p90.
    A monotonicity sanity check that catches reversed quantile bugs."""
    n_home, n_away = 12, 12
    dist = _make_dist_for_one_game(n_home, n_away)
    ens = sample_ensemble(
        dist, n=20,
        home_players=_players(n_home, id_start=100),
        away_players=_players(n_away, id_start=200),
        home_team="BOS", away_team="LAL", date_iso="2025-02-14",
        seed=123,
    )
    stats = ("pts", "fga", "ast", "reb", "tov")
    for side in ("home", "away"):
        for i in range(n_home if side == "home" else n_away):
            pl = getattr(ens.interval_low, side).players[i]
            pm = getattr(ens.mean, side).players[i]
            ph = getattr(ens.interval_high, side).players[i]
            for stat in stats:
                lo, me, hi = getattr(pl, stat), getattr(pm, stat), getattr(ph, stat)
                assert lo <= me <= hi, \
                    f"{side} player[{i}] {stat}: p10={lo} mean={me} p90={hi}"
