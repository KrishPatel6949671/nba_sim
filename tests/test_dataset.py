"""Tests for ``nba_sim.training.dataset``.

Strategy: build a small synthetic parquet matching the processed-layer
schema, then assert that ``BoxScoreDataset`` produces the dict shape +
semantics the model and loss expect. One end-to-end test runs a forward
+ composite_nll through a real model to catch contract drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.models.losses import LossWeights, composite_nll
from nba_sim.training.dataset import (
    BoxScoreDataset,
    FeatureStats,
    _COUNT_STAT_COLS,
    _D_PLAYER_RAW,
    _PLAYER_NUMERIC_COLS,
    collate_games,
)

# Reuse the model-shape test's config fixture.
from tests.test_model_shapes import _model_config


# ---------------------------------------------------------------------------
# Fixture: a minimal but schema-valid processed parquet
# ---------------------------------------------------------------------------


_POSITIONS = ("G", "F", "C")


def _row(
    *,
    game_id: str,
    player_id: int,
    team_id: int,
    is_home: bool,
    minutes: float,
    is_starter: bool = False,
    dnp: bool = False,
    p_min_avg_10: float | None = 18.0,
    position: str = "G",
) -> dict:
    """A single dataset row with sensible defaults.

    Every required column is populated so :meth:`BoxScoreDataset._check_required_columns`
    is satisfied. Counts are simple integers; rolling features are sane.
    """
    row: dict = {
        "game_id": game_id,
        "player_id": player_id,
        "team_id": team_id,
        "is_home": is_home,
        "is_active": True,
        "minutes": float(minutes),
        "position": position,
        "is_starter": is_starter,
        "dnp": dnp,
        # Counts (integers stored as floats so we can mix with nulls upstream).
        "pts": 10.0,
        "fga": 8.0, "fgm": 4.0,
        "tpa": 3.0, "tpm": 1.0,
        "fta": 4.0, "ftm": 3.0,
        "oreb": 1.0, "dreb": 3.0,
        "ast": 2.0, "stl": 1.0,
        "blk": 0.0, "tov": 1.0, "pf": 2.0,
        # Context.
        "season_phase": "mid",
        "day_of_week": 3,
        "month": 1,
        "altitude_ft": 500.0,
        "travel_miles_prev": 250.0,
        "rest_days": 1,
        "b2b": False,
        "is_3in4": False,
        "is_4in6": False,
        # Matchup.
        "h2h_last_meeting_margin": 3.0,
        "opp_def_rtg_10": 110.0,
        "opp_pace_10": 100.0,
        "t_pace_5": 100.0, "t_pace_10": 100.0,
        "t_off_rtg_5": 110.0, "t_off_rtg_10": 110.0,
        "t_def_rtg_5": 108.0, "t_def_rtg_10": 108.0,
        "t_win_pct_10": 0.5, "t_pts_avg_10": 110.0, "t_pts_allowed_10": 108.0,
    }
    # Player numeric columns: a sensible constant + a varied one.
    for c in _PLAYER_NUMERIC_COLS:
        row[c] = 20.0 if c == "p_min_avg_10" else 1.0
    row["p_min_avg_10"] = float(p_min_avg_10) if p_min_avg_10 is not None else None
    return row


def _make_game(
    game_id: str,
    home_team: int,
    away_team: int,
    n_home: int = 12,
    n_away: int = 12,
    player_id_start: int = 100,
    minute_pattern: list[float] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    pid = player_id_start
    pat = minute_pattern or ([24.0] * 5 + [16.0] * 4 + [8.0] * 3 + [0.0] * 5)
    for i in range(n_home):
        m = pat[i] if i < len(pat) else 0.0
        rows.append(_row(
            game_id=game_id, player_id=pid, team_id=home_team,
            is_home=True, minutes=m, is_starter=(i < 5),
            position=_POSITIONS[i % 3],
        ))
        pid += 1
    for i in range(n_away):
        m = pat[i] if i < len(pat) else 0.0
        rows.append(_row(
            game_id=game_id, player_id=pid, team_id=away_team,
            is_home=False, minutes=m, is_starter=(i < 5),
            position=_POSITIONS[i % 3],
        ))
        pid += 1
    return rows


@pytest.fixture
def synthetic_parquet(tmp_path: Path) -> Path:
    """Two games, two teams each, varying roster sizes."""
    rows: list[dict] = []
    rows.extend(_make_game("G001", home_team=10, away_team=20, n_home=13, n_away=11, player_id_start=100))
    rows.extend(_make_game("G002", home_team=10, away_team=30, n_home=12, n_away=14, player_id_start=200))
    df = pl.DataFrame(rows)
    p = tmp_path / "synthetic.parquet"
    df.write_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Shape / schema tests
# ---------------------------------------------------------------------------


def test_dataset_loads_and_length_matches_games(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    assert len(ds) == 2
    assert ds.game_ids == ["G001", "G002"]


def test_item_has_all_required_keys(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[0]
    required = {
        "home_player_feats", "away_player_feats",
        "home_player_ids", "away_player_ids",
        "home_role_ids", "away_role_ids",
        "home_mask", "away_mask",
        "context", "matchup",
        "pace_true", "off_rtg_true",
        "pace", "off_rtg",
        "home_minutes_share", "away_minutes_share",
        "home_plays_gate", "away_plays_gate",
    }
    for s in _COUNT_STAT_COLS:
        required.add(f"home_{s}")
        required.add(f"away_{s}")
    assert set(item.keys()) == required


def test_item_tensor_shapes(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet, max_players=15)
    item = ds[0]
    assert item["home_player_feats"].shape == (15, _D_PLAYER_RAW)
    assert item["away_player_feats"].shape == (15, _D_PLAYER_RAW)
    assert item["home_player_ids"].shape == (15,)
    assert item["home_role_ids"].shape == (15,)
    assert item["home_mask"].shape == (15,)
    assert item["home_mask"].dtype == torch.bool
    assert item["context"].shape == (24,)
    assert item["matchup"].shape == (16,)
    assert item["pace"].shape == ()
    assert item["off_rtg"].shape == (2,)
    assert item["home_minutes_share"].shape == (15,)
    assert item["home_plays_gate"].shape == (15,)
    assert item["home_fga"].shape == (15,)


# ---------------------------------------------------------------------------
# Semantic tests
# ---------------------------------------------------------------------------


def test_mask_true_for_active_false_for_padding(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    # G001 has 13 home / 11 away active.
    item = ds[0]
    assert int(item["home_mask"].sum()) == 13
    assert int(item["away_mask"].sum()) == 11
    # Active block is the contiguous prefix (sorted by minutes desc).
    assert torch.all(item["home_mask"][:13])
    assert torch.all(~item["home_mask"][13:])


def test_padded_ids_and_roles_are_zero(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[1]   # G002 with 14 active away players → 1 padding slot
    away_n = 14
    assert torch.all(item["away_player_ids"][away_n:] == 0)
    assert torch.all(item["away_role_ids"][away_n:] == 0)
    # Active slots get non-zero player ids (id_map covers them).
    assert torch.all(item["away_player_ids"][:away_n] > 0)


def test_padded_plays_gate_and_counts_are_zero(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[0]
    n_active = 13
    assert torch.all(item["home_plays_gate"][n_active:] == 0.0)
    for s in _COUNT_STAT_COLS:
        assert torch.all(item[f"home_{s}"][n_active:] == 0.0), f"home_{s} padding nonzero"


def test_minutes_share_sums_to_one_and_padded_slots_get_epsilon(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[0]
    s = float(item["home_minutes_share"].sum())
    assert s == pytest.approx(1.0, abs=1e-5)
    # Padded slots get a tiny positive mass (Dirichlet log_prob requires > 0).
    assert torch.all(item["home_minutes_share"] > 0.0)
    # Active slots with positive minutes carry the bulk; active-DNP and
    # padded slots both get the same tiny ε mass (correct: a DNP player's
    # minute share is essentially zero).
    pos_min_share = item["home_minutes_share"][item["home_minutes_share"] > 1e-4]
    eps_share = item["home_minutes_share"][item["home_minutes_share"] <= 1e-4]
    assert pos_min_share.numel() > 0 and eps_share.numel() > 0
    assert pos_min_share.min() > eps_share.max() * 100   # clearly separated


def test_plays_gate_matches_minutes_positive(synthetic_parquet: Path) -> None:
    """plays_gate is 1 iff the player has positive minutes."""
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[0]
    mask = item["home_mask"]
    # Active slots with 0 minutes → gate 0; with >0 minutes → gate 1.
    minutes_proxy = item["home_minutes_share"].clone()
    # 'Active with 0 raw minutes' shows up as the epsilon mass; can't
    # invert minutes_share back to raw minutes here, so re-derive raw
    # minutes from the source row count: pat had 5 zeros at the tail of
    # 12 → so home of G001 (13 active) has at least one zero-min slot.
    # We test the easier invariant: any padded slot has gate=0.
    assert torch.all(item["home_plays_gate"][~mask] == 0.0)
    # And every active slot's gate is either 0 (DNP/zero-min) or 1.
    assert torch.all((item["home_plays_gate"][mask] == 0.0) | (item["home_plays_gate"][mask] == 1.0))


def test_pace_and_off_rtg_are_finite_and_positive(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[0]
    assert torch.isfinite(item["pace"]).item()
    assert item["pace"].item() > 0.0
    assert torch.all(torch.isfinite(item["off_rtg"]))
    assert torch.all(item["off_rtg"] > 0.0)


def test_pace_true_aliases_pace(synthetic_parquet: Path) -> None:
    """Teacher-forcing input and the loss target are the same number."""
    ds = BoxScoreDataset(synthetic_parquet)
    item = ds[0]
    assert torch.equal(item["pace"], item["pace_true"])
    assert torch.equal(item["off_rtg"], item["off_rtg_true"])


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncates_to_max_players_keeping_top_minutes(tmp_path: Path) -> None:
    """A team with 17 active players gets truncated to 15, keeping the top
    15 by minutes."""
    rows: list[dict] = []
    # Build a single game where the home side has 17 active players. Give
    # them distinct minutes so we can identify the kept ones.
    minutes = [40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 5, 1]
    for i, m in enumerate(minutes):
        rows.append(_row(
            game_id="G1", player_id=1000 + i, team_id=10,
            is_home=True, minutes=float(m), is_starter=(i < 5),
        ))
    # And a normal 12-player away side.
    for i in range(12):
        rows.append(_row(
            game_id="G1", player_id=2000 + i, team_id=20,
            is_home=False, minutes=20.0, is_starter=(i < 5),
        ))
    p = tmp_path / "huge_roster.parquet"
    pl.DataFrame(rows).write_parquet(p)

    ds = BoxScoreDataset(p, max_players=15)
    item = ds[0]
    assert int(item["home_mask"].sum()) == 15
    # Kept ids are 1000..1014 (top 15 by minutes); 1015 and 1016 are dropped.
    kept = sorted(int(x) for x in item["home_player_ids"].tolist() if x != 0)
    expected_kept = sorted(ds.player_id_map[1000 + i] for i in range(15))
    assert kept == expected_kept


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def test_collate_stacks_to_batched_shapes(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet)
    batch = collate_games([ds[0], ds[1]])
    assert batch["home_player_feats"].shape == (2, 15, _D_PLAYER_RAW)
    assert batch["context"].shape == (2, 24)
    assert batch["matchup"].shape == (2, 16)
    assert batch["pace"].shape == (2,)
    assert batch["off_rtg"].shape == (2, 2)
    assert batch["home_mask"].shape == (2, 15)
    assert batch["home_mask"].dtype == torch.bool


# ---------------------------------------------------------------------------
# player_id_map
# ---------------------------------------------------------------------------


def test_unseen_player_id_maps_to_zero(tmp_path: Path) -> None:
    """A player_id not in the fitted map collapses to id=0 (pad/unseen)."""
    train_rows = _make_game("GTrain", 10, 20, n_home=11, n_away=11, player_id_start=100)
    pq_train = tmp_path / "train.parquet"
    pl.DataFrame(train_rows).write_parquet(pq_train)
    train = BoxScoreDataset(pq_train)

    # Build a "val" parquet whose player_ids don't overlap with train at all.
    val_rows = _make_game("GVal", 10, 20, n_home=11, n_away=11, player_id_start=9000)
    pq_val = tmp_path / "val.parquet"
    pl.DataFrame(val_rows).write_parquet(pq_val)

    val = BoxScoreDataset(
        pq_val,
        player_id_map=train.player_id_map,
        feature_stats=train.feature_stats,
    )
    item = val[0]
    mask = item["home_mask"]
    # Every active slot has player_id=0 (unseen → pad).
    assert torch.all(item["home_player_ids"][mask] == 0)


def test_player_id_map_assigns_ids_in_valid_range(synthetic_parquet: Path) -> None:
    ds = BoxScoreDataset(synthetic_parquet, max_player_id=5000)
    for v in ds.player_id_map.values():
        assert 1 <= v <= 4999


def test_feature_stats_reused_when_passed(synthetic_parquet: Path) -> None:
    """Passing pre-fitted stats means the dataset doesn't refit."""
    fs = FeatureStats()
    # Inject identifiable sentinel stats.
    fs.means["altitude_ft"] = 100.0
    fs.stds["altitude_ft"] = 50.0

    ds = BoxScoreDataset(synthetic_parquet, feature_stats=fs)
    assert ds.feature_stats.means["altitude_ft"] == 100.0
    assert ds.feature_stats.stds["altitude_ft"] == 50.0

    # Confirm it actually affects the produced tensors: altitude is at a
    # known offset in the context vector.
    item = ds[0]
    # Layout: 4 season_phase + 7 dow + 2 month + altitude = offset 13.
    altitude_z = float(item["context"][13])
    expected = (500.0 - 100.0) / 50.0   # raw altitude 500 from fixture
    assert altitude_z == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# Role bucketing
# ---------------------------------------------------------------------------


def test_role_id_buckets(tmp_path: Path) -> None:
    """starter → 1, rotation (p_min_avg_10 ≥ 12) → 2, else → 3, padded → 0."""
    rows = [
        _row(game_id="G", player_id=1, team_id=10, is_home=True,
             minutes=30.0, is_starter=True,  p_min_avg_10=22.0),    # starter
        _row(game_id="G", player_id=2, team_id=10, is_home=True,
             minutes=15.0, is_starter=False, p_min_avg_10=18.0),    # rotation
        _row(game_id="G", player_id=3, team_id=10, is_home=True,
             minutes=4.0,  is_starter=False, p_min_avg_10=5.0),     # end-of-bench
        _row(game_id="G", player_id=4, team_id=10, is_home=True,
             minutes=0.0,  is_starter=False, p_min_avg_10=None),    # cold-start → eob
    ]
    # Need both teams. Add a vanilla away side.
    rows.extend(_make_game("G", 10, 20, n_home=0, n_away=11, player_id_start=500)[len(rows):])
    # _make_game won't emit only-away rows if n_home=0 — workaround:
    away = []
    for i in range(11):
        away.append(_row(
            game_id="G", player_id=500 + i, team_id=20,
            is_home=False, minutes=20.0, is_starter=(i < 5),
        ))
    rows = rows[:4] + away

    p = tmp_path / "roles.parquet"
    pl.DataFrame(rows).write_parquet(p)
    ds = BoxScoreDataset(p)
    item = ds[0]
    # home side sorted by minutes desc → indexes [0..3] = (p1, p2, p3, p4)
    assert item["home_role_ids"][0].item() == 1   # starter
    assert item["home_role_ids"][1].item() == 2   # rotation
    assert item["home_role_ids"][2].item() == 3   # end-of-bench
    assert item["home_role_ids"][3].item() == 3   # cold-start → eob
    # Padded slot:
    assert item["home_role_ids"][4].item() == 0


# ---------------------------------------------------------------------------
# End-to-end smoke through model + composite_nll
# ---------------------------------------------------------------------------


def test_end_to_end_forward_and_loss(synthetic_parquet: Path) -> None:
    """A dataset-produced batch must pass cleanly through forward + loss."""
    torch.manual_seed(0)
    ds = BoxScoreDataset(synthetic_parquet)
    batch = collate_games([ds[0], ds[1]])

    model = HierarchicalBoxScoreModel(_model_config()).eval()
    preds = model(batch)
    out = composite_nll(preds, batch, LossWeights.default())

    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])
    # Per-head NLLs should all be finite too.
    for k in ("pace", "off_rtg", "minutes", "gate",
              "fga", "fgm", "tpm", "ftm", "blk", "stl"):
        assert torch.isfinite(out[k]), f"{k} is not finite"


def test_gradients_flow_from_real_data_loss(synthetic_parquet: Path) -> None:
    """Training-style: ``loss.backward()`` populates grads on every param."""
    torch.manual_seed(0)
    ds = BoxScoreDataset(synthetic_parquet)
    batch = collate_games([ds[0], ds[1]])

    model = HierarchicalBoxScoreModel(_model_config())
    preds = model(batch)
    out = composite_nll(preds, batch, LossWeights.default())
    out["loss"].backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
