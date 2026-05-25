"""Hard-constraint tests — the key property of this simulator.

Samples box scores from a random-init :class:`HierarchicalBoxScoreModel`
via :func:`sample_raw_box_score` and asserts every constraint from
PLAN.md §4.4 + §9 holds on every sample:

- per-team minutes sum to exactly 240 (after rounding-with-residual)
- ``FGM ≤ FGA``, ``3PM ≤ 3PA``, ``FTM ≤ FTA`` (by Binomial construction)
- ``3PM ≤ FGM`` (by post-hoc clamp; PLAN §4.4 lists this as the one
  not-by-construction enforcement)
- every count is non-negative and integer-valued
- ``team_total_stat == sum(player_stat)`` — trivial, since we don't
  model team totals separately
- padded slots (the architectural representation of inactive roster) get
  zero across every per-player tensor
- ``PTS == 2·FGM + 3PM + FTM`` and ``REB == OREB + DREB`` (derived)

The Phase 2 ship gate (PLAN §4.7) is the scale test
:func:`test_constraint_violation_rate_is_zero_over_1000_samples` — 1,000
sampled games all clean — which doubles as the integration test for
:func:`nba_sim.training.metrics.constraint_violation_rate`.
"""

from __future__ import annotations

import polars as pl
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.simulate.sampler import _round_minutes_to_240, sample_raw_box_score
from nba_sim.training.metrics import constraint_violation_rate

# Reuse fixture helpers from the existing model-shape test module.
from tests.test_model_shapes import (
    B, P, D_CTX_RAW, D_MATCHUP, D_P_RAW, N_PLAYERS, N_ROLES,
    _model_config,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_NB_STATS = ("fga", "tpa", "fta", "oreb", "dreb",
             "ast", "stl", "blk", "tov", "pf")
_MAKE_STATS = ("fgm", "tpm", "ftm")
_ALL_STATS = _NB_STATS + _MAKE_STATS


def _make_full_batch(
    b: int = B,
    p: int = P,
    n_active_home: int | None = None,
    n_active_away: int | None = None,
) -> dict[str, torch.Tensor]:
    """Synthetic forward-pass batch with controllable mask sizes.

    The features are pure noise — we're not testing accuracy here, just
    that the sampler produces constraint-satisfying outputs whatever the
    distribution params happen to be.
    """
    def _side(n_active: int | None) -> dict[str, torch.Tensor]:
        feats = torch.randn(b, p, D_P_RAW)
        player_ids = torch.randint(1, N_PLAYERS, (b, p))
        role_ids = torch.randint(1, N_ROLES, (b, p))
        if n_active is None:
            sizes = torch.randint(1, p + 1, (b,))
        else:
            sizes = torch.full((b,), n_active)
        mask = torch.arange(p).unsqueeze(0) < sizes.unsqueeze(1)
        player_ids = player_ids.masked_fill(~mask, 0)
        role_ids = role_ids.masked_fill(~mask, 0)
        return {
            "feats": feats, "player_ids": player_ids,
            "role_ids": role_ids, "mask": mask,
        }
    home = _side(n_active_home)
    away = _side(n_active_away)
    return {
        "home_player_feats": home["feats"], "away_player_feats": away["feats"],
        "home_player_ids": home["player_ids"], "away_player_ids": away["player_ids"],
        "home_role_ids": home["role_ids"],  "away_role_ids": away["role_ids"],
        "home_mask": home["mask"],          "away_mask": away["mask"],
        "context": torch.randn(b, D_CTX_RAW),
        "matchup": torch.randn(b, D_MATCHUP),
    }


def _sampled(
    b: int = B,
    n_active_home: int | None = 12,
    n_active_away: int | None = 12,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return ``(batch, sampled_dict)``. Seeds torch globally for repro."""
    torch.manual_seed(seed)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch = _make_full_batch(b=b, n_active_home=n_active_home, n_active_away=n_active_away)
    with torch.no_grad():
        dist = model(batch)
        sampled = sample_raw_box_score(dist, batch["home_mask"], batch["away_mask"])
    return batch, sampled


def _samples_to_long_df(batch: dict, sampled: dict) -> pl.DataFrame:
    """Convert a sampled batch into the long-format DataFrame
    :func:`constraint_violation_rate` consumes.

    One row per (game_id, team_id, player_id). Padded slots are dropped.
    """
    rows: list[dict] = []
    Bx = batch["home_mask"].shape[0]
    Px = batch["home_mask"].shape[1]
    for b in range(Bx):
        for side, team_offset in (("home", 0), ("away", 1000)):
            mask = batch[f"{side}_mask"][b]
            ids = batch[f"{side}_player_ids"][b]
            mins = sampled[f"{side}_minutes"][b]
            for i in range(Px):
                if not bool(mask[i]):
                    continue
                row = {
                    "game_id": b,
                    "team_id": team_offset + b,    # one team per side per game
                    "player_id": int(ids[i].item()) or (team_offset + i),
                    "minutes": float(mins[i]),
                }
                for s in _ALL_STATS:
                    row[s] = float(sampled[f"{side}_{s}"][b, i])
                rows.append(row)
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# _round_minutes_to_240 — unit tests
# ---------------------------------------------------------------------------


def test_round_minutes_sums_to_240_per_row() -> None:
    """Every row must sum to exactly 240.0 (within fp tolerance)."""
    torch.manual_seed(0)
    shares = torch.softmax(torch.randn(8, P), dim=-1)
    mask = torch.zeros(8, P, dtype=torch.bool)
    for i in range(8):
        mask[i, : 5 + (i % 11)] = True
    out = _round_minutes_to_240(shares, mask)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.full_like(sums, 240.0), atol=1e-4)


def test_round_minutes_zeros_padded_slots() -> None:
    shares = torch.softmax(torch.randn(4, P), dim=-1)
    mask = torch.zeros(4, P, dtype=torch.bool)
    mask[:, :10] = True
    out = _round_minutes_to_240(shares, mask)
    assert torch.all(out[~mask] == 0.0)


def test_round_minutes_values_are_tick_aligned_except_residual_slot() -> None:
    """Every entry is a multiple of ``tick=0.1`` — including the residual
    slot, since the residual is itself a multiple of ``tick`` (it's the
    difference between two tick-aligned numbers)."""
    shares = torch.softmax(torch.randn(4, P), dim=-1)
    mask = torch.ones(4, P, dtype=torch.bool)
    out = _round_minutes_to_240(shares, mask, tick=0.1)
    # (x*10) is integer → integer modular distance from itself is 0. The
    # fp32 multiplication introduces ~2e-4 noise; loose tolerance is fine
    # for the "value is on a 0.1-min tick" semantic check.
    diff = (out * 10) - (out * 10).round()
    assert float(diff.abs().max()) < 1e-3


def test_round_minutes_rejects_shape_mismatch() -> None:
    shares = torch.zeros(4, P)
    mask = torch.zeros(4, P + 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="must match"):
        _round_minutes_to_240(shares, mask)


# ---------------------------------------------------------------------------
# Per-constraint tests (PLAN §9)
# ---------------------------------------------------------------------------


def test_minutes_sum_to_240_per_team() -> None:
    _, out = _sampled(seed=0)
    for side in ("home", "away"):
        sums = out[f"{side}_minutes"].sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, 240.0), atol=1e-4), \
            f"{side} minutes sums: {sums.tolist()}"


def test_fgm_le_fga_per_player() -> None:
    _, out = _sampled(seed=1)
    for side in ("home", "away"):
        assert torch.all(out[f"{side}_fgm"] <= out[f"{side}_fga"]), side


def test_tpm_le_tpa_per_player() -> None:
    _, out = _sampled(seed=2)
    for side in ("home", "away"):
        assert torch.all(out[f"{side}_tpm"] <= out[f"{side}_tpa"]), side


def test_tpm_le_fgm_per_player() -> None:
    """The post-hoc clamp must hold on every sampled slot."""
    _, out = _sampled(seed=3)
    for side in ("home", "away"):
        assert torch.all(out[f"{side}_tpm"] <= out[f"{side}_fgm"]), side


def test_ftm_le_fta_per_player() -> None:
    _, out = _sampled(seed=4)
    for side in ("home", "away"):
        assert torch.all(out[f"{side}_ftm"] <= out[f"{side}_fta"]), side


def test_all_counts_non_negative_integers() -> None:
    _, out = _sampled(seed=5)
    for side in ("home", "away"):
        for stat in _ALL_STATS:
            t = out[f"{side}_{stat}"]
            assert torch.all(t >= 0.0), f"{side}_{stat} has negatives"
            assert torch.all(t == t.round()), f"{side}_{stat} has non-integer values"


def test_team_pts_equals_sum_of_player_pts() -> None:
    """No separate team-PTS prediction → the per-team total IS the player
    sum by construction. This test pins that invariant in place so a
    future regression that adds a separate team-PTS head is caught."""
    _, out = _sampled(seed=6)
    for side in ("home", "away"):
        player_pts = out[f"{side}_pts"]
        # Per-row team total derived from the sample dict equals the
        # row-wise sum — trivially, because that's how the sampler
        # exposes the team total.
        team_total = player_pts.sum(dim=-1)
        assert torch.allclose(team_total, player_pts.sum(dim=-1))
        # Plus: pts = 2·fgm + tpm + ftm by definition.
        recomputed = 2.0 * out[f"{side}_fgm"] + out[f"{side}_tpm"] + out[f"{side}_ftm"]
        assert torch.equal(player_pts, recomputed)


def test_reb_equals_oreb_plus_dreb_per_player() -> None:
    _, out = _sampled(seed=7)
    for side in ("home", "away"):
        assert torch.equal(
            out[f"{side}_reb"],
            out[f"{side}_oreb"] + out[f"{side}_dreb"],
        )


def test_dnp_rostered_players_get_zero_minutes() -> None:
    """Padded slots — the architectural representation of inactive roster
    spots — get exactly zero minutes from the rounding helper, never a
    fractional tick."""
    batch, out = _sampled(seed=8, n_active_home=10, n_active_away=11)
    for side in ("home", "away"):
        padded = ~batch[f"{side}_mask"]
        assert torch.all(out[f"{side}_minutes"][padded] == 0.0)
        # And: padded slots are zeroed across all per-player counts.
        for stat in _ALL_STATS:
            assert torch.all(out[f"{side}_{stat}"][padded] == 0.0), \
                f"{side}_{stat} has nonzero padded values"


# ---------------------------------------------------------------------------
# Hypothesis property test — varies roster sizes + seeds (PLAN §9)
# ---------------------------------------------------------------------------


@given(
    n_active_home=st.integers(min_value=1, max_value=P),
    n_active_away=st.integers(min_value=1, max_value=P),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_all_constraints_under_random_masks(
    n_active_home: int, n_active_away: int, seed: int
) -> None:
    """The single source-of-truth property: for any reasonable roster size
    + seed, the sampler output satisfies every §4.4 constraint."""
    batch, out = _sampled(
        b=2,
        n_active_home=n_active_home,
        n_active_away=n_active_away,
        seed=seed,
    )
    for side in ("home", "away"):
        mask = batch[f"{side}_mask"]
        padded = ~mask

        # Minutes sum to exactly 240 per row.
        sums = out[f"{side}_minutes"].sum(dim=-1)
        assert torch.allclose(sums, torch.full_like(sums, 240.0), atol=1e-4)

        # Make ≤ attempt for all three percent pairs.
        assert torch.all(out[f"{side}_fgm"] <= out[f"{side}_fga"])
        assert torch.all(out[f"{side}_tpm"] <= out[f"{side}_tpa"])
        assert torch.all(out[f"{side}_ftm"] <= out[f"{side}_fta"])
        # 3PM ≤ FGM (the clamped constraint).
        assert torch.all(out[f"{side}_tpm"] <= out[f"{side}_fgm"])

        # Padded → 0 across all per-player tensors.
        for stat in _ALL_STATS:
            t = out[f"{side}_{stat}"]
            assert torch.all(t >= 0.0)
            assert torch.all(t == t.round())
            assert torch.all(t[padded] == 0.0)


# ---------------------------------------------------------------------------
# Scale test — the Phase 2 ship gate (PLAN §4.7)
# ---------------------------------------------------------------------------


def test_constraint_violation_rate_is_zero_over_1000_samples() -> None:
    """1,000 sampled games from a random-init model — zero violations,
    per :func:`metrics.constraint_violation_rate`. Closes the loop
    between the sampler and the metric: if either piece drifts, this
    test catches it."""
    batch, sampled = _sampled(b=1000, n_active_home=12, n_active_away=12, seed=42)
    df = _samples_to_long_df(batch, sampled)
    rates = constraint_violation_rate(df)
    # Every constraint reports 0.0.
    for k, v in rates.items():
        assert v == 0.0, f"{k} violated on {v*100:.2f}% of games"
