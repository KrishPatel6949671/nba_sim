"""Tests for ``nba_sim.models.losses.composite_nll``.

Constructs a small synthetic batch with a known structure (mask, valid
Dirichlet targets, sensible counts) and verifies:

- All expected per-head keys are produced.
- Loss equals the weighted sum of per-head NLLs.
- Masking is honoured: changing values *only* in padded slots leaves
  every player-head NLL unchanged.
- Per-stat weight zeroing zeros that head's contribution to ``loss``.
- Gradients flow to every model parameter through ``loss.backward()``.
- Embedding shrinkage is included iff ``embedding_deltas`` is passed.
"""

from __future__ import annotations

import torch

from nba_sim.models.heads import BoxScoreDistribution
from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.models.losses import LossWeights, composite_nll

# Reuse the model-shape test's fixture helpers to avoid duplication.
from tests.test_model_shapes import (
    B,
    D_CTX_RAW,
    D_MATCHUP,
    D_P_RAW,
    N_PLAYERS,
    N_ROLES,
    P,
    _model_config,
)


def _make_batch_with_targets() -> tuple[dict, dict]:
    """Return (batch, targets) with internally consistent values.

    ``home_mask`` / ``away_mask`` are True for the first ``n_active``
    slots per batch row. Counts are non-negative; ``fgm <= fga`` etc.
    by construction. Dirichlet target rows sum to 1.
    """
    n_active = 10
    sizes = torch.full((B,), n_active)
    mask = torch.arange(P).unsqueeze(0) < sizes.unsqueeze(1)        # [B, P] bool

    def _ids(): return torch.randint(1, N_PLAYERS, (B, P)).masked_fill(~mask, 0)
    def _roles(): return torch.randint(1, N_ROLES, (B, P)).masked_fill(~mask, 0)
    def _feats(): return torch.randn(B, P, D_P_RAW)

    batch = {
        "home_player_feats": _feats(), "away_player_feats": _feats(),
        "home_player_ids": _ids(),     "away_player_ids": _ids(),
        "home_role_ids": _roles(),     "away_role_ids": _roles(),
        "home_mask": mask.clone(),     "away_mask": mask.clone(),
        "context": torch.randn(B, D_CTX_RAW),
        "matchup": torch.randn(B, D_MATCHUP),
        "pace_true": torch.full((B,), 100.0),
        "off_rtg_true": torch.full((B, 2), 110.0),
    }

    # Minute shares: each active slot gets ~25 min / 240, padded slots get ε,
    # then renormalize each row to sum to 1.
    minutes = torch.full((B, P), 1e-3)
    minutes[mask] = 24.0
    minutes = minutes / minutes.sum(dim=-1, keepdim=True)

    # Plausible per-player counts. We use the same numbers for home/away.
    fga = torch.full((B, P), 8.0); fgm = torch.full((B, P), 4.0)
    tpa = torch.full((B, P), 3.0); tpm = torch.full((B, P), 1.0)
    fta = torch.full((B, P), 4.0); ftm = torch.full((B, P), 3.0)
    oreb = torch.full((B, P), 1.0); dreb = torch.full((B, P), 3.0)
    ast = torch.full((B, P), 2.0); stl = torch.full((B, P), 1.0)
    blk = torch.full((B, P), 0.0); tov = torch.full((B, P), 1.0)
    pf = torch.full((B, P), 2.0)

    plays_gate = mask.float()

    targets = {
        "pace": batch["pace_true"],
        "off_rtg": batch["off_rtg_true"],
        "home_mask": mask, "away_mask": mask,
        "home_minutes_share": minutes, "away_minutes_share": minutes,
        "home_plays_gate": plays_gate, "away_plays_gate": plays_gate,
    }
    for side in ("home", "away"):
        targets[f"{side}_fga"] = fga; targets[f"{side}_fgm"] = fgm
        targets[f"{side}_tpa"] = tpa; targets[f"{side}_tpm"] = tpm
        targets[f"{side}_fta"] = fta; targets[f"{side}_ftm"] = ftm
        targets[f"{side}_oreb"] = oreb; targets[f"{side}_dreb"] = dreb
        targets[f"{side}_ast"] = ast; targets[f"{side}_stl"] = stl
        targets[f"{side}_blk"] = blk; targets[f"{side}_tov"] = tov
        targets[f"{side}_pf"] = pf
    return batch, targets


def _predict(model: HierarchicalBoxScoreModel, batch: dict) -> BoxScoreDistribution:
    return model(batch)


def test_composite_nll_produces_all_per_head_keys() -> None:
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch, targets = _make_batch_with_targets()
    preds = _predict(model, batch)
    out = composite_nll(preds, targets, LossWeights.default())

    expected = {
        "loss", "pace", "off_rtg", "minutes", "gate",
        "fga", "tpa", "fta", "fgm", "tpm", "ftm",
        "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
        "embedding_pool",
    }
    assert set(out.keys()) == expected
    for k in expected:
        assert out[k].dim() == 0, f"{k} should be a scalar, got shape {out[k].shape}"
        assert torch.isfinite(out[k]), f"{k} is not finite: {out[k]}"


def test_composite_loss_equals_weighted_sum_of_heads() -> None:
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch, targets = _make_batch_with_targets()
    preds = _predict(model, batch)
    w = LossWeights.default()
    out = composite_nll(preds, targets, w)

    expected_loss = (
        w.pace * out["pace"]
        + w.off_rtg * out["off_rtg"]
        + w.minutes * out["minutes"]
        + w.gate * out["gate"]
        + w.embedding_pool * out["embedding_pool"]
    )
    for stat in ("fga", "tpa", "fta", "fgm", "tpm", "ftm",
                 "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"):
        expected_loss = expected_loss + getattr(w, stat) * out[stat]

    assert torch.allclose(out["loss"], expected_loss, atol=1e-6)


def test_padded_slot_values_do_not_affect_player_head_nll() -> None:
    """The masked-mean reduction must ignore padded slot contributions —
    arbitrary garbage in padded slots leaves every per-player NLL unchanged."""
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch, targets = _make_batch_with_targets()
    preds = _predict(model, batch)
    w = LossWeights.default()
    out_clean = composite_nll(preds, targets, w)

    # Inject extreme values into padded slots of every per-player target.
    polluted = {k: v.clone() if torch.is_tensor(v) else v for k, v in targets.items()}
    for side in ("home", "away"):
        bad_mask = ~polluted[f"{side}_mask"]
        for stat in ("fga", "tpa", "fta", "fgm", "tpm", "ftm",
                     "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"):
            t = polluted[f"{side}_{stat}"]
            t[bad_mask] = 999.0
        polluted[f"{side}_plays_gate"][bad_mask] = 1.0
    out_polluted = composite_nll(preds, targets=polluted, weights=w)

    for head in ("gate", "fga", "tpa", "fta", "fgm", "tpm", "ftm",
                 "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"):
        assert torch.allclose(out_clean[head], out_polluted[head], atol=1e-6), \
            f"{head} changed when padded slots changed: {out_clean[head]} vs {out_polluted[head]}"


def test_zeroing_a_weight_zeros_its_loss_contribution() -> None:
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch, targets = _make_batch_with_targets()
    preds = _predict(model, batch)

    base = composite_nll(preds, targets, LossWeights.default())
    no_pace = LossWeights.default()
    no_pace.pace = 0.0
    zeroed = composite_nll(preds, targets, no_pace)

    expected_delta = base["pace"]   # weight was 1.0 → contribution was per["pace"]
    assert torch.allclose(base["loss"] - zeroed["loss"], expected_delta, atol=1e-6)


def test_embedding_pool_is_zero_when_no_deltas_passed() -> None:
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch, targets = _make_batch_with_targets()
    preds = _predict(model, batch)
    out = composite_nll(preds, targets, LossWeights.default(), embedding_deltas=None)
    assert out["embedding_pool"].item() == 0.0


def test_embedding_pool_added_when_deltas_passed() -> None:
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch, targets = _make_batch_with_targets()
    preds = _predict(model, batch)
    w = LossWeights.default()

    deltas = torch.randn(50, 32)
    out = composite_nll(preds, targets, w, embedding_deltas=deltas)
    expected = deltas.pow(2).sum(dim=-1).mean()
    assert torch.allclose(out["embedding_pool"], expected, atol=1e-6)


def test_gradients_flow_through_composite_loss() -> None:
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config())
    batch, targets = _make_batch_with_targets()
    preds = model(batch)
    out = composite_nll(preds, targets, LossWeights.default())
    out["loss"].backward()

    for name, p in model.named_parameters():
        if name in {
            "player_encoder.player_delta.weight",
            "player_encoder.role_embed.weight",
        }:
            assert p.grad is not None, f"{name} grad is None"
            assert torch.any(p.grad[1:] != 0), f"{name} non-padding grad is all zero"
        else:
            assert p.grad is not None, f"{name} grad is None"
