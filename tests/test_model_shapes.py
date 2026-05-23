"""Forward-pass shape tests for every module in :mod:`nba_sim.models`.

These tests exercise a random-init model on a small synthetic batch
and assert tensor shapes match PLAN.md §4.2. A separate test ensures
gradients flow to every parameter.
"""

from __future__ import annotations

import pytest
import torch

from torch.distributions import (
    Bernoulli,
    Binomial,
    Dirichlet,
    NegativeBinomial,
    Normal,
)

from nba_sim.models.encoders import (
    GameContextEncoder,
    PlayerEncoder,
    RosterAttentionPool,
)
from nba_sim.models.heads import (
    BoxScoreDistribution,
    DirichletMinutes,
    PlayerAllocHead,
    TeamHead,
)


# Match configs/model.yaml defaults so tests double as a sanity check on
# the published config.
B = 4
P = 15
D_P_RAW = 48
D_PLAYER_EMBED = 32
D_ROLE_EMBED = 8
D_OUT = 64
D_TEAM = 128
D_CTX_RAW = 24
D_CTX = 32
N_PLAYERS = 5000
N_ROLES = 4


def _make_batch(b: int = B, p: int = P, n_active: int | None = None) -> dict[str, torch.Tensor]:
    """Build a synthetic batch. ``n_active`` lets the test fix the mask."""
    feats = torch.randn(b, p, D_P_RAW)
    # player_id=0 is reserved for padding; sample real ids in [1, N_PLAYERS).
    player_ids = torch.randint(1, N_PLAYERS, (b, p))
    role_ids = torch.randint(1, N_ROLES, (b, p))
    if n_active is None:
        # vary roster sizes per batch entry, but keep at least 1 active.
        sizes = torch.randint(1, p + 1, (b,))
    else:
        sizes = torch.full((b,), n_active)
    mask = torch.arange(p).unsqueeze(0) < sizes.unsqueeze(1)   # [B, P] bool
    # Zero player/role ids for padded slots to mimic the dataloader contract.
    player_ids = player_ids.masked_fill(~mask, 0)
    role_ids = role_ids.masked_fill(~mask, 0)
    return {
        "feats": feats,
        "player_ids": player_ids,
        "role_ids": role_ids,
        "mask": mask,
    }


def _player_encoder() -> PlayerEncoder:
    return PlayerEncoder(
        d_player_raw=D_P_RAW,
        d_player_embed=D_PLAYER_EMBED,
        d_role_embed=D_ROLE_EMBED,
        d_out=D_TEAM,
        n_players=N_PLAYERS,
        n_roles=N_ROLES,
    )


def test_player_encoder_output_shape() -> None:
    enc = _player_encoder()
    batch = _make_batch()
    out = enc(batch["feats"], batch["player_ids"], batch["role_ids"], batch["mask"])
    assert out.shape == (B, P, D_TEAM)


def test_player_encoder_zeros_padded_slots() -> None:
    """Padded slot outputs must be exactly zero so downstream pooling is safe."""
    enc = _player_encoder()
    batch = _make_batch(n_active=10)
    out = enc(batch["feats"], batch["player_ids"], batch["role_ids"], batch["mask"])
    padded = out[~batch["mask"]]
    assert padded.shape[0] > 0
    assert torch.equal(padded, torch.zeros_like(padded))


def test_player_encoder_rejects_wrong_feature_dim() -> None:
    enc = _player_encoder()
    bad = torch.randn(B, P, D_P_RAW + 1)
    ids = torch.zeros(B, P, dtype=torch.long)
    mask = torch.ones(B, P, dtype=torch.bool)
    with pytest.raises(ValueError, match="D_p_raw"):
        enc(bad, ids, ids, mask)


def test_player_embedding_padding_idx_starts_at_zero() -> None:
    """padding_idx=0 means the row stays at zero so padded slots contribute 0."""
    enc = _player_encoder()
    assert torch.equal(
        enc.player_embed.weight[0], torch.zeros_like(enc.player_embed.weight[0])
    )
    assert torch.equal(
        enc.role_embed.weight[0], torch.zeros_like(enc.role_embed.weight[0])
    )


def test_roster_attention_pool_output_shape() -> None:
    pool = RosterAttentionPool(d_in=D_TEAM, n_heads=4)
    x = torch.randn(B, P, D_TEAM)
    mask = torch.ones(B, P, dtype=torch.bool)
    out = pool(x, mask)
    assert out.shape == (B, D_TEAM)


def test_roster_attention_pool_respects_mask() -> None:
    """Two batches that share the unmasked rows but differ on the masked
    rows must produce the same pooled output."""
    pool = RosterAttentionPool(d_in=D_TEAM, n_heads=4, dropout=0.0).eval()
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, :5] = True

    x_real = torch.randn(B, 5, D_TEAM)
    x_a = torch.zeros(B, P, D_TEAM)
    x_b = torch.zeros(B, P, D_TEAM)
    x_a[:, :5] = x_real
    x_b[:, :5] = x_real
    x_a[:, 5:] = torch.randn(B, P - 5, D_TEAM) * 100  # noise in masked slots
    x_b[:, 5:] = torch.randn(B, P - 5, D_TEAM) * 100  # different noise

    out_a = pool(x_a, mask)
    out_b = pool(x_b, mask)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_roster_attention_pool_rejects_indivisible_dims() -> None:
    with pytest.raises(ValueError, match="divisible"):
        RosterAttentionPool(d_in=65, n_heads=4)


def test_game_context_encoder_output_shape() -> None:
    enc = GameContextEncoder(d_in=D_CTX_RAW, d_out=D_CTX)
    ctx = torch.randn(B, D_CTX_RAW)
    out = enc(ctx)
    assert out.shape == (B, D_CTX)


def test_game_context_encoder_rejects_wrong_input_dim() -> None:
    enc = GameContextEncoder(d_in=D_CTX_RAW, d_out=D_CTX)
    with pytest.raises(ValueError, match="d_in"):
        enc(torch.randn(B, D_CTX_RAW + 1))


def test_encoder_gradients_flow_through_all_params() -> None:
    """After one backward pass on a composed encoder stack, every
    parameter that participated has a non-None grad."""
    enc = _player_encoder()
    pool = RosterAttentionPool(d_in=D_TEAM, n_heads=4)
    ctx_enc = GameContextEncoder(d_in=D_CTX_RAW, d_out=D_CTX)

    batch = _make_batch()
    per_player = enc(batch["feats"], batch["player_ids"], batch["role_ids"], batch["mask"])
    pooled = pool(per_player, batch["mask"])
    ctx_out = ctx_enc(torch.randn(B, D_CTX_RAW))
    loss = pooled.sum() + ctx_out.sum()
    loss.backward()

    # Player embedding row 0 is the padding row and is intentionally frozen
    # at zero by ``padding_idx=0`` — same for role row 0. Skip those.
    for module in (enc, pool, ctx_enc):
        for name, p in module.named_parameters():
            if name in {"player_embed.weight", "role_embed.weight"}:
                # Sum grads over the non-padding rows only.
                assert p.grad is not None, f"{name} grad is None"
                assert torch.any(p.grad[1:] != 0), f"{name} non-padding grad is all zero"
            else:
                assert p.grad is not None, f"{name} grad is None"


# ----------------------------------------------------------------------------
# heads.py
# ----------------------------------------------------------------------------

D_MATCHUP = 16
D_TEAM_PRED = 3   # pace + off_rtg_home + off_rtg_away


def _team_head() -> TeamHead:
    return TeamHead(d_in=2 * D_TEAM + D_CTX + D_MATCHUP)


def _player_alloc_head() -> PlayerAllocHead:
    return PlayerAllocHead(d_player=D_TEAM, d_team_pred=D_TEAM_PRED)


def test_team_head_output_shapes() -> None:
    head = _team_head()
    home = torch.randn(B, D_TEAM)
    away = torch.randn(B, D_TEAM)
    ctx = torch.randn(B, D_CTX)
    matchup = torch.randn(B, D_MATCHUP)
    pace_dist, off_dist = head(home, away, ctx, matchup)
    assert isinstance(pace_dist, Normal)
    assert isinstance(off_dist, Normal)
    assert pace_dist.batch_shape == (B,)
    assert off_dist.batch_shape == (B, 2)


def test_team_head_initial_means_match_config_defaults() -> None:
    """Init bias should place pace μ near 100 and off_rtg μ near 110."""
    head = _team_head().eval()
    home = torch.randn(B, D_TEAM)
    away = torch.randn(B, D_TEAM)
    ctx = torch.randn(B, D_CTX)
    matchup = torch.randn(B, D_MATCHUP)
    pace_dist, off_dist = head(home, away, ctx, matchup)
    # μ is bias + Wh; W is non-zero so allow a wide tolerance — what we're
    # really checking is "bias dominates at init".
    assert 80 < pace_dist.mean.mean().item() < 120
    assert 90 < off_dist.mean.mean().item() < 130


def test_team_head_initial_sigmas_match_config() -> None:
    """With log_sigma weights zeroed, init σ should exactly equal exp(bias)."""
    head = _team_head().eval()
    home = torch.randn(B, D_TEAM)
    away = torch.randn(B, D_TEAM)
    ctx = torch.randn(B, D_CTX)
    matchup = torch.randn(B, D_MATCHUP)
    pace_dist, off_dist = head(home, away, ctx, matchup)
    assert torch.allclose(pace_dist.stddev, torch.full_like(pace_dist.stddev, 3.0), atol=1e-5)
    assert torch.allclose(off_dist.stddev, torch.full_like(off_dist.stddev, 4.0), atol=1e-5)


def test_dirichlet_minutes_output_shape_and_normalization() -> None:
    head = DirichletMinutes(d_in=64)
    hidden = torch.randn(B, P, 64)
    mask = torch.ones(B, P, dtype=torch.bool)
    dist = head(hidden, mask)
    assert isinstance(dist, Dirichlet)
    assert dist.concentration.shape == (B, P)
    sample = dist.sample()
    assert torch.allclose(sample.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_dirichlet_minutes_padded_slots_get_negligible_mass() -> None:
    head = DirichletMinutes(d_in=64).eval()
    hidden = torch.randn(B, P, 64)
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, :5] = True
    dist = head(hidden, mask)
    # Concentration for masked slots is 1e-4 vs ~2.0 for unmasked.
    assert (dist.concentration[~mask] < 1e-3).all()
    # Sampled mass on masked slots should be tiny.
    samples = dist.sample(sample_shape=(200,))                # [200, B, P]
    mean_mass_padded = samples[..., 5:].mean().item()
    mean_mass_real = samples[..., :5].mean().item()
    assert mean_mass_padded < 0.01
    assert mean_mass_real > 0.1


def test_player_alloc_head_returns_all_distributions() -> None:
    head = _player_alloc_head()
    player_reps = torch.randn(B, P, D_TEAM)
    team_preds = torch.randn(B, D_TEAM_PRED)
    mask = torch.ones(B, P, dtype=torch.bool)
    out = head(player_reps, team_preds, mask)

    expected_count_keys = {"fga", "tpa", "fta", "oreb", "dreb",
                           "ast", "stl", "blk", "tov", "pf"}
    expected_percent_keys = {"fgm_logits", "tpm_logits", "ftm_logits"}
    assert set(out.keys()) == {"minutes", "plays_gate"} | expected_count_keys | expected_percent_keys

    assert isinstance(out["minutes"], Dirichlet)
    assert out["minutes"].concentration.shape == (B, P)
    assert isinstance(out["plays_gate"], Bernoulli)
    assert out["plays_gate"].logits.shape == (B, P)
    for k in expected_count_keys:
        assert isinstance(out[k], NegativeBinomial), f"{k} is {type(out[k])}"
        assert out[k].batch_shape == (B, P)
    for k in expected_percent_keys:
        assert isinstance(out[k], torch.Tensor)
        assert out[k].shape == (B, P)


def test_player_alloc_head_gate_zeros_padded_slots() -> None:
    head = _player_alloc_head().eval()
    player_reps = torch.randn(B, P, D_TEAM)
    team_preds = torch.randn(B, D_TEAM_PRED)
    mask = torch.zeros(B, P, dtype=torch.bool)
    mask[:, :7] = True
    out = head(player_reps, team_preds, mask)
    probs = out["plays_gate"].probs
    # Real slots: bias=1.5 ⇒ P≈0.82, but Wh can shift it — just assert > 0.01.
    assert (probs[mask] > 0.01).all()
    assert (probs[~mask] < 1e-3).all()


def test_player_alloc_head_rejects_dim_mismatch() -> None:
    head = _player_alloc_head()
    bad_reps = torch.randn(B, P, D_TEAM + 1)
    team_preds = torch.randn(B, D_TEAM_PRED)
    mask = torch.ones(B, P, dtype=torch.bool)
    with pytest.raises(ValueError, match="d_player"):
        head(bad_reps, team_preds, mask)
    good_reps = torch.randn(B, P, D_TEAM)
    bad_tp = torch.randn(B, D_TEAM_PRED + 1)
    with pytest.raises(ValueError, match="d_team_pred"):
        head(good_reps, bad_tp, mask)


def test_conditional_make_dist_produces_binomial() -> None:
    attempts = torch.tensor([[5.0, 3.0, 0.0], [10.0, 2.0, 7.0]])
    logits = torch.zeros_like(attempts)                       # p=0.5
    dist = BoxScoreDistribution.conditional_make_dist(attempts, logits)
    assert isinstance(dist, Binomial)
    assert dist.batch_shape == (2, 3)
    # log_prob defined and finite for valid counts.
    target = torch.tensor([[2.0, 1.0, 0.0], [5.0, 1.0, 3.0]])
    lp = dist.log_prob(target)
    assert torch.isfinite(lp).all()


def test_head_gradients_flow_to_every_parameter() -> None:
    """After one backward pass through team+alloc heads + a conditional
    Binomial, every parameter has a non-None grad."""
    team_head = _team_head()
    alloc_head = _player_alloc_head()
    home = torch.randn(B, D_TEAM)
    away = torch.randn(B, D_TEAM)
    ctx = torch.randn(B, D_CTX)
    matchup = torch.randn(B, D_MATCHUP)
    pace_dist, off_dist = team_head(home, away, ctx, matchup)

    # Use the team predictions as the conditioning signal — same plumbing
    # the hierarchical model will use.
    team_preds = torch.stack(
        [pace_dist.mean, off_dist.mean[:, 0], off_dist.mean[:, 1]], dim=-1
    )
    player_reps = torch.randn(B, P, D_TEAM)
    mask = torch.ones(B, P, dtype=torch.bool)
    out = alloc_head(player_reps, team_preds, mask)

    # Materialize all conditional Binomials. Must use .sample() not .mean —
    # Binomial requires integer-valued total_count.
    make = BoxScoreDistribution.conditional_make_dist
    fgm_dist = make(out["fga"].sample(), out["fgm_logits"])
    tpm_dist = make(out["tpa"].sample(), out["tpm_logits"])
    ftm_dist = make(out["fta"].sample(), out["ftm_logits"])

    # Composite scalar loss exercising every output path.
    loss = (
        -pace_dist.log_prob(torch.full((B,), 100.0)).mean()
        - off_dist.log_prob(torch.full((B, 2), 110.0)).mean()
        - out["minutes"].log_prob(out["minutes"].sample()).mean()
        - out["plays_gate"].log_prob(mask.float()).mean()
    )
    for stat in PlayerAllocHead._COUNT_STATS:
        loss = loss - out[stat].log_prob(torch.zeros(B, P)).mean()
    # Binomial log_probs exercise all three percent head logits.
    loss = loss - fgm_dist.log_prob(torch.zeros(B, P)).mean()
    loss = loss - tpm_dist.log_prob(torch.zeros(B, P)).mean()
    loss = loss - ftm_dist.log_prob(torch.zeros(B, P)).mean()
    loss.backward()

    for module in (team_head, alloc_head):
        for name, p in module.named_parameters():
            assert p.grad is not None, f"{name} grad is None"


# ----------------------------------------------------------------------------
# hierarchical.py
# ----------------------------------------------------------------------------

from nba_sim.models.hierarchical import HierarchicalBoxScoreModel    # noqa: E402


def _model_config() -> dict:
    return {
        "dims": {
            "d_player_raw": D_P_RAW,
            "d_player": 64,
            "d_team": D_TEAM,
            "d_context_raw": D_CTX_RAW,
            "d_context": D_CTX,
            "d_matchup": D_MATCHUP,
        },
        "embeddings": {
            "n_players": N_PLAYERS,
            "d_player_embed": D_PLAYER_EMBED,
            "n_teams": 40,
            "d_team_embed": 16,
            "n_roles": N_ROLES,
            "d_role": D_ROLE_EMBED,
        },
        "encoder": {
            "player_mlp_hidden": [128, 128],
            "player_mlp_dropout": 0.1,
            "roster_attention_heads": 4,
            "roster_attention_dropout": 0.1,
            "context_mlp_hidden": [64],
        },
        "team_head": {
            "hidden": [128, 64],
            "pace_noise_init": 3.0,
            "rtg_noise_init": 4.0,
        },
        "player_alloc_head": {
            "hidden": [128, 128],
            "dropout": 0.1,
            "dirichlet_alpha_init_scale": 2.0,
            "gate_bias_init": 1.5,
        },
    }


def _make_full_batch(b: int = B, p: int = P) -> dict[str, torch.Tensor]:
    home = _make_batch(b, p)
    away = _make_batch(b, p)
    return {
        "home_player_feats": home["feats"],
        "home_player_ids": home["player_ids"],
        "home_role_ids": home["role_ids"],
        "home_mask": home["mask"],
        "away_player_feats": away["feats"],
        "away_player_ids": away["player_ids"],
        "away_role_ids": away["role_ids"],
        "away_mask": away["mask"],
        "context": torch.randn(b, D_CTX_RAW),
        "matchup": torch.randn(b, D_MATCHUP),
    }


def test_hierarchical_forward_pass_end_to_end() -> None:
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch = _make_full_batch()
    out = model(batch)

    assert isinstance(out, BoxScoreDistribution)
    # Team-level
    assert out.pace.batch_shape == (B,)
    assert out.off_rtg.batch_shape == (B, 2)
    # Per-team Dirichlet minutes
    assert out.minutes_home.concentration.shape == (B, P)
    assert out.minutes_away.concentration.shape == (B, P)
    # Per-team Bernoulli gate
    assert out.plays_gate_home.logits.shape == (B, P)
    assert out.plays_gate_away.logits.shape == (B, P)
    # Per-team NB heads
    for side in ("home", "away"):
        for stat in ("fga", "tpa", "fta", "oreb", "dreb",
                     "ast", "stl", "blk", "tov", "pf"):
            d = getattr(out, f"{stat}_{side}")
            assert isinstance(d, NegativeBinomial), f"{stat}_{side}"
            assert d.batch_shape == (B, P), f"{stat}_{side}"
    # Per-team percent logits (tensors, not Distributions)
    for side in ("home", "away"):
        for stat in ("fgm", "tpm", "ftm"):
            t = getattr(out, f"{stat}_probs_{side}")
            assert t.shape == (B, P), f"{stat}_probs_{side}"


def test_hierarchical_teacher_forces_team_preds_when_provided() -> None:
    """When pace_true/off_rtg_true are in batch, the player-alloc head's
    conditioning should differ from the no-teacher-forcing case."""
    torch.manual_seed(0)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch = _make_full_batch()

    out_no_tf = model(batch)
    # Inject teacher-forced values way off from the predicted means.
    batch_tf = dict(batch)
    batch_tf["pace_true"] = torch.full((B,), 50.0)        # very low pace
    batch_tf["off_rtg_true"] = torch.full((B, 2), 50.0)   # very low off_rtg
    out_tf = model(batch_tf)

    # Team head outputs are unaffected by teacher-forcing — same exact dist params.
    assert torch.equal(out_no_tf.pace.mean, out_tf.pace.mean)
    assert torch.equal(out_no_tf.off_rtg.mean, out_tf.off_rtg.mean)
    # But the player-alloc heads condition on a different signal, so their
    # outputs must differ. We assert on fga_home.logits because the Dirichlet
    # minutes head and the team-head σ projection have zeroed weights at init
    # (so init outputs depend only on bias) — the NB count heads don't.
    assert not torch.allclose(
        out_no_tf.fga_home.logits, out_tf.fga_home.logits
    )


def test_hierarchical_gradients_flow_to_every_parameter() -> None:
    model = HierarchicalBoxScoreModel(_model_config())
    batch = _make_full_batch()
    batch["pace_true"] = torch.full((B,), 100.0)
    batch["off_rtg_true"] = torch.full((B, 2), 110.0)
    out = model(batch)

    # Composite scalar loss exercising every output path.
    make = BoxScoreDistribution.conditional_make_dist
    loss = (
        -out.pace.log_prob(batch["pace_true"]).mean()
        - out.off_rtg.log_prob(batch["off_rtg_true"]).mean()
        - out.minutes_home.log_prob(out.minutes_home.sample()).mean()
        - out.minutes_away.log_prob(out.minutes_away.sample()).mean()
        - out.plays_gate_home.log_prob(batch["home_mask"].float()).mean()
        - out.plays_gate_away.log_prob(batch["away_mask"].float()).mean()
    )
    for side in ("home", "away"):
        for stat in ("fga", "tpa", "fta", "oreb", "dreb",
                     "ast", "stl", "blk", "tov", "pf"):
            d = getattr(out, f"{stat}_{side}")
            loss = loss - d.log_prob(torch.zeros(B, P)).mean()
        # Conditional Binomials exercise the percent-head logits.
        for stat, attempts_name in (("fgm", "fga"), ("tpm", "tpa"), ("ftm", "fta")):
            attempts = getattr(out, f"{attempts_name}_{side}").sample()
            logits = getattr(out, f"{stat}_probs_{side}")
            loss = loss - make(attempts, logits).log_prob(torch.zeros(B, P)).mean()

    loss.backward()

    for name, p in model.named_parameters():
        if name in {
            "player_encoder.player_embed.weight",
            "player_encoder.role_embed.weight",
        }:
            # padding_idx=0 row stays frozen at zero by design; rest must have grad.
            assert p.grad is not None, f"{name} grad is None"
            assert torch.any(p.grad[1:] != 0), f"{name} non-padding grad is all zero"
        else:
            assert p.grad is not None, f"{name} grad is None"


def test_hierarchical_freeze_unfreeze_player_heads() -> None:
    model = HierarchicalBoxScoreModel(_model_config())
    model.freeze_player_heads()
    assert all(not p.requires_grad for p in model.player_alloc_head.parameters())
    # Encoder + team head should still be trainable.
    assert any(p.requires_grad for p in model.team_head.parameters())
    assert any(p.requires_grad for p in model.player_encoder.parameters())
    model.unfreeze_player_heads()
    assert all(p.requires_grad for p in model.player_alloc_head.parameters())


def test_hierarchical_checkpoint_roundtrip(tmp_path) -> None:
    """save_checkpoint → from_checkpoint produces a model whose forward
    output equals the original on the same batch."""
    torch.manual_seed(42)
    model = HierarchicalBoxScoreModel(_model_config()).eval()
    batch = _make_full_batch()
    out_orig = model(batch)

    ckpt = tmp_path / "test.pt"
    model.save_checkpoint(ckpt)
    loaded = HierarchicalBoxScoreModel.from_checkpoint(ckpt).eval()
    out_loaded = loaded(batch)

    assert torch.allclose(out_orig.pace.mean, out_loaded.pace.mean)
    assert torch.allclose(out_orig.off_rtg.mean, out_loaded.off_rtg.mean)
    assert torch.allclose(
        out_orig.minutes_home.concentration,
        out_loaded.minutes_home.concentration,
    )
    assert torch.allclose(out_orig.fga_home.mean, out_loaded.fga_home.mean)


# Legacy stub name from before hierarchical.py existed — keep one alias for
# the original test ID listed in PLAN §4.7.
def test_gradients_flow_to_every_parameter() -> None:
    """Same as test_hierarchical_gradients_flow_to_every_parameter — kept
    under the PLAN.md §4.7 name so the ship-gate test ID is recognized."""
    test_hierarchical_gradients_flow_to_every_parameter()
