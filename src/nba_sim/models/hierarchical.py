"""Full hierarchical team -> player model.

Composition (see PLAN.md §4.1):

    HierarchicalBoxScoreModel
        ├── PlayerEncoder         (shared across home/away)
        ├── RosterAttentionPool   (shared across home/away)
        ├── GameContextEncoder
        ├── TeamHead              -> pace, off_rtg
        └── PlayerAllocHead       (shared across home/away) -> per-player distributions

Forward returns a :class:`BoxScoreDistribution` so that both the loss
(:mod:`nba_sim.models.losses`) and the sampler
(:mod:`nba_sim.simulate.sampler`) share the same output contract.

Batch dict contract (see :mod:`nba_sim.training.dataset`):

    {
        "home_player_feats": [B, P, D_p_raw],
        "home_player_ids":   [B, P] long,
        "home_role_ids":     [B, P] long,
        "home_mask":         [B, P] bool,
        "away_player_feats": ..., "away_player_ids": ..., ..., "away_mask": ...,
        "context":           [B, D_ctx_raw],
        "matchup":           [B, D_m],
        # Teacher-forcing keys (optional — present during training, absent at inference):
        "pace_true":         [B],
        "off_rtg_true":      [B, 2],
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from nba_sim.models.encoders import (
    GameContextEncoder,
    PlayerEncoder,
    RosterAttentionPool,
)
from nba_sim.models.heads import BoxScoreDistribution, PlayerAllocHead, TeamHead


class HierarchicalBoxScoreModel(nn.Module):
    """The primary v1 model."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Build from a parsed ``configs/model.yaml`` dict."""
        super().__init__()
        self.config = config

        dims = config["dims"]
        emb = config["embeddings"]
        enc = config["encoder"]
        team_cfg = config["team_head"]
        alloc_cfg = config["player_alloc_head"]

        d_team = dims["d_team"]
        d_ctx_raw = dims["d_context_raw"]
        d_ctx = dims["d_context"]
        d_matchup = dims["d_matchup"]

        self.player_encoder = PlayerEncoder(
            d_player_raw=dims["d_player_raw"],
            d_player_embed=emb["d_player_embed"],
            d_role_embed=emb["d_role"],
            d_out=d_team,
            n_players=emb["n_players"],
            n_roles=emb["n_roles"],
            hidden=tuple(enc["player_mlp_hidden"]),
            dropout=enc["player_mlp_dropout"],
        )
        self.roster_pool = RosterAttentionPool(
            d_in=d_team,
            n_heads=enc["roster_attention_heads"],
            dropout=enc["roster_attention_dropout"],
        )
        self.context_encoder = GameContextEncoder(
            d_in=d_ctx_raw,
            d_out=d_ctx,
            hidden=tuple(enc["context_mlp_hidden"]),
        )
        self.team_head = TeamHead(
            d_in=2 * d_team + d_ctx + d_matchup,
            hidden=tuple(team_cfg["hidden"]),
            pace_noise_init=team_cfg["pace_noise_init"],
            rtg_noise_init=team_cfg["rtg_noise_init"],
        )
        self.player_alloc_head = PlayerAllocHead(
            d_player=d_team,
            d_team_pred=3,
            hidden=tuple(alloc_cfg["hidden"]),
            dropout=alloc_cfg["dropout"],
            dirichlet_alpha_init_scale=alloc_cfg["dirichlet_alpha_init_scale"],
            gate_bias_init=alloc_cfg["gate_bias_init"],
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> BoxScoreDistribution:
        """Consume a padded batch and emit all distributions.

        Conditioning of the player-alloc head on team-level predictions:
        if ``batch`` contains ``pace_true`` and ``off_rtg_true`` (training
        teacher-forcing), those are used. Otherwise the team head's mean
        prediction is used (inference; the simulator handles stochastic
        sampling by computing samples once externally and re-feeding them
        as the "true" keys).
        """
        home_per_player = self.player_encoder(
            batch["home_player_feats"],
            batch["home_player_ids"],
            batch["home_role_ids"],
            batch["home_mask"],
        )
        away_per_player = self.player_encoder(
            batch["away_player_feats"],
            batch["away_player_ids"],
            batch["away_role_ids"],
            batch["away_mask"],
        )

        home_team_rep = self.roster_pool(home_per_player, batch["home_mask"])
        away_team_rep = self.roster_pool(away_per_player, batch["away_mask"])

        ctx = self.context_encoder(batch["context"])
        matchup = batch["matchup"]

        pace_dist, off_rtg_dist = self.team_head(
            home_team_rep, away_team_rep, ctx, matchup
        )

        if "pace_true" in batch and "off_rtg_true" in batch:
            pace_cond = batch["pace_true"]
            off_cond = batch["off_rtg_true"]
        else:
            pace_cond = pace_dist.mean
            off_cond = off_rtg_dist.mean
        team_preds = torch.stack(
            [pace_cond, off_cond[:, 0], off_cond[:, 1]], dim=-1
        )                                                              # [B, 3]

        home_alloc = self.player_alloc_head(
            home_per_player, team_preds, batch["home_mask"]
        )
        away_alloc = self.player_alloc_head(
            away_per_player, team_preds, batch["away_mask"]
        )

        return BoxScoreDistribution(
            pace=pace_dist,
            off_rtg=off_rtg_dist,
            minutes_home=home_alloc["minutes"],
            minutes_away=away_alloc["minutes"],
            plays_gate_home=home_alloc["plays_gate"],
            plays_gate_away=away_alloc["plays_gate"],
            fga_home=home_alloc["fga"],
            fga_away=away_alloc["fga"],
            tpa_home=home_alloc["tpa"],
            tpa_away=away_alloc["tpa"],
            fta_home=home_alloc["fta"],
            fta_away=away_alloc["fta"],
            fgm_probs_home=home_alloc["fgm_logits"],
            fgm_probs_away=away_alloc["fgm_logits"],
            tpm_probs_home=home_alloc["tpm_logits"],
            tpm_probs_away=away_alloc["tpm_logits"],
            ftm_probs_home=home_alloc["ftm_logits"],
            ftm_probs_away=away_alloc["ftm_logits"],
            oreb_home=home_alloc["oreb"],
            oreb_away=away_alloc["oreb"],
            dreb_home=home_alloc["dreb"],
            dreb_away=away_alloc["dreb"],
            ast_home=home_alloc["ast"],
            ast_away=away_alloc["ast"],
            stl_home=home_alloc["stl"],
            stl_away=away_alloc["stl"],
            blk_home=home_alloc["blk"],
            blk_away=away_alloc["blk"],
            tov_home=home_alloc["tov"],
            tov_away=away_alloc["tov"],
            pf_home=home_alloc["pf"],
            pf_away=away_alloc["pf"],
        )

    def freeze_player_heads(self) -> None:
        """For §5.3 curriculum: train only pace + off_rtg in epochs 0–4."""
        for p in self.player_alloc_head.parameters():
            p.requires_grad = False

    def unfreeze_player_heads(self) -> None:
        for p in self.player_alloc_head.parameters():
            p.requires_grad = True

    def save_checkpoint(self, path: str | Path) -> None:
        """Save ``{"config": ..., "state_dict": ...}`` so :meth:`from_checkpoint`
        can reconstruct the model architecture from the file alone."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "HierarchicalBoxScoreModel":
        """Load weights + config from a checkpoint file."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
