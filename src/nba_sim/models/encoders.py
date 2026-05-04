"""Encoders: player feature encoder, roster attention pool, game context encoder.

Shapes (``B`` batch, ``P=15`` padded roster slots, see PLAN.md §4.2):

    PlayerEncoder        : [B, P, D_p_raw] + ids [B, P] -> [B, P, D_t]
    RosterAttentionPool  : [B, P, D_t] + mask [B, P]   -> [B, D_t]
    GameContextEncoder   : dict of numerical+categorical -> [B, D_ctx]

The roster pool uses multi-head attention with a learned query vector so
the model picks out the "important" players for team-level prediction
without caring about their order.
"""

from __future__ import annotations

import torch
from torch import nn


class PlayerEncoder(nn.Module):
    """Per-player feature encoder with a learned player-embedding table."""

    def __init__(
        self,
        d_player_raw: int,
        d_player_embed: int,
        d_role_embed: int,
        d_out: int,
        n_players: int,
        n_roles: int,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        feats: torch.Tensor,         # [B, P, D_p_raw]
        player_ids: torch.Tensor,    # [B, P]
        role_ids: torch.Tensor,      # [B, P]
        mask: torch.Tensor,          # [B, P] bool — True if slot is real
    ) -> torch.Tensor:               # [B, P, D_out]
        raise NotImplementedError


class RosterAttentionPool(nn.Module):
    """Attention-pool a padded set of player encodings into a single team vector."""

    def __init__(self, d_in: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,             # [B, P, D_in]
        mask: torch.Tensor,          # [B, P]
    ) -> torch.Tensor:               # [B, D_in]
        raise NotImplementedError


class GameContextEncoder(nn.Module):
    """MLP over numerical + 1-hot categorical context features."""

    def __init__(self, d_in: int, d_out: int, hidden: tuple[int, ...] = (64,)) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
