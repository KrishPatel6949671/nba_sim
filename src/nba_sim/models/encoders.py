"""Encoders: player feature encoder, roster attention pool, game context encoder.

Shapes (``B`` batch, ``P=15`` padded roster slots, see PLAN.md §4.2):

    PlayerEncoder        : [B, P, D_p_raw] + ids [B, P] -> [B, P, D_out]
    RosterAttentionPool  : [B, P, D_in] + mask [B, P]   -> [B, D_in]
    GameContextEncoder   : [B, D_ctx_raw]               -> [B, D_out]

The roster pool uses multi-head attention with a learned query vector so
the model picks out the "important" players for team-level prediction
without caring about their order.

Padding convention: ``player_id=0`` and ``role_id=0`` are reserved for
padded / unknown slots. The mask (``True`` = real player) is the ground
truth — embedding lookups still happen for masked slots but their
contribution is zeroed downstream.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _build_mlp(
    d_in: int,
    hidden: Sequence[int],
    d_out: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = d_in
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LayerNorm(h))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, d_out))
    return nn.Sequential(*layers)


class PlayerEncoder(nn.Module):
    """Per-player feature encoder with PLAN §5.4 partial-pooling embedding.

    Each player's embedding is decomposed as ``e_i = role_centroid[role(i)]
    + δ_i`` with ``δ_i`` initialized to 0. Players with the same role share
    an identical embedding at init; only training pushes their ``δ_i``
    away from zero. An L2 penalty on ``δ_i`` (applied externally by
    :func:`nba_sim.models.losses.composite_nll`) keeps rare / seldom-seen
    players close to their role's prior centroid.

    Concatenates ``[raw_features, e_i]`` and pushes the result through an
    MLP to produce a ``d_out``-dim vector per player. Output for padded
    slots is zeroed using the mask so downstream pooling can ignore them.
    """

    def __init__(
        self,
        d_player_raw: int,
        d_player_embed: int,
        d_out: int,
        n_players: int,
        n_roles: int,
        hidden: tuple[int, ...] = (128, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_player_raw = d_player_raw
        self.d_out = d_out
        self.d_player_embed = d_player_embed

        # Role centroid: shared across all players of the same role; this
        # is the partial-pooling prior. padding_idx=0 keeps the padding
        # row at zero so padded slots contribute zero downstream.
        self.role_embed = nn.Embedding(n_roles, d_player_embed, padding_idx=0)
        # Per-player δ: initialized to zero so every player starts at its
        # role centroid. ``padding_idx=0`` additionally zeros the gradient
        # for the padding row. Real player IDs are in [1, n_players-1].
        self.player_delta = nn.Embedding(n_players, d_player_embed, padding_idx=0)
        nn.init.zeros_(self.player_delta.weight)

        d_concat = d_player_raw + d_player_embed
        self.mlp = _build_mlp(d_concat, hidden, d_out, dropout)

    def forward(
        self,
        feats: torch.Tensor,         # [B, P, D_p_raw]
        player_ids: torch.Tensor,    # [B, P] long
        role_ids: torch.Tensor,      # [B, P] long
        mask: torch.Tensor,          # [B, P] bool — True if slot is real
    ) -> torch.Tensor:               # [B, P, D_out]
        if feats.shape[-1] != self.d_player_raw:
            raise ValueError(
                f"PlayerEncoder expected D_p_raw={self.d_player_raw}, got {feats.shape[-1]}"
            )

        role_centroid = self.role_embed(role_ids)    # [B, P, d_player_embed]
        delta = self.player_delta(player_ids)        # [B, P, d_player_embed]
        e = role_centroid + delta                    # [B, P, d_player_embed]
        x = torch.cat([feats, e], dim=-1)
        out = self.mlp(x)                            # [B, P, d_out]

        # Zero padded slots so downstream code (pooling, heads) can rely
        # on the invariant that masked entries contribute nothing.
        return out * mask.unsqueeze(-1).to(out.dtype)

    def gather_active_deltas(
        self,
        player_ids: torch.Tensor,    # [B, P] long
        mask: torch.Tensor,          # [B, P] bool
    ) -> torch.Tensor:               # [N_active, d_player_embed]
        """Look up ``δ_i`` for masked-in player slots and flatten.

        Used by :meth:`HierarchicalBoxScoreModel.embedding_deltas_for_batch`
        to feed ``composite_nll(..., embedding_deltas=...)`` so the L2
        partial-pooling penalty fires only on active players.
        """
        delta = self.player_delta(player_ids)        # [B, P, D]
        return delta[mask]                           # [N_active, D]


class RosterAttentionPool(nn.Module):
    """Attention-pool a padded set of player encodings into a single team vector.

    A single learned query attends over the per-player encodings. Padded
    roster slots are masked out via ``key_padding_mask``.
    """

    def __init__(self, d_in: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if d_in % n_heads != 0:
            raise ValueError(f"d_in ({d_in}) must be divisible by n_heads ({n_heads})")
        self.d_in = d_in
        self.query = nn.Parameter(torch.randn(1, 1, d_in) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_in,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,             # [B, P, D_in]
        mask: torch.Tensor,          # [B, P] bool — True if real
    ) -> torch.Tensor:               # [B, D_in]
        b = x.shape[0]
        q = self.query.expand(b, -1, -1)                  # [B, 1, D_in]
        key_padding_mask = ~mask                          # True = ignore
        attended, _ = self.attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )                                                 # [B, 1, D_in]
        pooled = self.norm(attended.squeeze(1))           # [B, D_in]
        return self.dropout(pooled)


class GameContextEncoder(nn.Module):
    """MLP over numerical + 1-hot categorical context features."""

    def __init__(self, d_in: int, d_out: int, hidden: tuple[int, ...] = (64,)) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.mlp = _build_mlp(d_in, hidden, d_out, dropout=0.0)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        if ctx.shape[-1] != self.d_in:
            raise ValueError(
                f"GameContextEncoder expected d_in={self.d_in}, got {ctx.shape[-1]}"
            )
        return self.mlp(ctx)
