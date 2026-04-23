"""PyTorch Dataset / DataLoader for the processed parquet splits.

Reads the output of :func:`nba_sim.data.etl.interim_to_processed` and
collates each game into a padded batch of shape ``[B, P, D]`` plus a
mask indicating real vs. padding slots.

The dataset is **game-level** (one item per game), not player-level, so
that team-level predictions and player-level predictions share inputs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class BoxScoreDataset(Dataset):
    """One item per game, each item includes both teams' padded rosters."""

    def __init__(
        self,
        processed_parquet: str | Path,
        max_players: int = 15,
        split: str = "train",
    ) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def collate_games(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-game dicts into a padded batch. Emits ``mask_home``, ``mask_away``."""
    raise NotImplementedError
