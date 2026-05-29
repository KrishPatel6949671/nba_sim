"""Snapshot → model-batch assembly (v2PLAN.md §15).

**Phase 7 — stub only.** This module turns a refreshed ``data/snapshot/`` plus
a chosen ``(home, away)`` matchup into the single-game batch the model's
forward pass consumes. It is intentionally unimplemented in Phase 6; the
signature is fixed here so the package's public surface (Appendix C) and
``nba_sim.snapshot.__getattr__`` are stable.

When implemented it will:
  1. Read rosters / player_features / team_features / team_lastgame + as_of.json.
  2. Filter rosters to (home, away); sort each side by p_min_avg_10 desc to
     mirror the training-time roster ordering.
  3. Assemble per-side feature blocks identical to
     ``BoxScoreDataset._build_side`` — filling the four refresh-null player
     numerics (rest_days, travel_miles_prev, opp_def_rtg_vs_pos,
     opp_blk_allowed_vs_pos) and the four bool flags (is_starter, dnp,
     is_home, b2b) for the chosen matchup + as_of date.
  4. Assemble the 24-d context vector from as_of + team_lastgame lookups.
  5. Assemble the 16-d matchup vector from team_features.
  6. Adopt the train split's player_id_map + feature_stats (same train-vs-eval
     standardization discipline as v1).
"""

from __future__ import annotations

from pathlib import Path

import torch


def build_synthetic_game_batch(
    *,
    home_team: str,
    away_team: str,
    snapshot_dir: Path = Path("data/snapshot"),
    train_parquet: Path = Path("data/processed/train.parquet"),
    max_players: int = 15,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, str]], list[tuple[int, str]], str]:
    """Build a ``(model_batch, home_players, away_players, as_of_iso)`` tuple.

    ``model_batch`` is a ``dict[str, torch.Tensor]`` with ``B == 1`` that goes
    straight into ``model.forward(batch)`` — no teacher-forcing keys. See the
    module docstring for the assembly steps.

    **Phase 7.** Raises until implemented.
    """
    raise NotImplementedError("build_synthetic_game_batch is a Phase 7 deliverable")
