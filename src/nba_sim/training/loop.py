"""Training loop: curriculum, optimizer, scheduler, checkpointing, early stop.

Two-stage curriculum (PLAN §5.3):

1. **Team-level pretrain** (``team_level_pretrain_epochs``) — freeze player
   heads and zero out their loss weights; train only pace + off_rtg.
2. **Joint training** — unfreeze and optimize the full composite loss.

Orchestration:

- AdamW with cosine-with-warmup schedule (linear warmup → cosine to 0).
- bf16 ``autocast`` on CUDA only; plain fp32 elsewhere (no GradScaler needed).
- Grad clipping to ``optim.grad_clip_max_norm``.
- Per-epoch checkpoint to ``out_dir/ckpt_epoch_{n}.pt``.
- ``out_dir/best.pt`` tracks the val-composite-NLL minimum.
- Early stopping with patience over val composite NLL.
- Deterministic by default (PLAN §5.7 ship gate: "same seed → identical
  loss curve to 1e-6").

Entry points:

- :func:`train` — pure function, takes already-loaded config dicts.
- :func:`train_from_yaml` — resolves ``configs/{train,model,data}.yaml``
  and delegates to ``train``.
- ``python -m nba_sim.training.loop --config configs/train.yaml``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from nba_sim.models.hierarchical import HierarchicalBoxScoreModel
from nba_sim.models.losses import LossWeights, composite_nll
from nba_sim.training.dataset import BoxScoreDataset, collate_games


# ---------------------------------------------------------------------------
# Reproducibility + device helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG the training loop touches.

    With ``deterministic=True`` we also enable cuDNN deterministic mode
    and ``torch.use_deterministic_algorithms`` so the same seed reproduces
    the same loss trajectory to floating-point precision.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, AttributeError):
            pass
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker so multi-worker shuffles are deterministic."""
    base = torch.initial_seed() % 2**31
    random.seed(base + worker_id)
    np.random.seed((base + worker_id) % 2**31)


def _resolve_device(spec: str) -> torch.device:
    """``cuda | cpu | auto`` → torch.device, falling back to CPU when CUDA is missing."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(spec)


def _autocast_ctx(device: torch.device, precision: str):
    """bf16 autocast on CUDA; no-op everywhere else."""
    if device.type == "cuda" and precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------


def _team_only_loss_weights(base: LossWeights) -> LossWeights:
    """Zero out every per-player head; keep only pace + off_rtg.

    Used during the curriculum pretrain phase. The player heads are also
    frozen via ``model.freeze_player_heads()`` so they receive no grads;
    zeroing the weights additionally skips the wasted forward computation
    of their NLLs (the heads still run but their losses don't contribute).
    """
    return LossWeights(
        pace=base.pace,
        off_rtg=base.off_rtg,
        minutes=0.0,
        gate=0.0,
        fga=0.0, tpa=0.0, fta=0.0,
        fgm=0.0, tpm=0.0, ftm=0.0,
        oreb=0.0, dreb=0.0, ast=0.0, stl=0.0, blk=0.0, tov=0.0, pf=0.0,
        embedding_pool=0.0,
    )


def _make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay to 0. Hands back an ``LambdaLR`` callable."""
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move every tensor leaf to ``device``; pass through non-tensors."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Tracking (lazy; optional)
# ---------------------------------------------------------------------------


class _Tracker:
    """Light wrapper over wandb / tensorboard, swappable via config.

    A no-op when ``backend == "none"`` or when the optional dependency
    isn't installed — keeps the test path independent of wandb.
    """

    def __init__(self, tracking_cfg: dict[str, Any]) -> None:
        backend = tracking_cfg.get("backend", "none")
        self._wandb = None
        self._tb = None
        if backend == "wandb":
            # Default to offline mode so a missing ~/.netrc doesn't crash
            # the wandb-core subprocess. Users explicitly setting WANDB_MODE
            # (e.g. "online" after `wandb login`) override this.
            os.environ.setdefault("WANDB_MODE", "offline")
            os.environ.setdefault("WANDB_SILENT", "true")
            try:
                import wandb
                wandb.init(
                    project=tracking_cfg.get("project"),
                    entity=tracking_cfg.get("entity"),
                    config=tracking_cfg,
                    mode=os.environ.get("WANDB_MODE", "offline"),
                )
                self._wandb = wandb
            except Exception:
                # Don't crash the training run if wandb is misconfigured.
                self._wandb = None
        if backend == "tensorboard" or tracking_cfg.get("also_write_tensorboard"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter()
            except Exception:
                self._tb = None
        self._every_n = int(tracking_cfg.get("log_every_n_steps", 25))

    def log_step(self, metrics: dict[str, float], step: int) -> None:
        if step % self._every_n != 0:
            return
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)

    def log_epoch(self, metrics: dict[str, float], epoch: int) -> None:
        if self._wandb is not None:
            self._wandb.log({**metrics, "epoch": epoch})
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(f"epoch/{k}", v, epoch)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------


def train(
    *,
    train_parquet: str | Path,
    val_parquet: str | Path,
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    out_dir: str | Path = "models",
    verbose: bool = True,
) -> dict[str, Any]:
    """Run hierarchical-NN training end-to-end. Returns a summary dict.

    Parameters
    ----------
    train_parquet, val_parquet
        Paths to the processed splits (``data/processed/{train,val}.parquet``).
    model_config
        Loaded ``configs/model.yaml`` — passed straight to
        :class:`HierarchicalBoxScoreModel`.
    train_config
        Loaded ``configs/train.yaml`` (top-level dict). Read keys: ``run``,
        ``optim``, ``schedule``, ``loop``, ``curriculum``, ``early_stopping``,
        ``loss_weights``, ``tracking``.
    out_dir
        Where checkpoints + ``training_summary.json`` land. Created if missing.

    Returns
    -------
    dict
        ``{"best_val_nll": float, "best_epoch": int, "epochs_trained": int,
        "history": [...], "checkpoint_dir": str}``. Also written to
        ``out_dir/training_summary.json``.
    """
    run_cfg = train_config["run"]
    optim_cfg = train_config["optim"]
    sched_cfg = train_config["schedule"]
    loop_cfg = train_config["loop"]
    curriculum_cfg = train_config["curriculum"]
    es_cfg = train_config["early_stopping"]
    tracking_cfg = train_config.get("tracking", {"backend": "none"})

    set_seed(int(run_cfg["seed"]), deterministic=bool(run_cfg.get("deterministic", True)))
    device = _resolve_device(str(run_cfg.get("device", "auto")))
    precision = str(run_cfg.get("precision", "fp32"))

    # ---- datasets / loaders ----------------------------------------------
    train_ds = BoxScoreDataset(train_parquet)
    val_ds = BoxScoreDataset(
        val_parquet,
        player_id_map=train_ds.player_id_map,
        feature_stats=train_ds.feature_stats,
    )

    # Seeded generator for shuffling reproducibility.
    g = torch.Generator()
    g.manual_seed(int(run_cfg["seed"]))

    num_workers = int(loop_cfg.get("num_workers", 0))
    pin_memory = bool(loop_cfg.get("pin_memory", False)) and device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(loop_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_games,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(loop_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_games,
    )

    # ---- model / optimizer / scheduler -----------------------------------
    model = HierarchicalBoxScoreModel(model_config).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg["weight_decay"]),
        betas=tuple(optim_cfg["betas"]),
    )

    max_epochs = int(sched_cfg["max_epochs"])
    total_steps = max_epochs * max(1, len(train_loader))
    lr_lambda = _make_lr_lambda(int(sched_cfg["warmup_steps"]), total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    base_weights = LossWeights(**train_config["loss_weights"])
    team_only_weights = _team_only_loss_weights(base_weights)

    pretrain_epochs = int(curriculum_cfg["team_level_pretrain_epochs"])
    grad_clip = float(optim_cfg["grad_clip_max_norm"])
    es_enabled = bool(es_cfg["enabled"])
    patience_max = int(es_cfg["patience"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracker = _Tracker(tracking_cfg)
    best_val = math.inf
    best_epoch = -1
    patience_left = patience_max
    history: list[dict[str, float]] = []
    global_step = 0

    try:
        for epoch in range(max_epochs):
            in_pretrain = epoch < pretrain_epochs
            # Freeze on epoch 0; unfreeze at the curriculum boundary.
            if epoch == 0 and pretrain_epochs > 0:
                model.freeze_player_heads()
            if epoch == pretrain_epochs:
                model.unfreeze_player_heads()
            epoch_weights = team_only_weights if in_pretrain else base_weights

            # ---- train epoch ---------------------------------------------
            model.train()
            t0 = time.time()
            train_loss_sum = 0.0
            train_n = 0
            for batch in train_loader:
                batch = _move_batch(batch, device)
                opt.zero_grad(set_to_none=True)
                with _autocast_ctx(device, precision):
                    preds = model(batch)
                    deltas = model.embedding_deltas_for_batch(batch)
                    out = composite_nll(
                        preds, batch, epoch_weights, embedding_deltas=deltas
                    )
                    loss = out["loss"]
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )
                opt.step()
                scheduler.step()

                B = batch["pace"].shape[0]
                train_loss_sum += float(loss.detach().item()) * B
                train_n += B
                global_step += 1
                tracker.log_step(
                    {
                        "train/loss": float(loss.detach().item()),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )
            train_loss = train_loss_sum / max(1, train_n)

            # ---- val epoch -----------------------------------------------
            model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = _move_batch(batch, device)
                    with _autocast_ctx(device, precision):
                        preds = model(batch)
                        deltas = model.embedding_deltas_for_batch(batch)
                        out = composite_nll(
                            preds, batch, epoch_weights, embedding_deltas=deltas
                        )
                    B = batch["pace"].shape[0]
                    val_loss_sum += float(out["loss"].item()) * B
                    val_n += B
            val_loss = val_loss_sum / max(1, val_n)

            # ---- checkpoint + best tracking ------------------------------
            ckpt_path = out_dir / f"ckpt_epoch_{epoch}.pt"
            model.save_checkpoint(ckpt_path)

            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                best_epoch = epoch
                model.save_checkpoint(out_dir / "best.pt")
                patience_left = patience_max
            else:
                patience_left -= 1

            dt = time.time() - t0
            record = {
                "epoch": epoch,
                "in_pretrain": in_pretrain,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": dt,
            }
            history.append(record)
            tracker.log_epoch(record, epoch)
            if verbose:
                marker = "*" if in_pretrain else " "
                print(
                    f"[epoch {epoch:>3d}{marker}] "
                    f"train={train_loss:.4f} val={val_loss:.4f} "
                    f"best_val={best_val:.4f} lr={record['lr']:.2e} "
                    f"({dt:.1f}s)",
                    flush=True,
                )

            if es_enabled and patience_left <= 0:
                if verbose:
                    print(
                        f"early stop at epoch {epoch}: "
                        f"patience exhausted (best={best_val:.4f} @ epoch {best_epoch})",
                        flush=True,
                    )
                break
    finally:
        tracker.close()

    summary = {
        "best_val_nll": best_val,
        "best_epoch": best_epoch,
        "epochs_trained": history[-1]["epoch"] + 1 if history else 0,
        "history": history,
        "checkpoint_dir": str(out_dir),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Evaluate (a thin "load + val composite NLL" helper; the full per-stat MAE
# evaluator with reliability plots lives in evaluate.py / scripts/evaluate.py).
# ---------------------------------------------------------------------------


def evaluate(
    *,
    checkpoint: str | Path,
    parquet: str | Path,
    train_parquet: str | Path,
    train_config: dict[str, Any],
    device: str = "auto",
) -> dict[str, float]:
    """Load a checkpoint and compute composite NLL on a parquet split.

    ``train_parquet`` is needed so the val/test dataset can re-use the
    train-fitted ``player_id_map`` + ``feature_stats`` — same discipline
    as the training loop, no test-set leakage into normalization stats.
    """
    dev = _resolve_device(device)
    model = HierarchicalBoxScoreModel.from_checkpoint(checkpoint).to(dev).eval()

    train_ds = BoxScoreDataset(train_parquet)
    eval_ds = BoxScoreDataset(
        parquet,
        player_id_map=train_ds.player_id_map,
        feature_stats=train_ds.feature_stats,
    )
    loader = DataLoader(
        eval_ds,
        batch_size=int(train_config["loop"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_games,
    )
    weights = LossWeights(**train_config["loss_weights"])

    loss_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, dev)
            preds = model(batch)
            deltas = model.embedding_deltas_for_batch(batch)
            out = composite_nll(
                preds, batch, weights, embedding_deltas=deltas
            )
            B = batch["pace"].shape[0]
            loss_sum += float(out["loss"].item()) * B
            n += B
    return {"composite_nll": loss_sum / max(1, n), "n_games": n}


# ---------------------------------------------------------------------------
# YAML entry point
# ---------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def train_from_yaml(
    train_yaml: str | Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive :func:`train` from a single ``configs/train.yaml`` path.

    Resolves ``model_config`` and ``data_config`` references in
    ``train.yaml`` relative to the repo root (``train.yaml``'s grandparent).
    Then derives the processed parquet paths from
    ``data.yaml:paths.processed``.
    """
    train_yaml = Path(train_yaml).resolve()
    train_cfg = _load_yaml(train_yaml)
    if overrides:
        train_cfg = _deep_merge(train_cfg, overrides)
    # configs/train.yaml → repo root is two parents up.
    repo_root = train_yaml.parent.parent

    model_cfg = _load_yaml(_resolve_path(train_cfg["model_config"], repo_root))
    data_cfg = _load_yaml(_resolve_path(train_cfg["data_config"], repo_root))

    processed_dir = _resolve_path(data_cfg["paths"]["processed"], repo_root)
    return train(
        train_parquet=processed_dir / "train.parquet",
        val_parquet=processed_dir / "val.parquet",
        model_config=model_cfg,
        train_config=train_cfg,
        out_dir=_resolve_path(train_cfg["paths"]["checkpoint_dir"], repo_root),
    )


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the hierarchical NBA NN.")
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to configs/train.yaml.")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override schedule.max_epochs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override run.seed.")
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.max_epochs is not None:
        overrides["schedule"] = {"max_epochs": args.max_epochs}
    if args.seed is not None:
        overrides["run"] = {"seed": args.seed}

    summary = train_from_yaml(args.config, overrides=overrides)
    print(f"best_val_nll={summary['best_val_nll']:.6f} "
          f"@ epoch {summary['best_epoch']} "
          f"({summary['epochs_trained']} trained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
