"""
optimizer.py — Optimizer / Scheduler 生成

設計方針:
- Backbone の lr を他のモジュールより小さくする (backbone_lr_factor)
- AdamW / Adam / SGD に対応
- Cosine / Step / MultiStep scheduler に対応
- warmup を手動で実装
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, LambdaLR

from ..config.config import OptimizerConfig, SchedulerConfig


def build_optimizer(
    model: nn.Module,
    cfg: OptimizerConfig,
) -> torch.optim.Optimizer:
    """
    Backbone に対して小さい lr を設定した optimizer を構築する。
    """
    # Backbone パラメータと non-backbone パラメータを分ける
    backbone_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            "params": other_params,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
        },
    ]
    if backbone_params and cfg.backbone_lr_factor > 0:
        param_groups.append({
            "params": backbone_params,
            "lr": cfg.lr * cfg.backbone_lr_factor,
            "weight_decay": cfg.weight_decay,
        })
    elif backbone_params:
        # backbone_lr_factor = 0 → backbone を完全に freeze
        for p in backbone_params:
            p.requires_grad_(False)

    name = cfg.optimizer.lower()
    if name == "adamw":
        optimizer = AdamW(param_groups)
    elif name == "adam":
        optimizer = Adam(param_groups)
    elif name == "sgd":
        optimizer = SGD(param_groups, momentum=cfg.momentum, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: SchedulerConfig,
    skip_warmup: bool = False,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Scheduler を構築する。warmup は LambdaLR で実装。

    Args:
        skip_warmup: True のとき warmup を無効化する。
                     Trainer がステップ単位 warmup を自前で行う場合に使用。
    """
    name = cfg.scheduler.lower()
    warmup_epochs = 0 if skip_warmup else cfg.warmup_epochs

    def warmup_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        return 1.0

    if name == "none":
        if warmup_epochs > 0:
            return LambdaLR(optimizer, lr_lambda=warmup_lambda)
        return None

    if name == "cosine":
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.total_epochs - warmup_epochs,
            eta_min=cfg.eta_min,
        )
    elif name == "step":
        base_scheduler = StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )
    elif name == "multistep":
        milestones = [m - warmup_epochs for m in cfg.milestones if m > warmup_epochs]
        base_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=cfg.gamma)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    if warmup_epochs > 0:
        return _WarmupScheduler(optimizer, warmup_epochs, base_scheduler)

    return base_scheduler


class _WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    warmup 期間は linear warmup, その後は base_scheduler に従う。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: torch.optim.lr_scheduler._LRScheduler,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            factor = (self.last_epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * factor for base_lr in self.base_lrs]
        # warmup 後は base_scheduler の lr を返す
        return self.base_scheduler.get_last_lr()

    def step(self) -> None:
        if self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step()
        super().step()
