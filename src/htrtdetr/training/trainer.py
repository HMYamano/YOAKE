"""
trainer.py — 学習ループ

機能:
- AMP (Automatic Mixed Precision) 対応
- gradient clipping
- checkpoint 保存 (best / last)
- resume
- early stopping
- CSV / JSON ログ保存
- wandb optional
- seed 固定
- staged training: set_stage() で切り替え

使い方:
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ..config.config import TrainConfig, LossConfig
from ..models.ht_rtdetr import HTRTDETR
from ..training.losses import CombinedLoss
from ..training.optimizer import build_optimizer, build_scheduler
from ..utils.misc import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, Timer, GPUMemoryTracker, move_batch_to_device
)
from ..utils.logging import get_logger, MetricsLogger, WandbLogger


class Trainer:
    """
    YOAKE の学習を管理するクラス。

    stage に応じて active な loss / module を切り替える。
    """

    def __init__(
        self,
        model: HTRTDETR,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        loss_cfg: LossConfig,
        num_classes: int = 1,
        num_actions: int = 5,
        max_ids: int = 50,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg

        # デバイス
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # 再現性
        set_seed(train_cfg.seed, train_cfg.deterministic)

        # Loss
        self.criterion = CombinedLoss(
            loss_cfg,
            num_classes=num_classes,
            num_actions=num_actions,
            max_ids=max_ids,
        )
        self.criterion.set_stage(train_cfg.stage)

        # Optimizer & Scheduler
        from ..config.config import OptimizerConfig, SchedulerConfig
        self.optimizer = build_optimizer(model, OptimizerConfig())
        self.scheduler = build_scheduler(
            self.optimizer, SchedulerConfig(total_epochs=train_cfg.max_epochs)
        )

        # AMP
        self.scaler = GradScaler() if train_cfg.use_amp else None

        # ログ
        Path(train_cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(
            "trainer",
            log_file=str(Path(train_cfg.output_dir) / "train.log"),
        )
        self.metrics_logger = MetricsLogger(
            train_cfg.output_dir, prefix=f"stage{train_cfg.stage}_metrics"
        )
        self.wandb = WandbLogger(
            enabled=train_cfg.use_wandb,
            project=train_cfg.wandb_project,
            run_name=train_cfg.wandb_run_name,
        )

        # 状態
        self.start_epoch = 0
        self.best_metric = float("inf")
        self.no_improve_count = 0

        # resume
        if train_cfg.resume:
            self._resume(train_cfg.resume)

    def _resume(self, path: str) -> None:
        self.logger.info(f"Resuming from {path}")
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler,
            strict=False, map_location=str(self.device)
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_metric = ckpt.get("metrics", {}).get("val_loss", float("inf"))
        self.logger.info(f"Resumed at epoch {self.start_epoch}")

    def train(self) -> None:
        """メインの学習ループ"""
        self.logger.info(
            f"Start training Stage {self.cfg.stage} | "
            f"Epochs: {self.cfg.max_epochs} | Device: {self.device}"
        )

        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            # --- Train epoch ---
            train_metrics = self._train_epoch(epoch)

            # --- Val epoch ---
            if (epoch + 1) % self.cfg.val_interval == 0:
                val_metrics = self._val_epoch(epoch)
            else:
                val_metrics = {}

            # --- Scheduler step ---
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Logging ---
            all_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            self.metrics_logger.log(all_metrics, step=epoch)
            self.wandb.log(all_metrics, step=epoch)
            self._log_epoch(epoch, train_metrics, val_metrics)

            # --- Checkpoint ---
            if self.cfg.save_last:
                save_checkpoint(
                    str(Path(self.cfg.output_dir) / "last.pth"),
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=all_metrics,
                )

            val_loss = val_metrics.get("val_loss", float("inf"))
            if self.cfg.save_best and val_loss < self.best_metric:
                self.best_metric = val_loss
                save_checkpoint(
                    str(Path(self.cfg.output_dir) / f"stage{self.cfg.stage}_best.pth"),
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=all_metrics,
                )
                self.no_improve_count = 0
                self.logger.info(f"[Epoch {epoch}] Best model saved (val_loss={val_loss:.4f})")
            else:
                self.no_improve_count += 1

            # --- Early stopping ---
            if self.no_improve_count >= self.cfg.early_stopping_patience:
                self.logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.no_improve_count} epochs)"
                )
                break

        self.wandb.finish()
        self.logger.info("Training completed.")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """1 epoch の学習"""
        self.model.train()
        self.criterion.train()

        meters = {
            "loss": AverageMeter("loss"),
            "loss_detection": AverageMeter("loss_det"),
            "loss_action": AverageMeter("loss_act"),
            "loss_id": AverageMeter("loss_id"),
        }
        epoch_start = time.time()

        for step, batch in enumerate(self.train_loader):
            batch = move_batch_to_device(batch, self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=(self.scaler is not None)):
                loss_dict = self._forward_loss(batch)

            total_loss = loss_dict.get("total_loss", loss_dict.get("loss_detection", 0.0))

            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.__dict__.get("grad_clip_norm", 0.1)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 0.1
                )
                self.optimizer.step()

            # メトリクス更新
            bs = batch["images"].shape[0] if "images" in batch else 1
            meters["loss"].update(total_loss.item(), bs)
            for key in ("loss_detection", "loss_action", "loss_id"):
                if key in loss_dict:
                    meters[key].update(loss_dict[key].item(), bs)

            if (step + 1) % self.cfg.log_interval == 0:
                self.logger.info(
                    f"[Epoch {epoch}][Step {step+1}/{len(self.train_loader)}] "
                    + " | ".join(f"{m}" for m in meters.values() if m.count > 0)
                )

        epoch_time = time.time() - epoch_start
        return {
            "train_loss": meters["loss"].avg,
            "train_loss_detection": meters["loss_detection"].avg,
            "train_loss_action": meters["loss_action"].avg,
            "train_loss_id": meters["loss_id"].avg,
            "epoch_time": epoch_time,
        }

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """1 epoch の検証"""
        self.model.eval()
        self.criterion.eval()

        total_loss = AverageMeter("val_loss")

        for batch in self.val_loader:
            batch = move_batch_to_device(batch, self.device)

            with autocast(enabled=(self.scaler is not None)):
                loss_dict = self._forward_loss(batch)

            loss = loss_dict.get("total_loss", 0.0)
            bs = batch["images"].shape[0] if "images" in batch else 1
            total_loss.update(loss.item() if isinstance(loss, torch.Tensor) else loss, bs)

        return {"val_loss": total_loss.avg}

    def _forward_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """batch を forward して loss を計算する"""
        stage = self.cfg.stage

        if stage == 1:
            # Single frame
            images = batch["images"]        # (B, 3, H, W)
            targets = batch["targets"]      # List[Dict]
            output = self.model.forward_single_frame(images)
            return self.criterion(output, targets, stage=1)

        else:
            # Sequence
            images = batch["images"]        # (B, T, 3, H, W)
            targets = batch["targets"]      # List[List[Dict]]

            output = self.model(images)

            # 最終フレームの targets を使う
            last_targets = [t[-1] for t in targets] if isinstance(targets[0], list) else targets
            return self.criterion(output, last_targets, stage=stage)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict,
        val_metrics: Dict,
    ) -> None:
        msg = (
            f"[Epoch {epoch}/{self.cfg.max_epochs-1}] "
            f"train_loss={train_metrics.get('train_loss', 0):.4f}"
        )
        if val_metrics:
            msg += f" | val_loss={val_metrics.get('val_loss', 0):.4f}"
        self.logger.info(msg)
