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
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        optimizer_cfg=None,
        scheduler_cfg=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg

        # デバイス
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
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
        opt_cfg = optimizer_cfg if optimizer_cfg is not None else OptimizerConfig()
        sch_cfg = scheduler_cfg if scheduler_cfg is not None else SchedulerConfig(
            total_epochs=train_cfg.max_epochs
        )
        self.optimizer = build_optimizer(model, opt_cfg)
        self.scheduler = build_scheduler(self.optimizer, sch_cfg)

        # AMP
        self.scaler = GradScaler('cuda') if train_cfg.use_amp else None

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

    # ------------------------------------------------------------------
    # Ultralytics-style display helpers
    # ------------------------------------------------------------------

    def _loss_col_names(self) -> List[str]:
        """Stage に応じた loss カラム名リストを返す"""
        names = ["box_loss"]
        if self.cfg.stage >= 2:
            names.append("act_loss")
        if self.cfg.stage >= 3:
            names.append("id_loss")
        return names

    def _gpu_mem(self) -> str:
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_reserved() / 1E9:.3g}G"
        return "0G"

    def _print_train_header(self) -> None:
        """エポック開始前にヘッダー行を表示する"""
        cols = ["Epoch", "GPU_mem"] + self._loss_col_names() + ["Instances", "Size"]
        print(("\n" + "%11s" * len(cols)) % tuple(cols))

    def _print_val_results(self, val_metrics: Dict) -> None:
        """バリデーション結果をUltralytics風に表示する"""
        if not val_metrics:
            return
        if "val_AP50" in val_metrics:
            header = "%22s%11s%11s%11s" % ("Class", "Images", "Instances", "AP50")
            n_imgs = len(self.val_loader.dataset) if hasattr(self.val_loader, "dataset") else "-"
            row = "%22s%11s%11s%11.4g" % ("all", n_imgs, "-", val_metrics["val_AP50"])
        else:
            header = "%22s%11s" % ("Class", "val_loss")
            row = "%22s%11.4g" % ("all", val_metrics.get("val_loss", 0.0))
        print(header)
        print(row)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """メインの学習ループ"""
        self.logger.info(
            f"Start training Stage {self.cfg.stage} | "
            f"Epochs: {self.cfg.max_epochs} | Device: {self.device}"
        )

        self._print_train_header()

        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            # --- Train epoch ---
            train_metrics = self._train_epoch(epoch)

            # --- Val epoch ---
            if (epoch + 1) % self.cfg.val_interval == 0:
                val_metrics = self._val_epoch(epoch)
                self._print_val_results(val_metrics)
            else:
                val_metrics = {}

            # --- Scheduler step ---
            if self.scheduler is not None:
                self.scheduler.step()

            # --- Logging ---
            all_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            self.metrics_logger.log(all_metrics, step=epoch)
            self.wandb.log(all_metrics, step=epoch)
            train_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            val_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            self.logger.info(f"Epoch {epoch}/{self.cfg.max_epochs}  [{train_str}]  [{val_str}]")

            # --- Checkpoint ---
            if self.cfg.save_last:
                save_checkpoint(
                    str(Path(self.cfg.output_dir) / "last.pth"),
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=all_metrics,
                )

            # Stage 1: AP50 (高いほど良い) で best を決定
            # その他: val_loss (低いほど良い) で best を決定
            if self.cfg.stage == 1 and "val_AP50" in val_metrics:
                metric_for_save = -val_metrics["val_AP50"]
                metric_display = f"AP50={val_metrics['val_AP50']:.4f}"
            else:
                metric_for_save = val_metrics.get("val_loss", float("inf"))
                metric_display = f"val_loss={metric_for_save:.4f}"

            if self.cfg.save_best and metric_for_save < self.best_metric:
                self.best_metric = metric_for_save
                save_checkpoint(
                    str(Path(self.cfg.output_dir) / f"stage{self.cfg.stage}_best.pth"),
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=all_metrics,
                )
                self.no_improve_count = 0
                self.logger.info(f"[Epoch {epoch}] Best model saved ({metric_display})")
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
        """1 epoch の学習 (Ultralytics 風表示)"""
        self.model.train()
        self.criterion.train()

        meters = {
            "loss": AverageMeter("loss"),
            "loss_detection": AverageMeter("loss_det"),
            "loss_action": AverageMeter("loss_act"),
            "loss_id": AverageMeter("loss_id"),
        }
        epoch_start = time.time()

        loss_cols = self._loss_col_names()  # e.g. ["box_loss", "act_loss"]
        # ヘッダー行のカラム数に合わせた書式
        # Epoch  GPU_mem  [loss_cols...]  Instances  Size
        n_fixed = 2  # Epoch + GPU_mem
        n_trail = 2  # Instances + Size
        fmt_desc = "%11s" * n_fixed + "%11.4g" * len(loss_cols) + "%11s" * n_trail

        # 画像サイズ (最初の batch から推定)
        img_size = "?"

        batch_bar = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(batch_bar):
            batch = move_batch_to_device(batch, self.device)

            if step == 0 and "images" in batch:
                h, w = batch["images"].shape[-2:]
                img_size = f"{h}x{w}"

            self.optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', enabled=(self.scaler is not None)):
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

            # メトリクス更新
            bs = batch["images"].shape[0] if "images" in batch else 1
            n_inst = sum(
                len(t["boxes"]) for t in (batch["targets"] if isinstance(batch["targets"][0], dict)
                                          else [t[-1] for t in batch["targets"]])
            ) if "targets" in batch else 0
            meters["loss"].update(total_loss.item(), bs)
            for key in ("loss_detection", "loss_action", "loss_id"):
                if key in loss_dict:
                    meters[key].update(loss_dict[key].item(), bs)

            # loss_cols の順に平均値を収集
            loss_vals = []
            col_to_meter = {
                "box_loss": "loss_detection",
                "act_loss": "loss_action",
                "id_loss":  "loss_id",
            }
            for col in loss_cols:
                m = meters[col_to_meter[col]]
                loss_vals.append(m.avg if m.count > 0 else 0.0)

            # Ultralytics 風の描画: desc が 1 行分のデータ
            epoch_str = f"{epoch + 1}/{self.cfg.max_epochs}"
            desc = fmt_desc % (
                epoch_str, self._gpu_mem(),
                *loss_vals,
                str(n_inst), img_size,
            )
            batch_bar.set_description(desc)

        batch_bar.close()

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

        # Stage 1 では AP50 も計算する
        from ..evaluation.evaluator import DetectionEvaluator
        from ..utils.misc import cxcywh_to_xyxy as _cxcywh_to_xyxy
        evaluator = DetectionEvaluator(iou_thresholds=[0.5]) if self.cfg.stage == 1 else None

        val_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.max_epochs}   val",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        for batch in val_bar:
            batch = move_batch_to_device(batch, self.device)

            with autocast('cuda', enabled=(self.scaler is not None)):
                if self.cfg.stage == 1:
                    images = batch["images"]
                    targets = batch["targets"]
                    output = self.model.forward_single_frame(images)
                    loss_dict = self.criterion(output, targets, stage=1)

                    # AP50 計算
                    probs = output.pred_logits.softmax(dim=-1)[:, :, :-1]
                    scores, labels = probs.max(dim=-1)
                    pred_xyxy = _cxcywh_to_xyxy(output.pred_boxes)
                    for b in range(images.shape[0]):
                        mask = scores[b] > 0.05
                        evaluator.update(
                            pred_xyxy[b][mask].cpu(),
                            scores[b][mask].cpu(),
                            labels[b][mask].cpu(),
                            targets[b]["boxes"].cpu(),
                            targets[b]["class_ids"].cpu(),
                            box_format="xyxy",
                        )
                else:
                    loss_dict = self._forward_loss(batch)

            loss = loss_dict.get("total_loss", 0.0)
            bs = batch["images"].shape[0] if "images" in batch else 1
            total_loss.update(loss.item() if isinstance(loss, torch.Tensor) else loss, bs)
            val_bar.set_postfix({"val_loss": f"{total_loss.avg:.4f}"})

        val_bar.close()

        result = {"val_loss": total_loss.avg}
        if evaluator is not None:
            det_metrics = evaluator.compute()
            result["val_AP50"] = det_metrics.get("AP50", 0.0)
        return result

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
            if "val_AP50" in val_metrics:
                msg += f" | val_AP50={val_metrics['val_AP50']:.4f}"
        self.logger.info(msg)
