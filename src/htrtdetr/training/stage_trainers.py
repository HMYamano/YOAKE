"""
stage_trainers.py — Stage-specific Training Loops  [DEPRECATED]

.. deprecated::
   このモジュールは旧来の段階別 Trainer 実装です。
   正式な Trainer は ``src/htrtdetr/training/trainer.py`` の ``Trainer`` クラスです。

   新しいコードでは以下を使用してください:

     from htrtdetr.training import Trainer

   または CLI から:

     yoake train stage=<1-4>

   このモジュールは互換性のために残してありますが、今後削除される予定です。

各 stage に特化した trainer クラス。
実際に gradient が流れ、checkpoint が保存される形で実装。

Stage 1: Detector のみ (FlyDetectionDataset — single frame)
Stage 2: Action head (TrackWindowDataset — per-track geo features)
Stage 3: ID head    (TrackWindowDataset — per-track geo features)
Stage 4: Unified    (SceneSequenceDataset — full scene + visual)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

from ..config.config import HTRTDETRConfig
from ..utils.misc import AverageMeter, save_checkpoint, load_checkpoint
from .losses import DetectionLoss, ActionLoss, IDLoss, CombinedLoss
from .matching import collect_matched_gt

try:
    from torch.amp import GradScaler, autocast
    _AMP_AVAILABLE = True
except ImportError:
    _AMP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------

class BaseTrainer:
    """共通の学習ループ基底クラス"""

    def __init__(
        self,
        cfg: HTRTDETRConfig,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
    ):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_amp = cfg.train.use_amp and _AMP_AVAILABLE and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_amp else None

        self.start_epoch = 0
        self.best_metric = float("inf")

    def _make_optimizer(self, params=None):
        if params is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
        cfg = self.cfg.optimizer
        if cfg.optimizer.lower() == "adamw":
            return torch.optim.AdamW(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == "sgd":
            return torch.optim.SGD(
                params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
            )
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def _make_scheduler(self, optimizer, num_epochs: int):
        cfg = self.cfg.scheduler
        name = cfg.scheduler.lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=cfg.eta_min
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.step_size, gamma=cfg.gamma
            )
        elif name == "onecycle":
            return None  # set up in train() with loader length
        return None

    def resume(self, checkpoint_path: Optional[str]) -> None:
        """checkpoint から resume する"""
        if checkpoint_path and Path(checkpoint_path).exists():
            state = load_checkpoint(
                checkpoint_path, self.model
            )
            self.start_epoch = state.get("epoch", 0)
            self.best_metric = state.get("best_metric", float("inf"))
            print(f"Resumed from {checkpoint_path} (epoch {self.start_epoch})")
        else:
            if checkpoint_path:
                print(f"Checkpoint not found: {checkpoint_path}, starting fresh")

    def _step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """AMP 対応の backward + optimizer step"""
        optimizer.zero_grad()
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.grad_clip_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.grad_clip_norm)
            optimizer.step()

    def _log_epoch(
        self,
        epoch: int,
        num_epochs: int,
        metrics: Dict[str, float],
        lr: float,
        elapsed: float,
        eta: float,
    ) -> None:
        train_parts = []
        val_parts = []
        for k, v in metrics.items():
            if k.startswith("val_"):
                val_parts.append(f"{k[4:]}={v:.4f}")
            else:
                train_parts.append(f"{k}={v:.4f}")

        mem_str = ""
        if torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            mem_str = f"  GPU={mb:.0f}MB"
            torch.cuda.reset_peak_memory_stats(self.device)

        elapsed_str = _fmt_time(elapsed)
        eta_str = _fmt_time(eta)

        print(
            f"\n[Epoch {epoch:3d}/{num_epochs}]"
            f"  lr={lr:.2e}"
            f"  time={elapsed_str}  ETA={eta_str}"
            f"{mem_str}"
        )
        if train_parts:
            print(f"  Train: " + "  ".join(train_parts))
        if val_parts:
            print(f"  Val  : " + "  ".join(val_parts))
        print()

    def _save(self, epoch: int, metric: float, tag: str = "last") -> None:
        state = {
            "epoch": epoch,
            "best_metric": self.best_metric,
        }
        save_checkpoint(
            str(self.output_dir / f"checkpoint_{tag}.pth"),
            self.model,
            epoch=epoch,
            extra=state,
        )
        if metric < self.best_metric:
            self.best_metric = metric
            save_checkpoint(
                str(self.output_dir / "checkpoint_best.pth"),
                self.model,
                epoch=epoch,
                extra=state,
            )
            print(f"  ★ New best: {metric:.4f}")


# ---------------------------------------------------------------------------
# Stage 1 Trainer — Detector
# ---------------------------------------------------------------------------

class Stage1Trainer(BaseTrainer):
    """
    Stage 1: 空間 Detector の学習。
    入力: 単フレーム画像
    損失: Hungarian matching + class focal + bbox L1 + GIoU
    """

    def __init__(self, cfg, model, device, output_dir):
        super().__init__(cfg, model, device, output_dir)
        self.loss_fn = DetectionLoss(
            cfg.loss,
            num_classes=cfg.model.detector.head.num_classes,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        resume: Optional[str] = None,
    ) -> None:
        self.resume(resume)
        self.model.set_stage(1)
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer, num_epochs)

        print(f"\n{'='*60}")
        print(f"  Stage 1 Training  ({num_epochs} epochs)")
        print(f"  Train batches: {len(train_loader)}"
              + (f"  Val batches: {len(val_loader)}" if val_loader else ""))
        print(f"{'='*60}\n")

        epoch_times = []
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, optimizer, epoch, num_epochs)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, epoch, num_epochs)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            avg_epoch_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])
            remaining = num_epochs - epoch
            eta = avg_epoch_time * remaining

            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._log_epoch(epoch, num_epochs, metrics, optimizer.param_groups[0]["lr"],
                            elapsed, eta)

            loss_for_save = val_metrics.get("loss_detection", train_metrics.get("loss_detection", 0.0))
            self._save(epoch, loss_for_save, tag="last")

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
        epoch: int, num_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}
        total = len(loader)

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [train]", total=total)
        for step, batch in enumerate(pbar, 1):
            images = batch["images"].to(self.device)       # (B, 3, H, W)
            targets = _move_targets_to_device(batch["targets"], self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model.forward_single_frame(images)
                losses = self.loss_fn(out.pred_logits, out.pred_boxes, targets)
                loss = losses["loss_detection"]

            self._step(loss, optimizer)

            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), images.shape[0])

            if _TQDM_AVAILABLE:
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()},
                                 refresh=False)
            elif step % max(1, total // 10) == 0:
                loss_str = "  ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
                print(f"  step {step:4d}/{total}  {loss_str}")

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader,
                   epoch: int = 0, num_epochs: int = 0) -> Dict[str, float]:
        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [val  ]", total=len(loader))
        for batch in pbar:
            images = batch["images"].to(self.device)
            targets = _move_targets_to_device(batch["targets"], self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model.forward_single_frame(images)
                losses = self.loss_fn(out.pred_logits, out.pred_boxes, targets)

            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), images.shape[0])

            if _TQDM_AVAILABLE:
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()},
                                 refresh=False)

        return {k: m.avg for k, m in meters.items()}


# ---------------------------------------------------------------------------
# Stage 2 Trainer — Action Head (geo features)
# ---------------------------------------------------------------------------

class Stage2Trainer(BaseTrainer):
    """
    Stage 2: Action head の学習。
    入力: per-track T フレームの geometric features
    損失: Focal cross-entropy (action classification)
    """

    def __init__(self, cfg, model, device, output_dir):
        super().__init__(cfg, model, device, output_dir)
        self.loss_fn = ActionLoss(
            cfg.loss,
            num_actions=cfg.model.action_head.num_actions,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
        resume: Optional[str] = None,
    ) -> None:
        self.resume(resume)
        self.model.set_stage(2)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._make_optimizer(trainable)
        scheduler = self._make_scheduler(optimizer, num_epochs)

        print(f"\n{'='*60}")
        print(f"  Stage 2 Training  ({num_epochs} epochs)")
        print(f"  Train batches: {len(train_loader)}"
              + (f"  Val batches: {len(val_loader)}" if val_loader else ""))
        print(f"{'='*60}\n")

        epoch_times = []
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, optimizer, epoch, num_epochs)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, epoch, num_epochs)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            eta = sum(epoch_times[-10:]) / len(epoch_times[-10:]) * (num_epochs - epoch)

            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._log_epoch(epoch, num_epochs, metrics, optimizer.param_groups[0]["lr"],
                            elapsed, eta)

            loss_for_save = val_metrics.get("loss", train_metrics.get("loss", 0.0))
            self._save(epoch, loss_for_save, tag="last")

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
        epoch: int, num_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        meter_loss = AverageMeter()
        meter_acc = AverageMeter()
        total = len(loader)

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [train]", total=total)
        for step, batch in enumerate(pbar, 1):
            geo = batch["geo_features"].to(self.device)
            gt_action = batch["action_ids"].to(self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model.forward_geo_sequence(geo)
                action_logits = out["action_logits"]
                loss = self.loss_fn(action_logits, gt_action)

            self._step(loss, optimizer)

            B = geo.shape[0]
            meter_loss.update(loss.item(), B)
            with torch.no_grad():
                acc = _accuracy(action_logits, gt_action)
            meter_acc.update(acc, B)

            if _TQDM_AVAILABLE:
                pbar.set_postfix(loss=f"{meter_loss.avg:.4f}", acc=f"{meter_acc.avg:.4f}",
                                 refresh=False)
            elif step % max(1, total // 10) == 0:
                print(f"  step {step:4d}/{total}  loss={meter_loss.avg:.4f}  acc={meter_acc.avg:.4f}")

        return {"loss": meter_loss.avg, "acc": meter_acc.avg}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader,
                   epoch: int = 0, num_epochs: int = 0) -> Dict[str, float]:
        self.model.eval()
        meter_loss = AverageMeter()
        meter_acc = AverageMeter()

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [val  ]", total=len(loader))
        for batch in pbar:
            geo = batch["geo_features"].to(self.device)
            gt_action = batch["action_labels"].to(self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model.forward_geo_sequence(geo)
                loss = self.loss_fn(out["action_logits"], gt_action)

            B = geo.shape[0]
            meter_loss.update(loss.item(), B)
            meter_acc.update(_accuracy(out["action_logits"], gt_action), B)

            if _TQDM_AVAILABLE:
                pbar.set_postfix(loss=f"{meter_loss.avg:.4f}", acc=f"{meter_acc.avg:.4f}",
                                 refresh=False)

        return {"loss": meter_loss.avg, "acc": meter_acc.avg}


# ---------------------------------------------------------------------------
# Stage 3 Trainer — ID Head (geo features)
# ---------------------------------------------------------------------------

class Stage3Trainer(BaseTrainer):
    """
    Stage 3: Memory-based ID head の学習。
    入力: per-track T フレームの geometric features
    損失: ID classification (focal CE) + optional triplet metric loss
    """

    def __init__(self, cfg, model, device, output_dir):
        super().__init__(cfg, model, device, output_dir)
        max_ids = cfg.model.id_head.max_ids
        self.loss_fn = IDLoss(cfg.loss, max_ids=max_ids)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
        resume: Optional[str] = None,
    ) -> None:
        self.resume(resume)
        self.model.set_stage(3)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = self._make_optimizer(trainable)
        scheduler = self._make_scheduler(optimizer, num_epochs)

        print(f"\n{'='*60}")
        print(f"  Stage 3 Training  ({num_epochs} epochs)")
        print(f"  Train batches: {len(train_loader)}"
              + (f"  Val batches: {len(val_loader)}" if val_loader else ""))
        print(f"{'='*60}\n")

        epoch_times = []
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, optimizer, epoch, num_epochs)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, epoch, num_epochs)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            eta = sum(epoch_times[-10:]) / len(epoch_times[-10:]) * (num_epochs - epoch)

            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._log_epoch(epoch, num_epochs, metrics, optimizer.param_groups[0]["lr"],
                            elapsed, eta)

            loss_for_save = val_metrics.get("loss_id", train_metrics.get("loss_id", 0.0))
            self._save(epoch, loss_for_save, tag="last")

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
        epoch: int, num_epochs: int,
    ) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}
        total = len(loader)

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [train]", total=total)
        for step, batch in enumerate(pbar, 1):
            geo = batch["geo_features"].to(self.device)
            gt_ids = batch["local_track_ids"].to(self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model.forward_geo_sequence(geo)
                id_logits = out["id_logits"]
                id_emb = out["id_embeddings"]
                losses = self.loss_fn(id_logits, gt_ids, id_emb)
                loss = losses["loss_id"]

            self._step(loss, optimizer)

            B = geo.shape[0]
            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), B)

            if _TQDM_AVAILABLE:
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()},
                                 refresh=False)
            elif step % max(1, total // 10) == 0:
                loss_str = "  ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
                print(f"  step {step:4d}/{total}  {loss_str}")

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader,
                   epoch: int = 0, num_epochs: int = 0) -> Dict[str, float]:
        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [val  ]", total=len(loader))
        for batch in pbar:
            geo = batch["geo_features"].to(self.device)
            gt_ids = batch["local_track_ids"].to(self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model.forward_geo_sequence(geo)
                losses = self.loss_fn(out["id_logits"], gt_ids, out["id_embeddings"])

            B = geo.shape[0]
            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), B)

            if _TQDM_AVAILABLE:
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()},
                                 refresh=False)

        return {k: m.avg for k, m in meters.items()}


# ---------------------------------------------------------------------------
# Stage 4 Trainer — Unified Fine-tuning
# ---------------------------------------------------------------------------

class Stage4Trainer(BaseTrainer):
    """
    Stage 4: 全モジュールの統合 fine-tuning。
    入力: シーン全体の T フレームシーケンス (visual + geo features)
    損失: detection + action + ID + temporal smooth
    """

    def __init__(self, cfg, model, device, output_dir):
        super().__init__(cfg, model, device, output_dir)
        self.loss_fn = CombinedLoss(
            cfg.loss,
            num_classes=cfg.model.detector.head.num_classes,
            num_actions=cfg.model.action_head.num_actions,
            max_ids=cfg.model.id_head.max_ids,
        )
        self.loss_fn.set_stage(4)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
        resume: Optional[str] = None,
        stage1_ckpt: Optional[str] = None,
        stage2_ckpt: Optional[str] = None,
        stage3_ckpt: Optional[str] = None,
    ) -> None:
        # 前 stage の checkpoint をロード (resume より優先度低い)
        if stage1_ckpt and Path(stage1_ckpt).exists():
            from ..utils.misc import load_model_weights
            load_model_weights(stage1_ckpt, self.model, strict=False)
            print(f"Loaded stage1 weights from {stage1_ckpt}")
        if stage2_ckpt and Path(stage2_ckpt).exists():
            from ..utils.misc import load_model_weights
            load_model_weights(stage2_ckpt, self.model, strict=False)
            print(f"Loaded stage2 weights from {stage2_ckpt}")
        if stage3_ckpt and Path(stage3_ckpt).exists():
            from ..utils.misc import load_model_weights
            load_model_weights(stage3_ckpt, self.model, strict=False)
            print(f"Loaded stage3 weights from {stage3_ckpt}")

        self.resume(resume)
        self.model.set_stage(4)
        optimizer = self._make_optimizer()
        scheduler = self._make_scheduler(optimizer, num_epochs)

        print(f"\n{'='*60}")
        print(f"  Stage 4 Training  ({num_epochs} epochs)")
        print(f"  Train batches: {len(train_loader)}"
              + (f"  Val batches: {len(val_loader)}" if val_loader else ""))
        print(f"{'='*60}\n")

        epoch_times = []
        for epoch in range(self.start_epoch + 1, num_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader, optimizer, epoch, num_epochs)

            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._val_epoch(val_loader, epoch, num_epochs)

            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            epoch_times.append(elapsed)
            eta = sum(epoch_times[-10:]) / len(epoch_times[-10:]) * (num_epochs - epoch)

            metrics = {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            self._log_epoch(epoch, num_epochs, metrics, optimizer.param_groups[0]["lr"],
                            elapsed, eta)

            loss_for_save = val_metrics.get("total_loss", train_metrics.get("total_loss", 0.0))
            self._save(epoch, loss_for_save, tag="last")

    def _train_epoch(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer,
        epoch: int, num_epochs: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        meters: Dict[str, AverageMeter] = {}
        total = len(loader)

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [train]", total=total)
        for step, batch in enumerate(pbar, 1):
            # SceneSequenceDataset バッチ
            images = batch["images"].to(self.device)     # (B, T, 3, H, W)
            # last-frame targets (detection loss)
            targets = _move_targets_to_device(batch["targets"], self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model(images)

                # Detection loss (last frame)
                det_losses = {}
                if out.pred_logits is not None:
                    from .losses import DetectionLoss
                    if not hasattr(self, "_det_loss"):
                        self._det_loss = DetectionLoss(
                            self.cfg.loss,
                            num_classes=self.cfg.model.detector.head.num_classes,
                        ).to(self.device)
                    det_losses = self._det_loss(out.pred_logits, out.pred_boxes, targets)

                # Action loss (IoU-based GT matching)
                act_losses = {}
                if out.action_logits is not None and out.det_results is not None:
                    gt_actions = collect_matched_gt(
                        out.det_results, targets, "action_ids", iou_threshold=0.5
                    )
                    if gt_actions.numel() > 0:
                        from .losses import ActionLoss
                        if not hasattr(self, "_act_loss"):
                            self._act_loss = ActionLoss(
                                self.cfg.loss,
                                num_actions=self.cfg.model.action_head.num_actions,
                            ).to(self.device)
                        act_losses["loss_action"] = self._act_loss(
                            out.action_logits, gt_actions.to(self.device)
                        )

                # ID loss (IoU-based GT matching)
                id_losses = {}
                if out.id_logits is not None and out.det_results is not None:
                    gt_tracks = collect_matched_gt(
                        out.det_results, targets, "track_ids", iou_threshold=0.5
                    )
                    if gt_tracks.numel() > 0:
                        from .losses import IDLoss
                        if not hasattr(self, "_id_loss"):
                            self._id_loss = IDLoss(
                                self.cfg.loss,
                                max_ids=self.cfg.model.id_head.max_ids,
                            ).to(self.device)
                        id_losses = self._id_loss(
                            out.id_logits,
                            gt_tracks.to(self.device),
                            out.id_embeddings,
                        )

                # Combine
                all_losses = {**det_losses, **act_losses, **id_losses}
                loss = sum(
                    v for k, v in all_losses.items()
                    if k.startswith("loss_") and v.requires_grad
                )
                if isinstance(loss, (int, float)):
                    continue
                all_losses["total_loss"] = loss

            self._step(loss, optimizer)

            B = images.shape[0]
            for k, v in all_losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item() if isinstance(v, torch.Tensor) else v, B)

            if _TQDM_AVAILABLE:
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()},
                                 refresh=False)
            elif step % max(1, total // 10) == 0:
                loss_str = "  ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
                print(f"  step {step:4d}/{total}  {loss_str}")

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader,
                   epoch: int = 0, num_epochs: int = 0) -> Dict[str, float]:
        self.model.eval()
        meters: Dict[str, AverageMeter] = {}

        pbar = _make_pbar(loader, desc=f"Ep {epoch:3d}/{num_epochs} [val  ]", total=len(loader))
        for batch in pbar:
            images = batch["images"].to(self.device)
            targets = _move_targets_to_device(batch["targets"], self.device)

            ctx = autocast("cuda") if self.use_amp else _null_ctx()
            with ctx:
                out = self.model(images)
                if out.pred_logits is not None:
                    if not hasattr(self, "_det_loss"):
                        from .losses import DetectionLoss
                        self._det_loss = DetectionLoss(
                            self.cfg.loss,
                            num_classes=self.cfg.model.detector.head.num_classes,
                        ).to(self.device)
                    losses = self._det_loss(out.pred_logits, out.pred_boxes, targets)
                else:
                    losses = {}

            B = images.shape[0]
            for k, v in losses.items():
                if k not in meters:
                    meters[k] = AverageMeter()
                meters[k].update(v.item(), B)

            if _TQDM_AVAILABLE:
                pbar.set_postfix({k: f"{m.avg:.4f}" for k, m in meters.items()},
                                 refresh=False)

        return {k: m.avg for k, m in meters.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """秒を h:mm:ss 形式に変換"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _make_pbar(loader, desc: str, total: int):
    """tqdm が使える場合はプログレスバー、そうでなければ素のイテレータを返す"""
    if _TQDM_AVAILABLE:
        return tqdm(loader, desc=desc, total=total, ncols=100, leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]")
    return loader


def _move_targets_to_device(targets, device):
    """List[Dict[str, Tensor]] を指定デバイスに移動する"""
    result = []
    for t in targets:
        moved = {}
        for k, v in t.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            else:
                moved[k] = v
        result.append(moved)
    return result


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """ignore_index=-1 を除いた accuracy"""
    valid = labels >= 0
    if valid.sum() == 0:
        return 0.0
    pred = logits[valid].argmax(dim=-1)
    return (pred == labels[valid]).float().mean().item()


class _null_ctx:
    """AMP が使えないときの no-op context manager"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
