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

from contextlib import nullcontext
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config.config import TrainConfig, LossConfig
from ..models.ht_rtdetr import HTRTDETR
from ..training.losses import CombinedLoss
from ..training.metrics import select_best_metric, is_better, init_best_val
from ..training.optimizer import build_optimizer, build_scheduler
from ..utils.misc import (
    set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, Timer, GPUMemoryTracker, move_batch_to_device,
    ModelEMA,
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
        use_dummy: bool = False,
        metrics_cfg=None,
        full_cfg: Optional[Any] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.use_dummy = use_dummy
        self.full_cfg = full_cfg

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
        sch_cfg = scheduler_cfg if scheduler_cfg is not None else SchedulerConfig()
        self.optimizer = build_optimizer(model, opt_cfg)

        # ステップ単位 warmup を Trainer 自身が管理するため、
        # Scheduler には warmup を持たせない (skip_warmup=True)。
        # T_max を warmup 後の残りエポック数に合わせる。
        # total_epochs は常に max_epochs と同期させる（YAML 側の値がズレていても正しく動く）。
        from dataclasses import replace as _dc_replace
        sch_cfg = _dc_replace(sch_cfg, total_epochs=train_cfg.max_epochs)

        self._warmup_epochs: int = sch_cfg.warmup_epochs
        self._warmup_iters: Optional[int] = None  # _train_epoch 初回呼び出し時に確定
        self._base_lrs: List[float] = [pg["lr"] for pg in self.optimizer.param_groups]

        sch_cfg_no_warmup = _dc_replace(
            sch_cfg,
            warmup_epochs=0,
            total_epochs=max(sch_cfg.total_epochs - sch_cfg.warmup_epochs, 1),
        )
        self.scheduler = build_scheduler(self.optimizer, sch_cfg_no_warmup, skip_warmup=True)

        # EMA (YOLOv5 スタイル) — val AP50 の安定化に必須
        use_ema = getattr(train_cfg, "use_ema", True)
        ema_decay = getattr(train_cfg, "ema_decay", 0.9999)
        self.ema: Optional[ModelEMA] = ModelEMA(model, decay=ema_decay) if use_ema else None

        # AMP
        self.scaler = GradScaler('cuda') if train_cfg.use_amp else None

        # MetricsConfig (ベスト指標選択用)
        self.metrics_cfg = metrics_cfg
        self.num_actions = num_actions

        # ログ / ディレクトリ
        out = Path(train_cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "weights").mkdir(exist_ok=True)
        (out / "val").mkdir(exist_ok=True)

        self.logger = get_logger(
            "trainer",
            log_file=str(out / "train.log"),
        )
        self.metrics_logger = MetricsLogger(
            train_cfg.output_dir, prefix="results"
        )

        # args.yaml — ハイパーパラメータをテキストで記録
        import dataclasses, yaml as _yaml  # noqa: E401
        args_path = out / "args.yaml"
        if not args_path.exists():
            try:
                cfg_dict = dataclasses.asdict(train_cfg) if dataclasses.is_dataclass(train_cfg) else {}
                args_path.write_text(_yaml.dump(cfg_dict, allow_unicode=True), encoding="utf-8")
            except Exception:
                pass  # yaml 未インストール等でも学習は継続

        # DummyDataset マーカーファイル & W&B run_name プレフィックス
        if use_dummy:
            marker = Path(train_cfg.output_dir) / "DUMMY_MODE"
            marker.write_text(
                "This run used DummyDataset (smoke-test mode).\n"
                "Metrics have NO real meaning.\n"
            )
            self.logger.warning(
                "=" * 60 + "\n"
                "  DUMMY MODE: DummyDataset を使用中。\n"
                "  メトリクスは意味を持ちません。\n"
                "  マーカーファイル: %s\n" % str(marker) +
                "=" * 60
            )

        _run_name = train_cfg.wandb_run_name
        if use_dummy:
            _run_name = f"[DUMMY] {_run_name}" if _run_name else "[DUMMY]"
        self.wandb = WandbLogger(
            enabled=train_cfg.use_wandb,
            project=train_cfg.wandb_project,
            run_name=_run_name,
        )

        # 状態 — ベスト指標の方向性 (higher_is_better) を事前に確定
        _, _mn, _hib = select_best_metric(train_cfg.stage, {}, metrics_cfg)
        self.start_epoch = 0
        self.best_metric = init_best_val(_hib)
        self._higher_is_better = _hib
        self._best_metric_name = _mn
        self.no_improve_count = 0
        self._last_act_eval: Optional[Dict[str, Any]] = None
        self._last_trk_eval: Optional[Dict[str, Any]] = None

        # resume
        if train_cfg.resume:
            self._resume(train_cfg.resume)

    def _checkpoint_extra(self) -> Dict[str, Any]:
        import dataclasses

        extra: Dict[str, Any] = {}
        if self.ema is not None:
            extra["ema_state"] = self.ema.state_dict()
        model_cfg = getattr(self.model, "cfg", None)
        if dataclasses.is_dataclass(model_cfg):
            extra["model_cfg"] = dataclasses.asdict(model_cfg)
        if self.full_cfg is not None and dataclasses.is_dataclass(self.full_cfg):
            extra["config"] = dataclasses.asdict(self.full_cfg)
        return extra

    def _resume(self, path: str) -> None:
        self.logger.info(f"Resuming from {path}")
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler,
            strict=False, map_location=str(self.device)
        )
        self.start_epoch = ckpt.get("epoch", 0) + 1
        saved_m = ckpt.get("metrics", {})
        self.best_metric = saved_m.get(
            self._best_metric_name,
            init_best_val(self._higher_is_better),
        )
        if self.ema is not None and "ema_state" in ckpt.get("extra", {}):
            self.ema.load_state_dict(ckpt["extra"]["ema_state"])
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
        n_imgs = len(self.val_loader.dataset) if hasattr(self.val_loader, "dataset") else "-"
        stage = self.cfg.stage
        if stage == 1 and "val_AP50" in val_metrics:
            header = "%22s%11s%11s%11s" % ("Class", "Images", "Instances", "AP50")
            row = "%22s%11s%11s%11.4g" % ("all", n_imgs, "-", val_metrics["val_AP50"])
        elif stage == 2 and "macro_f1" in val_metrics:
            header = "%22s%11s%11s%11s" % ("Class", "Images", "macro_f1", "accuracy")
            row = "%22s%11s%11.4g%11.4g" % (
                "all", n_imgs,
                val_metrics["macro_f1"], val_metrics.get("val_accuracy", 0.0),
            )
        elif stage == 3 and "idf1" in val_metrics:
            header = "%22s%11s%11s%11s" % ("Class", "Images", "IDF1", "IDSW")
            row = "%22s%11s%11.4g%11g" % (
                "all", n_imgs,
                val_metrics["idf1"], val_metrics.get("IDSW", 0),
            )
        elif stage == 4:
            ap50 = val_metrics.get("val_AP50", 0.0)
            mf1 = val_metrics.get("macro_f1", 0.0)
            idf1 = val_metrics.get("idf1", 0.0)
            from .metrics import compute_composite_score
            composite = compute_composite_score(ap50, mf1, idf1)
            header = "%22s%11s%11s%11s%11s" % ("Class", "Images", "AP50", "macro_f1", "composite")
            row = "%22s%11s%11.4g%11.4g%11.4g" % ("all", n_imgs, ap50, mf1, composite)
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

        # クラス不均衡対策の重み計算 (stage 2/4 のみ)
        if self.cfg.stage in (2, 4):
            self._compute_and_set_class_weights()

        self._print_train_header()

        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            # --- DummyDataset 使用中の警告 (毎エポック) ---
            if self.use_dummy:
                self.logger.warning(
                    f"[Epoch {epoch}/{self.cfg.max_epochs}] "
                    "DUMMY MODE — メトリクスは無意味です (smoke-test only)"
                )

            # --- Train epoch ---
            train_metrics = self._train_epoch(epoch)

            # --- Val epoch ---
            if (epoch + 1) % self.cfg.val_interval == 0:
                val_metrics = self._val_epoch(epoch)
                self._print_val_results(val_metrics)
            else:
                val_metrics = {}

            # --- Scheduler step (warmup 期間は Trainer がステップ単位で LR を管理) ---
            if self.scheduler is not None and epoch >= self._warmup_epochs:
                self.scheduler.step()

            # --- 現在の LR (先頭 param_group) ---
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- ベスト指標 ---
            metric_val, metric_name, higher = select_best_metric(
                self.cfg.stage, val_metrics, self.metrics_cfg
            )
            improved = bool(
                self.cfg.save_best and val_metrics and is_better(metric_val, self.best_metric, higher)
            )
            best_for_outputs = metric_val if improved else self.best_metric

            # composite score (Stage 4)
            composite = 0.0
            if self.cfg.stage == 4 and val_metrics:
                from .metrics import compute_composite_score
                composite = compute_composite_score(
                    val_metrics.get("val_AP50", 0.0),
                    val_metrics.get("macro_f1", 0.0),
                    val_metrics.get("idf1", 0.0),
                )

            # weighted_f1 (Stage 2/4)
            weighted_f1 = 0.0
            if self._last_act_eval is not None:
                pcf = self._last_act_eval.get("per_class_f1", {})
                cm  = self._last_act_eval.get("confusion_matrix", [])
                if pcf and cm:
                    total_sup = sum(sum(row) for row in cm)
                    weighted_f1 = (
                        sum(f1 * sum(cm[i]) for i, f1 in enumerate(pcf.values()))
                        / max(total_sup, 1)
                    )

            # --- results.csv: 全ステージ共通列を揃える ---
            all_metrics: Dict[str, Any] = {
                "epoch":                epoch,
                "train_loss":           train_metrics.get("train_loss", ""),
                "val_loss":             val_metrics.get("val_loss", ""),
                "val_loss_kind":        val_metrics.get("val_loss_kind", ""),
                "ap50":                 val_metrics.get("val_AP50", ""),
                "macro_f1":             val_metrics.get("macro_f1", ""),
                "weighted_f1":          weighted_f1 if weighted_f1 else "",
                "idf1":                 val_metrics.get("idf1", ""),
                "idp":                  val_metrics.get("idp", ""),
                "idr":                  val_metrics.get("idr", ""),
                "idsw":                 val_metrics.get("IDSW", ""),
                "composite_score":      composite if composite else "",
                "lr":                   current_lr,
                "primary_metric":       metric_name if val_metrics else "",
                "primary_metric_value": metric_val  if val_metrics else "",
                "best_so_far":          best_for_outputs if val_metrics else "",
                # internal detail
                "train_loss_det":       train_metrics.get("train_loss_detection", ""),
                "train_loss_act":       train_metrics.get("train_loss_action", ""),
                "train_loss_id":        train_metrics.get("train_loss_id", ""),
                "epoch_time":           train_metrics.get("epoch_time", ""),
            }
            self.metrics_logger.log(all_metrics, step=epoch)
            self.wandb.log(all_metrics, step=epoch)
            train_str = "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()
                                  if isinstance(v, float))
            val_str   = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()
                                  if isinstance(v, float))
            self.logger.info(f"Epoch {epoch}/{self.cfg.max_epochs}  [{train_str}]  [{val_str}]")

            # --- Checkpoint ---
            _ckpt_extra = self._checkpoint_extra()
            _out = Path(self.cfg.output_dir)
            if self.cfg.save_last:
                save_checkpoint(
                    str(_out / "weights" / "last.pth"),
                    self.model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=all_metrics, extra=_ckpt_extra,
                )

            # --- val/epoch_NNN.json ---
            if val_metrics:
                _vj = _out / "val" / f"epoch_{epoch:03d}.json"
                try:
                    _vj.write_text(
                        json.dumps({**val_metrics,
                                    "epoch": epoch,
                                    "primary_metric": metric_name,
                                    "primary_metric_value": float(metric_val),
                                    "composite_score": composite,
                                    "weighted_f1": weighted_f1},
                                   indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            # --- Stage 2: confusion matrix + per-class metrics ---
            if val_metrics and self._last_act_eval is not None:
                from .plots import save_confusion_matrix, save_per_class_metrics
                _cm = self._last_act_eval.get("confusion_matrix", [])
                if _cm:
                    _act_names = (
                        getattr(self.criterion, "action_names", None)
                        or [str(i) for i in range(self.num_actions)]
                    )
                    (_out / "plots").mkdir(exist_ok=True)
                    save_confusion_matrix(
                        _cm, _act_names,
                        str(_out / "plots" / "confusion_matrix_latest.png"),
                        title=f"Confusion Matrix — Stage {self.cfg.stage} epoch {epoch}",
                    )
                    save_per_class_metrics(
                        {**self._last_act_eval, "weighted_f1": weighted_f1},
                        str(_out / "plots"),
                        prefix=f"epoch_{epoch:03d}_",
                    )

            # --- metrics_latest.json ---
            if val_metrics:
                from .plots import save_metrics_latest
                save_metrics_latest(
                    val_metrics={k: v for k, v in val_metrics.items()
                                 if isinstance(v, (int, float, str, bool))},
                    output_dir=str(_out),
                    epoch=epoch,
                    primary_metric_name=metric_name,
                    primary_metric_value=float(metric_val),
                    best_so_far=float(best_for_outputs),
                    higher_is_better=higher,
                    class_counts=getattr(self, "_class_counts", None),
                    imbalance_strategy=getattr(
                        self.criterion.cfg, "imbalance_strategy", None),
                )

            # --- History plots (5 エポックごと + 最終エポック) ---
            if val_metrics and ((epoch + 1) % 5 == 0
                                or epoch == self.cfg.max_epochs - 1):
                from .plots import save_metrics_history
                save_metrics_history(
                    str(_out / "results.csv"), str(_out), stage=self.cfg.stage
                )

            # --- ベスト指標でベストモデルを保存 ---
            if improved:
                self.best_metric = metric_val
                _save_model = self.ema.ema if self.ema is not None else self.model
                save_checkpoint(
                    str(_out / "weights" / "best.pth"),
                    _save_model, self.optimizer, self.scheduler,
                    epoch=epoch, metrics=all_metrics, extra=_ckpt_extra,
                )
                self.no_improve_count = 0
                self.logger.info(
                    f"[Epoch {epoch}] Best model saved ({metric_name}={metric_val:.4f})"
                )
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

        nb = len(self.train_loader)

        # ステップ単位 warmup の総ステップ数を初回に確定
        if self._warmup_iters is None:
            self._warmup_iters = max(self._warmup_epochs * nb, 100)

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
            total=nb,
            dynamic_ncols=True,
            leave=True,
        )

        for step, batch in enumerate(batch_bar):
            batch = move_batch_to_device(batch, self.device)

            if step == 0 and "images" in batch:
                h, w = batch["images"].shape[-2:]
                img_size = f"{h}x{w}"

            # --- ステップ単位 Warmup (YOLOv5 スタイル) ---
            # warmup 期間中は毎ステップ LR を線形スケーリング
            ni = epoch * nb + step  # global step
            if ni < self._warmup_iters:
                lr_factor = (ni + 1) / self._warmup_iters
                for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                    pg["lr"] = base_lr * lr_factor

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

            # --- EMA 更新 ---
            if self.ema is not None:
                self.ema.update(self.model)

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
    def _val_epoch(self, epoch: int) -> Dict[str, Any]:
        """1 epoch の検証 (EMA モデルが有効なときは EMA で評価)。

        ステージ別プライマリ指標:
          Stage 1 → val_AP50 (DetectionEvaluator)
          Stage 2 → macro_f1 (ActionEvaluator)
          Stage 3 → idf1     (SimpleTrackingEvaluator)
          Stage 4 → composite (上記 3 つ全て)
        """
        eval_model = self.ema.ema if self.ema is not None else self.model
        eval_model.eval()
        self.criterion.eval()

        stage = self.cfg.stage
        total_loss = AverageMeter("val_loss")

        from ..evaluation.evaluator import (
            DetectionEvaluator, ActionEvaluator, SimpleTrackingEvaluator,
        )
        from ..utils.misc import cxcywh_to_xyxy as _cxcywh_to_xyxy
        from .matching import assign_gt_to_detections

        det_eval = DetectionEvaluator(iou_thresholds=[0.5]) if stage in (1, 4) else None
        act_eval = ActionEvaluator(num_actions=self.num_actions) if stage in (2, 4) else None
        trk_eval = SimpleTrackingEvaluator() if stage in (3, 4) else None
        action_loss = getattr(self.criterion, "action_loss", None)
        disable_class_weights_for_val = (
            stage in (2, 4) and hasattr(action_loss, "temporarily_disable_class_weights")
        )

        val_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.max_epochs}   val",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

        for batch in val_bar:
            batch = move_batch_to_device(batch, self.device)
            images = batch["images"]
            targets = batch["targets"]

            with (
                action_loss.temporarily_disable_class_weights()
                if disable_class_weights_for_val
                else nullcontext()
            ):
                with autocast('cuda', enabled=(self.scaler is not None)):
                    if stage == 1:
                        output = eval_model.forward_single_frame(images)
                        last_targets = targets
                    else:
                        output = eval_model(images)
                        last_targets = (
                            [t[-1] for t in targets]
                            if isinstance(targets[0], list)
                            else targets
                        )
                    loss_dict = self.criterion(output, last_targets, stage=stage)

            loss = loss_dict.get("total_loss", 0.0)
            bs = images.shape[0]
            total_loss.update(loss.item() if isinstance(loss, torch.Tensor) else loss, bs)

            # ── Detection evaluation (Stage 1, 4) ─────────────────────────
            if det_eval is not None and output.pred_boxes is not None:
                probs = output.pred_logits.softmax(dim=-1)[:, :, :-1]
                scores, labels = probs.max(dim=-1)
                pred_xyxy = _cxcywh_to_xyxy(output.pred_boxes)
                for b in range(bs):
                    mask = scores[b] > 0.05
                    det_eval.update(
                        pred_xyxy[b][mask].cpu(),
                        scores[b][mask].cpu(),
                        labels[b][mask].cpu(),
                        last_targets[b]["boxes"].cpu(),
                        last_targets[b].get("class_ids", torch.zeros(0, dtype=torch.long)).cpu(),
                        box_format="xyxy",
                    )

            # ── Action / ID evaluation (Stage 2, 3, 4) ────────────────────
            if act_eval is not None or trk_eval is not None:
                det_results = output.det_results or [
                    {"boxes": images.new_zeros((0, 4))} for _ in range(bs)
                ]
                action_batches = (
                    output.split_flattened_tensor(output.action_logits)
                    if output.det_results is not None
                    else [None] * bs
                )
                id_batches = (
                    output.split_flattened_tensor(output.id_logits)
                    if output.det_results is not None
                    else [None] * bs
                )

                for b in range(bs):
                    det_b = det_results[b] if b < len(det_results) else {"boxes": images.new_zeros((0, 4))}
                    pred_boxes_b = det_b.get("boxes", images.new_zeros((0, 4)))
                    tgt_b = last_targets[b]

                    # IoU で valid detections → GT ボックスをマッチング
                    asgn = assign_gt_to_detections(pred_boxes_b, tgt_b, iou_threshold=0.5)
                    matched = asgn["matched_gt_idx"] >= 0
                    n_det = int(pred_boxes_b.shape[0])

                    # Action
                    if act_eval is not None and b < len(action_batches):
                        al = action_batches[b]
                        if al is not None:
                            pred_acts = al.argmax(dim=-1)
                            gt_acts = asgn.get(
                                "matched_action_ids",
                                torch.full((n_det,), -1, dtype=torch.long, device=al.device),
                            )
                            valid = matched & (gt_acts >= 0)
                            if valid.any():
                                act_eval.update(
                                    pred_acts[valid].cpu().tolist(),
                                    gt_acts[valid].cpu().tolist(),
                                )

                    # Tracking ID
                    if trk_eval is not None and b < len(id_batches):
                        il = id_batches[b]
                        if il is not None:
                            pred_ids = il.argmax(dim=-1)
                            gt_ids = asgn.get(
                                "matched_track_ids",
                                torch.full((n_det,), -1, dtype=torch.long, device=il.device),
                            )
                            valid = matched & (gt_ids >= 0)
                            if valid.any():
                                sequence_key = None
                                if "meta" in batch and b < len(batch["meta"]):
                                    sequence_key = batch["meta"][b].get("video_id")
                                trk_eval.update(
                                    pred_ids[valid].cpu().tolist(),
                                    gt_ids[valid].cpu().tolist(),
                                    sequence_key=sequence_key,
                                )

            val_bar.set_postfix({"val_loss": f"{total_loss.avg:.4f}"})

        val_bar.close()

        result: Dict[str, Any] = {"val_loss": total_loss.avg, "val_loss_kind": "unweighted"}
        if det_eval is not None:
            dm = det_eval.compute()
            result["val_AP50"] = dm.get("AP50", 0.0)
        if act_eval is not None:
            am = act_eval.compute()
            result["macro_f1"] = am.get("macro_f1", 0.0)
            result["val_accuracy"] = am.get("accuracy", 0.0)
            # 完全な評価結果 (confusion matrix 等) を train() が参照できるよう保持
            self._last_act_eval: Optional[Dict] = am
        else:
            self._last_act_eval = None
        if trk_eval is not None:
            tm = trk_eval.compute()
            result["idf1"] = tm.get("idf1", 0.0)
            result["idp"] = tm.get("idp", 0.0)
            result["idr"] = tm.get("idr", 0.0)
            result["IDSW"] = float(tm.get("IDSW", 0))
            self._last_trk_eval: Optional[Dict] = tm
        else:
            self._last_trk_eval = None
        return result

    # ------------------------------------------------------------------
    # Class imbalance helpers
    # ------------------------------------------------------------------

    def _compute_and_set_class_weights(self) -> None:
        """訓練データの行動クラス頻度から逆頻度重みを計算して ActionLoss に設定する。

        LossConfig.imbalance_strategy == "class_weight" のときのみ実行。
        "sampler" が指定された場合は警告を出して class_weight に fallback する。
        """
        from .losses import ActionLoss

        strategy = getattr(self.criterion.cfg, "imbalance_strategy", "none")

        # sampler は未実装 → class_weight に fallback
        if strategy == "sampler":
            self.logger.warning(
                "imbalance_strategy='sampler' は未実装です。"
                " 'class_weight' に fallback します。"
            )
            strategy = "class_weight"
            self.criterion.cfg.imbalance_strategy = "class_weight"

        if strategy != "class_weight":
            self.logger.info(f"imbalance_strategy={strategy!r} — class weight 計算をスキップ")
            return

        action_loss_mod = getattr(self.criterion, "action_loss", None)
        if not isinstance(action_loss_mod, ActionLoss):
            self.logger.warning("ActionLoss が見つかりません。class weight をスキップします。")
            return

        self.logger.info("行動クラス頻度を集計中 (imbalance_strategy=class_weight)...")
        counts = [0] * self.num_actions
        scanned = 0
        max_scan = 200  # 最大スキャンバッチ数 (速度優先)

        for batch in self.train_loader:
            targets = batch.get("targets", [])
            if targets and isinstance(targets[0], list):
                targets = [t[-1] for t in targets]
            for tgt in targets:
                aids = tgt.get("action_ids", None)
                if aids is not None:
                    for aid in (aids.tolist() if hasattr(aids, "tolist") else list(aids)):
                        if 0 <= int(aid) < self.num_actions:
                            counts[int(aid)] += 1
            scanned += 1
            if scanned >= max_scan:
                break

        self.logger.info(f"  クラス頻度 (先頭 {scanned} バッチ): {counts}")

        if sum(counts) == 0:
            self.logger.warning(
                "  action_ids が見つかりません。class_weight をスキップします。"
                " データに action_ids フィールドが含まれているか確認してください。"
            )
            return

        smoothing = getattr(self.criterion.cfg, "class_weight_smoothing", 1.0)
        clip_min  = getattr(self.criterion.cfg, "class_weight_clip_min",  0.1)
        clip_max  = getattr(self.criterion.cfg, "class_weight_clip_max", 10.0)

        weights = ActionLoss.compute_class_weights(counts, smoothing, clip_min, clip_max)
        action_loss_mod.set_class_weights(weights)
        self._class_counts: List[int] = counts

        self.logger.info(
            f"  class weights 設定: {[round(w, 3) for w in weights.tolist()]}  "
            f"(smoothing={smoothing}, clip=[{clip_min},{clip_max}])"
        )

        # args.yaml に class_counts と strategy を記録
        _out = Path(self.cfg.output_dir)
        try:
            import yaml as _yaml
            args_path = _out / "args.yaml"
            _d = _yaml.safe_load(args_path.read_text("utf-8")) if args_path.exists() else {}
            _d = _d or {}
            _d["class_counts"] = counts
            _d["imbalance_strategy"] = strategy
            _d["class_weights"] = [round(w, 4) for w in weights.tolist()]
            args_path.write_text(_yaml.dump(_d, allow_unicode=True), "utf-8")
        except Exception:
            pass

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
