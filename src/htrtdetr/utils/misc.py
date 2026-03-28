"""
misc.py — 汎用ユーティリティ

内容:
- seed 固定
- checkpoint 保存・読み込み
- bbox 変換 (xywh ↔ xyxy ↔ cxcywh)
- GIoU 計算
- AverageMeter
"""

from __future__ import annotations

import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 再現性
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Python / NumPy / PyTorch の乱数を固定する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # 古い PyTorch バージョンへの互換性
            pass


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """モデルとトレーニング状態を保存する"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """checkpoint を読み込む。欠損 key は無視して部分 load も可能。"""
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(
        ckpt["model_state"], strict=strict
    )
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt


def load_model_weights(
    path: str,
    model: nn.Module,
    strict: bool = False,
    prefix_to_remove: str = "",
) -> None:
    """model weights のみを読み込む (optimizer 等は無視)"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)  # state dict が直接の場合にも対応
    if prefix_to_remove:
        state = {
            k[len(prefix_to_remove):]: v
            for k, v in state.items()
            if k.startswith(prefix_to_remove)
        }
    model.load_state_dict(state, strict=strict)


# ---------------------------------------------------------------------------
# Bounding box 変換
# ---------------------------------------------------------------------------

def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """[x1, y1, x2, y2] → [cx, cy, w, h]"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[cx, cy, w, h] → [x1, y1, x2, y2]"""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """[x, y, w, h] → [x1, y1, x2, y2]"""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def normalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """pixel 座標を [0, 1] に正規化"""
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return boxes / scale


def denormalize_boxes(boxes: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """[0, 1] 正規化座標を pixel 座標に戻す"""
    scale = boxes.new_tensor([img_w, img_h, img_w, img_h])
    return boxes * scale


# ---------------------------------------------------------------------------
# IoU / GIoU
# ---------------------------------------------------------------------------

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: (N, 4) [x1, y1, x2, y2]
    boxes2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    return inter_area / union_area.clamp(min=1e-6)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    GIoU: https://arxiv.org/abs/1902.09630
    boxes1: (N, 4), boxes2: (M, 4) [x1, y1, x2, y2]
    Returns: (N, M) GIoU matrix
    """
    iou = box_iou(boxes1, boxes2)

    # 最小包囲矩形
    enclose_x1 = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    enclose_y1 = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    enclose_x2 = torch.max(boxes1[:, None, 2], boxes2[None, :, 2])
    enclose_y2 = torch.max(boxes1[:, None, 3], boxes2[None, :, 3])

    enclose_area = (enclose_x2 - enclose_x1).clamp(0) * (
        enclose_y2 - enclose_y1
    ).clamp(0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union_area = area1[:, None] + area2[None, :] - inter_area

    giou = iou - (enclose_area - union_area) / enclose_area.clamp(min=1e-6)
    return giou


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """シンプルな greedy NMS。boxes: (N, 4) [xyxy], scores: (N,)"""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    try:
        from torchvision.ops import nms as tv_nms
        return tv_nms(boxes.float(), scores.float(), iou_threshold)
    except ImportError:
        # torchvision がない場合の手動実装
        _, order = scores.sort(descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            iou_vals = box_iou(boxes[i:i+1], boxes[rest])[0]
            order = rest[iou_vals <= iou_threshold]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ---------------------------------------------------------------------------
# 計測ユーティリティ
# ---------------------------------------------------------------------------

class AverageMeter:
    """学習中の metric 平均を追跡するクラス"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class Timer:
    """コードブロックの実行時間を計測する context manager"""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed = time.perf_counter() - self._start


class GPUMemoryTracker:
    """GPU メモリ使用量を追跡する"""

    @staticmethod
    def current_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / 1024 ** 2

    @staticmethod
    def peak_mb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / 1024 ** 2

    @staticmethod
    def reset_peak() -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# その他
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """モデルのパラメータ数を返す"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_module(module: nn.Module) -> None:
    """モジュールのすべてのパラメータを freeze する"""
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    """モジュールのすべてのパラメータを unfreeze する"""
    for p in module.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Model EMA (YOLOv5 スタイル)
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Exponential Moving Average of model weights.

    YOLOv5 と同じ実装:
      decay(t) = d * (1 - exp(-t / tau))
    で更新ステップ数 t に応じて decay を徐々に大きくし、
    学習初期は強めに更新・後期は安定化させる。

    val では self.ema を使うことで AP50 の振動を大幅に抑制できる。
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float = 2000,
    ) -> None:
        self.ema = deepcopy(model).eval()
        self.updates = 0
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_((1.0 - d) * msd[k].detach())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_state": self.ema.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.ema.load_state_dict(state["ema_state"])
        self.updates = state.get("updates", 0)


def get_device(prefer_gpu: bool = True) -> torch.device:
    """利用可能なデバイスを返す"""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """batch 内の Tensor を全て指定デバイスに移動する (再帰的)"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        converted = [move_batch_to_device(x, device) for x in batch]
        return type(batch)(converted)
    return batch
