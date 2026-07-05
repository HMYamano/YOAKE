"""
_common.py — 検出段が共有するモデルロード / 前処理 / 変換ヘルパー

学習済み HTRTDETR / RT-DETR detector を checkpoint からロードし、フレーム画像を
detector 入力テンソルへ前処理し、正規化 cxcywh 検出を画素 xyxy の Detection へ変換する。
既存 inference/inferencer.py と同じ前処理 (ImageNet 正規化 + リサイズ) を用いて
JSON パリティを保つ。
"""

from __future__ import annotations

import contextlib
from typing import List, Optional, Tuple

import numpy as np

from ...pipeline.schema import Detection

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_size(image_size_param, cfg) -> Tuple[int, int]:
    """image_size 指定 (int / (H,W) / None→cfg) を (H, W) に正規化する。

    cfg.data.image_size は (H, W) タプル。Inferencer/CLI と同じく非正方でも両次元を保つ。
    """
    s = image_size_param if image_size_param is not None else cfg.data.image_size
    if isinstance(s, int):
        return (int(s), int(s))
    return (int(s[0]), int(s[1]))


def inference_ctx(device: str):
    """Inferencer と同じ推論コンテキスト (no_grad + CUDA autocast) を返す。

    device が cuda のときのみ mixed precision を有効化 (fp16)。CPU では autocast 無効。
    """
    import torch
    try:
        from torch.cuda.amp import autocast
    except Exception:  # pragma: no cover
        from torch.amp import autocast  # type: ignore
    use_amp = str(device).startswith("cuda") and torch.cuda.is_available()
    stack = contextlib.ExitStack()
    stack.enter_context(torch.no_grad())
    stack.enter_context(autocast(enabled=use_amp))
    return stack


def load_htrtdetr(checkpoint: str, device: str):
    """checkpoint から HTRTDETR と復元 config をロードする (stage4, eval)。"""
    import torch

    from ...cli import load_runtime_config_from_checkpoint
    from ...models import build_model
    from ...utils.misc import load_model_weights

    cfg = load_runtime_config_from_checkpoint(checkpoint)
    if cfg is None:
        raise ValueError(
            f"checkpoint '{checkpoint}' に config メタデータがありません。"
            "config 付きで保存された checkpoint が必要です。"
        )
    cfg.model.detector.backbone.pretrained = False
    model = build_model(cfg.model)
    model.set_stage(4)
    load_model_weights(checkpoint, model, strict=False)
    model = model.to(device).eval()
    return model, cfg


def preprocess_frame(image_rgb: np.ndarray, size: Tuple[int, int]):
    """np.ndarray(H,W,3) RGB → 正規化テンソル (1,3,H,W)。size=(H,W)。"""
    import torch

    H, W = size
    try:
        from PIL import Image
        pil = Image.fromarray(image_rgb).resize((W, H), Image.BILINEAR)
        img = np.asarray(pil, dtype=np.float32) / 255.0
    except Exception:  # pragma: no cover
        import cv2
        img = cv2.resize(image_rgb, (W, H)).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def detections_from_adapter(
    det: dict, orig_w: int, orig_h: int,
    with_dense: bool = True,
) -> List[Detection]:
    """
    DetectionFeatureAdapter.extract_valid_detections の 1画像分 dict を Detection 列へ。

    det: {"boxes": (N,4) cxcywh normalized, "scores": (N,), "class_ids": (N,),
          "features": (N,D), "track_ids"?: (N,), "id_scores"?: (N,)}
    """
    import torch

    boxes = det.get("boxes")
    if boxes is None or (hasattr(boxes, "shape") and boxes.shape[0] == 0):
        return []
    boxes = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.asarray(boxes)
    scores = _to_np(det.get("scores"))
    class_ids = _to_np(det.get("class_ids"))
    feats = _to_np(det.get("features")) if with_dense else None
    track_ids = _to_np(det.get("track_ids"))

    out: List[Detection] = []
    for i in range(boxes.shape[0]):
        cx, cy, w, h = boxes[i]
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        d = Detection(
            bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
            score=float(scores[i]) if scores is not None else 1.0,
            class_id=int(class_ids[i]) if class_ids is not None else 0,
        )
        if feats is not None and i < len(feats):
            d.dense_feature = feats[i].astype(np.float32)
        # track_id は L3 (ここでは fused) が付与。raw_score は L1b の所有領域なので触らない。
        if track_ids is not None and i < len(track_ids):
            d.track_id = int(track_ids[i])
        out.append(d)
    return out


def _to_np(x) -> Optional[np.ndarray]:
    import torch
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
