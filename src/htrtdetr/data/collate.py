"""
collate.py — DataLoader 用 collate 関数

設計方針:
- フレームごとに個体数が異なるため、可変長を適切に処理する
- padding ベースと list ベースの両方の出力を提供
- Stage 1 (single frame) と Stage 2-4 (sequence) で異なる collate を使う
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Single frame collate (Stage 1: detection)
# ---------------------------------------------------------------------------

def collate_single_frame(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SingleFrameDataset の出力を batch にまとめる。
    個体数が異なるため、bbox等は list のままにする。

    Returns:
        images: (B, 3, H, W)
        targets: List[Dict] (B 個要素)
          - boxes: (N_i, 4)
          - class_ids: (N_i,)
          - track_ids: (N_i,)
          - action_ids: (N_i,)
        meta: List[Dict]
    """
    images = torch.stack([item["image"] for item in batch], dim=0)

    targets = []
    meta = []
    for item in batch:
        targets.append({
            "boxes": item["boxes"],
            "class_ids": item["class_ids"],
            "track_ids": item["track_ids"],
            "action_ids": item["action_ids"],
        })
        meta.append({
            "frame_index": item["frame_index"],
            "image_path": item["image_path"],
            "orig_size": item["orig_size"],
        })

    return {"images": images, "targets": targets, "meta": meta}


# ---------------------------------------------------------------------------
# Sequence collate (Stage 2-4: temporal / unified)
# ---------------------------------------------------------------------------

def collate_sequence(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    SlidingWindowDataset の出力を batch にまとめる。

    Returns:
        images: (B, T, 3, H, W)
        targets: List[List[Dict]]  (B x T 個要素)
          - 各要素は1フレームの targets dict
        meta: List[Dict]
    """
    images = torch.stack([item["images"] for item in batch], dim=0)  # (B, T, 3, H, W)

    # B x T の targets リスト (個体数はフレームごとに可変)
    all_targets: List[List[Dict]] = []
    for item in batch:
        frame_targets = []
        T = len(item["boxes"])
        for t in range(T):
            frame_targets.append({
                "boxes": item["boxes"][t],
                "class_ids": item["class_ids"][t],
                "track_ids": item["track_ids"][t],
                "action_ids": item["action_ids"][t],
            })
        all_targets.append(frame_targets)

    meta = []
    for item in batch:
        meta.append({
            "video_id": item["video_id"],
            "window_start": item["window_start"],
            "frame_indices": item["frame_indices"],
        })

    return {"images": images, "targets": all_targets, "meta": meta}


# ---------------------------------------------------------------------------
# Padded collate (sequence with padding — 固定長テンソルが必要な場合)
# ---------------------------------------------------------------------------

def collate_sequence_padded(
    batch: List[Dict[str, Any]],
    max_objects: int = 50,
) -> Dict[str, Any]:
    """
    個体数を max_objects に pad する版。
    モデルが固定 shape を期待する場合に使う。

    Returns:
        images: (B, T, 3, H, W)
        boxes: (B, T, max_objects, 4)  — padding は 0
        class_ids: (B, T, max_objects)  — padding は -1
        track_ids: (B, T, max_objects)  — padding は -1
        action_ids: (B, T, max_objects) — padding は -1
        obj_mask: (B, T, max_objects)   — True: valid object
    """
    B = len(batch)
    T = batch[0]["images"].shape[0]
    H, W = batch[0]["images"].shape[-2:]

    images = torch.stack([item["images"] for item in batch], dim=0)

    boxes = torch.zeros(B, T, max_objects, 4)
    class_ids = torch.full((B, T, max_objects), -1, dtype=torch.long)
    track_ids = torch.full((B, T, max_objects), -1, dtype=torch.long)
    action_ids = torch.full((B, T, max_objects), -1, dtype=torch.long)
    obj_mask = torch.zeros(B, T, max_objects, dtype=torch.bool)

    for b, item in enumerate(batch):
        for t in range(T):
            n = item["boxes"][t].shape[0]
            n_fill = min(n, max_objects)
            if n_fill > 0:
                boxes[b, t, :n_fill] = item["boxes"][t][:n_fill]
                class_ids[b, t, :n_fill] = item["class_ids"][t][:n_fill]
                track_ids[b, t, :n_fill] = item["track_ids"][t][:n_fill]
                action_ids[b, t, :n_fill] = item["action_ids"][t][:n_fill]
                obj_mask[b, t, :n_fill] = True

    meta = [
        {
            "video_id": item["video_id"],
            "window_start": item["window_start"],
            "frame_indices": item["frame_indices"],
        }
        for item in batch
    ]

    return {
        "images": images,
        "boxes": boxes,
        "class_ids": class_ids,
        "track_ids": track_ids,
        "action_ids": action_ids,
        "obj_mask": obj_mask,
        "meta": meta,
    }


def get_collate_fn(mode: str = "sequence", max_objects: int = 50):
    """mode に応じた collate 関数を返す"""
    if mode == "single":
        return collate_single_frame
    elif mode == "sequence":
        return collate_sequence
    elif mode == "sequence_padded":
        return lambda batch: collate_sequence_padded(batch, max_objects=max_objects)
    else:
        raise ValueError(f"Unknown collate mode: {mode}")
