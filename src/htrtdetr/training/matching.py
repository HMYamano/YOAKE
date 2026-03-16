"""
matching.py — GT-Prediction Matching Utilities

学習時に予測 bbox と GT bbox を対応づけるためのユーティリティ。
Stage 4 (unified) で detection 出力に GT action / track ID を割り当てるために使う。

主な用途:
  - det_adapter が返す valid detections に GT 情報を付与
  - IoU ベースの greedy マッチング (Hungarian に比べて軽量)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..utils.misc import box_iou, cxcywh_to_xyxy


def assign_gt_to_detections(
    pred_boxes: torch.Tensor,   # (N_pred, 4) cxcywh normalized
    target: Dict,               # {boxes: (N_gt,4) xyxy, class_ids, track_ids, action_ids}
    iou_threshold: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    各予測 bbox に対して最も IoU の高い GT を割り当てる (greedy)。

    Args:
        pred_boxes: (N_pred, 4) normalized cxcywh
        target: dict with keys 'boxes' (xyxy), optional 'class_ids', 'track_ids', 'action_ids'
        iou_threshold: この IoU 未満は unmatched (-1) として扱う

    Returns:
        dict:
          'matched_gt_idx'   (N_pred,) long  — GT インデックス、-1=unmatched
          'matched_class_ids'  (N_pred,) long  — -1=unmatched  (boxes_only のとき省略)
          'matched_track_ids'  (N_pred,) long
          'matched_action_ids' (N_pred,) long
    """
    N_pred = pred_boxes.shape[0]
    device = pred_boxes.device

    result: Dict[str, torch.Tensor] = {
        "matched_gt_idx": torch.full((N_pred,), -1, dtype=torch.long, device=device),
    }

    gt_boxes = target.get("boxes", None)
    if gt_boxes is None or gt_boxes.shape[0] == 0 or N_pred == 0:
        for key in ("class_ids", "track_ids", "action_ids"):
            if key in target:
                result[f"matched_{key}"] = torch.full(
                    (N_pred,), -1, dtype=torch.long, device=device
                )
        return result

    # Convert pred cxcywh → xyxy
    pred_xyxy = cxcywh_to_xyxy(pred_boxes)   # (N_pred, 4)
    gt_xyxy = gt_boxes.to(device)             # (N_gt, 4) already xyxy

    iou = box_iou(pred_xyxy, gt_xyxy)         # (N_pred, N_gt)
    max_iou, gt_idx = iou.max(dim=1)          # (N_pred,)

    # マッチング: IoU >= threshold のみ有効
    matched_gt_idx = gt_idx.clone()
    matched_gt_idx[max_iou < iou_threshold] = -1
    result["matched_gt_idx"] = matched_gt_idx

    # 各種 GT 属性を割り当て
    for attr_key in ("class_ids", "track_ids", "action_ids"):
        if attr_key not in target:
            continue
        gt_attr = target[attr_key].to(device)  # (N_gt,)
        safe_idx = matched_gt_idx.clamp(min=0)
        matched_attr = gt_attr[safe_idx]
        matched_attr[matched_gt_idx < 0] = -1
        result[f"matched_{attr_key}"] = matched_attr

    return result


def batch_assign_gt_to_detections(
    det_results: List[Dict],   # B 個の検出結果 (各 dict に 'boxes' key)
    targets: List[Dict],       # B 個の GT dict
    iou_threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """
    バッチ全体に対して GT 割り当てを行う。

    Args:
        det_results: List[{boxes: (N_b, 4) cxcywh, features: (N_b, D), ...}]
        targets:     List[{boxes: (N_gt, 4) xyxy, class_ids, track_ids, action_ids}]

    Returns:
        assignments: List[Dict] — 各バッチ要素の割り当て結果
    """
    assignments = []
    for det, tgt in zip(det_results, targets):
        pred_boxes = det.get("boxes", torch.zeros(0, 4))
        assignment = assign_gt_to_detections(pred_boxes, tgt, iou_threshold)
        assignments.append(assignment)
    return assignments


def collect_matched_gt(
    det_results: List[Dict],
    targets: List[Dict],
    key: str,               # 'action_ids' | 'track_ids' | 'class_ids'
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    全バッチの検出に対してマッチした GT 属性を収集して 1D Tensor に concat する。

    Returns:
        gt_labels: (N_total,) long — -1 は unmatched (損失で ignore)
    """
    all_labels = []
    for det, tgt in zip(det_results, targets):
        pred_boxes = det.get("boxes", torch.zeros(0, 4))
        asgn = assign_gt_to_detections(pred_boxes, tgt, iou_threshold)
        matched_key = f"matched_{key}"
        if matched_key in asgn:
            all_labels.append(asgn[matched_key])
        else:
            device = pred_boxes.device
            all_labels.append(
                torch.full((pred_boxes.shape[0],), -1, dtype=torch.long, device=device)
            )
    if not all_labels:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(all_labels, dim=0)
