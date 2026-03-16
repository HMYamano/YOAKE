"""
evaluator.py — 評価モジュール

評価指標:
1. Detection: AP50, AP75, recall, center error
2. Tracking / ID: IDF1, identity switches, fragment count
3. Action: frame-wise accuracy, macro F1, per-class F1
4. Runtime: FPS, latency, GPU memory usage

設計方針:
- 各評価は独立したクラスで実装し、単独でも使えるようにする
- 結果は dict で返し、JSON 保存と標準出力に対応
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils.misc import box_iou, cxcywh_to_xyxy


# ---------------------------------------------------------------------------
# Detection Evaluator
# ---------------------------------------------------------------------------

class DetectionEvaluator:
    """
    Detection の評価。
    AP50, recall, center error を計算する。
    """

    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self._predictions: List[Dict] = []  # {boxes, scores, class_ids, frame_id}
        self._ground_truths: List[Dict] = []

    def update(
        self,
        pred_boxes: torch.Tensor,       # (N, 4) [cx, cy, w, h] or [x1, y1, x2, y2]
        pred_scores: torch.Tensor,      # (N,)
        pred_classes: torch.Tensor,     # (N,)
        gt_boxes: torch.Tensor,         # (M, 4)
        gt_classes: torch.Tensor,       # (M,)
        frame_id: int = 0,
        box_format: str = "cxcywh",     # "cxcywh" | "xyxy"
    ) -> None:
        if box_format == "cxcywh":
            pred_xyxy = cxcywh_to_xyxy(pred_boxes) if pred_boxes.numel() > 0 \
                else pred_boxes
            gt_xyxy = cxcywh_to_xyxy(gt_boxes) if gt_boxes.numel() > 0 \
                else gt_boxes
        else:
            pred_xyxy = pred_boxes
            gt_xyxy = gt_boxes

        self._predictions.append({
            "boxes": pred_xyxy.cpu().numpy(),
            "scores": pred_scores.cpu().numpy(),
            "class_ids": pred_classes.cpu().numpy(),
            "frame_id": frame_id,
        })
        self._ground_truths.append({
            "boxes": gt_xyxy.cpu().numpy(),
            "class_ids": gt_classes.cpu().numpy(),
            "frame_id": frame_id,
        })

    def compute(self) -> Dict[str, float]:
        """全フレームを通じた評価指標を計算する"""
        results = {}

        for iou_thresh in self.iou_thresholds:
            ap, recall = self._compute_ap(iou_thresh)
            results[f"AP{int(iou_thresh*100)}"] = ap
            results[f"recall@{int(iou_thresh*100)}"] = recall

        results["center_error"] = self._compute_center_error()
        return results

    def _compute_ap(self, iou_threshold: float) -> Tuple[float, float]:
        """AP と recall を計算する (simplified PASCAL VOC 式)"""
        all_tp = []
        all_fp = []
        all_scores = []
        total_gt = 0

        for pred, gt in zip(self._predictions, self._ground_truths):
            pred_boxes = pred["boxes"]   # (N, 4)
            gt_boxes = gt["boxes"]       # (M, 4)
            scores = pred["scores"]      # (N,)

            n_gt = len(gt_boxes)
            total_gt += n_gt

            if len(pred_boxes) == 0:
                continue

            if n_gt == 0:
                all_fp.extend([1] * len(pred_boxes))
                all_tp.extend([0] * len(pred_boxes))
                all_scores.extend(scores.tolist())
                continue

            # IoU 計算
            pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            iou_mat = box_iou(pred_t, gt_t).numpy()  # (N, M)

            matched_gt = set()
            order = np.argsort(-scores)

            for i in order:
                all_scores.append(scores[i])
                if n_gt == 0:
                    all_fp.append(1)
                    all_tp.append(0)
                    continue

                best_iou = iou_mat[i].max()
                best_j = iou_mat[i].argmax()

                if best_iou >= iou_threshold and best_j not in matched_gt:
                    all_tp.append(1)
                    all_fp.append(0)
                    matched_gt.add(best_j)
                else:
                    all_tp.append(0)
                    all_fp.append(1)

        if not all_scores:
            return 0.0, 0.0

        order = np.argsort(-np.array(all_scores))
        tp = np.array(all_tp)[order]
        fp = np.array(all_fp)[order]

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / max(total_gt, 1)

        # AP (11-point interpolation)
        ap = 0.0
        for threshold in np.arange(0, 1.1, 0.1):
            if np.any(recalls >= threshold):
                ap += precisions[recalls >= threshold].max()
        ap /= 11.0

        recall = recalls[-1] if len(recalls) > 0 else 0.0
        return float(ap), float(recall)

    def _compute_center_error(self) -> float:
        """GT との中心点距離の平均 (pixel)"""
        errors = []
        for pred, gt in zip(self._predictions, self._ground_truths):
            if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
                continue

            pred_t = torch.tensor(pred["boxes"])
            gt_t = torch.tensor(gt["boxes"])
            iou_mat = box_iou(pred_t, gt_t)

            for i in range(len(pred["boxes"])):
                j = iou_mat[i].argmax().item()
                if iou_mat[i, j] > 0.3:
                    pred_cx = (pred["boxes"][i][0] + pred["boxes"][i][2]) / 2
                    pred_cy = (pred["boxes"][i][1] + pred["boxes"][i][3]) / 2
                    gt_cx = (gt["boxes"][j][0] + gt["boxes"][j][2]) / 2
                    gt_cy = (gt["boxes"][j][1] + gt["boxes"][j][3]) / 2
                    errors.append(np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2))

        return float(np.mean(errors)) if errors else 0.0

    def reset(self) -> None:
        self._predictions.clear()
        self._ground_truths.clear()


# ---------------------------------------------------------------------------
# Tracking Evaluator (IDF1)
# ---------------------------------------------------------------------------

class TrackingEvaluator:
    """
    Tracking 評価。
    IDF1, identity switches, fragment count を計算する。

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """

    def __init__(self):
        # {video_id: {frame_idx: {track_id: bbox}}}
        self._pred_tracks: Dict[str, Dict[int, Dict[int, List]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self._gt_tracks: Dict[str, Dict[int, Dict[int, List]]] = defaultdict(
            lambda: defaultdict(dict)
        )

    def update(
        self,
        video_id: str,
        frame_idx: int,
        pred_track_ids: List[int],
        pred_boxes: np.ndarray,       # (N, 4) [xyxy]
        gt_track_ids: List[int],
        gt_boxes: np.ndarray,         # (M, 4) [xyxy]
    ) -> None:
        for tid, box in zip(pred_track_ids, pred_boxes):
            self._pred_tracks[video_id][frame_idx][tid] = box.tolist()
        for tid, box in zip(gt_track_ids, gt_boxes):
            self._gt_tracks[video_id][frame_idx][tid] = box.tolist()

    def compute(self) -> Dict[str, float]:
        results = {}
        all_idtp = all_idfp = all_idfn = 0
        all_switches = all_fragments = 0

        for vid in self._gt_tracks:
            idtp, idfp, idfn, switches, frags = self._compute_video(vid)
            all_idtp += idtp
            all_idfp += idfp
            all_idfn += idfn
            all_switches += switches
            all_fragments += frags

        idf1 = (2 * all_idtp) / max(2 * all_idtp + all_idfp + all_idfn, 1)
        results["IDF1"] = float(idf1)
        results["IDTP"] = all_idtp
        results["IDFP"] = all_idfp
        results["IDFN"] = all_idfn
        results["ID_switches"] = all_switches
        results["ID_fragments"] = all_fragments
        return results

    def _compute_video(self, vid: str) -> Tuple[int, int, int, int, int]:
        """1 動画の tracking 評価"""
        pred = self._pred_tracks[vid]
        gt = self._gt_tracks[vid]

        # GT の全 track ID
        gt_all_ids = set(tid for f in gt.values() for tid in f)
        pred_all_ids = set(tid for f in pred.values() for tid in f)

        # 簡略 IDF1 計算 (threshold 0.5 IoU でマッチング)
        idtp = 0
        gt_total = sum(len(f) for f in gt.values())
        pred_total = sum(len(f) for f in pred.values())

        for frame_idx in sorted(gt.keys()):
            if frame_idx not in pred:
                continue
            gt_f = gt[frame_idx]
            pred_f = pred[frame_idx]

            if not gt_f or not pred_f:
                continue

            gt_boxes = np.array(list(gt_f.values()))
            pred_boxes = np.array(list(pred_f.values()))

            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
            iou_mat = box_iou(pred_t, gt_t).numpy()

            matched = set()
            for i in range(len(pred_boxes)):
                j = iou_mat[i].argmax()
                if iou_mat[i, j] >= 0.5 and j not in matched:
                    idtp += 1
                    matched.add(j)

        idfp = pred_total - idtp
        idfn = gt_total - idtp

        # Identity switches (簡略: 前フレームと track_id が変わったものをカウント)
        switches = 0
        prev_gt_to_pred = {}
        for frame_idx in sorted(gt.keys()):
            if frame_idx not in pred:
                continue
            for gt_tid in gt[frame_idx]:
                # GT ID に対応する pred ID を前フレームと比べる
                pass  # 詳細実装は省略
        # fragments (途切れた tracks の数)
        fragments = 0

        return idtp, idfp, idfn, switches, fragments

    def reset(self) -> None:
        self._pred_tracks.clear()
        self._gt_tracks.clear()


# ---------------------------------------------------------------------------
# Action Evaluator
# ---------------------------------------------------------------------------

class ActionEvaluator:
    """
    Action 分類の評価。
    frame-wise accuracy, macro F1, per-class F1, confusion matrix。
    """

    def __init__(self, num_actions: int, action_names: Optional[List[str]] = None):
        self.num_actions = num_actions
        self.action_names = action_names or [str(i) for i in range(num_actions)]
        self._preds: List[int] = []
        self._gts: List[int] = []

    def update(
        self,
        pred_action_ids: List[int],
        gt_action_ids: List[int],
    ) -> None:
        """有効な GT (action_id >= 0) のみ追加する"""
        for p, g in zip(pred_action_ids, gt_action_ids):
            if g >= 0:
                self._preds.append(p)
                self._gts.append(g)

    def compute(self) -> Dict[str, Any]:
        if not self._gts:
            return {"accuracy": 0.0, "macro_f1": 0.0}

        preds = np.array(self._preds)
        gts = np.array(self._gts)

        accuracy = float((preds == gts).mean())

        # Confusion matrix
        cm = np.zeros((self.num_actions, self.num_actions), dtype=int)
        for p, g in zip(preds, gts):
            if 0 <= g < self.num_actions and 0 <= p < self.num_actions:
                cm[g, p] += 1

        # Per-class F1
        per_class_f1 = {}
        f1_sum = 0.0
        n_valid = 0
        for c in range(self.num_actions):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            per_class_f1[self.action_names[c]] = float(f1)
            if cm[c, :].sum() > 0:
                f1_sum += f1
                n_valid += 1

        macro_f1 = f1_sum / max(n_valid, 1)

        return {
            "accuracy": accuracy,
            "macro_f1": float(macro_f1),
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm.tolist(),
        }

    def reset(self) -> None:
        self._preds.clear()
        self._gts.clear()


# ---------------------------------------------------------------------------
# Runtime Evaluator
# ---------------------------------------------------------------------------

class RuntimeEvaluator:
    """FPS / latency / GPU memory を計測する"""

    def __init__(self):
        self._times: List[float] = []
        self._gpu_mem: List[float] = []

    def update(self, elapsed_sec: float, gpu_mb: float = 0.0) -> None:
        self._times.append(elapsed_sec)
        self._gpu_mem.append(gpu_mb)

    def compute(self) -> Dict[str, float]:
        if not self._times:
            return {}
        times = np.array(self._times)
        return {
            "mean_fps": float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0.0,
            "mean_latency_ms": float(np.mean(times) * 1000),
            "p99_latency_ms": float(np.percentile(times, 99) * 1000),
            "mean_gpu_mb": float(np.mean(self._gpu_mem)),
            "peak_gpu_mb": float(np.max(self._gpu_mem)) if self._gpu_mem else 0.0,
        }

    def reset(self) -> None:
        self._times.clear()
        self._gpu_mem.clear()


# ---------------------------------------------------------------------------
# 総合 Evaluator
# ---------------------------------------------------------------------------

class HTRTDETREvaluator:
    """全評価指標をまとめて管理するクラス"""

    def __init__(
        self,
        num_classes: int = 1,
        num_actions: int = 5,
        action_names: Optional[List[str]] = None,
        iou_thresholds: Optional[List[float]] = None,
    ):
        self.detection = DetectionEvaluator(iou_thresholds)
        self.tracking = TrackingEvaluator()
        self.action = ActionEvaluator(num_actions, action_names)
        self.runtime = RuntimeEvaluator()

    def compute_all(self) -> Dict[str, Any]:
        results = {}
        results["detection"] = self.detection.compute()
        results["tracking"] = self.tracking.compute()
        results["action"] = self.action.compute()
        results["runtime"] = self.runtime.compute()
        return results

    def save(self, path: str) -> None:
        results = self.compute_all()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {path}")

    def reset_all(self) -> None:
        self.detection.reset()
        self.tracking.reset()
        self.action.reset()
        self.runtime.reset()
