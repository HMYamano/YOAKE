"""
Evaluation helpers for detection, action classification, tracking, and runtime.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils.misc import box_iou, cxcywh_to_xyxy


class DetectionEvaluator:
    """
    Detection metrics: class-aware global AP, recall, and center error.

    AP/recall are computed with class-aware matching, but the returned scores are
    aggregate metrics across all classes rather than per-class AP tables.
    """

    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self._predictions: List[Dict[str, Any]] = []
        self._ground_truths: List[Dict[str, Any]] = []

    def update(
        self,
        pred_boxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_classes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_classes: torch.Tensor,
        frame_id: int = 0,
        box_format: str = "cxcywh",
    ) -> None:
        if box_format == "cxcywh":
            pred_xyxy = cxcywh_to_xyxy(pred_boxes) if pred_boxes.numel() > 0 else pred_boxes
            gt_xyxy = cxcywh_to_xyxy(gt_boxes) if gt_boxes.numel() > 0 else gt_boxes
        else:
            pred_xyxy = pred_boxes
            gt_xyxy = gt_boxes

        self._predictions.append(
            {
                "boxes": pred_xyxy.cpu().numpy(),
                "scores": pred_scores.cpu().numpy(),
                "class_ids": pred_classes.cpu().numpy(),
                "frame_id": frame_id,
            }
        )
        self._ground_truths.append(
            {
                "boxes": gt_xyxy.cpu().numpy(),
                "class_ids": gt_classes.cpu().numpy(),
                "frame_id": frame_id,
            }
        )

    def compute(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for iou_thresh in self.iou_thresholds:
            ap, recall = self._compute_ap(iou_thresh)
            results[f"AP{int(iou_thresh * 100)}"] = ap
            results[f"recall@{int(iou_thresh * 100)}"] = recall
        results["center_error"] = self._compute_center_error()
        return results

    def _compute_ap(self, iou_threshold: float) -> Tuple[float, float]:
        all_tp: List[int] = []
        all_fp: List[int] = []
        all_scores: List[float] = []
        total_gt = 0

        for pred, gt in zip(self._predictions, self._ground_truths):
            pred_boxes = pred["boxes"]
            pred_classes = pred["class_ids"]
            gt_boxes = gt["boxes"]
            gt_classes = gt["class_ids"]
            scores = pred["scores"]

            n_gt = len(gt_boxes)
            total_gt += n_gt

            if len(pred_boxes) == 0:
                continue

            if n_gt == 0:
                all_fp.extend([1] * len(pred_boxes))
                all_tp.extend([0] * len(pred_boxes))
                all_scores.extend(scores.tolist())
                continue

            pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            iou_mat = box_iou(pred_t, gt_t).numpy()

            matched_gt = set()
            order = np.argsort(-scores)
            for i in order:
                all_scores.append(float(scores[i]))
                same_class = np.where(gt_classes == pred_classes[i])[0]
                if len(same_class) == 0:
                    all_tp.append(0)
                    all_fp.append(1)
                    continue
                same_class_ious = iou_mat[i, same_class]
                best_local = int(same_class_ious.argmax())
                best_iou = float(same_class_ious[best_local])
                best_j = int(same_class[best_local])
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

        ap = 0.0
        for threshold in np.arange(0, 1.1, 0.1):
            if np.any(recalls >= threshold):
                ap += float(precisions[recalls >= threshold].max())
        ap /= 11.0
        recall = float(recalls[-1]) if len(recalls) > 0 else 0.0
        return float(ap), recall

    def _compute_center_error(self) -> float:
        errors = []
        for pred, gt in zip(self._predictions, self._ground_truths):
            if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
                continue

            pred_t = torch.tensor(pred["boxes"])
            gt_t = torch.tensor(gt["boxes"])
            iou_mat = box_iou(pred_t, gt_t)
            for i in range(len(pred["boxes"])):
                same_class = np.where(gt["class_ids"] == pred["class_ids"][i])[0]
                if len(same_class) == 0:
                    continue
                same_class_ious = iou_mat[i, same_class]
                best_local = int(same_class_ious.argmax().item())
                j = int(same_class[best_local])
                if float(same_class_ious[best_local]) > 0.3:
                    pred_cx = (pred["boxes"][i][0] + pred["boxes"][i][2]) / 2
                    pred_cy = (pred["boxes"][i][1] + pred["boxes"][i][3]) / 2
                    gt_cx = (gt["boxes"][j][0] + gt["boxes"][j][2]) / 2
                    gt_cy = (gt["boxes"][j][1] + gt["boxes"][j][3]) / 2
                    errors.append(np.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2))
        return float(np.mean(errors)) if errors else 0.0

    def reset(self) -> None:
        self._predictions.clear()
        self._ground_truths.clear()


class TrackingEvaluator:
    """A fuller tracking evaluator used outside the training loop."""

    def __init__(self):
        self._pred_tracks: Dict[str, Dict[int, Dict[int, List[float]]]] = defaultdict(lambda: defaultdict(dict))
        self._gt_tracks: Dict[str, Dict[int, Dict[int, List[float]]]] = defaultdict(lambda: defaultdict(dict))

    def update(
        self,
        video_id: str,
        frame_idx: int,
        pred_track_ids: List[int],
        pred_boxes: np.ndarray,
        gt_track_ids: List[int],
        gt_boxes: np.ndarray,
    ) -> None:
        for tid, box in zip(pred_track_ids, pred_boxes):
            self._pred_tracks[video_id][frame_idx][tid] = box.tolist()
        for tid, box in zip(gt_track_ids, gt_boxes):
            self._gt_tracks[video_id][frame_idx][tid] = box.tolist()

    def compute(self) -> Dict[str, float]:
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
        return {
            "IDF1": float(idf1),
            "IDTP": all_idtp,
            "IDFP": all_idfp,
            "IDFN": all_idfn,
            "ID_switches": all_switches,
            "ID_fragments": all_fragments,
        }

    def _compute_video(self, vid: str) -> Tuple[int, int, int, int, int]:
        pred = self._pred_tracks[vid]
        gt = self._gt_tracks[vid]
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
                j = int(iou_mat[i].argmax())
                if float(iou_mat[i, j]) >= 0.5 and j not in matched:
                    idtp += 1
                    matched.add(j)

        idfp = pred_total - idtp
        idfn = gt_total - idtp
        switches = 0
        fragments = 0
        return idtp, idfp, idfn, switches, fragments

    def reset(self) -> None:
        self._pred_tracks.clear()
        self._gt_tracks.clear()


class ActionEvaluator:
    """Action metrics: accuracy, macro F1, per-class F1, confusion matrix."""

    def __init__(self, num_actions: int, action_names: Optional[List[str]] = None):
        self.num_actions = num_actions
        self.action_names = action_names or [str(i) for i in range(num_actions)]
        self._preds: List[int] = []
        self._gts: List[int] = []

    def update(self, pred_action_ids: List[int], gt_action_ids: List[int]) -> None:
        for pred_id, gt_id in zip(pred_action_ids, gt_action_ids):
            if gt_id >= 0:
                self._preds.append(pred_id)
                self._gts.append(gt_id)

    def compute(self) -> Dict[str, Any]:
        if not self._gts:
            return {"accuracy": 0.0, "macro_f1": 0.0}

        preds = np.array(self._preds)
        gts = np.array(self._gts)
        accuracy = float((preds == gts).mean())

        cm = np.zeros((self.num_actions, self.num_actions), dtype=int)
        for pred_id, gt_id in zip(preds, gts):
            if 0 <= gt_id < self.num_actions and 0 <= pred_id < self.num_actions:
                cm[gt_id, pred_id] += 1

        per_class_f1: Dict[str, float] = {}
        f1_sum = 0.0
        n_valid = 0
        for c in range(self.num_actions):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            per_class_f1[self.action_names[c]] = float(f1)
            if int(cm[c, :].sum()) > 0:
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


class RuntimeEvaluator:
    """Runtime metrics: FPS, latency, and GPU memory."""

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


class SimpleTrackingEvaluator:
    """
    Lightweight tracking proxy for training-time validation.

    Inputs are assumed to be spatially matched already. This proxy focuses on
    whether predicted IDs agree with GT IDs and whether the predicted ID tied
    to a GT changes across updates.
    """

    def __init__(self):
        self._idtp = 0
        self._idfp = 0
        self._idfn = 0
        self._idsw = 0
        self._prev_pred_for_gt: Dict[Any, int] = {}

    def update(
        self,
        pred_ids: List[int],
        gt_ids: List[int],
        matched_mask: Optional[List[bool]] = None,
        sequence_key: Optional[Any] = None,
    ) -> None:
        if matched_mask is None:
            pairs = list(zip(pred_ids, gt_ids))
            unmatched_gt_count = max(len(gt_ids) - len(pred_ids), 0)
            unmatched_pred_count = max(len(pred_ids) - len(gt_ids), 0)
        else:
            pairs = [(p, g) for p, g, m in zip(pred_ids, gt_ids, matched_mask) if m]
            unmatched_gt_count = max(len(gt_ids) - len(pairs), 0)
            unmatched_pred_count = sum(1 for m in matched_mask if not m)

        mismatched_pairs = 0
        for pred_id, gt_id in pairs:
            if pred_id == gt_id:
                self._idtp += 1
            else:
                mismatched_pairs += 1

            scoped_gt_id = (sequence_key, gt_id) if sequence_key is not None else gt_id
            prev_pred = self._prev_pred_for_gt.get(scoped_gt_id)
            if prev_pred is not None and prev_pred != pred_id:
                self._idsw += 1
            self._prev_pred_for_gt[scoped_gt_id] = pred_id

        self._idfp += unmatched_pred_count + mismatched_pairs
        self._idfn += unmatched_gt_count + mismatched_pairs

    def compute(self) -> Dict[str, float]:
        denom = max(2 * self._idtp + self._idfp + self._idfn, 1)
        idf1 = (2 * self._idtp) / denom
        idp = self._idtp / max(self._idtp + self._idfp, 1)
        idr = self._idtp / max(self._idtp + self._idfn, 1)
        return {
            "idf1": float(idf1),
            "idp": float(idp),
            "idr": float(idr),
            "IDTP": self._idtp,
            "IDFP": self._idfp,
            "IDFN": self._idfn,
            "IDSW": self._idsw,
        }

    def reset(self) -> None:
        self._idtp = 0
        self._idfp = 0
        self._idfn = 0
        self._idsw = 0
        self._prev_pred_for_gt.clear()


class HTRTDETREvaluator:
    """Container for all evaluators."""

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
        return {
            "detection": self.detection.compute(),
            "tracking": self.tracking.compute(),
            "action": self.action.compute(),
            "runtime": self.runtime.compute(),
        }

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
