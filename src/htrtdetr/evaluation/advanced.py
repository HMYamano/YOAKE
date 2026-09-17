"""
advanced.py — 追加の評価指標 (rotated AP / MOT / multi-head 行動 / latency)

既存 ``evaluator.py`` の指標はそのまま維持し、本モジュールで補う:

Detection
  * rotated IoU ベースの AP (OBB detector の正当な評価)
  * 面積帯ごとの AP (small / medium / large) — 小個体性能を切り出して見る

Tracking
  * MOTA / IDF1 / ID switches / fragmentations — **実際に数える**
    (既存 TrackingEvaluator は ID switch を 0 固定にしていた)
  * HOTA (近似) — 検出精度と関連付け精度を分けて見る指標

Behavior
  * balanced accuracy / per-class precision-recall-F1
  * multi-head の head ごと評価

Performance
  * 段ごと latency / FPS / VRAM
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RotatedDetectionEvaluator",
    "MOTEvaluator",
    "MultiHeadActionEvaluator",
    "BehaviorEvaluator",
    "PerformanceEvaluator",
]

_EPS = 1e-9

# COCO 準拠の面積帯 (画素^2)。small が小個体検出の評価軸になる。
AREA_RANGES: Dict[str, Tuple[float, float]] = {
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}


def _rotated_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N,5) x (M,5) の rotated IoU 行列。torch 非依存の numpy 実装を使う。"""
    from ..stages._geometry import rotated_iou_np

    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    out = np.zeros((len(pred), len(gt)), dtype=np.float32)
    for i in range(len(pred)):
        for j in range(len(gt)):
            out[i, j] = rotated_iou_np(pred[i], gt[j])
    return out


def _aabb_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N,4) x (M,4) xyxy の IoU 行列。"""
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_g = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    return (inter / np.clip(area_p + area_g - inter, _EPS, None)).astype(np.float32)


def _average_precision(tp: np.ndarray, fp: np.ndarray, n_gt: int) -> float:
    """
    スコア降順に並んだ tp/fp 列から AP を計算する (全点補間)。

    11 点補間より細かく、COCO の連続補間に近い値になる。
    """
    if n_gt == 0 or len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.clip(tp_cum + fp_cum, _EPS, None)

    # precision を単調非増加に均す (標準的な補間)
    precision = np.flip(np.maximum.accumulate(np.flip(precision)))
    # recall の増分で積分
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    return float(np.sum(np.diff(recall) * precision[1:]))


class RotatedDetectionEvaluator:
    """
    OBB 検出の評価。rotated IoU による AP と面積帯別 AP を出す。

    OBB が無い (AABB のみの) 予測 / GT を渡した場合は theta=0 として扱われるので、
    軸平行 detector の評価にもそのまま使える。

    Example::

        ev = RotatedDetectionEvaluator(iou_thresholds=[0.5, 0.75])
        ev.update(pred_obb, pred_scores, pred_classes, gt_obb, gt_classes)
        print(ev.compute())
    """

    def __init__(
        self,
        iou_thresholds: Optional[Sequence[float]] = None,
        use_rotated: bool = True,
        area_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.iou_thresholds = list(iou_thresholds or [0.5, 0.75])
        self.use_rotated = bool(use_rotated)
        self.area_ranges = dict(area_ranges or AREA_RANGES)
        self._frames: List[Dict[str, np.ndarray]] = []

    def update(
        self,
        pred_obb: np.ndarray,      # (N, 5) [cx,cy,w,h,theta] 画素座標
        pred_scores: np.ndarray,   # (N,)
        pred_classes: np.ndarray,  # (N,)
        gt_obb: np.ndarray,        # (M, 5)
        gt_classes: np.ndarray,    # (M,)
    ) -> None:
        self._frames.append({
            "pred_obb": np.asarray(pred_obb, dtype=np.float32).reshape(-1, 5),
            "pred_scores": np.asarray(pred_scores, dtype=np.float32).reshape(-1),
            "pred_classes": np.asarray(pred_classes, dtype=np.int64).reshape(-1),
            "gt_obb": np.asarray(gt_obb, dtype=np.float32).reshape(-1, 5),
            "gt_classes": np.asarray(gt_classes, dtype=np.int64).reshape(-1),
        })

    def compute(self) -> Dict[str, float]:
        results: Dict[str, float] = {}
        for thr in self.iou_thresholds:
            ap = self._ap_at(thr, area=None)
            key = f"{'r' if self.use_rotated else ''}AP{int(thr * 100)}"
            results[key] = ap

        # mAP: 0.50:0.05:0.95 の平均 (COCO 準拠)
        coco_thrs = np.arange(0.5, 0.96, 0.05)
        results[f"{'r' if self.use_rotated else ''}mAP"] = float(
            np.mean([self._ap_at(float(t), area=None) for t in coco_thrs])
        )

        # 面積帯別 AP50 (小個体性能を切り出す)
        for name in self.area_ranges:
            results[f"AP50_{name}"] = self._ap_at(0.5, area=name)
        return results

    def _ap_at(self, iou_thr: float, area: Optional[str]) -> float:
        """指定 IoU 閾値・面積帯での AP (全クラス平均)。"""
        classes = set()
        for f in self._frames:
            classes.update(f["pred_classes"].tolist())
            classes.update(f["gt_classes"].tolist())
        if not classes:
            return 0.0

        aps: List[float] = []
        for cls in sorted(classes):
            tp_all, fp_all, score_all = [], [], []
            n_gt = 0

            for f in self._frames:
                p_mask = f["pred_classes"] == cls
                g_mask = f["gt_classes"] == cls
                pred = f["pred_obb"][p_mask]
                scores = f["pred_scores"][p_mask]
                gt = f["gt_obb"][g_mask]

                if area is not None:
                    lo, hi = self.area_ranges[area]
                    gt_area = gt[:, 2] * gt[:, 3] if len(gt) else np.zeros(0)
                    keep = (gt_area >= lo) & (gt_area < hi)
                    gt = gt[keep]

                n_gt += len(gt)
                if len(pred) == 0:
                    continue

                order = np.argsort(-scores)
                pred, scores = pred[order], scores[order]

                if len(gt) == 0:
                    tp_all.append(np.zeros(len(pred)))
                    fp_all.append(np.ones(len(pred)))
                    score_all.append(scores)
                    continue

                iou = (
                    _rotated_iou_matrix(pred, gt) if self.use_rotated
                    else _aabb_iou_matrix(_obb_to_xyxy(pred), _obb_to_xyxy(gt))
                )
                tp = np.zeros(len(pred))
                fp = np.zeros(len(pred))
                used = set()
                for i in range(len(pred)):
                    j = int(iou[i].argmax())
                    if float(iou[i, j]) >= iou_thr and j not in used:
                        tp[i] = 1
                        used.add(j)
                    else:
                        fp[i] = 1
                tp_all.append(tp)
                fp_all.append(fp)
                score_all.append(scores)

            if n_gt == 0:
                continue
            if not score_all:
                aps.append(0.0)
                continue

            scores = np.concatenate(score_all)
            order = np.argsort(-scores)
            aps.append(_average_precision(
                np.concatenate(tp_all)[order], np.concatenate(fp_all)[order], n_gt
            ))

        return float(np.mean(aps)) if aps else 0.0

    def reset(self) -> None:
        self._frames.clear()


def _obb_to_xyxy(obb: np.ndarray) -> np.ndarray:
    """OBB (N,5) → 外接 AABB (N,4)。"""
    if len(obb) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h, th = obb.T
    c, s = np.cos(th), np.sin(th)
    ex = np.abs(w * c) / 2 + np.abs(h * s) / 2
    ey = np.abs(w * s) / 2 + np.abs(h * c) / 2
    return np.stack([cx - ex, cy - ey, cx + ex, cy + ey], axis=1).astype(np.float32)


class MOTEvaluator:
    """
    MOT 標準指標を実際に数える評価器。

    * MOTA  = 1 - (FN + FP + IDSW) / GT
    * IDF1  = 2·IDTP / (2·IDTP + IDFP + IDFN)
    * IDSW  = 同一 GT track に割り当てられた予測 ID が変わった回数
    * Frag  = GT track の追跡が途切れた回数
    * HOTA  ≈ sqrt(DetA · AssA) の近似

    既存 ``TrackingEvaluator`` は IDSW / Frag を 0 固定にしていたため、
    ID の入れ替わりを評価できなかった。本クラスはフレーム順に GT↔予測の対応を
    追い、実際の切り替わりを数える。
    """

    def __init__(self, iou_threshold: float = 0.5, use_rotated: bool = False):
        self.iou_threshold = float(iou_threshold)
        self.use_rotated = bool(use_rotated)
        # video -> frame -> {track_id: box}
        self._pred: Dict[str, Dict[int, Dict[int, np.ndarray]]] = defaultdict(dict)
        self._gt: Dict[str, Dict[int, Dict[int, np.ndarray]]] = defaultdict(dict)

    def update(
        self,
        video_id: str,
        frame_idx: int,
        pred_track_ids: Sequence[int],
        pred_boxes: np.ndarray,
        gt_track_ids: Sequence[int],
        gt_boxes: np.ndarray,
    ) -> None:
        """boxes は use_rotated=True なら (N,5) OBB、False なら (N,4) xyxy。"""
        self._pred[video_id][frame_idx] = {
            int(t): np.asarray(b, dtype=np.float32)
            for t, b in zip(pred_track_ids, pred_boxes)
        }
        self._gt[video_id][frame_idx] = {
            int(t): np.asarray(b, dtype=np.float32)
            for t, b in zip(gt_track_ids, gt_boxes)
        }

    def compute(self) -> Dict[str, float]:
        tot_gt = tot_fp = tot_fn = tot_idsw = tot_frag = 0
        tot_tp = 0
        idtp = idfp = idfn = 0
        iou_scores: List[float] = []
        assoc_scores: List[float] = []

        for vid in self._gt:
            r = self._compute_video(vid)
            tot_gt += r["n_gt"]
            tot_tp += r["tp"]
            tot_fp += r["fp"]
            tot_fn += r["fn"]
            tot_idsw += r["idsw"]
            tot_frag += r["frag"]
            idtp += r["idtp"]
            idfp += r["idfp"]
            idfn += r["idfn"]
            iou_scores.extend(r["iou"])
            assoc_scores.extend(r["assoc"])

        mota = 1.0 - (tot_fn + tot_fp + tot_idsw) / max(tot_gt, 1)
        idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)
        det_a = tot_tp / max(tot_tp + tot_fp + tot_fn, 1)
        ass_a = float(np.mean(assoc_scores)) if assoc_scores else 0.0
        hota = float(np.sqrt(max(det_a, 0.0) * max(ass_a, 0.0)))

        return {
            "MOTA": float(mota),
            "IDF1": float(idf1),
            "HOTA": hota,
            "DetA": float(det_a),
            "AssA": ass_a,
            "ID_switches": int(tot_idsw),
            "ID_fragments": int(tot_frag),
            "MOTP_iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
            "num_gt": int(tot_gt),
            "TP": int(tot_tp), "FP": int(tot_fp), "FN": int(tot_fn),
        }

    def _compute_video(self, vid: str) -> Dict[str, Any]:
        pred_v = self._pred.get(vid, {})
        gt_v = self._gt[vid]

        last_match: Dict[int, int] = {}      # gt_id -> 直近に割り当てられた pred_id
        was_tracked: Dict[int, bool] = {}    # gt_id -> 前フレームで追跡されていたか
        idsw = frag = 0
        tp = fp = fn = n_gt = 0
        # IDF1 用: (gt_id, pred_id) の共起回数
        cooc: Dict[Tuple[int, int], int] = defaultdict(int)
        gt_count: Dict[int, int] = defaultdict(int)
        pred_count: Dict[int, int] = defaultdict(int)
        iou_scores: List[float] = []
        # HOTA の AssA 用: TP ごとの (gt_id, pred_id)
        tp_pairs: List[Tuple[int, int]] = []

        for fi in sorted(gt_v.keys()):
            gt_f = gt_v[fi]
            pred_f = pred_v.get(fi, {})
            n_gt += len(gt_f)
            for t in gt_f:
                gt_count[t] += 1
            for t in pred_f:
                pred_count[t] += 1

            matches = self._match_frame(gt_f, pred_f, iou_scores)
            tp_pairs.extend(matches)
            matched_gt = {g for g, _ in matches}

            for gid, pid in matches:
                cooc[(gid, pid)] += 1
                prev = last_match.get(gid)
                if prev is not None and prev != pid:
                    idsw += 1
                last_match[gid] = pid
                # 途切れてから再開したら fragmentation
                if was_tracked.get(gid) is False:
                    frag += 1
                was_tracked[gid] = True

            for gid in gt_f:
                if gid not in matched_gt:
                    was_tracked[gid] = False

            tp += len(matches)
            fp += len(pred_f) - len(matches)
            fn += len(gt_f) - len(matches)

        # IDF1: GT track と予測 track の最大共起で 1 対 1 に対応づける
        idtp = self._id_tp(cooc, gt_count, pred_count)
        idfn = sum(gt_count.values()) - idtp
        idfp = sum(pred_count.values()) - idtp

        # --- HOTA の AssA ---
        # TP ごとに、その (gt_id, pred_id) の組がどれだけ一貫しているかを測る:
        #   A(c) = TPA / (TPA + FNA + FPA)
        #     TPA = その組で一致したフレーム数
        #     FNA = その gt が他の pred に付いた or 未検出だったフレーム数
        #     FPA = その pred が他の gt に付いた or 誤検出だったフレーム数
        # ID が入れ替わると TPA が割れて A(c) が下がるので、検出精度とは独立に
        # 「関連付けの正しさ」だけを評価できる (mean IoU では測れない量)。
        assoc: List[float] = []
        for gid, pid in tp_pairs:
            tpa = cooc[(gid, pid)]
            fna = gt_count[gid] - tpa
            fpa = pred_count[pid] - tpa
            assoc.append(tpa / max(tpa + fna + fpa, 1))

        return {
            "n_gt": n_gt, "tp": tp, "fp": fp, "fn": fn,
            "idsw": idsw, "frag": frag,
            "idtp": idtp, "idfp": idfp, "idfn": idfn,
            "iou": iou_scores, "assoc": assoc,
        }

    def _match_frame(
        self,
        gt_f: Dict[int, np.ndarray],
        pred_f: Dict[int, np.ndarray],
        iou_scores: List[float],
    ) -> List[Tuple[int, int]]:
        """1 フレーム内で GT↔予測を IoU 最大で 1 対 1 に対応づける。"""
        if not gt_f or not pred_f:
            return []
        gt_ids = list(gt_f.keys())
        pred_ids = list(pred_f.keys())
        gt_boxes = np.stack([gt_f[g] for g in gt_ids])
        pred_boxes = np.stack([pred_f[p] for p in pred_ids])

        iou = (
            _rotated_iou_matrix(pred_boxes, gt_boxes) if self.use_rotated
            else _aabb_iou_matrix(pred_boxes, gt_boxes)
        )

        matches: List[Tuple[int, int]] = []
        used_g, used_p = set(), set()
        # IoU 降順の貪欲 1 対 1 対応 (Hungarian と実務上ほぼ一致し、依存が要らない)
        for pi, gi in sorted(
            ((i, j) for i in range(len(pred_ids)) for j in range(len(gt_ids))),
            key=lambda x: -iou[x[0], x[1]],
        ):
            if iou[pi, gi] < self.iou_threshold:
                break
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            matches.append((gt_ids[gi], pred_ids[pi]))
            iou_scores.append(float(iou[pi, gi]))
        return matches

    @staticmethod
    def _id_tp(cooc, gt_count, pred_count) -> int:
        """共起が最大になる GT↔予測 track の 1 対 1 対応から IDTP を求める。"""
        pairs = sorted(cooc.items(), key=lambda kv: -kv[1])
        used_g, used_p = set(), set()
        total = 0
        for (gid, pid), n in pairs:
            if gid in used_g or pid in used_p:
                continue
            used_g.add(gid)
            used_p.add(pid)
            total += n
        return total

    def reset(self) -> None:
        self._pred.clear()
        self._gt.clear()


class BehaviorEvaluator:
    """
    行動分類の評価 (balanced accuracy / per-class precision-recall-F1 を追加)。

    既存 ``ActionEvaluator`` の accuracy / macro F1 / confusion matrix に加え、
    クラス不均衡下で意味のある balanced accuracy と、クラスごとの precision /
    recall を出す (動物行動では稀な行動ほど重要なことが多いため)。
    """

    def __init__(self, num_actions: int, action_names: Optional[List[str]] = None):
        self.num_actions = int(num_actions)
        self.action_names = action_names or [str(i) for i in range(num_actions)]
        self._cm = np.zeros((self.num_actions, self.num_actions), dtype=np.int64)

    def update(self, pred_ids: Sequence[int], gt_ids: Sequence[int]) -> None:
        for p, g in zip(pred_ids, gt_ids):
            if 0 <= g < self.num_actions and 0 <= p < self.num_actions:
                self._cm[int(g), int(p)] += 1

    def compute(self) -> Dict[str, Any]:
        cm = self._cm
        total = int(cm.sum())
        if total == 0:
            return {"accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0}

        accuracy = float(np.trace(cm) / total)
        per_class: Dict[str, Dict[str, float]] = {}
        recalls, f1s = [], []

        for c in range(self.num_actions):
            tp = int(cm[c, c])
            fp = int(cm[:, c].sum() - tp)
            fn = int(cm[c, :].sum() - tp)
            support = int(cm[c, :].sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, _EPS)
            per_class[self.action_names[c]] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
            if support > 0:
                recalls.append(recall)
                f1s.append(f1)

        return {
            "accuracy": accuracy,
            # balanced accuracy = クラスごと recall の平均 (不均衡に強い)
            "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

    def reset(self) -> None:
        self._cm[:] = 0


class MultiHeadActionEvaluator:
    """
    multi-head 行動分類の評価。head ごとに独立した ``BehaviorEvaluator`` を持つ。

    Example::

        ev = MultiHeadActionEvaluator({
            "locomotion": ["stationary", "walking", "running"],
            "social": ["none", "courtship", "chase"],
        })
        ev.update("locomotion", preds, gts)
        ev.compute()   # {"locomotion": {...}, "social": {...}, "mean_macro_f1": ...}
    """

    def __init__(self, heads: Dict[str, List[str]]):
        self.heads = {k: list(v) for k, v in heads.items()}
        self._evals = {
            name: BehaviorEvaluator(len(classes), classes)
            for name, classes in self.heads.items()
        }

    def update(self, head: str, pred_ids: Sequence[int], gt_ids: Sequence[int]) -> None:
        if head not in self._evals:
            raise KeyError(f"未知の head '{head}'。既知: {sorted(self._evals)}")
        self._evals[head].update(pred_ids, gt_ids)

    def compute(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {name: ev.compute() for name, ev in self._evals.items()}
        f1s = [v["macro_f1"] for v in out.values() if isinstance(v, dict)]
        accs = [v["accuracy"] for v in out.values() if isinstance(v, dict)]
        out["mean_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
        out["mean_accuracy"] = float(np.mean(accs)) if accs else 0.0
        return out

    def reset(self) -> None:
        for ev in self._evals.values():
            ev.reset()


class PerformanceEvaluator:
    """
    段ごとの latency / FPS / VRAM を集計する。

    ``StreamingInferencer.LatencyBreakdown`` と同じ段名を使うので、streaming と
    batch の内訳を同じ形で比較できる。
    """

    STAGES = ("preprocess", "detector", "tracking", "temporal", "behavior", "total")

    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = {s: [] for s in self.STAGES}
        self._gpu_mb: List[float] = []

    def update(self, timings: Dict[str, float], gpu_mb: float = 0.0) -> None:
        """timings は秒単位の {段名: 経過時間}。"""
        for stage, sec in timings.items():
            if stage in self._samples:
                self._samples[stage].append(float(sec))
        if gpu_mb:
            self._gpu_mb.append(float(gpu_mb))

    def compute(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for stage, vals in self._samples.items():
            if not vals:
                continue
            arr = np.asarray(vals) * 1000.0
            out[f"{stage}_mean_ms"] = float(arr.mean())
            out[f"{stage}_p95_ms"] = float(np.percentile(arr, 95))

        total = self._samples.get("total") or []
        if total:
            out["fps"] = float(1.0 / max(np.mean(total), _EPS))
            out["frames"] = len(total)
        if self._gpu_mb:
            out["mean_gpu_mb"] = float(np.mean(self._gpu_mb))
            out["peak_gpu_mb"] = float(np.max(self._gpu_mb))
        return out

    def reset(self) -> None:
        for v in self._samples.values():
            v.clear()
        self._gpu_mb.clear()
