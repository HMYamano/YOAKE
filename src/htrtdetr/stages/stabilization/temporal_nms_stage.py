"""
temporal_nms_stage.py — L1b: 時系列検出安定化 (ID なし・学習不要)

「物体検出のみ」と「検出+ID」の中間。ID を振らずに検出だけを時系列で安定化する。
やることは3つ:
  1. 持続性フィルタ: 隣接フレームに対応を持たず 1 フレームだけ孤立した検出を誤検出として除去
  2. 短ギャップ補間: t-1 と t+1 に高IoUで対応があるのに t で欠けている箱を線形補間で復元
  3. スコア平滑化: 対応づいた鎖に沿ってスコアを移動平均/EMA で均す (raw_score に原値を残す)

L3 トラッキングとの違いは「窓内・一時的な使い捨て IoU 連結」であること。速度/向きは
出さない (それは L4 の責務)。座標平滑化は既定 OFF (箱位置は補間復元のみ)。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_DET_SCORE, CAP_TEMPORAL_STABLE, Detection, FrameWindow,
)
from ...pipeline.stage import TIER_DETECTION, PipelineContext, StageBase
from .._geometry import iou_xyxy, linear_interp_box


class _Node:
    __slots__ = ("frame", "det", "interp")

    def __init__(self, frame: int, det: Detection, interp: bool = False):
        self.frame = frame
        self.det = det
        self.interp = interp


@register("L1b_stabilization.temporal_nms")
class StabilizationStage(StageBase):
    name = "L1b_stabilization.temporal_nms"
    tier = TIER_DETECTION
    # どの L1 バックエンドの後・L2 以降の前に走る (ワイルドカードが単一の真実源)
    after = frozenset({"L1_detection.*"})
    requires = frozenset({CAP_BOXES, CAP_DET_SCORE})
    provides = frozenset({CAP_TEMPORAL_STABLE})   # マーカー (箱はその場で書き換える)

    def __init__(
        self,
        iou_link: float = 0.5,
        min_persist: int = 2,
        max_gap: int = 1,
        score_smooth: str = "ema",
        smooth_boxes: bool = False,
        ema_alpha: float = 0.5,
    ):
        self.iou_link = float(iou_link)
        self.min_persist = int(min_persist)
        self.max_gap = int(max_gap)
        self.score_smooth = score_smooth
        self.smooth_boxes = bool(smooth_boxes)
        self.ema_alpha = float(ema_alpha)

    # ------------------------------------------------------------------ #
    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        frames = window.frames
        chains = self._build_chains(frames)

        removals: Dict[int, set] = {fr.frame_index: set() for fr in frames}
        frame_by_idx = {fr.frame_index: fr for fr in frames}

        for chain in chains:
            real_nodes = [n for n in chain if not n.interp]
            # 1) 持続性フィルタ
            if len(real_nodes) < self.min_persist:
                for n in real_nodes:
                    removals[n.frame].add(id(n.det))
                continue
            # 2) 短ギャップ補間
            self._interpolate_gaps(chain, frame_by_idx)
            # 3) スコア平滑化 (+ 任意で座標平滑化)
            self._smooth(chain)

        # 削除を適用
        for fr in frames:
            drop = removals[fr.frame_index]
            if drop:
                fr.detections = [d for d in fr.detections if id(d) not in drop]

        window.available.add(CAP_TEMPORAL_STABLE)
        return window

    # ------------------------------------------------------------------ #
    def _build_chains(self, frames: List) -> List[List[_Node]]:
        """max_gap までの先読みを許す IoU 連結で鎖 (使い捨てトラック) を作る。"""
        ordered = sorted(frames, key=lambda f: f.frame_index)
        active: List[List[_Node]] = []
        finished: List[List[_Node]] = []

        for fr in ordered:
            t = fr.frame_index
            # 期限切れの鎖を閉じる
            still: List[List[_Node]] = []
            for ch in active:
                if t - ch[-1].frame > self.max_gap + 1:
                    finished.append(ch)
                else:
                    still.append(ch)
            active = still

            dets = list(fr.detections)
            assigned = [False] * len(dets)
            # 各鎖を最良の検出に貪欲マッチ
            for ch in active:
                last = ch[-1].det
                best_j, best_iou = -1, self.iou_link
                for j, d in enumerate(dets):
                    if assigned[j] or d.class_id != last.class_id:
                        continue
                    v = iou_xyxy(last.bbox, d.bbox)
                    if v >= best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0:
                    assigned[best_j] = True
                    ch.append(_Node(t, dets[best_j]))
            # 未割り当て検出は新しい鎖
            for j, d in enumerate(dets):
                if not assigned[j]:
                    active.append([_Node(t, d)])

        finished.extend(active)
        return finished

    def _interpolate_gaps(self, chain: List[_Node], frame_by_idx: Dict) -> None:
        real = [n for n in chain if not n.interp]
        real.sort(key=lambda n: n.frame)
        new_nodes: List[_Node] = []
        for a, b in zip(real, real[1:]):
            gap = b.frame - a.frame - 1
            if 1 <= gap <= self.max_gap:
                for k in range(1, gap + 1):
                    t = a.frame + k
                    fr = frame_by_idx.get(t)
                    if fr is None:
                        continue
                    frac = k / (gap + 1)
                    box = linear_interp_box(a.det.bbox, b.det.bbox, frac)
                    d = Detection(
                        bbox=box,
                        score=float(min(a.det.score, b.det.score)),
                        class_id=int(a.det.class_id),
                        interpolated=True,
                        raw_score=None,
                    )
                    fr.detections.append(d)
                    new_nodes.append(_Node(t, d, interp=True))
        chain.extend(new_nodes)
        chain.sort(key=lambda n: n.frame)

    def _smooth(self, chain: List[_Node]) -> None:
        nodes = sorted(chain, key=lambda n: n.frame)
        # スコア平滑化
        if self.score_smooth and self.score_smooth != "none":
            scores = [n.det.score for n in nodes]
            for n in nodes:
                if n.det.raw_score is None:
                    n.det.raw_score = float(n.det.score)
            if self.score_smooth == "mean":
                m = float(np.mean(scores))
                for n in nodes:
                    n.det.score = m
            elif self.score_smooth == "ema":
                acc: Optional[float] = None
                a = self.ema_alpha
                for n in nodes:
                    acc = n.det.score if acc is None else a * n.det.score + (1 - a) * acc
                    n.det.score = float(acc)
        # 任意: 座標平滑化 (速度/向きは出さない)
        if self.smooth_boxes and len(nodes) >= 3:
            boxes = np.stack([n.det.bbox.astype(np.float32) for n in nodes], axis=0)
            sm = boxes.copy()
            sm[1:-1] = (boxes[:-2] + boxes[1:-1] + boxes[2:]) / 3.0
            for n, b in zip(nodes, sm):
                if not n.interp:
                    n.det.bbox = b
