"""
bytetrack_stage.py — L3: 外周 ByteTrack 風トラッカ (dense 不要・YOLO と組める)

高スコア検出を先に IoU で既存トラックへ関連付け、残ったトラックを低スコア検出で救済
する二段関連付け。dense_features を必要とせず boxes だけで track_id を振れるので、
非DETR系 (YOLO) の検出とも組み合わせられる。

既定は「1動画=1窓」で窓内を時系列順に処理する。sliding window で使う場合は
ctx.state にトラッカ状態を持ち回る。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import CAP_BOXES, CAP_TRACK_ID, FrameWindow
from ...pipeline.stage import TIER_TRACKING, PipelineContext, StageBase
from .._geometry import iou_xyxy, greedy_match


class _Trk:
    __slots__ = ("id", "box", "score", "class_id", "lost")

    def __init__(self, tid: int, box: np.ndarray, score: float, class_id: int):
        self.id = tid
        self.box = box
        self.score = score
        self.class_id = class_id
        self.lost = 0


@register("L3_tracking.bytetrack")
class ByteTrackStage(StageBase):
    name = "L3_tracking.bytetrack"
    tier = TIER_TRACKING
    requires = frozenset({CAP_BOXES})
    provides = frozenset({CAP_TRACK_ID})

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_iou: float = 0.3,
        track_buffer: int = 30,
        class_aware: bool = True,
    ):
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_iou = float(match_iou)
        self.track_buffer = int(track_buffer)
        self.class_aware = bool(class_aware)

    def reset_state(self, ctx: PipelineContext) -> None:
        # 1動画ごとにトラッカ状態をリセット (track_id を持ち越さない)
        ctx.state.pop(self.name, None)

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        st = ctx.state.setdefault(self.name, {"tracks": [], "next_id": 0})
        tracks: List[_Trk] = st["tracks"]

        for fr in sorted(window.frames, key=lambda f: f.frame_index):
            dets = fr.detections
            high = [i for i, d in enumerate(dets) if d.score >= self.track_thresh]
            low = [i for i, d in enumerate(dets)
                   if self.low_thresh <= d.score < self.track_thresh]

            # --- 一次関連付け: 高スコア検出 ↔ 全トラック ---
            matched_hi, un_tracks, un_hi = self._associate(tracks, dets, high)
            for ti, di in matched_hi:
                self._update(tracks[ti], dets[di])
            # --- 二次関連付け: 残トラック ↔ 低スコア検出 (救済) ---
            rem_tracks = [tracks[i] for i in un_tracks]
            matched_lo, un_rem, _ = self._associate(rem_tracks, dets, low)
            for ti, di in matched_lo:
                self._update(rem_tracks[ti], dets[di])
            # --- 未マッチ高スコア検出 → 新規トラック ---
            for di in un_hi:
                t = _Trk(st["next_id"], dets[di].bbox.copy(),
                         float(dets[di].score), int(dets[di].class_id))
                st["next_id"] += 1
                dets[di].track_id = t.id
                tracks.append(t)
            # --- 未マッチトラック → lost++、バッファ超過で除去 ---
            matched_track_objs = {id(tracks[ti]) for ti, _ in matched_hi} | \
                                 {id(rem_tracks[ti]) for ti, _ in matched_lo}
            survivors: List[_Trk] = []
            for t in tracks:
                if id(t) in matched_track_objs:
                    t.lost = 0
                    survivors.append(t)
                else:
                    t.lost += 1
                    if t.lost <= self.track_buffer:
                        survivors.append(t)
            tracks = survivors
            st["tracks"] = tracks

        window.available.add(CAP_TRACK_ID)
        return window

    def _associate(self, tracks: List[_Trk], dets: List, det_idx: List[int]):
        if not tracks or not det_idx:
            return [], list(range(len(tracks))), list(det_idx)
        cost = np.ones((len(tracks), len(det_idx)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for k, di in enumerate(det_idx):
                if self.class_aware and t.class_id != dets[di].class_id:
                    continue
                cost[i, k] = 1.0 - iou_xyxy(t.box, dets[di].bbox)
        matches, un_t, un_k = greedy_match(cost, max_cost=1.0 - self.match_iou)
        # k は det_idx 内の位置なので実 index へ戻す
        matches = [(i, det_idx[k]) for i, k in matches]
        un_det = [det_idx[k] for k in un_k]
        return matches, un_t, un_det

    def _update(self, t: _Trk, det) -> None:
        t.box = det.bbox.copy()
        t.score = float(det.score)
        t.lost = 0
        det.track_id = t.id


@register("L3_tracking.iou")
class IoUTrackStage(ByteTrackStage):
    """単閾値の素朴 IoU トラッカ (ByteTrack の低スコア救済を無効化した簡易版)。"""
    name = "L3_tracking.iou"

    def __init__(self, match_iou: float = 0.3, track_buffer: int = 30,
                 track_thresh: float = 0.1, class_aware: bool = True):
        super().__init__(track_thresh=track_thresh, low_thresh=track_thresh,
                         match_iou=match_iou, track_buffer=track_buffer,
                         class_aware=class_aware)
