"""
obb_track_stage.py — L3: OBB 対応トラッカ (rotated IoU + 向き一致 + 運動予測)

既存 ``ByteTrackStage`` の二段関連付け (高スコア先行 → 低スコアで救済) をそのまま
受け継ぎつつ、association のコストを OBB 向けに拡張する:

    similarity = w_iou    * rotated IoU      (無ければ軸平行 IoU へ自動 fallback)
               + w_orient * 体軸向きの一致度
               + w_motion * 等速度予測との一致度

細長い個体が接近して並ぶ場面では、軸平行 IoU だと隣の個体の箱と大きく重なって
しまい ID が入れ替わりやすい。回転を考慮すると重なりが実体に即した値になるので、
分離性能が上がる。向き一致は「すれ違う 2 個体」の取り違えを減らす。

torch 非依存 (numpy のみ) なので、YOLO-OBB など外部検出器とも組める。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import CAP_BOXES, CAP_TRACK_ID, FrameWindow
from ...pipeline.stage import TIER_TRACKING, PipelineContext, StageBase
from .._geometry import (
    greedy_match,
    iou_xyxy,
    orientation_consistency_np,
    rotated_iou_np,
)


class _ObbTrack:
    """OBB トラックの内部状態。"""

    __slots__ = ("id", "obb", "bbox", "score", "class_id", "lost", "velocity")

    def __init__(self, tid: int, obb: np.ndarray, bbox: np.ndarray,
                 score: float, class_id: int):
        self.id = tid
        self.obb = obb
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.lost = 0
        self.velocity = np.zeros(2, dtype=np.float32)

    def predict(self) -> np.ndarray:
        """等速度で次フレームの OBB を予測する。"""
        p = self.obb.copy()
        p[:2] = p[:2] + self.velocity
        return p


@register("L3_tracking.obb")
class OBBTrackStage(StageBase):
    """
    OBB 対応の二段関連付けトラッカ。

    Args:
        track_thresh: 一次関連付けに使う検出スコアの下限
        low_thresh: 二次関連付け (救済) の下限
        match_thresh: 関連付け成立に必要な類似度
        track_buffer: 消失を許容するフレーム数
        iou_weight / orientation_weight / motion_weight: コストの重み
        use_rotated_iou: OBB があれば rotated IoU を使う
        class_aware: 異なるクラス間の関連付けを禁止するか
    """

    name = "L3_tracking.obb"
    tier = TIER_TRACKING
    requires = frozenset({CAP_BOXES})
    provides = frozenset({CAP_TRACK_ID})

    def __init__(
        self,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        track_buffer: int = 30,
        iou_weight: float = 1.0,
        orientation_weight: float = 0.0,
        motion_weight: float = 0.0,
        use_rotated_iou: bool = True,
        class_aware: bool = True,
    ):
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.iou_weight = float(iou_weight)
        self.orientation_weight = float(orientation_weight)
        self.motion_weight = float(motion_weight)
        self.use_rotated_iou = bool(use_rotated_iou)
        self.class_aware = bool(class_aware)

        total = self.iou_weight + self.orientation_weight + self.motion_weight
        if total <= 0:
            raise ValueError(
                "L3_tracking.obb: association の重みが全て 0 です "
                "(iou_weight / orientation_weight / motion_weight のいずれかを > 0 に)"
            )
        self._weight_sum = total

    def reset_state(self, ctx: PipelineContext) -> None:
        """1 動画ごとにトラッカ状態をリセットする (track_id を持ち越さない)。"""
        ctx.state.pop(self.name, None)

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        st = ctx.state.setdefault(self.name, {"tracks": [], "next_id": 0})
        tracks: List[_ObbTrack] = st["tracks"]

        for fr in sorted(window.frames, key=lambda f: f.frame_index):
            dets = fr.detections
            high = [i for i, d in enumerate(dets) if d.score >= self.track_thresh]
            low = [i for i, d in enumerate(dets)
                   if self.low_thresh <= d.score < self.track_thresh]

            # 一次: 高スコア検出 ↔ 全トラック
            matched_hi, un_tracks, un_hi = self._associate(tracks, dets, high)
            for ti, di in matched_hi:
                self._update(tracks[ti], dets[di])

            # 二次: 残トラック ↔ 低スコア検出 (救済)
            rem = [tracks[i] for i in un_tracks]
            matched_lo, _, _ = self._associate(rem, dets, low)
            for ti, di in matched_lo:
                self._update(rem[ti], dets[di])

            # 未マッチ高スコア検出 → 新規トラック
            for di in un_hi:
                d = dets[di]
                t = _ObbTrack(
                    st["next_id"], d.get_obb().copy(), d.bbox.copy(),
                    float(d.score), int(d.class_id),
                )
                st["next_id"] += 1
                d.track_id = t.id
                tracks.append(t)

            # 未マッチトラック → lost++、バッファ超過で除去
            matched_objs = {id(tracks[ti]) for ti, _ in matched_hi} | \
                           {id(rem[ti]) for ti, _ in matched_lo}
            survivors: List[_ObbTrack] = []
            for t in tracks:
                if id(t) in matched_objs:
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

    # ------------------------------------------------------------------ #
    # association
    # ------------------------------------------------------------------ #

    def _associate(self, tracks: List[_ObbTrack], dets: List, det_idx: List[int]):
        """類似度を組み立てて貪欲マッチングする。"""
        if not tracks or not det_idx:
            return [], list(range(len(tracks))), list(det_idx)

        cost = np.ones((len(tracks), len(det_idx)), dtype=np.float32)
        for i, t in enumerate(tracks):
            pred_obb = t.predict() if self.motion_weight > 0 else t.obb
            for k, di in enumerate(det_idx):
                d = dets[di]
                if self.class_aware and t.class_id != int(d.class_id):
                    continue

                sim = 0.0
                d_obb = d.get_obb()
                if self.iou_weight > 0:
                    if self.use_rotated_iou and (d.has_obb or t.obb is not None):
                        sim += self.iou_weight * rotated_iou_np(d_obb, pred_obb)
                    else:
                        sim += self.iou_weight * iou_xyxy(d.bbox, t.bbox)
                if self.orientation_weight > 0:
                    sim += self.orientation_weight * orientation_consistency_np(
                        float(d_obb[4]), float(t.obb[4])
                    )
                if self.motion_weight > 0:
                    # 予測中心との距離をトラックのサイズで正規化して [0,1] に落とす
                    dist = float(np.linalg.norm(d_obb[:2] - pred_obb[:2]))
                    scale = max(float(np.mean(t.obb[2:4])), 1e-6)
                    sim += self.motion_weight * float(np.exp(-dist / scale))

                cost[i, k] = 1.0 - sim / self._weight_sum

        matches, un_t, un_k = greedy_match(cost, max_cost=1.0 - self.match_thresh)
        matches = [(i, det_idx[k]) for i, k in matches]
        return matches, un_t, [det_idx[k] for k in un_k]

    def _update(self, t: _ObbTrack, det) -> None:
        """マッチしたトラックを更新する (速度も推定)。"""
        new_obb = det.get_obb().copy()
        t.velocity = (new_obb[:2] - t.obb[:2]).astype(np.float32)
        t.obb = new_obb
        t.bbox = det.bbox.copy()
        t.score = float(det.score)
        t.lost = 0
        det.track_id = t.id
