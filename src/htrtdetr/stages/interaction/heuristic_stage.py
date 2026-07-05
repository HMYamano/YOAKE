"""
heuristic_stage.py — L5: 有向相互作用のヒューリスティック判定 (学習不要)

速度/幾何の非対称性 (facing_AB ≠ facing_BA, 接近, 追従) から court / chase /
investigate を判定する。L5 の最初のバックエンドとして、学習なしで「誰が誰に」の枠を
すぐ埋める。窓内でフレームごとにペアを採点し、連続区間を DirectedInteraction に集約。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_DIRECTED_INTER, CAP_KINEMATICS, CAP_TRACK_ID,
    DirectedInteraction, FrameWindow, Track,
)
from ...pipeline.stage import TIER_INTERACTION, PipelineContext, StageBase
from .pair_features import build_pair_feature


@register("L5_interaction.heuristic")
class HeuristicInteractionStage(StageBase):
    name = "L5_interaction.heuristic"
    tier = TIER_INTERACTION
    requires = frozenset({CAP_TRACK_ID, CAP_KINEMATICS})
    provides = frozenset({CAP_DIRECTED_INTER})

    def __init__(
        self,
        max_pair_dist_px: float = 120.0,
        approach_thresh: float = 0.0,
        facing_thresh_rad: float = 1.05,   # ~60deg 以内なら「向いている」
        court_speed_px: float = 2.0,       # これ以下で近接=court
        min_frames: int = 3,
    ):
        self.max_pair_dist_px = float(max_pair_dist_px)
        self.approach_thresh = float(approach_thresh)
        self.facing_thresh = float(facing_thresh_rad)
        self.court_speed_px = float(court_speed_px)
        self.min_frames = int(min_frames)

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        tracks: Dict[int, Track] = window.tracks
        if len(tracks) < 2:
            window.available.add(CAP_DIRECTED_INTER)
            return window

        pos = {tid: {f: i for i, f in enumerate(t.frames)} for tid, t in tracks.items()}
        tids = sorted(tracks)

        interactions: List[DirectedInteraction] = []
        # 有向なので (A,B) と (B,A) を両方見る
        for a in tids:
            for b in tids:
                if a == b:
                    continue
                interactions.extend(self._pair(tracks[a], tracks[b], pos[a], pos[b]))

        window.interactions.extend(interactions)
        window.available.add(CAP_DIRECTED_INTER)
        return window

    def _pair(self, A: Track, B: Track, posA: Dict, posB: Dict) -> List[DirectedInteraction]:
        common = sorted(set(A.frames) & set(B.frames))
        if len(common) < self.min_frames:
            return []
        prev_d: Optional[float] = None
        # frame -> (relation or None, score)
        labels: List = []
        for f in common:
            feat = build_pair_feature(A, B, f, posA, posB, prev_dist=prev_d)
            if feat is None:
                labels.append((f, None, 0.0))
                continue
            d, facing_AB, facing_BA, range_rate, approach_A, approach_B, sa, sb, sdiff, behind = feat
            prev_d = d
            rel, score = self._classify(
                d, facing_AB, facing_BA, range_rate, approach_A, sa, behind
            )
            labels.append((f, rel, score))
        return self._merge_intervals(A.track_id, B.track_id, labels)

    def _classify(self, d, facing_AB, facing_BA, range_rate, approach_A, sa, behind):
        if d > self.max_pair_dist_px:
            return None, 0.0
        a_faces_b = facing_AB < self.facing_thresh
        approaching = approach_A > self.approach_thresh or range_rate < 0
        if not a_faces_b:
            return None, 0.0
        # chase: A が B の後方から接近しつつ追従、両者移動
        if behind > 0.5 and approaching and sa > self.court_speed_px:
            return "chase", float(np.clip(0.5 + approach_A / (sa + 1e-6), 0.0, 1.0))
        # court: 近接・低速で向き合い
        if d < self.max_pair_dist_px * 0.5 and sa <= self.court_speed_px:
            return "court", float(np.clip(1.0 - d / (self.max_pair_dist_px * 0.5), 0.0, 1.0))
        # investigate: 向いて接近
        if approaching:
            return "investigate", float(np.clip(0.4 + (-range_rate) / (d + 1e-6), 0.0, 1.0))
        return None, 0.0

    def _merge_intervals(self, a_tid, b_tid, labels) -> List[DirectedInteraction]:
        out: List[DirectedInteraction] = []
        cur_rel: Optional[str] = None
        seg_start: Optional[int] = None
        seg_scores: List[float] = []
        prev_f: Optional[int] = None

        def flush(end_f):
            if cur_rel is not None and seg_start is not None and len(seg_scores) >= self.min_frames:
                out.append(DirectedInteraction(
                    agent_track_id=int(a_tid), patient_track_id=int(b_tid),
                    relation=cur_rel, score=float(np.mean(seg_scores)),
                    t_start=int(seg_start), t_end=int(end_f),
                ))

        for f, rel, score in labels:
            contiguous = prev_f is None or f == prev_f + 1
            if rel == cur_rel and contiguous:
                if rel is not None:
                    seg_scores.append(score)
            else:
                flush(prev_f if prev_f is not None else f)
                cur_rel = rel
                seg_start = f if rel is not None else None
                seg_scores = [score] if rel is not None else []
            prev_f = f
        flush(prev_f if prev_f is not None else 0)
        return out
