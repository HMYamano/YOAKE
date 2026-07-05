"""
kinematic_stage.py — L4: 運動 (速度 / 速さ / 進行方向) と窓内トラック集約。

track_id 付きの検出列から、各トラックの中心軌跡・速度・向きを計算し、
各 Detection に velocity/speed/heading を書き戻し、FrameWindow.tracks を埋める。

既存 data/feature_builder.py (GeometricFeatureBuilder) と同じ運動量 (dx,dy,speed) を
px/frame 単位で素直に計算する。track_id が無い検出は対象外 (運動には同一性が要る)。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_KINEMATICS, CAP_TRACK_ID, FrameWindow, Track,
)
from ...pipeline.stage import TIER_MOTION, PipelineContext, StageBase
from .._geometry import box_center


@register("L4_motion.kinematic")
class KinematicStage(StageBase):
    name = "L4_motion.kinematic"
    tier = TIER_MOTION
    requires = frozenset({CAP_TRACK_ID, CAP_BOXES})
    provides = frozenset({CAP_KINEMATICS})

    def __init__(self, smooth_window: int = 3):
        # 偶数窓は移動平均が半フレームずれるため奇数に丸める
        w = max(1, int(smooth_window))
        self.smooth_window = w if w % 2 == 1 else w + 1

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        # track_id -> [(frame_index, Detection)]
        by_track: Dict[int, List] = {}
        for fr in window.frames:
            for det in fr.detections:
                if det.track_id is None:
                    continue
                by_track.setdefault(int(det.track_id), []).append((fr.frame_index, det))

        for tid, items in by_track.items():
            items.sort(key=lambda x: x[0])
            frames = [f for f, _ in items]
            dets = [d for _, d in items]
            centers = np.stack([box_center(d.bbox) for d in dets], axis=0)  # (T,2)
            centers_s = self._smooth(centers)

            vel = np.zeros_like(centers_s)
            if len(centers_s) >= 2:
                dt = np.diff(np.asarray(frames, dtype=np.float32))
                dt[dt == 0] = 1.0
                vel[1:] = (centers_s[1:] - centers_s[:-1]) / dt[:, None]
                vel[0] = vel[1]
            speeds = np.linalg.norm(vel, axis=1)                            # (T,)
            headings = np.arctan2(vel[:, 1], vel[:, 0])                     # (T,)

            for i, det in enumerate(dets):
                det.velocity = vel[i].astype(np.float32)
                det.speed = float(speeds[i])
                det.heading = float(headings[i])

            window.tracks[tid] = Track(
                track_id=tid,
                frames=frames,
                centers=centers_s.astype(np.float32),
                velocities=vel.astype(np.float32),
                headings=headings.astype(np.float32),
                speeds=speeds.astype(np.float32),
            )

        window.available.add(CAP_KINEMATICS)
        return window

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        w = self.smooth_window
        if w <= 1 or len(arr) < w:
            return arr.astype(np.float32)
        pad = w // 2
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(w, dtype=np.float32) / w
        out = np.stack([
            np.convolve(padded[:, d], kernel, mode="valid") for d in range(arr.shape[1])
        ], axis=1)
        return out[: len(arr)].astype(np.float32)
