"""
keypoint_stage.py — pose (任意段): キーポイント/体軸推定の拡張点

AP10K 等の動物 pose 推定器を差し込み、各検出に keypoints/orientation を付ける任意段。
向き情報は L5 の facing 判定を体軸ベースへ格上げできる。pose 推定器は外部依存なので、
`predictor` (callable) を注入するか checkpoint を指定して使う。未指定なら setup で
明確に失敗する。

predictor 契約:
    predictor(image: np.ndarray(H,W,3), boxes_xyxy: np.ndarray(N,4)) -> np.ndarray(N, K, 3)
    各検出のキーポイント (x, y, conf)。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_IMAGE, CAP_POSE, FrameWindow,
)
from ...pipeline.stage import TIER_MOTION, PipelineContext, StageBase


@register("pose.keypoint")
class KeypointStage(StageBase):
    name = "pose.keypoint"
    tier = TIER_MOTION
    after = frozenset({"L1_detection.*"})
    requires = frozenset({CAP_BOXES, CAP_IMAGE})
    provides = frozenset({CAP_POSE})

    def __init__(self, predictor: Optional[Callable] = None, checkpoint: Optional[str] = None):
        self._predictor = predictor
        self.checkpoint = checkpoint

    def setup(self, ctx: PipelineContext) -> None:
        if self._predictor is None:
            raise NotImplementedError(
                "pose.keypoint には pose 推定器が必要です。predictor(callable) を注入するか、"
                "AP10K 等の pose backend を実装して checkpoint=... を渡してください。"
            )

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        for fr in window.frames:
            if fr.image is None or not fr.detections:
                continue
            boxes = np.stack([np.asarray(d.bbox, np.float32) for d in fr.detections], 0)
            kpts = self._predictor(fr.image, boxes)   # (N, K, 3)
            for d, kp in zip(fr.detections, kpts):
                d.keypoints = np.asarray(kp, np.float32)
                d.orientation = _axis_orientation(d.keypoints)
        window.available.add(CAP_POSE)
        return window


def _axis_orientation(kpts: np.ndarray) -> Optional[float]:
    """先頭2キーポイント (例: 頭→尾) から体軸方向 rad を推定する簡易実装。"""
    if kpts is None or len(kpts) < 2:
        return None
    head, tail = kpts[0], kpts[1]
    if head[2] < 0.1 or tail[2] < 0.1:
        return None
    return float(np.arctan2(head[1] - tail[1], head[0] - tail[0]))
