"""
geometric_stage.py — L2: 静的位置関係 (近接 / 重なり / 上下)。学習・モデル不要。

各フレーム内の検出ペアについて幾何関係を計算し、FrameResult.relations に格納する。
近接・重なりは対称 (src<dst で1本)、上下は有向。src/dst は「そのフレーム内の
detection index」。
"""

from __future__ import annotations

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_STATIC_RELATION, FrameWindow, Relation,
)
from ...pipeline.stage import TIER_STATIC, PipelineContext, StageBase
from .._geometry import iou_xyxy, box_center


@register("L2_static.geometric")
class GeometricRelationStage(StageBase):
    name = "L2_static.geometric"
    tier = TIER_STATIC
    requires = frozenset({CAP_BOXES})
    provides = frozenset({CAP_STATIC_RELATION})

    def __init__(
        self,
        near_dist_px: float = 40.0,
        overlap_iou: float = 0.1,
        use_vertical: bool = True,
        vertical_margin_px: float = 5.0,
    ):
        self.near_dist_px = float(near_dist_px)
        self.overlap_iou = float(overlap_iou)
        self.use_vertical = bool(use_vertical)
        self.vertical_margin_px = float(vertical_margin_px)

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        for fr in window.frames:
            dets = fr.detections
            fr.relations = []
            n = len(dets)
            centers = [box_center(d.bbox) for d in dets]
            for i in range(n):
                for j in range(i + 1, n):
                    ci, cj = centers[i], centers[j]
                    dist = float(np.linalg.norm(ci - cj))
                    iou = iou_xyxy(dets[i].bbox, dets[j].bbox)

                    if iou >= self.overlap_iou:
                        fr.relations.append(
                            Relation(src=i, dst=j, kind="overlap", value=iou, directed=False)
                        )
                    if dist <= self.near_dist_px:
                        fr.relations.append(
                            Relation(src=i, dst=j, kind="near", value=dist, directed=False)
                        )
                    if self.use_vertical:
                        dy = ci[1] - cj[1]
                        if dy < -self.vertical_margin_px:
                            # i は j より上 (画像座標では y 小)
                            fr.relations.append(
                                Relation(src=i, dst=j, kind="above", value=float(-dy), directed=True)
                            )
                        elif dy > self.vertical_margin_px:
                            fr.relations.append(
                                Relation(src=j, dst=i, kind="above", value=float(dy), directed=True)
                            )
        window.available.add(CAP_STATIC_RELATION)
        return window
