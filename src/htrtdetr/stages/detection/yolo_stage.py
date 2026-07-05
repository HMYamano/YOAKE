"""
yolo_stage.py — L1: YOLO 検出 (dense_features を出さない → 差し替えの試金石)

ultralytics YOLO を検出バックエンドとしてラップする。DETR の query 特徴は出さないので
capabilities から dense_features を外す。これにより「YOLO × 神経系トラッカ(dense必須)」を
strict で事前エラーにし、「YOLO × bytetrack(dense不要)」なら通る、という依存解決を実証する。

ultralytics 未インストールでも import は通り、setup 時に明確なメッセージで失敗する。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_CLASS, CAP_DET_SCORE, CAP_IMAGE, Detection, FrameWindow,
)
from ...pipeline.stage import TIER_DETECTION, PipelineContext, StageBase


@register("L1_detection.yolo")
class YOLOStage(StageBase):
    name = "L1_detection.yolo"
    tier = TIER_DETECTION
    requires = frozenset({CAP_IMAGE})
    # dense_features は出さない (非DETR系)
    provides = frozenset({CAP_BOXES, CAP_CLASS, CAP_DET_SCORE})

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        score_threshold: float = 0.3,
        imgsz: int = 640,
    ):
        self.weights = weights
        self.score_threshold = float(score_threshold)
        self.imgsz = int(imgsz)
        self._model = None

    def setup(self, ctx: PipelineContext) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "L1_detection.yolo には ultralytics が必要です: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.weights)

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        for fr in window.frames:
            if fr.image is None:
                continue
            res = self._model.predict(
                fr.image, imgsz=self.imgsz, conf=self.score_threshold, verbose=False
            )[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy()
                for i in range(len(xyxy)):
                    dets.append(Detection(
                        bbox=xyxy[i].astype(np.float32),
                        score=float(conf[i]),
                        class_id=int(cls[i]),
                    ))
            fr.detections = dets
        window.available.update(self.provides)
        return window

    def teardown(self) -> None:
        self._model = None
