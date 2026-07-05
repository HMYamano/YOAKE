"""
rtdetr_stage.py — L1: RT-DETR 検出 (per-frame, DETR 系なので dense_features を出す)

既存 RTDETRDetector を checkpoint からロードし、各フレームを単独で検出する
(temporal / ID は使わない = 「検出のみ」モード)。query 特徴を dense_feature として
各 Detection に付ける (L3/L5 の神経系が使う)。
"""

from __future__ import annotations

from typing import Optional, Tuple

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_CLASS, CAP_DENSE_FEATURES, CAP_DET_SCORE, CAP_IMAGE, FrameWindow,
)
from ...pipeline.stage import TIER_DETECTION, PipelineContext, StageBase
from ._common import (
    detections_from_adapter, inference_ctx, load_htrtdetr, preprocess_frame, resolve_size,
)


@register("L1_detection.rtdetr")
class RTDETRStage(StageBase):
    name = "L1_detection.rtdetr"
    tier = TIER_DETECTION
    requires = frozenset({CAP_IMAGE})
    provides = frozenset({CAP_BOXES, CAP_CLASS, CAP_DET_SCORE, CAP_DENSE_FEATURES})

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        score_threshold: Optional[float] = None,
        image_size: Optional[int] = None,
    ):
        self.checkpoint = checkpoint
        # None のときは checkpoint の cfg.model.detector.score_threshold を使う
        self.score_threshold = None if score_threshold is None else float(score_threshold)
        self.image_size = image_size
        self._model = None
        self._device = "cpu"
        self._size: Tuple[int, int] = (640, 640)

    def setup(self, ctx: PipelineContext) -> None:
        import torch

        if not self.checkpoint:
            raise ValueError("L1_detection.rtdetr には checkpoint=... が必要です。")
        self._device = ctx.device if ctx.device != "cuda" or torch.cuda.is_available() else "cpu"
        self._model, cfg = load_htrtdetr(self.checkpoint, self._device)
        self._size = resolve_size(self.image_size, cfg)
        if self.score_threshold is None:
            self.score_threshold = float(cfg.model.detector.score_threshold)

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        model = self._model
        for fr in window.frames:
            if fr.image is None:
                continue
            H0 = fr.height or fr.image.shape[0]
            W0 = fr.width or fr.image.shape[1]
            tensor = preprocess_frame(fr.image, self._size).to(self._device)
            with inference_ctx(self._device):
                det_out = model.detector(tensor)
                dets = model.det_adapter.extract_valid_detections(
                    det_out.pred_logits, det_out.pred_boxes, det_out.query_features,
                    score_threshold=self.score_threshold,
                )
            fr.detections = detections_from_adapter(dets[0], W0, H0, with_dense=True)
        window.available.update(self.provides)
        return window

    def teardown(self) -> None:
        self._model = None
