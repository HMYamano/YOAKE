"""
fused_stage.py — L1: 学習済み統合モデル HTRTDETR を合成バックエンドとして使う

1つの学習済み HTRTDETR を sliding window で回し、boxes/class/score/dense_feature に加え
GRU メモリ由来の track_id まで一括供給する。既存 inference/inferencer.py と同じ
最終フレーム推論を再現するので、JSON パリティを保つ「学習済み資産を捨てない」経路。

複数 capability を1passで出す合成バックエンドなので L1 tier に置くが、実質 L1+L3 を
まとめて満たす (track_id も provides)。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_BOXES, CAP_CLASS, CAP_DENSE_FEATURES, CAP_DET_SCORE, CAP_IMAGE,
    CAP_TRACK_ID, FrameWindow,
)
from ...pipeline.stage import TIER_DETECTION, PipelineContext, StageBase
from ._common import (
    detections_from_adapter, inference_ctx, load_htrtdetr, preprocess_frame, resolve_size,
)


@register("L1_detection.fused_htrtdetr")
class FusedHTRTDETRStage(StageBase):
    name = "L1_detection.fused_htrtdetr"
    tier = TIER_DETECTION
    requires = frozenset({CAP_IMAGE})
    provides = frozenset({CAP_BOXES, CAP_CLASS, CAP_DET_SCORE,
                          CAP_DENSE_FEATURES, CAP_TRACK_ID})

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        image_size: Optional[int] = None,
    ):
        self.checkpoint = checkpoint
        self.image_size = image_size
        self._model = None
        self._memory = None
        self._device = "cpu"
        self._size: Tuple[int, int] = (640, 640)

    def setup(self, ctx: PipelineContext) -> None:
        import torch

        if not self.checkpoint:
            raise ValueError("L1_detection.fused_htrtdetr には checkpoint=... が必要です。")
        self._device = ctx.device if ctx.device != "cuda" or torch.cuda.is_available() else "cpu"
        self._model, cfg = load_htrtdetr(self.checkpoint, self._device)
        self._size = resolve_size(self.image_size, cfg)
        self._reset_memory()

    def _reset_memory(self) -> None:
        import torch
        if self._model is not None:
            self._memory = self._model.create_memory_list(1, torch.device(self._device))[0]

    def reset_state(self, ctx: PipelineContext) -> None:
        # 1動画ごとに GRU ID メモリをリセット (Inferencer.reset_memory 相当)
        self._reset_memory()

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        import torch

        model = self._model
        ws = max(1, ctx.window_size)
        buffer: List = []
        for fr in sorted(window.frames, key=lambda f: f.frame_index):
            if fr.image is None:
                continue
            H0 = fr.height or fr.image.shape[0]
            W0 = fr.width or fr.image.shape[1]
            buffer.append(preprocess_frame(fr.image, self._size))
            if len(buffer) > ws:
                buffer.pop(0)
            images = torch.cat(buffer, dim=0).unsqueeze(0).to(self._device)  # (1,T,3,H,W)
            with inference_ctx(self._device):
                out = model(images, memory_list=[self._memory])
            det = out.det_results[0] if out.det_results else {}
            fr.detections = detections_from_adapter(det, W0, H0, with_dense=True)
        window.available.update(self.provides)
        return window

    def teardown(self) -> None:
        self._model = None
        self._memory = None
