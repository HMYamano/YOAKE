"""
memory_id_stage.py — L3: 学習済み GRU メモリ ID ヘッド (神経系トラッカ)

既存 models/id_head/MemoryIDHead を窓内でフレーム順に forward_inference で回し、
GRU メモリと照合して track_id を振る。神経系ラベルなので dense_features を要求する。

fidelity について:
  最も忠実な神経系トラッキングは、temporal 融合を含む統合モデル全体を回す
  `L1_detection.fused_htrtdetr` 経由で得られる (そちらは track_id を直接供給する)。
  本スタンドアロン段は dense_feature を id ヘッドへ直接渡す (temporal 融合を省く) ため
  近似であり、その旨を明示する。checkpoint が必要。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_APPEARANCE_EMB, CAP_BOXES, CAP_DENSE_FEATURES, CAP_TRACK_ID, FrameWindow,
)
from ...pipeline.stage import TIER_TRACKING, PipelineContext, StageBase


@register("L3_tracking.memory_id")
class MemoryIDStage(StageBase):
    name = "L3_tracking.memory_id"
    tier = TIER_TRACKING
    requires = frozenset({CAP_BOXES, CAP_DENSE_FEATURES})
    provides = frozenset({CAP_TRACK_ID, CAP_APPEARANCE_EMB})

    def __init__(self, checkpoint: Optional[str] = None):
        self.checkpoint = checkpoint
        self._head = None
        self._memory = None
        self._device = "cpu"

    def setup(self, ctx: PipelineContext) -> None:
        import torch

        from ...cli import load_runtime_config_from_checkpoint
        from ...models.id_head import MemoryIDHead

        if not self.checkpoint:
            raise ValueError(
                "L3_tracking.memory_id には checkpoint=... が必要です "
                "(学習済み id_head を含む .pth)。dense 不要の追跡なら "
                "L3_tracking.bytetrack を使ってください。"
            )
        self._device = ctx.device if ctx.device != "cuda" or torch.cuda.is_available() else "cpu"
        cfg = load_runtime_config_from_checkpoint(self.checkpoint)
        if cfg is None:
            raise ValueError(
                f"checkpoint '{self.checkpoint}' に config メタデータがありません。"
                "config 付きで保存された checkpoint が必要です。"
            )
        self._head = MemoryIDHead(cfg.model.id_head).to(self._device).eval()
        payload = torch.load(self.checkpoint, map_location=self._device, weights_only=True)
        state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
        sub = {k[len("id_head."):]: v for k, v in state.items() if k.startswith("id_head.")}
        if sub:
            self._head.load_state_dict(sub, strict=False)
        self._memory = self._head.create_memory(torch.device(self._device))
        self._feat_dim = cfg.model.id_head.feature_dim

    def reset_state(self, ctx: PipelineContext) -> None:
        # 1動画ごとに GRU ID メモリをリセット
        import torch
        if self._head is not None:
            self._memory = self._head.create_memory(torch.device(self._device))

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        import torch

        for fr in sorted(window.frames, key=lambda f: f.frame_index):
            dets = fr.detections
            valid = [d for d in dets if d.dense_feature is not None]
            if not valid:
                continue
            W = float(fr.width or ctx.state.get("width", 640))
            H = float(fr.height or ctx.state.get("height", 640))

            feats = np.stack([np.asarray(d.dense_feature, np.float32) for d in valid], 0)
            if feats.shape[1] != self._feat_dim:
                raise ValueError(
                    f"dense_feature 次元 {feats.shape[1]} が id_head.feature_dim "
                    f"{self._feat_dim} と一致しません。fused_htrtdetr 由来の特徴を使ってください。"
                )
            # xyxy(px) -> cxcywh(normalized)
            boxes = np.stack([np.asarray(d.bbox, np.float32) for d in valid], 0)
            cx = (boxes[:, 0] + boxes[:, 2]) * 0.5 / W
            cy = (boxes[:, 1] + boxes[:, 3]) * 0.5 / H
            bw = (boxes[:, 2] - boxes[:, 0]) / W
            bh = (boxes[:, 3] - boxes[:, 1]) / H
            cxcywh = np.stack([cx, cy, bw, bh], 1).astype(np.float32)

            with torch.no_grad():
                out = self._head.forward_inference(
                    torch.from_numpy(feats).to(self._device),
                    torch.from_numpy(cxcywh).to(self._device),
                    self._memory,
                )
            tids = out["track_ids"].cpu().numpy()
            embs = out["embeddings"].cpu().numpy()
            for d, tid, emb in zip(valid, tids, embs):
                d.track_id = int(tid)
                d.embedding = emb.astype(np.float32)

        window.available.add(CAP_TRACK_ID)
        window.available.add(CAP_APPEARANCE_EMB)
        return window

    def teardown(self) -> None:
        self._head = None
        self._memory = None
