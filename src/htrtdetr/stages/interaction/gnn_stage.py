"""
gnn_stage.py — L5: GNN / ペア注意 Transformer (混雑シーン向け・拡張点)

多個体・混雑・第三者干渉のあるシーンで、有向グラフ上の辺関係を群れ文脈込みで推論する
ための拡張点。2個体シーンには過剰であり、学習データ (混雑シーン) が前提。現状は
明示的な未実装マーカーとして登録され、config から参照はできるが setup で明確に失敗する
(黙って劣化しない)。
"""

from __future__ import annotations

from typing import Optional

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_DIRECTED_INTER, CAP_KINEMATICS, CAP_TRACK_ID, FrameWindow,
)
from ...pipeline.stage import TIER_INTERACTION, PipelineContext, StageBase


@register("L5_interaction.gnn")
class GNNInteractionStage(StageBase):
    name = "L5_interaction.gnn"
    tier = TIER_INTERACTION
    requires = frozenset({CAP_TRACK_ID, CAP_KINEMATICS})
    provides = frozenset({CAP_DIRECTED_INTER})

    def __init__(self, checkpoint: Optional[str] = None, **kwargs):
        self.checkpoint = checkpoint

    def setup(self, ctx: PipelineContext) -> None:
        raise NotImplementedError(
            "L5_interaction.gnn は混雑シーン向けの拡張点で未実装です。"
            "2個体〜疎なシーンでは L5_interaction.heuristic か .learned_pair を使ってください。"
        )

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:  # pragma: no cover
        raise NotImplementedError
