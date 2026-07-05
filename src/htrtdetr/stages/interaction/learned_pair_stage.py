"""
learned_pair_stage.py — L5: 学習型の順序付きペア分類 ('none' クラス込み)

既存 InteractionFeatureComputer が「ノード単位・(ほぼ)対称」だったのを、順序付き
ペア (A→B) 単位に置き換えて拡張する。各フレームのペア非対称特徴 (pair_features) を
小さな MLP/1D-conv で court/chase/mount/investigate/none に分類し、窓内で集約する。

checkpoint があれば学習済み重みをロードする。無い場合も構造検証のため未学習で走る
(出力は当てにならない旨を warning する)。
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

from ...pipeline.registry import register
from ...pipeline.schema import (
    CAP_DIRECTED_INTER, CAP_KINEMATICS, CAP_TRACK_ID,
    DirectedInteraction, FrameWindow, Track,
)
from ...pipeline.stage import TIER_INTERACTION, PipelineContext, StageBase
from .pair_features import PAIR_FEAT_DIM, build_pair_feature

DEFAULT_RELATIONS = ["none", "court", "chase", "mount", "investigate"]


def _build_head(pair_dim: int, hidden: int, relations: List[str]):
    import torch.nn as nn

    class DirectedInteractionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(pair_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            )
            self.classifier = nn.Linear(hidden, len(relations))

        def forward(self, x):
            return self.classifier(self.encoder(x))

    return DirectedInteractionHead()


@register("L5_interaction.learned_pair")
class LearnedPairStage(StageBase):
    name = "L5_interaction.learned_pair"
    tier = TIER_INTERACTION
    # pose / static_relation はあれば使う任意入力
    requires = frozenset({CAP_TRACK_ID, CAP_KINEMATICS})
    provides = frozenset({CAP_DIRECTED_INTER})

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        hidden: int = 64,
        relations: Optional[List[str]] = None,
        max_pair_dist_px: float = 120.0,
        min_frames: int = 3,
        score_thresh: float = 0.5,
        allow_untrained: bool = False,
    ):
        self.checkpoint = checkpoint
        self.allow_untrained = bool(allow_untrained)
        self.hidden = int(hidden)
        self.relations = relations or list(DEFAULT_RELATIONS)
        self.max_pair_dist_px = float(max_pair_dist_px)
        self.min_frames = int(min_frames)
        self.score_thresh = float(score_thresh)
        self._head = None
        self._device = "cpu"

    def setup(self, ctx: PipelineContext) -> None:
        import torch

        self._device = ctx.device if ctx.device != "cuda" or torch.cuda.is_available() else "cpu"
        self._head = _build_head(PAIR_FEAT_DIM, self.hidden, self.relations).to(self._device).eval()
        if self.checkpoint:
            payload = torch.load(self.checkpoint, map_location=self._device, weights_only=True)
            state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
            sub = {k.split("interaction_head.")[-1]: v
                   for k, v in state.items() if "interaction_head." in k}
            if sub:
                self._head.load_state_dict(sub, strict=False)
        elif not self.allow_untrained:
            raise ValueError(
                "L5_interaction.learned_pair は未学習です。学習済み重みを checkpoint=... で "
                "渡してください。構造確認のため未学習で走らせる場合は allow_untrained=true を "
                "明示してください (出力は当てになりません)。相互作用を学習なしで得るなら "
                "L5_interaction.heuristic を使ってください。"
            )
        else:
            warnings.warn(
                "L5_interaction.learned_pair が checkpoint 無し・allow_untrained=true で動作 "
                "しています (未学習。出力は構造確認用で信頼できません)。",
                stacklevel=2,
            )

    def process(self, window: FrameWindow, ctx: PipelineContext) -> FrameWindow:
        import torch

        tracks: Dict[int, Track] = window.tracks
        if len(tracks) < 2:
            window.available.add(CAP_DIRECTED_INTER)
            return window

        pos = {tid: {f: i for i, f in enumerate(t.frames)} for tid, t in tracks.items()}
        tids = sorted(tracks)
        none_idx = self.relations.index("none") if "none" in self.relations else -1

        for a in tids:
            for b in tids:
                if a == b:
                    continue
                A, B = tracks[a], tracks[b]
                common = sorted(set(A.frames) & set(B.frames))
                if len(common) < self.min_frames:
                    continue
                feats, frames = [], []
                prev_d = None
                for f in common:
                    fe = build_pair_feature(A, B, f, pos[a], pos[b], prev_dist=prev_d)
                    if fe is None:
                        continue
                    prev_d = float(fe[0])
                    if fe[0] > self.max_pair_dist_px:
                        continue
                    feats.append(fe)
                    frames.append(f)
                if len(feats) < self.min_frames:
                    continue
                x = torch.from_numpy(np.stack(feats, 0)).float().to(self._device)
                with torch.no_grad():
                    probs = torch.softmax(self._head(x), dim=-1).cpu().numpy()
                self._aggregate(a, b, frames, probs, none_idx, window)

        window.available.add(CAP_DIRECTED_INTER)
        return window

    def _aggregate(self, a, b, frames, probs, none_idx, window) -> None:
        pred = probs.argmax(axis=1)
        cur, seg_start, seg_scores, prev_f = None, None, [], None

        def flush(end_f):
            if cur is not None and seg_start is not None and len(seg_scores) >= self.min_frames:
                window.interactions.append(DirectedInteraction(
                    agent_track_id=int(a), patient_track_id=int(b),
                    relation=self.relations[cur], score=float(np.mean(seg_scores)),
                    t_start=int(seg_start), t_end=int(end_f),
                ))

        for k, f in enumerate(frames):
            c = int(pred[k])
            sc = float(probs[k, c])
            is_none = (c == none_idx) or (sc < self.score_thresh)
            contiguous = prev_f is None or f == prev_f + 1
            eff = None if is_none else c
            if eff == cur and contiguous:
                if eff is not None:
                    seg_scores.append(sc)
            else:
                flush(prev_f if prev_f is not None else f)
                cur = eff
                seg_start = f if eff is not None else None
                seg_scores = [sc] if eff is not None else []
            prev_f = f
        flush(prev_f if prev_f is not None else 0)

    def teardown(self) -> None:
        self._head = None
