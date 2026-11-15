"""
test_clip_stream.py — the keyframe (YOWO-style) parts of the model.

* the clip tube: tracking-free temporal evidence, one more additive group
* strided track history, so streaming reads the spacing training used
* the unified model and the streaming inferencer with the tube switched on
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from htrtdetr.config.config import (
    BehaviorConfig,
    ExplainabilityConfig,
    InferenceConfig,
    build_model_config,
)
from htrtdetr.inference import StreamingInferencer
from htrtdetr.models import build_model
from htrtdetr.models.behavior import BehaviorBranch
from htrtdetr.models.behavior.clip import ClipTubeEncoder
from htrtdetr.tracking import TrackFeatureCache

N, T, C = 4, 6, 32


def _inputs():
    hist = torch.rand(N, T, 5) * 0.5 + 0.25
    hist[:, :, 2:4] = 0.1
    return {
        "obb_history": hist,
        "history_mask": torch.ones(N, T, dtype=torch.bool),
        "spatial_features": torch.randn(2, C, 16, 16),
        "batch_index": torch.tensor([0, 0, 1, 1]),
        "detector_features": torch.randn(N, 24),
        "temporal_features": torch.randn(N, 20),
    }


def _branch(**overrides):
    cfg = BehaviorConfig(feature_dim=24, actor_roi_size=6, context_roi_size=6,
                         clip_channels=8, clip_roi_size=5)
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    return BehaviorBranch(cfg, num_actions=5, feature_channels=C, detector_dim=24,
                          temporal_dim=20, explain_cfg=ExplainabilityConfig())


# ══════════════════════════════════════════════════════════════════════════
# The tube itself
# ══════════════════════════════════════════════════════════════════════════

class TestClipTube:
    def test_shapes_and_empty_input(self):
        enc = ClipTubeEncoder(in_channels=C, out_dim=24, reduced_channels=8, roi_size=5)
        frames = [enc.reduce_frame(torch.randn(2, C, 16, 16)) for _ in range(T)]
        assert frames[0].shape == (2, 8, 16, 16)
        obb = torch.tensor([[0.5, 0.5, 0.1, 0.05, 0.3], [0.3, 0.6, 0.1, 0.05, 0.0]])
        out = enc(frames, obb, torch.tensor([0, 1]))
        assert out.shape == (2, 24)
        assert enc(frames, obb[:0], torch.zeros(0, dtype=torch.long)).shape == (0, 24)

    def test_it_sees_motion_that_a_single_frame_cannot(self):
        """Same keyframe, different past: a static clip and a moving one must
        not encode the same. That difference is the whole reason it exists."""
        torch.manual_seed(0)
        enc = ClipTubeEncoder(in_channels=C, out_dim=24, reduced_channels=8, roi_size=5).eval()
        key = torch.randn(1, C, 16, 16)
        static = [key] * T
        moving = [torch.roll(key, shifts=(T - 1 - t), dims=-1) for t in range(T)]
        obb = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.0]])
        idx = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            a = enc([enc.reduce_frame(f) for f in static], obb, idx)
            b = enc([enc.reduce_frame(f) for f in moving], obb, idx)
        assert not torch.allclose(a, b, atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════
# As an evidence group
# ══════════════════════════════════════════════════════════════════════════

class TestClipGroup:
    def _clip_frames(self, branch):
        return [branch.clip_encoder.reduce_frame(torch.randn(2, C, 16, 16)) for _ in range(T)]

    def test_off_unless_asked_so_old_checkpoints_still_load_strictly(self):
        assert BehaviorConfig().use_clip is None
        b = _branch()
        assert "clip" not in b.active_groups()
        assert not hasattr(b, "clip_encoder")

    def test_evidence_still_sums_to_the_logits(self):
        b = _branch(use_clip=True)
        assert "clip" in b.active_groups()
        p = b(**_inputs(), clip_frames=self._clip_frames(b))
        assert "clip" in p.evidence
        assert torch.allclose(sum(p.evidence.values()), p.logits, atol=1e-4)

    def test_ablation_removes_the_parameters(self):
        with_clip = sum(p.numel() for p in _branch(use_clip=True).parameters())
        without = sum(p.numel() for p in _branch(use_clip=False).parameters())
        assert with_clip > without

    def test_missing_frames_is_a_clear_error(self):
        b = _branch(use_clip=True)
        with pytest.raises(ValueError, match="clip_frames"):
            b(**_inputs())

    def test_gradient_reaches_the_per_frame_reduction(self):
        b = _branch(use_clip=True)
        b(**_inputs(), clip_frames=self._clip_frames(b)).logits.sum().backward()
        grad = b.clip_encoder.reduce[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0


# ══════════════════════════════════════════════════════════════════════════
# Strided history
# ══════════════════════════════════════════════════════════════════════════

def test_strided_history_ends_at_the_newest_frame():
    cache = TrackFeatureCache(window=7, feat_dim=2)          # (T-1)*s+1 with T=4, s=2
    for f in range(9):
        cache.update(f, [0], torch.full((1, 2), float(f)),
                     torch.tensor([[0.1 * f, 0.5, 0.1, 0.05, 0.0]]))
    feats, obbs, mask, ids = cache.stacked_history(stride=2)
    assert feats.shape == (1, 4, 2) and ids == [0]
    assert feats[0, :, 0].tolist() == [2.0, 4.0, 6.0, 8.0]
    assert mask.all()
    full, *_ = cache.stacked_history()
    assert full.shape == (1, 7, 2)
    empty = TrackFeatureCache(window=7, feat_dim=2).stacked_history(stride=2)
    assert empty[0].shape == (0, 4, 2)


# ══════════════════════════════════════════════════════════════════════════
# The unified model and streaming, with the tube on
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def clip_model():
    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.head.num_queries = 8
    cfg.detector.score_threshold = 0.0
    cfg.behavior.use_clip = True
    cfg.behavior.clip_channels = 16
    m = build_model(cfg)
    m.set_stage(2)
    return m


def test_training_forward_uses_every_frame_of_the_clip(clip_model):
    clip_model.train()
    images = torch.randn(2, 4, 3, 64, 64)
    out = clip_model(images)
    assert out.action_logits is not None
    assert out.behavior is not None and "clip" in out.behavior.evidence
    out.action_logits.sum().backward()
    grad = clip_model.behavior.clip_encoder.reduce[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0
    clip_model.zero_grad(set_to_none=True)


def test_streaming_reads_history_at_the_training_stride(clip_model):
    si = StreamingInferencer(
        clip_model,
        InferenceConfig(use_amp=False, use_channels_last=False),
        action_names=[str(i) for i in range(5)],
        window_size=4,
        image_size=(64, 64),
        clip_stride=3,
    )
    assert si.span == 10
    assert si.tracker.cache.window == 10
    rng = np.random.default_rng(3)
    last = None
    for i in range(12):
        last = si.process_frame(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), i)
    assert len(si._clip_maps) == 10
    frames = si._clip_frames()
    assert len(frames) == 4
    # newest map last, then every third one before it
    maps = list(si._clip_maps)
    assert frames[-1] is maps[-1] and frames[-2] is maps[-4]
    assert last is not None and last.action_probs is not None
    payload = last.to_json()
    assert "action_probs" in payload and "class_ids" in payload
    assert len(payload["action_probs"]) == len(payload["boxes"])
    si.reset()
    assert len(si._clip_maps) == 0


def test_a_frozen_detector_stays_in_eval_mode_while_the_rest_trains(clip_model):
    """Stage 2/3 freeze the detector; its BatchNorm must not train on tiny
    per-timestep batches or rewrite its running statistics on context frames."""
    clip_model.set_stage(2)
    clip_model.train()
    assert clip_model.training and not clip_model.detector.training
    assert clip_model.behavior.training
    clip_model.set_stage(4)
    clip_model.train()
    assert clip_model.detector.training
    clip_model.set_stage(2)
