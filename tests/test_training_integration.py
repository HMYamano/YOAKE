"""
test_training_integration.py — 学習ループの疎通テスト

「forward は通るが勾配が届いていない」という壊れ方を防ぐため、
**stage ごとに正しいモジュールへ勾配が流れる**ことを直接検証する。
"""

from __future__ import annotations

import pytest
import torch

from htrtdetr.config.config import LossConfig, build_model_config
from htrtdetr.models import build_model
from htrtdetr.training.losses import CombinedLoss
from htrtdetr.utils.misc import cxcywh_to_xyxy

_B, _T, _HW = 2, 3, 128


def _model(obb: bool = False, stage: int = 1):
    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.head.num_queries = 10
    cfg.detector.score_threshold = 0.0        # 検出を必ず出す
    if obb:
        cfg.detector.type = "rtdetr_obb"
        cfg.detector.obb.enabled = True
    m = build_model(cfg)
    m.set_stage(stage)
    m.train()
    return m, cfg


def _matching_targets(out, obb: bool = False):
    """
    モデル自身の予測を GT にして、IoU マッチングが必ず成立する targets を作る。

    ランダム重みでは予測と固定 GT が重ならず、全ラベルが -1 (ignore) になって
    損失が 0 になってしまう。それでは「勾配が流れるか」を検証できない。

    Stage 1 の出力は det_results を持たない (検出のみで早期 return する) ので、
    その場合は pred_boxes から直接 GT を作る。
    """
    targets = []
    for b in range(_B):
        if out.det_results is not None:
            boxes = out.det_results[b]["boxes"][:2].detach()
            gt_obb = out.det_results[b].get("obb")
            gt_obb = gt_obb[:2].detach() if gt_obb is not None else None
        else:
            boxes = out.pred_boxes[b][:2].detach()
            gt_obb = out.pred_obb[b][:2].detach() if out.pred_obb is not None else None

        n = boxes.shape[0]
        t = {
            "boxes": cxcywh_to_xyxy(boxes),
            "class_ids": torch.zeros(n, dtype=torch.long),
            "track_ids": torch.arange(n),
            "action_ids": torch.full((n,), 2, dtype=torch.long),
        }
        if obb and gt_obb is not None:
            t["obb"] = gt_obb
            t["obb_valid"] = torch.ones(n, dtype=torch.bool)
        targets.append(t)
    return targets


def _grad_prefixes(model):
    """勾配が実際に入ったパラメータの先頭モジュール名の集合。"""
    return {
        name.split(".")[0]
        for name, p in model.named_parameters()
        if p.grad is not None and float(p.grad.norm()) > 0
    }


class TestStageGradients:
    def test_stage1_trains_detector_only(self):
        m, cfg = _model(stage=1)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5)
        loss.set_stage(1)
        losses = loss(out, _matching_targets(out), stage=1)
        losses["total_loss"].backward()
        assert "detector" in _grad_prefixes(m)

    def test_stage2_trains_behavior_not_detector(self):
        """Stage 2 は detector を freeze し、temporal / behavior を学習する。"""
        m, _ = _model(stage=2)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5)
        loss.set_stage(2)
        losses = loss(out, _matching_targets(out), stage=2)
        assert float(losses["loss_action"].detach()) > 0, "action loss が 0 (GT 未マッチ)"
        losses["loss_action"].backward()
        got = _grad_prefixes(m)
        assert "temporal" in got
        assert "behavior" in got
        assert "detector" not in got, "Stage 2 で detector が freeze されていない"

    def test_stage3_trains_id_head_not_detector(self):
        m, _ = _model(stage=3)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5)
        loss.set_stage(3)
        losses = loss(out, _matching_targets(out), stage=3)
        losses["loss_id"].backward()
        got = _grad_prefixes(m)
        assert "id_head" in got or "feature_router" in got
        assert "detector" not in got

    def test_stage4_trains_everything(self):
        m, _ = _model(stage=4)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5)
        loss.set_stage(4)
        losses = loss(out, _matching_targets(out), stage=4)
        losses["total_loss"].backward()
        got = _grad_prefixes(m)
        assert "detector" in got
        assert "temporal" in got


class TestOBBTraining:
    def test_obb_losses_present_and_backward(self):
        m, cfg = _model(obb=True, stage=1)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5,
                            obb_cfg=cfg.detector.obb)
        loss.set_stage(1)
        losses = loss(out, _matching_targets(out, obb=True), stage=1)
        assert "loss_obb" in losses
        assert "loss_obb_angle" in losses
        assert "loss_obb_probiou" in losses
        losses["total_loss"].backward()
        assert "detector" in _grad_prefixes(m)

    def test_angle_head_receives_gradient(self):
        """角度ヘッドに実際に勾配が届くこと (OBB 学習が動く条件)。"""
        m, cfg = _model(obb=True, stage=1)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5,
                            obb_cfg=cfg.detector.obb)
        loss.set_stage(1)
        losses = loss(out, _matching_targets(out, obb=True), stage=1)
        losses["total_loss"].backward()
        angle_grads = [
            float(p.grad.norm())
            for n, p in m.named_parameters()
            if "angle_head" in n and p.grad is not None
        ]
        assert angle_grads and max(angle_grads) > 0

    def test_aabb_loss_keys_unchanged_for_legacy_config(self):
        """OBB を使わない構成では損失キーが従来と完全に同じ (後方互換)。"""
        m, _ = _model(obb=False, stage=1)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5)
        loss.set_stage(1)
        losses = loss(out, _matching_targets(out), stage=1)
        assert not any(k.startswith("loss_obb") for k in losses)


class TestNumericalHealth:
    @pytest.mark.parametrize("obb", [False, True])
    def test_no_nan_or_inf_in_losses(self, obb):
        m, cfg = _model(obb=obb, stage=4)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5,
                            obb_cfg=cfg.detector.obb if obb else None)
        loss.set_stage(4)
        losses = loss(out, _matching_targets(out, obb=obb), stage=4)
        for k, v in losses.items():
            assert torch.isfinite(v).all(), f"{k} が NaN/Inf"

    def test_empty_targets_do_not_break(self):
        m, _ = _model(stage=4)
        out = m(torch.randn(_B, _T, 3, _HW, _HW))
        empty = [{
            "boxes": torch.zeros(0, 4),
            "class_ids": torch.zeros(0, dtype=torch.long),
            "track_ids": torch.zeros(0, dtype=torch.long),
            "action_ids": torch.zeros(0, dtype=torch.long),
        } for _ in range(_B)]
        loss = CombinedLoss(LossConfig(), num_classes=1, num_actions=5)
        loss.set_stage(4)
        losses = loss(out, empty, stage=4)
        assert torch.isfinite(losses["total_loss"])
        losses["total_loss"].backward()   # 例外が出ないこと


class TestTrackAlignedVsQueryAligned:
    def test_both_modes_run(self):
        """query-aligned (非推奨) も後方互換として動き続けること。"""
        for track_aligned in (True, False):
            cfg = build_model_config("small")
            cfg.detector.backbone.pretrained = False
            cfg.detector.head.num_queries = 8
            cfg.detector.score_threshold = 0.0
            cfg.track_aligned_temporal = track_aligned
            m = build_model(cfg)
            m.set_stage(2)
            m.eval()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with torch.no_grad():
                    out = m(torch.randn(1, _T, 3, _HW, _HW))
            assert out.action_logits is not None

    def test_query_aligned_emits_deprecation_warning(self):
        cfg = build_model_config("small")
        cfg.detector.backbone.pretrained = False
        cfg.detector.head.num_queries = 8
        cfg.detector.score_threshold = 0.0
        cfg.track_aligned_temporal = False
        m = build_model(cfg)
        m.set_stage(2)
        m.eval()
        with pytest.warns(DeprecationWarning, match="query index"):
            with torch.no_grad():
                m(torch.randn(1, _T, 3, _HW, _HW))

    def test_track_aligned_populates_track_ids(self):
        cfg = build_model_config("small")
        cfg.detector.backbone.pretrained = False
        cfg.detector.head.num_queries = 8
        cfg.detector.score_threshold = 0.0
        m = build_model(cfg)
        m.set_stage(2)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(1, _T, 3, _HW, _HW))
        assert out.track_ids is not None
        assert out.behavior_rows is not None


class TestFullFeatureConfig:
    """
    全機能 (OBB + P2 + pose + multi-head + 全 behavior stream) を有効にした
    構成が実際に組み立てられ、forward まで通ることを固定する。

    個々の機能は別テストで検証しているが、「同時に有効にすると次元や配線が
    合わない」という組合せ起因の壊れ方はここでしか捕まえられない。
    """

    @pytest.fixture(scope="class")
    @classmethod
    def full_model(cls):
        from htrtdetr.config.config import (
            enable_obb, get_variant_config, clamp_temporal_branches, validate_config,
        )

        cfg = get_variant_config("small", stage=4)
        clamp_temporal_branches(cfg)
        enable_obb(cfg, use_p2=True)
        cfg.model.detector.backbone.pretrained = False
        cfg.model.detector.head.num_queries = 8
        cfg.model.detector.score_threshold = 0.0
        cfg.model.detector.pose.enabled = True
        cfg.model.detector.pose.num_keypoints = 3
        cfg.model.tracking.orientation_weight = 0.3
        cfg.model.tracking.motion_weight = 0.2
        cfg.model.behavior.use_pose = True
        cfg.model.behavior.classifier_type = "multi_head"
        cfg.model.behavior.heads = {
            "locomotion": ["stationary", "walking", "running"],
            "social": ["none", "courtship", "chase", "aggression"],
        }
        validate_config(cfg)
        m = build_model(cfg.model)
        m.set_stage(4)
        return m

    def test_all_behavior_groups_active(self, full_model):
        assert set(full_model.behavior.active_groups()) == {
            "actor", "context", "detector", "motion",
            "orientation", "interaction", "temporal", "pose",
        }

    def test_forward_produces_every_output(self, full_model):
        full_model.eval()
        with torch.no_grad():
            out = full_model(torch.randn(1, 4, 3, _HW, _HW))
        assert out.pred_obb is not None
        assert out.pred_keypoints is not None
        assert out.pred_polarity is not None
        assert out.track_ids is not None
        assert out.behavior is not None
        assert set(out.behavior.head_logits) == {"locomotion", "social"}

    def test_p2_level_present(self, full_model):
        assert full_model.detector.get_encoder_strides() == [4, 8, 16, 32]

    def test_explanation_api(self, full_model):
        full_model.eval()
        with torch.no_grad():
            out = full_model(torch.randn(1, 4, 3, _HW, _HW))
        gates = out.behavior_gates(0, 0)
        assert "pose" in gates and "interaction" in gates
        assert all(0.0 <= v <= 1.0 for v in gates.values())

    def test_train_forward(self, full_model):
        full_model.train()
        out = full_model(torch.randn(1, 4, 3, _HW, _HW))
        assert out.behavior is not None
