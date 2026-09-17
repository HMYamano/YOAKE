"""
test_obb_detection.py — OBB detector と OBB 損失のテスト

検証項目:
  * detector backend の選択 (rtdetr_aabb / rtdetr_obb)
  * P2 (stride 4) の有効化
  * OBB 出力の形状と整合 (pred_boxes は常に OBB の外接矩形)
  * OBB 損失 (角度 / ProbIoU) と derived GT のマスク
  * AABB 構成が完全に後方互換であること
"""

from __future__ import annotations

import math

import pytest
import torch

from htrtdetr.config.config import (
    ConfigValidationError,
    LossConfig,
    OBBConfig,
    build_model_config,
    clamp_temporal_branches,
    enable_obb,
    get_variant_config,
    validate_config,
)
from htrtdetr.geometry.obb import encode_angle, obb_to_cxcywh
from htrtdetr.models.detector import available_detectors, build_detector
from htrtdetr.training.losses import DetectionLoss, MultiHeadActionLoss, OBBLoss, PoseLoss


def _cfg(obb=False, p2=False, pose=0):
    c = build_model_config("small")
    c.detector.backbone.pretrained = False
    c.detector.head.num_queries = 12
    if obb:
        c.detector.type = "rtdetr_obb"
        c.detector.obb.enabled = True
    if p2:
        c.detector.fpn.use_p2 = True
        c.detector.fpn.num_levels = 4
    if pose:
        c.detector.pose.enabled = True
        c.detector.pose.num_keypoints = pose
    return c


# ---------------------------------------------------------------------------
# Detector backend
# ---------------------------------------------------------------------------

class TestDetectorBackends:
    def test_registered_backends(self):
        assert "rtdetr_aabb" in available_detectors()
        assert "rtdetr_obb" in available_detectors()

    def test_aabb_has_no_obb_output(self):
        d = build_detector(_cfg().detector)
        out = d(torch.randn(2, 3, 128, 128))
        assert out.pred_obb is None
        assert out.pred_angles is None
        assert not out.has_obb

    def test_obb_outputs_shapes(self):
        c = _cfg(obb=True)
        d = build_detector(c.detector)
        out = d(torch.randn(2, 3, 128, 128))
        Q = c.detector.head.num_queries
        assert out.pred_obb.shape == (2, Q, 5)
        assert out.pred_angles.shape == (2, Q, 2)
        assert out.pred_boxes.shape == (2, Q, 4)
        assert out.has_obb

    def test_pred_boxes_is_circumscribing_obb(self):
        """OBB 有効時も pred_boxes は「OBB の外接矩形」という一貫した意味を保つ。"""
        d = build_detector(_cfg(obb=True).detector)
        out = d(torch.randn(1, 3, 128, 128))
        expected = obb_to_cxcywh(out.pred_obb)
        assert torch.allclose(out.pred_boxes, expected, atol=1e-5)

    def test_angle_head_initialized_to_zero_rotation(self):
        """
        角度ヘッドは theta=0 から学習を始める。回転 GT が少ないデータでも
        AABB 相当の状態から滑らかに学習が進むようにするため。
        """
        d = build_detector(_cfg(obb=True).detector)
        out = d(torch.randn(2, 3, 128, 128))
        assert out.pred_obb[..., 4].abs().max() < 1e-5

    def test_obb_type_without_enabled_flag_raises(self):
        c = _cfg()
        c.detector.type = "rtdetr_obb"      # obb.enabled は False のまま
        with pytest.raises(ValueError, match="obb.enabled"):
            build_detector(c.detector)

    def test_unknown_backend_raises(self):
        c = _cfg()
        c.detector.type = "nonexistent"
        with pytest.raises(ValueError, match="未知の detector backend"):
            build_detector(c.detector)

    def test_yolo_backend_points_to_pipeline(self):
        c = _cfg()
        c.detector.type = "yolo_obb"
        with pytest.raises(ValueError, match="orchestration"):
            build_detector(c.detector)


class TestP2:
    def test_p2_adds_stride4_level(self):
        d = build_detector(_cfg(p2=True).detector)
        assert d.get_encoder_strides() == [4, 8, 16, 32]
        out = d(torch.randn(1, 3, 128, 128))
        assert len(out.encoder_features) == 4
        # 128/4 = 32
        assert out.encoder_features[0].shape[-1] == 32

    def test_without_p2_is_three_levels(self):
        d = build_detector(_cfg().detector)
        assert d.get_encoder_strides() == [8, 16, 32]
        assert len(d(torch.randn(1, 3, 128, 128)).encoder_features) == 3

    def test_p2_not_in_memory_reduces_tokens(self):
        """p2_in_memory=False なら decoder memory のトークン数が減る。"""
        c = _cfg(p2=True)
        c.detector.fpn.p2_in_memory = False
        d = build_detector(c.detector)
        out = d(torch.randn(1, 3, 128, 128))
        # FPN は 4 レベルのまま (特徴強化には使う)
        assert len(out.encoder_features) == 4
        assert out.pred_boxes.shape[1] == c.detector.head.num_queries

    def test_p2_works_with_obb(self):
        d = build_detector(_cfg(obb=True, p2=True).detector)
        out = d(torch.randn(1, 3, 128, 128))
        assert out.pred_obb is not None
        assert len(out.encoder_features) == 4


class TestPoseHead:
    def test_disabled_by_default(self):
        out = build_detector(_cfg().detector)(torch.randn(1, 3, 128, 128))
        assert out.pred_keypoints is None
        assert out.pred_polarity is None

    def test_keypoint_shapes(self):
        c = _cfg(obb=True, pose=3)
        out = build_detector(c.detector)(torch.randn(2, 3, 128, 128))
        Q = c.detector.head.num_queries
        assert out.pred_keypoints.shape == (2, Q, 3, 3)
        assert out.pred_polarity.shape == (2, Q, 2)

    def test_keypoint_coords_are_normalized(self):
        c = _cfg(obb=True, pose=2)
        out = build_detector(c.detector)(torch.randn(1, 3, 128, 128))
        xy = out.pred_keypoints[..., :2]
        assert xy.min() >= 0.0 and xy.max() <= 1.0


# ---------------------------------------------------------------------------
# OBB Loss
# ---------------------------------------------------------------------------

class TestOBBLoss:
    GT = torch.tensor([[0.5, 0.5, 0.2, 0.1, 0.4], [0.2, 0.3, 0.1, 0.05, -0.9]])

    def test_perfect_prediction_is_near_zero(self):
        loss = OBBLoss(LossConfig(), OBBConfig(enabled=True))
        out = loss(self.GT.clone(), encode_angle(self.GT[:, 4]), self.GT)
        assert float(out["loss_obb_angle"]) == pytest.approx(0.0, abs=1e-5)
        assert float(out["loss_obb_probiou"]) < 0.01

    def test_wrong_angle_is_penalized(self):
        loss = OBBLoss(LossConfig(), OBBConfig(enabled=True))
        bad = self.GT.clone()
        bad[:, 4] += math.pi / 2
        out = loss(bad, encode_angle(bad[:, 4]), self.GT)
        assert float(out["loss_obb_angle"]) > 0.5

    def test_derived_gt_masks_angle_loss_only(self):
        """
        AABB 由来の代用 GT (obb_source="derived") は角度損失から除外される。
        位置・サイズの情報は有効なので ProbIoU 項は残る。
        """
        loss = OBBLoss(LossConfig(), OBBConfig(enabled=True))
        bad = self.GT.clone()
        bad[:, 4] += math.pi / 2
        out = loss(bad, encode_angle(bad[:, 4]), self.GT,
                   angle_valid=torch.zeros(2, dtype=torch.bool))
        assert float(out["loss_obb_angle"]) == pytest.approx(0.0, abs=1e-6)
        assert float(out["loss_obb_probiou"]) > 0.1

    def test_partial_mask(self):
        loss = OBBLoss(LossConfig(), OBBConfig(enabled=True))
        bad = self.GT.clone()
        bad[:, 4] += math.pi / 2
        out = loss(bad, encode_angle(bad[:, 4]), self.GT,
                   angle_valid=torch.tensor([True, False]))
        assert float(out["loss_obb_angle"]) > 0

    def test_empty_input(self):
        loss = OBBLoss(LossConfig(), OBBConfig(enabled=True))
        out = loss(torch.zeros(0, 5), torch.zeros(0, 2), torch.zeros(0, 5))
        assert float(out["loss_obb"]) == 0.0

    def test_gradient_flows(self):
        loss = OBBLoss(LossConfig(), OBBConfig(enabled=True))
        pred = self.GT.clone().requires_grad_(True)
        ang = encode_angle(self.GT[:, 4]).clone().requires_grad_(True)
        loss(pred, ang, self.GT)["loss_obb"].backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        assert ang.grad is not None


class TestDetectionLossWithOBB:
    @pytest.fixture
    def preds(self):
        g = torch.Generator().manual_seed(0)
        B, Q = 2, 16
        boxes = torch.rand(B, Q, 4, generator=g).clamp(0.05, 0.9)
        return {
            "pred_logits": torch.randn(B, Q, 2, generator=g),
            "pred_boxes": boxes,
            "pred_obb": torch.cat([boxes, torch.randn(B, Q, 1, generator=g)], -1),
            "pred_angles": torch.randn(B, Q, 2, generator=g),
        }

    def _targets(self, with_obb: bool):
        t = {
            "boxes": torch.tensor([[0.1, 0.1, 0.3, 0.2]]),
            "class_ids": torch.tensor([0]),
        }
        if with_obb:
            t["obb"] = torch.tensor([[0.2, 0.15, 0.2, 0.1, 0.3]])
            t["obb_valid"] = torch.tensor([True])
        return [dict(t) for _ in range(2)]

    def test_aabb_loss_keys_unchanged(self, preds):
        """OBB 無効時の損失キーは従来と完全に同一 (後方互換)。"""
        loss = DetectionLoss(LossConfig(), num_classes=1)
        out = loss(preds["pred_logits"], preds["pred_boxes"], self._targets(False))
        assert set(out) == {"loss_class", "loss_bbox", "loss_giou", "loss_detection"}

    def test_obb_loss_adds_keys(self, preds):
        loss = DetectionLoss(LossConfig(), num_classes=1, obb_cfg=OBBConfig(enabled=True))
        out = loss(preds["pred_logits"], preds["pred_boxes"], self._targets(True),
                   pred_obb=preds["pred_obb"], pred_angles=preds["pred_angles"])
        assert "loss_obb" in out and "loss_obb_angle" in out
        assert float(out["loss_detection"]) > 0

    def test_gt_without_obb_masks_angle(self, preds):
        """GT に OBB が無ければ角度損失は 0 になる (学習を壊さない)。"""
        loss = DetectionLoss(LossConfig(), num_classes=1, obb_cfg=OBBConfig(enabled=True))
        out = loss(preds["pred_logits"], preds["pred_boxes"], self._targets(False),
                   pred_obb=preds["pred_obb"], pred_angles=preds["pred_angles"])
        assert float(out["loss_obb_angle"]) == pytest.approx(0.0, abs=1e-6)

    def test_rotated_matching_runs(self, preds):
        cfg = OBBConfig(enabled=True, use_rotated_matching=True)
        loss = DetectionLoss(LossConfig(), num_classes=1, obb_cfg=cfg)
        out = loss(preds["pred_logits"], preds["pred_boxes"], self._targets(True),
                   pred_obb=preds["pred_obb"], pred_angles=preds["pred_angles"])
        assert torch.isfinite(out["loss_detection"])

    def test_empty_targets(self, preds):
        loss = DetectionLoss(LossConfig(), num_classes=1, obb_cfg=OBBConfig(enabled=True))
        empty = [{"boxes": torch.zeros(0, 4), "class_ids": torch.zeros(0, dtype=torch.long)}] * 2
        out = loss(preds["pred_logits"], preds["pred_boxes"], empty,
                   pred_obb=preds["pred_obb"], pred_angles=preds["pred_angles"])
        assert torch.isfinite(out["loss_detection"])


class TestPoseLoss:
    def test_keypoint_loss_masks_invisible(self):
        loss = PoseLoss(LossConfig())
        pred = torch.zeros(2, 3, 3)
        gt = torch.zeros(2, 3, 3)
        gt[..., :2] = 1.0
        gt[..., 2] = 0.0             # 全て不可視 → 座標損失は 0
        out = loss(pred, gt)
        assert float(out["loss_pose_keypoint"]) == pytest.approx(0.0, abs=1e-6)

    def test_visible_keypoints_contribute(self):
        loss = PoseLoss(LossConfig())
        pred = torch.zeros(1, 2, 3)
        gt = torch.zeros(1, 2, 3)
        gt[0, 0, :2] = 0.5
        gt[0, 0, 2] = 2.0            # 可視
        out = loss(pred, gt)
        assert float(out["loss_pose_keypoint"]) > 0

    def test_polarity_loss(self):
        loss = PoseLoss(LossConfig())
        out = loss(None, None,
                   pred_polarity=torch.tensor([[2.0, -2.0]]),
                   gt_polarity=torch.tensor([1]))
        assert float(out["loss_pose_polarity"]) > 0

    def test_no_gt_is_zero(self):
        loss = PoseLoss(LossConfig())
        out = loss(torch.zeros(1, 2, 3), None)
        assert float(out["loss_pose"]) == 0.0


class TestMultiHeadLoss:
    def test_per_head_losses(self):
        loss = MultiHeadActionLoss(LossConfig())
        out = loss(
            {"loco": torch.randn(4, 3), "social": torch.randn(4, 2)},
            {"loco": torch.tensor([0, 1, 2, 0]), "social": torch.tensor([0, 1, 0, 1])},
        )
        assert "loss_action_loco" in out and "loss_action_social" in out
        assert float(out["loss_action"]) > 0

    def test_missing_head_gt_is_skipped(self):
        """GT が無い head は学習対象から外れる (部分アノテーション対応)。"""
        loss = MultiHeadActionLoss(LossConfig())
        out = loss(
            {"loco": torch.randn(4, 3), "social": torch.randn(4, 2)},
            {"loco": torch.tensor([0, 1, 2, 0])},
        )
        assert "loss_action_loco" in out
        assert "loss_action_social" not in out

    def test_all_ignored_labels(self):
        loss = MultiHeadActionLoss(LossConfig())
        out = loss({"loco": torch.randn(2, 3)}, {"loco": torch.tensor([-1, -1])})
        assert float(out["loss_action"]) == 0.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestOBBConfig:
    def test_enable_obb_helper(self):
        cfg = get_variant_config("small", stage=1)
        clamp_temporal_branches(cfg)
        enable_obb(cfg, use_p2=True)
        validate_config(cfg)
        assert cfg.model.detector.type == "rtdetr_obb"
        assert cfg.model.detector.fpn.num_levels == 4

    def test_p2_without_num_levels_rejected(self):
        cfg = get_variant_config("small", stage=1)
        clamp_temporal_branches(cfg)
        cfg.model.detector.fpn.use_p2 = True     # num_levels は 3 のまま
        with pytest.raises(ConfigValidationError, match="num_levels"):
            validate_config(cfg)

    def test_orientation_weight_requires_obb(self):
        cfg = get_variant_config("small", stage=1)
        clamp_temporal_branches(cfg)
        cfg.model.tracking.orientation_weight = 0.5
        with pytest.raises(ConfigValidationError, match="orientation_weight"):
            validate_config(cfg)

    def test_rotated_iou_falls_back_without_obb(self):
        """use_rotated_iou=True + AABB は有効な組合せ (自動 fallback)。"""
        cfg = get_variant_config("small", stage=1)
        clamp_temporal_branches(cfg)
        assert cfg.model.tracking.use_rotated_iou is True
        assert cfg.model.detector.obb.enabled is False
        validate_config(cfg)     # 例外が出ないこと

    def test_yaml_roundtrip_preserves_obb(self, tmp_path):
        cfg = get_variant_config("small", stage=1)
        clamp_temporal_branches(cfg)
        enable_obb(cfg)
        p = tmp_path / "c.yaml"
        cfg.save_yaml(str(p))
        from htrtdetr.config.config import HTRTDETRConfig
        back = HTRTDETRConfig.from_yaml(str(p))
        validate_config(back)
        assert back.model.detector.obb.enabled
        assert back.model.detector.type == "rtdetr_obb"


# ---------------------------------------------------------------------------
# ONNX export wrappers
# ---------------------------------------------------------------------------

class TestONNXWrappers:
    """
    ONNX export ラッパーの疎通テスト。

    `HTRTDETROutput` は属性アクセス型で subscript できないので、ラッパーが
    `out["pred_logits"]` と書いていると TypeError になる (実際にそうなっていた)。
    onnx パッケージ無しでも動くよう、ここでは wrapper の forward だけを検証する。
    """

    def _wrappers(self):
        import importlib.util
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "tools" / "export_onnx.py"
        spec = importlib.util.spec_from_file_location("_export_onnx", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_detector_wrapper_returns_logits_and_boxes(self):
        from htrtdetr.models import build_model

        mod = self._wrappers()
        c = _cfg()
        m = build_model(c)
        m.set_stage(1)
        m.eval()
        with torch.no_grad():
            out = mod.DetectorWrapper(m)(torch.randn(1, 3, 128, 128))
        assert len(out) == 2
        assert out[0].shape == (1, c.detector.head.num_queries, 2)
        assert out[1].shape == (1, c.detector.head.num_queries, 4)

    def test_obb_wrapper_returns_obb(self):
        from htrtdetr.models import build_model

        mod = self._wrappers()
        c = _cfg(obb=True)
        m = build_model(c)
        m.set_stage(1)
        m.eval()
        with torch.no_grad():
            out = mod.OBBDetectorWrapper(m)(torch.randn(1, 3, 128, 128))
        assert len(out) == 3
        assert out[2].shape == (1, c.detector.head.num_queries, 5)

    def test_obb_wrapper_rejects_aabb_model(self):
        from htrtdetr.models import build_model

        mod = self._wrappers()
        m = build_model(_cfg())
        m.set_stage(1)
        m.eval()
        with pytest.raises(ValueError, match="OBB detector"):
            mod.OBBDetectorWrapper(m)(torch.randn(1, 3, 128, 128))
