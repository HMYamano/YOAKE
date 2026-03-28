"""
test_model_forward.py — HTRTDETR モデルの forward パステスト

build_model_config("small") (pretrained=False にオーバーライド) を使い、
各 stage の forward が正しい出力形状と型を返すかを検証する。
"""

from __future__ import annotations

import pytest
import torch

from htrtdetr.config.config import build_model_config
from htrtdetr.models import build_model


# ---------------------------------------------------------------------------
# 共通定数
# ---------------------------------------------------------------------------

_B = 2       # バッチサイズ
_T = 4       # シーケンス長
_H = 128     # 画像高さ
_W = 128     # 画像幅


# ---------------------------------------------------------------------------
# 共通フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg():
    """small バリアント + pretrained=False の ModelConfig"""
    c = build_model_config("small")
    c.detector.backbone.pretrained = False
    return c


@pytest.fixture(scope="module")
def num_queries(cfg):
    return cfg.detector.head.num_queries


@pytest.fixture(scope="module")
def num_classes_with_bg(cfg):
    """num_classes + background (1) = 出力の最終次元"""
    return cfg.detector.head.num_classes + 1


@pytest.fixture(scope="module")
def model_stage1(cfg):
    m = build_model(cfg)
    m.set_stage(1)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_stage2(cfg):
    m = build_model(cfg)
    m.set_stage(2)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_stage3(cfg):
    m = build_model(cfg)
    m.set_stage(3)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_stage4(cfg):
    m = build_model(cfg)
    m.set_stage(4)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Stage 1: forward_single_frame
# ---------------------------------------------------------------------------

class TestStage1ForwardSingleFrame:
    def test_pred_logits_shape(self, model_stage1, num_queries, num_classes_with_bg):
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = model_stage1.forward_single_frame(images)
        expected = (_B, num_queries, num_classes_with_bg)
        assert out.pred_logits.shape == expected, (
            f"期待 {expected}, 実際 {out.pred_logits.shape}"
        )

    def test_pred_boxes_shape(self, model_stage1, num_queries):
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = model_stage1.forward_single_frame(images)
        assert out.pred_boxes.shape == (_B, num_queries, 4)

    def test_pred_boxes_in_unit_range(self, model_stage1):
        """pred_boxes は [0, 1] の正規化座標であること"""
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = model_stage1.forward_single_frame(images)
        boxes = out.pred_boxes
        assert boxes.min().item() >= 0.0 - 1e-4
        assert boxes.max().item() <= 1.0 + 1e-4

    def test_no_nan_in_outputs(self, model_stage1):
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = model_stage1.forward_single_frame(images)
        assert not torch.isnan(out.pred_logits).any(), "pred_logits に NaN"
        assert not torch.isnan(out.pred_boxes).any(), "pred_boxes に NaN"

    def test_stage1_has_no_action_logits(self, model_stage1):
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = model_stage1.forward_single_frame(images)
        assert out.action_logits is None

    def test_stage1_has_no_id_logits(self, model_stage1):
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = model_stage1.forward_single_frame(images)
        assert out.id_logits is None


# ---------------------------------------------------------------------------
# Stage 1: freeze チェック
# ---------------------------------------------------------------------------

class TestStage1Freeze:
    def test_detector_params_are_trainable(self, model_stage1):
        trainable = [
            n for n, p in model_stage1.named_parameters()
            if p.requires_grad and "detector" in n
        ]
        assert len(trainable) > 0, "Stage 1: detector のパラメータが全て freeze されている"

    def test_temporal_params_are_frozen(self, model_stage1):
        """Stage 1 では temporal module は freeze されているはず"""
        trainable_temporal = [
            n for n, p in model_stage1.named_parameters()
            if p.requires_grad and "temporal" in n
        ]
        assert len(trainable_temporal) == 0, (
            f"Stage 1: temporal params が freeze されていない: {trainable_temporal[:3]}"
        )


# ---------------------------------------------------------------------------
# Stage 2: sequence forward (action head)
# ---------------------------------------------------------------------------

class TestStage2Forward:
    def test_pred_logits_shape(self, model_stage2, num_queries, num_classes_with_bg):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage2(images)
        expected = (_B, num_queries, num_classes_with_bg)
        assert out.pred_logits.shape == expected, (
            f"期待 {expected}, 実際 {out.pred_logits.shape}"
        )

    def test_pred_boxes_shape(self, model_stage2, num_queries):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage2(images)
        assert out.pred_boxes.shape == (_B, num_queries, 4)

    def test_no_nan_in_outputs(self, model_stage2):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage2(images)
        assert not torch.isnan(out.pred_logits).any()
        assert not torch.isnan(out.pred_boxes).any()

    def test_stage2_action_logits_shape_or_none(self, model_stage2, cfg):
        """action_logits は (N_det, num_actions) または None"""
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage2(images)
        if out.action_logits is not None:
            assert out.action_logits.ndim == 2
            assert out.action_logits.shape[-1] == cfg.action_head.num_actions

    def test_stage2_no_id_logits(self, model_stage2):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage2(images)
        assert out.id_logits is None


# ---------------------------------------------------------------------------
# Stage 3: sequence forward (ID head)
# ---------------------------------------------------------------------------

class TestStage3Forward:
    def test_pred_logits_shape(self, model_stage3, num_queries, num_classes_with_bg):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage3(images)
        assert out.pred_logits.shape == (_B, num_queries, num_classes_with_bg)

    def test_stage3_id_logits_shape_or_none(self, model_stage3):
        """id_logits は (N_det, max_ids+1) または None"""
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage3(images)
        if out.id_logits is not None:
            assert out.id_logits.ndim == 2

    def test_stage3_no_action_logits(self, model_stage3):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage3(images)
        assert out.action_logits is None

    def test_no_nan_in_outputs(self, model_stage3):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage3(images)
        assert not torch.isnan(out.pred_logits).any()
        assert not torch.isnan(out.pred_boxes).any()


# ---------------------------------------------------------------------------
# Stage 4: sequence forward (全モジュール)
# ---------------------------------------------------------------------------

class TestStage4Forward:
    def test_pred_logits_shape(self, model_stage4, num_queries, num_classes_with_bg):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage4(images)
        assert out.pred_logits.shape == (_B, num_queries, num_classes_with_bg)

    def test_pred_boxes_shape(self, model_stage4, num_queries):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage4(images)
        assert out.pred_boxes.shape == (_B, num_queries, 4)

    def test_no_nan_in_outputs(self, model_stage4):
        images = torch.randn(_B, _T, 3, _H, _W)
        with torch.no_grad():
            out = model_stage4(images)
        assert not torch.isnan(out.pred_logits).any()
        assert not torch.isnan(out.pred_boxes).any()

    def test_all_params_trainable_in_stage4(self, model_stage4):
        """Stage 4 では全モジュールが学習可能であること"""
        frozen = [
            n for n, p in model_stage4.named_parameters()
            if not p.requires_grad
        ]
        assert len(frozen) == 0, (
            f"Stage 4: 以下のパラメータが freeze されている: {frozen[:5]}"
        )


# ---------------------------------------------------------------------------
# set_stage の冪等性と切り替え
# ---------------------------------------------------------------------------

class TestSetStage:
    def test_set_stage_is_idempotent(self, cfg, num_queries, num_classes_with_bg):
        m = build_model(cfg)
        m.set_stage(1)
        m.set_stage(1)  # 2 回連続でも壊れないこと
        m.eval()
        images = torch.randn(_B, 3, _H, _W)
        with torch.no_grad():
            out = m.forward_single_frame(images)
        assert out.pred_logits.shape == (_B, num_queries, num_classes_with_bg)

    def test_switch_stage_changes_trainable_params(self, cfg):
        m = build_model(cfg)
        m.set_stage(1)
        stage1_trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        m.set_stage(4)
        stage4_trainable = {n for n, p in m.named_parameters() if p.requires_grad}
        assert len(stage4_trainable) > len(stage1_trainable), (
            "Stage 4 は Stage 1 より多くのパラメータが学習可能であるべき"
        )
