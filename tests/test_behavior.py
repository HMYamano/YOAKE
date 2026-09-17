"""
test_behavior.py — 行動認識ブランチのテスト

検証項目:
  * actor (回転 crop) / context (拡大 crop) の 2-stream ROI
  * kinematics (運動 / 向き / 個体間) の意味的な正しさ
  * evidence 分解の加法性 (説明が本物であること)
  * single / hierarchical / multi_head の 3 分類方式
  * ablation で本当にパラメータが消えること
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from htrtdetr.config.config import BehaviorConfig, ExplainabilityConfig
from htrtdetr.models.behavior import (
    INTERACTION_FEATURE_NAMES,
    MOTION_FEATURE_NAMES,
    ORIENTATION_FEATURE_NAMES,
    BehaviorBranch,
    ContextROIExtractor,
    EvidenceBehaviorHead,
    HierarchicalClassifier,
    KinematicsEncoder,
    MultiHeadClassifier,
    RotatedROIExtractor,
    compute_interaction_features,
    compute_motion_features,
    compute_orientation_features,
)

_MOT = {n: i for i, n in enumerate(MOTION_FEATURE_NAMES)}
_ORI = {n: i for i, n in enumerate(ORIENTATION_FEATURE_NAMES)}
_INT = {n: i for i, n in enumerate(INTERACTION_FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# ROI: actor (rotated) + context (expanded)
# ---------------------------------------------------------------------------

class TestROI:
    @pytest.fixture
    def diagonal_bar(self):
        """45 度の棒が描かれた特徴マップ。"""
        H = W = 64
        fm = torch.zeros(1, 1, H, W)
        for i in range(-20, 21):
            y = int(H / 2 + i * math.sin(math.pi / 4))
            x = int(W / 2 + i * math.cos(math.pi / 4))
            fm[0, 0, max(0, min(H - 1, y)), max(0, min(W - 1, x))] = 1.0
        return fm

    def test_orientation_normalization_straightens_the_body(self, diagonal_bar):
        """
        actor crop は体軸方向を正規化するので、斜めの個体が横向きに揃う。
        これが「向きに依らず同じ姿勢を同じ特徴に写す」ことの直接の確認。
        """
        ext = RotatedROIExtractor(output_size=9, expand=1.0)
        obb = torch.tensor([[0.5, 0.5, 45 / 64, 10 / 64, math.pi / 4]])
        out = ext(diagonal_bar, obb, torch.zeros(1, dtype=torch.long))
        rows = out[0, 0].sum(dim=1)
        assert rows.argmax().item() == 4, "棒が中央行に揃っていない"
        assert rows[4] > 5 * rows[[0, 1, 2, 7, 8]].max(), "エネルギーが中央行に集中していない"

    def test_without_normalization_body_stays_diagonal(self, diagonal_bar):
        ext = RotatedROIExtractor(output_size=9, expand=1.0,
                                  orientation_normalize=False)
        obb = torch.tensor([[0.5, 0.5, 45 / 64, 10 / 64, math.pi / 4]])
        rows = ext(diagonal_bar, obb, torch.zeros(1, dtype=torch.long))[0, 0].sum(dim=1)
        # 正規化ありのときのような一点集中にはならない
        assert rows[4] < 5 * rows[[0, 8]].max() + 1e-6

    def test_output_shape(self):
        ext = RotatedROIExtractor(output_size=7)
        fm = torch.randn(2, 8, 16, 16)
        obb = torch.rand(5, 5)
        bi = torch.tensor([0, 0, 1, 1, 1])
        assert ext(fm, obb, bi).shape == (5, 8, 7, 7)

    def test_empty_roi(self):
        ext = RotatedROIExtractor(output_size=7)
        out = ext(torch.randn(1, 8, 16, 16), torch.zeros(0, 5),
                  torch.zeros(0, dtype=torch.long))
        assert out.shape == (0, 8, 7, 7)

    def test_gradient_flows_to_feature_map(self):
        ext = RotatedROIExtractor(output_size=5)
        fm = torch.randn(1, 4, 16, 16, requires_grad=True)
        obb = torch.tensor([[0.5, 0.5, 0.3, 0.2, 0.4]])
        ext(fm, obb, torch.zeros(1, dtype=torch.long)).sum().backward()
        assert fm.grad is not None and fm.grad.abs().sum() > 0

    def test_context_box_is_larger_than_individual(self):
        """設計原則 2: 個体 = tight、行動 = expanded。"""
        ext = ContextROIExtractor(output_size=8, context_scale=2.0, square=False)
        obb = torch.tensor([[0.5, 0.5, 0.10, 0.04, 0.0]])
        ctx = ext.context_boxes(obb)
        assert (ctx[0, 2] - ctx[0, 0]) > 0.10 * 1.9
        assert (ctx[0, 3] - ctx[0, 1]) > 0.04 * 1.9

    def test_context_roi_is_axis_aligned(self):
        """
        context は回転させない。回転させると「相手がどちら側にいるか」という
        social context が失われるため。
        """
        H = W = 32
        fm = torch.zeros(1, 1, H, W)
        fm[0, 0, :, W // 2:] = 1.0   # 右半分だけ明るい
        ext = ContextROIExtractor(output_size=8, context_scale=1.0, square=False)
        obb = torch.tensor([[0.5, 0.5, 1.0, 1.0, math.pi / 2]])   # 90 度回転
        out = ext(fm, obb, torch.zeros(1, dtype=torch.long))
        left = out[0, 0, :, :4].mean()
        right = out[0, 0, :, 4:].mean()
        assert right > left, "context crop が回転してしまっている"


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

class TestKinematics:
    @pytest.fixture
    def walking_and_still(self):
        """track 0: 一定速度で右へ / track 1: 静止。"""
        T = 8
        h = torch.zeros(2, T, 5)
        for t in range(T):
            h[0, t] = torch.tensor([0.1 + 0.02 * t, 0.5, 0.08, 0.03, 0.0])
            h[1, t] = torch.tensor([0.8, 0.5, 0.08, 0.03, math.pi / 2])
        return h, torch.ones(2, T, dtype=torch.bool)

    def test_speed(self, walking_and_still):
        h, m = walking_and_still
        f = compute_motion_features(h, m)
        assert float(f[0, _MOT["speed"]]) == pytest.approx(0.02, abs=1e-5)
        assert float(f[1, _MOT["speed"]]) == pytest.approx(0.0, abs=1e-6)

    def test_straightness_of_linear_motion(self, walking_and_still):
        h, m = walking_and_still
        f = compute_motion_features(h, m)
        assert float(f[0, _MOT["straightness"]]) == pytest.approx(1.0, abs=1e-3)

    def test_zigzag_has_lower_straightness(self):
        T = 9
        h = torch.zeros(1, T, 5)
        for t in range(T):
            h[0, t] = torch.tensor([0.02 * t, 0.5 + 0.05 * (t % 2), 0.08, 0.03, 0.0])
        f = compute_motion_features(h, torch.ones(1, T, dtype=torch.bool))
        assert float(f[0, _MOT["straightness"]]) < 0.8

    def test_masked_gap_does_not_fake_velocity(self, walking_and_still):
        """検出が飛んだフレームがあっても速度が偽の急変を起こさない。"""
        h, m = walking_and_still
        m2 = m.clone()
        m2[0, 3] = False
        f = compute_motion_features(h, m2)
        assert float(f[0, _MOT["speed"]]) == pytest.approx(0.02, abs=1e-5)

    def test_acceleration(self):
        T = 6
        h = torch.zeros(1, T, 5)
        pos = 0.0
        for t in range(T):
            pos += 0.01 * t          # 速度が毎フレーム増える → 一定加速度
            h[0, t] = torch.tensor([pos, 0.5, 0.08, 0.03, 0.0])
        f = compute_motion_features(h, torch.ones(1, T, dtype=torch.bool))
        assert float(f[0, _MOT["accel_magnitude"]]) > 0

    def test_heading_vs_body_forward_vs_sideways(self):
        """前進 (体軸と進行方向が一致) と横歩きを区別できる。"""
        T = 4
        fwd = torch.zeros(1, T, 5)
        side = torch.zeros(1, T, 5)
        for t in range(T):
            fwd[0, t] = torch.tensor([0.1 * t, 0.5, 0.08, 0.03, 0.0])
            side[0, t] = torch.tensor([0.1 * t, 0.5, 0.08, 0.03, math.pi / 2])
        m = torch.ones(1, T, dtype=torch.bool)
        f_fwd = compute_orientation_features(fwd, m)
        f_side = compute_orientation_features(side, m)
        assert float(f_fwd[0, _ORI["heading_vs_body"]]) == pytest.approx(0.0, abs=1e-4)
        assert float(f_side[0, _ORI["heading_vs_body"]]) > 1.0

    def test_angular_velocity(self):
        T = 4
        h = torch.zeros(1, T, 5)
        for t in range(T):
            h[0, t] = torch.tensor([0.5, 0.5, 0.08, 0.03, 0.1 * t])
        f = compute_orientation_features(h, torch.ones(1, T, dtype=torch.bool))
        assert float(f[0, _ORI["angular_velocity"]]) == pytest.approx(0.1, abs=1e-4)

    def test_approach_rate_sign(self):
        """接近で正、離反で負。chase / avoid の区別に直結する。"""
        obb = torch.tensor([[0.2, 0.5, 0.08, 0.03, 0.0], [0.8, 0.5, 0.08, 0.03, 0.0]])
        toward = compute_interaction_features(obb, torch.tensor([[0.02, 0.0], [0.0, 0.0]]))
        away = compute_interaction_features(obb, torch.tensor([[-0.02, 0.0], [0.0, 0.0]]))
        assert float(toward[0, _INT["approach_rate"]]) > 0
        assert float(away[0, _INT["approach_rate"]]) < 0

    def test_facing_is_asymmetric(self):
        """
        A→B と B→A を別次元で持つ (対称な特徴では courtship/chase を区別できない)。
        A は B を向き、B は横を向く配置で確認する。
        """
        obb = torch.tensor([
            [0.2, 0.5, 0.08, 0.03, 0.0],           # A: +x 方向 = B の方を向く
            [0.8, 0.5, 0.08, 0.03, math.pi / 2],   # B: +y 方向 = A を向かない
        ])
        f = compute_interaction_features(obb, torch.zeros(2, 2))
        assert float(f[0, _INT["facing_to_nn"]]) == pytest.approx(0.0, abs=1e-5)
        assert float(f[0, _INT["faced_by_nn"]]) > 1.0

    def test_single_individual_has_defined_features(self):
        f = compute_interaction_features(
            torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0]]), torch.zeros(1, 2)
        )
        assert f.shape == (1, len(INTERACTION_FEATURE_NAMES))
        assert float(f[0, _INT["nn_distance"]]) == pytest.approx(1.0)
        assert torch.isfinite(f).all()

    def test_group_index_isolates_scenes(self):
        """別バッチ要素の個体を近傍として拾わない。"""
        obb = torch.tensor([[0.2, 0.5, 0.08, 0.03, 0.0], [0.8, 0.5, 0.08, 0.03, 0.0]])
        f = compute_interaction_features(
            obb, torch.zeros(2, 2), group_index=torch.tensor([0, 1])
        )
        assert float(f[0, _INT["nn_distance"]]) == pytest.approx(1.0)

    def test_all_features_finite(self):
        g = torch.Generator().manual_seed(0)
        h = torch.rand(5, 6, 5, generator=g)
        m = torch.rand(5, 6, generator=g) > 0.3
        m[:, -1] = True
        assert torch.isfinite(compute_motion_features(h, m)).all()
        assert torch.isfinite(compute_orientation_features(h, m)).all()
        assert torch.isfinite(compute_interaction_features(h[:, -1], torch.zeros(5, 2))).all()

    def test_encoder_ablation_drops_groups(self):
        enc = KinematicsEncoder(output_dim=8, use_motion=False)
        out = enc(torch.rand(3, 4, 5), torch.ones(3, 4, dtype=torch.bool))
        assert "motion" not in out
        assert set(out) == {"orientation", "interaction"}


# ---------------------------------------------------------------------------
# Evidence decomposition
# ---------------------------------------------------------------------------

class TestEvidenceHead:
    GROUPS = {"appearance": 12, "motion": 8, "interaction": 8, "temporal": 10}

    def _feats(self, n=5):
        return {k: torch.randn(n, d) for k, d in self.GROUPS.items()}

    def test_evidence_sums_to_logits(self):
        """
        説明の加法性: 各 group の寄与を足すと最終 logit に一致する。
        これが成り立つから寄与値をそのまま根拠として提示できる。
        """
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16)
        p = h(self._feats())
        assert torch.allclose(sum(p.evidence.values()), p.logits, atol=1e-5)

    def test_explain_returns_all_groups(self):
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16)
        ex = h(self._feats()).explain(0)
        assert set(ex) == set(self.GROUPS)
        assert all(isinstance(v, float) for v in ex.values())

    def test_explain_for_specific_action(self):
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16)
        p = h(self._feats())
        for aid in range(5):
            contrib = sum(p.explain(0, action_id=aid).values())
            assert contrib == pytest.approx(float(p.logits[0, aid].detach()), abs=1e-4)

    def test_gates_in_range(self):
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16,
                                 use_gating=True)
        g = h(self._feats()).gate_weights(0)
        assert set(g) == set(self.GROUPS)
        assert all(0.0 <= v <= 1.0 for v in g.values())

    def test_no_evidence_mode_has_no_explanation(self):
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16,
                                 use_evidence=False)
        p = h(self._feats())
        assert p.logits.shape == (5, 5)
        assert p.evidence == {}

    def test_missing_group_raises_clear_error(self):
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16)
        feats = self._feats()
        del feats["motion"]
        with pytest.raises(KeyError, match="motion"):
            h(feats)

    def test_backward(self):
        h = EvidenceBehaviorHead(self.GROUPS, num_actions=5, hidden_dim=16)
        h(self._feats()).logits.sum().backward()
        assert all(p.grad is not None for p in h.parameters() if p.requires_grad)


class TestHierarchicalClassifier:
    HIER = {
        "_root": ["Immobile", "Locomotion", "Active"],
        "Active": ["Self-directed", "Social"],
        "Self-directed": ["grooming", "feeding"],
        "Social": ["courtship", "chase", "aggression"],
    }
    LEAVES = ["Immobile", "Locomotion", "grooming", "feeding",
              "courtship", "chase", "aggression"]

    def test_leaf_logprobs_form_a_distribution(self):
        """条件付き確率の積なので、葉の確率の合計は 1 になる。"""
        clf = HierarchicalClassifier(16, self.HIER, self.LEAVES)
        out = clf(torch.randn(4, 16))
        probs = out["leaf_logits"].exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)

    def test_shape(self):
        clf = HierarchicalClassifier(16, self.HIER, self.LEAVES)
        assert clf(torch.randn(3, 16))["leaf_logits"].shape == (3, len(self.LEAVES))

    def test_unreachable_leaf_raises(self):
        with pytest.raises(ValueError, match="到達"):
            HierarchicalClassifier(16, self.HIER, ["nonexistent"])

    def test_missing_root_raises(self):
        with pytest.raises(ValueError, match="根ノード"):
            HierarchicalClassifier(16, {"Active": ["a"]}, ["a"])


class TestMultiHeadClassifier:
    HEADS = {
        "locomotion": ["stationary", "walking", "running"],
        "social": ["none", "courtship", "chase", "aggression"],
    }

    def test_head_shapes(self):
        clf = MultiHeadClassifier(16, self.HEADS)
        out = clf(torch.randn(4, 16))
        assert out["locomotion"].shape == (4, 3)
        assert out["social"].shape == (4, 4)

    def test_empty_heads_rejected(self):
        with pytest.raises(ValueError, match="head"):
            MultiHeadClassifier(16, {})


# ---------------------------------------------------------------------------
# BehaviorBranch (統合)
# ---------------------------------------------------------------------------

class TestBehaviorBranch:
    N, T, C = 4, 8, 32

    @pytest.fixture
    def inputs(self):
        hist = torch.rand(self.N, self.T, 5) * 0.5 + 0.25
        hist[:, :, 2:4] = 0.1
        return {
            "obb_history": hist,
            "history_mask": torch.ones(self.N, self.T, dtype=torch.bool),
            "spatial_features": torch.randn(2, self.C, 16, 16),
            "batch_index": torch.tensor([0, 0, 1, 1]),
            "detector_features": torch.randn(self.N, 24),
            "temporal_features": torch.randn(self.N, 20),
        }

    def _branch(self, **overrides):
        cfg = BehaviorConfig(feature_dim=24, actor_roi_size=6, context_roi_size=6)
        if overrides:
            cfg = dataclasses.replace(cfg, **overrides)
        return BehaviorBranch(
            cfg, num_actions=5, feature_channels=self.C,
            detector_dim=24, temporal_dim=20,
            explain_cfg=ExplainabilityConfig(),
        )

    def test_all_streams_active(self, inputs):
        b = self._branch()
        assert set(b.active_groups()) == {
            "actor", "context", "detector", "motion",
            "orientation", "interaction", "temporal",
        }
        p = b(**inputs)
        assert p.logits.shape == (self.N, 5)
        assert torch.allclose(sum(p.evidence.values()), p.logits, atol=1e-4)

    @pytest.mark.parametrize("flag", [
        "use_actor", "use_context", "use_detector_feat",
        "use_motion", "use_orientation", "use_interaction", "use_temporal",
    ])
    def test_ablation_removes_group_and_params(self, inputs, flag):
        """
        ablation は「ゼロ埋め」ではなく「group ごと消す」。パラメータが実際に
        減ることを確認する (そうでないと公平な比較にならない)。
        """
        full = self._branch()
        ablated = self._branch(**{flag: False})
        assert len(ablated.active_groups()) < len(full.active_groups())
        assert sum(p.numel() for p in ablated.parameters()) \
            < sum(p.numel() for p in full.parameters())
        out = ablated(**inputs)
        assert out.logits.shape == (self.N, 5)

    def test_hierarchical_classifier(self, inputs):
        b = BehaviorBranch(
            dataclasses.replace(
                BehaviorConfig(feature_dim=24, actor_roi_size=6, context_roi_size=6),
                classifier_type="hierarchical",
                hierarchy={"_root": ["Immobile", "Active"],
                           "Active": ["groom", "court", "chase"]},
            ),
            num_actions=4, feature_channels=self.C, detector_dim=24, temporal_dim=20,
            action_names=["Immobile", "groom", "court", "chase"],
            explain_cfg=ExplainabilityConfig(),
        )
        assert b(**inputs).logits.shape == (self.N, 4)

    def test_multi_head_classifier(self, inputs):
        b = self._branch(
            classifier_type="multi_head",
            heads={"locomotion": ["stat", "walk"], "social": ["none", "court", "chase"]},
        )
        p = b(**inputs)
        assert p.head_logits["locomotion"].shape == (self.N, 2)
        assert p.head_logits["social"].shape == (self.N, 3)

    def test_context_boxes_exposed_for_gui(self, inputs):
        b = self._branch(context_scale=2.5)
        ctx = b.context_boxes(inputs["obb_history"][:, -1])
        assert ctx.shape == (self.N, 4)

    def test_missing_required_input_raises(self, inputs):
        b = self._branch()
        bad = dict(inputs)
        bad["spatial_features"] = None
        with pytest.raises(ValueError, match="spatial_features"):
            b(**bad)

    def test_no_enabled_stream_raises(self):
        with pytest.raises(ValueError, match="feature group"):
            self._branch(
                use_actor=False, use_context=False, use_detector_feat=False,
                use_motion=False, use_orientation=False,
                use_interaction=False, use_temporal=False,
            )

    def test_backward(self, inputs):
        b = self._branch()
        b(**inputs).logits.sum().backward()
        grads = [p.grad for p in b.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
