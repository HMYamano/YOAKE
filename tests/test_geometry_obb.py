"""
test_geometry_obb.py — OBB 幾何ユーティリティのテスト

geometry/obb.py (torch) と stages/_geometry.py (numpy) の両方を検証し、
両実装が同じ値を返すことも固定する (幾何は 2 か所に複製されているため)。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from htrtdetr.geometry.obb import (
    aabb_to_obb,
    canonicalize_obb,
    corners_to_obb,
    decode_angle,
    encode_angle,
    encoded_to_obb,
    expand_aabb,
    normalize_angle,
    obb_area,
    obb_to_aabb,
    obb_to_context_aabb,
    obb_to_corners,
    obb_to_cxcywh,
    obb_to_encoded,
    probiou,
    rotated_iou,
    rotated_nms,
)
from htrtdetr.stages._geometry import (
    aabb_to_obb_np,
    normalize_angle_np,
    obb_to_aabb_np,
    obb_to_corners_np,
    orientation_consistency_np,
    rotated_iou_np,
)


def _angle_diff_mod_pi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """矩形の周期 pi を考慮した角度差 (同じ箱なら 0)。"""
    return ((a - b + math.pi / 2) % math.pi) - math.pi / 2


# ---------------------------------------------------------------------------
# 角度の正規化 / 符号化
# ---------------------------------------------------------------------------

class TestAngle:
    def test_normalize_angle_range(self):
        """normalize_angle は常に半開区間 [-pi/2, pi/2) に入る。"""
        th = torch.linspace(-4 * math.pi, 4 * math.pi, 257)
        out = normalize_angle(th)
        assert out.min() >= -math.pi / 2 - 1e-6
        assert out.max() < math.pi / 2

    def test_normalize_preserves_box(self):
        """正規化しても表す矩形は変わらない (差は pi の整数倍)。"""
        th = torch.linspace(-4 * math.pi, 4 * math.pi, 129)
        k = (th - normalize_angle(th)) / math.pi
        assert torch.allclose(k, k.round(), atol=1e-5)

    def test_encode_decode_roundtrip(self):
        """decode(encode(theta)) は theta と同じ箱を表す。"""
        th = torch.linspace(-math.pi, math.pi, 129)
        back = decode_angle(encode_angle(th))
        assert _angle_diff_mod_pi(back, th).abs().max() < 1e-5

    def test_angle_encoding_is_continuous_at_boundary(self):
        """
        周期境界 (±pi/2) で符号化が飛ばないこと。

        これが 2 倍角を使う理由そのもの。生の (sin t, cos t) だと
        theta=+pi/2-eps と theta=-pi/2+eps (ほぼ同じ箱) の符号が正反対になる。
        """
        eps = 1e-3
        a = encode_angle(torch.tensor([math.pi / 2 - eps]))
        b = encode_angle(torch.tensor([-math.pi / 2 + eps]))
        assert (a - b).abs().max() < 1e-2, "2 倍角符号化が境界で不連続"

    def test_decode_is_scale_invariant(self):
        """未正規化の出力 (ノルム != 1) でも角度が正しく復号される。"""
        th = torch.tensor([0.3, -1.1, 1.4])
        enc = encode_angle(th)
        for scale in (0.01, 1.0, 37.0):
            back = decode_angle(enc * scale)
            assert _angle_diff_mod_pi(back, th).abs().max() < 1e-5

    def test_obb_encoded_roundtrip(self):
        obb = torch.tensor([[5.0, 6.0, 4.0, 2.0, 0.6]])
        back = encoded_to_obb(obb_to_encoded(obb))
        assert torch.allclose(back[:, :4], obb[:, :4], atol=1e-5)
        assert _angle_diff_mod_pi(back[:, 4], obb[:, 4]).abs().max() < 1e-5


# ---------------------------------------------------------------------------
# 表現間の変換
# ---------------------------------------------------------------------------

class TestConversions:
    def test_obb_to_corners_shape_and_area(self):
        obb = torch.tensor([[0.0, 0.0, 10.0, 4.0, 0.0]])
        c = obb_to_corners(obb)
        assert c.shape == (1, 4, 2)
        # theta=0 なら軸平行の頂点になる
        expected = torch.tensor([[[-5.0, -2.0], [5.0, -2.0], [5.0, 2.0], [-5.0, 2.0]]])
        assert torch.allclose(c, expected, atol=1e-5)

    def test_corners_to_obb_roundtrip(self):
        obb = torch.tensor([
            [50.0, 60.0, 40.0, 10.0, 0.5],
            [10.0, 10.0, 4.0, 4.0, -1.2],
            [0.0, 0.0, 20.0, 5.0, 0.0],
        ])
        back = corners_to_obb(obb_to_corners(obb))
        canon = canonicalize_obb(obb)
        assert torch.allclose(back[:, :4], canon[:, :4], atol=1e-4)
        assert _angle_diff_mod_pi(back[:, 4], canon[:, 4]).abs().max() < 1e-4

    def test_obb_to_aabb_of_rotated_square(self):
        """45 度回した一辺 10 の正方形の外接矩形は半対角 = 10/sqrt(2)。"""
        obb = torch.tensor([[0.0, 0.0, 10.0, 10.0, math.pi / 4]])
        aabb = obb_to_aabb(obb)
        half = 10 * math.sqrt(2) / 2
        assert torch.allclose(
            aabb, torch.tensor([[-half, -half, half, half]]), atol=1e-4
        )

    def test_aabb_obb_roundtrip(self):
        aabb = torch.tensor([[10.0, 20.0, 30.0, 50.0]])
        assert torch.allclose(obb_to_aabb(aabb_to_obb(aabb)), aabb, atol=1e-5)

    def test_obb_to_cxcywh_is_circumscribing(self):
        """OBB → cxcywh は外接矩形なので theta!=0 では元より大きくなる。"""
        obb = torch.tensor([[0.0, 0.0, 10.0, 2.0, math.pi / 4]])
        box = obb_to_cxcywh(obb)
        assert box[0, 2] > obb[0, 2] * 0.7   # 外接なので幅が広がる
        assert box[0, 3] > obb[0, 3]

    def test_obb_area(self):
        obb = torch.tensor([[0.0, 0.0, 10.0, 4.0, 1.0]])
        assert float(obb_area(obb)) == pytest.approx(40.0)

    def test_degenerate_box_is_safe(self):
        """幅・高さ 0 の退化した箱でも例外にならない。"""
        obb = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.3]])
        assert float(obb_area(obb)) == 0.0
        assert torch.isfinite(obb_to_corners(obb)).all()
        assert float(rotated_iou(obb, obb)[0, 0]) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 厳密な rotated IoU
# ---------------------------------------------------------------------------

class TestRotatedIoU:
    def test_identical_boxes(self):
        obb = torch.tensor([[50.0, 60.0, 40.0, 10.0, 0.5], [0.0, 0.0, 6.0, 6.0, -1.0]])
        assert torch.allclose(rotated_iou(obb, obb).diag(), torch.ones(2), atol=1e-4)

    def test_disjoint_boxes(self):
        a = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        b = torch.tensor([[100.0, 100.0, 2.0, 2.0, 0.0]])
        assert float(rotated_iou(a, b)) == pytest.approx(0.0, abs=1e-6)

    def test_axis_aligned_matches_analytic(self):
        """10x10 が 5 ずれた場合: 交差 50 / 合併 150 = 1/3。"""
        a = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
        b = torch.tensor([[5.0, 0.0, 10.0, 10.0, 0.0]])
        assert float(rotated_iou(a, b)) == pytest.approx(1 / 3, abs=1e-4)

    def test_rotated_45_matches_analytic(self):
        """同心の正方形 (一辺 10) と 45 度回転版の交差は正八角形。"""
        s = torch.tensor([[0.0, 0.0, 10.0, 10.0, 0.0]])
        s45 = torch.tensor([[0.0, 0.0, 10.0, 10.0, math.pi / 4]])
        inter = 200 * (math.sqrt(2) - 1)
        expected = inter / (200 - inter)
        assert float(rotated_iou(s, s45)) == pytest.approx(expected, abs=1e-4)

    def test_rotation_separates_close_elongated_boxes(self):
        """
        OBB を使う動機の核心: 細長い個体が並んだとき、軸平行 IoU は大きくなるが
        rotated IoU は実体どおり小さくなる。
        """
        from htrtdetr.utils.misc import box_iou

        a = torch.tensor([[0.0, 0.0, 40.0, 6.0, math.pi / 4]])
        b = torch.tensor([[0.0, 0.0, 40.0, 6.0, -math.pi / 4]])   # 直交して交差
        r_iou = float(rotated_iou(a, b))
        a_iou = float(box_iou(obb_to_aabb(a), obb_to_aabb(b)))
        assert a_iou > 0.9, "軸平行 IoU はほぼ完全一致に見える"
        assert r_iou < 0.2, "rotated IoU は実際の重なりだけを見る"

    def test_matrix_shape(self):
        a = torch.rand(5, 5)
        b = torch.rand(3, 5)
        assert rotated_iou(a, b).shape == (5, 3)

    def test_empty_input(self):
        a = torch.zeros(0, 5)
        b = torch.rand(3, 5)
        assert rotated_iou(a, b).shape == (0, 3)

    def test_iou_symmetry(self):
        g = torch.Generator().manual_seed(0)
        a = torch.rand(6, 5, generator=g) * 50
        b = torch.rand(6, 5, generator=g) * 50
        a[:, 2:4] = a[:, 2:4].clamp(min=1.0)
        b[:, 2:4] = b[:, 2:4].clamp(min=1.0)
        assert torch.allclose(rotated_iou(a, b), rotated_iou(b, a).T, atol=1e-4)


class TestRotatedNMS:
    def test_suppresses_duplicates(self):
        obb = torch.tensor([
            [0.0, 0.0, 10.0, 4.0, 0.0],
            [0.3, 0.1, 10.0, 4.0, 0.02],   # ほぼ重複
            [100.0, 100.0, 10.0, 4.0, 0.0],
        ])
        keep = rotated_nms(obb, torch.tensor([0.9, 0.8, 0.7]), iou_threshold=0.5)
        assert keep.tolist() == [0, 2]

    def test_empty(self):
        assert rotated_nms(torch.zeros(0, 5), torch.zeros(0)).numel() == 0


# ---------------------------------------------------------------------------
# 微分可能な ProbIoU
# ---------------------------------------------------------------------------

class TestProbIoU:
    def test_identical_is_near_one(self):
        obb = torch.tensor([[0.0, 0.0, 20.0, 5.0, 0.3]])
        assert float(probiou(obb, obb)) > 0.99

    def test_far_apart_is_near_zero(self):
        a = torch.tensor([[0.0, 0.0, 20.0, 5.0, 0.3]])
        b = torch.tensor([[500.0, 500.0, 20.0, 5.0, 0.3]])
        assert float(probiou(a, b)) < 0.05

    def test_penalizes_wrong_angle(self):
        a = torch.tensor([[0.0, 0.0, 20.0, 5.0, 0.0]])
        same = torch.tensor([[0.0, 0.0, 20.0, 5.0, 0.0]])
        rot = torch.tensor([[0.0, 0.0, 20.0, 5.0, math.pi / 2]])
        assert float(probiou(a, same)) > float(probiou(a, rot))

    def test_gradient_flows_to_all_params(self):
        pred = torch.tensor([[1.0, 2.0, 18.0, 6.0, 0.1]], requires_grad=True)
        target = torch.tensor([[0.0, 0.0, 20.0, 5.0, 0.4]])
        (1 - probiou(pred, target)).sum().backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        # 中心・サイズ・角度のすべてに勾配が届く
        assert pred.grad.abs().sum() > 0
        assert pred.grad[0, 4].abs() > 0, "角度に勾配が流れていない"


# ---------------------------------------------------------------------------
# Behavior context box
# ---------------------------------------------------------------------------

class TestContextBox:
    def test_expand_scales_about_center(self):
        aabb = torch.tensor([[10.0, 10.0, 30.0, 20.0]])   # 20x10, center (20,15)
        out = expand_aabb(aabb, scale=2.0)
        assert torch.allclose(out, torch.tensor([[0.0, 5.0, 40.0, 25.0]]), atol=1e-5)

    def test_expand_scale_one_is_identity(self):
        aabb = torch.tensor([[10.0, 10.0, 30.0, 20.0]])
        assert torch.allclose(expand_aabb(aabb, 1.0), aabb, atol=1e-5)

    def test_square_option(self):
        aabb = torch.tensor([[0.0, 0.0, 20.0, 10.0]])
        out = expand_aabb(aabb, 1.0, square=True)
        w = out[0, 2] - out[0, 0]
        h = out[0, 3] - out[0, 1]
        assert float(w) == pytest.approx(float(h))

    def test_clipping_to_image(self):
        aabb = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
        out = expand_aabb(aabb, 4.0, image_size=(50, 50))
        assert out.min() >= 0
        assert out[0, 2] <= 50 and out[0, 3] <= 50

    def test_context_is_larger_than_individual(self):
        """行動 context は必ず個体の箱より大きい (設計原則 2)。"""
        obb = torch.tensor([[50.0, 50.0, 20.0, 6.0, 0.4]])
        tight = obb_to_aabb(obb)
        ctx = obb_to_context_aabb(obb, scale=2.0)
        assert ctx[0, 0] < tight[0, 0] and ctx[0, 1] < tight[0, 1]
        assert ctx[0, 2] > tight[0, 2] and ctx[0, 3] > tight[0, 3]


# ---------------------------------------------------------------------------
# torch 実装と numpy 実装の一致
# ---------------------------------------------------------------------------

class TestTorchNumpyParity:
    """
    幾何は torch (モデル用) と numpy (torch 非依存の段用) の 2 か所にある。
    両者がズレると段プレーンとモデルプレーンで結果が食い違うため、ここで固定する。
    """

    @pytest.fixture
    def samples(self):
        rng = np.random.default_rng(12345)
        return [
            np.array([
                rng.uniform(-50, 50), rng.uniform(-50, 50),
                rng.uniform(1, 40), rng.uniform(1, 40), rng.uniform(-2, 2),
            ], dtype=np.float32)
            for _ in range(60)
        ]

    def test_corners_match(self, samples):
        for a in samples:
            t = obb_to_corners(torch.from_numpy(a)[None])[0].numpy()
            n = obb_to_corners_np(a)
            assert np.abs(t - n).max() < 1e-4

    def test_aabb_match(self, samples):
        for a in samples:
            t = obb_to_aabb(torch.from_numpy(a)[None])[0].numpy()
            n = obb_to_aabb_np(a)
            assert np.abs(t - n).max() < 1e-4

    def test_rotated_iou_match(self, samples):
        for a, b in zip(samples[::2], samples[1::2]):
            t = float(rotated_iou(torch.from_numpy(a)[None], torch.from_numpy(b)[None]))
            n = rotated_iou_np(a, b)
            assert abs(t - n) < 1e-4, f"torch={t} numpy={n}"

    def test_normalize_angle_match(self):
        for th in np.linspace(-10, 10, 101):
            t = float(normalize_angle(torch.tensor([float(th)]))[0])
            n = normalize_angle_np(float(th))
            assert abs(((t - n + math.pi / 2) % math.pi) - math.pi / 2) < 1e-5

    def test_aabb_to_obb_match(self):
        aabb = np.array([10.0, 20.0, 30.0, 50.0], dtype=np.float32)
        t = aabb_to_obb(torch.from_numpy(aabb)[None])[0].numpy()
        n = aabb_to_obb_np(aabb)
        assert np.abs(t - n).max() < 1e-5


class TestOrientationConsistency:
    def test_identical_is_one(self):
        assert orientation_consistency_np(0.7, 0.7) == pytest.approx(1.0)

    def test_perpendicular_is_zero(self):
        assert orientation_consistency_np(0.0, math.pi / 2) == pytest.approx(0.0, abs=1e-6)

    def test_period_pi(self):
        """矩形の周期は pi なので theta と theta+pi は完全一致とみなす。"""
        assert orientation_consistency_np(0.3, 0.3 + math.pi) == pytest.approx(1.0, abs=1e-6)
