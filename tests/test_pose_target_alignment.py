"""
test_pose_target_alignment.py — 部分的な pose アノテーションで GT が別画像へ漏れないこと

=== 何を守るテストか ===

pose (keypoints) は画像単位で任意のアノテーションなので、1 バッチの中に
「keypoint GT のある画像」と「無い画像」が混ざる。GT 収集が後者を読み飛ばすと、

    prediction: (2, K, 3)      GT: (1, K, 3)

となり、PyTorch の broadcasting が成立して**エラーにならないまま**画像 A の GT が
画像 B の予測にも適用される。損失は有限値で返り、学習も進むので、この壊れ方は
テストが無い限り誰にも気付かれない。

そこで:
  * GT の無い画像には可視性 0 のプレースホルダ行を入れて行数を揃える
  * それでも揃わなければ ``PoseLoss`` が例外で止める (黙って broadcast させない)
"""

from __future__ import annotations

import pytest
import torch

from htrtdetr.config.config import LossConfig
from htrtdetr.training.losses import PoseLoss
from htrtdetr.training.matching import collect_matched_gt, collect_matched_gt_tensor

K = 3


def _det(n: int, with_kp: bool = True):
    """n 件の予測を持つ det_results 要素。bbox は全部同じ場所に置く。"""
    d = {"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]).repeat(n, 1)}
    if with_kp:
        d["keypoints"] = torch.zeros(n, K, 3)
    return d


def _target(n_gt: int, keypoints: bool, value: float = 1.0):
    t = {
        "boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]]).repeat(n_gt, 1),
        "class_ids": torch.zeros(n_gt, dtype=torch.long),
    }
    if keypoints:
        kp = torch.full((n_gt, K, 3), value)
        kp[..., 2] = 2.0                      # visible
        t["keypoints"] = kp
    return t


def _n_pred(det_results) -> int:
    return sum(int(d["boxes"].shape[0]) for d in det_results)


class TestKeypointGtAlignment:
    def test_all_images_annotated(self):
        dets = [_det(1), _det(1)]
        targets = [_target(1, True, 1.0), _target(1, True, 2.0)]
        gt = collect_matched_gt_tensor(dets, targets, "keypoints")
        assert gt.shape[0] == _n_pred(dets) == 2
        assert gt[0, 0, 0].item() == pytest.approx(1.0)
        assert gt[1, 0, 0].item() == pytest.approx(2.0)

    def test_only_some_images_annotated_keeps_row_count(self):
        """
        P0-4 回帰。修正前は GT の無い画像を ``continue`` で飛ばしていたので
        gt.shape[0] == 1 になり、画像 A の GT が画像 B にも適用されていた。
        """
        dets = [_det(1), _det(1)]
        targets = [_target(1, True, 5.0), _target(1, False)]
        gt = collect_matched_gt_tensor(dets, targets, "keypoints")

        assert gt.shape[0] == _n_pred(dets) == 2, "予測と GT の行数が揃っていない"
        assert gt[0, 0, 0].item() == pytest.approx(5.0)
        assert torch.count_nonzero(gt[1]) == 0, (
            "keypoint の無い画像にプレースホルダではなく実 GT が入った"
        )

    def test_unannotated_image_is_excluded_from_the_loss(self):
        """プレースホルダは可視性 0 なので損失に寄与しない。"""
        loss = PoseLoss(LossConfig())
        pred = torch.zeros(2, K, 3, requires_grad=True)

        gt_both = torch.zeros(2, K, 3)
        gt_both[..., 2] = 2.0
        gt_both[..., :2] = 1.0
        full = loss(pred, gt_both)["loss_pose_keypoint"]

        gt_partial = gt_both.clone()
        gt_partial[1] = 0.0      # 2 枚目は未アノテーション (v=0)
        partial = loss(pred, gt_partial)["loss_pose_keypoint"]

        assert full.item() > 0
        assert partial.item() == pytest.approx(full.item()), (
            "未アノテーション画像が損失の分母/分子に混ざっている"
        )

    def test_batch_without_any_keypoints_yields_no_keypoint_loss(self):
        dets = [_det(1), _det(1)]
        targets = [_target(1, False), _target(1, False)]
        gt = collect_matched_gt_tensor(dets, targets, "keypoints")
        assert gt.shape[0] == 0

        loss = PoseLoss(LossConfig())
        out = loss(torch.zeros(2, K, 3, requires_grad=True), None)
        assert "loss_pose_keypoint" not in out
        assert out["loss_pose"].item() == pytest.approx(0.0)

    def test_varying_prediction_counts_per_image(self):
        dets = [_det(3), _det(1), _det(2)]
        targets = [_target(1, True), _target(1, False), _target(2, True)]
        gt = collect_matched_gt_tensor(dets, targets, "keypoints")
        assert gt.shape[0] == _n_pred(dets) == 6
        # 2 枚目 (index 3) はプレースホルダ
        assert torch.count_nonzero(gt[3]) == 0

    def test_unmatched_predictions_get_zero_rows(self):
        """IoU が閾値未満の予測はゼロ行 = 可視性 0 で損失から外れる。"""
        dets = [{"boxes": torch.tensor([[0.05, 0.05, 0.02, 0.02]])}]
        dets[0]["keypoints"] = torch.zeros(1, K, 3)
        targets = [_target(1, True, 9.0)]
        gt = collect_matched_gt_tensor(dets, targets, "keypoints")
        assert gt.shape[0] == 1
        assert torch.count_nonzero(gt[0]) == 0

    def test_image_with_no_predictions_contributes_no_rows(self):
        dets = [_det(0), _det(2)]
        targets = [_target(1, True), _target(2, True, 3.0)]
        gt = collect_matched_gt_tensor(dets, targets, "keypoints")
        assert gt.shape[0] == _n_pred(dets) == 2

    def test_target_with_key_but_zero_gt_objects(self):
        dets = [_det(2)]
        t = {"boxes": torch.zeros(0, 4), "keypoints": torch.zeros(0, K, 3)}
        gt = collect_matched_gt_tensor(dets, [t], "keypoints")
        assert gt.shape[0] == 2
        assert torch.count_nonzero(gt) == 0


class TestPoseLossRefusesToBroadcast:
    def test_mismatched_keypoint_rows_raise(self):
        loss = PoseLoss(LossConfig())
        with pytest.raises(ValueError, match="pose keypoint"):
            loss(torch.zeros(2, K, 3), torch.zeros(1, K, 3))

    def test_mismatched_polarity_rows_raise(self):
        loss = PoseLoss(LossConfig())
        with pytest.raises(ValueError, match="pose polarity"):
            loss(None, None, torch.zeros(3, 2), torch.zeros(1, dtype=torch.long))

    def test_matching_rows_do_not_raise(self):
        loss = PoseLoss(LossConfig())
        out = loss(torch.zeros(2, K, 3), torch.zeros(2, K, 3))
        assert torch.isfinite(out["loss_pose"]).all()


class TestPolarityAlignment:
    def test_polarity_rows_match_predictions_even_when_partially_annotated(self):
        dets = [_det(2, with_kp=False), _det(1, with_kp=False)]
        targets = [
            {
                "boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]]),
                "polarity": torch.tensor([1], dtype=torch.long),
            },
            {"boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]])},   # polarity なし
        ]
        gt = collect_matched_gt(dets, targets, key="polarity")
        assert gt.shape[0] == _n_pred(dets) == 3
        assert gt[2].item() == -1, "polarity の無い画像は ignore (-1) になるべき"
