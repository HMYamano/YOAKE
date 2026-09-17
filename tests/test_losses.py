import math

import pytest
import torch
import torch.nn.functional as F

from htrtdetr.config.config import LossConfig
from htrtdetr.training.losses import ActionLoss


class TestComputeClassWeights:
    def test_uniform_distribution_gives_uniform_weights(self):
        counts = [100, 100, 100]
        w = ActionLoss.compute_class_weights(counts)
        assert all(abs(w[i].item() - w[0].item()) < 0.01 for i in range(3))

    def test_imbalanced_rare_class_gets_higher_weight(self):
        counts = [1000, 10, 1]
        w = ActionLoss.compute_class_weights(counts)
        assert w[2].item() > w[1].item() > w[0].item()

    def test_clipping_applied(self):
        w = ActionLoss.compute_class_weights([10000, 1], clip_min=0.1, clip_max=5.0)
        assert w.min().item() >= 0.1 - 1e-6
        assert w.max().item() <= 5.0 + 1e-6

    def test_zero_count_no_nan(self):
        w = ActionLoss.compute_class_weights([100, 0, 50], smoothing=1.0)
        assert not any(math.isnan(v.item()) for v in w)
        assert not any(math.isinf(v.item()) for v in w)

    def test_all_zeros_no_nan(self):
        w = ActionLoss.compute_class_weights([0, 0, 0], smoothing=1.0)
        assert not any(math.isnan(v.item()) for v in w)
        assert not any(math.isinf(v.item()) for v in w)


class TestActionLossForward:
    def _make_loss(self, strategy="class_weight"):
        cfg = LossConfig(imbalance_strategy=strategy)
        return ActionLoss(cfg, num_actions=3)

    def test_forward_class_weight_no_error(self):
        loss_fn = self._make_loss("class_weight")
        loss_fn.set_class_weights(ActionLoss.compute_class_weights([100, 10, 1]))
        logits = torch.randn(8, 3)
        gt = torch.tensor([0, 1, 2, 0, 1, 2, -1, -1])
        loss = loss_fn(logits, gt)

        assert isinstance(loss, torch.Tensor)
        assert not math.isnan(loss.item())
        assert not math.isinf(loss.item())
        assert loss.item() >= 0.0

    def test_forward_focal_no_error(self):
        loss_fn = self._make_loss("focal")
        logits = torch.randn(6, 3)
        gt = torch.tensor([0, 1, 2, 0, -1, -1])
        loss = loss_fn(logits, gt)
        assert not math.isnan(loss.item())

    def test_forward_none_no_error(self):
        loss_fn = self._make_loss("none")
        logits = torch.randn(4, 3)
        gt = torch.tensor([0, 1, 2, -1])
        loss = loss_fn(logits, gt)
        assert not math.isnan(loss.item())

    def test_empty_batch_returns_zero(self):
        loss_fn = self._make_loss("class_weight")
        logits = torch.zeros(0, 3)
        gt = torch.zeros(0, dtype=torch.long)
        loss = loss_fn(logits, gt)
        assert loss.item() == 0.0

    def test_all_ignore_returns_finite(self):
        loss_fn = self._make_loss("class_weight")
        loss_fn.set_class_weights(ActionLoss.compute_class_weights([10, 10, 10]))
        logits = torch.randn(4, 3)
        gt = torch.tensor([-1, -1, -1, -1])
        loss = loss_fn(logits, gt)
        assert not math.isnan(loss.item())

    def test_all_ignore_does_not_create_class_zero_gradients(self):
        loss_fn = self._make_loss("class_weight")
        loss_fn.set_class_weights(ActionLoss.compute_class_weights([10, 10, 10]))
        logits = torch.randn(4, 3, requires_grad=True)
        gt = torch.tensor([-1, -1, -1, -1])

        loss = loss_fn(logits, gt)
        loss.backward()

        assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))

    def test_class_weight_upweights_rare_class(self):
        torch.manual_seed(42)
        logits = torch.zeros(3, 3)
        logits[:, 0] = 10.0
        gt = torch.tensor([2, 2, 2])

        loss_none = ActionLoss(LossConfig(imbalance_strategy="none"), num_actions=3)
        loss_cw = ActionLoss(LossConfig(imbalance_strategy="class_weight"), num_actions=3)
        loss_cw.set_class_weights(torch.tensor([0.1, 0.1, 10.0]))

        l_none = loss_none(logits, gt).item()
        l_cw = loss_cw(logits, gt).item()
        assert l_cw > l_none

    def test_temporarily_disable_class_weights_matches_unweighted_ce(self):
        loss_fn = self._make_loss("class_weight")
        loss_fn.set_class_weights(torch.tensor([0.1, 2.0, 5.0]))
        logits = torch.tensor(
            [[2.0, 0.5, -1.0], [0.2, 1.4, -0.3]],
            dtype=torch.float32,
        )
        gt = torch.tensor([0, 1], dtype=torch.long)

        weighted = loss_fn(logits, gt)
        with loss_fn.temporarily_disable_class_weights():
            unweighted = loss_fn(logits, gt)

        expected = F.cross_entropy(logits, gt, ignore_index=-1) * loss_fn.cfg.w_action
        assert unweighted.item() == pytest.approx(expected.item())
        assert weighted.item() != pytest.approx(unweighted.item())


class TestPosePolarity:
    """head-tail polarity の GT 経路。

    ``predict_polarity`` は pose を有効にすると既定で真になるのに、この経路は
    一度も通ったことがなかった: ``_pose_losses`` が import していない
    ``collect_matched_gt`` を呼んでいて、``polarity`` を持つデータセットは
    学習ではなく ``NameError`` に到達していた。GT 側も ``polarity`` を運ぶ
    キーとして登録されておらず、仮に import しても全件 -1 (= 全部 ignore) が
    返り、``cross_entropy`` が NaN を返すだけだった。
    """

    def test_matched_polarity_reaches_the_detections(self):
        from htrtdetr.training.matching import collect_matched_gt

        det = [{"boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5]])}]
        target = [{
            "boxes": torch.tensor([[0.0, 0.0, 0.5, 0.5]]),
            "polarity": torch.tensor([1], dtype=torch.long),
        }]
        gt = collect_matched_gt(det, target, key="polarity")
        assert gt.tolist() == [1], "polarity never reached the loss (all -1 = all ignored)"

    def test_a_batch_with_no_labelled_polarity_costs_nothing_and_is_not_nan(self):
        from htrtdetr.training.losses import PoseLoss

        loss = PoseLoss(LossConfig())
        pred = torch.zeros(4, 2, requires_grad=True)
        gt = torch.full((4,), -1, dtype=torch.long)
        out = loss(None, None, pred, gt)
        assert not torch.isnan(out["loss_pose"]).any()
        assert out["loss_pose"].item() == pytest.approx(0.0)
        assert "loss_pose_polarity" not in out

    def test_a_labelled_batch_still_produces_a_polarity_loss(self):
        from htrtdetr.training.losses import PoseLoss

        loss = PoseLoss(LossConfig())
        pred = torch.tensor([[2.0, -2.0], [-2.0, 2.0], [0.0, 0.0]])
        gt = torch.tensor([0, 1, -1], dtype=torch.long)
        out = loss(None, None, pred, gt)
        assert "loss_pose_polarity" in out
        assert out["loss_pose_polarity"].item() > 0.0
        assert not torch.isnan(out["loss_pose"]).any()
