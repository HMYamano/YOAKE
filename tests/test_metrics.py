import pytest
from torch import nn
from torch.utils.data import DataLoader

from htrtdetr.config.config import LossConfig, TrainConfig
from htrtdetr.training.metrics import (
    compute_composite_score,
    init_best_val,
    is_better,
    select_best_metric,
)


class TestSelectBestMetric:
    def test_stage1_uses_val_loss(self):
        # Stage 1 は AP50 が DETR 初期学習で不安定なため val_loss を基準にする
        v, name, higher = select_best_metric(1, {"val_AP50": 0.75, "val_loss": 0.3})
        assert name == "val_loss"
        assert higher is False
        assert v == pytest.approx(0.3)

    def test_stage2_uses_macro_f1(self):
        v, name, higher = select_best_metric(2, {"macro_f1": 0.65, "val_loss": 0.4})
        assert name == "macro_f1"
        assert higher is True
        assert v == pytest.approx(0.65)

    def test_stage3_uses_idf1(self):
        v, name, higher = select_best_metric(3, {"idf1": 0.58, "IDF1": 0.58, "val_loss": 0.5})
        assert name == "idf1"
        assert higher is True
        assert v == pytest.approx(0.58)

    def test_stage4_uses_composite(self):
        v, name, higher = select_best_metric(4, {"val_AP50": 0.8, "macro_f1": 0.7, "idf1": 0.6})
        assert name == "composite"
        assert higher is True
        assert v == pytest.approx(0.71)

    def test_stage4_ap50_primary_uses_val_ap50(self):
        class FakeCfg:
            stage2_primary = "macro_f1"
            stage3_primary = "idf1"
            stage4_primary = "ap50"
            composite_ap50 = 0.4
            composite_macro_f1 = 0.3
            composite_idf1 = 0.3

        v, name, higher = select_best_metric(4, {"val_AP50": 0.82, "AP50": 0.1}, FakeCfg())
        assert name == "AP50"
        assert higher is True
        assert v == pytest.approx(0.82)

    def test_stage2_val_loss_fallback(self):
        class FakeCfg:
            stage2_primary = "val_loss"
            stage3_primary = "idf1"
            stage4_primary = "composite"
            composite_ap50 = 0.4
            composite_macro_f1 = 0.3
            composite_idf1 = 0.3

        v, name, higher = select_best_metric(2, {"macro_f1": 0.5, "val_loss": 0.3}, FakeCfg())
        assert name == "val_loss"
        assert higher is False
        assert v == pytest.approx(0.3)

    def test_non_finite_val_loss_becomes_inf(self):
        class FakeCfg:
            stage2_primary = "val_loss"
            stage3_primary = "idf1"
            stage4_primary = "composite"
            composite_ap50 = 0.4
            composite_macro_f1 = 0.3
            composite_idf1 = 0.3

        v, name, higher = select_best_metric(2, {"val_loss": float("nan")}, FakeCfg())
        assert name == "val_loss"
        assert higher is False
        assert v == float("inf")


class TestCompositeScore:
    def test_basic_calculation(self):
        assert compute_composite_score(0.8, 0.7, 0.6) == pytest.approx(0.71)

    def test_custom_weights(self):
        assert compute_composite_score(1.0, 0.0, 0.0, w_ap50=1.0, w_macro_f1=0.0, w_idf1=0.0) == pytest.approx(1.0)

    def test_non_finite_inputs_are_sanitized(self):
        score = compute_composite_score(float("nan"), float("inf"), 0.5)
        assert score == pytest.approx(0.15)


class TestIsBetter:
    def test_higher_better(self):
        assert is_better(0.9, 0.5, True) is True
        assert is_better(0.3, 0.5, True) is False
        assert is_better(0.5, 0.5, True) is False

    def test_lower_better(self):
        assert is_better(0.2, 0.5, False) is True
        assert is_better(0.8, 0.5, False) is False
        assert is_better(0.5, 0.5, False) is False


class TestInitBestVal:
    def test_initial_values(self):
        assert init_best_val(True) == float("-inf")
        assert init_best_val(False) == float("inf")


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)


class TestTrainerBestSoFar:
    def test_best_so_far_uses_current_epoch_improvement(self, monkeypatch, tmp_path):
        from htrtdetr.training.trainer import Trainer
        import htrtdetr.training.trainer as trainer_mod
        import htrtdetr.training.plots as plots_mod

        captured = {}
        monkeypatch.setattr(trainer_mod, "save_checkpoint", lambda *args, **kwargs: None)
        monkeypatch.setattr(plots_mod, "save_metrics_latest", lambda **kwargs: captured.update(kwargs))

        trainer = Trainer(
            model=_TinyModel(),
            train_loader=DataLoader([0], batch_size=1),
            val_loader=DataLoader([0], batch_size=1),
            train_cfg=TrainConfig(
                stage=2,
                output_dir=str(tmp_path),
                max_epochs=1,
                early_stopping_patience=10,
                use_amp=False,
                use_ema=False,
                val_interval=1,
            ),
            loss_cfg=LossConfig(),
            num_classes=1,
            num_actions=3,
            max_ids=5,
        )

        monkeypatch.setattr(trainer, "_print_train_header", lambda: None)
        monkeypatch.setattr(trainer, "_print_val_results", lambda metrics: None)
        monkeypatch.setattr(trainer, "_compute_and_set_class_weights", lambda: None)
        monkeypatch.setattr(
            trainer,
            "_train_epoch",
            lambda epoch: {
                "train_loss": 0.5,
                "train_loss_detection": 0.0,
                "train_loss_action": 0.0,
                "train_loss_id": 0.0,
                "epoch_time": 0.1,
            },
        )
        monkeypatch.setattr(
            trainer,
            "_val_epoch",
            lambda epoch: {
                "val_loss": 0.4,
                "val_loss_kind": "unweighted",
                "macro_f1": 0.7,
                "idp": 0.8,
                "idr": 0.75,
            },
        )

        trainer.train()

        assert captured["best_so_far"] == pytest.approx(0.7)
        assert captured["val_metrics"]["val_loss_kind"] == "unweighted"
        assert captured["val_metrics"]["idp"] == pytest.approx(0.8)
        assert captured["val_metrics"]["idr"] == pytest.approx(0.75)


class TestSimpleTrackingEvaluator:
    def test_mismatched_ids_are_not_true_positives(self):
        from htrtdetr.evaluation.evaluator import SimpleTrackingEvaluator

        evaluator = SimpleTrackingEvaluator()
        evaluator.update([10], [1])
        result = evaluator.compute()

        assert result["IDTP"] == 0
        assert result["IDFP"] == 1
        assert result["IDFN"] == 1
        assert result["idf1"] == 0.0
        assert result["idp"] == 0.0
        assert result["idr"] == 0.0

    def test_id_switch_tracks_gt_identity_changes(self):
        from htrtdetr.evaluation.evaluator import SimpleTrackingEvaluator

        evaluator = SimpleTrackingEvaluator()
        evaluator.update([1], [1])
        evaluator.update([2], [1])
        result = evaluator.compute()

        assert result["IDSW"] == 1

    def test_id_precision_and_recall_are_exposed(self):
        from htrtdetr.evaluation.evaluator import SimpleTrackingEvaluator

        evaluator = SimpleTrackingEvaluator()
        evaluator.update([1, 2], [1, 3])
        result = evaluator.compute()

        assert result["IDTP"] == 1
        assert result["IDFP"] == 1
        assert result["IDFN"] == 1
        assert result["idp"] == pytest.approx(0.5)
        assert result["idr"] == pytest.approx(0.5)

    def test_sequence_key_prevents_cross_video_id_switches(self):
        from htrtdetr.evaluation.evaluator import SimpleTrackingEvaluator

        evaluator = SimpleTrackingEvaluator()
        evaluator.update([1], [7], sequence_key="video_a")
        evaluator.update([2], [7], sequence_key="video_b")
        result = evaluator.compute()

        assert result["IDSW"] == 0
