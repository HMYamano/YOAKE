import pytest
import torch

from htrtdetr.evaluation.evaluator import DetectionEvaluator
from htrtdetr.models.ht_rtdetr import HTRTDETROutput


class TestDetectionEvaluator:
    def test_class_mismatch_does_not_count_as_true_positive(self):
        evaluator = DetectionEvaluator(iou_thresholds=[0.5])
        pred_boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
        pred_scores = torch.tensor([0.9], dtype=torch.float32)
        pred_classes = torch.tensor([1], dtype=torch.long)
        gt_boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
        gt_classes = torch.tensor([0], dtype=torch.long)

        evaluator.update(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_classes=pred_classes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            box_format="xyxy",
        )
        metrics = evaluator.compute()

        assert metrics["AP50"] == pytest.approx(0.0)
        assert metrics["recall@50"] == pytest.approx(0.0)

    def test_matching_same_class_keeps_true_positive(self):
        evaluator = DetectionEvaluator(iou_thresholds=[0.5])
        pred_boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
        pred_scores = torch.tensor([0.9], dtype=torch.float32)
        pred_classes = torch.tensor([0], dtype=torch.long)
        gt_boxes = torch.tensor([[0.1, 0.1, 0.4, 0.4]], dtype=torch.float32)
        gt_classes = torch.tensor([0], dtype=torch.long)

        evaluator.update(
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_classes=pred_classes,
            gt_boxes=gt_boxes,
            gt_classes=gt_classes,
            box_format="xyxy",
        )
        metrics = evaluator.compute()

        assert metrics["AP50"] == pytest.approx(1.0, abs=1e-5)
        assert metrics["recall@50"] == pytest.approx(1.0, abs=1e-5)


class TestHTRTDETROutput:
    def test_split_flattened_tensor_uses_valid_detection_counts(self):
        output = HTRTDETROutput(
            pred_logits=torch.zeros(2, 4, 2),
            pred_boxes=torch.zeros(2, 4, 4),
            query_features=torch.zeros(2, 4, 8),
            det_results=[
                {"boxes": torch.zeros(2, 4)},
                {"boxes": torch.zeros(1, 4)},
            ],
        )
        chunks = output.split_flattened_tensor(torch.arange(9).view(3, 3))

        assert [chunk.shape[0] for chunk in chunks] == [2, 1]
        assert torch.equal(chunks[0], torch.tensor([[0, 1, 2], [3, 4, 5]]))
        assert torch.equal(chunks[1], torch.tensor([[6, 7, 8]]))
