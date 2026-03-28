import torch

from htrtdetr.training.matching import (
    assign_gt_to_detections,
    batch_assign_gt_to_detections,
    collect_matched_gt,
)


def _target(boxes_xyxy, action_ids=None, track_ids=None):
    tgt = {"boxes": torch.tensor(boxes_xyxy, dtype=torch.float32)}
    if action_ids is not None:
        tgt["action_ids"] = torch.tensor(action_ids, dtype=torch.long)
    if track_ids is not None:
        tgt["track_ids"] = torch.tensor(track_ids, dtype=torch.long)
    return tgt


class TestAssignGtToDetections:
    def test_perfect_overlap_assigns_correct_action_id(self):
        pred_boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
        target = _target([[0.3, 0.3, 0.7, 0.7]], action_ids=[2], track_ids=[5])
        result = assign_gt_to_detections(pred_boxes, target, iou_threshold=0.5)

        assert result["matched_gt_idx"][0].item() == 0
        assert result["matched_action_ids"][0].item() == 2
        assert result["matched_track_ids"][0].item() == 5

    def test_no_overlap_returns_minus_one(self):
        pred_boxes = torch.tensor([[0.1, 0.1, 0.1, 0.1]], dtype=torch.float32)
        target = _target([[0.7, 0.7, 0.9, 0.9]], action_ids=[1], track_ids=[3])
        result = assign_gt_to_detections(pred_boxes, target, iou_threshold=0.5)

        assert result["matched_gt_idx"][0].item() == -1
        assert result["matched_action_ids"][0].item() == -1
        assert result["matched_track_ids"][0].item() == -1

    def test_empty_gt_all_unmatched(self):
        pred_boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32)
        target = _target([], action_ids=[], track_ids=[])
        result = assign_gt_to_detections(pred_boxes, target, iou_threshold=0.5)
        assert result["matched_gt_idx"][0].item() == -1

    def test_empty_pred_returns_empty(self):
        pred_boxes = torch.zeros(0, 4)
        target = _target([[0.3, 0.3, 0.7, 0.7]], action_ids=[1])
        result = assign_gt_to_detections(pred_boxes, target)
        assert result["matched_gt_idx"].shape[0] == 0

    def test_duplicate_predictions_do_not_match_same_gt(self):
        pred_boxes = torch.tensor(
            [
                [0.5, 0.5, 0.4, 0.4],
                [0.5, 0.5, 0.4, 0.4],
            ],
            dtype=torch.float32,
        )
        target = _target([[0.3, 0.3, 0.7, 0.7]], action_ids=[7], track_ids=[11])
        result = assign_gt_to_detections(pred_boxes, target, iou_threshold=0.5)

        matched = result["matched_gt_idx"].tolist()
        assert matched.count(0) == 1
        assert matched.count(-1) == 1

    def test_target_without_action_ids(self):
        pred_boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
        target = {"boxes": torch.tensor([[0.3, 0.3, 0.7, 0.7]])}
        result = assign_gt_to_detections(pred_boxes, target)
        assert "matched_action_ids" not in result
        assert result["matched_gt_idx"][0].item() == 0


class TestCollectMatchedGt:
    def _make_det(self, boxes_cxcywh):
        return {"boxes": torch.tensor(boxes_cxcywh, dtype=torch.float32)}

    def test_collect_action_ids_basic(self):
        det_results = [self._make_det([[0.5, 0.5, 0.4, 0.4]])]
        targets = [_target([[0.3, 0.3, 0.7, 0.7]], action_ids=[3])]
        gt = collect_matched_gt(det_results, targets, key="action_ids")
        assert gt.shape[0] == 1
        assert gt[0].item() == 3

    def test_collect_track_ids_unmatched(self):
        det_results = [self._make_det([[0.1, 0.1, 0.05, 0.05]])]
        targets = [_target([[0.8, 0.8, 0.9, 0.9]], track_ids=[10])]
        gt = collect_matched_gt(det_results, targets, key="track_ids")
        assert gt[0].item() == -1

    def test_collect_empty_batch(self):
        gt = collect_matched_gt([], [], key="action_ids")
        assert gt.shape[0] == 0


class TestBatchAssign:
    def test_batch_size_2(self):
        det_results = [
            {"boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]])},
            {"boxes": torch.tensor([[0.1, 0.1, 0.05, 0.05]])},
        ]
        targets = [
            _target([[0.3, 0.3, 0.7, 0.7]], action_ids=[1]),
            _target([[0.8, 0.8, 0.9, 0.9]], action_ids=[2]),
        ]
        results = batch_assign_gt_to_detections(det_results, targets)
        assert len(results) == 2
        assert results[0]["matched_action_ids"][0].item() == 1
        assert results[1]["matched_action_ids"][0].item() == -1
