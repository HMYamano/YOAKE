"""Regression tests for expired tracks and competing re-identification candidates."""

import pytest
import torch

from htrtdetr.tracking import OBBAssociator, OnlineTracker, TrackFeatureCache


def _tracker():
    return OnlineTracker(
        OBBAssociator(iou_weight=1.0, appearance_weight=0.0),
        TrackFeatureCache(window=4, feat_dim=1, max_age=0, reid_max_age=2),
        reid_threshold=0.8,
    )


def test_expired_tracks_release_embeddings():
    tracker = _tracker()
    tracker.update(
        0, torch.tensor([[0.5, 0.5, 0.1, 0.1, 0.0]]), torch.ones(1, 1),
        det_embed=torch.tensor([[1.0, 0.0]]),
    )
    for frame in range(1, 4):
        tracker.update(frame, torch.empty(0, 5), torch.empty(0, 1))
    assert len(tracker.cache) == 0
    assert tracker._embeddings == {}


@pytest.mark.parametrize("reverse", [False, True])
def test_reid_assigns_competing_detections_to_distinct_valid_tracks(reverse):
    tracker = _tracker()
    boxes = torch.tensor([[0.2, 0.2, 0.1, 0.1, 0.0], [0.8, 0.8, 0.1, 0.1, 0.0]])
    tracker.update(0, boxes, torch.ones(2, 1), det_embed=torch.eye(2))
    tracker.update(1, torch.empty(0, 5), torch.empty(0, 1))
    # Both detections prefer track 0, but only the first can also match track 1.
    embeddings = torch.tensor([[0.8, 0.6], [1.0, 0.0]])
    features = torch.tensor([[10.0], [20.0]])
    if reverse:
        embeddings = embeddings.flip(0)
        features = features.flip(0)
    result = tracker.update(2, boxes, features, det_embed=embeddings)
    assert all(obj.is_reid for obj in result)
    assert [obj.track_id for obj in result] == ([0, 1] if reverse else [1, 0])
    assert tracker.cache.get(0).last_feature().item() == 20.0
    assert tracker.cache.get(1).last_feature().item() == 10.0


def test_history_of_expired_ids_returns_empty_tensors():
    cache = _tracker().cache
    cache.update(0, [7], torch.ones(1, 1), torch.ones(1, 5))
    for frame in range(1, 4):
        cache.update(frame, [], torch.empty(0, 1), torch.empty(0, 5))
    features, boxes, mask, ids = cache.stacked_history([7])
    assert features.shape == (0, 4, 1)
    assert boxes.shape == (0, 4, 5)
    assert mask.shape == (0, 4)
    assert ids == []
