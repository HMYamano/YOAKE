"""
Tests for :mod:`htrtdetr.data.keyframe_clip` — keyframe labels in, clips out.

A tiny synthetic video is written per test session. Every frame carries its
own index as a grey level in a corner patch, so a test can read back *which*
frame a clip actually contains rather than only its shape: the failure this
dataset exists to prevent is the right shape with the wrong frames in it.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from htrtdetr.annotation.keyframes import ClipSpec, labeling_meta  # noqa: E402
from htrtdetr.data.keyframe_clip import (  # noqa: E402
    KeyframeClipDataset,
    collate_keyframe_clips,
    collate_keyframe_single,
    context_image_name,
    resize_obb,
    transform_obb_for_flip,
)
from htrtdetr.training.targets import last_frame_targets  # noqa: E402

W, H, N_FRAMES = 96, 64, 80
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _level(index: int) -> int:
    """Grey level that encodes a frame index (distinct for 0..N_FRAMES)."""
    return 10 + 3 * index


def _frame(index: int) -> np.ndarray:
    img = np.full((H, W, 3), 90, dtype=np.uint8)
    img[:12, :12] = _level(index)                    # the index patch
    x = 20 + index % 50
    cv2.rectangle(img, (x, 30), (x + 14, 36), (230, 230, 230), -1)
    return img


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("kf_video") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(_frame(i))
    writer.release()
    return path


def _annotation(root: Path, video: Path, keyframes, *, clip=ClipSpec(length=4, stride=3)):
    frames = []
    for f in keyframes:
        frames.append({
            "frame_index": f,
            "image_path": context_image_name("vid", f),
            "objects": [
                {"object_id": 1, "bbox": [20.0, 28.0, 40.0, 38.0], "class_id": 0,
                 "track_id": 5, "action_id": 1,
                 "obb": {"cx": 30.0, "cy": 33.0, "w": 20.0, "h": 6.0, "angle": 0.4}},
                {"object_id": 2, "bbox": [60.0, 10.0, 70.0, 20.0], "class_id": 0,
                 "track_id": -1, "action_id": -1},
            ],
        })
    data = {
        "meta": {"version": "2.0", **({"labeling": labeling_meta(clip)} if clip else {})},
        "class_names": ["fly"],
        "action_names": ["stationary", "walking"],
        "videos": [{
            "video_id": "vid", "video_path": str(video), "fps": 30.0,
            "width": W, "height": H, "num_frames": N_FRAMES, "frames": frames,
        }],
    }
    path = root / "annotations.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _patch_level(tensor: "torch.Tensor") -> float:
    """Undo the ImageNet normalisation and read the index patch's grey level."""
    img = tensor.permute(1, 2, 0).numpy() * _STD + _MEAN
    return float(img[1:5, 1:5].mean() * 255.0)


# ══════════════════════════════════════════════════════════════════════════
# Geometry
# ══════════════════════════════════════════════════════════════════════════

def _corner_set(obb):
    from htrtdetr.data.keyframe_clip import _corners
    return sorted(map(tuple, np.round(_corners(np.asarray(obb, dtype=np.float32))[0], 3)))


def test_flip_mirrors_the_rotated_box_with_the_image():
    obb = np.array([[30.0, 20.0, 16.0, 6.0, 0.5]], dtype=np.float32)
    flipped = transform_obb_for_flip(obb, W, H, hflip=True, vflip=False)
    from htrtdetr.data.keyframe_clip import _corners
    mirrored = _corners(obb)[0].copy()
    mirrored[:, 0] = W - mirrored[:, 0]
    assert _corner_set(flipped) == sorted(map(tuple, np.round(mirrored, 3)))
    both = transform_obb_for_flip(obb, W, H, hflip=True, vflip=True)
    rotated = _corners(obb)[0].copy()
    rotated[:, 0] = W - rotated[:, 0]
    rotated[:, 1] = H - rotated[:, 1]
    assert _corner_set(both) == sorted(map(tuple, np.round(rotated, 3)))


def test_resize_is_exact_for_a_uniform_scale_and_keeps_area_otherwise():
    obb = np.array([[30.0, 20.0, 16.0, 6.0, 0.5]], dtype=np.float32)
    uniform = resize_obb(obb, 2.0, 2.0)
    assert np.allclose(uniform[0], [60.0, 40.0, 32.0, 12.0, 0.5], atol=1e-4)
    stretched = resize_obb(obb, 1.0, 1.5)
    assert stretched[0, 2] * stretched[0, 3] == pytest.approx(16 * 6 * 1.5, rel=1e-4)
    assert stretched[0, 0] == pytest.approx(30.0) and stretched[0, 1] == pytest.approx(30.0)
    axis = resize_obb(np.array([[10.0, 10.0, 8.0, 4.0, 0.0]], dtype=np.float32), 2.0, 3.0)
    assert np.allclose(axis[0], [20.0, 30.0, 16.0, 12.0, 0.0], atol=1e-4)


# ══════════════════════════════════════════════════════════════════════════
# The dataset
# ══════════════════════════════════════════════════════════════════════════

def test_clip_from_video_has_the_right_frames_and_only_the_keyframe_has_targets(tmp_path, video):
    ann = _annotation(tmp_path, video, [30, 60])
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False, frame_source="video")
    assert len(ds) == 2
    sample = ds[0]
    assert sample["images"].shape == (4, 3, 64, 96)
    assert sample["frame_indices"] == [21, 24, 27, 30]
    assert sample["clip_valid"] == [True] * 4
    levels = [_patch_level(sample["images"][t]) for t in range(4)]
    for got, index in zip(levels, sample["frame_indices"]):
        assert abs(got - _level(index)) < 6, (levels, sample["frame_indices"])
    assert [len(b) for b in sample["boxes"]] == [0, 0, 0, 2]
    key = {k: sample[k][-1] for k in ("boxes", "obb", "obb_valid", "action_ids", "track_ids")}
    assert key["action_ids"].tolist() == [1, -1]
    assert key["track_ids"].tolist() == [-1, -1], "track ids are not labels here"
    assert key["obb_valid"].tolist() == [True, False]
    assert key["obb"][0, 4].item() == pytest.approx(0.4, abs=1e-4)
    assert key["boxes"][0].tolist() == pytest.approx([20 / 96, 28 / 64, 40 / 96, 38 / 64], abs=1e-5)
    assert ds.stats["video_reads"] > 0


def test_clip_near_the_start_repeats_frame_zero_and_says_so(tmp_path, video):
    ann = _annotation(tmp_path, video, [5])
    sample = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False)[0]
    assert sample["frame_indices"] == [0, 0, 2, 5]
    assert sample["clip_valid"] == [False, False, True, True]


def test_exported_images_are_preferred_and_the_video_is_not_touched(tmp_path, video):
    ann = _annotation(tmp_path, video, [30])
    clip = ClipSpec(length=4, stride=3)
    frames, _ = clip.indices(30, N_FRAMES)
    for f in frames:
        out = tmp_path / context_image_name("vid", f)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), _frame(f))
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False, frame_source="images")
    sample = ds[0]
    assert ds.stats["video_reads"] == 0
    assert abs(_patch_level(sample["images"][0]) - _level(21)) < 6


def test_a_missing_context_image_falls_back_to_the_video(tmp_path, video):
    ann = _annotation(tmp_path, video, [30])
    key = tmp_path / context_image_name("vid", 30)
    key.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(key), _frame(30))
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False, frame_source="auto")
    sample = ds[0]
    assert ds.stats["video_reads"] == 3
    assert abs(_patch_level(sample["images"][0]) - _level(21)) < 6


def test_unreadable_keyframes_are_dropped_not_blacked_out(tmp_path, video):
    ann = _annotation(tmp_path, tmp_path / "gone.avi", [30, 60])
    with pytest.raises(FileNotFoundError):
        KeyframeClipDataset(str(ann), frame_source="auto")
    key = tmp_path / context_image_name("vid", 30)
    key.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(key), _frame(30))
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False)
    assert len(ds) == 1 and ds.dropped == [("vid", 60)]
    sample = ds[0]
    # no video and only the keyframe image: the context repeats it, counted
    assert ds.stats["substituted"] == 3
    assert all(abs(_patch_level(sample["images"][t]) - _level(30)) < 6 for t in range(4))


def test_augmentation_is_the_same_for_every_frame_of_a_clip(tmp_path, video, monkeypatch):
    ann = _annotation(tmp_path, video, [30])
    plain = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False)[0]
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=True,
                             brightness=0.0, contrast=0.0, saturation=0.0)
    monkeypatch.setattr(random, "random", lambda: 0.0)       # force the flip
    flipped = ds[0]
    for t in range(4):
        assert torch.allclose(flipped["images"][t], plain["images"][t].flip(-1), atol=1e-5)
    box = flipped["boxes"][-1][0].tolist()
    assert box == pytest.approx([1 - 40 / 96, 28 / 64, 1 - 20 / 96, 38 / 64], abs=1e-5)
    assert flipped["obb"][-1][0, 4].item() == pytest.approx(-0.4, abs=1e-4)


def test_single_frame_mode_feeds_stage_one_with_rotated_targets(tmp_path, video):
    ann = _annotation(tmp_path, video, [30, 60])
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False, single_frame=True)
    batch = collate_keyframe_single([ds[0], ds[1]])
    assert batch["images"].shape == (2, 3, 64, 96)
    assert len(batch["targets"]) == 2
    assert batch["targets"][0]["obb"].shape == (2, 5)
    assert batch["targets"][0]["obb_valid"].tolist() == [True, False]


def test_clip_collate_has_the_sequence_shape_the_trainer_expects(tmp_path, video):
    ann = _annotation(tmp_path, video, [30, 60])
    ds = KeyframeClipDataset(str(ann), image_size=(64, 96), augment=False)
    batch = collate_keyframe_clips([ds[0], ds[1]])
    assert batch["images"].shape == (2, 4, 3, 64, 96)
    keys = last_frame_targets(batch["targets"])
    assert [len(t["boxes"]) for t in keys] == [2, 2]
    assert "obb" in keys[0] and "obb_valid" in keys[0]
    assert batch["meta"][1]["keyframe"] == 60


def test_require_behavior_skips_keyframes_with_nothing_to_teach(tmp_path, video):
    ann = _annotation(tmp_path, video, [30, 60])
    data = json.loads(ann.read_text(encoding="utf-8"))
    for obj in data["videos"][0]["frames"][1]["objects"]:
        obj["action_id"] = -1
    ann.write_text(json.dumps(data), encoding="utf-8")
    ds = KeyframeClipDataset(str(ann), augment=False, require_behavior=True)
    assert [f.frame_index for _v, f in ds.samples] == [30]


def test_centered_clips_are_refused_for_training(tmp_path, video):
    ann = _annotation(tmp_path, video, [30], clip=ClipSpec(length=3, stride=2, anchor="center"))
    with pytest.raises(ValueError, match="anchor"):
        KeyframeClipDataset(str(ann))
