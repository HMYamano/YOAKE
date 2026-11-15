"""
Tests for :mod:`htrtdetr.annotation.keyframes` — the keyframe labelling scheme.

Pure logic, no torch / dearpygui / cv2 / numpy, so this runs on either of the
project's interpreters in well under a second (the purity itself is asserted
by ``tests/test_annotation_core.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htrtdetr.annotation import keyframes as kf  # noqa: E402


def _obj(action=-1, track=-1):
    return {"object_id": 1, "bbox": [0, 0, 10, 10], "class_id": 0,
            "track_id": track, "action_id": action}


def _dataset(frames_by_video, *, labeling=None, num_frames=1000):
    meta = {"version": "2.0"}
    if labeling is not None:
        meta["labeling"] = labeling
    return {
        "meta": meta,
        "class_names": ["fly"],
        "action_names": ["stationary", "walking", "grooming"],
        "videos": [
            {"video_id": vid, "num_frames": num_frames, "fps": 30.0,
             "frames": [{"frame_index": f, "image_path": "", "objects": objs}
                        for f, objs in frames]}
            for vid, frames in frames_by_video.items()
        ],
    }


# ══════════════════════════════════════════════════════════════════════════
# ClipSpec
# ══════════════════════════════════════════════════════════════════════════

class TestClipSpec:
    def test_end_anchor_puts_keyframe_last_and_counts_back_by_stride(self):
        clip = kf.ClipSpec(length=4, stride=3)
        frames, valid = clip.indices(100)
        assert frames == (91, 94, 97, 100)
        assert valid == (True, True, True, True)
        assert clip.key_position == 3
        assert clip.span == 10

    def test_before_the_start_is_clamped_and_marked_invalid(self):
        """Clamping keeps the clip length the model needs; the mask keeps the
        repeated frame from being read as an animal standing still."""
        frames, valid = kf.ClipSpec(length=4, stride=3).indices(4)
        assert frames == (0, 0, 1, 4)
        assert valid == (False, False, True, True)

    def test_center_anchor_clamps_at_the_end_of_the_video(self):
        clip = kf.ClipSpec(length=5, stride=2, anchor="center")
        frames, valid = clip.indices(98, num_frames=100)
        assert frames == (94, 96, 98, 99, 99)
        assert valid == (True, True, True, False, False)

    def test_clip_is_defined_by_video_position_not_by_labelled_frames(self):
        """The regression this class exists for: two keyframes ten seconds
        apart are never each other's context."""
        clip = kf.ClipSpec(length=8, stride=4)
        frames, _ = clip.indices(600)
        assert min(frames) == 600 - 28
        assert 300 not in frames

    @pytest.mark.parametrize("kwargs", [
        {"length": 0}, {"stride": 0}, {"anchor": "middle"},
    ])
    def test_rejects_nonsense(self, kwargs):
        with pytest.raises(ValueError):
            kf.ClipSpec(**kwargs)

    def test_round_trips_through_its_dict_and_tolerates_garbage(self):
        clip = kf.ClipSpec(length=16, stride=2, anchor="end")
        assert kf.ClipSpec.from_dict(clip.to_dict()) == clip
        assert kf.ClipSpec.from_dict("not a dict") == kf.ClipSpec()
        assert kf.ClipSpec.from_dict({"length": "x"}) == kf.ClipSpec()

    def test_seconds_and_description(self):
        clip = kf.ClipSpec(length=8, stride=4)
        assert clip.seconds(30.0) == pytest.approx(28 / 30)
        assert kf.describe_clip(clip, 30.0) == "8 frames, every 4th - 0.93 s at 30 fps"
        assert kf.describe_clip(kf.ClipSpec(length=3, stride=1)) == "3 frames, every frame"
        assert "every 2nd" in kf.describe_clip(kf.ClipSpec(length=3, stride=2))


# ══════════════════════════════════════════════════════════════════════════
# The dataset's declaration
# ══════════════════════════════════════════════════════════════════════════

class TestDeclaration:
    def test_absent_labeling_reads_dense(self):
        """Every file written before the scheme existed keeps its old meaning."""
        assert kf.label_mode_of(_dataset({})) == kf.DENSE
        assert kf.label_mode_of({}) == kf.DENSE

    def test_declared_keyframe_with_its_clip(self):
        data = _dataset({}, labeling=kf.labeling_meta(kf.ClipSpec(length=6, stride=2)))
        assert kf.label_mode_of(data) == kf.KEYFRAME
        assert kf.clip_spec_of(data) == kf.ClipSpec(length=6, stride=2)

    def test_works_on_a_dataset_object_too(self):
        meta = kf.set_labeling({}, kf.ClipSpec(length=4, stride=1))
        data = SimpleNamespace(meta=meta, videos=[])
        assert kf.label_mode_of(data) == kf.KEYFRAME
        assert kf.clip_spec_of(data).length == 4

    def test_resolve_label_mode(self):
        keyframe = _dataset({}, labeling=kf.labeling_meta())
        assert kf.resolve_label_mode("auto", keyframe) == kf.KEYFRAME
        assert kf.resolve_label_mode("auto", _dataset({})) == kf.DENSE
        assert kf.resolve_label_mode("auto") == kf.DENSE
        assert kf.resolve_label_mode("Keyframe", _dataset({})) == kf.KEYFRAME
        with pytest.raises(ValueError):
            kf.resolve_label_mode("keyframes")


# ══════════════════════════════════════════════════════════════════════════
# Export, counting, checks
# ══════════════════════════════════════════════════════════════════════════

def test_context_frames_is_the_union_of_every_clip():
    clip = kf.ClipSpec(length=3, stride=2)
    assert kf.context_frames([10, 12], clip) == [6, 8, 10, 12]
    assert kf.context_frames([1], clip) == [0, 1]


def test_summary_counts_behaviors_and_leftover_track_ids():
    data = _dataset({
        "a": [(10, [_obj(0), _obj(1), _obj(-1)]), (40, [])],
        "b": [(5, [_obj(1, track=3)])],
        "empty": [],
    })
    s = kf.summarize(data, data["action_names"])
    assert (s.videos, s.keyframes, s.empty_keyframes) == (2, 3, 1)
    assert (s.instances, s.with_behavior, s.without_behavior) == (4, 3, 1)
    assert s.per_action == (1, 2, 0)
    assert s.with_track_ids == 1
    sentence = s.sentence(data["action_names"])
    assert sentence.startswith("3 keyframes, 4 animals, 3 with a behavior")
    assert "walking 2" in sentence and "grooming" not in sentence
    assert kf.summarize({}, ()).sentence() == "No keyframes yet."


def test_check_reports_each_problem_once_in_words():
    data = _dataset({
        "a": [(2, [_obj(0, track=1)]), (2, [_obj(7)]), (5000, [_obj(1)])],
    }, labeling=kf.labeling_meta(kf.ClipSpec(length=4, stride=2)), num_frames=1000)
    warnings = kf.check(data, data["action_names"])
    text = " | ".join(warnings)
    assert "appear twice" in text
    assert "past the end" in text
    assert "no name" in text
    assert "repeat the first frame" in text
    assert "carry a track id" in text
    assert "No examples yet of: grooming" in text


def test_check_is_quiet_on_a_clean_dataset():
    data = _dataset({"a": [(100, [_obj(0)]), (200, [_obj(1)]), (300, [_obj(2)])]},
                    labeling=kf.labeling_meta())
    assert kf.check(data, data["action_names"]) == []


def test_check_says_when_only_the_detector_can_learn():
    data = _dataset({"a": [(100, [_obj(-1)])]}, labeling=kf.labeling_meta())
    assert any("only the detector" in w for w in kf.check(data, data["action_names"]))


# ══════════════════════════════════════════════════════════════════════════
# Conversion from the old scheme
# ══════════════════════════════════════════════════════════════════════════

def test_to_keyframes_thins_dense_labels_and_drops_identity():
    dense = _dataset({"a": [(f, [_obj(1, track=7)]) for f in range(0, 100)]})
    out = kf.to_keyframes(dense, every=30, clip=kf.ClipSpec(length=4, stride=2))
    frames = [f["frame_index"] for f in out["videos"][0]["frames"]]
    assert frames == [0, 30, 60, 90]
    assert all(o["track_id"] == -1 for f in out["videos"][0]["frames"] for o in f["objects"])
    assert all(o["action_id"] == 1 for f in out["videos"][0]["frames"] for o in f["objects"])
    assert kf.label_mode_of(out) == kf.KEYFRAME
    assert kf.clip_spec_of(out) == kf.ClipSpec(length=4, stride=2)
    # the input is untouched
    assert len(dense["videos"][0]["frames"]) == 100
    assert dense["videos"][0]["frames"][0]["objects"][0]["track_id"] == 7
    assert "labeling" not in dense["meta"]


def test_to_keyframes_keeps_sparse_labels_and_can_keep_ids():
    sparse = _dataset({"a": [(5, [_obj(0, track=2)]), (400, [_obj(1, track=2)])]})
    out = kf.to_keyframes(sparse, every=15, keep_track_ids=True)
    frames = out["videos"][0]["frames"]
    assert [f["frame_index"] for f in frames] == [5, 400]
    assert frames[0]["objects"][0]["track_id"] == 2


# ══════════════════════════════════════════════════════════════════════════
# Annotator helpers
# ══════════════════════════════════════════════════════════════════════════

def test_keyframe_marks_and_frame_status():
    annotations = {
        30: [SimpleNamespace(action_id=2), SimpleNamespace(action_id=-1)],
        10: [SimpleNamespace(action_id=0)],
    }
    assert kf.keyframe_marks(annotations) == [(10, 0), (30, 2)]
    assert kf.frame_status(annotations[30]) == (2, 1)
    assert kf.keyframe_sentence(annotations[30]) == "2 animals - 1 still needs a behavior."
    assert kf.keyframe_sentence(annotations[10]) == "1 animal - every one has a behavior."
    assert kf.keyframe_sentence([]) == "No animals boxed on this frame."


def test_context_window_leans_on_the_past_and_stays_in_the_video():
    assert kf.context_window(100, 30.0) == (70, 115)
    assert kf.context_window(3, 30.0, num_frames=10) == (0, 9)
    # the floor holds when the fps is tiny or missing
    lo, hi = kf.context_window(100, 1.0)
    assert (100 - lo, hi - 100) == (6, 6)


class TestContextLoop:
    def test_plays_the_range_in_a_loop_and_hands_the_keyframe_back(self):
        loop = kf.ContextLoop()
        assert not loop.active
        loop.start(anchor=50, lo=48, hi=51, now=0.0)
        assert loop.active and loop.position == 48
        seen = [loop.advance() for _ in range(6)]
        assert seen == [49, 50, 51, 48, 49, 50]
        assert loop.offset_seconds(10.0) == pytest.approx(0.0)
        assert loop.stop() == 50
        assert not loop.active

    def test_is_rate_limited_to_the_video_and_capped(self):
        loop = kf.ContextLoop()
        loop.start(anchor=10, lo=0, hi=20, now=0.0)
        assert loop.tick(0.01, fps=25.0) is None
        assert loop.tick(0.05, fps=25.0) == 1
        assert loop.tick(0.06, fps=240.0) is None      # capped at 30 fps
        assert loop.tick(0.05 + 1 / 30 + 1e-6, fps=240.0) == 2

    def test_progress_and_validation(self):
        loop = kf.ContextLoop()
        loop.start(anchor=5, lo=0, hi=10)
        loop.position = 5
        assert loop.progress() == pytest.approx(0.5)
        with pytest.raises(ValueError):
            loop.start(anchor=5, lo=10, hi=0)
