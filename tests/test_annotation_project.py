"""Project persistence and batch contracts; no GUI context or trained weights."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from htrtdetr.annotation import sampling
from htrtdetr.annotation.batch import (
    BatchOptions,
    BatchRunner,
    VideoInput,
    VideoResult,
    connect_detections,
    sample_project,
)
from htrtdetr.annotation.features import FrameFeature
from htrtdetr.annotation.intervals import BehaviorInterval, IntervalTrack
from htrtdetr.annotation.objects import AnnObject
from htrtdetr.annotation.project import ProjectSession, scan_folder
from htrtdetr.annotation.quality import ReviewItem
from htrtdetr.annotation.review_queue import ReviewQueue
from htrtdetr.annotation.session import VideoSession, video_scoped
from htrtdetr.data.annotation import load_annotation
from htrtdetr.gui.auto_label import Candidate, ReviewSession


@pytest.fixture
def project(tmp_path):
    project = ProjectSession(
        name="animals",
        class_names=["fly", "worm"],
        action_names=["walking", "resting"],
        directory=tmp_path,
    )
    for name in ("a.mp4", "b.mp4"):
        path = tmp_path / name
        path.write_bytes(b"video placeholder")
        video = project.add_video(path)
        video.annotations[0] = [
            AnnObject([1, 2, 11, 12], track_id=1, class_id=int(name == "b.mp4"))
        ]
        video.num_frames = 100
    return project


def test_create_save_reopen_and_disjoint_frame_zero(project, tmp_path):
    project.videos["a"].annotations[0][0].keyframe = False
    project.videos["a"].current_frame = 12
    project.videos["b"].current_frame = 53
    project.last_video = "b"
    project.save(tmp_path / "project.yoake.json")
    reopened = ProjectSession.open(tmp_path / "project.yoake.json")
    assert reopened.name == "animals"
    assert list(reopened.videos) == ["a", "b"]
    assert reopened.last_video == "b"
    assert reopened.videos["a"].current_frame == 12
    assert reopened.videos["b"].current_frame == 53
    assert reopened.videos["a"].annotations[0][0].class_id == 0
    assert reopened.videos["b"].annotations[0][0].class_id == 1
    assert not reopened.videos["a"].annotations[0][0].keyframe
    assert reopened.videos["b"].annotations[0][0].keyframe
    assert len(load_annotation(str(tmp_path / "annotations.json")).videos) == 2


def test_relative_paths_survive_moving_project(project, tmp_path):
    project.save(tmp_path / "project.yoake.json")
    raw = json.loads((tmp_path / "project.yoake.json").read_text())
    assert raw["videos"][0]["path"] == "a.mp4"
    moved = tmp_path.parent / (tmp_path.name + "_moved")
    shutil.copytree(tmp_path, moved)
    reopened = ProjectSession.open(moved / "project.yoake.json")
    assert reopened.videos["a"].path == moved / "a.mp4"
    assert not reopened.videos["a"].missing


def test_missing_file_retains_annotations_and_relinks(project, tmp_path):
    project.save(tmp_path / "project.yoake.json")
    (tmp_path / "a.mp4").unlink()
    reopened = ProjectSession.open(tmp_path / "project.yoake.json")
    assert reopened.videos["a"].missing
    assert reopened.videos["a"].annotations[0]
    located = tmp_path / "found.mp4"
    located.write_bytes(b"found")
    reopened.relink("a", located)
    assert not reopened.videos["a"].missing
    with pytest.raises(ValueError):
        reopened.relink("b", located)


def test_duplicate_stems_have_stable_distinct_ids(tmp_path):
    project = ProjectSession()
    first = project.add_video(tmp_path / "one" / "same.mp4")
    second = project.add_video(tmp_path / "two" / "same.mp4")
    assert first.video_id != second.video_id
    assert project.add_video(first.path) is first


def test_folder_import_filters_hidden_and_optional_recursion(tmp_path):
    for name in ("a.MP4", "b.avi", ".hidden.mp4", "notes.txt"):
        (tmp_path / name).touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.webm").touch()
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.mov").touch()
    assert [p.name for p in scan_folder(tmp_path)] == ["a.MP4", "b.avi"]
    assert {p.name for p in scan_folder(tmp_path, recursive=True)} == {"a.MP4", "b.avi", "c.webm"}


@pytest.mark.parametrize("obb_source", ["annotated", "derived"])
def test_class_obb_behavior_keypoints_provenance_roundtrip(project, tmp_path, obb_source):
    obj = AnnObject(
        [10, 10, 30, 20],
        class_id=1,
        track_id=1,
        action_id=1,
        obb=[20, 15, 20, 10, 0.2],
        obb_source=obb_source,
        source="auto",
        score=0.83,
        provenance={"model": "model-v2", "human_review": "accepted"},
        keypoints=[[20, 15, 2]],
    )
    project.videos["b"].annotations[0] = [obj]
    project.videos["b"].intervals = IntervalTrack([BehaviorInterval(1, 1, 0, 5, source="human")])
    project.save(tmp_path / "project.yoake.json")
    loaded = ProjectSession.open(tmp_path / "project.yoake.json").videos["b"]
    actual = loaded.annotations[0][0]
    assert (actual.class_id, actual.action_id, actual.track_id) == (1, 1, 1)
    assert actual.obb == pytest.approx(obj.obb)
    assert actual.obb_source == obb_source
    assert actual.provenance == obj.provenance
    assert actual.keypoints == obj.keypoints
    assert loaded.intervals.intervals[0].source == "human"


def test_candidates_decisions_and_predictions_persist_outside_ground_truth(project, tmp_path):
    video = project.videos["b"]
    video.annotations.clear()
    video.review = ReviewSession(
        {0: [Candidate([1, 1, 10, 10], class_id=1), Candidate([20, 20, 40, 40])]}
    )
    video.review.reject(0, 0)
    video.review.edit(0, 1, class_id=1)
    video.predictions = {0: [Candidate([1, 1, 10, 10], class_id=1)]}
    item = ReviewItem("low_confidence", 0, 0, video_id="b")
    video.queue = ReviewQueue([item])
    video.queue.accept(item)
    project.save(tmp_path / "project.yoake.json")
    loaded = ProjectSession.open(tmp_path / "project.yoake.json").videos["b"]
    assert loaded.review.state(0, 0) == "rejected"
    assert loaded.review.is_edited(0, 1)
    assert loaded.review.candidates(0)[1].class_id == 1
    assert loaded.queue.decision(item).done
    assert loaded.predictions[0][0].class_id == 1
    assert not loaded.annotations
    assert load_annotation(str(tmp_path / "annotations.json")).videos[1].frames == []


def test_legacy_without_video_path_binds_by_identity(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "videos": [
                    {
                        "video_id": "a",
                        "frames": [{"frame_index": 0, "objects": [{"bbox": [1, 1, 10, 10]}]}],
                    },
                    {
                        "video_id": "b",
                        "frames": [
                            {
                                "frame_index": 0,
                                "objects": [{"bbox": [20, 20, 40, 40], "class_id": 1}],
                            }
                        ],
                    },
                ]
            }
        )
    )
    project = ProjectSession.from_annotation(path)
    first = project.add_video(tmp_path / "a.mp4")
    assert first.video_id == "a"
    assert len(project.videos) == 2
    assert first.annotations[0][0].bbox == [1, 1, 10, 10]
    assert project.videos["b"].annotations[0][0].class_id == 1


def test_reject_duplicate_video_ids_instead_of_overwriting(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"videos": [{"video_id": "a"}, {"video_id": "a"}]}))
    with pytest.raises(ValueError, match="Duplicate"):
        ProjectSession.from_annotation(path)


def test_failed_save_preserves_committed_project(project, tmp_path, monkeypatch):
    import htrtdetr.annotation.project as module

    path = tmp_path / "project.yoake.json"
    project.save(path)
    before = path.read_bytes()
    project.videos["a"].annotations[0][0].class_id = 1
    actual = module.atomic_json

    def fail_manifest(target, data):
        if target == path:
            raise OSError("simulated disk failure")
        return actual(target, data)

    monkeypatch.setattr(module, "atomic_json", fail_manifest)
    with pytest.raises(OSError):
        project.save(path)
    assert path.read_bytes() == before
    assert ProjectSession.open(path).videos["a"].annotations[0][0].class_id == 0


def test_video_properties_keep_assignment_history_and_selection_local(project):
    @video_scoped
    class Canvas:
        pass

    app = Canvas()
    app._video_session = project.videos["a"]
    app.current_frame_idx = 12
    app._undo = ["edit A"]
    app._selected = app.annotations[0][0]
    app._video_session = project.videos["b"]
    assert app._undo == [] and app._selected is None
    app._undo.append("edit B")
    app.annotations = {0: [AnnObject([40, 40, 50, 50], class_id=1)]}
    app._video_session = project.videos["a"]
    assert app.current_frame_idx == 12
    assert app._undo == ["edit A"]
    assert app.annotations[0][0].bbox == [1, 2, 11, 12]


def test_scopes_and_explicit_progress(project):
    project.selected_videos = {"b"}
    project.videos["a"].status = "done"
    assert [v.video_id for v in project.scope("Selected videos")] == ["b"]
    assert [v.video_id for v in project.scope("All unfinished videos")] == ["b"]
    assert len(project.scope("Entire project")) == 2
    assert project.scope("All unannotated videos") == []
    assert project.videos["b"].display_status == "in_progress"
    project.videos["a"].review = ReviewSession({1: [Candidate([1, 1, 5, 5])]})
    assert project.videos["a"].display_status == "needs_review"


def test_review_identity_includes_video():
    a = ReviewItem("jump", 0, 1, (1,), video_id="a")
    b = replace(a, video_id="b")
    queue = ReviewQueue([a, b])
    queue.accept(a)
    assert not queue.decision(b).done


@pytest.fixture
def real_videos(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    paths = []
    for name in ("a", "b"):
        path = tmp_path / f"{name}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            pytest.skip("MJPG encoder unavailable")
        for i in range(12):
            image = np.full((48, 64, 3), 30, np.uint8)
            cv2.rectangle(image, (i + 5, 5), (i + 15, 15), (180, 180, 180), -1)
            writer.write(image)
        writer.release()
        paths.append(path)
    return [VideoInput(p.stem, p) for p in paths]


def test_batch_cache_cancel_and_one_video_failure(real_videos, tmp_path):
    calls = []

    def predictor(video, *args):
        calls.append(video.video_id)
        return {0: [Candidate([1, 1, 10, 10], class_id=1)]}

    runner = BatchRunner(tmp_path, tmp_path, predictor=predictor)
    options = BatchOptions(detect=True, sample=False, weights="fake-model")
    videos = [real_videos[0], VideoInput("missing", tmp_path / "missing.mp4"), real_videos[1]]
    first = runner.run(videos, options)
    assert [r.status for r in first] == ["completed", "failed", "completed"]
    second = runner.run(real_videos, options)
    assert [r.status for r in second] == ["cached", "cached"]
    assert calls == ["a", "b"]
    assert second[1].candidates[0][0].class_id == 1
    runner.cancel.set()
    assert runner.run(real_videos, options) == []


def test_cancel_keeps_finished_video_and_does_not_start_next(real_videos, tmp_path):
    runner = BatchRunner(tmp_path, tmp_path)
    events = []

    def emit(event):
        events.append(event)
        if event["kind"] == "video":
            runner.cancel.set()

    results = runner.run(real_videos, BatchOptions(sample=False), emit)
    assert [r.video_id for r in results] == ["a"]
    assert events[-1]["cancelled"]


@pytest.mark.parametrize("balanced", [True, False])
def test_project_sampling_total_budget_and_video_balance(balanced):
    results = [
        VideoResult("short", features=[FrameFeature(i, embedding=(i / 10, 1)) for i in range(10)]),
        VideoResult(
            "long", features=[FrameFeature(i, embedding=(i / 1000, 1)) for i in range(1000)]
        ),
    ]
    request = sampling.SamplingRequest(
        target=sampling.TARGET_DETECTION,
        budget=sampling.Budget(samples=10),
        strategy=sampling.UNIFORM,
    )
    sample_project(results, request, balanced=balanced)
    assert sum(r.selection.total for r in results) == 10
    if balanced:
        assert [r.selection.total for r in results] == [5, 5]


def test_real_batch_sampling_cache_and_current_video(real_videos, tmp_path):
    runner = BatchRunner(tmp_path, tmp_path)
    options = BatchOptions(
        stride=2,
        request=sampling.SamplingRequest(
            target=sampling.TARGET_DETECTION, budget=sampling.Budget(samples=4)
        ),
    )
    first = runner.run(real_videos[:1], options)
    assert first[0].status == "completed", first[0].error
    assert first[0].selection.total == 4
    second = runner.run(real_videos[:1], options)
    assert second[0].status == "cached"
    assert first[0].selection.frames == second[0].selection.frames


def test_track_linking_keeps_classes_and_video_local_ids():
    cands = {
        0: [Candidate([0, 0, 10, 10], class_id=1)],
        1: [Candidate([1, 0, 11, 10], class_id=1)],
        2: [Candidate([2, 0, 12, 10], class_id=0)],
    }
    linked = connect_detections(cands)
    assert linked[0][0].track_id == linked[1][0].track_id
    assert linked[1][0].track_id != linked[2][0].track_id
    assert connect_detections(cands)[0][0].track_id == linked[0][0].track_id
    assert cands[0][0].track_id == -1


def test_import_without_gui_dependencies_is_safe():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
sys.modules['dearpygui'] = None
sys.modules['cv2'] = None
from htrtdetr.annotator import AnnotationApp, AnnObject
assert AnnObject([0,0,10,10], class_id=2).class_id == 2
try:
    AnnotationApp()
except RuntimeError as exc:
    assert 'DearPyGui' in str(exc)
else:
    raise AssertionError('GUI startup must report the missing dependency')
"""
    env = {**os.environ, "PYTHONPATH": str(root / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("installed", [True, False])
def test_torch_capability_is_testable_independently_of_environment(monkeypatch, installed):
    from htrtdetr.gui import auto_label

    original = auto_label._has_module
    monkeypatch.setattr(
        auto_label, "_has_module", lambda name: installed if name == "torch" else original(name)
    )
    assert auto_label.probe(Path(__file__).resolve().parents[1]).torch is installed


def test_quality_does_not_flag_manual_boxes_as_low_confidence():
    from htrtdetr.annotation.batch import check_video
    from htrtdetr.annotation.quality import LOW_CONFIDENCE, QualityConfig

    objects = {0: [AnnObject([1, 1, 10, 10], track_id=1)]}
    assert not any(i.kind == LOW_CONFIDENCE for i in check_video({}, objects, QualityConfig(), 2))


def test_behavior_clips_never_extend_past_a_video():
    results = [
        VideoResult("short", num_frames=12, fps=25, features=[FrameFeature(i) for i in range(12)])
    ]
    sample_project(
        results,
        sampling.SamplingRequest(
            target=sampling.TARGET_BEHAVIOR, budget=sampling.Budget(samples=1), clip_seconds=3
        ),
    )
    assert all(c.end < 12 for c in results[0].selection.clips)


def test_snapshot_retention_is_bounded(project, tmp_path):
    for _ in range(5):
        project.save(tmp_path / "project.yoake.json")
    assert len(list((tmp_path / ".yoake" / "snapshots").glob("*.json"))) == 2


# ══════════════════════════════════════════════════════════════════════════
# The snapshot's cost, which is a freeze the user sees
# ══════════════════════════════════════════════════════════════════════════
#
# ``snapshot`` runs on the **main thread** on purpose: the worker that writes
# the file must never read structures the user is still editing. That makes
# the detach itself a frozen window, and after a detector run it is a long
# one — 28,800 candidates a video, twenty videos in this study, every four
# seconds. So the detach is cheap (``Candidate.to_dict``,
# ``ReviewSession.saved_rows``) and is not repeated for videos nobody touched.
#
# The cache is the risky half: a stale row is lost work. These tests pin the
# three rules that make it safe.


def _rows(state):
    return {row["video_id"]: row for row in state["videos"]}


def test_a_cached_row_is_identical_to_a_fresh_one(project, tmp_path):
    video = project.videos["b"]
    video.review = ReviewSession({0: [Candidate([1, 1, 10, 10], class_id=1)]})
    video.review.reject(0, 0)
    video.predictions = {0: [Candidate([2, 2, 20, 20])]}
    video.visited.add(7)

    fresh = _rows(project.snapshot(tmp_path, always=tuple(project.videos))[1])
    cached = _rows(project.snapshot(tmp_path)[1])
    assert json.dumps(cached, sort_keys=True) == json.dumps(fresh, sort_keys=True)


def test_touching_a_video_invalidates_its_row(project, tmp_path):
    project.snapshot(tmp_path, always=tuple(project.videos))
    project.videos["a"].status = "done"
    project.videos["a"].touch()
    assert _rows(project.snapshot(tmp_path)[1])["a"]["status"] == "done"


def test_an_untouched_video_is_not_re_detached(project, tmp_path):
    """The whole point: nineteen videos nobody edited cost nothing."""
    project.snapshot(tmp_path, always=tuple(project.videos))
    calls = []
    real = type(project)._video_row
    type(project)._video_row = lambda self, v, b: (calls.append(v.video_id), real(self, v, b))[1]
    try:
        project.snapshot(tmp_path, always=("a",))
    finally:
        type(project)._video_row = real
    assert calls == ["a"], f"re-detached {calls}, expected only the open video"


def test_always_forces_a_rebuild_even_without_a_touch(project, tmp_path):
    """``visited`` and the playhead move without marking a video dirty.

    A frame step must not make the save chip say "Unsaved", so ``_visit_frame``
    does not call ``touch()``. The autosave therefore names the open video in
    ``always``, and this is what makes that enough.
    """
    project.snapshot(tmp_path, always=tuple(project.videos))
    project.videos["a"].visited.add(42)
    assert 42 not in _rows(project.snapshot(tmp_path)[1])["a"]["visited"]
    assert 42 in _rows(project.snapshot(tmp_path, always=("a",))[1])["a"]["visited"]


def test_a_deliberate_save_never_uses_the_cache(project, tmp_path):
    """Ctrl+S is the user asking for certainty, so it rebuilds everything."""
    project.snapshot(tmp_path, always=tuple(project.videos))
    project.videos["a"].visited.add(99)  # changed without a touch
    project.save(tmp_path / "project.yoake.json")
    reopened = ProjectSession.open(tmp_path / "project.yoake.json")
    assert 99 in reopened.videos["a"].visited


def test_a_different_destination_is_not_a_cache_hit(project, tmp_path):
    """Paths in a row are relative to the base, so the base is part of the key."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    here = _rows(project.snapshot(tmp_path)[1])["a"]["path"]
    there = _rows(project.snapshot(other)[1])["a"]["path"]
    assert here != there, "the export reused rows written for another folder"


def test_to_dict_matches_asdict_exactly(project):
    """The fast detach must be the same answer, not a similar one."""
    from dataclasses import asdict as _asdict

    candidate = Candidate(
        [1, 2, 3, 4], obb=[1.0, 2.0, 3.0, 4.0, 0.5], score=0.75, track_id=3,
        action_id=2, class_id=1, source="track", provenance={"model": "best.pth"},
    )
    assert candidate.to_dict() == _asdict(candidate)
    assert Candidate([0, 0, 1, 1]).to_dict() == _asdict(Candidate([0, 0, 1, 1]))


def test_saved_rows_matches_the_public_accessors():
    session = ReviewSession({
        5: [Candidate([1, 1, 2, 2]), Candidate([3, 3, 4, 4])],
        1: [Candidate([5, 5, 6, 6])],
    })
    session.reject(5, 0)
    session.edit(1, 0, class_id=1)
    from dataclasses import asdict as _asdict

    expected = [
        {
            "frame": frame,
            "candidates": [
                {"candidate": _asdict(c), "state": session.state(frame, i),
                 "edited": session.is_edited(frame, i)}
                for i, c in enumerate(session.candidates(frame))
            ],
        }
        for frame in session.frames
    ]
    assert session.saved_rows() == expected


def test_a_replaced_session_under_the_same_id_is_not_a_cache_hit(project, tmp_path):
    """A fresh ``VideoSession`` restarts ``revision`` at 0.

    Remove and re-add a video, or let ``_sync_project`` swap the open session
    in under the same id, and the revision counter says "unchanged" about a
    completely different object. The cache therefore checks identity, not only
    the number.
    """
    project.snapshot(tmp_path, always=tuple(project.videos))
    replacement = VideoSession(
        video_id="a", path=project.videos["a"].path, num_frames=100, status="done"
    )
    project.videos["a"] = replacement
    assert _rows(project.snapshot(tmp_path)[1])["a"]["status"] == "done"


def test_a_removed_video_does_not_linger_in_the_cache(project, tmp_path):
    project.snapshot(tmp_path, always=tuple(project.videos))
    assert "b" in project._row_cache
    del project.videos["b"]
    project.snapshot(tmp_path)
    assert "b" not in project._row_cache, "a removed video kept its candidates alive"
