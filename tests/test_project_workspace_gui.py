"""Real viewport smoke tests for the multi-video workspace."""

from __future__ import annotations

import json
import time

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from htrtdetr.annotation.batch import BatchOptions, VideoResult
from htrtdetr.annotation.intervals import BehaviorInterval, IntervalTrack
from htrtdetr.annotation.project import ProjectSession
from htrtdetr.annotator import AnnotationApp
from htrtdetr.gui.auto_label import Candidate, ReviewSession

pytestmark = pytest.mark.gui


def wait_for_ui(app, predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app._pump_ui()
        time.sleep(0.01)
    app._pump_ui()
    assert predicate()


def test_export_rewrites_imported_image_paths_and_keeps_all_videos(app, tmp_path):
    app._create_object([5, 5, 25, 25])
    app._video_session.frame_metadata[0] = {"image_path": "old/location.png"}
    app.project.metadata["image_root"] = "old-root"
    app.switch_video("b")
    app._create_object([10, 10, 30, 30])
    out = tmp_path / "export"
    app.export_project(out)
    wait_for_ui(app, lambda: not app._export_thread.is_alive())
    data = json.loads((out / "annotations.json").read_text(encoding="utf-8"))
    assert data["meta"]["image_root"] == ""
    assert len(data["videos"]) == 2
    for video in data["videos"]:
        frame = video["frames"][0]
        assert frame["image_path"] == f"images/{video['video_id']}/000000.jpg"
        assert cv2.imread(str(out / frame["image_path"])) is not None
    reopened = ProjectSession.open(out / "project.yoake.json")
    assert reopened.videos["a"].frame_metadata[0]["image_path"] == "images/a/000000.jpg"


def test_export_into_the_project_never_moves_the_project(app, tmp_path):
    """Export writes a self-contained copy; the study stays where it lives.

    The default destination is inside the project (``datasets/frames``), and the
    export writes a manifest so the copy can be opened on its own. Those two
    facts together used to mean the live session quietly adopted the export as
    its home and left the real folder behind.
    """
    home = tmp_path / "study"
    app.project_dir = home
    app._create_object([5, 5, 25, 25])
    app.save_annotation()
    assert (home / "project.yoake.json").is_file()

    out = home / "datasets" / "frames"
    app.export_project(out)
    wait_for_ui(app, lambda: not app._export_thread.is_alive())

    assert app.project_dir == home
    assert app.project.directory == home
    assert (home / "annotations.json").is_file()
    # The copy stands on its own, and opening it does not disturb the original.
    assert ProjectSession.open(out / "project.yoake.json").videos
    assert ProjectSession.open(home / "project.yoake.json").videos


def test_collect_videos_copies_the_captures_into_the_folder(app, tmp_path):
    home = tmp_path / "study"
    app.project_dir = home
    app._create_object([5, 5, 25, 25])
    app.save_annotation()
    outside = [v.path for v in app.project.videos.values()]
    assert all(not str(p).startswith(str(home)) for p in outside)

    app.collect_videos()
    wait_for_ui(app, lambda: not app._collect_thread.is_alive())
    wait_for_ui(app, lambda: all(str(v.path).startswith(str(home))
                                 for v in app.project.videos.values()))
    assert sorted(p.name for p in (home / "videos").glob("*.avi")) == ["a.avi", "b.avi"]
    for original in outside:
        assert original.is_file()  # a copy, never a move: the source is untouched


def test_async_native_tracker_keeps_class_and_origin_video(app):
    seed = app._create_object([5, 5, 25, 25])
    seed.class_id, seed.action_id = 1, 1
    video = app._video_session

    class Tracker:
        def update(self, image):
            return True, (6, 6, 20, 20)

    app._trackers[seed.track_id] = Tracker()
    video.transient["tracker_seeds"] = {seed.track_id: seed}
    video.transient["tracker_frames"] = {seed.track_id: 0}
    app._queue_tracker_step(1, app._current_bgr)
    app.switch_video("b")
    wait_for_ui(app, lambda: not video.transient["tracker_busy"])
    candidate = video.review.candidates(1)[0]
    assert candidate.class_id == 1 and candidate.action_id == 1
    assert candidate.track_id == seed.track_id
    assert 1 not in video.annotations
    assert app._review is None


def test_shrinking_and_deleting_behavior_updates_saved_actions(app):
    from htrtdetr.annotation.objects import AnnObject

    for frame in range(5):
        app.annotations[frame] = [AnnObject([5, 5, 25, 25], track_id=1, action_id=1)]
    interval = BehaviorInterval(1, 1, 0, 4, source="human")
    app._intervals = IntervalTrack([interval])
    app._sel_interval = interval
    app.goto_frame(2)
    app._insp_move_interval("end")
    assert [app.annotations[f][0].action_id for f in range(5)] == [1, 1, 1, -1, -1]
    app.undo()
    assert all(app.annotations[f][0].action_id == 1 for f in range(5))
    app._sel_interval = app._intervals.intervals[0]
    app._insp_delete_interval()
    assert all(app.annotations[f][0].action_id == -1 for f in range(5))
    app.undo()
    assert len(app._intervals.intervals) == 1
    assert all(app.annotations[f][0].action_id == 1 for f in range(5))


@pytest.fixture(scope="module")
def viewport():
    app = AnnotationApp()
    dpg.create_viewport(title="YOAKE project smoke tests", width=1280, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    app.class_names = ["fly", "worm"]
    app.action_names = ["walking", "resting"]
    app._build_main_window()
    yield app
    if app.cap is not None:
        app.cap.release()
    app._stop_sam()
    dpg.destroy_context()


@pytest.fixture
def app(viewport, tmp_path):
    for name in ("a.avi", "b.avi"):
        writer = cv2.VideoWriter(
            str(tmp_path / name), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
        )
        assert writer.isOpened()
        for index in range(20):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            cv2.rectangle(frame, (10 + index, 10), (30 + index, 30), (200, 200, 200), -1)
            writer.write(frame)
        writer.release()
    project = ProjectSession("test", ["fly", "worm"], ["walking", "resting"], directory=tmp_path)
    project.settings["onboarding_done"] = True
    viewport._adopt_project(project)
    viewport._add_videos([tmp_path / "a.avi", tmp_path / "b.avi"])
    for _ in range(2):
        dpg.render_dearpygui_frame()
    return viewport


def test_create_folder_annotate_switch_save_restart_reopen(app, tmp_path, monkeypatch):
    import htrtdetr.annotator as module

    # Drive the actual setup and folder callbacks used by a new user.
    app._show_setup_screen()
    monkeypatch.setattr(module, "_tk_ask_dir", lambda *args, **kw: tmp_path)
    app._setup_add_folder()
    dpg.set_value("setup_proj_name", "experiment")
    dpg.set_value("setup_proj_dir", str(tmp_path / "project"))
    dpg.set_value("setup_classes", "fly, worm")
    app._on_new_project_confirmed()
    assert list(app.project.videos) == ["a", "b"]
    first = app._create_object([10, 10, 30, 30])
    app._insp_set_class("worm")
    app.goto_frame(7)
    app.switch_video("b")
    second = app._create_object([40, 20, 60, 40])
    app.goto_frame(13)
    assert app.switch_video("a")
    assert app.current_frame_idx == 7
    assert app.annotations[0][0] is first
    app.save_annotation()
    manifest = tmp_path / "project" / "project.yoake.json"
    assert manifest.exists()
    # A fresh project model simulates the state after restarting the application.
    reopened = ProjectSession.open(manifest)
    app._adopt_project(reopened)
    app.switch_video(reopened.last_video)
    assert app.current_frame_idx == 7
    assert app.annotations[0][0].class_id == 1
    assert app.switch_video("b")
    assert app.current_frame_idx == 13
    assert app.annotations[0][0].bbox == second.bbox


def test_undo_redo_never_crosses_videos_or_candidates(app):
    app._create_object([5, 5, 25, 25])
    a = app._video_session
    app.switch_video("b")
    app._review = ReviewSession({0: [Candidate([40, 20, 60, 40], class_id=1, track_id=1)]})
    app.accept_candidate(0)
    assert app.annotations[0][0].class_id == 1
    app.undo()
    assert app.annotations == {}
    assert app._review.state(0, 0) == "pending"
    assert len(a.annotations[0]) == 1
    app.redo()
    assert app._review.state(0, 0) == "accepted"
    assert app.switch_video("a")
    app.undo()
    assert app.annotations == {}
    assert len(app.project.videos["b"].annotations[0]) == 1


def test_switch_preserves_selection_view_behavior_and_predictions(app):
    first = app._create_object([5, 5, 25, 25])
    app._intervals = IntervalTrack([BehaviorInterval(1, 1, 0, 10, source="human")])
    app._predictions = {0: [Candidate([5, 5, 25, 25], class_id=1)]}
    app._zoom = 1.7
    namespace_a = app._sam_frame_id(0)
    app.switch_video("b")
    assert not app._intervals.intervals and not app._predictions
    assert app._sam_frame_id(0) != namespace_a
    app.switch_video("a")
    assert app._selected is first
    assert app._zoom == pytest.approx(1.7)
    assert app._predictions[0][0].class_id == 1
    assert app._intervals.intervals[0].source == "human"


def test_next_review_crosses_videos_and_keeps_ground_truth(app):
    app._create_object([5, 5, 25, 25])
    a, b = app.project.videos.values()
    a.review = ReviewSession({5: [Candidate([6, 6, 26, 26], class_id=1)]})
    b.review = ReviewSession({8: [Candidate([7, 7, 27, 27], class_id=1)]})
    app._project_review_key = None
    assert app.review_next_project()
    assert app._video_session is a and app.current_frame_idx == 5
    assert app.review_next_project()
    assert app._video_session is b and app.current_frame_idx == 8
    assert len(a.annotations[0]) == 1
    assert not b.annotations


def test_late_batch_result_is_applied_to_origin_video(app):
    origin = app.project.videos["a"]
    app.switch_video("b")
    result = VideoResult(
        "a",
        candidates={3: [Candidate([1, 1, 10, 10], class_id=1)]},
        num_frames=20,
        width=96,
        height=64,
    )
    app._batch_event(app.project, {"kind": "video", "result": result}, BatchOptions(detect=True))
    assert app._review is None
    assert origin.review.candidates(3)[0].class_id == 1
    assert not origin.annotations
    other = ProjectSession()
    app._batch_event(other, {"kind": "video", "result": result}, BatchOptions(detect=True))
    assert len(origin.review.candidates(3)) == 1


def test_candidates_cache_rerun_preserves_rejection(app):
    video = app._video_session
    video.review = ReviewSession({0: [Candidate([1, 1, 10, 10], class_id=1)]})
    video.review.reject(0, 0)
    app._merge_video_candidates(video, {0: [Candidate([1, 1, 10, 10], class_id=1)]})
    assert len(video.review.candidates(0)) == 1
    assert video.review.state(0, 0) == "rejected"


def test_end_of_video_can_skip_done_videos(app):
    app.project.settings["end_of_video"] = "next_unfinished"
    app.goto_frame(20)
    assert app._video_session.video_id == "b"
    app.switch_video("a")
    app.project.videos["b"].status = "done"
    app.goto_frame(20)
    assert app._video_session.video_id == "a"
    assert app.current_frame_idx == 19


def test_copy_and_class_edit_keep_obb_and_history(app):
    obj = app._create_object([5, 5, 25, 25])
    obj.make_obb()
    app._insp_set_class("worm")
    app.continue_animal("Copy to next frame")
    assert app.current_frame_idx == 1
    assert app.annotations[1][0].class_id == 1
    assert app.annotations[1][0].obb == obj.obb
    app.undo()
    assert 1 not in app.annotations


def test_browser_batch_help_build_with_current_project(app):
    app.show_video_browser()
    app.open_batch_dialog("sample")
    app.show_project_review()
    app.show_onboarding()
    app._show_help()
    for tag in (
        "project_videos",
        "project_batch",
        "project_review",
        "guided_annotation",
        "help_win",
    ):
        assert dpg.does_item_exist(tag)
        dpg.hide_item(tag)


def test_existing_json_then_open_video_never_resets_annotations(app, tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps(
            {
                "class_names": ["fly", "worm"],
                "videos": [
                    {
                        "video_id": "a",
                        "frames": [
                            {"frame_index": 0, "objects": [{"bbox": [3, 3, 20, 20], "class_id": 1}]}
                        ],
                    },
                    {
                        "video_id": "b",
                        "frames": [
                            {
                                "frame_index": 0,
                                "objects": [{"bbox": [30, 30, 50, 50], "class_id": 0}],
                            }
                        ],
                    },
                ],
            }
        )
    )
    app._load_annotation_file(legacy)
    app._load_video(tmp_path / "b.avi")
    assert app.annotations[0][0].bbox == [30, 30, 50, 50]
    app._load_video(tmp_path / "a.avi")
    assert app.annotations[0][0].class_id == 1


def test_selected_video_batch_and_entire_project_sampling(app):
    from htrtdetr.annotation.sampling import TARGET_DETECTION, Budget, SamplingRequest

    options = BatchOptions(
        stride=2, request=SamplingRequest(target=TARGET_DETECTION, budget=Budget(samples=4))
    )
    app.project.selected_videos = {"b"}
    assert app.start_project_batch("Selected videos", options)
    deadline = time.monotonic() + 10
    while app._batch_thread.is_alive() and time.monotonic() < deadline:
        app._pump_ui()
        time.sleep(0.01)
    app._pump_ui()
    assert app.project.videos["a"].selection is None
    assert app.project.videos["b"].selection.total == 4
    assert app.start_project_batch("Entire project", options)
    while app._batch_thread.is_alive() and time.monotonic() < deadline:
        app._pump_ui()
        time.sleep(0.01)
    app._pump_ui()
    assert [v.selection.total for v in app.project.videos.values()] == [2, 2]


def test_behavior_interval_undo_restores_boundaries(app):
    for index in range(10):
        app.goto_frame(index)
        obj = app._create_object([5, 5, 25, 25])
        obj.action_id = 0
    interval = BehaviorInterval(1, 0, 0, 9, source="human")
    app._intervals = IntervalTrack([interval])
    app._sel_interval = interval
    app.goto_frame(4)
    app._insp_move_interval("start")
    assert app._intervals.intervals[0].start == 4
    app.undo()
    assert app._intervals.intervals[0].start == 0


def test_autosave_does_not_acknowledge_newer_edits(app, monkeypatch):
    import threading

    from htrtdetr.gui import workspace

    started, release = threading.Event(), threading.Event()
    write = workspace.save_snapshot

    def slow_save(*args):
        started.set()
        assert release.wait(5)
        write(*args)

    monkeypatch.setattr(workspace, "save_snapshot", slow_save)
    app._create_object([5, 5, 25, 25])
    app._save_due = time.time() - 1
    app._tick_autosave()
    assert started.wait(5)
    app._create_object([40, 30, 60, 50])
    release.set()
    app._workspace_save_thread.join(5)
    app._pump_ui()
    assert app._dirty
    assert app._video_session.dirty
