"""
tests/test_keyframe_mode_gui.py — the annotator's keyframe labelling scheme.

``docs/KEYFRAME_BEHAVIOR.md``: on chosen frames a box and a behavior for every
animal; no track ids, no intervals; ``P`` plays the motion around the frame
while labels stay on it.

The project-file half runs anywhere. The annotator half needs a real window,
like ``test_track_identity.py``, and is marked ``gui``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.annotation import keyframes  # noqa: E402
from htrtdetr.annotation.project import ProjectSession  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# The project file (no GUI)
# ══════════════════════════════════════════════════════════════════════════

def test_an_older_project_without_the_key_stays_on_tracks():
    """Absent reads tracks: nothing written before keyframes changes behaviour."""
    assert "label_mode" not in ProjectSession().settings


def test_a_keyframe_project_exports_a_file_that_says_so(tmp_path):
    project = ProjectSession(name="kf", action_names=["still", "walk"])
    project.settings["label_mode"] = "keyframe"
    project.settings["clip"] = {"length": 6, "stride": 2, "anchor": "end"}
    data = project.dataset(tmp_path)
    assert keyframes.label_mode_of(data) == keyframes.KEYFRAME
    assert keyframes.clip_spec_of(data) == keyframes.ClipSpec(length=6, stride=2)
    project.settings["label_mode"] = "tracks"
    assert "labeling" not in project.dataset(tmp_path).meta


def test_opening_a_keyframe_file_opens_a_keyframe_project(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps({
        "meta": {"version": "2.0", "labeling": keyframes.labeling_meta(
            keyframes.ClipSpec(length=4, stride=3))},
        "class_names": ["fly"], "action_names": ["still", "walk"],
        "videos": [{"video_id": "v", "fps": 30.0, "width": 64, "height": 48,
                    "num_frames": 100, "frames": [
                        {"frame_index": 40, "image_path": "", "objects": [
                            {"object_id": 1, "bbox": [1, 1, 9, 9], "class_id": 0,
                             "track_id": -1, "action_id": 1}]}]}],
    }), encoding="utf-8")
    project = ProjectSession.from_annotation(path)
    assert project.settings["label_mode"] == "keyframe"
    assert project.settings["use_track_ids"] is False
    assert project.settings["clip"]["stride"] == 3


# ══════════════════════════════════════════════════════════════════════════
# The annotator, against a real window
# ══════════════════════════════════════════════════════════════════════════

try:
    import cv2
    import dearpygui.dearpygui as dpg
    import numpy as np

    from htrtdetr import annotator as A
    from htrtdetr.gui import keyframe_mode

    _HAS_GUI = True
except ImportError:  # the torch interpreter has no dearpygui and no cv2
    _HAS_GUI = False

gui_only = pytest.mark.skipif(not _HAS_GUI, reason="needs dearpygui and OpenCV")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """90 frames; each frame's index is written into its top-left pixels."""
    folder = tmp_path_factory.mktemp("yoake_keyframes")
    path = folder / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    for index in range(90):
        frame = np.full((120, 160, 3), 40, np.uint8)
        frame[:10, :10] = 2 * index + 10
        cv2.ellipse(frame, (30 + index, 60), (12, 5), 0, 0, 360, (215, 215, 225), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def instance():
    app = A.AnnotationApp()
    dpg.create_viewport(title="yoake keyframe tests", width=1280, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    app.class_names = ["fly"]
    app.action_names = ["still", "walk", "groom"]
    app._build_main_window()
    app._update_action_buttons()
    yield app
    dpg.destroy_context()


@pytest.fixture
def app(instance, video, tmp_path):
    instance.stop_context_loop(restore=False)
    instance.annotations = {}
    instance._undo.clear()
    instance._redo.clear()
    instance._selected = None
    instance.project_dir = tmp_path
    instance._load_video(video)
    instance.set_label_mode("tracks")
    instance.set_use_track_ids(True)
    instance.set_label_mode("keyframe")
    instance.goto_frame(45)
    return instance


@pytest.mark.gui
@gui_only
def test_keyframe_mode_asks_for_no_track_id_and_keeps_the_stored_flag(app):
    assert app.keyframe_mode and app.label_mode == "keyframe"
    assert app.use_track_ids is False
    assert app._create_object([10, 10, 40, 40]).track_id == -1
    # the stored flag is untouched, so switching back restores it
    assert app.project.settings["use_track_ids"] is True
    app.set_label_mode("tracks")
    assert app.use_track_ids is True


@pytest.mark.gui
@gui_only
def test_boxes_are_coloured_by_behavior_and_grey_until_given_one(app):
    obj = app._create_object([10, 10, 40, 40])
    col, fill = app._keyframe_box_colors(obj, (1, 2, 3, 255), (1, 2, 3, 50))
    assert fill[3] == 0, "an unanswered box is unfilled"
    app._selected = obj
    app.set_action(1)
    col1, fill1 = app._keyframe_box_colors(obj, (1, 2, 3, 255), (1, 2, 3, 50))
    assert col1 != col and fill1[3] > 0
    assert app._keyframe_box_label(obj, "walk") == "walk"


@pytest.mark.gui
@gui_only
def test_the_behavior_lane_shows_one_tick_per_labelled_animal(app):
    a = app._create_object([10, 10, 40, 40])
    b = app._create_object([60, 10, 90, 40])
    a.action_id, b.action_id = 2, -1
    marks = app._behavior_lane()
    assert [(m.start, m.end, m.action_id, m.source) for m in marks] == [(45, 45, 2, "human")]
    assert "1 still needs a behavior" in app._keyframe_note()
    app.rebuild_copilot()
    assert app._intervals.intervals == [], "keyframe labels are never intervals"


@pytest.mark.gui
@gui_only
def test_the_context_loop_plays_around_the_frame_and_leaves_labels_on_it(app):
    app._create_object([10, 10, 40, 40])
    assert app.start_context_loop()
    loop = app._context_loop_state()
    assert loop.active and loop.anchor == 45
    assert (loop.lo, loop.hi) == (15, 60)            # 1.0 s before, 0.5 s after at 30 fps
    assert app.current_frame_idx == 45, "the loop never moves the labelled frame"
    shown = loop.advance()
    app._show_context_frame(shown)
    assert app.current_frame_idx == 45
    app.set_action(1) if app._selected is not None else None
    app.stop_context_loop()
    assert not app.context_loop_active
    assert len(app.annotations[45]) == 1


@pytest.mark.gui
@gui_only
def test_p_toggles_and_esc_or_moving_ends_the_loop(app):
    actions = app._actions()
    actions["view.context_loop"]()
    assert app.context_loop_active
    app._escape()
    assert not app.context_loop_active
    app.toggle_context_loop()
    assert app.context_loop_active
    app.goto_frame(50)
    assert not app.context_loop_active and app.current_frame_idx == 50


@pytest.mark.gui
@gui_only
def test_the_setup_screen_defaults_to_keyframes(instance):
    assert keyframe_mode.LABEL_MODE_CHOICES[0].startswith("Keyframe")
    settings: dict = {}
    with dpg.window(tag="kf_setup_probe", show=False):
        instance._build_setup_label_mode(400)
    try:
        instance._apply_setup_label_mode(settings)
        assert settings == {"label_mode": "keyframe", "use_track_ids": False}
        dpg.set_value("setup_label_mode", keyframe_mode.LABEL_MODE_CHOICES[1])
        instance._apply_setup_label_mode(settings)
        assert settings == {"label_mode": "tracks", "use_track_ids": True}
    finally:
        dpg.delete_item("kf_setup_probe")
