"""
tests/test_track_identity.py — the project-level "do we follow individuals?" switch.

Some experiments count animals frame by frame and never ask which fly is which.
For those, a track id is not merely unused: it is a field the annotator keeps
asking a human to fill in, a set of checks that can never fire, and a column in
the saved file that asserts an identity nobody established.

The switch lives on the project (``settings["use_track_ids"]``), defaults to
True so every file written before it existed behaves exactly as it did, and is
read by both halves of the tool — the annotator and the analysis page.

The pure half of this runs anywhere. The GUI half needs a real window for the
same reason ``test_workflows.py`` does, and is marked ``gui``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.annotation import quality  # noqa: E402
from htrtdetr.annotation.project import ProjectSession  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# The setting itself (no GUI)
# ══════════════════════════════════════════════════════════════════════════


def test_a_new_project_follows_individuals():
    """The default has to be the old behaviour, or every existing project
    silently changes meaning the first time it is reopened."""
    assert ProjectSession().settings["use_track_ids"] is True


def test_the_setting_survives_a_round_trip(tmp_path):
    project = ProjectSession()
    project.settings["use_track_ids"] = False
    path = tmp_path / "project.yoake.json"
    project.save(path)
    assert ProjectSession.open(path).settings["use_track_ids"] is False


def test_a_file_written_before_the_setting_existed_reads_as_tracking(tmp_path):
    """An older project.yoake.json has no such key. Absent must mean True:
    reading it as False would turn off checks the user was relying on."""
    project = ProjectSession()
    project.settings.pop("use_track_ids")
    path = tmp_path / "project.yoake.json"
    project.save(path)
    reopened = ProjectSession.open(path)
    assert reopened.settings.get("use_track_ids", True) is True


# ══════════════════════════════════════════════════════════════════════════
# Which checks are worth running
# ══════════════════════════════════════════════════════════════════════════


def test_without_ids_only_the_single_frame_checks_run():
    keys = quality.detector_keys(use_track_ids=False)
    assert set(keys) == {
        quality.LOW_CONFIDENCE,
        "count",
        quality.OVERLAP,
        quality.MERGED_DETECTION,
    }
    for kind in quality.TRACK_DEPENDENT:
        assert kind not in keys


def test_with_ids_every_detector_runs():
    assert set(quality.detector_keys()) == set(quality.DETECTORS)


def test_every_track_dependent_key_is_a_real_detector():
    """A typo here would silently disable nothing, which is the worst outcome:
    the check would keep running and the list would keep looking correct."""
    for kind in quality.TRACK_DEPENDENT:
        assert kind in quality.DETECTORS


def test_analyze_honours_the_shorter_list():
    """The switch is wired to ``analyze``'s existing ``enabled``, not to a new
    branch inside each detector."""

    class C:
        def __init__(self, bbox, track_id):
            self.bbox, self.obb, self.score = bbox, None, 0.9
            self.track_id, self.action_id = track_id, 0

    # One track that teleports: a position jump with ids, nothing without.
    candidates = {
        0: [C([10, 10, 20, 20], 1)],
        1: [C([11, 10, 21, 20], 1)],
        2: [C([12, 10, 22, 20], 1)],
        3: [C([500, 400, 510, 410], 1)],
    }
    tracked = quality.analyze(candidates, num_frames=4)
    untracked = quality.analyze(
        candidates, enabled=quality.detector_keys(use_track_ids=False), num_frames=4
    )
    assert any(i.kind == quality.POSITION_JUMP for i in tracked)
    assert not any(i.kind in quality.TRACK_DEPENDENT for i in untracked)


# ══════════════════════════════════════════════════════════════════════════
# The annotator, against a real window
# ══════════════════════════════════════════════════════════════════════════

# Imported rather than ``importorskip``-ed: a module-level skip would take the
# pure tests above with it, and those are exactly the ones that have to run on
# the interpreter that has torch but no dearpygui (see docs/MIGRATION.md and
# tests/test_annotation_core.py on why the two interpreters are disjoint).
try:
    import cv2
    import dearpygui.dearpygui as dpg
    import numpy as np

    from htrtdetr import annotator as A

    _HAS_GUI = True
except ImportError:  # torch interpreter: no dearpygui, no cv2
    _HAS_GUI = False

gui_only = pytest.mark.skipif(not _HAS_GUI, reason="needs dearpygui and OpenCV")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A real 30-frame clip: Continue this animal decodes the next frame."""
    folder = tmp_path_factory.mktemp("yoake_tracks")
    path = folder / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 240))
    for index in range(30):
        frame = np.full((240, 320, 3), 28, np.uint8)
        cv2.ellipse(frame, (40 + index * 4, 120), (12, 5), 0, 0, 360, (215, 215, 225), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def instance():
    app = A.AnnotationApp()
    dpg.create_viewport(title="yoake track identity tests", width=1280, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    app.class_names = ["fly", "worm"]
    app.action_names = list(A._DEFAULT_ACTIONS)
    app._build_main_window()
    app._update_action_buttons()
    yield app
    dpg.destroy_context()


@pytest.fixture
def app(instance, video, tmp_path):
    instance.annotations = {}
    instance._undo.clear()
    instance._redo.clear()
    instance._selected = None
    instance.project_dir = tmp_path
    instance._load_video(video)
    instance.goto_frame(0)
    instance.set_use_track_ids(True)
    instance.active_track_id = 1
    return instance


@pytest.mark.gui
@gui_only
def test_a_box_drawn_without_ids_asserts_no_identity(app):
    app.set_use_track_ids(False)
    obj = app._create_object([10, 10, 40, 40])
    assert obj.track_id == -1, "-1 is the file format's own 'no identity'"
    assert obj.to_dict()["track_id"] == -1


@pytest.mark.gui
@gui_only
def test_turning_ids_back_on_gives_new_boxes_an_id_again(app):
    app.set_use_track_ids(False)
    app._create_object([10, 10, 40, 40])
    app.set_use_track_ids(True)
    assert app._create_object([50, 50, 80, 80]).track_id == app.active_track_id


@pytest.mark.gui
@gui_only
def test_switching_off_never_rewrites_what_is_already_saved(app):
    """Deleting ids on a checkbox would destroy work that switching back on
    could not restore, so the switch changes what happens *next*."""
    kept = app._create_object([10, 10, 40, 40])
    assert kept.track_id == 1
    app.set_use_track_ids(False)
    assert kept.track_id == 1


@pytest.mark.gui
@gui_only
def test_the_track_only_actions_refuse_and_say_where_the_switch_is(app):
    app.set_use_track_ids(False)
    before = app.active_track_id
    app.new_track()
    assert app.active_track_id == before
    assert "Tracking" in dpg.get_value("status_text")


@pytest.mark.gui
@gui_only
def test_following_boxes_between_frames_cannot_be_turned_on_without_ids(app):
    """Every tracker is keyed by track id. With one id for everything they
    would all overwrite each other, so the feature is refused, not broken."""
    app.set_use_track_ids(False)
    app._set_auto_track(True)
    assert app._auto_track is False


@pytest.mark.gui
@gui_only
def test_turning_ids_off_turns_auto_track_off_with_them(app):
    app._auto_track = True
    app.set_use_track_ids(False)
    assert app._auto_track is False


@pytest.mark.gui
@gui_only
def test_the_inspector_hides_the_track_controls(app):
    app.set_use_track_ids(False)
    app._refresh_inspector()
    assert not dpg.is_item_shown("insp_n_track_block")
    app.set_use_track_ids(True)
    app._refresh_inspector()
    assert dpg.is_item_shown("insp_n_track_block")


@pytest.mark.gui
@gui_only
def test_the_box_label_names_the_class_when_there_is_no_id(app):
    """The label has to carry *something*: with ids it is the identity, and
    without them the class is the only thing the box actually claims."""
    app.set_use_track_ids(False)
    app._create_object([10, 10, 40, 40])
    app._render_canvas()  # must not raise with track_id == -1
    app.set_use_track_ids(True)


@pytest.mark.gui
@gui_only
def test_the_checks_drop_the_track_detectors(app):
    app.set_use_track_ids(False)
    app.rebuild_copilot()
    assert all(i.kind not in quality.TRACK_DEPENDENT for i in app._queue.items)


@pytest.mark.gui
@gui_only
def test_continue_this_animal_falls_back_to_copying_the_box(app):
    app.set_use_track_ids(False)
    app.project.settings["continue_method"] = "Fast tracking"
    obj = app._create_object([10, 10, 40, 40])
    app._select(obj)
    app.continue_animal()
    # Copy to next frame is the part that still means something.
    assert len(app.annotations.get(1, [])) == 1
    assert app.annotations[1][0].track_id == -1
