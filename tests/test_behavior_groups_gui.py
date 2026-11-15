"""
tests/test_behavior_groups_gui.py — multi-head behaviors in the annotator.

``docs/KEYFRAME_BEHAVIOR.md`` §15.3: a keyframe project may group its
behaviors; then a key answers its own group and keeps the others, the box is
filled only when every group is answered, the Inspector has one menu per group,
and editing the vocabulary keeps every answer already given by name.

Needs a real window, like ``test_keyframe_mode_gui.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.annotation import behaviors  # noqa: E402

try:
    import cv2
    import dearpygui.dearpygui as dpg
    import numpy as np

    from htrtdetr import annotator as A
    from htrtdetr.gui import behavior_groups as BG
    from htrtdetr.gui import keyframe_mode

    _HAS_GUI = True
except ImportError:  # the torch interpreter may have no dearpygui
    _HAS_GUI = False

pytestmark = [
    pytest.mark.gui,
    pytest.mark.skipif(not _HAS_GUI, reason="needs dearpygui and OpenCV"),
]

GROUPS = "locomotion: still, walk, run\nsocial: none, courtship, copulation"


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    folder = tmp_path_factory.mktemp("yoake_groups")
    path = folder / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    for index in range(60):
        frame = np.full((120, 160, 3), 40, np.uint8)
        cv2.ellipse(frame, (30 + index, 60), (12, 5), 0, 0, 360, (215, 215, 225), -1)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def instance():
    app = A.AnnotationApp()
    dpg.create_viewport(title="yoake behavior groups", width=1280, height=820)
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
    instance.set_label_mode("keyframe")
    instance.set_behavior_vocabulary(
        behaviors.BehaviorVocabulary.single(["still", "walk", "groom"]), remap=False
    )
    instance.goto_frame(20)
    if dpg.does_item_exist(BG.EDITOR_TAG):
        dpg.delete_item(BG.EDITOR_TAG)
    return instance


def _groups(app):
    app.set_behavior_vocabulary(behaviors.parse_vocabulary(GROUPS, True), remap=False)
    return app.behavior_vocabulary


def _box(app):
    obj = app._create_object([10, 10, 40, 40])
    app._selected = obj
    return obj


def test_a_key_answers_its_group_and_undo_takes_it_back(app):
    vocab = _groups(app)
    assert vocab.multi and app.action_names == list(vocab.names)
    obj = _box(app)
    app.set_action(1)                                  # walk
    app.set_action(4)                                  # courtship
    assert obj.action_ids == [1, 4] and obj.action_id == 1
    assert app._behavior_text(obj) == "walk + courtship"
    assert "still open" not in app._behavior_status(obj)
    app.set_action(2)                                  # run replaces walk, keeps courtship
    assert obj.action_ids == [2, 4]
    app._actions()["edit.undo"]()
    assert app.annotations[20][0].action_ids == [1, 4]


def test_a_box_is_filled_only_once_every_group_is_answered(app):
    from htrtdetr.gui import theme

    _groups(app)
    obj = _box(app)
    app.set_action(1)
    col, fill = app._keyframe_box_colors(obj, (1, 2, 3, 255), (1, 2, 3, 50))
    assert col == theme.class_color(1) and fill[3] == 0, "social is still open"
    assert "still open: social" in app._behavior_status(obj)
    app.set_action(3)
    col, fill = app._keyframe_box_colors(obj, (1, 2, 3, 255), (1, 2, 3, 50))
    assert col == theme.class_color(1) and fill[3] > 0
    assert app._keyframe_box_label(obj, app._behavior_text(obj)) == "walk + none"


def test_the_inspector_has_one_menu_per_group(app):
    _groups(app)
    obj = _box(app)
    app.set_action(0)
    app._refresh_sidebar()
    insp = app.inspector
    assert dpg.does_item_exist(insp._t("b_group_1"))
    assert dpg.get_value(insp._t("b_group_0")) == "still"
    assert dpg.get_value(insp._t("b_group_1")) == "Not set"
    assert not dpg.get_item_configuration(insp._t("b_action"))["show"]
    app._insp_set_group_behavior((1, "copulation"))
    assert obj.action_ids == [0, 5]
    app._insp_set_group_behavior((0, "Not set"))
    assert obj.action_ids == [-1, 5] and obj.action_id == -1


def test_the_note_and_the_lane_follow_the_groups(app):
    _groups(app)
    a, b = _box(app), _box(app)
    behaviors.set_labels(a, app.behavior_vocabulary, [1, 4])
    behaviors.set_labels(b, app.behavior_vocabulary, [0, -1])
    assert app._keyframe_note() == "2 animals - 1 still needs a behavior (social)."
    ticks = sorted(m.action_id for m in app._behavior_lane())
    assert ticks == [0, 1, 4]


def test_the_buttons_carry_a_heading_per_group(app):
    _groups(app)
    texts, buttons = [], []
    for item in dpg.get_item_children("action_buttons_group", 1):
        kind = dpg.get_item_type(item)
        if kind.endswith("mvText"):
            texts.append(dpg.get_value(item))
        elif kind.endswith("mvButton"):
            buttons.append(dpg.get_item_label(item))
    assert texts == ["locomotion", "social"]
    assert buttons[3] == "[3] None" and len(buttons) == 6


def test_editing_the_vocabulary_keeps_every_answer(app):
    _groups(app)
    obj = _box(app)
    app.set_action(1)
    app.set_action(4)
    app.open_behavior_editor()
    assert dpg.get_value(BG.EDITOR_TAG + "_text") == GROUPS
    dpg.set_value(BG.EDITOR_TAG + "_text",
                  "locomotion: still, walk, run, jump\nsocial: none, courtship, copulation")
    app._apply_behavior_editor()
    assert not dpg.does_item_exist(BG.EDITOR_TAG)
    assert obj.action_ids == [1, 5], "courtship moved to 5 and the answer with it"
    assert app.project.action_groups["locomotion"][-1] == "jump"
    assert app.project.settings["behaviors"] == behaviors.MULTI_HEAD

    app.open_behavior_editor()
    dpg.set_value(BG.EDITOR_TAG + "_scheme", BG.BEHAVIOR_SCHEME_CHOICES[0])
    app._on_editor_scheme(BG.BEHAVIOR_SCHEME_CHOICES[0])
    assert dpg.get_value(BG.EDITOR_TAG + "_text").startswith("still, walk, run, jump, none")
    app._apply_behavior_editor()
    assert not app.behavior_vocabulary.multi
    assert obj.action_id == 1 and obj.action_ids == []
    assert app.project.settings["behaviors"] == behaviors.SINGLE


def test_a_bad_edit_is_explained_and_changes_nothing(app):
    _groups(app)
    app.open_behavior_editor()
    dpg.set_value(BG.EDITOR_TAG + "_text", "locomotion: still")
    app._apply_behavior_editor()
    assert dpg.does_item_exist(BG.EDITOR_TAG)
    assert "at least two" in dpg.get_value(BG.EDITOR_TAG + "_error")
    assert app.behavior_vocabulary.heads == ("locomotion", "social")
    dpg.delete_item(BG.EDITOR_TAG)


def test_add_behavior_asks_for_the_group(app):
    _groups(app)
    app.add_behavior_dialog()
    assert dpg.does_item_exist(BG.EDITOR_TAG)
    dpg.delete_item(BG.EDITOR_TAG)


def test_the_tracks_scheme_reads_the_same_names_as_one_list(app):
    _groups(app)
    app.set_label_mode("tracks")
    try:
        assert not app.behavior_vocabulary.multi
        assert app.behavior_vocabulary.names == tuple(app.action_names)
        obj = _box(app)
        app.set_action(4)
        assert obj.action_id == 4 and not obj.action_ids
    finally:
        app.set_label_mode("keyframe")


def test_the_setup_screen_offers_groups_with_keyframes_only(instance):
    with dpg.window(tag="groups_setup_probe", show=False):
        instance._build_setup_label_mode(400)
        with dpg.collapsing_header(tag="setup_actions_fold"):
            instance._build_setup_behaviors("still, walk")
    try:
        vocab = instance._read_setup_behaviors(["x", "y"])
        assert vocab is not None and not vocab.multi and vocab.names == ("still", "walk")

        dpg.set_value("setup_behavior_scheme", BG.BEHAVIOR_SCHEME_CHOICES[1])
        instance._on_setup_behavior_scheme(BG.BEHAVIOR_SCHEME_CHOICES[1])
        assert dpg.get_value("setup_actions").startswith("locomotion:")
        vocab = instance._read_setup_behaviors(["x", "y"])
        assert vocab is not None and vocab.multi and vocab.heads == ("locomotion", "social")

        dpg.set_value("setup_label_mode", keyframe_mode.LABEL_MODE_CHOICES[1])   # tracks
        assert instance._read_setup_behaviors(["x", "y"]) is None
        assert "Keyframes" in dpg.get_value("setup_actions_error")

        dpg.set_value("setup_label_mode", keyframe_mode.LABEL_MODE_CHOICES[0])
        dpg.set_value("setup_actions", "social none, courtship")
        assert instance._read_setup_behaviors(["x", "y"]) is None
        assert "group name" in dpg.get_value("setup_actions_error")

        dpg.set_value("setup_behavior_scheme", BG.BEHAVIOR_SCHEME_CHOICES[0])
        dpg.set_value("setup_actions", "")
        vocab = instance._read_setup_behaviors(["x", "y"])
        assert vocab is not None and vocab.names == ("x", "y"), "an empty field keeps the defaults"
    finally:
        dpg.delete_item("groups_setup_probe")
