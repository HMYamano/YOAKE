"""
tests/test_gui_dialogs.py — every dialog fits, and its buttons are reachable.

The bug this file exists for: a dialog is created with a fixed ``width`` and
``height``, its content grows past that, and the button the user opened it for
ends up below the bottom edge. It is invisible in review — the code reads fine
— and it is invisible at the author's window size. It only shows up on a
smaller screen, or once one more setting is added.

So the viewport here is **900x600**, the annotator's own declared minimum
(``create_viewport(min_width=900, min_height=600)``), not the 1280x820 the rest
of the suite uses. Every dialog is opened, rendered, and then asked two
questions:

1. Is the window itself inside the client area?
2. Is its primary button inside the window?

Both are measured from dearpygui rather than derived from the constants, so a
layout that happens to work stays passing and a layout that only works on the
author's screen does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

dpg = pytest.importorskip("dearpygui.dearpygui")
pytest.importorskip("cv2")
pytest.importorskip("numpy")

from htrtdetr import annotator as A  # noqa: E402
from htrtdetr.gui import widgets  # noqa: E402

pytestmark = pytest.mark.gui

#: The smallest window the annotator lets the user make.
WIDTH, HEIGHT = 900, 600


@pytest.fixture(scope="module")
def app():
    instance = A.AnnotationApp()
    dpg.create_viewport(title="yoake dialog tests", width=WIDTH, height=HEIGHT)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    instance.class_names = list(A._DEFAULT_CLASSES)
    instance.action_names = list(A._DEFAULT_ACTIONS)
    instance._build_main_window()
    instance._update_action_buttons()
    yield instance
    dpg.destroy_context()


def _pump(frames: int = 3) -> None:
    for _ in range(frames):
        dpg.render_dearpygui_frame()


def _client() -> tuple[int, int]:
    return dpg.get_viewport_client_width(), dpg.get_viewport_client_height()


def _assert_on_screen(tag: str) -> tuple[float, float, float, float]:
    """The window is inside the client area. Returns its rect."""
    assert dpg.does_item_exist(tag), f"{tag} was never created"
    vw, vh = _client()
    x, y = dpg.get_item_pos(tag)
    w, h = dpg.get_item_rect_size(tag)
    assert x >= 0 and y >= 0, f"{tag} starts off screen at ({x}, {y})"
    assert x + w <= vw + 1, f"{tag} reaches x={x + w} in a {vw}px client area"
    assert y + h <= vh + 1, f"{tag} reaches y={y + h} in a {vh}px client area"
    return x, y, w, h


def _assert_button_inside(window: str, button: str) -> None:
    """The button the dialog exists for is inside the dialog, not under it."""
    assert dpg.does_item_exist(button), f"{button} is missing from {window}"
    _wx, wy, _ww, wh = _assert_on_screen(window)
    _bx, by = dpg.get_item_rect_min(button)
    _bw, bh = dpg.get_item_rect_size(button)
    assert by + bh <= wy + wh + 1, (
        f"{button} ends at y={int(by + bh)}, below {window}'s bottom edge at "
        f"{int(wy + wh)} - it cannot be clicked without resizing the window"
    )


# ══════════════════════════════════════════════════════════════════════════
# The geometry helper itself
# ══════════════════════════════════════════════════════════════════════════


def test_a_dialog_larger_than_the_viewport_is_shrunk(app):
    w, h, x, y = widgets.fit_dialog(2000, 2000)
    vw, vh = _client()
    assert w < vw and h < vh
    assert x >= 0 and y >= 0
    assert x + w <= vw and y + h <= vh


def test_a_dialog_that_already_fits_keeps_its_size(app):
    w, h, _x, _y = widgets.fit_dialog(400, 300)
    assert (w, h) == (400, 300)


def test_refitting_pulls_a_registered_dialog_back_inside(app):
    """What the viewport-resize callback relies on: a dialog opened in a large
    window and then squeezed has to come back, not hang over the edge."""
    tag = "refit_probe_win"
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.window(tag=tag, label="probe", width=300, height=200, pos=[10, 10]):
        dpg.add_text("probe")
    widgets.place_dialog(tag, 3000, 3000)
    _pump()
    _assert_on_screen(tag)
    widgets.refit_dialogs()
    _pump()
    _assert_on_screen(tag)
    dpg.delete_item(tag)
    widgets.forget_dialog(tag)


# ══════════════════════════════════════════════════════════════════════════
# The main window, at the size the user is allowed to shrink it to
# ══════════════════════════════════════════════════════════════════════════


def test_the_main_window_regions_fit_the_minimum_viewport(app):
    """``test_workflows`` pins this at 1280x820. The minimum is the size that
    actually breaks things, and it is a size the user can reach by dragging."""
    _pump()
    vw, vh = _client()
    for tag in ("canvas", "timeline", "navbar", "statusbar"):
        if not dpg.does_item_exist(tag):
            continue
        try:
            x, y = dpg.get_item_rect_min(tag)
            w, h = dpg.get_item_rect_size(tag)
        except (KeyError, SystemError, TypeError):
            continue  # a child_window has no rect_min
        assert y + h <= vh + 1, f"{tag} reaches y={int(y + h)} in {vh}px"
        assert x + w <= vw + 1, f"{tag} reaches x={int(x + w)} in {vw}px"


def test_the_timeline_is_sized_from_the_client_width(app):
    """``get_viewport_width`` includes the window frame. Building the timeline
    from it put its right edge a border-width past the client area — visible
    only at startup, and only as a few clipped pixels."""
    vw, _vh = _client()
    assert app._timeline_width() <= vw


# ══════════════════════════════════════════════════════════════════════════
# The dialogs
# ══════════════════════════════════════════════════════════════════════════


def test_settings_keeps_done_on_screen(app):
    """Settings is five tabs deep and Checks alone is seven sliders and two
    buttons tall. Done used to scroll away with them."""
    app.open_settings("checks")
    _pump()
    _assert_button_inside(app.settings.tag, app.settings._t("done"))
    app.settings.hide()
    _pump()


def test_every_settings_tab_keeps_done_on_screen(app):
    from htrtdetr.gui import settings as settings_mod

    for name in settings_mod.TABS:
        app.open_settings(name.lower())
        _pump()
        _assert_button_inside(app.settings.tag, app.settings._t("done"))
    app.settings.hide()
    _pump()


def test_smart_annotate_keeps_its_action_on_screen(app):
    app.open_smart_annotate()
    _pump()
    _assert_button_inside(app.smart_dialog.tag, app.smart_dialog._t("go"))
    app.smart_dialog.hide()
    _pump()


def test_first_labels_need_no_model_or_sampling_switch(app):
    dialog = app.smart_dialog
    dialog.set_capability(predict_reason="No trained detector")
    dialog.show(video_name="new.mp4", num_frames=300, sample=False, prelabel=True)
    _pump()
    assert dialog.options()["sample"]
    assert not dialog.options()["generate_candidates"]
    assert not dialog.request().prelabelled
    assert not dpg.is_item_shown(dialog._t("sample"))
    assert not dpg.is_item_shown(dialog._t("balanced"))
    # Loading a checkpoint is optional and sits behind Advanced.
    assert dpg.get_item_parent(dialog._t("pick_model")) == dialog._t("model_setup")
    assert not dpg.get_value(dialog._t("advanced"))
    # The first-session choices must be visible without scrolling at 900x600.
    for suffix in ("scope", "target", "budget_mode", "samples", "advanced", "estimate_note"):
        _assert_button_inside(dialog.tag, dialog._t(suffix))
        assert dpg.is_item_visible(dialog._t(suffix))
    dialog.hide()


def test_completed_sampling_has_one_next_action_and_a_reachable_reset(app):
    from htrtdetr.annotation import sampling

    dialog = app.smart_dialog
    dialog.show(video_name="new.mp4", num_frames=300)
    selection = sampling.Selection(frames=[1, 20])
    dialog.finish(selection, predictions=3)
    _pump()
    assert not dpg.is_item_shown(dialog._t("form"))
    assert not dpg.is_item_shown(dialog._t("review"))
    for suffix in ("apply", "again"):
        button = dialog._t(suffix)
        _assert_button_inside(dialog.tag, button)
        bx, _ = dpg.get_item_rect_min(button)
        bw, _ = dpg.get_item_rect_size(button)
        wx, _, ww, _ = _assert_on_screen(dialog.tag)
        assert bx + bw <= wx + ww
    dialog._on_reset_clicked()
    assert dpg.is_item_shown(dialog._t("form"))
    dialog.hide()
    _pump()


def test_smart_annotate_keeps_start_annotating_on_screen(app):
    """The result phase adds four more blocks to an already-long form, and the
    button it produces is the entire point of having run the analysis."""
    app.open_smart_annotate()
    app.smart_dialog.finish(None, summary="test")
    _pump()
    _assert_button_inside(app.smart_dialog.tag, app.smart_dialog._t("apply"))
    app.smart_dialog.hide()
    _pump()


def test_the_project_entry_points_open_one_dialog(app):
    """Every entry point opens one window, whatever scope it means by it.

    There used to be a ``mode`` argument choosing between two captions for
    this window, and the second caption named a checkbox the window hides
    whenever the project has no checkpoint - which is every new project. The
    scope is the only thing an entry point now decides, and the Run button has
    to stay on screen for each of them.
    """
    for scope in ("", "project"):
        app.open_smart_annotate(scope)
        _pump()
        _assert_button_inside(app.smart_dialog.tag, app.smart_dialog._t("go"))
        assert dpg.get_item_label(app.smart_dialog.tag) == "Pick frames to label"
    app.smart_dialog.hide()
    _pump()


def test_a_finished_run_offers_the_review_it_produced(app):
    """A run that ends in "3 completed, 1 cached" is a dead end.

    The finished state has to carry the user into the review, so the count and
    the next action are the same button - and that button has to be reachable,
    which is what the rest of this file is about.
    """
    app.open_smart_annotate("project")
    app.smart_dialog.finish(None, predictions=412, issues=37)
    _pump()
    review = app.smart_dialog._t("review")
    assert dpg.is_item_shown(review)
    label = dpg.get_item_label(review)
    assert "412" in label and "37" in label, label
    _assert_button_inside(app.smart_dialog.tag, review)
    app.smart_dialog.hide()
    _pump()


def test_a_run_that_found_nothing_offers_no_review_button(app):
    """A button that does nothing when pressed is worse than no button."""
    app.open_smart_annotate("project")
    app.smart_dialog.finish(None, predictions=0, issues=0)
    _pump()
    assert not dpg.is_item_shown(app.smart_dialog._t("review"))
    assert dpg.is_item_shown(app.smart_dialog._t("apply"))
    assert dpg.get_item_label(app.smart_dialog._t("apply")) == "Close"
    app.smart_dialog.hide()
    _pump()


def test_the_train_round_dialog_fits_and_offers_start(app):
    """The round has no form, so Start is the only thing to reach - and it is
    the thing that must never be the part below the fold."""
    app.round_dialog.show(plan="312 labelled frames in 6 videos.")
    _pump()
    _assert_on_screen(app.round_dialog.tag)
    _assert_button_inside(app.round_dialog.tag, app.round_dialog._t("go"))
    assert not dpg.get_value(app.round_dialog._t("adjust")), "the knobs stay folded"
    app.round_dialog.hide()
    _pump()


def test_the_finished_round_offers_the_next_step_not_close(app):
    """A round exists to make the next pass better, so the button says so."""
    dialog = app.round_dialog
    dialog.show(plan="x")
    dialog.finish(title="Round 1 finished.", body="This round's model is now the "
                  "project's model.", where="Kept in runs/rounds/round-001")
    _pump()
    _assert_button_inside(dialog.tag, dialog._t("analyze"))
    assert dpg.is_item_shown(dialog._t("analyze"))
    assert not dpg.is_item_shown(dialog._t("adopt")), "nothing to override"
    assert not dpg.is_item_shown(dialog._t("go"))
    dialog.hide()
    _pump()


def test_a_round_that_scored_worse_offers_the_override(app):
    dialog = app.round_dialog
    dialog.show(plan="x")
    dialog.finish(title="Round 2 finished.", body="Not better.", offer_adopt=True)
    _pump()
    assert dpg.is_item_shown(dialog._t("adopt"))
    _assert_button_inside(dialog.tag, dialog._t("adopt"))
    dialog.hide()
    _pump()


def test_stopping_a_round_is_not_an_error_and_stays_fixable(app):
    """A stop is a stop: the form comes back, with no error styling."""
    dialog = app.round_dialog
    started = []
    dialog.on_start, saved = started.append, dialog.on_start
    try:
        dialog.show(plan="x")
        dialog._on_start_clicked()
        assert dialog.running and started
        assert not dpg.is_item_shown(dialog._t("go")), "Start is gone while it runs"
        assert dpg.is_item_shown(dialog._t("stop"))
        dialog.stopped("Stopped at pass 3.")
        _pump()
        assert not dialog.running
        assert dpg.is_item_shown(dialog._t("go")), "the form comes back"
        assert dpg.get_item_label(dialog._t("go")) == "Start again"
    finally:
        dialog.on_start = saved
    dialog.hide()
    _pump()


def test_a_round_without_a_project_folder_refuses_before_it_spawns(app):
    """Training writes into the project folder, so there has to be one - and
    the refusal names where to make one rather than only that it is off."""
    saved, app.project_dir = app.project_dir, None
    try:
        assert "Settings > Project" in app.train_round_blocked()
        app.round_dialog.show(plan="x")
        app.round_dialog._on_start_clicked()
        _pump()
        assert not app.round_dialog.running
        assert dpg.is_item_shown(app.round_dialog._t("go"))
    finally:
        app.project_dir = saved
    app.round_dialog.hide()
    _pump()


def test_analyze_is_locked_until_a_model_exists(app):
    """Running a detector needs a detector, and the sentence says the button
    one to the left makes one - not 'go and find a .pth'."""
    from htrtdetr.gui import widgets

    app._ensure_project().settings.pop("weights", None)
    reason = app.analyze_blocked()
    assert "Train makes one" in reason

    app.refresh_loop_gates()
    _pump()
    assert widgets.is_locked("top_analyze"), "dimmed, not dead"
    assert "Train makes one" in widgets.lock_reason("top_analyze")

    app.open_smart_annotate(preset="analyze")
    _pump()
    assert not app.smart_dialog.visible, "no window opens with nothing to run"


def test_the_gate_lifts_the_moment_a_model_exists(app, tmp_path):
    from htrtdetr.gui import widgets

    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"x")
    app._ensure_project().settings["weights"] = str(checkpoint)
    app.refresh_loop_gates()
    _pump()
    assert not widgets.is_locked("top_analyze")
    app._ensure_project().settings.pop("weights", None)
    app.refresh_loop_gates()
    _pump()
    assert widgets.is_locked("top_analyze")


def test_the_video_browser_fits(app):
    app.show_video_browser()
    _pump()
    _assert_on_screen("project_videos")
    dpg.hide_item("project_videos")


def test_the_help_window_keeps_close_on_screen(app):
    app._show_help()
    _pump()
    _assert_on_screen("help_win")
    dpg.delete_item("help_win")
    widgets.forget_dialog("help_win")


def test_the_setup_screen_keeps_create_project_on_screen(app):
    """The first window a new user ever sees, and the one with the most
    content: name, two label lists, the track switch, a folder and videos."""
    app._show_setup_screen()
    _pump()
    _assert_on_screen("setup_win")
    dpg.delete_item("setup_win")
    widgets.forget_dialog("setup_win")
    _pump()


def test_the_guided_annotation_steps_keep_their_buttons_on_screen(app):
    """Four steps, four different button rows. A wizard whose Next is off the
    bottom is a wizard with one step."""
    from htrtdetr.gui import onboarding

    for step in range(len(onboarding.STEPS)):
        onboarding.show_guide(app, step)
        _pump()
        _assert_on_screen("guided_annotation")
    dpg.delete_item("guided_annotation")
    widgets.forget_dialog("guided_annotation")


def test_the_unavailable_dialog_fits(app):
    app._show_unavailable("Fast tracking", "No tracker in this OpenCV build.")
    _pump()
    _assert_button_inside("unavailable_win", "unavailable_close")
    dpg.delete_item("unavailable_win")
    widgets.forget_dialog("unavailable_win")


# ══════════════════════════════════════════════════════════════════════════
# Activity
# ══════════════════════════════════════════════════════════════════════════


def test_the_activity_window_fits_and_keeps_close_on_screen(app):
    """The window a user opens *because* something already went wrong.

    It is the last place that may need resizing before it can be read: by the
    time it is open, the user is already looking for a reason.
    """
    app.activity.build()
    app.show_activity()
    _pump()
    _assert_on_screen("activity")
    _assert_button_inside("activity", "activity_close")
    app.activity.hide()


def test_the_activity_window_shows_what_the_status_line_said(app):
    app.activity.build()
    app._status("Could not read that checkpoint: no such file")
    app.show_activity()
    _pump()
    body = dpg.get_value("activity_messages")
    assert "Could not read that checkpoint" in body
    assert "! " in body, "an error must be marked, not only coloured"
    app.activity.hide()


def test_the_problems_filter_hides_the_chatter(app):
    app.activity.build()
    app.activity.clear()
    app._status("Frame 12/160  |  Annotated: 0/160 (0%)")
    app._status("Could not open that video")
    app.show_activity()
    _pump()
    app.activity.set_filter("problems")
    _pump()
    body = dpg.get_value("activity_messages")
    assert "Could not open that video" in body
    assert "Frame 12/160" not in body
    app.activity.set_filter("all")
    app.activity.hide()


def test_the_pill_reports_the_error_count(app):
    app.activity.clear()
    app._status("Could not do the thing")
    _pump()
    assert dpg.does_item_exist("top_activity")
    assert "1 error" in dpg.get_item_label("top_activity")
    app.activity.clear()
    app._refresh_activity_pill()
    assert dpg.get_item_label("top_activity") == "Activity"


def test_the_activity_window_toggles_rather_than_stacking(app):
    app.activity.build()
    app.toggle_activity()
    _pump()
    assert app.activity.visible
    app.toggle_activity()
    _pump()
    assert not app.activity.visible


def test_the_activity_window_shows_a_run_log_off_disk(app, tmp_path):
    """The traceback that killed a round is in a file, not in this process.

    Training and inference both write to ``stdout=<file>`` rather than a pipe
    (a pipe nobody drains blocks the child once the OS buffer fills), so the
    only complete record of a failed run is on disk. This pane is the way to
    it without a file browser.
    """
    folder = tmp_path / "runs" / "rounds" / "round-004" / "stage1"
    folder.mkdir(parents=True)
    (folder / "round.log").write_text(
        "Epoch 1/40\nRuntimeError: CUDA out of memory\n", encoding="utf-8"
    )
    previous = app.project_dir
    app.project_dir = tmp_path
    try:
        app.activity.build()
        app.show_activity()
        _pump()
        assert "round-004" in dpg.get_value("activity_logs")
        assert "CUDA out of memory" in dpg.get_value("activity_runlog")
    finally:
        app.activity.hide()
        app.project_dir = previous
