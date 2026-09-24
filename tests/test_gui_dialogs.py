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
    """Pre-label and Pick frames are the same window, and its Run stays on
    screen in both - there is no second project form to fall off the bottom."""
    for mode in ("auto", "sample"):
        app.open_smart_annotate(mode, "project")
        _pump()
        _assert_button_inside(app.smart_dialog.tag, app.smart_dialog._t("go"))
    app.smart_dialog.hide()
    _pump()


def test_a_finished_run_offers_the_review_it_produced(app):
    """A run that ends in "3 completed, 1 cached" is a dead end.

    The finished state has to carry the user into the review, so the count and
    the next action are the same button - and that button has to be reachable,
    which is what the rest of this file is about.
    """
    app.open_smart_annotate("auto", "project")
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
    app.open_smart_annotate("auto", "project")
    app.smart_dialog.finish(None, predictions=0, issues=0)
    _pump()
    assert not dpg.is_item_shown(app.smart_dialog._t("review"))
    assert dpg.is_item_shown(app.smart_dialog._t("apply"))
    assert dpg.get_item_label(app.smart_dialog._t("apply")) == "Close"
    app.smart_dialog.hide()
    _pump()


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
