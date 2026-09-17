"""Canvas gestures in a real viewport — the paths a mouse actually takes.

These cover four failures that every unit test in the suite passed straight
through, because each one lives in the join between a handler and the renderer
rather than inside either:

* the Pan tool stopped after a single mouse delta, because ``_render_canvas``
  cleared ``_mode``;
* moving, resizing or rotating a box recorded neither an undo step nor a dirty
  mark, so Ctrl+Z undid the *previous* edit and the autosave never fired;
* ``propagate_from_prev`` pushed an undo entry before its guards, so a stray
  Space on frame 0 dirtied a clean project and cleared the redo stack;
* every frame step seeked the capture, and every toolbar label wider than its
  button was silently clipped.
"""

from __future__ import annotations

import pytest

dpg = pytest.importorskip("dearpygui.dearpygui")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from htrtdetr.annotation.project import ProjectSession  # noqa: E402
from htrtdetr.annotator import AnnotationApp  # noqa: E402
from htrtdetr.gui import shortcuts, widgets  # noqa: E402

pytestmark = pytest.mark.gui

FRAME_W, FRAME_H, FRAMES = 96, 64, 24


@pytest.fixture(scope="module")
def viewport():
    app = AnnotationApp()
    dpg.create_viewport(title="YOAKE canvas smoke tests", width=1280, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    app.class_names = ["fly"]
    app.action_names = ["walking", "resting"]
    app._build_main_window()
    yield app
    if app.cap is not None:
        app.cap.release()
    app._stop_sam()
    dpg.destroy_context()


@pytest.fixture
def app(viewport, tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (FRAME_W, FRAME_H)
    )
    assert writer.isOpened()
    for index in range(FRAMES):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10 + index, 10), (30 + index, 30), (200, 200, 200), -1)
        writer.write(frame)
    writer.release()

    project = ProjectSession("canvas", ["fly"], ["walking", "resting"], directory=tmp_path)
    project.settings["onboarding_done"] = True
    viewport._adopt_project(project)
    viewport._add_videos([path])
    for _ in range(2):
        dpg.render_dearpygui_frame()
    viewport.goto_frame(0)
    viewport.set_tool("select")
    viewport.deselect(render=False)
    viewport._undo.clear()
    viewport._redo.clear()
    viewport._dirty = False
    return viewport


def _scripted_mouse(monkeypatch, app, image_points):
    """Feed ``_get_canvas_mouse`` a fixed path, in image coordinates.

    The handlers read the real cursor, which a test cannot move, so the one
    function that asks for it is replaced and everything downstream — the
    hit tests, the pan arithmetic, the renderer — runs for real.
    """
    points = [app._i2c(x, y) for x, y in image_points]
    state = {"i": 0}

    def next_point():
        index = min(state["i"], len(points) - 1)
        state["i"] += 1
        return points[index]

    monkeypatch.setattr(app, "_get_canvas_mouse", next_point)
    monkeypatch.setattr(app, "_is_over_canvas", lambda: True)
    monkeypatch.setattr(app, "_begin_scrub", lambda: False)


# ══════════════════════════════════════════════════════════════════════════
# The Pan tool
# ══════════════════════════════════════════════════════════════════════════

def test_pan_keeps_panning_for_the_whole_drag(app, monkeypatch):
    """Every drag event must move the view, not only the first.

    ``_render_canvas`` used to clear ``_mode`` whenever the Pan tool was armed,
    and ``_drag_pan`` renders — so the first event panned, dropped the mode and
    the press origin, and every later event returned at ``_mode == "idle"``.
    """
    app.set_tool("pan")
    app.zoom_by(4.0)  # zoomed in, so there is room to pan into
    app._pan_x = app._pan_y = 0.0
    _scripted_mouse(monkeypatch, app, [(40, 30), (44, 30), (52, 30)])

    app._on_mouse_press(None, None)
    assert app._mode == "pan" and app._pan_origin is not None

    app._on_mouse_drag(None, None)
    after_first = app._pan_x
    assert after_first != 0.0, "the first drag event did not pan at all"
    assert app._mode == "pan", "rendering retired the pan mode mid-drag"

    app._on_mouse_drag(None, None)
    assert app._pan_x != after_first, "the second drag event was ignored"

    app._on_mouse_release(None, None)
    assert app._mode == "idle" and app._pan_origin is None
    app.set_tool("select")
    app.fit_view()


# ══════════════════════════════════════════════════════════════════════════
# Undo and dirtiness for a dragged box
# ══════════════════════════════════════════════════════════════════════════

def test_moving_a_box_is_one_undo_step_and_marks_the_project_dirty(app, monkeypatch):
    obj = app._create_object([10, 10, 30, 30])
    app.deselect(render=False)
    app._undo.clear()
    app._redo.clear()
    app._dirty = False
    _scripted_mouse(monkeypatch, app, [(20, 20), (26, 24)])

    app._on_mouse_press(None, None)
    assert app._mode == "move" and app._pending_snap is not None
    app._on_mouse_drag(None, None)
    moved = list(obj.bbox)
    assert moved != [10, 10, 30, 30]
    app._on_mouse_release(None, None)

    assert len(app._undo) == 1, "a move left no undo step behind"
    assert app._dirty, "a move did not mark the project dirty, so it never autosaved"

    app.undo()
    assert app.annotations[app.current_frame_idx][0].bbox == [10, 10, 30, 30]


def test_a_click_that_only_selects_leaves_no_undo_step(app, monkeypatch):
    """P7: snapshot at press, push at release **only if it moved**."""
    app._create_object([10, 10, 30, 30])
    app.deselect(render=False)
    app._undo.clear()
    app._dirty = False
    _scripted_mouse(monkeypatch, app, [(20, 20)])

    app._on_mouse_press(None, None)
    app._on_mouse_release(None, None)

    assert app._undo == [], "selecting a box pushed an undo step"
    assert not app._dirty, "selecting a box marked the project dirty"


def test_resizing_a_box_is_undoable(app, monkeypatch):
    obj = app._create_object([10, 10, 30, 30])
    app._select(obj)
    app._undo.clear()
    app._dirty = False
    # Press on the bottom-right handle, in canvas pixels, then drag outward.
    _scripted_mouse(monkeypatch, app, [(30, 30), (40, 38)])

    app._on_mouse_press(None, None)
    assert app._mode == "resize", "the press did not land on a resize handle"
    app._on_mouse_drag(None, None)
    assert obj.bbox != [10, 10, 30, 30]
    app._on_mouse_release(None, None)

    assert len(app._undo) == 1 and app._dirty
    app.undo()
    assert app.annotations[app.current_frame_idx][0].bbox == [10, 10, 30, 30]


# ══════════════════════════════════════════════════════════════════════════
# Guards that must not spend an undo step
# ══════════════════════════════════════════════════════════════════════════

def test_propagate_on_the_first_frame_changes_nothing_at_all(app):
    """Space is a bare key and frame 0 has no previous frame."""
    app.goto_frame(0)
    sentinel = {"video_id": "sentinel", "frame": 0, "frames": {}}
    app._redo.append(sentinel)
    app._dirty = False

    app.propagate_from_prev()

    assert app._undo == [], "a refused propagation still pushed an undo step"
    assert app._redo == [sentinel], "a refused propagation cleared the redo stack"
    assert not app._dirty, "a refused propagation marked the project dirty"


def test_propagate_onto_a_frame_that_already_has_every_track_is_refused(app):
    app.goto_frame(0)
    app._create_object([10, 10, 30, 30])
    app.goto_frame(1)
    app.propagate_from_prev()
    assert len(app.annotations[1]) == 1

    app._undo.clear()
    app._dirty = False
    app.propagate_from_prev()

    assert len(app.annotations[1]) == 1, "the boxes were propagated twice"
    assert app._undo == [] and not app._dirty


def test_converting_a_box_that_is_already_that_shape_is_refused(app):
    obj = app._create_object([10, 10, 30, 30])
    app._select(obj)
    app._undo.clear()
    app._dirty = False

    app.convert_selected_to_aabb()  # it is already axis-aligned

    assert app._undo == [] and not app._dirty


# ══════════════════════════════════════════════════════════════════════════
# Frame stepping
# ══════════════════════════════════════════════════════════════════════════

class _CountingCapture:
    """A capture that reports how often it was asked to seek."""

    def __init__(self, cap):
        self._cap = cap
        self.seeks = 0

    def set(self, prop, value):
        self.seeks += 1
        return self._cap.set(prop, value)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def test_stepping_forward_decodes_sequentially_instead_of_seeking(app, monkeypatch):
    """Next-frame is the most-used gesture; it must not re-decode a GOP."""
    counting = _CountingCapture(app.cap)
    monkeypatch.setattr(app, "cap", counting)
    app._decoded_idx = None

    assert app._read_frame_bgr(0) is not None
    first_seeks = counting.seeks
    assert first_seeks == 1, "the first read has nowhere to continue from"

    for index in range(1, 6):
        assert app._read_frame_bgr(index) is not None
    assert counting.seeks == first_seeks, "stepping forward one frame still seeked"

    # Backwards is genuinely a jump and still takes the seek.
    assert app._read_frame_bgr(2) is not None
    assert counting.seeks == first_seeks + 1
    app._decoded_idx = None


# ══════════════════════════════════════════════════════════════════════════
# Nothing in the chrome may clip its own label
# ══════════════════════════════════════════════════════════════════════════

def _chrome_buttons():
    import htrtdetr.annotator as module

    tags = [f"rail_{tool}" for tool, *_rest in module._TOOLS]
    tags += ["rail_propagate", "rail_sticky"]
    tags += [f"rail_{a.replace('.', '_')}" for a in
             ("view.zoom_in", "view.zoom_out", "view.fit")]
    tags += ["top_smart", "top_review", "top_undo", "top_redo", "top_save",
             "top_panels", "top_settings", "nav_propagate"]
    return tags


def test_no_toolbar_button_is_narrower_than_its_own_label(app):
    """ImGui clips a too-long label at the frame edge, silently.

    ``Undo (Ctrl+Z)`` needs 111 px and was given 90, so the accelerator the
    button existed to teach was exactly the part cut off; the whole tool rail
    was 44 px wide and every label in it was 49 px.
    """
    offenders = []
    for tag in _chrome_buttons():
        if not dpg.does_item_exist(tag):
            continue
        label = dpg.get_item_label(tag) or ""
        width = int(dpg.get_item_configuration(tag).get("width") or 0)
        if width <= 0:
            continue
        needed = dpg.get_text_size(label)[0] + 2 * widgets._FRAME_PAD_X
        if needed > width:
            offenders.append((tag, label, width, needed))
    assert not offenders, f"clipped labels: {offenders}"


def test_the_rail_shows_the_key_for_every_tool(app):
    """The rail is where the keymap is learned, so the key has to be on it."""
    import htrtdetr.annotator as module

    for tool, _name, _word, action in module._TOOLS:
        key = shortcuts.label_for(action)
        label = dpg.get_item_label(f"rail_{tool}") or ""
        assert key and key in label, f"{tool} does not show its key ({key!r} not in {label!r})"


#: The viewport's declared ``min_width``. Everything in the chrome has to be
#: reachable at this size, because the user is allowed to be here.
MIN_VIEWPORT_W = 900


def test_the_top_bar_fits_inside_the_smallest_allowed_window(app):
    """Fixed widths pushed Panels and Settings off the right edge at 900 px.

    They were not clipped, they were **unreachable**: a horizontal group does
    not wrap, so the last two buttons of the row simply had no pixels. Sizing
    each button to its own label is what brings the row back inside the
    smallest window the viewport allows.
    """
    tags = ["top_smart", "top_review", "top_undo", "top_redo", "top_save",
            "top_panels", "top_settings"]
    widths = []
    for tag in tags:
        assert dpg.does_item_exist(tag), tag
        widths.append(dpg.get_item_rect_size(tag)[0])
    # The row also carries the YOAKE wordmark, the save pill and two spacers.
    chrome = 40 + 90 + 26
    total = sum(widths) + 8 * len(widths) + chrome
    assert total <= MIN_VIEWPORT_W, (
        f"the top bar needs {total:.0f} px and the window can be "
        f"{MIN_VIEWPORT_W} px wide: {list(zip(tags, widths))}"
    )


def test_the_tool_rail_can_be_scrolled_to_its_last_control(app):
    """At 600 px tall the rail is taller than the canvas.

    With ``no_scrollbar=True`` the zoom controls and the zoom readout below the
    fold did not exist for the user and nothing said why.
    """
    assert dpg.get_item_configuration("toolrail").get("no_scrollbar") is False
    for tag in ("rail_view_zoom_in", "rail_view_zoom_out", "rail_view_fit", "rail_zoom"):
        assert dpg.does_item_exist(tag), f"{tag} is not in the rail at all"
