"""
tests/test_gui_click_segment.py — the click-to-box engine, headless.

No display, no dearpygui, no torch. Every test here pins a property some
consumer depends on; the geometry tests use synthetic shapes whose true angle
and size are known exactly, so a regression shows up as a number, not a guess.

Design: ``docs/CLICK_SEGMENT.md``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from htrtdetr.gui import click_segment as cs

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = REPO_ROOT / "src" / "htrtdetr" / "gui" / "click_segment.py"

_HAS_CV2 = importlib.util.find_spec("cv2") is not None
needs_cv2 = pytest.mark.skipif(not _HAS_CV2, reason="OpenCV is not installed")

# Held at import time, on purpose. ``test_gui_auto_label.py`` pops torch out of
# ``sys.modules`` to prove auto_label never imports it, and torch cannot be
# imported a second time in one process — it re-registers its "triton" library
# and raises. A reference captured now stays valid after that pop; an import
# inside the test would explode depending on file order.
try:
    import torch as _torch
    from htrtdetr.geometry import obb as _geometry_obb
except Exception:                                    # noqa: BLE001
    _torch = None
    _geometry_obb = None


# --------------------------------------------------------------------------- #
# Helpers — synthetic animals with known ground truth
# --------------------------------------------------------------------------- #

def _ellipse_mask(size=200, centre=(100, 100), axes=(30, 8), angle_deg=25.0):
    """A filled ellipse: a fly body with no legs, whose axis we know exactly."""
    import cv2
    m = np.zeros((size, size), np.uint8)
    cv2.ellipse(m, centre, axes, angle_deg, 0, 360, 255, -1)
    return m


def _add_legs(mask, centre=(100, 100), angle_deg=25.0, n=6, length=20):
    """Thin lines radiating from the body — legs, antennae, a wing edge.

    They are 1 px wide, which is what makes them removable: the body-cut
    threshold is a fraction of the *body's* half-thickness, so anything this
    thin disappears without tuning a pixel count.
    """
    import cv2
    out = mask.copy()
    for i in range(n):
        a = math.radians(angle_deg + 30 + i * (300 / max(n - 1, 1)))
        x = int(centre[0] + length * math.cos(a))
        y = int(centre[1] + length * math.sin(a))
        cv2.line(out, centre, (x, y), 255, 1)
    return out


def _frame_with(mask, dark_object=True, bg=210, fg=40):
    """Turn a mask into a 3-channel frame: dark animal on a lit arena, or the
    reverse. Both must work from the same click, with nothing to configure."""
    base = np.full(mask.shape, bg if dark_object else fg, np.uint8)
    base[mask > 0] = fg if dark_object else bg
    return np.dstack([base, base, base])


def _angle_diff(a, b):
    """Smallest difference between two axes, which live mod pi."""
    d = (float(a) - float(b)) % math.pi
    return min(d, math.pi - d)


# --------------------------------------------------------------------------- #
# Module invariants
# --------------------------------------------------------------------------- #

def test_module_never_imports_dearpygui_or_torch():
    """The engine must stay headless and torch-free forever.

    An AST scan, not a substring search: the docstring legitimately names both.
    """
    imported = set()
    for node in ast.walk(ast.parse(MODULE_PATH.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "dearpygui" not in imported
    assert "torch" not in imported
    assert "subprocess" not in imported     # argv is data; the caller spawns


def test_module_scope_imports_are_stdlib_only():
    """Importing this module must not require OpenCV.

    The GUI imports it at startup to build the toolbar. If cv2 were imported at
    module scope, an OpenCV-less machine would lose the whole annotator instead
    of one tool.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.add(node.module.split(".")[0])
    assert top <= {"__future__", "math", "dataclasses", "typing"}


def test_min_body_px_agrees_with_the_annotator():
    """A box this module produces must be one the annotator will accept."""
    src = (REPO_ROOT / "src" / "htrtdetr" / "annotator.py").read_text(encoding="utf-8")
    for line in src.splitlines():
        if line.startswith("_MIN_BOX_PX"):
            assert int(line.split("=")[1]) == cs.MIN_BODY_PX
            break
    else:
        pytest.fail("annotator._MIN_BOX_PX not found")


# --------------------------------------------------------------------------- #
# Angle canonicalisation
# --------------------------------------------------------------------------- #

def test_normalize_angle_folds_into_the_half_open_range():
    for deg in range(-720, 721, 7):
        out = cs.normalize_angle(math.radians(deg))
        assert -math.pi / 2 <= out < math.pi / 2


def test_normalize_angle_preserves_the_axis():
    """Folding by pi may not move the line the angle names."""
    for deg in range(-720, 721, 11):
        th = math.radians(deg)
        assert _angle_diff(cs.normalize_angle(th), th) < 1e-9


def test_normalize_angle_matches_geometry_obb():
    """Two implementations of one rule; a test is what keeps them one rule.

    ``click_segment`` may never import torch, so it re-derives the fold that
    ``geometry/obb.py`` performs in tensors. This is what stops the two copies
    from drifting apart, boundary cases included.
    """
    if _torch is None or _geometry_obb is None:
        pytest.skip("torch is not installed")
    # Torch's C extension looks itself up in sys.modules to build a tensor, so
    # a test that popped it and did not put it back would break this one from a
    # distance. Repair it rather than fail on someone else's leak.
    sys.modules.setdefault("torch", _torch)

    angles = [math.radians(d) for d in range(-360, 361, 3)]
    angles += [-math.pi / 2, math.pi / 2, math.pi, -math.pi, 0.0]
    ours = [cs.normalize_angle(a) for a in angles]
    theirs = _geometry_obb.normalize_angle(
        _torch.tensor(angles, dtype=_torch.float64)).tolist()
    for a, b in zip(ours, theirs):
        assert abs(a - b) < 1e-9


# --------------------------------------------------------------------------- #
# mask -> OBB
# --------------------------------------------------------------------------- #

@needs_cv2
@pytest.mark.parametrize("angle_deg", [0.0, 25.0, 60.0, 90.0, 135.0, 170.0])
def test_mask_to_obb_recovers_a_known_angle(angle_deg):
    mask = _ellipse_mask(angle_deg=angle_deg)
    obb, why = cs.mask_to_obb(mask, (100, 100))
    assert why == "" and obb is not None
    assert _angle_diff(obb[4], math.radians(angle_deg)) < math.radians(2.0)


@needs_cv2
def test_mask_to_obb_recovers_a_known_size():
    """Axes (30, 8) rasterises to a 61 x 17 px body. Within 8%, or the boxes
    are wrong in a way that would reach the training set."""
    obb, why = cs.mask_to_obb(_ellipse_mask(axes=(30, 8), angle_deg=25.0), (100, 100))
    assert why == ""
    assert obb[2] == pytest.approx(61.0, rel=0.08)
    assert obb[3] == pytest.approx(17.0, rel=0.08)
    assert obb[0] == pytest.approx(100.0, abs=1.5)
    assert obb[1] == pytest.approx(100.0, abs=1.5)


@needs_cv2
def test_mask_to_obb_ignores_appendages():
    """The measured bug this whole module exists for.

    minAreaRect on a legged mask inflates the box and twists its axis. The
    fitted body must stay within 15% of the clean-ellipse fit on both sides,
    and the axis must not move more than 2 degrees.
    """
    import cv2

    clean = _ellipse_mask(axes=(30, 8), angle_deg=25.0)
    legged = _add_legs(clean, angle_deg=25.0)

    ref, _ = cs.mask_to_obb(clean, (100, 100))
    got, why = cs.mask_to_obb(legged, (100, 100))
    assert why == "" and got is not None

    assert got[2] == pytest.approx(ref[2], rel=0.15)
    assert got[3] == pytest.approx(ref[3], rel=0.15)
    assert _angle_diff(got[4], ref[4]) < math.radians(2.0)

    # And show that the naive reduction really does fail here, so this test
    # cannot quietly start passing for the wrong reason: minAreaRect wraps the
    # legs, which both inflates the box and tilts it off the body axis.
    (_c, (rw, rh), _a) = cv2.minAreaRect(cv2.findNonZero(legged))
    assert rw * rh > 1.4 * (ref[2] * ref[3])


@needs_cv2
def test_mask_to_obb_puts_the_body_axis_in_w():
    """`w` is the body axis, not "the longer side" — the convention the
    annotator and the training set both read."""
    obb, why = cs.mask_to_obb(_ellipse_mask(axes=(30, 8), angle_deg=25.0), (100, 100))
    assert why == ""
    assert obb[2] > obb[3]
    assert -math.pi / 2 <= obb[4] < math.pi / 2


@needs_cv2
def test_mask_to_obb_accepts_a_click_a_pixel_off_the_animal():
    """Thresholding is not exact at the edge; a click on the rim must still work."""
    mask = _ellipse_mask(axes=(30, 8), angle_deg=0.0)
    obb, why = cs.mask_to_obb(mask, (100, 108))     # just outside the minor axis
    assert why == "" and obb is not None


@needs_cv2
def test_mask_to_obb_reports_nothing_found_on_an_empty_mask():
    obb, why = cs.mask_to_obb(np.zeros((80, 80), np.uint8), (40, 40))
    assert obb is None and why == cs.NOTHING_FOUND


@needs_cv2
def test_mask_to_obb_reports_nothing_found_when_the_click_is_far_away():
    mask = _ellipse_mask(size=200, centre=(30, 30), axes=(8, 5))
    obb, why = cs.mask_to_obb(mask, (180, 180))
    assert obb is None and why == cs.NOTHING_FOUND


@needs_cv2
def test_mask_to_obb_reports_touches_edge():
    """A blob running off the window is expandable, not broken — the caller
    needs that told apart from a genuinely huge blob."""
    import cv2
    m = np.zeros((80, 80), np.uint8)
    cv2.rectangle(m, (0, 30), (50, 45), 255, -1)
    obb, why = cs.mask_to_obb(m, (25, 37))
    assert obb is None and why == cs.TOUCHES_EDGE


@needs_cv2
def test_mask_to_obb_reports_merged_background():
    import cv2
    m = np.zeros((80, 80), np.uint8)
    cv2.rectangle(m, (5, 5), (74, 74), 255, -1)     # fills the window, off the edge
    m[0, :] = 0
    obb, why = cs.mask_to_obb(m, (40, 40))
    assert obb is None and why == cs.MERGED_BACKGROUND


@needs_cv2
def test_mask_to_obb_reports_two_animals():
    """Two bodies meeting nose to nose are one blob with two cores of similar
    size, and refusing is the right answer: a single box over both would be
    silently wrong in the training set.

    This guard fires when the cores separate. Two animals lying flank to flank
    merge into one thick blob with no waist at all, and the local backend does
    fit one box over both — the limitation ``docs/CLICK_SEGMENT.md`` §5 names,
    and one of the reasons the SAM backend exists.
    """
    import cv2
    m = np.zeros((240, 240), np.uint8)
    cv2.ellipse(m, (90, 120), (30, 8), 0.0, 0, 360, 255, -1)
    cv2.ellipse(m, (150, 120), (30, 8), 0.0, 0, 360, 255, -1)
    obb, why = cs.mask_to_obb(m, (120, 120))
    assert obb is None and why == cs.TWO_ANIMALS


@needs_cv2
def test_mask_to_obb_keeps_one_animal_with_a_speck_beside_it():
    """A small satellite is dirt, not a second animal: the ratio decides, not
    the count, or every dust mote would block a click."""
    import cv2
    mask = _ellipse_mask(axes=(30, 8), angle_deg=0.0)
    cv2.circle(mask, (100, 88), 2, 255, -1)
    obb, why = cs.mask_to_obb(mask, (100, 100))
    assert why == "" and obb is not None


@needs_cv2
@pytest.mark.parametrize("angle_deg", [0.0, 25.0, 100.0])
def test_reduce_mask_is_what_mask_to_obb_returns(angle_deg):
    """``analysis/silhouette.py`` measures from the same reduction the click uses,
    so the two can never disagree about where an animal's axis is."""
    mask = _add_legs(_ellipse_mask(angle_deg=angle_deg), angle_deg=angle_deg)
    obb, why = cs.mask_to_obb(mask, (100, 100))
    parts, why_parts = cs.reduce_mask(mask, (100, 100))
    assert why == why_parts == "" and parts is not None
    assert parts.obb == obb
    assert parts.blob.shape == parts.core.shape == parts.body.shape == mask.shape
    assert parts.body.sum() <= parts.blob.sum()          # the legs are gone from the body
    assert math.hypot(*parts.axis) == pytest.approx(1.0)
    assert not parts.twin_core


@needs_cv2
def test_reduce_mask_can_measure_two_animals_instead_of_refusing():
    """The click refuses joined animals (a box over both would poison a training
    set); a measurement may instead keep the nearer one and say there were two."""
    import cv2
    m = np.zeros((240, 240), np.uint8)
    cv2.ellipse(m, (90, 120), (30, 8), 0.0, 0, 360, 255, -1)
    cv2.ellipse(m, (150, 120), (30, 8), 0.0, 0, 360, 255, -1)
    assert cs.reduce_mask(m, (120, 120)) == (None, cs.TWO_ANIMALS)
    parts, why = cs.reduce_mask(m, (100, 120), allow_twins=True)
    assert why == "" and parts is not None and parts.twin_core
    assert parts.centre[0] < 120                         # the axis is the nearer core's
    assert _angle_diff(parts.obb[4], 0.0) < math.radians(2)
    assert parts.obb[2] > 100                            # the extent is the pair's


@needs_cv2
def test_threshold_gray_is_the_bgr_path_without_the_conversion():
    import cv2
    frame = _frame_with(_ellipse_mask())
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(cs.threshold_gray(gray, (100, 100)), cs._otsu_mask(frame, (100, 100)))


# --------------------------------------------------------------------------- #
# The local backend, end to end
# --------------------------------------------------------------------------- #

@needs_cv2
@pytest.mark.parametrize("dark_object", [True, False])
def test_segment_local_fits_a_synthetic_animal(dark_object):
    """One gesture, both polarities, nothing for the user to configure."""
    frame = _frame_with(_ellipse_mask(axes=(30, 8), angle_deg=40.0),
                        dark_object=dark_object)
    res = cs.segment_local(frame, cs.ClickRequest(frame_index=0, x=100, y=100))
    assert res.ok, res.detail
    assert res.backend == "local"
    assert _angle_diff(res.obb[4], math.radians(40.0)) < math.radians(3.0)
    assert res.obb[2] == pytest.approx(60.0, rel=0.15)
    x1, y1, x2, y2 = res.bbox
    assert x1 < 100 < x2 and y1 < 100 < y2


@needs_cv2
def test_segment_local_widens_the_window_for_a_large_animal():
    """A body longer than the default window must not need a radius hint: the
    window doubles while the blob still runs off its edge."""
    frame = _frame_with(_ellipse_mask(size=400, centre=(200, 200),
                                      axes=(90, 22), angle_deg=10.0))
    res = cs.segment_local(frame, cs.ClickRequest(frame_index=0, x=200, y=200))
    assert res.ok, res.detail
    assert res.obb[2] == pytest.approx(180.0, rel=0.20)


@needs_cv2
def test_segment_local_honours_an_explicit_roi():
    """Shift+drag pins the window instead of guessing it."""
    frame = _frame_with(_ellipse_mask(axes=(30, 8), angle_deg=0.0))
    res = cs.segment_local(frame, cs.ClickRequest(
        frame_index=0, x=100, y=100, roi=(60, 80, 140, 120)))
    assert res.ok, res.detail


@needs_cv2
def test_segment_local_reports_instead_of_guessing():
    """An empty arena must produce no box at all. A plausible-looking wrong box
    is worse than nothing: it reaches the training set unnoticed."""
    frame = np.full((200, 200, 3), 210, np.uint8)
    res = cs.segment_local(frame, cs.ClickRequest(frame_index=0, x=100, y=100))
    assert not res.ok
    assert res.obb is None and res.bbox is None
    assert res.detail in cs.DIAGNOSES.values()


def test_segment_local_handles_no_frame():
    res = cs.segment_local(None, cs.ClickRequest(frame_index=0, x=1, y=1))
    assert not res.ok and res.diagnosis == cs.NO_FRAME


@needs_cv2
def test_segment_local_handles_a_click_outside_the_frame():
    frame = np.full((50, 50, 3), 200, np.uint8)
    res = cs.segment_local(frame, cs.ClickRequest(frame_index=0, x=99, y=10))
    assert not res.ok and res.diagnosis == cs.OUT_OF_FRAME


@needs_cv2
def test_segment_local_never_raises_on_hostile_input():
    """Every failure is a result, never an exception — the annotator calls this
    straight from a mouse handler, where a traceback would kill the tool."""
    frame = _frame_with(_ellipse_mask())
    for req in (
        cs.ClickRequest(frame_index=0, x=0, y=0),
        cs.ClickRequest(frame_index=0, x=199, y=199),
        cs.ClickRequest(frame_index=0, x=100, y=100, radius_hint=1),
        cs.ClickRequest(frame_index=0, x=100, y=100, radius_hint=10_000),
        cs.ClickRequest(frame_index=0, x=100, y=100, roi=(0, 0, 0, 0)),
        cs.ClickRequest(frame_index=0, x=100, y=100, roi=(150, 150, 20, 20)),
    ):
        assert isinstance(cs.segment_local(frame, req), cs.ClickResult)


@needs_cv2
def test_segment_is_deterministic():
    frame = _frame_with(_ellipse_mask(angle_deg=33.0))
    req = cs.ClickRequest(frame_index=0, x=100, y=100)
    assert cs.segment(frame, req).obb == cs.segment(frame, req).obb


# --------------------------------------------------------------------------- #
# Backend registry
# --------------------------------------------------------------------------- #

def test_local_is_always_available():
    """The property the UI rests on: the tool button is never locked, because
    the default backend needs nothing installed."""
    assert "local" in cs.available_backends()


@needs_cv2
def test_unknown_backend_falls_back_to_local():
    frame = _frame_with(_ellipse_mask())
    res = cs.segment(frame, cs.ClickRequest(frame_index=0, x=100, y=100),
                     backend="sam")       # not built yet
    assert res.backend == "local"


# --------------------------------------------------------------------------- #
# Copy (GUI_DESIGN §3.2, §4.2, P5)
# --------------------------------------------------------------------------- #

def test_every_diagnosis_has_a_sentence():
    for name in (cs.NOTHING_FOUND, cs.MERGED_BACKGROUND, cs.TWO_ANIMALS,
                 cs.TOUCHES_EDGE, cs.OUT_OF_FRAME, cs.NO_FRAME):
        assert cs.DIAGNOSES[name].endswith(".")


def test_recoverable_diagnoses_name_the_escape_hatch():
    """A failure that only states the cause is a dead end. Each recoverable one
    must name what to do instead, right now."""
    for name in (cs.NOTHING_FOUND, cs.MERGED_BACKGROUND, cs.TWO_ANIMALS,
                 cs.TOUCHES_EDGE):
        assert "draw the box by hand with W" in cs.DIAGNOSES[name]


# --------------------------------------------------------------------------- #
# The annotator wiring (no window is ever created)
# --------------------------------------------------------------------------- #

@pytest.fixture
def app():
    """A constructed annotator with the dpg-touching methods stubbed out.

    ``AnnotationApp.__init__`` only sets state, so the click path can be driven
    headlessly. Only the three methods that would talk to a window are replaced.
    """
    dpg = pytest.importorskip("dearpygui.dearpygui")
    from htrtdetr.annotator import AnnotationApp
    from htrtdetr.gui import theme

    a = AnnotationApp()
    # Most tests exercise the deterministic, dependency-free fitter. Automatic
    # helper startup is covered separately with a controlled completion below.
    a._click_backend = "local"
    a.frame_w, a.frame_h = 200, 200
    a.current_frame_idx = 0
    a.status_messages = []
    a._status = a.status_messages.append
    a._render_canvas = lambda: None
    a._refresh_sidebar = lambda: None
    a._refresh_tool_ui = lambda: None
    yield a
    # ``AnnotationApp.__init__`` calls ``dpg.create_context()``. Seventy of
    # those in one module and never a ``destroy_context`` is a leak the module
    # itself survives - it never opens a window - but the next GUI module in
    # the same pytest process does not: by then dearpygui is deep enough in
    # stale contexts that a modal window opened with ``show=True`` is closed
    # again on its first rendered frame, and the failure lands in
    # ``test_gui_dialogs.py`` with nothing to say it came from here.
    dpg.destroy_context()
    theme.reset_cache()


@needs_cv2
def test_annotator_click_creates_a_rotated_box(app):
    app._current_bgr = _frame_with(_ellipse_mask(axes=(30, 8), angle_deg=35.0))
    app._click_to_box(100, 100)

    objects = app.annotations[0]
    assert len(objects) == 1
    obj = objects[0]
    assert obj.has_obb
    assert _angle_diff(obj.obb[4], math.radians(35.0)) < math.radians(3.0)
    # bbox is the OBB's bounding box, and the click is inside it
    x1, y1, x2, y2 = obj.bbox
    assert x1 < 100 < x2 and y1 < 100 < y2
    # The box is selected, so Delete removes it straight away — which is what
    # makes creating it accepted rather than pending defensible (§8.2).
    assert app._selected is obj


@needs_cv2
def test_annotator_click_records_provenance(app):
    """The box is a human's choice of target and a machine's choice of
    rectangle. It reaches disk, so it has to say where it came from."""
    app._current_bgr = _frame_with(_ellipse_mask())
    app._click_to_box(100, 100)

    obj = app.annotations[0][0]
    assert obj.source == "wand"
    assert obj.to_dict()["source"] == "wand"


def test_a_hand_drawn_box_writes_no_provenance_key():
    """A human-only project's annotations.json must stay byte-identical."""
    from htrtdetr.annotator import AnnObject

    assert "source" not in AnnObject(bbox=[1, 2, 3, 4]).to_dict()
    assert AnnObject.from_dict({"bbox": [1, 2, 3, 4]}).source == "human"
    assert AnnObject.from_dict({"bbox": [1, 2, 3, 4], "source": "wand"}).source == "wand"


@needs_cv2
def test_annotator_click_on_empty_arena_creates_nothing(app):
    """A miss must leave the frame untouched and say why."""
    app._current_bgr = np.full((200, 200, 3), 210, np.uint8)
    app._click_to_box(100, 100)

    assert app.annotations.get(0, []) == []
    assert app.status_messages[-1] in cs.DIAGNOSES.values()


@needs_cv2
def test_annotator_click_remembers_the_animal_size(app):
    """The next click starts its search at the size that just worked, instead
    of widening the window from scratch every time."""
    app._current_bgr = _frame_with(_ellipse_mask(axes=(30, 8), angle_deg=0.0))
    assert app._click_radius_hint is None
    app._click_to_box(100, 100)
    assert app._click_radius_hint == pytest.approx(61.0, rel=0.10)


def test_annotator_click_with_no_frame_says_so(app):
    app._current_bgr = None
    app._click_to_box(10, 10)
    assert app.annotations.get(0, []) == []
    assert app.status_messages[-1] == cs.DIAGNOSES[cs.NO_FRAME]


@needs_cv2
def test_automatic_click_starts_once_and_falls_back_without_losing_the_click(app):
    app._click_backend = "auto"
    app._current_bgr = _frame_with(_ellipse_mask())
    starts = []
    app._start_sam = lambda **kwargs: starts.append(kwargs)
    app._click_to_box(100, 100)
    app._click_to_box(100, 100)
    assert len(starts) == 1
    assert not starts[0]["report_failure"]
    assert not app.annotations.get(0)
    starts[0]["on_done"](False)
    assert len(app.annotations[0]) == 1
    app._click_to_box(100, 100)
    assert len(starts) == 1, "an unavailable helper must not delay every click"


@needs_cv2
def test_automatic_click_resumes_with_the_ready_helper(app):
    app._click_backend = "auto"
    app._current_bgr = _frame_with(_ellipse_mask())
    starts, requests = [], []
    app._start_sam = lambda **kwargs: starts.append(kwargs)
    app._request_sam_point = requests.append
    app._click_to_box(100, 100)
    app._sam_client = _fake_helper()
    starts[0]["on_done"](True)
    assert len(requests) == 1
    assert (requests[0].x, requests[0].y) == (100, 100)
    assert not app.annotations.get(0)


@pytest.mark.parametrize("change", ["frame", "image", "tool", "class", "backend"])
@needs_cv2
def test_automatic_click_does_not_replay_after_the_user_moves_on(app, change):
    app._click_backend = "auto"
    app._current_bgr = _frame_with(_ellipse_mask())
    starts = []
    app._start_sam = lambda **kwargs: starts.append(kwargs)
    app._click_to_box(100, 100)
    if change == "frame":
        app.current_frame_idx = 1
    elif change == "image":
        app._current_bgr = app._current_bgr.copy()
    elif change == "tool":
        app.set_tool("draw")
    elif change == "class":
        app.active_class_id = 1
    else:
        app._click_backend = "local"
    starts[0]["on_done"](False)
    assert not app.annotations
    assert not app._click_start_pending
    assert "cancelled" in app.status_messages[-1]


@needs_cv2
def test_the_click_tool_makes_one_box_and_steps_aside(app):
    """C fits a box, then hands the mouse back to Select.

    A fitted box is almost never right first time and the next thing the hand
    does is grab a handle. While the tool stayed armed that grab landed as a
    *second* click, dropping another box on top of the first one — so the tool
    that was supposed to save a drag cost a delete instead.
    """
    app.set_tool("click")
    app._current_bgr = _frame_with(_ellipse_mask(axes=(30, 8), angle_deg=20.0))
    app._click_to_box(100, 100)

    obj = app.annotations[0][0]
    assert app._tool == "select", "one click, one box, then back to Select"
    assert app._selected is obj, "and the new box is the one under the handles"


@needs_cv2
def test_keep_drawing_holds_the_click_tool_too(app):
    """The one opt-out, shared with W and E rather than invented for C."""
    app._sticky_draw = True
    app.set_tool("click")
    app._current_bgr = _frame_with(_ellipse_mask(axes=(30, 8)))
    app._click_to_box(100, 100)
    assert app._tool == "click"


def test_tool_switching_round_trips(app):
    """C selects Click to Box and C again returns to Select, like W does."""
    assert app._tool == "select"
    app.toggle_click_tool()
    assert app._tool == "click"
    app.toggle_click_tool()
    assert app._tool == "select"
    app.set_tool("click")
    app._escape()
    assert app._tool == "select"


def test_the_click_tool_has_a_shortcut_and_it_is_free():
    """One key, one action: C was unbound before this tool existed."""
    from htrtdetr.gui import shortcuts

    clicks = [b for b in shortcuts.BINDINGS if b.action == "tool.click_box"]
    assert len(clicks) == 1
    assert clicks[0].key == "C"
    others = [b for b in shortcuts.BINDINGS
              if b.key == "C" and b.action != "tool.click_box"]
    assert others == []


# --------------------------------------------------------------------------- #
# Backend selection in the annotator (Phase 3 and 5)
# --------------------------------------------------------------------------- #

def _predictions_file(tmp_path, frame=0, boxes=((80, 92, 120, 108),),
                      obbs=None, scores=(0.81,)):
    """A predictions.json in the shape `cli.run_predict` writes."""
    payload = {str(frame): {"boxes": [list(b) for b in boxes],
                            "scores": list(scores)}}
    if obbs is not None:
        payload[str(frame)]["obb"] = [list(o) for o in obbs]
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_model_backend_is_refused_until_predictions_are_loaded(app):
    """Offering a fit with no data behind it would fail on the first click."""
    app.set_click_backend("model")
    assert app._click_backend == "local"
    assert "No model predictions are loaded" in app.status_messages[-2]


def test_loading_predictions_enables_the_model_backend(app, tmp_path):
    assert app.load_predictions_for_clicks(_predictions_file(tmp_path))
    app.set_click_backend("model")
    assert app._click_backend == "model"
    assert app._predictions[0]


def test_empty_predictions_are_diagnosed_not_silently_accepted(app, tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"0": {"boxes": []}}), encoding="utf-8")
    assert not app.load_predictions_for_clicks(path)
    assert "no detections" in app.status_messages[-1]


def test_unreadable_predictions_say_why(app, tmp_path):
    path = tmp_path / "predictions.json"
    path.write_text("not json at all", encoding="utf-8")
    assert not app.load_predictions_for_clicks(path)
    assert "Could not read those predictions" in app.status_messages[-1]


@needs_cv2
def test_a_click_on_the_model_backend_takes_the_detection(app, tmp_path):
    """No inference runs: the click chooses among boxes that already exist."""
    app.load_predictions_for_clicks(
        _predictions_file(tmp_path, obbs=[[100.0, 100.0, 38.0, 14.0, 0.25]]))
    app.set_click_backend("model")
    app._current_bgr = _frame_with(_ellipse_mask())
    app._click_to_box(100, 100)

    obj = app.annotations[0][0]
    assert obj.obb[:4] == pytest.approx([100.0, 100.0, 38.0, 14.0])
    assert obj.source == "wand"
    assert "model's detection" in app.status_messages[-1]


def test_a_click_away_from_every_detection_creates_nothing(app, tmp_path):
    app.load_predictions_for_clicks(_predictions_file(tmp_path))
    app.set_click_backend("model")
    app._current_bgr = np.zeros((300, 300, 3), np.uint8)
    app._click_to_box(280, 280)
    assert app.annotations.get(0, []) == []
    assert app.status_messages[-1] == cs.DIAGNOSES[cs.NO_CANDIDATE]


def test_pick_candidate_prefers_the_box_the_click_is_inside():
    """Two overlapping detections: containment beats a nearer centre."""
    class C:
        def __init__(self, bbox, score):
            self.bbox, self.obb, self.score = bbox, None, score
    inside = C([90, 90, 130, 110], 0.5)       # click is in this one
    near = C([102, 60, 142, 80], 0.9)         # centre is closer, click is outside
    res = cs.pick_candidate([near, inside], cs.ClickRequest(0, 100, 100))
    assert res.ok and res.bbox == (90, 90, 130, 110)


def test_pick_candidate_ignores_a_detection_far_away():
    class C:
        bbox = [0, 0, 10, 10]
        obb = None
        score = 1.0
    res = cs.pick_candidate([C()], cs.ClickRequest(0, 500, 500))
    assert not res.ok and res.diagnosis == cs.NO_CANDIDATE


# --------------------------------------------------------------------------- #
# The SAM path in the annotator, against a real helper process
# --------------------------------------------------------------------------- #

@pytest.fixture
def app_with_helper(app):
    """The annotator wired to a live ``--backend stub`` helper.

    The stub answers deterministically, so these tests check the *annotator's*
    async handling — the generation guard, the result path, the cleanup — and
    not a model's quality.
    """
    from htrtdetr.gui import sidecar as sc

    spec = sc.SidecarSpec(python=sys.executable, repo_root=str(REPO_ROOT),
                          backend="stub")
    client = sc.SidecarClient(spec)
    client.start(timeout=60)
    app._sam_client = client
    app._click_backend = "sam"
    try:
        yield app
    finally:
        app._stop_sam()


def _drain_until(app, predicate, timeout=30.0):
    """Pump the sidecar the way the frame callback does, until something happens.

    Deliberately not ``client.wait_for``: that consumes the reply itself, so it
    would test the client rather than the annotator's drain.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app._drain_sam(app._sam_client)
        if predicate():
            return True
        time.sleep(0.01)
    return False


@needs_cv2
def test_a_sam_click_creates_a_box_when_the_reply_arrives(app_with_helper):
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app._click_to_box(100, 100)

    # Nothing yet: the helper has not answered, and the annotator must not have
    # guessed a box in the meantime.
    assert app.annotations.get(0, []) == []
    assert app._sam_waiting

    assert _drain_until(app, lambda: app.annotations.get(0)), "no reply arrived"

    obj = app.annotations[0][0]
    assert obj.source == "wand"
    assert obj.obb[0] == pytest.approx(100.0)      # the stub's fixed answer
    assert obj.obb[2] == pytest.approx(40.0)
    assert not app._sam_waiting


@needs_cv2
def test_a_reply_for_a_frame_the_user_left_is_discarded(app_with_helper):
    """GUI_DESIGN P9: a stale result is dropped, and because the user was
    waiting, the drop is announced instead of being silent."""
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app._click_to_box(100, 100)

    app.current_frame_idx = 7                      # the user moved on
    assert _drain_until(app, lambda: not app._sam_jobs), "no reply arrived"

    assert app.annotations.get(0, []) == []
    assert app.annotations.get(7, []) == []
    assert app.status_messages[-1] == "Frame changed; the segmentation was discarded."


@needs_cv2
def test_switching_away_from_sam_stops_the_helper(app_with_helper):
    app = app_with_helper
    assert app._sam_ready()
    app.set_click_backend("local")
    assert app._click_backend == "local"
    assert not app._sam_ready()
    assert app._sam_jobs == {}


@needs_cv2
def test_a_dead_helper_falls_back_to_local_instead_of_dead_ending(app_with_helper):
    """If the helper dies mid-session the tool must keep working."""
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app._sam_client.stop()

    app._click_to_box(100, 100)                    # _sam_ready() is now False
    assert app.annotations[0][0].obb is not None   # local answered instead


def test_stopping_sam_twice_is_harmless(app):
    app._stop_sam()
    app._stop_sam()
    assert app._sam_client is None


# --------------------------------------------------------------------------- #
# Find all in frame — machine-chosen targets, so they are reviewed (Phase 4)
# --------------------------------------------------------------------------- #

@needs_cv2
def test_find_all_produces_candidates_not_annotations(app_with_helper):
    """GUI_DESIGN P1, and the line this feature is on the other side of from a
    click: here the machine chose *what* to box, so nothing reaches the saved
    objects without a person saying so."""
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    assert app.find_all_in_frame("fly")
    assert _drain_until(app, lambda: app.candidates_here())

    assert len(app.candidates_here()) == 3          # the stub's fixed grid
    assert app.annotations.get(0, []) == []         # nothing saved
    assert "not saved yet" in app.status_messages[-1]


@needs_cv2
def test_accepting_one_candidate_saves_only_that_one(app_with_helper):
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app.find_all_in_frame("fly")
    _drain_until(app, lambda: app.candidates_here())

    index, cand = app.candidates_here()[0]
    assert app.accept_candidate(index)

    assert len(app.annotations[0]) == 1
    assert len(app.candidates_here()) == 2
    obj = app.annotations[0][0]
    assert obj.source == "auto"
    assert obj.score == pytest.approx(cand.score)
    assert obj.to_dict()["score"] == pytest.approx(cand.score, abs=1e-4)


@needs_cv2
def test_clicking_a_candidate_accepts_it(app_with_helper):
    """The click *is* the review act — no second keystroke to confirm."""
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app.find_all_in_frame("fly")
    _drain_until(app, lambda: app.candidates_here())

    _index, cand = app.candidates_here()[1]
    hit = app._hit_candidate(cand.obb[0], cand.obb[1])
    assert hit is not None
    app.accept_candidate(hit)
    assert len(app.annotations[0]) == 1
    assert len(app.candidates_here()) == 2


@needs_cv2
def test_accept_all_and_discard_all(app_with_helper):
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app.find_all_in_frame("fly")
    _drain_until(app, lambda: app.candidates_here())

    assert app.accept_all_candidates() == 3
    assert len(app.annotations[0]) == 3
    assert app.candidates_here() == []

    app.find_all_in_frame("fly")
    _drain_until(app, lambda: app.candidates_here())
    assert app.discard_all_candidates() == 3
    assert app.candidates_here() == []
    assert len(app.annotations[0]) == 3            # accepted ones untouched


@needs_cv2
def test_a_stale_concept_reply_is_discarded(app_with_helper):
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app.find_all_in_frame("fly")
    app.current_frame_idx = 4
    assert _drain_until(app, lambda: not app._sam_jobs)

    assert app.candidates_here() == []
    assert app.status_messages[-1] == "Frame changed; the search was discarded."


def test_find_all_without_a_helper_starts_it_instead_of_refusing(app):
    """It used to point at an unrelated Click to Box setting, in one line at
    the bottom of the window, which made the button look dead. Continue and
    clip propagation have always started the helper themselves."""
    asked = []
    app._start_sam = lambda **kwargs: (asked.append(kwargs.get("on_done")), False)[1]

    assert not app.find_all_in_frame("fly")

    assert asked, "the press must start the helper"
    assert "Fit with:" not in app.status_messages[-1]
    assert "fly" in app.status_messages[-1]


def test_find_all_replays_the_typed_word_when_the_helper_comes_up(app):
    """The replay must search for what was typed, not for the default class:
    the word is resolved before the gate and captured by the callback."""
    captured = []
    app._start_sam = lambda **kwargs: (captured.append(kwargs["on_done"]), False)[1]
    app.find_all_in_frame("wasp")

    searched = []
    app.find_all_in_frame = lambda text="": searched.append(text)
    captured[0](True)
    assert searched == ["wasp"]

    searched.clear()
    captured[0](False)
    assert searched == [], "a failed start must not run a search"


# --------------------------------------------------------------------------- #
# Starting the helper — off the UI thread, so the window keeps drawing
# --------------------------------------------------------------------------- #

def _cap(usable=True, reason=""):
    from htrtdetr.gui import sidecar as sc

    interpreter = sc.InterpreterInfo(
        python=sys.executable, ok=True, torch=True,
        modules=("torch", "sam2", "huggingface_hub"),
    )
    return sc.SidecarCapability(
        usable=usable, interpreter=interpreter if usable else None, reason=reason
    )


def _fake_helper(**hello):
    payload = {"backend": "sam2", "device": "cuda", "capabilities": ["point"]}
    payload.update(hello)
    return SimpleNamespace(hello=payload, is_running=lambda: True, stop=lambda: None)


def test_start_does_not_block_the_ui_thread(app, monkeypatch):
    """The whole point of the change.

    The probe and the helper's start used to run inline in the button
    callback, and dearpygui draws no frame until a callback returns — so the
    status line said "Looking for a Python with PyTorch..." and stayed there,
    with the window greyed out, for as long as both took. Measured on this
    machine: 1.2 s to probe, 2.7 s to start, and a cold ``import torch`` in the
    child costs tens of seconds on top.
    """
    from htrtdetr.annotator import sidecar_mod

    entered, release = threading.Event(), threading.Event()

    def slow_probe(*_args, **_kwargs):
        entered.set()
        release.wait(10.0)
        return _cap(usable=False, reason="probe finished")

    monkeypatch.setattr(sidecar_mod, "probe_interpreters", slow_probe)
    began = time.monotonic()
    try:
        assert app._start_sam() is False, "not ready yet is the honest answer"
        elapsed = time.monotonic() - began
        assert entered.wait(10.0), "the probe never ran"
        assert elapsed < 1.0, f"_start_sam held the UI thread for {elapsed:.1f}s"
    finally:
        release.set()


def test_one_start_however_many_times_the_button_is_pressed(app, monkeypatch):
    spawned = []
    monkeypatch.setattr(app, "_sam_start_worker", spawned.append)
    first, second = [], []

    assert app._start_sam(on_done=first.append) is False
    assert app._start_sam(on_done=second.append) is False
    assert len(spawned) == 1, "a second press must not start a second helper"

    app._finish_sam_start(spawned[0], _cap(), _fake_helper(), "")
    assert first == [True] and second == [True], "every press waiting must be answered"
    assert app._sam_starting is False
    assert "ready" in app.status_messages[-1]


def test_a_live_helper_answers_the_caller_without_starting_anything(app, monkeypatch):
    spawned = []
    monkeypatch.setattr(app, "_sam_start_worker", spawned.append)
    app._sam_client = _fake_helper()
    answered = []

    assert app._start_sam(on_done=answered.append) is True
    assert spawned == [] and answered == [True]


def test_a_failed_start_reports_one_line_and_releases_the_waiting_press(app, monkeypatch):
    spawned = []
    monkeypatch.setattr(app, "_sam_start_worker", spawned.append)
    answered = []
    app._start_sam(on_done=answered.append)

    app._finish_sam_start(
        spawned[0], _cap(usable=False, reason="No Python has PyTorch.\npip install ..."), None, ""
    )

    assert answered == [False], "the press must be told, not left hanging"
    assert app._sam_client is None
    assert app._sam_starting is False
    assert app.status_messages[-1] == "No Python has PyTorch."


def test_automatic_start_failure_does_not_interrupt_labelling_with_a_dialog(app, monkeypatch):
    spawned, dialogs, answered = [], [], []
    monkeypatch.setattr(app, "_sam_start_worker", spawned.append)
    monkeypatch.setattr(app, "_show_unavailable", lambda *args: dialogs.append(args))
    app._start_sam(on_done=answered.append, report_failure=False)
    app._finish_sam_start(spawned[0], _cap(usable=False, reason="No helper"), None, "")
    assert answered == [False]
    assert not dialogs


def test_stopping_while_starting_discards_the_helper_that_lands_later(app, monkeypatch):
    """Otherwise the user turns it off and a helper appears seconds later,
    holding the GPU, and the operation that was waiting runs after the no."""
    spawned = []
    monkeypatch.setattr(app, "_sam_start_worker", spawned.append)
    waiting = []
    app._start_sam(on_done=waiting.append)

    app._stop_sam()

    stopped = []
    late = _fake_helper()
    late.stop = lambda: stopped.append(True)
    app._finish_sam_start(spawned[0], _cap(), late, "")

    assert app._sam_client is None
    assert stopped == [True], "a helper nobody asked for any more must be stopped"
    assert waiting == [], "the operation waiting on it must not run after a stop"


@needs_cv2
def test_candidates_never_reach_the_saved_file(app_with_helper, tmp_path):
    """The guarantee stated on the control: an unreviewed run cannot poison a
    training set, because a candidate has no path to disk at all."""
    app = app_with_helper
    app._current_bgr = _frame_with(_ellipse_mask())
    app.video_path = tmp_path / "v.mp4"
    app.fps, app.frame_w, app.frame_h, app.num_frames = 25.0, 200, 200, 10
    app.find_all_in_frame("fly")
    _drain_until(app, lambda: app.candidates_here())

    out = tmp_path / "annotations.json"
    app._write_annotation(out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["videos"][0]["frames"] == []
