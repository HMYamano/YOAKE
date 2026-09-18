"""
Contract tests for the annotator's wiring — no viewport, no video.

These assert the joins between the annotator and the modules it now delegates
to. They matter because each join is a *name*, and a name that stops matching
fails silently: a keymap action with no implementation is a documented key that
does nothing, an Inspector callback with a typo is a button that does nothing,
and neither shows up in a visual check.

Creating the app is avoided entirely: ``AnnotationApp.__init__`` calls
``dpg.create_context()``, and a unit test must not. ``__new__`` gives an
instance whose bound methods exist, which is all these tests need.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("dearpygui.dearpygui")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from types import SimpleNamespace  # noqa: E402

from htrtdetr import annotator as A  # noqa: E402
from htrtdetr.gui import inspector as INS, shortcuts  # noqa: E402


@pytest.fixture
def app():
    """An un-initialised instance: bound methods, no dearpygui context."""
    return A.AnnotationApp.__new__(A.AnnotationApp)


# ══════════════════════════════════════════════════════════════════════════
# Keymap <-> implementation
# ══════════════════════════════════════════════════════════════════════════

def test_every_annotator_binding_has_an_implementation(app):
    """A key in the help table with no handler is a lie the user acts on."""
    actions = app._actions()
    missing = sorted({b.action for b in shortcuts.for_scope("annotator")}
                     - set(actions))
    assert not missing, f"bound but unimplemented: {missing}"


def test_every_implementation_is_bound_to_a_key(app):
    """The other direction: an action nothing can reach is dead code."""
    bound = {b.action for b in shortcuts.for_scope("annotator")}
    orphans = sorted(set(app._actions()) - bound)
    assert not orphans, f"implemented but unreachable: {orphans}"


def test_review_scope_is_handled_by_review_mode():
    """Review Mode's keys are routed by name, not re-bound in the annotator."""
    from htrtdetr.gui import review as RV
    mode = RV.ReviewMode.__new__(RV.ReviewMode)
    mode._active = False
    for binding in shortcuts.for_scope("review"):
        # Inactive Review Mode must decline every key, so the annotator's own
        # handlers keep working while it is closed.
        assert mode.handle_action(binding.action) is False


def test_the_action_scopes_the_annotator_registers():
    assert A.AnnotationApp._ACTION_SCOPES == ("annotator", "review")
    for scope in A.AnnotationApp._ACTION_SCOPES:
        assert scope in shortcuts.SCOPES


def test_every_tool_in_the_rail_is_a_real_tool():
    tools = {t[0] for t in A._TOOLS}
    assert tools <= set(A.AnnotationApp._TOOL_NAMES)
    for tool, _label, _icon, action in A._TOOLS:
        assert shortcuts.binding_for(action) is not None, \
            f"{tool} names an unbound action {action}"


def test_every_tool_has_a_chip_sentence():
    """The chip always names the *next* action, for every tool (§0.3.4)."""
    for tool in A.AnnotationApp._TOOL_NAMES:
        assert A._TOOL_CHIP.get(tool), f"{tool} has no chip text"


# ══════════════════════════════════════════════════════════════════════════
# Inspector <-> implementation
# ══════════════════════════════════════════════════════════════════════════

def test_every_inspector_callback_name_resolves(app):
    """The Inspector calls by name; a typo would be a silently dead button."""
    callbacks = app._inspector_callbacks()
    assert callbacks
    for name, fn in callbacks.items():
        assert callable(fn), f"{name} is not callable"


def test_the_inspector_asks_for_what_the_annotator_provides(app):
    """Every name the Inspector's widgets invoke must exist in the map.

    Extracted from the module source rather than hand-listed, so a new button
    in the Inspector fails this test until the annotator implements it.
    """
    source = (ROOT / "src" / "htrtdetr" / "gui" / "inspector.py").read_text(
        encoding="utf-8")
    import re
    asked = set(re.findall(r'self\._cb\("([a-z_]+)"', source))
    asked |= set(re.findall(r'self\._call\("([a-z_]+)"', source))
    provided = set(app._inspector_callbacks())
    missing = sorted(asked - provided)
    assert not missing, f"Inspector calls names the annotator does not provide: {missing}"


# ══════════════════════════════════════════════════════════════════════════
# The copilot's own invariants
# ══════════════════════════════════════════════════════════════════════════

def test_candidate_map_is_empty_without_a_review_session(app):
    app._review = None
    assert app._candidate_map() == {}


def test_accept_targets_is_a_no_op_without_a_session(app):
    app._review = None
    assert app._accept_targets([(0, 0)]) == 0


def test_theme_is_the_only_source_of_colour_in_the_new_modules():
    """No colour literal outside theme.py (the design contract's rule).

    Checked on the modules this work created or rewrote. ``annotator.py`` still
    has literals in code that was not touched; that is recorded as a remaining
    TODO rather than asserted here, so this test stays honest instead of being
    disabled.
    """
    import re
    # Only where a colour is actually *used*: a bare 3-tuple elsewhere is a
    # size, a budget or a range, and flagging those would make the test noise.
    pattern = re.compile(r"(?:color|fill)\s*=\s*\(\s*\d{1,3}\s*,")
    for name in ("gui/timeline.py", "gui/inspector.py", "gui/review.py",
                 "gui/smart_annotate.py"):
        text = (ROOT / "src" / "htrtdetr" / name).read_text(encoding="utf-8")
        hits = [line.strip() for line in text.splitlines() if pattern.search(line)]
        assert not hits, f"{name} contains a colour literal: {hits[:3]}"


def test_the_annotation_package_never_reaches_into_the_gui_renderers():
    """Dependency direction: gui -> annotation, never the other way.

    The logic package must stay usable from a headless script and from the
    training interpreter, which has no dearpygui at all.
    """
    import ast
    base = ROOT / "src" / "htrtdetr" / "annotation"
    renderers = {"timeline", "inspector", "review", "smart_annotate", "hub",
                 "canvas", "widgets", "theme", "analysis_panel", "training_panel"}
    for path in base.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                leaf = node.module.rsplit(".", 1)[-1]
                assert leaf not in renderers, \
                    f"{path.name} imports the renderer {leaf}"
                for alias in node.names:
                    assert alias.name not in renderers, \
                        f"{path.name} imports the renderer {alias.name}"


# ══════════════════════════════════════════════════════════════════════════
# Backward compatibility
# ══════════════════════════════════════════════════════════════════════════

def test_a_human_only_project_still_writes_v1_1_shaped_objects():
    """The byte-compatibility promise: no new keys on a hand-drawn box."""
    obj = A.AnnObject(bbox=[10, 20, 30, 40], track_id=3, action_id=1)
    record = obj.to_dict()
    assert set(record) == {"object_id", "bbox", "class_id", "track_id",
                           "action_id", "occluded", "is_crowd"}
    assert "source" not in record and "score" not in record
    assert "keyframe" not in record


def test_a_machine_box_carries_its_provenance_and_nothing_more():
    obj = A.AnnObject(bbox=[10, 20, 30, 40], source="auto", score=0.83)
    record = obj.to_dict()
    assert record["source"] == "auto"
    assert record["score"] == pytest.approx(0.83)
    assert "obb" not in record


def test_the_cli_accepts_the_flags_the_analysis_page_passes():
    """``analysis_panel`` launches the annotator with these; before this work
    the parser rejected ``--import-predictions`` and the hand-off died."""
    import argparse
    import inspect
    source = inspect.getsource(A.main)
    for flag in ("--import-predictions", "--video", "--load"):
        assert flag in source, f"main() does not accept {flag}"


# ══════════════════════════════════════════════════════════════════════════
# Video propagation: two passes, no GUI
# ══════════════════════════════════════════════════════════════════════════
#
# _propagate_worker runs off the render thread and touches only self._post,
# self._status and the context it is handed, so it can be driven unbound
# against a fake helper — no viewport, and no AnnotationApp instance. It does
# open a VideoCapture of its own, so these write a few frames to tmp_path; that
# is the one thing about it that cannot be faked without faking cv2.
#
# The decision under test is how many passes it asks for. SAM 2 propagates one
# way from the prompt, so a window with frames before the prompt needs two —
# and asking for one is invisible until someone counts the boxes.

class _FakeVideoClient:
    """Records every propagate call and answers with boxes for its own half."""

    def __init__(self, num_frames, prompt_offset, drop=()):
        self.num_frames = num_frames
        self.prompt_offset = prompt_offset
        self.drop = set(drop)          # directions that answer nothing
        self.calls = []
        self._replies = {}
        self._next = 0

    def propagate(self, *, frames_dir, first_frame, prompt_frame, points=None,
                  box=None, obj_id=1, backward=False, max_frames=0):
        self.calls.append(
            {"backward": backward, "first_frame": first_frame,
             "prompt_frame": prompt_frame, "obj_id": obj_id,
             "points": points, "box": box})
        self._next += 1
        prompt = prompt_frame
        last = first_frame + self.num_frames - 1
        span = range(first_frame, prompt + 1) if backward else range(prompt, last + 1)
        if backward in self.drop:
            self._replies[self._next] = None
        else:
            self._replies[self._next] = {
                "ok": True, "window": self.num_frames,
                "items": [{"frame": f, "obb": [10.0, 10.0, 6.0, 3.0, 0.0]}
                          for f in span]}
        return self._next

    def wait_for(self, request_id, timeout=0.0):
        return self._replies.get(request_id)


class _Recorder:
    """The parts of AnnotationApp _propagate_worker actually reaches."""

    def __init__(self):
        self._propagate_stop = False
        self.statuses = []
        self.finished = []

    def _post(self, fn):
        fn()

    def _status(self, text):
        self.statuses.append(text)

    def _finish_video_propagation(self, project, video, result, request):
        self.finished.append(result)


def _video_file(path, frames=40, size=(64, 48)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 5) % 255, np.uint8))
    writer.release()
    return path


def _run_worker(tmp_path, *, before, after, prompt, frames=40, drop=()):
    from htrtdetr.annotator import AnnotationApp
    from htrtdetr.gui import video_segment as vs

    video = SimpleNamespace(
        path=_video_file(tmp_path / "clip.mp4", frames=frames),
        num_frames=frames)
    request = vs.PropagationRequest(
        frame=prompt, box=(10.0, 10.0, 30.0, 30.0), track_id=5,
        before=before, after=after)
    first, _last = request.window(frames)
    client = _FakeVideoClient(
        num_frames=min(frames, request.window(frames)[1] - first + 1),
        prompt_offset=prompt - first, drop=drop)
    app = _Recorder()
    AnnotationApp._propagate_worker(app, request, (None, video, client))
    return app, client, request


def test_a_window_with_a_before_half_costs_two_passes(tmp_path):
    """One pass would export the before half, decode it, and drop it."""
    app, client, _request = _run_worker(tmp_path, before=10, after=10, prompt=20)

    assert [c["backward"] for c in client.calls] == [False, True]
    assert app.finished, "the worker must hand a result to the main thread"
    assert any("backward" in s for s in app.statuses), "say what it is doing"


def test_a_window_that_starts_at_the_prompt_costs_one_pass(tmp_path):
    """Nothing before the prompt, so the backward pass is pure waste."""
    _app, client, _request = _run_worker(tmp_path, before=0, after=15, prompt=5)

    assert [c["backward"] for c in client.calls] == [False]


def test_a_window_clamped_to_frame_zero_still_gets_its_backward_pass(tmp_path):
    """before=100 on frame 8 clamps to first=0, which is still before the
    prompt — the decision is on the clamped window, not on the request."""
    _app, client, _request = _run_worker(tmp_path, before=100, after=5, prompt=8)

    assert [c["backward"] for c in client.calls] == [False, True]
    assert client.calls[0]["first_frame"] == 0


def test_both_passes_are_asked_about_the_same_prompt_and_track(tmp_path):
    """A different prompt or obj_id between passes would produce two animals."""
    _app, client, request = _run_worker(tmp_path, before=10, after=10, prompt=20)

    forward, backward = client.calls
    assert forward["prompt_frame"] == backward["prompt_frame"] == request.frame
    assert forward["obj_id"] == backward["obj_id"] == 5
    assert forward["box"] == backward["box"]


def test_the_merged_result_covers_the_whole_window(tmp_path):
    app, _client, request = _run_worker(tmp_path, before=10, after=10, prompt=20)

    result = app.finished[0]
    first, last = request.window(40)
    assert result.frames == tuple(range(first, last + 1))
    assert "lost it" not in result.describe(track_id=5)


def test_one_dead_pass_still_delivers_the_other_half(tmp_path):
    """Half a tracklet the human can accept beats an error message."""
    app, _client, _request = _run_worker(
        tmp_path, before=10, after=10, prompt=20, drop=(True,))

    assert app.finished, "the forward half must survive a dead backward pass"
    assert app.finished[0].ok and app.finished[0].count


def test_two_dead_passes_report_a_timeout_and_change_nothing(tmp_path):
    app, _client, _request = _run_worker(
        tmp_path, before=10, after=10, prompt=20, drop=(True, False))

    assert not app.finished, "nothing may be handed on when nothing came back"
    assert any("did not answer" in s for s in app.statuses)


def test_pressing_propagate_starts_the_helper_itself(tmp_path):
    """Otherwise the feature is unreachable: the only other caller of
    _start_sam is set_click_backend("sam"), so video propagation would be
    gated behind an unrelated Click to Box setting."""
    app = A.AnnotationApp.__new__(A.AnnotationApp)
    started = []
    app._sam_client = None
    app._sam_ready = lambda: False
    app._start_sam = lambda **kwargs: (started.append(kwargs.get("on_done")), False)[1]
    app._propagate_thread = None
    app.statuses = []
    app._status = app.statuses.append
    shown = []
    app._show_unavailable = lambda feature, reason: shown.append((feature, reason))

    A.AnnotationApp.propagate_selection(app)

    assert started, "the button must start the helper, not send the user away"
    assert not shown, "a helper that is still starting is not an unavailable one"
    assert app.statuses and "as soon as it answers" in app.statuses[-1]


def test_the_press_is_replayed_once_the_helper_answers(tmp_path):
    """The start is asynchronous, so the first press cannot do the work. It
    must come back by itself rather than teaching the user to press twice."""
    app = A.AnnotationApp.__new__(A.AnnotationApp)
    app._sam_client = None
    app._sam_ready = lambda: False
    app._propagate_thread = None
    app.statuses = []
    app._status = app.statuses.append
    app._show_unavailable = lambda feature, reason: None
    captured = []
    app._start_sam = lambda **kwargs: (captured.append(kwargs["on_done"]), False)[1]

    A.AnnotationApp.propagate_selection(app)
    assert captured, "nothing was registered to run when the helper comes up"

    # The helper is now up, and the callback the press left behind runs.
    calls = []
    app.propagate_selection = lambda: calls.append(True)
    captured[0](True)
    assert calls == [True]

    # A helper that failed must not replay anything: the status already said so.
    calls.clear()
    captured[0](False)
    assert calls == []


def test_propagate_does_not_restart_a_running_helper(tmp_path):
    app = A.AnnotationApp.__new__(A.AnnotationApp)
    started = []
    app._sam_client = SimpleNamespace(
        is_running=lambda: True,
        hello={"backend": "sam2", "capabilities": ["point", "video"],
               "device": "cuda"})
    app._sam_ready = lambda: True
    app._start_sam = lambda **_kwargs: started.append(True)
    app._propagate_thread = None
    app.cap = None
    app.statuses = []
    app._status = app.statuses.append

    A.AnnotationApp.propagate_selection(app)

    assert not started, "_start_sam must not be called on a live helper"
    assert app.statuses == ["Open a video first."]
