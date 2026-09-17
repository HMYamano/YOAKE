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
pytest.importorskip("cv2")

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
