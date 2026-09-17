"""
tests/test_analysis_track_identity.py — the analysis page's half of the switch.

Step 3 asks the same question the annotator asks: *is this run to be read as
individuals, or as detections?* The answer changes the overlays, the inspector,
the timeline and the command the page builds — but never the file. A run that
carries track ids keeps carrying them; switching the box off is a statement
about how the page reads them.

The page is built inside the hub, because that is where it lives and because
building it is exactly the step that finds a tag created in the wrong order.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

dpg = pytest.importorskip("dearpygui.dearpygui")
pytest.importorskip("numpy")

from htrtdetr.gui import theme  # noqa: E402
from htrtdetr.gui.hub import Hub  # noqa: E402

pytestmark = pytest.mark.gui

FRAMES = 40


def _predictions(path: Path, *, tracked: bool) -> Path:
    """A two-animal run in the shape ``inference/streaming.py`` writes."""
    out = {}
    for index in range(FRAMES):
        out[str(index)] = {
            "boxes": [[10 + index, 20, 40 + index, 50], [100 - index, 20, 130 - index, 50]],
            "scores": [0.9, 0.8],
            "track_ids": [1, 2] if tracked else [-1, -1],
            "action_ids": [1, 2 if index < 20 else 3],
        }
    path.write_text(json.dumps(out), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def panel():
    # Hub() calls auto_label.ensure_child_env, which mutates os.environ in
    # place so every subprocess button in the suite inherits PYTHONPATH and
    # PYTHONIOENCODING. That is right for the app and wrong to leave behind
    # for the next test module, so it is restored on teardown.
    saved = dict(os.environ)
    hub = Hub()
    dpg.create_context()
    # What Hub.run() does, in the same order: a context, then the theme.
    # Skipping the theme leaves the previous module's cached ids in place.
    theme.setup()
    dpg.create_viewport(title="yoake analysis track tests", width=1280, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    hub.build_ui()
    hub.switch("analysis")
    for _ in range(3):
        dpg.render_dearpygui_frame()
    yield hub.analysis
    dpg.destroy_context()
    os.environ.clear()
    os.environ.update(saved)


def _pump(frames: int = 3) -> None:
    for _ in range(frames):
        dpg.render_dearpygui_frame()


def test_the_page_reads_runs_as_individuals_by_default(panel):
    panel.set_use_track_ids(True)
    assert panel.use_track_ids is True
    assert dpg.get_item_configuration("ana_tg_tid")["enabled"] is True


def test_switching_it_off_locks_the_overlays_that_need_identity(panel):
    """Locked, not hidden (P4): the user has to be able to see that the
    overlay exists and what would turn it back on."""
    panel.set_use_track_ids(False)
    _pump()
    for tag in ("ana_tg_tid", "ana_tg_traj", "ana_only_selected"):
        assert dpg.get_value(tag) is False, f"{tag} should be off"
        assert dpg.get_item_configuration(tag)["enabled"] is False, f"{tag} should be locked"
    panel.set_use_track_ids(True)


def test_switching_it_back_on_unlocks_them(panel):
    panel.set_use_track_ids(False)
    panel.set_use_track_ids(True)
    _pump()
    for tag in ("ana_tg_tid", "ana_tg_traj", "ana_only_selected"):
        assert dpg.get_item_configuration(tag)["enabled"] is True


def test_the_timeline_becomes_one_lane(panel, tmp_path):
    panel._load_predictions(_predictions(tmp_path / "tracked.json", tracked=True))
    panel.set_use_track_ids(True)
    _pump()
    assert "One lane" not in dpg.get_value("ana_tl_note")

    panel.set_use_track_ids(False)
    _pump()
    note = dpg.get_value("ana_tl_note")
    assert "One lane" in note and "per frame rather than per animal" in note
    panel.set_use_track_ids(True)


def test_a_run_with_no_ids_switches_itself_off(panel, tmp_path):
    """Offering a view of data that is not in the file is worse than not
    offering it, so the file wins over the checkbox."""
    panel.set_use_track_ids(True)
    panel._load_predictions(_predictions(tmp_path / "untracked.json", tracked=False))
    _pump()
    assert panel.use_track_ids is False
    assert dpg.get_value("ana_use_tracks") is False


def test_a_run_with_ids_leaves_the_choice_alone(panel, tmp_path):
    panel.set_use_track_ids(True)
    panel._load_predictions(_predictions(tmp_path / "tracked2.json", tracked=True))
    _pump()
    assert panel.use_track_ids is True


def test_the_inspector_renders_without_identity(panel, tmp_path):
    """The path that used to assume every detection had a track badge."""
    panel._load_predictions(_predictions(tmp_path / "untracked2.json", tracked=False))
    panel.goto(5)
    panel._select_index(0)
    _pump()  # must not raise
    assert panel.use_track_ids is False
