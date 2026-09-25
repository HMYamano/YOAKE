"""
The four annotator workflows, driven end to end through the real GUI objects.

These are not unit tests. Each one walks a whole user journey — open a video,
analyse it, review it, correct it, save it, load it back — against a real
``AnnotationApp`` with a real dearpygui window and a real video on disk. They
exist because every unit test in this suite can pass while the *joins* are
broken: a callback wired to the wrong name, an undo that restores one frame of
a three-hundred-frame action, a save that drops what was just accepted.

    Workflow A   open video -> Smart Annotate -> samples -> annotate -> save -> reload
    Workflow B   import predictions -> Review Inbox -> accept/reject -> tracklet -> undo -> save
    Workflow C   ID switch -> review -> swap -> undo -> redo -> save -> reload
    Workflow D   behavior interval -> adjust edges -> accept -> save -> reload

They open a window. That is unavoidable: ``render_dearpygui_frame`` on an
un-shown viewport **segfaults** on this machine (dpg 2.3.1, Python 3.14.3), so
there is no headless path, and a GUI whose workflows are only ever tested in
pieces is a GUI whose workflows break. One context is created for the whole
module and destroyed at the end.

Skip them with ``-m "not gui"`` when a window would be unwelcome.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

dpg = pytest.importorskip("dearpygui.dearpygui")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from htrtdetr import annotator as A  # noqa: E402
from htrtdetr.annotation import quality, review_queue  # noqa: E402

pytestmark = pytest.mark.gui

WIDTH, HEIGHT, FRAMES, FPS = 320, 240, 160, 25.0


# ══════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """160 frames of two blobs crossing in the middle, so an ID switch is real."""
    folder = tmp_path_factory.mktemp("yoake_wf")
    path = folder / "courtship_01.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for index in range(FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 28, np.uint8)
        left = int(40 + index * 1.5)
        right = int(280 - index * 1.5)
        cv2.ellipse(frame, (left, 120), (12, 5), index * 3, 0, 360, (215, 215, 225), -1)
        cv2.ellipse(frame, (right, 120), (12, 5), -index * 3, 0, 360, (200, 225, 210), -1)
        writer.write(frame)
    writer.release()
    assert path.exists()
    return path


@pytest.fixture(scope="module")
def predictions(video):
    """Streaming-writer shape, with one defect of each kind planted.

    Top-level frame index -> detections, which is what
    ``inference/streaming.py`` actually writes.
    """
    out = {}
    for index in range(FRAMES):
        left = 40 + index * 1.5
        right = 280 - index * 1.5
        if index == 60:
            left = 300  # teleport -> position jump
        boxes = [[left - 14, 113, left + 14, 127], [right - 14, 113, right + 14, 127]]
        scores = [0.95, 0.28 if 100 <= index < 108 else 0.93]
        tracks = [1, 2]
        actions = [1, 1 if index < 80 else 3]
        if 130 <= index < 138:  # track 2 vanishes -> missing animal
            boxes, scores = boxes[:1], scores[:1]
            tracks, actions = tracks[:1], actions[:1]
        out[str(index)] = {
            "boxes": boxes,
            "scores": scores,
            "track_ids": tracks,
            "action_ids": actions,
            "obb": [
                [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1], 0.2] for b in boxes
            ],
        }
    path = video.parent / "predictions.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def viewport():
    """One dearpygui context for the whole module."""
    app = A.AnnotationApp()
    dpg.create_viewport(title="yoake workflow tests", width=1280, height=820)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    yield app
    dpg.destroy_context()


@pytest.fixture
def app(viewport, video, tmp_path):
    """A built annotator with the video open and a project folder set."""
    instance = viewport
    if not dpg.does_item_exist("main_win"):
        instance.class_names = list(A._DEFAULT_CLASSES)
        instance.action_names = list(A._DEFAULT_ACTIONS)
        instance._build_main_window()
        instance._update_action_buttons()

    # Reset the per-test state without rebuilding the window, which dearpygui
    # does not enjoy doing repeatedly inside one context.
    instance.annotations = {}
    instance._undo.clear()
    instance._redo.clear()
    instance._review = None
    instance._selected = None
    instance._sel_tracklet = instance._sel_interval = instance._sel_issue = None
    instance._selection = None
    instance._dirty = False
    instance.quality_config = quality.QualityConfig()
    instance.project_dir = tmp_path
    instance._load_video(video)
    # The view is state as much as the annotations are: a pan left at the clamp
    # limit by an earlier test makes every later drag look like a no-op.
    instance.fit_view()
    instance.set_tool("select")
    instance.rebuild_copilot()
    _pump(instance, 2)
    return instance


def _pump(app, frames: int = 2) -> None:
    """One turn of the render loop, including the deferred work it carries.

    ``_tick_copilot`` is in that loop for the same reason ``_pump_ui`` is:
    accepting a candidate schedules the re-analysis rather than running it
    inline (it costs 0.8 s on an hour of video, and a held-down Y key used to
    buy one of those per frame). A test that accepts and then reads the review
    queue has to give the loop the same chance the window does, so the clock
    is wound past the quiet period here rather than slept through.
    """
    for _ in range(frames):
        app._pump_ui()
        app._copilot_due = 0.0
        app._tick_copilot()
        dpg.render_dearpygui_frame()


def _saved(app) -> dict:
    app.save_annotation()
    path = app.project_dir / "annotations.json"
    assert path.exists(), "save produced no annotations.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _reload(app, path: Path) -> int:
    """Load a file back and return how many objects came in."""
    app.annotations = {}
    app._load_annotation_file(path)
    return sum(len(v) for v in app.annotations.values())


# ══════════════════════════════════════════════════════════════════════════
# Workflow A — open, Smart Annotate, annotate, save, reload
# ══════════════════════════════════════════════════════════════════════════


def test_workflow_a_smart_annotate_to_saved_file(app):
    from htrtdetr.annotation import sampling

    assert app.num_frames == FRAMES and app.cap is not None

    # 1. Smart Annotate: the dialog collects a request, the worker produces a
    #    selection, and the timeline shows where the samples are.
    app.open_smart_annotate()
    _pump(app)
    assert app.smart_dialog.visible

    request = app.smart_dialog.request()
    assert request.fps == pytest.approx(FPS)
    # Close it before driving the analysis directly: a modal left open blocks
    # every later one, which is also true for a real user.
    app.smart_dialog.hide()
    _pump(app)

    app.start_analysis(
        sampling.SamplingRequest(
            target=sampling.TARGET_BOTH,
            budget=sampling.Budget(samples=12),
            strategy=sampling.DIVERSITY,
            seed=1,
            fps=FPS,
        ),
        {"stride": 8, "generate_candidates": False, "use_gpu": False},
    )

    import time

    deadline = time.time() + 90
    while app._analysis_thread.is_alive() and time.time() < deadline:
        _pump(app)
    _pump(app, 6)

    selection = app._selection
    assert selection is not None, "the analysis produced no selection"
    assert selection.total > 0
    assert all(0 <= f < FRAMES for f in selection.frames)
    assert selection.reasons, "every sample must carry a reason (Priority 6)"

    # 2. The selection reaches the timeline, which is how the user finds it.
    app.apply_selection(selection)
    _pump(app)
    assert app.timeline.model.samples or app.timeline.model.clips

    # 3. Annotate one of the recommended frames by hand.
    target = int(selection.frames[0]) if selection.frames else 0
    app.goto_frame(target)
    obj = app._create_object([50, 100, 90, 130], source="human")
    obj.track_id = 1
    obj.action_id = 2
    app._mark_dirty()
    _pump(app)
    assert len(app.annotations[target]) == 1

    # 4. Save, and 5. reload.
    data = _saved(app)
    assert data["meta"]["version"] in ("1.1", "2.0")
    assert _reload(app, app.project_dir / "annotations.json") == 1
    restored = next(iter(app.annotations.values()))[0]
    assert restored.track_id == 1 and restored.action_id == 2


def test_workflow_a_sampling_is_reproducible_from_its_metadata(app):
    """A recorded seed and strategy must give the same frames back (§41)."""
    from htrtdetr.annotation import features, sampling

    scanned = [
        features.FrameFeature(
            frame=i * 4, embedding=features.normalize([((i * 37) % 11) - 5, i % 7, 1.0, 0.5])
        )
        for i in range(60)
    ]
    request = sampling.SamplingRequest(
        budget=sampling.Budget(samples=10), strategy=sampling.DIVERSITY, seed=5
    )
    first = sampling.plan(request, scanned)
    meta = request.to_metadata()
    again = sampling.plan(
        sampling.SamplingRequest(
            budget=sampling.Budget(samples=10), strategy=meta["strategy"], seed=meta["seed"]
        ),
        scanned,
    )
    assert first.frames == again.frames


# ══════════════════════════════════════════════════════════════════════════
# Workflow B — import, review, accept, tracklet, undo, save
# ══════════════════════════════════════════════════════════════════════════


def test_workflow_b_import_review_accept_tracklet_undo_save(app, predictions):
    # 1. Import: everything arrives pending and nothing touches annotations.
    assert app.import_predictions(predictions)
    _pump(app)
    total = sum(len(v) for v in app._candidate_map().values())
    assert total > 250
    assert app.annotations == {}, "a candidate must never reach annotations"

    # 2. The checks turn those predictions into a short worklist.
    app.quality_config = app.quality_config.replace(expected_animals=2)
    app.rebuild_copilot()
    _pump(app)
    kinds = dict(app._queue.kinds())
    assert 2 <= len(app._queue) <= 20, f"{len(app._queue)} is not a worklist"
    assert quality.POSITION_JUMP in kinds
    assert quality.MISSING_ANIMAL in kinds

    # 3. The Review Inbox is a real surface over that queue.
    app.toggle_inbox()
    _pump(app)
    assert dpg.is_item_shown(app.inbox.tag)
    app.inbox.set_queue(app._queue, fps=app.fps)
    app.inbox.refresh()
    _pump(app)
    app.toggle_inbox()

    # 4. Accept and reject one item each, through the same path the UI uses.
    first = app._queue.filtered()[0]
    app.jump_to_issue(first)
    _pump(app)
    assert app._sel_issue is first
    app._insp_issue("accept")
    _pump(app)
    assert app._queue.decision(first).state == review_queue.ACCEPTED

    second = next(
        i for i in app._queue.items if app._queue.decision(i).state == review_queue.UNRESOLVED
    )
    app._queue.reject(second)
    app._on_review_decision(second, "reject")
    assert app._queue.decision(second).state == review_queue.REJECTED

    # 5. Accept a whole tracklet in one action.
    before = sum(len(v) for v in app.annotations.values())
    part = max(app._tracklets, key=lambda t: t.count)
    app._sel_tracklet = part
    pending = sum(1 for f, i in part.targets() if app._review.state(f, i) == "pending")
    app.accept_tracklet()
    _pump(app)
    gained = sum(len(v) for v in app.annotations.values()) - before
    assert gained == pending, f"{gained} boxes from one click, expected {pending}"
    assert gained > 50, "the whole point is that this is many boxes"

    # 6. One undo takes back the whole tracklet, not one frame of it.
    app.undo()
    _pump(app)
    assert sum(len(v) for v in app.annotations.values()) == before

    app.redo()
    _pump(app)
    assert sum(len(v) for v in app.annotations.values()) == before + gained

    # 7. Save. Accepted boxes are in; pending candidates are not.
    data = _saved(app)
    written = sum(len(f["objects"]) for v in data["videos"] for f in v["frames"])
    assert written == before + gained
    still_pending = sum(
        1
        for f in app._review.frames
        for i in range(len(app._review.candidates(f)))
        if app._review.state(f, i) == "pending"
    )
    assert still_pending > 0, "the fixture should leave some unreviewed"
    assert written < total, "pending candidates must not have been saved"


def test_workflow_b_a_rejected_item_is_not_reinstated_by_bulk_accept(app, predictions):
    app.import_predictions(predictions)
    app.rebuild_copilot()
    item = app._queue.filtered()[0]
    app._queue.reject(item)
    app.accept_issue_kind(item.kind)
    assert app._queue.decision(item).state == review_queue.REJECTED


# ══════════════════════════════════════════════════════════════════════════
# Workflow C — ID switch, swap, undo, redo, save, reload
# ══════════════════════════════════════════════════════════════════════════


def test_workflow_c_id_switch_swap_undo_redo_save_reload(app, predictions):
    app.import_predictions(predictions)
    app.quality_config = app.quality_config.replace(expected_animals=2)
    app.rebuild_copilot()
    _pump(app)

    # 1. The crossing is detected, with both tracks named and context either side.
    switches = [i for i in app._queue.items if i.kind == quality.ID_SWITCH_RISK]
    assert switches, "the two tracks cross; that must be flagged"
    item = switches[0]
    assert len(item.track_ids) == 2
    assert item.span > 2, "the item must cover before and after the contact"

    # 2. Review Mode opens on it and loops around it.
    app._queue.set_filter(kinds={quality.ID_SWITCH_RISK})
    assert app.open_review_mode() is None  # opens, or says nothing is left
    _pump(app)
    if app.review_mode.active:
        low, high = app.review_mode.loop
        assert low <= item.start and high >= item.end
        assert app.review_mode.playing, "the loop must play without being asked"
        moved = app.review_mode.advance_loop(high)
        assert moved == low, "the loop must wrap"
        app.review_mode.close()
    app._queue.clear_filter()

    # 3. Accept the tracklets so there are real boxes to swap.
    app._accept_targets(app._pending_targets(range(FRAMES)))
    _pump(app)
    probe = item.end + 5
    before = sorted(o.track_id for o in app.annotations.get(probe, []))
    assert before, "there must be boxes after the contact to swap"

    # 4. Swap.
    changed = app.swap_tracks(item.track_ids[0], item.track_ids[1], from_frame=item.end)
    _pump(app)
    assert changed > 0
    swapped = sorted(o.track_id for o in app.annotations.get(probe, []))

    # 5. Undo takes back the *whole* sweep, not the frame the user is on.
    app.undo()
    _pump(app)
    assert sorted(o.track_id for o in app.annotations.get(probe, [])) == before

    # 6. Redo re-applies all of it.
    app.redo()
    _pump(app)
    assert sorted(o.track_id for o in app.annotations.get(probe, [])) == swapped

    # 7. Save and reload: the swap survives the round trip.
    _saved(app)
    _reload(app, app.project_dir / "annotations.json")
    assert sorted(o.track_id for o in app.annotations.get(probe, [])) == swapped


def test_workflow_c_ids_correct_resolves_without_touching_anything(app, predictions):
    app.import_predictions(predictions)
    app.rebuild_copilot()
    switches = [i for i in app._queue.items if i.kind == quality.ID_SWITCH_RISK]
    if not switches:
        pytest.skip("no crossing in this fixture")
    item = switches[0]
    app._accept_targets(app._pending_targets(range(FRAMES)))
    probe = item.end + 5
    before = sorted(o.track_id for o in app.annotations.get(probe, []))

    app._sel_issue = item
    app._insp_swap(False)
    assert app._queue.decision(item).state == review_queue.ACCEPTED
    assert sorted(o.track_id for o in app.annotations.get(probe, [])) == before


# ══════════════════════════════════════════════════════════════════════════
# Workflow D — behavior intervals
# ══════════════════════════════════════════════════════════════════════════


def test_workflow_d_behavior_interval_adjust_accept_save_reload(app, predictions):
    app.import_predictions(predictions)
    app.rebuild_copilot()
    _pump(app)

    # 1. Per-frame behavior predictions became a handful of intervals.
    proposed = app._intervals.intervals
    assert len(proposed) >= 2, "the fixture changes behavior partway through"
    assert all(i.source == "auto" for i in proposed)

    # 2. The boxes have to exist for the labels to be written onto something.
    app._accept_targets(app._pending_targets(range(FRAMES)))
    _pump(app)

    interval = max(app._intervals.intervals, key=lambda i: i.span)
    app._sel_interval = interval
    app._refresh_inspector()
    _pump(app)

    # 3. Move the start edge to the current frame.
    new_start = interval.start + 4
    app.goto_frame(new_start)
    app._insp_move_interval("start")
    _pump(app)
    moved = app._sel_interval
    assert moved.start == new_start
    assert moved.source == "human", "an edited edge is the human's"

    # 4. Move the end edge.
    new_end = moved.end - 3
    app.goto_frame(new_end)
    app._insp_move_interval("end")
    _pump(app)
    assert app._sel_interval.end == new_end

    # 5. Relabel, then accept.
    label = app.action_names[2]
    app._insp_relabel_interval(label)
    _pump(app)
    assert app._sel_interval.action_id == 2

    app._insp_confirm_interval()
    _pump(app)
    assert app._sel_interval.source == "human"

    # 6. The interval was written onto the per-frame action_id - the whole
    #    compatibility story for behavior.
    inside = app.annotations.get(app._sel_interval.start + 1, [])
    tracked = [o for o in inside if o.track_id == app._sel_interval.track_id]
    assert tracked and tracked[0].action_id == 2

    # 7. Save and reload: the labels survive as ordinary action ids.
    data = _saved(app)
    assert any(
        o.get("action_id") == 2 for v in data["videos"] for f in v["frames"] for o in f["objects"]
    )
    _reload(app, app.project_dir / "annotations.json")
    assert any(o.action_id == 2 for objs in app.annotations.values() for o in objs)


def test_workflow_d_splitting_an_interval_keeps_both_halves(app, predictions):
    app.import_predictions(predictions)
    app.rebuild_copilot()
    interval = max(app._intervals.intervals, key=lambda i: i.span)
    app._sel_interval = interval
    app.goto_frame(interval.start + interval.span // 2)
    before = len(app._intervals)
    app._insp_split_interval()
    assert len(app._intervals) == before + 1


# ══════════════════════════════════════════════════════════════════════════
# Cross-cutting guarantees
# ══════════════════════════════════════════════════════════════════════════


def test_the_legacy_sidebar_is_gone_and_its_tags_still_resolve(app):
    """Priority 1, asserted: the panel was removed, the contract was kept."""
    assert not dpg.does_item_exist("sidebar_win")
    missing = [tag for tag in A.AnnotationApp._LEGACY_TAGS if not dpg.does_item_exist(tag)]
    assert not missing, f"these moved nowhere: {missing}"


def test_settings_owns_the_project_level_controls(app):
    for tag in (
        "auto_track_cb",
        "sticky_draw_cb",
        "classes_display",
        "actions_display",
        "project_dir_text",
    ):
        assert dpg.does_item_exist(tag)
    app.open_settings("labels")
    _pump(app)
    assert app.settings.visible
    app.settings.hide()
    _pump(app)
    assert not app.settings.visible


def test_video_propagation_is_locked_with_a_reason_when_sam2_is_absent(app):
    capability = app.video_capability()
    if capability.available:
        pytest.skip("this machine has a video-capable helper running")
    assert not capability.available
    assert capability.reason, "a locked feature must say why"
    assert dpg.does_item_exist(
        "rail_propagate"
    ), "the entry point exists even when the backend does not (P4)"


def test_an_absent_sam2_does_not_stop_anything_else(app, predictions):
    """The optional-capability promise: YOAKE runs without SAM 2."""
    assert app.import_predictions(predictions)
    app.rebuild_copilot()
    assert len(app._queue) > 0
    app.propagate_selection()  # no selection, no backend: must not raise
    _pump(app)
    assert dpg.does_item_exist("main_win")


def test_the_embedding_provider_always_returns_something(app):
    provider = app.embedding_provider()
    info = provider.info()
    assert isinstance(info.describe(), str)
    assert info.embedding_dimension >= 0


def test_autosave_writes_only_once_the_debounce_has_passed(app):
    import time

    app._create_object([10, 10, 40, 40], source="human")
    assert app._dirty
    app._tick_autosave()
    assert app._dirty, "autosave must not fire immediately"
    app._save_due = time.time() - 1
    app._tick_autosave()
    # Project autosave writes off-thread and acknowledges on the UI thread.
    deadline = time.time() + 5
    while app._dirty and time.time() < deadline:
        _pump(app)
        time.sleep(0.01)
    assert not app._dirty
    assert (app.project_dir / "annotations.json").exists()


def test_every_control_fits_inside_the_default_viewport(app):
    """The transport row and the status line must be *on screen*.

    They were not: at the default 1280x820 the navbar sat at y=824 and the
    status line at y=859, both below the bottom edge, so First/Prev/Next/Last
    and every status message were unreachable until the user resized the
    window. The bottom reservation is derived from the timeline's real height
    now, and this asserts the result rather than the arithmetic.

    The reflow is explicit because this fixture builds the window directly
    rather than through ``run()``, so nothing has wired
    ``set_viewport_resize_callback``. The real app gets this call from the
    toolkit the moment the OS clamps the window to the display - which it does
    here, asking for 820 and being given 561 - and without it the test would be
    measuring a layout built for a viewport that never existed.
    """
    app._on_viewport_resize()
    _pump(app, 2)
    height = dpg.get_viewport_client_height()
    width = dpg.get_viewport_client_width()
    for tag in ("canvas", "timeline", "navbar", "statusbar"):
        if not dpg.does_item_exist(tag):
            continue
        try:
            _w, h = dpg.get_item_rect_size(tag)
            _x, y = dpg.get_item_rect_min(tag)
        except Exception:
            continue
        assert y + h <= height + 1, f"{tag} extends to {int(y + h)} in a {height}px viewport"
    for tag in ("toolrail", "canvas_win", "right_col"):
        x, _y = dpg.get_item_pos(tag)
        w = dpg.get_item_width(tag)
        assert x + w <= width + 1, f"{tag} extends to {int(x + w)} in a {width}px client area"


def test_hiding_the_panels_widens_the_canvas(app):
    wide_before, _h = app._canvas_size()
    app.toggle_panels()
    _pump(app)
    wide_after, _h2 = app._canvas_size()
    assert wide_after > wide_before
    app.toggle_panels()
    _pump(app)
    assert app._canvas_size()[0] == wide_before


def test_the_canvas_reflows_when_a_timeline_lane_is_added(app):
    """The bottom reservation is derived, not a magic number: a new lane must
    shrink the canvas rather than push the controls off screen."""
    before = app._bottom_height()
    from htrtdetr.gui import timeline as TL

    app.timeline.lanes = app.timeline.lanes + (TL.Lane("extra", "Extra", 12),)
    assert app._bottom_height() == before + 15
    app.timeline.lanes = app.timeline.lanes[:-1]
    assert app._bottom_height() == before


def test_settings_round_trips_the_quality_config(app):
    app.quality_config = quality.QualityConfig(expected_animals=3, low_score=0.33)
    app.settings.load(
        config=app.quality_config, project_dir=app.project_dir, auto_track=False, sticky_draw=False
    )
    _pump(app)
    restored = app.settings.quality_config(quality.QualityConfig())
    assert restored.expected_animals == 3
    assert restored.low_score == pytest.approx(0.33, abs=0.01)


def test_settings_variable_count_turns_the_count_checks_off(app):
    app.quality_config = quality.QualityConfig(expected_animals=2)
    app.settings.load(
        config=app.quality_config, project_dir=app.project_dir, auto_track=False, sticky_draw=False
    )
    dpg.set_value("expected_mode", "Variable")
    _pump(app)
    assert app.settings.quality_config(quality.QualityConfig()).expected_animals is None


def test_the_review_loop_plays_and_wraps(app, predictions):
    app.import_predictions(predictions)
    app.rebuild_copilot()
    if not len(app._queue):
        pytest.skip("no issues in this fixture")
    assert app.open_review_mode() is None
    _pump(app)
    if not app.review_mode.active:
        pytest.skip("review mode declined to open")
    low, high = app.review_mode.loop
    assert high > low
    assert app.review_mode.advance_loop(low) == low + 1
    assert app.review_mode.advance_loop(high) == low, "the loop must wrap"
    # Paused, nothing advances; the pacing gate is the only thing that decides.
    app.review_mode.toggle_play()
    import time

    assert not app.review_mode.due(time.time(), app.fps)
    app.review_mode.toggle_play()
    app.review_mode.close()
    _pump(app)


def test_accepting_in_review_mode_advances_to_the_next_item(app, predictions):
    app.import_predictions(predictions)
    app.rebuild_copilot()
    if len(app._queue) < 2:
        pytest.skip("need at least two issues")
    app.open_review_mode()
    _pump(app)
    if not app.review_mode.active:
        pytest.skip("review mode declined to open")
    first = app.review_mode.item
    app.review_mode.decide("accept")
    _pump(app)
    assert app._queue.decision(first).state == review_queue.ACCEPTED
    assert app.review_mode.item is not first, "accept must move on by itself"
    app.review_mode.close()


# ══════════════════════════════════════════════════════════════════════════
# The tool mouse grammar
# ══════════════════════════════════════════════════════════════════════════
#
# Each tool is driven through the real press/drag/release handlers with the
# cursor faked onto a known image point. These exist because adding a tool to
# the rail and to _TOOL_NAMES does *not* give it behaviour: Pan and Delete were
# both selectable, both drew their border and caption, and both silently fell
# through to Select - so H drew a box instead of moving the view, and X
# selected a box instead of deleting it.


def _at(app, image_x, image_y):
    """Pretend the cursor is over this image pixel.

    The timeline is stubbed out as well as the canvas, and it has to be:
    ``_on_mouse_press`` gives the timeline first refusal through
    ``_begin_scrub``, and ``timeline.hovered()`` asks dearpygui where the
    **real** mouse pointer is. Stubbing only the canvas therefore left these
    tests reading the operator's cursor: with the pointer resting over the
    timeline strip of the test window every press was swallowed as a scrub,
    every draw produced no box, and the failure moved around depending on how
    many tests had run and where the window had been placed.
    """
    cx, cy = app._i2c(image_x, image_y)
    app._is_over_canvas = lambda: True
    app._get_canvas_mouse = lambda: (cx, cy)
    app.timeline.hovered = lambda: False
    return cx, cy


def _press_drag_release(app, start, end=None):
    _at(app, *start)
    app._on_mouse_press(None, None)
    if end is not None:
        _at(app, *end)
        app._on_mouse_drag(None, None)
    app._on_mouse_release(None, None)


def test_x_deletes_the_box_under_the_cursor(app):
    app.set_tool("select")
    app._create_object([60, 60, 120, 110], source="human")
    app._undo.clear()
    assert len(app.annotations[app.current_frame_idx]) == 1

    app.set_tool("delete")
    _press_drag_release(app, (90, 85))
    assert (
        app.annotations.get(app.current_frame_idx, []) == []
    ), "X must delete the box, not select it"

    app.undo()
    assert len(app.annotations.get(app.current_frame_idx, [])) == 1, "each delete is one undo step"


def test_x_on_empty_space_deletes_nothing_and_draws_nothing(app):
    app.set_tool("delete")
    _press_drag_release(app, (20, 20), (80, 70))
    assert app.annotations.get(app.current_frame_idx, []) == [], "Delete must never create a box"


def test_delete_is_sticky_and_esc_leaves_it(app):
    app.set_tool("select")
    app._create_object([40, 40, 90, 90], source="human")
    app._create_object([150, 40, 200, 90], source="human")
    app.set_tool("delete")
    _press_drag_release(app, (65, 65))
    assert app._tool == "delete", "an eraser stays an eraser"
    _press_drag_release(app, (175, 65))
    assert app.annotations.get(app.current_frame_idx, []) == []
    app._escape()
    assert app._tool == "select"


def test_h_pans_the_view_and_creates_nothing(app):
    app.set_tool("pan")
    before = (app._pan_x, app._pan_y)
    _press_drag_release(app, (100, 100), (160, 130))
    assert (app._pan_x, app._pan_y) != before, "H must move the view"
    assert app.annotations.get(app.current_frame_idx, []) == [], "Pan must never create a box"


def test_pan_is_clamped_so_the_image_cannot_be_lost(app):
    app.set_tool("pan")
    _at(app, 100, 100)
    app._on_mouse_press(None, None)
    app._pan_origin = (0.0, 0.0, 0.0, 0.0)
    app._get_canvas_mouse = lambda: (99999.0, 99999.0)
    app._on_mouse_drag(None, None)
    app._on_mouse_release(None, None)
    # The clamp keeps _PAN_KEEP pixels of canvas free on the far side, so the
    # bound is the canvas width, not the scaled image width.
    assert app._pan_x <= app._canvas_w - A._PAN_KEEP + 1, "the image must stay reachable"


def test_middle_drag_pans_from_any_tool(app):
    """The Pan tool is the discoverable form of a gesture that never goes away."""
    app.set_tool("draw")
    before = (app._pan_x, app._pan_y)
    _at(app, 100, 100)
    app._on_middle_press()
    _at(app, 150, 140)
    app._on_middle_drag()
    app._on_middle_release()
    assert (app._pan_x, app._pan_y) != before
    assert app.annotations.get(app.current_frame_idx, []) == []


def test_w_draws_an_upright_box_even_on_top_of_another(app):
    app.set_tool("select")
    app._create_object([50, 50, 150, 130], source="human")
    app.set_tool("draw")
    _press_drag_release(app, (70, 70), (120, 110))
    boxes = app.annotations.get(app.current_frame_idx, [])
    assert len(boxes) == 2, "W must start a new box even inside an existing one"
    assert boxes[-1].obb is None


def test_e_draws_a_rotated_box_and_leaves_it_selected(app):
    app.set_tool("obb")
    assert app._obb_mode, "the rotated tool is the old OBB mode, made visible"
    _press_drag_release(app, (60, 60), (140, 120))
    boxes = app.annotations.get(app.current_frame_idx, [])
    assert len(boxes) == 1
    assert boxes[0].obb is not None, "E must produce a rotated box"
    assert (
        app._selected is boxes[0]
    ), "it stays selected so the rotation handle is already grabbable"
    assert app._tool == "select", "one box, then back to Select"


def test_a_too_small_drag_is_refused_out_loud(app):
    app.set_tool("draw")
    _press_drag_release(app, (100, 100), (101, 101))
    assert app.annotations.get(app.current_frame_idx, []) == []
    assert "too small" in dpg.get_value("status_text").lower()


def test_keep_drawing_holds_the_tool(app):
    app._set_sticky_draw(True)
    app.set_tool("draw")
    _press_drag_release(app, (40, 40), (100, 90))
    assert app._tool == "draw", "Keep drawing means the tool stays armed"
    app._set_sticky_draw(False)


def test_every_tool_reachable_by_key_has_mouse_behaviour(app):
    """The bug this file exists for: a tool in the rail with no press branch.

    Driven rather than inspected - each tool gets a real press/drag/release and
    must do *something* characteristic, or leave the frame untouched. A tool
    that silently falls through to Select passes no branch of this.
    """
    from htrtdetr.gui import shortcuts

    for tool, _label, _icon, action in A._TOOLS:
        app.annotations = {}
        app._undo.clear()
        app._selected = None
        app.set_tool("select")
        app._create_object([80, 70, 140, 120], source="human")
        app._undo.clear()
        pan_before = (app._pan_x, app._pan_y)

        assert shortcuts.binding_for(action) is not None, f"{tool} is unbound"
        app.set_tool(tool)
        assert app._tool == tool, f"{tool} did not become active"

        _press_drag_release(app, (95, 85), (135, 115))
        boxes = app.annotations.get(app.current_frame_idx, [])
        moved = (app._pan_x, app._pan_y) != pan_before

        if tool == "delete":
            assert not boxes, "Delete must remove the box"
        elif tool == "pan":
            assert moved and len(boxes) == 1, "Pan must move the view only"
        elif tool in ("draw", "obb"):
            assert len(boxes) == 2, f"{tool} must add a box"
        elif tool == "click":
            assert len(boxes) >= 1, "Click to Box must not lose the frame"
        else:  # select
            assert len(boxes) == 1


# ══════════════════════════════════════════════════════════════════════════
# Workflow F — accepting in bulk stays responsive
# ══════════════════════════════════════════════════════════════════════════
#
# The re-analysis behind the Review Inbox is a whole-video computation: nine
# detectors over every pending candidate, plus the tracklet index and the
# interval merge. Measured on this project's own scale - one hour at 30 fps,
# detections every 15th frame, four animals - it costs about 0.8 s, and it
# used to run once per accepted candidate. Holding Y down through fifty frames
# bought fifty full re-analyses of the same video.
#
# So the accept path schedules it. These tests pin the two halves of that
# bargain: the accept itself must still be immediate and complete, and the
# analysis must actually catch up.

def test_accepting_does_not_re_analyse_once_per_candidate(app, predictions):
    """A burst of accepts is one rebuild, not one per accept."""
    app.import_predictions(predictions)
    _pump(app)

    calls = []
    real = type(app).rebuild_copilot
    type(app).rebuild_copilot = lambda self: (calls.append(1), real(self))[1]
    try:
        for frame in range(0, 40):
            app.goto_frame(frame)
            app.accept_frame()
        assert not calls, "accepting rebuilt the copilot inline"
        assert app._copilot_dirty, "nothing was scheduled either"
        _pump(app)
        assert len(calls) == 1, f"{len(calls)} rebuilds for one burst of accepts"
    finally:
        type(app).rebuild_copilot = real


def test_the_accept_itself_is_still_immediate(app, predictions):
    """Deferring the analysis must not defer the boxes.

    What the user is looking at - the canvas, the saved annotations, the undo
    entry - is microseconds of work and stays synchronous. Only the analysis
    that feeds the Inbox waits.
    """
    app.import_predictions(predictions)
    _pump(app)
    app.goto_frame(20)
    pending = len(app.candidates_here())
    assert pending, "the fixture must have candidates on this frame"

    before = len(app.annotations.get(20, []))
    app.accept_frame()
    assert len(app.annotations.get(20, [])) == before + pending, (
        "the boxes did not land until the scheduled rebuild"
    )
    assert not app.candidates_here(), "they are still drawn as proposals"
    assert app._undo, "one undo entry, right away"


def test_the_scheduled_rebuild_reaches_the_review_inbox(app, predictions):
    """The Inbox catches up on its own, without another click.

    Identity, not length: ``rebuild_copilot`` builds a fresh
    :class:`ReviewQueue` and carries the decisions across by content key, so a
    rebuild that changed nothing the user can see still replaced the object.
    That is exactly what has to happen for the Inbox to be looking at the
    current analysis rather than the one from before the accept.
    """
    app.import_predictions(predictions)
    _pump(app)
    assert len(app._queue), "the fixture plants defects, so there is something to review"
    before = app._queue

    targets = app._pending_targets(range(FRAMES))
    assert targets
    app._accept_targets(targets)
    assert app._copilot_dirty
    assert app._queue is before, "the accept re-analysed inline after all"

    _pump(app)
    assert not app._copilot_dirty, "the tick never ran the scheduled rebuild"
    assert app._queue is not before, "the Inbox is still showing the stale queue"
    assert app.inbox.queue is app._queue, "the Inbox was not handed the new one"


def test_switching_video_does_not_reuse_the_other_ones_timeline(app, predictions):
    """A frame count is not an identity.

    This study is twenty recordings of one hour each. Keying the reuse of the
    timeline model on ``num_frames`` would have drawn a freshly opened video
    with the previous one's annotations until something forced a rebuild.
    """
    app.import_predictions(predictions)
    _pump(app)
    assert app._timeline_model_video is not None
    built_for = app._timeline_model_video

    app._timeline_model_video = "some other video of the same length"
    app._render_timeline(playhead_only=True)
    assert app._timeline_model_video == built_for, (
        "the light path reused a model built for a different video"
    )


def test_a_pending_rebuild_is_visible_to_the_user(app, predictions):
    """A deferral nobody can see reads as a click the window ignored."""
    app.import_predictions(predictions)
    _pump(app)
    app.goto_frame(20)
    app.accept_frame()
    assert "Re-checking the candidates" in app.running_jobs()
    _pump(app)
    assert "Re-checking the candidates" not in app.running_jobs()
