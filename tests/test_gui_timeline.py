"""
Unit tests for :mod:`htrtdetr.gui.timeline` and :mod:`htrtdetr.gui.inspector`.

Only the parts that do not touch a viewport. Creating a dearpygui context and
rendering a frame **segfaults without ``show_viewport()``** on this machine
(measured: dpg 2.3.1, Python 3.14.3 — ``render_dearpygui_frame`` on an
un-shown viewport aborts), and a unit test must not open a window. So:

* the model assembly, the frame/pixel mapping, the lane layout and the marker
  hit test are tested here, because they are the parts with arithmetic in them
  and the parts a wrong answer in is invisible;
* the drawing itself is covered by ``python -m htrtdetr.annotator --selftest``,
  which does open a viewport and is a manual step.

The split is deliberate. Every bug this file can catch is an off-by-one, and
those are exactly the bugs that survive a visual check.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The training interpreter has no dearpygui (measured: miniconda 3.13.5 here has
# torch and no dearpygui; the GUI's 3.14.3 has dearpygui and no torch). Skip
# rather than fail collection, so the whole suite still runs on both.
pytest.importorskip("dearpygui.dearpygui")

from htrtdetr.annotation import intervals, quality  # noqa: E402
from htrtdetr.gui import timeline as TL  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Frame <-> pixel mapping
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def strip():
    line = TL.Timeline("t")
    line._width = 1000
    line.set_model(TL.TimelineModel(num_frames=2000, current=0))
    return line


def test_first_and_last_frame_are_both_reachable(strip):
    """Both ends must be clickable, or the last frame of every video is
    unreachable by mouse."""
    assert strip.frame_to_x(0) == pytest.approx(TL.Timeline.PAD)
    assert strip.frame_to_x(1999) == pytest.approx(1000 - TL.Timeline.PAD)
    assert strip.x_to_frame(TL.Timeline.PAD) == 0
    assert strip.x_to_frame(1000 - TL.Timeline.PAD) == 1999


def test_mapping_round_trips(strip):
    for frame in (0, 1, 500, 1234, 1999):
        assert abs(strip.x_to_frame(strip.frame_to_x(frame)) - frame) <= 2


def test_clicks_outside_the_strip_clamp_rather_than_overflow(strip):
    assert strip.x_to_frame(-500) == 0
    assert strip.x_to_frame(99999) == 1999


def test_a_single_frame_video_does_not_divide_by_zero():
    line = TL.Timeline("t")
    line._width = 400
    line.set_model(TL.TimelineModel(num_frames=1, current=0))
    assert line.frame_to_x(0) == pytest.approx(TL.Timeline.PAD)
    assert line.x_to_frame(200) == 0


def test_lane_tops_are_ordered_and_inside_the_drawlist(strip):
    tops = [strip._lane_top(lane.key) for lane in TL.LANES]
    assert tops == sorted(tops), "lanes must stack downward in declaration order"
    assert tops[0] >= TL.Timeline.SCRUB_H, "no lane may overlap the scrubber"
    last = TL.LANES[-1]
    assert tops[-1] + last.height <= strip.height


# ══════════════════════════════════════════════════════════════════════════
# Model assembly
# ══════════════════════════════════════════════════════════════════════════

def test_build_model_of_an_empty_project_is_the_empty_state():
    model = TL.build_model(num_frames=0, current=0)
    assert model.is_empty()
    assert model.summary() == "No video loaded"


def test_a_loaded_video_with_nothing_on_it_says_so():
    model = TL.build_model(num_frames=1000, current=0)
    assert not model.is_empty()
    assert model.summary() == "No annotations yet"
    assert not any(model.has_lane(lane.key) for lane in TL.LANES)


def test_candidates_split_into_predicted_and_needs_review():
    """A weak candidate belongs in the review lane, not the predicted one."""
    candidates = {
        10: [SimpleNamespace(score=0.95)],
        20: [SimpleNamespace(score=0.20)],
        30: [SimpleNamespace(score=0.95), SimpleNamespace(score=0.10)],
    }
    model = TL.build_model(num_frames=100, current=0, candidates=candidates)
    assert model.predicted == {10}
    assert model.needs_review == {20, 30}, "one weak box makes the frame weak"


def test_a_flagged_frame_needs_review_even_at_high_confidence():
    """Confidence is not the only reason to look: an ID switch is confident
    and wrong."""
    candidates = {f: [SimpleNamespace(score=0.99)] for f in range(50)}
    item = quality.ReviewItem(kind=quality.ID_SWITCH_RISK, start=20, end=25)
    model = TL.build_model(num_frames=100, current=0, candidates=candidates,
                           review_items=[item])
    assert model.needs_review == set(range(20, 26))
    assert 19 in model.predicted and 26 in model.predicted


def test_annotations_populate_verified_and_keyframes():
    annotations = {5: [SimpleNamespace(keyframe=True)],
                   6: [SimpleNamespace(keyframe=False)]}
    model = TL.build_model(num_frames=100, current=0, annotations=annotations)
    assert model.verified == {5, 6}
    assert model.keyframes == {5}, "only human-placed boxes anchor propagation"


def test_empty_object_lists_do_not_mark_a_frame_verified():
    model = TL.build_model(num_frames=100, current=0, annotations={5: []})
    assert model.verified == set()


def test_behavior_intervals_carry_their_confirmed_state():
    proposed = intervals.BehaviorInterval(1, 0, 10, 20)
    confirmed = intervals.BehaviorInterval(1, 2, 30, 40, source="human")
    model = TL.build_model(num_frames=100, current=0,
                           behavior_intervals=[proposed, confirmed])
    assert model.behaviors == [(10, 20, 0, False), (30, 40, 2, True)]


def test_a_selection_populates_samples_and_clips():
    selection = SimpleNamespace(frames=(3, 9, 27),
                                clips=(SimpleNamespace(start=40, end=80),))
    model = TL.build_model(num_frames=100, current=0, selection=selection)
    assert model.samples == {3, 9, 27}
    assert model.clips == [(40, 80)]
    assert model.has_lane("samples")


def test_issue_flagging_is_clamped_to_the_video():
    """An item whose padding ran off the end must not mark frames that do not
    exist — the status lane would then draw past its own right edge."""
    item = quality.ReviewItem(kind=quality.OVERLAP, start=95, end=200)
    candidates = {f: [SimpleNamespace(score=0.99)] for f in range(100)}
    model = TL.build_model(num_frames=100, current=0, candidates=candidates,
                           review_items=[item])
    assert max(model.needs_review) == 99


def test_summary_counts_each_category():
    model = TL.build_model(
        num_frames=1000, current=0,
        annotations={i: [SimpleNamespace(keyframe=True)] for i in range(5)},
        candidates={i: [SimpleNamespace(score=0.99)] for i in range(100, 110)},
        review_items=[quality.ReviewItem(kind=quality.OVERLAP, start=1, end=2)],
        selection=SimpleNamespace(frames=(700,), clips=()))
    text = model.summary()
    assert "5 verified" in text and "10 predicted" in text
    assert "1 to review" in text and "1 sampled" in text


# ══════════════════════════════════════════════════════════════════════════
# Markers
# ══════════════════════════════════════════════════════════════════════════

def test_the_three_shape_kinds_are_distinct():
    """Shape carries meaning, not only colour (§22)."""
    assert TL.MARKER_SHAPES[quality.ID_SWITCH_RISK] == "cross"
    assert TL.MARKER_SHAPES[quality.MISSING_ANIMAL] == "triangle"
    assert quality.LOW_CONFIDENCE not in TL.MARKER_SHAPES, "the default is a diamond"


def test_marker_hit_test_prefers_the_nearest(strip):
    strip._markers = [(100.0, 30.0, "a"), (108.0, 30.0, "b")]
    y = strip._lane_top("issues") + 2
    assert strip.marker_at(101.0, y) == "a"
    assert strip.marker_at(107.0, y) == "b"


def test_marker_hit_test_respects_its_tolerance(strip):
    strip._markers = [(100.0, 30.0, "a")]
    y = strip._lane_top("issues") + 2
    assert strip.marker_at(100.0 + TL.Timeline.HIT_PX, y) == "a"
    assert strip.marker_at(100.0 + TL.Timeline.HIT_PX + 1, y) is None


def test_a_click_in_another_lane_is_not_a_marker_hit(strip):
    """Otherwise scrubbing over the behavior lane opens review items."""
    strip._markers = [(100.0, 30.0, "a")]
    assert strip.marker_at(100.0, strip._lane_top("samples") + 2) is None
    assert strip.marker_at(100.0, 2.0) is None


# ══════════════════════════════════════════════════════════════════════════
# Inspector — the parts with no dpg in them
# ══════════════════════════════════════════════════════════════════════════

def test_inspector_context_derives_the_review_state():
    from htrtdetr.gui import inspector as INS
    from htrtdetr.annotation import status as S

    assert INS.InspectorContext(accepted=True).state == S.VERIFIED
    assert INS.InspectorContext(accepted=False, score=0.9).state == S.PREDICTED
    assert INS.InspectorContext(accepted=False, score=0.1).state == S.PENDING
    flagged = INS.InspectorContext(
        accepted=False, score=0.99,
        item=quality.ReviewItem(kind=quality.OVERLAP, start=0, end=0))
    assert flagged.state == S.PENDING


def test_inspector_context_labels_fall_back_rather_than_raising():
    from htrtdetr.gui import inspector as INS
    ctx = INS.InspectorContext(action_id=7, class_id=3,
                               action_names=("idle", "walk"), class_names=("fly",))
    assert ctx.action_label() == "Not set"
    assert ctx.class_label() == "class 3"
    assert INS.InspectorContext(action_id=1,
                                action_names=("idle", "walk")).action_label() == "walk"


def test_every_inspector_context_has_a_title():
    from htrtdetr.gui import inspector as INS
    for kind in INS.CONTEXTS:
        assert INS._TITLES.get(kind), f"{kind} has no title"


# ══════════════════════════════════════════════════════════════════════════
# Clicking a clip in the samples lane
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sampled(strip):
    strip.set_model(TL.TimelineModel(num_frames=2000, current=0,
                                     clips=[(100, 300), (900, 1000)]))
    return strip


def samples_y(line):
    return line._lane_top("samples") + 2


def test_a_click_inside_a_clip_finds_it(sampled):
    x = sampled.frame_to_x(200)
    assert sampled.clip_at(x, samples_y(sampled)) == (100, 300)


def test_a_click_beside_a_clip_finds_nothing(sampled):
    """No pixel tolerance here, unlike a marker: a clip is already as wide as
    its own frames, so a tolerance would select the clip next door."""
    x = sampled.frame_to_x(600)
    assert sampled.clip_at(x, samples_y(sampled)) is None


def test_both_ends_of_a_clip_are_clickable(sampled):
    y = samples_y(sampled)
    assert sampled.clip_at(sampled.frame_to_x(100), y) == (100, 300)
    assert sampled.clip_at(sampled.frame_to_x(300), y) == (100, 300)


def test_a_click_in_another_lane_is_not_a_clip_hit(sampled):
    """Otherwise scrubbing over the status lane selects clips."""
    x = sampled.frame_to_x(200)
    assert sampled.clip_at(x, sampled._lane_top("issues") + 2) is None
    assert sampled.clip_at(x, 2.0) is None


def test_the_narrowest_clip_wins_when_two_overlap(strip):
    """A short clip inside a long one is the harder of the two to hit, and the
    more specific answer to which clip was meant."""
    strip.set_model(TL.TimelineModel(num_frames=2000, current=0,
                                     clips=[(100, 900), (400, 500)]))
    y = samples_y(strip)
    assert strip.clip_at(strip.frame_to_x(450), y) == (400, 500)
    assert strip.clip_at(strip.frame_to_x(200), y) == (100, 900)


def test_no_clips_means_no_clip_hit(strip):
    assert strip.clip_at(strip.frame_to_x(200), samples_y(strip)) is None


# -- press order ----------------------------------------------------------

def press_at(line, x, y, monkeypatch):
    monkeypatch.setattr(line, "hovered", lambda: True)
    monkeypatch.setattr(line, "mouse_pos", lambda: (x, y))
    return line.on_press()


def test_a_marker_beats_a_clip(sampled, monkeypatch):
    """The issues lane is above the samples lane, so this is about precedence
    in code rather than in pixels - but marker_at is checked first and a
    reordering would silently swallow review clicks."""
    seen = []
    sampled.on_issue = lambda payload: seen.append(("issue", payload))
    sampled.on_clip = lambda clip: seen.append(("clip", clip))
    x = sampled.frame_to_x(200)
    sampled._markers = [(x, 30.0, "item")]
    assert press_at(sampled, x, sampled._lane_top("issues") + 2, monkeypatch)
    assert seen == [("issue", "item")]


def test_a_clip_click_selects_and_also_seeks(sampled, monkeypatch):
    """A clip is a span, so the frame aimed at is information: the clip pane's
    propagation reads the playhead to decide which frame the human was on."""
    seen = []
    sampled.on_clip = lambda clip: seen.append(clip)
    sampled.on_seek = lambda frame: seen.append(("seek", frame))
    x = sampled.frame_to_x(200)
    assert press_at(sampled, x, samples_y(sampled), monkeypatch)
    assert ("seek", 200) in seen or any(
        isinstance(s, tuple) and s[0] == "seek" and abs(s[1] - 200) <= 2 for s in seen
    ), f"the click should have seeked as well: {seen}"
    assert (100, 300) in seen


def test_an_empty_part_of_the_lane_falls_through_to_a_seek(sampled, monkeypatch):
    seen = []
    sampled.on_clip = lambda clip: seen.append(("clip", clip))
    sampled.on_seek = lambda frame: seen.append(("seek", frame))
    assert press_at(sampled, sampled.frame_to_x(600), samples_y(sampled), monkeypatch)
    assert not any(s[0] == "clip" for s in seen)
    assert any(s[0] == "seek" for s in seen)
    assert sampled._scrubbing, "an empty click must start a scrub"


def test_a_clip_click_without_a_handler_still_seeks(sampled, monkeypatch):
    """on_clip is optional; the timeline must not swallow the press."""
    seen = []
    sampled.on_clip = None
    sampled.on_seek = lambda frame: seen.append(frame)
    assert press_at(sampled, sampled.frame_to_x(200), samples_y(sampled), monkeypatch)
    assert seen, "the press was consumed and nothing happened"


# ══════════════════════════════════════════════════════════════════════════
# The inspector's three dispatch tables must agree
# ══════════════════════════════════════════════════════════════════════════

def test_every_context_has_a_subtitle_entry_and_a_filler():
    """The bug this guards: these three tables are edited together when a
    context is added, and missing one shows an empty pane under a correct
    heading - which reads as "there is nothing here" rather than as a bug."""
    from htrtdetr.gui import inspector as INS
    assert set(INS._TITLES) == set(INS.CONTEXTS)
    assert set(INS._SUBTITLES) == set(INS.CONTEXTS)
    assert set(INS._FILLERS) == set(INS.CONTEXTS)
    for kind, name in INS._FILLERS.items():
        assert hasattr(INS.Inspector, name), f"{kind} points at a missing {name}"


def test_the_clip_context_exists_and_is_not_the_interval_one():
    """A clip is a question and an interval is an answer; sharing a context
    would put edge edits on something with no label to edit."""
    from htrtdetr.gui import inspector as INS
    assert INS.CLIP in INS.CONTEXTS
    assert INS.CLIP != INS.INTERVAL
    assert INS._TITLES[INS.CLIP] == "Sampled clip"


def test_no_two_inspector_sections_share_a_tag():
    """The bug this guards, which cost a crash to find.

    Every section's widgets are created at build time, so two sections using
    the same tag is not a subtle conflict - dearpygui refuses the second one
    with ``Alias already exists`` and the whole window fails to build. Adding
    the clip section with a ``c_`` prefix collided with the *candidate*
    section's ``c_reason``, and nothing caught it until a test that opens a
    viewport tried to build the pane.

    A static check is worth having because the tests that build a window are
    the slowest ones in the suite and the last to run.
    """
    import re
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "htrtdetr" / "gui" / "inspector.py"
    ).read_text(encoding="utf-8")

    # Every method boundary, so a builder's span ends at the next method and
    # not at the next builder - otherwise the last builder swallows the
    # fillers, which legitimately name every section's tags.
    methods = list(re.finditer(r"\n    def (\w+)\(", source))
    assert methods, "no methods found - has the file been restructured?"
    spans = [
        (m.group(1), m.end(), methods[i + 1].start() if i + 1 < len(methods) else len(source))
        for i, m in enumerate(methods)
    ]
    builders = [s for s in spans if s[0].startswith("_build_")]
    assert builders, "no _build_* methods found - has the file been restructured?"

    owners: dict[str, str] = {}
    clashes = []
    for name, start, end in builders:
        for tag in set(re.findall(r'_t\("([a-z_0-9]+)"\)', source[start:end])):
            previous = owners.get(tag)
            if previous is not None and previous != name:
                clashes.append(f"{tag}: {previous} and {name}")
            owners[tag] = name

    assert not clashes, "inspector tags used by two sections: " + "; ".join(clashes)


# ══════════════════════════════════════════════════════════════════════════
# Cost: the lane rules the module states about itself
# ══════════════════════════════════════════════════════════════════════════
#
# The module's docstring makes one promise about cost - "O(width), not
# O(frames)" - and the behavior lane used to be the one lane that broke it.
# ``intervals.merge_to_intervals`` can only join frames within ``max_gap`` of
# each other, so a detector run at stride 15 yields one interval per box:
# 28,800 of them for an hour of video with four animals, drawn one rectangle
# each, rebuilt on every frame change. That is measurably the difference
# between a window that steps frames at 60 fps and one that does not.
#
# These tests count draw calls through a stubbed dearpygui rather than opening
# a viewport, so they run in the unit suite beside the arithmetic they protect.

class _DrawCounter:
    """The subset of dearpygui the renderer touches, counting instead of drawing."""

    def __init__(self):
        self.calls = {}
        self.items = {"t"}

    def _count(self, name):
        def fn(*_a, **kw):
            self.calls[name] = self.calls.get(name, 0) + 1
            tag = kw.get("tag")
            if tag:
                self.items.add(tag)
        return fn

    def __getattr__(self, name):
        if name.startswith(("draw_", "add_", "configure_", "set_", "delete_")):
            return self._count(name)
        raise AttributeError(name)

    def does_item_exist(self, tag):
        return tag in self.items

    @property
    def total_draws(self):
        return sum(n for k, n in self.calls.items() if k.startswith("draw_"))


@pytest.fixture
def counted(monkeypatch):
    stub = _DrawCounter()
    monkeypatch.setattr(TL, "dpg", stub)
    return stub


def _one_frame_intervals(num_frames, stride=15, tracks=4):
    """What a stride-15 detector run actually produces: nothing merges."""
    return [(f, f, k % 3, False)
            for f in range(0, num_frames, stride)
            for k in range(tracks)]


def test_the_behavior_lane_costs_the_window_not_the_video(counted):
    """28,800 unmergeable intervals must not be 28,800 rectangles."""
    line = TL.Timeline("t")
    line._width = 1100
    model = TL.TimelineModel(num_frames=108_000, current=0)
    model.behaviors = _one_frame_intervals(108_000)
    assert len(model.behaviors) == 28_800
    line.set_model(model)
    line.render()
    assert counted.total_draws < 4_000, (
        f"{counted.total_draws} draw calls for a 1,100-pixel strip - the "
        "behavior lane is not bucketing"
    )


def test_a_confirmed_interval_is_not_hidden_by_an_unconfirmed_neighbour(counted):
    """Inside one pixel column, a human's decision wins.

    Colour and fill are the only channel a 9-pixel bar has, so a stretch a
    human signed off must not be redrawn as a proposal because something
    unconfirmed shares its pixel.
    """
    line = TL.Timeline("t")
    line._width = 1100
    model = TL.TimelineModel(num_frames=108_000, current=0)
    # Two intervals one frame apart: the same pixel column, same behavior.
    model.behaviors = [(5000, 5000, 1, False), (5001, 5001, 1, True)]
    line.set_model(model)

    # The alpha of the fill is what says confirmed (190) or not (55), and it
    # is not observable any other way.
    fills = []
    real_rect = TL.dpg.draw_rectangle
    TL.dpg.draw_rectangle = lambda *a, **kw: (fills.append(kw.get("fill")),
                                              real_rect(*a, **kw))[1]
    try:
        line.render()
    finally:
        TL.dpg.draw_rectangle = real_rect

    alphas = [f[3] for f in fills if f is not None and len(f) == 4]
    assert 190 in alphas, "the confirmed interval lost its column to the unconfirmed one"
    assert 55 not in alphas, "both were drawn - they share a pixel, so one must win"


def test_a_wide_interval_is_still_drawn_whole(counted):
    """Bucketing must not shrink a behavior long enough to read into a tick."""
    line = TL.Timeline("t")
    line._width = 1100
    model = TL.TimelineModel(num_frames=1000, current=0)
    model.behaviors = [(100, 900, 0, True)]
    line.set_model(model)
    line.render()
    assert counted.calls.get("draw_rectangle", 0) >= 1


def test_moving_the_playhead_does_not_redraw_the_lanes(counted):
    """The frame-change path.

    Stepping a frame changes one integer on the model. Rebuilding four lanes
    for it cost 15 ms on an hour of video with a detector run in it, on a 16 ms
    budget - which is the whole of "the window went treacle after I ran the
    model".
    """
    line = TL.Timeline("t")
    line._width = 1100
    model = TL.TimelineModel(num_frames=108_000, current=0)
    model.needs_review = set(range(0, 108_000, 15))
    model.issues = [(f, f + 16, "low_confidence", 0.5) for f in range(0, 108_000, 108)]
    line.set_model(model)

    line.render()
    full = counted.total_draws
    assert full > 100, "the lanes did draw something to begin with"

    counted.calls.clear()
    for frame in range(0, 2000, 100):
        model.current = frame
        line.render_playhead()
    per_move = counted.total_draws / 20
    assert per_move <= 4, (
        f"{per_move} draw calls to move the playhead - the lanes are being "
        "rebuilt with it"
    )


def test_a_playhead_move_before_any_render_still_paints_the_lanes(counted):
    """No layer yet means a full render, not an empty strip."""
    line = TL.Timeline("t")
    line._width = 1100
    line.set_model(TL.TimelineModel(num_frames=1000, current=10))
    line.render_playhead()
    assert counted.total_draws > 4, "the lanes were never painted"
