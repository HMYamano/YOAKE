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
