"""
Unit tests for :mod:`htrtdetr.annotation` — the Annotation Copilot's logic.

The point of this file is the purity contract. Every module under test imports
on a machine with **no torch and no dearpygui**, which is the machine the GUI
actually runs on here (Python 3.14: dearpygui 2.3.1, OpenCV 4.13, torch absent).
``test_purity`` asserts that mechanically, because the contract is only worth
anything if it cannot rot silently.

Everything else builds synthetic candidates rather than reading a video, so the
whole file runs in well under a second and can be run from either interpreter.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htrtdetr.annotation import (  # noqa: E402
    cache, features, intervals, propagation, quality, review_queue, sampling,
    status, tracklet,
)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def box(cx, cy, w=20.0, h=8.0, *, score=0.9, tid=0, act=-1, theta=None):
    """One duck-typed candidate: whatever has bbox/obb/score/track_id works."""
    return SimpleNamespace(
        bbox=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        obb=None if theta is None else [cx, cy, w, h, theta],
        score=score, track_id=tid, action_id=act, class_id=0,
    )


@pytest.fixture
def two_flies():
    """60 frames, two tracks.

    Planted defects, one per detector, so a test can assert the detector that
    should fire does and the ones that should not do not:

    * track 1 teleports on frame 20          -> POSITION_JUMP
    * track 2 scores 0.40 on frames 30-35    -> LOW_CONFIDENCE
    * track 2 is absent on frames 45-50      -> MISSING_ANIMAL, TRACK_FRAGMENT
    * the two tracks cross around frame 50   -> ID_SWITCH_RISK, OVERLAP
    """
    frames = {}
    for f in range(60):
        a = box(100 + f * 2, 100, tid=1, score=0.95)
        if f == 20:
            a = box(400, 300, tid=1, score=0.95)
        b = box(300 - f * 2, 100, tid=2, score=0.40 if 30 <= f < 36 else 0.92)
        frames[f] = [a, b]
    for f in range(45, 51):
        frames[f] = [frames[f][0]]
    return frames


# ══════════════════════════════════════════════════════════════════════════
# The purity contract
# ══════════════════════════════════════════════════════════════════════════

MODULES = ("status", "tracklet", "quality", "review_queue", "intervals",
           "features", "sampling", "propagation", "cache")


def _imports(name):
    """``[(module, is_module_scope), ...]`` for one module of the package.

    Parsed with :mod:`ast`, not grepped: the modules talk *about* dearpygui and
    torch in their docstrings, and a substring check would fail on the prose
    that explains why the import is absent.
    """
    import ast
    path = (Path(__file__).resolve().parent.parent / "src" / "htrtdetr"
            / "annotation" / f"{name}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = {id(node) for node in tree.body}
    out = []
    for parent in ast.walk(tree):
        for node in ast.iter_child_nodes(parent):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.append((alias.name.split(".")[0], id(node) in top))
            elif isinstance(node, ast.ImportFrom) and node.module:
                out.append((node.module.split(".")[0], id(node) in top))
    return out


@pytest.mark.parametrize("name", MODULES)
def test_purity_no_gui_or_torch_anywhere(name):
    """No module in the package may import dearpygui or torch, at any scope.

    Not a style rule. The GUI's interpreter cannot import torch and the
    training interpreter cannot import dearpygui, so either import would make
    half the package unusable on the machine that needs it.
    """
    banned = {"dearpygui", "torch", "torchvision"}
    found = {module for module, _ in _imports(name)} & banned
    assert not found, f"{name}.py imports {', '.join(sorted(found))}"


@pytest.mark.parametrize("name", MODULES)
def test_heavy_imports_are_lazy(name):
    """``cv2`` and ``numpy`` may only be imported inside a function body.

    Importing them at module scope would make the package unimportable on a
    machine that has neither, which is the machine the unit tests must run on.
    """
    heavy = {module for module, module_scope in _imports(name)
             if module_scope and module in {"cv2", "numpy"}}
    assert not heavy, f"{name}.py imports {', '.join(sorted(heavy))} at module scope"


# ══════════════════════════════════════════════════════════════════════════
# status
# ══════════════════════════════════════════════════════════════════════════

def test_only_a_human_promotes_to_verified():
    """The rule the whole review loop rests on (P1)."""
    assert status.status_of(accepted=True, score=0.01) == status.VERIFIED
    assert status.status_of(accepted=False, score=0.999) == status.PREDICTED
    assert status.is_training_data(status.VERIFIED)
    assert not status.is_training_data(status.PREDICTED)
    assert not status.is_training_data(status.PENDING)


def test_one_issue_outranks_any_score():
    assert status.status_of(accepted=False, score=0.99, issues=["overlap"]) \
        == status.PENDING


def test_missing_score_reads_as_confident_not_as_doubt():
    """A hand-drawn box has no score. It must not queue itself for review."""
    assert status.status_of(accepted=False, score=None) == status.PREDICTED


def test_summarize_is_human_readable():
    assert status.summarize({}) == "Nothing annotated yet"
    text = status.summarize({status.VERIFIED: 230, status.PREDICTED: 1850,
                             status.PENDING: 42})
    assert text == "230 verified, 1,850 predicted, 42 need review"


# ══════════════════════════════════════════════════════════════════════════
# tracklet
# ══════════════════════════════════════════════════════════════════════════

def test_tracklets_split_on_a_gap_and_not_on_low_confidence(two_flies):
    parts = tracklet.build_tracklets(two_flies)
    ids = [(t.track_id, t.start, t.end) for t in parts]
    assert (1, 0, 59) in ids, "track 1 is continuous and must stay one tracklet"
    assert (2, 0, 44) in ids and (2, 51, 59) in ids, "track 2 must split at its gap"
    # The 0.40-confidence run inside track 2 must NOT have split it: hiding a
    # weak stretch behind two innocent-looking tracklets is the failure mode.
    assert len([t for t in parts if t.track_id == 2]) == 2


def test_tracklet_is_addressable_into_a_review_session(two_flies):
    part = tracklet.build_tracklets(two_flies)[0]
    for frame, index in part.targets():
        assert two_flies[frame][index].track_id == part.track_id


def test_tracklet_reports_its_own_weakest_frame():
    part = tracklet.Tracklet(track_id=3, frames=(1, 2, 3), indices=(0, 0, 0),
                             scores=(0.9, 0.4, 0.8))
    assert part.min_score == pytest.approx(0.4)
    assert part.mean_score == pytest.approx(0.7)
    assert part.count == 3 and part.span == 3 and part.is_contiguous


def test_tracklet_contains_is_false_inside_a_gap():
    part = tracklet.Tracklet(track_id=1, frames=(10, 13), indices=(0, 0),
                             scores=(1.0, 1.0))
    assert part.contains(10) and part.contains(13)
    assert not part.contains(11), "a frame with no candidate is not covered"
    assert part.gaps == ((11, 12),)
    assert part.span == 4 and part.count == 2


def test_untracked_candidates_never_merge():
    """track_id -1 means the producer made no identity claim. Grouping them
    would invent one."""
    cands = {0: [box(10, 10, tid=-1)], 1: [box(12, 10, tid=-1)]}
    parts = tracklet.build_tracklets(cands)
    assert len(parts) == 2
    assert all(t.count == 1 for t in parts)


def test_tracklet_index_lookups(two_flies):
    index = tracklet.TrackletIndex(tracklet.build_tracklets(two_flies))
    assert len(index.at_frame(10)) == 2
    assert len(index.at_frame(47)) == 1, "track 2 is absent there"
    assert index.of(10, 0).track_id == 1
    assert len(index.for_track(2)) == 2, "a fragmented track has two tracklets"
    assert index.total_candidates() == sum(len(v) for v in two_flies.values())


def test_slice_returns_none_rather_than_an_empty_tracklet():
    part = tracklet.Tracklet(track_id=1, frames=(5, 6, 7), indices=(0, 0, 0),
                             scores=(1.0, 1.0, 1.0))
    assert part.slice(6, 7).frames == (6, 7)
    assert part.slice(100, 200) is None


# ══════════════════════════════════════════════════════════════════════════
# quality
# ══════════════════════════════════════════════════════════════════════════

def test_analyze_compresses_predictions_into_a_short_queue(two_flies):
    """The whole value proposition: 114 predictions, a handful of questions."""
    cfg = quality.QualityConfig(expected_animals=2)
    items = quality.analyze(two_flies, config=cfg, frames=list(range(60)),
                            num_frames=60)
    predictions = sum(len(v) for v in two_flies.values())
    assert predictions > 100
    assert 3 <= len(items) <= 15, f"{len(items)} items is not a worklist"


def test_each_planted_defect_is_found(two_flies):
    cfg = quality.QualityConfig(expected_animals=2)
    kinds = {i.kind for i in quality.analyze(two_flies, config=cfg,
                                             frames=list(range(60)), num_frames=60)}
    for expected in (quality.POSITION_JUMP, quality.LOW_CONFIDENCE,
                     quality.MISSING_ANIMAL, quality.TRACK_FRAGMENT,
                     quality.ID_SWITCH_RISK):
        assert expected in kinds, f"{expected} was planted but not detected"


def test_a_clean_run_produces_an_empty_queue():
    """No false positives on well-behaved data — the property that decides
    whether anyone trusts the queue at all."""
    clean = {f: [box(100 + f, 100, tid=1, score=0.95),
                 box(100 + f, 400, tid=2, score=0.96)] for f in range(60)}
    items = quality.analyze(clean, config=quality.QualityConfig(expected_animals=2),
                            frames=list(range(60)), num_frames=60)
    assert items == [], f"clean data produced {[i.kind for i in items]}"


def test_count_detectors_stay_silent_without_an_expected_count():
    frames = {f: [box(100, 100, tid=1)] for f in range(20)}
    items = quality.detect_count_mismatch(frames, quality.QualityConfig())
    assert items == []


def test_a_frame_with_no_detections_at_all_is_still_checked():
    """The case the count detector exists for: an empty frame writes no entry,
    so scanning only the keys of ``candidates`` would never see it."""
    frames = {f: [box(100, 100, tid=1), box(200, 100, tid=2)] for f in range(20)}
    del frames[10]
    cfg = quality.QualityConfig(expected_animals=2)
    without = quality.detect_count_mismatch(frames, cfg)
    with_frames = quality.detect_count_mismatch(frames, cfg, frames=list(range(20)))
    assert without == []
    assert any(i.kind == quality.MISSING_ANIMAL and i.start <= 10 <= i.end
               for i in with_frames)


def test_position_jump_baseline_is_per_track_not_global():
    """A fast animal and a still one in one video must not share a threshold."""
    cands = {}
    for f in range(40):
        fast = box(10 + f * 30, 50, tid=1)       # 30 px/frame, every frame
        still = box(500, 500, tid=2)             # never moves
        cands[f] = [fast, still]
    items = quality.detect_position_jumps(cands, quality.QualityConfig())
    assert items == [], "steady motion at any speed is not a jump"


def test_orientation_uses_the_double_angle_convention():
    """179 degrees and 1 degree describe the same body axis. A head-tail flip
    is not representable in the encoding, so it is not an error to report."""
    cands = {0: [box(100, 100, tid=1, theta=math.radians(1))],
             1: [box(100, 100, tid=1, theta=math.radians(179))]}
    assert quality.detect_orientation_jumps(cands, quality.QualityConfig()) == []

    turned = {0: [box(100, 100, tid=1, theta=0.0)],
              1: [box(100, 100, tid=1, theta=math.radians(80))]}
    assert len(quality.detect_orientation_jumps(turned, quality.QualityConfig())) == 1


def test_merged_detection_is_worse_when_an_animal_is_also_missing():
    """A double-size box in a frame that lost an animal is the textbook merge
    and must outrank a merely large box."""
    cfg = quality.QualityConfig(expected_animals=2)
    normal = {f: [box(100, 100, tid=1), box(300, 100, tid=2)] for f in range(30)}
    normal[15] = [box(200, 100, w=60, h=24, tid=1)]     # one huge box, one animal
    items = quality.detect_merged_detections(normal, cfg)
    assert items and items[0].severity > 0.3
    assert "missing" in items[0].detail


def test_id_switch_item_covers_before_and_after_the_contact():
    """Without both sides the question is unanswerable (§15)."""
    cands = {}
    for f in range(60):
        cands[f] = [box(100 + f * 3, 100, tid=1), box(280 - f * 3, 100, tid=2)]
    items = quality.detect_id_switch_risk(
        cands, quality.QualityConfig(contact_pad=8))
    assert items, "a crossing must be flagged"
    item = items[0]
    assert item.span > 2 * 8, "the item must include padding on both sides"


def test_items_are_clamped_to_the_video(two_flies):
    items = quality.analyze(two_flies, config=quality.QualityConfig(expected_animals=2),
                            frames=list(range(60)), num_frames=60)
    assert all(0 <= i.start <= i.end <= 59 for i in items)


def test_a_broken_detector_does_not_lose_the_others(monkeypatch):
    def explode(*_a, **_kw):
        raise RuntimeError("synthetic")
    monkeypatch.setitem(quality.DETECTORS, quality.OVERLAP, explode)
    cands = {f: [box(100, 100, tid=1, score=0.1)] for f in range(20)}
    items = quality.analyze(cands, config=quality.QualityConfig())
    assert any(i.kind == quality.LOW_CONFIDENCE for i in items)
    assert any("could not run" in i.detail for i in items)


def test_severity_ramp_is_bounded():
    assert quality.ramp(0.0, 1.0, 2.0) == 0.0
    assert quality.ramp(1.5, 1.0, 2.0) == pytest.approx(0.5)
    assert quality.ramp(99.0, 1.0, 2.0) == 1.0
    assert quality.ramp(5.0, 2.0, 2.0) == 1.0, "a degenerate range must not divide by zero"


def test_every_issue_code_has_a_label_and_help():
    for code in quality.ISSUE_ORDER:
        assert quality.ISSUE_LABELS.get(code), f"{code} has no label"
        assert quality.ISSUE_HELP.get(code), f"{code} has no help text"


# ══════════════════════════════════════════════════════════════════════════
# review_queue
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def queue(two_flies):
    cfg = quality.QualityConfig(expected_animals=2)
    return review_queue.ReviewQueue(
        quality.analyze(two_flies, config=cfg, frames=list(range(60)), num_frames=60))


def test_queue_filters_by_reason(queue):
    kinds = dict(queue.kinds())
    assert kinds, "the fixture must produce issues"
    target = next(iter(kinds))
    queue.set_filter(kinds={target})
    assert all(i.kind == target for i in queue.filtered())
    queue.clear_filter()
    assert len(queue.filtered()) == len(queue)


def test_advance_stops_at_the_end_rather_than_wrapping(queue):
    for _ in range(len(queue) * 2):
        queue.advance()
    assert queue.current() is queue.filtered()[-1]


def test_next_unresolved_skips_what_was_skipped(queue):
    first = queue.current()
    queue.skip(first)
    nxt = queue.next_unresolved()
    assert nxt is not first
    assert queue.decision(first).state == review_queue.SKIPPED
    assert not queue.decision(first).done, "skipped is not finished work"


def test_progress_counts_skipped_separately(queue):
    queue.accept(queue.filtered()[0])
    queue.skip(queue.filtered()[1])
    p = queue.progress()
    assert p.resolved == 1 and p.skipped == 1
    assert p.remaining == len(queue) - 2
    assert "skipped" in p.describe()


def test_decisions_survive_a_reanalysis(queue, two_flies):
    queue.accept(queue.filtered()[0])
    carried = queue.decisions()
    cfg = quality.QualityConfig(expected_animals=2)
    fresh = review_queue.ReviewQueue(
        quality.analyze(two_flies, config=cfg, frames=list(range(60)), num_frames=60))
    assert fresh.carry_over(carried) == len(carried)
    assert fresh.progress().resolved == 1


def test_a_decision_key_ignores_severity_and_wording():
    """Tuning a threshold must not orphan a human's Accept."""
    a = quality.ReviewItem(kind=quality.OVERLAP, start=1, end=5,
                           track_ids=(1, 2), severity=0.2, detail="x")
    b = quality.ReviewItem(kind=quality.OVERLAP, start=1, end=5,
                           track_ids=(1, 2), severity=0.9, detail="y")
    assert a.key == b.key


def test_swap_decision_records_intent_without_mutating_anything(queue):
    item = next((i for i in queue if i.kind == quality.ID_SWITCH_RISK), None)
    assert item is not None
    decision = queue.swap_decision(item, swapped=True, from_frame=item.end)
    assert decision.state == review_queue.EDITED
    assert decision.data["swapped"] is True
    assert decision.data["from_frame"] == item.end
    assert decision.data["track_ids"] == list(item.track_ids)


def test_ids_correct_resolves_as_accepted(queue):
    item = next(i for i in queue if i.kind == quality.ID_SWITCH_RISK)
    assert queue.swap_decision(item, swapped=False).state == review_queue.ACCEPTED


def test_bulk_accept_never_reverses_a_rejection(queue):
    item = queue.filtered()[0]
    queue.reject(item)
    queue.accept_kind(item.kind)
    assert queue.decision(item).state == review_queue.REJECTED


def test_unknown_decision_state_raises(queue):
    with pytest.raises(ValueError, match="Unknown review decision"):
        queue.set_decision(queue.filtered()[0], "maybe")


def test_duplicate_items_collapse_to_one_job():
    item = quality.ReviewItem(kind=quality.OVERLAP, start=1, end=5, track_ids=(1, 2))
    q = review_queue.ReviewQueue([item, item])
    assert len(q) == 1


def test_empty_queue_answers_everything_without_raising():
    q = review_queue.ReviewQueue([])
    assert q.current() is None
    assert q.advance() is None and q.retreat() is None
    assert q.next_unresolved() is None and q.nearest(10) is None
    assert q.accept() is None
    assert q.progress().describe() == "Nothing to review"
    assert q.summary() == "No locations need review"


# ══════════════════════════════════════════════════════════════════════════
# intervals
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def behavior_run():
    """idle 0-19, walk 20-59 with a one-frame flicker, courtship 60-99."""
    seq = [0] * 20 + [1] * 40 + [2] * 40
    seq[35] = 2
    return {f: [SimpleNamespace(track_id=1, action_id=a, score=0.9)]
            for f, a in enumerate(seq)}


def test_intervals_replace_per_frame_labels(behavior_run):
    out = intervals.merge_to_intervals(behavior_run)
    assert len(out) == 3, "100 frames of three behaviors is three decisions"
    assert [i.action_id for i in out] == [0, 1, 2]
    assert (out[0].start, out[0].end) == (0, 19)


def test_smoothing_removes_a_single_frame_flicker(behavior_run):
    out = intervals.merge_to_intervals(behavior_run)
    walk = [i for i in out if i.action_id == 1]
    assert len(walk) == 1, "a one-frame flicker must not split the walk"


def test_unlabelled_frames_never_become_intervals():
    cands = {f: [SimpleNamespace(track_id=1, action_id=intervals.UNLABELLED,
                                 score=1.0)] for f in range(30)}
    assert intervals.merge_to_intervals(cands) == []


def test_intervals_round_trip_to_per_frame_action_ids(behavior_run):
    """The compatibility bridge: the file format does not change."""
    track = intervals.IntervalTrack(intervals.merge_to_intervals(behavior_run))
    flat = track.to_frame_actions()
    assert len(flat) == 100
    assert flat[0][1] == 0 and flat[30][1] == 1 and flat[90][1] == 2


def test_boundary_edits_are_clamped():
    iv = intervals.BehaviorInterval(track_id=1, action_id=0, start=10, end=20)
    assert iv.move_start(99).start == 20, "start cannot pass end"
    assert iv.move_end(0).end == 10, "end cannot pass start"
    assert iv.move_start(12).source == "human"


def test_split_refuses_to_make_an_empty_side():
    iv = intervals.BehaviorInterval(track_id=1, action_id=0, start=10, end=20)
    left, right = iv.split(15)
    assert (left.start, left.end, right.start, right.end) == (10, 14, 15, 20)
    for bad in (10, 21, 5):
        with pytest.raises(ValueError, match="must fall inside"):
            iv.split(bad)


def test_an_inverted_interval_is_rejected_at_construction():
    with pytest.raises(ValueError, match="ends before it starts"):
        intervals.BehaviorInterval(track_id=1, action_id=0, start=20, end=10)


def test_confirm_marks_human_and_clears_doubt():
    iv = intervals.BehaviorInterval(track_id=1, action_id=0, start=0, end=9,
                                    confidence=0.3, boundary_confidence=(0.2, 0.2))
    done = iv.confirm()
    assert done.source == "human" and done.confidence == 1.0
    assert done.boundary_confidence == (1.0, 1.0)


def test_timecode_matches_the_behavior_literature_format():
    assert intervals.timecode(int(84.2 * 25), 25.0) == "01:24.20"
    assert intervals.timecode(100, 0.0) == "f100", "a lying fps must not divide by zero"


def test_uncertain_boundaries_become_review_items():
    iv = intervals.BehaviorInterval(track_id=1, action_id=2, start=100, end=200,
                                    confidence=0.9, boundary_confidence=(0.3, 0.95))
    items = intervals.detect_interval_issues([iv])
    assert len(items) == 1
    assert items[0].kind == quality.BEHAVIOR_TRANSITION
    assert items[0].start <= 100 <= items[0].end


def test_a_low_confidence_interval_questions_the_label_not_the_edges():
    iv = intervals.BehaviorInterval(track_id=1, action_id=2, start=100, end=200,
                                    confidence=0.2, boundary_confidence=(0.1, 0.1))
    items = intervals.detect_interval_issues([iv])
    assert [i.kind for i in items] == [quality.BEHAVIOR_DISAGREEMENT]


def test_human_intervals_are_never_queued():
    iv = intervals.BehaviorInterval(track_id=1, action_id=2, start=0, end=9,
                                    confidence=0.01, source="human")
    assert intervals.detect_interval_issues([iv]) == []


def test_interval_track_stats_are_an_ethogram(behavior_run):
    track = intervals.IntervalTrack(intervals.merge_to_intervals(behavior_run))
    stats = track.stats(fps=25.0)
    assert stats[0]["frames"] == 20
    assert stats[2]["seconds"] == pytest.approx(40 / 25.0)


# ══════════════════════════════════════════════════════════════════════════
# features
# ══════════════════════════════════════════════════════════════════════════

def test_distance_of_an_unknown_frame_is_unrelated_not_identical():
    """A zero vector at distance 0 would be picked first by diversity."""
    assert features.distance((), (1.0, 0.0)) == 1.0
    assert features.distance((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert features.distance((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(2.0)


def test_farthest_point_prefix_property():
    """The 3-pick traversal is the prefix of the 6-pick traversal, which is why
    a budget can change without recomputing anything."""
    rng = random.Random(0)
    vectors = [features.normalize([rng.gauss(0, 1) for _ in range(8)])
               for _ in range(40)]
    short = features.farthest_point_order(vectors, 3, first=0)
    long = features.farthest_point_order(vectors, 6, first=0)
    assert long[:3] == short


def test_farthest_point_spreads_across_clusters():
    a = [features.normalize([1, 0, 0, 0.01 * i]) for i in range(20)]
    b = [features.normalize([0, 1, 0, 0.01 * i]) for i in range(20)]
    c = [features.normalize([0, 0, 1, 0.01 * i]) for i in range(20)]
    picked = features.farthest_point_order(a + b + c, 3, first=0)
    buckets = {p // 20 for p in picked}
    assert buckets == {0, 1, 2}, "one pick per cluster"


def test_detection_stats_on_an_empty_frame():
    stats = features.detection_stats([])
    assert stats["count"] == 0
    assert stats["min_distance"] == math.inf, "no pair is not zero distance"
    assert stats["mean_score"] == 1.0, "unknown must not read as uncertain"


def test_interaction_peaks_when_animals_touch():
    apart = features.FrameFeature(frame=0, min_distance=5.0)
    near = features.FrameFeature(frame=1, min_distance=0.2)
    overlapping = features.FrameFeature(frame=2, min_distance=0.1, max_overlap=0.8)
    assert apart.interaction == 0.0
    assert near.interaction > 0.5
    assert overlapping.interaction >= 0.8


def test_uncertainty_is_one_minus_mean_score():
    assert features.FrameFeature(frame=0, mean_score=0.7).uncertainty \
        == pytest.approx(0.3)
    assert features.FrameFeature(frame=0).uncertainty == 0.0


def test_scan_skips_unreadable_frames_rather_than_zero_filling():
    """A zero feature is a lie that diversity sampling would happily select."""
    out = features.scan(lambda i: None, [0, 1, 2])
    assert out == []


# ══════════════════════════════════════════════════════════════════════════
# sampling
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def scanned():
    """600 sparse frames: three visual scenes, two interaction bursts, one
    low-confidence stretch."""
    rng = random.Random(7)
    basis = ([1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0])
    out = []
    for i in range(600):
        scene = 0 if i < 200 else (1 if i < 400 else 2)
        base = basis[scene]
        vector = features.normalize(
            [base[j % 4] * 4 + rng.gauss(0, 0.5) for j in range(16)])
        touching = (200 <= i < 230) or (400 <= i < 420)
        out.append(features.FrameFeature(
            frame=i * 3, embedding=vector, count=2,
            min_distance=0.2 if touching else 4.0,
            max_overlap=0.8 if touching else 0.0,
            mean_score=0.45 if 300 <= i < 340 else 0.95))
    return out


def _scene_spread(selection):
    buckets = [0, 0, 0]
    for frame in selection.frames:
        buckets[0 if frame < 600 else (1 if frame < 1200 else 2)] += 1
    return buckets


def test_diversity_covers_every_scene(scanned):
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_DETECTION, budget=sampling.Budget(samples=30),
        strategy=sampling.DIVERSITY, seed=3), scanned)
    assert all(count > 0 for count in _scene_spread(sel)), \
        "diversity must not spend the budget in one scene"


def test_uncertainty_goes_where_the_model_was_unsure(scanned):
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_DETECTION, budget=sampling.Budget(samples=30),
        strategy=sampling.UNCERTAINTY, seed=3), scanned)
    assert all(900 <= f < 1020 for f in sel.frames), \
        "the low-confidence stretch is frames 900-1017"


def test_diversity_plus_uncertainty_keeps_coverage_while_favouring_hard(scanned):
    """The reason for the two-stage shape: a pure ranking of hard frames returns
    k frames from the same two seconds."""
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_DETECTION, budget=sampling.Budget(samples=30),
        strategy=sampling.DIVERSITY_UNCERTAINTY, seed=3), scanned)
    spread = _scene_spread(sel)
    assert all(count > 0 for count in spread), "coverage must survive"
    assert spread[1] > spread[0], "but the hard scene must still win"


def test_interaction_clips_land_on_the_interaction(scanned):
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_BEHAVIOR, budget=sampling.Budget(samples=4),
        strategy=sampling.INTERACTION, seed=3, fps=25.0), scanned)
    assert sel.clips
    truth = list(range(600, 690)) + list(range(1200, 1260))
    hits = sum(1 for c in sel.clips if any(f in truth for f in (c.start, c.middle, c.end)))
    assert hits >= 2, "at least the two real interaction bursts must be covered"


def test_clips_never_overlap(scanned):
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_BEHAVIOR, budget=sampling.Budget(samples=10),
        strategy=sampling.INTERACTION, seed=1, fps=25.0), scanned)
    for a, b in zip(sel.clips, sel.clips[1:]):
        assert not a.overlaps(b), "overlapping clips make a human watch twice"


def test_clip_window_respects_a_sparse_scan(scanned):
    """The scan holds every 3rd frame. A window of `half` indices is not a
    window of `half` frames, and conflating them smears the score."""
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_BEHAVIOR, budget=sampling.Budget(samples=3),
        strategy=sampling.INTERACTION, seed=1, fps=25.0, clip_seconds=3.0), scanned)
    assert all(c.span == 75 for c in sel.clips), "3 s at 25 fps is 75 frames"


def test_the_same_seed_gives_the_same_frames(scanned):
    request = sampling.SamplingRequest(
        budget=sampling.Budget(minutes=15), strategy=sampling.DIVERSITY_UNCERTAINTY,
        seed=3, fps=25.0)
    first = sampling.plan(request, scanned)
    second = sampling.plan(request, scanned)
    assert first.frames == second.frames
    assert [c.start for c in first.clips] == [c.start for c in second.clips]


def test_a_different_seed_gives_different_frames(scanned):
    base = sampling.SamplingRequest(budget=sampling.Budget(samples=30),
                                    strategy=sampling.DIVERSITY, seed=3)
    other = sampling.SamplingRequest(budget=sampling.Budget(samples=30),
                                     strategy=sampling.DIVERSITY, seed=99)
    assert sampling.plan(base, scanned).frames != sampling.plan(other, scanned).frames


def test_an_unknown_strategy_falls_back_instead_of_raising(scanned):
    """A stale strategy name in a saved project must not stop a video opening."""
    sel = sampling.plan(sampling.SamplingRequest(
        budget=sampling.Budget(samples=10), strategy="no-such-strategy"), scanned)
    assert sel.frames, "the fallback must still select something"
    assert sel.total <= 10, "the fallback must still respect the budget"


def test_the_both_target_splits_a_sample_budget_by_share():
    frames, clips = sampling.Budget(samples=100, behavior_share=0.35).counts(
        target=sampling.TARGET_BOTH, model=sampling.WorkloadModel())
    assert (frames, clips) == (65, 35)


def test_prelabelling_multiplies_what_a_budget_buys():
    model = sampling.WorkloadModel()
    blank, _ = sampling.Budget(minutes=15).counts(
        target=sampling.TARGET_DETECTION, model=model, prelabelled=False)
    assisted, _ = sampling.Budget(minutes=15).counts(
        target=sampling.TARGET_DETECTION, model=model, prelabelled=True)
    assert assisted > blank * 3, "auto pre-labelling is the whole argument"


def test_both_target_always_buys_at_least_one_of_each():
    frames, clips = sampling.Budget(samples=3).counts(
        target=sampling.TARGET_BOTH, model=sampling.WorkloadModel())
    assert frames >= 1 and clips >= 1


def test_sample_count_wins_over_a_duration():
    """A count is exact and a duration is an estimate."""
    frames, _ = sampling.Budget(samples=50, minutes=99).counts(
        target=sampling.TARGET_DETECTION, model=sampling.WorkloadModel())
    assert frames == 50


def test_rank_normalize_is_robust_to_outliers():
    out = sampling.rank_normalize([1.0, 2.0, 3.0, 1000.0])
    assert out == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]
    assert sampling.rank_normalize([]) == []
    assert sampling.rank_normalize([5.0]) == [0.5]
    assert sampling.rank_normalize([2.0, 2.0]) == [0.5, 0.5], "ties share a rank"


def test_request_metadata_is_json_safe():
    request = sampling.SamplingRequest(budget=sampling.Budget(minutes=15), seed=7)
    text = json.dumps(request.to_metadata())
    assert json.loads(text)["seed"] == 7


def test_selection_describes_itself_for_the_dialog(scanned):
    sel = sampling.plan(sampling.SamplingRequest(
        target=sampling.TARGET_BOTH, budget=sampling.Budget(minutes=15),
        strategy=sampling.DIVERSITY, seed=1, fps=25.0), scanned)
    text = sel.describe()
    assert "frames" in text and "clip" in text
    assert "Estimated review workload" in text


def test_empty_features_produce_an_empty_selection():
    sel = sampling.plan(sampling.SamplingRequest(budget=sampling.Budget(samples=50)), [])
    assert sel.frames == () and sel.clips == () and sel.total == 0


# ══════════════════════════════════════════════════════════════════════════
# propagation
# ══════════════════════════════════════════════════════════════════════════

def test_a_move_propagates_as_a_move():
    correction = propagation.Correction(
        frame=10, track_id=1,
        before_bbox=[100, 100, 120, 110], after_bbox=[100, 108, 120, 118])
    transform = correction.transform()
    assert transform.moved and not transform.resized
    assert transform.dy == pytest.approx(8.0)


def test_an_identity_edit_plans_nothing():
    correction = propagation.Correction(
        frame=10, track_id=1,
        before_bbox=[100, 100, 120, 110], after_bbox=[100, 100.1, 120, 110.1])
    out = propagation.plan(correction, propagation.PropagationScope(),
                           boxes={f: ([0, 0, 10, 10], None) for f in range(20)})
    assert out.count == 0
    assert "did not change" in out.describe()


def test_propagation_never_overwrites_human_work():
    """P3, asserted. The count is reported so the message cannot lie."""
    boxes = {f: ([100, 100, 120, 110], None) for f in range(20)}
    out = propagation.plan(
        propagation.Correction(frame=5, track_id=1,
                               before_bbox=[100, 100, 120, 110],
                               after_bbox=[100, 110, 120, 120]),
        propagation.PropagationScope(kind=propagation.N_FRAMES, n=10),
        boxes=boxes, num_frames=20, protected={7, 8, 9})
    assert out.skipped_protected == 3
    assert not {7, 8, 9} & set(out.frames)
    assert "already corrected" in out.describe()


def test_propagation_never_creates_a_box():
    boxes = {5: ([100, 100, 120, 110], None), 6: ([100, 100, 120, 110], None)}
    out = propagation.plan(
        propagation.Correction(frame=5, track_id=1,
                               before_bbox=[100, 100, 120, 110],
                               after_bbox=[100, 110, 120, 120]),
        propagation.PropagationScope(kind=propagation.N_FRAMES, n=5),
        boxes=boxes, num_frames=20)
    assert out.frames == (6,)
    assert out.skipped_missing == 4
    assert "no box on this track" in out.describe()


def test_next_keyframe_scope_stops_before_the_keyframe():
    """The keyframe is a box a human placed; the correction is not theirs."""
    got = propagation.resolve_frames(
        propagation.PropagationScope(kind=propagation.NEXT_KEYFRAME),
        frame=10, num_frames=100, keyframes=[0, 10, 25])
    assert got == list(range(11, 25))


def test_backward_propagation_is_opt_in():
    forward_only = propagation.resolve_frames(
        propagation.PropagationScope(kind=propagation.N_FRAMES, n=3),
        frame=10, num_frames=100)
    assert forward_only == [11, 12, 13]
    both = propagation.resolve_frames(
        propagation.PropagationScope(kind=propagation.N_FRAMES, n=3, backward=True),
        frame=10, num_frames=100)
    assert both == [7, 8, 9, 11, 12, 13]


def test_this_frame_only_propagates_nothing():
    assert propagation.resolve_frames(
        propagation.PropagationScope(kind=propagation.THIS_FRAME),
        frame=10, num_frames=100) == []


def test_rotation_is_about_each_box_own_centre():
    """Rotating about the edited box's centre would sweep the track round an arc."""
    boxes = {f: ([0, 0, 0, 0], [100.0 + f * 50, 200.0, 20.0, 8.0, 0.0])
             for f in range(5)}
    out = propagation.plan(
        propagation.Correction(
            frame=0, track_id=1, before_bbox=[90, 196, 110, 204],
            after_bbox=[90, 196, 110, 204],
            before_obb=[100.0, 200.0, 20.0, 8.0, 0.0],
            after_obb=[100.0, 200.0, 20.0, 8.0, math.radians(30)]),
        propagation.PropagationScope(kind=propagation.TO_END),
        boxes=boxes, num_frames=5)
    for edit in out.edits:
        original = boxes[edit.frame][1]
        assert edit.obb[0] == pytest.approx(original[0]), "centre must not move"
        assert edit.obb[4] == pytest.approx(math.radians(30))


def test_propagated_bbox_stays_the_obb_bounding_rectangle():
    """The annotator's invariant: every consumer that reads only bbox keeps
    working whether or not the box is rotated."""
    boxes = {1: ([90, 196, 110, 204], [100.0, 200.0, 20.0, 8.0, 0.0])}
    out = propagation.plan(
        propagation.Correction(
            frame=0, track_id=1, before_bbox=[90, 196, 110, 204],
            after_bbox=[90, 196, 110, 204],
            before_obb=[100.0, 200.0, 20.0, 8.0, 0.0],
            after_obb=[100.0, 200.0, 20.0, 8.0, math.radians(90)]),
        propagation.PropagationScope(kind=propagation.TO_END),
        boxes=boxes, num_frames=5)
    edit = out.edits[0]
    assert (edit.bbox[2] - edit.bbox[0]) == pytest.approx(8.0, abs=0.01)
    assert (edit.bbox[3] - edit.bbox[1]) == pytest.approx(20.0, abs=0.01)


def test_a_collapsing_scale_is_dropped_rather_than_written():
    boxes = {f: ([100, 100, 120, 110], None) for f in range(5)}
    out = propagation.plan(
        propagation.Correction(frame=0, track_id=1,
                               before_bbox=[100, 100, 120, 110],
                               after_bbox=[109, 104, 111, 106]),
        propagation.PropagationScope(kind=propagation.TO_END),
        boxes=boxes, num_frames=5, min_side=5.0)
    assert out.count == 0


def test_obb_pair_is_preferred_over_the_derived_aabb():
    """An AABB derived from a rotated box changes size when the box merely
    turns; reading that as a resize would make every propagated box grow."""
    correction = propagation.Correction(
        frame=0, track_id=1,
        before_bbox=[90, 196, 110, 204], after_bbox=[96, 190, 104, 210],
        before_obb=[100.0, 200.0, 20.0, 8.0, 0.0],
        after_obb=[100.0, 200.0, 20.0, 8.0, math.radians(90)])
    transform = correction.transform()
    assert transform.rotated
    assert not transform.resized, "a pure rotation is not a resize"


# ══════════════════════════════════════════════════════════════════════════
# cache
# ══════════════════════════════════════════════════════════════════════════

def test_cache_round_trip(tmp_path):
    store = cache.DiskCache(tmp_path)
    key = cache.CacheKey.for_video(tmp_path / "missing.mp4", model="stage1",
                                   stride=15)
    assert store.get(key, "features") is None
    assert store.put(key, "features", [{"frame": 0}])
    assert store.get(key, "features") == [{"frame": 0}]


def test_changing_any_input_changes_the_key(tmp_path):
    base = cache.CacheKey.for_video(tmp_path / "v.mp4", model="a", stride=15)
    assert base.digest() != cache.CacheKey.for_video(
        tmp_path / "v.mp4", model="b", stride=15).digest()
    assert base.digest() != base.with_settings(stride=30).digest()
    assert base.digest() == cache.CacheKey.for_video(
        tmp_path / "v.mp4", model="a", stride=15).digest()


def test_a_modified_video_misses_the_cache(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"first")
    before = cache.CacheKey.for_video(video)
    video.write_bytes(b"second and longer")
    assert cache.CacheKey.for_video(video).digest() != before.digest()


def test_a_corrupt_entry_reads_as_a_miss_not_as_a_crash(tmp_path):
    store = cache.DiskCache(tmp_path)
    key = cache.CacheKey.for_video(tmp_path / "v.mp4")
    store.put(key, "features", [1, 2, 3])
    store.path_for(key, "features").write_text("{not json", encoding="utf-8")
    assert store.get(key, "features") is None


def test_prune_keeps_the_newest(tmp_path):
    store = cache.DiskCache(tmp_path)
    for i in range(5):
        store.put(cache.CacheKey.for_video(tmp_path / "v.mp4", stride=i),
                  "features", [i])
    assert store.prune(keep=2) == 3
    assert len(store.entries()) == 2


def test_cache_disables_itself_rather_than_failing_every_write(tmp_path):
    store = cache.DiskCache(tmp_path / "nope")
    key = cache.CacheKey.for_video(tmp_path / "v.mp4")
    store.disable("test")
    assert store.put(key, "features", [1]) is False
    assert store.get(key, "features") is None


def test_clear_is_safe_when_nothing_exists(tmp_path):
    assert cache.DiskCache(tmp_path / "absent").clear() is True


def test_infinite_distance_survives_serialisation(tmp_path):
    """FrameFeature.min_distance is inf with fewer than two detections, and a
    bare float('inf') is not valid JSON for every reader."""
    store = cache.DiskCache(tmp_path)
    key = cache.CacheKey.for_video(tmp_path / "v.mp4")
    feature = features.FrameFeature(frame=0, min_distance=math.inf)
    assert store.put(key, "features", [feature])
    restored = store.get(key, "features")
    assert restored[0]["frame"] == 0
