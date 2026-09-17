"""
quality.py — the detectors that turn 18,000 predictions into 94 things to look at.

This module answers one question: *given a run of machine predictions and
nothing else, which moments would a careful human want to see?* It is
unsupervised by construction — there is no ground truth at review time, so
every detector here works from internal consistency (a track that teleports, a
box that doubles in area, two animals where the experiment has two but the
detector found one) rather than from correctness.

Three rules the detectors obey, and the reasons they are rules
--------------------------------------------------------------

**1. A detector emits intervals, never per-frame items.** A fly that is
occluded for 40 frames is *one* thing to look at, not 40. Every detector runs
its per-frame test and then merges consecutive hits through :func:`_runs`. This
is the difference between a queue with 94 entries and a queue with 18,000,
which is the difference between a feature and a list.

**2. A detector never guesses which side is wrong.** ``ID_SWITCH_RISK`` says
two tracks crossed and the identities are now ambiguous; it does not say they
swapped. The human decides, in one click, through
:mod:`htrtdetr.annotation.review_queue`. Guessing here would mean silently
rewriting track ids on evidence that was never strong enough — and a wrong
guess that the human then has to find and undo is worse than no guess.

**3. Every threshold is in :class:`QualityConfig`, and the defaults are
scale-free where they can be.** A position jump is measured in multiples of
that track's own median step, not in pixels, so the same config works on a
1024x1024 arena and a 4K one. Where a bare number is unavoidable it is a
config field with its unit in the name, because *Drosophila* in a 35 mm
chamber and mice in an open field do not share a constant
(``docs/GUI_DESIGN.md`` P12: refuse to let a number look more meaningful than
it is, and record the measurement next to the threshold).

Purity: no dearpygui, no torch, no cv2, no numpy. Candidates are duck-typed on
``bbox`` / ``obb`` / ``score`` / ``track_id``, so this runs on
:class:`htrtdetr.gui.auto_label.Candidate`, on annotator objects, and on plain
namespaces in tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# auto_label is documented and tested as pure Python (no dearpygui, no torch,
# cv2 lazy). Re-implementing polygon IoU here would be a second copy of the
# subtlest geometry in the suite, free to drift from the one the merge path
# uses. Import it instead.
from ..gui.auto_label import iou_any, obb_to_xyxy
from .tracklet import Tracklet, build_tracklets

__all__ = [
    # codes
    "LOW_CONFIDENCE",
    "MISSING_ANIMAL",
    "EXTRA_ANIMAL",
    "ID_SWITCH_RISK",
    "POSITION_JUMP",
    "SIZE_JUMP",
    "ORIENTATION_JUMP",
    "TRACK_FRAGMENT",
    "OVERLAP",
    "MERGED_DETECTION",
    "BEHAVIOR_DISAGREEMENT",
    "BEHAVIOR_TRANSITION",
    "UNUSUAL_APPEARANCE",
    "MODEL_DISAGREEMENT",
    "ISSUE_CODES",
    "ISSUE_LABELS",
    "ISSUE_HELP",
    "ISSUE_ORDER",
    # data
    "ReviewItem",
    "QualityConfig",
    # detectors
    "detect_low_confidence",
    "detect_count_mismatch",
    "detect_position_jumps",
    "detect_size_jumps",
    "detect_orientation_jumps",
    "detect_overlaps",
    "detect_merged_detections",
    "detect_track_fragments",
    "detect_id_switch_risk",
    "analyze",
    "DETECTORS",
    "TRACK_DEPENDENT",
    "detector_keys",
    # shared helper
    "ramp",
]

# ══════════════════════════════════════════════════════════════════════════
# Issue vocabulary
# ══════════════════════════════════════════════════════════════════════════

LOW_CONFIDENCE = "low_confidence"
MISSING_ANIMAL = "missing_animal"
EXTRA_ANIMAL = "extra_animal"
ID_SWITCH_RISK = "id_switch_risk"
POSITION_JUMP = "position_jump"
SIZE_JUMP = "size_jump"
ORIENTATION_JUMP = "orientation_jump"
TRACK_FRAGMENT = "track_fragment"
OVERLAP = "overlap"
MERGED_DETECTION = "merged_detection"
BEHAVIOR_DISAGREEMENT = "behavior_disagreement"
BEHAVIOR_TRANSITION = "behavior_transition"
UNUSUAL_APPEARANCE = "unusual_appearance"
MODEL_DISAGREEMENT = "model_disagreement"

#: Queue order: the cheapest decisions first, so a human working top-down
#: builds momentum before reaching the ones that need scrubbing.
ISSUE_ORDER: Tuple[str, ...] = (
    MISSING_ANIMAL,
    ID_SWITCH_RISK,
    MERGED_DETECTION,
    EXTRA_ANIMAL,
    POSITION_JUMP,
    SIZE_JUMP,
    ORIENTATION_JUMP,
    TRACK_FRAGMENT,
    OVERLAP,
    LOW_CONFIDENCE,
    BEHAVIOR_DISAGREEMENT,
    BEHAVIOR_TRANSITION,
    UNUSUAL_APPEARANCE,
    MODEL_DISAGREEMENT,
)

ISSUE_CODES: Tuple[str, ...] = ISSUE_ORDER

#: The headline a human reads before deciding. Sentence case, no trailing
#: period (§4.1), and short enough to be taken in at a glance during a review
#: pass where it will be read ninety-four times.
#:
#: ``missing_animal`` and ``extra_animal`` stay separate rather than collapsing
#: into one "animal count mismatch": they demand opposite actions - look for an
#: animal that was missed, or look for a box on a reflection - and a label that
#: does not say which costs a second of thought every single time.
ISSUE_LABELS: Dict[str, str] = {
    LOW_CONFIDENCE: "Low confidence detection",
    MISSING_ANIMAL: "Missing animal",
    EXTRA_ANIMAL: "Extra animal",
    ID_SWITCH_RISK: "Possible ID switch",
    POSITION_JUMP: "Sudden position jump",
    SIZE_JUMP: "Sudden box size change",
    ORIENTATION_JUMP: "Sudden orientation change",
    TRACK_FRAGMENT: "Track stops and restarts",
    OVERLAP: "Animals overlapping",
    MERGED_DETECTION: "Possible merged detection",
    BEHAVIOR_DISAGREEMENT: "Behavior disagreement",
    BEHAVIOR_TRANSITION: "Uncertain behavior boundary",
    UNUSUAL_APPEARANCE: "Unusual appearance",
    MODEL_DISAGREEMENT: "Models disagree",
}

#: What the human is being asked to check. Shown under the reason in Review
#: Mode, where it is the only instruction on screen.
ISSUE_HELP: Dict[str, str] = {
    LOW_CONFIDENCE: "The model was unsure. Check the box is on an animal and fits it.",
    MISSING_ANIMAL: "Fewer animals than this project expects. Look for one that was not detected.",
    EXTRA_ANIMAL: "More animals than this project expects. Look for a box on a reflection or on debris.",
    ID_SWITCH_RISK: "Two animals met here. Check the identities are the same after as before.",
    POSITION_JUMP: "A box moved much further in one frame than this track usually moves.",
    SIZE_JUMP: "A box changed area sharply. It may have grown onto a neighbour or shrunk onto a part.",
    ORIENTATION_JUMP: "A rotated box turned sharply. Check the body axis, not the head direction.",
    TRACK_FRAGMENT: "This track stops and restarts. Check it is the same animal on both sides.",
    OVERLAP: "Two boxes overlap heavily. Check they are two animals and not one animal twice.",
    MERGED_DETECTION: "One box is large enough to hold two animals. Check whether two were merged.",
    BEHAVIOR_DISAGREEMENT: "The behavior label disagrees with the frames around it.",
    BEHAVIOR_TRANSITION: "The model was unsure where this behavior starts or ends.",
    UNUSUAL_APPEARANCE: "This frame does not look like the rest of the video.",
    MODEL_DISAGREEMENT: "Two models produced different answers here.",
}


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ReviewItem:
    """One place in the video a human should look, and why.

    Attributes:
        kind: One of the issue codes above.
        start: First frame of the interval, inclusive.
        end: Last frame, **inclusive**. ``start == end`` is a single frame.
        track_ids: The tracks involved, ascending. Empty for whole-frame
            issues such as a count mismatch, where no single track is at fault.
        severity: ``0.0``-``1.0``, higher is worse. Used for sorting and for
            the queue's severity filter; it is deliberately **not** shown as a
            number in the UI, because a 0.63 invites a precision the detectors
            do not have (P12).
        detail: One English sentence with the actual measurement in it, e.g.
            ``"Moved 88 px in one frame; this track usually moves 9 px."``
            This is the line the human reads before deciding.
    """

    kind: str
    start: int
    end: int
    track_ids: Tuple[int, ...] = ()
    severity: float = 0.5
    detail: str = ""
    video_id: str = ""

    @property
    def span(self) -> int:
        """Frames covered, inclusive."""
        return self.end - self.start + 1

    @property
    def label(self) -> str:
        return ISSUE_LABELS.get(self.kind, self.kind)

    @property
    def help(self) -> str:
        return ISSUE_HELP.get(self.kind, "")

    @property
    def key(self) -> Tuple[str, int, int, Tuple[int, ...]]:
        """Stable identity, so a re-analysis can keep decisions already made.

        Severity and wording are excluded on purpose: tuning a threshold or
        fixing a typo in ``detail`` must not orphan a human's Accept.
        """
        local = (self.kind, self.start, self.end, self.track_ids)
        return (self.video_id, *local) if self.video_id else local

    def middle(self) -> int:
        """The frame to jump to. Midpoint, so the cause is on screen rather
        than just off the left edge."""
        return (self.start + self.end) // 2

    def context(self, pad: int, num_frames: int = 0) -> Tuple[int, int]:
        """``(first, last)`` for a loop that shows before, during and after.

        ``pad`` frames each side, clamped to the video. This is what Review
        Mode plays: without the before-and-after, an ID switch is impossible
        to judge (§15).
        """
        lo = max(0, self.start - int(pad))
        hi = self.end + int(pad)
        if num_frames > 0:
            hi = min(hi, num_frames - 1)
        return lo, max(lo, hi)

    def describe(self, fps: float = 0.0) -> str:
        """``Possible ID switch  frames 1400-1412 (0.5 s)  tracks 2, 3``"""
        head = f"{self.label}  frames {self.start}"
        if self.end != self.start:
            head += f"-{self.end}"
        if fps > 0:
            head += f" ({self.span / fps:.1f} s)"
        if self.track_ids:
            ids = ", ".join(str(t) for t in self.track_ids)
            head += f"  track{'s' if len(self.track_ids) != 1 else ''} {ids}"
        return head


@dataclass(frozen=True)
class QualityConfig:
    """Every threshold the detectors use, in one overridable place.

    Defaults are tuned for *Drosophila* in a small arena at 25-30 fps, which is
    YOAKE's primary case, and every one of them is a guess that a different
    preparation should change. They are not constants and no detector may
    inline one.
    """

    # ── experiment ───────────────────────────────────────────────────────
    #: ``None`` = variable. An integer turns on the count detectors (§14); it
    #: is the single highest-value setting in this file, because "this
    #: experiment has exactly two flies" is knowledge the detector does not
    #: have and the human always does.
    expected_animals: Optional[int] = None

    # ── confidence ───────────────────────────────────────────────────────
    #: Below this a detection is queued as LOW_CONFIDENCE.
    low_score: float = 0.45
    #: At or above this an unflagged detection reads as PREDICTED (see
    #: :mod:`htrtdetr.annotation.status`).
    confident_score: float = 0.70

    # ── motion ───────────────────────────────────────────────────────────
    #: A step is a jump when it exceeds this many times the track's own median
    #: step. 6x is roughly "went further in one frame than in the previous six".
    position_jump_factor: float = 6.0
    #: Floor for the baseline, as a fraction of the median box diagonal. Stops
    #: a stationary animal (median step ~0 px) from flagging every twitch.
    position_jump_floor_frac: float = 0.35

    # ── shape ────────────────────────────────────────────────────────────
    #: Area ratio between consecutive frames, either direction.
    size_jump_factor: float = 1.8
    #: Rotated-box turn in one frame. Measured mod 180 degrees because the
    #: model's double-angle encoding cannot represent a head-tail flip, so a
    #: 179-degree "turn" is a 1-degree turn (see ``geometry/obb.py``).
    orientation_jump_deg: float = 55.0

    # ── interaction ──────────────────────────────────────────────────────
    #: IoU at which two boxes count as overlapping.
    overlap_iou: float = 0.30
    #: Centre distance, in multiples of the mean box diagonal, at which two
    #: animals count as "in contact" for ID-switch risk.
    contact_distance_frac: float = 0.90
    #: Frames of before/after kept around a contact event.
    contact_pad: int = 8
    #: A box this many times the median area may hold two animals.
    merged_area_factor: float = 1.75

    # ── tracks ───────────────────────────────────────────────────────────
    #: A gap longer than this inside one track id is TRACK_FRAGMENT.
    fragment_gap: int = 2
    #: Tracklets shorter than this are noise the queue should not spend a
    #: human on.
    min_tracklet_length: int = 2

    # ── merging ──────────────────────────────────────────────────────────
    #: Frames of separation below which two hits of the same kind on the same
    #: track become one queue entry.
    merge_gap: int = 3
    #: Items below this severity are built but not queued by default.
    min_severity: float = 0.0
    #: Hard cap per kind, worst first. A pathological run must not produce a
    #: 50,000-row queue that no UI can render and no human can face.
    max_items_per_kind: int = 500

    def replace(self, **kw: Any) -> "QualityConfig":
        """A copy with fields overridden — the config is frozen on purpose."""
        return replace(self, **kw)


# ══════════════════════════════════════════════════════════════════════════
# Geometry helpers (pure Python — no numpy on the GUI thread)
# ══════════════════════════════════════════════════════════════════════════


def _bbox(cand: Any) -> Optional[List[float]]:
    """AABB of a candidate, from ``obb`` when present, else ``bbox``."""
    obb = getattr(cand, "obb", None)
    if obb is not None:
        try:
            return [float(v) for v in obb_to_xyxy(obb)]
        except (TypeError, ValueError, IndexError):
            pass
    box = getattr(cand, "bbox", None)
    if box is None:
        return None
    try:
        return [float(v) for v in box[:4]]
    except (TypeError, ValueError, IndexError):
        return None


def _center(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _diag(box: Sequence[float]) -> float:
    return math.hypot(max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def _theta(cand: Any) -> Optional[float]:
    obb = getattr(cand, "obb", None)
    if obb is None:
        return None
    try:
        return float(obb[4])
    except (TypeError, ValueError, IndexError):
        return None


def _angle_delta(a: float, b: float) -> float:
    """Smallest turn between two body axes, in degrees, **mod 180**.

    Orientation lives mod pi under the double-angle encoding the detector and
    the tracker both use, so a box that reads 179 degrees and one that reads 1
    degree describe the same axis and must differ by 2, not by 178. Getting
    this wrong makes every stationary animal look like it spins once a frame.
    """
    d = math.degrees(a - b) % 180.0
    return min(d, 180.0 - d)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def ramp(value: float, soft: float, hard: float) -> float:
    """Map ``value`` onto ``0..1`` between a soft and a hard threshold.

    Below ``soft`` is 0, at or above ``hard`` is 1, linear between. Used so
    severity means "how far past the threshold" rather than "the raw number",
    which is what makes severities from different detectors comparable enough
    to sort one queue.
    """
    if hard <= soft:
        return 1.0 if value >= hard else 0.0
    return max(0.0, min(1.0, (value - soft) / (hard - soft)))


# ══════════════════════════════════════════════════════════════════════════
# Run merging — the reason the queue has 94 rows and not 18,000
# ══════════════════════════════════════════════════════════════════════════


def _runs(
    hits: Sequence[Tuple[int, float, str]],
    *,
    kind: str,
    track_ids: Tuple[int, ...],
    max_gap: int,
) -> List[ReviewItem]:
    """Merge ``(frame, severity, detail)`` hits into interval items.

    The merged item keeps the **worst** hit's severity and its detail, because
    that is the frame the human will be shown and the number they will judge.
    Keeping the first hit's instead would describe an interval by its mildest
    moment.
    """
    if not hits:
        return []
    ordered = sorted(hits, key=lambda h: h[0])
    out: List[ReviewItem] = []
    start = prev = ordered[0][0]
    best_sev, best_detail = ordered[0][1], ordered[0][2]

    def flush() -> None:
        out.append(
            ReviewItem(
                kind=kind,
                start=start,
                end=prev,
                track_ids=track_ids,
                severity=best_sev,
                detail=best_detail,
            )
        )

    for frame, sev, detail in ordered[1:]:
        if frame - prev > max_gap:
            flush()
            start, best_sev, best_detail = frame, sev, detail
        elif sev > best_sev:
            best_sev, best_detail = sev, detail
        prev = frame
    flush()
    return out


def _cap(items: List[ReviewItem], limit: int) -> List[ReviewItem]:
    """Worst ``limit`` items of one kind. Ties broken by frame so the cut is
    deterministic across runs with the same seed."""
    if limit <= 0 or len(items) <= limit:
        return items
    return sorted(items, key=lambda i: (-i.severity, i.start))[:limit]


# ══════════════════════════════════════════════════════════════════════════
# Detectors
# ══════════════════════════════════════════════════════════════════════════


def detect_low_confidence(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
) -> List[ReviewItem]:
    """Runs of frames where a track's detection was weak."""
    per_track: Dict[int, List[Tuple[int, float, str]]] = {}
    for frame in sorted(candidates):
        for cand in candidates[frame] or ():
            score = float(getattr(cand, "score", 1.0) or 0.0)
            if score >= cfg.low_score:
                continue
            tid = int(getattr(cand, "track_id", -1))
            sev = ramp(cfg.low_score - score, 0.0, cfg.low_score)
            per_track.setdefault(tid, []).append(
                (frame, sev, f"Confidence {score:.2f}, below {cfg.low_score:.2f}.")
            )
    out: List[ReviewItem] = []
    for tid, hits in per_track.items():
        ids = () if tid < 0 else (tid,)
        out.extend(_runs(hits, kind=LOW_CONFIDENCE, track_ids=ids, max_gap=cfg.merge_gap))
    return _cap(out, cfg.max_items_per_kind)


def detect_count_mismatch(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
    *,
    frames: Optional[Sequence[int]] = None,
) -> List[ReviewItem]:
    """Frames holding the wrong number of animals (§14).

    Needs ``cfg.expected_animals``; returns nothing without it, because with a
    variable animal count *every* frame is consistent with the design and
    flagging any of them would be noise.

    ``frames`` names the frames that were actually processed. Without it only
    frames present in ``candidates`` are checked — which silently misses the
    worst case, a frame where the detector found **nothing at all** and so
    wrote no entry.
    """
    expected = cfg.expected_animals
    if expected is None or expected < 0:
        return []

    scan = sorted(int(f) for f in frames) if frames is not None else sorted(candidates)
    missing: List[Tuple[int, float, str]] = []
    extra: List[Tuple[int, float, str]] = []

    for frame in scan:
        n = len(candidates.get(frame) or ())
        if n < expected:
            short = expected - n
            missing.append(
                (
                    frame,
                    ramp(short, 0.0, max(1.0, float(expected))),
                    f"Found {n} of {expected} expected animal{'s' if expected != 1 else ''}.",
                )
            )
        elif n > expected:
            over = n - expected
            extra.append(
                (
                    frame,
                    ramp(over, 0.0, max(1.0, float(expected))),
                    f"Found {n}, but this project expects {expected}.",
                )
            )

    out = _runs(missing, kind=MISSING_ANIMAL, track_ids=(), max_gap=cfg.merge_gap)
    out += _runs(extra, kind=EXTRA_ANIMAL, track_ids=(), max_gap=cfg.merge_gap)
    return _cap(out, cfg.max_items_per_kind)


def detect_position_jumps(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
    *,
    tracklets: Optional[Sequence[Tracklet]] = None,
) -> List[ReviewItem]:
    """A box that moved much further in one frame than its track usually does.

    The baseline is the track's **own** median step, floored at a fraction of
    its median box diagonal. Per-track rather than global because a courting
    male and a stationary female in the same video have step distributions an
    order of magnitude apart, and one global threshold would flag every frame
    of the male or none of the female.
    """
    tracks = (
        tracklets
        if tracklets is not None
        else build_tracklets(candidates, max_gap=cfg.fragment_gap)
    )
    out: List[ReviewItem] = []

    for t in tracks:
        if t.count < 3:
            continue  # two points define one step; there is no "usually" yet
        boxes: List[Tuple[int, List[float]]] = []
        for frame, index in t.targets():
            row = candidates.get(frame) or ()
            if index < len(row):
                box = _bbox(row[index])
                if box is not None:
                    boxes.append((frame, box))
        if len(boxes) < 3:
            continue

        steps: List[Tuple[int, float]] = []
        for (fa, ba), (fb, bb) in zip(boxes, boxes[1:]):
            ca, cb = _center(ba), _center(bb)
            gap = max(1, fb - fa)
            steps.append((fb, math.hypot(cb[0] - ca[0], cb[1] - ca[1]) / gap))

        typical = _median([s for _, s in steps])
        floor = cfg.position_jump_floor_frac * _median([_diag(b) for _, b in boxes])
        baseline = max(typical, floor)
        if baseline <= 0:
            continue
        limit = cfg.position_jump_factor * baseline

        hits = [
            (
                frame,
                ramp(step / baseline, cfg.position_jump_factor, cfg.position_jump_factor * 3.0),
                f"Moved {step:.0f} px in one frame; this track usually moves {typical:.0f} px.",
            )
            for frame, step in steps
            if step > limit
        ]
        out.extend(_runs(hits, kind=POSITION_JUMP, track_ids=(t.track_id,), max_gap=cfg.merge_gap))

    return _cap(out, cfg.max_items_per_kind)


def detect_size_jumps(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
    *,
    tracklets: Optional[Sequence[Tracklet]] = None,
) -> List[ReviewItem]:
    """A box whose area changed sharply between consecutive frames.

    An animal's apparent area changes slowly. A sudden growth is usually the
    box swallowing a neighbour; a sudden shrink is usually the box collapsing
    onto a body part. Both are worth one look.
    """
    tracks = (
        tracklets
        if tracklets is not None
        else build_tracklets(candidates, max_gap=cfg.fragment_gap)
    )
    out: List[ReviewItem] = []

    for t in tracks:
        rows: List[Tuple[int, float]] = []
        for frame, index in t.targets():
            row = candidates.get(frame) or ()
            if index < len(row):
                box = _bbox(row[index])
                if box is not None and _area(box) > 0:
                    rows.append((frame, _area(box)))
        if len(rows) < 2:
            continue

        hits: List[Tuple[int, float, str]] = []
        for (_, aa), (fb, ab) in zip(rows, rows[1:]):
            ratio = max(ab / aa, aa / ab)
            if ratio > cfg.size_jump_factor:
                direction = "grew" if ab > aa else "shrank"
                hits.append(
                    (
                        fb,
                        ramp(ratio, cfg.size_jump_factor, cfg.size_jump_factor * 2.5),
                        f"Box {direction} {ratio:.1f}x in one frame.",
                    )
                )
        out.extend(_runs(hits, kind=SIZE_JUMP, track_ids=(t.track_id,), max_gap=cfg.merge_gap))

    return _cap(out, cfg.max_items_per_kind)


def detect_orientation_jumps(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
    *,
    tracklets: Optional[Sequence[Tracklet]] = None,
) -> List[ReviewItem]:
    """A rotated box whose body axis turned sharply.

    Only runs where OBBs exist. Angles compare mod 180 degrees
    (see :func:`_angle_delta`); a head-tail flip is not representable in the
    encoding and is therefore not an error to report.
    """
    tracks = (
        tracklets
        if tracklets is not None
        else build_tracklets(candidates, max_gap=cfg.fragment_gap)
    )
    out: List[ReviewItem] = []

    for t in tracks:
        rows: List[Tuple[int, float]] = []
        for frame, index in t.targets():
            row = candidates.get(frame) or ()
            if index < len(row):
                th = _theta(row[index])
                if th is not None:
                    rows.append((frame, th))
        if len(rows) < 2:
            continue

        hits: List[Tuple[int, float, str]] = []
        for (_, ta), (fb, tb) in zip(rows, rows[1:]):
            turn = _angle_delta(tb, ta)
            if turn > cfg.orientation_jump_deg:
                hits.append(
                    (
                        fb,
                        ramp(turn, cfg.orientation_jump_deg, 90.0),
                        f"Body axis turned {turn:.0f} degrees in one frame.",
                    )
                )
        out.extend(
            _runs(hits, kind=ORIENTATION_JUMP, track_ids=(t.track_id,), max_gap=cfg.merge_gap)
        )

    return _cap(out, cfg.max_items_per_kind)


def detect_overlaps(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
) -> List[ReviewItem]:
    """Frames where two boxes overlap heavily.

    Grouped **per pair of tracks**, so one long contact between the same two
    animals is one queue entry and not one per frame.
    """
    per_pair: Dict[Tuple[int, int], List[Tuple[int, float, str]]] = {}

    for frame in sorted(candidates):
        row = list(candidates[frame] or ())
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                a, b = row[i], row[j]
                box_a, box_b = _bbox(a), _bbox(b)
                if box_a is None or box_b is None:
                    continue
                # Rotated IoU when both sides carry an angle: two flies lying
                # side by side have circumscribing rectangles that overlap
                # heavily while the animals do not touch at all, so the AABB
                # answer here would flag every pair in the arena.
                iou = float(iou_any(box_a, box_b, getattr(a, "obb", None), getattr(b, "obb", None)))
                if iou < cfg.overlap_iou:
                    continue
                ta = int(getattr(a, "track_id", -1))
                tb = int(getattr(b, "track_id", -1))
                pair = tuple(sorted((ta, tb)))
                per_pair.setdefault(pair, []).append(  # type: ignore[arg-type]
                    (frame, ramp(iou, cfg.overlap_iou, 0.9), f"Boxes overlap {iou * 100:.0f}%.")
                )

    out: List[ReviewItem] = []
    for pair, hits in per_pair.items():
        ids = tuple(t for t in pair if t >= 0)
        out.extend(_runs(hits, kind=OVERLAP, track_ids=ids, max_gap=cfg.merge_gap))
    return _cap(out, cfg.max_items_per_kind)


def detect_merged_detections(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
) -> List[ReviewItem]:
    """One box big enough to hold two animals (§14).

    The baseline is the median box area over the **whole run**, which is the
    only robust estimate of "one animal" available without a model of the
    species. When ``expected_animals`` is set and the frame is also short by
    one, severity is raised: a box of double area in a frame that lost an
    animal is the textbook merge, and it should outrank a merely large box.
    """
    areas: List[float] = []
    for row in candidates.values():
        for cand in row or ():
            box = _bbox(cand)
            if box is not None and _area(box) > 0:
                areas.append(_area(box))
    typical = _median(areas)
    if typical <= 0:
        return []

    limit = cfg.merged_area_factor * typical
    expected = cfg.expected_animals
    per_track: Dict[int, List[Tuple[int, float, str]]] = {}

    for frame in sorted(candidates):
        row = list(candidates[frame] or ())
        short = expected is not None and len(row) < expected
        for cand in row:
            box = _bbox(cand)
            if box is None:
                continue
            area = _area(box)
            if area <= limit:
                continue
            ratio = area / typical
            sev = ramp(ratio, cfg.merged_area_factor, cfg.merged_area_factor * 2.0)
            detail = f"Box is {ratio:.1f}x the usual area."
            if short:
                sev = min(1.0, sev + 0.3)
                detail += " One animal is also missing from this frame."
            per_track.setdefault(int(getattr(cand, "track_id", -1)), []).append(
                (frame, sev, detail)
            )

    out: List[ReviewItem] = []
    for tid, hits in per_track.items():
        ids = () if tid < 0 else (tid,)
        out.extend(_runs(hits, kind=MERGED_DETECTION, track_ids=ids, max_gap=cfg.merge_gap))
    return _cap(out, cfg.max_items_per_kind)


def detect_track_fragments(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
    *,
    tracklets: Optional[Sequence[Tracklet]] = None,
) -> List[ReviewItem]:
    """A track id that stops and restarts.

    One item per gap, covering the gap itself, so jumping to it puts the human
    on the frames where the animal was not detected rather than on the frames
    where it was.
    """
    tracks = (
        tracklets
        if tracklets is not None
        else build_tracklets(candidates, max_gap=cfg.fragment_gap)
    )
    by_track: Dict[int, List[Tracklet]] = {}
    for t in tracks:
        if t.track_id >= 0 and t.count >= cfg.min_tracklet_length:
            by_track.setdefault(t.track_id, []).append(t)

    out: List[ReviewItem] = []
    for tid, parts in by_track.items():
        parts.sort(key=lambda t: t.start)
        for a, b in zip(parts, parts[1:]):
            gap = b.start - a.end - 1
            if gap <= 0:
                continue
            out.append(
                ReviewItem(
                    kind=TRACK_FRAGMENT,
                    start=a.end + 1,
                    end=b.start - 1,
                    track_ids=(tid,),
                    severity=ramp(gap, cfg.fragment_gap, cfg.fragment_gap + 30.0),
                    detail=f"Track {tid} is missing for {gap} frame{'s' if gap != 1 else ''}.",
                )
            )
    return _cap(out, cfg.max_items_per_kind)


def detect_id_switch_risk(
    candidates: Mapping[int, Sequence[Any]],
    cfg: QualityConfig,
) -> List[ReviewItem]:
    """Contact events where identities could have been exchanged (§15).

    A contact is a run of frames where two tracks are closer than
    ``contact_distance_frac`` of a box diagonal, or overlapping. The item
    covers the contact **plus ``contact_pad`` frames each side**, because the
    decision is not about the contact — it is about whether who-is-who is the
    same before and after, and that cannot be seen without both.

    The detector reports risk only. Deciding, and swapping, belongs to
    :meth:`htrtdetr.annotation.review_queue.ReviewQueue.swap_decision`.
    """
    per_pair: Dict[Tuple[int, int], List[Tuple[int, float, str]]] = {}

    for frame in sorted(candidates):
        row = [c for c in (candidates[frame] or ()) if int(getattr(c, "track_id", -1)) >= 0]
        boxes = [(int(getattr(c, "track_id", -1)), _bbox(c)) for c in row]
        boxes = [(t, b) for t, b in boxes if b is not None]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ta, ba = boxes[i]
                tb, bb = boxes[j]
                if ta == tb:
                    continue
                ca, cb = _center(ba), _center(bb)
                dist = math.hypot(cb[0] - ca[0], cb[1] - ca[1])
                reach = cfg.contact_distance_frac * (_diag(ba) + _diag(bb)) / 2.0
                if reach <= 0 or dist > reach:
                    continue
                closeness = 1.0 - (dist / reach)
                per_pair.setdefault((min(ta, tb), max(ta, tb)), []).append(
                    (
                        frame,
                        ramp(closeness, 0.0, 1.0),
                        f"Tracks {min(ta, tb)} and {max(ta, tb)} came within "
                        f"{dist:.0f} px of each other.",
                    )
                )

    out: List[ReviewItem] = []
    for (ta, tb), hits in per_pair.items():
        # A wider merge gap than elsewhere: two animals that touch, part for a
        # few frames and touch again are one ambiguity, not three.
        for item in _runs(
            hits,
            kind=ID_SWITCH_RISK,
            track_ids=(ta, tb),
            max_gap=max(cfg.merge_gap, cfg.contact_pad),
        ):
            out.append(
                replace(
                    item, start=max(0, item.start - cfg.contact_pad), end=item.end + cfg.contact_pad
                )
            )
    return _cap(out, cfg.max_items_per_kind)


# ══════════════════════════════════════════════════════════════════════════
# The whole pass
# ══════════════════════════════════════════════════════════════════════════

#: Detector registry. A caller turns one off by name rather than by editing
#: :func:`analyze`, and a future detector is added here and nowhere else.
DETECTORS: Dict[str, Any] = {
    LOW_CONFIDENCE: detect_low_confidence,
    "count": detect_count_mismatch,
    POSITION_JUMP: detect_position_jumps,
    SIZE_JUMP: detect_size_jumps,
    ORIENTATION_JUMP: detect_orientation_jumps,
    OVERLAP: detect_overlaps,
    MERGED_DETECTION: detect_merged_detections,
    TRACK_FRAGMENT: detect_track_fragments,
    ID_SWITCH_RISK: detect_id_switch_risk,
}

#: Detectors that accept a pre-built ``tracklets=`` argument, so the pass
#: builds tracklets once instead of once per detector.
_TRACKLET_AWARE = frozenset({POSITION_JUMP, SIZE_JUMP, ORIENTATION_JUMP, TRACK_FRAGMENT})

#: Detectors that have no meaning without track ids. Every one of them asks a
#: question of the form "did *this animal* change between these two frames",
#: and a project that does not follow individuals has not said which box is
#: which animal. Running them anyway does not produce wrong answers — an
#: untracked candidate becomes its own one-frame tracklet, which is below
#: ``min_tracklet_length`` — it produces a check that can never fire, listed in
#: the UI as though it could.
TRACK_DEPENDENT = frozenset(
    {POSITION_JUMP, SIZE_JUMP, ORIENTATION_JUMP, TRACK_FRAGMENT, ID_SWITCH_RISK}
)


def detector_keys(*, use_track_ids: bool = True) -> Tuple[str, ...]:
    """The detectors worth running, for a project that does or does not track.

    Pass the result as :func:`analyze`'s ``enabled``. Confidence, animal count,
    overlap and merged-box checks work on one frame at a time and stay on
    either way — they are the ones that matter most when there are no ids.
    """
    return tuple(k for k in DETECTORS if use_track_ids or k not in TRACK_DEPENDENT)


def analyze(
    candidates: Mapping[int, Sequence[Any]],
    *,
    config: Optional[QualityConfig] = None,
    frames: Optional[Sequence[int]] = None,
    enabled: Optional[Iterable[str]] = None,
    tracklets: Optional[Sequence[Tracklet]] = None,
    num_frames: int = 0,
) -> List[ReviewItem]:
    """Run every enabled detector and return one sorted queue.

    Args:
        candidates: ``{frame_index: [candidate, ...]}``.
        config: Thresholds. Defaults to :class:`QualityConfig`'s defaults.
        frames: Every frame that was processed, including ones with no
            detections. Pass it whenever you know it — without it, a frame
            where the detector found nothing is invisible to the count
            detectors, which is precisely the case they exist for.
        enabled: Detector keys from :data:`DETECTORS`. ``None`` runs all.
        tracklets: Pre-built tracklets, reused rather than rebuilt.
        num_frames: Length of the video. Used to clamp items whose context
            padding runs off the end — an item ending at frame 62 of a 60-frame
            video makes the timeline draw off its own right edge and makes
            "jump here" seek past the last frame.

    Returns:
        Items sorted by ``(kind rank, -severity, start)`` — grouped by reason
        so the filter chips have contiguous blocks, worst first inside each.
    """
    cfg = config or QualityConfig()
    keys = list(DETECTORS) if enabled is None else [k for k in enabled if k in DETECTORS]
    if tracklets is None and any(k in _TRACKLET_AWARE for k in keys):
        tracklets = build_tracklets(candidates, max_gap=cfg.fragment_gap)

    out: List[ReviewItem] = []
    for key in keys:
        fn = DETECTORS[key]
        try:
            if key in _TRACKLET_AWARE:
                out.extend(fn(candidates, cfg, tracklets=tracklets))
            elif key == "count":
                out.extend(fn(candidates, cfg, frames=frames))
            else:
                out.extend(fn(candidates, cfg))
        except Exception as exc:  # one broken detector must not lose the rest
            out.append(
                ReviewItem(
                    kind=key if key in ISSUE_LABELS else MODEL_DISAGREEMENT,
                    start=0,
                    end=0,
                    severity=0.0,
                    detail=f"This check could not run: {type(exc).__name__}: {exc}",
                )
            )

    if cfg.min_severity > 0:
        out = [i for i in out if i.severity >= cfg.min_severity]

    if num_frames > 0:
        last = int(num_frames) - 1
        out = [
            (
                i
                if (i.start >= 0 and i.end <= last)
                else replace(i, start=max(0, i.start), end=min(last, i.end))
            )
            for i in out
            if i.start <= last
        ]

    rank = {k: n for n, k in enumerate(ISSUE_ORDER)}
    out.sort(key=lambda i: (rank.get(i.kind, len(rank)), -i.severity, i.start))
    return out
