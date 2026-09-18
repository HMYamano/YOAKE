"""
intervals.py — behavior as spans of time, not as a label on every frame.

A behavior is not a property of a frame. *Courtship* happens from 01:24.20 to
01:31.85; the frames in the middle are not each separately courting. Annotating
it frame by frame asks a human to make 190 decisions where there are two — the
start and the end — and then makes them re-make all 190 whenever a boundary
moves.

So the editing unit here is the interval::

    Courtship   01:24.20 - 01:31.85   track 2   conf 0.91
    [ Accept ]  [ <- start ]  [ end -> ]  [ Relabel ]

**Storage does not change.** ``annotations.json`` keeps one ``action_id`` per
object per frame exactly as it always has, and :meth:`IntervalTrack.to_frame_actions`
projects intervals back onto it. Intervals are a *view*, built on load by
:func:`merge_to_intervals` and flattened on save. That is the whole
compatibility story: every existing file, every existing training path and
every existing analysis script keeps working, and no new file appears that
could desynchronise from the one that matters
(``docs/GUI_DESIGN.md`` §9.3 records what a second file costs).

Purity: no dearpygui, no torch, no cv2, no numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .quality import (
    BEHAVIOR_DISAGREEMENT, BEHAVIOR_TRANSITION, QualityConfig, ReviewItem, ramp,
)

__all__ = [
    "BehaviorInterval", "IntervalTrack",
    "merge_to_intervals", "detect_interval_issues", "timecode",
    "DEFAULT_MIN_FRAMES", "DEFAULT_MAX_GAP", "DEFAULT_SMOOTH_WINDOW",
    "UNLABELLED",
]

#: ``action_id`` meaning "not annotated" throughout YOAKE
#: (``data/annotation.py``: *-1 は未アノテーション*). Never becomes an interval.
UNLABELLED = -1

#: A run shorter than this is a flicker, not a behavior. 5 frames is 0.2 s at
#: 25 fps — shorter than any *Drosophila* behavior this project scores, and
#: long enough that a real brief event survives. A different preparation should
#: change it; it is an argument, not a constant.
DEFAULT_MIN_FRAMES = 5

#: Missing frames tolerated inside one interval. An animal lost for two frames
#: mid-behavior did not stop and restart the behavior.
DEFAULT_MAX_GAP = 2

#: Majority-filter width applied before run-length encoding. Odd, so there is
#: always a strict majority. ``0`` or ``1`` disables smoothing.
DEFAULT_SMOOTH_WINDOW = 5


def timecode(frame: int, fps: float) -> str:
    """``01:24.20`` — minutes, seconds, hundredths.

    The format the behavior literature uses, so an interval can be copied into
    a methods section without re-formatting. ``fps <= 0`` falls back to the
    frame number rather than dividing by zero, because a container that lies
    about its frame rate must not take the panel down.
    """
    if fps <= 0:
        return f"f{int(frame)}"
    total = frame / fps
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes:02d}:{seconds:05.2f}"


@dataclass(frozen=True)
class BehaviorInterval:
    """One behavior, one animal, one span of frames.

    Attributes:
        track_id: The animal. ``-1`` means the producer made no identity claim.
        action_id: Index into ``action_names``. Never :data:`UNLABELLED` — an
            unlabelled span is the *absence* of an interval, not an interval
            with a null label.
        start: First frame, inclusive.
        end: Last frame, **inclusive**.
        confidence: Mean model confidence over the span. ``1.0`` for a human
            interval, where there is nothing to be unconfident about.
        source: ``"human"`` once a human has set or confirmed the boundaries,
            ``"auto"`` while it is still a machine proposal. Same vocabulary
            and same meaning as ``AnnObject.source``.
        boundary_confidence: ``(start, end)`` confidence *at the edges*,
            which is what decides whether the boundary needs review. A span
            can be confidently *courtship* in the middle and completely
            ambiguous about where it began.
    """

    track_id: int
    action_id: int
    start: int
    end: int
    confidence: float = 1.0
    source: str = "auto"
    boundary_confidence: Tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Interval ends before it starts: {self.start} -> {self.end}")

    # ── extent ───────────────────────────────────────────────────────────
    @property
    def span(self) -> int:
        """Frames covered, inclusive. A 1-frame interval has span 1."""
        return self.end - self.start + 1

    def duration_seconds(self, fps: float) -> float:
        return self.span / fps if fps > 0 else 0.0

    def contains(self, frame: int) -> bool:
        return self.start <= int(frame) <= self.end

    def overlaps(self, other: "BehaviorInterval") -> bool:
        return self.start <= other.end and other.start <= self.end

    def frames(self) -> range:
        return range(self.start, self.end + 1)

    # ── editing — every one returns a new interval ───────────────────────
    def move_start(self, frame: int) -> "BehaviorInterval":
        """Drag the left boundary. Clamped so it cannot pass the right one."""
        return replace(self, start=min(int(frame), self.end), source="human")

    def move_end(self, frame: int) -> "BehaviorInterval":
        """Drag the right boundary. Clamped so it cannot pass the left one."""
        return replace(self, end=max(int(frame), self.start), source="human")

    def relabel(self, action_id: int) -> "BehaviorInterval":
        return replace(self, action_id=int(action_id), source="human")

    def confirm(self) -> "BehaviorInterval":
        """Accept as-is. Confidence becomes 1.0 because a human has now said so."""
        return replace(self, source="human", confidence=1.0,
                       boundary_confidence=(1.0, 1.0))

    def split(self, frame: int) -> Tuple["BehaviorInterval", "BehaviorInterval"]:
        """Cut at ``frame``; the cut frame starts the second half.

        Raises when the cut would produce an empty side, because an interval of
        zero frames is not a thing a caller can usefully be handed.
        """
        f = int(frame)
        if not (self.start < f <= self.end):
            raise ValueError(
                f"Cannot split {self.start}-{self.end} at {f}: "
                f"the cut must fall inside the interval and leave both sides non-empty.")
        return (replace(self, end=f - 1, source="human"),
                replace(self, start=f, source="human"))

    # ── presentation ─────────────────────────────────────────────────────
    def label(self, action_names: Sequence[str] = ()) -> str:
        if 0 <= self.action_id < len(action_names):
            return str(action_names[self.action_id])
        return f"action {self.action_id}"

    def describe(self, fps: float = 0.0, action_names: Sequence[str] = ()) -> str:
        """``Courtship  01:24.20 - 01:31.85  track 2  conf 0.91``"""
        name = self.label(action_names)
        if fps > 0:
            span = f"{timecode(self.start, fps)} - {timecode(self.end, fps)}"
        else:
            span = f"frames {self.start}-{self.end}"
        out = f"{name}  {span}"
        if self.track_id >= 0:
            out += f"  track {self.track_id}"
        if self.source != "human":
            out += f"  conf {self.confidence:.2f}"
        return out


# ══════════════════════════════════════════════════════════════════════════
# Building intervals from per-frame predictions
# ══════════════════════════════════════════════════════════════════════════

def _smooth(labels: List[int], window: int) -> List[int]:
    """Majority filter over ``window`` frames.

    Removes the single-frame flicker that per-frame classifiers always produce
    at behavior boundaries. Applied *before* run-length encoding, because the
    alternative — encoding first and cleaning up afterwards — has to invent a
    rule for merging three-way alternations and always gets one wrong.
    """
    if window < 3 or len(labels) < window:
        return list(labels)
    half = window // 2
    out: List[int] = []
    for i in range(len(labels)):
        lo, hi = max(0, i - half), min(len(labels), i + half + 1)
        counts: Dict[int, int] = {}
        for v in labels[lo:hi]:
            counts[v] = counts.get(v, 0) + 1
        best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == labels[i]))
        out.append(best[0])
    return out


def merge_to_intervals(
    candidates: Mapping[int, Sequence[Any]],
    *,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_gap: int = DEFAULT_MAX_GAP,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    min_confidence: float = 0.0,
    source: str = "auto",
) -> List[BehaviorInterval]:
    """Turn per-frame behavior predictions into interval candidates (§21).

    Args:
        candidates: ``{frame: [obj, ...]}`` where each object carries
            ``track_id``, ``action_id`` and optionally ``score``. Objects with
            ``action_id == UNLABELLED`` are ignored.
        min_frames: Runs shorter than this are absorbed into a neighbour when
            the neighbours agree, and dropped otherwise.
        max_gap: Missing frames tolerated inside one interval.
        smooth_window: Majority-filter width, applied per track.
        min_confidence: Frames scoring below this are treated as unlabelled,
            so a low-confidence stretch breaks an interval instead of silently
            extending it.
        source: Stamped on every interval produced.

    Returns:
        Intervals sorted by ``(start, track_id, action_id)``.
    """
    per_track: Dict[int, List[Tuple[int, int, float]]] = {}
    for frame in sorted(int(f) for f in candidates):
        for obj in candidates[frame] or ():
            action = int(getattr(obj, "action_id", UNLABELLED))
            if action == UNLABELLED or action < 0:
                continue
            score = float(getattr(obj, "score", 1.0) or 0.0)
            if score < min_confidence:
                continue
            tid = int(getattr(obj, "track_id", -1))
            per_track.setdefault(tid, []).append((frame, action, score))

    out: List[BehaviorInterval] = []
    for tid, rows in per_track.items():
        rows.sort(key=lambda r: r[0])
        # Split into contiguous segments first: smoothing across a 400-frame
        # gap would let a behavior from before the gap vote on frames after it.
        segment: List[Tuple[int, int, float]] = []
        for row in rows:
            if segment and row[0] - segment[-1][0] - 1 > max_gap:
                out.extend(_segment_intervals(tid, segment, min_frames,
                                              smooth_window, source))
                segment = []
            segment.append(row)
        if segment:
            out.extend(_segment_intervals(tid, segment, min_frames,
                                          smooth_window, source))

    out.sort(key=lambda i: (i.start, i.track_id, i.action_id))
    return out


def _segment_intervals(
    track_id: int,
    segment: Sequence[Tuple[int, int, float]],
    min_frames: int,
    smooth_window: int,
    source: str,
) -> List[BehaviorInterval]:
    """Run-length encode one contiguous segment of one track."""
    frames = [r[0] for r in segment]
    labels = _smooth([r[1] for r in segment], smooth_window)
    scores = [r[2] for r in segment]

    runs: List[List[int]] = []          # [label, first_pos, last_pos]
    for pos, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = pos
        else:
            runs.append([label, pos, pos])

    runs = _absorb_short(runs, min_frames)

    out: List[BehaviorInterval] = []
    for label, lo, hi in runs:
        window = scores[lo:hi + 1]
        mean = sum(window) / len(window) if window else 1.0
        out.append(BehaviorInterval(
            track_id=track_id,
            action_id=label,
            start=frames[lo],
            end=frames[hi],
            confidence=mean,
            source=source,
            boundary_confidence=(window[0] if window else 1.0,
                                 window[-1] if window else 1.0),
        ))
    return out


def _absorb_short(runs: List[List[int]], min_frames: int) -> List[List[int]]:
    """Fold runs shorter than ``min_frames`` into a neighbour, or drop them.

    A short run between two runs of the *same* label is a dropout and is
    absorbed, joining them. Between two *different* labels it is genuinely
    ambiguous, so it goes to the longer neighbour rather than to an arbitrary
    side; that keeps the result independent of scan direction.
    """
    if min_frames <= 1:
        return runs
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (label, lo, hi) in enumerate(runs):
            if hi - lo + 1 >= min_frames:
                continue
            prev = runs[i - 1] if i > 0 else None
            nxt = runs[i + 1] if i + 1 < len(runs) else None
            if prev is None and nxt is None:
                break
            if prev is not None and nxt is not None and prev[0] == nxt[0]:
                prev[2] = nxt[2]
                del runs[i:i + 2]
            elif prev is None:
                nxt[1] = lo
                del runs[i]
            elif nxt is None:
                prev[2] = hi
                del runs[i]
            else:
                target = prev if (prev[2] - prev[1]) >= (nxt[2] - nxt[1]) else nxt
                if target is prev:
                    prev[2] = hi
                else:
                    nxt[1] = lo
                del runs[i]
            changed = True
            break
    # A lone run shorter than min_frames survives: dropping it would silently
    # delete the only thing the model found in this segment.
    return runs


# ══════════════════════════════════════════════════════════════════════════
# The editable collection
# ══════════════════════════════════════════════════════════════════════════

class IntervalTrack:
    """Every behavior interval in one video, with the edits the timeline makes.

    Mutating methods return the new interval so a caller can select it
    immediately, and every mutation keeps the list sorted, so the timeline can
    render straight from :attr:`intervals` with no re-sort per frame.
    """

    def __init__(self, intervals: Iterable[BehaviorInterval] = ()) -> None:
        self._items: List[BehaviorInterval] = sorted(
            intervals, key=lambda i: (i.start, i.track_id, i.action_id))

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[BehaviorInterval]:
        return iter(self._items)

    @property
    def intervals(self) -> List[BehaviorInterval]:
        return list(self._items)

    # ── lookup ───────────────────────────────────────────────────────────
    def at_frame(self, frame: int, track_id: Optional[int] = None
                 ) -> List[BehaviorInterval]:
        f = int(frame)
        return [i for i in self._items
                if i.contains(f) and (track_id is None or i.track_id == track_id)]

    def for_track(self, track_id: int) -> List[BehaviorInterval]:
        return [i for i in self._items if i.track_id == int(track_id)]

    def tracks(self) -> List[int]:
        return sorted({i.track_id for i in self._items})

    def pending(self) -> List[BehaviorInterval]:
        """Intervals no human has confirmed — the behavior review worklist."""
        return [i for i in self._items if i.source != "human"]

    # ── mutation ─────────────────────────────────────────────────────────
    def add(self, interval: BehaviorInterval) -> BehaviorInterval:
        self._items.append(interval)
        self._sort()
        return interval

    def remove(self, interval: BehaviorInterval) -> bool:
        try:
            self._items.remove(interval)
        except ValueError:
            return False
        return True

    def replace(self, old: BehaviorInterval,
                new: BehaviorInterval) -> Optional[BehaviorInterval]:
        """Swap one interval for another. ``None`` when ``old`` is not here."""
        try:
            self._items[self._items.index(old)] = new
        except ValueError:
            return None
        self._sort()
        return new

    def confirm(self, interval: BehaviorInterval) -> Optional[BehaviorInterval]:
        return self.replace(interval, interval.confirm())

    def confirm_all(self, track_id: Optional[int] = None) -> int:
        """Accept every unconfirmed interval, optionally of one track only."""
        n = 0
        for pos, item in enumerate(self._items):
            if item.source == "human":
                continue
            if track_id is not None and item.track_id != track_id:
                continue
            self._items[pos] = item.confirm()
            n += 1
        return n

    def split(self, interval: BehaviorInterval,
              frame: int) -> Tuple[BehaviorInterval, BehaviorInterval]:
        left, right = interval.split(frame)
        self.remove(interval)
        self._items.extend((left, right))
        self._sort()
        return left, right

    def _sort(self) -> None:
        self._items.sort(key=lambda i: (i.start, i.track_id, i.action_id))

    # ── the compatibility bridge ─────────────────────────────────────────
    def to_frame_actions(self) -> Dict[int, Dict[int, int]]:
        """Project intervals back to ``{frame: {track_id: action_id}}``.

        This is what makes intervals free: the editing model changes, the file
        format does not. The annotator writes these ids onto its objects and
        ``annotations.json`` looks exactly as it always did.

        Later intervals win where two overlap. Overlap should not happen for one
        track — nothing here creates it — but a hand-edited pair can, and
        last-wins is the rule the timeline shows, since a later interval is
        drawn over an earlier one.
        """
        out: Dict[int, Dict[int, int]] = {}
        for interval in self._items:
            for frame in interval.frames():
                out.setdefault(frame, {})[interval.track_id] = interval.action_id
        return out

    def stats(self, fps: float = 0.0) -> Dict[int, Dict[str, float]]:
        """Per action: count, total frames, total seconds — the ethogram
        summary an experiment actually reports."""
        out: Dict[int, Dict[str, float]] = {}
        for item in self._items:
            row = out.setdefault(item.action_id,
                                 {"count": 0.0, "frames": 0.0, "seconds": 0.0})
            row["count"] += 1
            row["frames"] += item.span
            row["seconds"] += item.duration_seconds(fps)
        return out


# ══════════════════════════════════════════════════════════════════════════
# Behavior-side detectors
# ══════════════════════════════════════════════════════════════════════════

def detect_interval_issues(
    intervals: Sequence[BehaviorInterval],
    cfg: Optional[QualityConfig] = None,
    *,
    boundary_pad: int = 3,
) -> List[ReviewItem]:
    """Behavior issues, in the same :class:`ReviewItem` vocabulary as §13.

    Two kinds, and deliberately only two:

    ``BEHAVIOR_TRANSITION``
        The model was unsure *where* the interval starts or ends. The item
        covers the boundary plus ``boundary_pad`` frames each side, which is
        the span a human has to see to place the edge.

    ``BEHAVIOR_DISAGREEMENT``
        The interval as a whole scored low, so the *label* is in doubt rather
        than the edges.

    Both are emitted per interval, not per frame, for the same reason the rest
    of the queue is: a 200-frame uncertain courtship is one question.
    """
    cfg = cfg or QualityConfig()
    out: List[ReviewItem] = []

    for item in intervals:
        if item.source == "human":
            continue

        if item.confidence < cfg.low_score:
            out.append(ReviewItem(
                kind=BEHAVIOR_DISAGREEMENT,
                start=item.start, end=item.end,
                track_ids=() if item.track_id < 0 else (item.track_id,),
                severity=ramp(cfg.low_score - item.confidence, 0.0, cfg.low_score),
                detail=(f"Behavior scored {item.confidence:.2f} over "
                        f"{item.span} frames, below {cfg.low_score:.2f}."),
            ))
            continue   # the label is in doubt; its edges are not the question

        for edge, conf, where in (
            (item.start, item.boundary_confidence[0], "starts"),
            (item.end, item.boundary_confidence[1], "ends"),
        ):
            if conf >= cfg.confident_score:
                continue
            out.append(ReviewItem(
                kind=BEHAVIOR_TRANSITION,
                start=max(0, edge - boundary_pad),
                end=edge + boundary_pad,
                track_ids=() if item.track_id < 0 else (item.track_id,),
                severity=ramp(cfg.confident_score - conf, 0.0, cfg.confident_score),
                detail=f"Unsure where this behavior {where}: confidence {conf:.2f}.",
            ))

    return out
