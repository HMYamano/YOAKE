"""
tracklet.py — the unit a human accepts, instead of 300 separate boxes.

A detector that runs over 30 minutes of video produces tens of thousands of
candidates. Reviewing them one at a time is not slow, it is *impossible*: at
one second per decision, 18,000 candidates is five hours. The way out is not a
faster click — it is a **bigger unit**. A tracklet is one animal, tracked over
a run of frames, with one confidence summary and one Accept.

    Tracklet #14   frames 1400-1699   300 candidates
    confidence  min 0.71   mean 0.94
    issues      none
    [ Accept tracklet ]

What splits a track into tracklets
----------------------------------
A ``track_id`` is a claim the tracker makes about identity across the whole
video. A **tracklet** is the part of that claim which is continuous enough to
be one decision. Two things end a tracklet:

1. **A gap** longer than ``max_gap`` frames. The animal was not detected, so
   whatever the tracker says on the far side is a fresh claim, and accepting
   across the gap would accept that claim unexamined.
2. **Nothing else.** In particular a tracklet is *not* split by low confidence
   or by a flagged issue — those are attached to it
   (:meth:`Tracklet.with_issues`) so the human sees them on the Accept button,
   which is the whole point. Splitting on them would hide the problem by
   turning one suspicious tracklet into two innocent-looking ones.

This module is pure data. It never imports dearpygui, torch, cv2 or numpy, and
it accepts any object with ``track_id`` / ``score`` / ``bbox`` attributes, so
it works on :class:`htrtdetr.gui.auto_label.Candidate` without importing it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Tracklet", "TrackletIndex",
    "build_tracklets", "DEFAULT_MAX_GAP", "DEFAULT_MIN_LENGTH",
]

#: Missing frames tolerated inside one tracklet. One frame of miss is a
#: blink — a fly walking behind another for a single frame at 25 fps — and
#: splitting on it would double the number of decisions for nothing.
DEFAULT_MAX_GAP = 1

#: A tracklet shorter than this is still built. Filtering is the caller's
#: choice because a 2-frame tracklet is noise during bulk accept but is the
#: *whole point* when the queue is showing you track fragmentation.
DEFAULT_MIN_LENGTH = 1


@dataclass(frozen=True)
class Tracklet:
    """One animal over one continuous run of frames.

    Attributes:
        track_id: The tracker's identity claim. ``-1`` means the producer had
            no opinion; those candidates are each their own tracklet, because
            grouping boxes that nothing claims are the same animal would be
            inventing the claim.
        frames: The frames covered, ascending, with no duplicates.
        indices: For each frame, the position of this tracklet's candidate
            within that frame's candidate list. Parallel to :attr:`frames`.
            This is what makes a tracklet *addressable*: every entry is a
            ``(frame, index)`` pair that
            :meth:`htrtdetr.gui.auto_label.ReviewSession.accept` takes.
        scores: Per-frame confidence, parallel to :attr:`frames`.
        issues: Issue codes attached by
            :mod:`htrtdetr.annotation.quality`. Empty means no detector
            objected — **not** that the tracklet is correct.
    """

    track_id: int
    frames: Tuple[int, ...]
    indices: Tuple[int, ...]
    scores: Tuple[float, ...]
    issues: Tuple[str, ...] = ()

    # ── extent ───────────────────────────────────────────────────────────
    @property
    def start(self) -> int:
        """First frame, inclusive."""
        return self.frames[0]

    @property
    def end(self) -> int:
        """Last frame, **inclusive**. A 1-frame tracklet has ``start == end``."""
        return self.frames[-1]

    @property
    def count(self) -> int:
        """Number of candidates, which is what the Accept button spends."""
        return len(self.frames)

    @property
    def span(self) -> int:
        """Frames from start to end inclusive, gaps included."""
        return self.end - self.start + 1

    @property
    def is_contiguous(self) -> bool:
        return self.span == self.count

    @property
    def gaps(self) -> Tuple[Tuple[int, int], ...]:
        """Missing runs inside the tracklet as ``(first_missing, last_missing)``."""
        out: List[Tuple[int, int]] = []
        for a, b in zip(self.frames, self.frames[1:]):
            if b - a > 1:
                out.append((a + 1, b - 1))
        return tuple(out)

    # ── confidence ───────────────────────────────────────────────────────
    @property
    def min_score(self) -> float:
        """The weakest frame. This is the number that decides whether a bulk
        accept is safe, so it is shown *before* the mean everywhere."""
        return min(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def duration_seconds(self, fps: float) -> float:
        """Wall-clock length. ``fps <= 0`` returns 0.0 rather than raising —
        a video whose fps the container lied about must not crash the panel."""
        return self.span / fps if fps > 0 else 0.0

    # ── addressing ───────────────────────────────────────────────────────
    def targets(self) -> Tuple[Tuple[int, int], ...]:
        """Every ``(frame, index)`` this tracklet owns, in frame order.

        Feed straight to a review state machine::

            for frame, index in tracklet.targets():
                session.accept(frame, index)
        """
        return tuple(zip(self.frames, self.indices))

    def contains(self, frame: int) -> bool:
        """Is ``frame`` one of the frames this tracklet actually covers?

        A frame inside a gap answers ``False`` — it is inside :attr:`span` but
        the tracklet has no candidate there, and saying otherwise would make
        callers index a frame that is not in :attr:`frames`.
        """
        return int(frame) in self._frame_set

    def index_at(self, frame: int) -> Optional[int]:
        """The candidate index at ``frame``, or ``None`` if not covered."""
        try:
            return self.indices[self.frames.index(int(frame))]
        except ValueError:
            return None

    def score_at(self, frame: int) -> Optional[float]:
        try:
            return self.scores[self.frames.index(int(frame))]
        except ValueError:
            return None

    @property
    def _frame_set(self) -> frozenset:
        # Recomputed per call rather than cached: a frozen dataclass cannot
        # memoise without object.__setattr__, and these are small.
        return frozenset(self.frames)

    # ── derivation ───────────────────────────────────────────────────────
    def with_issues(self, issues: Iterable[str]) -> "Tracklet":
        """A copy carrying these issue codes, de-duplicated and sorted.

        Sorted so two tracklets flagged by the same detectors in a different
        order compare equal — the queue de-duplicates on equality.
        """
        return replace(self, issues=tuple(sorted(set(str(i) for i in issues))))

    def slice(self, start: int, end: int) -> Optional["Tracklet"]:
        """The part of this tracklet inside ``[start, end]`` inclusive.

        Returns ``None`` when the range covers no frame of it, so a caller can
        write ``if (part := t.slice(a, b)):`` without a length check.
        """
        keep = [i for i, f in enumerate(self.frames) if start <= f <= end]
        if not keep:
            return None
        return Tracklet(
            track_id=self.track_id,
            frames=tuple(self.frames[i] for i in keep),
            indices=tuple(self.indices[i] for i in keep),
            scores=tuple(self.scores[i] for i in keep),
            issues=self.issues,
        )

    # ── presentation ─────────────────────────────────────────────────────
    def label(self) -> str:
        """``Tracklet #14`` — the name the UI uses everywhere."""
        if self.track_id < 0:
            return "Untracked box"
        return f"Tracklet #{self.track_id}"

    def describe(self, fps: float = 0.0) -> str:
        """One line for a list row or an Accept button's tooltip."""
        head = f"{self.label()}  frames {self.start}-{self.end}"
        if fps > 0:
            head += f" ({self.duration_seconds(fps):.1f} s)"
        body = (f"{self.count} candidate{'s' if self.count != 1 else ''}, "
                f"min {self.min_score:.2f}, mean {self.mean_score:.2f}")
        if self.issues:
            body += f" - {len(self.issues)} issue{'s' if len(self.issues) != 1 else ''}"
        return f"{head}  {body}"


class TrackletIndex:
    """Every tracklet in one run, with the two lookups the UI actually does.

    The timeline asks *"what covers this frame?"* on every render; the
    inspector asks *"what does this candidate belong to?"* on every selection.
    Both are dict hits here rather than a scan over tracklets.
    """

    def __init__(self, tracklets: Sequence[Tracklet] = ()) -> None:
        self._tracklets: Tuple[Tracklet, ...] = tuple(tracklets)
        self._by_frame: Dict[int, List[int]] = {}
        self._by_target: Dict[Tuple[int, int], int] = {}
        for pos, t in enumerate(self._tracklets):
            for frame, index in t.targets():
                self._by_frame.setdefault(frame, []).append(pos)
                self._by_target[(frame, index)] = pos

    def __len__(self) -> int:
        return len(self._tracklets)

    def __iter__(self) -> Iterator[Tracklet]:
        return iter(self._tracklets)

    def __getitem__(self, pos: int) -> Tracklet:
        return self._tracklets[pos]

    @property
    def tracklets(self) -> Tuple[Tracklet, ...]:
        return self._tracklets

    def at_frame(self, frame: int) -> List[Tracklet]:
        """Every tracklet with a candidate on ``frame``."""
        return [self._tracklets[p] for p in self._by_frame.get(int(frame), ())]

    def of(self, frame: int, index: int) -> Optional[Tracklet]:
        """The tracklet owning one candidate, or ``None`` if it owns none."""
        pos = self._by_target.get((int(frame), int(index)))
        return None if pos is None else self._tracklets[pos]

    def for_track(self, track_id: int) -> List[Tracklet]:
        """Every tracklet of one track id, in frame order. More than one means
        the track was fragmented by a gap — which is itself worth reviewing."""
        return [t for t in self._tracklets if t.track_id == int(track_id)]

    def total_candidates(self) -> int:
        return sum(t.count for t in self._tracklets)

    def with_issues(self) -> List[Tracklet]:
        return [t for t in self._tracklets if t.issues]


def build_tracklets(
    candidates: Mapping[int, Sequence[Any]],
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    min_length: int = DEFAULT_MIN_LENGTH,
) -> List[Tracklet]:
    """Group per-frame candidates into tracklets.

    Args:
        candidates: ``{frame_index: [candidate, ...]}`` as
            :func:`htrtdetr.gui.auto_label.load_candidates` returns. Each
            candidate needs ``track_id`` and ``score``; anything else is
            ignored, so this works on candidates, on annotation objects, and
            on plain namespaces in tests.
        max_gap: Missing frames tolerated inside one tracklet. ``0`` demands
            perfect contiguity.
        min_length: Tracklets shorter than this are dropped.

    Returns:
        Tracklets sorted by ``(start, track_id)`` — the order a timeline draws
        them and a queue lists them, so callers never have to re-sort.

    Untracked candidates (``track_id < 0``) each become their own 1-frame
    tracklet. Grouping them would mean asserting an identity the producer
    explicitly declined to assert.
    """
    max_gap = max(0, int(max_gap))
    per_track: Dict[int, List[Tuple[int, int, float]]] = {}
    loose: List[Tuple[int, int, float]] = []

    for frame in sorted(int(f) for f in candidates):
        for index, cand in enumerate(candidates[frame] or ()):
            tid = int(getattr(cand, "track_id", -1))
            score = float(getattr(cand, "score", 1.0) or 0.0)
            if tid < 0:
                loose.append((frame, index, score))
            else:
                per_track.setdefault(tid, []).append((frame, index, score))

    out: List[Tracklet] = []

    for tid, rows in per_track.items():
        rows.sort(key=lambda r: r[0])
        run: List[Tuple[int, int, float]] = []
        for row in rows:
            if run and row[0] - run[-1][0] - 1 > max_gap:
                out.append(_make(tid, run))
                run = []
            run.append(row)
        if run:
            out.append(_make(tid, run))

    for frame, index, score in loose:
        out.append(Tracklet(track_id=-1, frames=(frame,),
                            indices=(index,), scores=(score,)))

    if min_length > 1:
        out = [t for t in out if t.count >= min_length]

    out.sort(key=lambda t: (t.start, t.track_id))
    return out


def _make(track_id: int, run: Sequence[Tuple[int, int, float]]) -> Tracklet:
    return Tracklet(
        track_id=int(track_id),
        frames=tuple(r[0] for r in run),
        indices=tuple(r[1] for r in run),
        scores=tuple(r[2] for r in run),
    )
