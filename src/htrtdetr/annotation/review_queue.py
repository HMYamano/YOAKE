"""
review_queue.py — the Review Inbox: a worklist, not a list of problems.

:mod:`htrtdetr.annotation.quality` finds the places worth looking at. This
module is what a human actually *works*: filter by reason, step forward, decide,
and know how much is left. It is the state machine behind both the Review Inbox
panel and Review Mode's ``Y`` / ``N`` / ``E`` keys.

Design notes that are load-bearing
----------------------------------

**Decisions are keyed by content, not by position.** ``ReviewItem.key`` is
``(kind, start, end, track_ids)``. Re-running the detectors after correcting
something keeps every decision whose item still exists and quietly drops the
ones that no longer do — which is what makes "fix a few, re-analyse, continue"
a usable loop instead of a reset button. Position-keyed decisions would
scramble on the first re-run.

**Resolving never mutates annotations.** A decision here records *what the
human said*; applying it is the caller's job, through the annotator's existing
accept/reject paths (P1: accept is the only act with consequences, and it has
exactly one implementation). This module therefore stays pure, and a queue can
be built, filtered and tested with no video, no model and no GUI.

**The queue is ordered once, at construction.** Navigation is over a stable
list, so the ``next`` key does not teleport when a decision changes an item's
severity. Re-sorting after every decision is the fastest way to make a review
loop feel broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .quality import ISSUE_LABELS, ISSUE_ORDER, ReviewItem

__all__ = [
    "UNRESOLVED", "ACCEPTED", "REJECTED", "EDITED", "SKIPPED",
    "DECISIONS", "DECISION_LABELS",
    "Decision", "ReviewQueue", "QueueProgress",
]

#: An item nobody has judged yet.
UNRESOLVED = "unresolved"
#: "The machine was right here." Promotes the candidates to verified.
ACCEPTED = "accepted"
#: "The machine was wrong here." Discards them.
REJECTED = "rejected"
#: "The machine was nearly right." The human changed something; the item is
#: done either way, and what changed is recorded by the annotator, not here.
EDITED = "edited"
#: "Not now." Deliberately distinct from unresolved: a skipped item is one the
#: human has *seen* and chosen to defer, so ``next_unresolved`` passes over it
#: and the progress line can honestly say how much is genuinely untouched.
SKIPPED = "skipped"

DECISIONS: Tuple[str, ...] = (UNRESOLVED, ACCEPTED, REJECTED, EDITED, SKIPPED)

DECISION_LABELS: Dict[str, str] = {
    UNRESOLVED: "Not reviewed",
    ACCEPTED: "Accepted",
    REJECTED: "Rejected",
    EDITED: "Edited",
    SKIPPED: "Skipped",
}

#: Decisions that count as finished work.
_DONE = frozenset({ACCEPTED, REJECTED, EDITED})


@dataclass(frozen=True)
class Decision:
    """What a human said about one item, and anything they said with it."""

    state: str = UNRESOLVED
    #: Free-form payload the caller attaches — for an ID-switch item this is
    #: where ``{"swapped": True, "from_frame": 1412}`` lives, so the annotator
    #: can replay what was done without the queue understanding it.
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.state in _DONE

    @property
    def label(self) -> str:
        return DECISION_LABELS.get(self.state, self.state)


@dataclass(frozen=True)
class QueueProgress:
    """Counts for the one line the panel shows above the list."""

    total: int = 0
    resolved: int = 0
    skipped: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.resolved - self.skipped)

    @property
    def fraction(self) -> float:
        return self.resolved / self.total if self.total else 0.0

    def describe(self) -> str:
        """``94 issues - 31 reviewed, 4 skipped, 59 left`` / ``Nothing to review``."""
        if self.total == 0:
            return "Nothing to review"
        if self.remaining == 0 and self.skipped == 0:
            return f"All {self.total:,} reviewed"
        parts = [f"{self.total:,} issue{'s' if self.total != 1 else ''}"]
        tail = [f"{self.resolved:,} reviewed"]
        if self.skipped:
            tail.append(f"{self.skipped:,} skipped")
        tail.append(f"{self.remaining:,} left")
        return f"{parts[0]} - " + ", ".join(tail)


class ReviewQueue:
    """An ordered worklist of :class:`ReviewItem` with per-item decisions.

    Typical use::

        queue = ReviewQueue(quality.analyze(candidates, config=cfg))
        queue.set_filter(kinds={quality.ID_SWITCH_RISK})
        item = queue.current()
        ...                                 # show it, play its context
        queue.accept()                      # or reject() / edit() / skip()
        queue.advance()
    """

    def __init__(self, items: Sequence[ReviewItem] = ()) -> None:
        rank = {k: n for n, k in enumerate(ISSUE_ORDER)}
        seen: set = set()
        ordered: List[ReviewItem] = []
        for item in items:
            if item.key in seen:
                continue          # two detectors flagging the same span is one job
            seen.add(item.key)
            ordered.append(item)
        ordered.sort(key=lambda i: (rank.get(i.kind, len(rank)), -i.severity, i.start))

        self._items: Tuple[ReviewItem, ...] = tuple(ordered)
        self._decisions: Dict[Tuple, Decision] = {}
        self._filter_kinds: Optional[frozenset] = None
        self._filter_min_severity: float = 0.0
        self._hide_resolved: bool = False
        self._cursor: int = 0

    # ── contents ─────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[ReviewItem]:
        return iter(self._items)

    @property
    def items(self) -> Tuple[ReviewItem, ...]:
        """Every item, unfiltered, in queue order."""
        return self._items

    def kinds(self) -> List[Tuple[str, int]]:
        """``[(kind, count), ...]`` in :data:`ISSUE_ORDER`, for the filter chips.

        Only kinds actually present appear — a chip for a reason with nothing
        behind it teaches the user the filter is broken.
        """
        counts: Dict[str, int] = {}
        for item in self._items:
            counts[item.kind] = counts.get(item.kind, 0) + 1
        return [(k, counts[k]) for k in ISSUE_ORDER if k in counts]

    def kind_label(self, kind: str) -> str:
        return ISSUE_LABELS.get(kind, kind)

    # ── filtering ────────────────────────────────────────────────────────
    def set_filter(
        self,
        *,
        kinds: Optional[Iterable[str]] = None,
        min_severity: float = 0.0,
        hide_resolved: bool = False,
    ) -> None:
        """Narrow the view. ``kinds=None`` means every kind.

        The cursor is re-anchored onto the filtered view so the current item
        stays current when it survives the filter, and falls to the start when
        it does not — never to a stale index into a list that changed length.
        """
        current = self.current()
        self._filter_kinds = None if kinds is None else frozenset(kinds)
        self._filter_min_severity = float(min_severity)
        self._hide_resolved = bool(hide_resolved)
        view = self.filtered()
        if current is not None and current in view:
            self._cursor = view.index(current)
        else:
            self._cursor = 0

    def clear_filter(self) -> None:
        self.set_filter(kinds=None, min_severity=0.0, hide_resolved=False)

    def filtered(self) -> List[ReviewItem]:
        """The items the filter currently admits, in queue order."""
        out = []
        for item in self._items:
            if self._filter_kinds is not None and item.kind not in self._filter_kinds:
                continue
            if item.severity < self._filter_min_severity:
                continue
            if self._hide_resolved and self.decision(item).done:
                continue
            out.append(item)
        return out

    # ── navigation ───────────────────────────────────────────────────────
    @property
    def cursor(self) -> int:
        return self._cursor

    def current(self) -> Optional[ReviewItem]:
        view = self.filtered()
        if not view:
            return None
        self._cursor = max(0, min(self._cursor, len(view) - 1))
        return view[self._cursor]

    def select(self, item: ReviewItem) -> bool:
        """Put the cursor on ``item``. ``False`` if the filter hides it."""
        view = self.filtered()
        if item not in view:
            return False
        self._cursor = view.index(item)
        return True

    def select_index(self, index: int) -> Optional[ReviewItem]:
        view = self.filtered()
        if not view:
            return None
        self._cursor = max(0, min(int(index), len(view) - 1))
        return view[self._cursor]

    def advance(self) -> Optional[ReviewItem]:
        """Next item in the filtered view; stops at the end rather than wrapping.

        Not wrapping is deliberate: wrapping past the last item silently
        restarts a review the human believed they had finished.
        """
        view = self.filtered()
        if not view:
            return None
        self._cursor = min(self._cursor + 1, len(view) - 1)
        return view[self._cursor]

    def retreat(self) -> Optional[ReviewItem]:
        view = self.filtered()
        if not view:
            return None
        self._cursor = max(self._cursor - 1, 0)
        return view[self._cursor]

    def next_unresolved(self) -> Optional[ReviewItem]:
        """The next item nobody has judged, searching forward then from the top.

        Skipped items are passed over — that is what skipping means — so a pass
        with ``next_unresolved`` terminates even when the human defers half
        the queue.
        """
        view = self.filtered()
        if not view:
            return None
        n = len(view)
        for offset in range(1, n + 1):
            pos = (self._cursor + offset) % n
            if self.decision(view[pos]).state == UNRESOLVED:
                self._cursor = pos
                return view[pos]
        return None

    def at_frame(self, frame: int) -> List[ReviewItem]:
        """Every item whose interval covers ``frame`` — what the timeline needs
        when the user clicks a marker."""
        f = int(frame)
        return [i for i in self._items if i.start <= f <= i.end]

    def nearest(self, frame: int) -> Optional[ReviewItem]:
        """The filtered item closest to ``frame``, for "jump to nearest issue"."""
        view = self.filtered()
        if not view:
            return None
        f = int(frame)

        def distance(item: ReviewItem) -> int:
            if item.start <= f <= item.end:
                return 0
            return item.start - f if f < item.start else f - item.end

        best = min(view, key=distance)
        self._cursor = view.index(best)
        return best

    # ── decisions ────────────────────────────────────────────────────────
    def decision(self, item: ReviewItem) -> Decision:
        return self._decisions.get(item.key, Decision())

    def set_decision(self, item: ReviewItem, state: str, **data: Any) -> Decision:
        """Record a decision. Unknown states raise rather than being stored —
        a typo'd state would silently never count as done."""
        if state not in DECISIONS:
            raise ValueError(
                f"Unknown review decision {state!r}. Expected one of: "
                + ", ".join(DECISIONS))
        decision = Decision(state=state, data=dict(data))
        if state == UNRESOLVED and not data:
            self._decisions.pop(item.key, None)
        else:
            self._decisions[item.key] = decision
        return decision

    def accept(self, item: Optional[ReviewItem] = None, **data: Any) -> Optional[Decision]:
        """``Y`` — the machine was right here."""
        return self._decide(item, ACCEPTED, data)

    def reject(self, item: Optional[ReviewItem] = None, **data: Any) -> Optional[Decision]:
        """``N`` — the machine was wrong here."""
        return self._decide(item, REJECTED, data)

    def edit(self, item: Optional[ReviewItem] = None, **data: Any) -> Optional[Decision]:
        """``E`` — the human changed something and the item is settled."""
        return self._decide(item, EDITED, data)

    def skip(self, item: Optional[ReviewItem] = None, **data: Any) -> Optional[Decision]:
        """Defer: seen, not judged, not counted as remaining."""
        return self._decide(item, SKIPPED, data)

    def reset(self, item: Optional[ReviewItem] = None) -> Optional[Decision]:
        return self._decide(item, UNRESOLVED, {})

    def swap_decision(self, item: ReviewItem, *, swapped: bool,
                      from_frame: Optional[int] = None) -> Decision:
        """The two answers an ID-switch item takes (§15).

        ``swapped=False`` is *IDs Correct* and resolves as accepted;
        ``swapped=True`` is *Swap IDs* and resolves as edited, carrying the
        frame from which the caller should renumber. The queue records the
        intent; renumbering the track is the annotator's job, because it is a
        mutation of annotations and those have exactly one owner.
        """
        frame = item.end if from_frame is None else int(from_frame)
        if swapped:
            return self.set_decision(item, EDITED, swapped=True,
                                     from_frame=frame,
                                     track_ids=list(item.track_ids))
        return self.set_decision(item, ACCEPTED, swapped=False)

    def _decide(self, item: Optional[ReviewItem], state: str,
                data: Dict[str, Any]) -> Optional[Decision]:
        target = item if item is not None else self.current()
        if target is None:
            return None
        return self.set_decision(target, state, **data)

    # ── bulk ─────────────────────────────────────────────────────────────
    def accept_kind(self, kind: str) -> int:
        """Accept every unresolved item of one reason. Returns the count.

        Only unresolved items are touched, so this can never silently reverse
        a human's earlier Reject on the same reason.
        """
        n = 0
        for item in self._items:
            if item.kind == kind and self.decision(item).state == UNRESOLVED:
                self.set_decision(item, ACCEPTED)
                n += 1
        return n

    def accept_visible(self) -> int:
        """Accept every unresolved item the filter currently shows."""
        n = 0
        for item in self.filtered():
            if self.decision(item).state == UNRESOLVED:
                self.set_decision(item, ACCEPTED)
                n += 1
        return n

    # ── reporting ────────────────────────────────────────────────────────
    def progress(self, *, filtered: bool = False) -> QueueProgress:
        pool = self.filtered() if filtered else list(self._items)
        resolved = sum(1 for i in pool if self.decision(i).done)
        skipped = sum(1 for i in pool if self.decision(i).state == SKIPPED)
        return QueueProgress(total=len(pool), resolved=resolved, skipped=skipped)

    def decisions(self) -> Dict[Tuple, Decision]:
        """Every recorded decision, for logging and for :meth:`carry_over`."""
        return dict(self._decisions)

    def carry_over(self, decisions: Dict[Tuple, Decision]) -> int:
        """Adopt decisions from an earlier queue. Returns how many survived.

        Called after a re-analysis: items that still exist keep their verdict,
        items that no longer exist are dropped with their verdict. This is what
        makes "correct a few frames, re-run the detectors, carry on" cheap.
        """
        keys = {i.key for i in self._items}
        kept = {k: v for k, v in decisions.items() if k in keys}
        self._decisions.update(kept)
        return len(kept)

    def summary(self) -> str:
        """One line for the dashboard: ``94 locations need review``."""
        p = self.progress()
        if p.total == 0:
            return "No locations need review"
        if p.remaining == 0:
            return f"All {p.total:,} locations reviewed"
        return (f"{p.remaining:,} location{'s' if p.remaining != 1 else ''} "
                f"need review")
