"""
status.py — the three annotation states, derived rather than stored.

``docs/GUI_DESIGN.md`` §7.1 already defines four *provenance* states on disk
(human/auto/track x accepted/pending). This module adds the **review-facing**
vocabulary on top of them without adding a field to the file format:

===========  ==============================================================
``VERIFIED`` A human looked at this box and kept it. It is training data.
``PREDICTED`` A machine produced it, nothing is wrong with it, and it is
             confident. It is *not* training data until a human says so.
``PENDING``  A machine produced it and something about it needs a human.
===========  ==============================================================

The states are **derived**, never stored, for one reason: a stored status is a
second source of truth that desynchronises from ``accepted`` the first time
anything writes one without the other (``docs/GUI_DESIGN.md`` §9.3 records
exactly that failure for the ``keyframe`` flag). :func:`status_of` is a pure
function of facts that already exist, so it cannot drift.

The promotion rule is the one invariant the whole review loop rests on
(P1): **only a human promotes to VERIFIED.** Nothing in this module or any
caller may turn PREDICTED into VERIFIED on a threshold alone.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "VERIFIED", "PREDICTED", "PENDING", "STATES",
    "STATE_LABELS", "STATE_HELP",
    "status_of", "is_training_data", "counts_by_status", "summarize",
    "DEFAULT_CONFIDENT_SCORE",
]

VERIFIED = "verified"
PREDICTED = "predicted"
PENDING = "pending"

#: Review order: the states a human still owes work on come last.
STATES: tuple[str, ...] = (VERIFIED, PREDICTED, PENDING)

STATE_LABELS: dict[str, str] = {
    VERIFIED: "Verified",
    PREDICTED: "Predicted",
    PENDING: "Needs review",
}

STATE_HELP: dict[str, str] = {
    VERIFIED: "You kept this box. It is saved and it trains the model.",
    PREDICTED: "The model is confident about this box. It is not saved until you accept it.",
    PENDING: "Something about this box needs your eyes. It is not saved.",
}

#: Score at or above which an unflagged machine box reads as PREDICTED rather
#: than PENDING. Deliberately a *default argument*, not a constant the callers
#: reach for: the right value depends on the detector and the species, and
#: :class:`~htrtdetr.annotation.quality.QualityConfig` is where a project
#: overrides it.
DEFAULT_CONFIDENT_SCORE = 0.70


def status_of(
    *,
    accepted: bool,
    score: Optional[float] = None,
    issues: Sequence[Any] = (),
    confident_score: float = DEFAULT_CONFIDENT_SCORE,
) -> str:
    """Derive the review state of one box.

    Args:
        accepted: ``True`` once a human has kept the box. This alone decides
            VERIFIED — no score and no absence of issues can substitute for it.
        score: Detector or tracker confidence. ``None`` means "no machine
            opinion", which is how a hand-drawn box and an accepted box both
            look; it is treated as confident because there is nothing to doubt.
        issues: Anything truthy and non-empty means a detector flagged this
            box. One flag is enough to demand review, however high the score.
        confident_score: The PREDICTED/PENDING boundary.

    Returns:
        One of :data:`VERIFIED`, :data:`PREDICTED`, :data:`PENDING`.
    """
    if accepted:
        return VERIFIED
    if issues:
        return PENDING
    if score is None:
        return PREDICTED
    return PREDICTED if float(score) >= float(confident_score) else PENDING


def is_training_data(state: str) -> bool:
    """Does a box in this state reach ``annotations.json``?

    Only VERIFIED does. This function exists so no caller has to re-derive the
    rule with a ``!=`` and get it backwards.
    """
    return state == VERIFIED


def counts_by_status(states: Sequence[str]) -> dict[str, int]:
    """Tally a sequence of states, with every known state present at zero."""
    out = {s: 0 for s in STATES}
    for s in states:
        if s in out:
            out[s] += 1
    return out


def summarize(counts: Mapping[str, int]) -> str:
    """One English sentence for a status tally.

    Shaped for the dashboard and the annotator status line::

        230 verified, 1,850 predicted, 42 need review
        Nothing annotated yet
    """
    verified = int(counts.get(VERIFIED, 0))
    predicted = int(counts.get(PREDICTED, 0))
    pending = int(counts.get(PENDING, 0))
    if verified == 0 and predicted == 0 and pending == 0:
        return "Nothing annotated yet"
    parts = []
    if verified:
        parts.append(f"{verified:,} verified")
    if predicted:
        parts.append(f"{predicted:,} predicted")
    if pending:
        parts.append(f"{pending:,} need review")
    return ", ".join(parts)
