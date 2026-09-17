"""
propagation.py — one correction, carried to the frames that need the same one.

The single most wasteful thing in frame-by-frame annotation is making the *same*
correction forty times. The detector places every box on this track eight pixels
too low for the whole of a 300-frame tracklet; the human fixes frame 1400 and
then fixes 1401, and 1402, and stops caring somewhere around 1430.

So a correction here is not an edit to one box. It is a **transform** — a
translation, a scale, a rotation — extracted from what the human did, and
applied to the same track in the frames around it::

    Correction applied.

    Re-propagate:
      (o) Until next keyframe
      ( ) Next 30 frames
      ( ) Next 100 frames
      ( ) This frame only

    [ Apply ]

Three rules
-----------

**Never overwrite human work** (``docs/GUI_DESIGN.md`` P3). Frames whose box for
this track a human already placed are protected, skipped, and *counted*, so the
message can say ``skipped 4 frames you had already corrected`` rather than
quietly doing less than it claimed.

**Never create a box.** Propagation corrects boxes that exist. A frame where the
track has no box is not a frame where the box is wrong — it is a frame where the
animal was not detected, and inventing a box there is
:func:`htrtdetr.gui.auto_label.track_forward`'s job, under its own button, with
its own confidence.

**The plan is data, and applying it is the caller's.** :func:`plan` returns
edits; the annotator applies them inside its own undo snapshot. That keeps this
module pure and keeps the undo stack owning every mutation, which is the only
way ``Ctrl+Z`` can take back a 60-frame propagation in one step.

Purity: no dearpygui, no torch, no cv2, no numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Set, Tuple)

__all__ = [
    "THIS_FRAME", "NEXT_KEYFRAME", "N_FRAMES", "TO_END", "SCOPES",
    "SCOPE_LABELS", "PropagationScope", "Transform", "Correction",
    "Edit", "PropagationPlan", "resolve_frames", "plan",
]

THIS_FRAME = "this_frame"
NEXT_KEYFRAME = "next_keyframe"
N_FRAMES = "n_frames"
TO_END = "to_end"

SCOPES: Tuple[str, ...] = (NEXT_KEYFRAME, N_FRAMES, THIS_FRAME, TO_END)

SCOPE_LABELS: Dict[str, str] = {
    NEXT_KEYFRAME: "Until next keyframe",
    N_FRAMES: "Next {n} frames",
    THIS_FRAME: "This frame only",
    TO_END: "To the end of the tracklet",
}


@dataclass(frozen=True)
class PropagationScope:
    """How far a correction should travel.

    :data:`NEXT_KEYFRAME` is the default because it is the one scope with a
    *reason* behind its extent: a keyframe is a box a human placed, so the run
    between two keyframes is exactly the stretch nobody has checked. A fixed
    frame count is a guess; the keyframe boundary is the data saying where the
    guess stops being safe.
    """

    kind: str = NEXT_KEYFRAME
    #: Frames each way for :data:`N_FRAMES`. Ignored by the other kinds.
    n: int = 30
    forward: bool = True
    #: Backward propagation is off by default: the human was scrubbing forward,
    #: so the frames behind them are the ones they have already looked at.
    backward: bool = False

    @property
    def label(self) -> str:
        return SCOPE_LABELS.get(self.kind, self.kind).format(n=self.n)


@dataclass(frozen=True)
class Transform:
    """What the human's edit did, as something repeatable.

    Translation is absolute pixels; scale is a ratio; rotation is radians. An
    edit is usually one of the three, and a :attr:`is_identity` transform is
    dropped rather than applied to sixty frames for no effect.

    Rotation is applied about each box's **own centre**, not about the edited
    box's centre. Rotating every box in a tracklet about one point would sweep
    the whole track around an arc, which is never what turning one box meant.
    """

    dx: float = 0.0
    dy: float = 0.0
    scale_w: float = 1.0
    scale_h: float = 1.0
    dtheta: float = 0.0

    #: Below this, a drag is a click that wobbled.
    EPS_PX = 0.5
    #: Below this, a resize is the pointer moving inside one pixel.
    EPS_SCALE = 0.01
    #: ~0.3 degrees.
    EPS_RAD = 0.005

    @property
    def is_identity(self) -> bool:
        return (abs(self.dx) < self.EPS_PX and abs(self.dy) < self.EPS_PX
                and abs(self.scale_w - 1.0) < self.EPS_SCALE
                and abs(self.scale_h - 1.0) < self.EPS_SCALE
                and abs(self.dtheta) < self.EPS_RAD)

    @property
    def moved(self) -> bool:
        return abs(self.dx) >= self.EPS_PX or abs(self.dy) >= self.EPS_PX

    @property
    def resized(self) -> bool:
        return (abs(self.scale_w - 1.0) >= self.EPS_SCALE
                or abs(self.scale_h - 1.0) >= self.EPS_SCALE)

    @property
    def rotated(self) -> bool:
        return abs(self.dtheta) >= self.EPS_RAD

    def describe(self) -> str:
        """``moved 8 px down, 1.2x wider`` — what will happen to the other frames."""
        parts: List[str] = []
        if self.moved:
            horizontal = "right" if self.dx > 0 else "left"
            vertical = "down" if self.dy > 0 else "up"
            if abs(self.dx) >= self.EPS_PX and abs(self.dy) >= self.EPS_PX:
                parts.append(f"moved {abs(self.dx):.0f} px {horizontal} and "
                             f"{abs(self.dy):.0f} px {vertical}")
            elif abs(self.dx) >= self.EPS_PX:
                parts.append(f"moved {abs(self.dx):.0f} px {horizontal}")
            else:
                parts.append(f"moved {abs(self.dy):.0f} px {vertical}")
        if self.resized:
            parts.append(f"resized {self.scale_w:.2f}x by {self.scale_h:.2f}x")
        if self.rotated:
            parts.append(f"turned {math.degrees(self.dtheta):+.0f} degrees")
        return ", ".join(parts) if parts else "unchanged"

    # ── application ──────────────────────────────────────────────────────
    def apply_bbox(self, bbox: Sequence[float]) -> List[float]:
        """Translate and scale an axis-aligned box about its own centre."""
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        hw = (x2 - x1) / 2.0 * self.scale_w
        hh = (y2 - y1) / 2.0 * self.scale_h
        cx += self.dx
        cy += self.dy
        return [cx - hw, cy - hh, cx + hw, cy + hh]

    def apply_obb(self, obb: Sequence[float]) -> List[float]:
        """Translate, scale and rotate a rotated box about its own centre."""
        cx, cy, w, h, theta = (float(v) for v in obb[:5])
        return [cx + self.dx, cy + self.dy,
                max(0.0, w * self.scale_w), max(0.0, h * self.scale_h),
                theta + self.dtheta]


@dataclass(frozen=True)
class Correction:
    """One box before and after a human touched it.

    Build it from the annotator's undo snapshot and the current object, and let
    :meth:`transform` work out what the human actually did — which is more
    reliable than the annotator tracking whether the gesture was a move, a
    resize or a rotate, because a single drag can be two of them.
    """

    frame: int
    track_id: int
    before_bbox: Sequence[float]
    after_bbox: Sequence[float]
    before_obb: Optional[Sequence[float]] = None
    after_obb: Optional[Sequence[float]] = None

    def transform(self) -> Transform:
        """Extract the repeatable part of the edit.

        Prefers the OBB pair when both sides have one, because an AABB derived
        from a rotated box changes size when the box merely *turns*, and reading
        that as a resize would make every propagated box grow.
        """
        if self.before_obb is not None and self.after_obb is not None:
            bx, by, bw, bh, bt = (float(v) for v in self.before_obb[:5])
            ax, ay, aw, ah, at = (float(v) for v in self.after_obb[:5])
            return Transform(
                dx=ax - bx, dy=ay - by,
                scale_w=(aw / bw) if bw > 1e-6 else 1.0,
                scale_h=(ah / bh) if bh > 1e-6 else 1.0,
                dtheta=at - bt,
            )

        b = [float(v) for v in self.before_bbox[:4]]
        a = [float(v) for v in self.after_bbox[:4]]
        bw, bh = b[2] - b[0], b[3] - b[1]
        aw, ah = a[2] - a[0], a[3] - a[1]
        return Transform(
            dx=((a[0] + a[2]) - (b[0] + b[2])) / 2.0,
            dy=((a[1] + a[3]) - (b[1] + b[3])) / 2.0,
            scale_w=(aw / bw) if bw > 1e-6 else 1.0,
            scale_h=(ah / bh) if bh > 1e-6 else 1.0,
        )


@dataclass(frozen=True)
class Edit:
    """One box the plan would change, with its new geometry."""

    frame: int
    track_id: int
    bbox: List[float]
    obb: Optional[List[float]] = None


@dataclass(frozen=True)
class PropagationPlan:
    """Everything the propagation would do, before any of it is done."""

    edits: Tuple[Edit, ...] = ()
    scope: PropagationScope = field(default_factory=PropagationScope)
    transform: Transform = field(default_factory=Transform)
    #: Frames skipped because a human had placed that box.
    skipped_protected: int = 0
    #: Frames skipped because this track has no box there.
    skipped_missing: int = 0

    @property
    def count(self) -> int:
        return len(self.edits)

    @property
    def frames(self) -> Tuple[int, ...]:
        return tuple(e.frame for e in self.edits)

    def describe(self) -> str:
        """The sentence shown after Apply.

        Names what was skipped and why, every time. A propagation that silently
        did 26 frames when the user asked for 30 is how people stop believing a
        feature (§4.4).
        """
        if not self.edits:
            if self.transform.is_identity:
                return "Nothing to propagate: the box did not change."
            if self.skipped_protected and not self.skipped_missing:
                return ("Nothing to propagate: every frame in range already has a "
                        "box you placed.")
            if self.skipped_missing and not self.skipped_protected:
                return ("Nothing to propagate: this track has no box in the frames "
                        "in range.")
            return "Nothing to propagate in this range."

        head = (f"Applied to {self.count} frame{'s' if self.count != 1 else ''} "
                f"({self.transform.describe()}).")
        tail: List[str] = []
        if self.skipped_protected:
            tail.append(f"{self.skipped_protected} you had already corrected")
        if self.skipped_missing:
            tail.append(f"{self.skipped_missing} with no box on this track")
        if tail:
            head += " Skipped " + " and ".join(tail) + "."
        return head


# ══════════════════════════════════════════════════════════════════════════
# Scope resolution
# ══════════════════════════════════════════════════════════════════════════

def resolve_frames(
    scope: PropagationScope,
    *,
    frame: int,
    num_frames: int = 0,
    keyframes: Iterable[int] = (),
    track_frames: Optional[Iterable[int]] = None,
) -> List[int]:
    """Which frames a scope covers, excluding ``frame`` itself.

    Args:
        scope: The user's choice.
        frame: Where the correction was made.
        num_frames: Video length, for clamping. ``0`` means unknown, and then
            forward propagation is bounded by ``track_frames`` or by ``n``.
        keyframes: Frames a human has placed a box on for this track. Used by
            :data:`NEXT_KEYFRAME` as the stopping point.
        track_frames: Every frame where this track has a box. Used by
            :data:`TO_END` and to bound :data:`NEXT_KEYFRAME` when no keyframe
            lies ahead.

    Returns:
        Ascending frame indices, never including ``frame``.
    """
    here = int(frame)
    if scope.kind == THIS_FRAME:
        return []

    track = sorted(int(f) for f in track_frames) if track_frames is not None else []
    last = (num_frames - 1) if num_frames > 0 else (track[-1] if track else here + scope.n)
    first = 0 if num_frames > 0 else (track[0] if track else max(0, here - scope.n))

    if scope.kind == N_FRAMES:
        ahead = min(last, here + max(0, scope.n))
        behind = max(first, here - max(0, scope.n))
    elif scope.kind == TO_END:
        ahead, behind = last, first
    else:  # NEXT_KEYFRAME
        keys = sorted(int(k) for k in keyframes if int(k) != here)
        nxt = next((k for k in keys if k > here), None)
        prv = next((k for k in reversed(keys) if k < here), None)
        # Stop *before* the keyframe: the keyframe is a box a human placed, and
        # the correction is not theirs to apply to it.
        ahead = (nxt - 1) if nxt is not None else last
        behind = (prv + 1) if prv is not None else first

    out: List[int] = []
    if scope.backward:
        out.extend(range(max(0, behind), here))
    if scope.forward:
        out.extend(range(here + 1, ahead + 1))
    return sorted(set(out))


# ══════════════════════════════════════════════════════════════════════════
# Planning
# ══════════════════════════════════════════════════════════════════════════

def plan(
    correction: Correction,
    scope: PropagationScope,
    *,
    boxes: Mapping[int, Tuple[Sequence[float], Optional[Sequence[float]]]],
    num_frames: int = 0,
    keyframes: Iterable[int] = (),
    protected: Iterable[int] = (),
    frame_size: Optional[Tuple[int, int]] = None,
    min_side: float = 5.0,
) -> PropagationPlan:
    """Work out every box the correction should also change.

    Args:
        correction: What the human did, on one frame, to one track.
        scope: How far it should travel.
        boxes: ``{frame: (bbox, obb_or_None)}`` — this track's current geometry
            in every frame it appears in. Frames missing from this mapping are
            frames where the track has no box, and are skipped and counted.
        num_frames: Video length, for clamping.
        keyframes: Frames where a human placed this track's box.
        protected: Frames that must not be touched. Usually the keyframes plus
            anything else the caller considers human work.
        frame_size: ``(width, height)``. Propagated boxes are clamped inside it.
        min_side: A propagated box smaller than this in either axis is dropped
            rather than written — a scale-down that collapses a box in one frame
            would collapse it in sixty.

    Returns:
        A :class:`PropagationPlan`. An identity transform returns an empty plan,
        so a caller may hand every mouse release to this function and let the
        plan decide whether anything happened.
    """
    transform = correction.transform()
    if transform.is_identity:
        return PropagationPlan(scope=scope, transform=transform)

    guard: Set[int] = {int(f) for f in protected}
    guard.add(int(correction.frame))

    targets = resolve_frames(
        scope,
        frame=correction.frame,
        num_frames=num_frames,
        keyframes=keyframes,
        track_frames=boxes.keys(),
    )

    edits: List[Edit] = []
    skipped_protected = 0
    skipped_missing = 0

    for frame in targets:
        if frame in guard:
            skipped_protected += 1
            continue
        current = boxes.get(frame)
        if current is None:
            skipped_missing += 1
            continue

        bbox, obb = current
        new_obb = transform.apply_obb(obb) if obb is not None else None
        new_bbox = (_obb_bounds(new_obb) if new_obb is not None
                    else transform.apply_bbox(bbox))

        if (new_bbox[2] - new_bbox[0]) < min_side or (new_bbox[3] - new_bbox[1]) < min_side:
            skipped_missing += 1
            continue

        if frame_size is not None:
            new_bbox = _clamp(new_bbox, frame_size)

        edits.append(Edit(frame=frame, track_id=correction.track_id,
                          bbox=new_bbox, obb=new_obb))

    return PropagationPlan(
        edits=tuple(edits),
        scope=scope,
        transform=transform,
        skipped_protected=skipped_protected,
        skipped_missing=skipped_missing,
    )


def _obb_bounds(obb: Sequence[float]) -> List[float]:
    """Circumscribing AABB of a rotated box.

    The annotator's invariant is that ``bbox`` is always the OBB's bounding
    rectangle (``AnnObject.sync_bbox``), so that every consumer which only reads
    ``bbox`` keeps working whether or not the box is rotated. Propagation must
    preserve it.
    """
    cx, cy, w, h, theta = (float(v) for v in obb[:5])
    hw, hh = w / 2.0, h / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(cx + lx * cos_t - ly * sin_t)
        ys.append(cy + lx * sin_t + ly * cos_t)
    return [min(xs), min(ys), max(xs), max(ys)]


def _clamp(bbox: Sequence[float], frame_size: Tuple[int, int]) -> List[float]:
    width, height = frame_size
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    return [max(0.0, min(x1, width)), max(0.0, min(y1, height)),
            max(0.0, min(x2, width)), max(0.0, min(y2, height))]
