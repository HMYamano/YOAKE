"""
video_segment.py — one prompt on one frame, carried across many.

``click_segment`` answers *what is under this click, on this frame*.
This module answers *and where does it go next*::

    point or box on frame N
        -> SAM 2 video predictor, in the sidecar
        -> a mask per frame
        -> mask_to_obb, in the sidecar, with the GUI's own rule
        -> Candidates
        -> a Tracklet
        -> quality checks
        -> the Review Inbox

Why the window goes through the filesystem
------------------------------------------
SAM 2's ``init_state`` wants a directory of JPEG frames. Passing an mp4 works
only on some SAM 2 versions and only with decord installed; a frame directory
works on all of them. The GUI has OpenCV for certain — the helper's interpreter
may not — so the GUI exports the window and the helper reads it. That also
keeps the pipe carrying five floats per frame instead of a megabyte of mask.

Why the window is bounded
-------------------------
:data:`MAX_WINDOW` frames, and nothing propagates across the whole video in one
go. SAM 2's video predictor holds per-frame state for the entire window in
memory, a 2,000-frame window will exhaust a laptop, and — the reason that
actually matters — a human cannot review 2,000 propagated boxes as one
decision anyway. The unit of propagation is the unit of review: a tracklet.

Optional, and says so
---------------------
SAM 2 is never imported here and never in the GUI's process. If it is not
installed, :func:`capability` returns a :class:`VideoCapability` whose
``reason`` is the sentence to show, and everything else in YOAKE keeps working.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MAX_WINDOW",
    "DEFAULT_WINDOW",
    "VideoCapability",
    "PropagationRequest",
    "PropagationResult",
    "capability",
    "export_window",
    "build_candidates",
    "result_from_reply",
    "UNAVAILABLE_MESSAGE",
]

#: Frames a single propagation may cover. See the module docstring.
MAX_WINDOW = 600
#: What the UI offers by default: 4 seconds each way at 25 fps.
DEFAULT_WINDOW = 100

#: Shown wherever video propagation is offered and cannot run (§35). Names the
#: package, what is unavailable, and what still works — never a bare ImportError.
UNAVAILABLE_MESSAGE = (
    "SAM 2 is not installed.\n\n"
    "Video propagation is unavailable. Click to Box, tracking and everything "
    "else still work.\n\n"
    "To enable it, install sam2 into the Python that runs the segmentation "
    "helper:  pip install sam2"
)


@dataclass(frozen=True)
class VideoCapability:
    """Whether one prompt can be carried across frames, and why not if not."""

    available: bool = False
    backend: str = ""
    device: str = ""
    reason: str = ""

    def __bool__(self) -> bool:
        return self.available

    def describe(self) -> str:
        """``Video propagation  SAM 2 on cuda`` / ``Unavailable - SAM2 backend
        not installed``."""
        if self.available:
            where = f" on {self.device}" if self.device else ""
            return f"{self.backend or 'SAM 2'}{where}"
        return "Unavailable - SAM2 backend not installed"


def capability(client: Any) -> VideoCapability:
    """What the running helper says it can do.

    Read from the helper's ``hello`` reply — **never** from the backend's name
    and never from a guess (``sidecar.py``'s own rule). A helper called
    ``sam2`` whose install is broken advertises no ``video`` capability, and
    that is the answer that matters.
    """
    if client is None or not getattr(client, "is_running", lambda: False)():
        return VideoCapability(
            reason="No segmentation helper is running. Start one from " "Settings > Assistance."
        )
    hello = getattr(client, "hello", None) or {}
    capabilities = tuple(hello.get("capabilities") or ())
    if "video" not in capabilities:
        return VideoCapability(backend=str(hello.get("backend") or ""), reason=UNAVAILABLE_MESSAGE)
    return VideoCapability(
        available=True,
        backend=str(hello.get("backend") or "SAM 2"),
        device=str(hello.get("device") or ""),
    )


@dataclass(frozen=True)
class PropagationRequest:
    """One prompt, and how far it should travel.

    Attributes:
        frame: The frame the human prompted on.
        points: ``[[x, y, label], ...]`` in image pixels; label 1 is positive.
        box: ``[x1, y1, x2, y2]`` in image pixels, as an alternative prompt.
        track_id: The track the resulting candidates are claimed for. The
            propagation produces one animal, so one track id, and letting the
            caller choose it is what makes the result a *tracklet* the human
            can accept in one click.
        before: Frames of window before :attr:`frame`.
        after: Frames of window after it.
    """

    frame: int
    points: Tuple[Tuple[float, float, int], ...] = ()
    box: Optional[Tuple[float, float, float, float]] = None
    track_id: int = 1
    before: int = DEFAULT_WINDOW
    after: int = DEFAULT_WINDOW
    class_id: int = 0
    action_id: int = -1

    def window(self, num_frames: int) -> Tuple[int, int]:
        """``(first, last)`` inclusive, clamped to the video and to
        :data:`MAX_WINDOW`."""
        first = max(0, self.frame - max(0, int(self.before)))
        last = self.frame + max(0, int(self.after))
        if num_frames > 0:
            last = min(last, num_frames - 1)
        last = min(last, first + MAX_WINDOW - 1)
        return first, max(first, last)

    def is_empty(self) -> bool:
        return not self.points and self.box is None


@dataclass(frozen=True)
class PropagationResult:
    """What came back: one box per frame the model kept the animal in."""

    boxes: Tuple[Tuple[int, List[float], List[float]], ...] = ()
    ok: bool = True
    detail: str = ""
    window: int = 0

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def frames(self) -> Tuple[int, ...]:
        return tuple(frame for frame, _obb, _bbox in self.boxes)

    def describe(self, track_id: int = -1) -> str:
        """The sentence shown when a propagation finishes.

        Says how much of the window was covered, because *the model lost the
        animal after 40 of 200 frames* and *the model tracked all 200* are
        completely different outcomes and a bare count hides which one happened.
        """
        if not self.ok:
            return self.detail or "Video propagation failed."
        if not self.boxes:
            return (
                "The model did not find the animal in any frame of this "
                "window. Try a point nearer its centre, or a box around it."
            )
        span = f"frames {min(self.frames):,}-{max(self.frames):,}"
        who = f" as track {track_id}" if track_id >= 0 else ""
        tail = ""
        if self.window and self.count < self.window:
            tail = f" The model lost it after {self.count} of " f"{self.window} frames."
        return (
            f"{self.count:,} boxes proposed across {span}{who}. "
            f"They are candidates until you accept them.{tail}"
        )


def export_window(
    read_frame: Callable[[int], Any],
    first: int,
    last: int,
    *,
    root: Optional[Path] = None,
    quality: int = 92,
    should_stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Optional[Path], int]:
    """Write frames ``first..last`` as ``00000.jpg`` upward. Returns ``(dir, n)``.

    Zero-padded because the helper sorts the directory by name, and a plain
    sort puts ``10.jpg`` before ``9.jpg`` — which would land every propagated
    box on the wrong frame, silently and consistently.

    Returns ``(None, 0)`` when nothing could be written, rather than raising:
    the caller turns that into a sentence, and a failed export must not take
    the annotator down.
    """
    try:
        import cv2
    except ImportError:
        return None, 0

    folder = Path(tempfile.mkdtemp(prefix="yoake_prop_", dir=str(root) if root else None))
    written = 0
    total = max(1, last - first + 1)
    try:
        for offset, index in enumerate(range(first, last + 1)):
            if should_stop is not None and should_stop():
                break
            frame = read_frame(index)
            if frame is None:
                # A hole in the middle would shift every later frame's offset,
                # so the window stops here instead of skipping.
                break
            cv2.imwrite(
                str(folder / f"{offset:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
            )
            written += 1
            if on_progress is not None:
                on_progress(offset + 1, total)
    except Exception:  # noqa: BLE001
        shutil.rmtree(folder, ignore_errors=True)
        return None, 0

    if written == 0:
        shutil.rmtree(folder, ignore_errors=True)
        return None, 0
    return folder, written


def result_from_reply(reply: Dict[str, Any]) -> PropagationResult:
    """Turn one ``propagate`` reply into a :class:`PropagationResult`."""
    if not reply.get("ok"):
        detail = str(reply.get("detail") or "")
        if reply.get("error") == "unsupported" and not detail:
            detail = UNAVAILABLE_MESSAGE
        return PropagationResult(
            ok=False, detail=detail or "The helper could not propagate that prompt."
        )
    boxes: List[Tuple[int, List[float], List[float]]] = []
    for item in reply.get("items") or ():
        try:
            frame = int(item["frame"])
            obb = [float(v) for v in item["obb"][:5]]
            bbox = [float(v) for v in item.get("bbox") or []][:4]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if len(bbox) != 4:
            bbox = _obb_bounds(obb)
        boxes.append((frame, obb, bbox))
    boxes.sort(key=lambda row: row[0])
    return PropagationResult(boxes=tuple(boxes), ok=True, window=int(reply.get("window", 0) or 0))


def build_candidates(
    result: PropagationResult,
    *,
    track_id: int,
    score: float = 0.9,
    action_id: int = -1,
) -> Dict[int, List[Any]]:
    """``{frame: [Candidate]}`` from a propagation, ready for a ReviewSession.

    ``source="track"`` rather than ``"auto"``: the box came from following one
    animal a human pointed at, not from a detector deciding what to box. The
    annotator's provenance vocabulary already distinguishes the two, and the
    distinction is what tells a later reader whether a human chose the subject.

    The score is uniform because SAM 2's video predictor does not report a
    per-frame confidence. Inventing a decay would be inventing information
    (``docs/GUI_DESIGN.md`` P12); the checks in
    :mod:`htrtdetr.annotation.quality` find the frames that went wrong from
    geometry instead, which is what they are for.
    """
    from .auto_label import Candidate

    out: Dict[int, List[Any]] = {}
    for frame, obb, bbox in result.boxes:
        out[int(frame)] = [
            Candidate(
                bbox=[int(round(v)) for v in bbox],
                obb=[float(v) for v in obb],
                score=float(score),
                track_id=int(track_id),
                action_id=int(action_id),
                source="track",
            )
        ]
    return out


def _obb_bounds(obb: Sequence[float]) -> List[float]:
    import math

    cx, cy, w, h, theta = (float(v) for v in obb[:5])
    hw, hh = w / 2.0, h / 2.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(cx + lx * cos_t - ly * sin_t)
        ys.append(cy + lx * sin_t + ly * cos_t)
    return [min(xs), min(ys), max(xs), max(ys)]
