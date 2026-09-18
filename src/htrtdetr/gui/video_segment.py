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
from typing import (Any, Callable, Dict, List, Mapping, Optional,
                    Sequence, Tuple)

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
    "merge_results",
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
    "To enable it, install it into the Python that runs the segmentation "
    "helper:  pip install sam2 huggingface_hub\n\n"
    "Both packages are needed. sam2 reaches its weights through "
    "huggingface_hub and does not depend on it, so installing sam2 alone "
    "leaves every load failing inside the helper."
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



# ══════════════════════════════════════════════════════════════════════════
# Clips: one export, every animal in it
# ══════════════════════════════════════════════════════════════════════════
#
# Smart Annotate proposes *clips* for behavior, and a clip is the natural unit
# to follow animals across: it is already the span a human agreed to look at.
# Propagating one animal at a time re-exported and re-encoded the same frames
# for each of them, which is the whole cost of a propagation.
#
# Measured here (RTX 5070 Ti, sam2-hiera-tiny, 40 frames, two animals):
#
#   animal 1   forward 4.40 s + backward 1.58 s = 5.98 s   <- pays init_state
#   animal 2   forward 1.50 s + backward 1.58 s = 3.09 s   <- 52% of the first
#
# The helper caches the inference state per ``frames_dir`` and ``reset_state``
# keeps the per-frame features, so the encode is paid once and every extra
# animal costs only its two propagation passes. That is why these functions
# hand back requests that all share one window: the saving is entirely in
# exporting and encoding once, and it evaporates if the caller exports per
# animal. It is also why there is no multi-object protocol op - it would save
# one pass per animal against a cost that is already amortised.


@dataclass(frozen=True)
class ClipPrompt:
    """One animal to follow through a clip, and the identity its boxes claim.

    Both a ``box`` and a centre ``point`` are worth sending: SAM 2 takes them
    together and the pair is a stronger prompt than either alone.
    """

    track_id: int
    points: Tuple[Tuple[float, float, int], ...] = ()
    box: Optional[Tuple[float, float, float, float]] = None
    class_id: int = 0
    action_id: int = -1


def clip_window(start: int, end: int, num_frames: int = 0) -> Tuple[int, int]:
    """``(first, last)`` inclusive for a clip, clamped to the video and MAX_WINDOW.

    Order-tolerant: a clip handed over backwards is read as the span it covers
    rather than rejected, because the caller is often a drag.
    """
    first = max(0, min(int(start), int(end)))
    last = max(int(start), int(end))
    if num_frames > 0:
        last = min(last, num_frames - 1)
    last = min(last, first + MAX_WINDOW - 1)
    return first, max(first, last)


def clip_prompt_frame(start: int, end: int, preferred: Optional[int] = None,
                      num_frames: int = 0) -> int:
    """Which frame the prompts are read from. The middle, unless asked.

    The middle is not arbitrary. SAM 2 propagates one way from the prompt, so a
    clip is covered by a forward pass and a backward one; prompting at the
    middle makes the longer of those two runs as short as it can be, and drift
    accumulates along a run. Prompting at an end doubles the worst-case run for
    no gain.

    A ``preferred`` frame - the one the human is looking at - is honoured but
    clamped into the window, because a prompt outside it has nothing to read.
    """
    first, last = clip_window(start, end, num_frames)
    if preferred is None:
        return (first + last) // 2
    return max(first, min(int(preferred), last))


def clip_requests(
    start: int,
    end: int,
    prompts: Sequence[ClipPrompt],
    *,
    preferred_frame: Optional[int] = None,
    num_frames: int = 0,
) -> Tuple[PropagationRequest, ...]:
    """One :class:`PropagationRequest` per animal, all over the same window.

    Every request carries the same ``frame``, ``before`` and ``after``, so
    :meth:`PropagationRequest.window` gives all of them the identical span and
    one export serves the lot. Empty prompts are dropped rather than sent: a
    request with neither a point nor a box cannot prompt anything, and the
    helper would answer with an empty result that reads like a lost animal.
    """
    first, last = clip_window(start, end, num_frames)
    frame = clip_prompt_frame(start, end, preferred_frame, num_frames)
    out = []
    for prompt in prompts:
        request = PropagationRequest(
            frame=frame,
            points=tuple(tuple(p) for p in prompt.points),
            box=tuple(prompt.box) if prompt.box else None,
            track_id=int(prompt.track_id),
            before=frame - first,
            after=last - frame,
            class_id=int(prompt.class_id),
            action_id=int(prompt.action_id),
        )
        if not request.is_empty():
            out.append(request)
    return tuple(out)



def clip_seed_frame(
    start: int,
    end: int,
    annotated: Mapping[int, Sequence[Any]],
    *,
    preferred: Optional[int] = None,
    num_frames: int = 0,
) -> Optional[int]:
    """Which frame inside the clip should seed the propagation.

    A propagation can only follow animals it was prompted with, so the frame
    that matters is the one carrying the **most** of them - seeding from a
    frame where one of two flies is drawn leaves the other one unproposed for
    the entire clip, and nothing in the result says why.

    Ties go to whichever candidate frame is nearest the middle, for the same
    reason :func:`clip_prompt_frame` defaults there: the clip is covered by a
    forward pass and a backward one, and the middle makes the longer of the two
    as short as possible.

    ``preferred`` - the frame the human is looking at - wins **only** when it
    is not beaten on count. Honouring it unconditionally would quietly drop
    animals; ignoring it would move the prompt away from the frame they just
    checked for no reason.

    Returns ``None`` when no frame in the clip has a box, which is a different
    answer from "frame 0" and the caller has to say something different about
    it.
    """
    first, last = clip_window(start, end, num_frames)
    middle = (first + last) // 2
    populated = [
        (f, len(annotated.get(f) or ()))
        for f in range(first, last + 1)
        if annotated.get(f)
    ]
    if not populated:
        return None
    best_count = max(count for _f, count in populated)
    if preferred is not None:
        want = max(first, min(int(preferred), last))
        if len(annotated.get(want) or ()) >= best_count:
            return want
    # -count first so the fullest frame wins, then distance to the middle.
    return min(populated, key=lambda item: (-item[1], abs(item[0] - middle), item[0]))[0]


def prompts_from_objects(objects: Sequence[Any], *, fallback_track: int = 1
                         ) -> Tuple[ClipPrompt, ...]:
    """Turn the boxes on one frame into one prompt per animal.

    Sends the box **and** its centre as a point, because SAM 2 takes both and
    the pair is a stronger prompt than either alone.

    An object with no track id of its own gets a synthetic one counting up from
    ``fallback_track``: the propagation's result is a tracklet, and a tracklet
    with no identity cannot be accepted as one animal. Ids already present are
    never renumbered.
    """
    out = []
    spare = int(fallback_track)
    used = {int(o.track_id) for o in objects if int(getattr(o, "track_id", -1)) >= 0}
    for obj in objects:
        bbox = [float(v) for v in tuple(obj.bbox)[:4]]
        if len(bbox) < 4:
            continue
        x0, y0, x1, y1 = bbox
        track = int(getattr(obj, "track_id", -1))
        if track < 0:
            while spare in used:
                spare += 1
            track = spare
            used.add(track)
        out.append(
            ClipPrompt(
                track_id=track,
                points=(((x0 + x1) / 2.0, (y0 + y1) / 2.0, 1),),
                box=(x0, y0, x1, y1),
                class_id=int(getattr(obj, "class_id", 0) or 0),
                action_id=int(getattr(obj, "action_id", -1)),
            )
        )
    return tuple(out)


def describe_clip_run(pairs: Sequence[Tuple[Any, Optional[PropagationResult]]],
                      *, clips: int = 1, skipped: int = 0) -> str:
    """The one sentence a finished clip run leaves in the status line.

    Counts animals, not boxes, first: *3 animals followed across 48 frames* is
    what the human asked for, and a bare box total hides an animal that was
    lost immediately. An animal the model never found is named separately for
    the same reason - it is the outcome worth acting on.

    ``skipped`` belongs here rather than in a status line at the start of the
    run: a clip with no box in it is skipped before any work happens, so the
    only message announcing it would be overwritten by the first progress
    update seconds later. The count has to survive to the end or it is not
    really reported.
    """
    followed = [(r, res) for r, res in pairs if res is not None and res.count]
    lost = [r for r, res in pairs if res is None or not res.count]
    missing = (
        f" {skipped} clip(s) had no box to follow and were skipped." if skipped else ""
    )
    if not pairs:
        return "Nothing to follow in this clip." + missing
    if not followed:
        return (
            f"The model found none of the {len(pairs)} animals in this clip. "
            "Try prompting on a frame where they are clearly separated." + missing
        )
    boxes = sum(res.count for _r, res in followed)
    frames = sorted({f for _r, res in followed for f in res.frames})
    where = f"frames {frames[0]:,}-{frames[-1]:,}" if frames else "no frames"
    scope = "" if clips <= 1 else f" over {clips} clips"
    tail = ""
    if lost:
        ids = ", ".join(str(r.track_id) for r in lost)
        tail = f" Track {ids} was not found and nothing was proposed for it."
    return (
        f"{len(followed)} animal(s) followed across {where}{scope}: "
        f"{boxes:,} candidate boxes. They are candidates until you accept "
        f"them.{tail}{missing}"
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


def merge_results(*results: Optional[PropagationResult]) -> PropagationResult:
    """The passes of one propagation, as a single result.

    **SAM 2 propagates in one direction only.** A forward pass covers the
    prompt frame and everything after it, and produces nothing before it — so
    a window built with :attr:`PropagationRequest.before` frames needs a second
    pass with ``backward=True`` or that half comes back empty while still
    having been decoded, exported and paid for. Measured on a 40-frame window
    prompted at offset 20: forward alone returns 20 boxes, backward alone
    returns 21, and the two together return all 40.

    The prompt frame is produced by **both** passes. The first result that
    carries a frame wins, so pass the forward result first and the tie is
    resolved toward it.

    ``window`` is the maximum and not the sum: both passes run over the same
    exported directory and the helper reports the size of that directory, so
    adding them would double it and make :meth:`PropagationResult.describe`
    claim the model lost an animal it never lost.

    A failed pass is not fatal — if the other one produced boxes they are kept,
    because half a tracklet the human can accept beats an error message. Only
    when nothing survives is the failure passed through.
    """
    kept = [r for r in results if r is not None]
    if not kept:
        return PropagationResult(ok=False, detail="Nothing was propagated.")

    merged: Dict[int, Tuple[int, List[float], List[float]]] = {}
    window = 0
    for result in kept:
        if not result.ok:
            continue
        window = max(window, int(result.window))
        for row in result.boxes:
            merged.setdefault(int(row[0]), row)

    if not merged:
        failed = [r for r in kept if not r.ok]
        if failed:
            return failed[0]
        return PropagationResult(ok=True, boxes=(), window=window)

    return PropagationResult(
        boxes=tuple(merged[frame] for frame in sorted(merged)),
        ok=True,
        window=window,
    )


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
