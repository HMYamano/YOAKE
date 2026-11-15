"""
keyframes.py — label one image, take the time from the video.

The keyframe labelling scheme (``docs/KEYFRAME_BEHAVIOR.md``): a human gives
**a frame position, and a box plus a behavior for every animal in that frame**.
No track ids, no intervals. Temporal context is the clip of video that ends at
the keyframe, which costs nothing to label because the video already has it -
the central idea of YOWO, re-cut here for several look-alike animals.

===================  =====================================================
:class:`ClipSpec`       Which frames before a keyframe make up its clip.
:func:`label_mode_of`   Is this dataset keyframe-labelled (``meta.labeling``)?
:func:`context_frames`  Every frame a round must export for its clips.
:func:`summarize`       Keyframes, animals, and behavior labels, counted.
:func:`check`           What looks wrong for a keyframe dataset, in words.
:func:`to_keyframes`    Thin an old track/interval dataset to keyframes.
:class:`ContextLoop`    The clock and range of the annotator's context loop.
===================  =====================================================

What a keyframe *means* is the whole contract, so it is stated once, here:

* The boxes on a keyframe are **every** animal in that frame. Anywhere without
  a box is background, and a keyframe with no boxes says "nothing here".
* ``action_id`` is what this animal is doing at this instant, judged with the
  motion around it in view. ``-1`` means the behavior was not judged: the box
  still teaches the detector, and the behavior loss skips it.
* ``track_id`` is not a label. Identity comes from tracking at analysis time.

Purity: no dearpygui, no torch, no cv2, no numpy - like the rest of
:mod:`htrtdetr.annotation`, so it is testable from either interpreter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "KEYFRAME", "TRACKS", "DENSE", "AUTO",
    "PROJECT_MODES", "DATASET_MODES", "ANCHORS",
    "DEFAULT_CLIP_LENGTH", "DEFAULT_CLIP_STRIDE",
    "ClipSpec", "describe_clip",
    "labeling_of", "label_mode_of", "clip_spec_of", "labeling_meta",
    "set_labeling", "resolve_label_mode",
    "context_frames",
    "KeyframeSummary", "summarize", "check", "to_keyframes",
    "keyframe_marks", "frame_status", "keyframe_sentence",
    "context_window", "ContextLoop",
]

#: The project setting and the dataset declaration that mean "keyframe labels".
KEYFRAME = "keyframe"
#: Project setting for the original workflow: track ids and behavior intervals.
TRACKS = "tracks"
#: Dataset reading for a file without ``meta.labeling``: frames are a dense
#: run of video, and the legacy sliding-window training path applies.
DENSE = "dense"
#: Training config value: read the mode out of the annotation file.
AUTO = "auto"

PROJECT_MODES: Tuple[str, ...] = (KEYFRAME, TRACKS)
DATASET_MODES: Tuple[str, ...] = (KEYFRAME, DENSE)

#: ``end`` puts the keyframe last, as YOWO does, so the clip is causal and the
#: streaming inferencer can produce the same input frame by frame. ``center``
#: is defined for offline use; the model trains on ``end`` only.
ANCHORS: Tuple[str, ...] = ("end", "center")

#: 8 frames, every 4th: 29 frames of video, 0.93 s at 30 fps. Long enough for a
#: courtship approach or a grooming bout to show motion; short enough that a
#: round on a few hundred keyframes trains in minutes (cost grows with length).
DEFAULT_CLIP_LENGTH = 8
DEFAULT_CLIP_STRIDE = 4

#: The fewest context frames the loop shows on each side, whatever the fps
#: claims - the same floor :func:`htrtdetr.gui.review.context_range` uses.
_MIN_LOOP_PAD = 6


# ══════════════════════════════════════════════════════════════════════════
# The clip
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClipSpec:
    """``length`` frames, ``stride`` apart, with the keyframe at ``anchor``.

    The clip is defined by **frame position in the video**, never by position
    in the list of labelled frames. That distinction is the reason this class
    exists: the legacy ``SlidingWindowDataset`` windows over annotated frames,
    so sparse labels made it treat frames ten seconds apart as consecutive.
    """

    length: int = DEFAULT_CLIP_LENGTH
    stride: int = DEFAULT_CLIP_STRIDE
    anchor: str = "end"

    def __post_init__(self) -> None:
        length, stride = int(self.length), int(self.stride)
        if length < 1:
            raise ValueError(f"clip length must be at least 1, got {self.length}")
        if stride < 1:
            raise ValueError(f"clip stride must be at least 1, got {self.stride}")
        if self.anchor not in ANCHORS:
            raise ValueError(f"clip anchor must be one of {ANCHORS}, got {self.anchor!r}")
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "stride", stride)

    @property
    def span(self) -> int:
        """Frames of video the clip covers, first to last inclusive."""
        return (self.length - 1) * self.stride + 1

    @property
    def key_position(self) -> int:
        """Index of the keyframe inside the clip."""
        return self.length - 1 if self.anchor == "end" else (self.length - 1) // 2

    def offsets(self) -> Tuple[int, ...]:
        """Frame offsets from the keyframe, oldest first. ``end``: ``(-28, ..., -4, 0)``."""
        k = self.key_position
        return tuple((i - k) * self.stride for i in range(self.length))

    def indices(self, frame: int, num_frames: int = 0) -> Tuple[Tuple[int, ...], Tuple[bool, ...]]:
        """The clip's frame numbers for keyframe ``frame``, and which are real.

        A position before the first frame (or past the last, which only a
        ``center`` clip can reach) is clamped onto the video and reported as
        not valid. Clamping rather than dropping keeps every clip the same
        length, which the model needs; the mask keeps the repetition honest -
        a repeated frame has zero motion and must not be read as stillness.

        ``num_frames <= 0`` means the length is unknown and only the start is
        clamped.
        """
        last = int(num_frames) - 1 if int(num_frames) > 0 else None
        out: List[int] = []
        valid: List[bool] = []
        for off in self.offsets():
            f = int(frame) + off
            ok = f >= 0 and (last is None or f <= last)
            f = max(0, f)
            if last is not None:
                f = min(f, last)
            out.append(f)
            valid.append(ok)
        return tuple(out), tuple(valid)

    def seconds(self, fps: float) -> float:
        """Duration the clip spans, first frame to keyframe, in seconds."""
        return (self.span - 1) / float(fps) if fps and fps > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"length": self.length, "stride": self.stride, "anchor": self.anchor}

    @classmethod
    def from_dict(cls, raw: Any) -> "ClipSpec":
        """Read a ``meta.labeling.clip`` value. Anything unreadable is the default."""
        if isinstance(raw, ClipSpec):
            return raw
        if not isinstance(raw, Mapping):
            return cls()
        try:
            return cls(
                length=int(raw.get("length", DEFAULT_CLIP_LENGTH)),
                stride=int(raw.get("stride", DEFAULT_CLIP_STRIDE)),
                anchor=str(raw.get("anchor", "end") or "end"),
            )
        except (TypeError, ValueError):
            return cls()


def describe_clip(clip: ClipSpec, fps: float = 0.0) -> str:
    """``8 frames, every 4th - 0.93 s at 30 fps``. For dialogs and logs."""
    every = "every frame" if clip.stride == 1 else f"every {_ordinal(clip.stride)}"
    text = f"{clip.length} frames, {every}"
    if fps and fps > 0:
        text += f" - {clip.seconds(fps):.2f} s at {fps:g} fps"
    return text


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ══════════════════════════════════════════════════════════════════════════
# Reading a dataset's declaration
# ══════════════════════════════════════════════════════════════════════════

def _meta(dataset: Any) -> Mapping[str, Any]:
    if isinstance(dataset, Mapping):
        meta = dataset.get("meta")
    else:
        meta = getattr(dataset, "meta", None)
    return meta if isinstance(meta, Mapping) else {}


def labeling_of(dataset: Any) -> Mapping[str, Any]:
    """``meta.labeling`` of a dict or a ``DatasetAnno``; ``{}`` when absent."""
    raw = _meta(dataset).get("labeling")
    return raw if isinstance(raw, Mapping) else {}


def label_mode_of(dataset: Any) -> str:
    """:data:`KEYFRAME` if the file declares it, else :data:`DENSE`.

    Absent means dense on purpose: every file written before this scheme
    existed keeps training exactly the way it did.
    """
    return KEYFRAME if labeling_of(dataset).get("mode") == KEYFRAME else DENSE


def clip_spec_of(dataset: Any, default: Optional[ClipSpec] = None) -> ClipSpec:
    """The clip the file asks for, or ``default`` (the module default if None)."""
    raw = labeling_of(dataset).get("clip")
    if raw is None:
        return default if default is not None else ClipSpec()
    return ClipSpec.from_dict(raw)


def labeling_meta(clip: Optional[ClipSpec] = None) -> Dict[str, Any]:
    """The ``meta.labeling`` value that declares a keyframe dataset."""
    return {"mode": KEYFRAME, "clip": (clip or ClipSpec()).to_dict()}


def set_labeling(meta: Dict[str, Any], clip: Optional[ClipSpec] = None) -> Dict[str, Any]:
    """Declare ``meta`` keyframe-labelled, in place. Returns it for chaining."""
    meta["labeling"] = labeling_meta(clip)
    return meta


def resolve_label_mode(configured: str, dataset: Any = None) -> str:
    """Turn a training config's ``label_mode`` into :data:`KEYFRAME` or :data:`DENSE`.

    ``auto`` reads the dataset; anything else must already be one of the two.
    An unknown value raises rather than falling back, because the two paths
    train different things and a typo must not pick one silently.
    """
    value = str(configured or AUTO).strip().lower()
    if value == AUTO:
        return label_mode_of(dataset) if dataset is not None else DENSE
    if value in (KEYFRAME, DENSE):
        return value
    raise ValueError(
        f"data.label_mode must be 'auto', 'keyframe' or 'dense', got {configured!r}"
    )


# ══════════════════════════════════════════════════════════════════════════
# What a round has to export
# ══════════════════════════════════════════════════════════════════════════

def context_frames(
    keyframes: Iterable[int], clip: ClipSpec, num_frames: int = 0
) -> List[int]:
    """Every frame the clips of ``keyframes`` touch, sorted and unique.

    The keyframes themselves are included (they are the last frame of their
    own clip). Overlapping clips share frames, so 200 keyframes with an
    8-frame clip cost at most 1,600 frames and usually fewer.
    """
    wanted = set()
    for key in keyframes:
        frames, _valid = clip.indices(int(key), num_frames)
        wanted.update(frames)
    return sorted(wanted)


# ══════════════════════════════════════════════════════════════════════════
# Counting and checking
# ══════════════════════════════════════════════════════════════════════════

def _videos(dataset: Any) -> List[Mapping[str, Any]]:
    """The video rows of a dict or a ``DatasetAnno``, as mappings."""
    if isinstance(dataset, Mapping):
        return [v for v in (dataset.get("videos") or []) if isinstance(v, Mapping)]
    rows = []
    for v in getattr(dataset, "videos", []) or []:
        rows.append({
            "video_id": getattr(v, "video_id", ""),
            "num_frames": getattr(v, "num_frames", 0),
            "frames": [
                {"frame_index": getattr(f, "frame_index", 0),
                 "objects": list(getattr(f, "objects", []) or [])}
                for f in getattr(v, "frames", []) or []
            ],
        })
    return rows


def _field(obj: Any, name: str, default: int) -> int:
    value = obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class KeyframeSummary:
    """How much keyframe labelling a dataset holds."""

    videos: int = 0
    keyframes: int = 0
    empty_keyframes: int = 0
    instances: int = 0
    with_behavior: int = 0
    #: Instances per action id, ``len(action_names)`` long.
    per_action: Tuple[int, ...] = ()
    #: Boxes that still carry a track id. Harmless - training ignores it -
    #: but worth saying, because it means the file came from the old scheme.
    with_track_ids: int = 0

    @property
    def without_behavior(self) -> int:
        return self.instances - self.with_behavior

    def sentence(self, action_names: Sequence[str] = ()) -> str:
        """``120 keyframes, 236 animals, 201 with a behavior (stationary 80, ...)``."""
        if not self.keyframes:
            return "No keyframes yet."
        text = (
            f"{self.keyframes:,} keyframe{'s' if self.keyframes != 1 else ''}, "
            f"{self.instances:,} animal{'s' if self.instances != 1 else ''}, "
            f"{self.with_behavior:,} with a behavior"
        )
        named = [
            f"{action_names[i]} {n:,}"
            for i, n in enumerate(self.per_action)
            if n and i < len(action_names)
        ]
        if named:
            text += " (" + ", ".join(named) + ")"
        return text + "."


def summarize(dataset: Any, action_names: Sequence[str] = ()) -> KeyframeSummary:
    """Count keyframes, animals and behavior labels."""
    n_actions = len(action_names)
    per_action = [0] * n_actions
    videos = keyframes = empty = instances = with_behavior = tracked = 0
    for video in _videos(dataset):
        frames = list(video.get("frames") or [])
        if not frames:
            continue
        videos += 1
        for frame in frames:
            keyframes += 1
            objects = list(frame.get("objects") or [])
            if not objects:
                empty += 1
            for obj in objects:
                instances += 1
                action = _field(obj, "action_id", -1)
                if action >= 0:
                    with_behavior += 1
                    if action < n_actions:
                        per_action[action] += 1
                if _field(obj, "track_id", -1) >= 0:
                    tracked += 1
    return KeyframeSummary(
        videos=videos,
        keyframes=keyframes,
        empty_keyframes=empty,
        instances=instances,
        with_behavior=with_behavior,
        per_action=tuple(per_action),
        with_track_ids=tracked,
    )


def check(
    dataset: Any,
    action_names: Sequence[str] = (),
    clip: Optional[ClipSpec] = None,
) -> List[str]:
    """Plain-language warnings for a keyframe dataset. Empty means none.

    Warnings, not errors: every item here still trains. They are the things a
    person would want to know before spending a round on it.
    """
    clip = clip or clip_spec_of(dataset)
    out: List[str] = []
    duplicates = 0
    past_end = 0
    clipped_start = 0
    bad_action = 0
    for video in _videos(dataset):
        num_frames = _field(video, "num_frames", 0)
        seen = set()
        for frame in video.get("frames") or []:
            index = _field(frame, "frame_index", 0)
            if index in seen:
                duplicates += 1
            seen.add(index)
            if num_frames > 0 and index >= num_frames:
                past_end += 1
            _frames, valid = clip.indices(index, num_frames)
            if not all(valid):
                clipped_start += 1
            for obj in frame.get("objects") or []:
                action = _field(obj, "action_id", -1)
                if action_names and action >= len(action_names):
                    bad_action += 1

    summary = summarize(dataset, action_names)
    if duplicates:
        out.append(f"{duplicates} keyframe(s) appear twice in the same video; "
                   "only one copy of each should be kept.")
    if past_end:
        out.append(f"{past_end} keyframe(s) lie past the end of their video.")
    if bad_action:
        out.append(f"{bad_action} box(es) have a behavior number with no name "
                   f"(there are {len(action_names)} behaviors).")
    if clipped_start:
        out.append(f"{clipped_start} keyframe(s) are closer to the start of the "
                   "video than one clip; their clips repeat the first frame.")
    if summary.with_track_ids:
        out.append(f"{summary.with_track_ids:,} box(es) carry a track id. Keyframe "
                   "training ignores it - identity comes from tracking.")
    if summary.instances and not summary.with_behavior:
        out.append("No box has a behavior yet, so only the detector can be trained.")
    missing = [
        str(action_names[i]) for i, n in enumerate(summary.per_action)
        if n == 0 and i < len(action_names)
    ]
    if summary.with_behavior and missing:
        out.append("No examples yet of: " + ", ".join(missing) + ".")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Converting the old scheme
# ══════════════════════════════════════════════════════════════════════════

def to_keyframes(
    dataset: Mapping[str, Any],
    *,
    every: int = 15,
    clip: Optional[ClipSpec] = None,
    keep_track_ids: bool = False,
) -> Dict[str, Any]:
    """Thin a track/interval dataset (dict form) to keyframes. Input untouched.

    Keeps labelled frames at least ``every`` frames apart, greedily from the
    first, so a dense hour of labels becomes one keyframe per half second at
    30 fps and a sparse file loses nothing. Behavior labels derived from
    intervals are kept as they are - they were per-frame ``action_id`` on disk
    all along. Track ids become ``-1`` unless ``keep_track_ids``: they are not
    labels under this scheme, and leaving them in invites code that reads them.
    """
    every = max(1, int(every))
    out = copy.deepcopy(dict(dataset))
    meta = dict(out.get("meta") or {})
    set_labeling(meta, clip or clip_spec_of(dataset))
    out["meta"] = meta
    videos = []
    for video in out.get("videos") or []:
        frames = sorted(
            (f for f in video.get("frames") or [] if isinstance(f, dict)),
            key=lambda f: int(f.get("frame_index", 0)),
        )
        kept = []
        last: Optional[int] = None
        for frame in frames:
            index = int(frame.get("frame_index", 0))
            if last is not None and index - last < every:
                continue
            last = index
            if not keep_track_ids:
                for obj in frame.get("objects") or []:
                    if isinstance(obj, dict):
                        obj["track_id"] = -1
            kept.append(frame)
        video["frames"] = kept
        videos.append(video)
    out["videos"] = videos
    return out


# ══════════════════════════════════════════════════════════════════════════
# For the annotator
# ══════════════════════════════════════════════════════════════════════════

def keyframe_marks(annotations: Mapping[int, Sequence[Any]]) -> List[Tuple[int, int]]:
    """``(frame, action_id)`` for every box with a behavior, frame order.

    The timeline's behavior lane in keyframe mode: one tick per labelled
    animal, coloured by what it is doing, instead of interval bars.
    """
    marks: List[Tuple[int, int]] = []
    for frame in sorted(annotations):
        for obj in annotations.get(frame) or ():
            action = _field(obj, "action_id", -1)
            if action >= 0:
                marks.append((int(frame), action))
    return marks


def frame_status(objects: Sequence[Any]) -> Tuple[int, int]:
    """``(boxes, boxes without a behavior)`` on one keyframe."""
    boxes = len(objects)
    missing = sum(1 for obj in objects if _field(obj, "action_id", -1) < 0)
    return boxes, missing


def keyframe_sentence(objects: Sequence[Any]) -> str:
    """``3 animals - 1 still needs a behavior.`` for the Inspector."""
    boxes, missing = frame_status(objects)
    if boxes == 0:
        return "No animals boxed on this frame."
    text = f"{boxes} animal{'s' if boxes != 1 else ''}"
    if missing:
        return text + f" - {missing} still need{'s' if missing == 1 else ''} a behavior."
    return text + " - every one has a behavior."


def context_window(
    frame: int,
    fps: float,
    *,
    before_s: float = 1.0,
    after_s: float = 0.5,
    num_frames: int = 0,
) -> Tuple[int, int]:
    """``(first, last)`` frames of the context loop around a keyframe.

    More before than after on purpose: the model sees the past (``end``
    anchor), so the person should judge from mostly the same evidence, plus a
    little of what happens next to settle an ambiguous instant.
    """
    rate = float(fps) if fps and fps > 0 else 25.0
    before = max(_MIN_LOOP_PAD, int(round(before_s * rate)))
    after = max(_MIN_LOOP_PAD, int(round(after_s * rate)))
    lo = max(0, int(frame) - before)
    hi = int(frame) + after
    if num_frames > 0:
        hi = min(hi, int(num_frames) - 1)
    return lo, max(lo, hi)


class ContextLoop:
    """The clock and range of the context loop. The GUI does the drawing.

    The loop never moves the keyframe: it shows other frames *about* the one
    being labelled, and :meth:`stop` hands the keyframe back so the canvas can
    return to it. Labels are only ever attached to that keyframe.
    """

    #: Playback is capped like Review Mode's loop: a 240 fps recording looped
    #: at 240 fps is a blur, and the render loop's own rate differs by machine.
    MAX_FPS = 30.0

    def __init__(self) -> None:
        self.anchor: Optional[int] = None
        self.lo = 0
        self.hi = 0
        self.position = 0
        self.speed = 1.0
        self._last = 0.0

    @property
    def active(self) -> bool:
        return self.anchor is not None

    def start(self, anchor: int, lo: int, hi: int, now: float = 0.0) -> None:
        if hi < lo:
            raise ValueError(f"loop ends before it starts: {lo} -> {hi}")
        self.anchor = int(anchor)
        self.lo, self.hi = int(lo), int(hi)
        self.position = self.lo
        self._last = float(now)

    def stop(self) -> Optional[int]:
        """End the loop and return the keyframe to go back to."""
        anchor, self.anchor = self.anchor, None
        return anchor

    def due(self, now: float, fps: float) -> bool:
        if not self.active:
            return False
        rate = min(float(fps) if fps and fps > 0 else 25.0, self.MAX_FPS)
        interval = 1.0 / max(1e-6, rate * max(self.speed, 0.05))
        if now - self._last < interval:
            return False
        self._last = float(now)
        return True

    def advance(self) -> int:
        """Next frame of the loop, wrapping from the last back to the first."""
        nxt = self.position + 1
        self.position = self.lo if nxt > self.hi or nxt < self.lo else nxt
        return self.position

    def tick(self, now: float, fps: float) -> Optional[int]:
        """The frame to show now, or None if it is not time yet."""
        return self.advance() if self.due(now, fps) else None

    def offset_seconds(self, fps: float) -> float:
        """Signed time from the keyframe to the frame on screen."""
        if self.anchor is None or not fps or fps <= 0:
            return 0.0
        return (self.position - self.anchor) / float(fps)

    def progress(self) -> float:
        """Where in the loop, 0 at its first frame and 1 at its last."""
        if self.hi <= self.lo:
            return 0.0
        return (self.position - self.lo) / float(self.hi - self.lo)
