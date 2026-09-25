"""
finalpass.py — the last pass: every frame scanned, and the few a human visits.

After about ten rounds of the active-learning loop the model has stopped
learning much from another two hundred corrected frames (``round.series_note``
says so out loud). What is left is the thing the project was for: run the model
over **all** 2.16M frames of the twenty one-hour videos, satisfy yourself that
the result is right, and export it as a dataset.

Two decisions shape this whole module, and both of them are refusals.

**The human does not look at every frame.** At one second a frame, 2.16M frames
is eight person-years. So the pass does not ask for a verdict per frame; it asks
for a verdict per *place*. The quality checks (:mod:`~htrtdetr.annotation.quality`)
already turn a run of predictions into the moments that are internally
inconsistent, and :func:`segments` turns those - plus the frames where the box
count, the track set or the confidence changes - into a tiling of the video into
spans that were flagged and spans that were not. The human is taken to the start
of each flagged span, plus :data:`SPOT_CHECKS` evenly spaced frames inside the
*clean* spans, and then signs the video off. That is about 134 stops and 25
minutes a video: eight hours for the study, against eight years.

The spot checks are not padding. Everything else in this file only ever shows
the human places where **the model already doubted itself**, and the worst
failure mode of a detector is not doubt - it is being stably, confidently wrong.
A second animal it never once detects produces no low score, no count change and
no broken track, so it produces no flagged span at all, and a review built only
from flags would sign off a video with half its data missing and never show the
human a single frame of it. See :func:`spot_check_frames`.

**The exported dataset is labels plus a path to the video.** Not an image tree:
2.16M JPEGs at ~40 KB each is ~86 GB and, at the 308 fps the decoder manages
end to end, about twenty hours of writing them out. The boxes themselves are
tens of MB, and a video file the study already has is a better store of pixels
than a folder of stills copied out of it.

The ledger
----------
The model's answer for a whole video lives in one JSON-lines file,
``datasets/final/<video_id>.jsonl`` (:meth:`~htrtdetr.annotation.layout.ProjectLayout.final_ledger`),
never in the project manifest and never in a review session: the manifest is
rewritten whole on every autosave and the session is deep-copied on every undo,
and 216,000 boxes routed through either turns a keystroke into a second of work.

Line 1 is a header object. Every other line is one frame **that has boxes**::

    {"video": "cam1", "weights": "cam1.mp4:8123:17...", "num_frames": 108000, "fps": 30.0}
    {"f": 1234, "b": [[x1, y1, x2, y2, class_id, score, track_id], ...]}

Three size decisions are baked into that row shape, and together they are the
difference between a file measured in tens of MB and one measured in hundreds:
a frame with no boxes is **absent** rather than written as an empty list; a box
is a positional list rather than an object, because seven keys repeated 216,000
times are megabytes of the word ``"track_id"``; and the score is rounded to
three decimals, because ``0.8232181072235107`` is 18 characters that say nothing
``0.823`` does not.

What the row does **not** carry is the oriented box. The ledger's job is to
decide *where a human should look* and to carry the axis-aligned answer into the
release; a project that annotates orientation still holds its human-drawn OBBs
in ``annotations.json``, which is what the exporter merges over the top.

Reading is by window, never whole. ``<video_id>.jsonl.idx`` holds a byte offset
every :data:`INDEX_STRIDE` frames, so the review dock jumping to frame 91,000
seeks there instead of parsing the 91,000 rows in front of it.

Purity: no dearpygui, no torch, no cv2, no numpy. Boxes are duck-typed - a
:class:`htrtdetr.gui.auto_label.Candidate`, an
:class:`htrtdetr.annotation.objects.AnnObject` and a plain
``[x1, y1, x2, y2, class_id, score, track_id]`` list are all accepted, because
this module sits between a detector's output and a human's edits and sees all
three.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Mapping, Optional,
                    Sequence, Tuple)

__all__ = [
    # the ledger
    "INDEX_STRIDE",
    "write_header",
    "append_rows",
    "read_header",
    "read_window",
    "iter_rows",
    "frames_written",
    "last_frame",
    # the stops
    "Segment",
    "MIN_SEGMENT",
    "SPOT_CHECKS",
    "STOP_BUDGET",
    "stop_budget_note",
    "segments",
    "spot_check_frames",
    "stops",
    # the state
    "PassState",
    "mark_verified",
    "covers",
    "visited",
    "can_sign_off",
    "sign_off",
    "unsign",
    "progress_sentence",
]


# ══════════════════════════════════════════════════════════════════════════
# The ledger
# ══════════════════════════════════════════════════════════════════════════

#: Frames between entries in the sidecar index.
#:
#: One hour at 30 fps is 108,000 frames, so this is 108 index entries - about
#: 3 KB, cheap enough to load whole every time a window is read. The other end
#: of the trade is how much a window read has to scan past its seek: at most
#: 1,000 rows, which at two animals a frame is tens of KB. A stride of 100
#: would make the index 30 KB to save reading 20 KB, and a stride of 10,000
#: would make the dock's first paint parse a tenth of the video.
INDEX_STRIDE = 1000

#: Decimal places kept for a score. Three is the last one a human could act on
#: differently; see the module docstring for what the rest cost.
_SCORE_DIGITS = 3


def _idx_path(path: Any) -> Path:
    """``<ledger>.idx``. A suffix, not a sibling with a different stem, so a
    stray ledger and its index cannot be separated by a copy or a rename."""
    p = Path(path)
    return p.with_name(p.name + ".idx")


def _box_row(box: Any) -> Optional[List[Any]]:
    """One box as ``[x1, y1, x2, y2, class_id, score, track_id]``, or ``None``.

    Accepts the three shapes this module can be handed (see the module
    docstring). ``None`` for anything without usable coordinates: a box that
    cannot be parsed is dropped rather than written as zeros, because a
    ``[0, 0, 0, 0]`` row in the ledger is indistinguishable from a real
    detection in the corner and would be reviewed as one.
    """
    if isinstance(box, (list, tuple)):
        if len(box) < 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in box[:4])
        except (TypeError, ValueError):
            return None
        class_id = int(box[4]) if len(box) > 4 else 0
        score = float(box[5]) if len(box) > 5 else 1.0
        track_id = int(box[6]) if len(box) > 6 else -1
    else:
        raw = getattr(box, "bbox", None)
        if raw is None:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in list(raw)[:4])
        except (TypeError, ValueError):
            return None
        class_id = int(getattr(box, "class_id", 0) or 0)
        score = _certainty(box)
        # The one field with no guard, and the only one whose ``None`` is
        # routine: a box drawn in a project that does not follow individuals
        # has no track. ``int(None)`` raises, and it raises inside the loop
        # that writes a whole chunk - so one untracked box would lose every
        # frame in that append, in a two-hour pass, silently retried.
        raw_track = getattr(box, "track_id", -1)
        track_id = -1 if raw_track is None else int(raw_track)
    return [
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
        class_id,
        round(score, _SCORE_DIGITS),
        track_id,
    ]


def _dumps(obj: Any) -> bytes:
    """Compact JSON as bytes.

    Bytes, and every read and write in this module is binary, because of
    Windows: a file opened in text mode translates ``"\\n"`` into ``"\\r\\n"`` on
    the way out, and ``tell()`` on a text stream returns an opaque cookie
    rather than a byte position. Either one alone makes the offsets in the
    index point into the middle of a line, and the failure appears only on the
    seek - long after the file was written, on the machine that wrote it.
    """
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def write_header(path: Any, *, video: str, weights: str, num_frames: int, fps: float) -> None:
    """Start a ledger, discarding whatever was there.

    Truncating is the point: the header names the checkpoint that produced the
    file (``weights``, a :func:`~htrtdetr.annotation.cache.video_fingerprint`),
    so a pass run with a different model is a different file and must not be
    appended to the old one. Boxes from two models interleaved in one ledger
    cannot be told apart afterwards - the rows carry no provenance - and the
    human would sign off a mixture.

    The index is truncated with it, for the same reason and one more: an index
    left behind from a longer previous run points past the end of the new file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "video": str(video),
        "weights": str(weights),
        "num_frames": int(num_frames),
        "fps": float(fps),
    }
    with open(p, "wb") as fh:
        fh.write(_dumps(header))
    _write_index(p, {"stride": INDEX_STRIDE, "offsets": [], "rows": 0, "last": -1,
                     "bytes": p.stat().st_size})


def _write_index(path: Path, index: Dict[str, Any]) -> None:
    with open(_idx_path(path), "w", encoding="utf-8") as fh:
        json.dump(index, fh, separators=(",", ":"))


def _read_index(path: Any) -> Dict[str, Any]:
    """The sidecar index, or an empty one.

    Never raises. The index is derived data - it can always be rebuilt by
    reading the ledger - so a corrupt or missing index must degrade into a
    slower read, not into a lost pass.
    """
    try:
        with open(_idx_path(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    offsets = [
        [int(f), int(o)]
        for f, o in (data.get("offsets") or [])
        if isinstance(f, (int, float)) and isinstance(o, (int, float))
    ]
    return {
        "stride": int(data.get("stride") or INDEX_STRIDE),
        "offsets": offsets,
        "rows": int(data.get("rows") or 0),
        "last": int(data.get("last", -1)),
        "bytes": int(data.get("bytes") or 0),
    }


def _size(path: Any) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def _scan(path: Any) -> Tuple[int, int, List[List[int]]]:
    """``(rows, last_frame, offsets)`` by reading the whole file.

    The slow path, taken only when the index disagrees with the file's size -
    which means the machine stopped in the middle of an append. It is here so
    that case costs one scan rather than the pass.
    """
    rows = 0
    last = -1
    offsets: List[List[int]] = []
    bucket = -1
    for frame, _boxes, offset in _iter_raw(path):
        rows += 1
        last = max(last, frame)
        if frame // INDEX_STRIDE > bucket:
            bucket = frame // INDEX_STRIDE
            offsets.append([frame, offset])
    return rows, last, offsets


def _state(path: Any) -> Dict[str, Any]:
    """The index, rebuilt first if it does not describe the file on disk."""
    index = _read_index(path)
    size = _size(path)
    if index and index.get("bytes") == size:
        return index
    rows, last, offsets = _scan(path)
    return {"stride": INDEX_STRIDE, "offsets": offsets, "rows": rows,
            "last": last, "bytes": size}


def _truncate_partial_line(path: Any) -> None:
    """Cut back to the last complete row before anything is appended.

    A two-hour pass **will** be interrupted, and what an interruption leaves is
    a half-written final line. :func:`_iter_raw` already drops that line when
    reading, which is why reading a killed ledger works. Appending to it does
    not: ``open(path, "ab")`` starts at the end of the fragment, so the
    fragment and the first new row fuse into one line that parses as neither.
    The cost is not the one frame that was in flight - it is that the fused
    line is dropped by every later read while the index says the rows are
    there, so ``frames_written`` and ``iter_rows`` disagree for the rest of the
    file's life, on a file a human is about to sign off.

    Does nothing to a file that ends in a newline, which is every file that was
    closed properly, so the resume path pays one stat and one small read.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return
    if not size:
        return
    with open(p, "r+b") as fh:
        fh.seek(max(0, size - 1))
        if fh.read(1) == b"\n":
            return
        # Walk back to the last newline. A ledger row is a few hundred bytes,
        # so this reads one block in practice; a file with no newline at all
        # is truncated to nothing, which is correct - it holds no complete row.
        window = min(size, 1 << 16)
        fh.seek(size - window)
        tail = fh.read(window)
        cut = tail.rfind(b"\n")
        fh.truncate(size - window + cut + 1 if cut >= 0 else 0)


def append_rows(path: Any, rows: Any) -> int:
    """Append frames to a ledger, in ascending order. Returns how many.

    ``rows`` is ``{frame: [box, ...]}`` or an iterable of ``(frame, boxes)``.
    Frames with no boxes are skipped, which is the absence the reader relies on
    (see the module docstring).

    **Frames at or below the ledger's watermark are ignored.** That single rule
    is what makes a final pass resumable: the pass is two hours of GPU time, it
    will be interrupted, and the obvious resume - re-run from a round number of
    frames back and append whatever comes out - would otherwise duplicate every
    frame in the overlap. A duplicated frame is not a crash; it is a second row
    for frame 91,000 that :func:`read_window` returns instead of the first,
    with different boxes, in a file the human has already signed off part of.
    Ignoring them also makes appending the same rows twice a no-op, so a caller
    that cannot tell whether its last write landed can simply repeat it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(rows.items()) if hasattr(rows, "items") else sorted(rows, key=lambda r: r[0])

    _truncate_partial_line(p)
    index = _state(p)
    last = int(index["last"])
    offset = _size(p)
    offsets: List[List[int]] = list(index["offsets"])
    bucket = offsets[-1][0] // INDEX_STRIDE if offsets else -1
    written = 0

    with open(p, "ab") as fh:
        for frame, boxes in items:
            frame = int(frame)
            if frame <= last:
                continue
            packed = [row for row in (_box_row(b) for b in (boxes or ())) if row is not None]
            if not packed:
                last = frame
                continue
            line = _dumps({"f": frame, "b": packed})
            if frame // INDEX_STRIDE > bucket:
                bucket = frame // INDEX_STRIDE
                offsets.append([frame, offset])
            fh.write(line)
            offset += len(line)
            last = frame
            written += 1

    _write_index(p, {"stride": INDEX_STRIDE, "offsets": offsets,
                     "rows": int(index["rows"]) + written, "last": last,
                     "bytes": _size(p)})
    return written


def _iter_raw(path: Any) -> Iterator[Tuple[int, List[List[Any]], int]]:
    """``(frame, boxes, byte offset)`` for every data row, header skipped.

    A row that does not parse is **dropped in silence**, and the last row of a
    ledger is the one that most often does not: a two-hour pass killed by a
    power cut or a closed laptop leaves a half-written final line. Raising
    there would mean the interruption cost the human the whole ledger rather
    than the one frame that was in flight. A header is recognised by having no
    ``"f"`` - by shape and not by being line 1 - so a ledger that somehow lost
    its header still reads.
    """
    try:
        fh = open(path, "rb")
    except OSError:
        return
    with fh:
        offset = 0
        for raw in fh:
            here = offset
            offset += len(raw)
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(obj, dict) or "f" not in obj:
                continue
            boxes = obj.get("b") or []
            if not isinstance(boxes, list):
                continue
            yield int(obj["f"]), boxes, here


def read_header(path: Any) -> Dict[str, Any]:
    """The header object, or ``{}`` when there is not one to read.

    ``{}`` rather than an exception for the case the caller actually has: a
    project whose final pass has never run, where every video asks this
    question once and none of them has a file yet.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.readline()
    except OSError:
        return {}
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {}
    if not isinstance(obj, dict) or "f" in obj:
        return {}
    return obj


def read_window(path: Any, lo: int, hi: int) -> Dict[int, List[List[Any]]]:
    """The rows for frames ``lo..hi``, **inclusive**, read with a seek.

    This is what the review dock calls every time the human is taken to a stop,
    and it is why the index exists: without it, showing frame 91,000 of a
    108,000-frame ledger means parsing 91,000 rows of JSON the human will never
    see, once per stop, 134 times a video. With it the read touches the one
    :data:`INDEX_STRIDE`-frame block the window starts in, plus however many
    blocks the window itself spans.

    Reading stops at the first row past ``hi`` rather than continuing to the
    end of the file, for the same reason.
    """
    lo, hi = int(lo), int(hi)
    if hi < lo:
        return {}

    # ``_state``, not ``_read_index``: an index whose ``bytes`` no longer match
    # the file describes a ledger that has since been truncated or extended,
    # and seeking to one of its offsets lands past the end of it. The read
    # would then return ``{}`` - indistinguishable from "no boxes in this
    # window" - so a damaged ledger would show as a stretch the model found
    # nothing in, which is exactly the stretch a human is about to sign off.
    index = _state(path)
    offsets = index.get("offsets") or []
    start = 0
    if offsets:
        frames = [f for f, _ in offsets]
        pos = bisect_right(frames, lo) - 1
        start = offsets[max(0, pos)][1]

    out: Dict[int, List[List[Any]]] = {}
    try:
        fh = open(path, "rb")
    except OSError:
        return {}
    with fh:
        fh.seek(start)
        for raw in fh:
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                # Either the truncated last line, or - when no index existed
                # and the scan began at byte 0 - the header. Neither is fatal.
                continue
            if not isinstance(obj, dict) or "f" not in obj:
                continue
            frame = int(obj["f"])
            if frame > hi:
                break
            if frame >= lo:
                boxes = obj.get("b") or []
                if isinstance(boxes, list):
                    out[frame] = boxes
    return out


def iter_rows(path: Any) -> Iterator[Tuple[int, List[List[Any]]]]:
    """``(frame, boxes)`` for the whole ledger, ascending.

    The exporter's entry point, and the one caller that legitimately wants
    every row. A generator rather than a list because the point of the ledger
    is that the whole video's boxes are never held in memory at once.
    """
    for frame, boxes, _offset in _iter_raw(path):
        yield frame, boxes


def frames_written(path: Any) -> int:
    """How many frame rows the ledger holds.

    Answered from the index when the index still describes the file, so the
    progress line can be drawn without reading tens of MB; by a scan when it
    does not, which is exactly the interrupted-write case where the cached
    count would be one too high.
    """
    return int(_state(path)["rows"])


def last_frame(path: Any) -> int:
    """The highest frame in the ledger, or ``-1``.

    The watermark an interrupted pass resumes from: the next scan starts at
    ``last_frame(path) + 1``. Note that it counts frames the pass *reached*,
    including ones it wrote nothing for because they held no boxes - a pass
    that ended in a long empty stretch must not re-scan it.
    """
    return int(_state(path)["last"])


# ══════════════════════════════════════════════════════════════════════════
# The stops a human visits
# ══════════════════════════════════════════════════════════════════════════

#: Frames below which a span is not a place. One second at 30 fps.
#:
#: A stop costs a seek, a look and a decision - call it ten seconds - whatever
#: its length, so a queue of 400 half-second spans is not a more thorough
#: review than a queue of 60 spans covering the same frames; it is the same
#: review made four times as long and abandoned halfway.
#:
#: The two short cases are handled in **opposite** directions, and that is the
#: safety property of :func:`segments`. A flagged span shorter than this is
#: **grown** to it, never dropped - a two-frame ID-switch risk becomes a
#: one-second span the human can actually see, and no place the quality checks
#: asked for can be merged away. A clean sliver shorter than this between two
#: flagged spans is **absorbed** into them, because sending someone back to
#: the video for a fifteen-frame gap between two things they are already
#: looking at is a stop that buys nothing.
MIN_SEGMENT = 30

#: Frames sampled from the clean spans of one video, whatever the checks said.
#:
#: Sized against the rest of the review: a video runs to roughly 94 flagged
#: spans, so 40 spot checks make ~134 stops, about 25 minutes, and eight hours
#: for the twenty-video study. They are the only part of the review that is not
#: chosen by the model - see :func:`spot_check_frames` for why that matters
#: more than the other 94 put together.
SPOT_CHECKS = 40

#: The most stops one video may put in front of a human.
#:
#: Sign-off is gated on visiting every stop, so this number is the difference
#: between a video that can be finished and one that cannot. 150 stops at the
#: ten seconds or so each takes is about twenty-five minutes a video, which is
#: eight hours over the twenty-video study - the budget the whole final pass
#: was sized against.
#:
#: Measured against the alternative. An hour in which one of two flies is
#: briefly lost every couple of seconds - routine when they climb over each
#: other - produces about 1,800 flagged segments. Unbounded, that is five hours
#: for one video of twenty, spent answering the same question about the same
#: two flies eighteen hundred times, and sign-off would never come.
STOP_BUDGET = 150


@dataclass(frozen=True)
class Segment:
    """One span of a video, and whether anything asked a human to look at it.

    ``end`` is **inclusive**, matching :class:`htrtdetr.annotation.sampling.Clip`
    and :class:`htrtdetr.annotation.quality.ReviewItem`. One convention for a
    range across the whole package: the mixed convention's off-by-one is
    invisible until it silently drops the last frame of every span, and here
    that last frame is one a human signed off.

    Attributes:
        start: First frame.
        end: Last frame, inclusive.
        kind: ``"flagged"`` (something wants a human) or ``"clean"``.
        reason: Why it is flagged, in the words the review dock shows. Empty
            for a clean segment - a clean segment is not an assertion that the
            frames are right, only that nothing objected.
    """

    start: int
    end: int
    kind: str = "clean"
    reason: str = ""

    @property
    def span(self) -> int:
        """Frames covered, inclusive."""
        return self.end - self.start + 1

    @property
    def flagged(self) -> bool:
        return self.kind == "flagged"

    def middle(self) -> int:
        return (self.start + self.end) // 2

    def seconds(self, fps: float) -> float:
        return self.span / fps if fps > 0 else 0.0


def _certainty(box: Any) -> float:
    """A box's score, where **no score means certain**.

    :class:`~htrtdetr.annotation.objects.AnnObject` declares ``score: float |
    None = None`` (``objects.py``), so a box a human drew by hand has the
    attribute and it is ``None``. A ``getattr(box, "score", 1.0) or 0.0``
    therefore finds the attribute, finds it falsy, and returns **0.0** - which
    makes the most trustworthy box in the project the lowest-confidence row in
    the ledger, and sends :func:`segments` to flag every frame the human has
    already finished. A missing score is the absence of a machine's doubt, not
    a machine's certainty that it is wrong.

    ``0.0`` written explicitly is still ``0.0``: only ``None`` and a missing
    attribute mean certain.
    """
    if isinstance(box, (list, tuple)):
        return float(box[5]) if len(box) > 5 and box[5] is not None else 1.0
    raw = getattr(box, "score", None)
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _score_of(box: Any) -> float:
    return _certainty(box)


def _track_of(box: Any) -> int:
    if isinstance(box, (list, tuple)):
        return int(box[6]) if len(box) > 6 else -1
    try:
        return int(getattr(box, "track_id", -1))
    except (TypeError, ValueError):
        return -1


def _runs_of(frames: Sequence[int], reason: str) -> List[Tuple[int, int, str]]:
    """Consecutive frames carrying the same reason, as ``(start, end, reason)``.

    Kept as runs rather than as one entry per frame because a video where the
    detector is unsure for a whole minute produces 1,800 identical reasons, and
    the segment builder would then carry 1,800 copies of one sentence.
    """
    out: List[Tuple[int, int, str]] = []
    for frame in frames:
        if out and frame == out[-1][1] + 1:
            out[-1] = (out[-1][0], frame, reason)
        else:
            out.append((frame, frame, reason))
    return out


def _reasons_for(spans: Sequence[Tuple[int, int, str]], start: int, end: int) -> str:
    """The distinct reasons overlapping ``start..end``, in the order found.

    Capped at three. A segment that collected nine reasons is not more
    informative than one that collected three; it is a sentence nobody reads
    in a dock that is 260 px wide.
    """
    seen: List[str] = []
    for a, b, reason in spans:
        if a > end:
            break
        if b >= start and reason not in seen:
            seen.append(reason)
    if len(seen) > 3:
        seen = seen[:3]
    return "; ".join(seen)


def segments(
    rows: Mapping[int, Sequence[Any]],
    issues: Iterable[Any] = (),
    *,
    num_frames: int,
    expected: Optional[int] = None,
    min_score: float = 0.0,
) -> List[Segment]:
    """Tile ``[0, num_frames)`` into flagged and clean spans.

    The result **covers every frame exactly once**, with no gap and no overlap.
    That is not tidiness: the two things built on top of this are "take me to
    every flagged span" and "spot-check the rest", and a frame that belongs to
    no segment is a frame in neither - it would be neither reviewed nor
    sampled, and nothing downstream could notice.

    A frame is flagged when any of these is true:

    * a :class:`~htrtdetr.annotation.quality.ReviewItem` covers it. These are
      the real checks - teleporting tracks, merged boxes, contact events - and
      this function does not repeat their work, it only places them.
    * the number of boxes changed from the previous frame. An animal appearing
      or disappearing is the single cheapest signal that something happened,
      and it fires in cases no per-frame check does.
    * the set of track ids changed. A frame with the same *count* but different
      ids is an identity swap or a re-detection, which is the error that
      quietly ruins a trajectory dataset.
    * a box scores below ``min_score``.
    * ``expected`` is set and the box count is not it. Passing the animal count
      the experiment has is the highest-value thing a caller can do here, for
      the reason :class:`~htrtdetr.annotation.quality.QualityConfig` gives: it
      is knowledge the model does not have and the human always does.

    Everything else is clean. Spans shorter than :data:`MIN_SEGMENT` are then
    tidied in the two opposite directions that constant documents: a short
    flagged span grows to it, a short clean gap between flagged spans is
    absorbed. Both moves can only ever make a human look at *more* frames, so
    no flagged interval can be swallowed - which is the one property of this
    function everything downstream is entitled to assume.
    """
    total = max(0, int(num_frames))
    if total <= 0:
        return []

    flags = bytearray(total)
    spans: List[Tuple[int, int, str]] = []

    def mark(lo: int, hi: int) -> None:
        for f in range(max(0, lo), min(total - 1, hi) + 1):
            flags[f] = 1

    for item in issues or ():
        start = int(getattr(item, "start", 0))
        end = int(getattr(item, "end", start))
        if end < 0 or start >= total:
            continue
        mark(start, end)
        label = getattr(item, "label", "") or getattr(item, "kind", "") or "flagged"
        spans.append((max(0, start), min(total - 1, end), str(label)))

    # ── the per-frame signals ────────────────────────────────────────────
    # Walked over every frame of the video and not only over the frames the
    # ledger holds: a frame with no row has *no boxes*, and "the count dropped
    # to zero" is the most important count change there is.
    count_hits: List[int] = []
    track_hits: List[int] = []
    score_hits: List[int] = []
    expect_hits: List[int] = []
    previous_count = -1
    previous_tracks: frozenset = frozenset()

    for frame in range(total):
        boxes = rows.get(frame) or ()
        count = len(boxes)
        tracks = frozenset(_track_of(b) for b in boxes)
        if previous_count >= 0 and count != previous_count:
            count_hits.append(frame)
        elif tracks != previous_tracks and frame > 0:
            # ``elif`` on purpose: when the count changed, the track set
            # changed too, and both reasons on one frame say the same thing
            # twice in a dock that shows three.
            track_hits.append(frame)
        if expected is not None and count != int(expected):
            expect_hits.append(frame)
        if min_score > 0.0 and any(_score_of(b) < min_score for b in boxes):
            score_hits.append(frame)
        previous_count, previous_tracks = count, tracks

    for frames, reason in (
        (count_hits, "the number of boxes changes"),
        (track_hits, "the tracks change"),
        (score_hits, f"a box scores below {min_score:.2f}"),
        (expect_hits, f"not {expected} animals"),
    ):
        if not frames:
            continue
        for frame in frames:
            flags[frame] = 1
        spans.extend(_runs_of(frames, reason))

    spans.sort(key=lambda s: (s[0], s[1]))
    return _tile(_flag_runs(flags), spans, total)


def _flag_runs(flags: bytearray) -> List[Tuple[int, int]]:
    """Runs of set bytes as inclusive ``(start, end)`` pairs."""
    out: List[Tuple[int, int]] = []
    start = -1
    for frame, hit in enumerate(flags):
        if hit and start < 0:
            start = frame
        elif not hit and start >= 0:
            out.append((start, frame - 1))
            start = -1
    if start >= 0:
        out.append((start, len(flags) - 1))
    return out


def _tile(
    runs: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int, str]],
    total: int,
) -> List[Segment]:
    """Grow, join and complement the flagged runs into a tiling of the video.

    Three moves, in this order, and the order matters:

    1. **Grow** every run to :data:`MIN_SEGMENT`, centred on what was flagged.
       A one-frame position jump is a place, and it is one the human cannot
       judge from one frame.
    2. **Join** runs separated by less than :data:`MIN_SEGMENT` clean frames.
       Growing creates most of these, and the alternative is two stops a second
       apart.
    3. **Complement**: everything left over is clean, and a leftover shorter
       than :data:`MIN_SEGMENT` at either end of the video is given to the
       flagged span beside it rather than left as a stop of its own.

    Nothing here can delete a flagged frame, so the tiling covers
    ``[0, total)`` exactly and every originally-flagged frame is inside a
    flagged segment.
    """
    grown: List[Tuple[int, int]] = []
    for a, b in runs:
        need = MIN_SEGMENT - (b - a + 1)
        if need > 0:
            a, b = a - need // 2, b + (need - need // 2)
            if a < 0:
                b, a = b - a, 0
            if b > total - 1:
                a, b = max(0, a - (b - (total - 1))), total - 1
        grown.append((a, b))

    joined: List[List[int]] = []
    for a, b in sorted(grown):
        if joined and a <= joined[-1][1] + MIN_SEGMENT:
            joined[-1][1] = max(joined[-1][1], b)
        else:
            joined.append([a, b])

    if joined:
        if joined[0][0] < MIN_SEGMENT:
            joined[0][0] = 0
        if total - 1 - joined[-1][1] < MIN_SEGMENT:
            joined[-1][1] = total - 1

    out: List[Segment] = []
    cursor = 0
    for a, b in joined:
        if a > cursor:
            out.append(Segment(cursor, a - 1, "clean", ""))
        out.append(Segment(a, b, "flagged", _reasons_for(spans, a, b)))
        cursor = b + 1
    if cursor < total:
        out.append(Segment(cursor, total - 1, "clean", ""))
    return out


def spot_check_frames(segments: Sequence[Segment], n: int = SPOT_CHECKS) -> Tuple[int, ...]:
    """``n`` evenly spaced frames drawn only from the **clean** spans.

    This is the one part of the final pass the model does not choose, and it
    exists because of the failure every other part of this module is blind to.

    Everything else here starts from the model's own doubt: a low score, a
    broken track, a count that changed, a box that grew. All of those require
    the model to have noticed something. A model that is *stably and
    confidently wrong* - one that has decided the second animal in the arena is
    background and never once detects it - produces none of them. Its output is
    perfectly self-consistent: the same one box, the same track id, high
    confidence, for all 108,000 frames. It generates **zero** flagged spans, so
    a review built only from flags takes the human to zero stops, and they sign
    off a video that is missing half its data in under a minute.

    Sampling the clean spans is what makes that detectable. With 40 spot checks
    spread through the video, an animal missing from every frame is missed by
    the checks and caught on the first stop - and unlike the flagged stops,
    their count does not fall as the model gets better, which is exactly the
    point: the better the model looks, the more of the review is spot checks.

    Evenly spaced across clean *frames*, not across clean *segments*: spacing
    by segment would spend the same number of looks on a 30-frame span as on a
    40,000-frame one, and it is the long untroubled stretches that a confident
    error hides in.
    """
    clean = [s for s in segments if not s.flagged]
    total = sum(s.span for s in clean)
    n = int(n)
    if total <= 0 or n <= 0:
        return ()

    picks: List[int] = []
    step = total / float(n)
    index = 0
    consumed = 0
    for i in range(n):
        # Midpoint of the i-th slice, so the first and last picks sit inside
        # the video rather than on its first and last clean frame.
        target = int((i + 0.5) * step)
        target = min(target, total - 1)
        while index < len(clean) - 1 and consumed + clean[index].span <= target:
            consumed += clean[index].span
            index += 1
        picks.append(clean[index].start + (target - consumed))
    return tuple(sorted(set(picks)))


def stops(
    segments: Sequence[Segment],
    spots: Iterable[int] = (),
    *,
    budget: int = STOP_BUDGET,
) -> Tuple[int, ...]:
    """Every frame the human is taken to, ascending, at most ``budget`` of them.

    The **start** of each flagged segment rather than its middle: a flagged
    span is flagged because something changed, and the change is at the front.
    Arriving in the middle of a 400-frame track fragment shows the human the
    consequence with the cause already off screen (the same reason
    :meth:`~htrtdetr.annotation.quality.ReviewItem.context` plays the frames
    either side).

    **The budget is not a nicety.** Sign-off is gated on visiting every stop,
    so an unbounded list is an unfinishable video. Measured on the shape this
    project actually produces: an hour in which one of two flies is briefly
    lost every couple of seconds - routine when they climb over each other -
    yields ~1,800 flagged segments. At ten seconds a stop that is five hours
    for one video of twenty, spent answering the same question about the same
    two flies eighteen hundred times.

    What survives the cut, in order:

    1. **Every spot check.** They are the only thing in this module that can
       catch a model which is *stably and confidently wrong*, and a model that
       flickers a lot is not evidence that it never misses anything.
    2. **The longest flagged spans.** A 400-frame fragment is a real loss of
       identity; a two-frame blip in the same place forty times over is one
       fact about the detector, and seeing it once is seeing it.

    What is dropped is dropped **out loud**: :func:`stop_budget_note` says how
    many and why, and the caller puts it on screen. A cap nobody is told about
    reads as "there was nothing else to check".
    """
    spot_frames = {int(f) for f in spots}
    flagged = sorted(
        (s for s in segments if s.flagged),
        key=lambda seg: (-(seg.end - seg.start + 1), seg.start),
    )
    room = max(0, int(budget) - len(spot_frames))
    out = set(spot_frames)
    out.update(int(seg.start) for seg in flagged[:room])
    return tuple(sorted(out))


def stop_budget_note(
    segments: Sequence[Segment],
    spots: Iterable[int] = (),
    *,
    budget: int = STOP_BUDGET,
) -> str:
    """What the budget left out, or ``""`` when it left nothing out.

    The repo's rule about bounded coverage: a workflow that caps what it shows
    has to say what it dropped, because silent truncation reads as having
    covered everything.
    """
    flagged = [seg for seg in segments if seg.flagged]
    room = max(0, int(budget) - len({int(f) for f in spots}))
    dropped = max(0, len(flagged) - room)
    if not dropped:
        return ""
    return (
        f"{dropped:,} shorter flagged places are not in this list. The model "
        "loses and refinds an animal constantly in this video; they are still "
        "marked on the timeline."
    )


# ══════════════════════════════════════════════════════════════════════════
# The state: what has been scanned, what has been signed off
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PassState:
    """Where the final pass has got to on one video.

    Stored in :attr:`htrtdetr.annotation.session.VideoSession.final_pass` and
    written into the project manifest by ``project._video_row``, which is why
    it is four small fields and no boxes: the manifest is rewritten whole on
    every autosave, and a state that grew with the video would turn a keystroke
    in a twenty-video project into a second of JSON.

    Attributes:
        weights: The checkpoint that scanned this video, as a
            :func:`~htrtdetr.annotation.cache.video_fingerprint`. Kept so a
            re-trained model invalidates the pass instead of silently leaving
            half a video scanned by one model and half by another.
        scanned: Frames written to the ledger. ``0`` means the pass has not
            run - it is the flag *and* the progress figure, because two fields
            that can disagree about whether a pass ran will eventually
            disagree.
        verified: Merged, sorted, **inclusive** spans the human has signed off.
        signed_off: Unix time the whole video was signed off, or ``None``.
    """

    weights: str = ""
    scanned: int = 0
    verified: Tuple[Tuple[int, int], ...] = ()
    signed_off: Optional[float] = None

    def __post_init__(self) -> None:
        """Normalise ``verified`` here, because here is the only place that can.

        :func:`covers` walks the spans in order and stops at the first one that
        starts past the frame it was asked about - which is a correct and fast
        thing to do over sorted, merged spans and a wrong one over any other
        kind. :func:`from_dict` and :func:`mark_verified` both normalise, so
        the invariant held for every path the code takes today and for none of
        the paths a test, a future caller or a hand-edited manifest takes:
        ``PassState(verified=((500, 599), (0, 99)))`` makes ``covers(state,
        50)`` return ``False`` and ``can_sign_off`` refuse forever, counting
        wrong while it does. A frozen dataclass has exactly one place to make
        an invariant true for every construction, and this is it.
        """
        object.__setattr__(self, "verified", _merge_spans(self.verified))

    @property
    def ran(self) -> bool:
        return self.scanned > 0

    @property
    def verified_frames(self) -> int:
        return sum(b - a + 1 for a, b in self.verified)

    @classmethod
    def from_dict(cls, data: Any) -> "PassState":
        """Read the manifest's version of this, tolerating anything.

        Nothing here raises on a malformed field. This dict has been through a
        JSON round trip, a project file edited by hand, and possibly a version
        of the annotator that did not have this feature; a manifest that will
        not open is a worse outcome than a pass that has to be re-run.
        """
        if not isinstance(data, dict):
            return cls()
        try:
            scanned = max(0, int(data.get("scanned") or 0))
        except (TypeError, ValueError):
            scanned = 0
        spans: List[Tuple[int, int]] = []
        for raw in data.get("verified") or ():
            try:
                a, b = int(raw[0]), int(raw[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            if b >= a:
                spans.append((a, b))
        when = data.get("signed_off")
        try:
            signed = float(when) if when is not None else None
        except (TypeError, ValueError):
            signed = None
        return cls(
            weights=str(data.get("weights") or ""),
            scanned=scanned,
            verified=_merge_spans(spans),
            signed_off=signed,
        )

    def to_dict(self) -> Dict[str, Any]:
        """The manifest form. Lists, not tuples: this goes through ``json``."""
        return {
            "weights": self.weights,
            "scanned": int(self.scanned),
            "verified": [[int(a), int(b)] for a, b in self.verified],
            "signed_off": self.signed_off,
        }


def _merge_spans(spans: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Sorted, merged, non-overlapping inclusive spans.

    Touching spans join: ``(0, 99)`` and ``(100, 199)`` become ``(0, 199)``,
    because with an inclusive end they describe one unbroken stretch and
    leaving them apart would make ``covers`` right but every count of "how many
    places" wrong.
    """
    ordered = sorted((int(a), int(b)) for a, b in spans if int(b) >= int(a))
    out: List[Tuple[int, int]] = []
    for a, b in ordered:
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return tuple(out)


def mark_verified(state: PassState, start: int, end: int) -> PassState:
    """A copy with ``start..end`` (inclusive) added to the verified spans.

    Additive and idempotent: re-verifying a span the human already signed off
    changes nothing, and verifying them out of order is fine. A review that is
    eight hours long across twenty sittings must not depend on the order the
    stops were visited in.
    """
    start, end = int(start), int(end)
    if end < start:
        start, end = end, start
    return replace(state, verified=_merge_spans([*state.verified, (start, end)]))


def covers(state: PassState, frame: int) -> bool:
    """Has the human signed off the span containing ``frame``?"""
    frame = int(frame)
    for a, b in state.verified:
        if a <= frame <= b:
            return True
        if a > frame:
            break
    return False


def visited(state: PassState, stops: Sequence[int]) -> int:
    """How many of ``stops`` fall inside a verified span."""
    return sum(1 for frame in stops if covers(state, frame))


def can_sign_off(state: PassState, stops: Sequence[int]) -> Tuple[bool, str]:
    """``(ready, why not)`` for signing off the whole video.

    The refusal is always **countable** - "37 of 134 places still to check" -
    rather than "not ready yet". This is the last gate before a dataset the
    study is built on, and a user who is told no without being told how much
    more there is cannot plan the afternoon and will look for the override.
    """
    if not state.ran:
        return False, "The final pass has not run on this video yet."
    total = len(stops)
    if total == 0:
        return True, ""
    done = visited(state, stops)
    left = total - done
    if left > 0:
        return False, f"{left} of {total} places still to check."
    return True, ""


def sign_off(state: PassState, when: float) -> PassState:
    """Record that the human signed this video off at ``when`` (unix time).

    Deliberately does not re-check :func:`can_sign_off`. The gate belongs to
    the dialog, where the user can be told what is unfinished and decide; a
    second refusal down here could only raise, in a callback, after the
    confirmation - and a human who has looked at the video and says it is right
    is the authority this whole module is built to serve.
    """
    return replace(state, signed_off=float(when))


def unsign(state: PassState) -> PassState:
    """Reopen a signed-off video, **keeping** the verified spans.

    Keeping them is the point: re-opening usually means one thing was noticed
    after the fact, not that the other 133 stops were imaginary. Clearing them
    would price a second look at a fresh 25 minutes and mean nobody takes one.
    """
    return replace(state, signed_off=None)


def _duration(frames: int, fps: float) -> str:
    if fps <= 0 or frames <= 0:
        return f"{max(0, int(frames)):,} frames"
    minutes = frames / fps / 60.0
    if minutes < 1.0:
        return f"{frames / fps:.0f} seconds"
    if minutes < 90.0:
        return f"{minutes:.0f} minutes"
    return f"{minutes / 60.0:.1f} hours"


def progress_sentence(
    state: PassState,
    stops: Sequence[int],
    *,
    num_frames: int,
    fps: float,
) -> str:
    """One sentence for the Review dock: how much of this video is done.

    In *places*, not frames or percent. The human's unit of work here is a
    stop - a seek, a look, a decision - so "94 of 134 places checked" is a
    number they can divide by their own pace to get a time, and "87% verified"
    is not: the last 13% might be two stops or two hundred.
    """
    total = len(stops)
    if not state.ran:
        return (
            "The final pass has not run on this video yet "
            f"({max(0, int(num_frames)):,} frames, {_duration(int(num_frames), fps)})."
        )
    if state.signed_off is not None:
        return (
            f"Signed off: {total} place{'s' if total != 1 else ''} checked across "
            f"{max(0, int(num_frames)):,} frames ({_duration(int(num_frames), fps)})."
        )
    if total == 0:
        return "Nothing was flagged in this video. Ready to sign off."
    done = visited(state, stops)
    if done >= total:
        return (
            f"All {total} places checked ({_duration(state.verified_frames, fps)} "
            "of video). Ready to sign off."
        )
    return f"{done} of {total} places checked in this video."
