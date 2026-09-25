"""release.py — the last step: turning a signed-off final pass into the dataset.

Ten rounds of the active-learning loop produced a detector. The final pass
(:mod:`htrtdetr.annotation.finalpass`) ran it over all ~2.16M frames of the
twenty one-hour videos and wrote what it found into a per-video ledger. A human
then walked the places the quality checks flagged plus a spread of spot checks,
and signed each video off. **This module is what is left**: read the human's own
labels and the signed-off ledger, and emit the dataset the study set out to
make.

Two decisions were made before this file existed, and both of them are why it
looks the way it does.

**Labels only, plus a path to the source video.**
    There is no image tree and there will not be one. 2.16M JPEGs at ~40 KB is
    about 86 GB and roughly 20 hours of decoding, to reproduce frames that are
    already sitting in the twenty .mp4 files the project points at. So every
    frame row here carries ``frame_index`` and an **empty** ``image_path``, and
    the video row carries ``video_path``. The empty string is deliberate and is
    not an oversight: :func:`htrtdetr.data.annotation.load_annotation` fills a
    missing ``image_path`` in with ``images/<video_id>/<frame>.jpg``, and that
    path names a JPEG nobody ever wrote. A consumer that silently reads black
    frames for a whole dataset is a worse failure than one that cannot find an
    image and says so.

**The human did not look at every frame, and the file must not pretend they
did.**
    What they looked at is ``verified``: merged inclusive spans on the per-video
    state. Every object in the release therefore carries
    ``provenance["human_review"]`` saying which of **four** things it is -
    ``"drawn"`` (the human drew this box), ``"accepted"`` (the human took a
    machine proposal into their annotations), ``"signed_off"`` (the model drew
    it inside a stretch the human was taken to and judged) and ``"sampled"``
    (the model drew it in a video the human signed off, in a stretch they were
    never shown). The last of those is most of the file, and it has its own
    word for exactly that reason: four different grades of evidence, and a
    dataset that flattens them into "ground truth" cannot be audited later.

The one rule underneath everything
----------------------------------
**A frame the human drew on wins whole**, never box by box. See
:func:`merge_frame`; it is the same rule as
``auto_label.MergePolicy.skip_frames_with_accepted``, for the same reason, and
it is the difference between a dataset and a dataset with the model's misses
quietly re-inserted into the frames a human had just finished fixing.

Purity: no dearpygui, no torch, no cv2, no numpy - not even lazily.
:mod:`finalpass` is not imported either: this module takes the ledger rows and
the pass state as plain arguments, so every decision in it can be tested without
a ledger on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

__all__ = [
    "HUMAN_DRAWN",
    "HUMAN_ACCEPTED",
    "MODEL_SIGNED_OFF",
    "MODEL_SAMPLED",
    "NOT_SCANNED",
    "NOT_SIGNED_OFF",
    "NO_VIDEO_PATH",
    "BYTES_PER_BOX",
    "BYTES_PER_FRAME",
    "BOXES_PER_FRAME",
    "merge_frame",
    "VideoPlan",
    "release_plan",
    "dataset_dict",
    "write_release",
    "summary_sentence",
    "size_estimate",
]


# ── how a box got into the release ───────────────────────────────────────
#
# Four values, not a boolean. "Was this checked by a human?" has four honest
# answers here and collapsing them loses the one that matters most, because it
# is the one most of the file carries: a box the human never individually saw,
# in a video they signed off after sampling ~150 places in it. That box is good
# evidence and it is not the same evidence as a drawn one, and a year from now
# the only place that distinction can still be recovered is this field.

#: The human drew this box themselves.
HUMAN_DRAWN = "drawn"

#: The human took a machine proposal into their own annotations - they looked at
#: this box and kept it. ``source`` on the object says which machine made it.
HUMAN_ACCEPTED = "accepted"

#: The model drew it, inside a stretch the human was taken to and judged. They
#: did not necessarily look at this exact frame - a flagged span is checked as a
#: span - but they looked at its neighbourhood.
MODEL_SIGNED_OFF = "signed_off"

#: The model drew it, in a video the human signed off, in a stretch they were
#: never taken to. This is the honest grade for most of a release and the reason
#: it exists as a separate word: the review model is ~134 stops in 108,000
#: frames, so calling the other 107,000 "signed_off" would flatten a sampled
#: guarantee into a per-frame one, and nobody auditing the dataset later could
#: tell the two apart. A consumer that wants only directly-checked boxes can
#: filter on this field; one that wants all of them can ignore it.
MODEL_SAMPLED = "sampled"


# ── why a video did not make it into the release ─────────────────────────
#
# Each reason is phrased to fit the frame "N videos <reason> and were left out."
# so that :func:`summary_sentence` can group them without a table of verbs.

#: The final pass has not been run over this video at all (``scanned == 0``).
NOT_SCANNED = "not scanned by the final pass"

#: Scanned, but the human has not signed it off. This is the common one, and it
#: is the whole reason :func:`release_plan` returns names rather than a count.
NOT_SIGNED_OFF = "not signed off"

#: Signed off, but the project has no path to the source video. In a
#: labels-only dataset that is fatal: the labels point at nothing.
NO_VIDEO_PATH = "missing a path to the source video"


# ── what the file costs ──────────────────────────────────────────────────

#: Bytes one object costs in the saved JSON.
#:
#: **Measured, not counted by hand.** The first version of this constant was
#: 60, arrived at by adding up the field names, and the file it described came
#: out 6.7 times larger than the number beside the Export button - "about 330
#: MB" for a release that wrote 2.2 GB. Two things were missing from the
#: arithmetic: ``provenance`` is a nested object with a ``human_review`` key in
#: it, and ``storage.atomic_json`` writes with ``indent=2``.
#:
#: The release is therefore written **compact**, by :func:`write_release`
#: rather than through ``atomic_json``: it is a machine artefact of over a
#: gigabyte and nobody reads it by hand, and the indentation was 37% of it.
#: Measured on the real record, compact: 410 bytes for a two-box frame row,
#: i.e. 205 per box with the frame's own wrapper already inside it.
BYTES_PER_BOX = 205

#: Bytes of wrapper per frame row beyond its boxes: ``frame_index``, the empty
#: ``image_path``, ``width``, ``height`` and the ``objects`` list. Small next to
#: the boxes, and kept separate so a project with one animal per frame is
#: quoted correctly rather than at half of a two-animal project.
BYTES_PER_FRAME = 40

#: Objects per frame, **measured on this project's own material**: an hour of
#: video at 30 fps with two animals in it is about 216,000 boxes, which is two
#: per frame. A project that films ten animals must pass its own figure.
BOXES_PER_FRAME = 2.0


# ══════════════════════════════════════════════════════════════════════════
# The merge rule
# ══════════════════════════════════════════════════════════════════════════

def merge_frame(
    human_objects: Optional[Sequence[Any]],
    model_boxes: Optional[Sequence[Sequence[float]]],
    *,
    looked_at: bool = True,
) -> List[Dict[str, Any]]:
    """The objects for one frame: **a frame the human drew on wins whole.**

    This is ``auto_label.MergePolicy.skip_frames_with_accepted`` again, moved to
    the export side and hardened from a policy flag into the only behaviour.
    Quoting that module's rule 1: *"A frame holding any accepted human box is
    skipped whole - human work is never overwritten."* The merge there works
    per frame rather than per box, and so does this one, for the reason that
    makes per-box merging so tempting and so wrong: a human who has finished a
    frame has decided about the **whole** frame, including the places they left
    empty. Adding the model's boxes back into the gaps re-inserts precisely the
    detections that human just rejected, with no record that they did.

    ``human_objects is None`` means the human never opened this frame, so the
    model's boxes stand. ``human_objects == []`` is **not** the same thing: it
    is a human saying "there is nothing here", it is a real label, and it beats
    a model that thinks otherwise. Treating the two as one value is the single
    easiest way to lose every negative example in the dataset, because the
    frames a human cleared are exactly the frames the model had something in.

    ``looked_at`` only grades the model's boxes: ``True`` when this frame is
    inside a stretch the human was taken to and judged, ``False`` when it is in
    a signed-off video they were never shown. It changes no box and drops none;
    it decides between :data:`MODEL_SIGNED_OFF` and :data:`MODEL_SAMPLED`, so
    that the difference between "a human looked at this" and "a human vouched
    for the video this is in" survives into the file.

    Returns new dicts every time. Nothing here mutates or aliases the project's
    live :class:`~htrtdetr.annotation.objects.AnnObject` instances or their
    ``provenance`` dicts - the export runs against the session the user is still
    editing.
    """
    if human_objects is not None:
        return [_human_record(obj, index) for index, obj in enumerate(human_objects, start=1)]
    return [
        _model_record(box, index, looked_at=looked_at)
        for index, box in enumerate(model_boxes or (), start=1)
    ]


def _human_record(obj: Any, object_id: int) -> Dict[str, Any]:
    """One human-owned object as a JSON dict, keeping everything it carries.

    Serialised through the object's own ``to_dict`` when it has one, rather than
    by listing the fields here. There is exactly one canonical serializer for
    this format (``data.annotation._object_to_dict``, which ``AnnObject.to_dict``
    delegates to) and re-deriving the field list in the exporter is how ``obb``
    and ``keypoints`` end up silently missing from the deliverable while every
    test still passes - the release is the one artefact nobody diffs against a
    previous version.
    """
    record: Dict[str, Any]
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        record = dict(to_dict())
    else:
        # Duck-typed fallback, so a test (or a future object type) does not have
        # to implement the serializer to be exportable.
        record = {
            "bbox": [float(v) for v in getattr(obj, "bbox", [0, 0, 0, 0])],
            "class_id": int(getattr(obj, "class_id", 0) or 0),
            "track_id": int(getattr(obj, "track_id", -1)),
            "action_id": int(getattr(obj, "action_id", -1)),
        }
    record["object_id"] = object_id
    record["bbox"] = [float(v) for v in record.get("bbox", [0, 0, 0, 0])]

    # ``source`` is where the box came FROM and is never rewritten - a box the
    # wand or the tracker made and the human kept still says so. What the human
    # did about it is ``human_review``, which is a different question.
    source = str(getattr(obj, "source", record.get("source", "human")) or "human")
    record["source"] = source
    record["score"] = getattr(obj, "score", record.get("score"))
    provenance = dict(record.get("provenance") or {})
    provenance["human_review"] = HUMAN_DRAWN if source == "human" else HUMAN_ACCEPTED
    record["provenance"] = provenance
    return record


def _model_record(
    box: Sequence[float], object_id: int, *, looked_at: bool = True
) -> Dict[str, Any]:
    """One ledger box as a JSON dict.

    A ledger box is ``[x1, y1, x2, y2, class_id, score, track_id]``. Shorter
    rows are accepted and defaulted rather than raised on: the ledger is written
    by a two-hour pass over 2.16M frames and losing the whole release to one
    truncated line at the end of it would be the wrong trade.
    """
    values = list(box)
    coords = [float(v) for v in values[:4]] + [0.0] * max(0, 4 - len(values))
    return {
        "object_id": object_id,
        "bbox": coords[:4],
        "class_id": int(values[4]) if len(values) > 4 else 0,
        "track_id": int(values[6]) if len(values) > 6 else -1,
        # The final pass detects; it does not classify behaviour. -1 is
        # "unannotated", which is true, rather than 0, which would silently
        # label two million frames with whatever action_names[0] happens to be.
        "action_id": -1,
        "source": "model",
        "score": float(values[5]) if len(values) > 5 else None,
        "provenance": {
            "human_review": MODEL_SIGNED_OFF if looked_at else MODEL_SAMPLED
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# What goes in, and what is named for not going in
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VideoPlan:
    """One video's contribution to the release, decided before anything is read.

    ``frames`` is the number of frames the release **covers** - the whole
    signed-off video - and not the number of frame rows that end up in the file.
    Those differ, and the covered figure is the honest one: a frame the model
    found nothing in is absent from the ledger and therefore absent from the
    file, but it is still a frame of video inside a recording a human signed
    off, and it is still part of what the dataset claims to describe. Counting
    rows instead would quietly shrink the headline number every time the
    detector got better at empty frames.

    ``human + model == frames`` by construction: every covered frame is owned by
    one of the two.
    """

    video_id: str
    video_path: str
    num_frames: int
    fps: float
    #: Frames of video the release covers (see above).
    frames: int
    #: Of those, the frames whose labels are the human's.
    human: int
    #: Of those, the frames whose labels come from the signed-off ledger.
    model: int
    #: The spans the human was actually taken to, merged and inclusive. They do
    #: not decide what is exported - a signed-off video ships whole - they
    #: decide what each box is *called*: a model box inside one of these is
    #: :data:`MODEL_SIGNED_OFF`, outside it is :data:`MODEL_SAMPLED`. Dropping
    #: the distinction would flatten a sampled guarantee into a per-frame one,
    #: and nobody auditing the file later could tell a frame a human opened
    #: from the 107,000 they did not.
    verified: Tuple[Tuple[int, int], ...] = ()
    #: Frame size, carried because the annotation format's video row has it and
    #: ``validate_annotation`` warns when it is missing. Boxes are in pixels, so
    #: a consumer that wants normalised coordinates needs this.
    width: int = 0
    height: int = 0
    #: The checkpoint fingerprint that drew this video's boxes, off the
    #: per-video pass state. It reaches the file's ``meta`` so the dataset can
    #: name the model that produced most of it.
    weights: str = ""


def release_plan(
    videos: Iterable[Any],
    states: Optional[Mapping[str, Any]] = None,
    *,
    annotations_for: Optional[Callable[[str], Mapping[int, Sequence[Any]]]] = None,
) -> Tuple[List[VideoPlan], List[Tuple[str, str]]]:
    """``(plans, excluded)`` - what the release contains, and what it does not.

    The second half of the return value is the whole point of this signature.
    The natural shape for this function is "return the videos that are ready",
    and that shape has one silent failure in it that costs a study: a video that
    was never signed off - because the human ran out of afternoon on video 18 -
    drops out of the export without a word, and the dataset ships with 17 videos
    in it under a name that says 20. Every exclusion here comes back with the
    video's id and a reason, so the caller can put it on screen and
    :func:`summary_sentence` can put it in one line.

    ``videos`` may be :class:`~htrtdetr.annotation.session.VideoSession` objects
    or plain dicts; ``states`` maps ``video_id`` to the per-video final-pass
    state (``weights``/``scanned``/``verified``/``signed_off``), and when a video
    is missing from it the video's own ``final_pass`` is used. Nothing is read
    from disk and nothing is imported from :mod:`finalpass`, so the whole
    decision is testable from four dicts.
    """
    plans: List[VideoPlan] = []
    excluded: List[Tuple[str, str]] = []
    lookup = dict(states or {})

    for video in videos:
        video_id = str(_field(video, "video_id", "") or "")
        state = lookup.get(video_id)
        if state is None:
            state = _field(video, "final_pass", None)
        scanned = int(_field(state, "scanned", 0) or 0)
        signed_off = _field(state, "signed_off", None)
        spans = _merge_spans(_field(state, "verified", ()) or ())

        if not scanned:
            excluded.append((video_id, NOT_SCANNED))
            continue
        if signed_off is None or not spans:
            # A sign-off timestamp with no spans behind it is the same failure
            # as no sign-off: nothing was actually checked.
            excluded.append((video_id, NOT_SIGNED_OFF))
            continue
        path = _video_path(video)
        if not path:
            excluded.append((video_id, NO_VIDEO_PATH))
            continue

        num_frames = int(_field(video, "num_frames", 0) or 0)
        spans = _clip_spans(spans, num_frames)
        # **A signed-off video ships whole.** The spans are evidence of what
        # the human looked at, not the scope of the export: the review model
        # is "check the flagged places and a spread of spot checks, then say
        # this video is good", and ``finalpass.can_sign_off`` already refuses
        # the sign-off until every one of those places has been visited. If
        # the export were the spans instead, a video the human declared good
        # would contribute the few hundred frames around their clicks and the
        # other 107,000 would be silently missing - which is the shape of
        # failure this function's second return value exists to prevent, one
        # level down.
        covered = num_frames or (max((end for _s, end in spans), default=-1) + 1)
        human = (
            annotations_for(video_id)
            if annotations_for is not None
            else _field(video, "annotations", None)
        )
        # Bounded, like the exported rows are. A human frame index past the
        # video's length is a frame the writer will drop, so counting it here
        # would quote a total the file does not contain - and, because
        # ``model`` is the remainder, would make ``model`` negative.
        human_frames = sum(
            1
            for frame in (human or {})
            if 0 <= int(frame) and (num_frames <= 0 or int(frame) < num_frames)
        )

        plans.append(
            VideoPlan(
                video_id=video_id,
                video_path=path,
                num_frames=num_frames,
                fps=float(_field(video, "fps", 0.0) or 0.0),
                frames=covered,
                human=human_frames,
                model=covered - human_frames,
                verified=spans,
                width=int(_field(video, "width", 0) or 0),
                height=int(_field(video, "height", 0) or 0),
                weights=str(_field(state, "weights", "") or ""),
            )
        )
    return plans, excluded


# ══════════════════════════════════════════════════════════════════════════
# The dataset
# ══════════════════════════════════════════════════════════════════════════

def dataset_dict(
    plans: Sequence[VideoPlan],
    rows_for: Callable[[str], Iterable[Tuple[int, Sequence[Sequence[float]]]]],
    annotations_for: Callable[[str], Mapping[int, Sequence[Any]]],
    *,
    class_names: Sequence[str],
    action_names: Sequence[str],
    description: str = "",
) -> Dict[str, Any]:
    """The release, in this repo's own annotation format, as a plain dict.

    Shaped so that :func:`htrtdetr.data.annotation.load_annotation` reads it back
    without a converter - it is the format ``yoake train`` already consumes, and
    inventing a release-only format here would mean the deliverable of the study
    is the one file in the project nothing else can open.

    ``rows_for(video_id)`` yields ``(frame, boxes)`` from the ledger, ascending,
    for frames that have boxes; ``annotations_for(video_id)`` gives the human's
    ``{frame: [AnnObject]}``. Both are callables rather than data because the
    ledger for twenty videos is tens of millions of boxes and this function must
    never need two videos in memory at once.

    Three rules decide what becomes a frame row:

    1. Every frame of a signed-off video, not only the spans the human clicked
       through. Signing off is a statement about the video; the spans say which
       parts of it were looked at directly, and that difference is carried on
       each object rather than by dropping 99% of the frames.
    2. A human frame is always a row, **even when it is empty**. That row is
       the human saying "nothing here", which is a label.
    3. A ledger frame is a row only when it has boxes. Writing out the empty
       frames as well is what turns a 300 MB file into a 3 GB one, and adds
       nothing: a frame absent from an annotation file has no objects.
    """
    videos: List[Dict[str, Any]] = []
    for plan in plans:
        human = dict(annotations_for(plan.video_id) or {})
        frames: List[Dict[str, Any]] = []
        for frame, objects in sorted(human.items()):
            index = int(frame)
            frames.append(_frame_row(index, merge_frame(objects, None), plan))
        for frame, boxes in rows_for(plan.video_id) or ():
            index = int(frame)
            if index in human or not boxes:
                continue
            frames.append(
                _frame_row(
                    index,
                    merge_frame(None, boxes, looked_at=_covers(plan.verified, index)),
                    plan,
                )
            )
        frames.sort(key=lambda row: row["frame_index"])
        videos.append(
            {
                "video_id": plan.video_id,
                # The locator for every frame in this row. With no image tree,
                # this path and ``frame_index`` are the only way back to a pixel.
                "video_path": plan.video_path,
                "fps": plan.fps,
                "width": plan.width,
                "height": plan.height,
                "num_frames": plan.num_frames,
                "frames": frames,
            }
        )

    return {
        "meta": _meta(plans, description),
        "class_names": list(class_names),
        "action_names": list(action_names),
        "videos": videos,
    }


def _meta(plans: Sequence[VideoPlan], description: str = "") -> Dict[str, Any]:
    """The header both writers share, so they cannot describe the file differently."""
    # Which checkpoint drew the boxes. Most of this file is model output, and
    # a dataset that cannot say which model produced it cannot be reproduced,
    # re-scored against a better one, or withdrawn if that model turns out to
    # have a defect. One fingerprint per video, because a project may have
    # scanned its videos across a retrain.
    weights = sorted({plan.weights for plan in plans if plan.weights})
    return {
        "weights": weights[0] if len(weights) == 1 else weights,
        "version": "1.1",
        "description": description or "YOAKE release: final pass, signed off",
        "created": str(date.today()),
        "fps_default": plans[0].fps if plans else 0.0,
        # Empty, and load_annotation preserves an explicit empty string.
        # There is no image tree; see this module's docstring.
        "image_root": "",
        "frames_from": "video",
        "note": (
            "Labels only. Each frame is located by the video's video_path and "
            "the frame's frame_index; no images were exported. Every object "
            "carries provenance.human_review, which is one of four grades: "
            "'drawn' (a human drew this box), 'accepted' (a human took a "
            "machine proposal into their labels), 'signed_off' (the model drew "
            "it inside a stretch the human was taken to and judged), and "
            "'sampled' (the model drew it in a video the human signed off, in "
            "a stretch they were never shown). The review was by sampling: "
            "about 150 places per video, not every frame."
        ),
    }


def write_release(
    path: Any,
    plans: Sequence[VideoPlan],
    rows_for: Callable[[str], Iterable[Tuple[int, Sequence[Sequence[float]]]]],
    annotations_for: Callable[[str], Mapping[int, Sequence[Any]]],
    *,
    class_names: Sequence[str],
    action_names: Sequence[str],
    description: str = "",
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Write the release to ``path``, one video at a time. Returns what it wrote.

    :func:`dataset_dict` builds the whole release as a Python object, which is
    correct for a test and wrong for the real thing: twenty hour-long videos is
    about 4.3M boxes, and the dict, the JSON string it is serialised into and
    the file all exist at once - measured at over two gigabytes of peak memory
    for a file of about 1.4 GB. This function never holds more than one video.

    **Compact, not indented.** ``storage.atomic_json`` writes ``indent=2``,
    which is right for a manifest a human opens and wrong for this: the
    indentation was 37% of a 2.2 GB file that nothing reads by hand.

    **Frames outside the video are dropped, and counted.** A ledger row at
    frame 999,999 of a 108,000-frame video is a bug somewhere upstream, but a
    dataset row nothing can seek to is a bug that travels. They come back in
    the return value rather than being silently skipped.

    Written to a temporary name and renamed, so an interrupted export leaves
    the previous release intact rather than a half-file with the right name.

    Returns ``{"frames", "boxes", "dropped", "bytes", "videos": {id: rows}}``.
    ``videos`` is what lets the caller notice a video whose ledger has gone
    missing: :func:`~htrtdetr.annotation.finalpass.iter_rows` yields nothing
    for a file that is not there, so without a count a moved ledger exports as
    a video that says 108,000 frames and contains none.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_name(out.name + ".part")
    totals = {"frames": 0, "boxes": 0, "dropped": 0, "bytes": 0, "videos": {}}
    meta = _meta(plans, description)

    with open(temp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write('{"meta": ')
        json.dump(meta, fh, ensure_ascii=False)
        fh.write(', "class_names": ')
        json.dump(list(class_names), fh, ensure_ascii=False)
        fh.write(', "action_names": ')
        json.dump(list(action_names), fh, ensure_ascii=False)
        fh.write(', "videos": [')
        for position, plan in enumerate(plans):
            if position:
                fh.write(", ")
            if progress is not None:
                progress(position, len(plans), plan.video_id)
            counts = _write_video(fh, plan, rows_for, annotations_for)
            totals["frames"] += counts["frames"]
            totals["boxes"] += counts["boxes"]
            totals["dropped"] += counts["dropped"]
            totals["videos"][plan.video_id] = counts["frames"]
        fh.write("]}")
    totals["bytes"] = temp.stat().st_size
    temp.replace(out)
    return totals


def _write_video(fh, plan, rows_for, annotations_for) -> Dict[str, int]:
    """One video object, streamed. The frames are sorted; the ledger is already.

    The human's frames are held (there are at most a few thousand of them, and
    they are already in memory in the session this runs beside); the ledger's
    are streamed straight through. Merging the two ascending streams is what
    keeps ``frame_index`` sorted without sorting a 108,000-element list.
    """
    human = dict(annotations_for(plan.video_id) or {})
    limit = int(plan.num_frames or 0)
    dropped = 0
    frames = boxes = 0

    def in_range(index: int) -> bool:
        return index >= 0 and (limit <= 0 or index < limit)

    fh.write('{"video_id": ')
    json.dump(plan.video_id, fh, ensure_ascii=False)
    fh.write(', "video_path": ')
    json.dump(plan.video_path, fh, ensure_ascii=False)
    fh.write(
        f', "fps": {plan.fps}, "width": {plan.width}, "height": {plan.height}, '
        f'"num_frames": {plan.num_frames}, "frames": ['
    )

    pending = sorted(int(f) for f in human)
    cursor = 0
    first = True
    written_to = -1

    def emit(index: int, objects: List[Dict[str, Any]]) -> None:
        nonlocal first, frames, boxes
        if not first:
            fh.write(", ")
        first = False
        json.dump(_frame_row(index, objects, plan), fh, ensure_ascii=False)
        frames += 1
        boxes += len(objects)

    for index, row in rows_for(plan.video_id) or ():
        index = int(index)
        while cursor < len(pending) and pending[cursor] < index:
            here = pending[cursor]
            cursor += 1
            if in_range(here):
                emit(here, merge_frame(human[here], None))
            else:
                dropped += 1
        if cursor < len(pending) and pending[cursor] == index:
            cursor += 1
            if in_range(index):
                emit(index, merge_frame(human[index], None))
            else:
                dropped += 1
            continue
        if not row:
            continue
        if not in_range(index):
            dropped += 1
            continue
        if index <= written_to:
            # Two rows for one frame would become two ``frame_index`` entries,
            # and ``validate_annotation`` rejects the **whole file** for that -
            # so one duplicated frame in one video of twenty costs the entire
            # release. ``append_rows`` cannot write one, but a ledger that was
            # concatenated by hand or copied from two runs can hold one.
            dropped += 1
            continue
        written_to = index
        emit(index, merge_frame(None, row, looked_at=_covers(plan.verified, index)))
    for here in pending[cursor:]:
        if in_range(here):
            emit(here, merge_frame(human[here], None))
        else:
            dropped += 1

    fh.write("]}")
    return {"frames": frames, "boxes": boxes, "dropped": dropped}


def _frame_row(index: int, objects: List[Dict[str, Any]], plan: VideoPlan) -> Dict[str, Any]:
    return {
        "frame_index": index,
        # Deliberately empty rather than omitted: see the module docstring.
        "image_path": "",
        "width": plan.width,
        "height": plan.height,
        "objects": objects,
    }


# ══════════════════════════════════════════════════════════════════════════
# What the user is told
# ══════════════════════════════════════════════════════════════════════════

def summary_sentence(
    plans: Sequence[VideoPlan], excluded: Sequence[Tuple[str, str]] = ()
) -> str:
    """One line: what the release holds, and what it leaves out.

    The second half is not optional and is not a warning the user can miss. The
    export that quietly ships 17 of 20 videos is the failure this whole module
    is arranged against, and the cheapest place to catch it is the sentence
    printed next to the button that writes the file.
    """
    frames = sum(plan.frames for plan in plans)
    if plans:
        lines = [
            f"{frames:,} frames from {len(plans)} "
            f"video{'s' if len(plans) != 1 else ''}."
        ]
    else:
        lines = ["No videos are ready to export."]

    grouped: Dict[str, int] = {}
    for _video_id, reason in excluded:
        grouped[reason] = grouped.get(reason, 0) + 1
    for reason, count in grouped.items():
        plural = count != 1
        lines.append(
            f"{count} video{'s' if plural else ''} "
            f"{'are' if plural else 'is'} {reason} and "
            f"{'were' if plural else 'was'} left out."
        )
    return " ".join(lines)


def size_estimate(
    plans: Sequence[VideoPlan], *, boxes_per_frame: float = BOXES_PER_FRAME
) -> str:
    """How big the annotation file will be, said honestly and in advance.

    Honestly means two things here. It means the arithmetic is visible -
    :data:`BYTES_PER_BOX` and :data:`BOXES_PER_FRAME` both carry where their
    number came from - and it means the sentence says it is an estimate rather
    than implying a measurement, because the true size depends on how many
    animals are in shot and nothing in this module has counted them.

    It is worth quoting at all because the number is large enough to matter and
    small enough to be a relief: the alternative the user declined was an image
    tree of about 86 GB, and hearing "a few hundred MB" is what makes the
    labels-only decision feel like the right one rather than a compromise.
    """
    frames = sum(plan.frames for plan in plans)
    boxes = int(round(frames * max(0.0, float(boxes_per_frame))))
    total = frames * BYTES_PER_FRAME + boxes * BYTES_PER_BOX
    return (
        f"about {_bytes(total)} of JSON - roughly {boxes:,} boxes at "
        f"{BYTES_PER_BOX} bytes each, plus a row per frame. No images are "
        f"written; the labels point back into the source videos."
    )


def _bytes(total: int) -> str:
    mb = total / (1024.0 * 1024.0)
    if mb < 1.0:
        return f"{total / 1024.0:.0f} KB"
    if mb < 1024.0:
        return f"{mb:.0f} MB"
    return f"{mb / 1024.0:.1f} GB"


# ══════════════════════════════════════════════════════════════════════════
# Spans and field access
# ══════════════════════════════════════════════════════════════════════════

def _field(holder: Any, name: str, default: Any = None) -> Any:
    """One field off a dataclass, an object or a dict.

    The callers of this module hold ``VideoSession`` objects; its tests hold
    dicts; the pass state is a dict in the manifest and a ``PassState`` in
    :mod:`finalpass`. Reading all three the same way is what lets this module
    avoid importing any of them.
    """
    if holder is None:
        return default
    if isinstance(holder, Mapping):
        value = holder.get(name, default)
    else:
        value = getattr(holder, name, default)
    return default if value is None else value


def _video_path(video: Any) -> str:
    """The path recorded for the source video, as a string, or ``""``.

    ``VideoSession`` keeps it on ``path`` as a ``Path``; the annotation format
    and the dict form call it ``video_path``. Not stat'd: a project that has
    been moved to another machine still has a correct relative path, and
    refusing to export because the drive is not mounted right now would be a
    check in the wrong place.
    """
    for name in ("video_path", "path"):
        value = _field(video, name, "")
        if value:
            return str(value)
    return ""


def _merge_spans(spans: Iterable[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    """Sorted, merged, inclusive ``(start, end)`` pairs.

    Merged even though the state is documented as already merged, because this
    is the cheap end of the trade: an unmerged or overlapping pair costs one
    sort here, and costs a duplicated ``frame_index`` in the release otherwise -
    which ``validate_annotation`` reports as an error and which
    ``load_annotation`` does not.

    Adjacent spans join (``end + 1 == next start``): they describe one
    continuous stretch of checked video and splitting them changes nothing
    except the number of comparisons.
    """
    rows: List[Tuple[int, int]] = []
    for span in spans or ():
        try:
            start, end = int(span[0]), int(span[1])
        except (IndexError, TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        rows.append((start, end))
    rows.sort()
    out: List[Tuple[int, int]] = []
    for start, end in rows:
        if out and start <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return tuple(out)


def _clip_spans(
    spans: Sequence[Tuple[int, int]], num_frames: int
) -> Tuple[Tuple[int, int], ...]:
    """Drop what falls outside ``[0, num_frames - 1]``.

    ``num_frames <= 0`` means the project does not know the length, so nothing
    is clipped - guessing zero there would empty the release. When it is known,
    clipping matters: a decoder that reports one more frame than the manifest
    recorded would otherwise put a frame index into the dataset that no consumer
    can seek to.
    """
    out: List[Tuple[int, int]] = []
    for start, end in spans:
        start = max(0, start)
        if num_frames > 0:
            end = min(end, num_frames - 1)
        if end >= start:
            out.append((start, end))
    return tuple(out)


def _covers(spans: Sequence[Tuple[int, int]], frame: int) -> bool:
    """Is this frame inside the signed-off spans? Ends are **inclusive**."""
    for start, end in spans:
        if start <= frame <= end:
            return True
        if frame < start:
            break
    return False
