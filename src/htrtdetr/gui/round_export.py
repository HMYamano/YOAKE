"""Turn the labels that exist right now into a dataset a trainer can read.

This is deliberately **not** ``WorkspaceMixin.export_project``. That one is the
*File > Save and Export Frames* command and it does three things a round must
not do:

1. It rewrites the live project. It sets ``meta["image_root"] = ""`` and
   overwrites every ``frame_metadata["image_path"]`` on the session, then saves
   that into the project's own ``annotations.json`` - so the user's label file
   ends up pointing at an ``images/`` folder that only exists inside the export.
   Acceptable once, when the user asked for an export; not acceptable every
   time they press Train.
2. It re-decodes every annotated frame every time. A round runs repeatedly over
   a growing label set, so the second round must pay only for what the first
   one did not already write.
3. It has no completion signal and no cancel. A round has to know when the
   frames are on disk, because the trainer reads them off disk.

The frames are decoded through :func:`video_source.read_frame_sequential`, the
same function the canvas uses to show them. That is the point: the box was
drawn on the frame that function returned, so the JPEG handed to the trainer
has to come from the same call. Decoding "more correctly" here than the
labelling path does would train the model on a frame the human never saw.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..annotation.round import FrameKey

__all__ = [
    "ExportInput",
    "ExportResult",
    "export_frames",
    "missing_images",
    "rewrite_image_paths",
    "drop_frames",
]


@dataclass(frozen=True)
class ExportInput:
    """One video's worth of work, detached from the session on the main thread."""

    video_id: str
    path: Optional[Path]
    frames: Tuple[int, ...] = ()


@dataclass
class ExportResult:
    written: int = 0
    reused: int = 0
    failed: List[str] = field(default_factory=list)
    #: Frames that could not be decoded, so the caller can drop them from the
    #: dataset instead of shipping an annotation whose image is missing - which
    #: the trainer would turn into a black frame without saying anything.
    missing: List[FrameKey] = field(default_factory=list)
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return not self.cancelled and not self.failed


def _image_name(video_id: str, index: int) -> str:
    return f"images/{video_id}/{index:06d}.jpg"


def export_frames(
    videos: Sequence[ExportInput],
    out: Path,
    *,
    cancel: Optional[threading.Event] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> ExportResult:
    """Write one JPEG per labelled frame under ``out/images/<video>/``.

    Runs on a worker thread: it touches no dearpygui and no session state, only
    the detached :class:`ExportInput` rows it was handed.

    Incremental by existence. A JPEG that is already there is kept, because the
    only thing that could have written it is a previous round of this same
    project at this same frame, and re-decoding an hour of video to produce a
    byte-identical file is the difference between a round that can be run every
    twenty labels and one that cannot.
    """
    import cv2

    from . import video_source

    def write_jpeg(path: Path, frame) -> bool:
        """``cv2.imwrite`` cannot write to a path with a non-ASCII character.

        It does not raise - it returns False - and on Windows it fails for any
        path outside the active code page. A project folder named in Japanese
        is the normal case for this tool, so every frame of every round would
        silently fail to export and the round would refuse with "0 frames
        survived". Encoding in memory and writing the bytes ourselves goes
        through Python's own (Unicode-correct) file API.
        """
        try:
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                return False
            path.write_bytes(buffer.tobytes())
            return True
        except (OSError, cv2.error):
            return False

    result = ExportResult()
    total = sum(len(v.frames) for v in videos) or 1
    done = 0

    for video in videos:
        if cancel is not None and cancel.is_set():
            result.cancelled = True
            return result
        if not video.frames:
            continue
        folder = out / "images" / video.video_id
        folder.mkdir(parents=True, exist_ok=True)

        wanted = [
            index for index in sorted(video.frames)
            if not (folder / f"{index:06d}.jpg").exists()
        ]
        result.reused += len(video.frames) - len(wanted)
        done += len(video.frames) - len(wanted)
        if not wanted:
            continue
        if video.path is None or not Path(video.path).exists():
            result.failed.append(f"{video.video_id}: the video file is missing")
            result.missing.extend((video.video_id, i) for i in wanted)
            done += len(wanted)
            continue

        cap = cv2.VideoCapture(str(video.path))
        decoded: Optional[int] = None
        try:
            if not cap.isOpened():
                result.failed.append(f"{video.video_id}: could not open the video")
                result.missing.extend((video.video_id, i) for i in wanted)
                done += len(wanted)
                continue
            for index in wanted:
                if cancel is not None and cancel.is_set():
                    result.cancelled = True
                    return result
                frame, decoded = video_source.read_frame_sequential(cap, index, decoded)
                if frame is None or not write_jpeg(folder / f"{index:06d}.jpg", frame):
                    result.missing.append((video.video_id, index))
                else:
                    result.written += 1
                done += 1
                if progress is not None:
                    progress(done / total, f"{video.video_id}: frame {index:,}")
        finally:
            cap.release()

    if result.missing:
        names = {vid for vid, _ in result.missing}
        result.failed.append(
            f"{len(result.missing)} frame(s) could not be read from "
            f"{', '.join(sorted(names))}"
        )
    return result


def missing_images(dataset: dict, root: Path) -> List[str]:
    """Image paths in ``dataset`` that are not on disk under ``root``.

    The check exists because its absence is invisible. The training dataset
    opens ``<root>/<image_path>`` with ``strict=False`` and substitutes a black
    image on any failure, so a wrong root produces a complete run, a falling
    loss curve and a checkpoint - trained entirely on black frames, with the
    boxes scaled by the wrong original size on top. The CLI will never say so;
    this is the only thing that will.
    """
    out: List[str] = []
    for video in dataset.get("videos") or []:
        for frame in video.get("frames") or []:
            rel = str(frame.get("image_path", ""))
            if not rel or not (root / rel).exists():
                out.append(rel or f"{video.get('video_id')}:{frame.get('frame_index')}")
    return out


def rewrite_image_paths(dataset: dict) -> Dict[str, int]:
    """Point every frame at the round's own image tree, in the dict only.

    The live project is not touched. ``meta["image_root"]`` is cleared in the
    copy because the training loader ignores it entirely and resolves against
    the folder holding the annotation file - leaving a stale value in would be
    a claim nothing reads and everyone believes.
    """
    counts: Dict[str, int] = {}
    meta = dataset.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["image_root"] = ""
    for video in dataset.get("videos") or []:
        vid = str(video.get("video_id", ""))
        for frame in video.get("frames") or []:
            frame["image_path"] = _image_name(vid, int(frame.get("frame_index", 0)))
            counts[vid] = counts.get(vid, 0) + 1
    return counts


def drop_frames(dataset: dict, keys: Sequence[FrameKey]) -> int:
    """Remove frames whose image could not be written. Returns how many went.

    Shipping an annotation with no image is the black-frame failure with extra
    steps, so a frame that failed to export leaves the dataset rather than the
    trainer finding out about it silently.
    """
    drop = set(keys)
    if not drop:
        return 0
    removed = 0
    videos = []
    for video in dataset.get("videos") or []:
        vid = str(video.get("video_id", ""))
        rows = []
        for frame in video.get("frames") or []:
            if (vid, int(frame.get("frame_index", 0))) in drop:
                removed += 1
            else:
                rows.append(frame)
        if rows:
            video["frames"] = rows
            videos.append(video)
    dataset["videos"] = videos
    return removed
