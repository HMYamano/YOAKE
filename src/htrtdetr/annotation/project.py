"""Project identity, dataset adapters and portable workspace persistence.

Opening a project never opens a capture. Project files reference a committed
dataset snapshot; a killed save cannot pair new review decisions with old GT.
The conventional annotations.json is the training/export copy of that snapshot.

A project is a **folder**, whose shape :mod:`htrtdetr.annotation.layout` owns:
annotations beside the manifest, training data under ``datasets/`` and
``runs/``, analysis output under ``analysis/``. Everything written into it is
relative to its root, and every capture outside it is additionally remembered
by absolute path, so the folder keeps working after it is moved or copied.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..data.annotation import (
    DatasetAnno,
    FrameAnno,
    VideoAnno,
    load_annotation,
    save_annotation,
)
from ..gui.auto_label import Candidate, ReviewSession
from .intervals import BehaviorInterval, IntervalTrack
from .layout import (
    PROJECT_FILE,
    ProjectLayout,
    anchor_path,
    find_project,
    is_project_root,
    locate_media,
    portable_path,
    project_folder_name,
    resolve_media,
)
from .objects import AnnObject
from .quality import ReviewItem
from .review_queue import ReviewQueue
from .session import VideoSession
from .storage import atomic_json

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
SCOPES = (
    "Current video",
    "Selected videos",
    "All unfinished videos",
    "All unannotated videos",
    "Entire project",
)
STATUS_LABELS = {
    "new": "New",
    "in_progress": "In progress",
    "needs_review": "Needs review",
    "done": "Done",
}


def scan_folder(folder: Path, *, recursive: bool = False) -> list[Path]:
    root = Path(folder).resolve()
    paths = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (
            p
            for p in paths
            if p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
            and not any(part.startswith(".") for part in p.relative_to(root).parts)
        ),
        key=lambda p: str(p).casefold(),
    )


def relative_path(path: Path | None, base: Path) -> str:
    """Deprecated alias kept for callers outside this package."""
    return portable_path(path, base)


def resolve_path(value: str, base: Path) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p


@dataclass
class ProjectSession:
    name: str = "untitled"
    class_names: list = field(default_factory=lambda: ["fly"])
    action_names: list = field(default_factory=list)
    videos: dict[str, VideoSession] = field(default_factory=dict)
    last_video: str = ""
    #: ``use_track_ids`` is the one setting that changes what the annotator
    #: asks of the user on every frame: with it off, boxes are per-frame
    #: detections, carry ``track_id = -1``, and the track-based checks are not
    #: run. It defaults to True so every project written before it existed
    #: keeps behaving exactly as it did.
    settings: dict = field(
        default_factory=lambda: {
            "end_of_video": "stay",
            "onboarding_done": False,
            "use_track_ids": True,
        }
    )
    metadata: dict = field(default_factory=dict)
    keypoint_names: list = field(default_factory=list)
    action_groups: dict = field(default_factory=dict)
    directory: Path | None = None
    selected_videos: set = field(default_factory=set)
    sampling_log: list = field(default_factory=list)
    identity: str = field(default_factory=lambda: uuid.uuid4().hex)
    #: Videos this open had to find again by name, so the UI can say so once
    #: instead of letting the user wonder whether it opened the right file.
    relocated: list = field(default_factory=list)

    # ── the folder ───────────────────────────────────────────────────────
    @property
    def layout(self) -> ProjectLayout | None:
        """Where everything this project owns belongs. ``None`` until it has a home."""
        return ProjectLayout(self.directory) if self.directory is not None else None

    @classmethod
    def create(cls, parent: Path, name: str, **kwargs) -> ProjectSession:
        """Make ``<parent>/<name>/`` with the full skeleton and save the manifest.

        Creating the folder here rather than at the first save is what lets the
        rest of the workflow — training, analysis, caches — address a project
        folder from the moment it is named, instead of guessing at one.
        """
        folder = Path(parent).expanduser() / project_folder_name(name)
        if is_project_root(folder):
            raise FileExistsError(f"{folder} already contains a YOAKE project.")
        project = cls(name=str(name).strip() or "untitled", directory=folder.resolve(), **kwargs)
        project.save(folder / PROJECT_FILE)
        return project

    # ── locating captures again ──────────────────────────────────────────
    @property
    def search_roots(self) -> list[Path]:
        """Folders the user has pointed at before, newest first.

        Relinking one video usually answers the question for all of them: the
        capture drive was remounted, or the study was restored beside its
        videos. Remembering the folder turns the second, third and twentieth
        broken link into no question at all.
        """
        return [Path(p) for p in self.settings.get("search_roots", []) if p]

    def remember_search_root(self, folder: Path) -> None:
        folder = Path(folder)
        folder = folder if folder.is_dir() else folder.parent
        known = [str(folder)] + [str(p) for p in self.search_roots if p != folder]
        self.settings["search_roots"] = known[:8]

    def import_media(self, *, move: bool = False) -> tuple[int, list[str]]:
        """Copy (or move) every referenced capture into ``videos/``.

        This is the difference between a folder that travels and a folder that
        merely remembers where things were. Returns ``(imported, failures)``.
        """
        layout = self.layout
        if layout is None:
            raise ValueError("The project has no folder yet. Save it first.")
        layout.videos.mkdir(parents=True, exist_ok=True)
        imported, failures = 0, []
        for video in self.videos.values():
            if video.path is None or layout.contains(video.path) or not video.path.is_file():
                continue
            target = layout.videos / video.path.name
            stem, suffix, index = target.stem, target.suffix, 2
            while target.exists() and not target.samefile(video.path):
                target = layout.videos / f"{stem}_{index}{suffix}"
                index += 1
            try:
                if move:
                    shutil.move(str(video.path), target)
                else:
                    shutil.copy2(video.path, target)
            except OSError as exc:
                failures.append(f"{video.video_id}: {exc}")
                continue
            video.path = target.resolve()
            video.touch()
            imported += 1
        return imported, failures

    def add_video(self, path: Path) -> VideoSession:
        path = Path(path).resolve()
        match = self.find_path(path)
        if match is not None:
            return match
        # A legacy JSON often has only video_id, with no path. Bind only when
        # the stem identifies exactly one unlinked video; never guess by index.
        legacy = [v for v in self.videos.values() if v.path is None and v.video_id == path.stem]
        if len(legacy) == 1:
            legacy[0].path = path
            legacy[0].touch()
            return legacy[0]
        stem = re.sub(r"[^\w.-]+", "_", path.stem).strip(".") or "video"
        vid = stem
        suffix = 2
        while vid in self.videos:
            vid = f"{stem}_{suffix}"
            suffix += 1
        video = VideoSession(video_id=vid, path=path)
        self.videos[vid] = video
        if not self.last_video:
            self.last_video = vid
        video.touch()
        return video

    def find_path(self, path: Path) -> VideoSession | None:
        target = os.path.normcase(str(Path(path).resolve()))
        return next(
            (
                v
                for v in self.videos.values()
                if v.path is not None and os.path.normcase(str(v.path.resolve())) == target
            ),
            None,
        )

    def relink(self, video_id: str, path: Path) -> None:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        other = self.find_path(path)
        if other is not None and other.video_id != video_id:
            raise ValueError("This file already belongs to another project video.")
        self.videos[video_id].path = path
        self.videos[video_id].touch()
        self.remember_search_root(path.parent)

    def relink_missing(self, folder: Path) -> list[str]:
        """Point every missing video at a same-named file under ``folder``.

        The one-at-a-time Locate dialog is the right tool for one broken link
        and the wrong tool for forty: a restored backup breaks all of them for
        the same reason, and answering "the videos are over here" once should
        fix all of them.
        """
        folder = Path(folder)
        self.remember_search_root(folder)
        relinked = []
        for video in self.videos.values():
            if not video.missing:
                continue
            name = video.path.name if video.path is not None else f"{video.video_id}.mp4"
            found = locate_media(name, [folder])
            if found is None or self.find_path(found) is not None:
                continue
            video.path = found.resolve()
            video.touch()
            relinked.append(video.video_id)
        return relinked

    def scope(self, scope: str, current: str = "") -> list[VideoSession]:
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope: {scope}")
        return [
            v
            for v in self.videos.values()
            if (scope == "Entire project")
            or (scope == "Current video" and v.video_id == current)
            or (scope == "Selected videos" and v.video_id in self.selected_videos)
            or (scope == "All unfinished videos" and v.display_status != "done")
            or (scope == "All unannotated videos" and not v.annotations)
        ]

    def dataset(self, base: Path) -> DatasetAnno:
        data = DatasetAnno(
            meta=copy.deepcopy(self.metadata),
            class_names=list(self.class_names),
            action_names=list(self.action_names),
            keypoint_names=list(self.keypoint_names),
            action_groups=copy.deepcopy(self.action_groups),
        )
        for video in self.videos.values():
            frames = []
            for index, objects in sorted(video.annotations.items()):
                extra = video.frame_metadata.get(index, {})
                frames.append(
                    FrameAnno(
                        frame_index=index,
                        image_path=extra.get(
                            "image_path", f"images/{video.video_id}/{index:06d}.jpg"
                        ),
                        width=extra.get("width", 0),
                        height=extra.get("height", 0),
                        objects=[o.to_annotation() for o in objects],
                    )
                )
            data.videos.append(
                VideoAnno(
                    video_id=video.video_id,
                    fps=video.fps,
                    width=video.width,
                    height=video.height,
                    num_frames=video.num_frames,
                    video_path=relative_path(video.path, base),
                    frames=frames,
                )
            )
        data.meta["version"] = (
            "2.0"
            if any(o.obb is not None for v in data.videos for f in v.frames for o in f.objects)
            else "1.1"
        )
        data.meta.setdefault("description", f"YOAKE project: {self.name}")
        data.meta.setdefault("image_root", "")
        return data

    @classmethod
    def from_annotation(cls, path: Path) -> ProjectSession:
        path = Path(path).resolve()
        data = load_annotation(str(path))
        project = cls(
            name=path.stem,
            class_names=data.class_names,
            action_names=data.action_names,
            metadata=data.meta,
            keypoint_names=data.keypoint_names,
            action_groups=data.action_groups,
            directory=path.parent,
        )
        for item in data.videos:
            if item.video_id in project.videos:
                raise ValueError(f"Duplicate video_id: {item.video_id}")
            video = VideoSession(
                video_id=item.video_id,
                path=resolve_path(item.video_path, path.parent),
                fps=item.fps,
                width=item.width,
                height=item.height,
                num_frames=item.num_frames,
            )
            for frame in item.frames:
                if frame.frame_index in video.annotations:
                    raise ValueError(f"Duplicate frame {item.video_id}:{frame.frame_index}")
                video.annotations[frame.frame_index] = [
                    AnnObject.from_dict({**asdict(obj), "keyframe": True}) for obj in frame.objects
                ]
                video.frame_metadata[frame.frame_index] = {
                    "image_path": frame.image_path,
                    "width": frame.width,
                    "height": frame.height,
                }
            video.next_track_id = 1 + max(
                (o.track_id for row in video.annotations.values() for o in row), default=0
            )
            video.active_track_id = video.next_track_id
            project.videos[video.video_id] = video
        project.last_video = next(iter(project.videos), "")
        return project

    def snapshot(self, base: Path) -> tuple[DatasetAnno, dict]:
        """Detach mutable data on the main thread before writing on a worker."""
        rows = []
        for video in self.videos.values():
            review = []
            if video.review is not None:
                for frame in video.review.frames:
                    review.append(
                        {
                            "frame": frame,
                            "candidates": [
                                {
                                    "candidate": asdict(c),
                                    "state": video.review.state(frame, i),
                                    "edited": video.review.is_edited(frame, i),
                                }
                                for i, c in enumerate(video.review.candidates(frame))
                            ],
                        }
                    )
            issues = [
                {"item": asdict(item), "decision": asdict(video.queue.decision(item))}
                for item in video.queue.items
            ]
            rows.append(
                {
                    "video_id": video.video_id,
                    "path": portable_path(video.path, base),
                    # Only for captures outside the folder, and only as a
                    # fallback: see layout.resolve_media for the order.
                    "abs_path": anchor_path(video.path, base),
                    "last_frame": video.current_frame,
                    "active_track_id": video.active_track_id,
                    "next_track_id": video.next_track_id,
                    "active_class_id": video.active_class_id,
                    "status": video.status,
                    "visited": sorted(video.visited),
                    "keyframes": {
                        str(f): {str(o.object_id): o.keyframe for o in objects}
                        for f, objects in video.annotations.items()
                    },
                    "ui": copy.deepcopy(video.ui),
                    "review": review,
                    "issues": issues,
                    "intervals": [asdict(i) for i in video.intervals.intervals],
                    "predictions": {
                        str(f): [asdict(c) for c in cs] for f, cs in video.predictions.items()
                    },
                    "selection": asdict(video.selection) if video.selection is not None else None,
                }
            )
        state = {
            # Still 1: the manifest only gained optional keys, and a reader of
            # either version ignores what it does not know. Bumping it would
            # make an older YOAKE refuse a project it can in fact open.
            "schema_version": 1,
            "name": self.name,
            "identity": self.identity,
            "last_video": self.last_video,
            "settings": copy.deepcopy(self.settings),
            "videos": rows,
            "sampling_log": copy.deepcopy(self.sampling_log),
        }
        return copy.deepcopy(self.dataset(base)), state

    def save(self, path: Path) -> None:
        path = Path(path).resolve()
        layout = ProjectLayout(path.parent).create()
        data, state = self.snapshot(path.parent)
        save_snapshot(path, data, state)
        layout.write_readme(self.name)
        self.directory = path.parent
        for video in self.videos.values():
            video.dirty = False

    @classmethod
    def open(cls, path: Path) -> ProjectSession:
        """Open a manifest, a project folder, or a bare annotations.json."""
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            manifest = find_project(path)
            if manifest is None:
                raise FileNotFoundError(f"{path} is not a YOAKE project folder.")
            path = manifest
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "schema_version" not in raw or "annotations" not in raw:
            return cls.from_annotation(path)
        if raw["schema_version"] != 1:
            raise ValueError(f"Unsupported project version: {raw['schema_version']}")
        annotation = resolve_path(raw["annotations"], path.parent)
        if annotation is None:
            raise ValueError("Project does not reference an annotation snapshot")
        project = cls.from_annotation(annotation)
        project.directory = path.parent
        project.name = raw.get("name", path.stem)
        project.identity = raw.get("identity", project.identity)
        project.last_video = raw.get("last_video", project.last_video)
        project.settings.update(raw.get("settings", {}))
        project.sampling_log = raw.get("sampling_log", [])
        project.relocated = []
        for row in raw.get("videos", []):
            video = project.videos[row["video_id"]]
            # The project may have been moved, or the captures may have been,
            # or both. resolve_media decides in that order; a name search that
            # succeeds is written back on the next save, so it happens once.
            found, how = resolve_media(
                row.get("path", ""),
                row.get("abs_path", ""),
                path.parent,
                search_roots=project.search_roots,
            )
            video.path = found
            if how == "found":
                project.relocated.append(video.video_id)
                video.touch()
            video.current_frame = max(0, int(row.get("last_frame", 0)))
            video.next_track_id = max(video.next_track_id, int(row.get("next_track_id", 1)))
            video.active_track_id = row.get("active_track_id", video.next_track_id)
            video.active_class_id = row.get("active_class_id", 0)
            video.status = row.get("status", "new")
            video.visited = set(row.get("visited", []))
            for frame, flags in row.get("keyframes", {}).items():
                for obj in video.annotations.get(int(frame), []):
                    obj.keyframe = bool(flags.get(str(obj.object_id), True))
            video.ui = row.get("ui", {})
            video.intervals = IntervalTrack(
                [BehaviorInterval(**i) for i in row.get("intervals", [])]
            )
            video.review = ReviewSession()
            for frame in row.get("review", []):
                index = int(frame["frame"])
                video.review.add_frame(
                    index, [Candidate(**c["candidate"]) for c in frame["candidates"]]
                )
                for i, candidate in enumerate(frame["candidates"]):
                    video.review._set_state(index, i, candidate["state"])
                    if candidate.get("edited"):
                        video.review._edited[index][i] = True
            from .tracklet import TrackletIndex, build_tracklets

            video.tracklets = TrackletIndex(build_tracklets(video.review.pending_candidates()))
            video.predictions = {
                int(f): [Candidate(**c) for c in cs] for f, cs in row.get("predictions", {}).items()
            }
            items = []
            for issue in row.get("issues", []):
                values = dict(issue["item"])
                values["track_ids"] = tuple(values.get("track_ids", ()))
                items.append(ReviewItem(**values))
            video.queue = ReviewQueue(items)
            for item, issue in zip(items, row.get("issues", [])):
                decision = issue["decision"]
                video.queue.set_decision(item, decision["state"], **decision.get("data", {}))
            if row.get("selection"):
                from .sampling import Clip, Selection

                values = dict(row["selection"])
                values["clips"] = tuple(Clip(**c) for c in values.get("clips", []))
                values["frames"] = tuple(values.get("frames", []))
                values["reasons"] = {int(k): v for k, v in values.get("reasons", {}).items()}
                video.selection = Selection(**values)
        return project


def save_snapshot(path: Path, data: DatasetAnno, state: dict) -> None:
    """Commit snapshot first, then atomically move the manifest to it."""
    base = path.parent
    layout = ProjectLayout(base)
    generation = layout.snapshots / f"{uuid.uuid4().hex}.json"
    # Dataset paths remain relative to the project directory. The manifest's
    # per-video paths are authoritative when reopening the internal snapshot.
    save_annotation(data, str(generation), preserve_derived=True)
    save_annotation(data, str(layout.annotations), preserve_derived=True)
    state = {**state, "annotations": portable_path(generation, base)}
    atomic_json(layout.sampling_log, state.get("sampling_log", []))
    atomic_json(path, state)
    # Keep the committed generation and one predecessor, without growing on
    # every autosave. This directory holds only workspace-owned snapshots.
    try:
        snapshots = sorted(
            (
                p
                for p in generation.parent.glob("*.json")
                if re.fullmatch(r"[0-9a-f]{32}\.json", p.name)
            ),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return  # Housekeeping cannot turn a committed save into a failed save.
    for old in snapshots[2:]:
        try:
            old.unlink()
        except OSError:
            pass
