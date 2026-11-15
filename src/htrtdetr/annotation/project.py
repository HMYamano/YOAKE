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
from typing import Collection

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
    #: ``video_id -> (session, (revision, base), row)`` from the last
    #: :meth:`snapshot`. Not saved and not part of the project's identity - it
    #: exists only so the autosave does not re-detach twenty videos'
    #: candidates every four seconds to write the nineteen that did not change.
    _row_cache: dict = field(default_factory=dict, repr=False, compare=False)

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
        if self.settings.get("label_mode") == "keyframe":
            # The file says what its frames mean, so training reads them as
            # keyframes (docs/KEYFRAME_BEHAVIOR.md) without being told again.
            from .keyframes import ClipSpec, set_labeling

            set_labeling(data.meta, ClipSpec.from_dict(self.settings.get("clip") or {}))
        else:
            data.meta.pop("labeling", None)
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
        from .keyframes import KEYFRAME, clip_spec_of, label_mode_of

        if label_mode_of(data) == KEYFRAME:
            # A file that declares keyframe labels opens as a keyframe project:
            # its frames are keyframes, and it has no track ids to edit.
            project.settings["label_mode"] = KEYFRAME
            project.settings["use_track_ids"] = False
            project.settings["clip"] = clip_spec_of(data).to_dict()
        return project

    def snapshot(
        self, base: Path, *, always: Collection[str] = ()
    ) -> tuple[DatasetAnno, dict]:
        """Detach mutable data on the main thread before writing on a worker.

        On the main thread deliberately: the worker must never read structures
        the user is still editing. That makes the detach itself a freeze, and
        after a detector has run it is a long one - 28,800 candidates a video,
        and this study is twenty one-hour recordings. So the detach is both as
        cheap as it can be made (:meth:`Candidate.to_dict`,
        :meth:`ReviewSession.saved_rows`) and **not repeated for videos that
        have not changed**.

        ``always`` names video ids whose row must be rebuilt whatever the
        cache thinks. The autosave passes the open video, because three of the
        saved fields - ``visited``, ``last_frame`` and ``ui`` - change as the
        user simply looks around and deliberately do **not** mark the video
        dirty (a frame step must not make the save chip say "Unsaved"). Every
        other write to a video goes through :meth:`VideoSession.touch`, which
        is what the cache is keyed on. :meth:`save` passes every id, so a
        deliberate Ctrl+S is never anything but a full, current detach.
        """
        # A removed video's row would otherwise sit here holding its
        # candidates for the rest of the session - and, worse, a video removed
        # and re-added under the same id gets a fresh session whose revision
        # starts at 0 again, which is a cache hit on somebody else's data.
        # Pruning covers the first; the ``is`` check below covers the second,
        # and is why the entry keeps a reference to the session it was built
        # from.
        live = set(self.videos)
        for stale in [vid for vid in self._row_cache if vid not in live]:
            del self._row_cache[stale]

        rows = []
        key = str(base)
        for video in self.videos.values():
            cached = self._row_cache.get(video.video_id)
            if (
                cached is not None
                and cached[0] is video
                and cached[1] == (video.revision, key)
                and video.video_id not in always
            ):
                rows.append(cached[2])
                continue
            row = self._video_row(video, base)
            self._row_cache[video.video_id] = (video, (video.revision, key), row)
            rows.append(row)
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

    def _video_row(self, video, base: Path) -> dict:
        """One video's saved form, fully detached from the live objects."""
        # ``saved_rows`` rather than a walk over the public accessors: it is
        # the largest thing in the file by two orders of magnitude once a
        # detector has run, and it was costing 0.41 s a video through
        # ``asdict`` and a copied list per frame. Same rows, a fifteenth of
        # the time.
        review = video.review.saved_rows() if video.review is not None else []
        issues = [
            {"item": asdict(item), "decision": asdict(video.queue.decision(item))}
            for item in video.queue.items
        ]
        return {
            "video_id": video.video_id,
            "path": portable_path(video.path, base),
            # Only for captures outside the folder, and only as a fallback:
            # see layout.resolve_media for the order.
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
            # The same candidates again, from the other side: a loaded
            # predictions.json. ``to_dict`` for the same reason.
            "predictions": {
                str(f): [c.to_dict() for c in cs] for f, cs in video.predictions.items()
            },
            "selection": asdict(video.selection) if video.selection is not None else None,
            # Where the final pass has got to. Small by construction - a
            # fingerprint, two flags and a list of merged spans - because the
            # boxes it is about live in the ledger beside this file.
            "final_pass": copy.deepcopy(video.final_pass) if video.final_pass else None,
        }

    def save(self, path: Path) -> None:
        path = Path(path).resolve()
        layout = ProjectLayout(path.parent).create()
        data, state = self.snapshot(path.parent, always=tuple(self.videos))
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
            if isinstance(row.get("final_pass"), dict):
                video.final_pass = dict(row["final_pass"])
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
