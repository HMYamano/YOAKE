"""Video-owned editing state and compatibility properties for the canvas.

Legacy canvas methods still say ``self.annotations``. The descriptor resolves
that name to the active VideoSession, so even assignments cannot detach a
frame dictionary from its video. There is no shared frame or track namespace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .intervals import IntervalTrack
from .review_queue import ReviewQueue
from .tracklet import TrackletIndex


@dataclass
class VideoSession:
    video_id: str = ""
    path: Path | None = None
    fps: float = 25.0
    width: int = 640
    height: int = 640
    num_frames: int = 0
    annotations: dict = field(default_factory=dict)
    frame_metadata: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    predictions_path: Path | None = None
    review: Any = None
    queue: ReviewQueue = field(default_factory=ReviewQueue)
    tracklets: TrackletIndex = field(default_factory=TrackletIndex)
    intervals: IntervalTrack = field(default_factory=IntervalTrack)
    selection: Any = None
    current_frame: int = 0
    active_track_id: int = 1
    next_track_id: int = 1
    active_class_id: int = 0
    undo: list = field(default_factory=list)
    redo: list = field(default_factory=list)
    dirty: bool = False
    revision: int = 0
    status: str = "new"
    visited: set = field(default_factory=set)
    ui: dict = field(default_factory=dict)
    transient: dict = field(default_factory=dict)
    #: The final pass over every frame of this video, as
    #: :mod:`htrtdetr.annotation.finalpass` shapes it. ``{}`` until the pass
    #: has run. The boxes themselves are **not** here - they are in the
    #: ledger on disk - because this dict is serialised into the project
    #: manifest on every autosave and an hour of video is 216,000 boxes.
    final_pass: dict = field(default_factory=dict)

    @property
    def missing(self) -> bool:
        return self.path is None or not self.path.is_file()

    @property
    def pending(self) -> int:
        return self.review.counts().pending if self.review is not None else 0

    @property
    def display_status(self) -> str:
        # Done is an explicit human decision, never inferred from box coverage.
        if self.pending or self.queue.progress().remaining:
            return "needs_review"
        if self.status == "done":
            return "done"
        return (
            "in_progress"
            if self.status == "in_progress" or self.annotations or self.visited
            else "new"
        )

    def touch(self) -> None:
        self.dirty = True
        self.revision += 1


VIDEO_FIELDS = {
    "video_path": "path",
    "fps": "fps",
    "frame_w": "width",
    "frame_h": "height",
    "num_frames": "num_frames",
    "current_frame_idx": "current_frame",
    "annotations": "annotations",
    "active_track_id": "active_track_id",
    "next_track_id": "next_track_id",
    "active_class_id": "active_class_id",
    "_predictions": "predictions",
    "_predictions_path": "predictions_path",
    "_review": "review",
    "_queue": "queue",
    "_tracklets": "tracklets",
    "_intervals": "intervals",
    "_selection": "selection",
    "_undo": "undo",
    "_redo": "redo",
}

# Native trackers and gesture state are intentionally not serialized. Keeping
# them video-local still prevents a press, late reply or undo from crossing videos.
TRANSIENT_DEFAULTS = {
    "_click_radius_hint": None,
    "_selected": None,
    "_sel_tracklet": None,
    "_sel_interval": None,
    "_sel_issue": None,
    "_current_bgr": None,
    "_pending_snap": None,
    "_right_click_obj": None,
    "_draw_start": None,
    "_draw_end": None,
    "_drag_start": None,
    "_drag_bbox_orig": None,
    "_drag_obb_orig": None,
    "_resize_handle": None,
    "_mode": "idle",
    "_moved": False,
    "_sam_embed_due": None,
    "_sam_waiting": False,
    "_zoom": 1.0,
    "_pan_x": 0.0,
    "_pan_y": 0.0,
    "_pan_origin": None,
    "_middle_pan": None,
    "_cursor_canvas": None,
    "_embedding_info": None,
}


class VideoField:
    def __init__(self, name: str, transient: bool = False, default=None):
        self.name, self.transient, self.default = name, transient, default

    def session(self, app) -> VideoSession:
        # Also supports headless tests constructing AnnotationApp via __new__.
        if "_video_session" not in app.__dict__:
            app.__dict__["_video_session"] = VideoSession()
        return app.__dict__["_video_session"]

    def __get__(self, app, owner=None):
        if app is None:
            return self
        session = self.session(app)
        if self.transient:
            if self.name not in session.transient:
                session.transient[self.name] = (
                    self.default() if callable(self.default) else self.default
                )
            return session.transient[self.name]
        return getattr(session, self.name)

    def __set__(self, app, value):
        session = self.session(app)
        if self.transient:
            session.transient[self.name] = value
        else:
            setattr(session, self.name, value)


def video_scoped(cls):
    for name, field_name in VIDEO_FIELDS.items():
        setattr(cls, name, VideoField(field_name))
    for name, default in TRANSIENT_DEFAULTS.items():
        setattr(cls, name, VideoField(name, True, default))
    cls._trackers = VideoField("_trackers", True, dict)
    return cls
