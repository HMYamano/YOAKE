"""
auto_label.py — YOAKE auto-annotation engine (pure Python, no GUI)
=================================================================

The engine behind "Auto-Annotate" in the annotator and "Send predictions to
annotator" in the analysis panel. It owns four things and nothing else:

1. **Capability probing** — is PyTorch here, is the package importable, which
   OpenCV tracker backend exists — plus the exact English sentence the GUI
   shows when the answer is "no".
2. **Child-process data** — the argv and the environment for
   ``python -m htrtdetr.cli predict``. It *builds* them; the caller spawns.
3. **Candidates** — reading ``predictions.json`` (all three known writers),
   remapping exported frame indices back onto real video frames, merging the
   result into an existing annotation set without ever destroying human work,
   and the accept / reject review state machine on top of that.
4. **A torch-free fallback** — propagating one box forward with OpenCV, so the
   tool is still useful on a machine that cannot run the detector.

Hard invariants (asserted by ``tests/test_gui_auto_label.py``):

* No ``dearpygui`` import, at module scope or inside a function.
* **No ``torch`` import, ever.** Capability detection uses
  ``importlib.util.find_spec`` only; importing torch costs seconds and pulls
  CUDA DLLs on a machine that may not want them.
* ``cv2`` (and ``numpy``) are imported lazily, inside the tracker functions, so
  this module imports on a machine without OpenCV.
* No ``subprocess`` call. ``predict_argv`` and ``child_env`` are data.
* Every public function is deterministic given its arguments — the only global
  state read is ``os.environ``, inside ``child_env`` / ``ensure_child_env``.

The central design rule, and the one everything else rests on: **a candidate is
never written to disk.** Machine output arrives unaccepted, review is the only
act with consequences, and therefore re-running the detector is always safe and
an unreviewed run can never poison a training set.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    # capability
    "Capability",
    "BackendStatus",
    "probe",
    "check_backend",
    "TORCH_MISSING_MESSAGE",
    "PACKAGE_MISSING_MESSAGE",
    "TRACKER_LABELS",
    # environment
    "child_env",
    "ensure_child_env",
    "device_env",
    "repo_root_for",
    # command
    "PredictRequest",
    "AutoLabelRequest",
    "AutoLabelResult",
    "predict_argv",
    "build_command",
    "predictions_path_for",
    # frame export
    "ExportPlan",
    "export_plan",
    "write_index_map",
    "read_index_map",
    "remap_frame_indices",
    # predictions
    "Candidate",
    "load_candidates",
    "filter_candidates",
    # merge
    "MergePolicy",
    "MergeResult",
    "merge_candidates",
    "merge_into",
    "describe",
    "iou_xyxy",
    "iou_obb",
    "iou_any",
    "obb_to_xyxy",
    # annotation records
    "provenance_fields",
    "candidate_to_record",
    "predictions_to_annotations",
    # review
    "PENDING",
    "ACCEPTED",
    "REJECTED",
    "ReviewCounts",
    "ReviewSession",
    # tracking
    "tracker_backend",
    "make_tracker",
    "lk_propagate",
    "track_forward",
]


# --------------------------------------------------------------------------- #
# 1. Capability probing
# --------------------------------------------------------------------------- #

TORCH_MISSING_MESSAGE = (
    "Automatic detection needs PyTorch, which is not installed in this Python "
    "environment.\n"
    "Install it with:  pip install torch torchvision\n"
    "You can also run the detector on another machine and bring the result back with\n"
    '"Import Predictions as Candidates...":\n'
    "python -m htrtdetr.cli predict source=<video> weights=<checkpoint> "
    "output_dir=<folder> score_threshold=0.30"
)

PACKAGE_MISSING_MESSAGE = "YOAKE is not installed in this Python environment. Run: pip install -e ."

TRACKER_LABELS: Dict[str, str] = {
    "csrt": "CSRT",
    "csrt_legacy": "CSRT (legacy)",
    "mil": "MIL",
    "lk": "optical flow",
    "none": "none",
}

_TRACKER_REASONS: Dict[str, str] = {
    "csrt": "Tracker: CSRT.",
    "csrt_legacy": "Tracker: CSRT (legacy OpenCV API).",
    "mil": ("Tracker: MIL. Install opencv-contrib-python for CSRT, which is more accurate."),
    "lk": (
        "No OpenCV tracker is available. Falling back to optical flow, which is "
        "less accurate on fast motion."
    ),
    "none": (
        "OpenCV is not installed in this Python environment, so tracking is "
        "unavailable. Install it with:  pip install opencv-python"
    ),
}


@dataclass(frozen=True)
class Capability:
    """What this Python environment can actually do, and what to say when it cannot."""

    torch: bool
    package: bool
    package_via: str  # "installed" | "src" | ""
    opencv: bool
    tracker: str  # "csrt" | "csrt_legacy" | "mil" | "lk" | "none"
    torch_reason: str = ""
    package_reason: str = ""
    tracker_reason: str = ""

    @property
    def can_detect(self) -> bool:
        """True when ``yoake predict`` can run in this interpreter."""
        return self.torch and self.package

    @property
    def can_track(self) -> bool:
        """True when Track forward works — it never needs torch."""
        return self.tracker != "none"

    @property
    def tracker_label(self) -> str:
        return TRACKER_LABELS.get(self.tracker, self.tracker)


def _has_module(name: str) -> bool:
    """find_spec without ever importing the module, and without raising."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _spec_origin(name: str) -> str:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        return ""
    if spec is None:
        return ""
    if spec.origin:
        return str(spec.origin)
    locations = getattr(spec, "submodule_search_locations", None) or []
    return str(list(locations)[0]) if locations else ""


def _probe_package(repo_root: Path) -> Tuple[bool, str]:
    """``(importable, via)``. ``"src"`` when the import resolves inside ``<root>/src``."""
    src_pkg = repo_root / "src" / "htrtdetr"
    origin = _spec_origin("htrtdetr")
    if origin:
        try:
            resolved = Path(origin).resolve()
            src_resolved = src_pkg.resolve()
            if resolved == src_resolved or src_resolved in resolved.parents:
                return True, "src"
        except (OSError, ValueError):
            pass
        return True, "installed"
    if (src_pkg / "__init__.py").exists():
        return True, "src"
    return False, ""


def probe(repo_root: Path) -> Capability:
    """Inspect this environment. Never imports torch; may import cv2.

    ``package_via == "src"`` is *not* an error: :func:`child_env` fixes the
    src-layout case silently, so ``package_reason`` stays empty and the user
    never sees a ``ModuleNotFoundError``.
    """
    root = Path(repo_root)
    torch_ok = _has_module("torch")
    package_ok, via = _probe_package(root)
    opencv_ok = _has_module("cv2")
    tracker = tracker_backend() if opencv_ok else "none"
    return Capability(
        torch=torch_ok,
        package=package_ok,
        package_via=via,
        opencv=opencv_ok,
        tracker=tracker,
        torch_reason="" if torch_ok else TORCH_MISSING_MESSAGE,
        package_reason="" if package_ok else PACKAGE_MISSING_MESSAGE,
        tracker_reason=_TRACKER_REASONS.get(tracker, ""),
    )


@dataclass(frozen=True)
class BackendStatus:
    """Answer to "can I press Run right now?", with the sentence to show if not."""

    ok: bool
    torch: bool
    package: bool
    checkpoint: bool
    message: str
    capability: Optional[Capability] = None

    def __bool__(self) -> bool:
        return self.ok


def check_backend(repo_root: Path, checkpoint: str = "") -> BackendStatus:
    """Runtime gate for the Auto-Annotate button.

    ``message`` is English and is shown verbatim in the locked tooltip. It names
    the cause, the command that fixes it, and the escape hatch.
    """
    cap = probe(repo_root)
    ckpt_ok = bool(checkpoint) and Path(checkpoint).is_file()

    if not cap.package:
        return BackendStatus(False, cap.torch, False, ckpt_ok, cap.package_reason, cap)
    if not cap.torch:
        return BackendStatus(False, False, True, ckpt_ok, cap.torch_reason, cap)
    if not checkpoint:
        return BackendStatus(
            False,
            True,
            True,
            False,
            "Select a detector checkpoint first. It is the .pth file written to "
            "runs/train/stage4/weights/best.pth by step 2.",
            cap,
        )
    if not ckpt_ok:
        return BackendStatus(
            False,
            True,
            True,
            False,
            f"Checkpoint not found: {checkpoint}. Check the path, or point it at "
            "runs/train/stage4/weights/best.pth.",
            cap,
        )
    return BackendStatus(True, True, True, True, "Ready to run auto-annotation.", cap)


# --------------------------------------------------------------------------- #
# 2. Child-process environment
# --------------------------------------------------------------------------- #


def repo_root_for(start: Path) -> Path:
    """Walk up from ``start`` to the directory that holds ``src/htrtdetr``.

    Falls back to ``start`` itself, so the caller always gets a usable path.
    """
    here = Path(start).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "src" / "htrtdetr" / "__init__.py").exists():
            return candidate
    return here


def child_env(repo_root: Path, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Environment for any ``python -m htrtdetr.*`` child process.

    - ``PYTHONPATH``       — ``<repo_root>/src`` prepended, preserving any
      existing value. Without it every GUI subprocess dies with
      ``ModuleNotFoundError: No module named 'htrtdetr'`` on a src-layout
      checkout that was never ``pip install -e .``-ed.
    - ``PYTHONIOENCODING`` — ``utf-8``. The child still prints some Japanese;
      reading the pipe as utf-8 makes it legible instead of a decode error.
    - ``PYTHONUNBUFFERED`` — ``1``. Otherwise stdout arrives in 8 KB bursts and
      a running job looks frozen.
    """
    env: Dict[str, str] = dict(os.environ if base is None else base)
    src = str(Path(repo_root) / "src")
    existing = env.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    key = os.path.normcase(os.path.normpath(src))
    parts = [p for p in parts if os.path.normcase(os.path.normpath(p)) != key]
    env["PYTHONPATH"] = os.pathsep.join([src, *parts])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def ensure_child_env(repo_root: Path) -> Dict[str, str]:
    """Apply :func:`child_env` to ``os.environ`` in place, idempotently.

    ``gui/runner.py`` spawns without ``env=``, so its children inherit
    ``os.environ``. Calling this once at Hub construction — before any panel is
    built — is what makes every subprocess button in the suite work.
    """
    os.environ.update(child_env(repo_root))
    return dict(os.environ)


def device_env(device: str, base: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Environment overlay that pins the compute device.

    YOAKE has **no** ``device=`` CLI key — every runner picks
    ``cuda if torch.cuda.is_available() else cpu`` (verified in
    ``training/trainer.py``, ``runners/val_runner.py`` and
    ``pipeline/builder.py``). ``CUDA_VISIBLE_DEVICES`` is therefore the only
    honest way to force one, so that is what this does instead of inventing a
    flag that the CLI would silently ignore.

    ``"auto"`` / ``""`` leaves the environment untouched.
    """
    env: Dict[str, str] = dict(os.environ if base is None else base)
    dev = (device or "auto").strip().lower()
    if dev in ("", "auto"):
        return env
    if dev == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif dev == "cuda":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif dev.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = dev.split(":", 1)[1]
    elif dev.isdigit():
        env["CUDA_VISIBLE_DEVICES"] = dev
    return env


# --------------------------------------------------------------------------- #
# 3. The subprocess command
# --------------------------------------------------------------------------- #


def _fmt_num(value: float) -> str:
    """Two decimals when that is lossless, full precision otherwise."""
    v = float(value)
    short = f"{v:.2f}"
    return short if float(short) == v else repr(v)


@dataclass(frozen=True)
class PredictRequest:
    """The minimal, verified ``yoake predict`` call.

    Every key below was confirmed against ``cli.py::run_predict`` and
    ``config/config.py``: ``source`` / ``weights`` / ``output_dir`` /
    ``score_threshold`` / ``window_size`` / ``max_frames`` / ``config`` are read
    there, and ``inference.show_*`` / ``model.detector.nms_threshold`` are real
    config fields.
    """

    source: str  # video file OR a directory of exported frames
    weights: str
    output_dir: str
    score_threshold: float = 0.30
    window_size: int = 16
    max_frames: Optional[int] = None
    config: str = ""  # only when the checkpoint has no embedded config
    nms_threshold: Optional[float] = None


def predict_argv(req: PredictRequest) -> List[str]:
    """Argument list **after** ``sys.executable``.

    Consumers spawn ``[sys.executable, *predict_argv(req)]`` with
    ``cwd=repo_root``, ``env=child_env(repo_root)``, ``stdin=DEVNULL``,
    ``stdout=PIPE``, ``stderr=STDOUT``, ``text=True``, ``encoding="utf-8"``,
    ``errors="replace"`` and, on Windows,
    ``CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP``.

    ``stdin=DEVNULL`` matters: ``cli.run_train`` calls ``input()`` on a missing
    dataset, and an inherited stdin hangs the GUI silently.

    The artefact to read back is always ``<output_dir>/predictions.json``.
    """
    argv: List[str] = [
        "-u",
        "-m",
        "htrtdetr.cli",
        "predict",
        f"source={req.source}",
        f"weights={req.weights}",
        f"output_dir={req.output_dir}",
        f"score_threshold={_fmt_num(req.score_threshold)}",
        f"window_size={int(req.window_size)}",
    ]
    if req.nms_threshold is not None:
        argv.append(f"model.detector.nms_threshold={_fmt_num(req.nms_threshold)}")
    if req.max_frames is not None:
        argv.append(f"max_frames={int(req.max_frames)}")
    if req.config:
        argv.append(f"config={req.config}")
    # The overlay is blanked because only predictions.json is read back. The CLI
    # writes output.mp4 regardless; the caller may delete it.
    argv += [
        "inference.show_bbox=false",
        "inference.show_id=false",
        "inference.show_action=false",
        "inference.show_confidence=false",
    ]
    return argv


@dataclass
class AutoLabelRequest:
    """Everything the Auto-Annotate dialog collects, in one object.

    The fields split into two groups, and the split is deliberate:

    **Sent to the CLI** — ``source`` / ``weights`` / ``output_dir`` /
    ``score_threshold`` / ``window_size`` / ``config`` / ``nms_threshold``.

    **Applied on this side, because the CLI has no such flag** — ``device``
    (see :func:`device_env`), ``class_filter`` and ``max_detections`` (applied
    by :func:`filter_candidates` after the run), and ``start_frame`` /
    ``frame_count`` / ``stride`` (``yoake predict`` has no start-frame option,
    so a range is exported to frames first; see :func:`export_plan`).
    """

    source: str = ""
    weights: str = ""
    output_dir: str = ""
    score_threshold: float = 0.30
    nms_threshold: Optional[float] = None
    window_size: int = 16
    config: str = ""
    # frame range — applied by exporting frames, see export_plan()
    start_frame: int = 0
    frame_count: Optional[int] = None  # None = to the end of the video
    stride: int = 1
    # applied on this side
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "cuda:N"
    max_detections: Optional[int] = None
    class_filter: Tuple[int, ...] = ()
    skip_frames_with_accepted: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_export(self) -> bool:
        """True when the range cannot be expressed with ``max_frames`` alone."""
        return int(self.start_frame) > 0 or int(self.stride) > 1

    def to_predict_request(
        self, *, source: str = "", max_frames: Optional[int] = None
    ) -> PredictRequest:
        """Narrow this to the verified CLI surface.

        ``source`` overrides the video path when a frame range was exported to a
        temporary directory; ``max_frames`` then counts within that directory.
        """
        if max_frames is None and not source and not self.needs_export:
            # Whole-video run from frame 0: max_frames alone expresses the range.
            max_frames = self.frame_count
        return PredictRequest(
            source=source or self.source,
            weights=self.weights,
            output_dir=self.output_dir,
            score_threshold=self.score_threshold,
            window_size=self.window_size,
            max_frames=max_frames,
            config=self.config,
            nms_threshold=self.nms_threshold,
        )

    def validate(self) -> List[str]:
        """English problems with this request; ``[]`` means it can run."""
        problems: List[str] = []
        if not self.source:
            problems.append("Open a video first.")
        if not self.weights:
            problems.append("Select a detector checkpoint first.")
        elif not Path(self.weights).is_file():
            problems.append(f"Checkpoint not found: {self.weights}")
        if not self.output_dir:
            problems.append("Choose an output folder for the run.")
        if not 0.05 <= float(self.score_threshold) <= 0.95:
            problems.append("Score threshold must be between 0.05 and 0.95.")
        if self.nms_threshold is not None and not 0.0 < float(self.nms_threshold) <= 1.0:
            problems.append("NMS threshold must be between 0.0 and 1.0.")
        if int(self.window_size) < 1:
            problems.append("Window size must be 1 or more.")
        if self.frame_count is not None and int(self.frame_count) < 1:
            problems.append("Frames must be 1 or more.")
        if int(self.stride) < 1:
            problems.append("Frame stride must be 1 or more.")
        return problems


def build_command(
    request: Any, *, python: str = "", source: str = "", max_frames: Optional[int] = None
) -> List[str]:
    """Full argv — ``sys.executable`` included — that runs YOAKE inference.

    ``build_command(req) == [sys.executable, *predict_argv(...)]``. Accepts
    either an :class:`AutoLabelRequest` or a :class:`PredictRequest`.
    """
    if isinstance(request, AutoLabelRequest):
        req = request.to_predict_request(source=source, max_frames=max_frames)
    else:
        req = request
        if source or max_frames is not None:
            req = replace(
                req,
                source=source or req.source,
                max_frames=max_frames if max_frames is not None else req.max_frames,
            )
    return [python or sys.executable, *predict_argv(req)]


def predictions_path_for(output_dir: Path) -> Path:
    """The artefact to read back is always ``<output_dir>/predictions.json``."""
    return Path(output_dir) / "predictions.json"


@dataclass
class AutoLabelResult:
    """Outcome of one auto-annotation run, ready for a status line."""

    ok: bool = False
    stopped: bool = False
    returncode: Optional[int] = None
    output_dir: str = ""
    predictions_path: str = ""
    duration_s: float = 0.0
    candidates: Dict[int, List["Candidate"]] = field(default_factory=dict)
    merge: Optional["MergeResult"] = None
    log_tail: Tuple[str, ...] = ()
    message: str = ""

    @property
    def total_candidates(self) -> int:
        return sum(len(v) for v in self.candidates.values())

    def describe(self) -> str:
        """One English sentence for the status bar."""
        if self.message:
            return self.message
        if self.merge is not None:
            return describe(self.merge, stopped=self.stopped)
        if self.stopped:
            return (
                f"Auto-annotation stopped. The {self.total_candidates} candidates "
                "already produced are kept."
            )
        if not self.ok:
            tail = self.log_tail[-1] if self.log_tail else ""
            detail = f" {tail}" if tail else ""
            return f"Auto-annotation failed (exit code {self.returncode}).{detail}"
        return f"Auto-annotation done: {self.total_candidates} candidates."


# --------------------------------------------------------------------------- #
# 4. Frame-range export
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExportPlan:
    """Which real frames to write, and how to get back from local indices."""

    frames: List[int]  # real video frame indices, ascending
    index_map: Dict[str, int]  # {"0": 412, "1": 413, ...} local -> real
    dir_name: str  # "<job_id>"

    @property
    def count(self) -> int:
        return len(self.frames)


def export_plan(
    start: int, count: int, num_frames: int, job_id: str, stride: int = 1
) -> ExportPlan:
    """Plan the frame export for a range.

    ``yoake predict`` has **no start-frame option** — ``max_frames`` counts from
    0, and an image directory is re-indexed from 0. A range therefore requires
    writing the frames out and remembering the offset.

    The consumer writes the JPEGs itself (it owns the ``VideoCapture``) but
    **must use sequential ``grab()`` + selective ``retrieve()``, never
    ``cap.set(CAP_PROP_POS_FRAMES)``** — seeking is off-by-one on non-integer
    fps and agrees at frame 0, so the error is near-impossible to notice.
    """
    start = max(0, int(start))
    stride = max(1, int(stride))
    total = max(0, int(num_frames))
    count = int(count)
    if count < 0:
        count = total
    frames: List[int] = []
    idx = start
    while idx < total and len(frames) < count:
        frames.append(idx)
        idx += stride
    return ExportPlan(
        frames=frames,
        index_map={str(i): f for i, f in enumerate(frames)},
        dir_name=str(job_id),
    )


def write_index_map(plan: ExportPlan, out_dir: Path) -> Path:
    """Write ``<out_dir>/index_map.json``. Returns the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index_map.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.index_map, f, indent=2)
    return path


def read_index_map(path: Path) -> Dict[str, int]:
    """Read an ``index_map.json``. Returns ``{}`` when it is absent."""
    p = Path(path)
    if p.is_dir():
        p = p / "index_map.json"
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{p.name}: could not be read ({exc}).") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{p.name}: expected an object of local index -> frame index.")
    out: Dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def remap_frame_indices(
    cands: Mapping[int, Sequence["Candidate"]], index_map: Mapping[str, int]
) -> Dict[int, List["Candidate"]]:
    """Local index -> real video frame.

    WITHOUT THIS EVERY CANDIDATE LANDS ON THE WRONG FRAME. An empty map is the
    identity, so a whole-video run can call this unconditionally.
    """
    if not index_map:
        return {int(k): list(v) for k, v in cands.items()}
    out: Dict[int, List["Candidate"]] = {}
    for local, items in cands.items():
        real = index_map.get(str(local))
        if real is None:
            continue
        out.setdefault(int(real), []).extend(items)
    return out


# --------------------------------------------------------------------------- #
# 5. Reading predictions.json
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """One machine proposal. Never written to disk while it is unaccepted."""

    bbox: List[int]  # [x1, y1, x2, y2], ints, image px
    obb: Optional[List[float]] = None  # [cx, cy, w, h, theta], theta in radians
    score: float = 1.0
    track_id: int = -1
    action_id: int = -1
    class_id: int = 0
    source: str = "auto"  # "auto" | "track"

    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.bbox[2]) - int(self.bbox[0])

    @property
    def height(self) -> int:
        return int(self.bbox[3]) - int(self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def copy(self) -> "Candidate":
        return Candidate(
            bbox=list(self.bbox),
            obb=list(self.obb) if self.obb is not None else None,
            score=float(self.score),
            track_id=int(self.track_id),
            action_id=int(self.action_id),
            class_id=int(self.class_id),
            source=str(self.source),
            provenance=dict(self.provenance),
        )


def _pad(value, n: int, default, cast):
    """Exactly ``gui/predictions.py::_as_list`` — one padding rule for the suite."""
    if value is None:
        return [default] * n
    out = []
    for v in value:
        try:
            out.append(cast(v))
        except (TypeError, ValueError):
            out.append(default)
    if len(out) < n:
        out = out + [default] * (n - len(out))
    elif len(out) > n:
        out = out[:n]
    return out


def _as_int_box(box: Sequence[float]) -> List[int]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def load_candidates(
    path: Path, *, min_score: float = 0.0, source: str = "auto"
) -> Dict[int, List[Candidate]]:
    """Read a ``predictions.json`` into ``{frame_index: [Candidate, ...]}``.

    Tolerant of all three known writers:

    * **streaming** (``cli.run_predict``, the default) — ``boxes``, ``scores``,
      ``track_ids``, ``action_ids``, optional ``obb`` / ``context_boxes`` /
      ``evidence`` / ``gates`` / ``head_predictions``; **no** ``class_ids``.
    * **batch** (``inference.mode=batch``) — four keys only.
    * **pipeline** (``yoake pipeline``) — ``boxes``, ``scores``, ``class_ids``,
      ``track_ids``, ``interpolated``, ``relations``; **no** ``action_ids``,
      **no** ``obb``.

    Missing fields are padded exactly as ``gui/predictions.py::_as_list`` does:
    ``scores -> 1.0``, ``track_ids -> -1``, ``action_ids -> -1``,
    ``class_ids -> 0``.

    ``min_score`` is **inclusive** at the threshold. Raises ``ValueError`` with
    the file name and the reason on malformed input; never lets a ``KeyError``
    or ``TypeError`` escape.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "predictions.json"
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise ValueError(
            f"{p.name}: the run produced no predictions.json. Check the log for " "the reason."
        ) from exc
    except OSError as exc:
        raise ValueError(f"{p.name}: could not be read ({exc}).") from exc
    except ValueError as exc:
        raise ValueError(f"{p.name}: is not valid JSON ({exc}).") from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"{p.name}: expected an object of frame index -> detections, "
            f"found {type(raw).__name__}."
        )

    out: Dict[int, List[Candidate]] = {}
    for key, val in raw.items():
        try:
            fi = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(val, dict):
            raise ValueError(f"{p.name}: frame {key} is not an object.")
        try:
            boxes = [_as_int_box(b) for b in (val.get("boxes") or [])]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"{p.name}: frame {key} has a malformed box ({exc}).") from exc

        n = len(boxes)
        if n == 0:
            continue
        scores = _pad(val.get("scores"), n, 1.0, float)
        track_ids = _pad(val.get("track_ids"), n, -1, int)
        action_ids = _pad(val.get("action_ids"), n, -1, int)
        class_ids = _pad(val.get("class_ids"), n, 0, int)
        try:
            obbs: List[Optional[List[float]]] = [
                [float(v) for v in o[:5]] if o is not None and len(o) >= 5 else None
                for o in (val.get("obb") or [])
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{p.name}: frame {key} has a malformed obb ({exc}).") from exc
        while len(obbs) < n:
            obbs.append(None)

        items: List[Candidate] = []
        for i in range(n):
            if float(scores[i]) < float(min_score):
                continue
            items.append(
                Candidate(
                    bbox=boxes[i],
                    obb=obbs[i],
                    score=float(scores[i]),
                    track_id=int(track_ids[i]),
                    action_id=int(action_ids[i]),
                    class_id=int(class_ids[i]),
                    source=source,
                )
            )
        if items:
            out[fi] = items
    return out


def filter_candidates(
    cands: Mapping[int, Sequence[Candidate]],
    *,
    min_score: float = 0.0,
    classes: Sequence[int] = (),
    max_per_frame: Optional[int] = None,
) -> Dict[int, List[Candidate]]:
    """Client-side filtering.

    ``yoake predict`` has no class-filter and no max-detections flag (verified
    against ``cli.py`` and ``config/config.py``), so both are applied here,
    after the run, rather than by inventing CLI keys that do not exist.
    """
    allow = {int(c) for c in classes} if classes else None
    out: Dict[int, List[Candidate]] = {}
    for fi, items in cands.items():
        kept = [
            c
            for c in items
            if float(c.score) >= float(min_score) and (allow is None or int(c.class_id) in allow)
        ]
        if max_per_frame is not None and int(max_per_frame) >= 0:
            kept = sorted(kept, key=lambda c: -float(c.score))[: int(max_per_frame)]
        if kept:
            out[int(fi)] = kept
    return out


# --------------------------------------------------------------------------- #
# 6. IoU and the merge rule
# --------------------------------------------------------------------------- #


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Axis-aligned IoU of two ``[x1, y1, x2, y2]`` boxes. 0.0 when disjoint."""
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def obb_to_xyxy(obb: Sequence[float]) -> List[float]:
    """``[cx, cy, w, h, theta]`` -> its circumscribing ``[x1, y1, x2, y2]``."""
    cx, cy, w, h, th = (float(v) for v in obb[:5])
    hw, hh = w / 2.0, h / 2.0
    c, s = math.cos(th), math.sin(th)
    xs, ys = [], []
    for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(cx + lx * c - ly * s)
        ys.append(cy + lx * s + ly * c)
    return [min(xs), min(ys), max(xs), max(ys)]


def _obb_corners(obb: Sequence[float]) -> List[Tuple[float, float]]:
    cx, cy, w, h, th = (float(v) for v in obb[:5])
    hw, hh = w / 2.0, h / 2.0
    c, s = math.cos(th), math.sin(th)
    return [
        (cx + lx * c - ly * s, cy + lx * s + ly * c)
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def _poly_area(poly: Sequence[Tuple[float, float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) / 2.0


def _clip_poly(
    subject: Sequence[Tuple[float, float]], clip: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman clip of one convex polygon by another."""
    out = list(subject)
    n = len(clip)
    for i in range(n):
        if not out:
            return []
        cx1, cy1 = clip[i]
        cx2, cy2 = clip[(i + 1) % n]
        ex, ey = cx2 - cx1, cy2 - cy1

        def side(p: Tuple[float, float]) -> float:
            return ex * (p[1] - cy1) - ey * (p[0] - cx1)

        new: List[Tuple[float, float]] = []
        prev = out[-1]
        prev_s = side(prev)
        for cur in out:
            cur_s = side(cur)
            if cur_s >= 0.0:
                if prev_s < 0.0:
                    t = prev_s / (prev_s - cur_s)
                    new.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
                new.append(cur)
            elif prev_s >= 0.0:
                t = prev_s / (prev_s - cur_s)
                new.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            prev, prev_s = cur, cur_s
        out = new
    return out


def iou_obb(a: Sequence[float], b: Sequence[float]) -> float:
    """Rotated IoU of two ``[cx, cy, w, h, theta]`` boxes.

    Delegates to ``htrtdetr.stages._geometry.rotated_iou_np`` — the torch-free
    numpy implementation that ``tests/test_geometry_obb.py`` already pins
    against the torch one in ``geometry/obb.py`` — so this module does not
    become a third definition of the same geometry. The pure-Python polygon path
    below is the fallback for a machine without numpy or without the package.
    """
    try:
        import numpy as np

        from ..stages._geometry import rotated_iou_np
    except Exception:  # pragma: no cover - only without numpy / outside the package
        pass
    else:
        return float(
            rotated_iou_np(
                np.asarray(list(a[:5]), dtype=float),
                np.asarray(list(b[:5]), dtype=float),
            )
        )

    ca, cb = _obb_corners(a), _obb_corners(b)
    inter = _poly_area(_clip_poly(ca, cb))
    if inter <= 0.0:
        return 0.0
    union = _poly_area(ca) + _poly_area(cb) - inter
    return float(inter / union) if union > 0.0 else 0.0


def iou_any(
    box_a: Sequence[float],
    box_b: Sequence[float],
    obb_a: Optional[Sequence[float]] = None,
    obb_b: Optional[Sequence[float]] = None,
) -> float:
    """Rotated IoU when both sides carry an angle, axis-aligned IoU otherwise."""
    if obb_a is not None and obb_b is not None:
        return iou_obb(obb_a, obb_b)
    return iou_xyxy(box_a, box_b)


@dataclass(frozen=True)
class MergePolicy:
    """How machine output is allowed to touch an existing annotation set.

    ``iou_suppress = 0.60`` **is a judgement, not a measurement.** It was chosen
    so a detection that is plainly the same animal as an existing box is
    dropped, while a neighbour in a crowded frame survives. Record the dataset
    and the false-merge rate here once there is a real video to measure against.
    """

    skip_frames_with_accepted: bool = True
    iou_suppress: float = 0.60
    replace_pending: bool = True
    min_score: float = 0.0


@dataclass
class MergeResult:
    added: int = 0
    frames_touched: int = 0
    frames_total: int = 0
    skipped_protected: int = 0
    suppressed_iou: int = 0
    below_threshold: int = 0
    replaced_pending: int = 0


def merge_candidates(
    existing: Mapping[int, Sequence[Any]],
    incoming: Mapping[int, Sequence[Candidate]],
    policy: MergePolicy,
    *,
    is_accepted: Callable[[Any], bool],
    is_human: Callable[[Any], bool],
    bbox_of: Callable[[Any], Sequence[float]],
    make_obj: Callable[[Candidate], Any],
    obb_of: Optional[Callable[[Any], Optional[Sequence[float]]]] = None,
) -> Tuple[Dict[int, List[Any]], MergeResult]:
    """Pure merge. The callables keep ``AnnObject`` out of this module.

    Rules, applied in this order, per frame:

    1. A frame holding any accepted human box is skipped whole when
       ``skip_frames_with_accepted`` — human work is never overwritten.
    2. ``replace_pending`` drops that frame's existing unaccepted objects, so a
       re-run supersedes rather than stacks.
    3. A candidate below ``min_score`` is dropped.
    4. A candidate whose maximum IoU against an **accepted** box in that frame
       reaches ``iou_suppress`` is dropped.
    """
    out: Dict[int, List[Any]] = {int(k): list(v) for k, v in existing.items()}
    res = MergeResult(frames_total=len(incoming))

    ordered = sorted(incoming.items(), key=lambda kv: int(kv[0]))
    for raw_fi, raw_items in ordered:
        fi = int(raw_fi)
        items = list(raw_items)
        current = out.get(fi, [])

        if policy.skip_frames_with_accepted and any(
            is_accepted(o) and is_human(o) for o in current
        ):
            res.skipped_protected += 1
            continue

        if policy.replace_pending:
            keep = [o for o in current if is_accepted(o)]
            res.replaced_pending += len(current) - len(keep)
            current = keep

        anchors = [o for o in current if is_accepted(o)]
        added_here = 0
        for cand in items:
            if float(cand.score) < float(policy.min_score):
                res.below_threshold += 1
                continue
            worst = 0.0
            for anchor in anchors:
                other_obb = obb_of(anchor) if obb_of is not None else None
                worst = max(worst, iou_any(bbox_of(anchor), cand.bbox, other_obb, cand.obb))
                if worst >= policy.iou_suppress:
                    break
            if worst >= policy.iou_suppress:
                res.suppressed_iou += 1
                continue
            current.append(make_obj(cand))
            added_here += 1

        if current:
            out[fi] = current
        else:
            out.pop(fi, None)
        if added_here:
            res.added += added_here
            res.frames_touched += 1
    return out, res


def describe(res: MergeResult, *, stopped: bool = False) -> str:
    """One English sentence: count first, next action after an em dash."""
    if stopped:
        return f"Auto-annotation stopped. The {res.added} candidates already " "produced are kept."
    if res.added > 0:
        return (
            f"Auto-annotation done: {res.added} candidates on {res.frames_touched} "
            f"of {res.frames_total} frames ({res.skipped_protected} frames protected, "
            f"{res.suppressed_iou} overlapped an existing box) — press Ctrl+Right to "
            "jump to the first one, Y to accept a frame, N to discard it."
        )
    # Distinct empty-result diagnoses, never one generic shrug.
    if res.frames_total == 0:
        return "The run produced no predictions.json. Check the log for the reason."
    if res.skipped_protected and res.skipped_protected >= res.frames_total:
        return (
            "Every frame in the range already has boxes you drew. Untick "
            '"Skip frames that already have boxes I drew" to re-run them.'
        )
    if res.below_threshold:
        return 'No detections above the score threshold. Lower "Score threshold" ' "and run again."
    if res.suppressed_iou:
        return (
            f"Every detection overlapped a box that is already there "
            f"({res.suppressed_iou} suppressed). Nothing new to review."
        )
    return 'No detections above the score threshold. Lower "Score threshold" ' "and run again."


# --------------------------------------------------------------------------- #
# 7. Provenance and annotator-native records
# --------------------------------------------------------------------------- #


def provenance_fields(source: str, score: Optional[float]) -> Dict[str, Any]:
    """The extra keys a non-human box adds to ``AnnObject.to_dict()``.

    Returns ``{}`` for a plain human box, which is what keeps a human-only
    project's ``annotations.json`` byte-identical to today (and
    ``tests/test_annotation_v2.py`` / ``tests/test_labeling_ops.py`` green).

    ``data/annotation.py::_parse_object`` reads named keys and ignores unknown
    ones, so these extras load harmlessly through the library path.
    """
    src = (source or "human").strip()
    if src == "human":
        return {}
    out: Dict[str, Any] = {"source": src}
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def candidate_to_record(
    cand: Candidate, *, object_id: int = 0, accepted: bool = False
) -> Dict[str, Any]:
    """One candidate as the annotator's native object record.

    The shape matches ``annotator.AnnObject.to_dict()`` exactly — same keys,
    same ``obb`` sub-object with ``angle`` in radians — plus ``keyframe``,
    ``accepted`` and the provenance keys, all of which
    ``AnnObject.from_dict`` and the library loader tolerate.
    """
    rec: Dict[str, Any] = {
        "object_id": int(object_id),
        "bbox": [int(v) for v in cand.bbox],
        "class_id": int(cand.class_id),
        "track_id": int(cand.track_id),
        "action_id": int(cand.action_id),
        "occluded": False,
        "is_crowd": False,
        "keyframe": False,
        "accepted": bool(accepted),
    }
    if cand.obb is not None:
        cx, cy, w, h, th = (float(v) for v in cand.obb[:5])
        rec["obb"] = {
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "w": round(w, 2),
            "h": round(h, 2),
            "angle": round(th, 6),
        }
    rec.update(provenance_fields(cand.source, cand.score))
    if cand.provenance:
        rec["provenance"] = dict(cand.provenance)
    if accepted:
        rec.setdefault("provenance", {})["human_review"] = "accepted"
    return rec


def predictions_to_annotations(
    cands: Mapping[int, Sequence[Candidate]],
    *,
    start_object_id: int = 1,
    accepted: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    """``{frame: [Candidate]}`` -> ``{frame: [annotator object record]}``.

    Every record is tagged with its provenance (``source``), its confidence
    (``score``) and an unconfirmed review state (``accepted=False``,
    ``keyframe=False``). Object ids are assigned in frame order from
    ``start_object_id``, so the caller can hand it ``max(existing id) + 1``.
    """
    out: Dict[int, List[Dict[str, Any]]] = {}
    oid = int(start_object_id)
    for fi in sorted(int(k) for k in cands.keys()):
        recs = []
        for cand in cands[fi]:
            recs.append(candidate_to_record(cand, object_id=oid, accepted=accepted))
            oid += 1
        if recs:
            out[fi] = recs
    return out


# Default accessors for annotator-style dict records, used by merge_into().
def _rec_accepted(o: Any) -> bool:
    if isinstance(o, Mapping):
        return bool(o.get("accepted", True))
    return bool(getattr(o, "accepted", True))


def _rec_human(o: Any) -> bool:
    if isinstance(o, Mapping):
        return str(o.get("source", "human")) == "human"
    return str(getattr(o, "source", "human")) == "human"


def _rec_bbox(o: Any) -> Sequence[float]:
    if isinstance(o, Mapping):
        return o["bbox"]
    return getattr(o, "bbox")


def _rec_obb(o: Any) -> Optional[Sequence[float]]:
    raw = o.get("obb") if isinstance(o, Mapping) else getattr(o, "obb", None)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        try:
            return [
                float(raw["cx"]),
                float(raw["cy"]),
                float(raw["w"]),
                float(raw["h"]),
                float(raw.get("angle", raw.get("theta", 0.0))),
            ]
        except (KeyError, TypeError, ValueError):
            return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    return vals[:5] if len(vals) >= 5 else None


def merge_into(
    existing: Mapping[int, Sequence[Dict[str, Any]]],
    suggestions: Mapping[int, Sequence[Candidate]],
    policy: Optional[MergePolicy] = None,
    *,
    next_object_id: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], MergeResult]:
    """:func:`merge_candidates` for plain annotator-style dict records.

    Human work is preserved: a frame that already holds an accepted box the user
    drew is skipped whole, and a candidate overlapping an accepted box is
    suppressed by IoU. Returns the new set and a summary of what was added,
    skipped and replaced.
    """
    pol = policy or MergePolicy()
    if next_object_id is None:
        ids = [
            int(o.get("object_id", 0))
            for objs in existing.values()
            for o in objs
            if isinstance(o, Mapping)
        ]
        next_object_id = (max(ids) + 1) if ids else 1

    counter = {"oid": int(next_object_id)}

    def _make(cand: Candidate) -> Dict[str, Any]:
        rec = candidate_to_record(cand, object_id=counter["oid"], accepted=False)
        counter["oid"] += 1
        return rec

    return merge_candidates(
        existing,
        suggestions,
        pol,
        is_accepted=_rec_accepted,
        is_human=_rec_human,
        bbox_of=_rec_bbox,
        make_obj=_make,
        obb_of=_rec_obb,
    )


# --------------------------------------------------------------------------- #
# 8. The review state machine
# --------------------------------------------------------------------------- #

PENDING = "pending"
ACCEPTED = "accepted"
REJECTED = "rejected"


@dataclass(frozen=True)
class ReviewCounts:
    pending: int = 0
    accepted: int = 0
    rejected: int = 0

    @property
    def total(self) -> int:
        return self.pending + self.accepted + self.rejected

    @property
    def reviewed(self) -> int:
        return self.accepted + self.rejected


class ReviewSession:
    """Accept / reject / edit over one run's candidates.

    The counters back the progress readout, :meth:`next_pending_frame` backs the
    ``Ctrl+Right`` jump that is what makes 400 candidates reviewable, and
    :meth:`accepted_candidates` is the only thing that ever reaches the
    annotator's saved objects.
    """

    def __init__(self, candidates: Optional[Mapping[int, Sequence[Candidate]]] = None):
        self._frames: Dict[int, List[Candidate]] = {}
        self._states: Dict[int, List[str]] = {}
        self._edited: Dict[int, List[bool]] = {}
        if candidates:
            self.set_candidates(candidates)

    # -- population ------------------------------------------------------
    def set_candidates(self, candidates: Mapping[int, Sequence[Candidate]]) -> None:
        """Replace everything. A re-run supersedes; it does not stack."""
        self._frames = {}
        self._states = {}
        self._edited = {}
        for fi, items in candidates.items():
            self.add_frame(int(fi), items)

    def add_frame(self, frame: int, candidates: Sequence[Candidate]) -> None:
        items = [c.copy() for c in candidates]
        if not items:
            return
        fi = int(frame)
        self._frames.setdefault(fi, []).extend(items)
        self._states.setdefault(fi, []).extend([PENDING] * len(items))
        self._edited.setdefault(fi, []).extend([False] * len(items))

    # -- reading ---------------------------------------------------------
    @property
    def frames(self) -> List[int]:
        return sorted(self._frames.keys())

    def candidates(self, frame: int) -> List[Candidate]:
        return list(self._frames.get(int(frame), []))

    def state(self, frame: int, index: int) -> str:
        states = self._states.get(int(frame), [])
        return states[index] if 0 <= index < len(states) else ""

    def is_edited(self, frame: int, index: int) -> bool:
        flags = self._edited.get(int(frame), [])
        return bool(flags[index]) if 0 <= index < len(flags) else False

    def counts(self, frame: Optional[int] = None) -> ReviewCounts:
        if frame is None:
            groups: Iterable[List[str]] = list(self._states.values())
        else:
            groups = [self._states.get(int(frame), [])]
        p = a = r = 0
        for states in groups:
            for s in states:
                if s == ACCEPTED:
                    a += 1
                elif s == REJECTED:
                    r += 1
                else:
                    p += 1
        return ReviewCounts(pending=p, accepted=a, rejected=r)

    def pending_frames(self) -> List[int]:
        return [fi for fi in self.frames if any(s == PENDING for s in self._states.get(fi, []))]

    def next_pending_frame(self, after: int) -> Optional[int]:
        """The next frame that still holds candidates, or ``None``."""
        for fi in self.pending_frames():
            if fi > int(after):
                return fi
        return None

    def prev_pending_frame(self, before: int) -> Optional[int]:
        for fi in reversed(self.pending_frames()):
            if fi < int(before):
                return fi
        return None

    # -- single-candidate operations -------------------------------------
    def _set_state(self, frame: int, index: int, state: str) -> bool:
        states = self._states.get(int(frame))
        if states is None or not (0 <= index < len(states)):
            return False
        states[index] = state
        return True

    def accept(self, frame: int, index: int) -> bool:
        return self._set_state(frame, index, ACCEPTED)

    def reject(self, frame: int, index: int) -> bool:
        return self._set_state(frame, index, REJECTED)

    def reset(self, frame: int, index: int) -> bool:
        return self._set_state(frame, index, PENDING)

    def edit(
        self,
        frame: int,
        index: int,
        *,
        bbox: Optional[Sequence[float]] = None,
        obb: Optional[Sequence[float]] = None,
        track_id: Optional[int] = None,
        action_id: Optional[int] = None,
        class_id: Optional[int] = None,
        accept: bool = False,
    ) -> Optional[Candidate]:
        """Correct a candidate in place. Returns the updated candidate, or None.

        Editing does not resolve it on its own — a corrected box is still a
        candidate until it is accepted — unless ``accept=True``.
        """
        items = self._frames.get(int(frame))
        if items is None or not (0 <= index < len(items)):
            return None
        cand = items[index]
        if bbox is not None:
            cand.bbox = _as_int_box(bbox)
        if obb is not None:
            cand.obb = [float(v) for v in obb[:5]]
            cand.bbox = _as_int_box(obb_to_xyxy(cand.obb))
        if track_id is not None:
            cand.track_id = int(track_id)
        if action_id is not None:
            cand.action_id = int(action_id)
        if class_id is not None:
            cand.class_id = int(class_id)
        self._edited[int(frame)][index] = True
        if accept:
            self.accept(frame, index)
        return cand

    # -- batch operations ------------------------------------------------
    def _bulk(self, frames: Iterable[int], state: str) -> int:
        n = 0
        for fi in frames:
            states = self._states.get(int(fi))
            if not states:
                continue
            for i, s in enumerate(states):
                if s == PENDING:
                    states[i] = state
                    n += 1
        return n

    def accept_frame(self, frame: int) -> int:
        """Accept every candidate in this frame. Returns how many changed."""
        return self._bulk([frame], ACCEPTED)

    def reject_frame(self, frame: int) -> int:
        """Discard every candidate in this frame. Returns how many changed."""
        return self._bulk([frame], REJECTED)

    def accept_all(self) -> int:
        return self._bulk(self.frames, ACCEPTED)

    def reject_all(self) -> int:
        return self._bulk(self.frames, REJECTED)

    def accept_above(self, threshold: float) -> int:
        """Accept every pending candidate whose score reaches ``threshold``."""
        n = 0
        for fi in self.frames:
            items = self._frames[fi]
            states = self._states[fi]
            for i, cand in enumerate(items):
                if states[i] == PENDING and float(cand.score) >= float(threshold):
                    states[i] = ACCEPTED
                    n += 1
        return n

    # -- output ----------------------------------------------------------
    def accepted_candidates(self) -> Dict[int, List[Candidate]]:
        """Only the accepted ones — the single thing allowed to reach disk."""
        out: Dict[int, List[Candidate]] = {}
        for fi in self.frames:
            keep = [c for c, s in zip(self._frames[fi], self._states[fi]) if s == ACCEPTED]
            if keep:
                out[fi] = [c.copy() for c in keep]
        return out

    def pending_candidates(self) -> Dict[int, List[Candidate]]:
        out: Dict[int, List[Candidate]] = {}
        for fi in self.frames:
            keep = [c for c, s in zip(self._frames[fi], self._states[fi]) if s == PENDING]
            if keep:
                out[fi] = [c.copy() for c in keep]
        return out

    def to_annotations(self, *, start_object_id: int = 1) -> Dict[int, List[Dict[str, Any]]]:
        """Accepted candidates as annotator object records."""
        return predictions_to_annotations(
            self.accepted_candidates(), start_object_id=start_object_id, accepted=True
        )

    def progress_text(self) -> str:
        """English readout for the review pill."""
        c = self.counts()
        if c.total == 0:
            return "No candidates"
        if c.pending == 0:
            return (
                f"All {c.total} candidates reviewed — {c.accepted} accepted, "
                f"{c.rejected} discarded."
            )
        return (
            f"{c.pending} of {c.total} candidates still to review "
            f"({c.accepted} accepted, {c.rejected} discarded)."
        )


# --------------------------------------------------------------------------- #
# 9. The torch-free tracking fallback
# --------------------------------------------------------------------------- #


def _cv2():
    """Import OpenCV lazily. Raises ``RuntimeError`` with an English message."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "Tracking needs OpenCV, which is not installed in this Python "
            "environment. Install it with:  pip install opencv-python"
        ) from exc
    return cv2


def tracker_backend() -> str:
    """Which OpenCV tracker this machine actually has.

    Probe chain, in order::

        cv2.TrackerCSRT_create         -> "csrt"
        cv2.legacy.TrackerCSRT_create  -> "csrt_legacy"
        cv2.TrackerMIL_create          -> "mil"
        (none of the above)            -> "lk"     pure Lucas-Kanade

    MEASURED on this machine (opencv 4.13.0): CSRT does not exist and there is
    no ``cv2.legacy``, so this returns ``"mil"``. ``annotator.py`` used to call
    ``cv2.TrackerCSRT_create()`` unguarded and raised ``AttributeError`` the
    moment auto-track stepped a frame.

    GOTURN / Nano / Vit / DaSiamRPN are **deliberately excluded**: they require
    downloaded ONNX weights and fail at init with an opaque error.
    """
    try:
        cv2 = _cv2()
    except RuntimeError:
        return "none"
    if hasattr(cv2, "TrackerCSRT_create"):
        return "csrt"
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        return "csrt_legacy"
    if hasattr(cv2, "TrackerMIL_create"):
        return "mil"
    return "lk"


def make_tracker() -> Tuple[Optional[Callable[[], Any]], str]:
    """``(factory, backend_name)``. ``factory`` is None when the backend is 'lk'."""
    backend = tracker_backend()
    if backend in ("lk", "none"):
        return None, backend
    cv2 = _cv2()
    if backend == "csrt":
        return cv2.TrackerCSRT_create, backend
    if backend == "csrt_legacy":
        return cv2.legacy.TrackerCSRT_create, backend
    return cv2.TrackerMIL_create, backend


def _to_gray(frame):
    cv2 = _cv2()
    if getattr(frame, "ndim", 2) == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def lk_propagate(prev_gray, next_gray, box: Sequence[int]) -> Optional[List[int]]:
    """Move one box with sparse optical flow.

    ``goodFeaturesToTrack`` inside the box -> ``calcOpticalFlowPyrLK`` -> median
    ``(dx, dy)``, plus a median scale change from pairwise point distances.

    Returns ``None`` (never raises) when fewer than 4 good points survive — a
    tracker that cannot follow is a normal outcome, not a crash.
    """
    try:
        cv2 = _cv2()
        import numpy as np
    except (RuntimeError, ImportError):
        return None

    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        h, w = prev_gray.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        roi = prev_gray[y1:y2, x1:x2]
        if roi.size < 16:
            return None
        pts = cv2.goodFeaturesToTrack(
            roi, maxCorners=64, qualityLevel=0.01, minDistance=3, blockSize=3
        )
        if pts is None or len(pts) < 4:
            return None
        p0 = pts.reshape(-1, 2).astype(np.float32)
        p0[:, 0] += x1
        p0[:, 1] += y1
        p0 = p0.reshape(-1, 1, 2)
        p1, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            next_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None or status is None:
            return None
        ok = status.reshape(-1).astype(bool)
        a = p0.reshape(-1, 2)[ok]
        b = p1.reshape(-1, 2)[ok]
        if len(a) < 4:
            return None

        dx = float(np.median(b[:, 0] - a[:, 0]))
        dy = float(np.median(b[:, 1] - a[:, 1]))

        # Median pairwise distance ratio = scale change, clamped so one bad
        # match cannot blow the box up.
        scale = 1.0
        n = min(len(a), 24)
        da = a[:n][:, None, :] - a[:n][None, :, :]
        db = b[:n][:, None, :] - b[:n][None, :, :]
        d0 = np.sqrt((da**2).sum(-1))
        d1 = np.sqrt((db**2).sum(-1))
        mask = d0 > 2.0
        if bool(mask.any()):
            scale = float(np.median(d1[mask] / d0[mask]))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        scale = max(0.85, min(1.18, scale))

        cx = (x1 + x2) / 2.0 + dx
        cy = (y1 + y2) / 2.0 + dy
        bw = (x2 - x1) * scale
        bh = (y2 - y1) * scale
        return [
            int(round(cx - bw / 2)),
            int(round(cy - bh / 2)),
            int(round(cx + bw / 2)),
            int(round(cy + bh / 2)),
        ]
    except Exception:
        return None


def _clamp_box(box: Sequence[float], frame_w: int, frame_h: int) -> List[int]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(x1, float(frame_w)))
    x2 = max(0.0, min(x2, float(frame_w)))
    y1 = max(0.0, min(y1, float(frame_h)))
    y2 = max(0.0, min(y2, float(frame_h)))
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def track_forward(
    read_frame: Callable[[int], Any],
    start_idx: int,
    box: Sequence[int],
    n: int,
    frame_w: int,
    frame_h: int,
    *,
    min_side: int = 5,
    should_stop: Optional[Callable[[], bool]] = None,
    backend: str = "",
) -> Tuple[List[Tuple[int, List[int]]], str]:
    """Propagate one box forward up to ``n`` frames.

    Returns ``([(frame_index, [x1, y1, x2, y2]), ...], stop_reason)``.
    ``stop_reason`` is one of ``""`` (completed), ``"lost"``, ``"too_small"``,
    ``"out_of_frame"``, ``"stopped"``, ``"end_of_video"``.

    Every returned box is clamped into ``[0, frame_w] x [0, frame_h]``.
    ``read_frame(i)`` must return a BGR ndarray or ``None``; the caller supplies
    it so this module never opens a ``VideoCapture``. ``should_stop`` is polled
    each frame so a UI Stop button works.

    Pass ``backend="lk"`` to force the optical-flow path.

    Performance note for consumers: MIL measures ~47 ms/frame here, so 30 frames
    x 4 tracks is ~6 s. Run this on a worker thread — it touches only plain
    lists and the caller's ``read_frame`` — and apply the results on the main
    thread.
    """
    out: List[Tuple[int, List[int]]] = []
    n = int(n)
    if n <= 0:
        return out, ""

    first = read_frame(int(start_idx))
    if first is None:
        return out, "end_of_video"

    if backend in ("lk", "none"):
        factory: Optional[Callable[[], Any]] = None
        chosen = backend
    else:
        factory, chosen = make_tracker()
    if chosen == "none":
        return out, "lost"

    cur = _clamp_box(box, frame_w, frame_h)
    tracker = None
    if factory is not None:
        try:
            tracker = factory()
            x1, y1, x2, y2 = cur
            tracker.init(first, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        except Exception:
            tracker = None
            factory = None

    prev_gray = _to_gray(first) if tracker is None else None

    for step in range(1, n + 1):
        if should_stop is not None and should_stop():
            return out, "stopped"
        idx = int(start_idx) + step
        frame = read_frame(idx)
        if frame is None:
            return out, "end_of_video"

        if tracker is not None:
            try:
                ok, rect = tracker.update(frame)
            except Exception:
                return out, "lost"
            if not ok:
                return out, "lost"
            x, y, w, h = (float(v) for v in rect)
            nxt: Sequence[float] = [x, y, x + w, y + h]
        else:
            gray = _to_gray(frame)
            moved = lk_propagate(prev_gray, gray, cur)
            prev_gray = gray
            if moved is None:
                return out, "lost"
            nxt = moved

        clamped = _clamp_box(nxt, frame_w, frame_h)
        if clamped[2] - clamped[0] <= 0 or clamped[3] - clamped[1] <= 0:
            return out, "out_of_frame"
        if (clamped[2] - clamped[0]) < min_side or (clamped[3] - clamped[1]) < min_side:
            return out, "too_small"
        out.append((idx, clamped))
        cur = clamped

    return out, ""
