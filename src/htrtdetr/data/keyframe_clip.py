"""
keyframe_clip.py — keyframe labels in, clips of video out (YOWO-style training).

A keyframe dataset (``docs/KEYFRAME_BEHAVIOR.md``) labels single frames: every
animal's box and behavior, no track ids. This module turns each keyframe into
a training sample:

* **clip mode** (Stage 2 / 4): the ``length`` frames ``stride`` apart that end
  at the keyframe, ``(T, 3, H, W)``. Only the keyframe carries targets; the
  context frames are unlabelled video, and the model's own detector and tracker
  turn them into per-animal history.
* **single-frame mode** (Stage 1): just the keyframe, for the detector.

Two things this does that the legacy datasets do not, each because its absence
fails silently:

1. **The clip is taken by frame position in the video**, through
   :class:`htrtdetr.annotation.keyframes.ClipSpec`. ``SlidingWindowDataset``
   windows over the list of *annotated* frames, so sparse labels made it learn
   frames seconds apart as consecutive.
2. **Rotated boxes reach the loss.** ``SingleFrameDataset`` reads only ``bbox``,
   so the OBB angle head was never trained from annotations; here every target
   carries ``obb`` and ``obb_valid`` (False for boxes with no true rotation, so
   the angle loss skips them instead of learning "everything is horizontal").

Frames come from the image tree a round exports (``images/<video>/<index>.jpg``
beside the annotation file) or, when an image is not there, straight from the
video through :func:`htrtdetr.gui.video_source.read_frame_sequential` - the
function the annotator's canvas decodes with, so the model trains on the frame
the human labelled. A keyframe whose pixels cannot be found anywhere is dropped
at construction and counted, never replaced by a black frame.
"""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..annotation import behaviors
from ..annotation.keyframes import ClipSpec, clip_spec_of
from .annotation import DatasetAnno, FrameAnno, VideoAnno, load_annotation

__all__ = [
    "KeyframeClipDataset",
    "collate_keyframe_clips",
    "collate_keyframe_single",
    "FRAME_SOURCES",
    "context_image_name",
    "transform_obb_for_flip",
    "resize_obb",
]

logger = logging.getLogger(__name__)

FRAME_SOURCES = ("auto", "images", "video")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def context_image_name(video_id: str, index: int) -> str:
    """Where a round writes frame ``index`` of ``video_id``, relative to its dataset."""
    return f"images/{video_id}/{int(index):06d}.jpg"


# ══════════════════════════════════════════════════════════════════════════
# Geometry of the augmentations, in pixels
# ══════════════════════════════════════════════════════════════════════════

def _wrap_theta(theta: np.ndarray) -> np.ndarray:
    """Fold into [-pi/2, pi/2): a rectangle repeats every pi."""
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def transform_obb_for_flip(obb: np.ndarray, width: float, height: float,
                           *, hflip: bool, vflip: bool) -> np.ndarray:
    """Mirror ``(N, 5)`` pixel OBBs ``[cx, cy, w, h, theta]`` with the image.

    A mirror reverses the sense of rotation, so theta changes sign under either
    flip; under both it is a 180 degree rotation, which a rectangle does not see.
    """
    out = np.array(obb, dtype=np.float32, copy=True)
    if out.size == 0:
        return out.reshape(0, 5)
    if hflip:
        out[:, 0] = width - out[:, 0]
        out[:, 4] = -out[:, 4]
    if vflip:
        out[:, 1] = height - out[:, 1]
        out[:, 4] = -out[:, 4]
    out[:, 4] = _wrap_theta(out[:, 4])
    return out


def _corners(obb: np.ndarray) -> np.ndarray:
    """``(N, 5)`` -> ``(N, 4, 2)``, the corner order of :func:`geometry.obb.obb_to_corners`."""
    cx, cy, w, h, th = (obb[:, i] for i in range(5))
    hw, hh = w * 0.5, h * 0.5
    lx = np.stack([-hw, hw, hw, -hw], axis=-1)
    ly = np.stack([-hh, -hh, hh, hh], axis=-1)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    gx = cx[:, None] + lx * c - ly * s
    gy = cy[:, None] + lx * s + ly * c
    return np.stack([gx, gy], axis=-1)


def resize_obb(obb: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Scale ``(N, 5)`` pixel OBBs by ``(sx, sy)``, which need not be equal.

    A non-uniform scale turns a rectangle into a parallelogram, so the box is
    refitted from its scaled corners: the first edge keeps its direction (it is
    the width axis, as in :func:`geometry.obb.corners_to_obb`) and the height is
    chosen so the area is preserved. With ``sx == sy`` this is exact.
    """
    obb = np.asarray(obb, dtype=np.float32)
    if obb.size == 0:
        return obb.reshape(0, 5)
    corners = _corners(obb)
    corners[..., 0] *= sx
    corners[..., 1] *= sy
    center = corners.mean(axis=1)
    e1 = corners[:, 1] - corners[:, 0]
    e2 = corners[:, 3] - corners[:, 0]
    w = np.linalg.norm(e1, axis=-1)
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    h = np.where(w > 1e-9, area / np.maximum(w, 1e-9), np.linalg.norm(e2, axis=-1))
    theta = _wrap_theta(np.arctan2(e1[:, 1], e1[:, 0]))
    return np.stack([center[:, 0], center[:, 1], w, h, theta], axis=-1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# The dataset
# ══════════════════════════════════════════════════════════════════════════

class KeyframeClipDataset(Dataset):
    """One sample per keyframe: its clip (or the keyframe alone).

    Args:
        dataset: a loaded :class:`DatasetAnno`, or a path to an annotation file.
        clip: which frames make the clip. ``None`` reads ``meta.labeling.clip``.
        image_size: ``(H, W)`` the frames are resized to.
        augment: flips and colour jitter, **identical for every frame of a
            clip** - jitter that changes from frame to frame is fake motion.
        data_root: folder that ``image_path`` / ``video_path`` are relative to.
            Defaults to the annotation file's folder when a path is given.
        frame_source: ``images`` (exported tree only), ``video`` (decode only),
            or ``auto`` (the image if it is there, else the video).
        single_frame: return Stage 1 samples (one image) instead of clips.
        require_behavior: drop keyframes with no behavior label at all.

    Multi-head behaviors (``meta.labeling.behaviors = "multi_head"``,
    ``docs/KEYFRAME_BEHAVIOR.md`` §15) add ``head_action_ids`` to the targets:
    ``(N, groups)``, each answer as a position inside its group, ``-1`` for a
    group not judged. ``action_ids`` stays the first group's flat answer.
    """

    def __init__(
        self,
        dataset: Any,
        clip: Optional[ClipSpec] = None,
        image_size: Tuple[int, int] = (640, 640),
        augment: bool = True,
        data_root: str = "",
        frame_source: str = "auto",
        single_frame: bool = False,
        require_behavior: bool = False,
        hflip: bool = True,
        vflip: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
    ) -> None:
        if isinstance(dataset, (str, Path)):
            path = Path(dataset)
            if not data_root:
                data_root = str(path.parent)
            dataset = load_annotation(str(path))
        if not isinstance(dataset, DatasetAnno):
            raise TypeError(f"expected a DatasetAnno or a path, got {type(dataset).__name__}")
        if frame_source not in FRAME_SOURCES:
            raise ValueError(f"frame_source must be one of {FRAME_SOURCES}, got {frame_source!r}")

        self.anno = dataset
        #: One behavior list, or groups. A malformed multi-head declaration
        #: raises here, before a round spends minutes on it.
        self.vocab = behaviors.vocabulary_of(dataset)
        self.clip = clip if clip is not None else clip_spec_of(dataset)
        if self.clip.anchor != "end" and not single_frame:
            # The model predicts for the last frame of its window, so a
            # centred clip would put the label on the wrong frame.
            raise ValueError("training needs clip anchor 'end' (the keyframe last)")
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.augment = bool(augment)
        self.data_root = Path(data_root) if data_root else Path(".")
        self.frame_source = frame_source
        self.single_frame = bool(single_frame)
        self.hflip, self.vflip = bool(hflip), bool(vflip)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.saturation = float(saturation)

        #: How often a context frame had to come from somewhere other than
        #: where it was asked for. Logged once by :meth:`report`.
        self.stats: Dict[str, int] = {"video_reads": 0, "substituted": 0}
        self._captures: Dict[str, List[Any]] = {}

        self.samples: List[Tuple[VideoAnno, FrameAnno]] = []
        self.dropped: List[Tuple[str, int]] = []
        for video in dataset.videos:
            for frame in video.frames:
                if require_behavior and not any(
                    behaviors.has_behavior(o, self.vocab) for o in frame.objects
                ):
                    continue
                if self._reachable(video, frame):
                    self.samples.append((video, frame))
                else:
                    self.dropped.append((video.video_id, int(frame.frame_index)))
        if self.dropped:
            sample = ", ".join(f"{v}:{f}" for v, f in self.dropped[:3])
            logger.warning(
                "%d keyframe(s) have no image and no readable video and were left out "
                "(for example %s).", len(self.dropped), sample,
            )
        if not self.samples and self.dropped:
            raise FileNotFoundError(
                f"None of the {len(self.dropped)} keyframes could be read: no exported "
                f"image under {self.data_root} and no video file. Export the frames or "
                "fix the video paths."
            )

    # ── pickling: captures do not cross into DataLoader workers ──────────
    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

    def __del__(self) -> None:  # pragma: no cover - best effort
        for entry in getattr(self, "_captures", {}).values():
            try:
                entry[0].release()
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.samples)

    # ── where pixels come from ───────────────────────────────────────────
    def _image_file(self, video: VideoAnno, index: int, frame: Optional[FrameAnno] = None) -> Path:
        if frame is not None and frame.image_path:
            return self.data_root / frame.image_path
        return self.data_root / context_image_name(video.video_id, index)

    def _video_file(self, video: VideoAnno) -> Optional[Path]:
        if not video.video_path:
            return None
        path = Path(video.video_path)
        if not path.is_absolute():
            path = self.data_root / path
        return path

    def _reachable(self, video: VideoAnno, frame: FrameAnno) -> bool:
        if self.frame_source in ("auto", "images"):
            if self._image_file(video, frame.frame_index, frame).is_file():
                return True
            if self.frame_source == "images":
                return False
        video_file = self._video_file(video)
        return video_file is not None and video_file.is_file()

    def _read_image(self, path: Path) -> Optional[np.ndarray]:
        """RGB uint8, or None. Bytes through numpy so non-ASCII paths work on Windows."""
        try:
            import cv2

            buffer = np.fromfile(str(path), dtype=np.uint8)
            if buffer.size == 0:
                return None
            bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except (OSError, ImportError):
            try:
                from PIL import Image

                return np.array(Image.open(path).convert("RGB"))
            except Exception:
                return None

    def _read_video_frames(self, video: VideoAnno, indices: Sequence[int]) -> Dict[int, np.ndarray]:
        """Decode ``indices`` (any order) from the video, sequentially where possible."""
        path = self._video_file(video)
        if path is None or not path.is_file():
            return {}
        from ..gui import video_source

        key = str(path)
        entry = self._captures.get(key)
        if entry is None:
            cap = video_source.open_capture(path)
            if cap is None:
                return {}
            entry = [cap, None]
            self._captures[key] = entry
        out: Dict[int, np.ndarray] = {}
        import cv2

        for index in sorted(set(int(i) for i in indices)):
            frame, entry[1] = video_source.read_frame_sequential(entry[0], index, entry[1])
            if frame is not None:
                out[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.stats["video_reads"] += 1
        return out

    def _load_frames(self, video: VideoAnno, key: FrameAnno,
                     indices: Sequence[int]) -> List[np.ndarray]:
        """The pixels for ``indices``; the keyframe's own image for its index."""
        images: Dict[int, np.ndarray] = {}
        wanted = list(dict.fromkeys(int(i) for i in indices))
        if self.frame_source in ("auto", "images"):
            for index in wanted:
                frame = key if index == key.frame_index else None
                image = self._read_image(self._image_file(video, index, frame))
                if image is not None:
                    images[index] = image
        missing = [i for i in wanted if i not in images]
        if missing and self.frame_source in ("auto", "video"):
            images.update(self._read_video_frames(video, missing))
        if key.frame_index not in images:
            raise FileNotFoundError(
                f"keyframe {video.video_id}:{key.frame_index} could not be read"
            )
        out: List[np.ndarray] = []
        for index in indices:
            image = images.get(int(index))
            if image is None:
                # A context frame that exists nowhere: repeat the nearest one we
                # have. Zero motion is the honest stand-in, and it is counted.
                nearest = min(images, key=lambda i: abs(i - int(index)))
                image = images[nearest]
                self.stats["substituted"] += 1
            out.append(image)
        return out

    def report(self) -> str:
        """One line for the training log."""
        return (
            f"{len(self.samples)} keyframes ({len(self.dropped)} unreadable, left out); "
            f"clip {self.clip.length}x{self.clip.stride}; "
            f"{self.stats['video_reads']} frames decoded from video, "
            f"{self.stats['substituted']} context frames substituted"
        )

    # ── one sample ───────────────────────────────────────────────────────
    def _targets(self, frame: FrameAnno, width: int, height: int) -> Dict[str, np.ndarray]:
        objects = list(frame.objects)
        n = len(objects)
        boxes = np.array([o.bbox for o in objects], dtype=np.float32).reshape(n, 4)
        obb = np.array([o.get_obb() for o in objects], dtype=np.float32).reshape(n, 5)
        out = {
            "boxes": boxes,
            "obb": obb,
            "obb_valid": np.array([o.has_obb for o in objects], dtype=bool),
            "class_ids": np.array([o.class_id for o in objects], dtype=np.int64),
            "action_ids": np.array([o.action_id for o in objects], dtype=np.int64),
            "is_crowd": np.array([o.is_crowd for o in objects], dtype=bool),
        }
        if self.vocab.multi:
            width = len(self.vocab.groups)
            out["head_action_ids"] = np.array(
                [behaviors.group_targets(o, self.vocab) for o in objects], dtype=np.int64
            ).reshape(n, width)
            # The flat number of the first group's answer, whatever the file
            # stored in action_id: one source of truth for the answer.
            out["action_ids"] = np.array(
                [behaviors.labels_of(o, self.vocab)[0] for o in objects], dtype=np.int64
            )
        return out

    def _jitter(self) -> Tuple[float, float, float]:
        def draw(amount: float) -> float:
            if amount <= 0 or random.random() >= 0.5:
                return 1.0
            return 1.0 + random.uniform(-amount, amount)
        return draw(self.brightness), draw(self.contrast), draw(self.saturation)

    @staticmethod
    def _apply_jitter(image: np.ndarray, factors: Tuple[float, float, float]) -> np.ndarray:
        b, c, s = factors
        if b == 1.0 and c == 1.0 and s == 1.0:
            return image
        x = image.astype(np.float32) * b
        if c != 1.0:
            mean = x.mean()
            x = (x - mean) * c + mean
        if s != 1.0:
            gray = x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            x = gray[..., None] + (x - gray[..., None]) * s
        return np.clip(x, 0, 255).astype(np.uint8)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        import cv2

        h, w = self.image_size
        if image.shape[0] != h or image.shape[1] != w:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        x = (image.astype(np.float32) / 255.0 - _MEAN) / _STD
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        video, key = self.samples[idx]
        if self.single_frame:
            indices: Tuple[int, ...] = (int(key.frame_index),)
            valid: Tuple[bool, ...] = (True,)
        else:
            indices, valid = self.clip.indices(int(key.frame_index), int(video.num_frames or 0))
        frames = self._load_frames(video, key, indices)
        h0, w0 = frames[-1].shape[:2]
        width = float(video.width or w0)
        height = float(video.height or h0)
        if (w0, h0) != (int(width), int(height)):
            # The decoded frame wins over the metadata: the boxes were drawn on it.
            width, height = float(w0), float(h0)
        targets = self._targets(key, int(width), int(height))

        hflip = self.augment and self.hflip and random.random() < 0.5
        vflip = self.augment and self.vflip and random.random() < 0.5
        jitter = self._jitter() if self.augment else (1.0, 1.0, 1.0)

        boxes = targets["boxes"].copy()
        obb = targets["obb"].copy()
        if hflip:
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        if vflip:
            boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
        if hflip or vflip:
            obb = transform_obb_for_flip(obb, width, height, hflip=hflip, vflip=vflip)

        tensors = []
        for image in frames:
            if hflip:
                image = image[:, ::-1]
            if vflip:
                image = image[::-1]
            image = self._apply_jitter(np.ascontiguousarray(image), jitter)
            tensors.append(self._to_tensor(image))

        H, W = self.image_size
        sx, sy = W / width, H / height
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        obb = resize_obb(obb, sx, sy)
        boxes_norm = boxes / np.array([W, H, W, H], dtype=np.float32)
        obb_norm = obb.copy()
        if len(obb_norm):
            obb_norm[:, [0, 2]] /= W
            obb_norm[:, [1, 3]] /= H

        n = len(boxes_norm)
        key_target = {
            "boxes": torch.from_numpy(boxes_norm.reshape(n, 4)),
            "obb": torch.from_numpy(obb_norm.reshape(n, 5)),
            "obb_valid": torch.from_numpy(targets["obb_valid"]),
            "class_ids": torch.from_numpy(targets["class_ids"]),
            "action_ids": torch.from_numpy(targets["action_ids"]),
            # Not labels in this scheme. Present so the shared collate and
            # loss code find the keys they always have; -1 is "no identity".
            "track_ids": torch.full((n,), -1, dtype=torch.long),
            "identity_ids": torch.full((n,), -1, dtype=torch.long),
        }
        if "head_action_ids" in targets:
            key_target["head_action_ids"] = torch.from_numpy(targets["head_action_ids"])

        if self.single_frame:
            return {
                "image": tensors[0],
                **key_target,
                "frame_index": int(key.frame_index),
                "image_path": key.image_path,
                "orig_size": (int(height), int(width)),
                "video_id": video.video_id,
            }

        empty = {
            "boxes": torch.zeros(0, 4), "obb": torch.zeros(0, 5),
            "obb_valid": torch.zeros(0, dtype=torch.bool),
            "class_ids": torch.zeros(0, dtype=torch.long),
            "action_ids": torch.zeros(0, dtype=torch.long),
            "track_ids": torch.zeros(0, dtype=torch.long),
            "identity_ids": torch.zeros(0, dtype=torch.long),
        }
        if "head_action_ids" in key_target:
            empty["head_action_ids"] = torch.zeros(
                (0, len(self.vocab.groups)), dtype=torch.long
            )
        T = len(tensors)
        per_frame = [empty] * (T - 1) + [key_target]
        sample: Dict[str, Any] = {"images": torch.stack(tensors, dim=0)}
        for name in empty:
            sample[name] = [frame_target[name] for frame_target in per_frame]
        sample.update({
            "frame_indices": [int(i) for i in indices],
            "clip_valid": [bool(v) for v in valid],
            "video_id": video.video_id,
            "window_start": int(indices[0]),
            "keyframe": int(key.frame_index),
        })
        return sample


# ══════════════════════════════════════════════════════════════════════════
# Collate
# ══════════════════════════════════════════════════════════════════════════

_TARGET_KEYS = ("boxes", "obb", "obb_valid", "class_ids", "action_ids",
                "track_ids", "identity_ids")
#: Present only for multi-head behaviors; carried when the sample has it.
_OPTIONAL_TARGET_KEYS = ("head_action_ids",)


def _target_keys(item: Dict[str, Any]) -> Tuple[str, ...]:
    return _TARGET_KEYS + tuple(k for k in _OPTIONAL_TARGET_KEYS if k in item)


def collate_keyframe_clips(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Clips to ``images (B, T, 3, H, W)`` and ``targets`` B x T.

    The same shape :func:`htrtdetr.data.collate.collate_sequence` produces, so
    the Trainer and the loss take it unchanged: ``last_frame_targets`` picks
    the keyframe, and every other frame's target is empty. Unlike that collate
    it keeps ``obb`` / ``obb_valid``, which the detection loss reads.
    """
    images = torch.stack([item["images"] for item in batch], dim=0)
    targets: List[List[Dict[str, Any]]] = []
    meta: List[Dict[str, Any]] = []
    for item in batch:
        T = len(item["boxes"])
        keys = _target_keys(item)
        targets.append([{k: item[k][t] for k in keys} for t in range(T)])
        meta.append({
            "video_id": item["video_id"],
            "window_start": item["window_start"],
            "frame_indices": item["frame_indices"],
            "clip_valid": item["clip_valid"],
            "keyframe": item["keyframe"],
        })
    return {"images": images, "targets": targets, "meta": meta}


def collate_keyframe_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Single keyframes to ``images (B, 3, H, W)`` and ``targets`` of length B."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [{k: item[k] for k in _target_keys(item)} for item in batch]
    meta = [
        {"frame_index": item["frame_index"], "image_path": item["image_path"],
         "orig_size": item["orig_size"], "video_id": item["video_id"]}
        for item in batch
    ]
    return {"images": images, "targets": targets, "meta": meta}
