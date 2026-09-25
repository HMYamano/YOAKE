"""detect_only.py — boxes and nothing else, several frames at a time.

The active-learning loop asks the model one question: *where are the animals
in these frames*. It does not ask who they are or what they are doing — a
round trains ``stage=1``, so the identity, temporal and behavior heads are
still at their random initialisation and everything they output is noise the
annotator discards. :mod:`~htrtdetr.inference.streaming` runs all of them
anyway, because it is built for the finished four-stage model, and it runs
them one frame at a time because it has to: the tracker is sequential.

Drop those two constraints and the same detector goes four to five times
faster on the same GPU. Measured here (RTX 5070 Ti, fp16 autocast,
``channels_last``, ResNet-18 at 640x640, detector only):

    batch   1     73.7 fps        batch   8    289.7 fps
    batch   2    127.0 fps        batch  16    350.6 fps
    batch   4    209.7 fps        batch  32    366.8 fps

Nothing about the model changed between those rows. At batch 1 the detector
is not compute-bound at all — swapping ResNet-18 for ResNet-50 costs nothing
measurable (15.0 ms against 15.8 ms) and dropping the input to 384x384 costs
nothing either, because the time goes on launching kernels rather than on
running them. Batching is the whole of the difference, and it is free: the
detector is per-frame, so a batch of sixteen frames finds the same animals as
sixteen batches of one. Measured against the streaming path on the same
weights, with the score threshold at zero so all 100 queries come through:
**0.35 px** worst-case box difference and 5e-5 worst-case score difference
over 120 frames x 100 boxes. Not bit-identical, and it cannot be - fp16
reductions run in a different order at a different batch size - but a third of
a pixel on a box a human is about to accept or reject is not a difference.

End to end through ``yoake predict`` on 120 frames of one animal with the
threshold at zero: streaming 6.8 fps, detect-only 160.8 fps. With no
detections at all - the streaming path's best case, where tracking, temporal
and behavior all return immediately - it is still 36.3 against 308.0.

**What this does not produce.** ``track_ids`` are all ``-1`` and
``action_ids`` are all ``-1``. There is no tracker here, so identity is left
to the caller: :func:`htrtdetr.annotation.batch.connect_detections` links
detections by IoU for dense runs, and a strided run deliberately has no
identity to give (detections 15 frames apart are not neighbours, and the
online tracker's habit of joining them anyway is a defect this path does not
inherit).

**No annotated video either.** The streaming path encodes an overlay mp4 as
it goes. For a candidate-generation pass over twenty hour-long recordings at
stride 15 that is 144,000 frames drawn and encoded for a file nobody opens.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..geometry.obb import cxcywh_to_obb
from ..utils.logging import get_logger
from ..utils.misc import cxcywh_to_xyxy, get_device

__all__ = ["DEFAULT_BATCH", "BatchedDetector", "run_detect_only"]

#: Frames per forward pass.
#:
#: Sixteen rather than thirty-two: the measured curve is steep to 8, nearly
#: flat from 16 (350 fps) to 32 (367 fps), and peak GPU memory doubles across
#: that last step (1.3 GB to 2.5 GB). The annotator is running in the same
#: process tree and often on the same card as the display, so the last 5% of
#: throughput is not worth a gigabyte.
DEFAULT_BATCH = 16

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BatchedDetector:
    """The detector of an :class:`HTRTDETR`, run over batches of frames.

    Holds no state between calls beyond the model itself, which is the point:
    the thing that forced one-frame-at-a-time was the tracker's sequential
    memory, and there is no tracker here.
    """

    def __init__(
        self,
        model,
        *,
        image_size: Tuple[int, int] = (640, 640),
        score_threshold: float = 0.3,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        use_channels_last: bool = True,
        batch_size: int = DEFAULT_BATCH,
    ) -> None:
        self.model = model
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.score_threshold = float(score_threshold)
        self.batch_size = max(1, int(batch_size))
        # ``get_device()``, not wherever the weights happen to sit. ``cli``
        # builds the model on the CPU and leaves the move to the inference
        # class - both of the other two do it in their own ``__init__`` - so a
        # runner that merely reads ``next(parameters()).device`` runs the whole
        # pass on the processor and reports it as a slow GPU.
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.detector = self.model.detector
        self.adapter = self.model.det_adapter
        self.logger = get_logger(self.__class__.__name__)

        self._amp = bool(use_amp) and self.device.type == "cuda"
        self._channels_last = bool(use_channels_last) and self.device.type == "cuda"
        if self._channels_last:
            # Only the detector is moved: nothing else in this path runs.
            self.detector = self.detector.to(memory_format=torch.channels_last)
        self.model.eval()
        #: Seconds spent inside the forward pass, for ``runtime_stats.json``.
        self.detector_seconds = 0.0
        self.frames_done = 0

    # ── preprocessing ────────────────────────────────────────────────────
    def preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        """One frame to a normalised CHW float32 array.

        Exactly :meth:`StreamingInferencer.preprocess` minus the device move,
        which happens once per batch instead of once per frame. Keeping the
        arithmetic identical is what makes the two paths give the same boxes.
        """
        height, width = self.image_size
        try:
            from PIL import Image

            img = (
                np.asarray(
                    Image.fromarray(frame_rgb).resize((width, height), Image.BILINEAR),
                    dtype=np.float32,
                )
                / 255.0
            )
        except ImportError:  # pragma: no cover - PIL is a hard dependency here
            import cv2

            img = cv2.resize(frame_rgb, (width, height)).astype(np.float32) / 255.0
        return ((img - _MEAN) / _STD).transpose(2, 0, 1)

    def _to_device(self, chunk: Sequence[np.ndarray]) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(np.stack(chunk)))
        if self.device.type == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        else:
            tensor = tensor.to(self.device)
        if self._channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    # ── the forward pass ─────────────────────────────────────────────────
    @torch.no_grad()
    def detect_batch(
        self, frames: Sequence[np.ndarray], sizes: Sequence[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """``[{boxes, scores, class_ids, obb}, ...]``, one entry per frame.

        ``sizes`` is each frame's original ``(height, width)``, because the
        boxes come back normalised and have to land in the pixels of the frame
        they were found in.
        """
        if not frames:
            return []
        batch = self._to_device([self.preprocess(f) for f in frames])

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", torch.float16, enabled=self._amp):
            out = self.detector(batch)
            rows = self.adapter.extract_valid_detections(
                out.pred_logits,
                out.pred_boxes,
                out.query_features,
                score_threshold=self.score_threshold,
                pred_obb=out.pred_obb,
                pred_keypoints=out.pred_keypoints,
                pred_polarity=out.pred_polarity,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.detector_seconds += time.perf_counter() - started
        self.frames_done += len(frames)

        return [
            self._to_pixels(row, sizes[index]) for index, row in enumerate(rows)
        ]

    def _to_pixels(self, row: Dict[str, torch.Tensor], size: Tuple[int, int]) -> Dict[str, Any]:
        height, width = size
        boxes = row["boxes"].float()
        count = int(boxes.shape[0])
        scale = np.array([width, height, width, height], dtype=np.float32)
        obb = row["obb"].float() if "obb" in row else cxcywh_to_obb(boxes)
        obb_px = obb.cpu().numpy().copy()
        if count:
            obb_px[:, [0, 2]] *= width
            obb_px[:, [1, 3]] *= height
        return {
            "boxes": (cxcywh_to_xyxy(boxes).cpu().numpy() * scale).tolist(),
            "scores": row["scores"].float().cpu().numpy().tolist(),
            "class_ids": row["class_ids"].cpu().numpy().astype(int).tolist(),
            "obb": obb_px.tolist(),
            # No tracker ran, and saying so is the honest answer. The caller
            # decides what identity means for the kind of pass it asked for.
            "track_ids": [-1] * count,
            "action_ids": [-1] * count,
        }

    # ── the whole source ─────────────────────────────────────────────────
    def run(self, frames: Iterable[Tuple[int, np.ndarray, float]]) -> Dict[str, Any]:
        """Consume a ``(index, rgb, fps)`` generator into ``{index: row}``.

        The generator is the same one the other two paths read, so a video, an
        image directory and a strided export all arrive here unchanged.
        """
        results: Dict[str, Any] = {}
        pending: List[np.ndarray] = []
        sizes: List[Tuple[int, int]] = []
        indices: List[int] = []

        def flush() -> None:
            if not pending:
                return
            for index, row in zip(indices, self.detect_batch(pending, sizes)):
                results[str(index)] = row
            pending.clear()
            sizes.clear()
            indices.clear()

        for frame_index, frame_rgb, _fps in frames:
            pending.append(frame_rgb)
            sizes.append(frame_rgb.shape[:2])
            indices.append(int(frame_index))
            if len(pending) >= self.batch_size:
                flush()
        flush()
        return results

    def runtime_stats(self) -> Dict[str, Any]:
        seconds = max(self.detector_seconds, 1e-9)
        return {
            "frames": self.frames_done,
            "fps": self.frames_done / seconds,
            "detector_ms": seconds / max(self.frames_done, 1) * 1000.0,
            "batch_size": self.batch_size,
            "mode": "detect_only",
        }


def run_detect_only(
    model,
    source: str,
    output_dir: str,
    *,
    image_size: Tuple[int, int] = (640, 640),
    score_threshold: float = 0.3,
    batch_size: int = DEFAULT_BATCH,
    max_frames: Optional[int] = None,
    use_amp: bool = True,
    use_channels_last: bool = True,
) -> Dict[str, Any]:
    """Detect over a video or image directory and write ``predictions.json``.

    The same file, in the same schema, that the other two predict modes write
    — ``htrtdetr.gui.auto_label.load_candidates`` cannot tell which one made
    it, which is the point. ``runtime_stats.json`` is written beside it.
    """
    from .inferencer import read_image_dir_frames, read_video_frames

    src = Path(source)
    if src.is_dir():
        frames = read_image_dir_frames(str(src))
    elif src.exists():
        frames = read_video_frames(str(src), max_frames=max_frames)
    else:
        raise FileNotFoundError(f"Input not found: {src}")

    runner = BatchedDetector(
        model,
        image_size=image_size,
        score_threshold=score_threshold,
        batch_size=batch_size,
        use_amp=use_amp,
        use_channels_last=use_channels_last,
    )
    results = runner.run(frames)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "predictions.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    stats = runner.runtime_stats()
    with open(out / "runtime_stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(
        f"Detect-only inference complete. frames={stats['frames']} "
        f"FPS={stats['fps']:.1f} (detector {stats['detector_ms']:.2f} ms/frame "
        f"at batch {stats['batch_size']})"
    )
    print(f"Results: {out / 'predictions.json'}")
    return results
