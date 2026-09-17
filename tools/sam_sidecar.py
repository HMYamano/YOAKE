#!/usr/bin/env python
"""
sam_sidecar.py — the segmentation helper the annotator talks to
===============================================================

Reads newline-delimited JSON requests on stdin and writes replies on stdout,
one message per line. Started by ``htrtdetr.gui.sidecar.SidecarClient``, which
owns the protocol; this file owns the models.

    python tools/sam_sidecar.py --backend stub
    python tools/sam_sidecar.py --backend sam2 --model facebook/sam2-hiera-tiny
    python tools/sam_sidecar.py --backend segment_anything --model vit_b \
        --checkpoint sam_vit_b_01ec64.pth

Run it under whichever interpreter has torch — **not necessarily the one running
the GUI**, which is the entire reason this process exists.

Three rules this file lives by:

1. **stdout is the protocol.** Nothing else may write there. Logs, warnings and
   tracebacks go to stderr, which the parent drains into its log pane.
2. **A failure is a reply, not a crash.** Any exception inside one request comes
   back as ``{"ok": false, "error": ..., "detail": ...}``; the loop keeps going.
   A model that cannot segment one point is not a reason to lose the helper.
3. **The mask is reduced here, by the GUI's own code.**
   ``click_segment.mask_to_obb`` is imported and called in this process, so the
   rule that turns a mask into a body-axis box has exactly one implementation,
   and the pipe carries five floats instead of a megabyte of mask.

Self-test, no model and no GUI needed::

    python tools/sam_sidecar.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The parent puts <repo>/src on PYTHONPATH (auto_label.child_env). Add it here
# too so the helper can be started by hand from a checkout.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from htrtdetr.gui import click_segment as cs          # noqa: E402
from htrtdetr.gui import sidecar as proto             # noqa: E402


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #

class Adapter:
    """What every backend must provide.

    ``set_image`` is the expensive half and is called once per frame;
    ``point`` is the cheap half and is called once per click. Keeping them
    apart is what makes a click feel instant on a frame the user has been
    looking at (see ``docs/CLICK_SEGMENT.md`` §6.4).
    """

    name = "adapter"
    capabilities: Tuple[str, ...] = ("point",)

    def set_image(self, frame_bgr) -> None:
        raise NotImplementedError

    def point(self, points: Sequence[Sequence[float]],
              box: Optional[Sequence[float]]) -> Tuple[Any, Optional[float]]:
        """Return ``(mask, score)`` for the best mask at these prompts."""
        raise NotImplementedError

    def concept(self, text: str, limit: int) -> List[Tuple[Any, Optional[float]]]:
        raise NotImplementedError


class StubAdapter(Adapter):
    """A helper with no model, for testing the pipe itself.

    It answers deterministically from the prompt alone, so a protocol test
    asserts exact numbers and never depends on a download, a GPU or a random
    seed. Every other adapter is this one plus a model.
    """

    name = "stub"
    capabilities = ("point", "box", "concept")
    BOX = (40.0, 16.0, 0.3)          # w, h, theta — fixed, so tests can assert it

    def __init__(self) -> None:
        self.shape: Tuple[int, int] = (0, 0)

    def set_image(self, frame_bgr) -> None:
        self.shape = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))

    def obb_at(self, x: float, y: float):
        w, h, theta = self.BOX
        return (float(x), float(y), w, h, theta)

    def point(self, points, box):
        if not points:
            raise ValueError("no points given")
        x, y = float(points[0][0]), float(points[0][1])
        return ("obb", self.obb_at(x, y)), 0.5

    def concept(self, text, limit):
        """A deterministic grid, so the review flow can be tested end to end."""
        height, width = self.shape or (256, 256)
        out = []
        count = max(0, min(int(limit), 3))
        for i in range(count):
            x = width * (i + 1) / (count + 1)
            y = height / 2.0
            out.append((("obb", self.obb_at(x, y)), 0.9 - 0.1 * i))
        return out


class _TorchImageAdapter(Adapter):
    """Shared plumbing for the three real backends.

    They differ only in how a model is loaded and how a prompt is passed; the
    BGR->RGB conversion, the mask choice and the error handling are the same,
    so they live here once.
    """

    capabilities = ("point", "box")

    def __init__(self, model: str, checkpoint: str, device: str) -> None:
        self.model = model
        self.checkpoint = checkpoint
        self.device = device or "cpu"
        self._predictor = None

    @staticmethod
    def _rgb(frame_bgr):
        return frame_bgr[:, :, ::-1].copy()

    @staticmethod
    def _best(masks, scores):
        """The highest-scoring mask, as a plain boolean array."""
        import numpy as np
        masks = np.asarray(masks)
        if masks.ndim == 4:                 # (1, k, H, W) from some APIs
            masks = masks[0]
        if masks.ndim == 2:
            return masks.astype(bool), None
        scores = None if scores is None else np.asarray(scores).reshape(-1)
        idx = 0 if scores is None else int(scores.argmax())
        best = masks[idx].astype(bool)
        return best, (None if scores is None else float(scores[idx]))


class Sam2Adapter(_TorchImageAdapter):
    """Meta's SAM 2 image predictor.

    ``SAM2ImagePredictor.from_pretrained`` pulls weights from the Hub, so
    ``--model facebook/sam2-hiera-tiny`` is enough and there is no checkpoint
    path to manage. Tiny is the one worth trying first on a CPU.
    """

    name = "sam2"

    def _load(self):
        if self._predictor is None:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            name = self.model or "facebook/sam2-hiera-tiny"
            self._predictor = SAM2ImagePredictor.from_pretrained(name, device=self.device)
        return self._predictor

    def set_image(self, frame_bgr) -> None:
        self._load().set_image(self._rgb(frame_bgr))

    def point(self, points, box):
        import numpy as np
        pred = self._load()
        coords = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float32)
        labels = np.array([int(p[2]) if len(p) > 2 else 1 for p in points], dtype=np.int32)
        masks, scores, _ = pred.predict(
            point_coords=coords, point_labels=labels,
            box=None if box is None else np.array(box, dtype=np.float32),
            multimask_output=True,
        )
        return self._best(masks, scores)


class SegmentAnythingAdapter(_TorchImageAdapter):
    """The original ``segment_anything`` package.

    Needs a downloaded ``.pth``; ``--model`` is the registry key (``vit_b`` /
    ``vit_l`` / ``vit_h``). MobileSAM installs under the same API, which is why
    this adapter is worth keeping even though SAM 2 supersedes it: on a CPU-only
    machine a small model is the difference between usable and not.
    """

    name = "segment_anything"

    def _load(self):
        if self._predictor is None:
            from segment_anything import SamPredictor, sam_model_registry
            key = self.model or "vit_b"
            if not self.checkpoint:
                raise RuntimeError(
                    "segment_anything needs --checkpoint pointing at the .pth weights")
            sam = sam_model_registry[key](checkpoint=self.checkpoint)
            sam.to(device=self.device)
            self._predictor = SamPredictor(sam)
        return self._predictor

    def set_image(self, frame_bgr) -> None:
        self._load().set_image(self._rgb(frame_bgr))

    def point(self, points, box):
        import numpy as np
        pred = self._load()
        coords = np.array([[float(p[0]), float(p[1])] for p in points], dtype=np.float32)
        labels = np.array([int(p[2]) if len(p) > 2 else 1 for p in points], dtype=np.int32)
        masks, scores, _ = pred.predict(
            point_coords=coords, point_labels=labels,
            box=None if box is None else np.array(box, dtype=np.float32),
            multimask_output=True,
        )
        return self._best(masks, scores)


class TransformersSamAdapter(_TorchImageAdapter):
    """SAM through ``transformers`` (``SamModel`` + ``SamProcessor``).

    The one that needs no extra package beyond transformers itself, and the one
    a newer SAM generation is most likely to arrive through. The image
    embedding is computed once in :meth:`set_image` and reused per click, the
    same split the other two make.
    """

    name = "transformers"

    def _load(self):
        if self._predictor is None:
            import torch
            from transformers import SamModel, SamProcessor
            name = self.model or "facebook/sam-vit-base"
            model = SamModel.from_pretrained(name).to(self.device).eval()
            processor = SamProcessor.from_pretrained(name)
            self._predictor = (model, processor, torch)
        return self._predictor

    def set_image(self, frame_bgr) -> None:
        model, processor, torch = self._load()
        inputs = processor(self._rgb(frame_bgr), return_tensors="pt").to(self.device)
        with torch.no_grad():
            self._embedding = model.get_image_embeddings(inputs["pixel_values"])
        self._inputs = inputs
        self._size = (int(frame_bgr.shape[0]), int(frame_bgr.shape[1]))

    def point(self, points, box):
        model, processor, torch = self._load()
        if getattr(self, "_embedding", None) is None:
            raise RuntimeError("no image has been set")
        pts = [[[float(p[0]), float(p[1])] for p in points]]
        labels = [[[int(p[2]) if len(p) > 2 else 1 for p in points]]]
        inputs = processor(
            self._rgb_placeholder(), input_points=[pts], input_labels=labels,
            return_tensors="pt",
        ).to(self.device)
        inputs.pop("pixel_values", None)
        with torch.no_grad():
            out = model(image_embeddings=self._embedding, **inputs, multimask_output=True)
        masks = processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), self._inputs["original_sizes"].cpu(),
            self._inputs["reshaped_input_sizes"].cpu(),
        )[0]
        scores = out.iou_scores.cpu().reshape(-1)
        return self._best(masks.numpy(), scores.numpy())

    def _rgb_placeholder(self):
        """A 1x1 image: the processor wants one, but the embedding is reused."""
        import numpy as np
        return np.zeros((1, 1, 3), dtype=np.uint8)


ADAPTERS = {
    "stub": lambda a: StubAdapter(),
    "sam2": lambda a: Sam2Adapter(a.model, a.checkpoint, a.device),
    "segment_anything": lambda a: SegmentAnythingAdapter(a.model, a.checkpoint, a.device),
    "transformers": lambda a: TransformersSamAdapter(a.model, a.checkpoint, a.device),
}


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

def _say(message: Dict[str, Any]) -> None:
    sys.stdout.write(proto.encode(message) + "\n")
    sys.stdout.flush()


def _warn(text: str) -> None:
    sys.stderr.write(text.rstrip() + "\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Video propagation and frame embedding
# --------------------------------------------------------------------------- #

class Sam2VideoAdapter:
    """SAM 2's video predictor: one prompt, masks across a window of frames.

    Kept separate from :class:`Sam2Adapter` because the two SAM 2 entry points
    are genuinely different objects with different state. The image predictor
    holds one encoded frame; the video predictor holds an *inference state* for
    a whole window and streams masks out of it. Sharing one class would mean a
    mode flag and two half-initialised predictors.

    ``init_state`` wants a **directory of JPEG frames**, which is what the GUI
    writes before asking. Passing an mp4 works only on some SAM 2 versions and
    only with decord installed; a frame directory works on all of them, and the
    GUI has OpenCV for certain while this process may not.
    """

    name = "sam2_video"
    capabilities = ("video",)

    def __init__(self, model: str = "", device: str = "cpu") -> None:
        self.model = model or "facebook/sam2-hiera-tiny"
        self.device = device or "cpu"
        self._predictor = None

    def available(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("sam2") is not None

    def _load(self):
        if self._predictor is None:
            from sam2.sam2_video_predictor import SAM2VideoPredictor
            self._predictor = SAM2VideoPredictor.from_pretrained(
                self.model, device=self.device)
        return self._predictor

    def propagate(self, frames_dir: str, *, prompt_offset: int,
                  points: Optional[Sequence[Sequence[float]]],
                  box: Optional[Sequence[float]], obj_id: int,
                  backward: bool, max_frames: int):
        """Yield ``(frame_offset, mask)`` for every frame the model tracked.

        ``prompt_offset`` is an index into the window, not a video frame index:
        the window's mapping back to real frames belongs to the caller, and
        doing it here would mean this process has to be told the stride too.
        """
        import numpy as np
        import torch

        predictor = self._load()
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(state)

        kwargs: Dict[str, Any] = {"inference_state": state,
                                  "frame_idx": int(prompt_offset),
                                  "obj_id": int(obj_id)}
        if points:
            kwargs["points"] = np.array(
                [[float(p[0]), float(p[1])] for p in points], dtype=np.float32)
            kwargs["labels"] = np.array(
                [int(p[2]) if len(p) > 2 else 1 for p in points], dtype=np.int32)
        if box is not None:
            kwargs["box"] = np.array([float(v) for v in box], dtype=np.float32)
        predictor.add_new_points_or_box(**kwargs)

        seen = 0
        for offset, obj_ids, logits in predictor.propagate_in_video(
                state, reverse=bool(backward)):
            if max_frames and seen >= max_frames:
                break
            seen += 1
            try:
                index = list(obj_ids).index(int(obj_id))
            except ValueError:
                continue
            mask = logits[index]
            if hasattr(mask, "detach"):
                mask = mask.detach().cpu().numpy()
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[0]
            # SAM 2 yields logits, not a binary mask. Thresholding at zero is
            # the model's own convention; using 0.5 here would silently shrink
            # every mask.
            yield int(offset), (mask > 0.0)


class DinoEmbeddingAdapter:
    """Frames in, one L2-normalised feature vector out.

    Deliberately **not** a classifier. Smart Sampling needs a metric that
    separates different-looking frames; what the vector *means* is irrelevant,
    which is why the interface is a vector and a dimension and nothing else
    (``EmbeddingProvider`` in ``htrtdetr.gui.embedding``).

    Tries three loaders, best first, so it works on whatever the torch
    interpreter happens to have:

    1. ``transformers`` ``AutoModel`` — covers ``facebook/dinov2-*`` and any
       other hub encoder, and is what most torch installs already have.
    2. ``torch.hub`` ``facebookresearch/dinov2`` — no transformers needed.
    3. ``torchvision`` ResNet-18 features — always present with torchvision,
       weaker but real, and far better than the caller falling back to a
       16x16 thumbnail.
    """

    name = "dino"
    capabilities = ("embedding",)

    #: Loader -> the model name it defaults to.
    DEFAULTS = {
        "transformers": "facebook/dinov2-small",
        "hub": "dinov2_vits14",
        "torchvision": "resnet18",
    }

    def __init__(self, model: str = "", device: str = "cpu") -> None:
        self.model = model
        self.device = device or "cpu"
        self._model = None
        self._processor = None
        self._loader = ""
        self._dimension = 0

    def available(self) -> bool:
        import importlib.util
        if importlib.util.find_spec("torch") is None:
            return False
        return any(importlib.util.find_spec(m) is not None
                   for m in ("transformers", "torchvision"))

    def _load(self):
        if self._model is not None:
            return self._model
        import importlib.util
        import torch

        name = self.model
        if importlib.util.find_spec("transformers") is not None:
            try:
                from transformers import AutoImageProcessor, AutoModel
                name = name or self.DEFAULTS["transformers"]
                self._processor = AutoImageProcessor.from_pretrained(name)
                self._model = AutoModel.from_pretrained(name).to(self.device).eval()
                self._loader = "transformers"
                self._dimension = int(self._model.config.hidden_size)
                return self._model
            except Exception as exc:                  # noqa: BLE001
                _warn(f"transformers loader failed ({exc}); trying torch.hub")

        try:
            self._model = torch.hub.load(
                "facebookresearch/dinov2", name or self.DEFAULTS["hub"])
            self._model = self._model.to(self.device).eval()
            self._loader = "hub"
            self._dimension = int(getattr(self._model, "embed_dim", 384))
            return self._model
        except Exception as exc:                      # noqa: BLE001
            _warn(f"torch.hub loader failed ({exc}); trying torchvision")

        from torchvision import models
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        self._model = net.to(self.device).eval()
        self._loader = "torchvision"
        self._dimension = 512
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def loader(self) -> str:
        """Which of the three loaders actually answered. Reported to the GUI so
        a slow or weak backend is visible rather than guessed at."""
        return self._loader

    def encode(self, paths: Sequence[str], batch_size: int = 16):
        """``[(index, vector), ...]`` for the JPEGs at ``paths``, in order."""
        import numpy as np
        import torch

        model = self._load()
        out: List[Tuple[int, List[float]]] = []
        batch_size = max(1, int(batch_size))

        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            images = [_read_rgb(p) for p in chunk]
            keep = [(i, im) for i, im in enumerate(images) if im is not None]
            if not keep:
                continue

            with torch.no_grad():
                if self._loader == "transformers":
                    inputs = self._processor(images=[im for _i, im in keep],
                                             return_tensors="pt").to(self.device)
                    features = model(**inputs).last_hidden_state[:, 0]
                else:
                    tensor = _to_tensor([im for _i, im in keep], self.device)
                    features = model(tensor)
                    if features.ndim > 2:
                        features = features.flatten(1)
                features = torch.nn.functional.normalize(features.float(), dim=1)

            vectors = features.cpu().numpy()
            for row, (offset, _im) in enumerate(keep):
                out.append((start + offset,
                            [round(float(v), 6) for v in vectors[row]]))
        return out


def _read_rgb(path: str):
    """One JPEG as an HxWx3 uint8 RGB array, or ``None`` if unreadable.

    PIL first because ``transformers`` processors want PIL images anyway, and
    because this process may not have OpenCV even though it has torch.
    """
    try:
        from PIL import Image
        import numpy as np
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"))
    except Exception:                                  # noqa: BLE001
        try:
            import cv2
            import numpy as np
            array = cv2.imread(str(path), cv2.IMREAD_COLOR)
            return None if array is None else cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        except Exception:                              # noqa: BLE001
            return None


def _to_tensor(images, device):
    """ImageNet-normalised 224x224 batch, for the two non-transformers loaders."""
    import numpy as np
    import torch

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")
    rows = []
    for image in images:
        array = np.asarray(image, dtype="float32") / 255.0
        try:
            import cv2
            array = cv2.resize(array, (224, 224), interpolation=cv2.INTER_AREA)
        except Exception:                              # noqa: BLE001
            from PIL import Image
            array = np.asarray(
                Image.fromarray((array * 255).astype("uint8")).resize((224, 224)),
                dtype="float32") / 255.0
        rows.append(((array - mean) / std).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(rows)).to(device)


def _window_paths(frames_dir: str) -> List[str]:
    """Every frame file in a window, in numeric order.

    Sorted by name, which is why the GUI writes ``00000.jpg`` zero-padded: a
    plain sort of ``9.jpg`` and ``10.jpg`` puts them the wrong way round and
    every propagated box lands on the wrong frame.
    """
    folder = Path(frames_dir)
    if not folder.is_dir():
        return []
    return [str(p) for p in sorted(folder.glob("*.jpg"))]


class Session:
    """One helper's state: the adapters, and which frame it currently holds.

    Three adapters, not one: the image predictor answers clicks, the video
    predictor answers propagation, and the encoder answers embedding requests.
    They are separate objects with separate state and separate optional
    dependencies, and each is built only when something asks for it - so a
    helper started for Click to Box never pays for SAM 2's video predictor.
    """

    def __init__(self, adapter: Adapter, *, model: str = "",
                 device: str = "cpu") -> None:
        self.adapter = adapter
        self.frame: Optional[int] = None
        self.shape: Tuple[int, int] = (0, 0)
        self.model = model
        self.device = device
        self._video = None
        self._embed = None

    def video(self):
        """The SAM 2 video predictor, or ``None`` when sam2 is not installed."""
        if self._video is None:
            candidate = Sam2VideoAdapter(self.model, self.device)
            if not candidate.available():
                return None
            self._video = candidate
        return self._video

    def embedder(self):
        """The visual encoder, or ``None`` when nothing can load one."""
        if self._embed is None:
            candidate = DinoEmbeddingAdapter(self.model, self.device)
            if not candidate.available():
                return None
            self._embed = candidate
        return self._embed

    def extra_capabilities(self) -> List[str]:
        """What this helper can do beyond its image adapter.

        Probed with ``importlib.util.find_spec`` only - nothing is imported and
        no weights are touched, so ``hello`` stays instant even where SAM 2 is
        installed and would take thirty seconds to load.
        """
        out: List[str] = []
        if Sam2VideoAdapter(self.model, self.device).available():
            out.append("video")
        if DinoEmbeddingAdapter(self.model, self.device).available():
            out.append("embedding")
        return out

    def load(self, frame_index: int, path: str) -> None:
        """Bring a frame in and encode it, unless it is already the current one."""
        import numpy as np
        if self.frame == frame_index:
            return
        array = np.load(path, allow_pickle=False)
        self.adapter.set_image(array)
        self.frame = frame_index
        self.shape = (int(array.shape[0]), int(array.shape[1]))

    def to_obb(self, mask_or_obb, at: Optional[Tuple[float, float]]):
        """Reduce whatever the adapter returned to one OBB.

        An adapter may return a mask (every real model does) or an OBB directly
        (the stub). Masks go through the GUI's own ``mask_to_obb`` so the fit is
        identical to the local backend's, legs and all.
        """
        if isinstance(mask_or_obb, tuple) and mask_or_obb and mask_or_obb[0] == "obb":
            return tuple(float(v) for v in mask_or_obb[1]), ""
        import numpy as np
        mask = np.asarray(mask_or_obb)
        point = at if at is not None else (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
        return cs.mask_to_obb(mask.astype(np.uint8) * 255, point)


def handle(req: Dict[str, Any], session: Session, args) -> Optional[Dict[str, Any]]:
    op = req.get("op")
    req_id = int(req.get("id", 0))
    frame = req.get("frame")

    if op == "hello":
        got = int(req.get("protocol", 0))
        if got != proto.PROTOCOL_VERSION:
            return {"id": req_id, "ok": False, "error": "protocol_mismatch",
                    "detail": (f"This helper speaks protocol {proto.PROTOCOL_VERSION}, "
                               f"the GUI speaks {got}. They are from different checkouts.")}
        capabilities = list(session.adapter.capabilities)
        for extra in session.extra_capabilities():
            if extra not in capabilities:
                capabilities.append(extra)
        return {"id": req_id, "ok": True, "backend": session.adapter.name,
                "capabilities": capabilities,
                "device": getattr(session.adapter, "device", "cpu"),
                "protocol": proto.PROTOCOL_VERSION}

    if op == "embed":
        session.load(int(frame), str(req["path"]))
        return {"id": req_id, "ok": True, "frame": frame}

    if op == "point":
        points = req.get("points") or []
        if not points:
            return {"id": req_id, "ok": False, "frame": frame, "error": cs.NOTHING_FOUND,
                    "detail": cs.DIAGNOSES[cs.NOTHING_FOUND]}
        if frame is not None and session.frame != int(frame):
            path = req.get("path")
            if not path:
                return {"id": req_id, "ok": False, "frame": frame, "error": "no_frame",
                        "detail": "That frame has not been sent to the helper yet."}
            session.load(int(frame), str(path))
        mask, score = session.adapter.point(points, req.get("box"))
        at = (float(points[0][0]), float(points[0][1]))
        obb, why = session.to_obb(mask, at)
        if obb is None:
            return {"id": req_id, "ok": False, "frame": frame, "error": why,
                    "detail": cs.DIAGNOSES.get(why, "No mask was produced there.")}
        height, width = session.shape if session.shape != (0, 0) else (None, None)
        return {"id": req_id, "ok": True, "frame": frame,
                "obb": [round(float(v), 4) for v in obb],
                "bbox": list(cs.obb_to_aabb(obb, width, height)),
                "score": None if score is None else round(float(score), 4)}

    if op == "concept":
        if "concept" not in session.adapter.capabilities:
            return {"id": req_id, "ok": False, "frame": frame, "error": "unsupported",
                    "detail": (f"The {session.adapter.name} backend cannot find animals "
                               f"from a description.")}
        found = session.adapter.concept(str(req.get("text", "")), int(req.get("max", 32)))
        height, width = session.shape if session.shape != (0, 0) else (None, None)
        items = []
        for mask, score in found:
            obb, _why = session.to_obb(mask, None)
            if obb is None:
                continue
            items.append({"obb": [round(float(v), 4) for v in obb],
                          "bbox": list(cs.obb_to_aabb(obb, width, height)),
                          "score": None if score is None else round(float(score), 4)})
        return {"id": req_id, "ok": True, "frame": frame, "items": items}

    if op == "propagate":
        video = session.video()
        if video is None:
            return {"id": req_id, "ok": False, "frame": frame,
                    "error": "unsupported",
                    "detail": ("SAM 2 is not installed in this helper's Python. "
                               "Video propagation is unavailable; everything "
                               "else still works.")}
        frames_dir = str(req.get("frames_dir") or "")
        paths = _window_paths(frames_dir)
        if not paths:
            return {"id": req_id, "ok": False, "frame": frame,
                    "error": "no_frames",
                    "detail": "No frames were found in " + repr(frames_dir) + "."}
        first = int(req.get("first_frame", 0))
        prompt_offset = max(0, min(int(frame or first) - first, len(paths) - 1))
        try:
            import numpy as np
            items = []
            for offset, mask in video.propagate(
                    frames_dir,
                    prompt_offset=prompt_offset,
                    points=req.get("points"),
                    box=req.get("box"),
                    obj_id=int(req.get("obj_id", 1)),
                    backward=bool(req.get("backward")),
                    max_frames=int(req.get("max_frames", 0) or 0)):
                binary = np.asarray(mask).astype(np.uint8)
                if not binary.any():
                    continue
                # The mask's own centre of area, so mask_to_obb keeps the blob
                # the model actually tracked rather than whatever happens to
                # sit at the window centre.
                ys, xs = np.nonzero(binary)
                at = (float(xs.mean()), float(ys.mean()))
                obb, why = cs.mask_to_obb(binary * 255, at)
                if obb is None:
                    continue
                height, width = binary.shape[:2]
                items.append({
                    "frame": first + int(offset),
                    "obb": [round(float(v), 4) for v in obb],
                    "bbox": list(cs.obb_to_aabb(obb, width, height)),
                })
        except Exception as exc:                        # noqa: BLE001
            _warn(traceback.format_exc())
            return {"id": req_id, "ok": False, "frame": frame,
                    "error": "propagate_failed",
                    "detail": type(exc).__name__ + ": " + str(exc)}
        return {"id": req_id, "ok": True, "frame": frame, "items": items,
                "window": len(paths)}

    if op == "embed_frames":
        embedder = session.embedder()
        if embedder is None:
            return {"id": req_id, "ok": False, "error": "unsupported",
                    "detail": ("No visual encoder is available in this helper's "
                               "Python. Install transformers or torchvision, or "
                               "leave it - sampling falls back to the built-in "
                               "frame descriptor.")}
        paths = _window_paths(str(req.get("frames_dir") or ""))
        if not paths:
            return {"id": req_id, "ok": False, "error": "no_frames",
                    "detail": "No frames were found to encode."}
        first = int(req.get("first_frame", 0))
        try:
            pairs = embedder.encode(paths, int(req.get("batch_size", 16)))
        except Exception as exc:                        # noqa: BLE001
            _warn(traceback.format_exc())
            return {"id": req_id, "ok": False, "error": "encode_failed",
                    "detail": type(exc).__name__ + ": " + str(exc)}
        return {"id": req_id, "ok": True,
                "model": embedder.name, "loader": embedder.loader,
                "dimension": embedder.dimension,
                "items": [{"frame": first + index, "vector": vector}
                          for index, vector in pairs]}

    if op == "drop":
        if frame is not None and session.frame == int(frame):
            session.frame = None
        return {"id": req_id, "ok": True, "frame": frame}

    if op == "shutdown":
        return None

    return {"id": req_id, "ok": False, "error": "unknown_op",
            "detail": f"This helper does not know the operation {op!r}."}


def serve(session: Session, args, stream=None) -> int:
    """Read requests until stdin closes. Closing stdin is the normal way out."""
    stream = stream or sys.stdin
    for line in stream:
        req = proto.decode(line)
        if req is None:
            continue
        if req.get("op") == "shutdown":
            return 0
        try:
            reply = handle(req, session, args)
        except Exception as exc:                     # noqa: BLE001 — rule 2
            _warn(traceback.format_exc())
            reply = {"id": int(req.get("id", 0)), "ok": False,
                     "frame": req.get("frame"),
                     "error": type(exc).__name__, "detail": str(exc)}
        if reply is None:
            return 0
        _say(reply)
    return 0


def selftest() -> int:
    """Drive the stub through the protocol in-process. No GUI, no model, no torch."""
    import io
    import numpy as np

    session = Session(StubAdapter())
    session.adapter.set_image(np.zeros((120, 200, 3), np.uint8))
    session.frame, session.shape = 0, (120, 200)
    args = argparse.Namespace(model="", checkpoint="", device="cpu")

    checks = [
        ({"op": "hello", "id": 1, "protocol": proto.PROTOCOL_VERSION},
         lambda r: r["ok"] and "point" in r["capabilities"]),
        ({"op": "hello", "id": 2, "protocol": 999},
         lambda r: not r["ok"] and r["error"] == "protocol_mismatch"),
        ({"op": "point", "id": 3, "frame": 0, "points": [[100, 60, 1]]},
         lambda r: r["ok"] and r["obb"][0] == 100.0 and r["obb"][2] == 40.0),
        ({"op": "point", "id": 4, "frame": 0, "points": []},
         lambda r: not r["ok"]),
        ({"op": "concept", "id": 5, "frame": 0, "text": "fly", "max": 3},
         lambda r: r["ok"] and len(r["items"]) == 3),
        ({"op": "nonsense", "id": 6},
         lambda r: not r["ok"] and r["error"] == "unknown_op"),
        # The two optional capabilities must fail as a *reply* when their
        # dependency is absent, never as a crash, and the reply must name the
        # missing package rather than shrugging.
        ({"op": "propagate", "id": 7, "frame": 0, "frames_dir": "/nope",
          "first_frame": 0, "points": [[10, 10, 1]]},
         lambda r: not r["ok"] and r["error"] in ("unsupported", "no_frames")
         and bool(r.get("detail"))),
        ({"op": "embed_frames", "id": 8, "frames_dir": "/nope",
          "first_frame": 0},
         lambda r: not r["ok"] and r["error"] in ("unsupported", "no_frames")
         and bool(r.get("detail"))),
    ]
    failed = 0
    for req, check in checks:
        reply = handle(req, session, args)
        ok = reply is not None and check(reply)
        print(("PASS  " if ok else "FAIL  ") + json.dumps(req))
        if not ok:
            print("      got: " + json.dumps(reply))
            failed += 1

    # And once through the real reader loop, so the framing is exercised too.
    out = io.StringIO()
    stdout, sys.stdout = sys.stdout, out
    try:
        serve(Session(StubAdapter()),
              args,
              stream=io.StringIO(
                  proto.encode({"op": "hello", "id": 1,
                                "protocol": proto.PROTOCOL_VERSION}) + "\n"
                  + "a stray line a library printed\n"
                  + proto.encode({"op": "shutdown"}) + "\n"))
    finally:
        sys.stdout = stdout
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    ok = len(lines) == 1 and (proto.decode(lines[0]) or {}).get("ok") is True
    print(("PASS  " if ok else "FAIL  ") + "serve() framing, stray line ignored")
    failed += 0 if ok else 1

    print(f"\n{'all checks passed' if not failed else f'{failed} check(s) failed'}")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--backend", default="stub", choices=sorted(ADAPTERS))
    parser.add_argument("--model", default="", help="model id or registry key")
    parser.add_argument("--checkpoint", default="", help="weights file, if the backend needs one")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selftest", action="store_true",
                        help="run the protocol against the stub and exit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        return selftest()

    try:
        adapter = ADAPTERS[args.backend](args)
    except Exception as exc:                          # noqa: BLE001
        _warn(f"could not build the {args.backend} backend: {exc}")
        return 2
    return serve(Session(adapter, model=args.model, device=args.device), args)


if __name__ == "__main__":
    sys.exit(main())
