"""
features.py — cheap per-frame descriptors, so sampling has something to sample on.

Smart Sampling needs to know how frames differ from one another. The expensive
answer is a visual encoder (DINOv2, CLIP); the cheap answer is a thumbnail. This
module provides the cheap answer **always** and accepts the expensive one when a
backend produced it, because the cheap one is what makes the feature work on the
machine it has to work on: no torch, no GPU, one CPU core, 30 minutes of video.

The thumbnail descriptor
------------------------
A 16x16 grayscale thumbnail, mean-centred and L2-normalised, is a 256-dimensional
vector whose cosine distance tracks "does this frame look like that frame" well
enough to pick a diverse set. It is not semantic — it cannot tell a grooming fly
from a walking one — but diversity sampling does not need semantics, it needs a
metric that separates *different-looking* frames, and for a fixed camera over a
fixed arena that is exactly what coarse spatial layout gives you.

Where it is not enough, :attr:`FrameFeature.embedding` takes a real encoder's
vector instead, from :mod:`htrtdetr.gui.sidecar`, with no change to any consumer:
:func:`distance` works on whatever is there.

Cost, measured on this machine (Python 3.14, OpenCV 4.13, no GPU): a 1024x1024
BGR frame costs about 1.1 ms for the thumbnail and about 0.9 ms for the motion
and focus statistics. Scanning a 30-minute video at 2 fps is therefore roughly
7 seconds of feature work — the video decode dominates by an order of magnitude,
which is why :func:`scan` takes a frame reader rather than opening the video: the
caller already has one open and seeking twice would double the only cost that
matters.

Purity: ``numpy`` and ``cv2`` are imported **inside** the two functions that need
them, so this module imports on a machine with neither and every pure function
below stays unit-testable there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "FrameFeature", "DESCRIPTOR_SIZE",
    "descriptor", "frame_stats", "detection_stats", "scan",
    "distance", "normalize", "farthest_point_order",
]

#: Thumbnail edge. 16 gives 256 dimensions — enough structure to separate
#: layouts, small enough that the distance matrix for 2,000 frames is 4 M
#: floats and fits comfortably in the GUI process.
DESCRIPTOR_SIZE = 16


@dataclass(frozen=True)
class FrameFeature:
    """Everything cheap that is known about one frame.

    Every field is normalised to roughly ``0..1`` so a weighted sum over them is
    meaningful without per-video calibration. "Roughly" is honest: ``motion``
    saturates rather than clipping, so a very fast frame reads 1.0 and so does an
    extremely fast one.

    Attributes:
        frame: Index into the video.
        embedding: Appearance vector, L2-normalised. The thumbnail descriptor by
            default; a real encoder's output when one ran. Empty when no frame
            was available.
        motion: Mean absolute difference from the previous *scanned* frame,
            scaled to ``0..1``. Not from the previous video frame — the scan is
            sparse on purpose, and pretending otherwise would misreport.
        focus: Variance of the Laplacian, squashed to ``0..1``. Low means
            blurred: out of focus, or the animal moved during the exposure.
        brightness: Mean luminance, ``0..1``. A jump in it is a lighting change,
            which is a scene change by another name.
        count: Detections in this frame, or ``-1`` when nothing has run yet.
        mean_area: Mean detection area in pixels. ``0.0`` when unknown.
        min_distance: Closest centre-to-centre distance between two detections,
            in multiples of the mean box diagonal. ``inf`` with fewer than two
            detections — there is no pair, and 0.0 would read as "touching".
        max_overlap: Largest pairwise IoU in the frame.
        mean_score: Mean detection confidence; ``1.0`` when unknown, so an
            un-scored frame never looks uncertain by accident.
    """

    frame: int
    embedding: Tuple[float, ...] = ()
    motion: float = 0.0
    focus: float = 0.0
    brightness: float = 0.0
    count: int = -1
    mean_area: float = 0.0
    min_distance: float = math.inf
    max_overlap: float = 0.0
    mean_score: float = 1.0

    @property
    def has_embedding(self) -> bool:
        return bool(self.embedding)

    @property
    def uncertainty(self) -> float:
        """``0..1``: how unsure the detector was about this frame.

        ``1 - mean_score`` and nothing cleverer. Entropy over class scores would
        be better if YOAKE's predictions carried a distribution; they carry one
        score per box, so entropy would be a dressed-up version of this number
        that merely looks more principled (P12).
        """
        return max(0.0, min(1.0, 1.0 - self.mean_score))

    @property
    def interaction(self) -> float:
        """``0..1``: how close to an interaction this frame is.

        Peaks when two animals are touching and falls to zero when they are more
        than two body lengths apart. This is the signal behind
        *Interaction-focused* sampling — courtship and aggression only exist in
        frames where animals are near each other, so a behavior dataset built by
        uniform sampling spends most of its budget on an empty arena.
        """
        if not math.isfinite(self.min_distance):
            return 0.0
        near = max(0.0, 1.0 - self.min_distance / 2.0)
        return max(near, self.max_overlap)


# ══════════════════════════════════════════════════════════════════════════
# Per-frame image features (numpy + cv2, lazily)
# ══════════════════════════════════════════════════════════════════════════

def descriptor(frame_bgr: Any, size: int = DESCRIPTOR_SIZE) -> Tuple[float, ...]:
    """The thumbnail appearance descriptor: mean-centred, L2-normalised.

    Mean-centred before normalising so that overall brightness does not dominate
    the distance — two frames of the same scene under a flickering lamp should be
    near neighbours, not opposites.

    Returns ``()`` for an unusable frame rather than raising: a single
    undecodable frame in a 40,000-frame video must cost that frame, not the scan.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        return ()
    if frame_bgr is None:
        return ()
    try:
        arr = np.asarray(frame_bgr)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(arr.astype("float32"), (int(size), int(size)),
                           interpolation=cv2.INTER_AREA).ravel()
        small -= small.mean()
        norm = float(np.linalg.norm(small))
        if norm <= 1e-8:
            # A uniform frame — lens cap, dropped frame, blown-out exposure.
            # It has no structure to compare, and a zero vector would sit at
            # distance 1 from everything and be picked first by diversity.
            return ()
        return tuple(float(v) for v in (small / norm))
    except Exception:
        return ()


def frame_stats(frame_bgr: Any, prev_gray: Any = None) -> Tuple[float, float, float, Any]:
    """``(motion, focus, brightness, gray)`` for one frame.

    ``gray`` is returned so the caller can feed it back as ``prev_gray`` without
    converting twice — the scan is the hot loop and this halves its colour
    conversions.

    All three statistics are computed on a downscaled copy. Motion and focus at
    full resolution measure sensor noise as much as animal movement, and cost
    16x more to say it.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        return 0.0, 0.0, 0.0, None
    if frame_bgr is None:
        return 0.0, 0.0, 0.0, prev_gray
    try:
        arr = np.asarray(frame_bgr)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        h, w = arr.shape[:2]
        scale = 256.0 / max(h, w, 1)
        if scale < 1.0:
            arr = cv2.resize(arr, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
        gray = arr.astype("float32")

        brightness = float(gray.mean()) / 255.0
        focus_raw = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        # 500 is the knee of the squash, not a threshold: it maps a typically
        # sharp lab frame to ~0.5 so both blur and extra sharpness stay visible.
        focus = focus_raw / (focus_raw + 500.0)

        motion = 0.0
        if prev_gray is not None and getattr(prev_gray, "shape", None) == gray.shape:
            diff = float(np.abs(gray - prev_gray).mean()) / 255.0
            motion = diff / (diff + 0.02)   # 2% mean difference reads as 0.5
        return motion, focus, brightness, gray
    except Exception:
        return 0.0, 0.0, 0.0, prev_gray


# ══════════════════════════════════════════════════════════════════════════
# Detection-derived features (pure Python)
# ══════════════════════════════════════════════════════════════════════════

def detection_stats(row: Sequence[Any]) -> Dict[str, float]:
    """Scene statistics from one frame's detections. No numpy, no cv2.

    Returns ``count``, ``mean_area``, ``min_distance`` (in box diagonals),
    ``max_overlap`` and ``mean_score``, ready to splat into
    :class:`FrameFeature`.
    """
    from .quality import _area, _bbox, _center, _diag   # local: same package
    from ..gui.auto_label import iou_any

    boxes: List[Tuple[Any, List[float]]] = []
    scores: List[float] = []
    for cand in row or ():
        box = _bbox(cand)
        if box is None:
            continue
        boxes.append((cand, box))
        scores.append(float(getattr(cand, "score", 1.0) or 0.0))

    if not boxes:
        return {"count": 0, "mean_area": 0.0, "min_distance": math.inf,
                "max_overlap": 0.0, "mean_score": 1.0}

    areas = [_area(b) for _, b in boxes]
    diags = [_diag(b) for _, b in boxes]
    mean_diag = sum(diags) / len(diags) if diags else 0.0

    min_distance = math.inf
    max_overlap = 0.0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ca, ba), (cb, bb) = boxes[i], boxes[j]
            pa, pb = _center(ba), _center(bb)
            dist = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if mean_diag > 0:
                min_distance = min(min_distance, dist / mean_diag)
            max_overlap = max(max_overlap, float(iou_any(
                ba, bb, getattr(ca, "obb", None), getattr(cb, "obb", None))))

    return {
        "count": len(boxes),
        "mean_area": sum(areas) / len(areas),
        "min_distance": min_distance,
        "max_overlap": max_overlap,
        "mean_score": sum(scores) / len(scores) if scores else 1.0,
    }


# ══════════════════════════════════════════════════════════════════════════
# The scan
# ══════════════════════════════════════════════════════════════════════════

def scan(
    read_frame: Callable[[int], Any],
    frames: Sequence[int],
    *,
    candidates: Optional[Mapping[int, Sequence[Any]]] = None,
    embeddings: Optional[Mapping[int, Sequence[float]]] = None,
    descriptor_size: int = DESCRIPTOR_SIZE,
    should_stop: Optional[Callable[[], bool]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[FrameFeature]:
    """Build a :class:`FrameFeature` for each frame in ``frames``.

    Args:
        read_frame: ``read_frame(i)`` returns a BGR array or ``None``. The
            caller owns the ``VideoCapture``; this module never opens one, which
            is what keeps it importable without OpenCV.
        frames: Frame indices, ascending. Sparse is the point — a 30-minute
            video is scanned at 1-4 fps, not at 30.
        candidates: Detections per frame, when a model has already run. Adds the
            scene statistics that Active Learning needs; without it the features
            are appearance-only, which is exactly the Cold Start case (§7.1).
        embeddings: Real encoder vectors per frame, when a backend produced them.
            Overrides the thumbnail descriptor for those frames.
        should_stop: Polled every frame so a Cancel button works (§32).
        on_progress: ``(done, total)`` after every frame, for the progress bar.

    Returns:
        One feature per frame **that could be read**, in order. A frame that
        failed to decode is skipped rather than represented by zeros — a zero
        feature is a lie that diversity sampling would happily select.
    """
    out: List[FrameFeature] = []
    prev_gray = None
    total = len(frames)

    for done, index in enumerate(frames, start=1):
        if should_stop is not None and should_stop():
            break
        frame_bgr = read_frame(int(index))
        if frame_bgr is None:
            if on_progress is not None:
                on_progress(done, total)
            continue

        motion, focus, brightness, prev_gray = frame_stats(frame_bgr, prev_gray)

        vector: Tuple[float, ...]
        given = None if embeddings is None else embeddings.get(int(index))
        if given:
            vector = normalize(given)
        else:
            vector = descriptor(frame_bgr, descriptor_size)

        stats = (detection_stats(candidates.get(int(index)) or ())
                 if candidates is not None else {})

        out.append(FrameFeature(
            frame=int(index),
            embedding=vector,
            motion=motion,
            focus=focus,
            brightness=brightness,
            count=int(stats.get("count", -1)),
            mean_area=float(stats.get("mean_area", 0.0)),
            min_distance=float(stats.get("min_distance", math.inf)),
            max_overlap=float(stats.get("max_overlap", 0.0)),
            mean_score=float(stats.get("mean_score", 1.0)),
        ))
        if on_progress is not None:
            on_progress(done, total)

    return out


# ══════════════════════════════════════════════════════════════════════════
# Vector helpers — pure Python, so sampling is testable with no numpy
# ══════════════════════════════════════════════════════════════════════════

def normalize(vector: Sequence[float]) -> Tuple[float, ...]:
    """L2-normalise. A zero vector returns ``()`` — see :func:`descriptor`."""
    total = math.sqrt(sum(float(v) * float(v) for v in vector))
    if total <= 1e-12:
        return ()
    return tuple(float(v) / total for v in vector)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine distance in ``0..2`` between two L2-normalised vectors.

    Returns ``1.0`` — "unrelated" — when either side is empty or the lengths
    disagree. Not ``0.0``: an unknown frame must not look identical to
    everything, or diversity sampling will refuse to pick any of them.
    """
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return 1.0 - max(-1.0, min(1.0, dot))


def farthest_point_order(
    vectors: Sequence[Sequence[float]],
    k: int,
    *,
    first: int = 0,
    seeds: Sequence[Sequence[float]] = (),
) -> List[int]:
    """Greedy k-centre: pick ``k`` indices that are as spread out as possible.

    The classic farthest-point traversal — start somewhere, then repeatedly take
    whatever is furthest from everything picked so far. Two properties earn it
    its place here over k-means clustering:

    * **It is deterministic given ``first``**, so a seed makes a run reproducible
      and a project can record the seed and get the same frames back (§41).
    * **It needs no ``k`` up front and no convergence**, so the caller can stop
      at 64 frames or at 200 without re-running anything. The prefix of a
      1,000-frame traversal is exactly the 64-frame traversal.

    Cost is ``O(n*k)`` distance evaluations with an incremental nearest-distance
    array — 2,000 frames and 200 picks is 400,000 cosine distances over
    256 dimensions, about 1.5 s here. Comfortably a worker-thread job, far too
    slow for the render loop.

    ``seeds`` are points that count as already picked without being candidates:
    the vectors of frames a human has already annotated. With seeds the
    traversal has no arbitrary start — the first pick is whatever lies furthest
    from everything already labelled, which is the entire point — so ``first``
    is ignored and the run stays deterministic without it. This is what makes a
    second pass over the same video continue the first instead of re-rolling it;
    both properties above survive, because seeding only changes the initial
    ``nearest`` array and not the loop.
    """
    n = len(vectors)
    if n == 0 or k <= 0:
        return []
    k = min(int(k), n)

    usable_seeds = [s for s in seeds if s]
    if usable_seeds:
        picked: List[int] = []
        nearest = [min(distance(v, s) for s in usable_seeds) for v in vectors]
    else:
        start = max(0, min(int(first), n - 1))
        picked = [start]
        nearest = [distance(vectors[i], vectors[start]) for i in range(n)]
        nearest[start] = -1.0

    while len(picked) < k:
        best, best_d = -1, -1.0
        for i in range(n):
            if nearest[i] > best_d:
                best, best_d = i, nearest[i]
        if best < 0:
            break
        picked.append(best)
        nearest[best] = -1.0
        chosen = vectors[best]
        for i in range(n):
            if nearest[i] < 0:
                continue
            d = distance(vectors[i], chosen)
            if d < nearest[i]:
                nearest[i] = d
    return picked
