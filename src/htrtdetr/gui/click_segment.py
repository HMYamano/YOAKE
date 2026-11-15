"""
click_segment.py — YOAKE "Click to Box" engine (pure Python, no GUI)
===================================================================

The engine behind the annotator's **Click to Box** tool: the user clicks one
animal, and a rotated box (OBB) that fits its body comes back.

Design and rationale: ``docs/CLICK_SEGMENT.md``. Two of the three backends live
here because they answer synchronously: ``local`` (OpenCV, needs nothing
installed) and ``model`` (:func:`pick_candidate`, which chooses among detections
Auto-Annotate already produced). The third, ``sam``, runs in another process and
answers later, so it lives in ``gui/sidecar.py`` — but it calls
:func:`mask_to_obb` from here, so all three fit a box the same way.

Hard invariants (asserted by ``tests/test_gui_click_segment.py``), identical to
the ones ``gui/auto_label.py`` keeps:

* No ``dearpygui`` import, at module scope or inside a function.
* **No ``torch`` import, ever.** The annotator runs on a Python that has no
  torch at all; anything that needs it lives behind a child process.
* ``cv2`` / ``numpy`` are imported lazily, inside the functions that use them,
  so this module imports on a machine without OpenCV.
* No ``subprocess`` call. Child-process argv and environment are *data*, built
  elsewhere and spawned by the caller.
* Every public function is deterministic given its arguments and never raises
  on bad input — a failure comes back as a :class:`ClickResult` carrying a
  named diagnosis and the exact sentence the status line should show.

The central design rule, and the reason this file exists at all: **the quality
of a click-to-box tool is decided by how a mask is reduced to a rectangle, not
by how good the mask is.** ``cv2.minAreaRect`` on a raw fly mask swallows the
legs and wings, which inflates the box *and* twists its axis. :func:`mask_to_obb`
is therefore the one place that reduction happens, and the SAM sidecar will
import this very function rather than growing a second copy of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

__all__ = [
    # request / result
    "ClickRequest",
    "ClickResult",
    # diagnoses
    "NOTHING_FOUND",
    "MERGED_BACKGROUND",
    "TWO_ANIMALS",
    "TOUCHES_EDGE",
    "OUT_OF_FRAME",
    "NO_FRAME",
    "NO_CANDIDATE",
    "DIAGNOSES",
    # constants
    "MIN_BODY_PX",
    "DEFAULT_RADIUS",
    "MAX_RADIUS",
    "CORE_FRACTION",
    "BODY_FRACTION",
    "MAX_AREA_FRACTION",
    "TWIN_CORE_RATIO",
    # geometry
    "normalize_angle",
    "obb_corners",
    "obb_to_aabb",
    "point_in_obb",
    # segmentation
    "MaskParts",
    "mask_to_obb",
    "reduce_mask",
    "threshold_gray",
    "segment_local",
    "segment",
    "pick_candidate",
    "BACKENDS",
    "BACKEND_LABELS",
    "available_backends",
]


# --------------------------------------------------------------------------- #
# 1. Constants
# --------------------------------------------------------------------------- #

#: Below this, a box is noise, not an animal. Same value as
#: ``annotator._MIN_BOX_PX`` — the two must agree or the annotator would reject
#: a box this module happily produced.
MIN_BODY_PX = 5

#: Half-side of the first search window, in image pixels. The window doubles
#: (up to :data:`MAX_RADIUS`) while the blob still runs off its edge, so this is
#: a starting guess, not a limit.
DEFAULT_RADIUS = 48

#: Refuse to search wider than this. Past it, Otsu is thresholding a scene
#: rather than an animal, and a wrong box is worse than no box.
MAX_RADIUS = 256

#: A pixel belongs to the "body core" when its distance to the nearest
#: background pixel is at least this fraction of the blob's maximum. Legs and
#: wings are thin, so they fall out; the thorax and abdomen stay. The core
#: decides the *axis*, where robustness matters more than exact size.
CORE_FRACTION = 0.40

#: Radius of the opening that decides the *extent*, as a fraction of the body's
#: half-thickness. A morphological opening deletes everything thinner than
#: twice its radius — legs, wings, antennae — while **preserving the size of
#: what survives**, which is exactly the property the extent needs and the
#: reason the extent is not measured on the raw blob (legs would inflate it)
#: nor on the core (which is deliberately shrunken). Scaling the radius by the
#: blob's own thickness is what keeps this free of any pixel-count tuning.
BODY_FRACTION = 0.15

#: A blob covering more than this fraction of the search window is not one
#: animal — the threshold ran into the background.
MAX_AREA_FRACTION = 0.40

#: Two body cores whose areas are this similar mean two touching animals, not
#: one animal with a dent in it.
TWIN_CORE_RATIO = 0.60

_HALF_PI = math.pi / 2.0
#: Matches ``geometry/obb.py::_ANGLE_SNAP``. Both files fold an angle into
#: ``[-pi/2, pi/2)`` and must agree at the boundary, which a test asserts.
_ANGLE_SNAP = 1e-6


# --------------------------------------------------------------------------- #
# 2. Diagnoses — P5: name the cause, and name what to do instead
# --------------------------------------------------------------------------- #

NOTHING_FOUND = "nothing_found"
MERGED_BACKGROUND = "merged_background"
TWO_ANIMALS = "two_animals"
TOUCHES_EDGE = "touches_edge"
OUT_OF_FRAME = "out_of_frame"
NO_FRAME = "no_frame"
NO_CANDIDATE = "no_candidate"

#: The exact sentence the status line shows for each failure. Every one names
#: the cause *and* the next action, and every one offers the same escape hatch:
#: draw the box by hand, which always works.
DIAGNOSES = {
    NOTHING_FOUND: (
        "Nothing found at that point. Click on the animal itself, or draw the "
        "box by hand with W."
    ),
    MERGED_BACKGROUND: (
        "The animal merged with the background here. Zoom in and click again, "
        "or draw the box by hand with W."
    ),
    TWO_ANIMALS: (
        "Two animals look joined at that point. Click nearer this one's "
        "centre, or draw the box by hand with W."
    ),
    TOUCHES_EDGE: (
        "The shape kept running past the search area. Click nearer its centre, "
        "or draw the box by hand with W."
    ),
    NO_CANDIDATE: (
        "The model found no detection at that point on this frame. Switch the "
        "fit to Local (OpenCV), or draw the box by hand with W."
    ),
    OUT_OF_FRAME: "That point is outside the frame.",
    NO_FRAME: "No frame is loaded yet.",
}


# --------------------------------------------------------------------------- #
# 3. Request / result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClickRequest:
    """One click, in image pixels.

    ``x`` / ``y`` are what ``annotator._c2i()`` returns, so zoom and pan are
    already undone by the time a request is built.
    """

    frame_index: int
    x: float
    y: float
    #: ``True`` = "this point is the animal", ``False`` = "this point is not"
    #: (Ctrl+click). The ``local`` backend ignores it — it has no notion of an
    #: accumulated prompt; the SAM backend will use it.
    polarity: bool = True
    #: Half-side of the first search window. ``None`` uses
    #: :data:`DEFAULT_RADIUS`. The annotator passes the last successful body
    #: length so a second click on a similar animal starts at the right scale.
    radius_hint: Optional[float] = None
    #: ``(x0, y0, x1, y1)`` from a Shift+drag, which pins the search window
    #: instead of guessing it.
    roi: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class ClickResult:
    """What one click produced — or why it produced nothing.

    A failure is a normal result, not an exception: it carries the diagnosis
    and the sentence to show. Callers branch on :attr:`ok`.
    """

    obb: Optional[Tuple[float, float, float, float, float]]
    bbox: Optional[Tuple[int, int, int, int]]
    score: Optional[float]
    backend: str
    detail: str
    #: ``""`` on success, otherwise a key of :data:`DIAGNOSES`.
    diagnosis: str = ""
    class_id: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.obb is not None


def _failed(diagnosis: str, backend: str) -> ClickResult:
    return ClickResult(
        obb=None,
        bbox=None,
        score=None,
        backend=backend,
        detail=DIAGNOSES.get(diagnosis, "Could not fit a box there."),
        diagnosis=diagnosis,
    )


# --------------------------------------------------------------------------- #
# 4. Geometry
# --------------------------------------------------------------------------- #


def normalize_angle(theta: float) -> float:
    """Fold ``theta`` into ``[-pi/2, pi/2)``.

    A rectangle has period pi, so this changes nothing geometrically — and,
    because a pi rotation maps a rectangle onto itself, it does not swap which
    side is ``w``. That is what lets :func:`mask_to_obb` promise "``w`` is the
    body axis" while still handing back a canonical angle.

    The float64 modulo can round to pi and pin the result at +pi/2, giving two
    canonical forms for one box; the snap-back reproduces
    ``geometry/obb.py::normalize_angle`` exactly, which a test asserts.
    """
    out = (float(theta) + _HALF_PI) % math.pi - _HALF_PI
    if out >= _HALF_PI - _ANGLE_SNAP:
        out -= math.pi
    return out


def obb_corners(obb: Tuple[float, float, float, float, float]):
    """The four corners of an OBB, in the same order as ``AnnObject.corners()``."""
    cx, cy, w, h, th = obb
    hw, hh = w / 2.0, h / 2.0
    c, s = math.cos(th), math.sin(th)
    return [
        (cx + lx * c - ly * s, cy + lx * s + ly * c)
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def obb_to_aabb(
    obb: Tuple[float, float, float, float, float],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Integer bounding box of an OBB, optionally clamped to a frame."""
    pts = obb_corners(obb)
    x1 = math.floor(min(p[0] for p in pts))
    y1 = math.floor(min(p[1] for p in pts))
    x2 = math.ceil(max(p[0] for p in pts))
    y2 = math.ceil(max(p[1] for p in pts))
    if width is not None:
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
    if height is not None:
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))
    return int(x1), int(y1), int(x2), int(y2)


# --------------------------------------------------------------------------- #
# 5. mask -> OBB — the whole quality question, in one function
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MaskParts:
    """What :func:`reduce_mask` derives from one mask on the way to an OBB.

    Everything is in the mask's own pixel coordinates. The three masks are
    ``uint8`` 0/1 arrays of the mask's shape:

    * ``blob`` — the connected component under the point (appendages included)
    * ``core`` — the distance-transform core the axis was taken from
    * ``body`` — the opened blob the extent was measured on

    ``axis`` is the body axis as a unit vector. Its sign is whatever the
    eigenvector came out as: head and tail are as unresolved here as in
    :func:`mask_to_obb`. ``extent`` is ``(u0, u1, v0, v1)``, where the body ends
    along ``axis`` and across it, measured from ``centre`` (the core's mean).
    ``thickness`` is the blob's largest distance to the background, i.e. its
    half-thickness. ``twin_core`` is only ever ``True`` when the caller asked
    for ``allow_twins``.
    """

    blob: Any
    core: Any
    body: Any
    centre: Tuple[float, float]
    axis: Tuple[float, float]
    extent: Tuple[float, float, float, float]
    thickness: float
    obb: Tuple[float, float, float, float, float]
    twin_core: bool = False


def mask_to_obb(
    mask: Any, at: Tuple[float, float]
) -> Tuple[Optional[Tuple[float, float, float, float, float]], str]:
    """Reduce a binary mask to a body-axis OBB.

    Returns ``(obb, "")`` on success and ``(None, diagnosis)`` otherwise, where
    ``diagnosis`` is a key of :data:`DIAGNOSES`. The work is done by
    :func:`reduce_mask`; this is its OBB, for callers that want nothing else.

    Four steps, in this order and for these reasons:

    1. **Keep only the blob under the click.** Anything else in the window is
       another animal or background texture.
    2. **Strip the appendages.** A distance transform, cut at
       :data:`CORE_FRACTION` of its own maximum, deletes everything thinner
       than 40% of the thickest part — legs, wings, antennae — and keeps the
       body. Cutting at a fraction of the blob's own maximum is what makes this
       scale-free: it needs no idea of how many pixels an animal is.
    3. **Take the axis from the core by PCA, the extent from the body cut.**
       PCA rather than ``minAreaRect`` because the axis stays put when the core
       is dented. The extent comes from the looser :data:`BODY_FRACTION` cut
       with its erosion added back, so the box covers the whole body and none
       of the legs — the raw blob would include them, and the core alone would
       be a box around the thorax.
    4. **Canonicalise the angle**, never the side assignment: ``w`` is the body
       axis whether or not it is the longer side. ``geometry/obb.py`` makes the
       same choice, and says why — forcing the long side into ``w`` makes size
       regression jump discontinuously where w and h cross.

    The head/tail direction is *not* resolved. Under the double-angle encoding
    the model uses, orientation lives mod pi and a head-tail flip is not
    representable, so guessing one would be inventing information.
    """
    parts, why = reduce_mask(mask, at)
    if parts is None:
        return None, why
    return parts.obb, ""


def reduce_mask(
    mask: Any, at: Tuple[float, float], allow_twins: bool = False
) -> Tuple[Optional[MaskParts], str]:
    """The four steps of :func:`mask_to_obb`, keeping what each one produced.

    Returns ``(parts, "")`` on success and ``(None, diagnosis)`` otherwise.

    With ``allow_twins=False`` (what :func:`mask_to_obb` uses) two cores of
    comparable size are refused as :data:`TWO_ANIMALS`, exactly as before:
    a box over both would be wrong in a training set. A measuring caller
    (``analysis/silhouette.py``) can instead pass ``allow_twins=True``: the
    axis is taken from the core nearest the point, the extent is still the
    blob's — so for two animals under one detection the box is the pair's —
    and ``parts.twin_core`` reports that the blob was really two animals,
    which, for animals in copula, is the measurement.
    """
    import cv2
    import numpy as np

    m = np.ascontiguousarray(mask)
    if m.ndim != 2:
        return None, NOTHING_FOUND
    m = (m > 0).astype(np.uint8)
    h, w = m.shape[:2]
    if h < 2 or w < 2 or int(m.sum()) == 0:
        return None, NOTHING_FOUND

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, NOTHING_FOUND

    # ── 1. the blob under the click ──────────────────────────────────────
    px, py = int(round(at[0])), int(round(at[1]))
    lab = 0
    if 0 <= px < w and 0 <= py < h:
        lab = int(labels[py, px])
    if lab == 0:
        # The click landed a pixel or two off the animal (thresholding is not
        # exact at the edge). Accept the nearest blob within a quarter of the
        # window, and only that.
        reach = 0.25 * max(h, w)
        best: Optional[Tuple[float, int]] = None
        for i in range(1, n):
            d = math.hypot(float(cents[i][0]) - px, float(cents[i][1]) - py)
            if d <= reach and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            return None, NOTHING_FOUND
        lab = best[1]

    area = int(stats[lab, cv2.CC_STAT_AREA])
    if area < MIN_BODY_PX * MIN_BODY_PX:
        return None, NOTHING_FOUND

    bx = int(stats[lab, cv2.CC_STAT_LEFT])
    by = int(stats[lab, cv2.CC_STAT_TOP])
    bw = int(stats[lab, cv2.CC_STAT_WIDTH])
    bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
    # Runs off the window: the caller can widen the window and try again, which
    # is a different situation from "this really is one huge blob".
    if bx == 0 or by == 0 or bx + bw >= w or by + bh >= h:
        return None, TOUCHES_EDGE
    if area > MAX_AREA_FRACTION * h * w:
        return None, MERGED_BACKGROUND

    comp = (labels == lab).astype(np.uint8)

    # ── 2. strip the appendages ──────────────────────────────────────────
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    dmax = float(dist.max())
    core = (dist >= CORE_FRACTION * dmax).astype(np.uint8) if dmax > 0 else comp
    if int(core.sum()) < 4:
        core = comp

    # Two cores of comparable size are two animals that touch. One core with a
    # small satellite is one animal, so the ratio - not the count - decides.
    twin = False
    nc, clabels, cstats, ccents = cv2.connectedComponentsWithStats(core, 8)
    if nc > 2:
        areas = sorted(
            (int(cstats[i, cv2.CC_STAT_AREA]) for i in range(1, nc)),
            reverse=True,
        )
        if areas[0] > 0 and areas[1] >= TWIN_CORE_RATIO * areas[0]:
            if not allow_twins:
                return None, TWO_ANIMALS
            twin = True
        # Otherwise keep the core nearest the click and drop the specks.
        pick, best_d = 1, None
        for i in range(1, nc):
            d = math.hypot(float(ccents[i][0]) - px, float(ccents[i][1]) - py)
            if best_d is None or d < best_d:
                pick, best_d = i, d
        core = (clabels == pick).astype(np.uint8)

    # ── 3. axis from the core, extent from the body cut ──────────────────
    ys, xs = np.nonzero(core)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = pts.mean(axis=0)
    centred = pts - mean
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    try:
        _evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, NOTHING_FOUND
    u = np.asarray(evecs[:, -1], dtype=np.float64)  # largest eigenvalue last
    if not np.isfinite(u).all() or float(np.hypot(u[0], u[1])) < 1e-12:
        u = np.array([1.0, 0.0])
    u = u / float(np.hypot(u[0], u[1]))
    v = np.array([-u[1], u[0]])

    # The body: the blob with everything thinner than one opening diameter
    # removed. Erode-then-dilate deletes the legs outright and gives back the
    # body at its original size, so the extent measured here needs no
    # correction term — which is what the raw blob and the core both lack.
    radius = max(1, int(round(BODY_FRACTION * dmax))) if dmax > 0 else 0
    body = comp
    if radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, kernel)
        if int(opened.sum()) >= 4:
            body = opened
    # An opening can split a blob at a waist; keep the piece under the click.
    nb, blabels, _bstats, bcents = cv2.connectedComponentsWithStats(body, 8)
    if nb > 2:
        pick, best_d = 1, None
        for i in range(1, nb):
            d = math.hypot(float(bcents[i][0]) - mean[0], float(bcents[i][1]) - mean[1])
            if best_d is None or d < best_d:
                pick, best_d = i, d
        body = (blabels == pick).astype(np.uint8)

    cys, cxs = np.nonzero(body)
    cpts = np.stack([cxs.astype(np.float64), cys.astype(np.float64)], axis=1) - mean
    pu = cpts @ u
    pv = cpts @ v
    u0, u1 = float(pu.min()), float(pu.max())
    v0, v1 = float(pv.min()), float(pv.max())
    # +1 because a single pixel spans one pixel, not zero.
    length = (u1 - u0) + 1.0
    breadth = (v1 - v0) + 1.0
    mid_u, mid_v = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    cx = float(mean[0] + mid_u * u[0] + mid_v * v[0])
    cy = float(mean[1] + mid_u * u[1] + mid_v * v[1])

    if length < MIN_BODY_PX or breadth < 1.0:
        return None, NOTHING_FOUND

    # ── 4. canonical angle, body axis stays in w ─────────────────────────
    theta = normalize_angle(math.atan2(float(u[1]), float(u[0])))
    return MaskParts(
        blob=comp,
        core=core,
        body=body,
        centre=(float(mean[0]), float(mean[1])),
        axis=(float(u[0]), float(u[1])),
        extent=(u0, u1, v0, v1),
        thickness=dmax,
        obb=(cx, cy, float(length), float(breadth), theta),
        twin_core=twin,
    ), ""


# --------------------------------------------------------------------------- #
# 6. Backend: local (OpenCV only)
# --------------------------------------------------------------------------- #


def _window(
    x: float, y: float, radius: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    """Search window around a point, clamped to the frame."""
    r = max(8.0, float(radius))
    x0 = int(max(0, math.floor(x - r)))
    y0 = int(max(0, math.floor(y - r)))
    x1 = int(min(width, math.ceil(x + r)))
    y1 = int(min(height, math.ceil(y + r)))
    return x0, y0, x1, y1


def _otsu_mask(roi_bgr: Any, at_local: Tuple[float, float]) -> Any:
    """Binary mask of the ROI, with the polarity chosen from the click itself.

    Deciding dark-on-light versus light-on-dark from the pixels under the
    cursor is what makes one gesture work on both a dark fly on a lit arena and
    a pale larva on a dark plate, with nothing for the user to configure.
    """
    import cv2

    return threshold_gray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY), at_local)


def threshold_gray(gray: Any, at_local: Tuple[float, float]) -> Any:
    """:func:`_otsu_mask` for a ROI that is already grey (``uint8``, 2-D).

    The silhouette measurement (``analysis/silhouette.py``) converts each frame
    to grey once and thresholds many ROIs of it, so it enters here.
    """
    import cv2
    import numpy as np

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    px, py = int(round(at_local[0])), int(round(at_local[1]))
    h, w = gray.shape[:2]
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    y0, y1 = max(0, py - 3), min(h, py + 4)
    x0, x1 = max(0, px - 3), min(w, px + 4)
    here = float(np.median(gray[y0:y1, x0:x1]))
    overall = float(np.median(gray))
    dark_object = here <= overall

    flag = cv2.THRESH_BINARY_INV if dark_object else cv2.THRESH_BINARY
    _thr, bw = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    # A 3x3 open removes single-pixel noise. It also thins legs slightly, which
    # is free help for the distance transform later.
    return cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


def _grabcut_mask(roi_bgr: Any, at_local: Tuple[float, float]) -> Optional[Any]:
    """Slower second opinion, used only after Otsu has already failed.

    Returns ``None`` if GrabCut cannot run (window too small, degenerate rect),
    which is a normal outcome and not an error.
    """
    import cv2
    import numpy as np

    h, w = roi_bgr.shape[:2]
    if h < 16 or w < 16:
        return None
    rx = max(1, w // 4)
    ry = max(1, h // 4)
    rect = (rx, ry, max(2, w - 2 * rx), max(2, h - 2 * ry))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return out.astype(np.uint8)


def segment_local(frame_bgr: Any, req: ClickRequest) -> ClickResult:
    """Fit an OBB to the animal under ``req``, using OpenCV only.

    Needs no model, no download and no torch, runs in the GUI process in a few
    milliseconds, and is therefore both the default backend and the fallback
    for every other one.
    """
    import numpy as np  # noqa: F401  (kept symmetrical with the cv2 import)

    backend = "local"
    if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
        return _failed(NO_FRAME, backend)
    height, width = frame_bgr.shape[:2]
    if not (0 <= req.x < width and 0 <= req.y < height):
        return _failed(OUT_OF_FRAME, backend)

    if req.roi is not None:
        x0, y0, x1, y1 = (int(v) for v in req.roi)
        x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
        windows = [(x0, y0, x1, y1)]
    else:
        base = float(req.radius_hint or DEFAULT_RADIUS)
        windows = []
        for step in range(3):
            r = min(base * (2**step), float(MAX_RADIUS))
            windows.append(_window(req.x, req.y, r, width, height))
            if r >= MAX_RADIUS:
                break

    obb = None
    why = NOTHING_FOUND
    win = windows[0]
    for win in windows:
        x0, y0, x1, y1 = win
        if x1 - x0 < 8 or y1 - y0 < 8:
            why = OUT_OF_FRAME
            continue
        roi = frame_bgr[y0:y1, x0:x1]
        at_local = (req.x - x0, req.y - y0)
        obb, why = mask_to_obb(_otsu_mask(roi, at_local), at_local)
        if obb is not None:
            break
        # Only a blob running off the window is worth a wider window; every
        # other failure would reproduce identically at any size.
        if why != TOUCHES_EDGE:
            break

    if obb is None:
        x0, y0, x1, y1 = win
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            at_local = (req.x - x0, req.y - y0)
            cut = _grabcut_mask(frame_bgr[y0:y1, x0:x1], at_local)
            if cut is not None:
                obb, grab_why = mask_to_obb(cut, at_local)
                if obb is None:
                    why = grab_why if why == TOUCHES_EDGE else why

    if obb is None:
        return _failed(why, backend)

    x0, y0 = win[0], win[1]
    cx, cy, w, h, theta = obb
    placed = (cx + x0, cy + y0, w, h, theta)
    bbox = obb_to_aabb(placed, width, height)
    if bbox[2] - bbox[0] < MIN_BODY_PX or bbox[3] - bbox[1] < MIN_BODY_PX:
        return _failed(NOTHING_FOUND, backend)

    return ClickResult(
        obb=placed,
        bbox=bbox,
        score=None,
        backend=backend,
        detail=f"Fitted a {w:.0f} x {h:.0f} px box from your click.",
    )


# --------------------------------------------------------------------------- #
# 7. Backend: model — pick from detections that already exist
# --------------------------------------------------------------------------- #


def point_in_obb(x: float, y: float, obb: Sequence[float]) -> bool:
    """Is the point inside this rotated rectangle?"""
    cx, cy, w, h, theta = (float(v) for v in obb)
    c, s = math.cos(-theta), math.sin(-theta)
    dx, dy = x - cx, y - cy
    lx, ly = dx * c - dy * s, dx * s + dy * c
    return abs(lx) <= w / 2.0 and abs(ly) <= h / 2.0


def pick_candidate(candidates: Sequence[Any], req: ClickRequest) -> ClickResult:
    """The detection under the click, chosen from ones already computed.

    Runs no model. Auto-Annotate has already produced these boxes, so this
    backend only decides *which* of them the user meant — and once a detector
    has been trained on this species it is the most accurate backend there is,
    because it was trained for exactly this and SAM was not.

    Preference order: a box containing the click beats one that merely lies
    near it, and among equals the nearest centre wins. A click that is neither
    inside a box nor within one box-length of one is a miss, not a nudge toward
    the nearest thing on the frame.
    """
    best: Optional[Tuple[Tuple[int, float], Any]] = None
    for cand in candidates or ():
        bbox = [float(v) for v in getattr(cand, "bbox", ()) or ()]
        if len(bbox) != 4:
            continue
        obb = getattr(cand, "obb", None)
        if obb is not None and len(obb) == 5:
            cx, cy = float(obb[0]), float(obb[1])
            reach = max(float(obb[2]), float(obb[3]))
            inside = point_in_obb(req.x, req.y, obb)
        else:
            cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
            reach = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            inside = bbox[0] <= req.x <= bbox[2] and bbox[1] <= req.y <= bbox[3]
        distance = math.hypot(req.x - cx, req.y - cy)
        if not inside and distance > max(reach, 1.0):
            continue
        rank = (0 if inside else 1, distance)
        if best is None or rank < best[0]:
            best = (rank, cand)

    if best is None:
        return _failed(NO_CANDIDATE, "model")

    cand = best[1]
    bbox = tuple(int(round(float(v))) for v in cand.bbox)
    obb = getattr(cand, "obb", None)
    if obb is not None and len(obb) == 5:
        placed = tuple(float(v) for v in obb)
        placed = (placed[0], placed[1], placed[2], placed[3], normalize_angle(placed[4]))
    else:
        placed = (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
            float(bbox[2] - bbox[0]),
            float(bbox[3] - bbox[1]),
            0.0,
        )
    score = getattr(cand, "score", None)
    return ClickResult(
        obb=placed,
        bbox=bbox,
        score=None if score is None else float(score),
        backend="model",
        class_id=getattr(cand, "class_id", 0),
        detail=(
            "Took the model's detection at that point"
            + (f" (score {float(score):.2f})." if score is not None else ".")
        ),
    )


# --------------------------------------------------------------------------- #
# 8. Backend registry
# --------------------------------------------------------------------------- #

#: Every backend the UI offers, in the order it lists them.
BACKENDS = ("local", "sam", "model")

#: Backends that answer *here*, synchronously, from the frame alone.
#: ``sam`` is missing on purpose: it lives in another process and answers later,
#: so it cannot have this signature (see ``gui/sidecar.py``). ``model`` is
#: missing for a different reason: it needs detections, which are not derivable
#: from the frame — call :func:`pick_candidate` with them instead.
_IMPLEMENTATIONS = {"local": segment_local}

#: Human names for the backend picker.
BACKEND_LABELS = {
    "local": "Local (OpenCV)",
    "sam": "SAM helper",
    "model": "Model predictions",
}


def available_backends() -> Tuple[str, ...]:
    """Backends that answer synchronously in this process, in UI order.

    ``local`` is always in the list: it needs nothing installed. That is the
    property that lets the tool button itself never be locked — only entries in
    the backend picker are.
    """
    return tuple(name for name in BACKENDS if name in _IMPLEMENTATIONS)


def segment(frame_bgr: Any, req: ClickRequest, backend: str = "local") -> ClickResult:
    """Dispatch to a backend, falling back to ``local`` if it is not available."""
    impl = _IMPLEMENTATIONS.get(backend)
    if impl is None:
        impl = _IMPLEMENTATIONS["local"]
    return impl(frame_bgr, req)
