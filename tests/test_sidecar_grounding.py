"""Open-vocabulary detection: the op that had a protocol and no model.

``concept`` - *Find all in frame* in the annotator - has always had a UI, a
protocol op, a capability guard and a review flow. The only adapter that
implemented it was ``StubAdapter``, so on every real backend the button
reported that the backend could not find animals from a description. SAM
cannot: it is promptable and answers "what is at this point", never "where are
the flies".

These tests cover the composition that fills the gap - a detector finds the
boxes, the image adapter turns each into a mask, and ``mask_to_obb`` gives the
rotation the detector has no concept of - without needing the weights.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

sam_sidecar = pytest.importorskip("sam_sidecar")

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class FakeBoxAdapter:
    """An image adapter that can take a box prompt, like SAM 2 can."""

    name = "fake"
    capabilities = ("point", "box")
    device = "cpu"

    def __init__(self):
        self.prompts = []

    def set_image(self, frame_bgr):
        self.shape = frame_bgr.shape[:2]

    def point(self, points, box):
        import numpy as np

        self.prompts.append((tuple(points), box))
        # A filled rectangle standing in for the mask SAM would return.
        mask = np.zeros((64, 64), dtype=bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask[y0:y1, x0:x1] = True
        return mask, 0.9


class FakePointOnlyAdapter(FakeBoxAdapter):
    """No box prompt - the degrade path."""

    capabilities = ("point",)


class FakeGrounder:
    """Stands in for OWLv2 so the composition is testable with no weights."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.seen = []
        self.images = 0

    def available(self):
        return True

    def set_image(self, frame_bgr):
        self.images += 1

    def detect(self, text, limit, threshold=0.05):
        self.seen.append((text, limit, threshold))
        return self.boxes[: int(limit)]


def session_with(adapter, grounder, *, frame=0):
    import numpy as np

    session = sam_sidecar.Session(adapter, device="cpu")
    session._ground = grounder
    session.frame = frame
    session._last_array = np.zeros((64, 64, 3), dtype=np.uint8)
    session.shape = (64, 64)
    return session


# ── the adapter's own contract ───────────────────────────────────────────

def test_available_tracks_whether_transformers_is_installed():
    adapter = sam_sidecar.OwlGroundingAdapter()
    assert adapter.available() is sam_sidecar.grounding_ready()


def test_detect_returns_nothing_before_a_frame_arrives():
    """The failure that made this look like "no animals here".

    ``Session.load`` was never called on the concept path, so a real detector
    had no image and answered with an empty list - indistinguishable, in the
    UI, from a frame with nothing in it.
    """
    assert sam_sidecar.OwlGroundingAdapter().detect("fly", 8) == []


def test_an_empty_description_is_not_a_query():
    adapter = sam_sidecar.OwlGroundingAdapter()
    adapter._image = object()      # past the no-frame guard
    assert adapter.detect("   ,  , ", 8) == []


# ── Session.ground: the composition ──────────────────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="needs numpy")
def test_ground_prompts_the_image_adapter_with_box_and_centre():
    adapter = FakeBoxAdapter()
    grounder = FakeGrounder([((10.0, 20.0, 30.0, 40.0), 0.7)])
    session = session_with(adapter, grounder)

    out = session.ground("fly", 8)

    assert grounder.images == 1, "the detector was not shown the frame"
    (points, box), = adapter.prompts
    assert box == (10.0, 20.0, 30.0, 40.0)
    # Box plus centre point: a box-only prompt is not well formed for every
    # adapter here, and the centre costs nothing.
    assert points == ((20.0, 30.0, 1),)
    assert len(out) == 1 and out[0][1] == 0.7


@pytest.mark.skipif(not HAS_NUMPY, reason="needs numpy")
def test_ground_degrades_to_a_flat_box_without_a_segmenter():
    """No box prompt available: return the axis-aligned box, rotation zero.

    A human can rotate a box that is in the right place. Returning nothing
    because the orientation is unknown would throw away the useful part.
    """
    grounder = FakeGrounder([((10.0, 20.0, 30.0, 40.0), 0.7)])
    session = session_with(FakePointOnlyAdapter(), grounder)

    (shape, score), = session.ground("fly", 8)

    assert shape[0] == "obb"
    cx, cy, w, h, theta = shape[1]
    assert (cx, cy, w, h) == (20.0, 30.0, 20.0, 20.0)
    assert theta == 0.0
    assert score == 0.7


@pytest.mark.skipif(not HAS_NUMPY, reason="needs numpy")
def test_ground_survives_a_segmenter_that_raises():
    class Exploding(FakeBoxAdapter):
        def point(self, points, box):
            raise RuntimeError("out of memory")

    grounder = FakeGrounder([((10.0, 20.0, 30.0, 40.0), 0.7)])
    session = session_with(Exploding(), grounder)

    (shape, _), = session.ground("fly", 8)
    assert shape[0] == "obb", "a failed refinement must not lose the detection"


@pytest.mark.skipif(not HAS_NUMPY, reason="needs numpy")
def test_the_threshold_reaches_the_detector():
    grounder = FakeGrounder([])
    session = session_with(FakeBoxAdapter(), grounder)
    session.ground("fly", 4, 0.25)
    assert grounder.seen == [("fly", 4, 0.25)]


# ── the handler: the missing frame load ──────────────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="needs numpy")
def test_concept_without_a_path_says_so_instead_of_finding_nothing():
    # StubAdapter implements concept itself, so it still answers.
    session = sam_sidecar.Session(sam_sidecar.StubAdapter(), device="cpu")
    session.frame = None
    reply = sam_sidecar.handle(
        {"op": "concept", "id": 1, "frame": 5, "text": "fly"}, session, None
    )
    assert reply["ok"] is True

    # An adapter that needs the grounder must be told the frame is missing
    # rather than reporting an empty frame.
    session = sam_sidecar.Session(FakeBoxAdapter(), device="cpu")
    session._ground = FakeGrounder([])
    session.frame = None
    reply = sam_sidecar.handle(
        {"op": "concept", "id": 2, "frame": 5, "text": "fly"}, session, None
    )
    assert reply["ok"] is False
    assert reply["error"] == "no_frame"


def test_concept_names_the_install_when_nothing_can_find_animals():
    session = sam_sidecar.Session(FakeBoxAdapter(), device="cpu")
    session.grounder = lambda: None
    reply = sam_sidecar.handle(
        {"op": "concept", "id": 3, "frame": 0, "text": "fly"}, session, None
    )
    assert reply["ok"] is False and reply["error"] == "unsupported"
    detail = reply["detail"]
    assert "pip install transformers" in detail
    # It must also say what still works, or the message is just a dead end.
    assert "Continue" in detail


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="needs transformers")
def test_concept_is_advertised_once_a_detector_is_installed():
    session = sam_sidecar.Session(sam_sidecar.StubAdapter(), device="cpu")
    assert "concept" in session.extra_capabilities()


# ── the real thing, over the pipe, with the real weights ─────────────────

@pytest.mark.live_model
def test_a_real_helper_finds_and_orients_an_animal_from_a_description():
    """End to end: text in, an oriented box on the animal out.

    The assertion is about the *pipeline*, not about recall. OWLv2 is a
    photograph model and these are drawn ellipses, so it fires on some and not
    others; measured here it found one of three planted shapes. What must hold
    is that a detection which does arrive is placed by the detector and
    oriented by SAM 2 - if the composition were broken, every box would come
    back axis-aligned or in the wrong place.
    """
    import math

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    sys.path.insert(0, str(ROOT / "src"))
    from htrtdetr.gui import sidecar

    size = 512
    centre = (300, 300)
    angle_deg = -40.0
    frame = np.full((size, size, 3), 210, np.uint8)
    cv2.ellipse(frame, centre, (46, 16), angle_deg, 0, 360, (35, 30, 30), -1)
    cv2.ellipse(frame, centre, (14, 13), angle_deg, 0, 360, (10, 10, 10), -1)

    spec = sidecar.SidecarSpec(
        python=sys.executable, repo_root=str(ROOT), backend="sam2",
        model="facebook/sam2-hiera-tiny",
    )
    client = sidecar.SidecarClient(spec)
    try:
        hello = client.start(timeout=300)
    except sidecar.SidecarError as exc:
        pytest.skip(f"helper would not start: {exc}")
    try:
        if "concept" not in (hello.get("capabilities") or []):
            pytest.skip("this helper advertises no open-vocabulary detector")
        path = client.frames.put(0, frame)
        request = client.send(
            "concept", frame=0, path=str(path), text="insect, fly", max=8
        )
        reply = client.wait_for(request, timeout=900)
    finally:
        client.stop()

    assert reply is not None, "the helper must answer"
    assert reply.get("ok"), reply.get("detail")
    items = reply.get("items") or []
    if not items:
        pytest.skip("the detector scored nothing above threshold on drawn shapes")

    on_target = [
        i for i in items
        if abs(i["obb"][0] - centre[0]) < 40 and abs(i["obb"][1] - centre[1]) < 40
    ]
    assert on_target, f"nothing landed on the animal: {[i['obb'] for i in items]}"

    cx, cy, w, h, theta = on_target[0]["obb"]
    assert abs(cx - centre[0]) < 10 and abs(cy - centre[1]) < 10
    # The drawn body is 92 x 32; the fit should be in that neighbourhood and
    # elongated the same way round.
    assert max(w, h) > min(w, h) * 1.5
    # theta is radians. A zero here would mean the box never reached
    # mask_to_obb - the detector alone has no notion of orientation.
    assert abs(math.degrees(theta) - angle_deg) < 20, (
        f"expected about {angle_deg} degrees, got {math.degrees(theta):.1f}"
    )
