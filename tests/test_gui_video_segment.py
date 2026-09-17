"""
Tests for video propagation and the embedding provider.

Both are *optional capabilities*. The property that matters most is therefore
not that they work when the model is installed — it is that they fail as a
sentence when it is not, and that YOAKE keeps running either way. This machine
has no SAM 2, so the unavailable path is the one exercised here, which is
exactly the path most users will hit first.

The protocol itself is tested against a real subprocess: ``tools/sam_sidecar.py``
under the stub backend, spawned with this interpreter. That covers the codec,
the framing and the unsupported-capability replies without needing torch.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("dearpygui.dearpygui")

from htrtdetr.gui import embedding, sidecar, video_segment  # noqa: E402

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


# ══════════════════════════════════════════════════════════════════════════
# Protocol
# ══════════════════════════════════════════════════════════════════════════

def test_the_new_ops_are_declared():
    assert "propagate" in sidecar.OPS
    assert "embed_frames" in sidecar.OPS
    assert "video" in sidecar.CAPABILITIES
    assert "embedding" in sidecar.CAPABILITIES


def test_protocol_version_did_not_change():
    """Adding an op is additive: an old helper answers ``unknown_op`` and the
    GUI reports the capability as missing. Bumping the version would instead
    make the two refuse each other outright, for no reason."""
    assert sidecar.PROTOCOL_VERSION == 1


def test_a_propagate_request_is_well_formed():
    message = sidecar.request_message(
        "propagate", 7, frames_dir="/tmp/w", first_frame=100, frame=140,
        points=[[10.0, 20.0, 1]], obj_id=3, backward=False)
    assert message["op"] == "propagate" and message["id"] == 7
    assert message["first_frame"] == 100 and message["obj_id"] == 3
    # None-valued fields are dropped, so an unused prompt does not travel.
    assert "box" not in message
    assert json.loads(sidecar.encode(message)) == message


# ══════════════════════════════════════════════════════════════════════════
# Capability reporting
# ══════════════════════════════════════════════════════════════════════════

def test_no_helper_reports_unavailable_with_a_next_step():
    cap = video_segment.capability(None)
    assert not cap
    assert "Settings" in cap.reason, "a locked feature must name the way to it"


def test_a_helper_without_video_is_unavailable_and_says_which_package():
    client = SimpleNamespace(is_running=lambda: True,
                             hello={"backend": "sam2", "capabilities": ["point"]})
    cap = video_segment.capability(client)
    assert not cap
    assert "sam2" in cap.reason.lower()
    assert "still work" in cap.reason, "must say what is unaffected"
    assert cap.describe() == "Unavailable - SAM2 backend not installed"


def test_capability_is_read_from_hello_not_from_the_backend_name():
    """A helper called sam2 whose install is broken advertises no video
    capability, and that is the answer that matters."""
    named = SimpleNamespace(is_running=lambda: True,
                            hello={"backend": "sam2", "capabilities": []})
    assert not video_segment.capability(named)

    capable = SimpleNamespace(
        is_running=lambda: True,
        hello={"backend": "sam2", "capabilities": ["point", "video"],
               "device": "cuda"})
    cap = video_segment.capability(capable)
    assert cap and cap.device == "cuda"
    assert "sam2" in cap.describe()


# ══════════════════════════════════════════════════════════════════════════
# The window
# ══════════════════════════════════════════════════════════════════════════

def test_the_window_is_clamped_to_the_video_and_to_the_maximum():
    request = video_segment.PropagationRequest(frame=10, before=100, after=100)
    assert request.window(num_frames=50) == (0, 49)

    wide = video_segment.PropagationRequest(frame=5000, before=5000, after=5000)
    first, last = wide.window(num_frames=100000)
    assert last - first + 1 <= video_segment.MAX_WINDOW


def test_an_empty_prompt_is_recognised():
    assert video_segment.PropagationRequest(frame=0).is_empty()
    assert not video_segment.PropagationRequest(
        frame=0, points=((1.0, 2.0, 1),)).is_empty()


def test_exported_frames_are_zero_padded(tmp_path):
    """A plain sort puts 10.jpg before 9.jpg, which would land every
    propagated box on the wrong frame - silently and consistently."""
    def read(index):
        return np.full((24, 32, 3), index % 255, np.uint8)

    folder, written = video_segment.export_window(read, 0, 11, root=tmp_path)
    assert folder is not None and written == 12
    names = sorted(p.name for p in folder.glob("*.jpg"))
    assert names[0] == "00000.jpg" and names[-1] == "00011.jpg"
    assert names == [f"{i:05d}.jpg" for i in range(12)]
    shutil.rmtree(folder, ignore_errors=True)


def test_export_stops_at_a_hole_rather_than_shifting_everything(tmp_path):
    def read(index):
        return None if index == 5 else np.zeros((16, 16, 3), np.uint8)

    folder, written = video_segment.export_window(read, 0, 9, root=tmp_path)
    assert written == 5, "frames after the hole would have the wrong offset"
    shutil.rmtree(folder, ignore_errors=True)


def test_export_of_nothing_returns_none_rather_than_raising(tmp_path):
    folder, written = video_segment.export_window(
        lambda _i: None, 0, 4, root=tmp_path)
    assert folder is None and written == 0


# ══════════════════════════════════════════════════════════════════════════
# Replies
# ══════════════════════════════════════════════════════════════════════════

def test_an_unsupported_reply_becomes_the_install_sentence():
    result = video_segment.result_from_reply(
        {"ok": False, "error": "unsupported", "detail": ""})
    assert not result.ok
    assert "SAM 2 is not installed" in result.detail


def test_a_good_reply_becomes_sorted_boxes():
    reply = {"ok": True, "window": 3, "items": [
        {"frame": 12, "obb": [10, 10, 4, 2, 0.0], "bbox": [8, 9, 12, 11]},
        {"frame": 10, "obb": [20, 20, 4, 2, 0.0], "bbox": [18, 19, 22, 21]},
    ]}
    result = video_segment.result_from_reply(reply)
    assert result.ok and result.frames == (10, 12)


def test_a_missing_bbox_is_derived_from_the_obb():
    reply = {"ok": True, "items": [
        {"frame": 0, "obb": [100.0, 100.0, 20.0, 10.0, 0.0]}]}
    result = video_segment.result_from_reply(reply)
    _frame, _obb, bbox = result.boxes[0]
    assert bbox == pytest.approx([90.0, 95.0, 110.0, 105.0])


def test_a_partial_propagation_says_it_lost_the_animal():
    reply = {"ok": True, "window": 200, "items": [
        {"frame": f, "obb": [10, 10, 4, 2, 0.0]} for f in range(40)]}
    text = video_segment.result_from_reply(reply).describe(track_id=3)
    assert "lost it after 40 of 200" in text


def test_an_empty_propagation_suggests_a_better_prompt():
    text = video_segment.result_from_reply({"ok": True, "items": []}).describe()
    assert "point nearer its centre" in text


def test_candidates_are_tracked_not_auto():
    """The subject was chosen by a human, so the provenance is 'track'. The
    distinction tells a later reader whether a human picked the animal."""
    reply = {"ok": True, "items": [
        {"frame": f, "obb": [10.0, 10.0, 4.0, 2.0, 0.0]} for f in range(5)]}
    result = video_segment.result_from_reply(reply)
    candidates = video_segment.build_candidates(result, track_id=7)
    assert set(candidates) == set(range(5))
    one = candidates[0][0]
    assert one.source == "track" and one.track_id == 7
    assert one.obb is not None


def test_propagated_candidates_group_into_one_tracklet():
    """The unit of propagation is the unit of review."""
    from htrtdetr.annotation import tracklet

    reply = {"ok": True, "items": [
        {"frame": f, "obb": [10.0 + f, 10.0, 4.0, 2.0, 0.0]} for f in range(30)]}
    candidates = video_segment.build_candidates(
        video_segment.result_from_reply(reply), track_id=4)
    parts = tracklet.build_tracklets(candidates)
    assert len(parts) == 1
    assert parts[0].count == 30 and parts[0].track_id == 4


# ══════════════════════════════════════════════════════════════════════════
# Embedding providers
# ══════════════════════════════════════════════════════════════════════════

def test_the_thumbnail_provider_is_always_available():
    provider = embedding.ThumbnailProvider()
    info = provider.info()
    assert info.available and info.embedding_dimension == 256


def test_the_thumbnail_provider_encodes_real_frames():
    rng = np.random.default_rng(0)

    def read(index):
        return rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)

    vectors = embedding.ThumbnailProvider().encode_frames(read, [0, 3, 6])
    assert set(vectors) == {0, 3, 6}
    assert all(len(v) == 256 for v in vectors.values())
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-6 for v in vectors.values())


def test_an_unreadable_frame_is_absent_not_zero_filled():
    """A zero vector sits at distance 1 from everything and diversity sampling
    would pick it first."""
    vectors = embedding.ThumbnailProvider().encode_frames(lambda _i: None, [0, 1])
    assert vectors == {}


def test_provider_falls_back_when_the_helper_cannot_embed():
    client = SimpleNamespace(is_running=lambda: True,
                             hello={"capabilities": ["point"]})
    provider = embedding.provider_for(client)
    assert isinstance(provider, embedding.ThumbnailProvider)


def test_provider_uses_the_helper_when_it_advertises_embedding():
    client = SimpleNamespace(is_running=lambda: True,
                             hello={"capabilities": ["point", "embedding"],
                                    "backend": "dino", "device": "cuda"})
    assert isinstance(embedding.provider_for(client), embedding.SidecarProvider)


def test_provider_falls_back_when_no_helper_is_running():
    client = SimpleNamespace(is_running=lambda: False, hello={})
    assert isinstance(embedding.provider_for(client), embedding.ThumbnailProvider)
    assert isinstance(embedding.provider_for(None), embedding.ThumbnailProvider)


def test_every_provider_satisfies_the_same_interface():
    """Swapping the encoder must change no caller."""
    for provider in (embedding.NullProvider(), embedding.ThumbnailProvider()):
        info = provider.info()
        assert isinstance(info.model_name, str)
        assert isinstance(info.embedding_dimension, int)
        assert isinstance(info.describe(), str)
        assert provider.encode_frames(lambda _i: None, []) == {}


def test_the_null_provider_reports_itself_honestly():
    info = embedding.NullProvider().info()
    assert not info.available
    assert info.describe() == "No visual encoder"


# ══════════════════════════════════════════════════════════════════════════
# A real helper subprocess
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def stub_client():
    """The real ``tools/sam_sidecar.py`` under the stub backend.

    Spawned with *this* interpreter, which has no torch — so the two optional
    capabilities are genuinely absent and their unavailable paths are the ones
    under test.
    """
    spec = sidecar.SidecarSpec(python=sys.executable, repo_root=str(ROOT),
                               backend="stub")
    client = sidecar.SidecarClient(spec)
    try:
        client.start(timeout=60)
    except sidecar.SidecarError as exc:
        pytest.skip(f"helper would not start: {exc}")
    yield client
    client.stop()


def test_a_real_helper_answers_hello_and_lists_capabilities(stub_client):
    assert stub_client.is_running()
    assert "point" in stub_client.capabilities
    assert stub_client.hello.get("backend") == "stub"


def test_video_is_unavailable_on_a_helper_without_sam2(stub_client):
    cap = video_segment.capability(stub_client)
    if cap.available:
        pytest.skip("this machine has sam2 installed; the available path "
                    "needs a model and weights")
    assert "SAM 2 is not installed" in cap.reason


def test_propagate_against_a_real_helper_fails_as_a_reply(stub_client, tmp_path):
    """The rule the whole sidecar exists for: a failure is a reply, not a crash
    and not a lost helper."""
    def read(index):
        return np.zeros((32, 32, 3), np.uint8)

    folder, written = video_segment.export_window(read, 0, 4, root=tmp_path)
    assert written == 5
    request = stub_client.propagate(frames_dir=str(folder), first_frame=0,
                                    prompt_frame=2, points=[[16.0, 16.0, 1]])
    reply = stub_client.wait_for(request, timeout=60)
    shutil.rmtree(folder, ignore_errors=True)

    assert reply is not None, "the helper must answer, even to say no"
    if not reply.get("ok"):
        assert reply.get("detail"), "a refusal must carry a sentence"
        assert reply.get("error") in ("unsupported", "no_frames",
                                      "propagate_failed")
    assert stub_client.is_running(), "a refused request must not kill the helper"


def test_embed_frames_against_a_real_helper_fails_as_a_reply(stub_client, tmp_path):
    def read(index):
        return np.zeros((32, 32, 3), np.uint8)

    folder, _written = video_segment.export_window(read, 0, 2, root=tmp_path)
    request = stub_client.embed_frames(frames_dir=str(folder), first_frame=0)
    reply = stub_client.wait_for(request, timeout=60)
    shutil.rmtree(folder, ignore_errors=True)

    assert reply is not None
    if not reply.get("ok"):
        assert reply.get("detail")
    assert stub_client.is_running()


def test_an_unknown_op_does_not_kill_the_helper(stub_client):
    request = stub_client.send("drop", frame=999)
    assert stub_client.wait_for(request, timeout=30) is not None
    assert stub_client.is_running()
