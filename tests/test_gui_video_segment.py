"""
Tests for video propagation and the embedding provider.

Both are *optional capabilities*. The property that matters most is therefore
not that they work when the model is installed — it is that they fail as a
sentence when it is not, and that YOAKE keeps running either way. That
unavailable path is the one most users hit first, so most of this file drives
it with no model present at all.

The protocol itself is tested against a real subprocess: ``tools/sam_sidecar.py``
under the stub backend, spawned with this interpreter. That covers the codec,
the framing and the unsupported-capability replies without needing torch.

Where SAM 2 *is* installed the ``live_model`` block at the end runs it for
real, against a synthetic scene whose ground truth is known exactly. Those
tests download a checkpoint on first use and are skipped everywhere else::

    pytest tests/test_gui_video_segment.py -q -m "not live_model"
"""

from __future__ import annotations

import importlib.util
import json
import math
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


def _has(module: str) -> bool:
    """Is a module importable here, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _helper_module():
    """``tools/sam_sidecar.py`` as a module, for the parts that are pure logic.

    The helper normally runs as a subprocess under a different interpreter.
    ``resolve_device`` and ``sam2_ready`` are plain functions with no model
    behind them, and testing them through a subprocess would say nothing extra.
    """
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return pytest.importorskip("sam_sidecar")


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

    Spawned with *this* interpreter. Whether that interpreter has torch is a
    property of the machine, not of the suite: on a checkout where the GUI and
    the models share one environment it does, and the optional capabilities are
    then genuinely present. Each test below therefore asks the helper what it
    can do instead of assuming, which is the same rule the GUI itself obeys.
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
    if "embedding" in stub_client.capabilities:
        pytest.skip("this machine has a visual encoder; the available path "
                    "downloads a model and is covered by the live tests")

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


# ══════════════════════════════════════════════════════════════════════════
# Two passes, one tracklet
# ══════════════════════════════════════════════════════════════════════════
#
# SAM 2 propagates in ONE direction from the prompt. Every test here exists
# because a window has frames on both sides of it, and a single forward call
# therefore returns half of what the window cost to export.

def _reply(frames, window=0):
    return {"ok": True, "window": window,
            "items": [{"frame": f, "obb": [10.0 + f, 10.0, 4.0, 2.0, 0.0]}
                      for f in frames]}


def test_merging_two_passes_covers_the_whole_window():
    forward = video_segment.result_from_reply(_reply(range(20, 40), window=40))
    backward = video_segment.result_from_reply(_reply(range(0, 21), window=40))
    assert forward.count == 20 and backward.count == 21

    merged = video_segment.merge_results(forward, backward)
    assert merged.frames == tuple(range(40)), "the two halves must join up"
    assert merged.count == 40


def test_the_merged_window_is_the_maximum_not_the_sum():
    """Both passes run over the same exported directory. Summing would make
    describe() claim the model lost an animal it never lost."""
    merged = video_segment.merge_results(
        video_segment.result_from_reply(_reply(range(20, 40), window=40)),
        video_segment.result_from_reply(_reply(range(0, 21), window=40)))
    assert merged.window == 40
    assert "lost it" not in merged.describe(track_id=1)


def test_the_prompt_frame_resolves_to_the_first_pass():
    """Both passes return the prompt frame. The caller passes forward first."""
    forward = video_segment.result_from_reply(
        {"ok": True, "items": [{"frame": 20, "obb": [1.0, 1.0, 4.0, 2.0, 0.0]}]})
    backward = video_segment.result_from_reply(
        {"ok": True, "items": [{"frame": 20, "obb": [9.0, 9.0, 4.0, 2.0, 0.0]}]})
    merged = video_segment.merge_results(forward, backward)
    assert merged.count == 1, "the shared frame must not be duplicated"
    assert merged.boxes[0][1][0] == pytest.approx(1.0)


def test_one_failed_pass_keeps_the_other_halfs_boxes():
    """Half a tracklet a human can accept beats an error message."""
    good = video_segment.result_from_reply(_reply(range(10), window=20))
    bad = video_segment.result_from_reply({"ok": False, "detail": "boom"})
    merged = video_segment.merge_results(good, bad)
    assert merged.ok and merged.count == 10


def test_merging_only_failures_passes_the_failure_through():
    bad = video_segment.result_from_reply({"ok": False, "detail": "boom"})
    merged = video_segment.merge_results(bad, None)
    assert not merged.ok and merged.detail == "boom"


def test_merging_nothing_is_a_failure_not_an_empty_success():
    merged = video_segment.merge_results(None, None)
    assert not merged.ok and merged.detail


def test_a_propagated_merge_is_still_one_tracklet():
    """The unit of propagation is the unit of review — across both passes."""
    from htrtdetr.annotation import tracklet

    merged = video_segment.merge_results(
        video_segment.result_from_reply(_reply(range(20, 40), window=40)),
        video_segment.result_from_reply(_reply(range(0, 21), window=40)))
    parts = tracklet.build_tracklets(
        video_segment.build_candidates(merged, track_id=4))
    assert len(parts) == 1 and parts[0].count == 40


# ══════════════════════════════════════════════════════════════════════════
# Installing it, and running it on the right device
# ══════════════════════════════════════════════════════════════════════════

def test_the_install_sentence_names_both_packages():
    """sam2 does not depend on huggingface_hub, and reaches its weights only
    through it. Naming one package sends the user to a broken install."""
    assert "sam2" in video_segment.UNAVAILABLE_MESSAGE
    assert "huggingface_hub" in video_segment.UNAVAILABLE_MESSAGE


def test_sam2_without_huggingface_hub_is_not_offered_as_a_backend():
    half = sidecar.InterpreterInfo(python="p", ok=True, torch=True,
                                   modules=("torch", "sam2"))
    assert "sam2" not in half.backends
    assert half.sam2_needs_hub

    whole = sidecar.InterpreterInfo(
        python="p", ok=True, torch=True,
        modules=("torch", "sam2", "huggingface_hub"))
    assert whole.backends[0] == "sam2"
    assert not whole.sam2_needs_hub


def test_a_half_installed_sam2_is_diagnosed_by_name():
    """The most confusing state there is: the package is present, so the user
    believes the install worked, and every load fails inside sam2."""
    reason = sidecar.locked_reason([
        sidecar.InterpreterInfo(python=r"C:\env\python.exe", ok=True,
                                torch=True, modules=("torch", "sam2"))])
    assert "huggingface_hub" in reason
    assert r"C:\env\python.exe" in reason, "name the interpreter to install into"


def test_the_helper_is_asked_for_auto_not_for_cpu():
    """Only the helper has torch, so only the helper can see a GPU. A spec
    that hardcodes cpu runs video propagation an order of magnitude slower on
    a machine with a GPU, and says nothing about it."""
    spec = sidecar.SidecarSpec(python="python", repo_root=str(ROOT))
    assert spec.device == "auto"
    assert "--device" in spec.argv() and "auto" in spec.argv()


def test_device_auto_resolves_in_the_helper_not_in_the_gui():
    helper = _helper_module()
    assert helper.resolve_device("cpu") == "cpu"
    assert helper.resolve_device("cuda") == "cuda", "an explicit device is honoured"
    assert helper.resolve_device("auto") in ("cuda", "mps", "cpu")
    # No device at all means auto, not cpu — the old default was the bug.
    assert helper.resolve_device("") == helper.resolve_device("auto")


def test_sam2_needs_both_packages_before_the_helper_offers_video():
    """``available()`` gates the ``video`` capability. Advertising a capability
    that cannot load is worse than not advertising it."""
    helper = _helper_module()
    assert helper.sam2_ready() == (
        _has("sam2") and _has("huggingface_hub")
    )


# ══════════════════════════════════════════════════════════════════════════
# Live SAM 2
# ══════════════════════════════════════════════════════════════════════════
#
# Everything above runs with no model. These run the real thing against a
# scene whose ground truth is exact, so a regression in the mask-to-box rule,
# the frame offsets or the merge shows up as a number rather than as a shrug.
#
# Skipped unless sam2 and huggingface_hub are both importable here. First run
# downloads ~150 MB of checkpoint.

_SAM2_HERE = _has("sam2") and _has("huggingface_hub") and _has("torch")


def live_model(test):
    """Tag it *and* gate it.

    The tag is what makes ``-m "not live_model"`` work on a machine that does
    have the model; the gate is what makes the file pass on one that does not.
    A skipif alone would leave the tests untaggable, which is how a CI run ends
    up silently downloading a checkpoint.
    """
    test = pytest.mark.live_model(test)
    return pytest.mark.skipif(
        not _SAM2_HERE,
        reason="sam2 + huggingface_hub not installed in this interpreter")(test)

#: The synthetic scene. Small and short on purpose: the point is correctness of
#: the plumbing, and a 60-frame window already exercises both passes.
_N = 30
_SIZE = 256
_RX, _RY = 22, 12
_PROMPT = 15  # the offset the "human" prompts on: frames exist on both sides


def _truth_obb(i: int):
    """Where the ellipse is on frame ``i``, as ``(cx, cy, w, h, theta)``."""
    return [40.0 + 4.0 * i,
            128.0 + 34.0 * math.sin(i / 5.0),
            2.0 * _RX, 2.0 * _RY,
            math.radians(i * 3.0)]


def _write_scene(folder: Path) -> None:
    rng = np.random.default_rng(0)
    background = rng.integers(20, 60, (_SIZE, _SIZE, 3), dtype=np.uint8)
    for i in range(_N):
        frame = background.copy()
        cx, cy, _w, _h, theta = _truth_obb(i)
        cv2.ellipse(frame, (int(cx), int(cy)), (_RX, _RY),
                    math.degrees(theta), 0, 360, (235, 230, 220), -1)
        cv2.imwrite(str(folder / f"{i:05d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])


@pytest.fixture(scope="module")
def sam2_scene(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sam2_scene")
    _write_scene(folder)
    return folder


@pytest.fixture(scope="module")
def sam2_client():
    """A real helper with the real model, on whatever device it finds."""
    spec = sidecar.SidecarSpec(python=sys.executable, repo_root=str(ROOT),
                               backend="sam2", model="facebook/sam2-hiera-tiny")
    client = sidecar.SidecarClient(spec)
    try:
        client.start(timeout=300)
    except sidecar.SidecarError as exc:
        pytest.skip(f"sam2 helper would not start: {exc}")
    yield client
    client.stop()


def _propagate(client, folder, *, backward, first=1000, timeout=900):
    centre = _truth_obb(_PROMPT)
    request = client.propagate(
        frames_dir=str(folder), first_frame=first,
        prompt_frame=first + _PROMPT,
        points=[[centre[0], centre[1], 1]],
        obj_id=3, backward=backward)
    reply = client.wait_for(request, timeout=timeout)
    assert reply is not None, "the helper must answer"
    return video_segment.result_from_reply(reply)


@pytest.fixture(scope="module")
def sam2_passes(sam2_client, sam2_scene):
    """Both passes, run once. Four tests below read the same two results.

    Module-scoped because a pass is a real model over a real window: running
    it per test would multiply the file's runtime by the number of assertions
    made about the same propagation, which is the wrong trade for a check that
    is deterministic given the seeded scene.
    """
    forward = _propagate(sam2_client, sam2_scene, backward=False)
    backward = _propagate(sam2_client, sam2_scene, backward=True)
    return forward, backward


@live_model
def test_a_real_sam2_helper_advertises_video_and_picks_a_device(sam2_client):
    assert "video" in sam2_client.capabilities
    cap = video_segment.capability(sam2_client)
    assert cap.available and cap.device in ("cuda", "mps", "cpu")
    assert cap.describe().startswith("sam2 on ")


@live_model
def test_one_forward_pass_covers_only_the_frames_after_the_prompt(sam2_passes):
    """The reason merge_results exists, measured rather than assumed."""
    forward, _backward = sam2_passes
    assert forward.ok and forward.count
    assert min(forward.frames) == 1000 + _PROMPT, (
        "a forward pass must start at the prompt; if it starts earlier, SAM 2 "
        "changed and merge_results is doing the wrong thing")
    assert forward.count < _N, "the window has frames before the prompt"


@live_model
def test_both_passes_merge_into_one_gapless_tracklet(sam2_passes):
    from htrtdetr.annotation import tracklet

    merged = video_segment.merge_results(*sam2_passes)

    assert merged.frames == tuple(range(1000, 1000 + _N)), "a gap in the merge"
    assert merged.window == _N
    assert "lost it" not in merged.describe(track_id=3)

    parts = tracklet.build_tracklets(
        video_segment.build_candidates(merged, track_id=3))
    assert len(parts) == 1 and parts[0].count == _N


@live_model
def test_the_propagated_boxes_land_on_the_animal(sam2_passes):
    """Rotated-box IoU against exact truth.

    Measured 0.88 mean / 0.86 worst on a 5060 Ti with sam2-hiera-tiny. The
    thresholds are deliberately well below that: this asserts the boxes are on
    the animal and on the right *frame*, not that the model has not improved.
    An off-by-one in the window offsets shows up here as a collapse, because
    the ellipse moves 4 px a frame.
    """
    from htrtdetr.gui.auto_label import iou_obb

    merged = video_segment.merge_results(*sam2_passes)

    ious = [iou_obb(obb, _truth_obb(frame - 1000))
            for frame, obb, _bbox in merged.boxes]
    assert len(ious) == _N
    assert min(ious) > 0.6, f"worst box is off the animal: {min(ious):.3f}"
    assert sum(ious) / len(ious) > 0.75


@live_model
def test_a_clean_propagation_raises_no_review_items(sam2_passes):
    """The quality checks triage the model's own output. A track this steady
    must not generate questions, or the Review Inbox is noise."""
    from htrtdetr.annotation import quality

    merged = video_segment.merge_results(*sam2_passes)
    candidates = video_segment.build_candidates(merged, track_id=3)

    items = quality.analyze(
        candidates,
        config=quality.QualityConfig(expected_animals=1),
        frames=list(range(1000, 1000 + _N)),
        num_frames=1000 + _N)
    assert items == [], f"unexpected review items: {[i.reason for i in items]}"
