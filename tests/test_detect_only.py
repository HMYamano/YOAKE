"""
test_detect_only.py — the boxes-only, batched detection path.

The active-learning loop asks the model where the animals are and nothing
else. :mod:`htrtdetr.inference.detect_only` answers that question without the
tracker, the temporal module, the behavior heads or the overlay video, which
is what lets the detector be batched — measured 36.3 fps to 308.0 fps through
``yoake predict`` on the same weights.

The bargain only holds if the boxes are the same boxes. That is what most of
this file checks:

* a batch of N frames finds what N batches of one find (to fp16 precision);
* the JSON is the schema ``load_candidates`` already reads;
* identity is ``-1`` and says so, rather than being invented;
* no ``output.mp4``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from htrtdetr.config.config import build_model_config
from htrtdetr.gui import auto_label
from htrtdetr.inference import BatchedDetector, run_detect_only
from htrtdetr.models import build_model

_H, _W = 96, 128


@pytest.fixture(scope="module")
def model():
    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.head.num_queries = 8
    cfg.detector.score_threshold = 0.0  # every query comes through
    m = build_model(cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8) for _ in range(7)]


def _runner(model, **kw):
    return BatchedDetector(
        model, image_size=(64, 64), score_threshold=0.0,
        device=torch.device("cpu"), use_amp=False, use_channels_last=False, **kw
    )


# ══════════════════════════════════════════════════════════════════════════
# The bargain: batching must not change the answer
# ══════════════════════════════════════════════════════════════════════════

def test_a_batch_finds_what_one_at_a_time_finds(model, frames):
    """The detector is per-frame, so the batch size must not be visible.

    This is the whole justification for the module. If batching moved the
    boxes, the speed-up would be a different model rather than the same one
    run better.
    """
    sizes = [f.shape[:2] for f in frames]
    together = _runner(model, batch_size=len(frames)).detect_batch(frames, sizes)
    alone = [
        _runner(model, batch_size=1).detect_batch([f], [s])[0]
        for f, s in zip(frames, sizes)
    ]
    for one, many in zip(alone, together):
        assert len(one["boxes"]) == len(many["boxes"])
        assert np.allclose(one["boxes"], many["boxes"], atol=1e-3)
        assert np.allclose(one["scores"], many["scores"], atol=1e-5)


def test_the_batch_size_does_not_change_how_many_frames_come_back(model, frames):
    sizes = [f.shape[:2] for f in frames]

    def indices(batch):
        gen = ((i, f, 30.0) for i, f in enumerate(frames))
        return sorted(_runner(model, batch_size=batch).run(gen), key=int)

    assert indices(1) == indices(3) == indices(16) == [str(i) for i in range(len(frames))]
    del sizes


def test_a_partial_last_batch_is_not_dropped(model, frames):
    """7 frames at batch 3 is 2 full batches and a remainder of one."""
    gen = ((i, f, 30.0) for i, f in enumerate(frames))
    out = _runner(model, batch_size=3).run(gen)
    assert len(out) == 7


# ══════════════════════════════════════════════════════════════════════════
# The file it writes
# ══════════════════════════════════════════════════════════════════════════

def test_the_json_is_the_schema_the_annotator_already_reads(model, frames, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    cv2 = pytest.importorskip("cv2")
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:06d}.jpg"), f)

    out = tmp_path / "out"
    run_detect_only(
        model, str(folder), str(out), image_size=(64, 64),
        score_threshold=0.0, batch_size=4, use_amp=False, use_channels_last=False,
    )
    payload = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    assert len(payload) == len(frames)
    row = payload["0"]
    assert set(row) >= {"boxes", "scores", "track_ids", "action_ids", "class_ids", "obb"}

    # The real check: the annotator's own loader accepts it without a branch.
    candidates = auto_label.load_candidates(out / "predictions.json")
    assert len(candidates) == len(frames)
    assert all(isinstance(c, auto_label.Candidate) for cs in candidates.values() for c in cs)


def test_it_does_not_write_an_overlay_video(model, frames, tmp_path):
    """For 20 hour-long recordings at stride 15 that is 144,000 frames drawn
    and encoded into a file nobody opens."""
    folder = tmp_path / "frames"
    folder.mkdir()
    cv2 = pytest.importorskip("cv2")
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:06d}.jpg"), f)
    out = tmp_path / "out"
    run_detect_only(model, str(folder), str(out), image_size=(64, 64),
                    score_threshold=0.0, batch_size=4, use_amp=False,
                    use_channels_last=False)
    assert not (out / "output.mp4").exists()
    assert (out / "runtime_stats.json").exists()


def test_the_runtime_stats_say_which_mode_ran(model, frames, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    cv2 = pytest.importorskip("cv2")
    for i, f in enumerate(frames):
        cv2.imwrite(str(folder / f"{i:06d}.jpg"), f)
    out = tmp_path / "out"
    run_detect_only(model, str(folder), str(out), image_size=(64, 64),
                    score_threshold=0.0, batch_size=4, use_amp=False,
                    use_channels_last=False)
    stats = json.loads((out / "runtime_stats.json").read_text(encoding="utf-8"))
    assert stats["mode"] == "detect_only"
    assert stats["frames"] == len(frames)
    assert stats["batch_size"] == 4


# ══════════════════════════════════════════════════════════════════════════
# Identity, which this path deliberately does not have
# ══════════════════════════════════════════════════════════════════════════

def test_identity_is_absent_rather_than_invented(model, frames):
    """``-1`` is the honest answer with no tracker in the loop.

    It also matters downstream: ``connect_detections`` links exactly the rows
    whose ``track_id`` is negative, so an invented id here would be an
    invented id that nothing ever revisits.
    """
    sizes = [f.shape[:2] for f in frames]
    rows = _runner(model, batch_size=4).detect_batch(frames, sizes)
    for row in rows:
        assert set(row["track_ids"]) <= {-1}
        assert set(row["action_ids"]) <= {-1}


def test_connect_detections_supplies_the_ids_for_a_dense_pass(model, frames):
    """The dense path's identity comes from IoU linking, and can.

    A strided pass has none to give - detections 15 frames apart are not
    neighbours - which is why ``connect_detections`` is gated on stride 1.
    """
    from htrtdetr.annotation.batch import connect_detections

    candidates = {
        i: [auto_label.Candidate(bbox=[10, 10, 30, 30], score=0.9, track_id=-1)]
        for i in range(4)
    }
    linked = connect_detections(candidates)
    ids = {c.track_id for row in linked.values() for c in row}
    assert ids and -1 not in ids, "the linker left detections without identity"
    assert len(ids) == 1, "a stationary animal over 4 frames is one track"


# ══════════════════════════════════════════════════════════════════════════
# How the annotator asks for it
# ══════════════════════════════════════════════════════════════════════════

def test_predict_argv_asks_for_detect_only_and_a_batch():
    argv = auto_label.predict_argv(auto_label.PredictRequest(
        source="clip.mp4", weights="best.pth", output_dir="out",
        detect_only=True, batch_size=8,
    ))
    assert "detect_only=true" in argv
    assert "batch_size=8" in argv


def test_predict_argv_stays_as_it_was_when_detect_only_is_off():
    """A caller that has not asked for it must produce the old command."""
    argv = auto_label.predict_argv(auto_label.PredictRequest(
        source="clip.mp4", weights="best.pth", output_dir="out",
    ))
    assert not any(a.startswith(("detect_only", "batch_size")) for a in argv)


def test_the_annotation_loop_asks_for_it_by_default():
    from htrtdetr.annotation.batch import BatchOptions

    assert BatchOptions().detect_only is True


def test_the_batch_size_is_left_to_the_runner_by_default():
    """0 means "the measured default", not "one frame at a time"."""
    from htrtdetr.annotation.batch import BatchOptions

    assert BatchOptions().detect_batch == 0
    argv = auto_label.predict_argv(auto_label.PredictRequest(
        source="s", weights="w", output_dir="o", detect_only=True, batch_size=0,
    ))
    assert "detect_only=true" in argv
    assert not any(a.startswith("batch_size") for a in argv)
