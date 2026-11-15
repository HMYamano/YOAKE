"""
tests/test_realtime.py — multiprocessing のリアルタイム推論 (inference/realtime.py)

最重要の検証: **ファイル入力ではリアルタイム経路の結果が逐次の ``yoake predict`` と
完全に一致する**こと。前処理を別プロセスで回しても、同じ関数で同じ入力を作るので
一致するはずで、もしズレるならプロセス分割が結果を変えている。

あわせて、ライブ入力向けの「詰まったら捨てる」がフレーム番号を正直に残すこと、
どこかの段が落ちても固まらずに例外になることを固定する。子プロセスは spawn で
起動するので、1 回の実行に数秒かかる。

設計: docs/REALTIME.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from htrtdetr.config.config import HTRTDETRConfig, InferenceConfig, build_model_config  # noqa: E402
from htrtdetr.inference.realtime import (  # noqa: E402
    FrameRing,
    RealtimeOptions,
    is_live_source,
    run_realtime,
)
from htrtdetr.inference.streaming import (  # noqa: E402
    StreamingInferencer,
    StreamingResult,
    prepare_input,
)
from htrtdetr.models import build_model  # noqa: E402

ACTIONS = ["idle", "walk", "groom", "court", "other"]
IMAGE_SIZE = (64, 64)


def _write_video(path: Path, frames: int = 24, W: int = 160, H: int = 120) -> Path:
    """暗い楕円 (ハエ) が 3 匹、右へ歩く動画。MJPG なら Windows / Linux どちらでも書ける。"""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    for f in range(frames):
        img = np.full((H, W, 3), 200, np.uint8)
        for k in range(3):
            cv2.ellipse(img, (30 + 45 * k + f, 40 + 20 * k), (12, 5), 15 * k, 0, 360,
                        (40, 40, 40), -1)
        writer.write(img)
    writer.release()
    return path


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.head.num_queries = 6
    cfg.detector.score_threshold = 0.0       # 常に検出が出るようにする
    m = build_model(cfg)
    m.set_stage(4)
    m.eval()
    return m


def _inferencer(model) -> StreamingInferencer:
    return StreamingInferencer(
        model,
        InferenceConfig(use_amp=False, use_channels_last=False),
        action_names=ACTIONS,
        window_size=6,
        image_size=IMAGE_SIZE,
    )


class _FakeInferencer:
    """モデル無しで配管だけを試すための推論器 (main プロセスに留まる)。"""

    def __init__(self, delay: float = 0.0, corners=None):
        self.delay = delay
        self.corners = corners
        self.frames = []

    def process_prepared(self, prepared, frame_shape, frame_index=None):
        assert prepared.shape == (*IMAGE_SIZE, 3) and prepared.dtype == np.float32
        if self.delay:
            time.sleep(self.delay)
        self.frames.append(frame_index)
        corners = self.corners
        if corners is None:
            corners = np.array([[[20, 30], [50, 30], [50, 42], [20, 42]]], np.float32)
        return StreamingResult(
            frame_index, np.array([[20, 30, 50, 42]], np.float32),
            corners=corners, track_ids=np.array([1]),
        )

    def runtime_stats(self):
        return {"frames": len(self.frames)}


def _frames_of(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["_meta"], {k: v for k, v in data.items() if k != "_meta"}


# --------------------------------------------------------------------------- #
# 部品
# --------------------------------------------------------------------------- #

def test_live_sources_are_cameras_and_streams(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert is_live_source("0") and is_live_source(1)
    assert is_live_source("rtsp://192.168.0.2/stream") and is_live_source("http://cam/x")
    assert not is_live_source(str(tmp_path))
    (tmp_path / "3").mkdir()
    assert not is_live_source("3")           # 同名のフォルダがあればそちら


def test_frame_ring_shares_slots_between_handles():
    ring = FrameRing.create(3, (5, 7, 3), (4, 4, 3))
    try:
        other = FrameRing.attach(ring.spec)
        ring.frame(2)[...] = 9
        ring.model_input(1)[...] = 0.5
        assert int(other.frame(2).sum()) == 9 * 5 * 7 * 3
        assert float(other.model_input(1).mean()) == 0.5
        assert int(other.frame(0).sum()) == 0
        assert FrameRing.nbytes(3, (5, 7, 3), (4, 4, 3)) >= 3 * (5 * 7 * 3 + 4 * 4 * 3 * 4)
        other.close()
    finally:
        ring.close()
        ring.unlink()


def test_options_are_validated():
    with pytest.raises(ValueError):
        RealtimeOptions(slots=1)
    with pytest.raises(ValueError):
        RealtimeOptions(drop_frames="sometimes")
    assert RealtimeOptions().drops(live=True) and not RealtimeOptions().drops(live=False)
    assert RealtimeOptions(drop_frames="when_full").drops(live=False)
    # 自動: ライブ入力は浅く (遅延を抑える)、ファイルは深く (揺らぎを吸収する)
    assert RealtimeOptions().slot_count(live=True) == 4
    assert RealtimeOptions().slot_count(live=False) == 8
    assert RealtimeOptions(slots=3).slot_count(live=True) == 3


def test_the_writer_stamps_live_frames_with_capture_time(tmp_path):
    """ライブ入力: 各フレームに捕まえた時刻 ``t``、``_meta.fps`` は実測の取り込み速度。

    カメラ無しで確かめるため、writer を子プロセスではなくこのプロセスで直接回す
    (キューと停止フラグはスレッド用のもので足りる)。
    """
    import queue as std_queue
    import signal
    import threading

    from htrtdetr.inference import realtime as R

    ring = FrameRing.create(2, (24, 32, 3), (*IMAGE_SIZE, 3))
    ctrl, items, free, reports, errors = (std_queue.Queue() for _ in range(5))
    ctrl.put((ring.spec, {
        "meta": {"fps": 25.0, "realtime": {"live": True}},
        "live": True, "fps": 25.0, "frame_size": (32, 24),
        "action_names": ACTIONS, "class_names": ["0"], "draw": {},
        "write_video": False, "display": False,
    }))
    for slot, index, t_capture in ((0, 0, 100.0), (1, 3, 100.1)):
        items.put((slot, index, t_capture, {
            "json": {"boxes": [], "track_ids": []}, "boxes": np.zeros((0, 4), np.float32),
            "scores": np.zeros(0), "class_ids": np.zeros(0), "track_ids": np.zeros(0),
            "action_ids": np.zeros(0), "corners": None,
        }))
    items.put(None)
    previous = signal.getsignal(signal.SIGINT)
    try:
        R._writer_main(ctrl, items, free, reports, errors, threading.Event(), str(tmp_path))
    finally:
        signal.signal(signal.SIGINT, previous)
        ring.close()
        ring.unlink()

    assert errors.empty()
    data = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))
    assert data["0"]["t"] == 0.0 and data["3"]["t"] == pytest.approx(0.1)
    assert data["_meta"]["fps"] == pytest.approx(30.0)          # 3 フレーム / 0.1 秒
    assert data["_meta"]["realtime"]["frames_written"] == 2
    assert sorted(free.get_nowait() for _ in range(2)) == [0, 1]  # スロットは返る


def test_live_frames_reads_a_stream_like_source(tmp_path):
    from htrtdetr.inference import realtime as R

    video = _write_video(tmp_path / "clip.avi", frames=6)
    frames = list(R._live_frames(str(video), max_frames=4))
    assert [index for index, _frame, _fps in frames] == [0, 1, 2, 3]
    assert frames[0][1].shape == (120, 160, 3) and frames[0][2] == pytest.approx(30.0)


def test_prepare_input_is_exactly_what_the_detector_is_fed(model):
    si = _inferencer(model)
    frame = np.random.default_rng(3).integers(0, 255, (48, 80, 3), dtype=np.uint8)
    direct = si.preprocess(frame)
    out = np.empty((*IMAGE_SIZE, 3), np.float32)
    staged = si._to_device(prepare_input(frame, IMAGE_SIZE, out=out))
    assert torch.equal(direct, staged)


def test_process_prepared_matches_process_frame(model):
    rng = np.random.default_rng(5)
    frames = [rng.integers(0, 255, (48, 80, 3), dtype=np.uint8) for _ in range(5)]
    a, b = _inferencer(model), _inferencer(model)
    for i, frame in enumerate(frames):
        ra = a.process_frame(frame, frame_index=i)
        rb = b.process_prepared(prepare_input(frame, IMAGE_SIZE), frame.shape, frame_index=i)
        assert ra.to_json() == rb.to_json()


# --------------------------------------------------------------------------- #
# パイプライン全体 (子プロセスを起動する)
# --------------------------------------------------------------------------- #

def test_realtime_matches_sequential_predict_exactly(model, tmp_path):
    from htrtdetr import cli

    video = _write_video(tmp_path / "clip.avi")
    cfg = HTRTDETRConfig()
    cfg.model = model.cfg
    cfg.inference.use_amp = False
    cfg.inference.use_channels_last = False
    cfg.inference.geometry = True
    cfg.inference.realtime_slots = 4

    cli._run_predict_streaming(cfg, model, str(video), str(tmp_path / "seq"), IMAGE_SIZE,
                               ACTIONS, ["0"], 6, None)
    cli._run_predict_realtime(cfg, model, str(video), str(tmp_path / "rt"), IMAGE_SIZE,
                              ACTIONS, ["0"], 6, None)

    seq_meta, seq = _frames_of(tmp_path / "seq" / "predictions.json")
    rt_meta, rt = _frames_of(tmp_path / "rt" / "predictions.json")
    assert len(seq) == 24
    assert rt == seq                                     # 検出・ID・行動・幾何まで全部
    assert all("geometry" in entry for entry in rt.values())
    assert rt_meta["geometry"] == seq_meta["geometry"]
    assert rt_meta["realtime"]["frames_written"] == 24 and not rt_meta["realtime"]["live"]

    lines = (tmp_path / "rt" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["frame"] for line in lines] == list(range(24))
    assert (tmp_path / "rt" / "geometry.csv").read_text(encoding="utf-8") == \
        (tmp_path / "seq" / "geometry.csv").read_text(encoding="utf-8")
    stats = json.loads((tmp_path / "rt" / "runtime_stats.json").read_text(encoding="utf-8"))
    assert stats["frames_processed"] == stats["frames_written"] == 24
    assert stats["frames_dropped"] == 0 and stats["geometry"]["frames"] == 24


def test_dropping_when_full_keeps_frame_numbers_honest(tmp_path):
    """ライブ入力の方針: 下流が詰まったら捨てる。捨てたフレームは欠番として残る。"""
    video = _write_video(tmp_path / "clip.avi", frames=40)
    fake = _FakeInferencer(delay=0.03)
    stats = run_realtime(
        fake, str(video), str(tmp_path / "out"),
        image_size=IMAGE_SIZE, action_names=ACTIONS, class_names=["0"],
        options=RealtimeOptions(slots=2, drop_frames="when_full", write_video=False),
    )
    assert stats["frames_read"] == 40
    assert stats["frames_dropped"] > 0
    assert stats["frames_processed"] + stats["frames_dropped"] == 40
    _meta, frames = _frames_of(tmp_path / "out" / "predictions.json")
    assert sorted(int(k) for k in frames) == fake.frames      # 処理した番号のまま
    assert set(fake.frames) < set(range(40))


def test_max_frames_stops_the_stream(tmp_path):
    video = _write_video(tmp_path / "clip.avi", frames=30)
    fake = _FakeInferencer()
    stats = run_realtime(
        fake, str(video), str(tmp_path / "out"),
        image_size=IMAGE_SIZE, action_names=ACTIONS, class_names=["0"],
        options=RealtimeOptions(max_frames=5, geometry=True, write_video=False),
    )
    assert fake.frames == [0, 1, 2, 3, 4]
    assert stats["frames_written"] == 5
    assert (tmp_path / "out" / "geometry.csv").exists()


def test_a_failing_stage_raises_instead_of_hanging(tmp_path):
    video = _write_video(tmp_path / "clip.avi", frames=20)
    fake = _FakeInferencer(corners="not corners")        # geometry プロセスで落ちる
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="geometry"):
        run_realtime(
            fake, str(video), str(tmp_path / "out"),
            image_size=IMAGE_SIZE, action_names=ACTIONS, class_names=["0"],
            options=RealtimeOptions(geometry=True, write_video=False),
        )
    assert time.monotonic() - t0 < 60.0


def test_a_missing_input_fails_before_any_process_starts(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_realtime(
            _FakeInferencer(), str(tmp_path / "nope.mp4"), str(tmp_path / "out"),
            image_size=IMAGE_SIZE, action_names=ACTIONS, class_names=["0"],
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _patch_predict(monkeypatch, calls):
    import htrtdetr.cli as cli_mod
    import htrtdetr.models as models_mod
    import htrtdetr.utils.misc as misc_mod

    class DummyModel:
        def set_stage(self, stage):
            calls["stage"] = stage

    monkeypatch.setattr(models_mod, "build_model", lambda cfg: DummyModel())
    monkeypatch.setattr(cli_mod, "load_runtime_config_from_checkpoint", lambda path: None)
    monkeypatch.setattr(misc_mod, "load_model_weights", lambda *a, **k: ([], []))

    def fake(name):
        def run(cfg, *args, **kwargs):
            calls[name] = cfg
        return run

    monkeypatch.setattr(cli_mod, "_run_predict_realtime", fake("realtime"))
    monkeypatch.setattr(cli_mod, "_run_predict_streaming", fake("streaming"))


def test_predict_flags_route_to_the_realtime_pipeline(monkeypatch, tmp_path):
    from htrtdetr.cli import run_predict

    calls: dict = {}
    _patch_predict(monkeypatch, calls)
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"dummy")
    run_predict([f"source={tmp_path}", f"weights={ckpt}", "realtime=true", "geometry=true",
                 "geometry_params.min_speed=0.5"])
    cfg = calls["realtime"]
    assert "streaming" not in calls
    assert cfg.inference.realtime and cfg.inference.geometry
    assert cfg.inference.geometry_params == {"min_speed": 0.5}


def test_predict_without_realtime_stays_sequential_and_can_add_geometry(monkeypatch, tmp_path):
    from htrtdetr.cli import run_predict

    calls: dict = {}
    _patch_predict(monkeypatch, calls)
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"dummy")
    run_predict([f"source={tmp_path}", f"weights={ckpt}", "geometry=true"])
    assert "realtime" not in calls
    assert calls["streaming"].inference.geometry


def test_predict_refuses_a_camera_without_realtime(monkeypatch, tmp_path):
    from htrtdetr.cli import run_predict

    calls: dict = {}
    _patch_predict(monkeypatch, calls)
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"dummy")
    with pytest.raises(SystemExit):
        run_predict(["source=0", f"weights={ckpt}"])
    assert "stage" not in calls                  # モデルを読む前に止まる


def test_predict_refuses_unknown_geometry_params(monkeypatch, tmp_path):
    from htrtdetr.cli import run_predict

    calls: dict = {}
    _patch_predict(monkeypatch, calls)
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"dummy")
    with pytest.raises(SystemExit):
        run_predict([f"source={tmp_path}", f"weights={ckpt}", "geometry=true",
                     "geometry_params.min_sped=0.5"])
