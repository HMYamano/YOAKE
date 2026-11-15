"""
test_streaming.py — streaming 推論のテスト

最重要の検証: **streaming と batch で検出結果が一致する**こと。
streaming は過去フレームを detector へ再投入しないので、もし両者がズレるなら
「再計算の省略」が結果を変えてしまっていることになる。

あわせて、detector の呼び出し回数が window 長に依存しないことも固定する
(これが高速化の実体そのもの)。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from htrtdetr.config.config import InferenceConfig, build_model_config
from htrtdetr.inference import StreamingInferencer
from htrtdetr.models import build_model

_H, _W = 96, 128


@pytest.fixture(scope="module")
def model():
    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.head.num_queries = 8
    cfg.detector.score_threshold = 0.0     # 常に検出が出るようにする
    m = build_model(cfg)
    m.set_stage(4)
    m.eval()
    return m


@pytest.fixture(scope="module")
def obb_model():
    cfg = build_model_config("small")
    cfg.detector.backbone.pretrained = False
    cfg.detector.type = "rtdetr_obb"
    cfg.detector.obb.enabled = True
    cfg.detector.head.num_queries = 8
    cfg.detector.score_threshold = 0.0
    m = build_model(cfg)
    m.set_stage(4)
    m.eval()
    return m


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(7)
    return [rng.integers(0, 255, (_H, _W, 3), dtype=np.uint8) for _ in range(10)]


def _make(model, window=6):
    return StreamingInferencer(
        model,
        InferenceConfig(use_amp=False, use_channels_last=False),
        action_names=["idle", "walk", "groom", "court", "other"],
        window_size=window,
        image_size=(64, 64),
    )


class TestStreamingBasics:
    def test_process_frame_returns_detections(self, model, frames):
        si = _make(model)
        r = si.process_frame(frames[0], frame_index=0)
        assert len(r) > 0
        assert r.boxes_xyxy.shape[1] == 4
        assert len(r.track_ids) == len(r)

    def test_boxes_are_in_original_pixel_coords(self, model, frames):
        si = _make(model)
        r = si.process_frame(frames[0], frame_index=0)
        # 元フレームのサイズにスケールされている (入力の 64x64 ではない)
        assert r.boxes_xyxy[:, 0].max() <= _W + 1
        assert r.boxes_xyxy[:, 1].max() <= _H + 1

    def test_reset_clears_tracks(self, model, frames):
        si = _make(model)
        for i, f in enumerate(frames[:3]):
            si.process_frame(f, frame_index=i)
        assert si.tracker.stats()["num_tracks"] > 0
        si.reset()
        assert si.tracker.stats()["num_tracks"] == 0
        assert si.tracker.stats()["next_id"] == 0

    def test_latency_breakdown_has_all_stages(self, model, frames):
        si = _make(model)
        for i, f in enumerate(frames[:3]):
            si.process_frame(f, frame_index=i)
        stats = si.runtime_stats()
        for stage in ("preprocess", "detector", "tracking", "temporal", "behavior"):
            assert f"{stage}_ms" in stats
        assert stats["frames"] == 3
        assert stats["fps"] > 0

    def test_run_over_sequence(self, model, frames):
        si = _make(model)
        results = si.run(list(enumerate(frames)))
        assert len(results) == len(frames)
        assert all(len(r) > 0 for r in results.values())

    def test_json_serializable(self, model, frames):
        import json
        si = _make(model)
        r = si.process_frame(frames[0], frame_index=0)
        json.dumps(r.to_json())   # 例外が出なければよい


class TestStreamingOBB:
    def test_obb_and_corners_present(self, obb_model, frames):
        si = _make(obb_model)
        r = si.process_frame(frames[0], frame_index=0)
        assert r.obb is not None and r.obb.shape[1] == 5
        assert r.corners is not None and r.corners.shape[1:] == (4, 2)

    def test_obb_in_pixel_coords(self, obb_model, frames):
        si = _make(obb_model)
        r = si.process_frame(frames[0], frame_index=0)
        assert r.obb[:, 0].max() <= _W + 1
        assert r.obb[:, 1].max() <= _H + 1

    def test_context_boxes_are_larger(self, obb_model, frames):
        """行動 context box は個体 OBB より広い (設計原則 2)。"""
        si = _make(obb_model)
        r = si.process_frame(frames[0], frame_index=0)
        if r.context_boxes is None:
            pytest.skip("context stream disabled")
        ctx_w = r.context_boxes[:, 2] - r.context_boxes[:, 0]
        assert np.all(ctx_w >= r.obb[:, 2] * 0.99)


class TestStreamingVsBatch:
    """
    streaming と batch の一致検証。

    検出 (bbox / score) は**厳密に一致**しなければならない: どちらも同じフレームを
    同じ detector に通すだけであり、streaming が省くのは「過去フレームの再計算」
    だけだからである。
    """

    def _batch_detect(self, model, si, frames, window):
        """旧方式: 毎フレーム窓全体をモデルに通す。"""
        out = []
        buf = []
        with torch.inference_mode():
            for f in frames:
                buf.append(si.preprocess(f))
                if len(buf) > window:
                    buf.pop(0)
                images = torch.cat(buf, dim=0).unsqueeze(0)
                res = model(images)
                det = res.det_results[0]
                out.append(det["boxes"].clone())
        return out

    def test_detections_match_exactly(self, model, frames):
        window = 4
        si = _make(model, window=window)
        batch_boxes = self._batch_detect(model, si, frames, window)

        si.reset()
        stream_boxes = []
        for i, f in enumerate(frames):
            r = si.process_frame(f, frame_index=i)
            stream_boxes.append(r)

        for i, (b, s) in enumerate(zip(batch_boxes, stream_boxes)):
            assert len(s) == b.shape[0], f"frame {i}: 検出数が違う"
            # streaming は画素座標なので正規化に戻して比較
            sx = s.boxes_xyxy.copy()
            sx[:, [0, 2]] /= _W
            sx[:, [1, 3]] /= _H
            from htrtdetr.utils.misc import cxcywh_to_xyxy
            bx = cxcywh_to_xyxy(b).cpu().numpy()
            assert np.allclose(sx, bx, atol=1e-4), f"frame {i}: bbox が一致しない"

    def test_detector_calls_are_independent_of_window(self, model, frames):
        """
        streaming の detector 呼び出し回数はフレーム数に等しく、window に依存しない。
        batch では window 倍に増える — これが高速化の実体。
        """
        calls = {"n": 0}
        orig = model.detector.forward

        def counting(x):
            calls["n"] += 1
            return orig(x)

        model.detector.forward = counting
        try:
            for window in (2, 8):
                calls["n"] = 0
                si = _make(model, window=window)
                for i, f in enumerate(frames):
                    si.process_frame(f, frame_index=i)
                assert calls["n"] == len(frames), (
                    f"window={window} で detector が {calls['n']} 回呼ばれた "
                    f"(期待 {len(frames)})"
                )
        finally:
            model.detector.forward = orig

    def test_streaming_is_faster_than_sliding_window(self, model, frames):
        """streaming が旧 sliding-window より速いこと (回数比がそのまま効く)。"""
        import time
        window = 8
        si = _make(model, window=window)

        t0 = time.perf_counter()
        self._batch_detect(model, si, frames, window)
        t_batch = time.perf_counter() - t0

        si.reset()
        t0 = time.perf_counter()
        for i, f in enumerate(frames):
            si.process_frame(f, frame_index=i)
        t_stream = time.perf_counter() - t0

        assert t_stream < t_batch, (
            f"streaming ({t_stream:.3f}s) が batch ({t_batch:.3f}s) より遅い"
        )


class TestStreamingEdgeCases:
    def test_empty_frames_expire_tracker_embeddings(self, model, frames, monkeypatch):
        si = _make(model)
        si.tracker.cache.max_age = 0
        si.tracker.cache.reid_max_age = 1
        si.process_frame(frames[0])
        assert si.tracker._embeddings
        monkeypatch.setattr(model.cfg.detector, "score_threshold", 1.1)
        for frame in frames[1:3]:
            assert len(si.process_frame(frame)) == 0
        assert len(si.tracker.cache) == 0
        assert si.tracker._embeddings == {}

    def test_pose_features_reach_behavior_in_track_order(self, frames):
        cfg = build_model_config("small")
        cfg.detector.backbone.pretrained = False
        cfg.detector.head.num_queries = 3
        cfg.detector.score_threshold = 0.0
        cfg.detector.pose.enabled = True
        cfg.detector.pose.num_keypoints = 2
        cfg.behavior.use_pose = True
        m = build_model(cfg).eval()
        si = _make(m)
        captured = []

        def capture_pose(module, args, kwargs):
            captured.append(kwargs["keypoints"].detach().clone())

        # Force history order to differ from detection order.
        original_history = si.tracker.track_history

        def reversed_history(stride=1):
            hist, obb, mask, ids = original_history(stride=stride)
            return hist.flip(0), obb.flip(0), mask.flip(0), ids[::-1]

        si.tracker.track_history = reversed_history
        handle = m.behavior.register_forward_pre_hook(capture_pose, with_kwargs=True)
        try:
            with torch.inference_mode():
                det = m.detector(si.preprocess(frames[0]))
                expected = m.det_adapter.extract_valid_detections(
                    det.pred_logits, det.pred_boxes, det.query_features,
                    score_threshold=0.0, pred_keypoints=det.pred_keypoints,
                )[0]["keypoints"]
            result = si.process_frame(frames[0])
        finally:
            handle.remove()
        assert result.action_probs is not None
        assert torch.allclose(captured[0], expected.flip(0))

    def test_no_detections_frame(self, frames):
        cfg = build_model_config("small")
        cfg.detector.backbone.pretrained = False
        cfg.detector.head.num_queries = 8
        cfg.detector.score_threshold = 1.1      # 何も通さない
        m = build_model(cfg)
        m.set_stage(4)
        m.eval()
        si = _make(m)
        r = si.process_frame(frames[0], frame_index=0)
        assert len(r) == 0
        assert r.boxes_xyxy.shape == (0, 4)

    def test_single_frame_sequence(self, model, frames):
        si = _make(model, window=8)
        r = si.process_frame(frames[0], frame_index=0)
        assert len(r) > 0, "窓が埋まる前でも推論できること"

    def test_frame_index_autoincrements(self, model, frames):
        si = _make(model)
        idx = [si.process_frame(f).frame_index for f in frames[:4]]
        assert idx == [0, 1, 2, 3]
