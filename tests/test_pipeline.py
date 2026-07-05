"""
test_pipeline.py — レイヤードパイプライン (pipeline/ + stages/) のテスト

モデル/重み不要の古典段だけで、schema・registry・resolver・runner・YAML config を
検証する。神経系段 (rtdetr/fused/memory_id) は build まで (setup 前) をテストする。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from htrtdetr import pipeline as P
import htrtdetr.stages  # noqa: F401 (registers backends)
from htrtdetr.pipeline import PipelineContext, PipelineRunner, create
from htrtdetr.pipeline.resolver import InvalidPipelineError, resolve
from htrtdetr.pipeline.schema import (
    CAP_BOXES, CAP_CLASS, CAP_DET_SCORE, CAP_IMAGE, CAP_KINEMATICS, CAP_TRACK_ID,
    Detection, FrameResult,
)

REPO = Path(__file__).resolve().parents[1]


def _synthetic_frames(n=10, drop_at=5):
    frames = []
    for t in range(n):
        ax, bx = 10 + t * 3, 40 + t * 3
        dets = [
            Detection(bbox=np.array([ax, 45, ax + 14, 59], np.float32), score=0.9, class_id=0),
            Detection(bbox=np.array([bx, 45, bx + 14, 59], np.float32), score=0.9, class_id=0),
        ]
        if t == drop_at:
            dets = [dets[0]]  # B が欠損 → 補間テスト
        frames.append(FrameResult(frame_index=t, detections=dets, width=640, height=480))
    return frames


def _classical_stages():
    return [
        create("L1b_stabilization.temporal_nms", iou_link=0.3, min_persist=2, max_gap=1),
        create("L2_static.geometric", near_dist_px=60),
        create("L3_tracking.bytetrack", track_thresh=0.3, match_iou=0.1),
        create("L4_motion.kinematic", smooth_window=3),
        create("L5_interaction.heuristic", max_pair_dist_px=200, min_frames=2),
    ]


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def test_expected_backends_registered():
    backends = set(P.available_backends())
    expected = {
        "L1_detection.rtdetr", "L1_detection.yolo", "L1_detection.fused_htrtdetr",
        "L1b_stabilization.temporal_nms", "L2_static.geometric",
        "L3_tracking.bytetrack", "L3_tracking.iou", "L3_tracking.memory_id",
        "L4_motion.kinematic", "pose.keypoint",
        "L5_interaction.heuristic", "L5_interaction.learned_pair", "L5_interaction.gnn",
    }
    assert expected <= backends


def test_unknown_backend_raises():
    with pytest.raises(P.UnknownBackendError):
        create("L1_detection.does_not_exist")


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
def test_topological_order_tier_and_after():
    runner = PipelineRunner(
        _classical_stages(),
        inputs={CAP_BOXES, CAP_CLASS, CAP_DET_SCORE},
        mode="strict",
        ctx=PipelineContext(device="cpu", window_size=0),
    )
    names = [s.name for s in runner.runnable]
    # L1b(stab) は L2 より前 (tier 同点だが after/tier で前に来る)
    assert names.index("L1b_stabilization.temporal_nms") < names.index("L2_static.geometric")
    # 追跡 → 運動 → 相互作用 の順
    assert names.index("L3_tracking.bytetrack") < names.index("L4_motion.kinematic")
    assert names.index("L4_motion.kinematic") < names.index("L5_interaction.heuristic")


def test_strict_raises_on_missing_capability():
    stages = [create("L1_detection.yolo"), create("L3_tracking.memory_id")]
    with pytest.raises(InvalidPipelineError):
        PipelineRunner(stages, inputs={CAP_IMAGE}, mode="strict")


def test_lenient_cascade_skip():
    stages = [
        create("L1_detection.yolo"),
        create("L3_tracking.memory_id"),   # needs dense_features → skip
        create("L4_motion.kinematic"),     # needs track_id → cascade skip
    ]
    runner = PipelineRunner(stages, inputs={CAP_IMAGE}, mode="lenient")
    runnable = {s.name for s in runner.runnable}
    skipped = {n for n, _ in runner.skipped}
    assert runnable == {"L1_detection.yolo"}
    assert "L3_tracking.memory_id" in skipped
    assert "L4_motion.kinematic" in skipped


def test_cycle_detection():
    # after で相互参照させて循環を作る
    a = create("L2_static.geometric")
    b = create("L2_static.geometric")
    a.name, b.name = "cyc.a", "cyc.b"
    a.after = frozenset({"cyc.b"})
    b.after = frozenset({"cyc.a"})
    with pytest.raises(InvalidPipelineError):
        resolve([a, b], inputs={CAP_BOXES})


# --------------------------------------------------------------------------- #
# end-to-end (classical)
# --------------------------------------------------------------------------- #
def test_full_classical_pipeline_runs():
    runner = PipelineRunner(
        _classical_stages(),
        inputs={CAP_BOXES, CAP_CLASS, CAP_DET_SCORE},
        mode="strict",
        ctx=PipelineContext(device="cpu", window_size=0),
    )
    out = list(runner.run(_synthetic_frames()))[0]

    # 2 トラックが窓全体で維持される
    assert set(out.tracks) == {0, 1}
    for tr in out.tracks.values():
        assert len(tr.frames) >= 9
        assert tr.speeds is not None and tr.velocities is not None
    # 欠損した B が補間で復元されている
    assert any(d.interpolated for _, d in out.iter_detections())
    # L2 関係と L5 有向相互作用が生成される
    assert sum(len(f.relations) for f in out.frames) > 0
    assert len(out.interactions) > 0
    for it in out.interactions:
        assert it.agent_track_id != it.patient_track_id
    assert CAP_KINEMATICS in out.available and CAP_TRACK_ID in out.available


def test_stabilization_removes_isolated_detection():
    frames = [
        FrameResult(frame_index=0, detections=[
            Detection(bbox=np.array([10, 10, 20, 20], np.float32), score=0.9, class_id=0)],
            width=640, height=480),
        # t=1: 孤立した誤検出 (前後に対応なし)
        FrameResult(frame_index=1, detections=[
            Detection(bbox=np.array([10, 10, 20, 20], np.float32), score=0.9, class_id=0),
            Detection(bbox=np.array([300, 300, 320, 320], np.float32), score=0.5, class_id=0)],
            width=640, height=480),
        FrameResult(frame_index=2, detections=[
            Detection(bbox=np.array([11, 11, 21, 21], np.float32), score=0.9, class_id=0)],
            width=640, height=480),
    ]
    stage = create("L1b_stabilization.temporal_nms", iou_link=0.3, min_persist=2, max_gap=0)
    runner = PipelineRunner([stage], inputs={CAP_BOXES, CAP_DET_SCORE}, mode="strict",
                            ctx=PipelineContext(device="cpu", window_size=0))
    out = list(runner.run(frames))[0]
    # 孤立検出 (300,300) は除去される
    boxes = [d.bbox.tolist() for _, d in out.iter_detections()]
    assert [300.0, 300.0, 320.0, 320.0] not in boxes


# --------------------------------------------------------------------------- #
# YAML configs
# --------------------------------------------------------------------------- #
def _chase_frames(n=10):
    frames = []
    for t in range(n):
        ax, bx = 10 + t * 4, 45 + t * 4   # A behind B, both moving +x fast
        frames.append(FrameResult(frame_index=t, width=640, height=480, detections=[
            Detection(bbox=np.array([ax, 45, ax + 14, 59], np.float32), score=0.9, class_id=0),
            Detection(bbox=np.array([bx, 45, bx + 14, 59], np.float32), score=0.9, class_id=0),
        ]))
    return frames


def test_heuristic_detects_chase_with_correct_agent():
    # Regression: `behind` feature was geometrically inverted -> chase never fired.
    stages = [
        create("L3_tracking.bytetrack", track_thresh=0.3, match_iou=0.1),
        create("L4_motion.kinematic", smooth_window=3),
        create("L5_interaction.heuristic", max_pair_dist_px=200, min_frames=2, court_speed_px=2.0),
    ]
    runner = PipelineRunner(stages, inputs={CAP_BOXES, CAP_CLASS, CAP_DET_SCORE}, mode="strict",
                            ctx=PipelineContext(device="cpu", window_size=0))
    out = list(runner.run(_chase_frames()))[0]
    chases = [i for i in out.interactions if i.relation == "chase"]
    assert chases, f"expected a chase, got {[i.relation for i in out.interactions]}"
    # agent は後方の追う個体 (track 0), patient は逃げる個体 (track 1)
    assert chases[0].agent_track_id == 0 and chases[0].patient_track_id == 1


def test_pair_feature_dim_consistent():
    from htrtdetr.stages.interaction.pair_features import PAIR_FEAT_DIM, PAIR_FEAT_NAMES
    assert PAIR_FEAT_DIM == 10
    assert len(PAIR_FEAT_NAMES) == PAIR_FEAT_DIM
    assert "iou" not in PAIR_FEAT_NAMES  # dead constant-zero dimension removed


def test_kinematic_smooth_window_coerced_odd():
    st = create("L4_motion.kinematic", smooth_window=4)
    assert st.smooth_window == 5  # even -> next odd (avoids half-frame lag)


def test_run_default_is_single_window_even_when_ctx_window_size_set():
    # Regression: outer split must NOT fall back to ctx.window_size (that is for the
    # fused stage's internal buffer). Default = 1 video = 1 window.
    stages = [create("L3_tracking.bytetrack", track_thresh=0.3, match_iou=0.1)]
    runner = PipelineRunner(stages, inputs={CAP_BOXES, CAP_CLASS, CAP_DET_SCORE}, mode="strict",
                            ctx=PipelineContext(device="cpu", window_size=4, stride=2))
    windows = list(runner.run(_chase_frames(10)))
    assert len(windows) == 1
    assert len(windows[0].frames) == 10


def test_explicit_windows_are_non_overlapping_and_dedup():
    from htrtdetr.pipeline.io import merge_windows
    stages = [
        create("L3_tracking.bytetrack", track_thresh=0.3, match_iou=0.1),
        create("L4_motion.kinematic", smooth_window=3),
        create("L5_interaction.heuristic", max_pair_dist_px=200, min_frames=2),
    ]
    runner = PipelineRunner(stages, inputs={CAP_BOXES, CAP_CLASS, CAP_DET_SCORE}, mode="strict",
                            ctx=PipelineContext(device="cpu"))
    frames = _chase_frames(12)
    # explicit window_size=4 with stride=2 must be forced NON-overlapping (stride>=ws)
    windows = list(runner.run(frames, window_size=4, stride=2))
    covered = [fr.frame_index for w in windows for fr in w.frames]
    assert sorted(covered) == list(range(12))          # every frame once, no overlap
    assert len(covered) == 12
    merged = merge_windows(windows)
    keys = [(i.agent_track_id, i.patient_track_id, i.relation, i.t_start, i.t_end)
            for i in merged.interactions]
    assert len(keys) == len(set(keys))                 # no duplicate interactions


@pytest.mark.parametrize("name", [
    "detect_only", "temporal_detect", "detect_track", "full_offline", "yolo_bytetrack",
])
def test_pipeline_yaml_configs_build(name):
    """各 config が (重み無しでも) 段を組み立て・依存解決できる。"""
    pytest.importorskip("yaml")
    yaml_path = REPO / "configs" / "pipeline" / f"{name}.yaml"
    assert yaml_path.exists(), yaml_path
    runner = P.build_from_yaml(str(yaml_path))
    assert isinstance(runner, PipelineRunner)
    assert len(runner.runnable) >= 1
