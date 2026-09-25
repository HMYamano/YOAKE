"""
tests/test_gui_auto_label.py — the auto-annotation engine, headless.

No display, no dearpygui, no torch. Every test here pins a property some
consumer depends on; where a test exists because a real bug was measured, the
docstring says which one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from htrtdetr.gui import auto_label as al

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Capability probing
# --------------------------------------------------------------------------- #


def test_probe_never_imports_torch(monkeypatch):
    original = al._has_module
    monkeypatch.setattr(
        al, "_has_module", lambda name: False if name == "torch" else original(name)
    )
    """Importing torch costs seconds and pulls CUDA DLLs; find_spec must be enough."""
    # Put torch back afterwards. On a machine that *has* torch, leaving it out
    # of sys.modules poisons the rest of the session: torch cannot be imported
    # twice in one process (it re-registers its "triton" library and raises),
    # and even a reference held from before the pop can no longer make a tensor
    # because the C extension looks itself up in sys.modules. The assertion
    # below still runs against the popped state, so nothing is weakened.
    saved = sys.modules.pop("torch", None)
    try:
        cap = al.probe(REPO_ROOT)
        assert "torch" not in sys.modules
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
    # This machine has no torch, so the locked path must be the one exercised.
    assert cap.torch is False
    assert cap.can_detect is False
    assert "pip install torch torchvision" in cap.torch_reason
    assert "Import Predictions as Candidates" in cap.torch_reason


def test_module_never_imports_dearpygui_or_torch():
    """auto_label must stay unit-testable headlessly, and torch-free forever.

    An AST scan, not a substring search: the docstring legitimately names both.
    """
    import ast

    src = (REPO_ROOT / "src" / "htrtdetr" / "gui" / "auto_label.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "dearpygui" not in imported
    assert "torch" not in imported
    assert "subprocess" not in imported  # this module builds argv, it never spawns


def test_probe_finds_src_layout(monkeypatch):
    """htrtdetr uninstalled but <root>/src/htrtdetr present -> package_via == 'src'."""
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "htrtdetr":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    cap = al.probe(REPO_ROOT)
    assert cap.package is True
    assert cap.package_via == "src"
    # The src-layout case is fixed silently by child_env(); never surfaced.
    assert cap.package_reason == ""


def test_probe_reports_a_missing_package(monkeypatch, tmp_path):
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name == "htrtdetr":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    cap = al.probe(tmp_path)  # an empty directory: no src/htrtdetr either
    assert cap.package is False
    assert cap.package_via == ""
    assert cap.package_reason == al.PACKAGE_MISSING_MESSAGE


def test_check_backend_no_torch_message(tmp_path, monkeypatch):
    original = al._has_module
    monkeypatch.setattr(
        al, "_has_module", lambda name: False if name == "torch" else original(name)
    )
    """The no-torch path is the common one here, so it must be excellent."""
    status = al.check_backend(REPO_ROOT, checkpoint=str(tmp_path / "best.pth"))
    assert bool(status) is False
    assert status.torch is False
    assert status.message == al.TORCH_MISSING_MESSAGE
    # cause, fix command, escape hatch — all three present.
    assert "not installed in this Python environment" in status.message
    assert "pip install torch torchvision" in status.message
    assert "python -m htrtdetr.cli predict" in status.message


def test_check_backend_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        al,
        "probe",
        lambda root: al.Capability(
            torch=True, package=True, package_via="src", opencv=True, tracker="mil"
        ),
    )
    missing = tmp_path / "nope.pth"
    status = al.check_backend(REPO_ROOT, checkpoint=str(missing))
    assert status.ok is False
    assert "Checkpoint not found" in status.message

    status = al.check_backend(REPO_ROOT, checkpoint="")
    assert status.ok is False
    assert "Select a detector checkpoint first." in status.message

    real = tmp_path / "best.pth"
    real.write_bytes(b"x")
    assert al.check_backend(REPO_ROOT, checkpoint=str(real)).ok is True


# --------------------------------------------------------------------------- #
# Child-process environment
# --------------------------------------------------------------------------- #


def test_child_env_prepends_src(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "already"))
    env = al.child_env(tmp_path)
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert parts[0] == str(tmp_path / "src")
    assert str(tmp_path / "already") in parts  # existing value preserved
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_child_env_actually_imports():
    """Directly proves the ModuleNotFoundError that broke every GUI button is fixed."""
    proc = subprocess.run(
        [sys.executable, "-c", "import htrtdetr"],
        env=al.child_env(REPO_ROOT),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_ensure_child_env_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    al.ensure_child_env(tmp_path)
    first = os.environ["PYTHONPATH"]
    al.ensure_child_env(tmp_path)
    second = os.environ["PYTHONPATH"]
    assert first == second
    assert second.split(os.pathsep).count(str(tmp_path / "src")) == 1


def test_device_env_has_no_cli_flag(monkeypatch):
    """YOAKE has no device= key, so the device is pinned through the environment."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert "CUDA_VISIBLE_DEVICES" not in al.device_env("auto")
    assert al.device_env("cuda:1")["CUDA_VISIBLE_DEVICES"] == "1"


def test_asking_for_the_cpu_actually_hides_the_gpu(monkeypatch):
    """"-1", not "". Measured on this machine with torch installed:

        CUDA_VISIBLE_DEVICES=""    is_available() True,  device_count() 0
        CUDA_VISIBLE_DEVICES="-1"  is_available() False, device_count() 0

    Every runner in this repo picks ``cuda if torch.cuda.is_available() else
    cpu``, so the empty value did not select the processor - it selected a GPU
    that is not there. "Use the GPU" unticked was a crash waiting for a device
    index, in the one situation where the user had already told us the GPU was
    not an option.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert al.device_env("cpu")["CUDA_VISIBLE_DEVICES"] == "-1"


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


def test_predict_argv_exact():
    req = al.PredictRequest(source="v.mp4", weights="best.pth", output_dir="out")
    assert al.predict_argv(req) == [
        "-u",
        "-m",
        "htrtdetr.cli",
        "predict",
        "source=v.mp4",
        "weights=best.pth",
        "output_dir=out",
        "score_threshold=0.30",
        "window_size=16",
        "inference.show_bbox=false",
        "inference.show_id=false",
        "inference.show_action=false",
        "inference.show_confidence=false",
    ]


def test_predict_argv_optional_tokens():
    argv = al.predict_argv(
        al.PredictRequest(
            source="frames",
            weights="w.pth",
            output_dir="out",
            score_threshold=0.45,
            window_size=8,
            max_frames=60,
            config="cfg.yaml",
            nms_threshold=0.5,
        )
    )
    assert "max_frames=60" in argv
    assert "config=cfg.yaml" in argv
    assert "score_threshold=0.45" in argv
    assert "window_size=8" in argv
    assert "model.detector.nms_threshold=0.50" in argv
    # and they are still absent when unset
    bare = al.predict_argv(al.PredictRequest("v", "w", "o"))
    assert not any(t.startswith("max_frames=") for t in bare)
    assert not any(t.startswith("config=") for t in bare)
    assert not any(t.startswith("model.detector.nms") for t in bare)


def test_build_command_is_executable_plus_argv():
    req = al.PredictRequest(source="v.mp4", weights="w.pth", output_dir="out")
    assert al.build_command(req) == [sys.executable, *al.predict_argv(req)]


def test_auto_label_request_to_predict_request():
    req = al.AutoLabelRequest(
        source="v.mp4",
        weights="w.pth",
        output_dir="out",
        score_threshold=0.25,
        frame_count=120,
    )
    assert req.needs_export is False
    argv = al.predict_argv(req.to_predict_request())
    assert "max_frames=120" in argv
    assert "source=v.mp4" in argv

    # A range that does not start at 0 must be exported to frames first.
    ranged = al.AutoLabelRequest(
        source="v.mp4", weights="w.pth", output_dir="out", start_frame=412, frame_count=30
    )
    assert ranged.needs_export is True
    argv = al.predict_argv(ranged.to_predict_request(source="cache/job1"))
    assert "source=cache/job1" in argv
    assert not any(t.startswith("max_frames=") for t in argv)


def test_auto_label_request_validate():
    problems = al.AutoLabelRequest().validate()
    assert "Open a video first." in problems
    assert "Select a detector checkpoint first." in problems
    bad = al.AutoLabelRequest(
        source="v", weights="w", output_dir="o", score_threshold=1.5
    ).validate()
    assert "Score threshold must be between 0.05 and 0.95." in bad


def test_predictions_path_for(tmp_path):
    assert al.predictions_path_for(tmp_path) == tmp_path / "predictions.json"


# --------------------------------------------------------------------------- #
# Frame export and remapping
# --------------------------------------------------------------------------- #


def test_export_plan_and_index_map(tmp_path):
    plan = al.export_plan(412, 3, num_frames=1000, job_id="job1")
    assert plan.frames == [412, 413, 414]
    assert plan.index_map == {"0": 412, "1": 413, "2": 414}
    assert plan.dir_name == "job1"

    path = al.write_index_map(plan, tmp_path)
    assert al.read_index_map(path) == {"0": 412, "1": 413, "2": 414}
    assert al.read_index_map(tmp_path) == plan.index_map
    assert al.read_index_map(tmp_path / "missing") == {}


def test_export_plan_clamps_and_strides():
    assert al.export_plan(8, 100, num_frames=10, job_id="j").frames == [8, 9]
    assert al.export_plan(0, 3, num_frames=10, job_id="j", stride=4).frames == [0, 4, 8]
    assert al.export_plan(-5, 2, num_frames=10, job_id="j").frames == [0, 1]


def test_remap_frames():
    """Without this every candidate lands on the wrong frame."""
    cands = {
        0: [al.Candidate(bbox=[1, 2, 3, 4])],
        1: [al.Candidate(bbox=[5, 6, 7, 8])],
    }
    out = al.remap_frame_indices(cands, {"0": 412, "1": 413})
    assert sorted(out) == [412, 413]
    assert out[412][0].bbox == [1, 2, 3, 4]
    # An empty map is the identity, so a whole-video run can call it blindly.
    assert sorted(al.remap_frame_indices(cands, {})) == [0, 1]


# --------------------------------------------------------------------------- #
# Reading predictions.json — all three known writers
# --------------------------------------------------------------------------- #


def _write(tmp_path, payload, name="predictions.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_load_candidates_streaming(tmp_path):
    """cli.run_predict default: obb present, class_ids absent."""
    path = _write(
        tmp_path,
        {
            "0": {
                "boxes": [[10, 10, 30, 40], [50, 50, 70, 90]],
                "scores": [0.91, 0.42],
                "track_ids": [1, 2],
                "action_ids": [0, 3],
                "obb": [[20.0, 25.0, 20.0, 30.0, 0.3], [60.0, 70.0, 20.0, 40.0, -0.2]],
                "evidence": [{"motion": 1.2}, {"motion": 0.1}],
            },
        },
    )
    cands = al.load_candidates(path)
    c0, c1 = cands[0]
    assert c0.bbox == [10, 10, 30, 40]
    assert c0.score == pytest.approx(0.91)
    assert c0.track_id == 1 and c0.action_id == 0
    assert c0.class_id == 0  # padded, exactly like predictions.py
    assert c0.obb == [20.0, 25.0, 20.0, 30.0, 0.3]
    assert c1.source == "auto"


def test_load_candidates_batch(tmp_path):
    """inference.mode=batch: four keys only."""
    path = _write(
        tmp_path,
        {
            "7": {
                "boxes": [[0, 0, 10, 10]],
                "scores": [0.55],
                "track_ids": [4],
                "action_ids": [2],
            },
        },
    )
    cands = al.load_candidates(path)
    assert list(cands) == [7]
    c = cands[7][0]
    assert c.obb is None
    assert c.class_id == 0


def test_load_candidates_pipeline(tmp_path):
    """yoake pipeline: class_ids present, action_ids missing -> padded to -1."""
    path = _write(
        tmp_path,
        {
            "3": {
                "boxes": [[1, 1, 9, 9]],
                "scores": [0.8],
                "class_ids": [2],
                "track_ids": [11],
                "interpolated": [False],
                "relations": [],
            },
        },
    )
    c = al.load_candidates(path)[3][0]
    assert c.class_id == 2
    assert c.action_id == -1
    assert c.track_id == 11


def test_load_candidates_pads_missing_parallel_lists(tmp_path):
    path = _write(tmp_path, {"0": {"boxes": [[0, 0, 4, 4], [5, 5, 9, 9]]}})
    a, b = al.load_candidates(path)[0]
    assert (a.score, a.track_id, a.action_id, a.class_id) == (1.0, -1, -1, 0)
    assert (b.score, b.track_id, b.action_id, b.class_id) == (1.0, -1, -1, 0)


def test_load_candidates_min_score_is_inclusive(tmp_path):
    path = _write(
        tmp_path,
        {
            "0": {
                "boxes": [[0, 0, 4, 4], [5, 5, 9, 9], [6, 6, 8, 8]],
                "scores": [0.30, 0.2999, 0.7],
            },
        },
    )
    kept = al.load_candidates(path, min_score=0.30)[0]
    assert [round(c.score, 4) for c in kept] == [0.30, 0.7]


def test_load_candidates_errors_are_readable(tmp_path):
    with pytest.raises(ValueError) as exc:
        al.load_candidates(tmp_path / "predictions.json")
    assert "no predictions.json" in str(exc.value)

    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        al.load_candidates(bad)
    assert "broken.json" in str(exc.value)

    listy = _write(tmp_path, [1, 2, 3], name="list.json")
    with pytest.raises(ValueError):
        al.load_candidates(listy)

    malformed = _write(tmp_path, {"0": {"boxes": [["a", 1, 2, 3]]}}, name="bad2.json")
    with pytest.raises(ValueError):
        al.load_candidates(malformed)


def test_filter_candidates_threshold_class_and_cap():
    cands = {
        0: [
            al.Candidate(bbox=[0, 0, 4, 4], score=0.9, class_id=0),
            al.Candidate(bbox=[5, 5, 9, 9], score=0.6, class_id=1),
            al.Candidate(bbox=[6, 6, 8, 8], score=0.2, class_id=0),
        ]
    }
    assert len(al.filter_candidates(cands, min_score=0.5)[0]) == 2
    assert [c.class_id for c in al.filter_candidates(cands, classes=(1,))[0]] == [1]
    capped = al.filter_candidates(cands, max_per_frame=1)[0]
    assert len(capped) == 1 and capped[0].score == pytest.approx(0.9)
    assert al.filter_candidates(cands, min_score=0.99) == {}


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #


def test_iou_xyxy():
    assert al.iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert al.iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    # 10x10 and 10x10 overlapping by 5x10 -> 50 / 150
    assert al.iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150, abs=1e-9)


def test_iou_obb_matches_axis_aligned_at_theta_zero():
    a = [5.0, 5.0, 10.0, 10.0, 0.0]
    b = [10.0, 5.0, 10.0, 10.0, 0.0]
    assert al.iou_obb(a, a) == pytest.approx(1.0, abs=1e-6)
    assert al.iou_obb(a, b) == pytest.approx(50 / 150, abs=1e-4)
    # a square rotated 45 degrees against itself is still 1.0
    import math

    r = [5.0, 5.0, 10.0, 10.0, math.pi / 4]
    assert al.iou_obb(r, r) == pytest.approx(1.0, abs=1e-6)
    assert al.iou_obb(a, [100.0, 100.0, 4.0, 4.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


def test_iou_any_dispatches_on_the_angle():
    import math

    box = [0, 0, 10, 10]
    obb_a = [5.0, 5.0, 10.0, 2.0, 0.0]
    obb_b = [5.0, 5.0, 10.0, 2.0, math.pi / 2]
    # As axis-aligned boxes these overlap completely; as rotated boxes they cross.
    assert al.iou_any(box, box) == pytest.approx(1.0)
    assert al.iou_any(box, box, obb_a, obb_b) < 0.3


def test_obb_to_xyxy():
    import math

    x1, y1, x2, y2 = al.obb_to_xyxy([0.0, 0.0, 2.0, 2.0, math.pi / 4])
    assert x1 == pytest.approx(-math.sqrt(2), abs=1e-6)
    assert y2 == pytest.approx(math.sqrt(2), abs=1e-6)


# --------------------------------------------------------------------------- #
# Merge policy
# --------------------------------------------------------------------------- #


def _human(bbox, oid=1):
    return {
        "object_id": oid,
        "bbox": list(bbox),
        "class_id": 0,
        "track_id": 1,
        "action_id": -1,
        "occluded": False,
        "is_crowd": False,
        "keyframe": True,
        "accepted": True,
    }


def test_merge_policy_protects_human_frames():
    existing = {5: [_human([0, 0, 10, 10])]}
    incoming = {5: [al.Candidate(bbox=[100, 100, 120, 120], score=0.9)]}
    merged, res = al.merge_into(existing, incoming)
    assert res.added == 0
    assert res.skipped_protected == 1
    assert merged[5] == existing[5]  # human work untouched


def test_merge_policy_iou_suppression():
    """0.61 suppressed, 0.59 kept, at the 0.60 default."""
    policy = al.MergePolicy(skip_frames_with_accepted=False)
    anchor = [0.0, 0.0, 100.0, 100.0]  # area 10000

    def overlap_box(target_iou):
        # b = [0,0,100,w] fully inside a -> iou = w*100 / 10000 = w/100
        w = target_iou * 100.0
        return [0.0, 0.0, 100.0, w]

    high = al.Candidate(bbox=[int(v) for v in overlap_box(0.61)], score=0.9)
    low = al.Candidate(bbox=[int(v) for v in overlap_box(0.59)], score=0.9)
    assert al.iou_xyxy(anchor, high.bbox) >= 0.60
    assert al.iou_xyxy(anchor, low.bbox) < 0.60

    existing = {0: [_human(anchor)]}
    merged, res = al.merge_into(existing, {0: [high]}, policy)
    assert res.added == 0 and res.suppressed_iou == 1

    merged, res = al.merge_into(existing, {0: [low]}, policy)
    assert res.added == 1 and res.suppressed_iou == 0
    assert merged[0][-1]["accepted"] is False
    assert merged[0][-1]["source"] == "auto"


def test_merge_policy_replaces_pending():
    """A second run supersedes the first frame's candidates rather than stacking."""
    policy = al.MergePolicy(skip_frames_with_accepted=False)
    first, _ = al.merge_into({}, {2: [al.Candidate(bbox=[0, 0, 5, 5], score=0.8)]}, policy)
    assert len(first[2]) == 1
    second, res = al.merge_into(first, {2: [al.Candidate(bbox=[9, 9, 14, 14], score=0.7)]}, policy)
    assert len(second[2]) == 1
    assert second[2][0].get("bbox") == [9, 9, 14, 14]
    assert res.replaced_pending == 1


def test_merge_policy_threshold_filtering():
    policy = al.MergePolicy(skip_frames_with_accepted=False, min_score=0.5)
    _, res = al.merge_into({}, {0: [al.Candidate(bbox=[0, 0, 5, 5], score=0.2)]}, policy)
    assert res.added == 0
    assert res.below_threshold == 1


def test_merge_candidates_callables_keep_annobject_out():
    """The four callables are what let the annotator use this without a cycle."""

    class Obj:
        def __init__(self, bbox, accepted=True, source="human"):
            self.bbox, self.accepted, self.source = bbox, accepted, source

    existing = {0: [Obj([0, 0, 10, 10])]}
    merged, res = al.merge_candidates(
        existing,
        {0: [al.Candidate(bbox=[50, 50, 60, 60], score=0.9)]},
        al.MergePolicy(skip_frames_with_accepted=False),
        is_accepted=lambda o: o.accepted,
        is_human=lambda o: o.source == "human",
        bbox_of=lambda o: o.bbox,
        make_obj=lambda c: Obj(c.bbox, accepted=False, source=c.source),
    )
    assert res.added == 1
    assert merged[0][-1].accepted is False
    assert merged[0][-1].source == "auto"


def test_merge_result_message():
    ok = al.MergeResult(
        added=12, frames_touched=4, frames_total=10, skipped_protected=2, suppressed_iou=3
    )
    assert al.describe(ok) == (
        "Auto-annotation done: 12 candidates on 4 of 10 frames "
        "(2 frames protected, 3 overlapped an existing box) — press Ctrl+Right "
        "to jump to the first one, Y to accept a frame, N to discard it."
    )

    assert al.describe(al.MergeResult(added=5), stopped=True) == (
        "Auto-annotation stopped. The 5 candidates already produced are kept."
    )

    # Three distinct empty-result diagnoses, never one generic shrug.
    assert al.describe(al.MergeResult()) == (
        "The run produced no predictions.json. Check the log for the reason."
    )
    assert al.describe(al.MergeResult(frames_total=6, skipped_protected=6)) == (
        "Every frame in the range already has boxes you drew. Untick "
        '"Skip frames that already have boxes I drew" to re-run them.'
    )
    assert al.describe(al.MergeResult(frames_total=6, below_threshold=40)) == (
        'No detections above the score threshold. Lower "Score threshold" and ' "run again."
    )


# --------------------------------------------------------------------------- #
# Provenance and annotator records
# --------------------------------------------------------------------------- #


def test_provenance_fields_keep_human_json_byte_identical():
    assert al.provenance_fields("human", None) == {}
    assert al.provenance_fields("human", 0.9) == {}
    assert al.provenance_fields("auto", 0.8234) == {"source": "auto", "score": 0.8234}
    assert al.provenance_fields("track", None) == {"source": "track"}


def test_predictions_to_annotations_matches_the_annotator_schema():
    cands = {
        4: [al.Candidate(bbox=[10, 20, 30, 40], score=0.82, track_id=3, action_id=1)],
        9: [al.Candidate(bbox=[1, 2, 3, 4], obb=[2.0, 3.0, 2.0, 2.0, 0.5], score=0.6)],
    }
    out = al.predictions_to_annotations(cands, start_object_id=100)
    rec = out[4][0]
    # exactly AnnObject.to_dict()'s keys, plus review state and provenance
    assert set(rec) >= {
        "object_id",
        "bbox",
        "class_id",
        "track_id",
        "action_id",
        "occluded",
        "is_crowd",
        "keyframe",
        "accepted",
        "source",
        "score",
    }
    assert rec["object_id"] == 100
    assert rec["accepted"] is False  # unconfirmed review state
    assert rec["keyframe"] is False
    assert rec["source"] == "auto"
    assert rec["score"] == pytest.approx(0.82)
    assert out[9][0]["object_id"] == 101
    assert out[9][0]["obb"] == {"cx": 2.0, "cy": 3.0, "w": 2.0, "h": 2.0, "angle": 0.5}


def test_annotation_records_load_through_the_library_parser():
    """data/annotation.py ignores unknown keys, so provenance loads harmlessly."""
    from htrtdetr.data.annotation import _parse_object

    rec = al.candidate_to_record(
        al.Candidate(bbox=[1, 2, 30, 40], score=0.7, track_id=5), object_id=1
    )
    parsed = _parse_object(rec, 0)
    assert parsed.bbox == [1.0, 2.0, 30.0, 40.0]
    assert parsed.track_id == 5


# --------------------------------------------------------------------------- #
# The review state machine
# --------------------------------------------------------------------------- #


def _session():
    return al.ReviewSession(
        {
            3: [
                al.Candidate(bbox=[0, 0, 5, 5], score=0.9),
                al.Candidate(bbox=[6, 6, 9, 9], score=0.4),
            ],
            11: [al.Candidate(bbox=[1, 1, 4, 4], score=0.7)],
        }
    )


def test_review_accept_and_reject_single():
    s = _session()
    assert s.state(3, 0) == al.PENDING
    assert s.accept(3, 0) is True
    assert s.state(3, 0) == al.ACCEPTED
    assert s.reject(3, 1) is True
    assert s.counts() == al.ReviewCounts(pending=1, accepted=1, rejected=1)
    assert s.accept(3, 99) is False  # out of range is a no-op, not a crash


def test_review_per_frame_and_batch():
    s = _session()
    assert s.accept_frame(3) == 2
    assert s.counts(3).accepted == 2
    assert s.reject_frame(11) == 1
    assert s.counts().pending == 0
    assert s.accept_frame(3) == 0  # nothing pending left to change


def test_review_accept_above_threshold():
    s = _session()
    assert s.accept_above(0.65) == 2  # 0.9 and 0.7, not 0.4
    assert s.counts().accepted == 2
    assert s.counts().pending == 1


def test_review_pending_frame_navigation():
    """Ctrl+Right / Ctrl+Left is what makes 400 candidates reviewable."""
    s = _session()
    assert s.pending_frames() == [3, 11]
    assert s.next_pending_frame(0) == 3
    assert s.next_pending_frame(3) == 11
    assert s.next_pending_frame(11) is None
    assert s.prev_pending_frame(11) == 3
    assert s.prev_pending_frame(3) is None
    s.accept_frame(3)
    assert s.pending_frames() == [11]


def test_review_edit_keeps_it_a_candidate():
    s = _session()
    cand = s.edit(3, 0, bbox=[2, 2, 8, 8], track_id=7)
    assert cand is not None
    assert cand.bbox == [2, 2, 8, 8] and cand.track_id == 7
    assert s.is_edited(3, 0) is True
    assert s.state(3, 0) == al.PENDING  # a corrected box is still a candidate
    s.edit(3, 1, obb=[5.0, 5.0, 4.0, 2.0, 0.0], accept=True)
    assert s.state(3, 1) == al.ACCEPTED
    assert s.candidates(3)[1].bbox == [3, 4, 7, 6]


def test_review_only_accepted_candidates_leave_the_session():
    s = _session()
    s.accept(3, 0)
    s.reject(3, 1)
    accepted = s.accepted_candidates()
    assert list(accepted) == [3] and len(accepted[3]) == 1
    assert list(s.pending_candidates()) == [11]
    recs = s.to_annotations()
    assert list(recs) == [3]
    assert recs[3][0]["accepted"] is True


def test_review_progress_text():
    s = _session()
    assert s.progress_text() == ("3 of 3 candidates still to review (0 accepted, 0 discarded).")
    s.accept_all()
    assert s.progress_text() == ("All 3 candidates reviewed — 3 accepted, 0 discarded.")
    assert al.ReviewSession().progress_text() == "No candidates"


def test_review_set_candidates_replaces_rather_than_stacks():
    s = _session()
    s.set_candidates({0: [al.Candidate(bbox=[0, 0, 2, 2])]})
    assert s.frames == [0]
    assert s.counts().total == 1


# --------------------------------------------------------------------------- #
# Torch-free tracking
# --------------------------------------------------------------------------- #


def test_tracker_backend_chain(monkeypatch):
    backend = al.tracker_backend()
    assert backend in {"csrt", "csrt_legacy", "mil", "lk"}
    # Measured on this machine (opencv 4.13.0): no CSRT, no cv2.legacy.
    assert backend == "mil"

    cv2 = pytest.importorskip("cv2")
    monkeypatch.delattr(cv2, "TrackerCSRT_create", raising=False)
    monkeypatch.delattr(cv2, "legacy", raising=False)
    monkeypatch.delattr(cv2, "TrackerMIL_create", raising=False)
    assert al.tracker_backend() == "lk"
    assert al.make_tracker() == (None, "lk")


def test_make_tracker_returns_a_usable_factory():
    factory, backend = al.make_tracker()
    if factory is None:
        assert backend in ("lk", "none")
    else:
        tracker = factory()
        assert hasattr(tracker, "init") and hasattr(tracker, "update")


def _synthetic_sequence(n=40, w=280, h=160, step=4, size=40):
    """A textured square translating `step` px/frame — trackable by MIL and by LK."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(7)
    patch = rng.integers(90, 256, size=(size, size), dtype=np.uint8)
    frames = []
    for i in range(n):
        img = np.full((h, w, 3), 20, dtype=np.uint8)
        x, y = 20 + i * step, 40
        img[y : y + size, x : x + size, :] = patch[:, :, None]
        frames.append(img)
    return frames, (20, 40, 20 + size, 40 + size), step, size


def test_track_forward_synthetic():
    frames, box, step, size = _synthetic_sequence()
    tracked, reason = al.track_forward(
        lambda i: frames[i] if 0 <= i < len(frames) else None,
        0,
        box,
        39,
        280,
        160,
    )
    assert len(tracked) >= 30, reason
    last_idx, last_box = tracked[-1]
    cx = (last_box[0] + last_box[2]) / 2.0
    cy = (last_box[1] + last_box[3]) / 2.0
    expect_cx = 20 + last_idx * step + size / 2.0
    expect_cy = 40 + size / 2.0
    assert abs(cx - expect_cx) < 12
    assert abs(cy - expect_cy) < 12
    for _idx, b in tracked:
        assert 0 <= b[0] <= b[2] <= 280
        assert 0 <= b[1] <= b[3] <= 160


def test_lk_propagate_synthetic():
    frames, box, step, size = _synthetic_sequence()
    tracked, reason = al.track_forward(
        lambda i: frames[i] if 0 <= i < len(frames) else None,
        0,
        box,
        39,
        280,
        160,
        backend="lk",
    )
    assert len(tracked) >= 30, reason
    last_idx, last_box = tracked[-1]
    cx = (last_box[0] + last_box[2]) / 2.0
    assert abs(cx - (20 + last_idx * step + size / 2.0)) < 12


def test_lk_propagate_returns_none_on_a_featureless_frame():
    np = pytest.importorskip("numpy")
    flat = np.full((80, 80), 128, dtype=np.uint8)
    assert al.lk_propagate(flat, flat, [10, 10, 50, 50]) is None
    tiny = np.full((4, 4), 128, dtype=np.uint8)
    assert al.lk_propagate(tiny, tiny, [0, 0, 2, 2]) is None


def test_track_forward_stop_reasons():
    frames, box, _step, _size = _synthetic_sequence(n=6)
    read = lambda i: frames[i] if 0 <= i < len(frames) else None

    tracked, reason = al.track_forward(read, 0, box, 20, 280, 160)
    assert reason == "end_of_video"
    assert len(tracked) <= 5

    tracked, reason = al.track_forward(read, 0, box, 5, 280, 160, should_stop=lambda: True)
    assert (tracked, reason) == ([], "stopped")

    assert al.track_forward(read, 99, box, 5, 280, 160) == ([], "end_of_video")
    assert al.track_forward(read, 0, box, 0, 280, 160) == ([], "")


def test_tracked_boxes_become_candidates_in_the_same_review_loop():
    """The torch-free fallback is one branch of one loop, not a second feature."""
    frames, box, _step, _size = _synthetic_sequence(n=10)
    tracked, _reason = al.track_forward(
        lambda i: frames[i] if 0 <= i < len(frames) else None,
        0,
        box,
        9,
        280,
        160,
    )
    cands = {fi: [al.Candidate(bbox=b, score=1.0, track_id=2, source="track")] for fi, b in tracked}
    merged, res = al.merge_into({}, cands, al.MergePolicy(skip_frames_with_accepted=False))
    assert res.added == len(tracked)
    any_rec = merged[next(iter(merged))][0]
    assert any_rec["source"] == "track"
    assert any_rec["accepted"] is False


def test_the_gpu_probe_answers_without_importing_torch():
    """"Can this Python train" and "is there a GPU" are different questions.

    torch imports and runs perfectly well on a machine with no NVIDIA device,
    about ten times slower - so a duration quoted off Capability.torch can be
    nine tenths short. The probe loads the driver library directly, which is
    ~40ms against several seconds for an import.
    """
    saved = sys.modules.pop("torch", None)
    try:
        count = al.cuda_device_count()
        assert isinstance(count, int) and count >= 0
        assert al.has_gpu() is (count > 0)
        assert "torch" not in sys.modules
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
