"""A keyframe project's round: export the clips, train the detector, then the behaviors.

``docs/KEYFRAME_BEHAVIOR.md`` §8.3. The trainer is faked exactly as in
``test_train_round_wiring.py``; what is under test is the wiring that is new
for keyframe projects:

- the context frames of every keyframe's clip are exported, not just the keyframe;
- stage 2 runs after stage 1, warm-started from the detector just trained;
- adoption still compares *detectors* across rounds (a behavior F1 is never set
  against a detector loss), and the stage-2 checkpoint is what gets adopted;
- too few behavior labels, or a failing stage 2, still leaves a usable detector.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

dpg = pytest.importorskip("dearpygui.dearpygui")
pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from htrtdetr import annotator as A  # noqa: E402
from htrtdetr.annotation.layout import ProjectLayout  # noqa: E402
from htrtdetr.annotation.objects import AnnObject  # noqa: E402

pytestmark = pytest.mark.gui


class _FakeProc:
    def __init__(self, code: int = 0) -> None:
        self._code = code

    def poll(self):
        return self._code

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return self._code

    def kill(self):
        pass


@pytest.fixture(scope="module")
def app():
    instance = A.AnnotationApp()
    dpg.create_viewport(title="yoake keyframe round", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    instance.class_names = ["fly"]
    instance.action_names = ["still", "walk"]
    instance._build_main_window()
    instance._update_action_buttons()
    yield instance
    dpg.destroy_context()


def _write_clip(path, frames=60, w=96, h=72):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for index in range(frames):
        frame = np.full((h, w, 3), 20, np.uint8)
        cv2.rectangle(frame, (10 + index % 30, 20), (30 + index % 30, 30), (60, 200, 90), -1)
        writer.write(frame)
    writer.release()


def _project(app, tmp_path, *, behaviors=True):
    from htrtdetr.annotation.project import ProjectSession
    from htrtdetr.annotation.session import VideoSession

    app.project = ProjectSession()
    app.project.settings["label_mode"] = "keyframe"
    app.project.settings["use_track_ids"] = False
    app.project.settings["clip"] = {"length": 3, "stride": 2, "anchor": "end"}
    app.project_dir = tmp_path
    app.project.class_names = ["fly"]
    app.project.action_names = ["still", "walk"]
    app.class_names = ["fly"]
    app.action_names = ["still", "walk"]
    app._round = None
    app._round_pending_weights = ""
    ProjectLayout(tmp_path).create()
    for name, count in (("one", 8), ("two", 6)):
        path = tmp_path / f"{name}.mp4"
        _write_clip(path)
        video = VideoSession()
        video.video_id = name
        video.path = path
        video.num_frames = 60
        video.width, video.height = 96, 72
        for k in range(count):
            video.annotations[10 + 6 * k] = [
                AnnObject(bbox=[10, 10, 40, 40], track_id=-1, class_id=0,
                          action_id=(k % 2) if behaviors else -1)
            ]
        app.project.videos[name] = video
    app._dirty = False
    return app


def _finished_run(run: Path, *, metric: str, score: float, higher: bool, best=True):
    (run / "weights").mkdir(parents=True, exist_ok=True)
    if best:
        (run / "weights" / "best.pth").write_bytes(b"x")
    (run / "weights" / "last.pth").write_bytes(b"x")
    rows = ["epoch,train_loss,val_loss,primary_metric,primary_metric_value,best_so_far"]
    rows += [f"{e},5.0,1.0,{metric},{score},{score}" for e in range(3)]
    (run / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run / "metrics_latest.json").write_text(json.dumps({
        "primary_metric": metric, "primary_metric_value": score,
        "best_so_far": score, "higher_is_better": higher}), encoding="utf-8")


def _run(app, monkeypatch, *, stage2_code=0):
    """Start a round; each fake spawn leaves a finished run in the current phase."""
    spawned = []

    def fake_popen(cmd, **kwargs):
        spawned.append([str(c) for c in cmd])
        state = app._round
        if state.phase == "behavior":
            _finished_run(state.run_dir, metric="macro_f1", score=0.6, higher=True)
            if stage2_code:
                (state.run_dir / "round.log").write_text("Traceback\nboom\n", encoding="utf-8")
            return _FakeProc(stage2_code)
        _finished_run(state.run_dir, metric="val_loss", score=2.0, higher=False)
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if spawned or app._round is None:
            break
    assert spawned, "the round never reached the trainer"
    for _ in range(3):                      # detector finishes, then behaviors finish
        if app._round is None:
            break
        app._round.polled = 0.0
        app.tick_train_round()
    return spawned


def test_the_clip_frames_are_exported_with_the_keyframes(app, tmp_path, monkeypatch):
    _project(app, tmp_path)
    _run(app, monkeypatch)
    images = tmp_path / "datasets" / "rounds" / "images" / "one"
    # keyframe 10 with a 3-frame clip every 2nd frame needs 6, 8 and 10
    for index in (6, 8, 10):
        assert (images / f"{index:06d}.jpg").is_file(), index
    data = json.loads((tmp_path / "datasets" / "rounds" / "round-001.json").read_text("utf-8"))
    assert data["meta"]["labeling"]["mode"] == "keyframe"
    assert data["meta"]["labeling"]["clip"]["stride"] == 2
    assert {f["frame_index"] for v in data["videos"] for f in v["frames"]} >= {10, 16}


def test_behaviors_train_after_the_detector_and_their_model_is_adopted(app, tmp_path, monkeypatch):
    _project(app, tmp_path)
    spawned = _run(app, monkeypatch)
    assert len(spawned) == 2, "detector, then behaviors"
    first, second = (" ".join(cmd) for cmd in spawned)
    assert "stage=1" in first and "stage=2" in second
    stage1_best = next(c for c in spawned[1] if c.startswith("init_weights="))
    assert stage1_best.endswith(str(Path("stage1") / "weights" / "best.pth"))
    assert app._round is None
    adopted = app.project.settings["weights"]
    assert adopted.endswith(str(Path("stage2") / "weights" / "best.pth")), adopted
    row = json.loads((tmp_path / "rounds.json").read_text("utf-8"))["rounds"][-1]
    assert (row["metric"], row["score"]) == ("val_loss", 2.0), "adoption compares detectors"
    assert (row["behavior_metric"], row["behavior_score"]) == ("macro_f1", 0.6)


def test_too_few_behavior_labels_trains_the_detector_only_and_says_so(app, tmp_path, monkeypatch):
    _project(app, tmp_path, behaviors=False)
    spawned = _run(app, monkeypatch)
    assert len(spawned) == 1
    assert app.project.settings["weights"].endswith(str(Path("stage1") / "weights" / "best.pth"))
    row = json.loads((tmp_path / "rounds.json").read_text("utf-8"))["rounds"][-1]
    assert "detector only" in row["behavior_note"]


def test_a_failing_behavior_phase_keeps_the_detector(app, tmp_path, monkeypatch):
    _project(app, tmp_path)
    spawned = _run(app, monkeypatch, stage2_code=1)
    assert len(spawned) == 2
    assert app._round is None
    assert app.project.settings["weights"].endswith(str(Path("stage1") / "weights" / "best.pth"))
    row = json.loads((tmp_path / "rounds.json").read_text("utf-8"))["rounds"][-1]
    assert "behavior model stopped" in row["behavior_note"]


def test_the_plan_says_a_keyframe_round_is_two_runs(app, tmp_path):
    """Quoted before the user commits (P17): the behaviors add a second run."""
    _project(app, tmp_path)
    sentence = app.round_plan_sentence({"minutes": 5.0, "use_gpu": False})
    assert "Then the behaviors train on the clip before each keyframe" in sentence
    assert "3 frames, every 2nd" in sentence
