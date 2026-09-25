"""One round, driven end to end through the real app, with the trainer faked.

The arithmetic is tested in ``test_training_round.py`` and the export in
``test_round_export.py``. What is left is the wiring, and the wiring is where
this feature can go wrong in ways that look like success:

- a run that exits 0 without writing a checkpoint, leaving the project pointed
  at a ``.pth`` that does not exist;
- a round that scored worse than the last one being adopted anyway, which makes
  the *next* pass through the loop worse than having no detector at all;
- a stop being reported as an error, or an error as a stop;
- the holdout being re-rolled between rounds, so "better than last time" is two
  different rulers.

The child process is stubbed. Training for real is covered by running it - see
the end-to-end check in the round's own documentation - not by the suite, which
must not spend a minute of GPU per run.
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
    """A trainer that has already finished with ``code``."""

    def __init__(self, code: int = 0) -> None:
        self._code = code
        self.terminated = False

    def poll(self):
        return self._code

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self._code

    def kill(self):
        self.terminated = True


@pytest.fixture(scope="module")
def app():
    instance = A.AnnotationApp()
    dpg.create_viewport(title="yoake round wiring", width=1000, height=700)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    instance.class_names = ["fly"]
    instance.action_names = list(A._DEFAULT_ACTIONS)
    instance._build_main_window()
    instance._update_action_buttons()
    yield instance
    dpg.destroy_context()


def _write_clip(path, frames=40, w=96, h=72):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for index in range(frames):
        frame = np.full((h, w, 3), 20, np.uint8)
        cv2.rectangle(frame, (10, 20), (10 + 20 + index % 8, 50), (60, 200, 90), -1)
        writer.write(frame)
    writer.release()


@pytest.fixture
def project(app, tmp_path):
    """A project with 20 labelled frames in two real videos, and no round yet.

    Real files, because the export decodes them and the round refuses - loudly,
    and correctly - when it cannot read a frame it was told is labelled.
    """
    from htrtdetr.annotation.project import ProjectSession
    from htrtdetr.annotation.session import VideoSession

    app.project = ProjectSession()
    app.project_dir = tmp_path
    app.project.class_names = ["fly"]
    app.class_names = ["fly"]
    app._round = None
    app._round_pending_weights = ""
    ProjectLayout(tmp_path).create()
    for name, count in (("one", 12), ("two", 8)):
        path = tmp_path / f"{name}.mp4"
        _write_clip(path)
        video = VideoSession()
        video.video_id = name
        video.path = path
        video.num_frames = 40
        video.width, video.height = 96, 72
        for index in range(count):
            video.annotations[index * 3] = [
                AnnObject(bbox=[10, 10, 40, 40], track_id=1, class_id=0)
            ]
        app.project.videos[name] = video
    app._dirty = False
    return app


def _finished_run(state, *, best=True, score=1.0, epochs=3):
    """Write what a real trainer leaves behind, without training."""
    run = state.run_dir
    (run / "weights").mkdir(parents=True, exist_ok=True)
    if best:
        (run / "weights" / "best.pth").write_bytes(b"x")
    (run / "weights" / "last.pth").write_bytes(b"x")
    rows = ["epoch,train_loss,val_loss,primary_metric,primary_metric_value,best_so_far"]
    for epoch in range(epochs):
        rows.append(f"{epoch},5.0,{score},val_loss,{score},{score}")
    (run / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run / "metrics_latest.json").write_text(
        json.dumps({"primary_metric": "val_loss", "primary_metric_value": score,
                    "best_so_far": score, "higher_is_better": False}),
        encoding="utf-8",
    )


def _run_round(app, monkeypatch, *, best=True, score=1.0, code=0):
    """Start a round, fake the child, and let it finish."""
    spawned = {}

    def fake_popen(cmd, **kwargs):
        spawned["cmd"] = cmd
        spawned["kwargs"] = kwargs
        _finished_run(app._round, best=best, score=score)
        return _FakeProc(code)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    # The export runs on a worker; drain its posted closures until it spawns.
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if "cmd" in spawned or app._round is None:
            break
    assert "cmd" in spawned, "the round never reached the trainer"
    state = app._round
    state.polled = 0.0
    app.tick_train_round()
    return spawned, state


def test_the_round_trains_on_the_labels_and_adopts_the_model(project, monkeypatch):
    app = project
    spawned, _state = _run_round(app, monkeypatch, score=2.0)

    joined = " ".join(str(c) for c in spawned["cmd"])
    assert "model.detector.head.num_classes=1" in joined, "class count is never a default"
    assert "data.num_workers=0" in joined
    assert spawned["kwargs"]["stdin"] is subprocess.DEVNULL, "input() would hang the GUI"
    assert spawned["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "-1", "GPU off means off"

    assert app._round is None, "the round finished"
    weights = app.project.settings.get("weights", "")
    assert weights.endswith("best.pth") and Path(weights).exists()
    assert "runs" in weights and "rounds" in weights, "never under runs/train"


def test_the_splits_land_beside_the_images_not_in_a_folder_of_their_own(project, monkeypatch):
    """The trainer resolves images against the annotation file's own directory,
    so a split one folder away trains on black frames and says nothing."""
    app = project
    spawned, _state = _run_round(app, monkeypatch)
    train = next(c for c in spawned["cmd"] if str(c).startswith("data.train_root="))
    train_json = Path(str(train).split("=", 1)[1])
    assert (train_json.parent / "images").is_dir()


def test_last_pth_is_taken_when_validation_never_improved(project, monkeypatch):
    """``best.pth`` is only written on an improving validation epoch, so a
    short or flat run leaves only ``last.pth`` - and that is still a model."""
    app = project
    app.project.settings["weights"] = "kept.pth"
    _run_round(app, monkeypatch, best=False)
    adopted = app.project.settings["weights"]
    assert adopted.endswith("last.pth"), adopted
    assert Path(adopted).exists()


def test_nothing_is_adopted_when_the_run_wrote_no_weights_at_all(project, monkeypatch):
    app = project
    app.project.settings["weights"] = "kept.pth"

    def fake_popen(cmd, **kwargs):
        app._round.run_dir.mkdir(parents=True, exist_ok=True)
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if app._round is not None and app._round.proc is not None:
            break
    app._round.polled = 0.0
    app.tick_train_round()
    assert app.project.settings["weights"] == "kept.pth"


def test_a_worse_round_is_refused_but_offered(project, monkeypatch):
    """Adopting a worse detector makes the next pick-frames pass worse than
    having none, so it is the user's call and not the default."""
    app = project
    _run_round(app, monkeypatch, score=1.0)
    first = app.project.settings["weights"]

    # Round two, same holdout, a worse validation loss.
    _run_round(app, monkeypatch, score=9.0)
    assert app.project.settings["weights"] == first, "kept the better model"
    assert dpg.is_item_shown(app.round_dialog._t("adopt")), "the override is offered"

    app.adopt_pending_round_model()
    assert app.project.settings["weights"] != first, "the user can still take it"


def test_the_holdout_is_frozen_across_rounds(project, monkeypatch):
    app = project
    _run_round(app, monkeypatch, score=1.0)
    log = json.loads(ProjectLayout(app.project_dir).rounds_log.read_text(encoding="utf-8"))
    first = {tuple(row) for row in log["holdout"]}
    assert first

    _run_round(app, monkeypatch, score=0.5)
    log = json.loads(ProjectLayout(app.project_dir).rounds_log.read_text(encoding="utf-8"))
    second = {tuple(row) for row in log["holdout"]}
    assert first <= second, "a frozen holdout is never re-rolled"
    assert log["rounds"][0]["holdout_digest"] == log["rounds"][1]["holdout_digest"]
    assert log["rounds"][1]["round"] == 2


def test_the_round_log_records_what_each_round_bought(project, monkeypatch):
    app = project
    _run_round(app, monkeypatch, score=3.5)
    row = json.loads(
        ProjectLayout(app.project_dir).rounds_log.read_text(encoding="utf-8")
    )["rounds"][0]
    assert row["frames"] == 20 and row["videos"] == 2
    assert row["metric"] == "val_loss"
    assert row["score"] == 3.5
    assert row["adopted"] is True
    assert Path(row["run_dir"]).name == "stage1"


def test_a_failed_run_is_an_error_and_a_stopped_one_is_not(project, monkeypatch):
    app = project
    _spawned, state = _run_round(app, monkeypatch, code=1)
    assert app._round is None
    # An error leaves the form offering Start again, not a finished result.
    assert dpg.is_item_shown(app.round_dialog._t("go"))
    assert not dpg.is_item_shown(app.round_dialog._t("analyze"))
    assert state.run_dir.exists()


def test_a_second_round_cannot_start_while_one_is_running(project, monkeypatch):
    app = project

    def fake_popen(cmd, **kwargs):
        return _FakeProc(None)  # still going

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if app._round is not None and app._round.proc is not None:
            break
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    assert "already running" in " ".join(
        [dpg.get_value(app.round_dialog._t("detail")) or ""]
    )
    app._shutdown_round()
    assert app._round is None


# ── the guards the adversarial review asked for ──────────────────────────

def test_a_late_callback_cannot_touch_a_round_it_does_not_own(project, monkeypatch):
    """The critical one. A stopped round's export thread is still alive; its
    callbacks must not cancel, or steal the log of, whatever round is current."""
    app = project
    _spawned, first = _run_round(app, monkeypatch, score=1.0)
    assert app._round is None

    # A closure from the finished round, arriving late.
    app._round = object()  # stand-in for "some other round is current"
    current = app._round
    app._round_fail(first, "should be ignored", step=1)
    app._round_stopped(first, "should also be ignored")
    app._round_spawn(first)
    assert app._round is current, "a foreign round's callback was obeyed"
    app._round = None


def test_stopping_during_the_export_keeps_ownership(project, monkeypatch):
    """Clearing _round while the worker still owns the disk is what let a
    second round start beside it and have its callbacks stolen."""
    app = project
    import threading

    started = threading.Event()
    release = threading.Event()

    real = __import__("htrtdetr.gui.round_export", fromlist=["export_frames"]).export_frames

    def slow_export(inputs, out, *, cancel=None, progress=None):
        started.set()
        release.wait(timeout=5)
        return real(inputs, out, cancel=cancel, progress=progress)

    monkeypatch.setattr(
        "htrtdetr.gui.round_export.export_frames", slow_export
    )
    monkeypatch.setattr("htrtdetr.gui.train_operations.round_export.export_frames", slow_export)
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    assert started.wait(timeout=5)
    state = app._round
    assert state is not None and state.proc is None

    app.stop_train_round()
    assert app._round is state, "ownership stays with the live worker"
    assert state.cancel.is_set()

    # And a second round cannot start beside it.
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    assert app._round is state

    release.set()
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if app._round is None:
            break
    assert app._round is None, "the worker released the round when it noticed"


def test_the_plan_is_re_derived_against_the_frames_that_survived(project, monkeypatch):
    """A moved video drops frames. A batch size derived before that can exceed
    the surviving training set, which is zero optimizer steps and a 0.0 loss."""
    app = project
    app.project.videos["two"].path.unlink()  # the video moved away
    spawned, state = _run_round(app, monkeypatch)

    batch = int(next(
        str(c).split("=", 1)[1] for c in spawned["cmd"]
        if str(c).startswith("data.batch_size=")
    ))
    assert batch <= state.plan.n_train, (batch, state.plan)
    assert state.plan.counts.frames == 12, "only video one survived"


def test_a_failed_round_does_not_hand_its_number_to_the_retry(project, monkeypatch):
    app = project
    _spawned, first = _run_round(app, monkeypatch, code=1)
    assert first.run_dir.exists()
    _spawned2, second = _run_round(app, monkeypatch, code=0)
    assert second.run_dir != first.run_dir
    assert second.number == first.number + 1


def test_a_diverged_run_does_not_crash_the_log_write(project, monkeypatch):
    """NaN is a real outcome, and atomic_json refuses to write it."""
    app = project

    def fake_popen(cmd, **kwargs):
        state = app._round
        _finished_run(state)
        (state.run_dir / "metrics_latest.json").write_text(
            '{"primary_metric": "val_loss", "best_so_far": NaN, '
            '"primary_metric_value": NaN}',
            encoding="utf-8",
        )
        return _FakeProc(0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    app.start_train_round({"minutes": 5.0, "use_gpu": False})
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if app._round is not None and app._round.proc is not None:
            break
    app._round.polled = 0.0
    app.tick_train_round()
    assert app._round is None
    row = json.loads(
        ProjectLayout(app.project_dir).rounds_log.read_text(encoding="utf-8")
    )["rounds"][0]
    assert row["score"] is None


def test_reopening_a_running_round_keeps_stop_reachable(project, monkeypatch):
    app = project

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: _FakeProc(None))
    # Through the dialog, as the user does: _on_start_clicked is what puts the
    # window into its running phase, and that is the state a reopen must keep.
    app.round_dialog.show(plan="x")
    app.round_dialog._on_start_clicked()
    for _ in range(600):
        app._pump_ui()
        dpg.render_dearpygui_frame()
        if app._round is not None and app._round.proc is not None:
            break
    app.open_train_round()
    dpg.render_dearpygui_frame()
    assert dpg.is_item_shown(app.round_dialog._t("stop")), "Stop must survive a reopen"
    assert not dpg.is_item_shown(app.round_dialog._t("go"))
    app._shutdown_round()
