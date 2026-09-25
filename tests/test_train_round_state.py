"""Round adoption and project state, without opening a GUI or training a model."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("dearpygui.dearpygui")

from htrtdetr.annotation.layout import ProjectLayout
from htrtdetr.annotation.round import LabelCounts, derive_plan
from htrtdetr.gui.train_operations import RoundState, TrainRoundMixin


class App(TrainRoundMixin):
    def __init__(self, root):
        self.project_dir = root
        self.project = SimpleNamespace(settings={}, videos={})
        self.class_names = ["fly"]
        self.round_dialog = Mock()
        self._status = Mock()
        self._mark_dirty = Mock()
        self._refresh_model_ui = Mock()

    def _ensure_project(self):
        return self.project


def finish_round(app, number, score):
    plan = derive_plan(
        LabelCounts(frames=20), number=number, holdout=4, val_kind="frames"
    )
    state = RoundState(plan, ProjectLayout(app.project_dir), {})
    state.project = app.project
    state.digest = "fixed-holdout"
    weights = state.run_dir / "weights" / "best.pth"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"checkpoint")
    (state.run_dir / "metrics_latest.json").write_text(
        json.dumps({"primary_metric": "val_loss", "best_so_far": score}),
        encoding="utf-8",
    )
    app._round = state
    app._round_adopt(state)
    return weights


def test_manual_adoption_becomes_the_next_comparison_baseline(tmp_path):
    app = App(tmp_path)
    finish_round(app, 1, 1.0)
    second = finish_round(app, 2, 9.0)
    app.adopt_pending_round_model()
    assert app.project.settings["weights"] == str(second)
    assert app._last_adopted() == (9.0, "fixed-holdout")
    third = finish_round(app, 3, 5.0)
    assert app.project.settings["weights"] == str(third)


def test_manual_adoption_cannot_change_another_project(tmp_path):
    app = App(tmp_path)
    finish_round(app, 1, 1.0)
    finish_round(app, 2, 9.0)
    app.project = SimpleNamespace(settings={"weights": "other.pth"}, videos={})
    app.adopt_pending_round_model()
    assert app.project.settings["weights"] == "other.pth"


def test_a_deleted_pending_checkpoint_is_not_adopted(tmp_path):
    app = App(tmp_path)
    first = finish_round(app, 1, 1.0)
    second = finish_round(app, 2, 9.0)
    second.unlink()
    app.adopt_pending_round_model()
    assert app.project.settings["weights"] == str(first)
    assert app._last_adopted() == (1.0, "fixed-holdout")


def test_export_uses_the_starting_project_even_after_a_project_switch(tmp_path, monkeypatch):
    from htrtdetr.data.annotation import DatasetAnno, FrameAnno, VideoAnno
    from htrtdetr.gui.round_export import ExportResult

    app = App(tmp_path)
    original = app.project
    state = RoundState(
        derive_plan(LabelCounts(frames=20), warm=True), ProjectLayout(tmp_path), {}
    )
    state.project = original
    state.init_weights = str(tmp_path / "original.pth")
    app._round = state
    app._post = Mock()
    app._capability = lambda: SimpleNamespace(torch=False)
    dataset = DatasetAnno(
        class_names=["fly"],
        videos=[VideoAnno(video_id="clip", frames=[
            FrameAnno(frame_index=i, image_path=f"{i}.jpg") for i in range(20)
        ])],
    )

    def export(*args, **kwargs):
        app.project = SimpleNamespace(settings={}, videos={})
        app.class_names = ["mouse", "rat", "fly"]
        app.project_dir = tmp_path / "other"
        return ExportResult(reused=20)

    monkeypatch.setattr("htrtdetr.gui.train_operations.round_export.export_frames", export)
    monkeypatch.setattr("htrtdetr.gui.train_operations.round_export.missing_images", lambda *_: [])
    app._round_export_worker(state, [], dataset)
    assert state.train_json is not None
    assert state.plan.counts.classes == 1
    assert state.plan.warm

    popen = Mock()
    monkeypatch.setattr("htrtdetr.gui.train_operations.subprocess.Popen", popen)
    app.warm_start_from = Mock(side_effect=AssertionError("must use the saved weights"))
    try:
        app._round_spawn(state)
        cmd = popen.call_args.args[0]
        assert f"init_weights={state.init_weights}" in cmd
    finally:
        app._round_close_log(state)


# ── warm start, and the two ways it must refuse ──────────────────────────
#
# ``init_weights`` loads with ``strict=False``, which is the right call for a
# class-count change (it raises otherwise) and the wrong one for everything
# else: a checkpoint whose decoder is a different depth loads "successfully"
# and leaves half the model at its random initialisation. The round would then
# report a warm start having thrown the decoder away.


def _log_row(app, **fields):
    weights = app.project_dir / "best.pth"
    weights.write_bytes(b"not really a checkpoint")
    row = {"round": 1, "classes": 1, "weights": str(weights),
           "decoder_layers": 2, **fields}
    ProjectLayout(app.project_dir).rounds_log.write_text(
        json.dumps({"rounds": [row]}), encoding="utf-8"
    )


def test_a_matching_previous_round_is_carried_over(tmp_path):
    app = App(tmp_path)
    _log_row(app)
    path, incompatible = app.warm_start_from(LabelCounts(frames=50, classes=1))
    assert path.endswith("best.pth")
    assert not incompatible


def test_a_changed_class_list_refuses_the_warm_start(tmp_path):
    """The detector head is sized by ``num_classes``."""
    app = App(tmp_path)
    _log_row(app, classes=1)
    _path, incompatible = app.warm_start_from(LabelCounts(frames=50, classes=3))
    assert incompatible


def test_a_different_decoder_depth_refuses_the_warm_start(tmp_path):
    app = App(tmp_path)
    _log_row(app, decoder_layers=4)
    _path, incompatible = app.warm_start_from(LabelCounts(frames=50, classes=1))
    assert incompatible, (
        "strict=False would have loaded this silently and left the decoder "
        "randomly initialised"
    )


def test_a_row_from_before_the_key_existed_is_read_as_the_old_architecture(tmp_path):
    """Those rounds were trained at the production default of 4 layers.

    Defaulting the missing key to the *current* value would make the first
    round after the change warm-start from a checkpoint it cannot use.
    """
    app = App(tmp_path)
    _log_row(app)
    path = ProjectLayout(app.project_dir).rounds_log
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["rounds"][0]["decoder_layers"]
    path.write_text(json.dumps(data), encoding="utf-8")

    _p, incompatible = app.warm_start_from(LabelCounts(frames=50, classes=1))
    assert incompatible
    assert App.LEGACY_DECODER_LAYERS == 4


def test_no_previous_round_means_no_warm_start(tmp_path):
    app = App(tmp_path)
    assert app.warm_start_from(LabelCounts(frames=50, classes=1)) == ("", False)
