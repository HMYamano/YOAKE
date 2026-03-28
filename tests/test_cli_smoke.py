import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
PYTHON = sys.executable


class TestCLIHelp:
    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [PYTHON, "-m", "htrtdetr.cli"] + args,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
        )

    def test_help_exits_zero(self):
        r = self._run(["--help"])
        assert r.returncode == 0

    def test_help_contains_commands(self):
        r = self._run(["--help"])
        for cmd in ("train", "val", "predict", "analyze"):
            assert cmd in r.stdout

    def test_help_mentions_primary_metrics(self):
        r = self._run(["--help"])
        assert "AP50" in r.stdout or "ap50" in r.stdout.lower()
        assert "macro" in r.stdout.lower()

    def test_help_mentions_output_structure(self):
        r = self._run(["--help"])
        assert "weights" in r.stdout
        assert "results.csv" in r.stdout

    def test_unknown_command_exits_nonzero(self):
        r = self._run(["foobar"])
        assert r.returncode != 0


class TestParseArgv:
    def setup_method(self):
        from htrtdetr.cli import parse_argv

        self.parse_argv = parse_argv

    def test_bare_yaml_detected(self):
        cp, _ = self.parse_argv(["myconfig.yaml", "train.lr=1e-4"])
        assert cp == "myconfig.yaml"

    def test_config_key_detected(self):
        cp, ov = self.parse_argv(["config=myconfig.yaml", "train.lr=1e-4"])
        assert cp == "myconfig.yaml"
        assert ov["train"]["lr"] == pytest.approx(1e-4)

    def test_deep_nesting(self):
        _, ov = self.parse_argv(["a.b.c.d=42"])
        assert ov["a"]["b"]["c"]["d"] == 42


class TestFindCheckpoint:
    def test_returns_empty_string_when_no_checkpoint(self, tmp_path):
        from htrtdetr.cli import find_checkpoint

        assert find_checkpoint(str(tmp_path), stage=1) == ""

    def test_finds_new_path_best_pth(self, tmp_path):
        from htrtdetr.cli import find_checkpoint

        ckpt = tmp_path / "runs" / "train" / "stage1" / "weights" / "best.pth"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"dummy")
        assert find_checkpoint(str(tmp_path), stage=1) == str(ckpt)

    def test_finds_legacy_path(self, tmp_path):
        from htrtdetr.cli import find_checkpoint

        ckpt = tmp_path / "outputs" / "stage1" / "stage1_best.pth"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"dummy")
        assert find_checkpoint(str(tmp_path), stage=1) == str(ckpt)


class TestPredictAnalyzeEntryPoints:
    def test_run_predict_uses_inferencer(self, monkeypatch, tmp_path):
        from htrtdetr.cli import run_predict
        import htrtdetr.models as models_mod
        import htrtdetr.inference as inference_mod
        import htrtdetr.utils.misc as misc_mod
        import htrtdetr.cli as cli_mod

        calls = {}

        class DummyModel:
            def set_stage(self, stage):
                calls["stage"] = stage

        class DummyInferencer:
            def __init__(self, model, cfg, action_names, class_names, window_size):
                calls["init"] = {
                    "cfg": cfg,
                    "window_size": window_size,
                    "action_names": action_names,
                    "class_names": class_names,
                }

            def run(self, input_path, output_dir, image_size, max_frames):
                calls["run"] = {
                    "input_path": input_path,
                    "output_dir": output_dir,
                    "image_size": image_size,
                    "max_frames": max_frames,
                }

        monkeypatch.setattr(models_mod, "build_model", lambda cfg: DummyModel())
        monkeypatch.setattr(cli_mod, "load_runtime_config_from_checkpoint", lambda path: None)
        monkeypatch.setattr(
            misc_mod,
            "load_model_weights",
            lambda *args, **kwargs: (calls.__setitem__("weights", True) or ([], [])),
        )
        monkeypatch.setattr(inference_mod, "Inferencer", DummyInferencer)

        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"dummy")
        run_predict(
            [
                f"source={tmp_path}",
                f"weights={ckpt}",
                f"output_dir={tmp_path / 'out'}",
                "window_size=8",
            ]
        )

        assert calls["stage"] == 4
        assert calls["run"]["output_dir"] == str(tmp_path / "out")
        assert calls["init"]["window_size"] == 8

    def test_run_predict_restores_config_from_checkpoint_metadata(self, monkeypatch, tmp_path):
        from htrtdetr.cli import run_predict
        from htrtdetr.config.config import HTRTDETRConfig
        import htrtdetr.models as models_mod
        import htrtdetr.inference as inference_mod
        import htrtdetr.utils.misc as misc_mod
        import htrtdetr.cli as cli_mod

        calls = {}
        cfg = HTRTDETRConfig()
        cfg.data.image_size = (320, 512)
        cfg.data.window_size = 12
        cfg.model.action_head.num_actions = 3
        cfg.eval.action_names = ["idle", "walk", "groom"]

        class DummyModel:
            def set_stage(self, stage):
                calls["stage"] = stage

        class DummyInferencer:
            def __init__(self, model, cfg, action_names, class_names, window_size):
                calls["init"] = {
                    "cfg": cfg,
                    "window_size": window_size,
                    "action_names": action_names,
                    "class_names": class_names,
                }

            def run(self, input_path, output_dir, image_size, max_frames):
                calls["run"] = {
                    "input_path": input_path,
                    "output_dir": output_dir,
                    "image_size": image_size,
                    "max_frames": max_frames,
                }

        monkeypatch.setattr(models_mod, "build_model", lambda cfg: DummyModel())
        monkeypatch.setattr(cli_mod, "load_runtime_config_from_checkpoint", lambda path: cfg)
        monkeypatch.setattr(misc_mod, "load_model_weights", lambda *args, **kwargs: ([], []))
        monkeypatch.setattr(inference_mod, "Inferencer", DummyInferencer)

        ckpt = tmp_path / "model.pth"
        ckpt.write_bytes(b"dummy")
        run_predict(
            [
                f"source={tmp_path}",
                f"weights={ckpt}",
                f"output_dir={tmp_path / 'out'}",
            ]
        )

        assert calls["stage"] == 4
        assert calls["init"]["window_size"] == 12
        assert calls["init"]["action_names"] == ["idle", "walk", "groom"]
        assert calls["run"]["image_size"] == (320, 512)

    def test_run_analyze_uses_existing_analysis_wrappers(self, monkeypatch, tmp_path):
        from htrtdetr.cli import run_analyze
        import htrtdetr.analysis as analysis_mod

        pred = tmp_path / "predictions.json"
        pred.write_text("{}")
        ann = tmp_path / "annotations.json"
        ann.write_text('{"videos": [], "class_names": [], "action_names": []}')
        calls = []

        monkeypatch.setattr(analysis_mod, "run_timeline_analysis", lambda p, o: calls.append(("timeline", p, o)))
        monkeypatch.setattr(analysis_mod, "run_distribution_analysis", lambda a, o: calls.append(("distribution", a, o)))
        monkeypatch.setattr(analysis_mod, "run_id_switch_analysis", lambda p, o: calls.append(("id_switches", p, o)))
        monkeypatch.setattr(analysis_mod, "run_confidence_analysis", lambda p, o: calls.append(("confidence", p, o)))

        run_analyze([f"mode=timeline", f"predictions={pred}", f"output_dir={tmp_path}"])
        run_analyze([f"mode=distribution", f"annotation={ann}", f"output_dir={tmp_path}"])
        run_analyze([f"mode=id_switches", f"predictions={pred}", f"output_dir={tmp_path}"])
        run_analyze([f"mode=confidence", f"predictions={pred}", f"output_dir={tmp_path}"])

        assert [name for name, *_ in calls] == ["timeline", "distribution", "id_switches", "confidence"]


class TestGuiCommandAssembly:
    def test_build_yoake_command_keeps_cli_contract(self):
        from htrtdetr.gui_cli import build_yoake_command

        cmd = build_yoake_command("val", "stage=2", checkpoint="runs/train/stage2/weights/best.pth")
        assert cmd[:3] == ["-m", "htrtdetr.cli", "val"]
        assert "checkpoint=runs/train/stage2/weights/best.pth" in cmd

    def test_specialized_gui_builders_share_the_same_contract(self):
        from htrtdetr.gui_cli import (
            build_analyze_command,
            build_predict_command,
            build_train_command,
            build_val_command,
        )

        train_cmd = build_train_command(2, "data/train/annotations.json", "data/val/annotations.json", "runs/train/stage2")
        val_cmd = build_val_command(2, "runs/train/stage2/weights/best.pth", "data/val/annotations.json", "runs/val/stage2")
        predict_cmd = build_predict_command("data/video.mp4", "runs/train/stage4/weights/best.pth", "runs/predict")
        analyze_cmd = build_analyze_command("timeline", "runs/analyze", predictions="runs/predict/predictions.json")

        assert "data.train_root=data/train/annotations.json" in train_cmd
        assert "data.val_root=data/val/annotations.json" in val_cmd
        assert "source=data/video.mp4" in predict_cmd
        assert "mode=timeline" in analyze_cmd

    def test_gui_can_pass_annotation_json_to_data_val_root(self):
        from htrtdetr.gui_cli import _yoake

        cmd = _yoake(
            "val",
            "stage=1",
            "checkpoint=foo.pth",
            "data.val_root=data/val/annotations.json",
            "output_dir=runs/val/stage1",
        )
        assert "data.val_root=data/val/annotations.json" in cmd

    def test_gui_source_uses_data_val_root_contract(self):
        gui_source = (REPO_ROOT / "tools" / "gui.py").read_text(encoding="utf-8", errors="ignore")
        assert "data.val_root=" in gui_source
        assert "build_yoake_command" in gui_source


class TestPathHelpers:
    def test_windows_absolute_path_is_absolute(self):
        from htrtdetr.cli import resolve_annotation_path

        path = resolve_annotation_path(r"C:\work\data\annotations.json")
        assert str(path).lower().endswith("annotations.json")
