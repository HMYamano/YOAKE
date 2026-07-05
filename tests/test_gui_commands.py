"""
test_gui_commands.py — GUI コマンド生成 / 予測ローダ / 学習モニタのパーサ

いずれも dearpygui / torch 非依存で実行できる純ロジック。
"""

from __future__ import annotations

import json

from htrtdetr.gui import commands
from htrtdetr.gui import monitor
from htrtdetr.gui import predictions


# ─── commands.predict_command (visualize→show_* バグ修正の検証) ────────────

def test_predict_command_visualize_off_maps_to_show_fields():
    cmd = commands.predict_command("v.mp4", "w.pth", "out",
                                   score_threshold=0.5, window_size=16, visualize=False)
    joined = " ".join(cmd)
    assert cmd[:3] == ["-m", "htrtdetr.cli", "predict"]
    # 実在しない visualize= は出さない
    assert "visualize=" not in joined
    # 実在する inference.show_* を false にする
    for field in ("show_bbox", "show_id", "show_action", "show_confidence"):
        assert f"inference.{field}=False" in cmd
    assert "source=v.mp4" in cmd and "weights=w.pth" in cmd and "output_dir=out" in cmd
    assert "score_threshold=0.5" in cmd and "window_size=16" in cmd


def test_predict_command_visualize_on_has_no_show_or_visualize():
    cmd = commands.predict_command("v.mp4", "w.pth", "out", visualize=True)
    joined = " ".join(cmd)
    assert "visualize=" not in joined
    assert "inference.show_" not in joined


def test_predict_command_max_frames_optional():
    assert any(a.startswith("max_frames=") for a in
               commands.predict_command("v", "w", "o", max_frames=100))
    assert not any(a.startswith("max_frames=") for a in
                   commands.predict_command("v", "w", "o", max_frames=None))


# ─── commands.train_command (window_size / imbalance を含む) ───────────────

def test_train_command_stage2_includes_window_and_imbalance():
    cmd = commands.train_command(2, "tr", "va", "out", epochs=30, batch_size=8,
                                 lr=5e-5, window_size=16, imbalance_strategy="class_weight")
    assert cmd[:4] == ["-m", "htrtdetr.cli", "train", "stage=2"]
    assert "data.train_root=tr" in cmd and "data.val_root=va" in cmd
    assert "train.output_dir=out" in cmd
    assert "train.max_epochs=30" in cmd and "data.batch_size=8" in cmd
    assert "data.window_size=16" in cmd
    assert "loss.imbalance_strategy=class_weight" in cmd


def test_train_command_stage1_omits_optional():
    cmd = commands.train_command(1, "tr", "va", "out", epochs=50, batch_size=4, lr=1e-4)
    assert "stage=1" in cmd
    assert not any(a.startswith("data.window_size=") for a in cmd)
    assert not any(a.startswith("loss.imbalance_strategy=") for a in cmd)


# ─── commands.analyze バリデーション ───────────────────────────────────────

def test_analyze_missing_field():
    assert commands.analyze_missing_field("timeline", "", "") == "predictions"
    assert commands.analyze_missing_field("id_switches", "", "") == "predictions"
    assert commands.analyze_missing_field("confidence", "", "") == "predictions"
    assert commands.analyze_missing_field("distribution", "", "") == "annotation"
    assert commands.analyze_missing_field("timeline", "p.json", "") is None
    assert commands.analyze_missing_field("distribution", "", "a.json") is None


def test_analyze_command():
    cmd = commands.analyze_command("timeline", "out", predictions="p.json")
    assert cmd[:3] == ["-m", "htrtdetr.cli", "analyze"]
    assert "mode=timeline" in cmd and "predictions=p.json" in cmd and "output_dir=out" in cmd


# ─── predictions.load_predictions ─────────────────────────────────────────

def test_load_predictions_fills_missing_fields(tmp_path):
    data = {
        "0": {"boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
              "scores": [0.9, 0.5], "track_ids": [1, 2], "action_ids": [0, 1]},
        "5": {"boxes": [[50, 50, 60, 60]]},  # scores/track/action 欠損, class_ids なし
    }
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    preds = predictions.load_predictions(str(p))
    assert set(preds.keys()) == {0, 5}
    assert len(preds[0]) == 2
    # 欠損は既定値で補完
    fd5 = preds[5]
    assert fd5.scores == [1.0]
    assert fd5.track_ids == [-1]
    assert fd5.action_ids == [-1]
    assert fd5.class_ids == [0]

    assert predictions.frame_range(preds) == (0, 5)
    timelines = predictions.build_track_timelines(preds)
    assert timelines[1] == {0: 0}
    assert timelines[2] == {0: 1}


# ─── monitor パーサ ─────────────────────────────────────────────────────────

def test_read_results_csv_missing_file(tmp_path):
    assert monitor.read_results_csv(str(tmp_path / "nope.csv")) == []


def test_series_parses_and_skips_non_numeric(tmp_path):
    csv_text = (
        "epoch,train_loss,val_loss,ap50\n"
        "0,1.5,2.0,0.10\n"
        "1,1.0,,0.20\n"        # val_loss 欠損 → その行は val_loss 系列でスキップ
        "2,0.5,1.0,\n"          # ap50 欠損
    )
    p = tmp_path / "results.csv"
    p.write_text(csv_text, encoding="utf-8")
    rows = monitor.read_results_csv(str(p))
    assert len(rows) == 3

    xs, ys = monitor.series(rows, "epoch", "train_loss")
    assert xs == [0.0, 1.0, 2.0] and ys == [1.5, 1.0, 0.5]

    xs, ys = monitor.series(rows, "epoch", "val_loss")
    assert xs == [0.0, 2.0] and ys == [2.0, 1.0]  # epoch1 はスキップ

    xs, ys = monitor.series(rows, "epoch", "ap50")
    assert xs == [0.0, 1.0] and ys == [0.10, 0.20]


def test_read_metrics_latest(tmp_path):
    assert monitor.read_metrics_latest(str(tmp_path / "nope.json")) is None
    p = tmp_path / "metrics_latest.json"
    p.write_text(json.dumps({"epoch": 3, "primary_metric": "macro_f1",
                             "primary_metric_value": 0.7, "best_so_far": 0.7}),
                 encoding="utf-8")
    m = monitor.read_metrics_latest(str(p))
    assert m["epoch"] == 3 and m["primary_metric"] == "macro_f1"


def test_metric_columns_for_stage():
    assert ("ap50", "AP50") in monitor.metric_columns_for_stage(1)
    assert any(c == "idf1" for c, _ in monitor.metric_columns_for_stage(3))
    assert any(c == "composite_score" for c, _ in monitor.metric_columns_for_stage(4))
