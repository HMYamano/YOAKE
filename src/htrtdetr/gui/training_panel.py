"""
training_panel.py — Training GUI (dearpygui)
============================================

Dataset 準備 / Stage 1–4 の学習・評価 / analyze を ``yoake`` CLI 経由で実行し、
``results.csv`` と ``metrics_latest.json`` を tail してライブに損失・指標をプロットする。

コマンド生成は ``htrtdetr.gui.commands`` に集約 (visualize バグ修正・window_size 追加
などはそこで担保)。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import dearpygui.dearpygui as dpg

from . import commands, dialogs
from .monitor import TrainingMonitor, metric_columns_for_stage, series
from .runner import CommandRunner

_LOSS_SERIES = [("train_loss", "train_loss"), ("val_loss", "val_loss")]
_METRIC_SERIES = ["ap50", "macro_f1", "idf1", "composite_score"]


class TrainingPanel:
    """Training ページ。ハブから ``build(parent)`` で構築する。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.tools = self.root / "tools"
        self.runs = self.root / "runs"
        self.runner = CommandRunner("train_log", "train_status", cwd=self.root)
        self.monitor: Optional[TrainingMonitor] = None
        self.monitor_stage = 1

    # ═══════════════════════════════════════════════════════════════════
    def build(self, parent: str) -> None:
        with dpg.child_window(tag="panel_training", parent=parent,
                              width=-1, height=-1, border=False, show=False):
            dpg.add_text("Training — 段階的学習", color=(120, 160, 220))
            dpg.add_separator()

            with dpg.group(horizontal=True):
                # 左: フォーム群
                with dpg.child_window(width=560, height=-150, border=False):
                    with dpg.tab_bar():
                        self._tab_dataset()
                        for stage in (1, 2, 3, 4):
                            self._tab_stage(stage)
                        self._tab_eval_analyze()
                # 右: ライブモニタ
                with dpg.child_window(width=-1, height=-150, border=True):
                    self._build_monitor()

            dpg.add_text("", tag="train_status", color=(180, 220, 180))
            with dpg.child_window(tag="train_log_win", width=-1, height=130, border=True):
                dpg.add_text("", tag="train_log", wrap=1100)

    # ── フォーム部品 ──────────────────────────────────────────────────
    def _field(self, label, tag, default="", browse=None, width=330):
        dpg.add_text(f"{label}:", indent=4)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag=tag, default_value=default, width=width)
            if browse is not None:
                dpg.add_button(label="参照", width=54, callback=browse)

    def _pick_json(self, tag):
        return lambda: self._set(tag, dialogs.pick_file("JSON/データ", dialogs.JSON_TYPES))

    def _pick_ckpt(self, tag):
        return lambda: self._set(tag, dialogs.pick_file("チェックポイント", dialogs.CKPT_TYPES))

    def _pick_dir(self, tag):
        return lambda: self._set(tag, dialogs.pick_dir())

    def _set(self, tag, value):
        if value:
            dpg.set_value(tag, value)

    # ── Dataset タブ ──────────────────────────────────────────────────
    def _tab_dataset(self):
        with dpg.tab(label="Dataset"):
            dpg.add_text("アノテーション変換 / 分割 / 統計", color=(160, 200, 240))
            dpg.add_separator()
            self._field("入力 (csv/mot17)", "ds_conv_in", browse=self._pick_json("ds_conv_in"))
            self._field("出力 JSON", "ds_conv_out",
                        browse=lambda: self._set("ds_conv_out", dialogs.save_file()))
            with dpg.group(horizontal=True):
                dpg.add_text("format:", indent=4)
                dpg.add_combo(["csv", "mot17"], tag="ds_conv_fmt", default_value="csv", width=120)
            dpg.add_button(label="変換実行", callback=lambda: self.runner.run([
                self.tools / "convert_annotations.py",
                f"input={dpg.get_value('ds_conv_in')}",
                f"output={dpg.get_value('ds_conv_out')}",
                f"format={dpg.get_value('ds_conv_fmt')}",
            ]))
            dpg.add_separator()
            self._field("アノテーション JSON", "ds_split_anno", browse=self._pick_json("ds_split_anno"))
            self._field("分割出力先", "ds_split_out", browse=self._pick_dir("ds_split_out"))
            with dpg.group(horizontal=True):
                dpg.add_text("train:", indent=4)
                dpg.add_input_float(tag="ds_train_ratio", default_value=0.7, width=70, format="%.2f", step=0)
                dpg.add_text("val:")
                dpg.add_input_float(tag="ds_val_ratio", default_value=0.15, width=70, format="%.2f", step=0)
            dpg.add_button(label="分割実行", callback=lambda: self.runner.run([
                self.tools / "build_splits.py",
                f"anno={dpg.get_value('ds_split_anno')}",
                f"output={dpg.get_value('ds_split_out')}",
                f"train_ratio={dpg.get_value('ds_train_ratio')}",
                f"val_ratio={dpg.get_value('ds_val_ratio')}",
            ]))
            dpg.add_separator()
            self._field("統計対象 JSON", "ds_stat_anno", browse=self._pick_json("ds_stat_anno"))
            dpg.add_button(label="統計表示", callback=lambda: self.runner.run([
                self.tools / "summarize_dataset.py", f"anno={dpg.get_value('ds_stat_anno')}",
            ]))
            dpg.add_spacer(height=4)
            dpg.add_text("※ 動画からのラベリングは左ナビ「Labeling」から。", color=(150, 150, 160))

    # ── Stage タブ ────────────────────────────────────────────────────
    def _tab_stage(self, stage: int):
        p = f"s{stage}"
        defaults = {1: (50, 4, 1e-4), 2: (30, 8, 5e-5), 3: (30, 8, 5e-5), 4: (20, 2, 1e-5)}
        ep, bs, lr = defaults[stage]
        with dpg.tab(label=f"Stage {stage}"):
            dpg.add_text(f"Stage {stage} 学習", color=(160, 200, 240))
            dpg.add_separator()
            self._field("Train データ", f"{p}_train", browse=self._pick_json(f"{p}_train"))
            self._field("Val データ", f"{p}_val", browse=self._pick_json(f"{p}_val"))
            self._field("出力ディレクトリ", f"{p}_out",
                        default=str(self.runs / "train" / f"stage{stage}"),
                        browse=self._pick_dir(f"{p}_out"))
            with dpg.group(horizontal=True):
                dpg.add_text("epochs:", indent=4)
                dpg.add_input_int(tag=f"{p}_epochs", default_value=ep, width=90)
                dpg.add_text(" batch:")
                dpg.add_input_int(tag=f"{p}_batch", default_value=bs, width=80)
                dpg.add_text(" lr:")
                dpg.add_input_float(tag=f"{p}_lr", default_value=lr, width=110, format="%.6f", step=0)
            if stage >= 2:
                with dpg.group(horizontal=True):
                    dpg.add_text("window_size:", indent=4)
                    dpg.add_input_int(tag=f"{p}_window", default_value=16, width=90)
                    if stage == 2:
                        dpg.add_text(" 不均衡対策:")
                        dpg.add_combo(["none", "class_weight", "focal"], tag=f"{p}_imb",
                                      default_value="class_weight", width=140)
            dpg.add_button(label=f"Stage {stage} 学習開始", height=32,
                           callback=lambda s=stage: self._start_train(s))
            dpg.add_separator()
            self._field("評価チェックポイント", f"{p}_ckpt", browse=self._pick_ckpt(f"{p}_ckpt"))
            self._field("評価 Val データ", f"{p}_eval_val", browse=self._pick_json(f"{p}_eval_val"))
            self._field("評価出力先", f"{p}_eval_out",
                        default=str(self.runs / "val" / f"stage{stage}"),
                        browse=self._pick_dir(f"{p}_eval_out"))
            dpg.add_button(label=f"Stage {stage} 評価実行", height=32,
                           callback=lambda s=stage: self._start_eval(s))

    # ── Eval / Analyze タブ ──────────────────────────────────────────
    def _tab_eval_analyze(self):
        with dpg.tab(label="Analyze"):
            dpg.add_text("解析 (timeline / distribution / id_switches / confidence)",
                         color=(160, 200, 240))
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("mode:", indent=4)
                dpg.add_combo(["timeline", "distribution", "id_switches", "confidence"],
                              tag="an_mode", default_value="timeline", width=160)
            self._field("予測 JSON (timeline/id_switches/confidence)", "an_pred",
                        browse=self._pick_json("an_pred"))
            self._field("アノテーション JSON (distribution)", "an_anno",
                        browse=self._pick_json("an_anno"))
            self._field("出力先", "an_out", default=str(self.runs / "analyze"),
                        browse=self._pick_dir("an_out"))
            dpg.add_button(label="解析実行", height=32, callback=self._start_analyze)

    # ═══════════════════════════════════════════════════════════════════
    # 実行
    # ═══════════════════════════════════════════════════════════════════
    def _start_train(self, stage: int):
        p = f"s{stage}"
        train_root = dpg.get_value(f"{p}_train")
        val_root = dpg.get_value(f"{p}_val")
        out = dpg.get_value(f"{p}_out")
        if not train_root:
            self.runner.log(f"[ERROR] Stage {stage}: Train データを指定してください。")
            return
        window = int(dpg.get_value(f"{p}_window")) if stage >= 2 and dpg.does_item_exist(f"{p}_window") else None
        imb = dpg.get_value(f"{p}_imb") if stage == 2 and dpg.does_item_exist(f"{p}_imb") else None
        cmd = commands.train_command(
            stage, train_root, val_root, out,
            epochs=int(dpg.get_value(f"{p}_epochs")),
            batch_size=int(dpg.get_value(f"{p}_batch")),
            lr=float(dpg.get_value(f"{p}_lr")),
            window_size=window,
            imbalance_strategy=imb,
        )
        self.attach_monitor(out, stage)
        self.runner.run(cmd)

    def _start_eval(self, stage: int):
        p = f"s{stage}"
        ckpt = dpg.get_value(f"{p}_ckpt")
        if not ckpt:
            self.runner.log(f"[ERROR] Stage {stage}: 評価チェックポイントを指定してください。")
            return
        cmd = commands.val_command(
            stage, ckpt, dpg.get_value(f"{p}_eval_val"), dpg.get_value(f"{p}_eval_out"),
        )
        self.runner.run(cmd)

    def _start_analyze(self):
        mode = dpg.get_value("an_mode")
        pred = dpg.get_value("an_pred")
        anno = dpg.get_value("an_anno")
        missing = commands.analyze_missing_field(mode, pred, anno)
        if missing:
            self.runner.log(f"[ERROR] mode={mode} には {missing}= が必要です。")
            return
        cmd = commands.analyze_command(mode, dpg.get_value("an_out"),
                                       predictions=pred, annotation=anno)
        self.runner.run(cmd)

    # ═══════════════════════════════════════════════════════════════════
    # ライブモニタ
    # ═══════════════════════════════════════════════════════════════════
    def _build_monitor(self):
        dpg.add_text("ライブ学習モニタ", color=(200, 200, 120))
        dpg.add_text("", tag="mon_summary", color=(180, 220, 180), wrap=520)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag="mon_dir", width=360, hint="runs/train/stageN")
            dpg.add_button(label="監視", callback=self._attach_monitor_manual)
        with dpg.plot(label="Loss", height=180, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="mon_loss_x")
            with dpg.plot_axis(dpg.mvYAxis, label="loss", tag="mon_loss_y"):
                for tag, label in _LOSS_SERIES:
                    dpg.add_line_series([], [], label=label, tag=f"mon_s_{tag}")
        with dpg.plot(label="Metrics", height=180, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="mon_metric_x")
            with dpg.plot_axis(dpg.mvYAxis, label="value", tag="mon_metric_y"):
                for col in _METRIC_SERIES:
                    dpg.add_line_series([], [], label=col, tag=f"mon_s_{col}")

    def attach_monitor(self, run_dir: str, stage: int):
        if not run_dir:
            return
        if self.monitor is not None:
            self.monitor.stop()
        self.monitor_stage = stage
        dpg.set_value("mon_dir", run_dir)
        self.monitor = TrainingMonitor(run_dir, self._on_monitor_update, interval=2.0)
        self.monitor.start()

    def _attach_monitor_manual(self):
        run_dir = dpg.get_value("mon_dir")
        # ディレクトリ名から stage を推測
        stage = self.monitor_stage
        for s in (1, 2, 3, 4):
            if f"stage{s}" in str(run_dir):
                stage = s
        self.attach_monitor(run_dir, stage)

    def _on_monitor_update(self, rows: List[Dict[str, str]], latest: Optional[dict]):
        for tag, col in _LOSS_SERIES:
            xs, ys = series(rows, "epoch", col)
            if dpg.does_item_exist(f"mon_s_{tag}"):
                dpg.set_value(f"mon_s_{tag}", [xs, ys])
        active_cols = {c for c, _ in metric_columns_for_stage(self.monitor_stage)}
        for col in _METRIC_SERIES:
            xs, ys = series(rows, "epoch", col) if col in active_cols else ([], [])
            if dpg.does_item_exist(f"mon_s_{col}"):
                dpg.set_value(f"mon_s_{col}", [xs, ys])
        for ax in ("mon_loss_x", "mon_loss_y", "mon_metric_x", "mon_metric_y"):
            if dpg.does_item_exist(ax):
                dpg.fit_axis_data(ax)
        if latest and dpg.does_item_exist("mon_summary"):
            dpg.set_value(
                "mon_summary",
                f"epoch {latest.get('epoch', '?')} | "
                f"{latest.get('primary_metric', '?')}={latest.get('primary_metric_value', '?')} | "
                f"best={latest.get('best_so_far', '?')}",
            )

    def teardown(self):
        if self.monitor is not None:
            self.monitor.stop()
