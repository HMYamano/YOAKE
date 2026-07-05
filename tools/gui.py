"""
YOAKE GUI - Step-by-step training and analysis interface
Built with DearPyGui (main UI) + tkinter (file dialogs)
"""

import dearpygui.dearpygui as dpg
import subprocess
import threading
import sys
import os
import json
from pathlib import Path
from datetime import datetime

try:
    from htrtdetr.gui_cli import (
        _yoake as _shared_yoake,
        build_yoake_command as _shared_build_yoake_command,
    )
except ImportError:
    _SRC = Path(__file__).resolve().parent.parent / "src"
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from htrtdetr.gui_cli import (
        _yoake as _shared_yoake,
        build_yoake_command as _shared_build_yoake_command,
    )

# 修正済みコマンドビルダ (visualize→show_* マッピング / analyze バリデーション)
from htrtdetr.gui.commands import (
    analyze_command as _analyze_command,
    analyze_missing_field as _analyze_missing_field,
    predict_command as _predict_command,
)

# tkinter for file dialogs only
import tkinter as tk
from tkinter import filedialog

# ─────────────────────────── constants ────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
TOOLS = ROOT / "tools"
CONFIGS = ROOT / "configs"
RUNS = ROOT / "runs"           # 新しい出力ルート
OUTPUTS = ROOT / "outputs"     # legacy (参照用のみ)


# 共有ヘルパ (htrtdetr.gui_cli) にルーティング。ローカル定義は廃止した。
_yoake = _shared_yoake
build_yoake_command = _shared_build_yoake_command

STEPS = [
    "Dataset",
    "Stage 1: Detector",
    "Stage 2: Action",
    "Stage 3: ID",
    "Stage 4: Unified",
    "Inference",
    "Analysis",
]

STEP_COLORS = {
    "Dataset":        (100, 180, 100),
    "Stage 1: Detector": (80, 140, 200),
    "Stage 2: Action":   (160, 100, 200),
    "Stage 3: ID":       (200, 140,  80),
    "Stage 4: Unified":  (200,  80,  80),
    "Inference":         (80, 180, 180),
    "Analysis":          (180, 180,  80),
}

WINDOW_W, WINDOW_H = 1200, 800
SIDEBAR_W = 200
LOG_H = 200

# ─────────────────────────── state ────────────────────────────────
current_step = [0]
active_process = [None]
log_lines = []
MAX_LOG = 500

# ─────────────────────────── helpers ──────────────────────────────

def _tk_pick_file(title="ファイルを選択", filetypes=(("All", "*.*"),)):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return path or ""


def _tk_pick_dir(title="フォルダを選択"):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path or ""


def _tk_save_file(title="保存先を選択", defaultextension=".json",
                  filetypes=(("JSON", "*.json"), ("All", "*.*"))):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title=title, defaultextension=defaultextension, filetypes=filetypes)
    root.destroy()
    return path or ""


def append_log(text: str):
    """Append text to log panel (thread-safe via dpg.set_value)."""
    ts = datetime.now().strftime("%H:%M:%S")
    log_lines.append(f"[{ts}] {text}")
    if len(log_lines) > MAX_LOG:
        log_lines.pop(0)
    dpg.set_value("log_text", "\n".join(log_lines))
    # auto-scroll
    dpg.set_y_scroll("log_panel", dpg.get_y_scroll_max("log_panel"))


def run_command(cmd: list[str], cwd=None):
    """Run command in background thread, stream output to log."""
    def _worker():
        append_log(f"$ {' '.join(str(c) for c in cmd)}")
        dpg.configure_item("btn_stop", enabled=True)
        dpg.configure_item("status_bar", default_value="実行中...")
        try:
            proc = subprocess.Popen(
                [sys.executable] + [str(c) for c in cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cwd or ROOT),
                encoding="utf-8",
                errors="replace",
            )
            active_process[0] = proc
            for line in proc.stdout:
                append_log(line.rstrip())
            proc.wait()
            code = proc.returncode
            msg = "完了 ✓" if code == 0 else f"エラー (code={code})"
            append_log(f"--- {msg} ---")
            dpg.configure_item("status_bar", default_value=msg)
        except FileNotFoundError as e:
            append_log(f"[ERROR] コマンドが見つかりません: {e}")
            append_log("  ヒント: pip install -e . で yoake をインストールしてください")
            dpg.configure_item("status_bar", default_value="エラー: コマンド未発見")
        except Exception as e:
            append_log(f"[ERROR] {e}")
            dpg.configure_item("status_bar", default_value="エラー")
        finally:
            active_process[0] = None
            dpg.configure_item("btn_stop", enabled=False)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def stop_command():
    proc = active_process[0]
    if proc and proc.poll() is None:
        proc.terminate()
        append_log("--- 中断されました ---")


def _run_analyze_ui():
    """analyze 実行前に mode 別必須フィールドを検証してから実行する。"""
    mode = dpg.get_value("ana_mode")
    pred = dpg.get_value("ana_pred")
    gt = dpg.get_value("ana_gt")
    missing = _analyze_missing_field(mode, pred, gt)
    if missing:
        append_log(f"[ERROR] mode={mode} には {missing}= が必要です。")
        dpg.configure_item("status_bar", default_value="エラー: 入力不足")
        return
    run_command(_analyze_command(mode, dpg.get_value("ana_output"),
                                 predictions=pred, annotation=gt))


# ─────────────────────────── sidebar ──────────────────────────────

def build_sidebar():
    with dpg.child_window(tag="sidebar", width=SIDEBAR_W, height=-1, border=False):
        dpg.add_spacer(height=10)
        dpg.add_text("YOAKE GUI", color=(220, 220, 220))
        dpg.add_separator()
        dpg.add_spacer(height=8)
        dpg.add_text("ステップ", color=(160, 160, 160))
        dpg.add_spacer(height=4)
        for i, name in enumerate(STEPS):
            col = STEP_COLORS[name]
            dpg.add_button(
                label=f"  {i+1}. {name}",
                tag=f"step_btn_{i}",
                width=SIDEBAR_W - 16,
                height=36,
                callback=lambda s, a, u: switch_step(u),
                user_data=i,
            )
            dpg.bind_item_theme(f"step_btn_{i}", make_btn_theme(col, i == 0))

        dpg.add_spacer(height=12)
        dpg.add_separator()
        dpg.add_spacer(height=8)

        # Stop button
        dpg.add_button(
            label="  ■ 中断",
            tag="btn_stop",
            width=SIDEBAR_W - 16,
            height=32,
            callback=lambda: stop_command(),
            enabled=False,
        )
        dpg.add_spacer(height=4)
        dpg.add_text("", tag="status_bar", color=(180, 220, 180), wrap=SIDEBAR_W - 16)


def make_btn_theme(color, active=False):
    tag = f"theme_btn_{color}_{active}"
    if dpg.does_item_exist(tag):
        return tag
    r, g, b = color
    bg = (r, g, b, 255) if active else (r // 3, g // 3, b // 3, 200)
    hov = (min(r + 40, 255), min(g + 40, 255), min(b + 40, 255), 255)
    with dpg.theme(tag=tag):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, bg)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hov)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,
                                (min(r + 60, 255), min(g + 60, 255), min(b + 60, 255)))
    return tag


def switch_step(idx: int):
    old = current_step[0]
    current_step[0] = idx
    # update sidebar colors
    for i, name in enumerate(STEPS):
        col = STEP_COLORS[name]
        dpg.bind_item_theme(f"step_btn_{i}", make_btn_theme(col, i == idx))
    # show/hide content panels
    for i in range(len(STEPS)):
        dpg.configure_item(f"panel_{i}", show=(i == idx))


# ─────────────────────────── panels ───────────────────────────────

def field_row(label: str, tag: str, default: str = "", hint: str = "",
              btn_label="参照", btn_cb=None, width: int = 380):
    """One-line: label + text input + optional browse button."""
    with dpg.group(horizontal=True):
        dpg.add_text(f"{label}:", indent=4)
    with dpg.group(horizontal=True):
        dpg.add_input_text(tag=tag, default_value=default,
                           hint=hint, width=width)
        if btn_cb:
            dpg.add_button(label=btn_label, width=60, callback=btn_cb)


def int_row(label: str, tag: str, default: int = 0, width: int = 100):
    with dpg.group(horizontal=True):
        dpg.add_text(f"{label}:", indent=4)
        dpg.add_input_int(tag=tag, default_value=default, width=width)


def float_row(label: str, tag: str, default: float = 0.0, width: int = 120):
    with dpg.group(horizontal=True):
        dpg.add_text(f"{label}:", indent=4)
        dpg.add_input_float(tag=tag, default_value=default, format="%.6f", width=width)


def section(label):
    dpg.add_spacer(height=6)
    dpg.add_text(label, color=(160, 200, 240))
    dpg.add_separator()
    dpg.add_spacer(height=2)


# ── Panel 0: Dataset ──────────────────────────────────────────────

def panel_dataset(parent):
    with dpg.child_window(tag="panel_0", parent=parent,
                          width=-1, height=-1, show=True, border=False):
        dpg.add_text("Dataset 準備", color=(220, 220, 220))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        with dpg.tab_bar():

            # --- Convert Tab ---
            with dpg.tab(label="アノテーション変換"):
                dpg.add_spacer(height=6)
                section("入力 (CSV/MOT17)")
                field_row("入力ファイル", "conv_input", hint="annotations.csv",
                          btn_cb=lambda: dpg.set_value(
                              "conv_input",
                              _tk_pick_file("入力ファイル選択",
                                            [("CSV/JSON", "*.csv *.json"), ("All", "*.*")])))
                field_row("出力 JSON", "conv_output", hint="annotations.json",
                          btn_cb=lambda: dpg.set_value(
                              "conv_output",
                              _tk_save_file("出力先選択", ".json")))
                with dpg.group(horizontal=True):
                    dpg.add_text("フォーマット:", indent=4)
                    dpg.add_combo(["csv", "mot17"], tag="conv_format",
                                  default_value="csv", width=120)
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="変換実行",
                    height=32,
                    callback=lambda: run_command([
                        TOOLS / "convert_annotations.py",
                        f"input={dpg.get_value('conv_input')}",
                        f"output={dpg.get_value('conv_output')}",
                        f"format={dpg.get_value('conv_format')}",
                    ]),
                )

            # --- Splits Tab ---
            with dpg.tab(label="Train/Val/Test 分割"):
                dpg.add_spacer(height=6)
                section("アノテーションファイル")
                field_row("アノテーション JSON", "split_anno",
                          btn_cb=lambda: dpg.set_value(
                              "split_anno",
                              _tk_pick_file("アノテーション JSON",
                                            [("JSON", "*.json"), ("All", "*.*")])))
                field_row("出力ディレクトリ", "split_output",
                          hint="data/splits/",
                          btn_cb=lambda: dpg.set_value(
                              "split_output", _tk_pick_dir("出力ディレクトリ")))
                dpg.add_spacer(height=4)
                section("分割比率")
                float_row("Train 比率", "split_train", 0.7)
                float_row("Val 比率",   "split_val",   0.15)
                float_row("Test 比率",  "split_test",  0.15)
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="分割実行",
                    height=32,
                    callback=lambda: run_command([
                        TOOLS / "build_splits.py",
                        f"anno={dpg.get_value('split_anno')}",
                        f"output={dpg.get_value('split_output')}",
                        f"train_ratio={dpg.get_value('split_train')}",
                        f"val_ratio={dpg.get_value('split_val')}",
                    ]),
                )

            # --- Visualize Tab ---
            with dpg.tab(label="データ確認"):
                dpg.add_spacer(height=6)
                field_row("アノテーション JSON", "vis_anno",
                          btn_cb=lambda: dpg.set_value(
                              "vis_anno",
                              _tk_pick_file("アノテーション JSON",
                                            [("JSON", "*.json"), ("All", "*.*")])))
                dpg.add_spacer(height=8)
                dpg.add_button(label="統計表示", height=32,
                               callback=lambda: run_command([
                                   TOOLS / "summarize_dataset.py",
                                   f"anno={dpg.get_value('vis_anno')}",
                               ]))
                dpg.add_button(label="可視化", height=32,
                               callback=lambda: run_command([
                                   TOOLS / "visualize_dataset.py",
                                   f"anno={dpg.get_value('vis_anno')}",
                               ]))


# ── Panel 1: Stage 1 ──────────────────────────────────────────────

def panel_stage1(parent):
    with dpg.child_window(tag="panel_1", parent=parent,
                          width=-1, height=-1, show=False, border=False):
        dpg.add_text("Stage 1: Spatial Detector 学習", color=(80, 140, 200))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        with dpg.tab_bar():

            with dpg.tab(label="学習"):
                dpg.add_spacer(height=6)
                section("データ")
                field_row("Train アノテーション", "s1_train_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s1_train_anno",
                              _tk_pick_file("Train JSON", [("JSON", "*.json")])))
                field_row("Val アノテーション", "s1_val_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s1_val_anno",
                              _tk_pick_file("Val JSON", [("JSON", "*.json")])))
                field_row("出力ディレクトリ", "s1_output",
                          default=str(RUNS / "train" / "stage1"),
                          btn_cb=lambda: dpg.set_value(
                              "s1_output", _tk_pick_dir("出力ディレクトリ")))
                dpg.add_spacer(height=4)
                section("ハイパーパラメータ")
                int_row("Epochs", "s1_epochs", 50)
                int_row("Batch Size", "s1_batch", 4)
                float_row("Learning Rate", "s1_lr", 1e-4)
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 1 学習開始",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "train", "stage=1",
                        f"data.train_root={dpg.get_value('s1_train_anno')}",
                        f"data.val_root={dpg.get_value('s1_val_anno')}",
                        f"train.output_dir={dpg.get_value('s1_output')}",
                        f"train.max_epochs={dpg.get_value('s1_epochs')}",
                        f"data.batch_size={dpg.get_value('s1_batch')}",
                        f"optimizer.lr={dpg.get_value('s1_lr')}",
                    )),
                )

            with dpg.tab(label="評価"):
                dpg.add_spacer(height=6)
                section("評価設定")
                field_row("アノテーション JSON", "s1_eval_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s1_eval_anno",
                              _tk_pick_file("アノテーション JSON", [("JSON", "*.json")])))
                field_row("チェックポイント (.pt)", "s1_ckpt",
                          hint="outputs/stage1/best.pt",
                          btn_cb=lambda: dpg.set_value(
                              "s1_ckpt",
                              _tk_pick_file("チェックポイント", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s1_eval_out",
                          default=str(RUNS / "val" / "stage1"),
                          btn_cb=lambda: dpg.set_value(
                              "s1_eval_out", _tk_pick_dir()))
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 1 評価実行",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "val", "stage=1",
                        f"checkpoint={dpg.get_value('s1_ckpt')}",
                        f"data.val_root={dpg.get_value('s1_eval_anno')}",
                        f"output_dir={dpg.get_value('s1_eval_out')}",
                    )),
                )


# ── Panel 2: Stage 2 ──────────────────────────────────────────────

def panel_stage2(parent):
    with dpg.child_window(tag="panel_2", parent=parent,
                          width=-1, height=-1, show=False, border=False):
        dpg.add_text("Stage 2: Action Head 学習", color=(160, 100, 200))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        with dpg.tab_bar():

            with dpg.tab(label="学習"):
                dpg.add_spacer(height=6)
                section("データ & チェックポイント")
                field_row("Train アノテーション", "s2_train_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s2_train_anno", _tk_pick_file("Train JSON", [("JSON", "*.json")])))
                field_row("Val アノテーション", "s2_val_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s2_val_anno", _tk_pick_file("Val JSON", [("JSON", "*.json")])))
                field_row("Stage 1 チェックポイント", "s2_s1_ckpt",
                          hint="outputs/stage1/best.pt",
                          btn_cb=lambda: dpg.set_value(
                              "s2_s1_ckpt", _tk_pick_file("Stage1 .pt", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s2_output",
                          default=str(RUNS / "train" / "stage2"),
                          btn_cb=lambda: dpg.set_value(
                              "s2_output", _tk_pick_dir()))
                dpg.add_spacer(height=4)
                section("ハイパーパラメータ")
                int_row("Epochs", "s2_epochs", 30)
                int_row("Batch Size", "s2_batch", 8)
                float_row("Learning Rate", "s2_lr", 5e-5)
                with dpg.group(horizontal=True):
                    dpg.add_text("クラス不均衡対策:", indent=4)
                    dpg.add_combo(
                        ["none", "class_weight", "focal"],
                        tag="s2_imbalance", default_value="class_weight", width=140,
                    )
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 2 学習開始",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "train", "stage=2",
                        f"data.train_root={dpg.get_value('s2_train_anno')}",
                        f"data.val_root={dpg.get_value('s2_val_anno')}",
                        f"train.output_dir={dpg.get_value('s2_output')}",
                        f"train.max_epochs={dpg.get_value('s2_epochs')}",
                        f"data.batch_size={dpg.get_value('s2_batch')}",
                        f"optimizer.lr={dpg.get_value('s2_lr')}",
                        f"loss.imbalance_strategy={dpg.get_value('s2_imbalance')}",
                    )),
                )

            with dpg.tab(label="評価"):
                dpg.add_spacer(height=6)
                field_row("アノテーション JSON", "s2_eval_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s2_eval_anno", _tk_pick_file("JSON", [("JSON", "*.json")])))
                field_row("チェックポイント", "s2_ckpt",
                          btn_cb=lambda: dpg.set_value(
                              "s2_ckpt", _tk_pick_file("PT", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s2_eval_out",
                          default=str(RUNS / "val" / "stage2"),
                          btn_cb=lambda: dpg.set_value("s2_eval_out", _tk_pick_dir()))
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 2 評価実行",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "val", "stage=2",
                        f"checkpoint={dpg.get_value('s2_ckpt')}",
                        f"data.val_root={dpg.get_value('s2_eval_anno')}",
                        f"output_dir={dpg.get_value('s2_eval_out')}",
                    )),
                )


# ── Panel 3: Stage 3 ──────────────────────────────────────────────

def panel_stage3(parent):
    with dpg.child_window(tag="panel_3", parent=parent,
                          width=-1, height=-1, show=False, border=False):
        dpg.add_text("Stage 3: ID Head 学習", color=(200, 140, 80))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        with dpg.tab_bar():

            with dpg.tab(label="学習"):
                dpg.add_spacer(height=6)
                section("データ & チェックポイント")
                field_row("Train アノテーション", "s3_train_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s3_train_anno", _tk_pick_file("Train JSON", [("JSON", "*.json")])))
                field_row("Val アノテーション", "s3_val_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s3_val_anno", _tk_pick_file("Val JSON", [("JSON", "*.json")])))
                field_row("Stage 1 チェックポイント", "s3_s1_ckpt",
                          hint="outputs/stage1/best.pt",
                          btn_cb=lambda: dpg.set_value(
                              "s3_s1_ckpt", _tk_pick_file("Stage1 .pt", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s3_output",
                          default=str(RUNS / "train" / "stage3"),
                          btn_cb=lambda: dpg.set_value("s3_output", _tk_pick_dir()))
                dpg.add_spacer(height=4)
                section("ハイパーパラメータ")
                int_row("Epochs", "s3_epochs", 30)
                int_row("Batch Size", "s3_batch", 8)
                float_row("Learning Rate", "s3_lr", 5e-5)
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 3 学習開始",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "train", "stage=3",
                        f"data.train_root={dpg.get_value('s3_train_anno')}",
                        f"data.val_root={dpg.get_value('s3_val_anno')}",
                        f"train.output_dir={dpg.get_value('s3_output')}",
                        f"train.max_epochs={dpg.get_value('s3_epochs')}",
                        f"data.batch_size={dpg.get_value('s3_batch')}",
                        f"optimizer.lr={dpg.get_value('s3_lr')}",
                    )),
                )

            with dpg.tab(label="評価"):
                dpg.add_spacer(height=6)
                field_row("アノテーション JSON", "s3_eval_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s3_eval_anno", _tk_pick_file("JSON", [("JSON", "*.json")])))
                field_row("チェックポイント", "s3_ckpt",
                          btn_cb=lambda: dpg.set_value(
                              "s3_ckpt", _tk_pick_file("PT", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s3_eval_out",
                          default=str(RUNS / "val" / "stage3"),
                          btn_cb=lambda: dpg.set_value("s3_eval_out", _tk_pick_dir()))
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 3 評価実行",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "val", "stage=3",
                        f"checkpoint={dpg.get_value('s3_ckpt')}",
                        f"data.val_root={dpg.get_value('s3_eval_anno')}",
                        f"output_dir={dpg.get_value('s3_eval_out')}",
                    )),
                )


# ── Panel 4: Stage 4 ──────────────────────────────────────────────

def panel_stage4(parent):
    with dpg.child_window(tag="panel_4", parent=parent,
                          width=-1, height=-1, show=False, border=False):
        dpg.add_text("Stage 4: Unified Fine-tuning", color=(200, 80, 80))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        with dpg.tab_bar():

            with dpg.tab(label="学習"):
                dpg.add_spacer(height=6)
                section("データ & チェックポイント")
                field_row("Train アノテーション", "s4_train_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s4_train_anno", _tk_pick_file("Train JSON", [("JSON", "*.json")])))
                field_row("Val アノテーション", "s4_val_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s4_val_anno", _tk_pick_file("Val JSON", [("JSON", "*.json")])))
                field_row("Stage 1 チェックポイント", "s4_s1_ckpt",
                          btn_cb=lambda: dpg.set_value(
                              "s4_s1_ckpt", _tk_pick_file("Stage1", [("PyTorch", "*.pt *.pth")])))
                field_row("Stage 2 チェックポイント", "s4_s2_ckpt",
                          btn_cb=lambda: dpg.set_value(
                              "s4_s2_ckpt", _tk_pick_file("Stage2", [("PyTorch", "*.pt *.pth")])))
                field_row("Stage 3 チェックポイント", "s4_s3_ckpt",
                          btn_cb=lambda: dpg.set_value(
                              "s4_s3_ckpt", _tk_pick_file("Stage3", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s4_output",
                          default=str(RUNS / "train" / "stage4"),
                          btn_cb=lambda: dpg.set_value("s4_output", _tk_pick_dir()))
                dpg.add_spacer(height=4)
                section("ハイパーパラメータ")
                int_row("Epochs", "s4_epochs", 20)
                int_row("Batch Size", "s4_batch", 2)
                float_row("Learning Rate", "s4_lr", 1e-5)
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Stage 4 学習開始",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "train", "stage=4",
                        f"data.train_root={dpg.get_value('s4_train_anno')}",
                        f"data.val_root={dpg.get_value('s4_val_anno')}",
                        f"train.output_dir={dpg.get_value('s4_output')}",
                        f"train.max_epochs={dpg.get_value('s4_epochs')}",
                        f"data.batch_size={dpg.get_value('s4_batch')}",
                        f"optimizer.lr={dpg.get_value('s4_lr')}",
                    )),
                )

            with dpg.tab(label="評価"):
                dpg.add_spacer(height=6)
                field_row("アノテーション JSON", "s4_eval_anno",
                          btn_cb=lambda: dpg.set_value(
                              "s4_eval_anno", _tk_pick_file("JSON", [("JSON", "*.json")])))
                field_row("チェックポイント", "s4_ckpt",
                          btn_cb=lambda: dpg.set_value(
                              "s4_ckpt", _tk_pick_file("PT", [("PyTorch", "*.pt *.pth")])))
                field_row("出力ディレクトリ", "s4_eval_out",
                          default=str(RUNS / "val" / "stage4"),
                          btn_cb=lambda: dpg.set_value("s4_eval_out", _tk_pick_dir()))
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Unified 評価実行",
                    height=36,
                    callback=lambda: run_command(_yoake(
                        "val", "stage=4",
                        f"checkpoint={dpg.get_value('s4_ckpt')}",
                        f"data.val_root={dpg.get_value('s4_eval_anno')}",
                        f"output_dir={dpg.get_value('s4_eval_out')}",
                    )),
                )


# ── Panel 5: Inference ───────────────────────────────────────────

def panel_inference(parent):
    with dpg.child_window(tag="panel_5", parent=parent,
                          width=-1, height=-1, show=False, border=False):
        dpg.add_text("Inference: 動画推論", color=(80, 180, 180))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        section("入力")
        field_row("入力動画 (.mp4)", "inf_input",
                  btn_cb=lambda: dpg.set_value(
                      "inf_input",
                      _tk_pick_file("動画ファイル", [("動画", "*.mp4 *.avi *.mov"), ("All", "*.*")])))
        field_row("チェックポイント", "inf_ckpt",
                  hint="outputs/stage4/best.pt",
                  btn_cb=lambda: dpg.set_value(
                      "inf_ckpt", _tk_pick_file("チェックポイント", [("PyTorch", "*.pt *.pth")])))
        field_row("出力ディレクトリ", "inf_output",
                  default=str(RUNS / "predict"),
                  btn_cb=lambda: dpg.set_value("inf_output", _tk_pick_dir()))

        dpg.add_spacer(height=4)
        section("推論設定")
        float_row("Score Threshold", "inf_thresh", 0.5)
        int_row("Window Size (フレーム)", "inf_window", 16)
        int_row("Max Frames (0=全フレーム)", "inf_maxframes", 0)
        with dpg.group(horizontal=True):
            dpg.add_text("可視化オーバーレイ:", indent=4)
            dpg.add_checkbox(tag="inf_vis", default_value=True)

        dpg.add_spacer(height=8)
        dpg.add_button(
            label="推論実行",
            height=36,
            callback=lambda: run_command(_predict_command(
                dpg.get_value("inf_input"),
                dpg.get_value("inf_ckpt"),
                dpg.get_value("inf_output"),
                score_threshold=float(dpg.get_value("inf_thresh")),
                window_size=int(dpg.get_value("inf_window")),
                max_frames=int(dpg.get_value("inf_maxframes")) or None,
                visualize=bool(dpg.get_value("inf_vis")),
            )),
        )


# ── Panel 6: Analysis ─────────────────────────────────────────────

def panel_analysis(parent):
    with dpg.child_window(tag="panel_6", parent=parent,
                          width=-1, height=-1, show=False, border=False):
        dpg.add_text("Analysis: 結果解析", color=(180, 180, 80))
        dpg.add_separator()
        dpg.add_spacer(height=8)

        section("入力")
        field_row("予測 JSON", "ana_pred",
                  btn_cb=lambda: dpg.set_value(
                      "ana_pred", _tk_pick_file("予測 JSON", [("JSON", "*.json")])))
        field_row("Ground Truth JSON (任意)", "ana_gt",
                  btn_cb=lambda: dpg.set_value(
                      "ana_gt", _tk_pick_file("GT JSON", [("JSON", "*.json")])))
        field_row("出力ディレクトリ", "ana_output",
                  default=str(RUNS / "analyze"),
                  btn_cb=lambda: dpg.set_value("ana_output", _tk_pick_dir()))

        dpg.add_spacer(height=4)
        section("解析モード")
        with dpg.group(horizontal=True):
            dpg.add_text("モード:", indent=4)
            dpg.add_combo(
                ["timeline", "distribution", "id_switches", "confidence"],
                tag="ana_mode", default_value="timeline", width=160,
            )
        dpg.add_spacer(height=8)

        with dpg.group(horizontal=True):
            dpg.add_button(
                label="解析実行",
                height=36,
                callback=lambda: _run_analyze_ui(),
            )
            dpg.add_spacer(width=8)
            dpg.add_button(
                label="アブレーション集計",
                height=36,
                callback=lambda: run_command([
                    TOOLS / "run_ablation.py",
                    f"output_dir={dpg.get_value('ana_output')}",
                ]),
            )


# ─────────────────────────── log panel ────────────────────────────

def build_log_panel(parent):
    with dpg.child_window(tag="log_panel", parent=parent,
                          width=-1, height=LOG_H, border=True):
        dpg.add_text("", tag="log_text", wrap=WINDOW_W - SIDEBAR_W - 32)


# ─────────────────────────── global theme ─────────────────────────

def apply_global_theme():
    with dpg.theme(tag="global_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        (22, 22, 30))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         (28, 28, 38))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         (40, 40, 55))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  (55, 55, 75))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,   (65, 65, 90))
            dpg.add_theme_color(dpg.mvThemeCol_Text,            (220, 220, 220))
            dpg.add_theme_color(dpg.mvThemeCol_Border,          (60, 60, 80))
            dpg.add_theme_color(dpg.mvThemeCol_Header,          (50, 80, 120, 180))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,   (70, 110, 160))
            dpg.add_theme_color(dpg.mvThemeCol_Tab,             (35, 35, 50))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,      (60, 80, 120))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,       (50, 80, 140))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         (18, 18, 26))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   (30, 30, 50))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     (20, 20, 28))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   (60, 60, 85))
            dpg.add_theme_color(dpg.mvThemeCol_Button,          (45, 80, 130))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   (60, 110, 170))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    (40, 70, 120))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  6)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   4)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    6, 4)
    dpg.bind_theme("global_theme")


# ─────────────────────────── main ─────────────────────────────────

def main():
    dpg.create_context()
    apply_global_theme()

    dpg.create_viewport(
        title="YOAKE GUI",
        width=WINDOW_W,
        height=WINDOW_H,
        resizable=True,
    )
    dpg.setup_dearpygui()

    with dpg.window(tag="main_win", label="YOAKE GUI",
                    no_title_bar=True, no_move=True, no_resize=True,
                    no_scrollbar=True):

        # top: sidebar + content area
        with dpg.group(horizontal=True):
            build_sidebar()

            # right column: content panels + log
            with dpg.group(horizontal=False):
                with dpg.child_window(tag="content_area",
                                      width=-1,
                                      height=-(LOG_H + 12),
                                      border=False):
                    panel_dataset("content_area")
                    panel_stage1("content_area")
                    panel_stage2("content_area")
                    panel_stage3("content_area")
                    panel_stage4("content_area")
                    panel_inference("content_area")
                    panel_analysis("content_area")

                dpg.add_separator()
                build_log_panel("main_win")

    # keep main window filling viewport
    def _resize_cb():
        w = dpg.get_viewport_width()
        h = dpg.get_viewport_height()
        dpg.set_item_width("main_win", w)
        dpg.set_item_height("main_win", h)
        dpg.set_item_pos("main_win", [0, 0])

    dpg.set_viewport_resize_callback(_resize_cb)
    _resize_cb()

    dpg.show_viewport()
    append_log("YOAKE GUI を起動しました。左のメニューからステップを選択してください。")
    append_log(f"  ROOT: {ROOT}")
    append_log(f"  出力先: {RUNS}")
    append_log("  学習・評価は yoake CLI (python -m htrtdetr.cli) 経由で実行されます。")

    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
