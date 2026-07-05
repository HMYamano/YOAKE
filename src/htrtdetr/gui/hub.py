"""
hub.py — YOAKE 統一 GUI ハブ (dearpygui)
========================================

YORU の ``app.py`` に相当する単一ウィンドウのランチャー/ハブ。左ナビから
**Labeling / Training / Video Analysis** を切り替える。

- Labeling: 範囲ラベリング対応のアノテータ (``htrtdetr.annotator``) を別プロセスで起動
  (dearpygui は 1 プロセス 1 コンテキストのため、注釈ツールは独立プロセスで動かす。
  これは YORU app.py が各モジュールを起動するのと同じ考え方)。
- Training: 段階的学習 + ライブモニタ (``TrainingPanel``)。
- Video Analysis: オフライン動画解析ビューア (``AnalysisPanel``)。

起動: ``python -m htrtdetr.gui`` / ``yoake gui`` / ``gui`` (エントリポイント)。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg

from . import dialogs
from .analysis_panel import AnalysisPanel
from .training_panel import TrainingPanel

WINDOW_W, WINDOW_H = 1360, 900
NAV_W = 190

_PAGES = [
    ("labeling", "🖉  Labeling", (100, 180, 120)),
    ("training", "⚙  Training", (90, 140, 210)),
    ("analysis", "▶  Video Analysis", (80, 200, 200)),
]


def _repo_root() -> Path:
    # src/htrtdetr/gui/hub.py -> repo root は 3 つ上
    return Path(__file__).resolve().parents[3]


class Hub:
    def __init__(self) -> None:
        self.root = _repo_root()
        self.current = "labeling"
        self.training = TrainingPanel(self.root)
        self.analysis = AnalysisPanel(self.root)

    # ═══════════════════════════════════════════════════════════════════
    def _theme(self) -> None:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 22, 30))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (28, 28, 38))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (40, 40, 55))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 80, 130))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 110, 170))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (35, 35, 50))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (50, 80, 140))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
        dpg.bind_theme(theme)

    # ── ナビ ─────────────────────────────────────────────────────────
    def _build_nav(self) -> None:
        with dpg.child_window(width=NAV_W, height=-1, border=False):
            dpg.add_spacer(height=8)
            dpg.add_text("YOAKE GUI", color=(230, 200, 120))
            dpg.add_text("Behavior Analysis", color=(140, 140, 150))
            dpg.add_separator()
            dpg.add_spacer(height=8)
            for key, label, _col in _PAGES:
                dpg.add_button(label=label, width=NAV_W - 16, height=40,
                               tag=f"nav_{key}",
                               callback=lambda s, a, u: self.switch(u), user_data=key)
            dpg.add_spacer(height=16)
            dpg.add_separator()
            dpg.add_text("出力ルート:", color=(150, 150, 160))
            dpg.add_text(str(self.root / "runs"), wrap=NAV_W - 12, color=(120, 120, 130))

    def switch(self, key: str) -> None:
        self.current = key
        for k, _l, _c in _PAGES:
            tag = f"panel_{k}"
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, show=(k == key))

    # ── Labeling ページ ──────────────────────────────────────────────
    def _build_labeling(self, parent: str) -> None:
        with dpg.child_window(tag="panel_labeling", parent=parent,
                              width=-1, height=-1, border=False, show=True):
            dpg.add_text("Labeling — 動画アノテーション (範囲ラベリング対応)",
                         color=(120, 200, 140))
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text(
                "動画を読み込み、bbox・track ID・行動(action) をラベル付けします。\n"
                "「範囲ラベリング」では、個体を選び [開始, 終了] フレームに行動/ID を\n"
                "一括付与し、キーフレーム間の bbox を線形補間できます (全フレーム手描き不要)。",
                color=(200, 200, 210))
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="アノテータを起動", height=40, width=200,
                               callback=lambda: self._launch_annotator())
                dpg.add_button(label="動画を指定して起動", height=40, width=200,
                               callback=self._launch_annotator_with_video)
                dpg.add_button(label="既存の注釈JSONを開いて起動", height=40, width=240,
                               callback=self._launch_annotator_with_json)
            dpg.add_spacer(height=8)
            dpg.add_text("", tag="labeling_status", color=(180, 220, 180))
            dpg.add_spacer(height=12)
            dpg.add_text("ヒント (アノテータ内キーボード):", color=(160, 200, 240))
            dpg.add_text(
                "  → / ← フレーム移動   Space 前フレームから伝播   A 自動追跡(CSRT)\n"
                "  0–4 選択boxに行動   Delete 削除   Ctrl+S 保存\n"
                "  範囲ラベリングはサイドバー「Range Labeling」から。",
                color=(170, 170, 180))

    def _launch_annotator(self, extra_args: Optional[list] = None) -> None:
        cmd = [sys.executable, "-m", "htrtdetr.annotator"] + (extra_args or [])
        try:
            subprocess.Popen(cmd, cwd=str(self.root))
            dpg.set_value("labeling_status", "アノテータを別ウィンドウで起動しました。")
        except Exception as exc:  # noqa: BLE001
            dpg.set_value("labeling_status", f"[ERROR] 起動失敗: {exc}")

    def _launch_annotator_with_video(self) -> None:
        path = dialogs.pick_file("動画を選択", dialogs.VIDEO_TYPES)
        if path:
            self._launch_annotator([path])

    def _launch_annotator_with_json(self) -> None:
        path = dialogs.pick_file("注釈 JSON", dialogs.JSON_TYPES)
        if path:
            self._launch_annotator(["--load", path])

    # ═══════════════════════════════════════════════════════════════════
    def build_ui(self) -> None:
        """ハブのウィジェットツリーを構築する (コンテキスト作成後・描画ループ前)。"""
        with dpg.window(tag="hub_win", no_title_bar=True, no_move=True,
                        no_resize=True, no_scrollbar=True):
            with dpg.group(horizontal=True):
                self._build_nav()
                with dpg.child_window(tag="content", width=-1, height=-1, border=False):
                    self._build_labeling("content")
                    self.training.build("content")
                    self.analysis.build("content")

        dpg.set_primary_window("hub_win", True)
        self.switch("labeling")

    def run(self, argv=None) -> None:
        dpg.create_context()
        self._theme()
        dpg.create_viewport(title="YOAKE GUI — Training / Video Analysis / Labeling",
                            width=WINDOW_W, height=WINDOW_H, resizable=True)
        dpg.setup_dearpygui()
        self.build_ui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            try:
                if self.current == "analysis":
                    self.analysis.tick()
            except Exception:  # noqa: BLE001 — 再生の例外で GUI を落とさない
                pass
            dpg.render_dearpygui_frame()

        self.training.teardown()
        dpg.destroy_context()


def main(argv=None) -> None:
    Hub().run(argv)


if __name__ == "__main__":
    main()
