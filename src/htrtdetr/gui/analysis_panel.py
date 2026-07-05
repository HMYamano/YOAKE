"""
analysis_panel.py — Video Analysis GUI (dearpygui)
==================================================

オフライン動画解析ビューア:
- ``yoake predict`` を subprocess 起動して推論、または既存 ``predictions.json`` を読込
- 元動画に bbox/ID/action を重畳して再生 (FrameCanvas)
- 個体 (track) 別の行動タイムライン + フレームジャンプ/スライダ + score しきい値

リアルタイム/Webカメラ/クローズドループは対象外 (オフライン専用)。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import dearpygui.dearpygui as dpg

from . import commands
from . import dialogs
from .canvas import FrameCanvas, track_color
from .predictions import FrameDetections, build_track_timelines, frame_range, load_predictions
from .runner import CommandRunner

_DEFAULT_ACTIONS = "idle,walk,groom,interact,other"
_ACTION_COLORS = [
    (120, 120, 120, 255), (33, 150, 243, 255), (76, 175, 80, 255),
    (255, 152, 0, 255), (233, 30, 99, 255), (156, 39, 176, 255),
    (0, 188, 212, 255), (205, 220, 57, 255),
]
_MAX_TIMELINE_TRACKS = 10
_TL_W, _TL_H, _TL_ROW, _TL_LABEL_W = 940, 190, 16, 64


def _action_color(action_id: int):
    if action_id is None or action_id < 0:
        return (60, 60, 70, 255)
    return _ACTION_COLORS[action_id % len(_ACTION_COLORS)]


class AnalysisPanel:
    """Video Analysis ページ。ハブから ``build(parent)`` で構築し ``tick()`` で再生する。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runner = CommandRunner("ana_log", "ana_status", cwd=self.root)
        self.canvas = FrameCanvas("ana_canvas", "ana_frame_tex", "ana_tex_reg")

        self.cap = None  # cv2.VideoCapture
        self.video_path: Optional[Path] = None
        self.num_frames = 0
        self.fps = 30.0
        self.cur_frame = 0
        self._frame_rgb: Optional[np.ndarray] = None

        self.preds: Dict[int, FrameDetections] = {}
        self.action_names: List[str] = _DEFAULT_ACTIONS.split(",")

        self.playing = False
        self._last_tick = 0.0

    # ═══════════════════════════════════════════════════════════════════
    # UI 構築
    # ═══════════════════════════════════════════════════════════════════
    def build(self, parent: str) -> None:
        with dpg.child_window(tag="panel_analysis", parent=parent,
                              width=-1, height=-1, border=False, show=False):
            dpg.add_text("Video Analysis — 動画解析", color=(80, 200, 200))
            dpg.add_separator()

            # ── 入力 / 推論 ─────────────────────────────────────────
            with dpg.collapsing_header(label="① 入力 と 推論", default_open=True):
                self._field("入力動画", "ana_video", browse=self._browse_video, width=440)
                self._field("チェックポイント (.pth)", "ana_weights",
                            browse=self._browse_weights, width=440)
                self._field("出力ディレクトリ", "ana_outdir",
                            default=str(self.root / "runs" / "predict"),
                            browse=lambda: self._set("ana_outdir", dialogs.pick_dir()))
                with dpg.group(horizontal=True):
                    dpg.add_text("score閾値:", indent=4)
                    dpg.add_input_float(tag="ana_score", default_value=0.3,
                                        format="%.2f", width=90, step=0)
                    dpg.add_text("  window:", )
                    dpg.add_input_int(tag="ana_window", default_value=16, width=90)
                    dpg.add_text("  max_frames(0=全):")
                    dpg.add_input_int(tag="ana_maxframes", default_value=0, width=90)
                with dpg.group(horizontal=True):
                    dpg.add_button(label="推論実行 (predict)", height=30,
                                   callback=self._run_predict)
                    dpg.add_button(label="予測JSONを読込", height=30,
                                   callback=self._browse_predictions)
                    dpg.add_button(label="出力から自動読込", height=30,
                                   callback=self._load_from_outdir)
                    dpg.add_button(label="■ 中断", height=30, callback=self.runner.stop)

            # ── 再生 ─────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_button(label="|◀", width=32, callback=lambda: self.goto(0))
                dpg.add_button(label="◀", width=30, callback=lambda: self.goto(self.cur_frame - 1))
                dpg.add_button(label="▶再生", width=64, callback=self._toggle_play, tag="ana_play_btn")
                dpg.add_button(label="▶", width=30, callback=lambda: self.goto(self.cur_frame + 1))
                dpg.add_button(label="▶|", width=32, callback=lambda: self.goto(self.num_frames - 1))
                dpg.add_text("速度:")
                dpg.add_combo(["0.5", "1.0", "2.0", "4.0"], tag="ana_speed",
                              default_value="1.0", width=70)
                dpg.add_text("表示score≥:")
                dpg.add_slider_float(tag="ana_disp_thr", default_value=0.0,
                                     min_value=0.0, max_value=1.0, width=140,
                                     callback=lambda: self._render())
                dpg.add_input_text(tag="ana_actions", default_value=_DEFAULT_ACTIONS,
                                   width=240, callback=self._on_actions_change, on_enter=True)
                dpg.add_button(label="行動名反映", callback=self._on_actions_change)
            dpg.add_slider_int(tag="ana_slider", default_value=0, min_value=0, max_value=0,
                               width=-1, callback=lambda s, a: self.goto(a))

            # ── キャンバス + タイムライン ──────────────────────────
            with dpg.child_window(tag="ana_canvas_win", width=-1, height=-260,
                                  no_scrollbar=True, border=True):
                dpg.add_drawlist(tag="ana_canvas", width=920, height=520)
            dpg.add_text("個体別 行動タイムライン (縦線=現在フレーム)", color=(160, 200, 240))
            with dpg.child_window(tag="ana_tl_win", width=-1, height=_TL_H + 14, border=True):
                dpg.add_drawlist(tag="ana_timeline", width=_TL_W, height=_TL_H)

            # ── ログ ─────────────────────────────────────────────────
            dpg.add_text("", tag="ana_status", color=(180, 220, 180))
            with dpg.child_window(tag="ana_log_win", width=-1, height=120, border=True):
                dpg.add_text("", tag="ana_log", wrap=900)

    def _field(self, label, tag, default="", browse=None, width=380):
        dpg.add_text(f"{label}:", indent=4)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag=tag, default_value=default, width=width)
            if browse is not None:
                dpg.add_button(label="参照", width=60, callback=browse)

    # ═══════════════════════════════════════════════════════════════════
    # ブラウズ / 読込
    # ═══════════════════════════════════════════════════════════════════
    def _set(self, tag, value):
        if value:
            dpg.set_value(tag, value)

    def _browse_video(self):
        path = dialogs.pick_file("動画を選択", dialogs.VIDEO_TYPES)
        if path:
            dpg.set_value("ana_video", path)
            self.load_video(Path(path))

    def _browse_weights(self):
        self._set("ana_weights", dialogs.pick_file("チェックポイント", dialogs.CKPT_TYPES))

    def _browse_predictions(self):
        path = dialogs.pick_file("predictions.json", dialogs.JSON_TYPES)
        if path:
            self._load_predictions(Path(path))

    def _load_from_outdir(self):
        outdir = dpg.get_value("ana_outdir")
        if outdir:
            self._load_predictions(Path(outdir) / "predictions.json")

    def _on_actions_change(self, *args):
        raw = dpg.get_value("ana_actions") or _DEFAULT_ACTIONS
        self.action_names = [s.strip() for s in raw.split(",") if s.strip()]
        self._render()
        self._build_timeline()

    # ═══════════════════════════════════════════════════════════════════
    # 推論実行
    # ═══════════════════════════════════════════════════════════════════
    def _run_predict(self):
        source = dpg.get_value("ana_video")
        weights = dpg.get_value("ana_weights")
        outdir = dpg.get_value("ana_outdir")
        if not source or not weights:
            self.runner.log("[ERROR] 入力動画とチェックポイントを指定してください。")
            return
        max_frames = int(dpg.get_value("ana_maxframes")) or None
        cmd = commands.predict_command(
            source, weights, outdir,
            score_threshold=float(dpg.get_value("ana_score")),
            window_size=int(dpg.get_value("ana_window")),
            max_frames=max_frames,
            visualize=True,
        )

        def _done(code: int):
            if code == 0:
                self._load_predictions(Path(outdir) / "predictions.json")

        self.runner.run(cmd, on_done=_done)

    def _load_predictions(self, path: Path):
        try:
            self.preds = load_predictions(str(path))
        except (OSError, ValueError) as exc:
            self.runner.log(f"[ERROR] 予測を読めません: {path} ({exc})")
            return
        self.runner.log(f"予測を読み込みました: {path}  ({len(self.preds)} フレーム)")
        # 動画未読込なら、予測から想定フレーム数を設定
        if self.cap is None:
            _, fmax = frame_range(self.preds)
            self.num_frames = max(self.num_frames, fmax + 1)
            if dpg.does_item_exist("ana_slider"):
                dpg.configure_item("ana_slider", max_value=max(self.num_frames - 1, 0))
        self._build_timeline()
        self._render()

    # ═══════════════════════════════════════════════════════════════════
    # 動画
    # ═══════════════════════════════════════════════════════════════════
    def load_video(self, path: Path):
        import cv2

        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.runner.log(f"[ERROR] 動画を開けません: {path}")
            return
        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if dpg.does_item_exist("ana_slider"):
            dpg.configure_item("ana_slider", max_value=max(self.num_frames - 1, 0))
        self.runner.log(f"動画を読み込みました: {path.name} "
                        f"({self.num_frames} フレーム, {self.fps:.1f} fps)")
        self.goto(0)

    def _read_frame(self, idx: int):
        import cv2

        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def goto(self, idx: int):
        if self.num_frames > 0:
            idx = max(0, min(idx, self.num_frames - 1))
        else:
            idx = max(0, idx)
        self.cur_frame = idx
        if self.cap is not None:
            self._frame_rgb = self._read_frame(idx)
        if dpg.does_item_exist("ana_slider"):
            dpg.set_value("ana_slider", idx)
        self._render()
        self._draw_cursor()

    # ═══════════════════════════════════════════════════════════════════
    # 再生ループ (ハブの描画ループから tick() が呼ばれる)
    # ═══════════════════════════════════════════════════════════════════
    def _toggle_play(self):
        self.playing = not self.playing
        if dpg.does_item_exist("ana_play_btn"):
            dpg.configure_item("ana_play_btn", label="⏸停止" if self.playing else "▶再生")
        self._last_tick = time.perf_counter()

    def tick(self):
        if not self.playing or self.cap is None or self.num_frames <= 0:
            return
        speed = float(dpg.get_value("ana_speed") or "1.0")
        now = time.perf_counter()
        if now - self._last_tick < 1.0 / max(self.fps * speed, 1e-3):
            return
        self._last_tick = now
        if self.cur_frame >= self.num_frames - 1:
            self.playing = False
            if dpg.does_item_exist("ana_play_btn"):
                dpg.configure_item("ana_play_btn", label="▶再生")
            return
        self.goto(self.cur_frame + 1)

    # ═══════════════════════════════════════════════════════════════════
    # 描画
    # ═══════════════════════════════════════════════════════════════════
    def _current_detections(self) -> List[dict]:
        fd = self.preds.get(self.cur_frame)
        if fd is None:
            return []
        thr = float(dpg.get_value("ana_disp_thr")) if dpg.does_item_exist("ana_disp_thr") else 0.0
        out = []
        for i in range(len(fd)):
            if fd.scores[i] < thr:
                continue
            aid = fd.action_ids[i]
            action = self.action_names[aid] if 0 <= aid < len(self.action_names) else ""
            out.append({
                "bbox": fd.boxes[i],
                "track_id": fd.track_ids[i],
                "action": action,
                "score": fd.scores[i],
            })
        return out

    def _render(self):
        if self._frame_rgb is not None:
            self.canvas.update_frame_data(self._frame_rgb)
        self.canvas.render(self._frame_rgb, self._current_detections(),
                           placeholder="動画を読み込み、推論または予測JSONを読み込んでください")

    # ── タイムライン ──────────────────────────────────────────────────
    def _build_timeline(self):
        if not dpg.does_item_exist("ana_timeline"):
            return
        dpg.delete_item("ana_timeline", children_only=True)
        if not self.preds:
            dpg.draw_text((16, 16), "予測なし", size=14, color=(120, 120, 140, 255),
                          parent="ana_timeline")
            return
        fmin, fmax = frame_range(self.preds)
        span = max(fmax - fmin, 1)
        timelines = build_track_timelines(self.preds)
        tracks = sorted(t for t in timelines if t >= 0)[:_MAX_TIMELINE_TRACKS]
        x0, x1 = _TL_LABEL_W, _TL_W - 12
        for row, tid in enumerate(tracks):
            y = 6 + row * _TL_ROW
            col = track_color(tid)
            dpg.draw_text((6, y + 1), f"ID{tid}", size=12, color=col, parent="ana_timeline")
            per_frame = timelines[tid]
            # run-length で同一 action の連続区間をまとめて矩形描画
            for (fa, fb, aid) in self._run_length(per_frame, fmin, fmax):
                xa = x0 + (fa - fmin) / span * (x1 - x0)
                xb = x0 + (fb - fmin) / span * (x1 - x0)
                acol = _action_color(aid)
                dpg.draw_rectangle((xa, y), (max(xb, xa + 1), y + _TL_ROW - 4),
                                   color=acol, fill=acol, parent="ana_timeline")
        # 凡例
        legy = 8 + len(tracks) * _TL_ROW + 4
        for i, name in enumerate(self.action_names[:len(_ACTION_COLORS)]):
            lx = _TL_LABEL_W + i * 120
            acol = _action_color(i)
            dpg.draw_rectangle((lx, legy), (lx + 12, legy + 12), color=acol, fill=acol,
                               parent="ana_timeline")
            dpg.draw_text((lx + 16, legy), name, size=12, color=(200, 200, 210, 255),
                          parent="ana_timeline")
        self._draw_cursor()

    @staticmethod
    def _run_length(per_frame: Dict[int, int], fmin: int, fmax: int):
        segments = []
        start = None
        prev_a = None
        prev_f = None
        for f in range(fmin, fmax + 1):
            a = per_frame.get(f)
            if a is None:
                if start is not None:
                    segments.append((start, prev_f, prev_a))
                    start = None
                prev_a = None
                continue
            if start is None:
                start, prev_a = f, a
            elif a != prev_a:
                segments.append((start, prev_f, prev_a))
                start, prev_a = f, a
            prev_f = f
        if start is not None:
            segments.append((start, prev_f, prev_a))
        return segments

    def _draw_cursor(self):
        if not dpg.does_item_exist("ana_timeline") or not self.preds:
            return
        if dpg.does_item_exist("ana_tl_cursor"):
            dpg.delete_item("ana_tl_cursor")
        fmin, fmax = frame_range(self.preds)
        span = max(fmax - fmin, 1)
        x0, x1 = _TL_LABEL_W, _TL_W - 12
        cx = x0 + (self.cur_frame - fmin) / span * (x1 - x0)
        dpg.draw_line((cx, 0), (cx, _TL_H), color=(255, 255, 255, 220), thickness=1.5,
                      parent="ana_timeline", tag="ana_tl_cursor")
