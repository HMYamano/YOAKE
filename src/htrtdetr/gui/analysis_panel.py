"""
analysis_panel.py - Video analysis GUI (Dear PyGui)
==================================================

Run inference, load predictions, inspect detections frame by frame, and review
track timelines in a single panel.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import dearpygui.dearpygui as dpg

from . import commands, dialogs, widgets
from .canvas import FrameCanvas, track_color
from .predictions import FrameDetections, build_track_timelines, frame_range, load_predictions
from .runner import CommandRunner

_DEFAULT_ACTIONS = "idle,walk,groom,interact,other"
_ACTION_COLORS = [
    (120, 120, 120, 255),
    (33, 150, 243, 255),
    (76, 175, 80, 255),
    (255, 152, 0, 255),
    (233, 30, 99, 255),
    (156, 39, 176, 255),
    (0, 188, 212, 255),
    (205, 220, 57, 255),
]
_MAX_TIMELINE_TRACKS = 10
_TL_W, _TL_H, _TL_ROW, _TL_LABEL_W = 1080, 210, 18, 72


def _action_color(action_id: int):
    if action_id is None or action_id < 0:
        return (60, 60, 70, 255)
    return _ACTION_COLORS[action_id % len(_ACTION_COLORS)]


class AnalysisPanel:
    """Video analysis page embedded inside the GUI hub."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.runner = CommandRunner("ana_log", "ana_status", cwd=self.root)
        self.canvas = FrameCanvas("ana_canvas", "ana_frame_tex", "ana_tex_reg")

        self.cap = None
        self.video_path: Optional[Path] = None
        self.num_frames = 0
        self.fps = 30.0
        self.cur_frame = 0
        self._frame_rgb: Optional[np.ndarray] = None

        self.preds: Dict[int, FrameDetections] = {}
        self.action_names: List[str] = _DEFAULT_ACTIONS.split(",")

        self.playing = False
        self._last_tick = 0.0

    def build(self, parent: str) -> None:
        with dpg.child_window(
            tag="panel_analysis",
            parent=parent,
            width=-1,
            height=-1,
            border=False,
            show=False,
        ):
            dpg.add_text("Video Analysis Workspace", color=(104, 214, 214))
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text(
                "Run inference on a video, load an existing predictions JSON file, and review "
                "detections with per-track action timelines.",
                wrap=980,
                color=(196, 202, 210),
            )
            dpg.add_spacer(height=10)

            with dpg.collapsing_header(label="1. Input Files", default_open=True):
                widgets.path_field(
                    "Input video", "ana_video",
                    field_key="video",
                    required=True,
                    browse=self._pick_video,
                    tooltip_text="MP4/AVI/MOV file to run detection and tracking on.",
                    width=470,
                )
                widgets.path_field(
                    "Checkpoint (.pth or .pt)", "ana_weights",
                    field_key="checkpoint",
                    required=True,
                    browse=self._pick_weights,
                    tooltip_text="Trained model weights, typically stage4 or stage2 checkpoint.",
                    width=470,
                )
                widgets.path_field(
                    "Output folder", "ana_outdir",
                    field_key="predict_output_dir",
                    default=str(self.root / "runs" / "predict"),
                    browse=self._pick_outdir,
                    tooltip_text=(
                        "Directory that receives predictions.json and overlay "
                        "videos. Also used by 'Load predictions.json From "
                        "Output Folder'."
                    ),
                    width=470,
                )
                dpg.add_text(
                    "Use the output folder both for new inference runs and for quickly loading "
                    "the latest predictions.json file.",
                    color=(170, 176, 184),
                    wrap=880,
                )

            with dpg.collapsing_header(label="2. Run or Load Predictions", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Score threshold:", indent=4)
                    dpg.add_input_float(tag="ana_score",
                                        default_value=0.3, format="%.2f",
                                        width=90, step=0)
                    widgets.tooltip(
                        "Detections with score below this value are discarded "
                        "during inference. Lower = more boxes.", wrap=280,
                    )
                    dpg.add_text("Window size:")
                    dpg.add_input_int(tag="ana_window",
                                      default_value=16, width=90)
                    widgets.tooltip(
                        "Number of frames processed together for temporal "
                        "action / ID heads.", wrap=260,
                    )
                    dpg.add_text("Max frames (0 = all):")
                    dpg.add_input_int(tag="ana_maxframes",
                                      default_value=0, width=100)
                    widgets.tooltip(
                        "Stop after this many frames. 0 processes the entire "
                        "video. Useful for a quick preview.", wrap=280,
                    )
                dpg.add_spacer(height=6)
                with dpg.group(horizontal=True):
                    widgets.primary_button(
                        "Run Inference", self._run_predict, height=34, width=170,
                    )
                    dpg.add_spacer(width=4)
                    dpg.add_button(label="Load Predictions JSON",
                                   height=32, callback=self._browse_predictions)
                    widgets.tooltip("Load predictions from an arbitrary predictions.json file.", wrap=260)
                    dpg.add_button(label="Load From Output Folder",
                                   height=32, callback=self._load_from_outdir)
                    widgets.tooltip(
                        "Load predictions.json from the Output folder set "
                        "above. Handy right after inference finishes.", wrap=280,
                    )
                    dpg.add_button(label="Stop", height=32,
                                   callback=self.runner.stop)
                    widgets.tooltip("Terminate the running inference process.", wrap=220)

            dpg.add_spacer(height=8)
            dpg.add_text("", tag="ana_summary", color=(184, 224, 186), wrap=1080)

            with dpg.collapsing_header(label="3. Review Playback", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="First", width=52, callback=lambda: self.goto(0))
                    dpg.add_button(label="Prev", width=52, callback=lambda: self.goto(self.cur_frame - 1))
                    dpg.add_button(label="Play", width=72, callback=self._toggle_play, tag="ana_play_btn")
                    dpg.add_button(label="Next", width=52, callback=lambda: self.goto(self.cur_frame + 1))
                    dpg.add_button(label="Last", width=52, callback=lambda: self.goto(self.num_frames - 1))
                    dpg.add_text("Speed:")
                    dpg.add_combo(["0.5", "1.0", "2.0", "4.0"], tag="ana_speed", default_value="1.0", width=74)
                    dpg.add_text("Show score >=")
                    dpg.add_slider_float(
                        tag="ana_disp_thr",
                        default_value=0.0,
                        min_value=0.0,
                        max_value=1.0,
                        width=180,
                        callback=lambda: self._render(),
                    )
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_text("Action names:", indent=4)
                    dpg.add_input_text(
                        tag="ana_actions",
                        default_value=_DEFAULT_ACTIONS,
                        width=360,
                        callback=self._on_actions_change,
                        on_enter=True,
                    )
                    dpg.add_button(label="Apply Action Names", callback=self._on_actions_change)
                dpg.add_slider_int(
                    tag="ana_slider",
                    default_value=0,
                    min_value=0,
                    max_value=0,
                    width=-1,
                    callback=lambda s, a: self.goto(a),
                )
                dpg.add_text("", tag="ana_frame_info", color=(170, 176, 184), wrap=1080)

            with dpg.child_window(tag="ana_canvas_win", width=-1, height=-290, no_scrollbar=True, border=True):
                dpg.add_drawlist(tag="ana_canvas", width=1100, height=560)

            dpg.add_text("Track action timeline (white line = current frame)", color=(166, 205, 241))
            with dpg.child_window(tag="ana_tl_win", width=-1, height=_TL_H + 18, border=True):
                dpg.add_drawlist(tag="ana_timeline", width=_TL_W, height=_TL_H)

            dpg.add_text("", tag="ana_status", color=(184, 224, 186), wrap=1080)
            with dpg.child_window(tag="ana_log_win", width=-1, height=120, border=True):
                dpg.add_text("", tag="ana_log", wrap=980)

        self._update_summary()

    def _field(self, label, tag, default="", browse=None, width=420):
        dpg.add_text(f"{label}:", indent=4)
        with dpg.group(horizontal=True):
            dpg.add_input_text(tag=tag, default_value=default, width=width)
            if browse is not None:
                dpg.add_button(label="Browse", width=70, callback=browse)

    def _set(self, tag, value):
        if value:
            dpg.set_value(tag, value)

    def _browse_video(self):
        path = dialogs.pick_file("Select video", dialogs.VIDEO_TYPES)
        if path:
            dpg.set_value("ana_video", path)
            self.load_video(Path(path))

    def _browse_weights(self):
        self._set("ana_weights", dialogs.pick_file("Select checkpoint", dialogs.CKPT_TYPES))

    def _pick_video(self) -> str:
        path = dialogs.pick_file("Select video", dialogs.VIDEO_TYPES)
        if path:
            self.load_video(Path(path))
        return path

    def _pick_weights(self) -> str:
        return dialogs.pick_file("Select checkpoint", dialogs.CKPT_TYPES)

    def _pick_outdir(self) -> str:
        return dialogs.pick_dir("Select output folder")

    def _browse_predictions(self):
        path = dialogs.pick_file("Select predictions JSON", dialogs.JSON_TYPES)
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
        self._update_frame_info()

    def _run_predict(self):
        source = dpg.get_value("ana_video")
        weights = dpg.get_value("ana_weights")
        outdir = dpg.get_value("ana_outdir")
        if not source or not weights:
            self.runner.log("[ERROR] An input video and checkpoint are required.")
            return
        max_frames = int(dpg.get_value("ana_maxframes")) or None
        cmd = commands.predict_command(
            source,
            weights,
            outdir,
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
            self.runner.log(f"[ERROR] Cannot read predictions: {path} ({exc})")
            return
        self.runner.log(f"Loaded predictions: {path} ({len(self.preds)} frames)")
        if self.cap is None:
            _, fmax = frame_range(self.preds)
            self.num_frames = max(self.num_frames, fmax + 1)
            if dpg.does_item_exist("ana_slider"):
                dpg.configure_item("ana_slider", max_value=max(self.num_frames - 1, 0))
        self._build_timeline()
        self._render()
        self._update_summary()

    def load_video(self, path: Path):
        import cv2

        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.runner.log(f"[ERROR] Cannot open video: {path}")
            return
        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if dpg.does_item_exist("ana_slider"):
            dpg.configure_item("ana_slider", max_value=max(self.num_frames - 1, 0))
        self.runner.log(f"Loaded video: {path.name} ({self.num_frames} frames, {self.fps:.1f} fps)")
        self.goto(0)
        self._update_summary()

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
        self._update_frame_info()

    def _toggle_play(self):
        self.playing = not self.playing
        if dpg.does_item_exist("ana_play_btn"):
            dpg.configure_item("ana_play_btn", label="Pause" if self.playing else "Play")
        self._last_tick = time.perf_counter()
        self._update_frame_info()

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
                dpg.configure_item("ana_play_btn", label="Play")
            self._update_frame_info()
            return
        self.goto(self.cur_frame + 1)

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
            out.append(
                {
                    "bbox": fd.boxes[i],
                    "track_id": fd.track_ids[i],
                    "action": action,
                    "score": fd.scores[i],
                }
            )
        return out

    def _render(self):
        if self._frame_rgb is not None:
            self.canvas.update_frame_data(self._frame_rgb)
        self.canvas.render(
            self._frame_rgb,
            self._current_detections(),
            placeholder="Load a video, then run inference or load a predictions JSON file.",
        )
        self._update_frame_info()

    def _update_summary(self):
        video_text = self.video_path.name if self.video_path is not None else "No video loaded"
        pred_frames = len(self.preds)
        pred_text = f"{pred_frames} prediction frames loaded" if pred_frames else "No predictions loaded"
        info = f"Video: {video_text} | Frames: {self.num_frames or 0} | FPS: {self.fps:.1f} | {pred_text}"
        if dpg.does_item_exist("ana_summary"):
            dpg.set_value("ana_summary", info)

    def _update_frame_info(self):
        if not dpg.does_item_exist("ana_frame_info"):
            return
        visible = len(self._current_detections())
        total = max(self.num_frames - 1, 0)
        state = "Playing" if self.playing else "Paused"
        thr = float(dpg.get_value("ana_disp_thr")) if dpg.does_item_exist("ana_disp_thr") else 0.0
        dpg.set_value(
            "ana_frame_info",
            f"Frame {self.cur_frame} / {total} | Visible detections: {visible} | Display threshold: {thr:.2f} | {state}",
        )

    def _build_timeline(self):
        if not dpg.does_item_exist("ana_timeline"):
            return
        dpg.delete_item("ana_timeline", children_only=True)
        if not self.preds:
            dpg.draw_text((16, 16), "No predictions loaded", size=14, color=(120, 120, 140, 255), parent="ana_timeline")
            return
        fmin, fmax = frame_range(self.preds)
        span = max(fmax - fmin, 1)
        timelines = build_track_timelines(self.preds)
        tracks = sorted(t for t in timelines if t >= 0)[:_MAX_TIMELINE_TRACKS]
        x0, x1 = _TL_LABEL_W, _TL_W - 18
        for row, tid in enumerate(tracks):
            y = 8 + row * _TL_ROW
            col = track_color(tid)
            dpg.draw_text((8, y + 1), f"ID {tid}", size=12, color=col, parent="ana_timeline")
            per_frame = timelines[tid]
            for fa, fb, aid in self._run_length(per_frame, fmin, fmax):
                xa = x0 + (fa - fmin) / span * (x1 - x0)
                xb = x0 + (fb - fmin) / span * (x1 - x0)
                acol = _action_color(aid)
                dpg.draw_rectangle(
                    (xa, y),
                    (max(xb, xa + 1), y + _TL_ROW - 4),
                    color=acol,
                    fill=acol,
                    parent="ana_timeline",
                )
        legend_y = 10 + len(tracks) * _TL_ROW + 8
        for i, name in enumerate(self.action_names[: len(_ACTION_COLORS)]):
            lx = _TL_LABEL_W + i * 122
            acol = _action_color(i)
            dpg.draw_rectangle((lx, legend_y), (lx + 12, legend_y + 12), color=acol, fill=acol, parent="ana_timeline")
            dpg.draw_text((lx + 16, legend_y), name, size=12, color=(204, 208, 214, 255), parent="ana_timeline")
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
        x0, x1 = _TL_LABEL_W, _TL_W - 18
        cx = x0 + (self.cur_frame - fmin) / span * (x1 - x0)
        dpg.draw_line(
            (cx, 0),
            (cx, _TL_H),
            color=(255, 255, 255, 220),
            thickness=1.5,
            parent="ana_timeline",
            tag="ana_tl_cursor",
        )
