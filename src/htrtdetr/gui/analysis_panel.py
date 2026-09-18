"""
analysis_panel.py - Video analysis GUI (Dear PyGui)
==================================================

Step 3 of the YOAKE hub: run inference on a video, load a ``predictions.json``,
and review the result frame by frame.

The page is three regions, top to bottom:

1. **Source and predictions** - which video, which checkpoint, where the run
   writes, and the buttons that produce or load a ``predictions.json``.
2. **Viewer** - transport controls, the frame canvas (wheel zoom, drag pan,
   click to select a detection) and, beside it, the inspector / display tabs.
3. **Timeline** - one lane per track showing the action over time. Click it to
   jump to a frame.

Notes for anyone editing this file
----------------------------------
* Hover and ``rect_min`` are always read from the **drawlist**, never from the
  enclosing ``child_window``: a child window has no ``rect_min`` at all, and its
  ``is_item_hovered`` goes False the moment a mouse button goes down, which
  kills every drag. Both were measured on this machine.
* Nothing here may import torch. The capability probe uses
  :mod:`htrtdetr.gui.auto_label`, which uses ``importlib.util.find_spec`` only.
* Every string the user can see is English.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import dearpygui.dearpygui as dpg

from . import auto_label, commands, dialogs, recents, shortcuts, theme, widgets
from .canvas import DEFAULT_TOGGLES, FrameCanvas, draw_overlays_bgr, track_color
from .predictions import (
    FrameDetections,
    build_head_timelines,
    build_track_timelines,
    build_track_trajectories,
    frame_range,
    load_predictions,
)
from .runner import CommandRunner

# toggle キー → チェックボックスの dpg タグ。
# ``bbox`` は以前タグが無く DEFAULT_TOGGLES の True に固定されていたため、
# 「Individual OBB」を外しても軸平行の箱が必ず出てしまっていた。
_TOGGLE_TAGS: Dict[str, str] = {
    "obb": "ana_tg_obb",
    "bbox": "ana_tg_bbox",
    "context_box": "ana_tg_context",
    "trajectory": "ana_tg_traj",
    "track_id": "ana_tg_tid",
    "pose": "ana_tg_pose",
    "action": "ana_tg_action",
    "confidence": "ana_tg_conf",
    "explanation": "ana_tg_expl",
}

#: ラベル と 1 行説明。用語は docs/GUI_DESIGN.md のグロッサリにそろえる。
_TOGGLE_LABELS: Tuple[Tuple[str, str, str], ...] = (
    ("obb", "Rotated box (OBB)",
     "The tight oriented box around one individual, with a line showing its heading"),
    ("bbox", "Axis-aligned box",
     "The upright box. Used when a detection has no rotated box, or when rotated boxes are off"),
    ("context_box", "Behavior context box",
     "The wider box the action head looked at. Drawn dashed so it cannot be mistaken for a detection"),
    ("trajectory", "Trajectory",
     "The path each track took up to the current frame. Never shows the future"),
    ("track_id", "Track ID",
     "The T-number in the label chip"),
    ("pose", "Pose",
     "Keypoints, when the run produced them"),
    ("action", "Action",
     "The predicted behavior name in the label chip"),
    ("confidence", "Confidence",
     "The detector score. It is not a calibrated probability"),
    ("explanation", "Explanation",
     "The top evidence groups beside each box, in logit units"),
)

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
_TL_W, _TL_H, _TL_ROW, _TL_LABEL_W = 1080, 210, 20, 96
_TL_TOP = 30          # frame ruler の高さ
_TL_MIN_H, _TL_MAX_H = 130, 240

_SPEEDS = ("0.25", "0.5", "1.0", "2.0", "4.0")

#: このページの編集可能ウィジェット。``get_focused_item()`` は水平 group の中に
#: 入力がある場合 group の方を返す (実測) ので、focus だけでは typing を検出
#: できない。各タグの ``is_item_active`` を直接見るのが確実。
_EDITABLE_TAGS: Tuple[str, ...] = (
    "ana_video", "ana_weights", "ana_outdir",
    "ana_score", "ana_window", "ana_maxframes",
    "ana_speed", "ana_frame_input", "ana_slider",
    "ana_disp_thr", "ana_actions", "ana_classes",
)

#: ビューア行の下に確保する高さの初期値。タイムラインの実高さに合わせて
#: ``_apply_viewer_reserve`` が調整する。
_VIEWER_RESERVE = 300

#: The empty-state copy. Says what is missing and how to produce it.
_EMPTY_CANVAS = (
    "Load a video and a predictions.json to review detections frame by frame.\n"
    "Pick a video above, then either run inference or load an existing "
    "predictions.json."
)
_EMPTY_TIMELINE = (
    "No predictions loaded. Run inference above, or load a predictions.json.\n"
    "A predictions.json is written to the output folder by "
    "'yoake predict source=<video> weights=<checkpoint> output_dir=<folder>'."
)
_EMPTY_INSPECTOR = (
    "Click a box on the video to see its ID, action, confidence and the "
    "evidence behind the decision."
)


def _action_color(action_id: Optional[int]):
    if action_id is None or action_id < 0:
        return (60, 60, 70, 255)
    return _ACTION_COLORS[action_id % len(_ACTION_COLORS)]


def _keys_for(action: str, scope: str = "annotator") -> str:
    """共有キーマップから、ある action の全キー表記を ``A / Left`` の形で返す。"""
    labels = [b.label for b in shortcuts.for_action(action, scope)]
    return " / ".join(labels) if labels else ""


def _exists(tag) -> bool:
    try:
        return bool(tag) and dpg.does_item_exist(tag)
    except SystemError:
        return False


class AnalysisPanel:
    """Video analysis page embedded inside the GUI hub."""

    #: Playback bindings owned by this page. Nav and view actions are pulled
    #: from the shared keymap so the labels can never drift; only the
    #: play/pause key is local to this screen.
    _LOCAL_BINDINGS = (
        shortcuts.Binding(action="playback.toggle", key="Spacebar", mods=(),
                          scope="analysis", group="Navigation",
                          help="Play or pause the video"),
    )
    _SHARED_ACTIONS = ("nav.prev", "nav.next", "nav.first", "nav.last",
                       "view.fit", "view.zoom_in", "view.zoom_out")

    def __init__(self, root: Path, project: Optional[Path] = None) -> None:
        self.root = Path(root)
        #: The project folder predictions and overlays are written into.
        #: ``None`` falls back to the source checkout's ``runs/predict``.
        self.project = Path(project) if project else None
        self.runner = CommandRunner("ana_log", "ana_status", cwd=self.root)
        self.canvas = FrameCanvas("ana_canvas", "ana_frame_tex", "ana_tex_reg")

        self.cap = None
        self.video_path: Optional[Path] = None
        self.pred_path: Optional[Path] = None
        self.num_frames = 0
        self.fps = 30.0
        self.cur_frame = 0
        self._frame_rgb: Optional[np.ndarray] = None
        self._decoded_idx = -2          # 直前にデコードしたフレーム (逐次読みの判定用)

        self.preds: Dict[int, FrameDetections] = {}
        self.action_names: List[str] = _DEFAULT_ACTIONS.split(",")
        self.class_names: List[str] = []

        self.playing = False
        self._last_tick = 0.0

        # トラック同一性をこの解析で使うか。予測ファイルの中身とは独立した
        # 「この解析でどう読むか」の宣言で、既定は使う。
        self.use_track_ids: bool = True

        # 表示 / 選択状態
        self._selected_track: Optional[int] = None
        self._selected_index: Optional[int] = None
        self._hover_index: Optional[int] = None
        self._trajectories: Optional[Dict[int, List]] = None
        self._head_timelines: Optional[Dict[str, Dict[int, Dict[int, str]]]] = None
        self._timeline_rows: List[Tuple[int, float]] = []   # (frame_min..) 用メタ
        self._timeline_span: Tuple[int, int] = (0, 0)
        self._timeline_width = _TL_W
        self._truncated_tracks = 0

        # マウス操作
        self._drag_mode: Optional[str] = None
        self._pan_anchor: Tuple[float, float] = (0.0, 0.0)
        self._btn_down: Dict[int, bool] = {}

        # メインスレッドで処理するための遅延フラグ (runner のコールバックは別スレッド)
        self._pending_predictions: Optional[Path] = None
        self._reported_errors: set = set()
        self._tl_open = True
        self._reserve = _VIEWER_RESERVE
        self._cap_probe: Optional[auto_label.Capability] = None
        self._built = False

    # ── where this page writes ──────────────────────────────────────────
    def _default_outdir(self) -> str:
        """``<project>/analysis/predict`` when a project is open.

        Predictions are a *result of this study*, not of the source checkout.
        Writing them next to the code is how a laptop ends up with six
        predictions.json files and no way to tell which video each came from.
        """
        if self.project is None:
            return str(self.root / "runs" / "predict")
        from ..annotation.layout import ProjectLayout

        return str(ProjectLayout(self.project).predictions)

    def set_project(self, path: Optional[Path]) -> None:
        """Follow the hub's active project, without discarding a typed path."""
        previous = self._default_outdir()
        self.project = Path(path) if path else None
        if not dpg.does_item_exist("ana_outdir"):
            return
        existing = dpg.get_value("ana_outdir") or ""
        if not existing or existing == previous:
            dpg.set_value("ana_outdir", self._default_outdir())
            widgets.refresh_all_indicators(["ana_outdir"])

    # ══════════════════════════════════════════════════════════════════
    # build
    # ══════════════════════════════════════════════════════════════════
    def build(self, parent: str) -> None:
        with dpg.child_window(
            tag="panel_analysis",
            parent=parent,
            width=-1,
            height=-1,
            border=False,
            show=False,
        ):
            widgets.page_title(
                "3. Analyze",
                "Run inference on a video, load a predictions.json, and review "
                "detections and per-track action timelines frame by frame.",
            )

            self._build_source_region()
            widgets.gap(2)
            self._build_viewer_region()
            widgets.gap(2)
            self._build_timeline_region()
            self._build_log_region()

        self._register_handlers()
        self._built = True
        self._sync_canvas_size()
        self._apply_capability_gate()
        self._apply_track_gate()
        self._update_summary()
        self._update_inspector()
        self._update_zoom_label()
        self._build_timeline()
        self._render()

    # ── region 1: source and predictions ──────────────────────────────
    def _build_source_region(self) -> None:
        with dpg.collapsing_header(label="Source and predictions",
                                   tag="ana_hdr_source", default_open=True):
            widgets.path_field(
                "Input video", "ana_video",
                field_key="video",
                required=True,
                browse=self._pick_video,
                tooltip_text="MP4, AVI, MOV or MKV file to review or run detection on.",
                width=470,
            )
            widgets.path_field(
                "Checkpoint (.pth or .pt)", "ana_weights",
                field_key="checkpoint",
                required=True,
                browse=self._pick_weights,
                tooltip_text=(
                    "Trained weights, normally the stage 4 or stage 2 checkpoint "
                    "from runs/train/. Only needed to run inference - loading an "
                    "existing predictions.json does not use it."
                ),
                width=470,
            )
            widgets.path_field(
                "Output folder", "ana_outdir",
                field_key="predict_output_dir",
                default=self._default_outdir(),
                browse=self._pick_outdir,
                tooltip_text=(
                    "Receives predictions.json and the overlay video. "
                    "'Load from output folder' reads predictions.json from here."
                ),
                width=470,
            )
            widgets.gap(1)

            with dpg.group(horizontal=True):
                dpg.add_text("Score threshold:", color=theme.TEXT_DIM)
                dpg.add_input_float(tag="ana_score", default_value=0.3,
                                    format="%.2f", width=90, step=0.0)
                widgets.tooltip(
                    "Detections below this score are discarded during inference. "
                    "Lower keeps more boxes. (default 0.30)", wrap=300,
                )
                dpg.add_spacer(width=theme.SP_M)
                dpg.add_text("Window size:", color=theme.TEXT_DIM)
                dpg.add_input_int(tag="ana_window", default_value=16, width=90)
                widgets.tooltip(
                    "Frames processed together by the temporal action and ID "
                    "heads. (default 16)", wrap=280,
                )
                dpg.add_spacer(width=theme.SP_M)
                dpg.add_text("Max frames:", color=theme.TEXT_DIM)
                dpg.add_input_int(tag="ana_maxframes", default_value=0, width=100)
                widgets.tooltip(
                    "Stop after this many frames. 0 = all frames. Useful for a "
                    "quick preview before committing to a long run.", wrap=300,
                )
            widgets.gap(1)

            dpg.add_checkbox(
                label="Use track identity in this analysis",
                tag="ana_use_tracks", default_value=True,
                callback=lambda s, a: self.set_use_track_ids(a),
            )
            widgets.tooltip(
                "On: detections are read as individuals - track IDs in the "
                "label chip, trajectories, one timeline lane per animal, and "
                "the overlay video labels each box with its ID.\n"
                "Off: detections are read one frame at a time. No IDs are "
                "shown or drawn and the timeline is a single lane. Nothing is "
                "deleted from predictions.json - this is how the run is read, "
                "not what it contains.\n"
                "It switches itself off when a loaded predictions.json has no "
                "track IDs in it.", wrap=340,
            )
            widgets.gap(1)

            with dpg.group(horizontal=True):
                widgets.primary_button("Run inference", self._run_predict,
                                       height=34, width=170, tag="ana_run_btn")
                dpg.add_spacer(width=theme.SP_S)
                widgets.secondary_button("Load predictions.json...",
                                         self._browse_predictions,
                                         height=32, width=200)
                widgets.tooltip("Load predictions from any predictions.json file.",
                                wrap=260)
                dpg.add_spacer(width=theme.SP_S)
                widgets.secondary_button("Load from output folder",
                                         self._load_from_outdir,
                                         height=32, width=200)
                widgets.tooltip(
                    "Load predictions.json from the output folder above. "
                    "Handy right after a run finishes.", wrap=280,
                )
                dpg.add_spacer(width=theme.SP_S)
                widgets.secondary_button("Stop", self._stop_run, height=32, width=90)
                widgets.tooltip("Stop the running inference process. "
                                "Frames already written are kept.", wrap=260)
            widgets.gap(1)
            dpg.add_text("", tag="ana_summary", color=theme.ACCEPTED, wrap=1000)
            dpg.add_text("", tag="ana_status", color=theme.TEXT_DIM, wrap=1000)

    # ── region 2: viewer ──────────────────────────────────────────────
    def _build_viewer_region(self) -> None:
        self._build_transport()
        with dpg.group(horizontal=True):
            with dpg.child_window(tag="ana_canvas_win", width=-392,
                                  height=-_VIEWER_RESERVE,
                                  no_scrollbar=True, border=True):
                # サイズは _sync_canvas_size() が親に合わせて毎フレーム調整する。
                # drawlist に固定サイズを与えると、はみ出した部分は clip されて
                # is_item_hovered が False になり、そこのクリックが全て死ぬ。
                dpg.add_drawlist(tag="ana_canvas", width=640, height=360)
                with dpg.group(tag="ana_canvas_empty", show=True):
                    widgets.gap(3)
                    widgets.empty_state(_EMPTY_CANVAS,
                                        "Choose a video...", self._pick_and_load_video,
                                        wrap=520)
            with dpg.child_window(tag="ana_side_win", width=-1,
                                  height=-_VIEWER_RESERVE, border=True):
                with dpg.tab_bar(tag="ana_side_tabs"):
                    with dpg.tab(label="Inspector", tag="ana_tab_inspector"):
                        widgets.gap(1)
                        dpg.add_group(tag="ana_inspector")
                    with dpg.tab(label="Display", tag="ana_tab_display"):
                        self._build_display_tab()
                    with dpg.tab(label="Export", tag="ana_tab_export"):
                        self._build_export_tab()

    def _build_transport(self) -> None:
        with dpg.group(horizontal=True):
            # 幅は 0 (内容に合わせる)。固定幅にするとキー表記が入ったラベルが
            # 途中で切れて "|< (H" のような読めない表示になる。
            widgets.toolbar_button(
                "First", self._first, tag="ana_btn_first",
                help_text=f"Go to the first frame ({shortcuts.label_for('nav.first')})")
            widgets.toolbar_button(
                "Prev", self._prev, tag="ana_btn_prev",
                help_text=f"Previous frame ({_keys_for('nav.prev')})")
            widgets.toolbar_button(
                "Play", self._toggle_play, tag="ana_play_btn", width=96,
                shortcut="Space", active=False,
                help_text="Play or pause the video")
            widgets.toolbar_button(
                "Next", self._next, tag="ana_btn_next",
                help_text=f"Next frame ({_keys_for('nav.next')})")
            widgets.toolbar_button(
                "Last", self._last, tag="ana_btn_last",
                help_text=f"Go to the last frame ({shortcuts.label_for('nav.last')})")

            dpg.add_spacer(width=theme.SP_M)
            dpg.add_text("Speed:", color=theme.TEXT_DIM)
            dpg.add_combo(list(_SPEEDS), tag="ana_speed",
                          default_value="1.0", width=74)
            widgets.tooltip("Playback rate as a multiple of the video's own frame "
                            "rate. Decoding may not keep up above 2.0.", wrap=280)

            dpg.add_spacer(width=theme.SP_M)
            dpg.add_text("Frame:", color=theme.TEXT_DIM)
            dpg.add_input_int(tag="ana_frame_input", default_value=0, width=104,
                              step=0, on_enter=True, callback=self._jump_to_typed)
            widgets.tooltip("Type a frame number and press Enter to jump there.",
                            wrap=240)
            widgets.secondary_button("Go", self._jump_to_typed, height=26, width=44)

            dpg.add_spacer(width=theme.SP_M)
            widgets.toolbar_button(
                "Fit", self._fit_view, tag="ana_btn_fit", width=64,
                shortcut=shortcuts.label_for("view.fit"),
                help_text="Fit the frame to the window and clear the pan")
            # +/- はキー表記をラベルに足さない ("- (-)" は読めない)。
            widgets.toolbar_button(
                "-", lambda: self._zoom_step(-1), tag="ana_btn_zoom_out", width=32,
                help_text=f"Zoom out ({shortcuts.label_for('view.zoom_out')})")
            dpg.add_text("100%", tag="ana_zoom_label", color=theme.TEXT_DIM)
            widgets.toolbar_button(
                "+", lambda: self._zoom_step(1), tag="ana_btn_zoom_in", width=32,
                help_text=f"Zoom in ({shortcuts.label_for('view.zoom_in')})")

            dpg.add_spacer(width=theme.SP_M)
            widgets.status_pill("ana_play_pill", "Paused", "idle")

        dpg.add_slider_int(tag="ana_slider", default_value=0, min_value=0,
                           max_value=0, width=-1, format="frame %d",
                           clamped=True,
                           callback=lambda s, a: self.goto(int(a)))
        widgets.tooltip("Drag to scrub through the video. Decoding is sequential, "
                        "so a long jump takes a moment.", wrap=280)
        dpg.add_text("", tag="ana_frame_info", color=theme.TEXT_DIM, wrap=1080)
        dpg.add_text(
            "Wheel: zoom at the cursor / middle-drag or drag the background: pan / "
            "click a box: select it / right-click: clear the selection",
            color=theme.TEXT_FAINT, wrap=1080,
        )

    def _build_display_tab(self) -> None:
        widgets.gap(1)
        dpg.add_text("Overlays", color=theme.ACCENT)
        dpg.add_separator()
        for key, label, help_text in _TOGGLE_LABELS:
            dpg.add_checkbox(label=label, tag=_TOGGLE_TAGS[key],
                             default_value=DEFAULT_TOGGLES[key],
                             callback=lambda: self._render())
            widgets.tooltip(help_text, wrap=300)
        dpg.add_text(
            "Solid rotated box = one individual. Dashed wider box = the context "
            "the action head used. Display only - the predictions file is "
            "never modified.",
            color=theme.TEXT_DIM, wrap=330,
        )

        widgets.gap(2)
        dpg.add_text("Filters", color=theme.ACCENT)
        dpg.add_separator()
        widgets.labeled_slider(
            "Show score >=", "ana_disp_thr", default=0.0,
            min_value=0.0, max_value=1.0, width=180,
            help_text="Hides low-scoring boxes on screen only. Saved predictions "
                      "are unaffected. (default 0.00)",
            callback=lambda: self._render(),
        )
        dpg.add_checkbox(label="Show only the selected track",
                         tag="ana_only_selected", default_value=False,
                         callback=lambda: self._render())
        widgets.tooltip("Hides every other individual so one animal can be "
                        "followed through a crowded frame.", wrap=300)

        widgets.gap(2)
        dpg.add_text("Names", color=theme.ACCENT)
        dpg.add_separator()
        dpg.add_text("Action names (comma separated):", color=theme.TEXT_DIM)
        dpg.add_input_text(tag="ana_actions", default_value=_DEFAULT_ACTIONS,
                           width=-1, callback=self._on_names_change, on_enter=True)
        widgets.tooltip("Maps action_id 0, 1, 2... onto readable names. The "
                        "order must match the training classes file.", wrap=300)
        dpg.add_text("Class names (comma separated, optional):", color=theme.TEXT_DIM)
        dpg.add_input_text(tag="ana_classes", default_value="", width=-1,
                           hint="fly,worm", callback=self._on_names_change,
                           on_enter=True)
        widgets.secondary_button("Apply names", self._on_names_change,
                                 height=28, width=-1)

    def _build_export_tab(self) -> None:
        widgets.gap(1)
        dpg.add_text("Current frame", color=theme.ACCENT)
        dpg.add_separator()
        widgets.secondary_button("Save frame as PNG...", self._export_frame_png,
                                 height=30, width=-1, tag="ana_exp_png")
        widgets.tooltip("Writes the current frame at full resolution with the "
                        "overlays that are switched on right now.", wrap=300)
        widgets.gap(2)

        dpg.add_text("Visible predictions", color=theme.ACCENT)
        dpg.add_separator()
        dpg.add_text(
            "Exports every detection that passes the display threshold, using "
            "the action and class names above. The source predictions.json is "
            "not modified.",
            color=theme.TEXT_DIM, wrap=330,
        )
        widgets.gap(1)
        widgets.secondary_button("Export visible predictions (JSON)...",
                                 self._export_predictions_json,
                                 height=30, width=-1, tag="ana_exp_json")
        widgets.secondary_button("Export visible predictions (CSV)...",
                                 self._export_predictions_csv,
                                 height=30, width=-1, tag="ana_exp_csv")
        widgets.gap(2)

        dpg.add_text("Correct these predictions", color=theme.ACCENT)
        dpg.add_separator()
        dpg.add_text(
            "Opens the annotator on this video with the loaded predictions as "
            "candidates. Nothing is saved as a label until you accept it there.",
            color=theme.TEXT_DIM, wrap=330,
        )
        widgets.gap(1)
        widgets.secondary_button("Send predictions to annotator",
                                 self._send_to_annotator,
                                 height=30, width=-1, tag="ana_send_annot")

    # ── region 3: timeline and log ────────────────────────────────────
    def _build_timeline_region(self) -> None:
        # 折りたためるようにしてある: track が多いとレーンが増えてビューアを
        # 圧迫するので、箱を見たいときは畳んで場所を返せる。
        with dpg.collapsing_header(label="Action timeline",
                                   tag="ana_hdr_timeline", default_open=True):
            dpg.add_text("", tag="ana_tl_note", color=theme.TEXT_DIM, wrap=1080)
            with dpg.child_window(tag="ana_tl_win", width=-1, height=_TL_H + 22,
                                  border=True):
                dpg.add_drawlist(tag="ana_timeline", width=_TL_W, height=_TL_H)
                with dpg.group(tag="ana_tl_empty", show=True):
                    widgets.gap(1)
                    widgets.empty_state(_EMPTY_TIMELINE,
                                        "Load predictions.json...",
                                        self._browse_predictions, wrap=760)

    def _build_log_region(self) -> None:
        with dpg.collapsing_header(label="Run log", tag="ana_hdr_log",
                                   default_open=False):
            with dpg.child_window(tag="ana_log_win", width=-1, height=150,
                                  border=True):
                dpg.add_text("", tag="ana_log", wrap=980)
                theme.bind_mono(dpg.last_item())

    # ══════════════════════════════════════════════════════════════════
    # capability gate
    # ══════════════════════════════════════════════════════════════════
    def _capability(self) -> auto_label.Capability:
        if self._cap_probe is None:
            self._cap_probe = auto_label.probe(self.root)
        return self._cap_probe

    def _apply_capability_gate(self) -> None:
        """torch が無い環境では Run inference を locked にする (traceback を吐かない)。"""
        cap = self._capability()
        if not _exists("ana_run_btn"):
            return
        if not cap.package and cap.package_reason:
            widgets.locked("ana_run_btn", cap.package_reason)
        elif not cap.torch:
            widgets.locked("ana_run_btn", cap.torch_reason)
        else:
            widgets.unlock("ana_run_btn",
                           "Runs 'yoake predict' on the video above and loads the "
                           "resulting predictions.json when it finishes.")

    # ══════════════════════════════════════════════════════════════════
    # handlers
    # ══════════════════════════════════════════════════════════════════
    def _register_handlers(self) -> None:
        if _exists("ana_handlers"):
            return
        with dpg.handler_registry(tag="ana_handlers"):
            dpg.add_mouse_wheel_handler(callback=self._on_wheel)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
            for button in (dpg.mvMouseButton_Left, dpg.mvMouseButton_Middle):
                dpg.add_mouse_down_handler(button=button, callback=self._on_mouse_down)
                dpg.add_mouse_drag_handler(button=button, threshold=1.0,
                                           callback=self._on_mouse_drag)
                dpg.add_mouse_release_handler(button=button,
                                              callback=self._on_mouse_release)
            dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right,
                                        callback=self._on_right_click)
            for binding in self._bindings():
                try:
                    key = shortcuts.resolve(dpg, binding)
                except AttributeError:
                    continue
                dpg.add_key_press_handler(key=key, callback=self._wrap(binding))

    def _bindings(self) -> List[shortcuts.Binding]:
        """このページが listen する binding 一覧 (共有キーマップ + ローカル)。"""
        out: List[shortcuts.Binding] = []
        for action in self._SHARED_ACTIONS:
            out.extend(shortcuts.for_action(action, "annotator"))
        out.extend(self._LOCAL_BINDINGS)
        return out

    def shortcut_rows(self) -> List[Tuple[str, str]]:
        """``widgets.kbd_table`` に渡せる (key, description) 行。"""
        seen: set = set()
        rows: List[Tuple[str, str]] = []
        for b in self._bindings():
            if b.label in seen:
                continue
            seen.add(b.label)
            rows.append(b.row())
        return rows

    # ── guards ────────────────────────────────────────────────────────
    def _page_visible(self) -> bool:
        if not _exists("panel_analysis"):
            return False
        try:
            return bool(dpg.get_item_configuration("panel_analysis").get("show", False))
        except (KeyError, SystemError, TypeError):
            return False

    def _typing(self) -> bool:
        """入力ウィジェットに打ち込んでいる間はショートカットを殺す。

        判定は ``is_item_active``。``is_item_focused`` はウィンドウ内の最初の入力が
        自動で focus を取るため常に真になり、ガードとして役に立たない。

        さらに ``get_focused_item()`` だけでは足りない: 入力が水平 group の中にある
        と dpg は **group** の方を返す (実測。Frame 入力欄で確認)。focus 経由の判定に
        加えて、このページの編集可能ウィジェットを直接見る。
        """
        for tag in _EDITABLE_TAGS:
            if not _exists(tag):
                continue
            try:
                if dpg.is_item_active(tag):
                    return True
            except (KeyError, SystemError, TypeError):
                continue
        try:
            focused = dpg.get_focused_item()
        except SystemError:
            return False
        if not focused:
            return False
        try:
            kind = dpg.get_item_type(focused)
            active = dpg.is_item_active(focused)
        except (KeyError, SystemError, TypeError):
            return False
        return active and any(kind.endswith(k) for k in shortcuts.EDITING_TYPES)

    def _wrap(self, binding: shortcuts.Binding):
        action = self._actions().get(binding.action)

        def _cb(sender=None, app_data=None):
            if action is None or not self._page_visible() or self._typing():
                return
            if not shortcuts.mods_held(dpg, binding):
                return
            if not binding.mods and dpg.is_key_down(dpg.mvKey_ModCtrl):
                return          # Ctrl+1..4 (hub page switch) must not fire a bare key
            action()

        return _cb

    def _actions(self) -> Dict[str, object]:
        return {
            "nav.prev": self._prev,
            "nav.next": self._next,
            "nav.first": self._first,
            "nav.last": self._last,
            "view.fit": self._fit_view,
            "view.zoom_in": lambda: self._zoom_step(1),
            "view.zoom_out": lambda: self._zoom_step(-1),
            "playback.toggle": self._toggle_play,
        }

    # ── mouse ─────────────────────────────────────────────────────────
    def _on_wheel(self, sender, app_data) -> None:
        if not self._page_visible() or not self.canvas.is_hovered():
            return
        pos = self.canvas.mouse_canvas()
        if pos is None:
            return
        notches = 1.0 if float(app_data) > 0 else -1.0
        self.canvas.zoom_at(pos[0], pos[1], notches)
        self._update_zoom_label()
        self.canvas.refresh_overlay()

    def _on_mouse_move(self, sender, app_data) -> None:
        if not self._page_visible() or self._drag_mode is not None:
            return
        if self._frame_rgb is None or not self.canvas.is_hovered():
            if self._hover_index is not None:
                self._hover_index = None
                self.canvas.refresh_overlay(hovered_index=None)
            return
        idx = self.canvas.hit_test_mouse(self._visible_detections())
        if idx != self._hover_index:
            self._hover_index = idx
            self.canvas.refresh_overlay(hovered_index=idx)

    def _on_mouse_down(self, sender, app_data) -> None:
        button = int(app_data[0]) if isinstance(app_data, (list, tuple)) else int(app_data)
        if self._btn_down.get(button):
            return                       # down handler fires every frame while held
        self._btn_down[button] = True
        if not self._page_visible():
            return

        if button == dpg.mvMouseButton_Left and self._timeline_hovered():
            self._seek_from_timeline()
            return
        if not self.canvas.is_hovered():
            return

        if button == dpg.mvMouseButton_Middle:
            self._begin_pan()
            return

        # 左ボタン: 個体の上なら選択、背景なら pan
        idx = self.canvas.hit_test_mouse(self._visible_detections())
        if idx is None:
            self._begin_pan()
        else:
            self._select_index(idx)

    def _begin_pan(self) -> None:
        self._drag_mode = "pan"
        self._pan_anchor = self.canvas.pan

    def _on_mouse_drag(self, sender, app_data) -> None:
        if self._drag_mode != "pan" or not self._page_visible():
            return
        try:
            dx, dy = float(app_data[1]), float(app_data[2])
        except (IndexError, TypeError, ValueError):
            return
        self.canvas.set_pan(self._pan_anchor[0] + dx, self._pan_anchor[1] + dy)
        self.canvas.refresh_overlay()

    def _on_mouse_release(self, sender, app_data) -> None:
        button = int(app_data) if isinstance(app_data, (int, float)) else int(app_data[0])
        self._btn_down[button] = False
        self._drag_mode = None

    def _on_right_click(self, sender, app_data) -> None:
        if not self._page_visible() or not self.canvas.is_hovered():
            return
        if self._selected_track is None and self._selected_index is None:
            return
        self._selected_track = None
        self._selected_index = None
        self._render()
        widgets.status_text("ana_status", "Selection cleared.", "idle")

    # ══════════════════════════════════════════════════════════════════
    # file pickers
    # ══════════════════════════════════════════════════════════════════
    def _pick_video(self, initialdir: str = "") -> str:
        path = dialogs.pick_file("Select video", dialogs.VIDEO_TYPES)
        if path:
            self.load_video(Path(path))
        return path

    def _pick_and_load_video(self) -> None:
        path = self._pick_video()
        if path and _exists("ana_video"):
            dpg.set_value("ana_video", path)
            widgets.refresh_all_indicators(["ana_video"])

    def _pick_weights(self, initialdir: str = "") -> str:
        return dialogs.pick_file("Select checkpoint", dialogs.CKPT_TYPES)

    def _pick_outdir(self, initialdir: str = "") -> str:
        return dialogs.pick_dir("Select output folder")

    def _browse_video(self) -> None:
        self._pick_and_load_video()

    def _browse_weights(self) -> None:
        path = self._pick_weights()
        if path and _exists("ana_weights"):
            dpg.set_value("ana_weights", path)

    def _browse_predictions(self) -> None:
        path = dialogs.pick_file("Select predictions.json", dialogs.JSON_TYPES)
        if path:
            self._load_predictions(Path(path))

    def _load_from_outdir(self) -> None:
        outdir = dpg.get_value("ana_outdir") if _exists("ana_outdir") else ""
        if not outdir:
            widgets.status_text("ana_status",
                                "Set an output folder first.", "warn")
            return
        self._load_predictions(Path(outdir) / "predictions.json")

    # ══════════════════════════════════════════════════════════════════
    # names
    # ══════════════════════════════════════════════════════════════════
    def _on_names_change(self, *args) -> None:
        raw = (dpg.get_value("ana_actions") if _exists("ana_actions") else "") or _DEFAULT_ACTIONS
        self.action_names = [s.strip() for s in raw.split(",") if s.strip()]
        raw_cls = (dpg.get_value("ana_classes") if _exists("ana_classes") else "") or ""
        self.class_names = [s.strip() for s in raw_cls.split(",") if s.strip()]
        self._render()
        self._build_timeline()

    def _action_name(self, action_id: int) -> str:
        if 0 <= action_id < len(self.action_names):
            return self.action_names[action_id]
        return "" if action_id < 0 else f"action {action_id}"

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class {class_id}"

    # ══════════════════════════════════════════════════════════════════
    # run inference
    # ══════════════════════════════════════════════════════════════════
    def _run_predict(self) -> None:
        if widgets.is_locked("ana_run_btn"):
            widgets.status_text("ana_status",
                                widgets.lock_reason("ana_run_btn").splitlines()[0],
                                "warn")
            return
        source = dpg.get_value("ana_video")
        weights = dpg.get_value("ana_weights")
        outdir = dpg.get_value("ana_outdir")
        if not source or not weights:
            widgets.status_text("ana_status",
                                "An input video and a checkpoint are required.",
                                "error")
            self.runner.log("[ERROR] An input video and a checkpoint are required.")
            return
        if not Path(source).exists():
            widgets.status_text("ana_status", f"Video not found: {source}", "error")
            return
        if not Path(weights).exists():
            widgets.status_text("ana_status", f"Checkpoint not found: {weights}", "error")
            return
        if not outdir:
            outdir = self._default_outdir()
            dpg.set_value("ana_outdir", outdir)

        recents.add("video", source)
        recents.add("checkpoint", weights)
        recents.add("predict_output_dir", outdir)

        max_frames = int(dpg.get_value("ana_maxframes")) or None
        cmd = commands.predict_command(
            source, weights, outdir,
            score_threshold=float(dpg.get_value("ana_score")),
            window_size=int(dpg.get_value("ana_window")),
            max_frames=max_frames,
            visualize=True,
            track=self.use_track_ids,
        )
        target = Path(outdir) / "predictions.json"

        def _done(code: int) -> None:
            # runner はワーカースレッドから呼ぶ。dpg には触らず、tick() で拾う。
            if code == 0:
                self._pending_predictions = target

        widgets.status_pill("ana_play_pill", "Running", "running")
        widgets.status_text("ana_status", "Running inference...", "running")
        if _exists("ana_hdr_log"):
            dpg.set_value("ana_hdr_log", True)
        self.runner.run(cmd, on_done=_done)

    def set_use_track_ids(self, value: bool) -> None:
        """Read this analysis with or without individual identity."""
        self.use_track_ids = bool(value)
        if _exists("ana_use_tracks"):
            dpg.set_value("ana_use_tracks", self.use_track_ids)
        if not self.use_track_ids:
            self._selected_track = None
            self._selected_index = None
        self._apply_track_gate()
        self._build_timeline()
        self._render()
        self._update_inspector()

    def _apply_track_gate(self) -> None:
        """Lock, never hide, the overlays that need identity (P4)."""
        reason = (
            ""
            if self.use_track_ids
            else "This analysis is not using track identity. Tick 'Use track "
            "identity in this analysis' above to switch it back on."
        )
        for tag in ("ana_tg_tid", "ana_tg_traj", "ana_only_selected"):
            if not _exists(tag):
                continue
            if reason:
                dpg.set_value(tag, False)
                widgets.locked(tag, reason)
            else:
                widgets.unlock(tag)

    def _stop_run(self) -> None:
        if not self.runner.is_running():
            widgets.status_text("ana_status", "No inference is running.", "idle")
            return
        self.runner.stop()
        widgets.status_text("ana_status", "Inference stopped.", "warn")
        widgets.status_pill("ana_play_pill", "Stopped", "warn")

    # ══════════════════════════════════════════════════════════════════
    # predictions
    # ══════════════════════════════════════════════════════════════════
    def _load_predictions(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            msg = (f"No predictions.json at {path}. Run inference above, or point "
                   f"'Output folder' at a folder that has one.")
            widgets.status_text("ana_status", msg, "error")
            self.runner.log(f"[ERROR] {msg}")
            return
        try:
            self.preds = load_predictions(str(path))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            msg = f"Cannot read {path.name}: {exc}"
            widgets.status_text("ana_status", msg, "error")
            self.runner.log(f"[ERROR] {msg} ({path})")
            return

        self.pred_path = path
        self.runner.log(f"Loaded predictions: {path} ({len(self.preds)} frames)")
        widgets.status_text(
            "ana_status",
            f"Loaded {len(self.preds)} frames of predictions from {path.name}.",
            "ok",
        )
        # 派生キャッシュは予測が変わるたびに捨てる (古い軌跡が残らないように)
        self._trajectories = None
        self._head_timelines = None
        self._selected_track = None
        self._selected_index = None
        self._hover_index = None
        # A run with no ids in it cannot be read as individuals, whatever the
        # checkbox says, so the checkbox follows the file rather than offering
        # a view of the data that does not exist.
        if not self._predictions_have_tracks():
            if self.use_track_ids:
                self.runner.log(
                    "These predictions carry no track IDs. Track identity is "
                    "switched off for this analysis."
                )
            self.use_track_ids = False
            if _exists("ana_use_tracks"):
                dpg.set_value("ana_use_tracks", False)
        self._apply_track_gate()
        if self.cap is None:
            _, fmax = frame_range(self.preds)
            self.num_frames = max(self.num_frames, fmax + 1)
            self._sync_slider_range()
        if self.video_path is not None and _exists("ana_hdr_source"):
            # 両方そろったら入力欄を畳んでビューアに場所を譲る
            try:
                dpg.set_value("ana_hdr_source", False)
            except SystemError:
                pass
        self._build_timeline()
        self._render()
        self._update_summary()

    # ══════════════════════════════════════════════════════════════════
    # video
    # ══════════════════════════════════════════════════════════════════
    def load_video(self, path: Path) -> None:
        import cv2

        path = Path(path)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            msg = f"Cannot open the video: {path}. Check its path and format."
            widgets.status_text("ana_status", msg, "error")
            self.runner.log(f"[ERROR] {msg}")
            return
        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._decoded_idx = -2
        self._sync_slider_range()
        if _exists("ana_canvas_empty"):
            dpg.configure_item("ana_canvas_empty", show=False)
        if _exists("ana_canvas"):
            dpg.configure_item("ana_canvas", show=True)
        self.runner.log(
            f"Loaded video: {path.name} ({self.num_frames} frames, {self.fps:.1f} fps)")
        widgets.status_text(
            "ana_status",
            f"Loaded {path.name} - {self.num_frames} frames at {self.fps:.1f} fps.",
            "ok",
        )
        self._sync_canvas_size()
        self.canvas.fit()
        self.goto(0)
        self._update_summary()
        self._update_zoom_label()

    def _sync_slider_range(self) -> None:
        if _exists("ana_slider"):
            dpg.configure_item("ana_slider", max_value=max(self.num_frames - 1, 0))

    def _read_frame(self, idx: int):
        """フレームを読む。連番なら seek せず逐次 read する。

        デコード位置の管理は ``gui/video_source`` と共有する (annotator 側と同じ
        処理を 2 箇所に持つと片方だけ直って挙動が分かれるため)。``_decoded_idx``
        の -2 は「位置不明」を表す既存の番兵なので、helper の ``None`` に読み替える。
        """
        import cv2

        from . import video_source

        if self.cap is None:
            return None
        here = self._decoded_idx if self._decoded_idx >= 0 else None
        bgr, decoded = video_source.read_frame_sequential(self.cap, idx, here)
        self._decoded_idx = -2 if decoded is None else decoded
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def goto(self, idx: int) -> None:
        idx = int(idx)
        if self.num_frames > 0:
            idx = max(0, min(idx, self.num_frames - 1))
        else:
            idx = max(0, idx)
        self.cur_frame = idx
        if self.cap is not None:
            self._frame_rgb = self._read_frame(idx)
        if _exists("ana_slider"):
            dpg.set_value("ana_slider", idx)
        if _exists("ana_frame_input"):
            dpg.set_value("ana_frame_input", idx)
        self._hover_index = None
        self._selected_index = None
        self._render()
        self._draw_cursor()

    def _first(self) -> None:
        self.goto(0)

    def _last(self) -> None:
        self.goto(max(self.num_frames - 1, 0))

    def _prev(self) -> None:
        self.goto(self.cur_frame - 1)

    def _next(self) -> None:
        self.goto(self.cur_frame + 1)

    def _jump_to_typed(self, *args) -> None:
        if not _exists("ana_frame_input"):
            return
        self.goto(int(dpg.get_value("ana_frame_input")))

    def _toggle_play(self) -> None:
        if self.cap is None:
            widgets.status_text("ana_status", "Open a video first.", "warn")
            return
        self.playing = not self.playing
        if _exists("ana_play_btn"):
            dpg.configure_item("ana_play_btn",
                               label="Pause (Space)" if self.playing else "Play (Space)")
            widgets.set_toolbar_active("ana_play_btn", self.playing)
        widgets.status_pill("ana_play_pill",
                            "Playing" if self.playing else "Paused",
                            "running" if self.playing else "idle")
        self._last_tick = time.perf_counter()
        self._update_frame_info()

    # ── zoom ──────────────────────────────────────────────────────────
    def _fit_view(self) -> None:
        self.canvas.fit()
        self._update_zoom_label()
        self.canvas.refresh_overlay()

    def _zoom_step(self, notches: float) -> None:
        self.canvas.zoom_step(notches)
        self._update_zoom_label()
        self.canvas.refresh_overlay()

    def _update_zoom_label(self) -> None:
        if _exists("ana_zoom_label"):
            dpg.set_value("ana_zoom_label", f"{self.canvas.zoom_percent()}%")

    # ══════════════════════════════════════════════════════════════════
    # tick
    # ══════════════════════════════════════════════════════════════════
    def tick(self) -> None:
        widgets.tick_toasts()

        # ワーカースレッドが置いていった仕事をメインスレッドで拾う
        pending, self._pending_predictions = self._pending_predictions, None
        if pending is not None:
            widgets.status_pill("ana_play_pill", "Paused", "idle")
            self._load_predictions(pending)

        self._sync_timeline_width()
        open_now = self._timeline_open()
        if open_now != self._tl_open:
            self._tl_open = open_now
            self._apply_viewer_reserve()
        if self._sync_canvas_size():
            self._render()

        if not self.playing or self.cap is None or self.num_frames <= 0:
            return
        try:
            speed = float(dpg.get_value("ana_speed") or "1.0")
        except (TypeError, ValueError):
            speed = 1.0
        now = time.perf_counter()
        if now - self._last_tick < 1.0 / max(self.fps * speed, 1e-3):
            return
        self._last_tick = now
        if self.cur_frame >= self.num_frames - 1:
            self.playing = False
            if _exists("ana_play_btn"):
                dpg.configure_item("ana_play_btn", label="Play (Space)")
                widgets.set_toolbar_active("ana_play_btn", False)
            widgets.status_pill("ana_play_pill", "Paused", "idle")
            self._update_frame_info()
            return
        self.goto(self.cur_frame + 1)

    def teardown(self) -> None:
        """Release the video and stop any running inference. Called on GUI close."""
        self.playing = False
        try:
            if self.runner.is_running():
                self.runner.stop()
        except Exception:  # noqa: BLE001 - teardown must never raise
            pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:  # noqa: BLE001
                pass
            self.cap = None
        self._frame_rgb = None
        self._decoded_idx = -2

    # ══════════════════════════════════════════════════════════════════
    # detections for the current frame
    # ══════════════════════════════════════════════════════════════════
    def _current_detections(self) -> List[dict]:
        """現フレームの検出を canvas / inspector が読む dict のリストにする。"""
        fd = self.preds.get(self.cur_frame)
        if fd is None:
            return []
        out: List[dict] = []
        for i in range(len(fd)):
            aid = fd.action_ids[i]
            cid = fd.class_ids[i] if i < len(fd.class_ids) else 0
            entry = {
                "bbox": fd.boxes[i],
                "track_id": fd.track_ids[i],
                "action": self._action_name(aid),
                "action_id": aid,
                "class_id": cid,
                "class": self._class_name(cid),
                "score": fd.scores[i],
                "index": i,
            }
            obb = fd.obb_of(i)
            if obb is not None:
                entry["obb"] = obb
            if i < len(fd.context_boxes):
                entry["context_box"] = fd.context_boxes[i]
            ev = fd.evidence_of(i)
            if ev:
                entry["evidence"] = ev
            gates = fd.gates_of(i)
            if gates:
                entry["gates"] = gates
            heads = fd.heads_of(i)
            if heads:
                entry["heads"] = heads
            out.append(entry)
        return out

    def _display_threshold(self) -> float:
        if not _exists("ana_disp_thr"):
            return 0.0
        try:
            return float(dpg.get_value("ana_disp_thr"))
        except (TypeError, ValueError):
            return 0.0

    def _visible_detections(self) -> List[dict]:
        """表示フィルタを通した検出。hit_test と描画はこの同じリストを使う。"""
        thr = self._display_threshold()
        only_sel = bool(dpg.get_value("ana_only_selected")) if _exists("ana_only_selected") else False
        out = []
        for det in self._current_detections():
            if float(det["score"]) < thr:
                continue
            if only_sel and self._selected_track is not None \
                    and det["track_id"] != self._selected_track:
                continue
            out.append(det)
        return out

    def _toggles(self) -> Dict[str, bool]:
        """チェックボックスの現在値を FrameCanvas の toggle dict にまとめる。"""
        tg = dict(DEFAULT_TOGGLES)
        for key, tag in _TOGGLE_TAGS.items():
            if _exists(tag):
                tg[key] = bool(dpg.get_value(tag))
        return tg

    def _select_index(self, idx: Optional[int]) -> None:
        dets = self._visible_detections()
        if idx is None or not (0 <= idx < len(dets)):
            self._selected_index = None
            self._selected_track = None
        else:
            self._selected_index = idx
            self._selected_track = dets[idx]["track_id"]
        self._render()

    # ══════════════════════════════════════════════════════════════════
    # render
    # ══════════════════════════════════════════════════════════════════
    def _render(self) -> None:
        if not self._built:
            return
        if self._frame_rgb is not None:
            self.canvas.update_frame_data(self._frame_rgb)
        if _exists("ana_canvas_empty"):
            dpg.configure_item("ana_canvas_empty", show=self._frame_rgb is None)
        if _exists("ana_canvas"):
            dpg.configure_item("ana_canvas", show=self._frame_rgb is not None)

        tg = self._toggles()
        dets = self._visible_detections()
        traj = None
        if tg.get("trajectory"):
            if self._trajectories is None:
                self._trajectories = build_track_trajectories(self.preds)
            # 現フレームまでの軌跡だけを描く (未来を見せない)
            traj = {
                tid: [p for p in pts if p[0] <= self.cur_frame]
                for tid, pts in self._trajectories.items()
            }
        self.canvas.render(
            self._frame_rgb,
            dets,
            placeholder="",
            toggles=tg,
            trajectories=traj,
            selected_track=self._selected_track,
            selected_index=self._selected_index,
            hovered_index=self._hover_index,
        )
        self._update_frame_info()
        self._update_inspector()
        self._update_zoom_label()

    def _update_summary(self) -> None:
        video_text = self.video_path.name if self.video_path is not None else "no video"
        pred_text = (f"{len(self.preds)} prediction frames"
                     if self.preds else "no predictions")
        info = (f"Video: {video_text}  |  Frames: {self.num_frames or 0}  |  "
                f"FPS: {self.fps:.1f}  |  {pred_text}")
        if _exists("ana_summary"):
            dpg.set_value("ana_summary", info)

    def _update_frame_info(self) -> None:
        if not _exists("ana_frame_info"):
            return
        total = max(self.num_frames - 1, 0)
        shown = len(self._visible_detections())
        raw = len(self._current_detections())
        thr = self._display_threshold()
        state = "Playing" if self.playing else "Paused"
        hidden = raw - shown
        hidden_text = f" ({hidden} hidden by the filters)" if hidden > 0 else ""
        dpg.set_value(
            "ana_frame_info",
            f"Frame {self.cur_frame} / {total}  |  {shown} detections{hidden_text}"
            f"  |  Show score >= {thr:.2f}  |  {state}",
        )

    # ══════════════════════════════════════════════════════════════════
    # inspector
    # ══════════════════════════════════════════════════════════════════
    def _update_inspector(self) -> None:
        """選択中の個体の詳細と evidence を右パネルへ書き出す。"""
        if not _exists("ana_inspector"):
            return
        dpg.delete_item("ana_inspector", children_only=True)
        parent = "ana_inspector"

        if self._selected_track is None:
            dpg.push_container_stack(parent)
            widgets.empty_state(_EMPTY_INSPECTOR, wrap=330)
            dpg.pop_container_stack()
            return

        dets = self._visible_detections()
        det = None
        if self._selected_index is not None and 0 <= self._selected_index < len(dets):
            det = dets[self._selected_index]
        if det is None:
            det = next((d for d in dets if d["track_id"] == self._selected_track), None)

        if det is None:
            dpg.add_text(f"Track {self._selected_track} is not in this frame.",
                         color=theme.TEXT_DIM, wrap=330, parent=parent)
            self._inspector_track_nav(parent)
            return

        dpg.push_container_stack(parent)
        try:
            with dpg.group(horizontal=True):
                if self.use_track_ids:
                    widgets.track_badge(int(det["track_id"]))
                dpg.add_text(det.get("action") or "no action",
                             color=theme.TEXT)
                theme.bind_h2(dpg.last_item())
            widgets.gap(1)

            rows = []
            if self.use_track_ids:
                rows.append(
                    ("Track", f"T{int(det['track_id'])}"
                     if int(det["track_id"]) >= 0 else "unassigned")
                )
            rows += [
                ("Class", det.get("class", "")),
                ("Action", det.get("action") or "none"),
                ("Confidence", f"{float(det['score']):.2f}"),
            ]
            bbox = det.get("bbox")
            if bbox is not None:
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                rows.append(("Box (px)",
                             f"{x1:.0f}, {y1:.0f} to {x2:.0f}, {y2:.0f}"
                             f"  ({x2 - x1:.0f} x {y2 - y1:.0f})"))
            if det.get("obb"):
                cx, cy, w, h, th = (float(v) for v in det["obb"][:5])
                rows.append(("Rotated box",
                             f"{w:.0f} x {h:.0f} at {math.degrees(th):+.1f} deg"))
            self._kv_table(rows)

            dpg.add_text(
                "The detector's confidence is not a calibrated probability.",
                color=theme.TEXT_FAINT, wrap=330,
            )
            widgets.gap(1)
            self._inspector_track_nav(parent)

            if det.get("heads"):
                widgets.gap(1)
                dpg.add_text("Heads", color=theme.ACCENT)
                dpg.add_separator()
                self._kv_table([(str(h), str(c)) for h, c in det["heads"].items()])

            if det.get("gates"):
                widgets.gap(1)
                dpg.add_text("Gates", color=theme.ACCENT)
                dpg.add_separator()
                self._kv_table([(str(g), f"{float(v):.2f}")
                                for g, v in det["gates"].items()])

            ev = det.get("evidence") or {}
            widgets.gap(1)
            dpg.add_text("Evidence", color=theme.ACCENT)
            dpg.add_separator()
            if ev:
                total = sum(ev.values())
                span = max(abs(v) for v in ev.values()) or 1.0
                for group, val in sorted(ev.items(), key=lambda kv: -abs(kv[1])):
                    bar = "#" * max(1, int(abs(val) / span * 14))
                    colr = theme.PENDING if val >= 0 else theme.ACCENT
                    dpg.add_text(f"{group:<12s} {val:+6.2f}  {bar}", color=colr)
                    theme.bind_mono(dpg.last_item())
                dpg.add_text(f"{'total':<12s} {total:+6.2f}", color=theme.TEXT)
                theme.bind_mono(dpg.last_item())
                dpg.add_text(
                    "Contributions are in logit units and sum to the final score.",
                    color=theme.TEXT_FAINT, wrap=330,
                )
            else:
                dpg.add_text(
                    "This run recorded no evidence. Re-run inference with "
                    "explainability.enabled=true to see why each action was chosen.",
                    color=theme.TEXT_DIM, wrap=330,
                )
        finally:
            dpg.pop_container_stack()

    def _kv_table(self, rows: Sequence[Tuple[str, str]]) -> None:
        if not rows:
            return
        with dpg.table(header_row=False, borders_innerH=False,
                       borders_outerH=False, borders_innerV=False,
                       borders_outerV=False, width=-1,
                       policy=dpg.mvTable_SizingStretchProp):
            dpg.add_table_column(init_width_or_weight=0.38)
            dpg.add_table_column(init_width_or_weight=0.62)
            for key, value in rows:
                with dpg.table_row():
                    dpg.add_text(str(key), color=theme.TEXT_DIM)
                    dpg.add_text(str(value), color=theme.TEXT)

    def _predictions_have_tracks(self) -> bool:
        """True when any detection in the loaded run claims an identity."""
        return any(
            int(t) >= 0 for fd in self.preds.values() for t in fd.track_ids
        )

    def _track_frames(self, track_id: int) -> List[int]:
        return sorted(f for f, fd in self.preds.items()
                      if track_id in [int(t) for t in fd.track_ids])

    def _inspector_track_nav(self, parent: str) -> None:
        """選択 track が居るフレームへ前後に飛ぶ (タイムラインを目で追わずに済む)。"""
        if self._selected_track is None or not self.use_track_ids:
            return
        frames = self._track_frames(int(self._selected_track))
        if not frames:
            return
        prev_f = max((f for f in frames if f < self.cur_frame), default=None)
        next_f = min((f for f in frames if f > self.cur_frame), default=None)
        dpg.push_container_stack(parent)
        try:
            dpg.add_text(
                f"Track T{int(self._selected_track)} appears in {len(frames)} frames "
                f"({frames[0]}-{frames[-1]}).",
                color=theme.TEXT_DIM, wrap=330,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="< Previous appearance", height=26, width=160,
                               enabled=prev_f is not None,
                               callback=(lambda s, a, u, f=prev_f: self.goto(f))
                               if prev_f is not None else None)
                theme.bind(dpg.last_item(), theme.ghost_theme())
                dpg.add_button(label="Next appearance >", height=26, width=150,
                               enabled=next_f is not None,
                               callback=(lambda s, a, u, f=next_f: self.goto(f))
                               if next_f is not None else None)
                theme.bind(dpg.last_item(), theme.ghost_theme())
        finally:
            dpg.pop_container_stack()

    # ══════════════════════════════════════════════════════════════════
    # timeline
    # ══════════════════════════════════════════════════════════════════
    def _sync_canvas_size(self) -> bool:
        """drawlist を親の child_window に合わせる。合っていなければ True を返す。

        固定サイズの drawlist が親より大きいと、はみ出した領域は clip されて
        ``is_item_hovered`` が False になる。そこに描いた箱はクリックできない
        (実測: 1100x560 の drawlist が 966x223 の窓に入ると、縦 223px より下の
        検出は選択不能になる)。fit 計算も drawlist のサイズを読むので、
        ここがずれると画像が見えない領域にレターボックスされる。
        """
        if not _exists("ana_canvas_win") or not _exists("ana_canvas"):
            return False
        try:
            w, h = dpg.get_item_rect_size("ana_canvas_win")
            cfg = dpg.get_item_configuration("ana_canvas")
        except (KeyError, SystemError, TypeError):
            return False
        want_w = max(240, int(w) - 16)
        want_h = max(160, int(h) - 16)
        if abs(int(cfg.get("width", 0)) - want_w) <= 4                 and abs(int(cfg.get("height", 0)) - want_h) <= 4:
            return False
        dpg.configure_item("ana_canvas", width=want_w, height=want_h)
        return True

    def _timeline_open(self) -> bool:
        if not _exists("ana_hdr_timeline"):
            return True
        try:
            return bool(dpg.get_value("ana_hdr_timeline"))
        except (KeyError, SystemError, TypeError):
            return True

    def _apply_viewer_reserve(self) -> None:
        """タイムラインの実高さに合わせてビューア行の高さを取り直す。"""
        if not self._timeline_open():
            reserve = 118          # 畳んだヘッダ + ログヘッダ + 余白だけ
        else:
            reserve = _VIEWER_RESERVE
            if _exists("ana_tl_win"):
                try:
                    reserve = int(dpg.get_item_configuration("ana_tl_win")
                                  .get("height", _TL_H + 22)) + 118
                except (KeyError, SystemError, TypeError, ValueError):
                    reserve = _VIEWER_RESERVE
            reserve = max(200, min(400, reserve))
        self._reserve = reserve
        for tag in ("ana_canvas_win", "ana_side_win"):
            if _exists(tag):
                dpg.configure_item(tag, height=-reserve)

    def _sync_timeline_width(self) -> None:
        """ウィンドウ幅が変わったらタイムラインを引き伸ばす。"""
        if not _exists("ana_tl_win") or not self.preds:
            return
        try:
            w, _h = dpg.get_item_rect_size("ana_tl_win")
        except (KeyError, SystemError, TypeError):
            return
        target = max(560, int(w) - 20)
        if abs(target - self._timeline_width) > 8:
            self._timeline_width = target
            self._build_timeline()

    def _build_timeline(self) -> None:
        if not _exists("ana_timeline"):
            return
        dpg.delete_item("ana_timeline", children_only=True)

        if not self.preds:
            if _exists("ana_tl_empty"):
                dpg.configure_item("ana_tl_empty", show=True)
            if _exists("ana_timeline"):
                dpg.configure_item("ana_timeline", show=False)
            if _exists("ana_tl_note"):
                dpg.set_value("ana_tl_note", "")
            if _exists("ana_tl_win"):
                dpg.configure_item("ana_tl_win", height=_TL_MIN_H + 22)
            self._apply_viewer_reserve()
            return

        if _exists("ana_tl_empty"):
            dpg.configure_item("ana_tl_empty", show=False)
        dpg.configure_item("ana_timeline", show=True)

        fmin, fmax = frame_range(self.preds)
        span = max(fmax - fmin, 1)
        self._timeline_span = (fmin, fmax)
        width = self._timeline_width
        x0, x1 = _TL_LABEL_W, width - 18

        self._draw_time_ruler(fmin, fmax, span, x0, x1)

        if self._head_timelines is None:
            self._head_timelines = build_head_timelines(self.preds)

        if self._head_timelines:
            # multi-head: head ごとに別レーンで描く
            row = self._draw_multi_head_timeline(fmin, fmax, span, x0, x1)
        else:
            # single / hierarchical: track ごとに 1 レーン
            row = self._draw_single_timeline(fmin, fmax, span, x0, x1)

        legend_y = _TL_TOP + row * _TL_ROW + 10
        dpg.draw_text((8, legend_y), "Actions:", size=12,
                      color=theme.rgba(theme.TEXT_DIM), parent="ana_timeline")
        for i, name in enumerate(self.action_names[: len(_ACTION_COLORS)]):
            lx = _TL_LABEL_W + i * 122
            acol = _action_color(i)
            dpg.draw_rectangle((lx, legend_y), (lx + 12, legend_y + 12),
                               color=acol, fill=acol, rounding=2.0,
                               parent="ana_timeline")
            dpg.draw_text((lx + 17, legend_y), name, size=12,
                          color=theme.rgba(theme.TEXT), parent="ana_timeline")

        height = max(_TL_MIN_H, min(_TL_MAX_H, legend_y + 30))
        dpg.configure_item("ana_timeline", width=width, height=height)
        if _exists("ana_tl_win"):
            dpg.configure_item("ana_tl_win", height=height + 22)
        self._apply_viewer_reserve()
        self._update_timeline_note()
        self._draw_cursor()

    def _update_timeline_note(self) -> None:
        if not _exists("ana_tl_note"):
            return
        tail = ("Click the timeline to jump to a frame. Collapse this section to "
                "give the video more room.")
        if not self.use_track_ids:
            dpg.set_value(
                "ana_tl_note",
                "One lane: this analysis is not using track identity, so the "
                f"behavior is shown per frame rather than per animal. {tail}",
            )
            return
        if self._truncated_tracks > 0:
            dpg.set_value(
                "ana_tl_note",
                f"Showing {_MAX_TIMELINE_TRACKS} of {self._truncated_tracks} tracks - "
                f"sorted by frame count. {tail}",
            )
        else:
            dpg.set_value("ana_tl_note", tail)

    def _draw_time_ruler(self, fmin: int, fmax: int, span: int,
                         x0: float, x1: float) -> None:
        """上端にフレーム番号の目盛りを描く (どこを見ているかが判る)。"""
        dpg.draw_line((x0, _TL_TOP - 6), (x1, _TL_TOP - 6),
                      color=theme.rgba(theme.BORDER_SUBTLE), thickness=1.0,
                      parent="ana_timeline")
        ticks = 8
        for i in range(ticks + 1):
            frac = i / ticks
            fx = x0 + frac * (x1 - x0)
            frame = int(round(fmin + frac * span))
            dpg.draw_line((fx, _TL_TOP - 12), (fx, _TL_TOP - 6),
                          color=theme.rgba(theme.BORDER_SUBTLE), thickness=1.0,
                          parent="ana_timeline")
            dpg.draw_text((fx - 10, 4), str(frame), size=11,
                          color=theme.rgba(theme.TEXT_FAINT), parent="ana_timeline")

    def _lane_background(self, row: int, x0: float, x1: float) -> None:
        """行の帯。長いタイムラインで行が混ざらないようにする。"""
        y = _TL_TOP + row * _TL_ROW
        if row % 2 == 0:
            dpg.draw_rectangle((x0 - 2, y - 2), (x1 + 2, y + _TL_ROW - 4),
                               color=(0, 0, 0, 0),
                               fill=theme.rgba(theme.SURFACE_RAISE, 90),
                               parent="ana_timeline")

    def _draw_single_timeline(self, fmin, fmax, span, x0, x1) -> int:
        """track ごとに 1 レーンの行動タイムラインを描き、使った行数を返す。

        トラック同一性を使わない解析では、レーンは「どの個体か」ではなく
        「このフレームで何が起きているか」を表す 1 本になる。
        """
        timelines = build_track_timelines(self.preds)
        if not self.use_track_ids:
            return self._draw_untracked_timeline(timelines, fmin, fmax, span, x0, x1)
        ranked = sorted(((t, len(tl)) for t, tl in timelines.items() if t >= 0),
                        key=lambda kv: -kv[1])
        self._truncated_tracks = len(ranked) if len(ranked) > _MAX_TIMELINE_TRACKS else 0
        tracks = [t for t, _n in ranked[:_MAX_TIMELINE_TRACKS]]
        for row, tid in enumerate(tracks):
            self._lane_background(row, x0, x1)
            y = _TL_TOP + row * _TL_ROW
            col = track_color(tid)
            dpg.draw_rectangle((8, y), (14, y + _TL_ROW - 6), color=col, fill=col,
                               rounding=2.0, parent="ana_timeline")
            dpg.draw_text((19, y), f"T{tid}", size=12, color=col,
                          parent="ana_timeline")
            dpg.draw_text((19 + 34, y + 1), f"{len(timelines[tid])}f", size=11,
                          color=theme.rgba(theme.TEXT_FAINT), parent="ana_timeline")
            for fa, fb, aid in self._run_length(timelines[tid], fmin, fmax):
                xa = x0 + (fa - fmin) / span * (x1 - x0)
                xb = x0 + (fb - fmin) / span * (x1 - x0)
                acol = _action_color(aid)
                dpg.draw_rectangle(
                    (xa, y), (max(xb, xa + 1), y + _TL_ROW - 6),
                    color=acol, fill=acol, rounding=2.0, parent="ana_timeline",
                )
        return len(tracks)

    def _draw_untracked_timeline(self, timelines, fmin, fmax, span, x0, x1) -> int:
        """One lane: the behavior seen on each frame, identity set aside.

        Merging the per-track timelines is honest here in a way that per-track
        lanes are not: without ids, "lane 3" would name an individual the run
        never claimed to have followed.
        """
        self._truncated_tracks = 0
        merged: Dict[int, str] = {}
        for tl in timelines.values():
            for frame, action in tl.items():
                merged.setdefault(frame, action)
        self._lane_background(0, x0, x1)
        y = _TL_TOP
        dpg.draw_text((8, y), "All", size=12,
                      color=theme.rgba(theme.TEXT_DIM), parent="ana_timeline")
        dpg.draw_text((19 + 34, y + 1), f"{len(merged)}f", size=11,
                      color=theme.rgba(theme.TEXT_FAINT), parent="ana_timeline")
        for fa, fb, aid in self._run_length(merged, fmin, fmax):
            xa = x0 + (fa - fmin) / span * (x1 - x0)
            xb = x0 + (fb - fmin) / span * (x1 - x0)
            acol = _action_color(aid)
            dpg.draw_rectangle(
                (xa, y), (max(xb, xa + 1), y + _TL_ROW - 6),
                color=acol, fill=acol, rounding=2.0, parent="ana_timeline",
            )
        return 1

    def _draw_multi_head_timeline(self, fmin, fmax, span, x0, x1) -> int:
        """
        multi-head の場合、``locomotion`` / ``social`` / ``self_directed`` などを
        head ごとの別レーンとして描く。

        同じ個体が「歩きながら求愛する」ような同時成立を、レーンを縦に並べることで
        一目で読めるようにする (単一レーンでは表現できない)。
        """
        row = 0
        heads = sorted(self._head_timelines)
        self._truncated_tracks = 0
        # head 内のクラス名 → 色 (head をまたいで色が衝突しないよう head ごとに採番)
        for head in heads:
            per_track = self._head_timelines[head]
            ranked = sorted(((t, len(tl)) for t, tl in per_track.items() if t >= 0),
                            key=lambda kv: -kv[1])
            if len(ranked) > _MAX_TIMELINE_TRACKS:
                self._truncated_tracks = max(self._truncated_tracks, len(ranked))
            tracks = [t for t, _n in ranked[:_MAX_TIMELINE_TRACKS]]
            if not tracks:
                continue
            y_head = _TL_TOP + row * _TL_ROW
            dpg.draw_text((8, y_head), head[:12], size=12,
                          color=theme.rgba(theme.ACCENT), parent="ana_timeline")
            row += 1

            classes = sorted({c for tl in per_track.values() for c in tl.values()})
            cls_id = {c: i for i, c in enumerate(classes)}
            for tid in tracks:
                self._lane_background(row, x0, x1)
                y = _TL_TOP + row * _TL_ROW
                dpg.draw_text((19, y), f"T{tid}", size=11,
                              color=track_color(tid), parent="ana_timeline")
                numeric = {f: cls_id[c] for f, c in per_track[tid].items()}
                for fa, fb, aid in self._run_length(numeric, fmin, fmax):
                    xa = x0 + (fa - fmin) / span * (x1 - x0)
                    xb = x0 + (fb - fmin) / span * (x1 - x0)
                    acol = _action_color(aid)
                    dpg.draw_rectangle(
                        (xa, y), (max(xb, xa + 1), y + _TL_ROW - 6),
                        color=acol, fill=acol, rounding=2.0, parent="ana_timeline",
                    )
                row += 1
        return row

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

    def _draw_cursor(self) -> None:
        if not _exists("ana_timeline") or not self.preds:
            return
        if _exists("ana_tl_cursor"):
            dpg.delete_item("ana_tl_cursor")
        for tag in ("ana_tl_cursor_label", "ana_tl_cursor_text"):
            if _exists(tag):
                dpg.delete_item(tag)
        fmin, fmax = self._timeline_span
        span = max(fmax - fmin, 1)
        x0, x1 = _TL_LABEL_W, self._timeline_width - 18
        cx = x0 + (self.cur_frame - fmin) / span * (x1 - x0)
        cx = max(x0, min(x1, cx))
        try:
            height = dpg.get_item_configuration("ana_timeline").get("height", _TL_H)
        except (KeyError, SystemError, TypeError):
            height = _TL_H
        dpg.draw_line((cx, _TL_TOP - 12), (cx, height),
                      color=(255, 255, 255, 220), thickness=1.5,
                      parent="ana_timeline", tag="ana_tl_cursor")
        # ルーラの目盛り数字と重なるので、必ず暗い板を敷いてから書く
        lx = min(cx + 4, x1 - 42)
        dpg.draw_rectangle((lx - 2, 2), (lx + 34, 17), color=(0, 0, 0, 0),
                           fill=(12, 14, 18, 230), rounding=2.0,
                           parent="ana_timeline", tag="ana_tl_cursor_label")
        dpg.draw_text((lx + 2, 3), str(self.cur_frame), size=11,
                      color=(255, 255, 255, 240), parent="ana_timeline",
                      tag="ana_tl_cursor_text")

    def _timeline_hovered(self) -> bool:
        if not _exists("ana_timeline") or not self.preds:
            return False
        try:
            return bool(dpg.is_item_hovered("ana_timeline"))
        except (KeyError, SystemError, TypeError):
            return False

    def _seek_from_timeline(self) -> None:
        """タイムラインのクリック位置のフレームへ飛ぶ。

        原点は **drawlist** の ``rect_min`` から取る。child_window には ``rect_min``
        が存在せず ``KeyError`` になる。ここを裸の except で握り潰すと、外した
        クリックと壊れたクリックが区別できなくなる。
        """
        try:
            ox, _oy = dpg.get_item_rect_min("ana_timeline")
        except (KeyError, SystemError, TypeError, ValueError) as exc:
            self._report_once("timeline_origin",
                              f"Cannot locate the timeline on screen ({exc}).")
            return
        mx, _my = dpg.get_mouse_pos(local=False)
        x0, x1 = _TL_LABEL_W, self._timeline_width - 18
        rel = mx - ox
        if rel < x0 or rel > x1 or x1 <= x0:
            return
        fmin, fmax = self._timeline_span
        span = max(fmax - fmin, 1)
        frame = int(round(fmin + (rel - x0) / (x1 - x0) * span))
        self.goto(frame)

    def _report_once(self, key: str, message: str) -> None:
        """同じ種類の失敗を毎フレーム報告しない (ログが流れて意味を失う)。"""
        if key in self._reported_errors:
            return
        self._reported_errors.add(key)
        self.runner.log(f"[ERROR] {message}")
        widgets.status_text("ana_status", message, "error")

    # ══════════════════════════════════════════════════════════════════
    # export
    # ══════════════════════════════════════════════════════════════════
    def _export_frame_png(self) -> None:
        if self._frame_rgb is None:
            widgets.status_text("ana_status",
                                "Open a video first. There is no frame to save.",
                                "warn")
            return
        import cv2

        default = f"frame_{self.cur_frame:06d}.png"
        path = dialogs.save_file("Save frame as PNG", ".png",
                                 [("PNG", "*.png"), ("All", "*.*")])
        if not path:
            return
        if not Path(path).suffix:
            path = str(Path(path).with_suffix(".png"))
        bgr = cv2.cvtColor(self._frame_rgb, cv2.COLOR_RGB2BGR)
        try:
            composed = draw_overlays_bgr(
                bgr, self._visible_detections(), self._toggles(),
                trajectories=self._trajectory_prefix(),
                selected_track=self._selected_track,
            )
            ok = cv2.imwrite(path, composed)
        except (OSError, ValueError, TypeError) as exc:
            widgets.status_text("ana_status", f"Save error: {exc}", "error")
            return
        if not ok:
            widgets.status_text("ana_status",
                                f"Cannot write {path}. Check the folder and the "
                                f"file extension.", "error")
            return
        widgets.status_text("ana_status",
                            f"Saved frame {self.cur_frame} to {Path(path).name} "
                            f"({default} by default).", "ok")
        self.runner.log(f"Saved frame {self.cur_frame}: {path}")

    def _trajectory_prefix(self) -> Optional[Dict[int, List]]:
        if not self._toggles().get("trajectory"):
            return None
        if self._trajectories is None:
            self._trajectories = build_track_trajectories(self.preds)
        return {tid: [p for p in pts if p[0] <= self.cur_frame]
                for tid, pts in self._trajectories.items()}

    def _visible_everywhere(self) -> Dict[int, List[dict]]:
        """全フレームについて、表示閾値を通った検出を集める。"""
        thr = self._display_threshold()
        out: Dict[int, List[dict]] = {}
        for fi in sorted(self.preds):
            fd = self.preds[fi]
            rows = []
            for i in range(len(fd)):
                if float(fd.scores[i]) < thr:
                    continue
                aid = int(fd.action_ids[i])
                cid = int(fd.class_ids[i]) if i < len(fd.class_ids) else 0
                rows.append({
                    "bbox": [float(v) for v in fd.boxes[i]],
                    "score": float(fd.scores[i]),
                    "track_id": int(fd.track_ids[i]),
                    "action_id": aid,
                    "action": self._action_name(aid),
                    "class_id": cid,
                    "class": self._class_name(cid),
                    "obb": fd.obb_of(i),
                })
            if rows:
                out[fi] = rows
        return out

    def _export_predictions_json(self) -> None:
        if not self.preds:
            widgets.status_text("ana_status",
                                "No predictions loaded. Run inference or load a "
                                "predictions.json first.", "warn")
            return
        path = dialogs.save_file("Export visible predictions", ".json",
                                 dialogs.JSON_TYPES)
        if not path:
            return
        data = self._visible_everywhere()
        payload = {}
        for fi, rows in data.items():
            entry = {
                "boxes": [r["bbox"] for r in rows],
                "scores": [r["score"] for r in rows],
                "track_ids": [r["track_id"] for r in rows],
                "action_ids": [r["action_id"] for r in rows],
                "class_ids": [r["class_id"] for r in rows],
            }
            obbs = [r["obb"] for r in rows]
            if any(o is not None for o in obbs):
                entry["obb"] = [list(o) if o is not None else [] for o in obbs]
            payload[str(fi)] = entry
        try:
            Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            widgets.status_text("ana_status", f"Export error: {exc}", "error")
            return
        total = sum(len(v) for v in data.values())
        widgets.status_text(
            "ana_status",
            f"Exported {total} detections across {len(data)} frames to "
            f"{Path(path).name}. The source predictions file is unchanged.", "ok")
        self.runner.log(f"Exported predictions: {path}")

    def _export_predictions_csv(self) -> None:
        if not self.preds:
            widgets.status_text("ana_status",
                                "No predictions loaded. Run inference or load a "
                                "predictions.json first.", "warn")
            return
        path = dialogs.save_file("Export visible predictions", ".csv",
                                 [("CSV", "*.csv"), ("All", "*.*")])
        if not path:
            return
        data = self._visible_everywhere()
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow([
                    "frame", "track_id", "class_id", "class", "action_id",
                    "action", "score", "x1", "y1", "x2", "y2",
                    "obb_cx", "obb_cy", "obb_w", "obb_h", "obb_theta_deg",
                ])
                for fi, rows in data.items():
                    for r in rows:
                        x1, y1, x2, y2 = (r["bbox"] + [0, 0, 0, 0])[:4]
                        obb = r["obb"]
                        if obb:
                            oc = [f"{float(obb[0]):.2f}", f"{float(obb[1]):.2f}",
                                  f"{float(obb[2]):.2f}", f"{float(obb[3]):.2f}",
                                  f"{math.degrees(float(obb[4])):.2f}"]
                        else:
                            oc = ["", "", "", "", ""]
                        writer.writerow([
                            fi, r["track_id"], r["class_id"], r["class"],
                            r["action_id"], r["action"], f"{r['score']:.4f}",
                            f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                            *oc,
                        ])
        except (OSError, TypeError, ValueError) as exc:
            widgets.status_text("ana_status", f"Export error: {exc}", "error")
            return
        total = sum(len(v) for v in data.values())
        widgets.status_text(
            "ana_status",
            f"Exported {total} rows across {len(data)} frames to "
            f"{Path(path).name}.", "ok")
        self.runner.log(f"Exported predictions: {path}")

    # ══════════════════════════════════════════════════════════════════
    # hand-off to the annotator
    # ══════════════════════════════════════════════════════════════════
    def _send_to_annotator(self) -> None:
        """現在の predictions を候補としてアノテータで開く (レビューの往復を閉じる)。"""
        if self.pred_path is None or not Path(self.pred_path).exists():
            widgets.status_text(
                "ana_status",
                "Load a predictions.json first. The annotator imports it as "
                "candidates, which are never saved as labels until you accept them.",
                "warn")
            return
        video = dpg.get_value("ana_video") if _exists("ana_video") else ""
        if not video and self.video_path is not None:
            video = str(self.video_path)
        if not video:
            widgets.status_text("ana_status",
                                "Choose an input video first.", "warn")
            return

        cmd = [sys.executable, "-m", "htrtdetr.annotator",
               "--import-predictions", str(self.pred_path),
               "--video", str(video)]
        kwargs = {
            "cwd": str(self.root),
            "env": auto_label.child_env(self.root),
            "stdin": subprocess.DEVNULL,
        }
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flags:
            kwargs["creationflags"] = flags
        try:
            proc = subprocess.Popen(cmd, **kwargs)
        except (OSError, ValueError) as exc:
            msg = f"The annotator could not start: {exc}"
            widgets.status_text("ana_status", msg, "error")
            self.runner.log(f"[ERROR] {msg}")
            return
        widgets.status_text(
            "ana_status",
            f"Annotator starting (PID {proc.pid}) with {len(self.preds)} frames of "
            f"predictions as candidates. Nothing is saved until you accept it there.",
            "ok")
        self.runner.log("$ " + " ".join(cmd))
