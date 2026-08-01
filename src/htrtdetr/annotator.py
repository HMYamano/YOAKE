#!/usr/bin/env python3
"""
YOAKE Annotation Tool — Dear PyGui Edition
===========================================
GUI-based video annotation tool producing YOAKE JSON v1.1 datasets.

Usage:
    python tools/annotate.py [video_path] [--load annotations.json]

Keyboard shortcuts (when canvas is focused):
    → / N        Next frame
    ← / P        Previous frame
    Space        Propagate boxes from previous frame
    Delete       Delete selected box
    0–4          Set action for selected box
    T            Set track ID (input field)
    A            Toggle auto-track mode
    Ctrl+S       Save annotation
    Escape       Deselect
"""

from __future__ import annotations

import copy
import json
import sys
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    print("Error: dearpygui not installed.  Run: pip install dearpygui")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed.  Run: pip install opencv-python")
    sys.exit(1)

import numpy as np

try:  # 範囲ラベリング用の純粋関数 (dearpygui 非依存・単体テスト可)
    from .gui.labeling_ops import clamp_bbox, resolve_range_boxes, round_bbox
except ImportError:  # スクリプト直接実行時のフォールバック
    from htrtdetr.gui.labeling_ops import clamp_bbox, resolve_range_boxes, round_bbox

# ─── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_ACTIONS = ["stationary", "walking", "grooming", "courtship", "aggression"]
_DEFAULT_CLASSES = ["fly"]

_TRACK_PALETTE = [
    (230, 25, 75),   (60, 180, 75),   (255, 225, 25),  (67, 99, 216),
    (245, 130, 49),  (145, 30, 180),  (66, 212, 244),  (240, 50, 230),
    (191, 239, 69),  (250, 190, 212), (70, 153, 144),  (220, 190, 255),
    (154, 99, 36),   (255, 250, 200), (128, 0, 0),     (170, 255, 195),
    (128, 128, 0),   (255, 216, 177), (0, 0, 117),     (169, 169, 169),
]

_MIN_BOX_PX = 5
_HANDLE_R = 7       # resize-handle radius in canvas pixels
_FRAME_TEX = "yoake_frame_tex"
_SIDEBAR_W = 280

# dearpygui 2.x では mvKey_Control が廃止され mvKey_ModCtrl になった (1.x フォールバック付き)
_KEY_CTRL = getattr(dpg, "mvKey_ModCtrl", getattr(dpg, "mvKey_Control", 17))


def _track_color(track_id: int, alpha: int = 255) -> Tuple[int, int, int, int]:
    if track_id < 0:
        return (140, 140, 140, alpha)
    r, g, b = _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]
    return (r, g, b, alpha)


# ─── tkinter file-dialog helpers ─────────────────────────────────────────────

def _tk_open_file(title: str, filetypes: list) -> Optional[Path]:
    """Show a native OS open-file dialog (tkinter). Returns Path or None."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return Path(path) if path else None


def _tk_save_file(title: str, filetypes: list,
                  initial_file: str = "annotations.json") -> Optional[Path]:
    """Show a native OS save-file dialog (tkinter). Returns Path or None."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title=title, filetypes=filetypes,
        initialfile=initial_file, defaultextension=".json",
    )
    root.destroy()
    return Path(path) if path else None


def _tk_ask_dir(title: str) -> Optional[Path]:
    """Show a native OS directory picker (tkinter). Returns Path or None."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(path) if path else None


# ─── Data Model ───────────────────────────────────────────────────────────────

_gid = 0


class AnnObject:
    def __init__(self, bbox: List[int], track_id: int = -1, action_id: int = -1,
                 occluded: bool = False, is_crowd: bool = False,
                 keyframe: bool = True) -> None:
        global _gid
        _gid += 1
        self.object_id = _gid
        self.bbox = list(bbox)
        self.track_id = track_id
        self.action_id = action_id
        self.occluded = occluded
        self.is_crowd = is_crowd
        # keyframe: True = 手動で置いた box (補間のアンカー)、False = 補間/自動生成 box。
        # 編集用フラグで、保存 JSON (v1.1) には出力しない。
        self.keyframe = keyframe

    def to_dict(self) -> dict:
        return {
            "object_id": self.object_id,
            "bbox": [int(v) for v in self.bbox],
            "class_id": 0,
            "track_id": self.track_id,
            "action_id": self.action_id,
            "occluded": self.occluded,
            "is_crowd": self.is_crowd,
        }

    @staticmethod
    def from_dict(d: dict) -> "AnnObject":
        global _gid
        obj = AnnObject(
            bbox=d["bbox"],
            track_id=d.get("track_id", -1),
            action_id=d.get("action_id", -1),
            occluded=d.get("occluded", False),
            is_crowd=d.get("is_crowd", False),
            keyframe=d.get("keyframe", True),  # 読み込んだ box はキーフレーム扱い
        )
        if "object_id" in d:
            obj.object_id = d["object_id"]
            _gid = max(_gid, d["object_id"])
        return obj


# ─── Application ──────────────────────────────────────────────────────────────

class AnnotationApp:

    def __init__(self) -> None:
        # ── project config ───────────────────────────────────────────
        self.class_names: List[str] = []
        self.action_names: List[str] = []
        self.project_name: str = "untitled"

        # ── video state ──────────────────────────────────────────────
        self.video_path: Optional[Path] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.num_frames: int = 0
        self.fps: float = 25.0
        self.frame_w: int = 640
        self.frame_h: int = 640
        self.current_frame_idx: int = 0
        self._current_bgr: Optional[np.ndarray] = None

        # ── annotation state ─────────────────────────────────────────
        self.annotations: Dict[int, List[AnnObject]] = {}
        self.next_track_id: int = 1
        self.active_track_id: int = 1

        # ── range labeling state ─────────────────────────────────────
        self.range_start: int = 0
        self.range_end: int = 0

        # ── canvas interaction state ──────────────────────────────────
        self._mode: str = "idle"        # idle | draw | move | resize
        self._draw_start: Optional[Tuple[float, float]] = None
        self._draw_end: Optional[Tuple[float, float]] = None
        self._selected: Optional[AnnObject] = None
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_bbox_orig: Optional[List[int]] = None
        self._resize_handle: Optional[str] = None
        self._right_click_obj: Optional[AnnObject] = None

        # ── display geometry ──────────────────────────────────────────
        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._canvas_w: int = 800
        self._canvas_h: int = 600

        # ── auto-tracking ────────────────────────────────────────────
        self._auto_track: bool = False
        self._trackers: Dict[int, cv2.Tracker] = {}  # track_id -> CSRT tracker

        # ── annotation save path ─────────────────────────────────────
        self.annotation_path: Optional[Path] = None
        # プロジェクト保存ディレクトリ。設定されていれば Ctrl+S はここへ
        # annotations.json を静かに書き込む (ダイアログを開かない)。
        self.project_dir: Optional[Path] = None

        dpg.create_context()
        self._setup_theme()

    # ═══════════════════════════════════════════════════════════════════
    # Theme
    # ═══════════════════════════════════════════════════════════════════

    def _setup_theme(self) -> None:
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       (18,  18,  30,  255))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        (24,  24,  42,  255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (40,  40,  68,  255))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55,  55,  90,  255))
                dpg.add_theme_color(dpg.mvThemeCol_Button,         (40,  80, 120,  255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (55, 105, 155,  255))
                dpg.add_theme_color(dpg.mvThemeCol_Header,         (40,  80, 120,  200))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (55, 105, 155,  200))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,        (15,  15,  26,  255))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,  (20,  20,  40,  255))
                dpg.add_theme_color(dpg.mvThemeCol_Tab,            (30,  30,  55,  255))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive,      (50,  80, 130,  255))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,      (100, 180, 255, 255))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     (100, 160, 230, 255))
                dpg.add_theme_color(dpg.mvThemeCol_Text,           (210, 210, 220, 255))
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  4)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   4)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,    4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     6, 5)
        dpg.bind_theme(global_theme)

    # ═══════════════════════════════════════════════════════════════════
    # Project Setup Screen
    # ═══════════════════════════════════════════════════════════════════

    def _show_setup_screen(self) -> None:
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        w, h = 520, 560

        if dpg.does_item_exist("setup_win"):
            dpg.delete_item("setup_win")

        with dpg.window(label="YOAKE - Project Setup", tag="setup_win",
                        width=w, height=h, no_resize=False, no_close=True,
                        pos=[(vw - w) // 2, (vh - h) // 2]):

            dpg.add_text("New Project", color=(230, 25, 75, 255))
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text(
                "Create a project, define labels, and choose an optional folder for autosave.",
                color=(170, 170, 180, 255),
                wrap=w - 40,
            )
            dpg.add_spacer(height=8)

            dpg.add_text("Project name")
            dpg.add_input_text(tag="setup_proj_name", default_value="my_project",
                               width=-1)
            dpg.add_spacer(height=8)

            dpg.add_text("Class labels (comma-separated, for example: fly,worm)")
            dpg.add_input_text(tag="setup_classes",
                               default_value=", ".join(_DEFAULT_CLASSES), width=-1)
            dpg.add_spacer(height=8)

            dpg.add_text("Action labels (comma-separated, in display order)")
            dpg.add_input_text(tag="setup_actions",
                               default_value=", ".join(_DEFAULT_ACTIONS),
                               width=-1, multiline=False)
            dpg.add_spacer(height=8)

            dpg.add_text("Project folder (optional - Save writes annotations.json here)")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="setup_proj_dir", default_value="", width=-80)
                dpg.add_button(label="Browse", width=72,
                               callback=self._on_setup_pick_proj_dir)
            dpg.add_spacer(height=12)
            dpg.add_separator()
            dpg.add_spacer(height=8)

            dpg.add_button(label="Create Project",
                           callback=self._on_new_project_confirmed,
                           width=-1, height=36)

            dpg.add_spacer(height=12)
            dpg.add_text("Or continue from an existing annotation file",
                         color=(140, 140, 150, 255))
            dpg.add_spacer(height=6)
            dpg.add_button(label="Load Existing Annotation JSON...",
                           callback=self._on_load_project_clicked,
                           width=-1, height=36)

    def _on_setup_pick_proj_dir(self) -> None:
        p = _tk_ask_dir(title="Select Project Directory")
        if p is not None and dpg.does_item_exist("setup_proj_dir"):
            dpg.set_value("setup_proj_dir", str(p))

    def _on_new_project_confirmed(self) -> None:
        self.project_name = dpg.get_value("setup_proj_name").strip() or "untitled"
        raw_cls = dpg.get_value("setup_classes")
        raw_act = dpg.get_value("setup_actions")
        self.class_names = [s.strip() for s in raw_cls.split(",") if s.strip()]
        self.action_names = [s.strip() for s in raw_act.split(",") if s.strip()]
        if not self.class_names:
            self.class_names = ["fly"]
        if not self.action_names:
            self.action_names = list(_DEFAULT_ACTIONS)
        raw_dir = dpg.get_value("setup_proj_dir").strip()
        self.project_dir = Path(raw_dir) if raw_dir else None
        dpg.delete_item("setup_win")
        self._build_main_window()
        self._update_action_buttons()
        self._refresh_project_dir_display()

    def _on_load_project_clicked(self) -> None:
        path = _tk_open_file(
            title="Load Annotation JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is None:
            return
        dpg.delete_item("setup_win")
        self._build_main_window()
        self._load_annotation_file(path)

    # ═══════════════════════════════════════════════════════════════════
    # Main Window
    # ═══════════════════════════════════════════════════════════════════

    def _build_main_window(self) -> None:
        if dpg.does_item_exist("main_win"):
            dpg.delete_item("main_win")

        with dpg.window(tag="main_win", label="YOAKE Annotation Tool",
                        no_title_bar=True, no_move=True, no_resize=True,
                        no_scrollbar=True):
            self._build_menubar()

            # ── horizontal split: canvas | sidebar ──────────────────
            with dpg.group(horizontal=True):
                self._build_canvas_panel()
                self._build_sidebar()

            # ── navigation bar ────────────────────────────────────────
            self._build_navbar()

        dpg.set_primary_window("main_win", True)
        self._build_key_handlers()
        self._build_mouse_handlers()

        # texture registry (created once)
        if not dpg.does_item_exist("tex_reg"):
            with dpg.texture_registry(tag="tex_reg", show=False):
                pass

    # ─── Menu bar ────────────────────────────────────────────────────

    def _build_menubar(self) -> None:
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Open Video...  Ctrl+O",
                                  callback=self._open_video_dialog)
                dpg.add_menu_item(label="Load Annotation...",
                                  callback=self._load_annotation_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="Set Project Folder...",
                                  callback=self._set_project_dir_dialog)
                dpg.add_menu_item(label="Save  Ctrl+S",
                                  callback=self.save_annotation)
                dpg.add_menu_item(label="Save As...",
                                  callback=self._save_as_dialog)
                dpg.add_menu_item(label="Save and Export Frames...",
                                  callback=self._save_and_export_dialog)
                dpg.add_separator()
                dpg.add_menu_item(label="New Project...",
                                  callback=self._show_setup_screen)

            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Delete Selected  Del",
                                  callback=self.delete_selected)
                dpg.add_menu_item(label="Propagate from Prev  Space",
                                  callback=self.propagate_from_prev)

            with dpg.menu(label="Labels"):
                dpg.add_menu_item(label="Add Class Label...",
                                  callback=self._add_class_label_dialog)
                dpg.add_menu_item(label="Add Action Label...",
                                  callback=self._add_action_label_dialog)

            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Keyboard Shortcuts",
                                  callback=self._show_help)

    # ─── Canvas panel ────────────────────────────────────────────────

    def _build_canvas_panel(self) -> None:
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        cw = max(vw - _SIDEBAR_W - 20, 400)
        ch = max(vh - 110, 300)

        with dpg.child_window(tag="canvas_win", width=cw, height=ch,
                              no_scrollbar=True, border=False):
            dpg.add_drawlist(tag="canvas", width=cw, height=ch)

    # ─── Sidebar ─────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        vh = dpg.get_viewport_height()
        sh = max(vh - 110, 300)

        with dpg.child_window(tag="sidebar_win", width=_SIDEBAR_W, height=sh,
                              no_scrollbar=False, border=True):
            # Title
            dpg.add_text("YOAKE Annotator", color=(230, 25, 75, 255))
            dpg.add_text("Draw, track, and label behavior", color=(150, 150, 160, 255))
            dpg.add_separator()
            dpg.add_spacer(height=4)

            with dpg.collapsing_header(label="Quick Start", default_open=True):
                dpg.add_text(
                    "1. Open a video.\n"
                    "2. Drag on the canvas to create a box.\n"
                    "3. Assign a track and action.\n"
                    "4. Use Range Labeling to update multiple frames faster.",
                    color=(170, 170, 180, 255),
                    wrap=_SIDEBAR_W - 16,
                )
                dpg.add_spacer(height=4)
                dpg.add_text(
                    "Tip: click a box to edit it, or use the object list below to jump to a selection.",
                    color=(140, 140, 150, 255),
                    wrap=_SIDEBAR_W - 16,
                )

            dpg.add_spacer(height=4)

            # ── Active track ──────────────────────────────────────────
            with dpg.collapsing_header(label="Active Track", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Track ID:")
                    dpg.add_input_int(tag="active_tid", default_value=1, width=80,
                                      min_value=0, max_value=9999,
                                      callback=self._on_active_tid_change)
                    dpg.add_button(label="New Track", width=90,
                                   callback=self.new_track)
                dpg.add_spacer(height=2)
                dpg.add_text("", tag="active_track_color_label")

            dpg.add_spacer(height=4)

            # ── Range Labeling ────────────────────────────────────────
            with dpg.collapsing_header(label="Range Labeling",
                                       default_open=True):
                dpg.add_text(
                             "Select a track, choose a frame span, and apply track or action changes across that range.",
                             color=(150, 150, 160), wrap=_SIDEBAR_W - 16)
                with dpg.group(horizontal=True):
                    dpg.add_text("Start:", indent=4)
                    dpg.add_input_int(tag="range_start", default_value=0, width=70,
                                      min_value=0, max_value=0)
                    dpg.add_button(label="Use Current", width=88,
                                   callback=lambda: dpg.set_value(
                                       "range_start", self.current_frame_idx))
                with dpg.group(horizontal=True):
                    dpg.add_text("End:", indent=4)
                    dpg.add_input_int(tag="range_end", default_value=0, width=70,
                                      min_value=0, max_value=0)
                    dpg.add_button(label="Use Current", width=88,
                                   callback=lambda: dpg.set_value(
                                       "range_end", self.current_frame_idx))
                with dpg.group(horizontal=True):
                    dpg.add_text("Track ID:", indent=4)
                    dpg.add_input_int(tag="range_track", default_value=1, width=80,
                                      min_value=0, max_value=9999)
                with dpg.group(horizontal=True):
                    dpg.add_text("Action:", indent=4)
                    dpg.add_combo(items=["(no change)"], tag="range_action",
                                  default_value="(no change)", width=-1)
                dpg.add_checkbox(label="Interpolate boxes", tag="range_interp",
                                 default_value=True)
                dpg.add_button(label="Apply to Range", width=-1,
                               height=30, callback=self._apply_range_ui)
                dpg.add_button(label="Remove Track From Range", width=-1,
                               callback=self._clear_range_ui)

            dpg.add_spacer(height=4)

            # ── Objects in frame ──────────────────────────────────────
            with dpg.collapsing_header(label="Objects in Current Frame", default_open=True):
                dpg.add_listbox(items=[], tag="obj_listbox", num_items=6,
                                width=-1, callback=self._on_list_select)

            dpg.add_spacer(height=4)

            # ── Selected box ──────────────────────────────────────────
            with dpg.collapsing_header(label="Selected Object", default_open=True):
                with dpg.group(horizontal=True):
                    dpg.add_text("Track ID:", indent=4)
                    dpg.add_input_text(tag="sel_tid", width=60,
                                       callback=self._on_sel_tid_change,
                                       on_enter=True)
                dpg.add_spacer(height=2)
                dpg.add_text("Action:", indent=4)
                dpg.add_combo(items=["(none)"], tag="sel_action", width=-1,
                              callback=self._on_sel_action_change)
                dpg.add_spacer(height=2)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="Occluded", tag="sel_occluded",
                                     callback=self._on_sel_flags_change)
                    dpg.add_checkbox(label="Is Crowd", tag="sel_crowd",
                                     callback=self._on_sel_flags_change)

            dpg.add_spacer(height=4)

            # ── Quick action buttons ───────────────────────────────────
            with dpg.collapsing_header(label="Quick Actions (0-4)", default_open=True):
                dpg.add_group(tag="action_buttons_group")

            dpg.add_spacer(height=4)

            # ── Auto-track ────────────────────────────────────────────
            with dpg.collapsing_header(label="Auto-Track", default_open=True):
                dpg.add_checkbox(label="Enable auto-track mode (A)",
                                 tag="auto_track_cb",
                                 callback=self._on_auto_track_toggle)
                dpg.add_text(
                             "When enabled, moving to the next frame automatically updates every initialized tracker.",
                             color=(140, 140, 150, 255))
                dpg.add_spacer(height=4)
                dpg.add_button(label="Reset All Trackers",
                               width=-1, callback=self._reset_all_trackers)
                dpg.add_button(label="Reset Selected Tracker",
                               width=-1, callback=self._reset_selected_tracker)

            dpg.add_spacer(height=4)

            # ── Labels management ─────────────────────────────────────
            with dpg.collapsing_header(label="Labels", default_open=False):
                dpg.add_text("Classes:", color=(160, 200, 255, 255))
                dpg.add_text("", tag="classes_display")
                dpg.add_button(label="Add Class Label",
                               callback=self._add_class_label_dialog, width=-1)
                dpg.add_spacer(height=4)
                dpg.add_text("Actions:", color=(160, 200, 255, 255))
                dpg.add_text("", tag="actions_display")
                dpg.add_button(label="Add Action Label",
                               callback=self._add_action_label_dialog, width=-1)

            dpg.add_spacer(height=4)

            # ── Frame info ────────────────────────────────────────────
            with dpg.collapsing_header(label="Frame Info", default_open=True):
                dpg.add_text("", tag="frame_info_text",
                             color=(120, 120, 130, 255))
                dpg.add_spacer(height=4)
                dpg.add_text("Project folder:", color=(160, 200, 255, 255))
                dpg.add_text("(not set)", tag="project_dir_text",
                             color=(120, 120, 130, 255), wrap=_SIDEBAR_W - 16)
                dpg.add_button(label="Change...", width=-1,
                               callback=self._set_project_dir_dialog)

    def _update_action_buttons(self) -> None:
        if not dpg.does_item_exist("action_buttons_group"):
            return
        dpg.delete_item("action_buttons_group", children_only=True)
        action_colors = [
            (85,  85,  85,  255), (21,  101, 192, 255), (46,  125, 50, 255),
            (230, 81,  0,   255), (183, 27,  28,  255),
        ]
        for i, name in enumerate(self.action_names):
            col = action_colors[i % len(action_colors)]
            label = f"[{i}] {name.capitalize()}"
            with dpg.theme() as btn_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button,        col)
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,
                                        (min(col[0]+30,255), min(col[1]+30,255),
                                         min(col[2]+30,255), 255))
            btn = dpg.add_button(label=label, width=-1, height=28,
                                 parent="action_buttons_group",
                                 callback=lambda s, a, u: self.set_action(u),
                                 user_data=i)
            dpg.bind_item_theme(btn, btn_theme)

        # Update action combo
        if dpg.does_item_exist("sel_action"):
            dpg.configure_item("sel_action",
                               items=["(none)"] + list(self.action_names))
        # Update range-labeling action combo
        if dpg.does_item_exist("range_action"):
            dpg.configure_item("range_action",
                               items=["(no change)", "(none)"] + list(self.action_names))

    # ─── Navigation bar ──────────────────────────────────────────────

    def _build_navbar(self) -> None:
        with dpg.group(horizontal=True, tag="navbar"):
            dpg.add_button(label="First", width=52,
                           callback=lambda: self.goto_frame(0))
            dpg.add_button(label="Prev", width=48,
                           callback=lambda: self.goto_frame(self.current_frame_idx - 1))
            dpg.add_text("Frame:")
            dpg.add_input_int(tag="frame_input", default_value=0, width=80,
                              min_value=0, max_value=0,
                              callback=lambda s, a: self.goto_frame(a),
                              on_enter=True)
            dpg.add_text("/ 0", tag="total_frames_text")
            dpg.add_button(label="Next", width=48,
                           callback=lambda: self.goto_frame(self.current_frame_idx + 1))
            dpg.add_button(label="Last", width=52,
                           callback=lambda: self.goto_frame(self.num_frames - 1))
            dpg.add_spacer(width=12)
            dpg.add_button(label="Propagate  (Space)",
                           callback=self.propagate_from_prev)
            dpg.add_spacer(width=12)

        # Progress / status bar
        with dpg.group(horizontal=True, tag="statusbar"):
            dpg.add_progress_bar(tag="progress_bar", default_value=0.0,
                                 width=-200, height=16)
            dpg.add_text("", tag="status_text", color=(100, 160, 200, 255))

    # ─── Mouse handlers ──────────────────────────────────────────────

    def _build_mouse_handlers(self) -> None:
        # dearpygui 2.x に add_mouse_press_handler は無い。押下相当は
        # add_mouse_click_handler (押した瞬間に発火) を使う。
        with dpg.handler_registry(tag="mouse_handlers"):
            dpg.add_mouse_click_handler(button=0,
                                        callback=self._on_mouse_press)
            dpg.add_mouse_drag_handler(button=0, threshold=1.0,
                                       callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=0,
                                          callback=self._on_mouse_release)
            dpg.add_mouse_click_handler(button=1,
                                        callback=self._on_right_click)

    # ─── Key handlers ────────────────────────────────────────────────

    def _build_key_handlers(self) -> None:
        with dpg.handler_registry(tag="key_handlers"):
            dpg.add_key_press_handler(dpg.mvKey_Right,
                                      callback=lambda: self.goto_frame(self.current_frame_idx + 1))
            dpg.add_key_press_handler(dpg.mvKey_Left,
                                      callback=lambda: self.goto_frame(self.current_frame_idx - 1))
            dpg.add_key_press_handler(dpg.mvKey_N,
                                      callback=lambda: self.goto_frame(self.current_frame_idx + 1))
            dpg.add_key_press_handler(dpg.mvKey_P,
                                      callback=lambda: self.goto_frame(self.current_frame_idx - 1))
            dpg.add_key_press_handler(dpg.mvKey_Spacebar,
                                      callback=self.propagate_from_prev)
            dpg.add_key_press_handler(dpg.mvKey_Delete,
                                      callback=self.delete_selected)
            dpg.add_key_press_handler(dpg.mvKey_Escape,
                                      callback=self.deselect)
            dpg.add_key_press_handler(dpg.mvKey_A,
                                      callback=self._toggle_auto_track_key)
            dpg.add_key_press_handler(dpg.mvKey_S, callback=self._ctrl_s_handler)
            for i in range(5):
                dpg.add_key_press_handler(
                    getattr(dpg, f"mvKey_{i}"),
                    callback=lambda s, a, u: self.set_action(u),
                    user_data=i,
                )

    # ═══════════════════════════════════════════════════════════════════
    # Canvas Rendering
    # ═══════════════════════════════════════════════════════════════════

    def _get_canvas_mouse(self) -> Tuple[float, float]:
        """Mouse position in canvas-local pixel coords."""
        mx, my = dpg.get_mouse_pos(local=False)
        try:
            cx, cy = dpg.get_item_rect_min("canvas_win")
        except Exception:
            cx, cy = 0, 0
        return float(mx - cx), float(my - cy)

    def _recompute_scale(self) -> None:
        try:
            cw, ch = dpg.get_item_rect_size("canvas_win")
        except Exception:
            cw, ch = self._canvas_w, self._canvas_h
        self._canvas_w = int(cw) if cw > 1 else self._canvas_w
        self._canvas_h = int(ch) if ch > 1 else self._canvas_h
        if self.frame_w > 0 and self.frame_h > 0:
            scale = min(self._canvas_w / self.frame_w,
                        self._canvas_h / self.frame_h)
            self._scale = scale
            self._offset_x = (self._canvas_w - self.frame_w * scale) / 2
            self._offset_y = (self._canvas_h - self.frame_h * scale) / 2

    def _i2c(self, x: float, y: float) -> Tuple[float, float]:
        return x * self._scale + self._offset_x, y * self._scale + self._offset_y

    def _c2i(self, cx: float, cy: float) -> Tuple[float, float]:
        s = max(self._scale, 1e-9)
        return (cx - self._offset_x) / s, (cy - self._offset_y) / s

    def _render_canvas(self) -> None:
        if not dpg.does_item_exist("canvas"):
            return
        dpg.delete_item("canvas", children_only=True)
        self._recompute_scale()
        cw, ch = self._canvas_w, self._canvas_h

        if self._current_bgr is None:
            dpg.draw_text((cw // 2 - 160, ch // 2), "Open a video file (File -> Open Video)",
                          color=(80, 80, 100, 255), size=16, parent="canvas")
            return

        # Frame image
        if dpg.does_item_exist(_FRAME_TEX):
            dw = int(self.frame_w * self._scale)
            dh = int(self.frame_h * self._scale)
            dpg.draw_image(_FRAME_TEX,
                           pmin=(self._offset_x, self._offset_y),
                           pmax=(self._offset_x + dw, self._offset_y + dh),
                           parent="canvas")

        # Annotated boxes
        for obj in self.annotations.get(self.current_frame_idx, []):
            self._draw_box(obj, selected=(obj is self._selected))

        # Rubber-band while drawing
        if self._mode == "draw" and self._draw_start and self._draw_end:
            xs = sorted([self._draw_start[0], self._draw_end[0]])
            ys = sorted([self._draw_start[1], self._draw_end[1]])
            p1 = self._i2c(xs[0], ys[0])
            p2 = self._i2c(xs[1], ys[1])
            dpg.draw_rectangle(p1, p2, color=(255, 255, 255, 200),
                                fill=(255, 255, 255, 30),
                                thickness=1.5, parent="canvas")

    def _draw_dashed_rect(self, c1: Tuple[float, float], c2: Tuple[float, float],
                          color, thickness: float = 1.5,
                          dash: float = 7.0, gap: float = 5.0) -> None:
        """破線の矩形を drawlist に描く (dpg に破線 API が無いため線分で近似)。"""
        import math

        x1, y1 = c1
        x2, y2 = c2

        def _edge(ax, ay, bx, by):
            length = math.hypot(bx - ax, by - ay)
            if length < 1.0:
                return
            ux, uy = (bx - ax) / length, (by - ay) / length
            d = 0.0
            while d < length:
                e = min(d + dash, length)
                dpg.draw_line((ax + ux * d, ay + uy * d),
                              (ax + ux * e, ay + uy * e),
                              color=color, thickness=thickness, parent="canvas")
                d += dash + gap

        _edge(x1, y1, x2, y1)
        _edge(x2, y1, x2, y2)
        _edge(x2, y2, x1, y2)
        _edge(x1, y2, x1, y1)

    def _draw_box(self, obj: AnnObject, selected: bool = False) -> None:
        x1, y1, x2, y2 = obj.bbox
        c1 = self._i2c(x1, y1)
        c2 = self._i2c(x2, y2)
        col = _track_color(obj.track_id)
        col_fill = _track_color(obj.track_id, alpha=50)
        thick = 3.0 if selected else 1.5
        if getattr(obj, "keyframe", True):
            dpg.draw_rectangle(c1, c2, color=col, fill=col_fill,
                                thickness=thick, parent="canvas")
        else:
            # 補間/自動生成 box は破線で描き、手動キーフレームと区別する
            self._draw_dashed_rect(c1, c2, col, thickness=thick)

        # Label
        action_s = (self.action_names[obj.action_id]
                    if 0 <= obj.action_id < len(self.action_names) else "?")
        tid_s = f"T{obj.track_id}" if obj.track_id >= 0 else "T?"
        label = f"{tid_s}|{action_s}"
        dpg.draw_text((c1[0] + 3, c1[1] + 2), label,
                      color=(0, 0, 0, 200), size=13, parent="canvas")
        dpg.draw_text((c1[0] + 2, c1[1] + 1), label,
                      color=(255, 255, 255, 220), size=13, parent="canvas")

        if obj.occluded:
            dpg.draw_text((c2[0] - 28, c1[1] + 2), "OCC",
                          color=(255, 230, 0, 220), size=11, parent="canvas")

        # Track indicator (colored dot)
        has_tracker = obj.track_id in self._trackers
        dot_col = (100, 255, 100, 220) if has_tracker else col
        dpg.draw_circle((c1[0] + 6, c2[1] - 6), 5,
                        color=dot_col, fill=dot_col, parent="canvas")

        # Resize handles when selected
        if selected:
            for hx, hy in [c1, (c2[0], c1[1]), (c1[0], c2[1]), c2]:
                dpg.draw_circle((hx, hy), _HANDLE_R,
                                color=col, fill=(255, 255, 255, 200),
                                parent="canvas")

    # ═══════════════════════════════════════════════════════════════════
    # Mouse Interaction
    # ═══════════════════════════════════════════════════════════════════

    def _is_over_canvas(self) -> bool:
        return dpg.does_item_exist("canvas_win") and dpg.is_item_hovered("canvas_win")

    def _on_mouse_press(self, sender, app_data) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)

        # Resize handle?
        if self._selected is not None:
            h = self._hit_handle(cx, cy, self._selected)
            if h is not None:
                self._mode = "resize"
                self._resize_handle = h
                self._drag_start = (ix, iy)
                self._drag_bbox_orig = list(self._selected.bbox)
                return

        # Existing box?
        objects = self.annotations.get(self.current_frame_idx, [])
        for obj in reversed(objects):
            bx1, by1, bx2, by2 = obj.bbox
            if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                self._select(obj)
                self._mode = "move"
                self._drag_start = (ix, iy)
                self._drag_bbox_orig = list(obj.bbox)
                return

        # Draw new box
        self.deselect(render=False)
        self._mode = "draw"
        ixc = max(0.0, min(ix, float(self.frame_w)))
        iyc = max(0.0, min(iy, float(self.frame_h)))
        self._draw_start = (ixc, iyc)
        self._draw_end = (ixc, iyc)

    def _on_mouse_drag(self, sender, app_data) -> None:
        if self._mode == "idle" or not self._is_over_canvas():
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)
        ixc = max(0.0, min(ix, float(self.frame_w)))
        iyc = max(0.0, min(iy, float(self.frame_h)))

        if self._mode == "draw":
            self._draw_end = (ixc, iyc)

        elif self._mode == "move" and self._selected is not None:
            dx = ix - self._drag_start[0]
            dy = iy - self._drag_start[1]
            ox1, oy1, ox2, oy2 = self._drag_bbox_orig
            w, h = ox2 - ox1, oy2 - oy1
            nx1 = int(max(0.0, min(ox1 + dx, self.frame_w - w)))
            ny1 = int(max(0.0, min(oy1 + dy, self.frame_h - h)))
            self._selected.bbox = [nx1, ny1, nx1 + w, ny1 + h]
            self._selected.keyframe = True  # 手動移動でキーフレーム化 (補間アンカー)
            # Re-init tracker on manual move
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        elif self._mode == "resize" and self._selected is not None:
            ox1, oy1, ox2, oy2 = self._drag_bbox_orig
            ixi, iyi = int(ixc), int(iyc)
            h_name = self._resize_handle
            if h_name == "tl":
                nx1, ny1, nx2, ny2 = ixi, iyi, ox2, oy2
            elif h_name == "tr":
                nx1, ny1, nx2, ny2 = ox1, iyi, ixi, oy2
            elif h_name == "bl":
                nx1, ny1, nx2, ny2 = ixi, oy1, ox2, iyi
            else:
                nx1, ny1, nx2, ny2 = ox1, oy1, ixi, iyi
            nx1, nx2 = sorted([nx1, nx2])
            ny1, ny2 = sorted([ny1, ny2])
            if nx2 - nx1 >= _MIN_BOX_PX and ny2 - ny1 >= _MIN_BOX_PX:
                self._selected.bbox = [nx1, ny1, nx2, ny2]
                self._selected.keyframe = True  # 手動リサイズでキーフレーム化
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        self._render_canvas()

    def _on_mouse_release(self, sender, app_data) -> None:
        if self._mode == "draw" and self._draw_start and self._draw_end:
            xs = sorted([int(self._draw_start[0]), int(self._draw_end[0])])
            ys = sorted([int(self._draw_start[1]), int(self._draw_end[1])])
            if xs[1] - xs[0] >= _MIN_BOX_PX and ys[1] - ys[0] >= _MIN_BOX_PX:
                obj = self._create_object([xs[0], ys[0], xs[1], ys[1]])
                # Initialize tracker in auto-track mode
                if self._auto_track and self._current_bgr is not None:
                    self._init_tracker(obj.track_id, self.current_frame_idx, obj.bbox)
        elif self._mode == "move" and self._selected is not None:
            # Re-initialize tracker from new position
            if self._auto_track and self._current_bgr is not None:
                self._init_tracker(self._selected.track_id,
                                   self.current_frame_idx, self._selected.bbox)
        elif self._mode == "resize" and self._selected is not None:
            if self._auto_track and self._current_bgr is not None:
                self._init_tracker(self._selected.track_id,
                                   self.current_frame_idx, self._selected.bbox)

        self._mode = "idle"
        self._draw_start = self._draw_end = None
        self._drag_start = self._drag_bbox_orig = None
        self._resize_handle = None
        self._render_canvas()
        self._refresh_sidebar()

    def _on_right_click(self, sender, app_data) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)
        objects = self.annotations.get(self.current_frame_idx, [])
        for obj in reversed(objects):
            bx1, by1, bx2, by2 = obj.bbox
            if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                self._select(obj)
                self._right_click_obj = obj
                self._show_context_menu(obj)
                return

    def _show_context_menu(self, obj: AnnObject) -> None:
        if dpg.does_item_exist("ctx_menu"):
            dpg.delete_item("ctx_menu")
        mx, my = dpg.get_mouse_pos(local=False)
        with dpg.window(tag="ctx_menu", popup=True, no_title_bar=True,
                        no_move=True, pos=(mx, my), min_size=(160, 10)):
            dpg.add_text(f"Track {obj.track_id}", color=(200, 200, 100, 255))
            dpg.add_separator()
            dpg.add_menu_item(label="Set Track ID...",
                              callback=self._prompt_track_id_dialog)
            dpg.add_separator()
            for i, name in enumerate(self.action_names):
                dpg.add_menu_item(
                    label=f"Action: {name}",
                    callback=lambda s, a, u: self.set_action(u),
                    user_data=i,
                )
            dpg.add_separator()
            if self._auto_track:
                dpg.add_menu_item(label="Init Tracker for This Box",
                                  callback=lambda: self._init_tracker_from_selected())
                dpg.add_menu_item(label="Remove Tracker for This Track",
                                  callback=lambda: self._reset_selected_tracker())
            dpg.add_separator()
            dpg.add_menu_item(label="Delete Box", callback=self.delete_selected)

    def _hit_handle(self, cx: float, cy: float, obj: AnnObject) -> Optional[str]:
        x1, y1, x2, y2 = obj.bbox
        corners = {
            "tl": self._i2c(x1, y1), "tr": self._i2c(x2, y1),
            "bl": self._i2c(x1, y2), "br": self._i2c(x2, y2),
        }
        for name, (hx, hy) in corners.items():
            if abs(cx - hx) <= _HANDLE_R + 3 and abs(cy - hy) <= _HANDLE_R + 3:
                return name
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Video Loading
    # ═══════════════════════════════════════════════════════════════════

    def _open_video_dialog(self) -> None:
        path = _tk_open_file(
            title="Open Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if path is not None:
            self._load_video(path)

    def _load_video(self, path: Path) -> None:
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self._status(f"Cannot open: {path.name}")
            return

        self.cap = cap
        self.video_path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.annotations = {}
        self.current_frame_idx = 0
        self._selected = None
        self._trackers.clear()
        self.annotation_path = path.parent / (path.stem + "_annotations.json")

        # Create frame texture
        if dpg.does_item_exist(_FRAME_TEX):
            dpg.delete_item(_FRAME_TEX)
        blank = np.zeros(self.frame_w * self.frame_h * 4, dtype=np.float32)
        if not dpg.does_item_exist("tex_reg"):
            with dpg.texture_registry(tag="tex_reg", show=False):
                dpg.add_raw_texture(self.frame_w, self.frame_h, blank,
                                    tag=_FRAME_TEX,
                                    format=dpg.mvFormat_Float_rgba)
        else:
            dpg.push_container_stack("tex_reg")
            dpg.add_raw_texture(self.frame_w, self.frame_h, blank,
                                tag=_FRAME_TEX,
                                format=dpg.mvFormat_Float_rgba)
            dpg.pop_container_stack()

        if dpg.does_item_exist("frame_input"):
            dpg.configure_item("frame_input", max_value=self.num_frames - 1)
        if dpg.does_item_exist("total_frames_text"):
            dpg.set_value("total_frames_text", f"/ {self.num_frames - 1}")
        for _rtag in ("range_start", "range_end"):
            if dpg.does_item_exist(_rtag):
                dpg.configure_item(_rtag, max_value=max(self.num_frames - 1, 0))
                dpg.set_value(_rtag, 0)

        dpg.set_viewport_title(f"YOAKE Annotation Tool — {path.name}")
        self.goto_frame(0)
        self._status(f"Loaded: {path.name}  ({self.num_frames} frames, "
                     f"{self.fps:.1f} fps, {self.frame_w}×{self.frame_h})")

    def _read_frame_bgr(self, idx: int) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    # ═══════════════════════════════════════════════════════════════════
    # Frame Navigation
    # ═══════════════════════════════════════════════════════════════════

    def goto_frame(self, idx: int) -> None:
        if self.cap is None:
            return
        idx = max(0, min(idx, self.num_frames - 1))

        # Auto-track: run trackers when advancing one frame
        if self._auto_track and idx == self.current_frame_idx + 1 and self._trackers:
            frame_bgr = self._read_frame_bgr(idx)
            if frame_bgr is not None:
                self._run_trackers_for_frame(idx, frame_bgr)

        self.current_frame_idx = idx
        frame_bgr = self._read_frame_bgr(idx)
        self._current_bgr = frame_bgr

        if frame_bgr is not None and dpg.does_item_exist(_FRAME_TEX):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
            data = (rgba.astype(np.float32) / 255.0).flatten()
            dpg.set_value(_FRAME_TEX, data)

        self.deselect(render=False)
        if dpg.does_item_exist("frame_input"):
            dpg.set_value("frame_input", idx)

        self._render_canvas()
        self._refresh_sidebar()

        annotated = len(self.annotations)
        pct = annotated / max(self.num_frames, 1)
        if dpg.does_item_exist("progress_bar"):
            dpg.set_value("progress_bar", pct)
        self._status(
            f"Frame {idx}/{self.num_frames - 1}  |  "
            f"Annotated: {annotated}/{self.num_frames} ({pct * 100:.0f}%)"
        )

    # ═══════════════════════════════════════════════════════════════════
    # Selection
    # ═══════════════════════════════════════════════════════════════════

    def _select(self, obj: AnnObject) -> None:
        self._selected = obj
        self._refresh_sidebar()
        self._render_canvas()

    def deselect(self, render: bool = True) -> None:
        self._selected = None
        self._refresh_sidebar()
        if render:
            self._render_canvas()

    # ═══════════════════════════════════════════════════════════════════
    # Annotation Management
    # ═══════════════════════════════════════════════════════════════════

    def _create_object(self, bbox: List[int]) -> AnnObject:
        obj = AnnObject(bbox=bbox, track_id=self.active_track_id, action_id=-1)
        self.annotations.setdefault(self.current_frame_idx, []).append(obj)
        self._select(obj)
        return obj

    def delete_selected(self) -> None:
        if self._selected is None:
            return
        objs = self.annotations.get(self.current_frame_idx, [])
        if self._selected in objs:
            objs.remove(self._selected)
        if self._selected.track_id in self._trackers:
            del self._trackers[self._selected.track_id]
        self.deselect()

    def new_track(self) -> None:
        self.next_track_id += 1
        self.active_track_id = self.next_track_id
        if dpg.does_item_exist("active_tid"):
            dpg.set_value("active_tid", self.next_track_id)
        if dpg.does_item_exist("range_track"):
            dpg.set_value("range_track", self.active_track_id)
        self._update_active_track_color()
        self._status(f"New track ID: {self.next_track_id}")

    def propagate_from_prev(self) -> None:
        if self.cap is None:
            return
        prev = self.current_frame_idx - 1
        if prev < 0:
            self._status("No previous frame to propagate from.")
            return
        prev_objs = self.annotations.get(prev, [])
        if not prev_objs:
            self._status("Previous frame has no annotations.")
            return
        existing_tids = {o.track_id for o in
                         self.annotations.get(self.current_frame_idx, [])}
        count = 0
        for obj in prev_objs:
            if obj.track_id not in existing_tids:
                global _gid
                new_obj = copy.deepcopy(obj)
                _gid += 1
                new_obj.object_id = _gid
                self.annotations.setdefault(self.current_frame_idx, []).append(new_obj)
                existing_tids.add(obj.track_id)
                count += 1
        self._render_canvas()
        self._refresh_sidebar()
        self._status(f"Propagated {count} object(s) from frame {prev}.")

    # ═══════════════════════════════════════════════════════════════════
    # Range Labeling  (範囲ラベリング)
    # ═══════════════════════════════════════════════════════════════════

    def _track_obj(self, frame_idx: int, track_id: int) -> Optional[AnnObject]:
        """指定フレームで track_id を持つ最初のオブジェクトを返す (無ければ None)。"""
        for obj in self.annotations.get(frame_idx, []):
            if obj.track_id == track_id:
                return obj
        return None

    def apply_range(self, track_id: int, action_id: Optional[int],
                    f_start: int, f_end: int, interpolate: bool) -> None:
        """区間 [f_start, f_end] の track に行動/bbox を一括適用する。

        - ``interpolate=True`` かつ区間内にキーフレーム box が 2 個以上あれば、隣接
          キーフレーム間を線形補間して中間フレームの box を生成する (端の外側は
          最も近いキーフレームを保持)。キーフレームが 1 個なら全域その box を保持。
        - ``action_id`` が ``None`` の場合は行動を変更しない。box が無く geometry も
          得られないフレームはスキップ (既存 box があれば行動だけ更新)。
        保存形式は v1.1 のまま (生成 box は通常の per-frame object)。
        """
        if self.cap is None:
            self._status("No video loaded.")
            return
        if f_end < f_start:
            f_start, f_end = f_end, f_start
        f_start = max(0, f_start)
        f_end = min(self.num_frames - 1, f_end) if self.num_frames else f_end

        anchors: List[Tuple[int, List[int]]] = []
        for f in range(f_start, f_end + 1):
            obj = self._track_obj(f, track_id)
            if obj is not None and obj.keyframe:
                anchors.append((f, list(obj.bbox)))

        # 補間 ON かつアンカーがあれば box_map を生成 (純関数 resolve_range_boxes)
        box_map = (resolve_range_boxes(anchors, f_start, f_end)
                   if interpolate and anchors else {})
        anchor_frames = {f for f, _ in anchors}
        count = 0
        for f in range(f_start, f_end + 1):
            obj = self._track_obj(f, track_id)
            box = box_map.get(f)
            if obj is None:
                if box is None:
                    continue  # 配置すべき box が無い → スキップ
                placed = round_bbox(clamp_bbox(box, self.frame_w, self.frame_h))
                obj = AnnObject(bbox=placed, track_id=track_id, action_id=-1,
                                keyframe=False)
                self.annotations.setdefault(f, []).append(obj)
            elif box is not None and f not in anchor_frames:
                obj.bbox = round_bbox(clamp_bbox(box, self.frame_w, self.frame_h))
                obj.keyframe = False
            if action_id is not None:
                obj.action_id = action_id
            count += 1

        self.next_track_id = max(self.next_track_id, track_id)
        self._render_canvas()
        self._refresh_sidebar()
        note = ""
        if interpolate and len(anchors) < 2:
            note = ("(keyframe held)" if anchors
                    else "(no box -> action only)")
        self._status(f"Range [{f_start},{f_end}] track {track_id}: "
                     f"applied to {count} frame(s) {note}")

    def clear_range(self, track_id: int, f_start: int, f_end: int) -> None:
        """区間 [f_start, f_end] の指定 track のオブジェクトを削除する。"""
        if f_end < f_start:
            f_start, f_end = f_end, f_start
        removed = 0
        for f in range(f_start, f_end + 1):
            objs = self.annotations.get(f, [])
            keep = [o for o in objs if o.track_id != track_id]
            removed += len(objs) - len(keep)
            if keep:
                self.annotations[f] = keep
            elif f in self.annotations:
                del self.annotations[f]
        self.deselect()
        self._render_canvas()
        self._refresh_sidebar()
        self._status(f"Range [{f_start},{f_end}] track {track_id}: "
                     f"removed {removed} object(s).")

    def _range_action_id(self) -> Optional[int]:
        val = dpg.get_value("range_action") if dpg.does_item_exist("range_action") else None
        if val in (None, "(no change)"):
            return None
        if val == "(none)":
            return -1
        return self.action_names.index(val) if val in self.action_names else None

    def _apply_range_ui(self) -> None:
        self.apply_range(
            track_id=int(dpg.get_value("range_track")),
            action_id=self._range_action_id(),
            f_start=int(dpg.get_value("range_start")),
            f_end=int(dpg.get_value("range_end")),
            interpolate=bool(dpg.get_value("range_interp")),
        )

    def _clear_range_ui(self) -> None:
        self.clear_range(
            track_id=int(dpg.get_value("range_track")),
            f_start=int(dpg.get_value("range_start")),
            f_end=int(dpg.get_value("range_end")),
        )

    def set_action(self, action_id: int) -> None:
        if self._selected is None:
            self._status(f"No box selected. Action [{action_id}] was ignored.")
            return
        self._selected.action_id = action_id
        name = (self.action_names[action_id]
                if action_id < len(self.action_names) else str(action_id))
        self._status(f"Action set to {name}")
        self._refresh_sidebar()
        self._render_canvas()

    # ═══════════════════════════════════════════════════════════════════
    # Auto-Tracking
    # ═══════════════════════════════════════════════════════════════════

    def _init_tracker(self, track_id: int, frame_idx: int,
                      bbox: List[int]) -> bool:
        """Initialize a CSRT tracker for the given track and bounding box."""
        frame_bgr = self._read_frame_bgr(frame_idx)
        if frame_bgr is None:
            return False
        tracker = cv2.TrackerCSRT_create()
        x1, y1, x2, y2 = bbox
        tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        self._trackers[track_id] = tracker
        self._status(f"Tracker initialized for T{track_id}.")
        return True

    def _init_tracker_from_selected(self) -> None:
        if self._selected is None:
            return
        self._init_tracker(self._selected.track_id,
                           self.current_frame_idx, self._selected.bbox)
        self._render_canvas()

    def _run_trackers_for_frame(self, frame_idx: int,
                                frame_bgr: np.ndarray) -> None:
        """Run all active trackers on frame_bgr and create annotation objects."""
        existing_tids = {o.track_id for o in
                         self.annotations.get(frame_idx, [])}
        for track_id, tracker in list(self._trackers.items()):
            if track_id in existing_tids:
                continue
            success, xywh = tracker.update(frame_bgr)
            if success:
                x, y, w, h = [int(v) for v in xywh]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(self.frame_w, x + w)
                y2 = min(self.frame_h, y + h)
                if x2 - x1 >= _MIN_BOX_PX and y2 - y1 >= _MIN_BOX_PX:
                    # Find action_id from previous frame if available
                    prev_action = -1
                    for obj in self.annotations.get(frame_idx - 1, []):
                        if obj.track_id == track_id:
                            prev_action = obj.action_id
                            break
                    global _gid
                    new_obj = AnnObject(bbox=[x1, y1, x2, y2],
                                        track_id=track_id,
                                        action_id=prev_action)
                    self.annotations.setdefault(frame_idx, []).append(new_obj)
            else:
                # Tracker lost — keep it but stop updating
                self._status(f"Tracker lost for T{track_id} at frame {frame_idx}.")

    def _reset_all_trackers(self) -> None:
        self._trackers.clear()
        self._status("All trackers reset.")
        self._render_canvas()

    def _reset_selected_tracker(self) -> None:
        if self._selected is None:
            return
        tid = self._selected.track_id
        if tid in self._trackers:
            del self._trackers[tid]
            self._status(f"Tracker for T{tid} removed.")
        self._render_canvas()

    def _on_auto_track_toggle(self, sender, app_data) -> None:
        self._auto_track = app_data
        if not self._auto_track:
            self._trackers.clear()
        self._status(f"Auto-track: {'ON' if self._auto_track else 'OFF'}")

    def _toggle_auto_track_key(self) -> None:
        if not dpg.does_item_exist("auto_track_cb"):
            return
        new_val = not dpg.get_value("auto_track_cb")
        dpg.set_value("auto_track_cb", new_val)
        self._auto_track = new_val
        if not self._auto_track:
            self._trackers.clear()
        self._status(f"Auto-track: {'ON' if self._auto_track else 'OFF'}")

    # ═══════════════════════════════════════════════════════════════════
    # Label Management
    # ═══════════════════════════════════════════════════════════════════

    def _add_class_label_dialog(self) -> None:
        self._label_dialog("Add Class Label", self._on_add_class)

    def _add_action_label_dialog(self) -> None:
        self._label_dialog("Add Action Label", self._on_add_action)

    def _label_dialog(self, title: str, confirm_cb) -> None:
        if dpg.does_item_exist("label_dlg"):
            dpg.delete_item("label_dlg")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        with dpg.window(label=title, tag="label_dlg", modal=True,
                        width=300, height=120,
                        pos=[(vw - 300) // 2, (vh - 120) // 2],
                        no_resize=True):
            dpg.add_input_text(tag="label_dlg_input", width=-1,
                               hint="New label name")
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add", width=100, callback=confirm_cb)
                dpg.add_button(label="Cancel", width=100,
                               callback=lambda: dpg.delete_item("label_dlg"))

    def _on_add_class(self) -> None:
        name = dpg.get_value("label_dlg_input").strip()
        if name and name not in self.class_names:
            self.class_names.append(name)
            self._refresh_labels_display()
        if dpg.does_item_exist("label_dlg"):
            dpg.delete_item("label_dlg")

    def _on_add_action(self) -> None:
        name = dpg.get_value("label_dlg_input").strip()
        if name and name not in self.action_names:
            self.action_names.append(name)
            self._update_action_buttons()
            self._refresh_labels_display()
        if dpg.does_item_exist("label_dlg"):
            dpg.delete_item("label_dlg")

    def _refresh_labels_display(self) -> None:
        if dpg.does_item_exist("classes_display"):
            dpg.set_value("classes_display",
                          "\n".join(f"  [{i}] {n}"
                                    for i, n in enumerate(self.class_names)))
        if dpg.does_item_exist("actions_display"):
            dpg.set_value("actions_display",
                          "\n".join(f"  [{i}] {n}"
                                    for i, n in enumerate(self.action_names)))

    # ═══════════════════════════════════════════════════════════════════
    # Sidebar Refresh
    # ═══════════════════════════════════════════════════════════════════

    def _refresh_sidebar(self) -> None:
        if not dpg.does_item_exist("obj_listbox"):
            return

        objects = self.annotations.get(self.current_frame_idx, [])
        items = []
        sel_idx = None
        for i, obj in enumerate(objects):
            action_s = (self.action_names[obj.action_id]
                        if 0 <= obj.action_id < len(self.action_names) else "—")
            trk_mark = "🔵" if obj.track_id in self._trackers else "  "
            items.append(f"{trk_mark} T{obj.track_id:<3}  {action_s:<14}"
                         f"{'[OCC]' if obj.occluded else ''}")
            if obj is self._selected:
                sel_idx = i

        dpg.configure_item("obj_listbox", items=items)
        if sel_idx is not None:
            dpg.set_value("obj_listbox", items[sel_idx])

        # Selected box props
        if self._selected is not None:
            dpg.set_value("sel_tid", str(self._selected.track_id))
            action_val = (self.action_names[self._selected.action_id]
                          if 0 <= self._selected.action_id < len(self.action_names)
                          else "(none)")
            dpg.set_value("sel_action", action_val)
            dpg.set_value("sel_occluded", self._selected.occluded)
            dpg.set_value("sel_crowd", self._selected.is_crowd)
        else:
            dpg.set_value("sel_tid", "")
            dpg.set_value("sel_action", "(none)")
            dpg.set_value("sel_occluded", False)
            dpg.set_value("sel_crowd", False)

        # Frame info
        annotated = len(self.annotations)
        if self.cap is not None:
            info = (
                f"Frame : {self.current_frame_idx}/{max(self.num_frames-1,0)}\n"
                f"Objects: {len(objects)}\n"
                f"Annotated: {annotated}/{self.num_frames}\n"
                f"Size : {self.frame_w}×{self.frame_h}\n"
                f"FPS  : {self.fps:.1f}\n"
                f"Trackers: {len(self._trackers)} active"
            )
        else:
            info = "No video loaded."
        if dpg.does_item_exist("frame_info_text"):
            dpg.set_value("frame_info_text", info)

        self._refresh_labels_display()

    def _update_active_track_color(self) -> None:
        if not dpg.does_item_exist("active_track_color_label"):
            return
        r, g, b, _ = _track_color(self.active_track_id)
        dpg.configure_item("active_track_color_label",
                           color=(r, g, b, 255),
                           default_value=f"  ■ Track {self.active_track_id}")

    def _on_active_tid_change(self, sender, app_data) -> None:
        self.active_track_id = max(0, app_data)
        self.next_track_id = max(self.next_track_id, self.active_track_id)
        if dpg.does_item_exist("range_track"):
            dpg.set_value("range_track", self.active_track_id)
        self._update_active_track_color()

    def _on_list_select(self, sender, app_data) -> None:
        if app_data is None:
            return
        objects = self.annotations.get(self.current_frame_idx, [])
        for obj in objects:
            action_s = (self.action_names[obj.action_id]
                        if 0 <= obj.action_id < len(self.action_names) else "—")
            trk_mark = "🔵" if obj.track_id in self._trackers else "  "
            label = (f"{trk_mark} T{obj.track_id:<3}  {action_s:<14}"
                     f"{'[OCC]' if obj.occluded else ''}")
            if label == app_data:
                self._select(obj)
                return

    def _on_sel_tid_change(self, sender, app_data) -> None:
        if self._selected is None:
            return
        try:
            tid = int(app_data)
            self._selected.track_id = tid
            self.next_track_id = max(self.next_track_id, tid)
            self._render_canvas()
            self._refresh_sidebar()
        except ValueError:
            pass

    def _on_sel_action_change(self, sender, app_data) -> None:
        if self._selected is None:
            return
        if app_data in self.action_names:
            self._selected.action_id = self.action_names.index(app_data)
        elif app_data == "(none)":
            self._selected.action_id = -1
        self._render_canvas()

    def _on_sel_flags_change(self) -> None:
        if self._selected is None:
            return
        if dpg.does_item_exist("sel_occluded"):
            self._selected.occluded = dpg.get_value("sel_occluded")
        if dpg.does_item_exist("sel_crowd"):
            self._selected.is_crowd = dpg.get_value("sel_crowd")

    def _prompt_track_id_dialog(self) -> None:
        if self._selected is None:
            return
        if dpg.does_item_exist("tid_dlg"):
            dpg.delete_item("tid_dlg")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        current = self._selected.track_id
        with dpg.window(label="Set Track ID", tag="tid_dlg", modal=True,
                        width=260, height=100,
                        pos=[(vw - 260) // 2, (vh - 100) // 2],
                        no_resize=True):
            dpg.add_input_int(tag="tid_dlg_input", default_value=current,
                              min_value=0, max_value=9999, width=-1)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Set", width=100,
                               callback=self._on_track_id_dialog_confirm)
                dpg.add_button(label="Cancel", width=100,
                               callback=lambda: dpg.delete_item("tid_dlg"))

    def _on_track_id_dialog_confirm(self) -> None:
        if self._selected is None or not dpg.does_item_exist("tid_dlg_input"):
            return
        tid = dpg.get_value("tid_dlg_input")
        self._selected.track_id = tid
        self.next_track_id = max(self.next_track_id, tid)
        dpg.delete_item("tid_dlg")
        self._render_canvas()
        self._refresh_sidebar()

    # ═══════════════════════════════════════════════════════════════════
    # File I/O
    # ═══════════════════════════════════════════════════════════════════

    def save_annotation(self) -> None:
        if self.cap is None:
            self._status("No video loaded.")
            return
        # プロジェクトディレクトリが指定されていれば、静かに annotations.json を上書き。
        if self.project_dir is not None:
            path = self.project_dir / "annotations.json"
            self._write_annotation(path)
            return
        # 未設定なら従来通りダイアログで保存先を選ばせる。
        initial = (self.annotation_path.name
                   if self.annotation_path else "annotations.json")
        path = _tk_save_file(
            title="Save Annotation",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_file=initial,
        )
        if path is not None:
            self._write_annotation(path)

    def _save_as_dialog(self) -> None:
        """プロジェクトディレクトリを設定していても、明示的にダイアログで保存先を選ぶ。"""
        if self.cap is None:
            self._status("No video loaded.")
            return
        initial = (self.annotation_path.name
                   if self.annotation_path else "annotations.json")
        path = _tk_save_file(
            title="Save Annotation As",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_file=initial,
        )
        if path is not None:
            self._write_annotation(path)

    def _set_project_dir_dialog(self) -> None:
        """メニューからプロジェクト保存ディレクトリを変更する。"""
        p = _tk_ask_dir(title="Select Project Directory")
        if p is None:
            return
        self.project_dir = p
        self._refresh_project_dir_display()
        self._status(f"Project folder set to {p}")

    def _refresh_project_dir_display(self) -> None:
        if dpg.does_item_exist("project_dir_text"):
            dpg.set_value(
                "project_dir_text",
                str(self.project_dir) if self.project_dir else "(not set)",
            )

    def _write_annotation(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        video_id = self.video_path.stem if self.video_path else "video_001"

        frames = []
        for fi in sorted(self.annotations.keys()):
            frames.append({
                "frame_index": fi,
                "image_path": f"images/{video_id}/{fi:06d}.jpg",
                "objects": [o.to_dict() for o in self.annotations[fi]],
            })

        data = {
            "meta": {
                "version": "1.1",
                "description": f"Annotated with YOAKE Annotation Tool — {self.project_name}",
                "created": datetime.now().isoformat(timespec="seconds"),
                "fps_default": self.fps,
                "image_root": "",
            },
            "class_names": list(self.class_names),
            "action_names": list(self.action_names),
            "videos": [{
                "video_id": video_id,
                "fps": self.fps,
                "width": self.frame_w,
                "height": self.frame_h,
                "num_frames": self.num_frames,
                "frames": frames,
            }],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.annotation_path = path
        self._status(f"Saved to {path.name}")

    def _save_and_export_dialog(self) -> None:
        if self.cap is None:
            self._status("No video loaded.")
            return
        # プロジェクトディレクトリが設定されていればそこへ出力、無ければダイアログ。
        out = self.project_dir
        if out is None:
            out = _tk_ask_dir(title="Select Output Directory")
            if out is None:
                return
        video_id = self.video_path.stem if self.video_path else "video_001"
        img_dir = out / "images" / video_id
        img_dir.mkdir(parents=True, exist_ok=True)

        self._write_annotation(out / "annotations.json")

        frame_indices = sorted(self.annotations.keys())
        for fi in frame_indices:
            frame_bgr = self._read_frame_bgr(fi)
            if frame_bgr is not None:
                cv2.imwrite(str(img_dir / f"{fi:06d}.jpg"), frame_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])

        self._status(f"Exported {len(frame_indices)} frames to {img_dir}")

    def _load_annotation_dialog(self) -> None:
        path = _tk_open_file(
            title="Load Annotation JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is not None:
            self._load_annotation_file(path)

    def _load_annotation_file(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self._status(f"Load error: {exc}")
            return

        # Restore labels from JSON
        if "class_names" in data:
            self.class_names = data["class_names"]
        if "action_names" in data:
            self.action_names = data["action_names"]
        if dpg.does_item_exist("action_buttons_group"):
            self._update_action_buttons()

        self.annotations = {}
        for video in data.get("videos", []):
            for frame in video.get("frames", []):
                fi = frame["frame_index"]
                self.annotations[fi] = [
                    AnnObject.from_dict(o) for o in frame.get("objects", [])
                ]

        max_tid = max(
            (obj.track_id for objs in self.annotations.values()
             for obj in objs if obj.track_id >= 0),
            default=0,
        )
        self.next_track_id = max_tid + 1
        self.active_track_id = self.next_track_id
        if dpg.does_item_exist("active_tid"):
            dpg.set_value("active_tid", self.next_track_id)

        self._render_canvas()
        self._refresh_sidebar()
        self.annotation_path = path
        # 読み込んだ JSON の親ディレクトリを既定のプロジェクトディレクトリにする。
        # (既に手動で別ディレクトリを設定していた場合は上書きしない。)
        if self.project_dir is None:
            self.project_dir = path.parent
            self._refresh_project_dir_display()
        self._status(f"Loaded: {path.name}")

    # ═══════════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════════

    def _status(self, msg: str) -> None:
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", msg)

    def _ctrl_s_handler(self) -> None:
        if dpg.is_key_down(_KEY_CTRL):
            self.save_annotation()

    def _show_help(self) -> None:
        if dpg.does_item_exist("help_win"):
            dpg.delete_item("help_win")
        vw, vh = dpg.get_viewport_width(), dpg.get_viewport_height()
        txt = (
            "Keyboard Shortcuts\n"
            "------------------\n"
            "Right / N      Next frame\n"
            "Left / P       Previous frame\n"
            "Space          Propagate boxes from previous frame\n"
            "A              Toggle auto-track mode\n"
            "Delete         Delete selected box\n"
            "0 - 4          Set action for selected box\n"
            "Ctrl+S         Save annotation JSON\n"
            "Escape         Deselect\n\n"
            "Mouse\n"
            "-----\n"
            "Drag on empty area   Draw a new bounding box\n"
            "Click on box         Select a box\n"
            "Drag selected box    Move a box\n"
            "Drag corner dot      Resize a box\n"
            "Right-click          Open the context menu\n\n"
            "Auto-Track (A)\n"
            "--------------\n"
            "Draw a box while auto-track mode is on to initialize a tracker automatically.\n"
            "When you move to the next frame, the CSRT tracker updates every active box.\n"
            "A green dot means the tracker is active for that object.\n"
        )
        with dpg.window(label="Help / Shortcuts", tag="help_win",
                        width=380, height=420, modal=False,
                        pos=[(vw - 380) // 2, (vh - 420) // 2]):
            dpg.add_text(txt)
            dpg.add_spacer(height=8)
            dpg.add_button(label="Close", width=-1,
                           callback=lambda: dpg.delete_item("help_win"))

    def _on_viewport_resize(self) -> None:
        """Called when the viewport is resized — update canvas and sidebar sizes."""
        if not dpg.does_item_exist("canvas_win"):
            return
        vw = dpg.get_viewport_width()
        vh = dpg.get_viewport_height()
        cw = max(vw - _SIDEBAR_W - 20, 400)
        ch = max(vh - 110, 300)
        dpg.configure_item("canvas_win", width=cw, height=ch)
        dpg.configure_item("canvas", width=cw, height=ch)
        if dpg.does_item_exist("sidebar_win"):
            dpg.configure_item("sidebar_win", height=ch)
        self._render_canvas()

    # ═══════════════════════════════════════════════════════════════════
    # Run
    # ═══════════════════════════════════════════════════════════════════

    def run(self, video: Optional[Path] = None,
            load_json: Optional[Path] = None) -> None:
        dpg.create_viewport(title="YOAKE Annotation Tool",
                            width=1280, height=820,
                            min_width=900, min_height=600)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

        # Show project setup screen first
        self._show_setup_screen()

        # Handle CLI arguments after first render
        if video or load_json:
            def _deferred():
                if dpg.does_item_exist("setup_win"):
                    # auto-confirm setup with defaults
                    self.class_names = list(_DEFAULT_CLASSES)
                    self.action_names = list(_DEFAULT_ACTIONS)
                    dpg.delete_item("setup_win")
                    self._build_main_window()
                    self._update_action_buttons()
                if video:
                    self._load_video(video)
                if load_json:
                    self._load_annotation_file(load_json)
            dpg.set_frame_callback(2, _deferred)

        dpg.start_dearpygui()
        dpg.destroy_context()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="annotate",
                                     description="YOAKE GUI Annotation Tool")
    parser.add_argument("video", nargs="?", help="Video file to open")
    parser.add_argument("--load", metavar="JSON",
                        help="Load existing annotation JSON")
    args = parser.parse_args(argv)

    app = AnnotationApp()
    app.run(
        video=Path(args.video) if args.video else None,
        load_json=Path(args.load) if args.load else None,
    )


if __name__ == "__main__":
    main()
