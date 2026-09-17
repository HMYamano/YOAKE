#!/usr/bin/env python3
"""
YOAKE Annotation Tool — Dear PyGui Edition
==========================================
Video annotation for animal-behavior datasets. Draw boxes, track them forward,
label actions, and save YOAKE JSON (v1.1, or v2.0 when rotated boxes are used).

Usage:
    python -m htrtdetr.annotator [video] [--load annotations.json]
                                 [--import-predictions predictions.json]
    python -m htrtdetr.annotator --selftest        # synthetic-input acceptance run

The keyboard map is generated from ``htrtdetr.gui.shortcuts`` — press F1 in the
application for the authoritative list. The toolbar labels, the tooltips and the
help window all come from that same table, so they cannot drift apart.
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
    from .gui import auto_label, shortcuts, theme, widgets
except ImportError:  # スクリプト直接実行時のフォールバック
    from htrtdetr.gui.labeling_ops import clamp_bbox, resolve_range_boxes, round_bbox
    from htrtdetr.gui import auto_label, shortcuts, theme, widgets


# ─── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_ACTIONS = ["stationary", "walking", "grooming", "courtship", "aggression"]
_DEFAULT_CLASSES = ["fly"]

# 20 色のトラックパレット。theme.TRACK_PALETTE が唯一の定義元で、
# アノテータと解析キャンバスで同じ色が出るようにしている。
_TRACK_PALETTE = [tuple(c) for c in theme.TRACK_PALETTE]

_MIN_BOX_PX = 5
_HANDLE_R = 7             # resize-handle radius in canvas pixels
_ROT_HANDLE_OFFSET = 26   # rotation handle distance from the box edge (canvas px)
_FRAME_TEX = "yoake_frame_tex"
_SIDEBAR_W = 300
_CHROME_H = 152           # menubar + toolbar + navbar + status bar
_UNDO_DEPTH = 50
_ZOOM_MIN = 0.2
_ZOOM_MAX = 20.0
_PAN_KEEP_PX = 64         # never let the frame leave the canvas entirely

# dearpygui 2.x では mvKey_Control が廃止され mvKey_ModCtrl になった (1.x フォールバック付き)
_KEY_CTRL = getattr(dpg, "mvKey_ModCtrl", getattr(dpg, "mvKey_Control", 17))
_KEY_SHIFT = getattr(dpg, "mvKey_ModShift", getattr(dpg, "mvKey_Shift", 16))

#: 共有テーブルに無い、アノテータ専用の追加バインド。
#: (action, key suffix, mods, display label, help)
_EXTRA_BINDINGS: Tuple[Tuple[str, str, Tuple[str, ...], str, str], ...] = (
    ("edit.redo", "Y", ("Ctrl",), "Ctrl+Y", "Redo the change you just undid"),
)

_TOOLS = ("select", "draw", "obb", "pan", "delete")

_TOOL_NAMES = {
    "select": "Select",
    "draw": "Draw Box",
    "obb": "Draw Rotated Box",
    "pan": "Pan",
    "delete": "Delete Box",
}

_TOOL_CHIPS = {
    "select": "Tool: Select - click a box to edit it, or press W to draw.",
    "draw": "Tool: Draw Box - drag on the image to create a box. Esc cancels.",
    "obb": "Tool: Draw Rotated Box - drag, then turn the yellow handle or press [ / ].",
    "pan": "Tool: Pan - drag to move the image. F fits it to the window.",
    "delete": "Tool: Delete Box - click a box to remove it. Esc stops.",
}

_TOOL_CAPTIONS = {
    "draw": "DRAW BOX - Esc to cancel",
    "obb": "DRAW ROTATED BOX - Esc to cancel",
    "pan": "PAN - F to fit",
    "delete": "DELETE - Esc to stop",
}

_TOOL_TAGS = {
    "select": "tb_tool_select",
    "draw": "tb_tool_draw",
    "obb": "tb_tool_obb",
    "pan": "tb_tool_pan",
    "delete": "tb_tool_delete",
}


def _track_color(track_id: int, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Shared track palette (theme.TRACK_PALETTE), indexed ``tid % 20``."""
    return theme.track_color(track_id, alpha)


def _palette_color(index: int, alpha: int = 255) -> Tuple[int, int, int, int]:
    r, g, b = _TRACK_PALETTE[int(index) % len(_TRACK_PALETTE)]
    return (r, g, b, alpha)


# ─── tkinter file-dialog helpers ─────────────────────────────────────────────

def _tk_open_file(title: str, filetypes: list,
                  initialdir: str = "") -> Optional[Path]:
    """Show a native OS open-file dialog (tkinter). Returns Path or None."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    kwargs = {"title": title, "filetypes": filetypes}
    if initialdir:
        kwargs["initialdir"] = initialdir
    path = filedialog.askopenfilename(**kwargs)
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


def _tk_ask_dir(title: str, initialdir: str = "") -> Optional[Path]:
    """Show a native OS directory picker (tkinter). Returns Path or None."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    kwargs = {"title": title}
    if initialdir:
        kwargs["initialdir"] = initialdir
    path = filedialog.askdirectory(**kwargs)
    root.destroy()
    return Path(path) if path else None


def _repo_root() -> Path:
    """Repository root, so child processes can import ``htrtdetr`` from ``src``."""
    try:
        return auto_label.repo_root_for(Path(__file__))
    except Exception:
        return Path(__file__).resolve().parents[2]
