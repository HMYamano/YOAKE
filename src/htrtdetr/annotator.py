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
import math
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import dearpygui.dearpygui as dpg
except ImportError:
    dpg = None

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np

try:  # GUI 補助モジュール (相対 import で失敗したら絶対 import)
    from .gui import auto_label, click_segment
    from .gui import sidecar as sidecar_mod  # noqa: F401  (re-export: サイドカーの起動は gui.sam_session に移したが、テストと外部コードが htrtdetr.annotator.sidecar_mod をmonkeypatch する)
    from .gui import canvas_view, file_dialogs, history as history_mod, sam_session
    from .gui.review_mode import ReviewModeMixin
    from .gui import video_controller
    from .gui.video_controller import VideoControllerMixin
    from .gui.canvas_view import CanvasViewMixin
    from .gui.history import EditHistoryMixin
    from .gui.sam_session import SamSessionMixin
    from .gui.auto_label import TRACKER_LABELS, make_tracker

    from .gui import activity as activity_mod
    from .gui import chrome
    from .gui.final_operations import FinalPassMixin
    from .gui.context_loop import ContextLoopMixin
    from .gui.keyframe_mode import KeyframeModeMixin
    from .gui.round_work import RoundWorkMixin
    from .gui.train_operations import TrainRoundMixin

    if dpg is not None:
        from .gui import menubar
    if dpg is not None:
        from .gui import inspector as inspector_mod
    if dpg is not None:
        from .gui import review as review_mod
    if dpg is not None:
        from .gui import settings as settings_mod
    from .gui import shortcuts

    if dpg is not None:
        from .gui import smart_annotate as smart_annotate_mod
    if dpg is not None:
        from .gui import final_pass as final_pass_mod
    if dpg is not None:
        from .gui import release_export as release_export_mod
    if dpg is not None:
        from .gui import train_round as train_round_mod
    if dpg is not None:
        from .gui import theme
    if dpg is not None:
        from .gui import timeline as timeline_mod
    if dpg is not None:
        from .gui import widgets
    from .annotation import intervals as intervals_mod
    from .annotation import propagation as propagation_mod
    from .annotation import quality as quality_mod
    from .annotation import review_queue as review_queue_mod
    from .annotation import round as round_mod
    from .annotation import sampling as sampling_mod
    from .annotation import tracklet as tracklet_mod
except ImportError:  # スクリプト直接実行時のフォールバック
    from htrtdetr.gui import auto_label, click_segment
    from htrtdetr.gui import sidecar as sidecar_mod  # noqa: F401  (re-export: サイドカーの起動は gui.sam_session に移したが、テストと外部コードが htrtdetr.annotator.sidecar_mod をmonkeypatch する)
    from htrtdetr.gui import canvas_view, file_dialogs, history as history_mod, sam_session
    from htrtdetr.gui.review_mode import ReviewModeMixin
    from htrtdetr.gui import video_controller
    from htrtdetr.gui.video_controller import VideoControllerMixin
    from htrtdetr.gui.canvas_view import CanvasViewMixin
    from htrtdetr.gui.history import EditHistoryMixin
    from htrtdetr.gui.sam_session import SamSessionMixin
    from htrtdetr.gui.auto_label import TRACKER_LABELS, make_tracker

    from htrtdetr.gui import activity as activity_mod
    from htrtdetr.gui import chrome
    from htrtdetr.gui.final_operations import FinalPassMixin
    from htrtdetr.gui.context_loop import ContextLoopMixin
    from htrtdetr.gui.keyframe_mode import KeyframeModeMixin
    from htrtdetr.gui.round_work import RoundWorkMixin
    from htrtdetr.gui.train_operations import TrainRoundMixin

    if dpg is not None:
        from htrtdetr.gui import menubar
    if dpg is not None:
        from htrtdetr.gui import inspector as inspector_mod
    if dpg is not None:
        from htrtdetr.gui import review as review_mod
    if dpg is not None:
        from htrtdetr.gui import settings as settings_mod
    from htrtdetr.gui import shortcuts

    if dpg is not None:
        from htrtdetr.gui import smart_annotate as smart_annotate_mod
    if dpg is not None:
        from htrtdetr.gui import final_pass as final_pass_mod
    if dpg is not None:
        from htrtdetr.gui import release_export as release_export_mod
    if dpg is not None:
        from htrtdetr.gui import train_round as train_round_mod
    if dpg is not None:
        from htrtdetr.gui import theme
    if dpg is not None:
        from htrtdetr.gui import timeline as timeline_mod
    if dpg is not None:
        from htrtdetr.gui import widgets
    from htrtdetr.annotation import intervals as intervals_mod
    from htrtdetr.annotation import propagation as propagation_mod
    from htrtdetr.annotation import quality as quality_mod
    from htrtdetr.annotation import review_queue as review_queue_mod
    from htrtdetr.annotation import round as round_mod
    from htrtdetr.annotation import sampling as sampling_mod
    from htrtdetr.annotation import tracklet as tracklet_mod

# ─── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_ACTIONS = ["stationary", "walking", "grooming", "courtship", "aggression"]
_DEFAULT_CLASSES = ["fly"]

_TRACK_PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (67, 99, 216),
    (245, 130, 49),
    (145, 30, 180),
    (66, 212, 244),
    (240, 50, 230),
    (191, 239, 69),
    (250, 190, 212),
    (70, 153, 144),
    (220, 190, 255),
    (154, 99, 36),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 216, 177),
    (0, 0, 117),
    (169, 169, 169),
]

_MIN_BOX_PX = 5
#: 投機 encode の遅延。値は ``gui.sam_session`` に移したが、この名前で
#: import していたコードのために別名を残す。
_SAM_EMBED_DELAY = sam_session.SAM_EMBED_DELAY

#: Click to Box のバックエンド選択に出すツールチップ。
#: P17 (取り返しのつかない待ちは事前に値段を言う) に従い、SAM の初回コストを
#: 選ぶ前に書いておく。数値はこの機械での実測 (docs/CLICK_SEGMENT.md §6.4)。
_BACKEND_TIPS = {
    "local": (
        "OpenCV only. Runs here, in milliseconds, and needs nothing installed. "
        "This is the default and the fallback for everything else."
    ),
    "sam": (
        "A SAM model in a helper process, started on demand.\n"
        "The helper loads once (tens of seconds), then the first click on each "
        "frame waits for that frame to be encoded - about 1.6 s per frame on "
        "this machine's CPU. Frames you sit on are encoded before you click.\n"
        "Needs a Python with PyTorch and a SAM package; if none is found this "
        "stays on Local and says why."
    ),
    "model": (
        "Your own trained detector's output, read from a predictions.json - no "
        "inference is run. A click takes the detection under it.\n"
        "Once a model has been trained on this species this is the most "
        "accurate fit, because it was trained for exactly this.\n"
        "Load a file with File > Load Predictions > As boxes for Click animal..."
    ),
}

#: Canvas modes that edit the selected box by dragging it. One undo step each,
#: pushed at release and only if the box actually changed (P7).
_DRAG_EDIT_MODES = ("move", "resize", "rotate")

_HANDLE_R = 7  # resize-handle radius in canvas pixels
_ROT_HANDLE_OFFSET = 26  # rotation handle distance from the box edge (canvas px)
#: フレームテクスチャのタグ。値は ``gui.video_controller`` に移した。
_FRAME_TEX = video_controller.FRAME_TEX
_SIDEBAR_W = 280
_TIMELINE_H = 26  # シークバー (drawlist) の高さ
_TIMELINE_PAD = 10  # シークバー左右の余白 (px)
_BOTTOM_H = 160  # キャンバス下に確保する高さ (シークバー + ナビ + status)

#: Rail width and button width. They describe the rail, so they live with it
#: in :mod:`gui.chrome`; these names are kept because ``_canvas_size`` and the
#: layout tests read them from here.
_RAIL_W = chrome.RAIL_W
_RAIL_BTN_W = chrome.RAIL_BTN_W

#: Right column width. 280 rather than 320: the Inspector shows one context at
#: a time now, so it needs room for a label and a control and nothing else, and
#: every pixel it gives back goes to the image.
_RIGHT_W = 280

#: Height of the menubar plus the top bar, above the canvas.
#:
#: Measured, not guessed: at 1280x820 the canvas starts at y=83. It used to be
#: assumed to be 42, which is why the transport controls and the status line
#: ended up **below the bottom of the window** at the default viewport size -
#: First/Prev/Next/Last were simply unreachable until the user resized.
_TOP_H = 114

#: What sits under the canvas, besides the timeline itself: the transport row,
#: the status row, and dearpygui's spacing between them. Added to
#: ``Timeline.height`` rather than hard-coded, so adding a timeline lane
#: reflows the canvas instead of pushing the controls off screen.
_UNDER_TIMELINE_H = 76

#: 逐次デコードで grab に切り替える上限。値は ``gui.video_controller``
#: に移した (実際に使うのはそちら)。
_SEQ_SKIP_MAX = video_controller.SEQ_SKIP_MAX

#: Undo depth, per frame. Lives beside the code that applies it
#: (``gui.history``); this name is kept for backward compatibility.
_UNDO_DEPTH = history_mod.UNDO_DEPTH

#: Autosave debounce. The value now lives beside the code that applies it
#: (``gui.history``); this name is kept for anything that imported it from
#: ``htrtdetr.annotator``.
_AUTOSAVE_DEBOUNCE = history_mod.AUTOSAVE_DEBOUNCE

#: Zoom / pan limits. The values now live beside the code that applies them
#: (``gui.canvas_view``); these names are kept so anything that imported them
#: from ``htrtdetr.annotator`` keeps working.
_ZOOM_MIN, _ZOOM_MAX = canvas_view.ZOOM_MIN, canvas_view.ZOOM_MAX
_PAN_KEEP = canvas_view.PAN_KEEP

#: The six tools, in rail order: ``(tool, full name, rail word, action)``.
#: Defined in :mod:`gui.chrome`, which builds the rail out of it, and
#: re-exported here because ``_refresh_tool_ui`` and the layout tests have
#: always read ``annotator._TOOLS``.
_TOOLS: Tuple[Tuple[str, str, str, str], ...] = chrome.TOOLS

#: One sentence per tool, shown in the canvas chip. Always names the *next*
#: action, never just the state (docs/GUI_DESIGN.md §0.3.4).
_TOOL_CHIP: Dict[str, str] = {
    "select": "Select - click a box to edit it, or press W to draw one.",
    "draw": "Draw Box - drag on the image to create a box. Esc cancels.",
    "obb": "Draw Rotated Box - drag, then turn the handle or press [ / ].",
    "click": "Click animal - click one animal; a box is fitted and selected.",
    "pan": "Pan - drag to move the image. F fits it to the window.",
    "delete": "Delete Box - click a box to remove it. Esc stops.",
}

# dearpygui 2.x では mvKey_Control が廃止され mvKey_ModCtrl になった (1.x フォールバック付き)
_KEY_CTRL = getattr(dpg, "mvKey_ModCtrl", getattr(dpg, "mvKey_Control", 17))


#: 4 隅の算出は幾何なので ``gui.canvas_view`` に置いた。レビュー画面と
#: 通常キャンバスの両方が使うため、描画経路ごとに書き分けない。
_corners_of = canvas_view.corners_of


def _repo_root() -> Path:
    """リポジトリのルート。子プロセスの cwd と PYTHONPATH の基準になる。

    実装は ``gui.sam_session.repo_root`` に一本化してある (サイドカーの起動が
    この値を最も必要とするため)。2 か所で別々に数え直すと、片方だけ階層が
    ずれても気付けない。
    """
    return sam_session.repo_root()


# ─── OpenCV tracker backend ───────────────────────────────────────────────────
# トラッカー実装は OpenCV のビルドによって置き場所が違い、contrib 無しのビルドには
# CSRT/KCF が同梱されていない。無い API を直接呼ぶと AttributeError で落ちるので、
# 検出は gui.auto_label の実装 (csrt -> csrt_legacy -> mil) をそのまま使う。
_TRACKER_FACTORY, _TRACKER_BACKEND = make_tracker()
_TRACKER_NAME = TRACKER_LABELS.get(_TRACKER_BACKEND, _TRACKER_BACKEND)
_TRACKER_HINT = (
    "This OpenCV build has no box tracker. Install CSRT with:  " "pip install opencv-contrib-python"
)


def _track_color(track_id: int, alpha: int = 255) -> Tuple[int, int, int, int]:
    if track_id < 0:
        return (140, 140, 140, alpha)
    r, g, b = _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]
    return (r, g, b, alpha)


# ─── tkinter file-dialog helpers ─────────────────────────────────────────────


# ダイアログ本体は ``gui.file_dialogs`` へ移した (gui 側のモジュールが
# ``from ..annotator import _tk_open_file`` と逆向きに import していたため)。
# 旧名は外部コードのために残すが、**呼び出し時に解決する薄いラッパ**にする。
#
# ``_tk_open_file = file_dialogs.open_file`` と束縛してしまうと、
# ``file_dialogs.open_file`` を差し替えてもこちらは元の関数を指したままになる。
# OS の modal ダイアログはテストを無言でハングさせるので、差し替え口は
# ``gui.file_dialogs`` の 1 か所に揃えておく (tests/conftest.py の安全網もそこを見る)。

def _tk_open_file(title: str, filetypes: list) -> Optional[Path]:
    return file_dialogs.open_file(title, filetypes)


def _tk_save_file(
    title: str, filetypes: list, initial_file: str = "annotations.json"
) -> Optional[Path]:
    return file_dialogs.save_file(title, filetypes, initial_file)


def _tk_ask_dir(title: str) -> Optional[Path]:
    return file_dialogs.ask_dir(title)


# ─── Data Model ───────────────────────────────────────────────────────────────

from htrtdetr.annotation.objects import AnnObject
from htrtdetr.annotation.objects import _parse_obb as _parse_obb
from htrtdetr.annotation.session import video_scoped
from htrtdetr.gui.workspace import WorkspaceMixin


@video_scoped
class AnnotationApp(
    activity_mod.ActivityMixin,
    CanvasViewMixin,
    ContextLoopMixin,
    EditHistoryMixin,
    KeyframeModeMixin,
    FinalPassMixin,
    ReviewModeMixin,
    RoundWorkMixin,
    SamSessionMixin,
    TrainRoundMixin,
    VideoControllerMixin,
    WorkspaceMixin,
):

    def __init__(self) -> None:
        if dpg is None or cv2 is None:
            raise RuntimeError(
                "The annotator needs DearPyGui and OpenCV. Install: pip install -e .[annotate]"
            )
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
        #: Frame index the capture last decoded, or ``None`` when its position
        #: is unknown. What lets :meth:`_read_frame_bgr` step without seeking.
        self._decoded_idx: Optional[int] = None
        #: Tag of the texture the frame is pushed into. An attribute and not
        #: the module constant, because a texture dearpygui refuses to delete
        #: has to be abandoned under a new tag rather than reused at the wrong
        #: size (:meth:`_install_video_capture`).
        self._frame_tex: str = _FRAME_TEX
        self._tex_serial: int = 1

        # ── annotation state ─────────────────────────────────────────
        self.annotations: Dict[int, List[AnnObject]] = {}
        self.next_track_id: int = 1
        self.active_track_id: int = 1

        # ── canvas interaction state ──────────────────────────────────
        self._mode: str = "idle"  # idle | draw | move | resize | scrub
        self._draw_start: Optional[Tuple[float, float]] = None
        self._draw_end: Optional[Tuple[float, float]] = None
        self._selected: Optional[AnnObject] = None
        self._drag_start: Optional[Tuple[float, float]] = None
        self._drag_bbox_orig: Optional[List[int]] = None
        self._drag_obb_orig: Optional[List[float]] = None
        self._resize_handle: Optional[str] = None
        self._right_click_obj: Optional[AnnObject] = None
        # OBB モード: 新規に描く箱を回転矩形として作るか (既定 False = 従来の AABB)
        self._obb_mode: bool = False
        # ツールモード (LabelImg 準拠):
        #   "select" = 既存ボックスの選択 / 移動 / リサイズ
        #   "draw"   = クリック&ドラッグで必ず新規ボックスを描く (W キー)
        #   "click"  = クリック 1 点から個体を推定して箱を作る (C キー)
        # 既存ボックスの中や上に重ねて描けるようにするためのモード分離。
        self._tool: str = "select"
        # Click to Box (docs/CLICK_SEGMENT.md)。"local" は追加依存ゼロで常に
        # 使えるので、ツール自体が使えなくなることはない。
        self._click_backend: str = "auto"
        self._click_auto_attempted = False
        self._click_start_pending = False
        self._click_radius_hint: Optional[float] = None
        # SAM サイドカー。使うと宣言されるまで probe も起動もしない。
        self._sam_client = None  # sidecar.SidecarClient
        self._sam_cap = None  # sidecar.SidecarCapability
        self._sam_jobs: Dict[int, dict] = {}  # request id -> 依頼時の文脈
        self._sam_pump_armed: bool = False
        self._sam_embed_due: Optional[float] = None
        self._sam_waiting: bool = False
        # 起動は非同期。押した操作は「用意ができたら」ここに積んで replay する。
        self._sam_starting: bool = False
        self._sam_report_failure = True
        self._sam_on_ready: List[Callable[[bool], None]] = []
        # _stop_sam が飛んだ後に古い起動が着地して勝手にヘルパーを据えるのを防ぐ。
        self._sam_start_token: int = 0
        # concept モードの候補 (auto_label.ReviewSession)。accepted になるまで
        # self.annotations には入らない = ディスクに届かない (GUI_DESIGN P1)。
        self._review = None
        # model バックエンド用の predictions.json (推論は走らせない)
        self._predictions: Dict[int, list] = {}
        self._predictions_path: Optional[Path] = None
        # LabelImg は 1 個描くと編集モードへ戻る。連続で描きたい場合のための粘着指定。
        self._sticky_draw: bool = False

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

        # ── view transform (zoom / pan) ──────────────────────────────
        # _scale = _fit_scale * _zoom, and _offset = centering + _pan. Because
        # _i2c / _c2i compose both, every hit test, handle test and draw call
        # is correct under zoom by construction — nothing else needs touching
        # (docs/GUI_DESIGN.md §0.3.5).
        self._fit_scale: float = 1.0
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._cursor_canvas: Optional[Tuple[float, float]] = None
        #: ``(press_x, press_y, pan_x_at_press, pan_y_at_press)`` while panning.
        self._pan_origin: Optional[Tuple[float, float, float, float]] = None
        self._middle_pan: Optional[Tuple[float, float, float, float]] = None

        # ── undo ─────────────────────────────────────────────────────
        # Per-frame snapshot stack. _pending_snap is taken on mouse press and
        # pushed on release **only if the object moved**, so a click that
        # selects does not leave an undo step behind (P7).
        self._undo: List[Dict[str, Any]] = []
        self._redo: List[Dict[str, Any]] = []
        self._pending_snap: Optional[Dict[str, Any]] = None
        self._moved: bool = False

        # ── dirty state / autosave ───────────────────────────────────
        self._dirty: bool = False
        self._save_due: Optional[float] = None
        self._last_saved: Optional[float] = None

        # ── the Annotation Copilot ───────────────────────────────────
        # Everything the AI proposes lives here and never in self.annotations
        # until a human accepts it. That is the one invariant the whole review
        # loop rests on (docs/GUI_DESIGN.md P1).
        self.quality_config = quality_mod.QualityConfig()
        self._embedding_info = None
        self._tracklets = tracklet_mod.TrackletIndex()
        self._queue = review_queue_mod.ReviewQueue()
        self._intervals = intervals_mod.IntervalTrack()
        self._selection = None  # sampling.Selection
        self._sel_tracklet = None  # the tracklet in the inspector
        self._sel_interval = None
        self._sel_issue = None
        self._sel_clip = None  # the sampled clip in the inspector
        self._analysis_thread = None
        self._analysis_stop = False
        self._propagate_thread = None
        self._propagate_stop = False
        #: Inspector and Inbox hidden, so the image has the whole width. A
        #: long annotation session spends most of its time looking at pixels.
        self._panels_hidden = False
        self._ui_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        #: Set when the pending set changed and the copilot has not caught up.
        #: See the note above :meth:`rebuild_copilot`.
        self._copilot_dirty = False
        self._copilot_due = 0.0
        self._activity_polled = 0.0
        #: Which video the timeline's current model was assembled for. ``None``
        #: means "no model yet", which forces the full path. See
        #: :meth:`_render_timeline`.
        self._timeline_model_video = None

        #: Everything the app has said, kept so it can be read back (§ the
        #: Activity window). Built before any other view, because the views
        #: below are allowed to record into it while they are being made and
        #: a failure during construction is exactly the one worth keeping.
        self.activity_log = activity_mod.ActivityLog()
        self.activity = activity_mod.ActivityWindow(
            self.activity_log,
            running=self.running_jobs,
            project_root=lambda: self.project_dir,
            on_open_folder=self._open_folder,
        )

        self.inspector = inspector_mod.Inspector("insp", callbacks=self._inspector_callbacks())
        self.timeline = timeline_mod.Timeline(
            "timeline",
            on_seek=lambda f: self.goto_frame(f, light=True),
            on_issue=self._on_timeline_marker,
            on_clip=self._on_timeline_clip,
            on_sample=self._goto_sample,
        )
        self.inbox = review_mod.ReviewInbox(
            on_select=self.jump_to_issue,
            on_start_review=self.open_review_mode,
            on_accept_kind=self.accept_issue_kind,
            on_reanalyze=self.open_smart_annotate,
            on_next_chosen=self.goto_next_chosen_frame,
            on_train=lambda: self.open_train_round(),
            on_next_final=self.goto_next_final_stop,
            on_final_checked=self.mark_final_stop_checked,
            on_sign_off=self.sign_off_video,
        )
        self.review_mode = review_mod.ReviewMode(
            on_seek=lambda f: self.goto_frame(f, light=True),
            on_decision=self._on_review_decision,
            on_edit=self._on_review_edit,
            on_swap=self._on_review_swap,
            on_exit=self._on_review_exit,
        )
        self.settings = settings_mod.SettingsDialog(callbacks=self._settings_callbacks())
        #: The running active-learning round, or ``None``. One attribute for
        #: the whole feature: the plan, the worker, the child and the clock all
        #: hang off it, so "is a round going" has one answer.
        self._round = None
        self._round_pending_weights = ""
        self._round_pending_project = None
        self.round_dialog = train_round_mod.TrainRoundDialog(
            on_start=self.start_train_round,
            on_stop=self.stop_train_round,
            on_analyze=lambda: self.open_smart_annotate(preset="analyze"),
            on_adopt=self.adopt_pending_round_model,
            on_options=self.round_plan_sentence,
        )
        self.build_final_dialogs(final_pass_mod, release_export_mod)
        self.smart_dialog = smart_annotate_mod.SmartAnnotateDialog(
            on_analyze=self.start_analysis,
            on_cancel=self.stop_analysis,
            on_apply=self.apply_selection,
            on_review=lambda: self.review_next_project(),
            on_pick_model=lambda: self.open_settings("model"),
            on_scope=self.scope_summary,
        )

        self._ensure_project()
        dpg.create_context()
        self._setup_theme()

    # ═══════════════════════════════════════════════════════════════════
    # Theme
    # ═══════════════════════════════════════════════════════════════════

    def _setup_theme(self) -> None:
        """Adopt the suite's design tokens.

        This used to be sixteen RGBA literals that existed only here, so a box
        drawn in the annotator and the same box reviewed in the analysis panel
        were different colours. :mod:`htrtdetr.gui.theme` is the one place a
        colour is chosen for the whole suite, and it self-checks its own
        contrast ratios (``python -m htrtdetr.gui.theme --check``).
        """
        theme.setup()

    # ═══════════════════════════════════════════════════════════════════
    # Project Setup Screen
    # ═══════════════════════════════════════════════════════════════════

    #: Preferred size of the setup window. It is only a preference:
    #: :func:`widgets.place_dialog` shrinks it to fit a small viewport, and the
    #: form scrolls inside it while Create Project stays pinned at the bottom.
    _SETUP_W, _SETUP_H = 620, 700

    def _show_setup_screen(self) -> None:
        w, h = self._SETUP_W, self._SETUP_H

        if dpg.does_item_exist("setup_win"):
            dpg.delete_item("setup_win")

        with dpg.window(
            label="YOAKE - Project Setup",
            tag="setup_win",
            width=w,
            height=h,
            no_resize=False,
            no_close=True,
        ):
            # Two buttons start a project and both used to sit at the far end
            # of a form taller than the window. They are pinned now; only the
            # form above them scrolls.
            with widgets.dialog_body("setup_body", footer=96):
                dpg.add_text("New Project", color=theme.ACCENT)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_text(
                    "Create a project, define labels, and choose an optional folder for autosave.",
                    color=theme.TEXT_DIM,
                    wrap=w - 60,
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Project name")
                dpg.add_input_text(
                    tag="setup_proj_name",
                    default_value="my_project",
                    width=-1,
                    callback=self._on_setup_name_changed,
                )
                dpg.add_spacer(height=8)

                dpg.add_text("Class labels (comma-separated, for example: fly,worm)")
                dpg.add_input_text(
                    tag="setup_classes", default_value=", ".join(_DEFAULT_CLASSES), width=-1
                )
                dpg.add_spacer(height=8)

                # Behaviors are optional and they are the answer most likely to
                # change once the user has watched an hour of their own video,
                # so the field is folded away with a working default rather
                # than presented as a decision to make before frame 0. Classes
                # stay open: the class list is what every box is labelled with.
                with dpg.collapsing_header(
                    label="Behaviors (optional - defaults are fine for now)",
                    tag="setup_actions_fold",
                    default_open=False,
                ):
                    # One behavior per animal, or groups (gui/behavior_groups.py).
                    self._build_setup_behaviors(", ".join(_DEFAULT_ACTIONS), w - 80)
                dpg.add_spacer(height=8)

                # The labelling scheme: decided once, before the first box.
                self._build_setup_label_mode(w - 60)
                dpg.add_spacer(height=8)

                # ── where the project lives ───────────────────────────
                # A folder, not a file. Everything the study produces -
                # annotations, training data, analysis output, caches - is
                # written inside it, and every path it records is relative to
                # it, so the folder can be moved, copied or handed over whole.
                dpg.add_text("Project folder (created if it does not exist)")
                with dpg.group(horizontal=True):
                    dpg.add_input_text(
                        tag="setup_proj_dir",
                        default_value=str(self._setup_default_dir()),
                        width=-90,
                        callback=self._on_setup_dir_edited,
                    )
                    dpg.add_button(label="Browse", width=80, callback=self._on_setup_pick_proj_dir)
                dpg.add_text("", tag="setup_dir_preview", color=theme.TEXT_DIM, wrap=w - 60)
                dpg.add_checkbox(
                    label="Copy the videos into the project folder",
                    tag="setup_copy_media",
                    default_value=False,
                )
                # One line, not four. Leaving it off is safe and reversible -
                # Project > Copy Videos into Project Folder does it later - so
                # this does not need the paragraph a one-way door would.
                dpg.add_text(
                    "Off is fine: the project finds them where they are. You "
                    "can copy them in later from the Project menu.",
                    color=theme.TEXT_DIM,
                    wrap=w - 60,
                )
                dpg.add_text("", tag="setup_error", color=theme.DANGER, wrap=w - 60)
                dpg.add_spacer(height=12)
                dpg.add_separator()
                dpg.add_spacer(height=8)

                dpg.add_text("Videos can be added now or from Project Videos later.")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Add Files...", callback=lambda: self._setup_add_files())
                    dpg.add_button(label="Add Folder...", callback=lambda: self._setup_add_folder())
                    dpg.add_checkbox(label="Include subfolders", tag="setup_recursive")
                dpg.add_text("0 videos selected", tag="setup_video_count")

            widgets.dialog_footer()
            # The primary action of the first screen anyone ever sees, and it
            # was a bare dpg.add_button - so it rendered in the default grey,
            # indistinguishable from the "or load an existing file" button
            # under it. The accent fill is what the design tokens reserve for
            # exactly this (§1.5), and it is the only thing on the screen that
            # says which of the two buttons is the way forward.
            widgets.primary_button(
                "Create Project", self._on_new_project_confirmed, width=-1, height=34
            )
            widgets.secondary_button(
                "Or load an existing annotation JSON...",
                self._on_load_project_clicked,
                width=-1,
                height=28,
            )
        self._setup_dir_edited = False
        self._setup_dir_parent = None
        self._refresh_setup_dir_preview()
        widgets.place_dialog("setup_win", w, h)

    # ── the project folder, on the first screen anyone sees ──────────────
    def _setup_default_dir(self, name: str = "") -> Path:
        """Somewhere sensible by default, so a study is never left homeless.

        The folder used to be optional, and a project without one autosaved
        into a temporary recovery directory - which is exactly how annotations,
        runs and analysis output end up in three unrelated places.
        """
        from .annotation.layout import project_folder_name

        parent = getattr(self, "_setup_dir_parent", None) or Path.home() / "YOAKE"
        raw = name or (
            dpg.get_value("setup_proj_name") if dpg.does_item_exist("setup_proj_name") else ""
        )
        return Path(parent) / project_folder_name(raw or "my_project")

    def _on_setup_dir_edited(self, *_args) -> None:
        self._setup_dir_edited = True
        self._refresh_setup_dir_preview()

    def _on_setup_name_changed(self, *_args) -> None:
        if not getattr(self, "_setup_dir_edited", False) and dpg.does_item_exist("setup_proj_dir"):
            dpg.set_value("setup_proj_dir", str(self._setup_default_dir()))
        self._refresh_setup_dir_preview()

    def _refresh_setup_dir_preview(self) -> None:
        if not dpg.does_item_exist("setup_dir_preview"):
            return
        from .annotation.layout import FOLDERS, is_project_root

        raw = dpg.get_value("setup_proj_dir").strip()
        if not raw:
            dpg.set_value(
                "setup_dir_preview",
                "No folder: the project autosaves to a temporary recovery folder "
                "until you use Save As.",
            )
            return
        folders = ", ".join(f"{name}/" for _a, name, _d in FOLDERS)
        existing = "Opens the project already in this folder. " if is_project_root(raw) else ""
        dpg.set_value("setup_dir_preview", f"{existing}Creates {folders} and annotations.json here.")

    def _on_setup_pick_proj_dir(self) -> None:
        p = _tk_ask_dir(title="Select a folder to create the project in")
        if p is None or not dpg.does_item_exist("setup_proj_dir"):
            return
        # A folder that already holds a project is the one the user means;
        # anything else is the parent the new project folder goes into.
        from .annotation.layout import is_project_root

        if is_project_root(p):
            self._setup_dir_edited = True
            dpg.set_value("setup_proj_dir", str(p))
        else:
            self._setup_dir_parent = Path(p)
            self._setup_dir_edited = False
            dpg.set_value("setup_proj_dir", str(self._setup_default_dir()))
        self._refresh_setup_dir_preview()

    def _on_new_project_confirmed(self) -> None:
        from .annotation.layout import PROJECT_FILE, is_project_root

        raw_dir = dpg.get_value("setup_proj_dir").strip()
        # Creating a project on top of another one would overwrite a study with
        # an empty one. Offer to open it instead, on the screen they are on.
        if raw_dir and is_project_root(raw_dir):
            if dpg.does_item_exist("setup_error"):
                dpg.set_value(
                    "setup_error",
                    f"{raw_dir} already holds a project. Opening it instead of "
                    "creating an empty one.",
                )
            dpg.delete_item("setup_win")
            widgets.forget_dialog("setup_win")
            self._build_main_window()
            self.open_project(Path(raw_dir) / PROJECT_FILE)
            return

        vocab = self._read_setup_behaviors(_DEFAULT_ACTIONS)
        if vocab is None:
            return
        self._preserve_project_to_recovery()
        from .annotation.project import ProjectSession
        from .annotation.session import VideoSession

        self.project = ProjectSession()
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self._video_session = VideoSession()
        self.project_name = dpg.get_value("setup_proj_name").strip() or "untitled"
        raw_cls = dpg.get_value("setup_classes")
        self.class_names = [s.strip() for s in raw_cls.split(",") if s.strip()]
        if not self.class_names:
            self.class_names = ["fly"]
        self._apply_setup_behaviors(vocab)
        self.project_dir = Path(raw_dir).expanduser() if raw_dir else None
        # "At video end" is not asked on this screen any more: it is a question
        # about finishing video 1 put to someone who has not opened it, and it
        # is in Settings > Project where it can be answered by someone who has.
        # The default stays what it was.
        self.project.settings["end_of_video"] = (
            "next_unfinished"
            if dpg.does_item_exist("setup_end")
            and dpg.get_value("setup_end") == "Open next unfinished video"
            else "stay"
        )
        self._apply_setup_label_mode(self.project.settings)
        copy_media = dpg.does_item_exist("setup_copy_media") and bool(
            dpg.get_value("setup_copy_media")
        )
        dpg.delete_item("setup_win")
        widgets.forget_dialog("setup_win")
        self._build_main_window()
        self._update_action_buttons()
        self._refresh_track_ui()
        self._sync_project()
        self._add_videos(getattr(self, "_setup_videos", []))
        self._setup_videos = []
        if self.project_dir is not None:
            self.project.save(self.project_dir / PROJECT_FILE)
            self._status(f"Project folder created: {self.project_dir}")
            if copy_media:
                self.collect_videos()
        self._refresh_workspace()
        if not self.project.settings.get("onboarding_done"):
            self.show_onboarding()
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

        with dpg.window(
            tag="main_win",
            label="YOAKE Annotation Tool",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
        ):
            self._build_menubar()
            self._build_topbar()

            # ── the four regions ────────────────────────────────────
            # Tool Rail | Canvas | Inspector, over one Timeline. The rail is
            # muscle memory, the canvas is where the eye is, the inspector
            # answers "what did I select", and the timeline answers "where in
            # the video is anything" - which is the question the old
            # single-lane seek bar could not answer at all.
            with dpg.group(horizontal=True):
                self._build_toolrail()
                self._build_canvas_panel()
                self._build_right_column()

            self._build_navbar()

        dpg.set_primary_window("main_win", True)
        self.settings.build(
            tracker_label="" if _TRACKER_FACTORY is None else _TRACKER_NAME,
            tracker_hint=_TRACKER_HINT,
            backend_tips=_BACKEND_TIPS,
        )
        self.settings.load(
            config=self.quality_config,
            project_dir=self.project_dir,
            auto_track=self._auto_track,
            sticky_draw=self._sticky_draw,
            use_track_ids=self.use_track_ids,
        )
        self._refresh_backend_ui()
        self._refresh_labels_display()
        self.smart_dialog.build()
        self.round_dialog.build()
        # Built hidden, with the rest of the dialogs, rather than on first
        # press: the point of the pill is that the log is one click away, and
        # a window that has to be constructed first is one click and a stutter.
        self.activity.build()
        self._refresh_activity_pill()
        self.review_mode.build(
            width=max(dpg.get_viewport_width() - 120, 700),
            height=max(dpg.get_viewport_height() - 90, 520),
        )
        self._refresh_model_ui()
        self._refresh_propagation_ui()
        self._refresh_history_ui()
        self._refresh_workspace()
        self._build_key_handlers()
        self._build_mouse_handlers()

        # texture registry (created once)
        if not dpg.does_item_exist("tex_reg"):
            with dpg.texture_registry(tag="tex_reg", show=False):
                pass

    # ─── Menu bar ────────────────────────────────────────────────────

    def _build_menubar(self) -> None:
        """The five menus. Built in :mod:`gui.menubar`, which owns the list."""
        menubar.build(self)

    # ─── Canvas panel ────────────────────────────────────────────────

    def _bottom_height(self) -> int:
        """Pixels the timeline and the transport rows need, together.

        Derived from the timeline's own height so a new lane reflows the canvas
        rather than pushing the controls out of the window.
        """
        return self.timeline.height + _UNDER_TIMELINE_H

    def _canvas_size(self) -> Tuple[int, int]:
        """``(width, height)`` of the canvas for the current viewport.

        One function, called by the builder, the resize handler and the review
        canvas. They used to carry the same arithmetic twice, and changing only
        one of them clips the canvas (``docs/GUI_DESIGN.md`` §0.2.6).

        The right column collapses to nothing when the Inspector is hidden, so
        ``Hide panels`` really does give the whole width to the image.
        """
        vw = dpg.get_viewport_client_width()
        vh = dpg.get_viewport_client_height()
        right = 0 if self._panels_hidden else _RIGHT_W
        # Client dimensions exclude the Windows title bar and window borders.
        # Leave room for two 12px window paddings and two 8px column gaps.
        return (max(vw - _RAIL_W - right - 40, 360), max(vh - _TOP_H - self._bottom_height(), 240))

    def _build_canvas_panel(self) -> None:
        cw, ch = self._canvas_size()
        with dpg.child_window(
            tag="canvas_win", width=cw, height=ch, no_scrollbar=True, border=False
        ):
            dpg.add_drawlist(tag="canvas", width=cw, height=ch)

    # ─── Top bar and tool rail ───────────────────────────────────────

    def _build_topbar(self) -> None:
        """Two rows: where am I, and the loop. Built in :mod:`gui.chrome`."""
        chrome.topbar(self)

    def _build_toolrail(self) -> None:
        """Tools, assistance, view. Built in :mod:`gui.chrome`."""
        chrome.toolrail(self)

    # ─── Right column ────────────────────────────────────────────────

    def _build_right_column(self) -> None:
        """Inspector and Review Inbox share one column; one is visible at a time.

        Below them, the project-level panels the old sidebar kept permanently
        open, now collapsed: they are settings, not properties of a selection,
        and a setting that is always on screen is a setting that competes with
        the work (§4, §27).
        """
        _cw, ch = self._canvas_size()
        with dpg.child_window(
            tag="right_col", width=_RIGHT_W, height=ch, border=True, no_scrollbar=False
        ):
            self.inspector.build(parent="right_col", height=-1, width=_RIGHT_W - 24)
            self.inbox.build(parent="right_col", height=-1, width=_RIGHT_W - 24)
            self._show_item(self.inbox.tag, False)

    # ─── Sidebar ─────────────────────────────────────────────────────

    # ─── The legacy sidebar is gone ──────────────────────────────────
    #
    # It was eleven always-open panels in a 280 px column. Everything in it now
    # lives where it belongs, and the tags moved with the widgets so no callback
    # or refresh helper had to change:
    #
    #   Frame Info, Active Track, Objects in Frame, per-frame candidates,
    #   Find all in frame          -> Inspector, "nothing selected" context
    #   Selected Object, Quick Actions, Box Shape conversions
    #                              -> Inspector, "box selected" context
    #   Select / Draw / Click tool buttons, Keep drawing, zoom
    #                              -> Tool Rail
    #   Quick Start                -> the canvas empty state
    #   Labels, Auto-Track, Click to Box backend, Project folder
    #                              -> Settings dialog (built hidden at startup,
    #                                 so its tags always exist)
    #
    # ``_legacy_tag_map`` below records the moves, and
    # ``tests/test_annotator_wiring.py`` asserts every one of them still
    # resolves after the window is built.

    #: ``old tag -> where it lives now``. Kept as data so the migration is
    #: checkable rather than a claim in a comment.
    _LEGACY_TAGS: Dict[str, str] = {
        "active_tid": "inspector: nothing",
        "active_track_color_label": "inspector: nothing",
        "obj_listbox": "inspector: nothing",
        "candidate_count": "inspector: nothing",
        "accept_all_btn": "inspector: nothing",
        "discard_all_btn": "inspector: nothing",
        # ``concept_text`` was here until the find-all button started naming
        # the class itself. The widget is gone, so the promise is too: this
        # map records moves, and a tag that leads nowhere is not a move.
        "find_all_btn": "inspector: nothing",
        "action_buttons_group": "inspector: box",
        "auto_track_cb": "settings: tracking",
        "sticky_draw_cb": "settings: assistance",
        "classes_display": "settings: labels",
        "actions_display": "settings: labels",
        "project_dir_text": "settings: project",
        "tool_mode_text": "topbar",
    }

    # _update_action_buttons lives in gui/behavior_groups.py (groups get headings).

    # ─── Navigation bar ──────────────────────────────────────────────

    def _build_navbar(self) -> None:
        # ── the multi-lane timeline ──────────────────────────────────
        # Four lanes over one frame axis: annotation status, review markers,
        # behavior intervals, and Smart Annotate samples. It replaces a
        # single-lane seek bar that could only answer "have I been here".
        self.timeline.resize(self._timeline_width())
        self.timeline.build(parent="main_win")

        # Transport, timeline summary and status share one row. They used to be
        # three stacked rows costing 76 px of image; none of them needs a line
        # to itself, and the summary belongs next to the timeline it describes.
        with dpg.group(horizontal=True, tag="navbar"):
            # Widths through widgets.fit_width, not hand-picked numbers: at the
            # theme's FramePadding a 52 px button holds 32 px of text, so
            # "First" was drawn as "Firs".
            dpg.add_button(
                label="First",
                width=widgets.fit_width("First", 52),
                callback=lambda: self.goto_frame(0),
            )
            dpg.add_button(
                label="Prev",
                width=widgets.fit_width("Prev", 48),
                callback=lambda: self.goto_frame(self.current_frame_idx - 1),
            )
            widgets.attach_tooltip(dpg.last_item(), shortcuts.hint_for("nav.prev"))
            dpg.add_text("Frame:")
            dpg.add_input_int(
                tag="frame_input",
                default_value=0,
                width=140,
                min_value=0,
                max_value=0,
                callback=lambda s, a: self.goto_frame(a),
                on_enter=True,
            )
            dpg.add_text("/ 0", tag="total_frames_text")
            dpg.add_button(
                label="Next",
                width=widgets.fit_width("Next", 48),
                callback=lambda: self.goto_frame(self.current_frame_idx + 1),
            )
            widgets.attach_tooltip(dpg.last_item(), shortcuts.hint_for("nav.next"))
            dpg.add_button(
                label="Last",
                width=widgets.fit_width("Last", 52),
                callback=lambda: self.goto_frame(self.num_frames - 1),
            )
            dpg.add_spacer(width=10)
            # "Copy from prev", not "Propagate": the rail's Continue button
            # and the Review menu's clip entries are the other two ways to
            # carry a box forward, and between them this one concept had six
            # names. The word that survives says what the press does.
            widgets.secondary_button(
                shortcuts.with_key("Copy from prev", "edit.propagate"),
                lambda *_: self.propagate_from_prev(),
                width=150,
                height=26,
                tag="nav_propagate",
            )
            widgets.attach_tooltip("nav_propagate", shortcuts.help_for("edit.propagate"))
            dpg.add_spacer(width=14)
            dpg.add_text("", tag="timeline_summary", color=theme.TEXT_FAINT)

        with dpg.group(horizontal=True, tag="statusbar"):
            dpg.add_text("", tag="status_text", color=theme.ACCENT)

    # ─── Sequence (seek) bar ─────────────────────────────────────────

    def _timeline_width(self) -> int:
        """シークバーの実幅 (px)。生成前は viewport 幅から見積もる。

        ``get_viewport_client_width`` を使う。``get_viewport_width`` は
        ウィンドウ枠込みの幅なので、その値で作ると枠のぶん (このマシンでは 16px)
        だけ右へはみ出す。リサイズ経路は最初から client 幅を使っていたので、
        「起動直後だけ数 px 切れている」という気づきにくい形で出ていた。
        """
        if dpg.does_item_exist("timeline"):
            try:
                w, _h = dpg.get_item_rect_size("timeline")
                if w > 1:
                    return int(w)
            except (KeyError, SystemError, TypeError):
                pass
        return max(widgets.viewport_size()[0] - 24, 400)

    # The three mapping helpers now delegate to the Timeline, which owns the
    # axis. They are kept as methods because the canvas code, the scrub
    # handlers and the tests all call them by name.
    def _frame_to_x(self, idx: int) -> float:
        return self.timeline.frame_to_x(idx)

    def _x_to_frame(self, x: float) -> int:
        return self.timeline.x_to_frame(x)

    def _timeline_mouse_x(self) -> float:
        """マウス x をシークバーローカル座標に変換する。"""
        pos = self.timeline.mouse_pos()
        if pos is not None:
            return pos[0]
        mx, _my = dpg.get_mouse_pos(local=False)
        return float(mx)

    def _render_timeline(self, *, playhead_only: bool = False) -> None:
        """Rebuild the timeline model from live state and redraw it.

        Called after every mutation and every frame change. Assembling the
        model here rather than inside the renderer keeps the renderer a pure
        function of data, which is what lets it be unit-tested with no
        viewport.

        ``playhead_only`` is the frame-change path. Stepping a frame changes
        exactly one field of the model - where the playhead is - and nothing
        else it draws: not the annotations, not the pending set, not the
        review queue, not the samples. Reassembling all of it cost 6 ms on an
        hour of video with a detector run in it, on a budget of 16 ms, for a
        result identical but for one integer. So the frame-change path moves
        the integer and redraws; every mutation still goes through the full
        rebuild, which is what keeps the cached model honest.

        The reuse is keyed on **which video the model was built for**, not on
        its length. This study is twenty recordings of one hour each: a frame
        count is not an identity here, and "same number of frames" would have
        let a freshly opened video be drawn with the previous one's
        annotations until something else happened to force a rebuild.
        """
        if not self.timeline.alive():
            return
        here = getattr(self._video_session, "video_id", None)
        if playhead_only and self._timeline_model_video == here:
            self.timeline.model.current = int(self.current_frame_idx)
            self.timeline.render_playhead()
            return
        self._timeline_model_video = here
        self.timeline.set_model(
            timeline_mod.build_model(
                num_frames=self.num_frames,
                current=self.current_frame_idx,
                fps=self.fps,
                annotations=self.annotations,
                candidates=self._candidate_map(),
                review_items=list(self._queue.items),
                behavior_intervals=self._behavior_lane(),
                selection=self._selection,
                confident_score=self.quality_config.confident_score,
            )
        )
        self.timeline.render()
        if dpg.does_item_exist("timeline_summary"):
            dpg.set_value("timeline_summary", self.timeline.model.summary())

    def _candidate_map(self) -> Dict[int, list]:
        """``{frame: [Candidate, ...]}`` for whatever is pending right now.

        The review session is the only holder of candidates; this reads it
        without copying the objects, so the timeline and the detectors see
        exactly what the canvas draws (P2: one predicate for draw and for
        everything that reasons about what is drawn).
        """
        session = self._review
        if session is None:
            return {}
        return {frame: session.candidates(frame) for frame in session.frames}

    def _begin_scrub(self) -> bool:
        """シークバー上で押されたら scrub モードに入る。処理したら True。

        The timeline gets first refusal: a press on a review marker opens that
        item instead of seeking, because a 5-pixel marker that merely moves the
        playhead near the problem reads as a missed click.
        """
        if self.cap is None or self._mode != "idle":
            return False
        if not self.timeline.hovered():
            return False
        if self.timeline.on_press():
            # A marker hit is handled and finished; a seek leaves the timeline
            # scrubbing and we mirror that into our own mode machine.
            if self.timeline.scrubbing:
                self._mode = "scrub"
                return True
            return True
        return False

    def _scrub_to_mouse(self) -> None:
        idx = self._x_to_frame(self._timeline_mouse_x())
        if idx == self.current_frame_idx:
            self._render_timeline()
            return
        # ドラッグ中は自動トラッキングとサイドバー更新を止めて軽くする
        self.goto_frame(idx, light=True)

    def _end_scrub(self) -> None:
        self._mode = "idle"
        self.timeline.on_release()
        self._refresh_sidebar()
        self._render_canvas()
        self._render_timeline()

    # ─── Mouse handlers ──────────────────────────────────────────────

    def _build_mouse_handlers(self) -> None:
        if dpg.does_item_exist("mouse_handlers"):
            dpg.delete_item("mouse_handlers")
        # dearpygui 2.x に add_mouse_press_handler は無い。押下相当は
        # add_mouse_click_handler (押した瞬間に発火) を使う。
        with dpg.handler_registry(tag="mouse_handlers"):
            dpg.add_mouse_click_handler(button=0, callback=self._on_mouse_press)
            dpg.add_mouse_drag_handler(button=0, threshold=1.0, callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=0, callback=self._on_mouse_release)
            dpg.add_mouse_click_handler(button=1, callback=self._on_right_click)
            # Middle-drag pans from *every* tool, always. The Pan tool is the
            # discoverable form of a gesture that never goes away.
            dpg.add_mouse_click_handler(button=2, callback=self._on_middle_press)
            dpg.add_mouse_drag_handler(button=2, threshold=1.0, callback=self._on_middle_drag)
            dpg.add_mouse_release_handler(button=2, callback=self._on_middle_release)

    # ─── Key handlers ────────────────────────────────────────────────

    _ACTION_SCOPES = ("annotator", "review")

    def _actions(self) -> Dict[str, Callable[[], None]]:
        """Every keymap action this window implements, by name.

        One table, consumed by both the key handlers and the Help window, so a
        binding that exists in :mod:`htrtdetr.gui.shortcuts` and has no
        implementation here shows up as a missing key in the table rather than
        as a key that silently does nothing.
        """
        return {
            # tools
            "tool.select": lambda: self.set_tool("select"),
            "tool.draw_box": self.toggle_draw_tool,
            "tool.draw_rotated": lambda: self.set_tool("obb"),
            "tool.click_box": self.toggle_click_tool,
            "tool.pan": lambda: self.set_tool("pan"),
            "tool.delete": lambda: self.set_tool("delete"),
            "tool.cancel": self._escape,
            # view
            "view.fit": self.fit_view,
            "view.zoom_in": lambda: self.zoom_by(1.25),
            "view.zoom_out": lambda: self.zoom_by(0.8),
            "view.activity": self.toggle_activity,
            "view.context_loop": self.toggle_context_loop,
            # navigation
            "nav.prev": lambda: self.goto_frame(self.current_frame_idx - 1),
            "nav.next": lambda: self.goto_frame(self.current_frame_idx + 1),
            "nav.first": lambda: self.goto_frame(0),
            "nav.last": lambda: self.goto_frame(self.num_frames - 1),
            "review.next_pending_frame": self._jump_next_pending,
            "review.prev_pending_frame": self._jump_prev_pending,
            # review
            "review.cycle": self._cycle_selection,
            "review.accept_all": self._accept_all_here,
            "review.mark_checked": self.mark_frame_checked,
            "review.discard_all": self._discard_all_here,
            "auto.track_forward": self._track_forward_selected,
            "auto.annotate": lambda: self.open_smart_annotate(),
            "auto.toggle_autotrack": self._toggle_auto_track_key,
            # edit
            "edit.delete": self.delete_selected,
            "edit.propagate": self.propagate_from_prev,
            "edit.undo": self.undo,
            "edit.redo": self.redo,
            "edit.new_track": self.new_track,
            "obb.rotate_ccw": lambda: self.rotate_selected(-5),
            "obb.rotate_cw": lambda: self.rotate_selected(5),
            # file
            "file.save": self.save_annotation,
            "file.save_as": self._save_as_dialog,
            "file.open_video": self._open_video_dialog,
            "help.shortcuts": self._show_help,
            **{f"action.set_{i}": (lambda n=i: self.set_action(n)) for i in range(10)},
        }

    def _typing(self) -> bool:
        """Is the user typing into a field right now?

        ``is_item_active``, never ``is_item_focused``: the first input widget
        in a window auto-focuses, so focus alone is sticky and useless. Without
        this guard, typing a track id into ``sel_tid`` fires A, N, R and
        **Delete**, and the last of those deletes the selected box - the worst
        live bug the annotator had (``docs/GUI_DESIGN.md`` §0.3.8).
        """
        focused = dpg.get_focused_item()
        if not focused:
            return False
        try:
            kind = dpg.get_item_type(focused)
        except (SystemError, TypeError):
            return False
        return any(kind.endswith(k) for k in shortcuts.EDITING_TYPES) and dpg.is_item_active(
            focused
        )

    def _wrap(self, binding, action: Callable[[], None]) -> Callable:
        """Guard one binding: typing, modifiers, and the Review Mode override."""

        def callback(_sender=None, _app_data=None) -> None:
            if binding.action not in shortcuts.ALWAYS_ALLOWED and self._typing():
                return
            if binding.mods and not shortcuts.mods_held(dpg, binding):
                return
            if not binding.mods and dpg.is_key_down(_KEY_CTRL):
                return  # do not fire a bare W while Ctrl is held
            # Review Mode is modal: while it is up, only its own scope acts,
            # so Delete cannot reach a box the user cannot even see.
            if self.review_mode.active:
                if binding.scope == "review":
                    self.review_mode.handle_action(binding.action)
                return
            if binding.scope != "annotator":
                return
            try:
                action()
            except Exception as exc:  # a key must never kill the app
                self._status(f"{binding.label} failed: {type(exc).__name__}: {exc}")

        return callback

    def _build_key_handlers(self) -> None:
        """Register every binding from the one keymap.

        The annotator used to bind fifteen keys inline with no typing guard,
        and its own docstring advertised a ``T`` binding that was never
        registered. Generating the handlers from
        :data:`htrtdetr.gui.shortcuts.BINDINGS` makes that class of drift
        impossible: the handler, the tooltip, the rail label and the help table
        all come from the same row.
        """
        if dpg.does_item_exist("key_handlers"):
            dpg.delete_item("key_handlers")

        actions = self._actions()
        missing: List[str] = []
        with dpg.handler_registry(tag="key_handlers"):
            for scope in self._ACTION_SCOPES:
                for binding in shortcuts.for_scope(scope):
                    action = actions.get(binding.action)
                    if action is None and scope == "annotator":
                        missing.append(binding.action)
                        continue
                    if action is None:
                        # Review-scope actions are handled by ReviewMode itself.
                        action = lambda b=binding: self.review_mode.handle_action(b.action)
                    try:
                        key = shortcuts.resolve(dpg, binding)
                    except AttributeError:
                        missing.append(f"{binding.action} ({binding.key})")
                        continue
                    dpg.add_key_press_handler(key, callback=self._wrap(binding, action))
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)

        if missing:
            # Loud, not silent: an unimplemented binding is a key that looks
            # documented and does nothing.
            self._status("Unbound actions: " + ", ".join(sorted(set(missing))))

    # ═══════════════════════════════════════════════════════════════════
    # Canvas Rendering
    # ═══════════════════════════════════════════════════════════════════

    def _render_canvas(self) -> None:
        if not dpg.does_item_exist("canvas"):
            return
        dpg.delete_item("canvas", children_only=True)
        self._recompute_scale()
        cw, ch = self._canvas_w, self._canvas_h

        if self._current_bgr is None:
            self._draw_empty_state(cw, ch)
            return

        # Frame image
        if dpg.does_item_exist(self._frame_tex):
            dw = int(self.frame_w * self._scale)
            dh = int(self.frame_h * self._scale)
            dpg.draw_image(
                self._frame_tex,
                pmin=(self._offset_x, self._offset_y),
                pmax=(self._offset_x + dw, self._offset_y + dh),
                parent="canvas",
            )

        # Annotated boxes
        for obj in self.annotations.get(self.current_frame_idx, []):
            self._draw_box(obj, selected=(obj is self._selected))

        # Pending candidates: dashed amber, never solid. A candidate must not be
        # mistakable for an annotation - it is not saved and may never be.
        for _index, cand in self.candidates_here():
            self._draw_candidate(cand)

        # Rubber-band while drawing.
        #
        # Nothing here may touch ``_mode``. This used to clear it whenever the
        # Pan tool was armed, and ``_drag_pan`` renders - so the first drag
        # event panned once, dropped the mode and the press origin, and every
        # later event returned at ``_mode == "idle"``. The Pan tool moved the
        # image by a single mouse delta and then went dead until the button was
        # released and pressed again. A renderer reads state; it never retires
        # it. ``_on_mouse_release`` owns that.
        if self._mode == "draw" and self._draw_start and self._draw_end:
            xs = sorted([self._draw_start[0], self._draw_end[0]])
            ys = sorted([self._draw_start[1], self._draw_end[1]])
            p1 = self._i2c(xs[0], ys[0])
            p2 = self._i2c(xs[1], ys[1])
            dpg.draw_rectangle(
                p1,
                p2,
                color=(255, 255, 255, 200),
                fill=(255, 255, 255, 30),
                thickness=1.5,
                parent="canvas",
            )

        # Last, so they sit over everything: the in-canvas tool feedback.
        self._draw_tool_overlay(cw, ch)
        if self.context_loop_active:
            self._draw_context_overlay(cw, ch)

    def _draw_empty_state(self, cw: int, ch: int) -> None:
        """Say what to do, not just that there is nothing here (§23).

        A blank canvas is indistinguishable from a broken one. This is the
        first screen of a new project, so it carries the whole no-model recipe
        in three numbered lines rather than a list of keys: the one question a
        first-time user has here is "where do I start", and the honest answer
        is not "frame 0" - it is "let it choose the frames".

        The wording matches the buttons exactly. It used to say *Project
        Videos*, and the button has read *Videos* since the top bar was built.
        """
        lines = [
            ("Open a video to start.", theme.TEXT, 18),
            ("", theme.TEXT, 8),
            (
                shortcuts.with_key("1.  Videos > Add Files / Add Folder", "file.open_video"),
                theme.TEXT_DIM,
                14,
            ),
            (
                shortcuts.with_key(
                    "2.  Pick frames to label - no trained model needed", "auto.annotate"
                ),
                theme.TEXT_DIM,
                14,
            ),
            (
                f"3.  {shortcuts.label_for('tool.click_box')} clicks an animal into a box, "
                f"{shortcuts.label_for('tool.draw_box')} draws one by hand",
                theme.TEXT_DIM,
                14,
            ),
        ]
        y = ch // 2 - 48
        for text, color, size in lines:
            if text:
                dpg.draw_text(
                    (max(24, cw // 2 - 190), y),
                    text,
                    color=theme.rgba(color),
                    size=size,
                    parent="canvas",
                )
            y += size + 8

    def _draw_tool_overlay(self, cw: int, ch: int) -> None:
        """Channels three and four of "which tool is armed" (§0.3.4).

        A coloured border just inside the canvas edge plus a caption, and a
        crosshair at the cursor. Both are drawn **inside the canvas**, because
        the user's eye is on the image and not on a rail 600 px away; the rail
        button's fill alone is a state you have to go and look for.

        Select pays nothing: the method returns on its first comparison.
        """
        if self._tool == "select":
            return
        danger = self._tool == "delete"
        color = theme.rgba(theme.DANGER if danger else theme.ACCENT, 210)
        dpg.draw_rectangle((2, 2), (cw - 2, ch - 2), color=color, thickness=2.0, parent="canvas")
        caption = {
            "draw": "DRAW BOX - Esc to cancel",
            "obb": "DRAW ROTATED BOX - Esc to cancel",
            "click": "CLICK TO BOX - Esc to leave",
            "pan": "PAN - F fits the frame",
            "delete": "DELETE - Esc to stop",
        }.get(self._tool, "")
        if caption:
            dpg.draw_text((12, 10), caption, color=color, size=13, parent="canvas")

        cursor = self._cursor_canvas
        if cursor is not None and self._tool in ("draw", "obb", "delete"):
            guide = theme.rgba(theme.DANGER if danger else theme.ACCENT, 90)
            dpg.draw_line(
                (0, cursor[1]), (cw, cursor[1]), color=guide, thickness=1.0, parent="canvas"
            )
            dpg.draw_line(
                (cursor[0], 0), (cursor[0], ch), color=guide, thickness=1.0, parent="canvas"
            )

    def _draw_dashed_rect(
        self,
        c1: Tuple[float, float],
        c2: Tuple[float, float],
        color,
        thickness: float = 1.5,
        dash: float = 7.0,
        gap: float = 5.0,
    ) -> None:
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
                dpg.draw_line(
                    (ax + ux * d, ay + uy * d),
                    (ax + ux * e, ay + uy * e),
                    color=color,
                    thickness=thickness,
                    parent="canvas",
                )
                d += dash + gap

        _edge(x1, y1, x2, y1)
        _edge(x2, y1, x2, y2)
        _edge(x2, y2, x1, y2)
        _edge(x1, y2, x1, y1)

    def _draw_box(self, obj: AnnObject, selected: bool = False) -> None:
        x1, y1, x2, y2 = obj.bbox
        c1 = self._i2c(x1, y1)
        c2 = self._i2c(x2, y2)
        # Without track ids every box would share the "untracked" colour, which
        # makes a crowded frame unreadable. Colour by class instead: that is
        # the identity such a project does assert.
        key = obj.track_id if self.use_track_ids else obj.class_id
        col = _track_color(key)
        col_fill = _track_color(key, alpha=50)
        col, col_fill = self._keyframe_box_colors(obj, col, col_fill)
        thick = 3.0 if selected else 1.5

        if obj.has_obb:
            # --- 回転矩形: 4 頂点の polygon で描く ---
            pts = [self._i2c(px, py) for px, py in obj.corners()]
            if getattr(obj, "keyframe", True):
                dpg.draw_polygon(pts, color=col, fill=col_fill, thickness=thick, parent="canvas")
            else:
                for i in range(4):
                    self._draw_dashed_line(pts[i], pts[(i + 1) % 4], col, thick)
            # 体軸 (w 方向) を示す線 — 向きが一目で判るようにする
            mid_front = ((pts[1][0] + pts[2][0]) / 2, (pts[1][1] + pts[2][1]) / 2)
            center = self._i2c(obj.obb[0], obj.obb[1])
            dpg.draw_line(center, mid_front, color=col, thickness=thick, parent="canvas")
        elif getattr(obj, "keyframe", True):
            dpg.draw_rectangle(c1, c2, color=col, fill=col_fill, thickness=thick, parent="canvas")
        else:
            # 補間/自動生成 box は破線で描き、手動キーフレームと区別する
            self._draw_dashed_rect(c1, c2, col, thickness=thick)

        # Label
        action_s = self._behavior_text(obj)
        if self.keyframe_mode:
            label = self._keyframe_box_label(obj, action_s)
        elif self.use_track_ids:
            label = f"{'T' + str(obj.track_id) if obj.track_id >= 0 else 'T?'}|{action_s}"
        else:
            class_s = (
                self.class_names[obj.class_id]
                if 0 <= obj.class_id < len(self.class_names)
                else "?"
            )
            label = f"{class_s}|{action_s}"
        dpg.draw_text((c1[0] + 3, c1[1] + 2), label, color=(0, 0, 0, 200), size=13, parent="canvas")
        dpg.draw_text(
            (c1[0] + 2, c1[1] + 1), label, color=(255, 255, 255, 220), size=13, parent="canvas"
        )

        if obj.occluded:
            dpg.draw_text(
                (c2[0] - 28, c1[1] + 2), "OCC", color=(255, 230, 0, 220), size=11, parent="canvas"
            )

        # Track indicator (colored dot)
        has_tracker = obj.track_id in self._trackers
        dot_col = (100, 255, 100, 220) if has_tracker else col
        dpg.draw_circle((c1[0] + 6, c2[1] - 6), 5, color=dot_col, fill=dot_col, parent="canvas")

        # Resize handles when selected
        if selected:
            if obj.has_obb:
                # 回転矩形: 4 隅ハンドル + 体軸方向に伸びる回転ハンドル
                for hx, hy in (self._i2c(px, py) for px, py in obj.corners()):
                    dpg.draw_circle(
                        (hx, hy), _HANDLE_R, color=col, fill=(255, 255, 255, 200), parent="canvas"
                    )
                rot = self._rotation_handle_pos(obj)
                ctr = self._i2c(obj.obb[0], obj.obb[1])
                dpg.draw_line(ctr, rot, color=(255, 230, 0, 200), thickness=1.5, parent="canvas")
                dpg.draw_circle(
                    rot,
                    _HANDLE_R,
                    color=(255, 230, 0, 255),
                    fill=(255, 230, 0, 180),
                    parent="canvas",
                )
            else:
                for hx, hy in [c1, (c2[0], c1[1]), (c1[0], c2[1]), c2]:
                    dpg.draw_circle(
                        (hx, hy), _HANDLE_R, color=col, fill=(255, 255, 255, 200), parent="canvas"
                    )

    def _draw_candidate(self, cand) -> None:
        """未承認の候補を描く。

        破線・琥珀色・塗りなし。承認済みの box (実線) と一目で区別がつくことが
        必要条件です。候補は保存されず、消えるかもしれないものなので、
        アノテーションと同じ見た目にしてはいけません。
        """
        colour = (255, 190, 60, 230)
        if cand.obb and len(cand.obb) == 5:
            pts = [self._i2c(x, y) for x, y in click_segment.obb_corners(cand.obb)]
            for i in range(4):
                self._draw_dashed_line(pts[i], pts[(i + 1) % 4], colour, 1.5)
        else:
            x1, y1, x2, y2 = cand.bbox
            self._draw_dashed_rect(self._i2c(x1, y1), self._i2c(x2, y2), colour)
        cx, cy = self._i2c(cand.bbox[0], cand.bbox[1])
        dpg.draw_text(
            (cx + 2, cy - 16), f"? {cand.score:.2f}", color=colour, size=13, parent="canvas"
        )

    def _draw_dashed_line(
        self, a, b, color, thickness: float = 1.5, dash: float = 7.0, gap: float = 5.0
    ) -> None:
        """破線の線分 (dpg に破線 API が無いため線分で近似)。"""
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1.0:
            return
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        d = 0.0
        while d < length:
            e = min(d + dash, length)
            dpg.draw_line(
                (a[0] + ux * d, a[1] + uy * d),
                (a[0] + ux * e, a[1] + uy * e),
                color=color,
                thickness=thickness,
                parent="canvas",
            )
            d += dash + gap

    def _rotation_handle_pos(self, obj: AnnObject) -> Tuple[float, float]:
        """回転ハンドルの canvas 座標 (体軸 +w 方向へ少し離した位置)。"""
        cx, cy, w, h, th = obj.get_obb()
        # 箱のサイズに依らず掴みやすいよう、最低でも一定距離だけ外に出す
        dist = max(w / 2 + _ROT_HANDLE_OFFSET / max(self._scale, 1e-6), w / 2 + 1)
        return self._i2c(cx + dist * math.cos(th), cy + dist * math.sin(th))

    # ═══════════════════════════════════════════════════════════════════
    # Mouse Interaction
    # ═══════════════════════════════════════════════════════════════════

    def _is_over_canvas(self) -> bool:
        """
        マウスが描画キャンバス上にあるか。

        child_window の ``is_item_hovered()`` は ImGui の仕様上、マウスボタンを
        押している間 (active item がいる間) は False を返す。これをドラッグ判定に
        使うとボタンを押した瞬間に False になり、ドラッグが一切届かなくなる
        (= ボックスが描けない)。drawlist は押下中も True を返すので、そちらを見る。
        """
        if dpg.does_item_exist("canvas") and dpg.is_item_hovered("canvas"):
            return True
        # ドラッグ中はキャンバスの外へはみ出しても操作を続けられるようにする
        # (座標は frame 境界へクランプ済みなので安全)。
        if self._mode != "idle" and dpg.is_mouse_button_down(0):
            return True
        return dpg.does_item_exist("canvas_win") and dpg.is_item_hovered("canvas_win")

    def _on_mouse_press(self, sender, app_data) -> None:
        # シークバー上の押下はフレーム移動 (そのままドラッグでスクラブ)
        if self._begin_scrub():
            return
        if not self._is_over_canvas() or self.cap is None:
            return
        if self.context_loop_active:  # bring the labelled frame back, then click
            self.stop_context_loop()
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)

        # Pan (H): drag moves the view. The image must not gain a box while
        # the user is only looking around, which is exactly what happened
        # before this branch existed - the press fell through to Select and
        # started a rubber band.
        if self._tool == "pan":
            self._mode = "pan"
            self._pan_origin = (cx, cy, self._pan_x, self._pan_y)
            return

        # Delete Box (X): click removes the box under the cursor, one undo step
        # each. Sticky, because it is an eraser and you rarely delete just one.
        if self._tool == "delete":
            victim = self._hit_object(ix, iy)
            if victim is None:
                self._status("Nothing under the cursor. Esc leaves Delete.")
                return
            self._push_undo()
            self._mark_dirty()
            self.annotations.get(self.current_frame_idx, []).remove(victim)
            if self._selected is victim:
                self._selected = None
            self._render_canvas()
            self._refresh_sidebar()
            self._status(
                f"Deleted a box on track {victim.track_id}. "
                f"{shortcuts.label_for('edit.undo')} takes it back."
            )
            return

        # Click to Box (C): 押した瞬間に 1 個体ぶんの箱を作る。ドラッグは
        # 伴わないので _mode は idle のままにし、release 側には何も残さない。
        if self._tool == "click":
            self._click_to_box(ix, iy)
            return

        # Draw tools (W / E): 既存の箱には触れず、必ず新しい箱を描き始める。
        # これにより既存ボックスの内側や重なった位置にも描ける (LabelImg と同じ)。
        # E is the same gesture as W; the only difference is that the box it
        # makes is rotated, which _create_object reads from _obb_mode.
        if self._tool in ("draw", "obb"):
            self.deselect(render=False)
            self._mode = "draw"
            ixd = max(0.0, min(ix, float(self.frame_w)))
            iyd = max(0.0, min(iy, float(self.frame_h)))
            self._draw_start = (ixd, iyd)
            self._draw_end = (ixd, iyd)
            return

        # Resize handle?
        if self._selected is not None:
            h = self._hit_handle(cx, cy, self._selected)
            if h is not None:
                self._mode = "rotate" if h == "rot" else "resize"
                self._begin_drag_edit()
                self._resize_handle = h
                self._drag_start = (ix, iy)
                self._drag_bbox_orig = list(self._selected.bbox)
                self._drag_obb_orig = list(self._selected.obb) if self._selected.has_obb else None
                return

        # A pending candidate? Clicking one accepts exactly it. The click is
        # the review act, and it is checked before existing boxes because a
        # candidate is drawn on top and is what the user is looking at.
        cand_index = self._hit_candidate(ix, iy)
        if cand_index is not None:
            self.accept_candidate(cand_index)
            self._status("Candidate accepted. The rest are still unsaved.")
            self._render_canvas()
            self._refresh_sidebar()
            return

        # Existing box?
        hit = self._hit_object(ix, iy)
        if hit is not None:
            self._select(hit)
            self._mode = "move"
            self._begin_drag_edit()
            self._drag_start = (ix, iy)
            self._drag_bbox_orig = list(hit.bbox)
            self._drag_obb_orig = list(hit.obb) if hit.has_obb else None
            return

        # Draw new box
        self.deselect(render=False)
        self._mode = "draw"
        ixc = max(0.0, min(ix, float(self.frame_w)))
        iyc = max(0.0, min(iy, float(self.frame_h)))
        self._draw_start = (ixc, iyc)
        self._draw_end = (ixc, iyc)

    def _on_mouse_drag(self, sender, app_data) -> None:
        if self._mode == "scrub":
            # バーの外へはみ出しても掴んだままスクラブを続ける (端でクランプ)
            self._scrub_to_mouse()
            return
        if self._mode == "idle" or not self._is_over_canvas():
            return
        cx, cy = self._get_canvas_mouse()
        ix, iy = self._c2i(cx, cy)
        ixc = max(0.0, min(ix, float(self.frame_w)))
        iyc = max(0.0, min(iy, float(self.frame_h)))

        if self._mode == "pan":
            self._drag_pan(cx, cy)
            return

        if self._mode == "draw":
            self._draw_end = (ixc, iyc)

        elif self._mode == "move" and self._selected is not None:
            dx = ix - self._drag_start[0]
            dy = iy - self._drag_start[1]
            if self._drag_obb_orig is not None:
                # OBB は中心を動かすだけ (w/h/theta は保つ)
                ocx, ocy, ow, oh, oth = self._drag_obb_orig
                self._selected.obb = [
                    max(0.0, min(ocx + dx, float(self.frame_w))),
                    max(0.0, min(ocy + dy, float(self.frame_h))),
                    ow,
                    oh,
                    oth,
                ]
                self._selected.sync_bbox()
            else:
                ox1, oy1, ox2, oy2 = self._drag_bbox_orig
                w, h = ox2 - ox1, oy2 - oy1
                nx1 = int(max(0.0, min(ox1 + dx, self.frame_w - w)))
                ny1 = int(max(0.0, min(oy1 + dy, self.frame_h - h)))
                self._selected.bbox = [nx1, ny1, nx1 + w, ny1 + h]
            self._selected.keyframe = True  # 手動移動でキーフレーム化 (補間アンカー)
            self._moved = True
            # Re-init tracker on manual move
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        elif self._mode == "rotate" and self._selected is not None:
            # 中心からマウスへの向きをそのまま体軸方向にする (直感的な回転)
            cx, cy = self._selected.obb[0], self._selected.obb[1]
            self._selected.obb[4] = math.atan2(iy - cy, ix - cx)
            self._selected.sync_bbox()
            self._selected.keyframe = True
            self._moved = True
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        elif self._mode == "resize" and self._selected is not None:
            if self._drag_obb_orig is not None:
                self._moved |= self._resize_obb(ixc, iyc)
            else:
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
                    self._moved = True
            if self._auto_track and self._selected.track_id in self._trackers:
                del self._trackers[self._selected.track_id]

        self._render_canvas()

    def _resize_obb(self, ix: float, iy: float) -> bool:
        """
        回転矩形の隅ハンドルドラッグ。``True`` なら実際に形が変わった。

        マウス位置を box ローカル座標へ戻し、対角の隅を固定したまま w/h を決める。
        こうすると「掴んだ隅だけが動く」という軸平行の操作感が、回転していても
        そのまま成り立つ。

        戻り値は undo のためにある: 最小サイズを下回って何もしなかったドラッグで
        undo ステップを積むと、Ctrl+Z が「何も起きない 1 回」になってしまう。
        """
        obj = self._selected
        ocx, ocy, ow, oh, oth = self._drag_obb_orig
        idx = int(self._resize_handle[1])  # "c0".."c3"
        c, s = math.cos(-oth), math.sin(-oth)

        # マウスをローカル座標へ
        dx, dy = ix - ocx, iy - ocy
        lx = dx * c - dy * s
        ly = dx * s + dy * c

        # 対角の固定点 (ローカル座標)
        signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        fx = -signs[idx][0] * ow / 2
        fy = -signs[idx][1] * oh / 2

        new_w = abs(lx - fx)
        new_h = abs(ly - fy)
        if new_w < _MIN_BOX_PX or new_h < _MIN_BOX_PX:
            return False

        # 新しい中心 = 固定点と可動点の中点 (ローカル) → グローバルへ戻す
        mid_lx = (lx + fx) / 2
        mid_ly = (ly + fy) / 2
        cf, sf = math.cos(oth), math.sin(oth)
        obj.obb = [
            ocx + mid_lx * cf - mid_ly * sf,
            ocy + mid_lx * sf + mid_ly * cf,
            new_w,
            new_h,
            oth,
        ]
        obj.sync_bbox()
        obj.keyframe = True
        return True

    def _on_mouse_release(self, sender, app_data) -> None:
        if self._mode == "scrub":
            self._end_scrub()
            return
        if self._mode == "draw" and self._draw_start and self._draw_end:
            xs = sorted([int(self._draw_start[0]), int(self._draw_end[0])])
            ys = sorted([int(self._draw_start[1]), int(self._draw_end[1])])
            if xs[1] - xs[0] >= _MIN_BOX_PX and ys[1] - ys[0] >= _MIN_BOX_PX:
                obj = self._create_object([xs[0], ys[0], xs[1], ys[1]])
                # Initialize tracker in auto-track mode
                if self._auto_track and self._current_bgr is not None:
                    self._init_tracker(obj.track_id, self.current_frame_idx, obj.bbox)
                if self._tool in ("draw", "obb") and not self._sticky_draw:
                    # LabelImg と同じく 1 個描いたら選択ツールへ戻る。
                    # 回転ボックスは選択したまま戻るので、そのまま回転ハンドルを
                    # 掴める (一番よくある次の操作)。
                    self.set_tool("select")
                    self._select(obj)
                else:
                    self._status(
                        "Box created. " + shortcuts.with_key("Draw another", "tool.draw_box")
                    )
            elif self._tool in ("draw", "obb"):
                self._status(
                    f"The box was too small. Drag an area of at least "
                    f"{_MIN_BOX_PX} x {_MIN_BOX_PX} pixels."
                )
        elif self._mode in _DRAG_EDIT_MODES and self._selected is not None:
            # Rotate belongs here too: turning a box by its handle is an edit
            # like any other, and leaving it out meant a rotated box kept a
            # tracker initialised from the orientation it no longer has.
            self._commit_drag_edit()
            if self._auto_track and self._current_bgr is not None:
                self._init_tracker(
                    self._selected.track_id, self.current_frame_idx, self._selected.bbox
                )

        self._mode = "idle"
        self._draw_start = self._draw_end = None
        self._drag_start = self._drag_bbox_orig = None
        self._drag_obb_orig = None
        self._resize_handle = None
        self._pending_snap = None
        self._moved = False
        self._pan_origin = None
        self._render_canvas()
        self._refresh_sidebar()

    def _begin_drag_edit(self) -> None:
        """Snapshot before a move / resize / rotate drag starts (P7).

        Taken at **press** because that is the only moment the box is still in
        its old shape, and pushed at release only if it actually moved - so a
        click that merely selects leaves no undo step behind.
        """
        self._pending_snap = self._snapshot()
        self._moved = False

    def _commit_drag_edit(self) -> None:
        """Close a move / resize / rotate drag: one undo step, one dirty mark.

        Dragging a box was the most common edit in the annotator and the only
        one that recorded **nothing**: no undo entry and no dirty flag. Ctrl+Z
        after a misplaced drag therefore undid whatever came before it - and
        the save chip still read *No changes*, so the autosave never fired and
        closing the window threw the correction away. ``_pending_snap`` and
        ``_moved`` were declared for exactly this and never used.
        """
        if not self._moved or self._pending_snap is None:
            return
        self._push_undo(self._pending_snap)
        self._pending_snap = None
        self._moved = False
        self._mark_dirty()
        verb = {"move": "Moved", "resize": "Resized", "rotate": "Rotated"}.get(self._mode, "Edited")
        self._status(
            f"{verb} the box on track {self._selected.track_id}. "
            f"{shortcuts.label_for('edit.undo')} takes it back."
        )

    def _on_right_click(self, sender, app_data) -> None:
        if not self._is_over_canvas() or self.cap is None:
            return
        # 描画 / クリック中の右クリックはキャンセル (メニューは出さない)
        if self._tool in ("draw", "obb", "click", "pan", "delete"):
            self._escape()
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
        with dpg.window(
            tag="ctx_menu",
            popup=True,
            no_title_bar=True,
            no_move=True,
            pos=(mx, my),
            min_size=(160, 10),
        ):
            if self.use_track_ids:
                dpg.add_text(f"Track {obj.track_id}", color=(200, 200, 100, 255))
                dpg.add_separator()
                dpg.add_menu_item(label="Set Track ID...", callback=self._prompt_track_id_dialog)
                dpg.add_separator()
            # --- OBB 操作 ---
            if obj.has_obb:
                angle_deg = math.degrees(obj.obb[4])
                dpg.add_text(f"OBB  angle: {angle_deg:+.1f}°", color=(255, 230, 0, 220))
                dpg.add_menu_item(label="Rotate +15°", callback=lambda: self.rotate_selected(15))
                dpg.add_menu_item(label="Rotate -15°", callback=lambda: self.rotate_selected(-15))
                dpg.add_menu_item(
                    label="Convert to axis-aligned box", callback=self.convert_selected_to_aabb
                )
            else:
                dpg.add_menu_item(
                    label="Convert to rotated box (OBB)", callback=self.convert_selected_to_obb
                )
            dpg.add_separator()
            for i, name in enumerate(self.action_names):
                dpg.add_menu_item(
                    label=f"Action: {name}",
                    callback=lambda s, a, u: self.set_action(u),
                    user_data=i,
                )
            dpg.add_separator()
            if self._auto_track and self.use_track_ids:
                dpg.add_menu_item(
                    label="Init Tracker for This Box",
                    callback=lambda: self._init_tracker_from_selected(),
                )
                dpg.add_menu_item(
                    label="Remove Tracker for This Track",
                    callback=lambda: self._reset_selected_tracker(),
                )
            dpg.add_separator()
            dpg.add_menu_item(label="Delete Box", callback=self.delete_selected)

    def _hit_handle(self, cx: float, cy: float, obj: AnnObject) -> Optional[str]:
        """
        ハンドル判定。OBB のときは回転ハンドルを**先に**見る。

        回転ハンドルは体軸の外側にあり隅ハンドルと重なりうるので、優先順を
        固定しておかないと掴む対象が状況で変わってしまう。
        """
        if obj.has_obb:
            rx, ry = self._rotation_handle_pos(obj)
            if abs(cx - rx) <= _HANDLE_R + 3 and abs(cy - ry) <= _HANDLE_R + 3:
                return "rot"
            names = ("c0", "c1", "c2", "c3")
            for name, (px, py) in zip(names, obj.corners()):
                hx, hy = self._i2c(px, py)
                if abs(cx - hx) <= _HANDLE_R + 3 and abs(cy - hy) <= _HANDLE_R + 3:
                    return name
            return None

        x1, y1, x2, y2 = obj.bbox
        corners = {
            "tl": self._i2c(x1, y1),
            "tr": self._i2c(x2, y1),
            "bl": self._i2c(x1, y2),
            "br": self._i2c(x2, y2),
        }
        for name, (hx, hy) in corners.items():
            if abs(cx - hx) <= _HANDLE_R + 3 and abs(cy - hy) <= _HANDLE_R + 3:
                return name
        return None

    @staticmethod
    def _point_in_obb(px: float, py: float, obj: AnnObject) -> bool:
        """点が回転矩形の内部にあるか (box ローカル座標へ戻して判定)。"""
        cx, cy, w, h, th = obj.get_obb()
        c, s = math.cos(-th), math.sin(-th)
        dx, dy = px - cx, py - cy
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        return abs(lx) <= w / 2 and abs(ly) <= h / 2

    def _hit_object(self, ix: float, iy: float) -> Optional[AnnObject]:
        """画像座標 (ix, iy) にある最前面のオブジェクトを返す。"""
        for obj in reversed(self.annotations.get(self.current_frame_idx, [])):
            if obj.has_obb:
                if self._point_in_obb(ix, iy, obj):
                    return obj
            else:
                bx1, by1, bx2, by2 = obj.bbox
                if bx1 <= ix <= bx2 and by1 <= iy <= by2:
                    return obj
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Video Loading
    # ═══════════════════════════════════════════════════════════════════

    def _select(self, obj: AnnObject) -> None:
        # A box selection supersedes whatever the Inspector was showing: the
        # user just pointed at something, and that is the context now.
        self._sel_tracklet = self._sel_interval = self._sel_issue = None
        self._sel_clip = None
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

    def _create_object(self, bbox: List[int], source: str = "human") -> AnnObject:
        self._push_undo()
        self._mark_dirty()
        obj = AnnObject(
            bbox=bbox,
            # -1 is the file format's own "no identity asserted", so a project
            # that does not track needs no new field and no new reader.
            track_id=self.active_track_id if self.use_track_ids else -1,
            action_id=-1,
            source=source,
            class_id=getattr(self, "active_class_id", 0),
        )
        if self._obb_mode:
            # OBB モードでは theta=0 の回転矩形として作る。以後は回転ハンドルで
            # 向きを付けられる (最初から回転させる必要はない)。
            obj.make_obb()
        self.annotations.setdefault(self.current_frame_idx, []).append(obj)
        self._select(obj)
        return obj

    # ─── Track identity: the one project-level yes/no ────────────────

    @property
    def use_track_ids(self) -> bool:
        """Does this project follow individuals across frames?

        Stored on the project, not on the app, because it describes the
        experiment rather than the session: reopening a project that does not
        track must not silently start asking for track ids again. Absent from
        an older file it reads True, so nothing written before this existed
        changes behaviour. Keyframe labelling reads False (the flag is kept).
        """
        if self.keyframe_mode:
            return False
        return bool(self._ensure_project().settings.get("use_track_ids", True))

    def set_use_track_ids(self, value: Any) -> None:
        """Turn track identity on or off for the whole project.

        Turning it *off* leaves the ids already on disk alone. Rewriting them
        to -1 would silently destroy work on the strength of a checkbox, and
        turning the switch back on could not undo it.
        """
        value = bool(value)
        project = self._ensure_project()
        if project.settings.get("use_track_ids", True) == value:
            return
        project.settings["use_track_ids"] = value
        if not value:
            self._trackers.clear()
            self._auto_track = False
        self._mark_dirty()
        self._refresh_track_ui()
        self.rebuild_copilot()
        self._render_canvas()
        self._refresh_sidebar()
        self._status(
            "Boxes now carry a track id, and the track-based checks are back on."
            if value
            else "Track ids are off. Boxes are per-frame detections and the "
            "track-based checks are not run."
        )

    def _refresh_track_ui(self) -> None:
        """Show or hide every control that only means something with track ids.

        The Inspector hides its own two track controls from
        ``InspectorContext.use_track_ids``; this handles the Settings mirror
        and the auto-track checkbox, which are not part of a selection.
        """
        self.settings.set_use_track_ids(self.use_track_ids)
        self._refresh_inspector()

    # ─── Tool mode (select / draw) ───────────────────────────────────

    #: Every tool the canvas understands. ``obb``, ``pan`` and ``delete`` are
    #: new; the old hidden "OBB mode" checkbox becomes a visible tool with a
    #: visible active state (docs/GUI_DESIGN.md §0.3.2).
    _TOOL_NAMES = ("select", "draw", "obb", "click", "pan", "delete")

    def set_tool(self, tool: str) -> None:
        """ツールを切り替える。未知の名前は Select へ落ちる。"""
        self._tool = tool if tool in self._TOOL_NAMES else "select"
        # The rotated-box tool *is* the old OBB mode. Keeping the flag in sync
        # means the drawing code needs no change and the sidebar mirror still
        # reads true.
        self._obb_mode = self._tool == "obb"
        if dpg.does_item_exist("obb_mode_cb"):
            dpg.set_value("obb_mode_cb", self._obb_mode)
        self._status(_TOOL_CHIP.get(self._tool, ""))
        self._refresh_tool_ui()
        self._render_canvas()

    def toggle_draw_tool(self) -> None:
        """W キー: 描画モードのオン / オフ (LabelImg の 'Create RectBox')。"""
        self.set_tool("select" if self._tool == "draw" else "draw")

    def toggle_click_tool(self) -> None:
        """C キー: Click to Box のオン / オフ。"""
        self.set_tool("select" if self._tool == "click" else "click")

    # ─── Click to Box ────────────────────────────────────────────────

    def _click_to_box(self, ix: float, iy: float) -> None:
        """クリック 1 点から個体を推定し、回転矩形を作る。

        設計と根拠は ``docs/CLICK_SEGMENT.md``。ここは薄い接着層に徹し、
        推定そのものは ``gui/click_segment.py`` (dpg 非依存・単体テスト可) が持つ。

        失敗したときは**箱を作らない**。それらしい矩形を置く方が、
        気づかれないまま学習データへ入る分だけ質が悪い。
        """
        if self._current_bgr is None:
            self._status(click_segment.DIAGNOSES[click_segment.NO_FRAME])
            return
        req = click_segment.ClickRequest(
            frame_index=self.current_frame_idx,
            x=float(ix),
            y=float(iy),
            radius_hint=self._click_radius_hint,
        )

        backend = self._click_backend
        if backend == "auto":
            if self._prepare_auto_click(ix, iy):
                return
            backend = (
                "sam"
                if self._sam_ready()
                else ("model" if self._predictions.get(self.current_frame_idx) else "local")
            )
        # SAM だけは別プロセスで走るので答えが後から来る。ここでは投げるだけで、
        # 適用は _pump_sidecar → _apply_sam_reply が行う (GUI_DESIGN P9)。
        if backend == "sam" and self._sam_ready():
            self._request_sam_point(req)
            return

        try:
            if backend == "model":
                res = click_segment.pick_candidate(
                    self._predictions.get(self.current_frame_idx, []), req
                )
                if not res.ok and self._click_backend == "auto":
                    res = click_segment.segment(self._current_bgr, req, backend="local")
            else:
                res = click_segment.segment(self._current_bgr, req, backend=backend)
        except Exception as exc:  # noqa: BLE001 — マウスハンドラを落とさない
            self._status(f"Click animal failed: {exc}")
            return
        self._apply_click_result(res)

    def _apply_click_result(self, res) -> None:
        """``ClickResult`` を 1 個の box にする。バックエンド共通の出口。

        同期 (local / model) も非同期 (sam) もここに合流するので、「クリックで
        できた箱」の性質 —— provenance・選択状態・トラッカ初期化・status ——
        はバックエンドによって変わりません。
        """
        if not res.ok:
            self._status(res.detail)
            return
        cx, cy, w, h, theta = res.obb
        obj = self._create_object(list(res.bbox), source="wand")
        obj.obb = [cx, cy, w, h, theta]
        obj.obb_source = "annotated"
        if getattr(res, "class_id", None) is not None:
            obj.class_id = res.class_id
        obj.sync_bbox()
        # 次のクリックは同じくらいの大きさの個体を狙うはずなので、探索窓の
        # 初期値を実測値から決める (小さすぎて何度も広げ直すのを避ける)。
        self._click_radius_hint = max(float(w), float(h))
        if self._auto_track and self.use_track_ids:
            self._init_tracker(obj.track_id, self.current_frame_idx, obj.bbox)

        # One click, one box, then back to Select — the same rule W and E
        # follow. A fitted box is almost never right first time, and the next
        # thing the hand does is grab a handle; leaving the tool armed meant
        # that grab dropped a second box on top of the first one instead.
        # "Keep drawing after each box" opts back into the sticky behaviour.
        if self._tool == "click" and not self._sticky_draw:
            self.set_tool("select")
            self._select(obj)
            self._status(
                f"{res.detail} Drag a handle to adjust it, or press "
                f"{shortcuts.label_for('tool.click_box')} for the next animal."
            )
        else:
            self._status(
                f"{res.detail} Adjust the handles if it is off, or press Delete to remove it."
            )
        self._render_canvas()
        self._refresh_sidebar()

    # ─── Click to Box: backends ──────────────────────────────────────

    def load_predictions_for_clicks(self, path: Path) -> bool:
        """``predictions.json`` を model バックエンド用に読む。推論はしない。"""
        try:
            from .gui.auto_label import load_candidates
        except ImportError:  # pragma: no cover
            from htrtdetr.gui.auto_label import load_candidates
        try:
            frames = load_candidates(Path(path))
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not read those predictions: {exc}")
            return False
        total = sum(len(v) for v in frames.values())
        if not total:
            self._status(
                f"{Path(path).name} has no detections in it. Check the run's "
                f"score threshold, or keep the Local (OpenCV) fit."
            )
            return False
        self._predictions = frames
        self._predictions_path = Path(path)
        self._status(
            f"Loaded {total} detections on {len(frames)} frames from "
            f"{Path(path).name}. Click animal can now use Model predictions."
        )
        self._refresh_backend_ui()
        return True

    # ─── Find all in frame (concept prompt) ──────────────────────────
    #
    # ここだけは Click to Box と扱いが逆になります。クリックでは**人が対象を
    # 選んで**いるので箱は即 accepted で作りますが、concept では機械が
    # 「何を囲むか」まで決めるので、GUI_DESIGN P1 のとおり候補は
    # **ディスクに触れない pending** として置き、人の承認を必ず挟みます。
    # 判断基準はモデルの種類ではなく「対象を選んだのは誰か」です。

    def _review_session(self):
        if self._review is None:
            try:
                from .gui.auto_label import ReviewSession
            except ImportError:  # pragma: no cover
                from htrtdetr.gui.auto_label import ReviewSession
            self._review = ReviewSession()
        return self._review

    def candidates_here(self) -> list:
        """このフレームの pending 候補。accepted/rejected は含めない。"""
        if self._review is None:
            return []
        try:
            from .gui.auto_label import PENDING
        except ImportError:  # pragma: no cover
            from htrtdetr.gui.auto_label import PENDING
        fi = self.current_frame_idx
        return [
            (i, c)
            for i, c in enumerate(self._review.candidates(fi))
            if self._review.state(fi, i) == PENDING
        ]

    def find_all_in_frame(self, text: str = "") -> bool:
        """テキストでこのフレームの個体をまとめて検出させる (SAM3 系のみ)。"""
        text = (text or "").strip() or (self.class_names[0] if self.class_names else "animal")
        if not self._sam_ready():
            # Pressing this IS asking to use the helper — the same reading
            # Continue and clip propagation have always taken. It used to send
            # the user to an unrelated Click to Box setting instead, and said
            # so in one line at the bottom of the window, so the button looked
            # dead. The resolved text is captured, so the replay searches for
            # what was typed rather than for the default.
            self._start_sam(on_done=lambda ok: self.find_all_in_frame(text) if ok else None)
            if not self._sam_ready():
                self._status(
                    f"Starting the segmentation helper. The search for "
                    f"'{text}' will run as soon as it answers."
                )
                return False
        client = self._sam_client
        if "concept" not in client.capabilities:
            self._status(
                f"The {client.backend_name} backend cannot find animals from a "
                f"description. Click them one at a time with C instead."
            )
            return False
        if self._current_bgr is None:
            self._status(click_segment.DIAGNOSES[click_segment.NO_FRAME])
            return False
        frame_index = self.current_frame_idx
        try:
            path = client.frames.put(self._sam_frame_id(frame_index), self._current_bgr)
            req_id = client.send(
                "concept", frame=self._sam_frame_id(frame_index), path=str(path), text=text, max=64
            )
        except Exception as exc:  # noqa: BLE001
            self._status(f"The segmentation helper stopped: {exc}")
            self._stop_sam()
            return False
        self._sam_jobs[req_id] = {"kind": "concept", "frame": frame_index, "text": text}
        self._sam_waiting = True
        self._arm_sam_pump()
        self._status(f"Looking for every '{text}' in this frame...")
        return True

    def _apply_concept_reply(self, msg: dict, job: dict) -> None:
        try:
            from .gui.auto_label import Candidate
        except ImportError:  # pragma: no cover
            from htrtdetr.gui.auto_label import Candidate
        items = msg.get("items") or []
        if not items:
            self._status(
                f"No '{job['text']}' was found in this frame. Try clicking one "
                f"with C, or draw a box by hand with W."
            )
            return
        cands = [
            Candidate(
                bbox=[int(v) for v in it["bbox"]],
                obb=[float(v) for v in it["obb"]],
                score=float(it.get("score") or 1.0),
                source="auto",
                class_id=self.active_class_id,
            )
            for it in items
            if it.get("bbox") and it.get("obb")
        ]
        self._review_session().add_frame(job["frame"], cands)
        self._mark_dirty()
        self._status(
            f"Found {len(cands)} candidates. They are not saved yet - accept the "
            f"good ones, or discard them all."
        )
        self._render_canvas()
        self._refresh_candidate_ui()

    def accept_candidate(self, index: int) -> bool:
        """候補 1 つを本物の box にする。**ここが唯一ディスクに届く経路**。"""
        count = self._accept_targets([(self.current_frame_idx, index)])
        if count:
            self._select(self.annotations[self.current_frame_idx][-1])
        return bool(count)

    def accept_all_candidates(self) -> int:
        pending = self.candidates_here()
        for index, _cand in reversed(pending):
            self.accept_candidate(index)
        if pending:
            self._status(f"Accepted {len(pending)} candidates on this frame.")
            self._render_canvas()
            self._refresh_sidebar()
        return len(pending)

    def discard_all_candidates(self) -> int:
        if self._review is None:
            return 0
        self._push_undo()
        n = self._review.reject_frame(self.current_frame_idx)
        if n:
            self._mark_dirty()
        if n:
            self._status(f"Discarded {n} candidates. Accepted boxes are untouched.")
            self._render_canvas()
            self._refresh_candidate_ui()
        return n

    def _hit_candidate(self, ix: float, iy: float) -> Optional[int]:
        """クリック点にある pending 候補の index (手前から)。"""
        for index, cand in reversed(self.candidates_here()):
            if cand.obb and click_segment.point_in_obb(ix, iy, cand.obb):
                return index
            x1, y1, x2, y2 = cand.bbox
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return index
        return None

    def _refresh_candidate_ui(self) -> None:
        pending = self.candidates_here()
        if dpg.does_item_exist("candidate_count"):
            dpg.set_value(
                "candidate_count",
                f"{len(pending)} candidates on this frame" if pending else "No candidates",
            )
        for tag in ("accept_all_btn", "discard_all_btn"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=bool(pending))

    def _load_predictions_dialog(self) -> None:
        path = _tk_open_file(
            "Load predictions.json for Click animal", [("JSON", "*.json"), ("All files", "*.*")]
        )
        if path and self.load_predictions_for_clicks(path):
            self.set_click_backend("model")

    def _refresh_backend_ui(self) -> None:
        for name in ("auto", *click_segment.BACKENDS):
            tag = f"click_backend_{name}"
            if dpg.does_item_exist(tag):
                on = self._click_backend == name
                dpg.configure_item(
                    tag,
                    label=("> " if on else "  ")
                    + click_segment.BACKEND_LABELS.get(name, "Automatic (best available)"),
                )

    def _on_sticky_draw_toggle(self, sender=None, app_data=None) -> None:
        """dearpygui-signature shim over :meth:`_set_sticky_draw`."""
        self._set_sticky_draw(app_data)

    def _refresh_tool_ui(self) -> None:
        """Make the active tool impossible to misread.

        Three redundant channels, one of them inside the canvas, because the
        user's eye is on the image and not on a rail 600 px away
        (docs/GUI_DESIGN.md §0.3.4): the rail button fills with the accent, the
        status line names the next action, and ``_render_canvas`` draws a
        coloured frame and caption for every tool but Select.
        """
        for tool, _label, _word, _action in _TOOLS:
            tag = f"rail_{tool}"
            if dpg.does_item_exist(tag):
                widgets.set_toolbar_active(tag, self._tool == tool)

        if dpg.does_item_exist("tool_mode_text"):
            dpg.set_value("tool_mode_text", _TOOL_CHIP.get(self._tool, ""))
            dpg.configure_item(
                "tool_mode_text", color=(theme.TEXT_DIM if self._tool == "select" else theme.ACCENT)
            )

    def _escape(self) -> None:
        """Esc: 描画ツールを抜けて選択も解除する。

        Every tool returns to Select, including the sticky ones - Delete in
        particular, which otherwise keeps erasing on the next click. A running
        context loop is ended first (the canvas is not showing this frame).
        """
        if self.context_loop_active:
            self.stop_context_loop()
            return
        if self._mode in ("pan",):
            self._mode = "idle"
            self._pan_origin = None
        if self._mode == "draw":
            # 描きかけの矩形を捨てる
            self._mode = "idle"
            self._draw_start = self._draw_end = None
        if self._tool != "select":
            self.set_tool("select")
        self.deselect()

    def toggle_obb_mode(self) -> None:
        """新規ボックスを OBB として作るかを切り替える。"""
        self._obb_mode = not self._obb_mode
        if dpg.does_item_exist("obb_mode_cb"):
            dpg.set_value("obb_mode_cb", self._obb_mode)
        self._status(
            f"OBB mode {'ON - new boxes are rotated' if self._obb_mode else 'OFF - axis-aligned boxes'}"
        )

    def convert_selected_to_obb(self) -> None:
        """選択中のボックスを回転矩形へ昇格させる (theta=0 から開始)。"""
        if self._selected is None:
            self._status("Select a box first.")
            return
        if self._selected.has_obb:
            self._status("This box is already a rotated box.")
            return
        self._push_undo()
        self._mark_dirty()
        self._selected.make_obb()
        self._selected.keyframe = True
        self._render_canvas()
        self._refresh_sidebar()
        self._status("Converted to a rotated box (rotate with the yellow handle)")

    def convert_selected_to_aabb(self) -> None:
        """選択中の回転矩形を軸平行ボックスへ戻す (外接矩形になる)。

        undo を積むのはガードを抜けたあと。抜けられなかった呼び出しで積むと、
        何も変えていない undo ステップが 1 つ増え、しかも ``_push_undo`` が
        redo スタックを消すので「やり直せたはずの操作」が消える。
        """
        if self._selected is None:
            self._status("Select a box first.")
            return
        if not self._selected.has_obb:
            self._status("This box is already an axis-aligned box.")
            return
        self._push_undo()
        self._mark_dirty()
        self._selected.drop_obb()
        self._selected.keyframe = True
        self._render_canvas()
        self._refresh_sidebar()
        self._status("Reverted to an axis-aligned box (bounding rectangle)")

    def rotate_selected(self, degrees: float) -> None:
        """選択中の OBB を相対角度で回す (キーボードからの微調整用)。"""
        if self._selected is None:
            self._status("Select a box first, then [ and ] turn it.")
            return
        self._push_undo()
        self._mark_dirty()
        self._selected.make_obb()
        self._selected.obb[4] += math.radians(degrees)
        self._selected.sync_bbox()
        self._selected.keyframe = True
        self._render_canvas()
        self._status(f"Rotate {degrees:+.0f}° -> {math.degrees(self._selected.obb[4]):.1f}°")

    def delete_selected(self) -> None:
        if self._selected is None:
            self._status("No box is selected.")
            return
        self._push_undo()
        self._mark_dirty()
        objs = self.annotations.get(self.current_frame_idx, [])
        if self._selected in objs:
            objs.remove(self._selected)
        if self._selected.track_id in self._trackers:
            del self._trackers[self._selected.track_id]
        self.deselect()

    def new_track(self) -> None:
        if not self._require_track_ids("A new track id"):
            return
        used = {o.track_id for row in self.annotations.values() for o in row}
        if self._review is not None:
            used.update(c.track_id for f in self._review.frames for c in self._review.candidates(f))
        self.next_track_id = max(self.next_track_id, max(used, default=0)) + 1
        self.active_track_id = self.next_track_id
        if dpg.does_item_exist("active_tid"):
            dpg.set_value("active_tid", self.next_track_id)
        self._update_active_track_color()
        self._status(f"New track ID: {self.next_track_id}")

    def propagate_from_prev(self) -> None:
        """Copy the previous frame's boxes onto this one.

        The undo step is pushed **after** every guard, not before. Space is a
        bare key and it is easy to hit on frame 0, on an empty previous frame
        or on a frame that is already annotated; pushing first meant each of
        those left an undo entry that restores nothing, marked a clean project
        Unsaved, and - because ``_push_undo`` clears the redo stack - threw
        away a redo the user could still have wanted.
        """
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
        # With track ids, one box per track: re-propagating must not stack a
        # second copy on a track that is already here. Without them there is no
        # identity to compare, so an already-annotated frame is left alone
        # rather than doubled.
        # ``get``, not ``setdefault``: a guard that returns must not leave an
        # empty list behind, or the frame counts an annotation it does not have.
        existing = self.annotations.get(self.current_frame_idx, [])
        existing_tids = {o.track_id for o in existing}
        count = 0
        if not self.use_track_ids and existing:
            self._status("This frame already has boxes. Delete them first to propagate again.")
            return
        if self.use_track_ids and all(o.track_id in existing_tids for o in prev_objs):
            self._status("Every track from the previous frame is already here.")
            return
        self._push_undo()
        self._mark_dirty()
        existing = self.annotations.setdefault(self.current_frame_idx, [])
        for obj in prev_objs:
            if not self.use_track_ids or obj.track_id not in existing_tids:
                record = obj.to_dict()
                record.pop("object_id", None)
                new_obj = AnnObject.from_dict(record)
                existing.append(new_obj)
                existing_tids.add(obj.track_id)
                count += 1
        self._render_canvas()
        self._refresh_sidebar()
        self._status(f"Propagated {count} object(s) from frame {prev}.")

    def set_action(self, action_id: int) -> None:
        if self._selected is None:
            self._status(f"No box selected. Action [{action_id}] was ignored.")
            return
        if not self._behavior_would_change(self._selected, action_id):
            return
        self._push_undo()
        self._mark_dirty()
        # One behavior, or this group's answer only (gui/behavior_groups.py).
        self._apply_behavior(self._selected, action_id)
        self._status(self._behavior_status(self._selected))
        self._refresh_sidebar()
        self._render_canvas()

    # ═══════════════════════════════════════════════════════════════════
    # Auto-Tracking
    # ═══════════════════════════════════════════════════════════════════

    def _init_tracker(self, track_id: int, frame_idx: int, bbox: List[int]) -> bool:
        """Initialize a tracker for the given track and bounding box.

        使用するトラッカーは OpenCV のビルドに合わせて選ばれる
        (contrib があれば CSRT、無ければ MIL などにフォールバック)。
        """
        if _TRACKER_FACTORY is None:
            self._status(_TRACKER_HINT)
            return False
        frame_bgr = self._read_frame_bgr(frame_idx)
        if frame_bgr is None:
            return False
        x1, y1, x2, y2 = bbox
        w, h = int(x2 - x1), int(y2 - y1)
        if w < _MIN_BOX_PX or h < _MIN_BOX_PX:
            return False
        try:
            tracker = _TRACKER_FACTORY()
            tracker.init(frame_bgr, (int(x1), int(y1), w, h))
        except Exception as exc:  # cv2.error など (箱が画像外など)
            self._status(f"Could not initialize the tracker for T{track_id}: {exc}")
            return False
        self._trackers[track_id] = tracker
        self._video_session.transient.setdefault("tracker_frames", {})[track_id] = frame_idx
        seed = next(
            (o for o in self.annotations.get(frame_idx, []) if o.track_id == track_id), None
        )
        if seed is not None:
            self._video_session.transient.setdefault("tracker_seeds", {})[track_id] = copy.deepcopy(
                seed
            )
        self._status(f"{_TRACKER_NAME} tracker initialized for T{track_id}.")
        return True

    def _init_tracker_from_selected(self) -> None:
        if self._selected is None:
            return
        self._init_tracker(self._selected.track_id, self.current_frame_idx, self._selected.bbox)
        self._render_canvas()

    def _run_trackers_for_frame(self, frame_idx: int, frame_bgr: np.ndarray) -> None:
        """Keep the legacy A shortcut, with worker-owned tracker updates."""
        self._queue_tracker_step(frame_idx, frame_bgr)

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

    def _on_auto_track_toggle(self, sender=None, app_data=None) -> None:
        """dearpygui-signature shim over :meth:`_set_auto_track`."""
        self._set_auto_track(app_data)

    def _on_obb_mode_toggle(self, sender=None, app_data=None) -> None:
        """Legacy hook for the old hidden OBB-mode checkbox.

        The mode is a visible tool now (``Draw Rotated Box``), so this routes
        to :meth:`set_tool` rather than flipping a flag the rail would then
        contradict.
        """
        self.set_tool("obb" if bool(app_data) else "draw")

    def _toggle_auto_track_key(self) -> None:
        """Ctrl+T. Works whether or not the Settings window has been opened."""
        self._set_auto_track(not self._auto_track)
        if dpg.does_item_exist("auto_track_cb"):
            dpg.set_value("auto_track_cb", self._auto_track)

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
        with dpg.window(
            label=title,
            tag="label_dlg",
            modal=True,
            width=320,
            height=140,
            no_resize=True,
        ):
            dpg.add_input_text(tag="label_dlg_input", width=-1, hint="New label name")
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add", width=100, callback=confirm_cb)
                dpg.add_button(
                    label="Cancel", width=100, callback=lambda: dpg.delete_item("label_dlg")
                )
        widgets.place_dialog("label_dlg", 320, 140)

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
            dpg.set_value(
                "classes_display", "\n".join(f"  [{i}] {n}" for i, n in enumerate(self.class_names))
            )
        if dpg.does_item_exist("actions_display"):
            dpg.set_value("actions_display", self._behavior_display_text())

    # ═══════════════════════════════════════════════════════════════════
    # Sidebar Refresh
    # ═══════════════════════════════════════════════════════════════════

    def _refresh_sidebar(self) -> None:
        """Push live state into every panel that shows it.

        Kept under its old name because roughly a dozen call sites invoke it
        after a mutation, and renaming it would have been churn with no reader
        benefit. What it refreshes changed completely: the eleven-panel sidebar
        is gone, so this now drives the Inspector, the object list that moved
        into it, the save chip and the timeline.

        The selected box's own fields are *not* written here any more. The
        Inspector owns them and fills them in :meth:`show`, so there is one
        writer per widget instead of two that could disagree.
        """
        self._refresh_candidate_ui()
        self._refresh_inspector()
        self._refresh_save_chip()

        if dpg.does_item_exist("obj_listbox"):
            objects = self.annotations.get(self.current_frame_idx, [])
            rows, selected_row = [], None
            for obj in objects:
                action = self._behavior_text(obj, "-")
                mark = "*" if obj.track_id in self._trackers else " "
                flags = " [occluded]" if obj.occluded else ""
                name = (
                    f"T{obj.track_id:<3}"
                    if self.use_track_ids
                    else (
                        self.class_names[obj.class_id]
                        if 0 <= obj.class_id < len(self.class_names)
                        else "box"
                    )
                )
                rows.append(f"{mark} {name} {action}{flags}")
                if obj is self._selected:
                    selected_row = rows[-1]
            self._obj_rows = rows
            dpg.configure_item("obj_listbox", items=rows)
            if selected_row is not None:
                dpg.set_value("obj_listbox", selected_row)

        self._refresh_labels_display()
        self._render_timeline()

    # ─── Settings dialog wiring ──────────────────────────────────────

    def _settings_callbacks(self) -> Dict[str, Callable]:
        """What the Settings dialog can invoke, by name.

        The same shape as :meth:`_inspector_callbacks`: the dialog is a view
        with no knowledge of the annotator, and a missing name is a no-op
        rather than an error.
        """
        return {
            "end_of_video": self._set_end_behavior,
            "continue_method": self._set_continue_method,
            "guided_help": self.show_onboarding,
            "set_project_dir": self._set_project_dir_dialog,
            "add_class": self._add_class_label_dialog,
            "add_action": self.add_behavior_dialog,
            "edit_behaviors": self.open_behavior_editor,
            "auto_track": self._set_auto_track,
            "use_track_ids": self.set_use_track_ids,
            "label_mode": self.set_label_mode,
            "sticky_draw": self._set_sticky_draw,
            "click_backend": self.set_click_backend,
            "pick_weights": self._pick_model_checkpoint,
            "model_changed": self._on_model_changed,
            "reset_trackers": self._reset_all_trackers,
            "reset_selected_tracker": self._reset_selected_tracker,
            "quality_changed": self._on_quality_changed,
            "recheck": self._recheck_video,
            "closed": self._on_settings_closed,
        }

    def open_settings(self, tab: str = "") -> None:
        settings = self._ensure_project().settings
        self.settings.load(
            config=self.quality_config,
            project_dir=self.project_dir,
            auto_track=self._auto_track,
            sticky_draw=self._sticky_draw,
            use_track_ids=self.use_track_ids,
            weights=settings.get("weights", ""),
            config_path=settings.get("config", ""),
            label_mode=self.label_mode,
        )
        self._refresh_model_ui()
        dpg.set_value(
            "settings_end_behavior",
            (
                "Open next unfinished video"
                if self.project.settings.get("end_of_video") == "next_unfinished"
                else "Stay on last frame"
            ),
        )
        dpg.set_value(
            "settings_continue_method", self.project.settings.get("continue_method", "Automatic")
        )
        self.settings.show(tab)

    def _on_settings_closed(self) -> None:
        self._refresh_inspector()
        self._render_canvas()

    def _set_auto_track(self, value: Any) -> None:
        """Auto-track, from the Settings checkbox.

        Kept separate from ``_on_auto_track_toggle`` because that one is a
        dearpygui callback signature and this takes a plain bool, so the key
        binding and the checkbox can share one implementation.
        """
        if value and not self._require_track_ids("Following boxes between frames"):
            if dpg.does_item_exist("auto_track_cb"):
                dpg.set_value("auto_track_cb", False)
            return
        self._auto_track = bool(value)
        if not self._auto_track:
            self._trackers.clear()
        elif _TRACKER_FACTORY is None:
            self._status("Auto-track: " + _TRACKER_HINT)
            return
        self._status(f"Auto-track is {'on' if self._auto_track else 'off'}.")

    def _set_sticky_draw(self, value: Any) -> None:
        self._sticky_draw = bool(value)
        if dpg.does_item_exist("rail_sticky"):
            widgets.set_toolbar_active("rail_sticky", self._sticky_draw)
        self._status(
            "Keep drawing is on - the draw and click tools stay active after each box."
            if self._sticky_draw
            else "Keep drawing is off - the tool returns to Select after each box."
        )

    def toggle_sticky_draw(self) -> None:
        """The rail's Keep-drawing button, kept in sync with Settings."""
        self._set_sticky_draw(not self._sticky_draw)
        if dpg.does_item_exist("sticky_draw_cb"):
            dpg.set_value("sticky_draw_cb", self._sticky_draw)

    def _on_quality_changed(self) -> None:
        """A threshold moved. Adopt it, but do not re-run the detectors yet.

        Re-checking a 40,000-frame run on every slider tick would make the
        sliders unusable, so the dialog offers an explicit *Re-check the video*
        and says so in its note.
        """
        self.quality_config = self.settings.quality_config(self.quality_config)

    def _recheck_video(self) -> None:
        self.quality_config = self.settings.quality_config(self.quality_config)
        self.rebuild_copilot()
        self._status(f"Re-checked. {self._queue.summary()}.")

    def _update_active_track_color(self) -> None:
        if not dpg.does_item_exist("active_track_color_label"):
            return
        red, green, blue, _alpha = _track_color(self.active_track_id)
        dpg.configure_item(
            "active_track_color_label",
            color=(red, green, blue, 255),
            default_value=f"[{self.active_track_id}]",
        )

    def _on_active_tid_change(self, sender=None, app_data=None) -> None:
        """dearpygui-signature shim over :meth:`_on_active_track_change`."""
        self._on_active_track_change(app_data)

    def _on_list_select(self, sender=None, app_data=None) -> None:
        """Select the box whose row was clicked.

        Matched by **position**, not by rebuilding the row's text. The old
        handler re-derived a label with different spacing and a different
        placeholder from the one the list actually rendered, so the comparison
        never matched and clicking a row silently did nothing. The rows are
        built once in ``_refresh_sidebar`` and remembered here.
        """
        if app_data is None:
            return
        rows = getattr(self, "_obj_rows", [])
        objects = self.annotations.get(self.current_frame_idx, [])
        if app_data in rows:
            index = rows.index(app_data)
            if 0 <= index < len(objects):
                self._select(objects[index])

    def _require_track_ids(self, what: str) -> bool:
        """Refuse a track-only action, and say where the switch is (§35)."""
        if self.use_track_ids:
            return True
        self._status(
            f"{what} needs track ids, which this project has switched off. "
            "Project > Tracking... turns them back on."
        )
        return False

    def _on_active_track_change(self, value) -> None:
        """The track id new boxes are given, from the Inspector's field."""
        if not self.use_track_ids:
            return
        try:
            self.active_track_id = max(0, int(value))
        except (TypeError, ValueError):
            return
        self.next_track_id = max(self.next_track_id, self.active_track_id)
        self._update_active_track_color()

    def _prompt_track_id_dialog(self) -> None:
        if self._selected is None or not self._require_track_ids("Track ids"):
            return
        if dpg.does_item_exist("tid_dlg"):
            dpg.delete_item("tid_dlg")
        current = self._selected.track_id
        with dpg.window(
            label="Set Track ID",
            tag="tid_dlg",
            modal=True,
            width=280,
            height=130,
            no_resize=True,
        ):
            dpg.add_input_int(
                tag="tid_dlg_input", default_value=current, min_value=0, max_value=9999, width=-1
            )
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Set", width=100, callback=self._on_track_id_dialog_confirm)
                dpg.add_button(
                    label="Cancel", width=100, callback=lambda: dpg.delete_item("tid_dlg")
                )
        widgets.place_dialog("tid_dlg", 280, 130)

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
        if not self._ensure_project().videos:
            self._status("No video loaded.")
            return
        # プロジェクトディレクトリが指定されていれば、静かに annotations.json を上書き。
        if self.project_dir is not None:
            path = self.project_dir / "annotations.json"
            self._write_annotation(path)
            return
        # 未設定なら従来通りダイアログで保存先を選ばせる。
        initial = self.annotation_path.name if self.annotation_path else "annotations.json"
        path = _tk_save_file(
            title="Save Annotation",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_file=initial,
        )
        if path is not None:
            self._write_annotation(path)

    def _save_as_dialog(self) -> None:
        """プロジェクトディレクトリを設定していても、明示的にダイアログで保存先を選ぶ。"""
        if not self._ensure_project().videos:
            self._status("No video loaded.")
            return
        initial = self.annotation_path.name if self.annotation_path else "annotations.json"
        path = _tk_save_file(
            title="Save Annotation As",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initial_file=initial,
        )
        if path is not None:
            self._write_annotation(path)

    def _set_project_dir_dialog(self) -> None:
        """メニューからプロジェクトフォルダを設定/移動する。

        設定した時点でフォルダ構造 (datasets/ runs/ analysis/ ...) を作り、
        マニフェストを書き出す。「後で保存すれば作られる」では、その間に走った
        学習や解析の出力が別の場所に残ってしまうため。
        """
        p = _tk_ask_dir(title="Select the project folder")
        if p is None:
            return
        from .annotation.layout import PROJECT_FILE

        self.project_dir = Path(p)
        self._sync_project()
        try:
            self.project.save(self.project_dir / PROJECT_FILE)
        except OSError as exc:
            self._status(f"Could not create the project folder: {exc}")
            return
        self.annotation_path = self.project_dir / "annotations.json"
        self._refresh_project_dir_display()
        self._refresh_workspace()
        self._status(f"Project folder set to {p}")

    def _open_folder(self, folder) -> None:
        """Show ``folder`` in the system file browser.

        Split out of :meth:`_open_project_folder` because the Activity window
        needs the same three platform branches to reach a run folder, and two
        copies of "which command opens a directory here" is two places to fix
        the day one of them is wrong.
        """
        import os
        import subprocess
        import sys

        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # noqa: S606 - a folder the user owns
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            self._status(f"Could not open {folder}: {exc}")

    def _open_project_folder(self) -> None:
        """Show the project folder in the system file browser."""
        if self.project_dir is None:
            self._status("No project folder yet. Use File > Set Project Folder.")
            return
        self._open_folder(self.project_dir)

    def _refresh_project_dir_display(self) -> None:
        # Setting the folder is what unblocks Train, so the gate is re-read
        # here. Without it the button stayed dimmed, still tooltipped "this
        # project has no folder yet", after the user had just given it one.
        self.refresh_loop_gates()
        if dpg.does_item_exist("project_dir_text"):
            dpg.set_value(
                "project_dir_text",
                str(self.project_dir) if self.project_dir else "(not set)",
            )

    def _write_annotation(self, path: Path) -> None:
        self._save_workspace(Path(path))

    def _save_and_export_dialog(self) -> None:
        """Export frames + annotations as a dataset, inside the project by default.

        ``datasets/frames`` rather than the project root: exported frames are
        training input, they can be regenerated, and there can be several of
        them. Mixing them into the root is how a project folder stops being
        readable after the third export.
        """
        from .annotation.layout import ProjectLayout

        out = ProjectLayout(self.project_dir).frames if self.project_dir else None
        if out is None:
            picked = _tk_ask_dir(title="Select Output Directory")
            out = Path(picked) if picked is not None else None
        if out is not None:
            self.export_project(Path(out))

    def _load_annotation_dialog(self) -> None:
        path = _tk_open_file(
            title="Load Annotation JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is not None:
            self._load_annotation_file(path)

    def _load_annotation_file(self, path: Path) -> None:
        self.open_project(Path(path))

    def _status(self, msg: str) -> None:
        """Say one sentence, and keep it.

        The status line holds exactly one message and forgets it the moment
        the next one arrives, which is right for the line and wrong for the
        session: the sentence that explains why a twenty-minute round failed
        used to survive until the next mouse move. Every message now also
        lands in the Activity log, classified by what it reads like, so it can
        be found again.
        """
        if dpg.does_item_exist("status_text"):
            dpg.set_value("status_text", msg)
        log = getattr(self, "activity_log", None)
        if log is None:
            return
        kind = activity_mod.classify(msg)
        log.record(msg, kind)
        # Only a problem redraws the chip here. ``_status`` is called on every
        # frame change ("Frame 812/108000 | Annotated: ..."), and re-labelling
        # and re-theming a button sixty times a second to say the same word is
        # the kind of cost this release exists to remove. Everything else the
        # chip reports moves on the order of seconds and is picked up by
        # :meth:`_tick_activity`; an error is the one thing that must not wait
        # even half of one.
        if kind in (activity_mod.WARN, activity_mod.ERROR):
            self._refresh_activity_pill()

    def _ctrl_s_handler(self) -> None:
        if dpg.is_key_down(_KEY_CTRL):
            self.save_annotation()

    def _show_help(self) -> None:
        if dpg.does_item_exist("help_win"):
            dpg.delete_item("help_win")
        from .gui.workspace_help import help_text

        with dpg.window(label="Help / Shortcuts", tag="help_win", width=720, height=650):
            # The help text is longer than any window it will ever be shown in,
            # so it scrolls and Close does not.
            with widgets.dialog_body("help_body"):
                dpg.add_text(help_text(), wrap=660)
            widgets.dialog_footer()
            dpg.add_button(
                label="Close", width=-1, height=32, callback=lambda: dpg.delete_item("help_win")
            )
        widgets.place_dialog("help_win", 720, 650)

    # ═══════════════════════════════════════════════════════════════════
    # The Annotation Copilot
    # ═══════════════════════════════════════════════════════════════════
    #
    # AI proposes -> human reviews -> human corrects -> AI re-propagates.
    #
    # Everything below obeys one rule, inherited from the candidate lifecycle
    # that was already here (docs/GUI_DESIGN.md P1): **a machine proposal never
    # reaches self.annotations, and therefore never reaches disk, until a human
    # accepts it.** Smart sampling, tracklets, the review queue and behavior
    # intervals are all views over the pending set; accepting is the only act
    # with consequences, and it has exactly one implementation.

    @staticmethod
    def _show_item(tag: str, visible: bool) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=bool(visible))

    # ─── Undo and redo ───────────────────────────────────────────────
    #
    # An undo entry is a **set of frame snapshots**, not one frame's list. The
    # first version stored one list, which was right for drawing and wrong for
    # everything else: Swap IDs rewrites a track across hundreds of frames, and
    # a one-frame snapshot would have restored the frame the user happened to
    # be on and left the other three hundred swapped. A half-undone swap is
    # worse than no undo at all, because the user believes it was taken back.
    #
    # Each entry also records where the user was, so undoing something that
    # happened elsewhere takes them there rather than silently changing a frame
    # they cannot see.

    def _pump_ui(self) -> None:
        """Run whatever the analysis thread queued, on the main thread.

        A worker thread must never touch dearpygui: dpg is not thread-safe and
        a cross-thread ``set_value`` corrupts the item tree in ways that
        surface minutes later somewhere else (GUI_DESIGN P9). The worker puts
        closures here; this drains them between frames.
        """
        for _ in range(32):  # bounded, so a flood cannot stall a frame
            try:
                job = self._ui_queue.get_nowait()
            except Exception:
                return
            try:
                job()
            except Exception as exc:
                self._status(f"Background update failed: {type(exc).__name__}: {exc}")

    def _post(self, job: Callable[[], None]) -> None:
        """Called from the worker thread. The only cross-thread channel."""
        self._ui_queue.put(job)

    # ─── Capability gating ───────────────────────────────────────────

    def _capability(self):
        return auto_label.probe(_repo_root())

    # ─── The project's detector checkpoint ───────────────────────────

    def _pick_model_checkpoint(self) -> None:
        path = _tk_open_file(
            "Detector checkpoint", [("Checkpoint", "*.pth *.pt"), ("All files", "*.*")]
        )
        if path is None:
            return
        if dpg.does_item_exist(self.settings._t("weights")):
            dpg.set_value(self.settings._t("weights"), str(path))
        self._on_model_changed()

    def _on_model_changed(self) -> None:
        """The Model tab edited the checkpoint; the project now owns it."""
        weights, config = self.settings.model_paths()
        self._ensure_project().settings.update(weights=weights, config=config)
        self._mark_dirty()
        self._refresh_model_ui()

    def _refresh_model_ui(self) -> None:
        """Say the same thing about the model everywhere it is said.

        The top bar chip, the Model tab's status line, Smart Annotate's locked
        options and the project batch's gate all answer "is there a model?",
        and they answer it from this one probe so they cannot disagree.
        """
        settings = self._ensure_project().settings
        weights = str(settings.get("weights", "") or "")
        try:
            status = auto_label.check_backend(_repo_root(), weights)
        except Exception:
            return
        name = Path(weights).name if weights else ""
        if dpg.does_item_exist("top_model"):
            widgets.status_pill(
                "top_model",
                f"Model: {name}" if name else "Model: none",
                "ok" if status.ok else "idle",
            )
        self.settings.set_model_status(status.message, status.ok)
        self._refresh_capability_gates()
        dialog = getattr(self, "batch_dialog", None)
        if dialog is not None and dpg.does_item_exist(dialog.tag):
            dialog.gate()

    def _refresh_capability_gates(self) -> None:
        """Lock what this machine cannot do, with the reason (P4)."""
        try:
            cap = self._capability()
        except Exception:
            return
        predict_reason = ""
        if not cap.torch:
            predict_reason = (
                cap.torch_reason
                or "PyTorch is not installed in this Python, so the " "detector cannot run here."
            )
        elif not cap.package:
            predict_reason = cap.package_reason
        elif not self._ensure_project().settings.get("weights"):
            # Not a deficiency: this option is the accelerator that arrives
            # *after* a model exists, and labelling does not wait for it. The
            # wording says so, and names what proposes boxes in the meantime,
            # so an empty checkbox does not read as a blocked workflow.
            predict_reason = (
                "Detector pre-labelling needs a trained checkpoint, and this "
                "project has none yet - so this option turns on later and "
                "nothing here waits for it.\n"
                "Train, in the top bar, makes one from the labels you already "
                "have, and sets it for you.\n"
                "Until then the boxes come from the segmentation helper: press C "
                "to click an animal, or select a box and press Continue."
            )
        self.refresh_loop_gates()
        self.smart_dialog.set_capability(
            gpu_reason=(
                ""
                if cap.torch
                else "PyTorch is not installed here, so there is no GPU path. "
                "Frame scanning still runs on the CPU."
            ),
            predict_reason=predict_reason,
        )

    # ─── Experiment-specific quality control (§14) ───────────────────

    def _on_list_select_value(self, value: Any) -> None:
        """The Inspector's object list hands over the row it selected."""
        self._on_list_select(None, value)

    def _find_all_clicked(self) -> None:
        """*Find every ``<class>`` in this frame*, from the Inspector.

        The word comes from the class the Inspector is set to, which is the
        word printed on the button. There used to be a text box above it whose
        placeholder was the same word; typing it and leaving it blank produced
        the same search, so it was a field that could only be answered wrong.
        """
        self.find_all_in_frame(self.active_class_name())

    def active_class_name(self) -> str:
        """The class new boxes get, as a word. ``"animal"`` if none is set."""
        names = self.class_names
        if not names:
            return "animal"
        index = min(max(getattr(self, "active_class_id", 0), 0), len(names) - 1)
        return str(names[index])

    # ─── Auto pre-label: importing an existing model's output (§9) ───

    def import_predictions(self, path: Path) -> bool:
        """Load ``predictions.json`` as **candidates**, and build the copilot.

        This is auto pre-labelling through the path the suite already has: the
        analysis page runs the detector over a whole video, writes a
        timestamped ``predictions.json``, and hands it here. Nothing is
        inferred in this process, which is what lets the annotator run on a
        Python with no torch.

        Everything that arrives is **pending** (P1). It is drawn dashed and
        amber, it is grouped into tracklets, it is checked by every detector in
        :mod:`htrtdetr.annotation.quality`, and none of it reaches
        ``annotations.json`` until a human accepts it.

        ``--import-predictions`` on the command line calls this, which is the
        flag ``analysis_panel``'s *Send Predictions to Annotator* button has
        been passing all along - the annotator simply never accepted it, so the
        hand-off died with an argparse error before this.
        """
        try:
            frames = auto_label.load_candidates(Path(path))
        except Exception as exc:  # noqa: BLE001
            self._status(f"Could not read those predictions: {exc}")
            return False

        total = sum(len(v) for v in frames.values())
        if not total:
            self._status(
                f"{Path(path).name} has no detections in it. Lower the run's "
                f"score threshold and predict again."
            )
            return False

        self._review_session().set_candidates(frames)
        # Keep the Click to Box model backend pointed at the same file, so the
        # two features cannot disagree about what the model said.
        self._predictions = frames
        self._predictions_path = Path(path)
        self._refresh_backend_ui()

        self._mark_dirty()
        self.rebuild_copilot()
        self._refresh_candidate_ui()
        self._render_canvas()

        issues = len(self._queue)
        clean = total - self._flagged_candidate_count()
        self._status(
            f"{total:,} predictions on {len(frames):,} frames. "
            + (
                f"{clean:,} look consistent; {issues:,} "
                f"location{'s' if issues != 1 else ''} need review - press "
                f"Review, or Ctrl+Right to step through the candidates."
                if issues
                else "No checks objected. Press Y to accept a frame, or Review to "
                "look through them."
            )
        )
        return True

    def _flagged_candidate_count(self) -> int:
        """How many pending candidates fall inside a review item.

        The complement is what the summary calls *consistent*, and it has to be
        counted rather than assumed: saying "18,332 are consistent" when the
        number is really the total minus the number of *items* would overstate
        the model every time an item spans forty frames.
        """
        session = self._review
        if session is None:
            return 0
        flagged: set = set()
        for item in self._queue.items:
            for frame in range(item.start, item.end + 1):
                for index, cand in enumerate(session.candidates(frame)):
                    if item.track_ids and cand.track_id not in item.track_ids:
                        continue
                    flagged.add((frame, index))
        return len(flagged)

    def _import_predictions_dialog(self) -> None:
        path = _tk_open_file(
            title="Load Predictions as Candidates",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path is not None:
            self.import_predictions(path)

    # ─── Smart Annotate ──────────────────────────────────────────────

    #: The two ways into the one engine window, as
    #: ``(caption, pre-label, strategy)``. They are **presets**, not modes: the
    #: window, the engine and the progress display are the same, and a preset
    #: only decides where the form starts.
    #:
    #: A caption naming a control the window is about to hide would be a lie,
    #: which is what retired the old *Pre-label videos with the model* entry.
    #: ``analyze`` cannot tell that lie because it is locked until a checkpoint
    #: exists - and once one does, every control it names is on screen.
    SMART_PRESETS: Dict[str, Tuple[str, bool, str]] = {
        "sample": ("Pick frames to label", False, ""),
        # Diversity + uncertainty, not uncertainty alone. A detector that has
        # seen two hundred frames of one species is not calibrated enough for
        # its confidence to rank a million frames on its own: pure uncertainty
        # piles the picks onto one hard-looking situation. The combined
        # strategy spreads them over the frames it found hardest, which is what
        # ``_spread_over_top`` was written for.
        "analyze": ("Find animals with the model", True, sampling_mod.DIVERSITY_UNCERTAINTY),
    }

    def open_smart_annotate(self, scope: str = "", preset: str = "sample") -> None:
        """The one dialog, opened on the scope and preset the entry point means.

        No video needs to be open: this runs over a scope, and the project
        entries are reached from Project Videos without opening anything.
        """
        project = self._ensure_project()
        if scope == "project" or (not scope and self.cap is None):
            scope = "Selected videos" if project.selected_videos else "Entire project"
        elif not scope:
            scope = "Current video"
        self._refresh_capability_gates()
        caption, prelabel, strategy = self.SMART_PRESETS.get(
            preset, self.SMART_PRESETS["sample"]
        )
        if prelabel and not project.settings.get("weights"):
            self._status(self.analyze_blocked())
            return
        name, frames, fps, videos = self.scope_summary(scope)
        if not videos:
            self._status("No videos in this scope. Add or select videos in Project Videos.")
            return
        self.smart_dialog.show(
            video_name=name,
            num_frames=frames,
            fps=fps,
            videos=videos,
            scope=scope,
            sample=True,
            prelabel=prelabel or None,
            strategy=strategy,
            title=caption,
        )

    def analyze_blocked(self) -> str:
        """Why *Find animals with the model* is off. ``""`` when it is not.

        Running a detector needs a detector, and in this window the way to get
        one is one button to the left - so the sentence says that rather than
        sending the user to a file browser for a checkpoint they do not have.
        """
        if not self._ensure_project().settings.get("weights"):
            return (
                "No model yet, so there is nothing to find animals with. "
                "Train makes one from the labels you already have."
            )
        try:
            cap = self._capability()
        except Exception:  # noqa: BLE001
            return ""
        if not cap.torch:
            return cap.torch_reason
        return ""

    def start_analysis(self, request, options: Dict[str, Any]) -> None:
        from .annotation.batch import BatchOptions

        project = self._sync_project()
        weights = options.get("weights") or project.settings.get("weights", "")
        detect = bool(options.get("generate_candidates"))
        if detect and not weights:
            self.smart_dialog.fail(
                "No detector checkpoint yet. Train makes one from the labels "
                "you already have, or turn off Generate annotation candidates "
                "and pick frames to label by hand."
            )
            return
        stride = max(1, int(options.get("stride", 15)))
        sampling_on = bool(options.get("sample", True))
        # Detect sparsely exactly when the detections are for *choosing*
        # frames. The sampler can only ever pick a scanned frame, so running
        # the detector on the fourteen frames between two scan positions is
        # work whose result is cached and never read: measured at 121 frames/s,
        # 20 one-hour videos cost 4.1 hours dense against 17 minutes at stride
        # 15. A pre-label-only run is the other intent - the user means to
        # review the whole video - so that stays dense.
        detect_stride = stride if sampling_on else 1
        batch = BatchOptions(
            detect=detect,
            weights=weights,
            config=project.settings.get("config", ""),
            device="cuda" if options.get("use_gpu") else "cpu",
            score_threshold=float(options.get("score_threshold", 0.3)),
            # Derived, not asked. A project that follows individuals has no
            # reason to leave its detections unconnected, and one that does
            # not has nothing to connect them into - so the checkbox that used
            # to ask was a question with one right answer either way.
            connect_tracks=bool(project.settings.get("use_track_ids", True)),
            sample=sampling_on,
            stride=stride,
            detect_stride=detect_stride,
            request=request,
            balanced=options.get("balanced", True),
            quality=self.quality_config,
        )
        self._smart_running = True
        if self.start_project_batch(options.get("scope", "Current video"), batch):
            self._analysis_thread = self._batch_thread
        else:
            self._smart_running = False
            self.smart_dialog.fail("Could not start. Check the video scope and detector settings.")

    def stop_analysis(self) -> None:
        self._analysis_stop = True
        self.stop_project_batch()

    def apply_selection(self, selection) -> None:
        """Take the user to the first sample and show the plan on the timeline."""
        selections = getattr(self, "_smart_selections", {})
        origin = next((vid for vid, value in selections.items() if value is selection), None)
        if origin and origin != self._video_session.video_id and not self.switch_video(origin):
            return
        self._selection = selection
        self._render_timeline()
        if selection.frames:
            self.goto_frame(int(selection.frames[0]))
        elif selection.clips:
            self.goto_frame(int(selection.clips[0].start))
        # Say how to reach the other ninety-eight. "They are marked on the
        # timeline" was true and useless: at an hour a video the samples lane
        # is a row of dots eighty-six frames wide apiece.
        key = shortcuts.label_for("review.next_pending_frame")
        self._status(
            f"Sample 1 of {len(selection.frames):,}"
            + (f" and {len(selection.clips):,} clips" if selection.clips else "")
            + f". {key} for the next one, or click a mark on the timeline."
        )

    # ─── Building the copilot views ──────────────────────────────────
    #
    # The rebuild is a **whole-video** computation: nine detectors over every
    # pending candidate, plus the tracklet index and the interval merge.
    # Measured on this project's own scale - one hour at 30 fps, detections
    # every 15th frame, four animals, so 7,200 frames and 28,800 boxes - it
    # costs about 0.8 s, and it was being run once per accepted candidate.
    # Holding Y down through fifty frames therefore bought fifty full
    # re-analyses of the same video: forty seconds of frozen window for a
    # result that only the last one of them produced.
    #
    # So accepting schedules instead of rebuilding. The canvas, the saved
    # annotations and the undo entry are all updated synchronously - those are
    # microseconds and they are what the user is looking at - and the analysis
    # that feeds the Review Inbox and the timeline catches up once the burst
    # of clicks has stopped. One rebuild per burst, not one per click.

    #: How long to wait after the last change before re-analysing. Long
    #: enough that a run of accepts coalesces into one rebuild, short enough
    #: that a single accept feels like it landed before the user has finished
    #: looking at the result.
    COPILOT_QUIET = 0.35

    def schedule_copilot_rebuild(self, delay: Optional[float] = None) -> None:
        """Mark the copilot stale. :meth:`_tick_copilot` does the work."""
        self._copilot_dirty = True
        self._copilot_due = time.monotonic() + (
            self.COPILOT_QUIET if delay is None else float(delay)
        )

    def _tick_copilot(self) -> None:
        """Run a scheduled rebuild once the user has stopped changing things."""
        if not getattr(self, "_copilot_dirty", False):
            return
        if time.monotonic() < getattr(self, "_copilot_due", 0.0):
            return
        self.rebuild_copilot()

    def rebuild_copilot(self) -> None:
        """Recompute tracklets, the review queue and behavior intervals.

        The immediate form, and still the right one for anything that changes
        the pending set wholesale - importing predictions, undo, *Re-check
        everything*. The per-candidate paths call
        :meth:`schedule_copilot_rebuild` instead.

        Decisions already made are carried across by content key, so "correct
        a few, re-analyse, carry on" does not reset the review (§13).
        """
        self._copilot_dirty = False
        candidates = self._candidate_map()
        cfg = self.quality_config

        parts = tracklet_mod.build_tracklets(candidates, max_gap=cfg.fragment_gap)
        items = quality_mod.analyze(
            candidates,
            config=cfg,
            frames=self._examined_frames(),
            enabled=quality_mod.detector_keys(use_track_ids=self.use_track_ids),
            tracklets=parts,
            num_frames=self.num_frames,
        )

        # Attach each tracklet's issues to it, so Accept tracklet can warn
        # before it spends one click on three hundred boxes.
        by_track: Dict[int, set] = {}
        for item in items:
            for track_id in item.track_ids:
                by_track.setdefault(track_id, set()).add(item.kind)
        parts = [p.with_issues(by_track.get(p.track_id, ())) for p in parts]
        self._tracklets = tracklet_mod.TrackletIndex(parts)

        from dataclasses import replace

        video_id = self._video_session.video_id
        items = [replace(i, video_id=video_id) for i in items]
        carried = self._queue.decisions()
        carried = {(video_id, *k) if video_id and len(k) == 4 else k: v for k, v in carried.items()}
        self._queue = review_queue_mod.ReviewQueue(items)
        self._queue.carry_over(carried)

        self._intervals = self._rebuilt_intervals(candidates)

        self.inbox.set_queue(self._queue, fps=self.fps)
        self._render_timeline()
        self._refresh_inspector()

    def _inconsistency_map(self) -> Dict[int, float]:
        """Per-frame temporal-inconsistency scores for Active Learning.

        Derived from the review queue rather than recomputed: the detectors
        have already found the inconsistent moments, and a second measure of
        the same thing would be a second thing to keep in step.
        """
        out: Dict[int, float] = {}
        temporal = {
            quality_mod.POSITION_JUMP,
            quality_mod.SIZE_JUMP,
            quality_mod.ORIENTATION_JUMP,
            quality_mod.TRACK_FRAGMENT,
            quality_mod.ID_SWITCH_RISK,
        }
        for item in self._queue.items:
            if item.kind not in temporal:
                continue
            for frame in range(item.start, item.end + 1):
                out[frame] = max(out.get(frame, 0.0), item.severity)
        return out

    # ─── Review Inbox ────────────────────────────────────────────────

    def toggle_panels(self) -> None:
        """Give the whole width to the image, or take it back."""
        self._panels_hidden = not self._panels_hidden
        self._show_item("right_col", not self._panels_hidden)
        widgets.set_toolbar_active("rail_panels", not self._panels_hidden)
        self._on_viewport_resize()
        self._status(
            "Panels hidden. Press Panels again to bring them back."
            if self._panels_hidden
            else "Panels shown."
        )

    def toggle_inbox(self) -> None:
        """Swap the right column between the Inspector and the Review Inbox."""
        showing = dpg.does_item_exist(self.inbox.tag) and dpg.is_item_shown(self.inbox.tag)
        self._show_item(self.inbox.tag, not showing)
        self._show_item(self.inspector.tag, showing)
        widgets.set_toolbar_active("top_review", not showing)
        if not showing:
            self.inbox.refresh()

    def jump_to_issue(self, item) -> None:
        """Select a review item: go there, show it, select what it is about."""
        video_id = getattr(item, "video_id", "")
        if video_id and video_id != self._video_session.video_id:
            if not self.switch_video(video_id):
                return
        self._sel_issue = item
        self._sel_tracklet = None
        self._sel_interval = None
        self._sel_clip = None
        self._queue.select(item)
        self.goto_frame(item.middle())
        # Select the box the item is about, when it names exactly one track, so
        # the user can act on it without hunting for it in a crowded frame.
        if len(getattr(item, "track_ids", ())) == 1:
            track_id = item.track_ids[0]
            for obj in self.annotations.get(self.current_frame_idx, []):
                if obj.track_id == track_id:
                    self._selected = obj
                    break
        self._refresh_inspector()

    def _on_timeline_marker(self, payload) -> None:
        """A click on a timeline marker: ``(start, end, kind)``."""
        start, end, kind = payload
        for item in self._queue.items:
            if (item.start, item.end, item.kind) == (start, end, kind):
                self.jump_to_issue(item)
                return
        self.goto_frame((start + end) // 2)

    def accept_issue_kind(self, kind: str) -> None:
        n = self._queue.accept_kind(kind)
        for item in self._queue.items:
            if item.kind == kind:
                self._accept_item_candidates(item)
        label = quality_mod.ISSUE_LABELS.get(kind, kind).lower()
        self._status(f"Accepted {n} {label} item{'s' if n != 1 else ''}.")
        self.inbox.refresh()
        self._render_timeline()

    def _accept_targets(self, targets: Sequence[Tuple[int, int]]) -> int:
        """Accept ``(frame, index)`` candidates into the saved annotations.

        The one path by which a proposal becomes data. Every batch action below
        funnels through it, so there is exactly one place where the promotion
        rule lives.
        """
        session = self._review
        if session is None or not targets:
            return 0
        # Accepting a tracklet writes into every frame it covers, so the undo
        # entry has to cover them all as well.
        self._push_undo(frames={int(f) for f, _i in targets} | {self.current_frame_idx})
        accepted = 0
        by_frame: Dict[int, List[int]] = {}
        for frame, index in targets:
            by_frame.setdefault(int(frame), []).append(int(index))

        for frame, indices in by_frame.items():
            row = session.candidates(frame)
            for index in sorted(set(indices)):
                if not (0 <= index < len(row)):
                    continue
                if session.state(frame, index) != auto_label.PENDING:
                    continue
                cand = row[index]
                obj = AnnObject(
                    bbox=[int(v) for v in cand.bbox],
                    track_id=int(cand.track_id),
                    action_id=int(cand.action_id),
                    keyframe=False,
                    obb=list(cand.obb) if cand.obb is not None else None,
                    source=cand.source or "auto",
                    score=float(cand.score),
                    class_id=cand.class_id,
                    provenance={**cand.provenance, "human_review": "accepted"},
                )
                self.annotations.setdefault(frame, []).append(obj)
                session.accept(frame, index)
                accepted += 1

        if accepted:
            self._mark_dirty()
            # Scheduled, not immediate: a burst of accepts is one analysis.
            self.schedule_copilot_rebuild()
            self._refresh_sidebar()
            self._render_canvas()
        return accepted

    def _pending_targets(self, frames) -> List[Tuple[int, int]]:
        session = self._review
        if session is None:
            return []
        out: List[Tuple[int, int]] = []
        for frame in frames:
            for index in range(len(session.candidates(frame))):
                if session.state(frame, index) == auto_label.PENDING:
                    out.append((int(frame), index))
        return out

    def accept_frame(self) -> None:
        n = self._accept_targets(self._pending_targets([self.current_frame_idx]))
        self._status(f"Accepted {n} candidate{'s' if n != 1 else ''} on this frame.")

    def accept_tracklet(self) -> None:
        """One click for a whole tracklet - the point of the abstraction (§12)."""
        part = self._sel_tracklet or self._tracklet_under_selection()
        session = self._review
        if part is None or session is None:
            self._status("Select a candidate first, then accept its tracklet.")
            return
        targets = [(f, i) for f, i in part.targets() if session.state(f, i) == auto_label.PENDING]
        n = self._accept_targets(targets)
        self._status(
            f"Accepted {n} candidate{'s' if n != 1 else ''} "
            f"across frames {part.start}-{part.end}."
        )

    def reject_tracklet(self) -> None:
        part = self._sel_tracklet or self._tracklet_under_selection()
        session = self._review
        if part is None or session is None:
            return
        n = 0
        for frame, index in part.targets():
            if session.state(frame, index) == auto_label.PENDING:
                session.reject(frame, index)
                n += 1
        self.schedule_copilot_rebuild()
        self._render_canvas()
        self._status(f"Discarded {n} candidate{'s' if n != 1 else ''}.")
        self._mark_dirty()

    def goto_tracklet(self) -> None:
        part = self._sel_tracklet or self._tracklet_under_selection()
        if part is not None:
            self.goto_frame(part.start)

    def accept_clip(self) -> None:
        """Every candidate inside the sampling clip around this frame."""
        span = None
        for clip in getattr(self._selection, "clips", ()) or ():
            if clip.start <= self.current_frame_idx <= clip.end:
                span = (clip.start, clip.end)
                break
        if span is None:
            half = max(1, int(round(1.5 * self.fps)))
            span = (max(0, self.current_frame_idx - half), self.current_frame_idx + half)
        n = self._accept_targets(self._pending_targets(range(span[0], span[1] + 1)))
        self._status(
            f"Accepted {n} candidate{'s' if n != 1 else ''} " f"in frames {span[0]}-{span[1]}."
        )

    def accept_visible(self) -> None:
        n = self._accept_targets(self._pending_targets([self.current_frame_idx]))
        self._status(f"Accepted {n} visible candidate{'s' if n != 1 else ''}.")

    def accept_similar(self, *, size_tolerance: float = 0.55, score_window: float = 0.15) -> None:
        """Accept candidates that look like the selected one (§18).

        Similarity is deliberately conservative: the same track, a comparable
        confidence, and a comparable size. A loose notion of "similar" that
        accepts a hundred boxes the user did not look at is exactly the failure
        this whole design exists to avoid.
        """
        session = self._review
        anchor = self._selected_candidate()
        if session is None or anchor is None:
            self._status("Select a candidate first.")
            return
        _frame, _index, cand = anchor
        area = max(1.0, float(cand.width * cand.height))
        low, high = 1.0 / (1.0 + size_tolerance), 1.0 + size_tolerance
        targets: List[Tuple[int, int]] = []
        for frame in session.frames:
            for index, other in enumerate(session.candidates(frame)):
                if session.state(frame, index) != auto_label.PENDING:
                    continue
                if other.track_id != cand.track_id:
                    continue
                if abs(other.score - cand.score) > score_window:
                    continue
                ratio = max(1.0, float(other.width * other.height)) / area
                if not (low <= ratio <= high):
                    continue
                targets.append((frame, index))
        n = self._accept_targets(targets)
        self._status(
            f"Accepted {n} similar candidate{'s' if n != 1 else ''} " f"on track {cand.track_id}."
        )

    def _accept_item_candidates(self, item) -> int:
        frames = range(item.start, item.end + 1)
        if not item.track_ids:
            return self._accept_targets(self._pending_targets(frames))
        session = self._review
        if session is None:
            return 0
        targets = [
            (f, i)
            for f in frames
            for i, c in enumerate(session.candidates(f))
            if session.state(f, i) == auto_label.PENDING and c.track_id in item.track_ids
        ]
        return self._accept_targets(targets)

    def _reject_item_candidates(self, item) -> int:
        session = self._review
        if session is None:
            return 0
        n = 0
        for frame in range(item.start, item.end + 1):
            for index, cand in enumerate(session.candidates(frame)):
                if session.state(frame, index) != auto_label.PENDING:
                    continue
                if item.track_ids and cand.track_id not in item.track_ids:
                    continue
                session.reject(frame, index)
                n += 1
        if n:
            self.schedule_copilot_rebuild()
            self._render_canvas()
        return n

    def _selected_candidate(self):
        """``(frame, index, Candidate)`` for the candidate under the cursor."""
        session = self._review
        if session is None:
            return None
        index = None
        if self._is_over_canvas():
            index = self._hit_candidate(*self._c2i(*self._get_canvas_mouse()))
        if index is None:
            index = getattr(self, "_sel_candidate_index", None)
        if index is None:
            return None
        row = session.candidates(self.current_frame_idx)
        if not (0 <= index < len(row)):
            return None
        return self.current_frame_idx, index, row[index]

    def _tracklet_under_selection(self):
        anchor = self._selected_candidate()
        if anchor is None:
            if self._selected is not None:
                parts = self._tracklets.for_track(self._selected.track_id)
                return parts[0] if parts else None
            return None
        frame, index, _cand = anchor
        return self._tracklets.of(frame, index)

    # ─── Video propagation (§10, §11) ────────────────────────────────
    #
    #   point or box on frame N
    #       -> SAM 2 video predictor, in the sidecar, never in this process
    #       -> a mask per frame
    #       -> mask_to_obb, with the same rule Click to Box uses
    #       -> Candidates, as one tracklet
    #       -> the quality checks
    #       -> the Review Inbox
    #
    # Optional throughout. With no SAM 2 the button is locked with the reason,
    # and nothing else in YOAKE changes.

    def video_capability(self):
        """What the running helper can do about video, as a value."""
        try:
            from .gui import video_segment as vs
        except ImportError:  # pragma: no cover
            from htrtdetr.gui import video_segment as vs
        return vs.capability(self._sam_client)

    def _refresh_propagation_ui(self) -> None:
        """Keep Continue unlocked, and say what it will actually do.

        It is never locked, and that is correct rather than an oversight: the
        method is resolved at press time and the last rung of the ladder -
        copying the box to the next frame - needs nothing installed. What the
        user cannot otherwise know is *which* rung they are on, so the tooltip
        names the ladder and the status line names the rung that ran.
        """
        tag = "rail_propagate"
        if dpg.does_item_exist(tag):
            widgets.unlock(
                tag,
                "Continue the selected animal into the frames that follow. "
                "Uses accurate propagation when the segmentation helper is "
                "running, fast tracking when OpenCV can, and otherwise copies "
                "the box to the next frame - so it always does something. "
                "Press it again while it is running to stop.",
            )

    def propagate_selection(self) -> None:
        """Carry the selected box - or the last click - across the video.

        The prompt is a **box** when something is selected and a **point**
        otherwise, because those are the two things the user can already have
        given. Asking for a third kind of gesture to use the feature would be a
        new thing to learn for no gain.
        """
        try:
            from .gui import video_segment as vs
        except ImportError:  # pragma: no cover
            from htrtdetr.gui import video_segment as vs

        # Pressing this IS asking to use the helper, so start it here rather
        # than sending the user to Settings to turn on an unrelated setting.
        # _start_sam is lazy and idempotent. It is also asynchronous, so the
        # press is remembered and replayed when the helper answers — a second
        # press would otherwise be a rule to learn from nowhere.
        if not self._sam_ready():
            self._start_sam(on_done=lambda ok: self.propagate_selection() if ok else None)
            if not self._sam_ready():
                self._status(
                    "Starting the segmentation helper. The propagation will "
                    "begin as soon as it answers."
                )
                return

        capability = self.video_capability()
        if not capability.available:
            self._show_unavailable("Video propagation", capability.reason)
            return
        if self.cap is None:
            self._status("Open a video first.")
            return
        if self._propagate_thread is not None and self._propagate_thread.is_alive():
            # Pressing the one control again is how a job this long gets
            # stopped (P6: a stop is a stop, never an error). There was no way
            # to stop it at all, and it can run for minutes. The wording is
            # exact about *when* it stops, because the pass already inside the
            # helper cannot be interrupted from here and promising otherwise
            # would be the second surprise.
            self._propagate_stop = True
            self._status(
                "Stopping the propagation. The pass already running has to "
                "finish first; nothing after it will start."
            )
            return

        obj = self._selected
        if obj is None:
            self._status("Select a box first, then press this to follow it " "across frames.")
            return

        track_id = obj.track_id if obj.track_id >= 0 else self.active_track_id
        request = vs.PropagationRequest(
            frame=self.current_frame_idx,
            box=tuple(float(v) for v in obj.bbox[:4]),
            track_id=int(track_id),
            class_id=obj.class_id,
            action_id=obj.action_id,
            before=vs.DEFAULT_WINDOW,
            after=vs.DEFAULT_WINDOW,
        )

        self._propagate_stop = False
        self._propagate_thread = threading.Thread(
            target=self._propagate_worker,
            args=(request, (self.project, self._video_session, self._sam_client)),
            name="yoake-propagate",
            daemon=True,
        )
        self._propagate_thread.start()
        self._status(f"Following track {track_id} across frames...")

    def _propagate_worker(self, request, context=None) -> None:
        """Export the window, ask the helper, build candidates. Off-thread."""
        try:
            from .gui import video_segment as vs
        except ImportError:  # pragma: no cover
            from htrtdetr.gui import video_segment as vs

        project, video, client = context or (self.project, self._video_session, self._sam_client)
        folder = None
        cap = None
        try:
            first, last = request.window(video.num_frames)
            # Its own capture: seeking one VideoCapture from two threads is not
            # safe, and sharing the annotator's would make the canvas jump.
            cap = cv2.VideoCapture(str(video.path))
            if not cap.isOpened():
                self._post(lambda: self._status("The video could not be opened for propagation."))
                return

            def read_frame(index: int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                return frame if ok else None

            folder, written = vs.export_window(
                read_frame,
                first,
                last,
                should_stop=lambda: self._propagate_stop,
                on_progress=lambda done, total: (
                    self._post(lambda d=done, t=total: self._status(f"Preparing frames {d}/{t}..."))
                    if done % 25 == 0
                    else None
                ),
            )
            if folder is None or written == 0:
                self._post(lambda: self._status("No frames could be prepared for propagation."))
                return

            # SAM 2 propagates one way from the prompt. The window has frames
            # on both sides of it, so it takes two passes - otherwise the
            # `before` half is exported, decoded and then silently discarded.
            # The backward pass is skipped entirely when there is no such half.
            def one_pass(backward: bool):
                req_id = client.propagate(
                    frames_dir=str(folder),
                    first_frame=first,
                    prompt_frame=request.frame,
                    points=[list(p) for p in request.points] or None,
                    box=list(request.box) if request.box else None,
                    obj_id=max(1, int(request.track_id)),
                    backward=backward,
                )
                reply = client.wait_for(req_id, timeout=600.0)
                return None if reply is None else vs.result_from_reply(reply)

            self._post(lambda: self._status("Following the animal forward..."))
            forward = one_pass(False)
            backward = None
            if first < request.frame and not self._propagate_stop:
                self._post(lambda: self._status("Following the animal backward..."))
                backward = one_pass(True)

            if forward is None and backward is None:
                self._post(
                    lambda: self._status("The helper did not answer in time. Nothing was changed.")
                )
                return

            # Forward first, so the prompt frame - which both passes return -
            # resolves to the forward pass.
            result = vs.merge_results(forward, backward)
            self._post(
                lambda r=result, q=request: self._finish_video_propagation(project, video, r, q)
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            self._post(lambda m=message: self._status(f"Video propagation failed. {m}"))
        finally:
            if cap is not None:
                cap.release()
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)

    def _show_unavailable(self, feature: str, reason: str) -> None:
        """Explain a missing optional backend, never raise an ImportError (§35)."""
        tag = "unavailable_win"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        with dpg.window(
            label=feature,
            tag=tag,
            modal=True,
            no_collapse=True,
            width=460,
            height=250,
        ):
            dpg.add_text(f"{feature} is unavailable", color=theme.PENDING)
            theme.bind_h2(dpg.last_item())
            widgets.gap(1)
            dpg.add_text(reason, color=theme.TEXT, wrap=420)
            widgets.gap(2)
            widgets.secondary_button(
                "Close",
                lambda *_: dpg.delete_item(tag),
                width=140,
                height=30,
                tag="unavailable_close",
            )
        widgets.place_dialog(tag, 460, 250)
        self._status(f"{feature} is unavailable.")

    # ─── Visual embedding provider (§5) ──────────────────────────────

    def embedding_provider(self):
        """The best encoder available right now.

        Asked for fresh on every scan rather than cached: the helper can be
        started, stopped or swapped between runs, and a cached provider would
        keep using an encoder that is no longer there.
        """
        try:
            from .gui import embedding as emb
        except ImportError:  # pragma: no cover
            from htrtdetr.gui import embedding as emb
        return emb.provider_for(self._sam_client)

    # ─── Correction re-propagation (§16) ─────────────────────────────

    def repropagate(self, correction, scope) -> None:
        """Carry one correction to the frames around it."""
        track_id = correction.track_id
        boxes: Dict[int, Tuple[Sequence[float], Optional[Sequence[float]]]] = {}
        keyframes: List[int] = []
        for frame, objects in self.annotations.items():
            for obj in objects:
                if obj.track_id != track_id:
                    continue
                boxes[frame] = (list(obj.bbox), list(obj.obb) if obj.obb else None)
                if obj.keyframe:
                    keyframes.append(frame)

        plan = propagation_mod.plan(
            correction,
            scope,
            boxes=boxes,
            num_frames=self.num_frames,
            keyframes=keyframes,
            protected=keyframes,
            frame_size=(self.frame_w, self.frame_h),
            min_side=_MIN_BOX_PX,
        )

        if plan.edits:
            self._push_undo(frames={e.frame for e in plan.edits} | {self.current_frame_idx})
            applied = 0
            for edit in plan.edits:
                for obj in self.annotations.get(edit.frame, []):
                    if obj.track_id != edit.track_id:
                        continue
                    obj.bbox = [int(round(v)) for v in edit.bbox]
                    obj.obb = list(edit.obb) if edit.obb is not None else None
                    applied += 1
                    break
            if applied:
                self._mark_dirty()
                self._render_canvas()
                self._render_timeline()
        self._status(plan.describe())

    # ─── Inspector wiring ────────────────────────────────────────────

    def _inspector_callbacks(self) -> Dict[str, Callable]:
        """Every action the Inspector can invoke, by name.

        A dict rather than a subclass so the Inspector stays a view with no
        knowledge of the annotator, and so a test can build one with an empty
        map and still render every section.
        """
        return {
            "continue_animal": self.continue_animal,
            "click_animal": lambda: self.set_tool("click"),
            "draw_animal": lambda: self.set_tool("draw"),
            "guided_help": self.show_onboarding,
            "set_class": self._insp_set_class,
            "set_track": self._insp_set_track,
            "set_action_name": self._insp_set_action,
            "set_group_behavior": self._insp_set_group_behavior,
            "set_occluded": lambda v: self._insp_set_flag("occluded", v),
            "set_crowd": lambda v: self._insp_set_flag("is_crowd", v),
            "toggle_obb": self._insp_toggle_obb,
            "delete": self.delete_selected,
            "accept": self.accept_frame,
            "reject": self._discard_all_here,
            "edit": lambda: self.set_tool("select"),
            "accept_frame": self.accept_frame,
            "accept_tracklet": self.accept_tracklet,
            "accept_clip": self.accept_clip,
            "accept_visible": self.accept_visible,
            "accept_similar": self.accept_similar,
            "reject_tracklet": self.reject_tracklet,
            "goto_tracklet": self.goto_tracklet,
            "clip_follow": self._insp_follow_clip,
            "clip_goto_seed": self._insp_goto_clip_seed,
            "clip_goto_start": self._insp_goto_clip_start,
            "confirm_interval": self._insp_confirm_interval,
            "relabel_interval": self._insp_relabel_interval,
            "interval_start_here": lambda: self._insp_move_interval("start"),
            "interval_end_here": lambda: self._insp_move_interval("end"),
            "split_interval": self._insp_split_interval,
            "delete_interval": self._insp_delete_interval,
            "ids_correct": lambda: self._insp_swap(False),
            "swap_ids": lambda: self._insp_swap(True),
            "issue_accept": lambda: self._insp_issue("accept"),
            "issue_reject": lambda: self._insp_issue("reject"),
            "issue_skip": lambda: self._insp_issue("skip"),
            "issue_prev": lambda: self._insp_step(-1),
            "issue_next": lambda: self._insp_step(1),
            # -- migrated from the legacy sidebar --
            "select_object": self._on_list_select_value,
            "set_active_track": self._on_active_track_change,
            "new_track": self.new_track,
            "find_all": lambda text: self.find_all_in_frame(text),
            "find_all_clicked": self._find_all_clicked,
            "mark_checked": self.mark_frame_checked,
        }

    def _refresh_inspector(self) -> None:
        """Show whatever is selected. The one place the context is decided."""
        self._refresh_round_block()
        if not dpg.does_item_exist(self.inspector.tag):
            return
        common = dict(
            frame=self.current_frame_idx,
            num_frames=self.num_frames,
            fps=self.fps,
            class_names=tuple(self.class_names),
            action_names=tuple(self.action_names),
            active_class_id=self.active_class_id,
            expected_animals=self.quality_config.expected_animals,
            verified_frames=len(self.annotations),
            pending_here=len(self.candidates_here()),
            use_track_ids=self.use_track_ids,
            chosen_frame=self.current_frame_idx in (self._selection.frame_set()
                                                    if self._selection is not None else ()),
            checked=self.current_frame_idx in self.annotations,
            keyframe_note=self._keyframe_note(),
        )

        if self._sel_issue is not None:
            self.inspector.show(
                inspector_mod.InspectorContext(
                    kind=inspector_mod.ISSUE,
                    item=self._sel_issue,
                    issue_progress=self._queue.progress().describe(),
                    **common,
                )
            )
            return
        if self._sel_interval is not None:
            self.inspector.show(
                inspector_mod.InspectorContext(
                    kind=inspector_mod.INTERVAL, interval=self._sel_interval, **common
                )
            )
            return
        if self._sel_tracklet is not None:
            self.inspector.show(
                inspector_mod.InspectorContext(
                    kind=inspector_mod.TRACKLET, tracklet=self._sel_tracklet, **common
                )
            )
            return

        if self._sel_clip is not None:
            try:
                from .gui import video_segment as vs
            except ImportError:  # pragma: no cover
                from htrtdetr.gui import video_segment as vs

            clip = self._sel_clip
            seed = vs.clip_seed_frame(
                clip.start, clip.end, self.annotations,
                num_frames=self.num_frames,
            )
            self.inspector.show(
                inspector_mod.InspectorContext(
                    kind=inspector_mod.CLIP,
                    clip=clip,
                    clip_seed=seed,
                    clip_animals=len(self.annotations.get(seed, [])) if seed is not None else 0,
                    clip_blocked=self.clip_propagation_blocked(),
                    **common,
                )
            )
            return

        here = len(self.annotations.get(self.current_frame_idx, []))
        obj = self._selected
        if obj is None:
            self.inspector.show(
                inspector_mod.InspectorContext(kind=inspector_mod.NOTHING, animals=here, **common)
            )
            return

        self.inspector.show(
            inspector_mod.InspectorContext(
                kind=inspector_mod.BOX,
                animals=here,
                track_id=obj.track_id,
                class_id=obj.class_id,
                action_id=obj.action_id,
                **self._inspector_behavior_fields(obj),
                score=obj.score,
                occluded=obj.occluded,
                is_crowd=obj.is_crowd,
                has_obb=obj.has_obb,
                angle_deg=math.degrees(obj.get_obb()[4]),
                source=obj.source,
                accepted=True,
                **common,
            )
        )

    # -- inspector actions -------------------------------------------
    def _on_timeline_clip(self, payload) -> None:
        """A click on a clip in the samples lane: ``(start, end)``.

        Selecting, not running. A propagation is a job of minutes that a click
        cannot take back, so the click puts the clip in the Inspector and the
        Inspector offers the button - the same shape as clicking a box, which
        selects it rather than editing it.
        """
        start, end = (int(v) for v in payload)
        clip = None
        for candidate in getattr(self._selection, "clips", ()) or ():
            if int(candidate.start) == start and int(candidate.end) == end:
                clip = candidate
                break
        if clip is None:
            # The timeline only draws clips it was given, so this means the
            # model and the selection have drifted; fall back to a plain span
            # rather than dropping the click.
            clip = sampling_mod.Clip(start=start, end=end)
        self._sel_clip = clip
        self._sel_tracklet = self._sel_interval = self._sel_issue = None
        self._selected = None
        self._refresh_sidebar()
        self._render_canvas()

    def clip_propagation_blocked(self) -> str:
        """Why a clip propagation could not run, or ``""``.

        Only reasons that pressing the button cannot resolve. The segmentation
        helper is deliberately **not** one of them: ``propagate_clip`` starts it
        on press, so reporting "no helper is running" here would lock a control
        over a condition that pressing it fixes. That is the same reasoning
        that keeps Continue permanently unlocked.
        """
        if not self.use_track_ids:
            return (
                "Following a clip needs track IDs, which this project has "
                "turned off. Turn them on in Project > Tracking."
            )
        if self.cap is None:
            return "Open a video first."
        return ""

    def _insp_follow_clip(self) -> None:
        clip = self._sel_clip
        if clip is None:
            return
        self.propagate_clip(clip.start, clip.end)

    def _insp_goto_clip_seed(self) -> None:
        clip = self._sel_clip
        if clip is None:
            return
        try:
            from .gui import video_segment as vs
        except ImportError:  # pragma: no cover
            from htrtdetr.gui import video_segment as vs

        seed = vs.clip_seed_frame(
            clip.start, clip.end, self.annotations, num_frames=self.num_frames
        )
        if seed is None:
            self._status("This clip has no annotated frame to go to yet.")
            return
        self.goto_frame(int(seed))

    def _insp_goto_clip_start(self) -> None:
        clip = self._sel_clip
        if clip is not None:
            self.goto_frame(int(clip.start))

    def _insp_set_class(self, name: str) -> None:
        if name not in self.class_names:
            return
        self.active_class_id = self.class_names.index(name)
        if self._selected is not None:
            self._push_undo()
            self._selected.class_id = self.active_class_id
            self._mark_dirty()
            self._refresh_sidebar()
            self._render_canvas()

    def _insp_set_track(self, value: int) -> None:
        if self._selected is None:
            return
        self._push_undo()
        self._selected.track_id = max(-1, int(value))
        self._mark_dirty()
        self._refresh_sidebar()
        self._render_canvas()

    def _insp_set_action(self, name: str) -> None:
        if self._selected is None:
            return
        self._push_undo()
        self._selected.action_id = (
            self.action_names.index(name) if name in self.action_names else -1
        )
        self._mark_dirty()
        self._refresh_sidebar()
        self._render_canvas()

    def _insp_set_flag(self, field_name: str, value: bool) -> None:
        if self._selected is None:
            return
        self._push_undo()
        setattr(self._selected, field_name, bool(value))
        self._mark_dirty()
        self._refresh_sidebar()

    def _insp_toggle_obb(self) -> None:
        if self._selected is None:
            return
        if self._selected.has_obb:
            self.convert_selected_to_aabb()
        else:
            self.convert_selected_to_obb()
        self._refresh_inspector()

    def _insp_confirm_interval(self) -> None:
        if self._sel_interval is None:
            return
        self._push_undo(frames=self.annotations.keys())
        self._sel_interval = self._intervals.confirm(self._sel_interval)
        self._apply_intervals()
        self._status("Behavior confirmed.")

    def _insp_relabel_interval(self, name: str) -> None:
        if self._sel_interval is None or name not in self.action_names:
            return
        self._push_undo(frames=self.annotations.keys())
        self._sel_interval = self._intervals.replace(
            self._sel_interval, self._sel_interval.relabel(self.action_names.index(name))
        )
        self._apply_intervals()

    def _insp_move_interval(self, edge: str) -> None:
        if self._sel_interval is None:
            return
        self._push_undo(frames=self.annotations.keys())
        previous = self._sel_interval
        moved = (
            self._sel_interval.move_start(self.current_frame_idx)
            if edge == "start"
            else self._sel_interval.move_end(self.current_frame_idx)
        )
        self._sel_interval = self._intervals.replace(self._sel_interval, moved)
        self._apply_intervals(previous=previous)

    def _insp_split_interval(self) -> None:
        if self._sel_interval is None:
            return
        self._push_undo(frames=self.annotations.keys())
        try:
            left, _right = self._intervals.split(self._sel_interval, self.current_frame_idx)
        except ValueError as exc:
            self._status(str(exc))
            return
        self._sel_interval = left
        self._apply_intervals()

    def _insp_delete_interval(self) -> None:
        if self._sel_interval is None:
            return
        self._push_undo(frames=self.annotations.keys())
        previous = self._sel_interval
        self._intervals.remove(self._sel_interval)
        self._sel_interval = None
        self._apply_intervals(previous=previous)

    def _apply_intervals(self, previous=None) -> None:
        """Write interval labels back onto the per-frame ``action_id``.

        This is the whole compatibility story for behavior: intervals are the
        editing model, ``annotations.json`` keeps one action per object per
        frame, and nothing downstream changes.
        """
        flat = self._intervals.to_frame_actions()
        if previous is not None:
            for frame in range(previous.start, previous.end + 1):
                if previous.track_id in flat.get(frame, {}):
                    continue
                for obj in self.annotations.get(frame, []):
                    if obj.track_id == previous.track_id and obj.action_id == previous.action_id:
                        obj.action_id = -1
        touched = 0
        for frame, per_track in flat.items():
            for obj in self.annotations.get(frame, []):
                want = per_track.get(obj.track_id)
                if want is not None and obj.action_id != want:
                    obj.action_id = want
                    touched += 1
        self._mark_dirty()
        self._refresh_sidebar()
        self._refresh_inspector()
        self._render_timeline()

    def _insp_swap(self, swapped: bool) -> None:
        if self._sel_issue is None:
            return
        self._queue.swap_decision(self._sel_issue, swapped=swapped)
        self._on_review_swap(self._sel_issue, swapped)
        self._insp_step(1)

    def _insp_issue(self, verdict: str) -> None:
        if self._sel_issue is None:
            return
        getattr(self._queue, verdict)(self._sel_issue)
        self._on_review_decision(self._sel_issue, verdict)
        self._insp_step(1)
        self._mark_dirty()

    def _insp_step(self, direction: int) -> None:
        item = self._queue.next_unresolved() if direction > 0 else self._queue.retreat()
        if item is None:
            self._sel_issue = None
            self._status("Nothing left to review.")
            self._refresh_inspector()
            return
        self.jump_to_issue(item)

    # ─── Keyboard actions that needed a home ─────────────────────────

    def sample_frames(self) -> List[int]:
        """The frames *Pick frames to label* chose for this video, in order."""
        frames = getattr(self._selection, "frames", None) or ()
        return sorted({int(f) for f in frames})

    def _goto_sample(self, frame: int) -> None:
        """Seek to a chosen frame and say which of them it is."""
        frames = self.sample_frames()
        self.goto_frame(int(frame))
        if frame in frames:
            self._status(
                f"Sample {frames.index(frame) + 1} of {len(frames)}. "
                f"{shortcuts.label_for('review.next_pending_frame')} for the next one."
            )

    def _jump_next_pending(self) -> None:
        """Ctrl+Right: the next thing worth looking at, candidates first.

        Candidates first, then the frames the sampler chose. The fallback is
        the whole point on day one: *Pick frames to label* runs without a
        checkpoint and is the recommended first action, but it produces no
        candidates - so until this key also stepped through its samples, it
        answered "No candidates are loaded." and the ninety-nine frames it had
        just chosen could only be reached by scrubbing a timeline on which one
        pixel is eighty-six frames.
        """
        session = self._review
        frame = None
        if session is not None:
            frame = session.next_pending_frame(self.current_frame_idx)
        if frame is not None:
            self.goto_frame(frame)
            return
        nxt = next((f for f in self.sample_frames() if f > self.current_frame_idx), None)
        if nxt is not None:
            self._goto_sample(nxt)
            return
        if session is not None:
            self._status("No candidates after this frame.")
        elif self.sample_frames():
            self._status("That was the last chosen frame in this video.")
        else:
            self._status(
                "Nothing to step through yet. "
                f"{shortcuts.label_for('auto.annotate')} chooses the frames "
                f"worth labelling."
            )

    def _jump_prev_pending(self) -> None:
        session = self._review
        frame = None
        if session is not None:
            frame = session.prev_pending_frame(self.current_frame_idx)
        if frame is not None:
            self.goto_frame(frame)
            return
        previous = [f for f in self.sample_frames() if f < self.current_frame_idx]
        if previous:
            self._goto_sample(previous[-1])

    def _cycle_selection(self) -> None:
        """Q: step through this frame's boxes, candidates first."""
        objects = self.annotations.get(self.current_frame_idx, [])
        if not objects:
            return
        if self._selected is None or self._selected not in objects:
            self._select(objects[0])
            return
        index = (objects.index(self._selected) + 1) % len(objects)
        self._select(objects[index])

    def _accept_all_here(self) -> None:
        self.accept_frame()

    def _discard_all_here(self) -> None:
        n = self.discard_all_candidates()
        self.schedule_copilot_rebuild()
        self._status(f"Discarded {n} candidate{'s' if n != 1 else ''}.")

    def _track_forward_selected(self) -> None:
        self.continue_animal("Fast tracking")

    def _on_viewport_resize(self) -> None:
        """Reflow the four regions. One size source, used by every region."""
        if not dpg.does_item_exist("canvas_win"):
            return
        vw = dpg.get_viewport_client_width()
        cw, ch = self._canvas_size()
        dpg.configure_item("canvas_win", width=cw, height=ch)
        dpg.configure_item("canvas", width=cw, height=ch)
        for tag in ("toolrail", "right_col"):
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, height=ch)
        if dpg.does_item_exist("right_col"):
            dpg.configure_item("right_col", show=not self._panels_hidden)
        self.timeline.resize(max(vw - 24, 400))  # vw is the *client* width
        # Dialogs are sized against the viewport they were opened in. Shrinking
        # the window without this leaves Settings, Smart Annotate or the batch
        # form hanging over the edge with their buttons out of reach.
        widgets.refit_dialogs()
        if self.review_mode.active:
            self.review_mode.resize(
                width=dpg.get_viewport_client_width() - 120,
                height=dpg.get_viewport_client_height() - 90,
            )
            self._render_review_canvas()
        self._clamp_pan()
        self._recompute_scale()
        self._render_canvas()
        self._render_timeline()

    # ═══════════════════════════════════════════════════════════════════
    # Run
    # ═══════════════════════════════════════════════════════════════════

    def run(
        self,
        video: Optional[Path] = None,
        load_json: Optional[Path] = None,
        import_predictions: Optional[Path] = None,
        folder: Optional[Path] = None,
        recursive: bool = False,
    ) -> None:
        dpg.create_viewport(
            title="YOAKE Annotation Tool", width=1280, height=820, min_width=900, min_height=600
        )
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_resize_callback(self._on_viewport_resize)

        # Show project setup screen first
        self._show_setup_screen()

        # Handle CLI arguments after first render
        if video or load_json or import_predictions or folder:

            def _deferred():
                if dpg.does_item_exist("setup_win"):
                    # auto-confirm setup with defaults
                    self.class_names = list(_DEFAULT_CLASSES)
                    self.action_names = list(_DEFAULT_ACTIONS)
                    dpg.delete_item("setup_win")
                    self._build_main_window()
                    self._update_action_buttons()
                if load_json:
                    self._load_annotation_file(load_json)
                if folder:
                    from .annotation.project import scan_folder

                    self._add_videos(scan_folder(folder, recursive=recursive))
                if video:
                    self._load_video(video)
                # After the video, so the candidates land on a real frame
                # count and the review queue can check the animal count.
                if import_predictions:
                    self.import_predictions(import_predictions)

            dpg.set_frame_callback(2, _deferred)

        try:
            # The render loop is explicit rather than dpg.start_dearpygui() so
            # the frame can carry the things that must happen between frames:
            # draining the analysis thread's queue (never from the thread
            # itself), watching the training child, the coalesced copilot
            # rebuild, the debounced autosave, the Activity readout and toast
            # expiry.
            while dpg.is_dearpygui_running():
                self._pump_ui()
                self._tick_review()
                self._tick_context_loop()
                self.tick_train_round()
                self._tick_copilot()
                self._tick_autosave()
                self._tick_activity()
                widgets.tick_toasts()
                dpg.render_dearpygui_frame()
        finally:
            # ヘルパーは stdin が閉じれば自分で終わるが、明示的に止める。
            # Windows で取り残されたプロセスが GPU/メモリを掴んだままになるのを
            # 防ぐため、ここは必ず通す。
            self._shutdown_workspace()
            self._shutdown_round()
            self._stop_sam()
            if self.cap is not None:
                self.cap.release()
        dpg.destroy_context()


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="annotate", description="YOAKE GUI Annotation Tool")
    parser.add_argument("video", nargs="?", help="Video file to open")
    parser.add_argument(
        "--project",
        metavar="PATH",
        help="Open a project folder, or the project.yoake.json inside it",
    )
    parser.add_argument("--folder", metavar="PATH", help="Add all videos from a folder")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders with --folder")
    parser.add_argument("--load", metavar="JSON", help="Load existing annotation JSON")
    parser.add_argument(
        "--import-predictions",
        metavar="JSON",
        dest="import_predictions",
        help="Load a predictions.json as candidates to review. This is the "
        "flag the analysis page's 'Send Predictions to Annotator' button "
        "passes.",
    )
    parser.add_argument(
        "--video",
        dest="video_opt",
        metavar="PATH",
        help="Video file to open (same as the positional form)",
    )
    args = parser.parse_args(argv)

    app = AnnotationApp()
    video = args.video or args.video_opt
    app.run(
        video=Path(video) if video else None,
        load_json=Path(args.project or args.load) if (args.project or args.load) else None,
        folder=Path(args.folder) if args.folder else None,
        recursive=args.recursive,
        import_predictions=(Path(args.import_predictions) if args.import_predictions else None),
    )


if __name__ == "__main__":
    main()
