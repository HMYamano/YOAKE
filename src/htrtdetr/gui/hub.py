"""
hub.py - Unified YOAKE GUI hub (Dear PyGui)
===========================================

Single-window shell for the whole YOAKE workflow:

- ``0. Workflow``  guided overview: what each step needs, does and produces
- ``1. Label``     launch the annotator, reopen recent projects, auto-annotate
- ``2. Train``     dataset preparation, staged training, live monitor
- ``3. Analyze``   inference, frame-by-frame review, track timelines

Design notes
------------
* 色 / 余白 / フォントはすべて :mod:`htrtdetr.gui.theme` のトークン経由。
  このファイルに色リテラルを書いてはいけない。
* キーバインドは :mod:`htrtdetr.gui.shortcuts` が単一の真実。ハブ側の
  ``Ctrl+1..4`` / ``F1`` もそこから生成する。
* 子プロセスの環境は :func:`htrtdetr.gui.auto_label.ensure_child_env` が
  ``Hub.__init__`` で一度だけ適用する。``gui/runner.py`` は ``env=`` を渡さず
  ``os.environ`` を継承するため、これを忘れると全ての実行ボタンが
  ``ModuleNotFoundError`` で静かに失敗する。

The annotator runs in a separate process because Dear PyGui uses a single
context per process.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from . import auto_label, dialogs, recents, shortcuts, theme, widgets
from .analysis_panel import AnalysisPanel
from .training_panel import TrainingPanel

__all__ = ["Hub", "main", "environment_rows", "EnvRow"]

# ─── geometry ─────────────────────────────────────────────────────────────
#: Preferred viewport size. Clamped to the display in :func:`_viewport_size`
#: so the window can never open larger than a 1366x768 laptop screen.
WINDOW_W, WINDOW_H = 1440, 920
NAV_W = 264
STATUS_H = 30
SIDE_COL_W = 400          # right-hand column on the Label page
CARD_PAD = 52             # child-window padding allowance for text wrapping

APP_NAME = "YOAKE"
APP_TAGLINE = "Animal behavior workflow"

_CREATE_NO_WINDOW = 0x08000000


# ══════════════════════════════════════════════════════════════════════════
# pages
# ══════════════════════════════════════════════════════════════════════════
_PAGES: Tuple[Dict[str, str], ...] = (
    {
        "key": "workflow",
        "label": "0. Workflow",
        "short": "Workflow",
        "subtitle": "Start here",
        "action": "page.workflow",
        "summary": "What each step needs, what you do, and the file it produces.",
    },
    {
        "key": "labeling",
        "label": "1. Label",
        "short": "Label",
        "subtitle": "Draw boxes, label actions",
        "action": "page.label",
        "summary": "The whole labelling loop lives here: it picks the frames, "
                   "proposes the boxes, and you review them. No model needed.",
    },
    {
        "key": "training",
        "label": "2. Train",
        "short": "Train",
        "subtitle": "Build a model from finished labels",
        "action": "page.train",
        "summary": "Once the labels are done: create splits, train, watch the "
                   "metrics. Nothing in step 1 waits on this.",
    },
    {
        "key": "analysis",
        "label": "3. Analyze",
        "short": "Analyze",
        "subtitle": "Review predictions",
        "action": "page.analyze",
        "summary": "Run inference, then step through detections, tracks and actions.",
    },
)

#: Accepted by :meth:`Hub.switch` alongside the canonical keys, so older call
#: sites that say ``switch("home")`` keep working.
_PAGE_ALIASES: Dict[str, str] = {
    "home": "workflow",
    "label": "labeling",
    "train": "training",
    "analyze": "analysis",
    "video analysis": "analysis",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _page_meta(key: str) -> Dict[str, str]:
    resolved = _PAGE_ALIASES.get(str(key).lower(), key)
    for page in _PAGES:
        if page["key"] == resolved:
            return page
    return _PAGES[0]


# ══════════════════════════════════════════════════════════════════════════
# environment probe  (cheap, and never imports torch)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class EnvRow:
    """One line of the environment report.

    ``state`` is ``ok`` | ``warn`` | ``missing`` and maps straight onto
    :func:`theme.status_dot`.
    """

    label: str
    value: str
    state: str
    detail: str = ""
    fix: str = ""

    @property
    def text(self) -> str:
        return f"{self.label} {self.value}".strip()


def _dist_version(*names: str) -> str:
    """Version of the first installed distribution, without importing it."""
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover
        return ""
    for name in names:
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001 - PackageNotFoundError and friends
            continue
    return ""


def _python_label() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


def _dpg_version() -> str:
    """Never call ``dpg.get_dearpygui_version()`` here.

    Measured: that entry point **segfaults** when no dearpygui context exists,
    and this report is also read before ``create_context()``. The distribution
    metadata and the package attribute answer the same question safely.
    """
    version = _dist_version("dearpygui")
    if version:
        return version
    try:
        import dearpygui
        return str(getattr(dearpygui, "__version__", "")) or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _cuda_state() -> Tuple[str, str, str]:
    """``(value, state, detail)`` for the NVIDIA driver, without torch.

    Loads the CUDA driver library directly (``nvcuda.dll`` / ``libcuda.so.1``)
    and asks it for a device count. Measured at ~40 ms here, and it answers the
    question the user actually has - "will training use the GPU?" - without
    paying for a torch import.
    """
    names = ("nvcuda.dll",) if os.name == "nt" else ("libcuda.so.1", "libcuda.so")
    lib = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return ("no NVIDIA driver found", "warn",
                "Training and inference will run on the CPU, which is much slower.")
    try:
        rc = int(lib.cuInit(0))
        count = ctypes.c_int(0)
        if rc == 0:
            lib.cuDeviceGetCount(ctypes.byref(count))
    except Exception:  # noqa: BLE001
        return ("driver found, could not be queried", "warn",
                "The CUDA driver did not answer. Training falls back to the CPU.")
    if rc == 0 and count.value > 0:
        noun = "device" if count.value == 1 else "devices"
        return (f"{count.value} {noun}", "ok",
                "A CUDA device is visible to this machine.")
    return ("driver present, no device", "warn",
            "The driver is installed but no GPU is visible. Training and "
            "inference will run on the CPU, which is much slower.")


_ENV_CACHE: Dict[str, Tuple[EnvRow, ...]] = {}


def environment_rows(root: Path, *, refresh: bool = False) -> Tuple[EnvRow, ...]:
    """The environment report shown in the sidebar and on ``0. Workflow``.

    Every value comes from ``importlib`` metadata, ``find_spec`` or the CUDA
    driver - nothing here imports torch or the model package. Cached per root
    because the answer cannot change while the app is running.
    """
    key = str(root)
    if not refresh and key in _ENV_CACHE:
        return _ENV_CACHE[key]

    cap = auto_label.probe(Path(root))
    cv_version = _dist_version("opencv-python", "opencv-contrib-python",
                               "opencv-python-headless")
    torch_version = _dist_version("torch")
    cuda_value, cuda_state, cuda_detail = _cuda_state()

    rows: List[EnvRow] = [
        EnvRow("Python", _python_label(), "ok",
               f"Interpreter: {sys.executable}"),
        EnvRow("Dear PyGui", _dpg_version(), "ok",
               "The toolkit this window is drawn with."),
        EnvRow(
            "OpenCV",
            cv_version or ("installed" if cap.opencv else "not installed"),
            "ok" if cap.opencv else "missing",
            cap.tracker_reason if cap.opencv else
            "Video decoding and box tracking both need OpenCV.",
            "" if cap.opencv else "pip install opencv-python",
        ),
        EnvRow(
            "PyTorch",
            torch_version or "not installed",
            "ok" if cap.torch else "warn",
            "Training and automatic detection can run in this environment."
            if cap.torch else
            "Training and automatic detection need PyTorch. Everything else - "
            "labeling, tracking a box forward, and reviewing a predictions.json "
            "produced elsewhere - works without it.",
            "" if cap.torch else "pip install torch torchvision",
        ),
        EnvRow("CUDA", cuda_value, cuda_state, cuda_detail),
        EnvRow(
            "YOAKE package",
            {"installed": "installed", "src": "source checkout"}.get(
                cap.package_via, "not importable"),
            "ok" if cap.package else "missing",
            "Imported from this repository; every subprocess is given the "
            "same path." if cap.package_via == "src" else
            ("Installed in this environment." if cap.package
             else cap.package_reason),
            "" if cap.package else "pip install -e .",
        ),
    ]
    result = tuple(rows)
    _ENV_CACHE[key] = result
    return result


def _viewport_size() -> Tuple[int, int]:
    """Preferred size, clamped so the window always fits the display."""
    dw, dh = 0, 0
    if os.name == "nt":
        try:
            user32 = ctypes.windll.user32       # type: ignore[attr-defined]
            dw = int(user32.GetSystemMetrics(0))
            dh = int(user32.GetSystemMetrics(1))
        except Exception:  # noqa: BLE001
            dw, dh = 0, 0
    if dw <= 0 or dh <= 0:
        try:
            import tkinter as tk
            probe = tk.Tk()
            probe.withdraw()
            dw, dh = probe.winfo_screenwidth(), probe.winfo_screenheight()
            probe.destroy()
        except Exception:  # noqa: BLE001
            dw, dh = 0, 0
    if dw <= 0 or dh <= 0:
        return WINDOW_W, WINDOW_H
    return (max(960, min(WINDOW_W, dw - 80)),
            max(640, min(WINDOW_H, dh - 80)))


def _ellipsize(text: str, limit: int = 58) -> str:
    """Trim from the left, so the filename - the useful half - always survives."""
    s = str(text)
    if len(s) <= limit:
        return s
    return "..." + s[-(limit - 3):]


def _copy_to_clipboard(text: str) -> bool:
    try:
        dpg.set_clipboard_text(str(text))
        return True
    except Exception:  # noqa: BLE001
        return False


# ══════════════════════════════════════════════════════════════════════════
# annotator launch watcher
# ══════════════════════════════════════════════════════════════════════════
class _LaunchWatch:
    """Watches a freshly spawned annotator for long enough to tell the truth.

    ``Popen`` succeeding proves nothing: a missing package, a bad path or an
    import error all exit within a second. The hub therefore reports
    "starting..." first and only claims the annotator is running once the
    process has survived :data:`GRACE_S`.
    """

    GRACE_S = 1.5
    MAX_LINES = 60

    def __init__(self, proc: subprocess.Popen, label: str) -> None:
        self.proc = proc
        self.label = label
        self.deadline = time.monotonic() + self.GRACE_S
        self.lines: Deque[str] = deque(maxlen=self.MAX_LINES)
        self._thread: Optional[threading.Thread] = None
        if proc.stdout is not None:
            self._thread = threading.Thread(target=self._read, daemon=True)
            self._thread.start()

    def _read(self) -> None:
        stream = self.proc.stdout
        if stream is None:
            return
        try:
            for line in stream:
                text = line.rstrip()
                if text:
                    self.lines.append(text)
        except (OSError, ValueError):
            pass

    def first_problem(self) -> str:
        """The most useful line the failed child printed, or a fallback."""
        for line in self.lines:
            low = line.lower()
            if "error" in low or "exception" in low or "traceback" in low:
                return line
        for line in reversed(self.lines):
            if line.strip():
                return line
        return f"it exited with code {self.proc.returncode}."


# ══════════════════════════════════════════════════════════════════════════
# the hub
# ══════════════════════════════════════════════════════════════════════════
class Hub:
    """The single-window shell. One instance per process."""

    def __init__(self) -> None:
        self.root = _repo_root()

        # 子プロセスが htrtdetr を import できるようにする。パネル構築より前。
        auto_label.ensure_child_env(self.root)

        self.cap = auto_label.probe(self.root)
        self.current = "workflow"

        #: The project folder all three working pages operate inside. It is
        #: remembered across sessions, because "which study am I in" is not a
        #: question a user should answer again on every launch - and because
        #: it decides where training and analysis write their output.
        self.project: Optional[Path] = recents.active_project()

        self.training = TrainingPanel(self.root, self.project)
        self.analysis = AnalysisPanel(self.root, self.project)
        # 追加のみ: パネル側は好きなときに ``self.hub.status(...)`` を呼べる。
        # コンストラクタ署名は変えていないので、使わないパネルにも無害。
        self.training.hub = self       # type: ignore[attr-defined]
        self.analysis.hub = self       # type: ignore[attr-defined]

        self._wrap_items: List[Tuple[object, int, Optional[int]]] = []
        self._launches: List[_LaunchWatch] = []
        self._reported_tick_errors: set = set()
        self._last_content_w = 0

    # ─── status bar ──────────────────────────────────────────────────
    def status(self, message: str, kind: str = "idle") -> None:
        """Transient one-line status, bottom left. Safe before the UI exists."""
        widgets.status_text("hub_status", message, kind)

    def context(self, message: str) -> None:
        """Persistent context, bottom right (project / output paths)."""
        widgets.status_text("hub_status_right", message, "idle")

    def _refresh_context(self) -> None:
        """Persistent right-hand context, degraded to fit the window.

        The status bar's right column is a fraction of the viewport, so a full
        pair of absolute paths is simply cut off on a narrow window. Three
        tiers - full paths, folder names, project only - keep it legible at
        every width instead.
        """
        try:
            vw = int(dpg.get_viewport_client_width())
        except Exception:  # noqa: BLE001
            vw = WINDOW_W
        chars = max(16, int((vw - 24) * 0.38 / 7.6))
        home = self.project if self.project is not None else self.root
        runs = self.training.runs
        label = "Project" if self.project is not None else "Source"
        if chars >= 60:
            half = max(14, (chars - 24) // 2)
            text = (f"{label}: {_ellipsize(str(home), half)}   |   "
                    f"Output: {_ellipsize(str(runs), half)}")
        elif chars >= 34:
            text = f"{label}: {home.name}   |   Output: {runs.name}/"
        else:
            text = f"{label}: {home.name}"
        self.context(text)

    # ─── responsive text ─────────────────────────────────────────────
    def _content_w(self) -> int:
        """Usable width of the page area, recomputed on every resize."""
        try:
            vw = int(dpg.get_viewport_client_width())
        except Exception:  # noqa: BLE001
            vw = WINDOW_W
        return max(360, vw - NAV_W - 56)

    def _track_wrap(self, item, margin: int = 24,
                    fixed: Optional[int] = None) -> None:
        """Remember a text item so its ``wrap`` follows the window width.

        dearpygui takes ``wrap`` in pixels and has no "wrap to the parent"
        value, so responsiveness has to be applied by hand. ``fixed`` pins the
        item to a column of known width instead of the page width.
        """
        if item is None:
            return
        self._wrap_items.append((item, int(margin), fixed))

    def _wrapped_text(self, text: str, *, color=None, margin: int = 24,
                      fixed: Optional[int] = None, tag: Optional[str] = None,
                      bullet: bool = False, indent: int = -1) -> object:
        kwargs: Dict[str, object] = {"color": color or theme.TEXT_DIM,
                                     "bullet": bullet}
        if tag:
            kwargs["tag"] = tag
        if indent >= 0:
            kwargs["indent"] = indent
        base = fixed if fixed is not None else self._content_w()
        kwargs["wrap"] = max(200, base - margin)
        dpg.add_text(text, **kwargs)
        item = dpg.last_item()
        self._track_wrap(item, margin, fixed)
        return item

    def _section(self, label: str, description: str = "", *,
                 fixed: Optional[int] = None,
                 margin: int = CARD_PAD) -> None:
        """A section heading whose description follows the window width.

        ``widgets.section_header`` takes a fixed ``wrap``; routing the
        description through :meth:`_wrapped_text` instead is what stops it
        being clipped when the window narrows.
        """
        widgets.section_header(label)
        if description:
            self._wrapped_text(description, margin=margin, fixed=fixed)
            widgets.gap(0)

    def _key_list(self, rows, width: int, *, key_w: int = 64) -> None:
        """A key legend for a narrow column.

        ``widgets.kbd_table`` wraps its description at a fixed 460 px, which a
        400 px sidebar clips. This wraps to the column instead.
        """
        for key_label, description in rows:
            with dpg.group(horizontal=True):
                dpg.add_button(label=str(key_label), width=key_w, height=22,
                               enabled=True)
                chip = dpg.last_item()
                theme.bind(chip, theme.chip_theme(theme.SURFACE_RAISE))
                theme.bind_mono(chip)
                dpg.add_text(str(description), color=theme.TEXT_DIM,
                             wrap=max(150, width - key_w - 22))
            widgets.gap(0)

    def _apply_wraps(self) -> None:
        width = self._content_w()
        if width == self._last_content_w:
            return
        self._last_content_w = width
        self._refresh_context()
        for item, margin, fixed in self._wrap_items:
            if not dpg.does_item_exist(item):
                continue
            base = fixed if fixed is not None else width
            try:
                dpg.configure_item(item, wrap=max(200, base - margin))
            except Exception:  # noqa: BLE001
                continue

    def _on_viewport_resize(self, sender=None, app_data=None) -> None:
        self._apply_wraps()
        widgets.refit_dialogs()

    # ══════════════════════════════════════════════════════════════════
    # navigation sidebar
    # ══════════════════════════════════════════════════════════════════
    def _build_nav(self) -> None:
        text_w = NAV_W - 30
        with dpg.child_window(tag="hub_nav", width=NAV_W, height=-STATUS_H,
                              border=False):
            widgets.gap(1)
            dpg.add_text(APP_NAME, color=theme.ACCENT)
            theme.bind_h1(dpg.last_item())
            dpg.add_text(APP_TAGLINE, color=theme.TEXT_DIM, wrap=text_w)
            widgets.gap(1)
            dpg.add_separator()
            widgets.gap(1)

            for page in _PAGES:
                self._build_nav_row(page)

            widgets.gap(1)
            dpg.add_separator()
            widgets.gap(1)

            # ── project context ──────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("Project", color=theme.TEXT_DIM)
                dpg.add_text(_ellipsize(str(self.root), 20),
                             tag="nav_project_path", color=theme.TEXT_FAINT)
            widgets.attach_tooltip("nav_project_path", str(self.root), wrap=420)
            with dpg.group(horizontal=True):
                dpg.add_text("Output ", color=theme.TEXT_DIM)
                dpg.add_text(_ellipsize(str(self.root / "runs"), 20),
                             tag="nav_output_path", color=theme.TEXT_FAINT)
            widgets.attach_tooltip("nav_output_path",
                                   f"{self.root / 'runs'}\n\nEvery run writes a "
                                   "timestamped folder here. Nothing is ever "
                                   "overwritten.", wrap=420)

            widgets.gap(1)
            dpg.add_separator()
            widgets.gap(1)

            # ── environment ──────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("Environment", color=theme.TEXT_DIM)
                widgets.help_icon(
                    "What this Python environment can do. A missing backend is "
                    "reported here at startup instead of surfacing as a failed "
                    "run later on.", wrap=340)
            widgets.gap(0)
            for idx, row in enumerate(environment_rows(self.root)):
                with dpg.group(horizontal=True):
                    dpg.add_text("●", tag=f"env_dot_{idx}",
                                 color=theme.status_dot(row.state))
                    dpg.add_text(row.text, tag=f"env_text_{idx}",
                                 color=theme.TEXT_FAINT, wrap=text_w - 18)
                tip = row.detail or row.text
                if row.fix:
                    tip = f"{tip}\n\nInstall it with:  {row.fix}"
                widgets.attach_tooltip(f"env_text_{idx}", tip, wrap=420)

            widgets.gap(1)
            dpg.add_separator()
            widgets.gap(1)
            half = (text_w - theme.SP_M) // 2
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    shortcuts.with_key("Keys", "help.shortcuts", scope="hub"),
                    lambda: self.show_shortcuts(), height=28, width=half,
                    tag="nav_shortcuts_btn")
                widgets.secondary_button("About", lambda: self.show_about(),
                                         height=28, width=half,
                                         tag="nav_about_btn")
            widgets.attach_tooltip("nav_shortcuts_btn",
                                   "Every key the hub and the annotator listen for.",
                                   wrap=320)
            widgets.attach_tooltip("nav_about_btn",
                                   "Version, environment report and folders.",
                                   wrap=320)

    def _build_nav_row(self, page: Dict[str, str]) -> None:
        key = page["key"]
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=4, height=30, tag=f"nav_bar_{key}"):
                dpg.draw_rectangle((0, 0), (4, 30), fill=theme.rgba(theme.BG, 0),
                                   color=theme.rgba(theme.BG, 0),
                                   tag=f"nav_bar_rect_{key}")
            dpg.add_button(
                label=page["label"],
                width=NAV_W - 38,
                height=30,
                tag=f"nav_{key}",
                callback=lambda s, a, u: self.switch(u),
                user_data=key,
            )
            widgets.attach_tooltip(
                f"nav_{key}",
                f"{page['subtitle']}. {page['summary']} "
                f"({shortcuts.label_for(page['action'], scope='hub')})",
                wrap=300)
        # Only the selected step shows its caption, so the sidebar costs one
        # extra line instead of four and the caption sits under the right row.
        dpg.add_text(page["subtitle"], tag=f"nav_sub_{key}", indent=14,
                     color=theme.ACCENT, wrap=NAV_W - 48, show=False)
        widgets.gap(0)

    def _refresh_nav_state(self) -> None:
        """Three redundant channels: the fill, the left bar and the caption."""
        for page in _PAGES:
            key = page["key"]
            active = key == self.current
            if dpg.does_item_exist(f"nav_{key}"):
                theme.bind(f"nav_{key}",
                           theme.accent_theme() if active else theme.ghost_theme())
            rect = f"nav_bar_rect_{key}"
            if dpg.does_item_exist(rect):
                fill = theme.rgba(theme.ACCENT) if active else theme.rgba(theme.BG, 0)
                try:
                    dpg.configure_item(rect, fill=fill, color=fill)
                except Exception:  # noqa: BLE001
                    pass
            sub = f"nav_sub_{key}"
            if dpg.does_item_exist(sub):
                dpg.configure_item(sub, show=active)

    def switch(self, key: str) -> None:
        """Show one page. Accepts the canonical keys and the legacy aliases."""
        resolved = _page_meta(key)["key"]
        self.current = resolved
        for page in _PAGES:
            panel_tag = f"panel_{page['key']}"
            if dpg.does_item_exist(panel_tag):
                dpg.configure_item(panel_tag, show=(page["key"] == resolved))
        self._refresh_nav_state()
        meta = _page_meta(resolved)
        self.status(f"{meta['label']} - {meta['summary']}")
        if resolved == "labeling":
            self._rebuild_recent_projects()

    # ══════════════════════════════════════════════════════════════════
    # 0. Workflow
    # ══════════════════════════════════════════════════════════════════
    _STEP_CARDS: Tuple[Dict[str, str], ...] = (
        {
            "key": "labeling",
            "needs": "A video file (.mp4, .avi, .mov, .mkv) - or an "
                     "annotations.json you saved earlier.",
            "does": "Draw a box around each subject, give it a track ID, and "
                    "label what it is doing. Range labeling applies one action "
                    "across a span of frames instead of one frame at a time. "
                    "Then repeat: Smart Annotate picks the next frames, Find "
                    "all in frame and Continue propose the boxes, and the "
                    "Review Inbox is where you accept or reject them.",
            "makes": "annotations.json in the project folder, beside "
                     "project.yoake.json.",
            "tip": "This step needs no trained model. The proposals come from "
                   "the segmentation helper, and each pass avoids the frames "
                   "the last one had you annotate. Candidates are never "
                   "written to disk until you accept them, so re-running "
                   "anything here is always safe.",
        },
        {
            "key": "training",
            "needs": "A label set you consider finished: annotations.json, "
                     "converted to the dataset layout and split into "
                     "train / val / test.",
            "does": "Convert the annotations, create the splits, then train the "
                    "stages in order. The monitor plots loss and metrics while "
                    "the run is going.",
            "makes": "<project>/runs/train/stage{N}/weights/best.pth, plus "
                     "results.csv for the plots.",
            "tip": "Each run writes to <project>/runs/train/stage{N}/ and is never "
                   "overwritten. Picking \"detection\" as the model mode forces "
                   "stage 1 regardless of the stage you chose - the log says so "
                   "when it happens.",
        },
        {
            "key": "analysis",
            "needs": "A checkpoint (.pth) from step 2 and a video to run it on.",
            "does": "Run inference, then step through the frames, click a box "
                    "to inspect its track, action and confidence, and read the "
                    "timeline for the whole video.",
            "makes": "predictions.json and output.mp4 in "
                     "<project>/analysis/predict/.",
            "tip": "The detector's confidence is not a calibrated probability. "
                   "Use the timeline to find the moments the model got wrong, "
                   "then fix those frames in step 1.",
        },
    )

    _RECOMMENDED_LOOP: Tuple[str, ...] = (
        "Step 1 is the loop, and it closes without a model. Run Smart Annotate "
        "to pick the frames, let Find all in frame or Continue propose the "
        "boxes, review them, and run Smart Annotate again - the next pass "
        "scores frames by how unlike your existing labels they are, so it "
        "carries on from where you stopped instead of re-rolling.",
        "Label in rounds of 20-50 frames rather than one long sitting. The "
        "point of the loop is that what you label changes what you are asked "
        "to label next, and that only happens between rounds.",
        "Steps 2 and 3 start when you decide the labels are good enough. Train "
        "stage 1 first; \"detection\" as the model mode forces it whatever the "
        "stage buttons say. Watch the validation curve, not the training one.",
        "Analyze on a video the model has not seen, and step through the "
        "timeline instead of trusting the summary numbers.",
        "A trained checkpoint then makes step 1 faster - Auto Annotate "
        "pre-labels whole videos and uncertainty joins the sampling - but it "
        "was never required to get here.",
    )

    def _build_workflow(self, parent: str) -> None:
        with dpg.child_window(tag="panel_workflow", parent=parent, width=-1,
                              height=-1, border=False, show=True):
            widgets.page_title(
                "0. Workflow",
                "Step 1 is a loop you repeat until the labels are good enough. "
                "Steps 2 and 3 turn those labels into a model and check it. "
                "Each step produces the file the next step needs.")

            self._build_capability_block()

            self._section(
                "The three steps",
                "Every card names what it needs before you start, what you do, "
                "and the file it leaves behind. Only step 1 repeats.")
            widgets.gap(1)
            for card in self._STEP_CARDS:
                self._build_step_card(card)
                widgets.gap(1)

            self._section("Recommended loop")
            for line in self._RECOMMENDED_LOOP:
                self._wrapped_text(line, margin=CARD_PAD)
                widgets.gap(0)

            self._section(
                "Keyboard",
                "These work anywhere in this window. The annotator has its own "
                "set - press F1 for the full list.")
            widgets.kbd_table(shortcuts.help_rows("hub"), tag="wf_kbd",
                              key_width=130)
            widgets.gap(1)
            widgets.secondary_button(
                shortcuts.with_key("Show all shortcuts", "help.shortcuts",
                                   scope="hub"),
                lambda: self.show_shortcuts(), height=30, width=280,
                tag="wf_shortcuts_btn")
            widgets.gap(3)

    def _build_capability_block(self) -> None:
        """Only rendered when something is missing. Silence means ready."""
        rows = environment_rows(self.root)
        problems = [r for r in rows if r.state != "ok"]
        fixes = [r.fix for r in problems if r.fix]

        widgets.section_header("This environment")
        if not problems:
            with dpg.group(horizontal=True):
                dpg.add_text("●", color=theme.status_dot("ok"))
                dpg.add_text("Everything the workflow needs is installed.",
                             color=theme.TEXT)
            widgets.gap(2)
            return

        for row in problems:
            with dpg.group(horizontal=True):
                dpg.add_text("●", color=theme.status_dot(row.state))
                dpg.add_text(f"{row.label}: {row.value}", color=theme.TEXT)
            if row.detail:
                self._wrapped_text(row.detail, margin=CARD_PAD + 16, indent=18)
            if row.fix:
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=14)
                    dpg.add_text(row.fix, color=theme.TEXT_DIM)
                    theme.bind_mono(dpg.last_item())
            widgets.gap(0)

        if fixes:
            combined = "\n".join(fixes)
            widgets.gap(1)
            widgets.secondary_button(
                "Copy the install commands",
                lambda: self._copy_commands(combined),
                height=28, width=260, tag="wf_copy_fix")
            widgets.attach_tooltip(
                "wf_copy_fix",
                "Copies every command above to the clipboard. Run them in the "
                "same Python environment this window is running in, then "
                "restart YOAKE.", wrap=380)
        widgets.gap(2)

    def _copy_commands(self, text: str) -> None:
        if _copy_to_clipboard(text):
            self.status("Install commands copied to the clipboard.", "ok")
            widgets.toast("Install commands copied to the clipboard.", "ok")
        else:
            self.status("Could not reach the clipboard. Select the text and "
                        "copy it by hand.", "warn")

    def _build_step_card(self, card: Dict[str, str]) -> None:
        meta = _page_meta(card["key"])
        key = card["key"]
        with dpg.child_window(width=-1, border=True, auto_resize_y=True,
                              tag=f"wf_card_{key}"):
            with dpg.group(horizontal=True):
                dpg.add_text(meta["label"], color=theme.TEXT)
                theme.bind_h2(dpg.last_item())
                dpg.add_spacer(width=theme.SP_L)
                widgets.secondary_button(
                    shortcuts.with_key(f"Open {meta['short']}", meta["action"],
                                       scope="hub"),
                    lambda s=None, a=None, u=key: self.switch(key),
                    height=26, width=200, tag=f"wf_open_{key}")
            widgets.gap(0)
            for label, body in (("Needs", card["needs"]),
                                ("You do", card["does"]),
                                ("Produces", card["makes"])):
                dpg.add_text(f"{label}:", color=theme.TEXT_DIM)
                self._wrapped_text(body, color=theme.TEXT,
                                   margin=CARD_PAD + 16, indent=12)
                widgets.gap(0)
            self._wrapped_text(f"Tip: {card['tip']}", color=theme.TEXT_FAINT,
                               margin=CARD_PAD + 16, indent=12)
            widgets.gap(0)

    # ══════════════════════════════════════════════════════════════════
    # 1. Label
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _label_capabilities() -> Tuple[str, ...]:
        """The "what you can do here" list, with keys pulled from the keymap.

        Every accelerator named here comes from ``shortcuts`` rather than being
        typed inline, so a rebind updates this copy too.
        """
        key = shortcuts.label_for
        return (
            f"Press {key('tool.draw_box')}, then drag on the image to draw a "
            f"box. {key('tool.cancel')} cancels.",
            "Give every subject a track ID so the model can learn identity "
            "across frames.",
            "Label an action on one box, or apply one action across a whole "
            "span of frames.",
            f"Track forward ({key('auto.track_forward')}) propagates a box you "
            "drew across later frames. It needs OpenCV only, never PyTorch.",
            f"Auto-annotate ({key('auto.annotate')}) runs a trained detector "
            "and brings the result back as candidates you accept or discard.",
            f"Save with {key('file.save')}. Candidates are never written to "
            "annotations.json.",
        )

    def _build_labeling(self, parent: str) -> None:
        with dpg.child_window(tag="panel_labeling", parent=parent, width=-1,
                              height=-1, border=False, show=False):
            widgets.page_title(
                "1. Label",
                "Draw boxes, assign tracks, and label actions. The annotator "
                "opens in its own window so it can own its keyboard.")

            with dpg.group(horizontal=True):
                with dpg.child_window(width=-SIDE_COL_W, height=-1,
                                      border=False, tag="labeling_main"):
                    self._build_project_bar()
                    self._build_label_launcher()
                    self._build_label_recents()
                    self._build_label_autolabel()
                with dpg.child_window(width=-1, height=-1, border=True,
                                      tag="labeling_side"):
                    self._build_label_side()

    # ══════════════════════════════════════════════════════════════════
    # the project folder
    # ══════════════════════════════════════════════════════════════════
    def _build_project_bar(self) -> None:
        self._section(
            "Project folder",
            "One folder per study. Annotations, the training data built from "
            "them, every run and every analysis result are written inside it, "
            "and every path it stores is relative - so the folder can be "
            "moved, copied or handed to someone else and still work.",
            margin=CARD_PAD + SIDE_COL_W)
        with dpg.group(horizontal=True):
            dpg.add_text("●", tag="hub_project_dot", color=theme.TEXT_FAINT)
            dpg.add_text("", tag="hub_project_path", color=theme.TEXT)
        self._track_wrap("hub_project_path", 44, fixed=560)
        with dpg.group(horizontal=True):
            widgets.secondary_button("Open project folder...",
                                     self._pick_project_folder,
                                     height=30, width=210, tag="hub_project_open")
            widgets.attach_tooltip(
                "hub_project_open",
                "Pick a folder that contains project.yoake.json. Steps 2 and 3 "
                "immediately point at the data and the output folders inside "
                "it.", wrap=340)
            dpg.add_spacer(width=theme.SP_S)
            dpg.add_button(label="Show in file browser", height=30, width=190,
                           tag="hub_project_reveal", callback=self._reveal_project)
            theme.bind(dpg.last_item(), theme.ghost_theme())
        widgets.gap(2)
        self._refresh_project_bar()

    def _refresh_project_bar(self) -> None:
        if not dpg.does_item_exist("hub_project_path"):
            return
        if self.project is None:
            dpg.set_value(
                "hub_project_path",
                "No project yet. 'Start a new labeling session' creates one, "
                "or open an existing project folder.")
            dpg.configure_item("hub_project_dot", color=theme.TEXT_FAINT)
        else:
            dpg.set_value("hub_project_path", str(self.project))
            dpg.configure_item("hub_project_dot", color=theme.status_dot("ok"))
        self._refresh_context()

    def use_project(self, path, *, quiet: bool = False) -> None:
        """Adopt ``path`` as the active project, and tell every page about it.

        Called from the folder picker and from opening a recent project: the
        two ways in must not be able to disagree, because a Train page still
        pointing at the previous study would write a run into it.
        """
        from ..annotation.layout import find_project

        manifest = find_project(path) if path else None
        if manifest is None:
            # Opening a loose annotation JSON is legitimate; it simply does not
            # name a project, and the pages keep the one they had.
            if not quiet:
                self.status(f"No project.yoake.json in {path}.", "error")
            return
        self.project = manifest.parent
        recents.set_active_project(self.project)
        self.training.set_project(self.project)
        self.analysis.set_project(self.project)
        self._refresh_project_bar()
        self._rebuild_recent_projects()
        self.status(f"Project: {self.project}", "ok")

    def _pick_project_folder(self) -> None:
        try:
            path = dialogs.pick_dir("Select a project folder")
        except Exception as exc:  # noqa: BLE001
            self.status(f"Could not open the folder picker: {exc}", "error")
            return
        if not path:
            return
        self.use_project(path)
        if self.project is not None:
            self._launch_annotator(["--project", str(self.project)],
                                   label=self.project.name)

    def _reveal_project(self) -> None:
        if self.project is None:
            self.status("No project folder yet.", "error")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(self.project))  # noqa: S606 - the user's own folder
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.project)])
            else:
                subprocess.Popen(["xdg-open", str(self.project)])
        except OSError as exc:
            self.status(f"Could not open {self.project}: {exc}", "error")

    def _build_label_launcher(self) -> None:
        self._section(
            "Open the annotator",
            "Start from nothing, from a video, or from a project you already "
            "saved.", margin=CARD_PAD + SIDE_COL_W)
        widgets.primary_button("Start a new labeling session",
                               lambda: self._launch_annotator(),
                               height=36, width=360, tag="lab_new_btn")
        widgets.attach_tooltip(
            "lab_new_btn",
            "Opens the annotator with no video loaded. You can create the "
            "project and pick the video from inside it.", wrap=340)
        widgets.gap(1)
        widgets.secondary_button("Open a video...",
                                 lambda: self._launch_with_video(),
                                 height=32, width=360, tag="lab_video_btn")
        widgets.attach_tooltip(
            "lab_video_btn",
            "Pick a video file; the annotator opens with it loaded on frame 0.",
            wrap=340)
        widgets.gap(0)
        widgets.secondary_button("Open an annotation JSON...",
                                 lambda: self._launch_with_json(),
                                 height=32, width=360, tag="lab_json_btn")
        widgets.attach_tooltip(
            "lab_json_btn",
            "Continue editing a saved project. The video it references is "
            "reopened with it.", wrap=340)
        widgets.gap(1)
        dpg.add_text("", tag="labeling_status", color=theme.TEXT_DIM,
                     wrap=SIDE_COL_W)
        self._track_wrap("labeling_status", 24, fixed=560)
        widgets.gap(2)

    def _build_label_recents(self) -> None:
        self._section(
            "Recent projects",
            "Paths you have used before, newest first. Shared with steps 2 "
            "and 3.", margin=CARD_PAD + SIDE_COL_W)
        with dpg.child_window(tag="labeling_recent_list", width=-1, height=164,
                              border=True):
            pass
        self._rebuild_recent_projects()
        widgets.gap(2)

    def _recent_entries(self, limit: int = 8) -> List[Tuple[str, str]]:
        """``[(kind, path), ...]`` merged from the shared recent-file store."""
        out: List[Tuple[str, str]] = []
        seen = set()
        for field, kind in (("annotation_json", "project"), ("video", "video")):
            for path in recents.get(field):
                if path in seen:
                    continue
                seen.add(path)
                out.append((kind, path))
        return out[:limit]

    def _rebuild_recent_projects(self) -> None:
        tag = "labeling_recent_list"
        if not dpg.does_item_exist(tag):
            return
        try:
            dpg.delete_item(tag, children_only=True)
        except Exception:  # noqa: BLE001
            return
        entries = self._recent_entries()
        if not entries:
            dpg.add_text(
                "No recent paths yet. Open a video or an annotation JSON once "
                "and it will be listed here.",
                parent=tag, color=theme.TEXT_DIM, wrap=520)
            return
        for idx, (kind, path) in enumerate(entries):
            row_tag = f"lab_recent_{idx}"
            with dpg.group(horizontal=True, parent=tag):
                dpg.add_button(
                    label="Open", width=64, height=24, tag=row_tag,
                    callback=lambda s, a, u: self._open_recent(*u),
                    user_data=(kind, path))
                theme.bind(row_tag, theme.ghost_theme())
                dpg.add_text("project" if kind == "project" else "video",
                             color=theme.TEXT_FAINT)
                dpg.add_text(_ellipsize(path, 52), color=theme.TEXT)
            widgets.attach_tooltip(row_tag, path, wrap=460)

    def _open_recent(self, kind: str, path: str) -> None:
        if not Path(path).exists():
            self.status(f"Not found: {path}. It may have been moved or deleted.",
                        "error")
            widgets.status_text(
                "labeling_status",
                f"Not found: {path}. It may have been moved or deleted.",
                "error")
            return
        if kind == "project":
            self.use_project(Path(path), quiet=True)
            self._launch_annotator(["--load", path], label=Path(path).name)
        else:
            self._launch_annotator([path], label=Path(path).name)

    def _build_label_autolabel(self) -> None:
        self._section(
            "Auto-annotate",
            "Let a trained detector propose boxes, then accept or discard them "
            "one frame at a time.", margin=CARD_PAD + SIDE_COL_W)
        cap = self.cap
        with dpg.group(horizontal=True):
            dpg.add_text("●", color=theme.status_dot("ok" if cap.torch else "warn"))
            dpg.add_text(
                "Model detection: available" if cap.torch
                else "Model detection: unavailable here",
                color=theme.TEXT)
        if cap.torch:
            self._wrapped_text(
                "Open the annotator and press "
                f"{shortcuts.label_for('auto.annotate')} to point it at a "
                "checkpoint. Results arrive as candidates - they are not saved "
                "as labels until you accept them.",
                margin=CARD_PAD, fixed=620)
        else:
            self._wrapped_text(cap.torch_reason, margin=CARD_PAD, fixed=620)
            widgets.gap(1)
            widgets.secondary_button(
                "Copy the install command",
                lambda: self._copy_commands("pip install torch torchvision"),
                height=28, width=260, tag="lab_copy_torch")

        widgets.gap(1)
        with dpg.group(horizontal=True):
            dpg.add_text("●", color=theme.status_dot(
                "ok" if cap.can_track else "missing"))
            dpg.add_text(f"Track forward: {cap.tracker_label}", color=theme.TEXT)
        self._wrapped_text(
            cap.tracker_reason or
            "Track forward propagates a box you already drew. It never needs "
            "PyTorch.", margin=CARD_PAD, fixed=620)
        widgets.gap(3)

    def _build_label_side(self) -> None:
        col = SIDE_COL_W - 28
        self._section("What you can do here", fixed=col)
        for line in self._label_capabilities():
            self._wrapped_text(line, margin=30, fixed=col, bullet=True)
            widgets.gap(0)
        widgets.gap(1)

        self._section(
            "Annotator shortcuts",
            "Generated from the same table the annotator registers, so this "
            "list cannot drift.", fixed=col, margin=8)
        for name, rows in shortcuts.help_groups("annotator"):
            dpg.add_text(name, color=theme.ACCENT)
            self._key_list(rows, col)
            widgets.gap(1)
        widgets.gap(2)

    # ══════════════════════════════════════════════════════════════════
    # global dialogs
    # ══════════════════════════════════════════════════════════════════
    def show_shortcuts(self, *_args) -> None:
        """The one keyboard reference for the whole suite."""
        tag = "hub_shortcuts_win"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        width, height, x, y = widgets.fit_dialog(820, 720)
        with dpg.window(label="Keyboard shortcuts", tag=tag, modal=True,
                        width=width, height=height, no_collapse=True,
                        pos=[x, y]):
            dpg.add_text(
                "Every binding in the suite. The annotator registers its own "
                "set in its own window.", color=theme.TEXT_DIM, wrap=width - 48)
            widgets.gap(1)
            with dpg.tab_bar():
                with dpg.tab(label="Annotator"):
                    # Negative height fills what is left minus the footer, so
                    # the Close button cannot be pushed off the bottom edge.
                    with dpg.child_window(width=-1, height=-72, border=False):
                        widgets.shortcut_table(
                            shortcuts.help_groups("annotator"),
                            note=self._changed_note(), key_width=132)
                with dpg.tab(label="Hub"):
                    with dpg.child_window(width=-1, height=-72, border=False):
                        widgets.kbd_table(shortcuts.help_rows("hub"),
                                          key_width=132)
                        widgets.gap(2)
                        dpg.add_text(
                            "Shortcuts are suppressed while you are typing in a "
                            "field, except Ctrl+S.",
                            color=theme.TEXT_DIM, wrap=width - 64)
            dpg.add_separator()
            widgets.secondary_button("Close",
                                     lambda: dpg.delete_item(tag),
                                     height=30, width=140,
                                     tag="hub_shortcuts_close")

    @staticmethod
    def _changed_note() -> str:
        lines = ["Changed in this version:"]
        for was, now, why in shortcuts.changed_rows():
            lines.append(f"  {was}  ->  {now}  ({why}.)")
        lines.append("")
        lines.append("Unchanged on purpose:")
        for line in shortcuts.UNCHANGED_ON_PURPOSE:
            lines.append(f"  {line}.")
        return "\n".join(lines)

    def show_about(self, *_args) -> None:
        tag = "hub_about_win"
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        rows = environment_rows(self.root)
        vw, vh = widgets.viewport_size()
        with dpg.window(label=f"About {APP_NAME}", tag=tag, modal=True,
                        width=560, autosize=True, no_resize=True,
                        pos=[max(12, (vw - 560) // 2), max(12, (vh - 520) // 2)]):
            dpg.add_text(f"{APP_NAME} {self._app_version()}", color=theme.TEXT)
            theme.bind_h2(dpg.last_item())
            dpg.add_text(
                "HT-RT-DETR: detection, tracking and action recognition for "
                "animal behavior video.", color=theme.TEXT_DIM, wrap=500)
            widgets.gap(2)
            dpg.add_separator()
            widgets.gap(1)
            widgets.kbd_table(
                [(r.label, r.value) for r in rows] +
                [("Project folder", str(self.root)),
                 ("Output folder", str(self.root / "runs"))],
                tag="about_env_table", key_width=140)
            widgets.gap(2)
            dpg.add_separator()
            widgets.gap(1)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Copy this report",
                    lambda: self._copy_commands(self._about_report()),
                    height=30, width=180, tag="about_copy_btn")
                widgets.secondary_button("Close",
                                         lambda: dpg.delete_item(tag),
                                         height=30, width=140,
                                         tag="about_close_btn")

    def _app_version(self) -> str:
        version = _dist_version("htrtdetr", "yoake")
        if version:
            return f"v{version}"
        init = self.root / "src" / "htrtdetr" / "__init__.py"
        try:
            for line in init.read_text(encoding="utf-8").splitlines():
                if line.startswith("__version__"):
                    return "v" + line.split("=", 1)[1].strip().strip("\"'")
        except (OSError, IndexError, ValueError):
            pass
        return "(version unknown)"

    def _about_report(self) -> str:
        lines = [f"{APP_NAME} {self._app_version()}"]
        lines += [f"{r.label}: {r.value}" for r in environment_rows(self.root)]
        lines.append(f"Project folder: {self.root}")
        lines.append(f"Output folder: {self.root / 'runs'}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════
    # annotator launch
    # ══════════════════════════════════════════════════════════════════
    def _launch_annotator(self, extra_args: Optional[List[str]] = None,
                          label: str = "") -> None:
        """Spawn the annotator and report what actually happened.

        The child inherits nothing by accident: it is given
        :func:`auto_label.child_env` so ``htrtdetr`` imports on a source
        checkout, and ``stdin=DEVNULL`` so a stray ``input()`` can never hang
        it invisibly behind the hub window.
        """
        cmd = [sys.executable, "-u", "-m", "htrtdetr.annotator"] + list(extra_args or [])
        name = label or "a new session"
        kwargs: Dict[str, object] = {
            "cwd": str(self.root),
            "env": auto_label.child_env(self.root),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            kwargs["creationflags"] = _CREATE_NO_WINDOW
        try:
            proc = subprocess.Popen(cmd, **kwargs)  # type: ignore[arg-type]
        except (OSError, ValueError) as exc:
            message = f"The annotator could not start: {exc}"
            widgets.status_text("labeling_status", message, "error")
            self.status(message, "error")
            return
        self._launches.append(_LaunchWatch(proc, name))
        message = f"The annotator is starting... ({name})"
        widgets.status_text("labeling_status", message, "running")
        self.status(message, "running")

    def _poll_launches(self) -> None:
        """Turn "it spawned" into "it is running" - or into the real reason."""
        if not self._launches:
            return
        still: List[_LaunchWatch] = []
        for watch in self._launches:
            code = watch.proc.poll()
            if code is None:
                if time.monotonic() >= watch.deadline:
                    message = f"Annotator running (PID {watch.proc.pid})."
                    widgets.status_text("labeling_status", message, "ok")
                    self.status(message, "ok")
                    continue
                still.append(watch)
                continue
            if code == 0:
                message = "The annotator closed."
                widgets.status_text("labeling_status", message, "idle")
                self.status(message, "idle")
            else:
                message = ("The annotator could not start: "
                           f"{watch.first_problem()}")
                widgets.status_text("labeling_status", message, "error")
                self.status(message, "error")
                widgets.toast("The annotator could not start.", "error")
        self._launches = still

    def _launch_with_video(self) -> None:
        try:
            path = dialogs.pick_file("Select a video", dialogs.VIDEO_TYPES)
        except Exception as exc:  # noqa: BLE001
            self.status(f"Could not open the file picker: {exc}", "error")
            return
        if not path:
            return
        recents.add("video", path)
        self._rebuild_recent_projects()
        self._launch_annotator([path], label=Path(path).name)

    def _launch_with_json(self) -> None:
        try:
            path = dialogs.pick_file("Select an annotation JSON",
                                     dialogs.JSON_TYPES)
        except Exception as exc:  # noqa: BLE001
            self.status(f"Could not open the file picker: {exc}", "error")
            return
        if not path:
            return
        recents.add("annotation_json", path)
        self.use_project(Path(path).parent, quiet=True)
        self._rebuild_recent_projects()
        self._launch_annotator(["--load", path], label=Path(path).name)

    # ══════════════════════════════════════════════════════════════════
    # keyboard
    # ══════════════════════════════════════════════════════════════════
    def _actions(self) -> Dict[str, Callable[[], None]]:
        return {
            "page.workflow": lambda: self.switch("workflow"),
            "page.label": lambda: self.switch("labeling"),
            "page.train": lambda: self.switch("training"),
            "page.analyze": lambda: self.switch("analysis"),
            "help.shortcuts": self.show_shortcuts,
        }

    @staticmethod
    def _typing() -> bool:
        """True while a text / number / combo widget is being edited.

        ``is_item_active`` rather than ``is_item_focused``: the first input in a
        window auto-focuses, so focus alone is sticky and would disable every
        shortcut for the rest of the session.
        """
        try:
            focused = dpg.get_focused_item()
        except Exception:  # noqa: BLE001
            return False
        if not focused:
            return False
        try:
            item_type = dpg.get_item_type(focused)
            if not any(item_type.endswith(k) for k in shortcuts.EDITING_TYPES):
                return False
            return bool(dpg.is_item_active(focused))
        except Exception:  # noqa: BLE001
            return False

    def _wrap_binding(self, binding, action: Callable[[], None]) -> Callable:
        def _cb(sender=None, app_data=None, user_data=None) -> None:
            if binding.action not in shortcuts.ALWAYS_ALLOWED and self._typing():
                return
            if binding.mods and not shortcuts.mods_held(dpg, binding):
                return
            if not binding.mods:
                try:
                    if dpg.is_key_down(dpg.mvKey_ModCtrl):
                        return
                except Exception:  # noqa: BLE001
                    pass
            action()
        return _cb

    def _build_key_handlers(self) -> None:
        actions = self._actions()
        missing = [b.action for b in shortcuts.for_scope("hub")
                   if b.action not in actions]
        if missing:
            raise RuntimeError(
                "hub shortcuts without an implementation: " + ", ".join(missing))
        with dpg.handler_registry(tag="hub_key_handlers"):
            for binding in shortcuts.for_scope("hub"):
                try:
                    key = shortcuts.resolve(dpg, binding)
                except AttributeError:
                    continue
                dpg.add_key_press_handler(
                    key, callback=self._wrap_binding(binding,
                                                     actions[binding.action]))

    # ══════════════════════════════════════════════════════════════════
    # window
    # ══════════════════════════════════════════════════════════════════
    def _build_status_bar(self) -> None:
        dpg.add_separator()
        with dpg.table(header_row=False, borders_innerV=False,
                       borders_outerV=False, borders_innerH=False,
                       borders_outerH=False, width=-1,
                       policy=dpg.mvTable_SizingStretchProp,
                       tag="hub_status_bar"):
            dpg.add_table_column(init_width_or_weight=1.0)
            dpg.add_table_column(init_width_or_weight=0.62)
            with dpg.table_row():
                dpg.add_text("", tag="hub_status", color=theme.TEXT_DIM)
                dpg.add_text("", tag="hub_status_right", color=theme.TEXT_FAINT)

    def build_ui(self) -> None:
        with dpg.window(tag="hub_win", no_title_bar=True, no_move=True,
                        no_resize=True, no_scrollbar=True):
            with dpg.group(horizontal=True):
                self._build_nav()
                with dpg.child_window(tag="content", width=-1, height=-STATUS_H,
                                      border=False):
                    self._build_workflow("content")
                    self._build_labeling("content")
                    self.training.build("content")
                    self.analysis.build("content")
            self._build_status_bar()

        dpg.set_primary_window("hub_win", True)
        self._build_key_handlers()
        self._refresh_context()
        self.switch("workflow")
        self._apply_wraps()

    def _tick(self) -> None:
        widgets.tick_toasts()
        self._poll_launches()
        self._apply_wraps()
        if self.current != "analysis":
            return
        try:
            self.analysis.tick()
        except Exception as exc:  # noqa: BLE001
            kind = type(exc).__name__
            if kind not in self._reported_tick_errors:
                self._reported_tick_errors.add(kind)
                self.status(f"Playback error: {exc}", "error")

    def run(self, argv=None) -> None:
        dpg.create_context()
        theme.setup()
        width, height = _viewport_size()
        dpg.create_viewport(
            title=f"{APP_NAME} - Workflow / Label / Train / Analyze",
            width=width, height=height, resizable=True,
            min_width=900, min_height=600,
        )
        dpg.setup_dearpygui()
        self.build_ui()
        try:
            dpg.set_viewport_resize_callback(self._on_viewport_resize)
        except Exception:  # noqa: BLE001
            pass
        dpg.show_viewport()

        while dpg.is_dearpygui_running():
            self._tick()
            dpg.render_dearpygui_frame()

        self.teardown()
        dpg.destroy_context()
        theme.reset_cache()

    def teardown(self) -> None:
        """Release every child process and timer the panels own."""
        for name in ("training", "analysis"):
            panel = getattr(self, name, None)
            stop = getattr(panel, "teardown", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001
                    continue
            else:
                runner = getattr(panel, "runner", None)
                runner_stop = getattr(runner, "stop", None)
                if callable(runner_stop):
                    try:
                        runner_stop()
                    except Exception:  # noqa: BLE001
                        pass


def main(argv=None) -> None:
    Hub().run(argv)


if __name__ == "__main__":
    main()
