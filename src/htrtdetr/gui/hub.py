"""
hub.py - Unified YOAKE GUI hub (Dear PyGui)
===========================================

Single-window launcher for the main YOAKE workflows:
- Home (workflow overview & quick jumps)
- Labeling
- Training
- Video Analysis

The annotator runs in a separate process because Dear PyGui uses a single
context per process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg

from . import dialogs, widgets
from .analysis_panel import AnalysisPanel
from .training_panel import TrainingPanel

WINDOW_W, WINDOW_H = 1440, 920
NAV_W = 260

_PAGES = [
    {
        "key": "home",
        "label": "Home",
        "subtitle": "Overview",
        "summary": "Start here. See the recommended workflow and jump to any workspace.",
        "accent": (238, 212, 136),
    },
    {
        "key": "labeling",
        "label": "Labeling",
        "subtitle": "Step 1",
        "summary": "Create or edit annotations for videos with range labeling and track tools.",
        "accent": (92, 168, 104),
    },
    {
        "key": "training",
        "label": "Training",
        "subtitle": "Step 2",
        "summary": "Prepare datasets, train each stage, and monitor runs from one screen.",
        "accent": (80, 136, 214),
    },
    {
        "key": "analysis",
        "label": "Video Analysis",
        "subtitle": "Step 3",
        "summary": "Run inference, review predictions frame by frame, and inspect track timelines.",
        "accent": (72, 188, 188),
    },
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _page_meta(key: str) -> dict:
    for page in _PAGES:
        if page["key"] == key:
            return page
    return _PAGES[0]


class Hub:
    def __init__(self) -> None:
        self.root = _repo_root()
        self.current = "home"
        self.training = TrainingPanel(self.root)
        self.analysis = AnalysisPanel(self.root)

    def _theme(self) -> None:
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 22, 28))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (24, 29, 38))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (34, 42, 54))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (44, 54, 69))
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 82, 124))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (67, 104, 152))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (80, 120, 170))
                dpg.add_theme_color(dpg.mvThemeCol_Header, (39, 55, 77))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (54, 78, 109))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (68, 96, 132))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, (32, 40, 52))
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, (52, 76, 110))
                dpg.add_theme_color(dpg.mvThemeCol_Text, (224, 228, 233))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (50, 59, 73))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 7)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 8)
        dpg.bind_theme(theme)

    # ─── navigation ──────────────────────────────────────────────────
    def _build_nav(self) -> None:
        with dpg.child_window(width=NAV_W, height=-1, border=False):
            dpg.add_spacer(height=10)
            dpg.add_text("YOAKE", color=(238, 212, 136))
            dpg.add_text("Animal behavior workflow", color=(146, 157, 171))
            dpg.add_spacer(height=6)
            dpg.add_text(
                "Pick a workspace below. Hover over a button to see what it does.",
                wrap=NAV_W - 22, color=(184, 190, 200),
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            for page in _PAGES:
                key = page["key"]
                dpg.add_button(
                    label=page["label"],
                    width=NAV_W - 16,
                    height=44,
                    tag=f"nav_{key}",
                    callback=lambda s, a, u: self.switch(u),
                    user_data=key,
                )
                widgets.tooltip(
                    f"{page['subtitle']} — {page['summary']}", wrap=280
                )
                dpg.add_text(page["subtitle"], indent=8,
                             color=(140, 152, 168), tag=f"nav_sub_{key}")
                dpg.add_spacer(height=2)

            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=8)
            dpg.add_text("Current Workspace", color=(150, 173, 198))
            dpg.add_text("", tag="nav_page_title", color=(226, 230, 235))
            dpg.add_text("", tag="nav_page_summary",
                         wrap=NAV_W - 22, color=(180, 186, 194))

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=8)
            dpg.add_text("Project Root", color=(150, 173, 198))
            dpg.add_text(str(self.root), wrap=NAV_W - 22, color=(132, 142, 154))
            dpg.add_spacer(height=6)
            dpg.add_text("Output Root", color=(150, 173, 198))
            dpg.add_text(str(self.root / "runs"),
                         wrap=NAV_W - 22, color=(132, 142, 154))

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text("Tips", color=(150, 173, 198))
            dpg.add_text(
                "* = required field\n"
                "Recent button reuses past paths\n"
                "Hover any label for details",
                wrap=NAV_W - 22, color=(160, 170, 182),
            )

    # ─── home page ───────────────────────────────────────────────────
    def _build_home(self, parent: str) -> None:
        with dpg.child_window(
            tag="panel_home",
            parent=parent,
            width=-1,
            height=-1,
            border=False,
            show=True,
        ):
            dpg.add_text("Welcome to YOAKE", color=(238, 212, 136))
            dpg.add_separator()
            dpg.add_spacer(height=8)
            dpg.add_text(
                "YOAKE helps you label animal behavior, train a multi-stage "
                "tracking model, and analyze video predictions — all from one "
                "window. Follow the three steps below.",
                wrap=980, color=(200, 208, 218),
            )
            dpg.add_spacer(height=14)

            for idx, key in enumerate(("labeling", "training", "analysis"), start=1):
                meta = _page_meta(key)
                with dpg.child_window(width=-1, height=112, border=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"  {idx}", color=meta["accent"])
                        with dpg.group():
                            dpg.add_text(meta["label"], color=(226, 230, 235))
                            dpg.add_text(
                                meta["summary"], wrap=780, color=(184, 192, 202)
                            )
                            dpg.add_spacer(height=4)
                            widgets.primary_button(
                                f"Open {meta['label']}",
                                lambda s, a, u=key: self.switch(u),
                                height=30, width=200,
                            )
                dpg.add_spacer(height=8)

            dpg.add_spacer(height=8)
            dpg.add_text("Keyboard & UX tips", color=(166, 205, 241))
            dpg.add_separator()
            dpg.add_spacer(height=4)
            dpg.add_text(
                "- A green dot next to a path input means the file exists; red "
                "means the path is invalid.\n"
                "- Use the 'Recent' button to reuse a previously chosen path "
                "instead of navigating dialogs again.\n"
                "- Hover over any label or '?' icon for details about the "
                "field.\n"
                "- The 'Watch Folder' button on the Training page connects "
                "the live loss / metric plots to any run directory.",
                wrap=980, color=(190, 198, 210),
            )

    # ─── labeling page ────────────────────────────────────────────────
    def _build_labeling(self, parent: str) -> None:
        with dpg.child_window(
            tag="panel_labeling",
            parent=parent,
            width=-1,
            height=-1,
            border=False,
            show=False,
        ):
            dpg.add_text("Labeling Workspace", color=(130, 214, 148))
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text(
                "Use the annotator to draw boxes, assign track IDs, and label actions. "
                "Range labeling can apply IDs and actions across multiple frames so you do "
                "not need to edit every frame by hand.",
                wrap=860, color=(198, 204, 212),
            )
            dpg.add_spacer(height=10)

            with dpg.group(horizontal=True):
                with dpg.child_window(width=460, height=260, border=True):
                    dpg.add_text("Open the Annotator", color=(176, 212, 244))
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_text(
                        "Start with a blank project, load a video directly, or continue from an "
                        "existing annotation JSON file.",
                        wrap=420, color=(182, 188, 196),
                    )
                    dpg.add_spacer(height=10)
                    widgets.primary_button(
                        "Start New Annotation Session",
                        lambda: self._launch_annotator(), height=38,
                    )
                    dpg.add_spacer(height=6)
                    dpg.add_button(
                        label="Open Annotator With a Video",
                        height=34, width=-1,
                        callback=self._launch_annotator_with_video,
                    )
                    widgets.tooltip("Pick a video file; the annotator opens with it loaded.")
                    dpg.add_spacer(height=6)
                    dpg.add_button(
                        label="Open Existing Annotation JSON",
                        height=34, width=-1,
                        callback=self._launch_annotator_with_json,
                    )
                    widgets.tooltip("Continue editing a saved annotation project.")
                    dpg.add_spacer(height=10)
                    dpg.add_text("", tag="labeling_status",
                                 color=(184, 224, 186), wrap=420)

                with dpg.child_window(width=-1, height=260, border=True):
                    dpg.add_text("Recommended Workflow", color=(176, 212, 244))
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_text(
                        "1. Open a video or load an annotation project.\n"
                        "2. Draw boxes for key frames and assign the correct track.\n"
                        "3. Use Range Labeling to fill action labels across frame spans.\n"
                        "4. Save the annotation JSON and continue with dataset preparation.",
                        color=(188, 194, 202),
                    )
                    dpg.add_spacer(height=10)
                    dpg.add_text("Useful Shortcuts", color=(150, 173, 198))
                    dpg.add_text(
                        "Right / Left: move frame\n"
                        "Space: propagate boxes from previous frame\n"
                        "A: toggle auto-track\n"
                        "0-4: assign action to the selected box\n"
                        "Ctrl+S: save",
                        color=(170, 176, 184),
                    )

    # ─── nav state ────────────────────────────────────────────────────
    def _refresh_nav_state(self) -> None:
        current = _page_meta(self.current)
        for page in _PAGES:
            key = page["key"]
            label = page["label"]
            active = key == self.current
            button_label = f">  {label}" if active else f"    {label}"
            if dpg.does_item_exist(f"nav_{key}"):
                dpg.configure_item(f"nav_{key}", label=button_label)
            sub_tag = f"nav_sub_{key}"
            if dpg.does_item_exist(sub_tag):
                dpg.configure_item(
                    sub_tag,
                    color=page["accent"] if active else (140, 152, 168),
                )
        if dpg.does_item_exist("nav_page_title"):
            dpg.set_value("nav_page_title", current["label"])
        if dpg.does_item_exist("nav_page_summary"):
            dpg.set_value("nav_page_summary", current["summary"])

    def switch(self, key: str) -> None:
        self.current = key
        for page in _PAGES:
            panel_tag = f"panel_{page['key']}"
            if dpg.does_item_exist(panel_tag):
                dpg.configure_item(panel_tag, show=(page["key"] == key))
        self._refresh_nav_state()

    # ─── annotator launch helpers ─────────────────────────────────────
    def _launch_annotator(self, extra_args: Optional[list] = None) -> None:
        cmd = [sys.executable, "-m", "htrtdetr.annotator"] + (extra_args or [])
        try:
            subprocess.Popen(cmd, cwd=str(self.root))
            if dpg.does_item_exist("labeling_status"):
                dpg.set_value("labeling_status",
                              "The annotator opened in a separate window.")
        except Exception as exc:  # noqa: BLE001
            if dpg.does_item_exist("labeling_status"):
                dpg.set_value("labeling_status",
                              f"[ERROR] Failed to launch the annotator: {exc}")

    def _launch_annotator_with_video(self) -> None:
        path = dialogs.pick_file("Select Video", dialogs.VIDEO_TYPES)
        if path:
            self._launch_annotator([path])

    def _launch_annotator_with_json(self) -> None:
        path = dialogs.pick_file("Select Annotation JSON", dialogs.JSON_TYPES)
        if path:
            self._launch_annotator(["--load", path])

    # ─── main window ──────────────────────────────────────────────────
    def build_ui(self) -> None:
        with dpg.window(
            tag="hub_win",
            no_title_bar=True,
            no_move=True,
            no_resize=True,
            no_scrollbar=True,
        ):
            with dpg.group(horizontal=True):
                self._build_nav()
                with dpg.child_window(tag="content", width=-1, height=-1, border=False):
                    self._build_home("content")
                    self._build_labeling("content")
                    self.training.build("content")
                    self.analysis.build("content")

        dpg.set_primary_window("hub_win", True)
        self.switch("home")

    def run(self, argv=None) -> None:
        dpg.create_context()
        self._theme()
        dpg.create_viewport(
            title="YOAKE GUI - Home / Labeling / Training / Video Analysis",
            width=WINDOW_W,
            height=WINDOW_H,
            resizable=True,
        )
        dpg.setup_dearpygui()
        self.build_ui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            try:
                if self.current == "analysis":
                    self.analysis.tick()
            except Exception:  # noqa: BLE001
                pass
            dpg.render_dearpygui_frame()

        self.training.teardown()
        dpg.destroy_context()


def main(argv=None) -> None:
    Hub().run(argv)


if __name__ == "__main__":
    main()
