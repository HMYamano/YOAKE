"""*Export the verified dataset* — the last window in the study.

    Export the verified dataset                              [x]

    1,836,400 frames from 17 videos.
    3 videos are not signed off and were left out:
      fly_18.mp4, fly_19.mp4, fly_20.mp4

    Writes datasets/release/annotations.json, about 480 MB, and a
    path to each source video beside it. No image files: two million
    frames as JPEGs is about 86 GB and twenty hours of decoding, and
    everything in this project reads the video directly.

    ────────────────────────────────────────────────────────
                              [ Export ]  [ Close ]

**The excluded videos are named.** That is the whole reason this window exists
rather than a menu item that just writes the file. An export that silently
covers seventeen of twenty videos is indistinguishable, on disk, from one that
covers all twenty - and the difference is three hours of a biologist's
recordings missing from the paper.

View only: no torch, no cv2, no work. The export itself runs on a worker thread
in :mod:`htrtdetr.gui.final_operations`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import dearpygui.dearpygui as dpg

from . import theme, widgets

__all__ = ["ReleaseExportDialog"]

_FORM, _RUNNING, _DONE = "form", "running", "done"


class ReleaseExportDialog:
    """One window, one job. Safe to build twice."""

    WIDTH, HEIGHT = 600, 480

    def __init__(
        self,
        tag: str = "release_export",
        *,
        on_export: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_open_folder: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tag = tag
        self.on_export = on_export
        self.on_open_folder = on_open_folder
        self._built = False
        self._phase = _FORM

    def _t(self, suffix: str) -> str:
        return f"{self.tag}_{suffix}"

    @staticmethod
    def _set(tag: str, value: Any) -> None:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)

    @staticmethod
    def _show(tag: str, visible: bool) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=bool(visible))

    @staticmethod
    def _call(fn: Optional[Callable], *args: Any) -> None:
        if fn is not None:
            fn(*args)

    def build(self) -> None:
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)
        with dpg.window(
            tag=self.tag,
            label="Export the verified dataset",
            show=False,
            modal=True,
            no_collapse=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            on_close=lambda *_: self.hide(),
        ):
            with widgets.dialog_body(self._t("body"), footer=64):
                dpg.add_text("", tag=self._t("summary"), color=theme.TEXT,
                             wrap=self.WIDTH - 60)
                widgets.gap(1)
                # Named, listed, and in the accent that means "this is a thing
                # you have not finished" rather than the one that means error:
                # an unsigned video is not a failure, it is work left.
                dpg.add_text("", tag=self._t("excluded"), color=theme.PENDING,
                             wrap=self.WIDTH - 60)
                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)
                dpg.add_text("", tag=self._t("what"), color=theme.TEXT_DIM,
                             wrap=self.WIDTH - 60)
                widgets.gap(1)
                dpg.add_progress_bar(tag=self._t("bar"), default_value=0.0,
                                     width=-1, height=14, show=False)
                dpg.add_text("", tag=self._t("detail"), color=theme.TEXT_DIM,
                             wrap=self.WIDTH - 60)
                with dpg.group(tag=self._t("result_group"), show=False):
                    widgets.gap(1)
                    dpg.add_separator()
                    widgets.gap(1)
                    dpg.add_text("", tag=self._t("result"), color=theme.ACCEPTED,
                                 wrap=self.WIDTH - 60)
                    dpg.add_text("", tag=self._t("where"), color=theme.TEXT_FAINT,
                                 wrap=self.WIDTH - 60)

            widgets.dialog_footer()
            with dpg.group(horizontal=True):
                widgets.primary_button(
                    "Export", lambda *_: self._on_export(), height=34, width=160,
                    tag=self._t("go"),
                )
                widgets.secondary_button(
                    "Open the folder", lambda *_: self._call(self.on_open_folder),
                    height=32, width=180, tag=self._t("open"),
                )
                widgets.secondary_button(
                    "Close", lambda *_: self.hide(), height=32, width=120,
                    tag=self._t("close"),
                )
        self._built = True
        self._set_phase(_FORM)

    def _set_phase(self, phase: str) -> None:
        self._phase = phase
        self._show(self._t("go"), phase != _RUNNING)
        self._show(self._t("open"), phase == _DONE)
        self._show(self._t("bar"), phase == _RUNNING)
        self._show(self._t("result_group"), phase == _DONE)
        if dpg.does_item_exist(self._t("go")):
            dpg.configure_item(self._t("go"), enabled=phase != _RUNNING)

    def show(self, *, summary: str = "", excluded: str = "", what: str = "") -> None:
        self.build()
        self._set(self._t("summary"), summary)
        self._set(self._t("excluded"), excluded)
        self._set(self._t("what"), what)
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        dpg.configure_item(self.tag, show=True)

    def hide(self) -> None:
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, show=False)

    @property
    def visible(self) -> bool:
        return dpg.does_item_exist(self.tag) and dpg.is_item_shown(self.tag)

    def _on_export(self) -> None:
        self._set_phase(_RUNNING)
        self._call(self.on_export, {})

    def set_progress(self, fraction: float, detail: str = "") -> None:
        self._set(self._t("bar"), max(0.0, min(1.0, float(fraction))))
        if detail:
            self._set(self._t("detail"), detail)

    def finish(self, result: str, where: str = "") -> None:
        self._set_phase(_DONE)
        self._set(self._t("result"), result)
        self._set(self._t("where"), where)

    def fail(self, message: str) -> None:
        self._set_phase(_FORM)
        self._set(self._t("detail"), message)
