"""*Check every frame* — the last pass, and nothing but a view.

    Check every frame                                        [x]

    20 videos, 2,159,740 frames. About 2 hours on your GPU.
    It runs one video at a time and picks up where it stopped, so you
    can close this window, or the machine, and start it again.
    Nothing you labelled changes. The boxes go to a separate file that
    you sign off video by video, and only what you sign off is exported.

       OK  fly_01.mp4          108,000 frames, 213,904 boxes
       OK  fly_02.mp4          107,940 frames, 209,110 boxes
       >   fly_03.mp4          epoch -            41,300 of 108,000
           fly_04.mp4
           ...
       [========                            ]

    ────────────────────────────────────────────────────────
                                    [ Start ]  [ Close ]

**This is the end of the loop, not another turn of it.** Ten rounds of *pick
50-200 frames, correct them, train* produce a detector; this produces the
dataset. The difference is in what comes back: a round returns the frames it
chose and throws the rest of its scan away (:func:`annotation.batch.keep_chosen`),
and this returns every frame of every video.

**Why it is a menu row and not a fifth button on the loop.** The top bar's
second row is the loop, and the loop is taken twenty times. This is taken once,
at the end, and it costs two hours of machine time and an evening of the user's.
A button beside *Analyze* would be pressed by habit on round three, which is
exactly when it is a two-hour mistake.

Like :mod:`htrtdetr.gui.train_round`, this module is the **view only**: it runs
nothing, spawns nothing, and imports neither torch nor cv2. The work is in
:mod:`htrtdetr.gui.final_operations`, on a worker thread, where it cannot
freeze this window. The per-video states are the shared five words from
:mod:`htrtdetr.gui.phases`, so a user who has watched a round can read this.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import dearpygui.dearpygui as dpg

from . import phases, theme, widgets

__all__ = ["FinalPassDialog"]

_FORM, _RUNNING, _STOPPED, _DONE = "form", "running", "stopped", "done"

#: Rows drawn in the per-video list. Twenty videos fit; a hundred do not, and
#: dearpygui draws every row whether or not it is scrolled into view.
MAX_ROWS = 40


class FinalPassDialog:
    """One window, one job. Safe to build twice."""

    WIDTH, HEIGHT = 620, 600

    def __init__(
        self,
        tag: str = "final_pass",
        *,
        on_start: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_review: Optional[Callable[[], None]] = None,
        on_export: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tag = tag
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_review = on_review
        self.on_export = on_export
        self.videos: List[str] = []
        self.steps: Optional[phases.PhaseList] = None
        self._built = False
        self._phase = _FORM

    # ── tiny dpg helpers ─────────────────────────────────────────────────
    def _t(self, suffix: str) -> str:
        return f"{self.tag}_{suffix}"

    @staticmethod
    def _set(tag: str, value: Any) -> None:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)

    @staticmethod
    def _get(tag: str, default: Any = None) -> Any:
        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    @staticmethod
    def _show(tag: str, visible: bool) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=bool(visible))

    @staticmethod
    def _call(fn: Optional[Callable], *args: Any) -> None:
        if fn is not None:
            fn(*args)

    # ── building ─────────────────────────────────────────────────────────
    def build(self, videos: Sequence[str] = ()) -> None:
        """Create the window, with one phase row per video in the project.

        The row list is rebuilt rather than reused because it is derived from
        the project, and a project can gain a video between two openings of
        this dialog. ``phases.PhaseList`` is built against a fixed label list,
        so a new list means a new one.
        """
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)
        self.videos = [str(v) for v in videos][:MAX_ROWS]
        self.steps = phases.PhaseList(self.tag, self.videos, wrap=self.WIDTH - 80)

        with dpg.window(
            tag=self.tag,
            label="Check every frame",
            show=False,
            modal=True,
            no_collapse=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            on_close=lambda *_: self.hide(),
        ):
            with widgets.dialog_body(self._t("body"), footer=64):
                dpg.add_text("", tag=self._t("quote"), color=theme.TEXT, wrap=self.WIDTH - 60)
                widgets.gap(1)
                # The guarantee, stated before the button and not after it.
                # This is the one operation in the annotator that produces
                # boxes the user never drew and never accepted one at a time,
                # so it owes them an explicit account of what it does not
                # touch (P3: machine output is never written to human work).
                dpg.add_text(
                    "Nothing you labelled changes. The boxes go to a separate "
                    "file that you sign off video by video, and only what you "
                    "sign off reaches the exported dataset.",
                    color=theme.TEXT_DIM,
                    wrap=self.WIDTH - 60,
                )
                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)
                if self.videos:
                    self.steps.build()
                else:
                    dpg.add_text(
                        "No videos in this project yet.", color=theme.TEXT_FAINT
                    )
                widgets.gap(1)
                dpg.add_progress_bar(tag=self._t("bar"), default_value=0.0, width=-1, height=14)
                dpg.add_text("", tag=self._t("detail"), color=theme.TEXT_DIM,
                             wrap=self.WIDTH - 60)
                widgets.gap(1)
                self._build_result()

            widgets.dialog_footer()
            self._build_actions()

        self._built = True
        self._set_phase(_FORM)

    def _build_result(self) -> None:
        with dpg.group(tag=self._t("result_group"), show=False):
            dpg.add_separator()
            widgets.gap(1)
            dpg.add_text("", tag=self._t("result_title"), color=theme.ACCEPTED)
            dpg.add_text("", tag=self._t("result_body"), color=theme.TEXT,
                         wrap=self.WIDTH - 60)

    def _build_actions(self) -> None:
        with dpg.group(horizontal=True, tag=self._t("actions")):
            widgets.primary_button(
                "Start", lambda *_: self._on_start(), height=34, width=160,
                tag=self._t("go"),
            )
            widgets.secondary_button(
                "Stop", lambda *_: self._call(self.on_stop), height=32, width=140,
                tag=self._t("stop"),
            )
            # The next action, as everywhere else in this loop. Finding the
            # boxes is the cheap half; checking them is the half that makes
            # them a dataset, so the button after a finished pass is the one
            # that starts the checking.
            widgets.primary_button(
                "Check what it found", lambda *_: self._call(self.on_review),
                height=34, width=240, tag=self._t("review"),
            )
            widgets.secondary_button(
                "Export the dataset...", lambda *_: self._call(self.on_export),
                height=32, width=200, tag=self._t("export"),
            )
            widgets.secondary_button(
                "Close", lambda *_: self.hide(), height=32, width=120,
                tag=self._t("close"),
            )

    # ── phase switching ──────────────────────────────────────────────────
    def _set_phase(self, phase: str) -> None:
        """Body and buttons move together, so they can never disagree."""
        self._phase = phase
        running = phase == _RUNNING
        done = phase == _DONE
        self._show(self._t("go"), not running and not done)
        self._show(self._t("stop"), running)
        self._show(self._t("review"), done)
        self._show(self._t("export"), done)
        self._show(self._t("close"), not running)
        self._show(self._t("result_group"), done or phase == _STOPPED)
        if dpg.does_item_exist(self._t("go")):
            dpg.configure_item(
                self._t("go"),
                label="Carry on" if phase == _STOPPED else "Start",
            )

    # ── the caller's interface ───────────────────────────────────────────
    def show(self, *, videos: Sequence[str] = (), quote: str = "") -> None:
        self.build(videos)
        self._set(self._t("quote"), quote)
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        dpg.configure_item(self.tag, show=True)

    def hide(self) -> None:
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, show=False)

    @property
    def visible(self) -> bool:
        return dpg.does_item_exist(self.tag) and dpg.is_item_shown(self.tag)

    @property
    def running(self) -> bool:
        return self._phase == _RUNNING

    def _on_start(self) -> None:
        self._set_phase(_RUNNING)
        self._call(self.on_start, {})

    def mark(self, index: int, state: str, detail: str = "") -> None:
        """One video's row. Safe to call from a marshalled worker callback."""
        if self.steps is not None:
            self.steps.set(index, state, detail)

    def advance_to(self, index: int, detail: str = "") -> None:
        if self.steps is not None:
            self.steps.advance_to(index, detail)

    def set_progress(self, fraction: float, detail: str = "") -> None:
        self._set(self._t("bar"), max(0.0, min(1.0, float(fraction))))
        if detail:
            self._set(self._t("detail"), detail)

    def finish(self, *, title: str, body: str) -> None:
        self._set_phase(_DONE)
        if self.steps is not None:
            self.steps.finish()
        self.set_progress(1.0)
        self._set(self._t("result_title"), title)
        self._set(self._t("result_body"), body)

    def stopped(self, message: str) -> None:
        """A stopped pass keeps what it wrote, and the message says which.

        Resumability is the only reason this operation is usable at all - two
        hours is longer than anyone will sit in front of a modal - so the
        sentence a stop produces has to be the one that makes starting again
        feel free rather than like beginning a two-hour job twice.
        """
        self._set_phase(_STOPPED)
        self._set(self._t("result_title"), "Stopped.")
        self._set(self._t("result_body"), message)

    def fail(self, message: str, index: Optional[int] = None) -> None:
        self._set_phase(_STOPPED)
        if self.steps is not None:
            self.steps.fail(index, "")
        self._set(self._t("result_title"), "That did not work.")
        self._set(self._t("result_body"), message)
