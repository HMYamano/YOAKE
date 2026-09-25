"""*Train on these labels* — the round dialog, and nothing but a view.

    Train on these labels                                    [x]

    312 labelled frames in 6 videos, 2 classes, 1,104 boxes.
    About 11 minutes on your GPU.
    60 frames from one video are held back, so the model is
    checked against a recording it never saw.
    Nothing you labelled changes, and nothing the model finds
    is written to annotations.json until you accept it.

       OK  Saving your labels
       OK  Collecting the frames        312 frames, 0 decoded
       >   Training the detector        epoch 14 of 96 - val loss 3.81 (best 3.77)
           Checking what it learned
       [========                            ]

     > Adjust this round...

    ────────────────────────────────────────────────────────
                                    [ Start ]  [ Close ]

**Zero decisions before it runs.** Every number is derived in
:mod:`htrtdetr.annotation.round` from the project itself, and the paragraph at
the top is that derivation read back in the user's units - which is the whole
of the cost quote a long job owes before it starts. A biologist has no way to
answer "epochs?" or "learning rate?", and a form that asks is a form that gets
a wrong answer or no answer at all. *Adjust this round...* is collapsed and
holds two things they genuinely can answer: how long they are willing to wait,
and whether to use the GPU.

Like :mod:`htrtdetr.gui.smart_annotate`, this module is the **view only**. It
runs nothing, spawns nothing, and imports neither torch nor cv2: the work
belongs on a worker thread or in a child process where it cannot freeze this
window. It reports what the caller tells it, and the phase list it reports into
is the shared one, so a user who has watched a sampling run can read a training
run without learning a second vocabulary.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import dearpygui.dearpygui as dpg

from . import phases, theme, widgets

__all__ = ["PHASES", "SAVE", "COLLECT", "TRAIN", "CHECK", "TrainRoundDialog"]

#: The four phases, in the order they happen. Named for what is going on rather
#: than for the subsystem doing it: "Collecting the frames" is what the user
#: sees happen, "exporting the dataset" is what the code calls it.
PHASES = (
    "Saving your labels",
    "Collecting the frames",
    "Training the detector",
    "Checking what it learned",
)
SAVE, COLLECT, TRAIN, CHECK = range(4)

#: How hard the round should try. Effort, not minutes and never epochs.
#:
#: It was minutes, and minutes were the wrong unit: the number of passes a
#: minute buys depends on the label count, so the same "12 minutes" meant four
#: hundred passes on a small project and three and a half thousand on a large
#: one - the same button quietly becoming a forty-five minute job as the study
#: grew. The budget is a number of passes now, and the plan paragraph converts
#: it back to minutes at the measured rate, so the quote and the work are the
#: same decision read two ways.
LENGTHS = ("Quick", "Normal", "Thorough")
_DEFAULT_LENGTH = "Normal"

_FORM, _RUNNING, _STOPPED, _DONE = "form", "running", "stopped", "done"


class TrainRoundDialog:
    """One window, one job. Safe to build twice."""

    WIDTH, HEIGHT = 580, 560

    def __init__(
        self,
        tag: str = "train_round",
        *,
        on_start: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        on_analyze: Optional[Callable[[], None]] = None,
        on_adopt: Optional[Callable[[], None]] = None,
        on_options: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> None:
        self.tag = tag
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_analyze = on_analyze
        self.on_adopt = on_adopt
        #: Called whenever a control changes, and asked for the new plan
        #: paragraph. The dialog never derives anything itself.
        self.on_options = on_options
        self.steps = phases.PhaseList(tag, PHASES)
        self._built = False
        self._phase = _FORM
        self._running = False
        self._gpu_reason = ""
        self._offer_adopt = False

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

    # ── building ─────────────────────────────────────────────────────────
    def build(self) -> None:
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)

        with dpg.window(
            tag=self.tag,
            label="Train on these labels",
            show=False,
            modal=True,
            no_collapse=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            on_close=lambda *_: self.hide(),
        ):
            # Body scrolls, buttons do not: the plan paragraph grows by a line
            # whenever the round has something extra to admit, and Start is the
            # thing that must never be the part below the fold.
            with widgets.dialog_body(self._t("body"), footer=64):
                dpg.add_text("", tag=self._t("plan"), color=theme.TEXT, wrap=self.WIDTH - 60)
                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)
                self.steps.build()
                widgets.gap(1)
                dpg.add_progress_bar(tag=self._t("bar"), default_value=0.0, width=-1, height=14)
                dpg.add_text(
                    "", tag=self._t("detail"), color=theme.TEXT_DIM, wrap=self.WIDTH - 60
                )
                widgets.gap(1)
                self._build_adjust()
                self._build_result()

            widgets.dialog_footer()
            self._build_actions()

        self._built = True
        self._set_phase(_FORM)

    def _build_adjust(self) -> None:
        with dpg.collapsing_header(
            label="Adjust this round...", default_open=False, tag=self._t("adjust")
        ):
            dpg.add_combo(
                list(LENGTHS),
                tag=self._t("length"),
                default_value=_DEFAULT_LENGTH,
                label="Effort",
                width=180,
                callback=lambda *_: self.refresh_plan(),
            )
            widgets.attach_tooltip(
                self._t("length"),
                "How hard this round should try. Normal is sized to keep the "
                "cycle turning rather than to produce a finished model - the "
                "paragraph above says what it comes to in minutes on this "
                "machine. Rounds after the first carry on from the last one, "
                "so they cost less again.",
            )
            dpg.add_checkbox(
                label="Use the GPU",
                tag=self._t("gpu"),
                default_value=True,
                callback=lambda *_: self.refresh_plan(),
            )
            widgets.attach_tooltip(
                self._t("gpu"),
                "Training runs on the processor without one, which works and "
                "is roughly ten times slower. The estimate above follows this.",
            )
            dpg.add_text(
                "Everything else is worked out from this project. The full "
                "controls - stages, learning rate, imbalance - are in the "
                "workflow window's Train page, which is where a final model "
                "is made.",
                color=theme.TEXT_FAINT,
                wrap=self.WIDTH - 80,
            )

    def _build_result(self) -> None:
        with dpg.group(tag=self._t("result_group"), show=False):
            dpg.add_separator()
            widgets.gap(1)
            dpg.add_text("", tag=self._t("result_title"), color=theme.ACCEPTED)
            dpg.add_text("", tag=self._t("result_body"), color=theme.TEXT, wrap=self.WIDTH - 60)
            widgets.gap(1)
            dpg.add_text("", tag=self._t("result_where"), color=theme.TEXT_FAINT,
                         wrap=self.WIDTH - 60)

    def _build_actions(self) -> None:
        with dpg.group(horizontal=True, tag=self._t("actions")):
            widgets.primary_button(
                "Start", self._on_start_clicked, height=34, width=160, tag=self._t("go")
            )
            widgets.secondary_button(
                "Stop", self._on_stop_clicked, height=32, width=140, tag=self._t("stop")
            )
            # The next action, not "Close". A round exists to make the next
            # pass better, so the button says what the next pass is - and it
            # says "next", because this button is taken ten times and the word
            # is what makes the tenth press feel like progress rather than a
            # repetition. It does not say "find animals in the videos": that
            # is the job at the end, once the rounds have stopped teaching,
            # and it lives in the Project menu where it cannot be pressed by
            # habit halfway through the loop.
            widgets.primary_button(
                "Pick the next frames to label",
                self._on_analyze_clicked,
                height=34,
                width=280,
                tag=self._t("analyze"),
            )
            widgets.attach_tooltip(
                self._t("analyze"),
                "Use the model you just trained to choose the next 50-200 "
                "frames worth your time, and bring its boxes back here to "
                "correct. It looks over the whole project but only returns "
                "what it chose.",
            )
            widgets.secondary_button(
                "Use this model anyway",
                self._on_adopt_clicked,
                height=32,
                width=200,
                tag=self._t("adopt"),
            )
            widgets.secondary_button(
                "Close", lambda *_: self.hide(), height=32, width=120, tag=self._t("close")
            )

    # ── phase switching ──────────────────────────────────────────────────
    def _set_phase(self, phase: str) -> None:
        """Body and buttons move together, so they can never disagree."""
        self._phase = phase
        running = phase == _RUNNING
        done = phase == _DONE
        self._show(self._t("adjust"), phase in (_FORM, _STOPPED))
        self._show(self._t("result_group"), done)
        self._show(self._t("go"), not running and not done)
        self._show(self._t("stop"), running)
        self._show(self._t("analyze"), done)
        self._show(self._t("adopt"), done and self._offer_adopt)
        self._show(self._t("close"), not running)
        if dpg.does_item_exist(self._t("go")):
            dpg.configure_item(
                self._t("go"), label="Start again" if phase == _STOPPED else "Start"
            )

    # ── the plan paragraph ───────────────────────────────────────────────
    def options(self) -> Dict[str, Any]:
        return {
            "effort": str(self._get(self._t("length"), _DEFAULT_LENGTH)),
            "use_gpu": bool(self._get(self._t("gpu"), False)) and not self._gpu_reason,
        }

    def refresh_plan(self) -> None:
        """Re-quote the cost. The dialog asks; the caller derives."""
        if not self._built or self.on_options is None:
            return
        try:
            self._set(self._t("plan"), self.on_options(self.options()))
        except Exception:  # noqa: BLE001 - a quote must never break the window
            pass

    def set_plan(self, text: str) -> None:
        self._set(self._t("plan"), text)

    def set_capability(self, *, gpu_reason: str = "") -> None:
        """Lock what this machine cannot do, with the reason (never hidden)."""
        self._gpu_reason = gpu_reason
        if not self._built:
            return
        if gpu_reason:
            self._set(self._t("gpu"), False)
            widgets.locked(self._t("gpu"), gpu_reason)
        else:
            widgets.unlock(self._t("gpu"))

    # ── callbacks ────────────────────────────────────────────────────────
    def _on_start_clicked(self, *_a: Any) -> None:
        if self._running:
            return
        self._running = True
        self._offer_adopt = False
        self.steps.reset()
        self.set_progress(0.0, "Starting.")
        self._set_phase(_RUNNING)
        if self.on_start is not None:
            self.on_start(self.options())

    def _on_stop_clicked(self, *_a: Any) -> None:
        if self.on_stop is not None:
            self.on_stop()
        # The caller confirms through :meth:`stopped`. Redrawing as finished
        # before the child has actually gone is a lie about a job that is
        # still holding a GPU.
        self.set_progress(None, "Stopping.")

    def _on_analyze_clicked(self, *_a: Any) -> None:
        self.hide()
        if self.on_analyze is not None:
            self.on_analyze()

    def _on_adopt_clicked(self, *_a: Any) -> None:
        if self.on_adopt is not None:
            self.on_adopt()
        self._offer_adopt = False
        self._show(self._t("adopt"), False)

    # ── the reporting surface the caller drives ──────────────────────────
    def set_step(self, index: int, state: str, detail: str = "") -> None:
        self.steps.set(index, state, detail)

    def advance_to_step(self, index: int, detail: str = "") -> None:
        self.steps.advance_to(index, detail)

    def set_progress(self, fraction: Optional[float], text: str = "") -> None:
        if fraction is not None and dpg.does_item_exist(self._t("bar")):
            dpg.set_value(self._t("bar"), max(0.0, min(1.0, float(fraction))))
        if text:
            self._set(self._t("detail"), text)

    def finish(self, *, title: str, body: str, where: str = "", offer_adopt: bool = False) -> None:
        """The round ended and produced something to say."""
        self._running = False
        self._offer_adopt = bool(offer_adopt)
        self.steps.finish()
        self.set_progress(1.0, "")
        self._set(self._t("result_title"), title)
        self._set(self._t("result_body"), body)
        self._set(self._t("result_where"), where)
        self._set_phase(_DONE)

    def fail(self, message: str, *, step: Optional[int] = None) -> None:
        """A failure explains itself and stays fixable in place."""
        self._running = False
        self.steps.fail(step, "")
        self.set_progress(None, message)
        self._set_phase(_STOPPED)

    def stopped(self, message: str = "") -> None:
        """A stop is a stop, never an error."""
        self._running = False
        self.set_progress(None, message or "Stopped.")
        self._set_phase(_STOPPED)

    # ── window ───────────────────────────────────────────────────────────
    def show(self, plan: str = "") -> None:
        if not self._built:
            self.build()
        self._running = False
        self._offer_adopt = False
        self.steps.reset()
        self.set_progress(0.0, "")
        self._set_phase(_FORM)
        if plan:
            self.set_plan(plan)
        else:
            self.refresh_plan()
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        dpg.configure_item(self.tag, show=True)

    def reopen(self) -> None:
        """Bring the window back **without resetting it**.

        Pressing Train while a round is going must not put the dialog back to
        the form: the form has no Stop button, and Stop is the only way to
        reach a running trainer. This is the difference between "show me what
        is happening" and "start over".
        """
        if not self._built:
            self.build()
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
        return self._running
