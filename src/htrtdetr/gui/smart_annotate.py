"""
smart_annotate.py — the dialog that replaces "which frames should I label?"

Opening a 30-minute video used to present 54,000 frames and no guidance. The
user scrolled to somewhere near the start, drew boxes until they were bored,
and produced a training set that was 200 near-identical frames of one pose.

Smart Annotate asks two questions instead — *what are you annotating* and *how
much time do you have* — and answers the rest:

    Scope          [ Entire project      v ]
    Videos         6 videos  (2:41:08, 241,900 frames)

    Annotate       ( ) Individuals   ( ) Behavior   (o) Individuals + behavior
    Budget         ( ) 50   (o) 100   ( ) 200 samples      [ by time ]
    Start without a trained model: pick frames, then click each animal.
    > Advanced     strategy, seed, clip length, scan stride, score threshold

    [ Run ]

    Reading the video ................ done
    Generating candidates ............ done    412 boxes
    Checking tracks and behavior ..... done
    Computing embeddings ............. working
    Choosing informative frames ...... waiting
    [=================                     ]

    Video 3 of 6: courtship_03.mp4
    [=========                             ]
    courtship_01: completed
    courtship_02: cached
                                                        [ Cancel ]

**This is the only dialog for the whole engine.** Auto Annotate Project, Smart
Sample Project and Smart Annotate were three windows in front of one
:func:`~htrtdetr.gui.workspace_operations.start_project_batch`, differing in
which checkbox started ticked. They are now two menu entries — *Pre-label
videos with the model* and *Pick frames to label* — into this form, which is
why it carries a scope, two progress bars and a per-video result list that the
single-video version never needed.

This module is the **view only**. It collects a
:class:`~htrtdetr.annotation.sampling.SamplingRequest`, hands it to the caller,
and renders whatever progress the caller reports back. It runs nothing, spawns
nothing and imports neither torch nor cv2 — the work belongs on a worker thread
or in a child process, where it cannot freeze this window (§31).

The estimate is live
--------------------
Changing the target, the budget or the *Generate candidates* checkbox updates
the estimated workload immediately, before anything runs. That is the whole
point of a budget: the user sees that 15 minutes buys 150 assisted frames or
36 unassisted ones, and picks accordingly (P17 — quote the cost before the user
commits).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import dearpygui.dearpygui as dpg

from ..annotation import sampling
from . import theme, widgets

__all__ = [
    "STEPS",
    "StepState",
    "SmartAnnotateDialog",
    "review_label",
    "step_for_operation",
]

#: The five phases, in order (§32). Named as nouns-in-progress so a half-done
#: list reads as a state, not as a log.
#:
#: The order is :class:`~htrtdetr.annotation.batch.BatchRunner`'s, not the one
#: these phases were first imagined in. The runner detects, then checks, then
#: encodes, then scans; a list written in any other order marches backwards
#: while the run is going forwards, and a progress display that goes backwards
#: reads as a broken run rather than as a mislabelled list.
STEPS: Tuple[str, ...] = (
    "Reading the video",
    "Generating annotation candidates",
    "Checking tracks and behavior",
    "Computing embeddings",
    "Choosing informative frames",
)

#: The runner's ``operation`` strings, by the step each one is. Matched as a
#: prefix so the runner can keep saying "(model loading may take a while)"
#: without this table knowing about it.
_OPERATION_STEPS: Tuple[Tuple[str, int], ...] = (
    ("Detecting animals", 1),
    ("Using cached predictions", 1),
    ("Checking tracks and behavior", 2),
    ("Encoding frames", 3),
    ("Choosing informative frames", 4),
    ("Completed", 4),
)


def step_for_operation(operation: str) -> int:
    """Which of :data:`STEPS` the runner is in, from what it says it is doing."""
    for prefix, index in _OPERATION_STEPS:
        if str(operation).startswith(prefix):
            return index
    return 0


def review_label(candidates: int, issues: int) -> str:
    """The finished-run review button's label, or ``""`` for nothing to do.

    The count is in the label because the button is the only place the run's
    outcome and the next action meet: "Review 412 candidates" says both what
    happened and what to press. An empty string is the caller's signal to show
    no button at all — one that does nothing when pressed teaches that the
    feature is broken.
    """
    parts = []
    if candidates > 0:
        parts.append(f"{candidates:,} candidate{'s' if candidates != 1 else ''}")
    if issues > 0:
        parts.append(f"{issues:,} location{'s' if issues != 1 else ''}")
    return "Review " + " and ".join(parts) if parts else ""


#: Phase states, and the glyph each shows. ASCII so it renders before the font
#: loads, and so it survives a console that is not UTF-8.
StepState = str
_STEP_GLYPH: Dict[StepState, str] = {
    "waiting": "  ",
    "running": "> ",
    "done": "OK",
    "skipped": "--",
    "failed": "!!",
}
_STEP_COLOR: Dict[StepState, Any] = {
    "waiting": theme.TEXT_FAINT,
    "running": theme.ACCENT,
    "done": theme.ACCEPTED,
    "skipped": theme.TEXT_FAINT,
    "failed": theme.DANGER,
}

_SAMPLE_CHOICES: Tuple[int, ...] = (50, 100, 200)
_MINUTE_CHOICES: Tuple[int, ...] = (5, 15, 30)

_TARGETS: Tuple[Tuple[str, str], ...] = (
    (sampling.TARGET_DETECTION, "Individuals"),
    (sampling.TARGET_BEHAVIOR, "Behavior"),
    (sampling.TARGET_BOTH, "Individuals + behavior"),
)


#: The phases the dialog can be in. The body and the button row are always
#: switched together, by :meth:`SmartAnnotateDialog._set_phase`, so the Cancel
#: button and the progress block can never disagree about whether a run is
#: happening. ``_STOPPED`` keeps the progress block and its last message on
#: screen while offering the form again — a failure explains itself and stays
#: fixable in place (§39).
_FORM, _RUNNING, _STOPPED, _DONE = "form", "running", "stopped", "done"


class SmartAnnotateDialog:
    """The modal. One instance per annotator; shown and hidden, never rebuilt."""

    WIDTH = 560
    HEIGHT = 660

    def __init__(
        self,
        tag: str = "smart_annotate",
        *,
        on_analyze: Optional[Callable[[sampling.SamplingRequest, Dict[str, Any]], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        on_apply: Optional[Callable[[Any], None]] = None,
        on_review: Optional[Callable[[], None]] = None,
        on_pick_model: Optional[Callable[[], None]] = None,
        on_scope: Optional[Callable[[str], Tuple[str, int, float, int]]] = None,
    ) -> None:
        self.tag = tag
        self.on_analyze = on_analyze
        self.on_cancel = on_cancel
        self.on_apply = on_apply
        self.on_review = on_review
        self.on_pick_model = on_pick_model
        #: ``scope -> (label, frames, fps, video count)``. The dialog never
        #: looks at the project itself; it asks, so that changing the scope
        #: can re-quote the size of what is about to run.
        self.on_scope = on_scope

        self.fps = 25.0
        self.num_frames = 0
        self.selection: Any = None
        self._running = False
        self._built = False
        self._phase = _FORM
        #: Why "Use the GPU" is locked, and why there is no model, if there
        #: is not. The GPU is locked in place (P4); the missing model is not,
        #: because "choose one" is an action and a dimmed checkbox is not.
        self._gpu_reason = ""
        self._predict_reason = ""
        #: ``True`` once the scope holds more than one video, which is when
        #: "3 of 6" and the per-video result lines start meaning something.
        self._project_mode = False
        self._review_label = ""
        self._step_states: Dict[int, StepState] = {}

    # ── plumbing ─────────────────────────────────────────────────────────
    def _t(self, suffix: str) -> str:
        return f"{self.tag}_{suffix}"

    @staticmethod
    def _set(tag: str, value: Any) -> None:
        if dpg.does_item_exist(tag):
            try:
                dpg.set_value(tag, value)
            except SystemError:
                pass

    @staticmethod
    def _get(tag: str, default: Any = None) -> Any:
        if not dpg.does_item_exist(tag):
            return default
        try:
            return dpg.get_value(tag)
        except SystemError:
            return default

    @staticmethod
    def _show(tag: str, visible: bool) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=bool(visible))

    # ── construction ─────────────────────────────────────────────────────
    def build(self) -> None:
        """Create the window, hidden. Safe to call twice."""
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)

        with dpg.window(
            tag=self.tag,
            label="Pick frames to label",
            show=False,
            modal=True,
            no_collapse=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            on_close=lambda *_: self.hide(),
        ):
            # Body scrolls, buttons do not. The form alone is taller than the
            # window once Advanced is open, and the result section adds four
            # more blocks under it, so "Analyze video" and "Start annotating"
            # both used to end up off the bottom exactly when they were the
            # thing to press.
            with widgets.dialog_body(self._t("body"), footer=64):
                dpg.add_text("", tag=self._t("video"), color=theme.TEXT)
                dpg.add_text("", tag=self._t("video_note"), color=theme.TEXT_DIM)
                self._build_form()
                self._build_estimate()

                self._build_progress()
                self._build_result()

            widgets.dialog_footer()
            self._build_actions()

        self._built = True
        self._set_phase(_FORM)
        self._refresh_estimate()

    def _build_actions(self) -> None:
        """The pinned button row. One phase's buttons are shown at a time."""
        with dpg.group(horizontal=True, tag=self._t("actions")):
            widgets.primary_button(
                "Pick frames", self._on_analyze_clicked, height=34, width=200,
                tag=self._t("go"),
            )
            widgets.secondary_button(
                "Cancel", self._on_cancel_clicked, height=32, width=140,
                tag=self._t("cancel"),
            )
            widgets.primary_button(
                "Start annotating", self._on_apply_clicked, height=34, width=230,
                tag=self._t("apply"),
            )
            # A run that pre-labelled but sampled nothing used to end on a
            # button reading "Close". The candidates it just produced are the
            # next action, so they get a button that says how many.
            widgets.primary_button(
                "Review", self._on_review_clicked, height=34, width=280,
                tag=self._t("review"),
            )
            widgets.secondary_button(
                "Change options",
                self._on_reset_clicked,
                height=32,
                width=150,
                tag=self._t("again"),
            )

    def _set_phase(self, phase: str, *, go_label: str = "") -> None:
        """Show the one phase the dialog is in, body and buttons together.

        Keep the settings in memory while showing just the current task.
        """
        self._phase = phase
        self._show(self._t("form"), phase in (_FORM, _STOPPED))
        self._show(self._t("estimate_card"), phase in (_FORM, _STOPPED))
        self._show(self._t("progress_group"), phase in (_RUNNING, _STOPPED))
        self._show(self._t("result_group"), phase == _DONE)
        self._show(self._t("go"), phase != _DONE)
        self._show(self._t("cancel"), phase == _RUNNING)
        self._show(self._t("apply"), phase == _DONE)
        self._show(self._t("review"), phase == _DONE and bool(self._review_label))
        self._show(self._t("again"), phase == _DONE)
        if dpg.does_item_exist(self._t("go")):
            dpg.configure_item(
                self._t("go"),
                enabled=phase != _RUNNING,
                label=go_label
                or ("Working..." if phase == _RUNNING else "Pick frames"),
            )

    # -- the form ---------------------------------------------------------
    def _build_form(self) -> None:
        with dpg.group(tag=self._t("form")):
            from ..annotation.project import SCOPES

            with dpg.group(tag=self._t("nomodel_group"), show=False):
                dpg.add_text(
                    "No trained model needed. Pick frames, then press C and click "
                    "each animal to fit a box. W draws a box by hand.",
                    color=theme.TEXT, wrap=self.WIDTH - 80,
                )
            dpg.add_combo(
                list(SCOPES),
                tag=self._t("scope"),
                default_value="Current video",
                label="Scope",
                width=300,
                callback=lambda *_: self._on_scope_change(),
            )
            dpg.add_checkbox(
                label="Balance samples across videos", tag=self._t("balanced"), default_value=True
            )
            dpg.add_combo(
                [label for _, label in _TARGETS],
                tag=self._t("target"), default_value=_TARGETS[2][1],
                label="Annotate", width=300,
                callback=lambda *_: self._refresh_estimate(),
            )
            dpg.add_text("Labeling budget", color=theme.TEXT_DIM)
            with dpg.group(horizontal=True):
                dpg.add_combo(
                    ["By sample count", "By time"],
                    tag=self._t("budget_mode"), default_value="By sample count",
                    width=180, callback=lambda *_: self._on_budget_mode(),
                )
                with dpg.group(tag=self._t("budget_samples")):
                    dpg.add_combo(
                        [f"{n} samples" for n in _SAMPLE_CHOICES],
                        tag=self._t("samples"), default_value="100 samples", width=150,
                        callback=lambda *_: self._refresh_estimate(),
                    )
                with dpg.group(tag=self._t("budget_minutes"), show=False):
                    dpg.add_combo(
                        [f"{n} minutes" for n in _MINUTE_CHOICES],
                        tag=self._t("minutes"), default_value="15 minutes", width=150,
                        callback=lambda *_: self._refresh_estimate(),
                    )

            with dpg.group(tag=self._t("prelabel_group")):
                dpg.add_checkbox(
                    label="Generate annotation candidates",
                    tag=self._t("prelabel"),
                    default_value=True,
                    callback=lambda *_: self._on_sample_toggle(),
                )
                widgets.attach_tooltip(
                    self._t("prelabel"),
                    "Run the configured detector on videos in scope so you review boxes "
                    "instead of drawing them. This is what makes a 15-minute "
                    "budget buy four times as many frames.",
                )
            # Not the same switch as the one below it. This one decides
            # whether frames are chosen at all; that one decides whether the
            # choosing is informed or evenly spaced. Off skips the frame scan
            # entirely, which is most of the cost of a pre-label-only run.
            dpg.add_checkbox(
                label="Choose which frames to label",
                tag=self._t("sample"),
                default_value=True,
                callback=lambda *_: self._on_sample_toggle(),
            )
            widgets.attach_tooltip(
                self._t("sample"),
                "On also proposes the frames and clips worth your time, and "
                "marks them on the timeline. Off only pre-labels, which skips "
                "the frame scan and is much faster.",
            )
            # Lives inside the pre-label group because that is the only thing
            # it affects: ``BatchOptions.device`` is read in two places and
            # both sit under ``if options.detect:``. With no checkpoint it was
            # a tickable switch that changed nothing, next to a message
            # explaining that there is no model - and its old tooltip claimed
            # it moved the encoder too, which the segmentation helper decides
            # for itself in its own process.
            with dpg.group(tag=self._t("gpu_group")):
                dpg.add_checkbox(label="Use the GPU", tag=self._t("gpu"), default_value=True)
                widgets.attach_tooltip(
                    self._t("gpu"),
                    "Runs the detector on the GPU when one is available. It "
                    "also runs on the CPU, more slowly. Frame scanning is CPU "
                    "work either way.",
                )

            with dpg.collapsing_header(
                label="Advanced", default_open=False, tag=self._t("advanced")
            ):
                with dpg.group(tag=self._t("model_setup"), show=False):
                    dpg.add_text(
                        "Already trained a detector? Add it to generate box candidates.",
                        color=theme.TEXT_DIM, wrap=self.WIDTH - 60,
                    )
                    widgets.secondary_button(
                        "Choose a model...", self._on_pick_model_clicked,
                        width=220, height=28, tag=self._t("pick_model"),
                    )
                    dpg.add_text(
                        "", tag=self._t("nomodel_reason"), color=theme.TEXT_DIM,
                        wrap=self.WIDTH - 60,
                    )
                dpg.add_combo(
                    [sampling.STRATEGY_LABELS[k] for k in sampling.STRATEGIES],
                    tag=self._t("strategy"),
                    default_value=sampling.STRATEGY_LABELS[sampling.DIVERSITY],
                    label="Strategy",
                    width=250,
                    callback=lambda *_: self._on_strategy_change(),
                )
                dpg.add_text(
                    "", tag=self._t("strategy_help"), color=theme.TEXT_DIM, wrap=self.WIDTH - 60
                )
                dpg.add_input_int(
                    label="Random seed",
                    tag=self._t("seed"),
                    default_value=0,
                    width=130,
                    step=1,
                    min_value=0,
                    min_clamped=True,
                )
                widgets.attach_tooltip(
                    self._t("seed"),
                    "The same seed selects the same frames. It is saved with "
                    "the project so a selection can be reproduced.",
                )
                dpg.add_input_float(
                    label="Clip length (s)",
                    tag=self._t("clip"),
                    default_value=3.0,
                    width=130,
                    step=0.5,
                    min_value=0.5,
                    min_clamped=True,
                    format="%.1f",
                    callback=lambda *_: self._refresh_estimate(),
                )
                dpg.add_input_int(
                    label="Scan every N frames",
                    tag=self._t("stride"),
                    default_value=15,
                    width=130,
                    step=5,
                    min_value=1,
                    min_clamped=True,
                    callback=lambda *_: self._refresh_estimate(),
                )
                widgets.attach_tooltip(
                    self._t("stride"),
                    "How densely the videos are scanned for the frames you "
                    "could be offered. Worked out from how much video is in "
                    "scope: a pool of about 20,000 places is already far more "
                    "than 200 picks can use, and every extra place is a frame "
                    "the detector runs on again next round. Lower is slower "
                    "and finds shorter events.",
                )
                # The one knob in here that belongs to the detector rather
                # than to the sampler, so it is the one that disappears with
                # the detector.
                with dpg.group(tag=self._t("threshold_group")):
                    dpg.add_input_float(
                        label="Candidate score threshold",
                        tag=self._t("threshold"),
                        default_value=0.3,
                        width=130,
                        step=0.05,
                        min_value=0.0,
                        max_value=1.0,
                        min_clamped=True,
                        max_clamped=True,
                        format="%.2f",
                    )
                    widgets.attach_tooltip(
                        self._t("threshold"),
                        "How sure the detector has to be before a box is "
                        "proposed. Lower proposes more and rejects more; "
                        "nothing it proposes is written until you accept it.",
                    )

    def _build_estimate(self) -> None:
        with dpg.child_window(tag=self._t("estimate_card"), height=92, border=True):
            dpg.add_text("", tag=self._t("estimate"), color=theme.TEXT, wrap=self.WIDTH - 60)
            dpg.add_text(
                "", tag=self._t("estimate_note"), color=theme.TEXT_FAINT, wrap=self.WIDTH - 60
            )

    def _build_progress(self) -> None:
        with dpg.group(tag=self._t("progress_group"), show=False):
            dpg.add_separator()
            widgets.gap(1)
            for index, label in enumerate(STEPS):
                with dpg.group(horizontal=True):
                    dpg.add_text("  ", tag=self._t(f"step_glyph_{index}"), color=theme.TEXT_FAINT)
                    dpg.add_text(label, tag=self._t(f"step_{index}"), color=theme.TEXT_FAINT)
                    dpg.add_text("", tag=self._t(f"step_detail_{index}"), color=theme.TEXT_FAINT)
            widgets.gap(1)
            dpg.add_progress_bar(tag=self._t("bar"), default_value=0.0, width=-1, height=14)
            dpg.add_text(
                "", tag=self._t("progress_text"), color=theme.TEXT_DIM, wrap=self.WIDTH - 40
            )
            # "Video 3 of 6" is not a step and cannot be one: the steps
            # describe one video and a project run has several. This is what
            # the old Auto Annotate Project window carried, and a six-video
            # run is unreadable without it -- including which video failed,
            # since one failure never stops the rest.
            with dpg.group(tag=self._t("project_group"), show=False):
                widgets.gap(1)
                dpg.add_text("", tag=self._t("scope_progress"), color=theme.TEXT)
                dpg.add_progress_bar(
                    tag=self._t("overall_bar"), default_value=0.0, width=-1, height=14
                )
                dpg.add_input_text(
                    tag=self._t("results"), multiline=True, readonly=True, height=72, width=-1
                )

    def _build_result(self) -> None:
        with dpg.group(tag=self._t("result_group"), show=False):
            dpg.add_separator()
            widgets.gap(1)
            dpg.add_text("", tag=self._t("result_title"), color=theme.ACCEPTED)
            dpg.add_text("", tag=self._t("result_body"), color=theme.TEXT, wrap=self.WIDTH - 40)
            widgets.gap(1)

            # Why each sample was chosen (Priority 6). A breakdown by reason,
            # in plain words: a user who cannot tell why 72 frames were picked
            # has been handed a black box, and one shown a raw score has been
            # handed a number they cannot act on. The reasons are the middle
            # ground, and the arithmetic stays behind Details.
            dpg.add_text(
                "", tag=self._t("result_reasons"), color=theme.TEXT_DIM, wrap=self.WIDTH - 40
            )
            theme.bind_mono(dpg.last_item())
            with dpg.collapsing_header(
                label="Details", default_open=False, tag=self._t("result_details")
            ):
                dpg.add_text(
                    "",
                    tag=self._t("result_detail_body"),
                    color=theme.TEXT_FAINT,
                    wrap=self.WIDTH - 60,
                )
                theme.bind_mono(dpg.last_item())

    # ── capability gating ────────────────────────────────────────────────
    def set_capability(self, *, gpu_reason: str = "", predict_reason: str = "") -> None:
        """Lock the options this machine cannot honour.

        Locked, never hidden (P4): a dimmed checkbox with the reason in its
        tooltip teaches that the feature exists and what would turn it on.
        Hiding it teaches that it does not exist.
        """
        self._gpu_reason = gpu_reason
        self._predict_reason = predict_reason
        if not self._built:
            return
        if gpu_reason:
            self._set(self._t("gpu"), False)
            widgets.locked(self._t("gpu"), gpu_reason)
        else:
            widgets.unlock(self._t("gpu"))
        if predict_reason:
            self._set(self._t("prelabel"), False)
            self._set(self._t("nomodel_reason"), predict_reason)
        self._show(self._t("prelabel_group"), not predict_reason)
        self._show(self._t("nomodel_group"), bool(predict_reason))
        self._show(self._t("model_setup"), bool(predict_reason))
        # The GPU switch and the score threshold are about the detector, so
        # they go where the detector went. The rest of Advanced - strategy,
        # seed, clip length, scan stride - tunes the sampler, which runs with
        # no checkpoint at all, so it stays.
        self._show(self._t("gpu_group"), not predict_reason)
        self._show(self._t("threshold_group"), not predict_reason)
        self._on_sample_toggle()

    # ── reading the form ─────────────────────────────────────────────────
    def request(self) -> sampling.SamplingRequest:
        """The current form, as a :class:`SamplingRequest`."""
        target_label = self._get(self._t("target"), _TARGETS[2][1])
        target = next(
            (key for key, label in _TARGETS if label == target_label), sampling.TARGET_BOTH
        )

        by_time = self._get(self._t("budget_mode"), "") == "By time"
        if by_time:
            minutes = _first_int(self._get(self._t("minutes"), "15 minutes"), 15)
            budget = sampling.Budget(minutes=float(minutes))
        else:
            samples = _first_int(self._get(self._t("samples"), "100 samples"), 100)
            budget = sampling.Budget(samples=samples)

        strategy = sampling.DIVERSITY
        label = self._get(self._t("strategy"), "")
        for key, text in sampling.STRATEGY_LABELS.items():
            if text == label:
                strategy = key
                break

        return sampling.SamplingRequest(
            target=target,
            budget=budget,
            strategy=strategy,
            seed=int(self._get(self._t("seed"), 0) or 0),
            fps=self.fps,
            clip_seconds=float(self._get(self._t("clip"), 3.0) or 3.0),
            prelabelled=self.options()["generate_candidates"],
        )

    def options(self) -> Dict[str, Any]:
        """Everything that is not part of the sampling request itself."""
        return {
            "scope": self._get(self._t("scope"), "Current video"),
            "balanced": bool(self._get(self._t("balanced"), True)),
            "use_gpu": bool(self._get(self._t("gpu"), False)) and not self._gpu_reason,
            "generate_candidates": (
                bool(self._get(self._t("prelabel"), False)) and not self._predict_reason
            ),
            "sample": bool(self._get(self._t("sample"), True))
            or not (self._get(self._t("prelabel"), False) and not self._predict_reason),
            "stride": max(1, int(self._get(self._t("stride"), 15) or 15)),
            "score_threshold": float(self._get(self._t("threshold"), 0.3) or 0.3),
        }

    # ── callbacks ────────────────────────────────────────────────────────
    def _on_budget_mode(self) -> None:
        by_time = self._get(self._t("budget_mode"), "") == "By time"
        self._show(self._t("budget_samples"), not by_time)
        self._show(self._t("budget_minutes"), by_time)
        self._refresh_estimate()

    def _on_sample_toggle(self) -> None:
        prelabel = self.options()["generate_candidates"]
        if not prelabel:
            self._set(self._t("sample"), True)
        self._show(self._t("sample"), prelabel)
        sample = bool(self._get(self._t("sample"), True))
        if dpg.does_item_exist(self._t("strategy")):
            dpg.configure_item(self._t("strategy"), enabled=sample)
        self._refresh_estimate()

    def _on_scope_change(self) -> None:
        """Re-quote the size of what the new scope covers."""
        if self.on_scope is None:
            return
        try:
            name, frames, fps, videos = self.on_scope(
                self._get(self._t("scope"), "Current video")
            )
        except Exception:
            return
        self.set_subject(name, frames, fps, videos)

    def _on_strategy_change(self) -> None:
        label = self._get(self._t("strategy"), "")
        for key, text in sampling.STRATEGY_LABELS.items():
            if text == label:
                self._set(self._t("strategy_help"), sampling.STRATEGY_HELP.get(key, ""))
                return
        self._set(self._t("strategy_help"), "")

    def _on_analyze_clicked(self, *_a: Any) -> None:
        if self._running:
            return
        self._running = True
        self.selection = None
        self._review_label = ""
        self._set_phase(_RUNNING)
        self.reset_steps()
        self.clear_results()
        self.set_progress(0.0, "Starting.")
        self.set_overall(0.0, "")
        if self.on_analyze is not None:
            self.on_analyze(self.request(), self.options())

    def _on_cancel_clicked(self, *_a: Any) -> None:
        if self.on_cancel is not None:
            self.on_cancel()
        # The caller confirms the stop through :meth:`stopped`; this only
        # reports that the request was made, because a Cancel that redraws as
        # finished before the worker has actually stopped is a lie (P6).
        self.set_progress(None, "Stopping.")

    def _on_apply_clicked(self, *_a: Any) -> None:
        if self.on_apply is not None and self.selection is not None:
            self.on_apply(self.selection)
        self.hide()

    def _on_review_clicked(self, *_a: Any) -> None:
        self.hide()
        if self.on_review is not None:
            self.on_review()

    def _on_pick_model_clicked(self, *_a: Any) -> None:
        if self.on_pick_model is not None:
            self.on_pick_model()

    def _on_reset_clicked(self, *_a: Any) -> None:
        self._running = False
        self._set_phase(_FORM)

    # ── the live estimate ────────────────────────────────────────────────
    def _refresh_estimate(self) -> None:
        if not self._built:
            return
        if not self._get(self._t("sample"), True):
            self._set(
                self._t("estimate"),
                "No frames will be selected. This run only pre-labels the "
                "videos in scope.",
            )
            self._set(
                self._t("estimate_note"),
                "Turn on 'Choose which frames to label' to have the run "
                "propose where to spend your time.",
            )
            return
        request = self.request()
        frames, clips = request.budget.counts(
            target=request.target, model=request.workload, prelabelled=request.prelabelled
        )

        parts: List[str] = []
        if frames:
            parts.append(f"{frames:,} frame{'s' if frames != 1 else ''}")
        if clips:
            seconds = clips * request.clip_seconds
            parts.append(
                f"{clips:,} clip{'s' if clips != 1 else ''} "
                f"of {request.clip_seconds:g} s ({seconds:.0f} s total)"
            )
        self._set(
            self._t("estimate"),
            (
                "This will select " + " and ".join(parts) + "."
                if parts
                else "Nothing would be selected."
            ),
        )

        estimate = request.workload.describe(frames, clips, request.prelabelled)
        note = f"Your labelling afterwards: {estimate}."
        if not request.prelabelled and not self._predict_reason:
            assisted = request.workload.describe(frames, clips, True)
            note += f" With candidates generated it would be about " f"{assisted.lstrip('~')}."
        scan = self._scan_estimate(request)
        if scan:
            # Said before the button, not discovered after it. The only number
            # here used to be the human one, so pressing Analyze over a
            # twenty-hour project began a scan of well over an hour that
            # nothing had mentioned.
            note = f"{scan} {note}"
        self._set(self._t("estimate_note"), note)

    #: Seconds to step over one frame without decoding it, measured on
    #: 640x480 30fps H.264 with the grab-forward reader in ``annotation.batch``
    #: (4.0 ms per scanned frame at stride 15, so 0.27 ms per frame stepped
    #: over).
    #:
    #: **The scan is priced per frame walked, not per frame scanned**, which is
    #: why a coarser stride barely moves this number: the reader demuxes every
    #: frame in the video either way. A previous version of this estimate
    #: divided by the stride and so promised that doubling it halved the wait.
    SECONDS_PER_WALKED_FRAME = 0.00027

    #: Seconds to seek to an arbitrary frame and decode it. The reader gives up
    #: on walking and seeks once the gap is over 48 frames, which caps the
    #: per-scanned-frame cost - measured ~16 ms, against the ~16.8 ms a
    #: stride-60 walk costs, which is where the two meet.
    SECONDS_PER_SEEK = 0.016

    def _scan_estimate(self, request) -> str:
        """``"Scanning takes about 7 minutes."`` for the scope, or ``""``."""
        if not self.num_frames or not self._get(self._t("sample"), True):
            return ""
        stride = max(1, int(self._get(self._t("stride"), 15) or 15))
        passes = 2 if self._encoder_pass_likely() else 1
        per_scanned = min(self.SECONDS_PER_SEEK, stride * self.SECONDS_PER_WALKED_FRAME)
        seconds = (self.num_frames / stride) * per_scanned * passes
        if seconds < 90:
            return f"Scanning takes about {max(5, int(round(seconds / 5) * 5))} seconds."
        if seconds < 5400:
            return f"Scanning takes about {max(2, int(round(seconds / 60)))} minutes."
        return f"Scanning takes about {seconds / 3600:.1f} hours."

    def _encoder_pass_likely(self) -> bool:
        """Whether the run reads every scanned frame twice.

        It does when an embedding encoder is available: one pass to encode and
        one to compute frame features, over the same list.
        """
        return True

    # ── the caller's interface ───────────────────────────────────────────
    def set_subject(
        self, video_name: str = "", num_frames: int = 0, fps: float = 25.0, videos: int = 1
    ) -> None:
        """Say what the run will cover: one video, or a scope's worth of them.

        ``videos`` is the count in scope. Above one, the run reports "3 of 6"
        and a line per video, because a single bar over six videos says
        nothing about which one is slow and nothing about which one failed.
        """
        self.fps = float(fps) if fps > 0 else 25.0
        self.num_frames = int(num_frames)
        self._project_mode = int(videos) > 1
        # The scan stride follows the size of the scope, so opening the dialog
        # on one video and on twenty does not propose the same scan. It is
        # written into the visible field rather than kept beside it: Advanced
        # is where a user overrides it for this run, and a hidden value the
        # field disagrees with is the kind of thing that gets reported as the
        # stride "not doing anything".
        if self.num_frames > 0:
            self._set(self._t("stride"), sampling.scan_stride(self.num_frames, fps=self.fps))
        self._show(self._t("balanced"), self._project_mode)

        # "Video 3 of 6", the overall bar and the per-video result list. Built
        # hidden, so this is the only thing that reveals them; without it
        # ``set_overall`` and ``add_result`` write into widgets nobody sees and
        # a six-video run reports as if it were one.
        self._show(self._t("project_group"), self._project_mode)
        self._set(self._t("video"), video_name or "No video")
        if num_frames > 0:
            minutes = num_frames / self.fps / 60.0
            self._set(
                self._t("video_note"),
                f"{num_frames:,} frames, {minutes:.0f} minutes at " f"{self.fps:g} fps",
            )
        elif self._project_mode:
            self._set(
                self._t("video_note"),
                "Total length appears once every video in scope has been read.",
            )
        else:
            self._set(self._t("video_note"), "")
        self._refresh_estimate()

    def show(
        self,
        *,
        video_name: str = "",
        num_frames: int = 0,
        fps: float = 25.0,
        videos: int = 1,
        scope: str = "",
        prelabel: Optional[bool] = None,
        sample: Optional[bool] = None,
        strategy: str = "",
        title: str = "",
    ) -> None:
        """Open the dialog for one video, or for a scope full of them.

        Everything here is a **preset**: where the one form starts, for the
        entry point that opened it. The window, the engine and the progress
        display do not change.

        ``title`` may name a control - *Find animals with the model* names the
        pre-label checkbox - only because the caller offering that caption is
        gated on a checkpoint existing, so the control it names is on screen
        whenever the caption is. Without that gate a caption could name the
        very checkbox :meth:`set_capability` is about to hide, which is what
        retired the last window that tried it.
        """
        if not self._built:
            self.build()
        if scope:
            self._set(self._t("scope"), scope)
        if prelabel is not None and not self._predict_reason:
            self._set(self._t("prelabel"), bool(prelabel))
        if sample is not None:
            self._set(self._t("sample"), bool(sample))
        if strategy:
            self._set(self._t("strategy"), sampling.STRATEGY_LABELS.get(strategy, strategy))
        if dpg.does_item_exist(self.tag):
            caption = title or "Pick frames to label"
            if self._predict_reason:
                caption = "Pick frames to label"
            dpg.configure_item(self.tag, label=caption)
        self.set_subject(video_name, num_frames, fps, videos)
        self._on_reset_clicked()
        self._on_sample_toggle()
        self._on_strategy_change()
        self._refresh_estimate()
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

    def set_step(self, index: int, state: StepState, detail: str = "") -> None:
        """Mark one phase. Safe from a worker thread's marshalled callback."""
        if not (0 <= index < len(STEPS)):
            return
        self._step_states[index] = state
        glyph = _STEP_GLYPH.get(state, "  ")
        color = _STEP_COLOR.get(state, theme.TEXT_FAINT)
        for tag, value in (
            (self._t(f"step_glyph_{index}"), glyph),
            (self._t(f"step_{index}"), STEPS[index]),
            (self._t(f"step_detail_{index}"), detail),
        ):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
                dpg.configure_item(tag, color=color)

    def reset_steps(self) -> None:
        """Back to waiting, for the next video in the scope."""
        self._step_states.clear()
        for index in range(len(STEPS)):
            self.set_step(index, "waiting")

    def advance_to_step(self, index: int, detail: str = "") -> None:
        """``index`` is running; everything before it is done.

        The runner reports what it is doing, not what it has finished, so the
        phases before the current one are completed by inference. Anything
        already marked failed or skipped keeps that mark.
        """
        for earlier in range(max(0, min(index, len(STEPS)))):
            if self._step_states.get(earlier) not in ("done", "skipped", "failed"):
                self.set_step(earlier, "done")
        self.set_step(index, "running", detail)

    def set_progress(self, fraction: Optional[float], text: str = "") -> None:
        """Move the bar and the line under it. ``None`` leaves the bar alone."""
        if fraction is not None and dpg.does_item_exist(self._t("bar")):
            dpg.set_value(self._t("bar"), max(0.0, min(1.0, float(fraction))))
        if text:
            self._set(self._t("progress_text"), text)

    def set_overall(self, fraction: Optional[float], text: str = "") -> None:
        """The second bar: how far through the scope, not through one video."""
        if fraction is not None and dpg.does_item_exist(self._t("overall_bar")):
            dpg.set_value(self._t("overall_bar"), max(0.0, min(1.0, float(fraction))))
        if text:
            self._set(self._t("scope_progress"), text)

    def add_result(self, line: str) -> None:
        """One finished video, with its status and the reason if it failed."""
        if not dpg.does_item_exist(self._t("results")):
            return
        current = self._get(self._t("results"), "") or ""
        self._set(self._t("results"), current + line.rstrip() + chr(10))

    def clear_results(self) -> None:
        self._set(self._t("results"), "")

    def finish(
        self,
        selection: Any,
        *,
        summary: str = "",
        predictions: int = 0,
        issues: int = 0,
        total_samples: Optional[int] = None,
    ) -> None:
        """Report success and offer the buttons that follow from it (§32)."""
        self._running = False
        self.selection = selection
        self._review_label = review_label(predictions, issues)
        self._set_phase(_DONE)

        self._set(
            self._t("result_title"),
            "Project analysis complete" if total_samples is not None else "Video analysis complete",
        )

        lines: List[str] = []
        if predictions:
            lines.append(f"{predictions:,} predictions generated")
        if issues:
            lines.append(f"{issues:,} locations need review")
        if selection is not None:
            total = selection.total if total_samples is None else total_samples
            lines.append(f"Recommended samples: {total:,}")
            estimate = selection.describe().splitlines()
            if total_samples is None and estimate and estimate[-1].startswith("Estimated"):
                lines.append(estimate[-1])
        if summary:
            lines.append(summary)
        self._set(self._t("result_body"), "\n".join(lines) or "Nothing to do.")

        self._set(self._t("result_reasons"), reason_breakdown(selection))
        self._set(self._t("result_detail_body"), reason_details(selection))

        total = 0 if selection is None else selection.total
        if total_samples is not None:
            total = total_samples
        # Two things a run can produce and two buttons for them. A run that
        # only pre-labelled has no samples to start on, and used to end on
        # "Close" with 412 unreviewed candidates behind it.
        if dpg.does_item_exist(self._t("apply")):
            dpg.configure_item(
                self._t("apply"),
                label=f"Start on {total:,} samples" if total else "Close",
                show=bool(total) or not self._review_label,
                enabled=True,
            )
        if dpg.does_item_exist(self._t("review")):
            dpg.configure_item(
                self._t("review"),
                label=self._review_label or "Review",
                show=bool(self._review_label) and not total,
            )
            # One primary action at a time: when there are samples to start
            # on, that is the point of the run and Review is the other door.
            theme.bind(
                self._t("review"),
                theme.ghost_theme() if total else theme.accent_theme(),
            )

    def fail(self, message: str, *, step: Optional[int] = None) -> None:
        """Report a failure without closing anything (§39).

        The dialog stays open with the settings intact, so the user can change
        one thing and run again rather than reopening and retyping.
        """
        self._running = False
        if step is not None:
            self.set_step(step, "failed")
        self._set_phase(_STOPPED, go_label="Try again")
        self.set_progress(None, message)

    def stopped(self, message: str = "") -> None:
        """A stop is a stop, never an error (P6)."""
        self._running = False
        self._set_phase(_STOPPED)
        self.set_progress(None, message or "Analysis stopped. Nothing was changed.")


def _first_int(text: Any, default: int) -> int:
    """``"100 samples"`` -> ``100``. A radio label is the value's only carrier
    in dearpygui, so the label is parsed rather than mapped through a dict that
    could fall out of step with it."""
    digits = ""
    for ch in str(text):
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    try:
        return int(digits) if digits else int(default)
    except ValueError:
        return int(default)


# ══════════════════════════════════════════════════════════════════════════
# Explaining a selection (Priority 6)
# ══════════════════════════════════════════════════════════════════════════

#: Internal reason -> the words a user reads. The internal names are the
#: sampler's signal names; these say what they *mean* for the video.
REASON_WORDS: Dict[str, str] = {
    "representative": "Representative",
    "uncertain": "Model uncertainty",
    "interaction": "Interaction",
    "rare": "Rare appearance",
    "novel": "Unlike what you have labelled",
    "inconsistent": "Tracking ambiguity",
    "random": "Random",
    "uniform": "Evenly spaced",
    "difficult": "Difficult",
}


def _words(reason: str) -> str:
    return REASON_WORDS.get(reason, str(reason).replace("_", " ").capitalize())


def reason_breakdown(selection: Any) -> str:
    """The block under the headline::

        38 Representative
        16 Interaction
         8 Rare appearance
         6 Tracking ambiguity

    Sorted by count, so the first line says what the sampler mostly did. No
    scores and no formulae: a number the user cannot act on is worse than no
    number at all (``docs/GUI_DESIGN.md`` P12).
    """
    if selection is None:
        return ""
    counts: Dict[str, int] = {}
    for reason in (getattr(selection, "reasons", None) or {}).values():
        label = _words(reason) + " frames"
        counts[label] = counts.get(label, 0) + 1
    for clip in getattr(selection, "clips", ()) or ():
        label = _words(getattr(clip, "reason", "representative")) + " clips"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return ""
    width = max(len(str(n)) for n in counts.values())
    return "\n".join(
        f"{n:>{width}}  {label}"
        for label, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )


def reason_details(selection: Any) -> str:
    """What sits behind the Details fold: the settings that produced this
    selection, and the frames it picked. Enough to reproduce a run, and nothing
    that needs interpreting."""
    if selection is None:
        return ""
    frames = list(getattr(selection, "frames", ()) or ())
    clips = list(getattr(selection, "clips", ()) or ())
    lines = [
        f"strategy   {getattr(selection, 'strategy', '?')}",
        f"seed       {getattr(selection, 'seed', 0)}",
        f"scanned    {getattr(selection, 'pool_size', 0):,} frames",
        f"selected   {len(frames):,} frames, {len(clips):,} clips",
    ]
    if frames:
        shown = ", ".join(str(f) for f in frames[:24])
        lines.append("frames     " + shown + (" ..." if len(frames) > 24 else ""))
    for clip in clips[:12]:
        lines.append(
            f"clip       {clip.start:,}-{clip.end:,}  " f"{_words(getattr(clip, 'reason', ''))}"
        )
    if len(clips) > 12:
        lines.append(f"           ... and {len(clips) - 12} more")
    return "\n".join(lines)
