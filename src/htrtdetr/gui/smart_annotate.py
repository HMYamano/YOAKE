"""
smart_annotate.py — the dialog that replaces "which frames should I label?"

Opening a 30-minute video used to present 54,000 frames and no guidance. The
user scrolled to somewhere near the start, drew boxes until they were bored,
and produced a training set that was 200 near-identical frames of one pose.

Smart Annotate asks two questions instead — *what are you annotating* and *how
much time do you have* — and answers the rest:

    Video          courtship_01.mp4  (27:14, 40,850 frames)

    Annotate       ( ) Individuals   ( ) Behavior   (o) Individuals + behavior
    Budget         ( ) 50   (o) 100   ( ) 200 samples      [ by time ]
    Options        [x] Generate annotation candidates
                   [x] Choose the informative samples automatically
                   [x] Use the GPU
    > Advanced     strategy, random seed, clip length, scan stride

    [ Analyze video ]

    Scanning frames .......... done   1,362 frames
    Computing embeddings ..... done
    Finding clips ............ working
    Generating predictions ... waiting
    Checking consistency ..... waiting
                                                        [ Cancel ]

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

__all__ = ["STEPS", "StepState", "SmartAnnotateDialog"]

#: The five phases, in order (§32). Named as nouns-in-progress so a half-done
#: list reads as a state, not as a log.
STEPS: Tuple[str, ...] = (
    "Scanning frames",
    "Computing embeddings",
    "Finding representative frames and clips",
    "Generating annotation candidates",
    "Checking temporal consistency",
)

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
    ) -> None:
        self.tag = tag
        self.on_analyze = on_analyze
        self.on_cancel = on_cancel
        self.on_apply = on_apply

        self.fps = 25.0
        self.num_frames = 0
        self.selection: Any = None
        self._running = False
        self._built = False
        self._phase = _FORM
        #: Why "Use the GPU" and "Generate candidates" are locked, if they are.
        self._gpu_reason = ""
        self._predict_reason = ""

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
            label="Smart Annotate",
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
            with widgets.dialog_body(self._t("body")):
                dpg.add_text("", tag=self._t("video"), color=theme.TEXT)
                dpg.add_text("", tag=self._t("video_note"), color=theme.TEXT_DIM)
                widgets.gap(2)

                self._build_form()
                widgets.gap(1)
                self._build_estimate()
                widgets.gap(2)

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
                "Analyze video", self._on_analyze_clicked, height=34, width=200,
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
            widgets.secondary_button(
                "Change the settings and run again",
                self._on_reset_clicked,
                height=32,
                width=260,
                tag=self._t("again"),
            )

    def _set_phase(self, phase: str, *, go_label: str = "") -> None:
        """Show the one phase the dialog is in, body and buttons together.

        The form itself is never hidden: the settings that produced a result
        are the settings the next run starts from.
        """
        self._phase = phase
        self._show(self._t("progress_group"), phase in (_RUNNING, _STOPPED))
        self._show(self._t("result_group"), phase == _DONE)
        self._show(self._t("go"), phase != _DONE)
        self._show(self._t("cancel"), phase == _RUNNING)
        self._show(self._t("apply"), phase == _DONE)
        self._show(self._t("again"), phase == _DONE)
        if dpg.does_item_exist(self._t("go")):
            dpg.configure_item(
                self._t("go"),
                enabled=phase != _RUNNING,
                label=go_label
                or ("Analyzing..." if phase == _RUNNING else "Analyze video"),
            )

    # -- the form ---------------------------------------------------------
    def _build_form(self) -> None:
        with dpg.group(tag=self._t("form")):
            from ..annotation.project import SCOPES

            dpg.add_combo(
                list(SCOPES),
                tag=self._t("scope"),
                default_value="Current video",
                label="Scope",
                width=300,
            )
            dpg.add_checkbox(
                label="Balance samples across videos", tag=self._t("balanced"), default_value=True
            )
            widgets.section_header("What are you annotating?")
            with dpg.group(horizontal=True):
                dpg.add_radio_button(
                    [label for _, label in _TARGETS],
                    tag=self._t("target"),
                    default_value=_TARGETS[2][1],
                    horizontal=False,
                    callback=lambda *_: self._refresh_estimate(),
                )
            widgets.gap(1)

            widgets.section_header(
                "How much time do you have?",
                "The count is exact. The duration is an estimate, and it says so.",
            )
            dpg.add_radio_button(
                ["By sample count", "By time"],
                tag=self._t("budget_mode"),
                default_value="By sample count",
                horizontal=True,
                callback=lambda *_: self._on_budget_mode(),
            )
            with dpg.group(tag=self._t("budget_samples")):
                dpg.add_radio_button(
                    [f"{n} samples" for n in _SAMPLE_CHOICES],
                    tag=self._t("samples"),
                    default_value="100 samples",
                    horizontal=True,
                    callback=lambda *_: self._refresh_estimate(),
                )
            with dpg.group(tag=self._t("budget_minutes"), show=False):
                dpg.add_radio_button(
                    [f"{n} minutes" for n in _MINUTE_CHOICES],
                    tag=self._t("minutes"),
                    default_value="15 minutes",
                    horizontal=True,
                    callback=lambda *_: self._refresh_estimate(),
                )
            widgets.gap(1)

            widgets.section_header("Options")
            dpg.add_checkbox(
                label="Generate annotation candidates",
                tag=self._t("prelabel"),
                default_value=True,
                callback=lambda *_: self._refresh_estimate(),
            )
            widgets.attach_tooltip(
                self._t("prelabel"),
                "Run the configured detector on videos in scope so you review boxes "
                "instead of drawing them. This is what makes a 15-minute "
                "budget buy four times as many frames.",
            )
            dpg.add_checkbox(
                label="Choose the informative samples automatically",
                tag=self._t("smart"),
                default_value=True,
                callback=lambda *_: self._on_smart_toggle(),
            )
            widgets.attach_tooltip(
                self._t("smart"),
                "Off means evenly spaced frames. On means frames chosen to "
                "cover what the video actually contains.",
            )
            dpg.add_checkbox(label="Use the GPU", tag=self._t("gpu"), default_value=True)
            widgets.attach_tooltip(
                self._t("gpu"),
                "Runs the detector and the encoder on the GPU when one is "
                "available. Everything here also runs on the CPU, more slowly.",
            )

            widgets.gap(1)
            with dpg.collapsing_header(
                label="Advanced", default_open=False, tag=self._t("advanced")
            ):
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
                    "How densely the video is scanned for features. Lower is "
                    "slower and finds shorter events.",
                )

    def _build_estimate(self) -> None:
        with dpg.child_window(tag=self._t("estimate_card"), height=76, border=True):
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
            widgets.locked(self._t("prelabel"), predict_reason)
        else:
            widgets.unlock(self._t("prelabel"))
        self._refresh_estimate()

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
        if not self._get(self._t("smart"), True):
            strategy = sampling.UNIFORM
        else:
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
            prelabelled=bool(self._get(self._t("prelabel"), True)),
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
            "stride": max(1, int(self._get(self._t("stride"), 15) or 15)),
        }

    # ── callbacks ────────────────────────────────────────────────────────
    def _on_budget_mode(self) -> None:
        by_time = self._get(self._t("budget_mode"), "") == "By time"
        self._show(self._t("budget_samples"), not by_time)
        self._show(self._t("budget_minutes"), by_time)
        self._refresh_estimate()

    def _on_smart_toggle(self) -> None:
        smart = bool(self._get(self._t("smart"), True))
        if dpg.does_item_exist(self._t("strategy")):
            dpg.configure_item(self._t("strategy"), enabled=smart)
        self._refresh_estimate()

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
        self._set_phase(_RUNNING)
        for index in range(len(STEPS)):
            self.set_step(index, "waiting")
        self.set_progress(0.0, "Starting.")
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

    def _on_reset_clicked(self, *_a: Any) -> None:
        self._running = False
        self._set_phase(_FORM)

    # ── the live estimate ────────────────────────────────────────────────
    def _refresh_estimate(self) -> None:
        if not self._built:
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
        note = f"Estimated review workload: {estimate}."
        if not request.prelabelled:
            assisted = request.workload.describe(frames, clips, True)
            note += f" With candidates generated it would be about " f"{assisted.lstrip('~')}."
        note += " The estimate is a guess, not a measurement."
        self._set(self._t("estimate_note"), note)

    # ── the caller's interface ───────────────────────────────────────────
    def show(self, *, video_name: str = "", num_frames: int = 0, fps: float = 25.0) -> None:
        """Open the dialog for one video."""
        if not self._built:
            self.build()
        self.fps = float(fps) if fps > 0 else 25.0
        self.num_frames = int(num_frames)

        self._set(self._t("video"), video_name or "No video")
        if num_frames > 0:
            minutes = num_frames / self.fps / 60.0
            self._set(
                self._t("video_note"),
                f"{num_frames:,} frames, {minutes:.0f} minutes at " f"{self.fps:g} fps",
            )
        else:
            self._set(self._t("video_note"), "")

        self._on_reset_clicked()
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

    def set_progress(self, fraction: Optional[float], text: str = "") -> None:
        """Move the bar and the line under it. ``None`` leaves the bar alone."""
        if fraction is not None and dpg.does_item_exist(self._t("bar")):
            dpg.set_value(self._t("bar"), max(0.0, min(1.0, float(fraction))))
        if text:
            self._set(self._t("progress_text"), text)

    def finish(
        self,
        selection: Any,
        *,
        summary: str = "",
        predictions: int = 0,
        issues: int = 0,
        total_samples: Optional[int] = None,
    ) -> None:
        """Report success and offer the one button that follows from it (§32)."""
        self._running = False
        self.selection = selection
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
        if dpg.does_item_exist(self._t("apply")):
            dpg.configure_item(
                self._t("apply"),
                label=f"Start on {total:,} samples" if total else "Close",
                enabled=True,
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
