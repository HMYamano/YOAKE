"""
settings.py — everything that describes the *project*, out of the way of the work.

The annotator's old right pane held eleven always-open panels. Four of them were
properties of whatever was selected, and those belong in the Inspector. Two were
tools, and those belong in the rail. The rest were **settings**: which OpenCV
tracker to use, which Click to Box backend, the label vocabularies, the project
folder, how many animals the experiment has. A setting is decided once and then
lived with, so it has no business competing with the image for attention on every
frame.

Six tabs, one window, built hidden at startup:

=============  ===========================================================
Project        Folder, expected animal count, autosave state
Labels         Class and action vocabularies
Tracking       Auto-track, the OpenCV backend this build actually has
Model          The detector checkpoint, and whether it can run here
Assistance     Click to Box backend, drawing behaviour
Checks         The nine detectors' thresholds, all of them advanced
=============  ===========================================================

**Built hidden, never lazily.** Every widget exists from startup, so the
annotator's refresh helpers — ``_refresh_labels_display``,
``_refresh_project_dir_display``, ``_refresh_backend_ui``,
``_toggle_auto_track_key`` — keep finding their tags whether or not the user has
ever opened this window. Building it on first open would make those helpers
silently no-op until then, which is the kind of bug that only shows up for
somebody else.

**The legacy tags are kept deliberately.** ``auto_track_cb``, ``sticky_draw_cb``,
``click_backend_*``, ``classes_display``, ``actions_display`` and
``project_dir_text`` moved here *with their names*, so eight existing callbacks
and four refresh helpers work unchanged. The widget moved; the contract did not.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import dearpygui.dearpygui as dpg

from ..annotation import quality
from . import click_segment, theme, widgets

__all__ = ["SettingsDialog", "TABS", "THRESHOLD_FIELDS"]

TABS: Tuple[str, ...] = ("Project", "Labels", "Tracking", "Model", "Assistance", "Checks")

#: The :class:`~htrtdetr.annotation.quality.QualityConfig` fields worth exposing,
#: with the sentence that says what changing one does. Every other field on that
#: dataclass stays code-level: a setting nobody can predict the effect of is a
#: setting that only generates support questions.
THRESHOLD_FIELDS: Tuple[Tuple[str, str, str, float, float], ...] = (
    (
        "low_score",
        "Low confidence below",
        "Detections weaker than this are queued for review.",
        0.05,
        0.95,
    ),
    (
        "confident_score",
        "Confident at or above",
        "Detections at least this strong read as predicted rather than pending.",
        0.1,
        0.99,
    ),
    (
        "position_jump_factor",
        "Position jump (x usual step)",
        "A box that moves this many times further than the track usually does "
        "in one frame is flagged.",
        2.0,
        20.0,
    ),
    (
        "size_jump_factor",
        "Size jump (x area)",
        "A box whose area changes by this factor in one frame is flagged.",
        1.2,
        5.0,
    ),
    (
        "orientation_jump_deg",
        "Orientation jump (degrees)",
        "A rotated box turning more than this in one frame is flagged. Measured "
        "mod 180, because a head-tail flip is not representable.",
        10.0,
        90.0,
    ),
    (
        "overlap_iou",
        "Overlap at IoU",
        "Two boxes overlapping this much are flagged as a possible double " "detection.",
        0.1,
        0.9,
    ),
    (
        "merged_area_factor",
        "Merged box (x median area)",
        "A box this many times the usual area may hold two animals.",
        1.2,
        4.0,
    ),
)


class SettingsDialog:
    """The project's settings. One window, five tabs, shown on demand."""

    WIDTH = 560
    HEIGHT = 620

    def __init__(
        self,
        tag: str = "settings",
        *,
        callbacks: Optional[Mapping[str, Callable]] = None,
    ) -> None:
        self.tag = tag
        self.callbacks: Dict[str, Callable] = dict(callbacks or {})
        self._built = False

    # ── plumbing ─────────────────────────────────────────────────────────
    def _t(self, suffix: str) -> str:
        return f"{self.tag}_{suffix}"

    def _call(self, name: str, *args: Any) -> None:
        fn = self.callbacks.get(name)
        if fn is not None:
            fn(*args)

    def _cb(self, name: str, *args: Any) -> Callable:
        return lambda *_dpg: self._call(name, *args)

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

    # ── construction ─────────────────────────────────────────────────────
    def build(
        self,
        *,
        tracker_label: str = "",
        tracker_hint: str = "",
        backend_tips: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Create the window, hidden. Safe to call twice."""
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)
        tips = dict(backend_tips or {})

        with dpg.window(
            tag=self.tag,
            label="Project settings",
            show=False,
            modal=True,
            no_collapse=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            on_close=lambda *_: self.hide(),
        ):
            # The tabs scroll; Done does not. The Checks tab is seven sliders
            # and two buttons tall, so with everything in one flow Done ended
            # up below the bottom edge of a 620 px window and the only way out
            # of Settings was the title bar's close box.
            with widgets.dialog_body(self._t("body")):
                with dpg.tab_bar(tag=self._t("tabs")):
                    self._build_project_tab()
                    self._build_labels_tab()
                    self._build_tracking_tab(tracker_label, tracker_hint)
                    self._build_model_tab()
                    self._build_assistance_tab(tips)
                    self._build_checks_tab()

            widgets.dialog_footer()
            with dpg.group(horizontal=True):
                widgets.primary_button(
                    "Done", lambda *_: self.hide(), width=140, height=32, tag=self._t("done")
                )
                dpg.add_text("", tag=self._t("note"), color=theme.TEXT_DIM)

        self._built = True

    # -- Project ----------------------------------------------------------
    def _build_project_tab(self) -> None:
        with dpg.tab(label="Project", tag=self._t("tab_project")):
            widgets.gap(1)
            dpg.add_text("Project folder", color=theme.TEXT)
            dpg.add_text(
                "(not set)", tag="project_dir_text", color=theme.TEXT_FAINT, wrap=self.WIDTH - 60
            )
            dpg.add_text(
                "Ctrl+S writes annotations.json here without asking, and "
                "autosave only runs once this is set.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )
            widgets.secondary_button(
                "Change folder...",
                self._cb("set_project_dir"),
                width=200,
                height=28,
                tag=self._t("pick_dir"),
            )

            dpg.add_combo(
                ["Stay on last frame", "Open next unfinished video"],
                label="At video end",
                tag="settings_end_behavior",
                default_value="Stay on last frame",
                width=270,
                callback=lambda s, a: self._call("end_of_video", a),
            )
            # No "Guided Annotation" button here. It is a Help topic, it is in
            # the Help menu, and a settings tab is not where anyone looks for
            # one.
            widgets.gap(2)
            widgets.section_header(
                "Animals per frame", "The single most useful thing you can tell the checks."
            )
            dpg.add_text(
                "With a fixed count, a frame holding fewer becomes a review "
                "item instead of a silent gap, and a box large enough for two "
                "is flagged as a possible merge. This is knowledge the "
                "detector does not have and you always do.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )
            dpg.add_radio_button(
                ["Fixed", "Variable"],
                tag="expected_mode",
                horizontal=True,
                default_value="Variable",
                callback=lambda *_: self._on_expected_change(),
            )
            dpg.add_input_int(
                label="Animals",
                tag="expected_count",
                default_value=2,
                width=140,
                min_value=0,
                min_clamped=True,
                step=1,
                callback=lambda *_: self._on_expected_change(),
            )

    # -- Labels -----------------------------------------------------------
    def _build_labels_tab(self) -> None:
        with dpg.tab(label="Labels", tag=self._t("tab_labels")):
            widgets.gap(1)
            widgets.section_header(
                "Classes", "What kind of animal a box holds. One class is the usual case."
            )
            dpg.add_text("", tag="classes_display", color=theme.TEXT, wrap=self.WIDTH - 60)
            widgets.secondary_button(
                "Add class...",
                self._cb("add_class"),
                width=200,
                height=28,
                tag=self._t("add_class"),
            )
            dpg.add_text(
                "Each box keeps its class. Change a selected box in the Inspector; "
                "new boxes use the active class.",
                color=theme.TEXT_FAINT,
                wrap=self.WIDTH - 60,
            )

            widgets.gap(2)
            widgets.section_header(
                "Behaviors", "Keys 0-9 set the selected box's behavior in this order."
            )
            dpg.add_text("", tag="actions_display", color=theme.TEXT, wrap=self.WIDTH - 60)
            widgets.secondary_button(
                "Add behavior...",
                self._cb("add_action"),
                width=200,
                height=28,
                tag=self._t("add_action"),
            )

    # -- Tracking ---------------------------------------------------------
    def _build_tracking_tab(self, tracker_label: str, tracker_hint: str) -> None:
        with dpg.tab(label="Tracking", tag=self._t("tab_tracking")):
            widgets.gap(1)
            widgets.section_header(
                "Does this experiment follow individuals?",
                "Decided once per project. Everything else on this tab depends "
                "on it.",
            )
            dpg.add_checkbox(
                label="Follow each individual across frames (track IDs)",
                tag="use_track_ids_cb",
                default_value=True,
                callback=lambda s, a: self._on_use_track_ids(a),
            )
            dpg.add_text(
                "On: every box carries a track ID, the Inspector offers one, "
                "and the checks look for ID switches, broken tracks and "
                "implausible per-track jumps.\n"
                "Off: boxes are per-frame detections with no identity. Nothing "
                "asks you for an ID, boxes are coloured by class, and only the "
                "single-frame checks run. Track IDs already saved are left "
                "alone.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )

            widgets.gap(2)
            widgets.section_header("Continuing an animal")
            dpg.add_combo(
                ["Automatic", "Fast tracking", "Accurate propagation", "Copy to next frame"],
                label="Continue with",
                tag="settings_continue_method",
                width=220,
                default_value="Automatic",
                callback=lambda s, a: self._call("continue_method", a),
            )
            dpg.add_checkbox(
                label="Follow boxes to the next frame",
                tag="auto_track_cb",
                callback=lambda s, a: self._call("auto_track", a),
            )
            dpg.add_text(
                "When on, stepping to the next frame moves every initialised "
                "tracker with the animal. Propagated boxes are drawn dashed: "
                "they are candidates to correct, not annotations.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )

            widgets.gap(1)
            if tracker_label:
                dpg.add_text(
                    f"Backend: {tracker_label}", color=theme.TEXT_DIM, tag=self._t("tracker_label")
                )
            else:
                dpg.add_text(
                    tracker_hint or "No box tracker in this OpenCV build.",
                    color=theme.DANGER,
                    wrap=self.WIDTH - 60,
                    tag=self._t("tracker_label"),
                )

            widgets.gap(2)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Reset all trackers",
                    self._cb("reset_trackers"),
                    width=200,
                    height=28,
                    tag=self._t("reset_all"),
                )
                widgets.secondary_button(
                    "Reset selected",
                    self._cb("reset_selected_tracker"),
                    width=180,
                    height=28,
                    tag=self._t("reset_sel"),
                )

    def _on_use_track_ids(self, value: Any) -> None:
        """Gate the rest of the tab, then tell the annotator."""
        self._gate_tracking(bool(value))
        self._call("use_track_ids", bool(value))

    def set_use_track_ids(self, value: bool) -> None:
        """Mirror the annotator's answer without firing the callback back.

        This dialog is the only writer of ``use_track_ids_cb`` and of the
        controls it gates, so the annotator asks rather than reaching in — two
        writers on one widget is how a checkbox and the thing it controls end
        up disagreeing.
        """
        self._set("use_track_ids_cb", bool(value))
        self._gate_tracking(bool(value))

    def _gate_tracking(self, enabled: bool) -> None:
        """Lock, never hide, the controls that need track ids (P4)."""
        reason = (
            ""
            if enabled
            else "This project does not follow individuals across frames, so "
            "there is no track to continue. Switch track IDs on above to use it."
        )
        for tag in (
            "settings_continue_method",
            "auto_track_cb",
            self._t("reset_all"),
            self._t("reset_sel"),
        ):
            if not dpg.does_item_exist(tag):
                continue
            if reason:
                widgets.locked(tag, reason)
            else:
                widgets.unlock(tag)
        if not enabled:
            self._set("auto_track_cb", False)

    # -- Model ------------------------------------------------------------
    def _build_model_tab(self) -> None:
        """Where the checkpoint lives.

        It used to live inside Auto Annotate Project, which made "no model" a
        greyed-out checkbox two dialogs deep, and made Smart Annotate tell the
        user to go and open a different dialog first. A checkpoint is a
        property of the project, not an argument to one run of one dialog, so
        it belongs here beside the folder and the label vocabularies.
        """
        with dpg.tab(label="Model", tag=self._t("tab_model")):
            widgets.gap(1)
            widgets.section_header(
                "Detector checkpoint",
                "The only thing in the annotator that needs a trained model.",
            )
            dpg.add_text(
                "Step 2 writes this file to runs/train/stage4/weights/best.pth. "
                "Until it exists, labelling runs without it: press C to click an "
                "animal, and the frame sampling below Pick frames to label needs "
                "no model at all.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )
            widgets.gap(1)
            dpg.add_input_text(
                label="Checkpoint",
                tag=self._t("weights"),
                width=self.WIDTH - 200,
                callback=lambda *_: self._call("model_changed"),
            )
            widgets.secondary_button(
                "Browse...",
                self._cb("pick_weights"),
                width=200,
                height=28,
                tag=self._t("pick_weights"),
            )
            dpg.add_input_text(
                label="Config (optional)",
                tag=self._t("config"),
                width=self.WIDTH - 200,
                callback=lambda *_: self._call("model_changed"),
            )
            widgets.gap(2)
            # Always on screen, whatever it says. "Why is that checkbox grey?"
            # is a question the user should never have to go looking for the
            # answer to, so the answer is stated where the setting is.
            dpg.add_text("Status", color=theme.TEXT)
            dpg.add_text(
                "", tag=self._t("model_status"), color=theme.TEXT_DIM, wrap=self.WIDTH - 60
            )

    def model_paths(self) -> Tuple[str, str]:
        """``(checkpoint, config)`` as the fields currently read."""
        return (
            str(self._get(self._t("weights"), "") or ""),
            str(self._get(self._t("config"), "") or ""),
        )

    def set_model_status(self, message: str, ok: bool = False) -> None:
        """Show what :func:`auto_label.check_backend` said, verbatim."""
        self._set(self._t("model_status"), message)
        if dpg.does_item_exist(self._t("model_status")):
            dpg.configure_item(
                self._t("model_status"), color=theme.ACCEPTED if ok else theme.TEXT_DIM
            )

    # -- Assistance -------------------------------------------------------
    def _build_assistance_tab(self, tips: Mapping[str, str]) -> None:
        with dpg.tab(label="Assistance", tag=self._t("tab_assist")):
            widgets.gap(1)
            with dpg.collapsing_header(label="Advanced assistance", default_open=False):
                widgets.section_header(
                    "Click animal", "Which fitter a click uses to find the animal under it."
                )
                for name in ("auto", *click_segment.BACKENDS):
                    tag = f"click_backend_{name}"
                    dpg.add_button(
                        label="  "
                        + (
                            "Automatic (best available)"
                            if name == "auto"
                            else click_segment.BACKEND_LABELS[name]
                        ),
                        tag=tag,
                        width=-1,
                        height=28,
                        callback=lambda s, a, u: self._call("click_backend", u),
                        user_data=name,
                    )
                    theme.bind(tag, theme.ghost_theme())
                    if tips.get(name):
                        widgets.attach_tooltip(tag, tips[name], wrap=360)

            widgets.gap(2)
            widgets.section_header("Drawing")
            dpg.add_checkbox(
                label="Keep drawing after each box",
                tag="sticky_draw_cb",
                callback=lambda s, a: self._call("sticky_draw", a),
            )
            dpg.add_text(
                "Off means Draw Box, Draw Rotated Box and Click animal each "
                "return to Select after one box, which is what LabelImg does "
                "and what leaves the new box ready to adjust. On is for laying "
                "down many boxes in a row.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )

    # -- Checks -----------------------------------------------------------
    def _build_checks_tab(self) -> None:
        with dpg.tab(label="Checks", tag=self._t("tab_checks")):
            widgets.gap(1)
            dpg.add_text(
                "What the checks flag for review. The defaults are tuned for "
                "Drosophila in a small arena at 25-30 fps and every one of "
                "them is a judgement, not a measurement - a different "
                "preparation should change them.",
                color=theme.TEXT_DIM,
                wrap=self.WIDTH - 60,
            )
            widgets.gap(1)

            for name, label, help_text, low, high in THRESHOLD_FIELDS:
                tag = self._t(f"q_{name}")
                dpg.add_slider_float(
                    label=label,
                    tag=tag,
                    width=200,
                    min_value=low,
                    max_value=high,
                    format="%.2f",
                    callback=lambda *_: self._on_threshold_change(),
                )
                widgets.attach_tooltip(tag, help_text, wrap=360)

            widgets.gap(2)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Restore defaults",
                    lambda *_: self._on_reset_thresholds(),
                    width=200,
                    height=28,
                    tag=self._t("q_reset"),
                )
                widgets.secondary_button(
                    "Re-check the video",
                    self._cb("recheck"),
                    width=200,
                    height=28,
                    tag=self._t("q_recheck"),
                )
            dpg.add_text("", tag=self._t("q_note"), color=theme.TEXT_DIM, wrap=self.WIDTH - 60)

    # ── reading and writing state ────────────────────────────────────────
    def load(
        self,
        *,
        config: quality.QualityConfig,
        project_dir: Any = None,
        auto_track: bool = False,
        sticky_draw: bool = False,
        use_track_ids: bool = True,
        weights: str = "",
        config_path: str = "",
    ) -> None:
        """Fill every control from the annotator's current state."""
        if not self._built:
            return
        self._set("project_dir_text", str(project_dir) if project_dir else "(not set)")
        self._set(self._t("weights"), str(weights or ""))
        self._set(self._t("config"), str(config_path or ""))
        self._set("expected_mode", "Fixed" if config.expected_animals is not None else "Variable")
        self._set("expected_count", int(config.expected_animals or 2))
        self._set("auto_track_cb", bool(auto_track) and bool(use_track_ids))
        self._set("sticky_draw_cb", bool(sticky_draw))
        self._set("use_track_ids_cb", bool(use_track_ids))
        self._gate_tracking(bool(use_track_ids))
        for name, *_rest in THRESHOLD_FIELDS:
            self._set(self._t(f"q_{name}"), float(getattr(config, name)))

    def quality_config(self, base: quality.QualityConfig) -> quality.QualityConfig:
        """``base`` with every threshold replaced by what the sliders show."""
        overrides: Dict[str, Any] = {}
        for name, *_rest in THRESHOLD_FIELDS:
            value = self._get(self._t(f"q_{name}"))
            if value is not None:
                overrides[name] = float(value)
        fixed = self._get("expected_mode", "Variable") == "Fixed"
        overrides["expected_animals"] = int(self._get("expected_count", 2) or 0) if fixed else None
        return base.replace(**overrides)

    # ── callbacks ────────────────────────────────────────────────────────
    def _on_expected_change(self) -> None:
        fixed = self._get("expected_mode", "Variable") == "Fixed"
        if dpg.does_item_exist("expected_count"):
            dpg.configure_item("expected_count", enabled=fixed)
        self._call("quality_changed")

    def _on_threshold_change(self) -> None:
        self._set(
            self._t("q_note"),
            "Changed. Press 'Re-check the video' to rebuild the review " "queue with these.",
        )
        self._call("quality_changed")

    def _on_reset_thresholds(self) -> None:
        defaults = quality.QualityConfig()
        for name, *_rest in THRESHOLD_FIELDS:
            self._set(self._t(f"q_{name}"), float(getattr(defaults, name)))
        self._on_threshold_change()

    # ── lifecycle ────────────────────────────────────────────────────────
    def show(self, tab: str = "") -> None:
        if not self._built:
            return
        # Selecting the tab must never be able to stop the window opening: a
        # caller naming a tab that does not exist should still get Settings.
        target = self._t(f"tab_{tab.lower()}") if tab else ""
        if target and dpg.does_item_exist(target):
            try:
                dpg.set_value(self._t("tabs"), target)
            except SystemError:
                pass
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        dpg.configure_item(self.tag, show=True)
        self._on_expected_change()

    def hide(self) -> None:
        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, show=False)
        self._call("closed")

    @property
    def visible(self) -> bool:
        return dpg.does_item_exist(self.tag) and dpg.is_item_shown(self.tag)
