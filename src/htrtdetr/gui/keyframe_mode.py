"""
keyframe_mode.py — the annotator's keyframe labelling scheme.

``docs/KEYFRAME_BEHAVIOR.md``: on chosen frames, a box and a behavior for every
animal, and nothing else. No track ids (identity comes from tracking at
analysis time), no intervals (bouts come from decoding the model's per-frame
probabilities along tracks). This mixin is everything the annotator does
differently in that scheme, kept out of ``annotator.py`` on purpose - that file
has a line budget, and a scheme is a responsibility.

=====================  ==================================================
Project setting        ``settings["label_mode"]``: ``keyframe`` | ``tracks``.
                       Absent reads ``tracks`` (older projects unchanged);
                       new projects are created as ``keyframe``.
Boxes                  Coloured by behavior; grey and unfilled until one is
                       given. Labelled with the behavior, not a track id.
Timeline               The behavior lane shows one tick per labelled animal.
Inspector              "3 animals - 1 still needs a behavior."
Setup screen           The scheme is the first question after the labels.
Behavior groups        Optional multi-head answers (one per group), in
                       :mod:`htrtdetr.gui.behavior_groups`, inherited here.
=====================  ==================================================

The context loop (``P``) lives beside it in :mod:`htrtdetr.gui.context_loop`.
"""

from __future__ import annotations

from typing import Any, List, Tuple

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

from ..annotation import intervals as intervals_mod
from ..annotation import keyframes
from .behavior_groups import BehaviorGroupsMixin

__all__ = ["KeyframeModeMixin", "LABEL_MODE_CHOICES"]

#: The setup screen's choice. First is the default for a new project; each
#: starts with the scheme's name, which is what the handlers read.
LABEL_MODE_CHOICES: Tuple[str, ...] = (
    "Keyframes - a box and a behavior for each animal (recommended)",
    "Tracks - track IDs on every box, behaviors as intervals",
)


class KeyframeModeMixin(BehaviorGroupsMixin):
    """Mixed into ``AnnotationApp``."""

    # ── the setting ──────────────────────────────────────────────────────
    @property
    def label_mode(self) -> str:
        """``keyframe`` or ``tracks``. Absent from a project file: ``tracks``."""
        value = self._ensure_project().settings.get("label_mode", keyframes.TRACKS)
        return keyframes.KEYFRAME if value == keyframes.KEYFRAME else keyframes.TRACKS

    @property
    def keyframe_mode(self) -> bool:
        return self.label_mode == keyframes.KEYFRAME

    def set_label_mode(self, value: Any) -> None:
        """Switch the project's labelling scheme. Never rewrites saved labels.

        Track ids and intervals already on disk stay where they are: a setting
        that destroyed work switching back could not restore would be a trap.
        """
        mode = keyframes.KEYFRAME if str(value or "").lower().startswith("keyframe") \
            else keyframes.TRACKS
        if self.label_mode == mode:
            return
        self._ensure_project().settings["label_mode"] = mode
        if mode == keyframes.KEYFRAME:
            self._trackers.clear()
            self._auto_track = False
        self.stop_context_loop()
        self._mark_dirty()
        self._refresh_track_ui()
        # Groups only count in keyframe mode, and so do the behavior colours.
        self._update_action_buttons()
        self._refresh_labels_display()
        if hasattr(getattr(self, "settings", None), "set_label_mode"):
            self.settings.set_label_mode(mode)
        self.rebuild_copilot()
        self._render_canvas()
        self._refresh_sidebar()
        self._status(
            "Keyframe labelling: give every animal on a frame a box and a behavior. "
            "No track IDs - tracking happens at analysis time. P plays the motion "
            "around the frame."
            if mode == keyframes.KEYFRAME
            else "Track labelling: boxes carry track IDs and behaviors are edited as intervals."
        )

    # ── drawing ──────────────────────────────────────────────────────────
    def _keyframe_box_colors(self, obj: Any, col: Tuple, fill: Tuple) -> Tuple[Tuple, Tuple]:
        """Box colours. Keyframe mode answers "what is it doing" with colour.

        The behavior's colour is the one the timeline lane uses, so a tick and
        a box that mean the same thing look the same. Unanswered is grey and
        unfilled; while the context loop plays, every box is unfilled so the
        motion under it shows.
        """
        if self.keyframe_mode:
            from . import theme

            # With behavior groups: the first group's colour, filled once all
            # groups are answered (BehaviorGroupsMixin._behavior_box_style).
            index, filled = self._behavior_box_style(obj)
            col = theme.class_color(index)
            fill = theme.class_color(index, 60 if filled else 0)
        if self.context_loop_active:
            fill = (0, 0, 0, 0)
        return col, fill

    def _keyframe_box_label(self, obj: Any, action_s: str) -> str:
        """The behavior, prefixed by the class only when there is more than one."""
        if len(self.class_names) > 1 and 0 <= obj.class_id < len(self.class_names):
            return f"{self.class_names[obj.class_id]}|{action_s}"
        return action_s

    def _keyframe_behavior_marks(self) -> List[Any]:
        """One confirmed one-frame "interval" per labelled behavior, for the lane.

        Drawn as ticks in the behavior's colour they show where each behavior
        has examples and where the video has none - which is where the next
        keyframes should go.
        """
        return [
            intervals_mod.BehaviorInterval(
                track_id=-1, action_id=action, start=frame, end=frame, source="human"
            )
            for frame, action in keyframes.keyframe_marks(
                self.annotations, self.behavior_vocabulary
            )
        ]

    def _behavior_lane(self) -> List[Any]:
        """What the timeline's behavior lane draws: ticks here, intervals in tracks mode."""
        return self._keyframe_behavior_marks() if self.keyframe_mode else self._intervals.intervals

    def _rebuilt_intervals(self, candidates: Any) -> "intervals_mod.IntervalTrack":
        """The copilot's interval view, rebuilt from the pending candidates.

        Keyframe labels are never intervals - a behavior is one answer per box
        per keyframe, and bouts come from the analysis - so there it is empty.
        In tracks mode human intervals are kept and machine ones fill the rest.
        """
        if self.keyframe_mode:
            return intervals_mod.IntervalTrack()
        human = [i for i in self._intervals.intervals if i.source == "human"]
        derived = intervals_mod.merge_to_intervals(candidates)
        return intervals_mod.IntervalTrack(
            human + [
                i for i in derived
                if not any(h.track_id == i.track_id and h.overlaps(i) for h in human)
            ]
        )

    def _keyframe_note(self) -> str:
        """The Inspector's progress line for this keyframe, or ``""``."""
        if not self.keyframe_mode or self.current_frame_idx not in self.annotations:
            return ""
        return keyframes.keyframe_sentence(
            self.annotations.get(self.current_frame_idx, []), self.behavior_vocabulary
        )

    # ── the setup screen ─────────────────────────────────────────────────
    def _build_setup_label_mode(self, wrap: int) -> None:
        """The scheme question on the first screen. Decided before the first box."""
        from . import theme

        dpg.add_text("What do you label?")
        dpg.add_radio_button(
            items=list(LABEL_MODE_CHOICES),
            tag="setup_label_mode",
            default_value=LABEL_MODE_CHOICES[0],
            callback=lambda s, a: self._on_setup_label_mode(a),
        )
        dpg.add_text(
            "Keyframes: on the frames you pick, a box and a behavior for every "
            "animal - nothing else. Who is who comes from tracking at analysis "
            "time, and behaviors become bouts there too. P plays the motion "
            "around a frame while you decide.",
            color=theme.TEXT_DIM,
            wrap=wrap,
        )
        dpg.add_text(
            "Tracks: every box carries a track ID and behaviors are edited as "
            "time intervals (the original workflow).",
            color=theme.TEXT_DIM,
            wrap=wrap,
        )
        # Only meaningful in the tracks scheme, so only shown there.
        dpg.add_checkbox(
            label="Follow each individual across frames (track IDs)",
            tag="setup_use_tracks",
            default_value=True,
            show=False,
        )

    def _on_setup_label_mode(self, value: Any) -> None:
        if dpg is not None and dpg.does_item_exist("setup_use_tracks"):
            dpg.configure_item(
                "setup_use_tracks", show=not str(value or "").startswith("Keyframe")
            )

    def _apply_setup_label_mode(self, settings: dict) -> None:
        """Write the setup screen's answer into a new project's settings."""
        keyframe = (
            dpg is not None
            and dpg.does_item_exist("setup_label_mode")
            and str(dpg.get_value("setup_label_mode") or "").startswith("Keyframe")
        )
        settings["label_mode"] = keyframes.KEYFRAME if keyframe else keyframes.TRACKS
        settings["use_track_ids"] = False if keyframe else bool(
            dpg.get_value("setup_use_tracks") if dpg is not None
            and dpg.does_item_exist("setup_use_tracks") else True
        )
