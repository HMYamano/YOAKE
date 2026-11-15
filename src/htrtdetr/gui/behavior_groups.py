"""
behavior_groups.py — the annotator's side of multi-head behaviors.

``docs/KEYFRAME_BEHAVIOR.md`` §15. By default a box gets one behavior. A
keyframe project may instead group its behaviors - exclusive inside a group,
simultaneous across groups - and then every box gets one answer per group::

    locomotion: still, walk, run
    social:     none, courtship, copulation

What changes in the annotator, and only in a keyframe project that chose groups:

=====================  ==================================================
Keys 0-9               Flat numbers through all groups. A key answers its
                       own group and leaves the others alone.
Set behavior buttons   One heading per group.
Inspector              One "Behavior" menu per group, with "Not set".
Boxes                  Coloured by the first group's answer; filled only
                       once every group is answered; labelled
                       ``walk + courtship`` (``?`` for an open group).
Settings > Labels      "Edit behaviors..." writes the groups as text;
                       answers follow their names through an edit.
Create Project         "One behavior per animal" or "Several groups".
=====================  ==================================================

The vocabulary itself - parsing, numbering, the per-box answer - is
:mod:`htrtdetr.annotation.behaviors`, which has no GUI and is unit-tested on
its own. This mixin is wiring. :class:`~htrtdetr.gui.keyframe_mode.KeyframeModeMixin`
inherits it, so ``annotator.py`` needs no import and no line of its budget.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover - GUI なしの環境
    dpg = None

from ..annotation import behaviors

__all__ = ["BehaviorGroupsMixin", "BEHAVIOR_SCHEME_CHOICES", "EDITOR_TAG"]

#: The Create Project choice and the editor's. First is the default.
BEHAVIOR_SCHEME_CHOICES: Tuple[str, ...] = (
    "One behavior per animal (recommended)",
    "Several behavior groups (multi-head)",
)

EDITOR_TAG = "behavior_editor"

#: Shown when someone switches to groups with the default list still there.
_GROUPS_EXAMPLE = "locomotion: still, walk, run\nsocial: none, courtship, copulation"

_SINGLE_HINT = "Comma-separated, in display order. Keys 0-9 follow this order."
_GROUPS_HINT = (
    "One group per line, written 'group: behavior, behavior, ...'. Each animal "
    "gets one behavior from every group - a fly can walk and court at once. "
    "Give a group 'none' if it can be off. Keys 0-9 count through all groups."
)

#: The tracks scheme's button colours, unchanged from before groups existed.
_TRACK_BUTTON_COLORS = (
    (85, 85, 85, 255),
    (21, 101, 192, 255),
    (46, 125, 50, 255),
    (230, 81, 0, 255),
    (183, 27, 28, 255),
)


def _is_groups(choice: Any) -> bool:
    return str(choice or "").startswith("Several")


class BehaviorGroupsMixin:
    """Mixed into ``AnnotationApp`` (through ``KeyframeModeMixin``)."""

    # ── the vocabulary ───────────────────────────────────────────────────
    @property
    def behavior_vocabulary(self) -> behaviors.BehaviorVocabulary:
        """The behaviors boxes are answered with: one list, or groups.

        Groups count only in a keyframe project; in the tracks scheme the same
        names are one flat list. Cached on its inputs, because the canvas asks
        once per box per redraw.
        """
        project = self._ensure_project()
        multi = bool(
            self.keyframe_mode
            and project.settings.get("behaviors") == behaviors.MULTI_HEAD
            and project.action_groups
        )
        groups = project.action_groups if multi else {}
        key = (
            multi,
            tuple(self.action_names),
            tuple((str(k), tuple(v)) for k, v in groups.items()),
        )
        cached = getattr(self, "_behavior_vocab_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        settings = {"behaviors": behaviors.MULTI_HEAD} if multi else {}
        vocab = behaviors.project_vocabulary(settings, self.action_names, groups)
        self._behavior_vocab_cache = (key, vocab)
        return vocab

    def set_behavior_vocabulary(
        self, vocab: behaviors.BehaviorVocabulary, *, remap: bool = True
    ) -> int:
        """Make ``vocab`` the project's behaviors. Returns boxes re-numbered.

        Every box in every video keeps its answers by name (see
        :func:`behaviors.remap`), so adding a behavior in the middle of a group
        does not turn later answers into their neighbours. Each box is
        re-numbered exactly once even if two views of a video share it.
        """
        project = self._ensure_project()
        old = self.behavior_vocabulary
        changed = 0
        if remap:
            boxes: Dict[int, Any] = {}
            touched = []
            for video in project.videos.values():
                found = [o for row in video.annotations.values() for o in row]
                for obj in found:
                    boxes[id(obj)] = obj
                if found:
                    touched.append(video)
            for row in self.annotations.values():
                for obj in row:
                    boxes[id(obj)] = obj
            changed = behaviors.remap(list(boxes.values()), old, vocab)
            if changed:
                for video in touched:
                    touch = getattr(video, "touch", None)
                    if callable(touch):
                        touch()
        self.action_names = list(vocab.names)
        project.action_names = list(vocab.names)
        if vocab.multi:
            project.action_groups = vocab.as_groups()
            project.settings["behaviors"] = behaviors.MULTI_HEAD
        else:
            if project.settings.get("behaviors") == behaviors.MULTI_HEAD:
                # Ours to clear: written when the groups were. A file's own
                # (hierarchical) action_groups are left alone otherwise.
                project.action_groups = {}
            project.settings["behaviors"] = behaviors.SINGLE
        self._behavior_vocab_cache = None
        self._mark_dirty()
        self._update_action_buttons()
        self._refresh_labels_display()
        self._refresh_sidebar()
        self._render_canvas()
        return changed

    # ── one box ──────────────────────────────────────────────────────────
    def _behavior_text(self, obj: Any, unset: str = "?") -> str:
        """What a box is doing, for its label and the object list."""
        vocab = self.behavior_vocabulary
        if vocab.multi:
            return behaviors.describe(obj, vocab, unset)
        a = getattr(obj, "action_id", -1)
        return str(self.action_names[a]) if 0 <= a < len(self.action_names) else unset

    def _behavior_would_change(self, obj: Any, action_id: int) -> bool:
        """Would answering ``action_id`` change this box? Asked before the undo push."""
        probe = {
            "action_id": getattr(obj, "action_id", -1),
            "action_ids": list(getattr(obj, "action_ids", None) or []),
        }
        return behaviors.set_behavior(probe, self.behavior_vocabulary, action_id)

    def _apply_behavior(self, obj: Any, action_id: int) -> None:
        behaviors.set_behavior(obj, self.behavior_vocabulary, action_id)

    def _behavior_status(self, obj: Any) -> str:
        """The status line after a behavior key: what the box now is, what is open."""
        vocab = self.behavior_vocabulary
        if not vocab.multi:
            return f"Action set to {self._behavior_text(obj, str(obj.action_id))}"
        text = f"Behavior: {behaviors.describe(obj, vocab)}"
        missing = behaviors.missing_groups(obj, vocab)
        if missing:
            text += " - still open: " + ", ".join(vocab.heads[h] for h in missing)
        return text

    def _behavior_box_style(self, obj: Any) -> Tuple[int, bool]:
        """``(colour index, filled)`` for a keyframe box.

        Colour comes from the first group's answer, the same colour the key's
        button and the timeline tick use. Filled means done: with groups,
        only once every group has its answer.
        """
        vocab = self.behavior_vocabulary
        labels = behaviors.labels_of(obj, vocab)
        if not vocab.multi:
            return labels[0], labels[0] >= 0
        return labels[0], all(a >= 0 for a in labels)

    # ── the Inspector ────────────────────────────────────────────────────
    def _inspector_behavior_fields(self, obj: Any) -> Dict[str, Any]:
        """Extra ``InspectorContext`` fields for a box: one menu per group."""
        vocab = self.behavior_vocabulary
        if not vocab.multi:
            return {}
        return {
            "behavior_groups": tuple((name, tuple(b)) for name, b in vocab.groups),
            "group_labels": behaviors.group_targets(obj, vocab),
        }

    def _insp_set_group_behavior(self, payload: Any) -> None:
        """The Inspector's per-group menu: ``(group index, behavior name)``."""
        if self._selected is None:
            return
        try:
            group, name = payload
            group = int(group)
        except (TypeError, ValueError):
            return
        vocab = self.behavior_vocabulary
        if not vocab.multi or not 0 <= group < len(vocab.groups):
            return
        options = vocab.behaviors_of(group)
        current = behaviors.labels_of(self._selected, vocab)[group]
        target = vocab.flat_id(group, options.index(name)) if name in options else -1
        if target == current:
            return
        self._push_undo()
        if target >= 0:
            behaviors.set_behavior(self._selected, vocab, target)
        else:
            behaviors.clear_group(self._selected, vocab, group)
        self._mark_dirty()
        self._status(self._behavior_status(self._selected))
        self._refresh_sidebar()
        self._render_canvas()

    # ── the Set behavior buttons ─────────────────────────────────────────
    def _update_action_buttons(self) -> None:
        """One button per behavior; with groups, a heading per group.

        Keyframe projects colour each button like the boxes and timeline ticks
        it produces. The tracks scheme keeps its original five colours.
        """
        if dpg is None or not dpg.does_item_exist("action_buttons_group"):
            return
        from . import theme

        dpg.delete_item("action_buttons_group", children_only=True)
        vocab = self.behavior_vocabulary
        keyframe = bool(self.keyframe_mode)
        for h, (group, names) in enumerate(vocab.groups):
            if vocab.multi:
                dpg.add_text(group, color=theme.TEXT_DIM, parent="action_buttons_group")
            for k, name in enumerate(names):
                i = vocab.offsets[h] + k
                col = (
                    theme.class_color(i) if keyframe
                    else _TRACK_BUTTON_COLORS[i % len(_TRACK_BUTTON_COLORS)]
                )
                with dpg.theme() as btn_theme:
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, col)
                        dpg.add_theme_color(
                            dpg.mvThemeCol_ButtonHovered,
                            tuple(min(c + 30, 255) for c in col[:3]) + (255,),
                        )
                btn = dpg.add_button(
                    label=f"[{i}] {name.capitalize()}",
                    width=-1,
                    height=28,
                    parent="action_buttons_group",
                    callback=lambda s, a, u: self.set_action(u),
                    user_data=i,
                )
                dpg.bind_item_theme(btn, btn_theme)
        if dpg.does_item_exist("sel_action"):
            dpg.configure_item("sel_action", items=["(none)"] + list(self.action_names))

    def _behavior_display_text(self) -> str:
        """Settings > Labels: the numbered behaviors, grouped when they are."""
        vocab = self.behavior_vocabulary
        if not vocab.multi:
            return "\n".join(f"  [{i}] {n}" for i, n in enumerate(self.action_names))
        lines: List[str] = []
        for h, (group, names) in enumerate(vocab.groups):
            lines.append(f"  {group}")
            lines.extend(f"    [{vocab.offsets[h] + k}] {n}" for k, n in enumerate(names))
        return "\n".join(lines)

    def add_behavior_dialog(self) -> None:
        """Settings' "Add behavior...": with groups it must ask which group."""
        if self.behavior_vocabulary.multi:
            self.open_behavior_editor()
        else:
            self._add_action_label_dialog()

    # ── the setup screen ─────────────────────────────────────────────────
    def _build_setup_behaviors(self, default_text: str, wrap: int = 560) -> None:
        """The Behaviors fold on Create Project: the scheme, then the names."""
        from . import theme

        dpg.add_radio_button(
            items=list(BEHAVIOR_SCHEME_CHOICES),
            tag="setup_behavior_scheme",
            default_value=BEHAVIOR_SCHEME_CHOICES[0],
            callback=lambda s, a: self._on_setup_behavior_scheme(a),
        )
        dpg.add_text(_SINGLE_HINT, tag="setup_actions_hint", color=theme.TEXT_DIM, wrap=wrap)
        dpg.add_input_text(
            tag="setup_actions", default_value=default_text, width=-1,
            multiline=True, height=64,
        )
        dpg.add_text("", tag="setup_actions_error", color=theme.ERROR, wrap=wrap, show=False)
        self._setup_actions_default = default_text

    def _on_setup_behavior_scheme(self, choice: Any) -> None:
        """Rewrite the field into the other form, so it is a valid starting point."""
        if dpg is None or not dpg.does_item_exist("setup_actions"):
            return
        text = str(dpg.get_value("setup_actions") or "")
        dpg.set_value("setup_actions_hint", _GROUPS_HINT if _is_groups(choice) else _SINGLE_HINT)
        dpg.configure_item("setup_actions_error", show=False)
        dpg.set_value("setup_actions", self._convert_behavior_text(text, _is_groups(choice)))

    def _convert_behavior_text(self, text: str, to_groups: bool) -> str:
        default = getattr(self, "_setup_actions_default", "")
        if to_groups:
            if behaviors._GROUP_SEP in text:
                return text
            if text.strip() == default.strip() or not text.strip():
                return _GROUPS_EXAMPLE
            return f"behavior{behaviors._GROUP_SEP} {text.strip()}"
        if behaviors._GROUP_SEP not in text:
            return text
        try:
            return ", ".join(behaviors.parse_vocabulary(text, True).names)
        except ValueError:
            return text

    def _read_setup_behaviors(self, defaults: Any) -> Optional[behaviors.BehaviorVocabulary]:
        """The vocabulary Create Project asked for, or None after saying what is wrong."""
        text = str(dpg.get_value("setup_actions") or "") if dpg is not None \
            and dpg.does_item_exist("setup_actions") else ""
        groups = dpg is not None and dpg.does_item_exist("setup_behavior_scheme") \
            and _is_groups(dpg.get_value("setup_behavior_scheme"))
        keyframe = dpg is not None and dpg.does_item_exist("setup_label_mode") \
            and str(dpg.get_value("setup_label_mode") or "").startswith("Keyframe")
        problem = ""
        vocab = None
        try:
            vocab = behaviors.parse_vocabulary(text, groups)
        except ValueError as exc:
            problem = str(exc)
        if vocab is not None and groups and not keyframe:
            problem = ("Behavior groups need the Keyframes scheme: the tracks scheme "
                       "edits behaviors as intervals, one at a time.")
        if problem:
            if dpg is not None and dpg.does_item_exist("setup_actions_error"):
                dpg.set_value("setup_actions_error", problem)
                dpg.configure_item("setup_actions_error", show=True)
            if dpg is not None and dpg.does_item_exist("setup_actions_fold"):
                dpg.set_value("setup_actions_fold", True)
            return None
        if vocab is None or not len(vocab):
            vocab = behaviors.BehaviorVocabulary.single(list(defaults))
        return vocab

    def _apply_setup_behaviors(self, vocab: behaviors.BehaviorVocabulary) -> None:
        """Write Create Project's vocabulary into the new project."""
        project = self._ensure_project()
        self.action_names = list(vocab.names)
        project.action_names = list(vocab.names)
        project.action_groups = vocab.as_groups() if vocab.multi else {}
        project.settings["behaviors"] = behaviors.MULTI_HEAD if vocab.multi else behaviors.SINGLE
        self._behavior_vocab_cache = None

    # ── Settings > Labels: the editor ────────────────────────────────────
    def open_behavior_editor(self) -> None:
        """Edit the behaviors as text: a comma list, or one group per line."""
        if dpg is None:
            return
        from . import theme, widgets

        if dpg.does_item_exist(EDITOR_TAG):
            dpg.delete_item(EDITOR_TAG)
        vocab = self.behavior_vocabulary
        with dpg.window(
            label="Edit behaviors", tag=EDITOR_TAG, modal=True, no_collapse=True,
            width=520, height=360, on_close=lambda: dpg.delete_item(EDITOR_TAG),
        ):
            dpg.add_radio_button(
                items=list(BEHAVIOR_SCHEME_CHOICES),
                tag=EDITOR_TAG + "_scheme",
                default_value=BEHAVIOR_SCHEME_CHOICES[1 if vocab.multi else 0],
                callback=lambda s, a: self._on_editor_scheme(a),
                enabled=bool(self.keyframe_mode),
            )
            if not self.keyframe_mode:
                dpg.add_text("Behavior groups need the Keyframes scheme (Settings > Tracking).",
                             color=theme.TEXT_FAINT, wrap=490)
            dpg.add_text(_GROUPS_HINT if vocab.multi else _SINGLE_HINT,
                         tag=EDITOR_TAG + "_hint", color=theme.TEXT_DIM, wrap=490)
            dpg.add_input_text(tag=EDITOR_TAG + "_text", default_value=vocab.to_text(),
                               multiline=True, width=-1, height=140)
            dpg.add_text("Answers already given follow their names: renaming a behavior "
                         "clears it, adding or reordering keeps it.",
                         color=theme.TEXT_FAINT, wrap=490)
            dpg.add_text("", tag=EDITOR_TAG + "_error", color=theme.ERROR, wrap=490, show=False)
            with dpg.group(horizontal=True):
                widgets.primary_button("Apply", self._apply_behavior_editor, width=120,
                                       height=28)
                widgets.secondary_button("Cancel", lambda: dpg.delete_item(EDITOR_TAG),
                                         width=100, height=28)
        widgets.place_dialog(EDITOR_TAG, 520, 360)

    def _on_editor_scheme(self, choice: Any) -> None:
        text = str(dpg.get_value(EDITOR_TAG + "_text") or "")
        dpg.set_value(EDITOR_TAG + "_hint", _GROUPS_HINT if _is_groups(choice) else _SINGLE_HINT)
        dpg.set_value(EDITOR_TAG + "_text", self._convert_behavior_text(text, _is_groups(choice)))

    def _apply_behavior_editor(self) -> None:
        groups = _is_groups(dpg.get_value(EDITOR_TAG + "_scheme")) and bool(self.keyframe_mode)
        try:
            vocab = behaviors.parse_vocabulary(dpg.get_value(EDITOR_TAG + "_text"), groups)
            if not len(vocab):
                raise ValueError("Give at least one behavior.")
        except ValueError as exc:
            dpg.set_value(EDITOR_TAG + "_error", str(exc))
            dpg.configure_item(EDITOR_TAG + "_error", show=True)
            return
        changed = self.set_behavior_vocabulary(vocab)
        dpg.delete_item(EDITOR_TAG)
        groups_text = (f"{len(vocab.groups)} groups" if vocab.multi
                       else f"{len(vocab)} behaviors")
        self._status(f"Behaviors updated: {groups_text}"
                     + (f"; {changed} box(es) re-numbered to keep their answers." if changed
                        else "."))
