"""
inspector.py — the right pane shows what you selected, and nothing else.

The annotator's old sidebar was eleven panels, all present at all times: Quick
Start, Active Track, Objects, Selected Object, Actions, Tools, Auto-Track,
Labels, Frame Info. Every one of them competed for attention whether or not it
had anything to do with what the user was touching, and the two that mattered
for the current selection were somewhere in the middle of the stack.

An inspector shows the properties of **the current selection**. Nothing
selected is itself a context — the frame's own properties — not an excuse to
show everything (§4).

=====================  ==================================================
Nothing selected       Frame, animals in it, annotation status, next action
A box                  Class, track, behavior, rotation, occlusion, delete
A candidate            Accept / Reject / Edit, confidence, why it is queued
A tracklet             Range, confidence summary, issues, Accept tracklet
A behavior interval    Label, start, end, confidence, confirm and adjust
A review item          The reason, what to check, and the one-click answers
A sampled clip         Range, why it was sampled, and Follow animals across it
=====================  ==================================================

Built once, shown by section
---------------------------
Every section is created at build time and toggled with ``show``, matching what
the hub already does with its pages (``docs/GUI_DESIGN.md`` §0.2.1). Rebuilding
the pane on every selection would churn twenty widgets per click, drop keyboard
focus mid-edit, and re-register callbacks against fresh tags for no gain.

Callbacks
---------
The annotator passes a ``{name: callable}`` map. A missing name is a no-op
rather than an error, so this module can be built against a caller that has not
implemented every action yet — the same *degrades gracefully on a missing tag*
contract :mod:`htrtdetr.gui.widgets` holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import dearpygui.dearpygui as dpg

from ..annotation import quality
from ..annotation import status as ann_status
from . import shortcuts, theme, widgets

__all__ = [
    "NOTHING",
    "BOX",
    "CANDIDATE",
    "TRACKLET",
    "INTERVAL",
    "ISSUE",
    "CLIP",
    "CONTEXTS",
    "InspectorContext",
    "Inspector",
]

NOTHING = "nothing"
BOX = "box"
CANDIDATE = "candidate"
TRACKLET = "tracklet"
INTERVAL = "interval"
ISSUE = "issue"
#: A span Smart Annotate proposed and nobody has labelled yet. Distinct from
#: INTERVAL, which is a span a human has already answered: a clip is a
#: question, an interval is an answer, and the actions on them share nothing.
CLIP = "clip"

CONTEXTS: Tuple[str, ...] = (NOTHING, BOX, CANDIDATE, TRACKLET, INTERVAL, ISSUE, CLIP)

_TITLES: Dict[str, str] = {
    NOTHING: "Frame",
    BOX: "Selected box",
    CANDIDATE: "Candidate",
    TRACKLET: "Tracklet",
    INTERVAL: "Behavior",
    ISSUE: "Review item",
    CLIP: "Sampled clip",
}


#: The one-line explanation under each title. Empty where the section speaks
#: for itself.
_SUBTITLES: Dict[str, str] = {
    NOTHING: "Nothing is selected.",
    BOX: "",
    CANDIDATE: "Not saved until you accept it.",
    TRACKLET: "One decision covers every frame below.",
    INTERVAL: "Edit the edges, not every frame.",
    ISSUE: "",
    CLIP: "Proposed for annotation. Nothing in it is labelled yet.",
}

#: Which filler runs for each context. Module level, and by name rather than by
#: bound method, so a test can assert that every context in :data:`CONTEXTS`
#: has a title, a subtitle and a filler. Adding a seventh context showed why
#: that matters: these three tables are edited together and nothing but a
#: reader was checking that they agreed, so missing one would have shown an
#: empty pane with the right heading.
_FILLERS: Dict[str, str] = {
    NOTHING: "_fill_nothing",
    BOX: "_fill_box",
    CANDIDATE: "_fill_candidate",
    TRACKLET: "_fill_tracklet",
    INTERVAL: "_fill_interval",
    ISSUE: "_fill_issue",
    CLIP: "_fill_clip",
}


@dataclass(frozen=True)
class InspectorContext:
    """What is selected, as one value the annotator hands over.

    Only the fields the ``kind`` needs are read. Anything else is ignored, so
    the annotator may build one of these with whatever it has to hand rather
    than assembling a different shape per context.
    """

    kind: str = NOTHING

    # -- frame context --
    frame: int = 0
    num_frames: int = 0
    fps: float = 25.0
    animals: int = 0
    expected_animals: Optional[int] = None
    verified_frames: int = 0
    pending_here: int = 0
    #: Is this frame one of the ones the current round asked for?
    chosen_frame: bool = False
    #: Has a human already decided about this frame - boxes drawn, or an
    #: explicit "nothing here"? Separate from ``animals`` because the two
    #: differ exactly in the case this flag exists for: an empty frame that
    #: has been signed off looks identical to one nobody has opened.
    checked: bool = False

    # -- box / candidate --
    track_id: int = -1
    #: Does this project follow individuals across frames? False hides every
    #: track control: a field the user must not fill in is worse than no field,
    #: because it invites a value that nothing will ever read.
    use_track_ids: bool = True
    active_class_id: int = 0
    class_id: int = 0
    action_id: int = -1
    score: Optional[float] = None
    occluded: bool = False
    is_crowd: bool = False
    has_obb: bool = False
    angle_deg: float = 0.0
    source: str = "human"
    accepted: bool = True

    # -- tracklet --
    tracklet: Any = None

    # -- behavior interval --
    interval: Any = None

    # -- review item --
    item: Any = None
    issue_progress: str = ""

    # -- sampled clip --
    clip: Any = None
    #: The frame inside the clip whose boxes would seed a propagation, and how
    #: many there are. Both come from the annotator rather than being worked
    #: out here: the rule for choosing the frame is the propagation's, and two
    #: implementations of it would drift.
    clip_seed: Optional[int] = None
    clip_animals: int = 0
    #: Empty when a clip propagation could run right now; otherwise the one
    #: sentence saying what would stop it.
    clip_blocked: str = ""

    # -- vocabulary --
    class_names: Sequence[str] = ()
    action_names: Sequence[str] = ()

    #: Keyframe labelling only: how far this frame is from done, as one
    #: sentence ("3 animals - 1 still needs a behavior."). Empty otherwise.
    keyframe_note: str = ""

    @property
    def state(self) -> str:
        """VERIFIED / PREDICTED / PENDING for the selected box."""
        issues = getattr(self.item, "kind", None)
        return ann_status.status_of(
            accepted=self.accepted, score=self.score, issues=(issues,) if issues else ()
        )

    def action_label(self) -> str:
        if 0 <= self.action_id < len(self.action_names):
            return str(self.action_names[self.action_id])
        return "Not set"

    def class_label(self) -> str:
        if 0 <= self.class_id < len(self.class_names):
            return str(self.class_names[self.class_id])
        return f"class {self.class_id}"


class Inspector:
    """The right pane. One child window, seven sections, one visible at a time."""

    WIDTH = 288

    def __init__(
        self, tag: str = "inspector", callbacks: Optional[Mapping[str, Callable]] = None
    ) -> None:
        self.tag = tag
        self.callbacks: Dict[str, Callable] = dict(callbacks or {})
        self.context = InspectorContext()
        self._built = False

    # ── plumbing ─────────────────────────────────────────────────────────
    def _call(self, name: str, *args: Any) -> None:
        """Invoke a caller-supplied action, or do nothing if there is none."""
        fn = self.callbacks.get(name)
        if fn is not None:
            fn(*args)

    def _cb(self, name: str, *args: Any) -> Callable:
        """A dearpygui callback that invokes ``name`` with fixed arguments."""
        return lambda *_dpg: self._call(name, *args)

    def _cb_passthrough(self, name: str) -> Callable:
        """A callback that hands the widget's own value to ``name``.

        Separate from :meth:`_cb` because dearpygui passes
        ``(sender, app_data, user_data)`` and swallowing ``app_data`` is how an
        input field ends up calling its handler with nothing.
        """

        def callback(_sender=None, app_data=None, _user=None) -> None:
            self._call(name, app_data)

        return callback

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
    def _show(tag: str, visible: bool) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=bool(visible))

    @staticmethod
    def _label(tag: str, text: str) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, label=text)

    def _key(self, action: str) -> str:
        """The key for an action, from the one keymap. Never hand-written —
        a hard-coded label is how the docstring and the handler drift apart."""
        return shortcuts.label_for(action)

    # ── construction ─────────────────────────────────────────────────────
    def build(self, parent: str, height: int = -1, width: int = 0) -> None:
        """Create the pane and every section. Safe to call twice.

        ``width`` overrides :attr:`WIDTH` so the host decides the column width
        once and both panes in it agree; ``-1`` fills the parent.
        """
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)
        if width:
            self.WIDTH = int(width) if width > 0 else self.WIDTH

        with dpg.child_window(
            tag=self.tag,
            parent=parent,
            width=(width if width else self.WIDTH),
            height=height,
            border=False,
        ):
            dpg.add_text("Frame", tag=self._t("title"), color=theme.TEXT)
            theme.bind_h2(dpg.last_item())
            dpg.add_text("", tag=self._t("subtitle"), color=theme.TEXT_DIM, wrap=self.WIDTH - 20)
            dpg.add_separator()
            widgets.gap(1)

            self._build_nothing()
            self._build_box()
            self._build_candidate()
            self._build_tracklet()
            self._build_interval()
            self._build_issue()
            self._build_clip()

        self._built = True
        self.show(InspectorContext())

    # -- nothing selected -------------------------------------------------
    def _build_nothing(self) -> None:
        """The frame's own properties, and what can be done to this frame.

        Four legacy sidebar panels land here — *Frame Info*, *Active Track*,
        *Objects in Current Frame* and the per-frame half of *Tools*. They keep
        their tag names (``active_tid``, ``obj_listbox``, ``candidate_count``,
        ``accept_all_btn``, ``discard_all_btn``, ``concept_text``) so the eight
        annotator callbacks that write into them work unchanged: the widget
        moved, the contract did not.
        """
        with dpg.group(tag=self._t("sec_nothing"), show=True):
            dpg.add_combo(
                [],
                tag=self._t("n_class"),
                label="Class",
                width=140,
                callback=lambda s, a: self._call("set_class", a),
            )
            # The rail owns tool selection. Explain the first action here
            # without adding a second button for the same click tool.
            dpg.add_text("Add a box", color=theme.TEXT_DIM)
            dpg.add_text(
                f"Press {self._key('tool.click_box')}, then click the centre of an animal. "
                f"Drag the handles to adjust the box. "
                f"Use {self._key('tool.draw_box')} to draw by hand.",
                wrap=self.WIDTH - 20,
            )
            # Labelled with the class the combo above is set to, because the
            # question it asks is "find every *what*". It used to be a blank
            # text box captioned "Find all in frame", which made the user type
            # a word the app already knew.
            with dpg.collapsing_header(label="Find all at once (optional)", default_open=False):
                dpg.add_text(
                    "Uses an installed AI helper. Check each dashed box before accepting it.",
                    color=theme.TEXT_DIM, wrap=self.WIDTH - 30,
                )
                widgets.secondary_button(
                    "Find all: animal",
                    self._cb("find_all_clicked"),
                    width=-1,
                    height=28,
                    tag="find_all_btn",
                )
            widgets.attach_tooltip(
                "find_all_btn",
                "Ask the segmentation helper for every animal in this frame "
                "at once, by name. The boxes arrive as candidates: dashed, "
                "unsaved, and yours to accept or discard.",
            )
            dpg.add_text("", tag=self._t("n_frame"), color=theme.TEXT)
            dpg.add_text("", tag=self._t("n_time"), color=theme.TEXT_DIM)
            dpg.add_text("", tag=self._t("n_animals"), color=theme.TEXT)
            dpg.add_text(
                "",
                tag=self._t("n_count_note"),
                color=theme.PENDING,
                wrap=self.WIDTH - 20,
                show=False,
            )
            dpg.add_text("", tag=self._t("n_status"), color=theme.TEXT_DIM, wrap=self.WIDTH - 20)

            # ── candidates on this frame ─────────────────────────────
            # Shown only when there are any: an Accept All that is always
            # present and usually disabled teaches the user to ignore it.
            widgets.gap(2)
            with dpg.group(tag=self._t("n_cand_group"), show=False):
                dpg.add_text("No candidates", tag="candidate_count", color=theme.CANDIDATE)
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Accept all",
                        tag="accept_all_btn",
                        enabled=False,
                        height=28,
                        width=(self.WIDTH - 32) // 2,
                        callback=self._cb("accept_frame"),
                    )
                    dpg.add_button(
                        label="Discard all",
                        tag="discard_all_btn",
                        enabled=False,
                        height=28,
                        width=(self.WIDTH - 32) // 2,
                        callback=self._cb("reject"),
                    )
                theme.bind("accept_all_btn", theme.accent_theme())
                theme.bind("discard_all_btn", theme.ghost_theme())

            # ── boxes on this frame ──────────────────────────────────
            widgets.gap(2)
            dpg.add_text("Boxes on this frame", color=theme.TEXT_DIM)
            dpg.add_listbox(
                items=[],
                tag="obj_listbox",
                num_items=5,
                width=-1,
                callback=self._cb_passthrough("select_object"),
            )
            widgets.attach_tooltip(
                "obj_listbox",
                "Every box on this frame. Click one to select it - this is the "
                "fastest way to reach a box that is underneath another.",
            )

            # ── the track new boxes get ──────────────────────────────
            # The whole block disappears when the project does not follow
            # individuals; the tags survive so nothing that writes into them
            # has to know.
            widgets.gap(2)
            with dpg.group(tag=self._t("n_track_block")):
                with dpg.group(horizontal=True):
                    dpg.add_text("New boxes use track", color=theme.TEXT_DIM)
                    dpg.add_text("", tag="active_track_color_label")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(
                        tag="active_tid",
                        default_value=1,
                        width=90,
                        min_value=0,
                        max_value=9999,
                        step=1,
                        callback=self._cb_passthrough("set_active_track"),
                    )
                    widgets.secondary_button(
                        "New",
                        self._cb("new_track"),
                        width=70,
                        height=26,
                        tag=self._t("n_new_track"),
                    )
                widgets.attach_tooltip(
                    self._t("n_new_track"),
                    f"Start a fresh track id for the next box you draw "
                    f"({self._key('edit.new_track')}).",
                )

            # ── this frame is one of the round's, and it is empty ──────
            # Only on a chosen frame with nothing on it, which is the only
            # state where "there is nothing here" is a decision rather than a
            # description. Everywhere else the frame's own boxes already say
            # it has been looked at.
            with dpg.group(tag=self._t("n_empty_group"), show=False):
                widgets.gap(1)
                widgets.secondary_button(
                    f"Nothing here - checked ({self._key('review.mark_checked')})",
                    self._cb("mark_checked"),
                    width=-1,
                    height=28,
                    tag=self._t("n_empty"),
                )
                widgets.attach_tooltip(
                    self._t("n_empty"),
                    "Record that you looked at this frame and there is nothing "
                    "to box. It counts as done for this round, and it trains "
                    "as an empty frame - which is a sample the model needs, "
                    "not a gap in the data.",
                )

            widgets.gap(2)
            dpg.add_text("", tag=self._t("n_hint"), color=theme.TEXT_DIM, wrap=self.WIDTH - 20)

    # -- a box ------------------------------------------------------------
    def _build_box(self) -> None:
        with dpg.group(tag=self._t("sec_box"), show=False):
            widgets.primary_button("Continue this animal", self._cb("continue_animal"), height=32)
            dpg.add_combo(
                ["Automatic", "Fast tracking", "Accurate propagation", "Copy to next frame"],
                default_value="Automatic",
                width=-1,
                callback=lambda s, a: self._call("continue_animal", a),
            )
            widgets.status_pill(self._t("b_state"), "Verified", "ok")
            widgets.gap(1)

            dpg.add_combo(
                (),
                tag=self._t("b_class"),
                label="Class",
                width=150,
                callback=lambda s, a: self._call("set_class", a),
            )
            dpg.add_input_int(
                tag=self._t("b_track"),
                label="Track",
                width=110,
                min_value=-1,
                min_clamped=True,
                step=1,
                callback=lambda s, a: self._call("set_track", int(a)),
            )
            widgets.attach_tooltip(
                self._t("b_track"),
                "Which individual this box is. -1 means no identity was "
                "asserted for it.",
            )
            dpg.add_combo(
                (),
                tag=self._t("b_action"),
                label="Behavior",
                width=150,
                callback=lambda s, a: self._call("set_action_name", a),
            )

            widgets.gap(1)
            dpg.add_text("", tag=self._t("b_geometry"), color=theme.TEXT_DIM, wrap=self.WIDTH - 20)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Rotate", self._cb("toggle_obb"), width=86, height=26, tag=self._t("b_obb")
                )
                dpg.add_text("", tag=self._t("b_angle"), color=theme.TEXT_DIM)
            widgets.attach_tooltip(
                self._t("b_obb"),
                "Turn this box into a rotated box, or back into an upright one. "
                "[ and ] then turn it 5 degrees at a time.",
            )

            widgets.gap(1)
            dpg.add_checkbox(
                label="Occluded",
                tag=self._t("b_occluded"),
                callback=lambda s, a: self._call("set_occluded", bool(a)),
            )
            dpg.add_checkbox(
                label="Is crowd",
                tag=self._t("b_crowd"),
                callback=lambda s, a: self._call("set_crowd", bool(a)),
            )

            # ── one-press behaviors ──────────────────────────────────
            # The old "Quick Actions (0-4)" panel, which sat in the sidebar
            # under a header that claimed five while the keys covered ten. It
            # belongs against the selection it acts on. Tag preserved so
            # ``_update_action_buttons`` fills it unchanged.
            widgets.gap(2)
            dpg.add_text("Set behavior", color=theme.TEXT_DIM)
            dpg.add_group(tag="action_buttons_group")

            widgets.gap(1)
            dpg.add_text("", tag=self._t("b_source"), color=theme.TEXT_FAINT, wrap=self.WIDTH - 20)
            widgets.gap(2)
            widgets.danger_button(
                f"Delete box  ({self._key('edit.delete')})",
                self._cb("delete"),
                height=28,
                tag=self._t("b_delete"),
            )

    # -- a candidate ------------------------------------------------------
    def _build_candidate(self) -> None:
        with dpg.group(tag=self._t("sec_candidate"), show=False):
            widgets.status_pill(self._t("c_state"), "Needs review", "pending")
            widgets.gap(1)
            dpg.add_text("", tag=self._t("c_score"), color=theme.TEXT)
            dpg.add_text("", tag=self._t("c_reason"), color=theme.PENDING, wrap=self.WIDTH - 20)
            widgets.gap(2)

            widgets.primary_button("Accept", self._cb("accept"), height=32, tag=self._t("c_accept"))
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Reject",
                    self._cb("reject"),
                    width=(self.WIDTH - 30) // 2,
                    height=28,
                    tag=self._t("c_reject"),
                )
                widgets.secondary_button(
                    "Edit",
                    self._cb("edit"),
                    width=(self.WIDTH - 30) // 2,
                    height=28,
                    tag=self._t("c_edit"),
                )

            widgets.gap(2)
            dpg.add_text("Accept more than one", color=theme.TEXT_DIM)
            dpg.add_separator()
            for label, name, help_text in (
                ("This frame", "accept_frame", "Accept every candidate on this frame."),
                (
                    "This tracklet",
                    "accept_tracklet",
                    "Accept every candidate of this track, across the frames it "
                    "runs without a gap.",
                ),
                (
                    "This clip",
                    "accept_clip",
                    "Accept every candidate inside the clip around this frame.",
                ),
                ("All visible", "accept_visible", "Accept every candidate currently on screen."),
                (
                    "Similar predictions",
                    "accept_similar",
                    "Accept candidates that look like this one - same track, "
                    "similar size, similar confidence.",
                ),
            ):
                tag = self._t(f"c_batch_{name}")
                widgets.secondary_button(label, self._cb(name), width=-1, height=26, tag=tag)
                widgets.attach_tooltip(tag, help_text)

    # -- a tracklet -------------------------------------------------------
    def _build_tracklet(self) -> None:
        with dpg.group(tag=self._t("sec_tracklet"), show=False):
            dpg.add_text("", tag=self._t("t_range"), color=theme.TEXT)
            dpg.add_text("", tag=self._t("t_count"), color=theme.TEXT_DIM)
            widgets.gap(1)
            dpg.add_text("Confidence", color=theme.TEXT_DIM)
            dpg.add_text("", tag=self._t("t_conf"), color=theme.TEXT)
            widgets.gap(1)
            dpg.add_text("", tag=self._t("t_issues"), color=theme.PENDING, wrap=self.WIDTH - 20)
            widgets.gap(2)
            widgets.primary_button(
                "Accept tracklet", self._cb("accept_tracklet"), height=34, tag=self._t("t_accept")
            )
            widgets.attach_tooltip(
                self._t("t_accept"),
                "Accept every candidate in this tracklet at once. This is the "
                "whole point of a tracklet: one decision instead of hundreds.",
            )
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Reject all",
                    self._cb("reject_tracklet"),
                    width=(self.WIDTH - 30) // 2,
                    height=28,
                    tag=self._t("t_reject"),
                )
                widgets.secondary_button(
                    "Go to start",
                    self._cb("goto_tracklet"),
                    width=(self.WIDTH - 30) // 2,
                    height=28,
                    tag=self._t("t_goto"),
                )

    # -- a behavior interval ----------------------------------------------
    def _build_interval(self) -> None:
        with dpg.group(tag=self._t("sec_interval"), show=False):
            widgets.status_pill(self._t("i_state"), "Proposed", "pending")
            widgets.gap(1)
            dpg.add_combo(
                (),
                tag=self._t("i_label"),
                label="Behavior",
                width=150,
                callback=lambda s, a: self._call("relabel_interval", a),
            )
            dpg.add_text("", tag=self._t("i_span"), color=theme.TEXT)
            dpg.add_text("", tag=self._t("i_conf"), color=theme.TEXT_DIM)
            widgets.gap(2)
            widgets.primary_button(
                "Accept interval", self._cb("confirm_interval"), height=32, tag=self._t("i_accept")
            )
            widgets.gap(1)
            dpg.add_text("Adjust the edges", color=theme.TEXT_DIM)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Start here",
                    self._cb("interval_start_here"),
                    width=(self.WIDTH - 30) // 2,
                    height=26,
                    tag=self._t("i_start"),
                )
                widgets.secondary_button(
                    "End here",
                    self._cb("interval_end_here"),
                    width=(self.WIDTH - 30) // 2,
                    height=26,
                    tag=self._t("i_end"),
                )
            widgets.attach_tooltip(
                self._t("i_start"), "Move this behavior's start to the frame you are on."
            )
            widgets.attach_tooltip(
                self._t("i_end"), "Move this behavior's end to the frame you are on."
            )
            widgets.gap(1)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Split here",
                    self._cb("split_interval"),
                    width=(self.WIDTH - 30) // 2,
                    height=26,
                    tag=self._t("i_split"),
                )
                widgets.danger_button(
                    "Delete",
                    self._cb("delete_interval"),
                    width=(self.WIDTH - 30) // 2,
                    height=26,
                    tag=self._t("i_delete"),
                )

    # -- a review item ----------------------------------------------------
    def _build_issue(self) -> None:
        with dpg.group(tag=self._t("sec_issue"), show=False):
            widgets.status_pill(self._t("r_kind"), "Review", "pending")
            widgets.gap(1)
            dpg.add_text("", tag=self._t("r_detail"), color=theme.TEXT, wrap=self.WIDTH - 20)
            widgets.gap(1)
            dpg.add_text("", tag=self._t("r_help"), color=theme.TEXT_DIM, wrap=self.WIDTH - 20)
            widgets.gap(1)
            dpg.add_text("", tag=self._t("r_where"), color=theme.TEXT_FAINT)
            widgets.gap(2)

            # The ID-switch pair, shown only for that kind: two buttons, one
            # click, no typing (§15).
            with dpg.group(tag=self._t("r_swap_group"), show=False):
                widgets.primary_button(
                    "IDs are correct", self._cb("ids_correct"), height=32, tag=self._t("r_ids_ok")
                )
                widgets.secondary_button(
                    "Swap IDs from here", self._cb("swap_ids"), height=30, tag=self._t("r_swap")
                )
                widgets.attach_tooltip(
                    self._t("r_swap"),
                    "Exchange these two track ids from this frame onward. " "Ctrl+Z takes it back.",
                )
                widgets.gap(1)

            with dpg.group(tag=self._t("r_plain_group"), show=True):
                widgets.primary_button(
                    "Accept", self._cb("issue_accept"), height=32, tag=self._t("r_accept")
                )
                with dpg.group(horizontal=True):
                    widgets.secondary_button(
                        "Reject",
                        self._cb("issue_reject"),
                        width=(self.WIDTH - 30) // 2,
                        height=28,
                        tag=self._t("r_reject"),
                    )
                    widgets.secondary_button(
                        "Skip",
                        self._cb("issue_skip"),
                        width=(self.WIDTH - 30) // 2,
                        height=28,
                        tag=self._t("r_skip"),
                    )

            widgets.gap(1)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Previous",
                    self._cb("issue_prev"),
                    width=(self.WIDTH - 30) // 2,
                    height=26,
                    tag=self._t("r_prev"),
                )
                widgets.secondary_button(
                    "Next",
                    self._cb("issue_next"),
                    width=(self.WIDTH - 30) // 2,
                    height=26,
                    tag=self._t("r_next"),
                )
            widgets.gap(1)
            dpg.add_text("", tag=self._t("r_progress"), color=theme.TEXT_DIM)

    # -- a sampled clip ---------------------------------------------------
    def _build_clip(self) -> None:
        """A clip is a question, so the pane's job is to let one answer start.

        The animal count is the number that decides the outcome, not a detail:
        a propagation can only follow animals it was prompted with, so a clip
        seeded from a frame holding one of two flies leaves the other one
        unproposed for its whole length. Showing the count and the frame it
        came from is what makes that visible *before* the run rather than
        afterwards.
        """
        with dpg.group(tag=self._t("sec_clip"), show=False):
            dpg.add_text("", tag=self._t("cl_range"), color=theme.TEXT)
            dpg.add_text("", tag=self._t("cl_time"), color=theme.TEXT_DIM)
            widgets.gap(1)
            dpg.add_text("Sampled because", color=theme.TEXT_DIM)
            dpg.add_text("", tag=self._t("cl_reason"), color=theme.TEXT,
                         wrap=self.WIDTH - 20)
            widgets.gap(1)
            dpg.add_text("Would follow", color=theme.TEXT_DIM)
            dpg.add_text("", tag=self._t("cl_seed"), color=theme.TEXT,
                         wrap=self.WIDTH - 20)
            widgets.gap(2)
            widgets.primary_button(
                "Follow animals across this clip",
                self._cb("clip_follow"),
                height=34,
                tag=self._t("cl_follow"),
            )
            widgets.attach_tooltip(
                self._t("cl_follow"),
                "Follows every animal on the seed frame across the whole clip, "
                "in one pass over its frames. The boxes arrive as candidates "
                "in the Review Inbox; nothing is saved until you accept them.",
            )
            dpg.add_text("", tag=self._t("cl_blocked"), color=theme.PENDING,
                         wrap=self.WIDTH - 20)
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Go to seed frame",
                    self._cb("clip_goto_seed"),
                    width=(self.WIDTH - 30) // 2,
                    height=28,
                    tag=self._t("cl_goto_seed"),
                )
                widgets.secondary_button(
                    "Go to start",
                    self._cb("clip_goto_start"),
                    width=(self.WIDTH - 30) // 2,
                    height=28,
                    tag=self._t("cl_goto"),
                )

    # ── display ──────────────────────────────────────────────────────────
    def show(self, context: InspectorContext) -> None:
        """Switch to a context and fill it in. The only public update path."""
        if not self._built:
            return
        self.context = context
        kind = context.kind if context.kind in CONTEXTS else NOTHING

        for name in CONTEXTS:
            self._show(self._t(f"sec_{name}"), name == kind)

        # One place decides whether a track control is on screen, and it is
        # the project's own answer rather than each filler's guess.
        self._show(self._t("n_track_block"), context.use_track_ids)
        self._show(self._t("b_track"), context.use_track_ids)

        self._set(self._t("title"), _TITLES.get(kind, "Inspector"))
        self._set(self._t("subtitle"), _SUBTITLES.get(kind, ""))
        getattr(self, _FILLERS[kind])(context)

    # -- fillers ----------------------------------------------------------
    def _fill_nothing(self, ctx: InspectorContext) -> None:
        self._set(self._t("n_frame"), f"Frame {ctx.frame:,} of {max(ctx.num_frames - 1, 0):,}")
        self._set(self._t("n_time"), _timecode(ctx.frame, ctx.fps))
        self._set(
            self._t("n_animals"),
            ctx.keyframe_note or f"{ctx.animals} animal{'s' if ctx.animals != 1 else ''} here",
        )

        # The count check, right where the count is shown. This is the cheapest
        # place in the whole app to catch a missed animal (§14).
        mismatch = ctx.expected_animals is not None and ctx.animals != ctx.expected_animals
        self._show(self._t("n_count_note"), mismatch)
        if mismatch:
            expected = ctx.expected_animals or 0
            self._set(
                self._t("n_count_note"),
                f"This project expects {expected}. "
                + (
                    "Look for one that was not detected."
                    if ctx.animals < expected
                    else "Look for a box on a reflection or on debris."
                ),
            )

        # The candidate block appears only when there is something to accept.
        # An Accept All that is permanently present and usually disabled is a
        # control the eye learns to skip.
        self._show(self._t("n_cand_group"), bool(ctx.pending_here))
        self._show(
            self._t("n_empty_group"),
            bool(ctx.chosen_frame) and not ctx.animals and not ctx.pending_here and not ctx.checked,
        )

        self._set(
            self._t("n_status"),
            f"{ctx.verified_frames:,} frame"
            f"{'s' if ctx.verified_frames != 1 else ''} annotated"
            + (
                f" - {ctx.pending_here} candidate" f"{'s' if ctx.pending_here != 1 else ''} here"
                if ctx.pending_here
                else ""
            ),
        )

        # Find-all is optional; its availability is resolved on first use.
        self._set(self._t("n_hint"), self._next_action_hint(ctx))
        if dpg.does_item_exist(self._t("n_class")):
            dpg.configure_item(self._t("n_class"), items=list(ctx.class_names))
            if ctx.class_names:
                self._set(
                    self._t("n_class"),
                    ctx.class_names[min(ctx.active_class_id, len(ctx.class_names) - 1)],
                )
        # The find-all button names the class it will search for, so the user
        # reads the question instead of typing the answer.
        self._label("find_all_btn", f"Find all: {self.find_all_word(ctx)}")

    @staticmethod
    def find_all_word(ctx: InspectorContext) -> str:
        """The noun *Find every ... in this frame* searches for.

        The active class, not ``class_names[0]``: a two-class project switches
        the combo to say which animal the next box is, and the button has to
        follow it or the two controls disagree on the same screen.
        """
        names = list(ctx.class_names)
        if not names:
            return "animal"
        return str(names[min(max(ctx.active_class_id, 0), len(names) - 1)])

    def set_find_all_available(self, available: bool, reason: str = "") -> None:
        """Show or lock the concept prompt, from the annotator's capability probe.

        Locked, never hidden (P4): dimming teaches that the feature exists and
        what would turn it on; hiding teaches that it does not exist.
        """
        if not dpg.does_item_exist("find_all_btn"):
            return
        if available:
            widgets.unlock("find_all_btn")
        else:
            widgets.locked(
                "find_all_btn",
                reason
                or "No segmentation helper can run on this machine. Install "
                "one with:  pip install sam2 huggingface_hub\n"
                "You can keep labelling without it: press C and click an "
                "animal, or W and draw a box.",
            )

    def _next_action_hint(self, ctx: InspectorContext) -> str:
        """The one sentence that says what to do now (§23).

        Ordered by what is most useful, not by what is most impressive: an
        unreviewed candidate outranks an empty frame, because ignoring it is
        how candidates silently fail to reach the training set.
        """
        if ctx.pending_here:
            return (
                f"Press {self._key('review.accept_all')} to accept the "
                f"candidates here, or "
                f"{self._key('review.discard_all')} to discard them."
            )
        if ctx.animals == 0:
            return (
                f"No boxes yet. No trained detector is needed: "
                f"press {self._key('tool.click_box')} and click an animal to start."
            )
        return (
            f"{self._key('nav.next')} and {self._key('nav.prev')} move "
            f"between frames. {self._key('file.save')} saves."
        )

    def _fill_box(self, ctx: InspectorContext) -> None:
        state = ctx.state
        widgets.status_pill(
            self._t("b_state"),
            ann_status.STATE_LABELS.get(state, state),
            {"verified": "ok", "predicted": "idle", "pending": "pending"}.get(state, "idle"),
        )

        if dpg.does_item_exist(self._t("b_class")):
            dpg.configure_item(self._t("b_class"), items=list(ctx.class_names) or ["fly"])
        self._set(self._t("b_class"), ctx.class_label())

        self._set(self._t("b_track"), int(ctx.track_id))

        options = ["Not set"] + list(ctx.action_names)
        if dpg.does_item_exist(self._t("b_action")):
            dpg.configure_item(self._t("b_action"), items=options)
        self._set(self._t("b_action"), ctx.action_label())

        self._set(self._t("b_geometry"), "Rotated box" if ctx.has_obb else "Upright box")
        self._label(self._t("b_obb"), "Make upright" if ctx.has_obb else "Make rotated")
        self._set(self._t("b_angle"), f"{ctx.angle_deg:+.0f} deg" if ctx.has_obb else "")

        self._set(self._t("b_occluded"), bool(ctx.occluded))
        self._set(self._t("b_crowd"), bool(ctx.is_crowd))

        origin = {
            "human": "You drew this box.",
            "auto": "The detector proposed this box.",
            "track": "A tracker placed this box.",
            "wand": "Click animal fitted this box.",
        }.get(ctx.source, f"Source: {ctx.source}")
        if ctx.score is not None:
            origin += f" Confidence {ctx.score:.2f}."
        self._set(self._t("b_source"), origin)

    def _fill_candidate(self, ctx: InspectorContext) -> None:
        score = ctx.score if ctx.score is not None else 1.0
        self._set(self._t("c_score"), f"Confidence {score:.2f}")

        reason = ""
        if ctx.item is not None:
            kind = getattr(ctx.item, "kind", "")
            reason = quality.ISSUE_HELP.get(kind, getattr(ctx.item, "detail", ""))
        elif score < 0.45:
            reason = quality.ISSUE_HELP[quality.LOW_CONFIDENCE]
        self._set(self._t("c_reason"), reason)

        state = ann_status.PENDING if reason else ann_status.PREDICTED
        widgets.status_pill(
            self._t("c_state"), ann_status.STATE_LABELS[state], "pending" if reason else "idle"
        )
        self._label(self._t("c_accept"), f"Accept  ({self._key('review.accept_all')})")

    def _fill_tracklet(self, ctx: InspectorContext) -> None:
        part = ctx.tracklet
        if part is None:
            return
        self._set(self._t("t_range"), f"{part.label()}   frames {part.start:,}-{part.end:,}")
        seconds = part.duration_seconds(ctx.fps)
        self._set(
            self._t("t_count"),
            f"{part.count:,} candidate{'s' if part.count != 1 else ''}"
            + (f" - {seconds:.1f} s" if seconds else ""),
        )
        # Minimum before mean: the weakest frame is what decides whether a bulk
        # accept is safe, so it is read first.
        self._set(self._t("t_conf"), f"minimum {part.min_score:.2f}   mean {part.mean_score:.2f}")

        if part.issues:
            labels = [quality.ISSUE_LABELS.get(i, i) for i in part.issues]
            self._set(self._t("t_issues"), "Flagged: " + ", ".join(labels))
        elif not part.is_contiguous:
            gaps = len(part.gaps)
            self._set(
                self._t("t_issues"), f"{gaps} gap{'s' if gaps != 1 else ''} inside this range."
            )
        else:
            self._set(self._t("t_issues"), "No checks objected to this tracklet.")

        self._label(self._t("t_accept"), f"Accept all {part.count:,}")

    def _fill_interval(self, ctx: InspectorContext) -> None:
        interval = ctx.interval
        if interval is None:
            return
        confirmed = getattr(interval, "source", "auto") == "human"
        widgets.status_pill(
            self._t("i_state"),
            "Confirmed" if confirmed else "Proposed",
            "ok" if confirmed else "pending",
        )

        if dpg.does_item_exist(self._t("i_label")):
            dpg.configure_item(self._t("i_label"), items=list(ctx.action_names))
        self._set(self._t("i_label"), interval.label(ctx.action_names))

        self._set(
            self._t("i_span"),
            f"{_timecode(interval.start, ctx.fps)} - "
            f"{_timecode(interval.end, ctx.fps)}   "
            f"({interval.span:,} frames)",
        )
        self._set(
            self._t("i_conf"),
            "Confirmed by you" if confirmed else f"Confidence {interval.confidence:.2f}",
        )

    def _fill_issue(self, ctx: InspectorContext) -> None:
        item = ctx.item
        if item is None:
            return
        kind = getattr(item, "kind", "")
        severity = float(getattr(item, "severity", 0.5))
        widgets.status_pill(
            self._t("r_kind"),
            quality.ISSUE_LABELS.get(kind, kind),
            "error" if severity >= 0.66 else "pending",
        )
        self._set(self._t("r_detail"), getattr(item, "detail", ""))
        self._set(self._t("r_help"), quality.ISSUE_HELP.get(kind, ""))

        where = f"Frames {item.start:,}"
        if item.end != item.start:
            where += f"-{item.end:,}"
        tracks = getattr(item, "track_ids", ())
        if tracks:
            where += (
                "   track" + ("s " if len(tracks) != 1 else " ") + ", ".join(str(t) for t in tracks)
            )
        self._set(self._t("r_where"), where)

        is_switch = kind == quality.ID_SWITCH_RISK
        self._show(self._t("r_swap_group"), is_switch)
        self._show(self._t("r_plain_group"), not is_switch)

        self._label(self._t("r_accept"), f"Accept  ({self._key('review.accept_all')})")
        self._label(self._t("r_reject"), f"Reject  ({self._key('review.discard_all')})")
        self._set(self._t("r_progress"), ctx.issue_progress)

    def _fill_clip(self, ctx: InspectorContext) -> None:
        clip = ctx.clip
        if clip is None:
            return
        start, end = int(clip.start), int(clip.end)
        length = end - start + 1
        self._set(self._t("cl_range"), f"Frames {start:,}-{end:,}   ({length:,} frames)")
        seconds = length / ctx.fps if ctx.fps > 0 else 0.0
        span = f"{_timecode(start, ctx.fps)} - {_timecode(end, ctx.fps)}"
        self._set(
            self._t("cl_time"), span + (f"   {seconds:.1f} s" if seconds else "")
        )

        reason = str(getattr(clip, "reason", "") or "").replace("_", " ") or "sampling"
        score = getattr(clip, "score", 0.0) or 0.0
        self._set(
            self._t("cl_reason"),
            reason + (f"   (score {float(score):.2f})" if score else ""),
        )

        if ctx.clip_seed is None or ctx.clip_animals <= 0:
            # Not a warning about the tool: a propagation follows what it was
            # prompted with, so with nothing drawn there is nothing to follow.
            self._set(
                self._t("cl_seed"),
                "Nothing yet. A propagation can only follow animals it was "
                "prompted with, so draw a box (W), click an animal (C), or use "
                "Find every animal in this frame, on one frame of this clip first.",
            )
        else:
            count = int(ctx.clip_animals)
            self._set(
                self._t("cl_seed"),
                f"{count} animal{'s' if count != 1 else ''} from frame "
                f"{int(ctx.clip_seed):,}, the fullest frame in this clip.",
            )

        blocked = ctx.clip_blocked
        if not blocked and (ctx.clip_seed is None or ctx.clip_animals <= 0):
            blocked = "There is no box inside this clip to follow."
        if blocked:
            widgets.locked(self._t("cl_follow"), blocked)
        else:
            widgets.unlock(self._t("cl_follow"))
        self._set(self._t("cl_blocked"), blocked)
        self._show(self._t("cl_goto_seed"), ctx.clip_seed is not None)


def _timecode(frame: int, fps: float) -> str:
    """``01:24.20`` — the same format :mod:`htrtdetr.annotation.intervals` uses."""
    if fps <= 0:
        return f"frame {frame}"
    total = frame / fps
    minutes = int(total // 60)
    return f"{minutes:02d}:{total - minutes * 60:05.2f}"
