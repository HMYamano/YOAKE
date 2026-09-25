"""A list of named phases that fills top-down while a long job runs.

Two dialogs need this now - *Pick frames to label* and *Train on these labels* -
and they need it to look and behave identically, because a user who has watched
one run learns to read the other at a glance. Before this module the second one
would have been a copy of the first's forty lines plus its two glyph tables,
and the copy would have drifted the first time either grew a state.

The states are ASCII on purpose: the list is the first thing drawn when a run
starts, which can be before the font has loaded, and it is also the thing most
likely to be read out of a screenshot pasted into a bug report.

``waiting`` / ``running`` / ``done`` / ``skipped`` / ``failed`` are the whole
vocabulary. In particular ``skipped`` exists and must be used: a run that did
not detect anything because it was not asked to has not *done* that phase, and
marking it done is the kind of small lie that makes the next failure
unreadable.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import dearpygui.dearpygui as dpg

from . import theme

__all__ = ["State", "GLYPH", "COLOR", "PhaseList"]

State = str

GLYPH: Dict[State, str] = {
    "waiting": "  ",
    "running": "> ",
    "done": "OK",
    "skipped": "--",
    "failed": "!!",
}

COLOR: Dict[State, Any] = {
    "waiting": theme.TEXT_FAINT,
    "running": theme.ACCENT,
    "done": theme.ACCEPTED,
    "skipped": theme.TEXT_FAINT,
    "failed": theme.DANGER,
}

#: States that a later phase's completion must not overwrite. ``advance_to``
#: completes the phases before the current one by inference, and a phase that
#: already failed or was skipped did not become done because the run moved on.
_STICKY = ("done", "skipped", "failed")


class PhaseList:
    """The widgets for one list of phases, plus the state behind them.

    ``owner`` is the tag prefix of the dialog that holds it, so two lists in
    one dearpygui context cannot collide.
    """

    def __init__(self, owner: str, labels: Sequence[str], *, wrap: int = 0) -> None:
        self.owner = owner
        self.labels = tuple(labels)
        self.wrap = wrap
        self._states: Dict[int, State] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def _t(self, suffix: str) -> str:
        return f"{self.owner}_phase_{suffix}"

    def build(self) -> None:
        """Three cells a row: the glyph, the label, and a detail that changes."""
        for index, label in enumerate(self.labels):
            with dpg.group(horizontal=True):
                dpg.add_text("  ", tag=self._t(f"glyph_{index}"), color=theme.TEXT_FAINT)
                dpg.add_text(label, tag=self._t(f"label_{index}"), color=theme.TEXT_FAINT)
                dpg.add_text("", tag=self._t(f"detail_{index}"), color=theme.TEXT_FAINT)

    def state(self, index: int) -> State:
        return self._states.get(index, "waiting")

    def set(self, index: int, state: State, detail: str = "") -> None:
        """Mark one phase. Safe to call from a marshalled worker callback."""
        if not (0 <= index < len(self.labels)):
            return
        self._states[index] = state
        glyph = GLYPH.get(state, "  ")
        colour = COLOR.get(state, theme.TEXT_FAINT)
        for tag, value in (
            (self._t(f"glyph_{index}"), glyph),
            (self._t(f"label_{index}"), self.labels[index]),
            (self._t(f"detail_{index}"), detail),
        ):
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, value)
                dpg.configure_item(tag, color=colour)

    def reset(self, first: int = 0) -> None:
        """Back to waiting, from ``first`` onwards.

        ``first`` is what keeps a per-item reset from erasing the phases that
        happened once for the whole job. A project batch resets its per-video
        phases at each new video; the phases before those describe work that
        was done once and is not being redone, and un-ticking them reads as a
        run that went backwards.
        """
        for index in range(max(0, first), len(self.labels)):
            self._states.pop(index, None)
            self.set(index, "waiting")

    def advance_to(self, index: int, detail: str = "") -> None:
        """``index`` is running; everything before it is done by inference.

        A runner reports what it is doing, not what it has finished, so this is
        how the earlier phases get their marks - except the sticky ones, which
        keep theirs.
        """
        for earlier in range(max(0, min(index, len(self.labels)))):
            if self._states.get(earlier) not in _STICKY:
                self.set(earlier, "done")
        self.set(index, "running", detail)

    def finish(self, *, state: State = "done", first: int = 0) -> None:
        """Mark everything still waiting or running, from ``first`` onwards."""
        for index in range(max(0, first), len(self.labels)):
            if self._states.get(index) not in _STICKY:
                self.set(index, state)

    def fail(self, index: Optional[int], detail: str = "") -> None:
        """Mark the phase that failed, leaving the ones before it alone."""
        if index is None:
            index = max(
                (i for i, s in self._states.items() if s == "running"), default=0
            )
        self.set(index, "failed", detail)
