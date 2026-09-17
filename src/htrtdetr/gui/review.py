"""
review.py — the Review Inbox, and the mode that empties it fast.

Two surfaces over one :class:`~htrtdetr.annotation.review_queue.ReviewQueue`:

:class:`ReviewInbox`
    A docked list. Filter chips per reason, a row per item, a progress line.
    This is where a user *sees* that 18,426 predictions came with 94 questions,
    and where they choose which kind of question to answer first.

:class:`ReviewMode`
    A full-bleed overlay with the video, one sentence, and three keys. No
    toolbar, no inspector, no sidebar. It exists because the Inbox is a good
    way to understand a queue and a bad way to empty one: at 94 items, every
    pixel that is not the video or the reason is a pixel the user has to look
    past, ninety-four times.

    ┌───────────────────────────────────────────────┐
    │  14 / 94        Possible ID switch            │
    │                                               │
    │                  [ video ]                    │
    │                                               │
    │  Two animals met here. Check the identities   │
    │  are the same after as before.                │
    │                                               │
    │  [ IDs are correct ]  [ Swap IDs from here ]  │
    │  Y accept   N reject   E edit   <- ->  Esc    │
    └───────────────────────────────────────────────┘

The loop is the feature
-----------------------
Review Mode plays the item's context on repeat — half a second before, the
problem, half a second after — because an ID switch cannot be judged from a
still frame, and scrubbing to find the before and after by hand costs more than
the decision does. The caller owns playback; this module owns the *range* and
tells the caller when it changes.

Neither class stores annotations or mutates anything. Decisions go to the queue
and to the caller's callbacks, which is where the one implementation of accept
lives (P1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

import dearpygui.dearpygui as dpg

from . import shortcuts, theme, widgets
from ..annotation import quality, review_queue

__all__ = ["ReviewInbox", "ReviewMode", "context_range"]


def context_range(item: Any, *, fps: float, num_frames: int,
                  pad_seconds: float = 0.75) -> Tuple[int, int]:
    """``(first, last)`` to loop for one review item.

    ``pad_seconds`` each side, floored at 6 frames so the loop is still a loop
    on a video whose fps the container under-reported, and clamped to the
    video so the player is never asked to seek past the end.
    """
    pad = max(6, int(round(pad_seconds * max(fps, 1.0))))
    start = int(getattr(item, "start", 0))
    end = int(getattr(item, "end", start))
    lo = max(0, start - pad)
    hi = end + pad
    if num_frames > 0:
        hi = min(hi, num_frames - 1)
    return lo, max(lo, hi)


# ══════════════════════════════════════════════════════════════════════════
# The docked list
# ══════════════════════════════════════════════════════════════════════════

class ReviewInbox:
    """The dockable queue list: filter chips, rows, progress, bulk actions."""

    WIDTH = 320
    #: Rows rendered at once. A queue capped at 500 per kind can still reach
    #: four figures, and dearpygui draws every row whether or not it is
    #: scrolled into view, so the list is paged rather than complete.
    PAGE = 120

    def __init__(
        self,
        tag: str = "review_inbox",
        *,
        on_select: Optional[Callable[[Any], None]] = None,
        on_start_review: Optional[Callable[[], None]] = None,
        on_accept_kind: Optional[Callable[[str], None]] = None,
        on_reanalyze: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tag = tag
        self.on_select = on_select
        self.on_start_review = on_start_review
        self.on_accept_kind = on_accept_kind
        self.on_reanalyze = on_reanalyze
        self.queue = review_queue.ReviewQueue()
        self.fps = 25.0
        self._built = False
        self._page = 0

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
    def _show(tag: str, visible: bool) -> None:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, show=bool(visible))

    # ── construction ─────────────────────────────────────────────────────
    def build(self, parent: str, height: int = -1, width: int = 0) -> None:
        """Create the panel. ``width`` overrides :attr:`WIDTH` so the inbox and
        the inspector share one column width chosen by the host."""
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)
        if width > 0:
            self.WIDTH = int(width)

        with dpg.child_window(tag=self.tag, parent=parent,
                              width=(width if width else self.WIDTH),
                              height=height, border=False):
            dpg.add_text("Review", color=theme.TEXT)
            theme.bind_h2(dpg.last_item())
            dpg.add_text("", tag=self._t("progress"), color=theme.TEXT_DIM,
                         wrap=self.WIDTH - 20)
            dpg.add_separator()
            widgets.gap(1)

            with dpg.group(tag=self._t("empty"), show=True):
                widgets.empty_state(
                    "Nothing to review yet.\n\n"
                    "Run Smart Annotate, or load predictions, and the places "
                    "worth a second look will appear here.",
                    "Smart Annotate...",
                    lambda *_: self._call(self.on_reanalyze),
                    wrap=self.WIDTH - 24, tag=self._t("empty_state"))

            with dpg.group(tag=self._t("body"), show=False):
                widgets.primary_button("Review them one by one",
                                       lambda *_: self._call(self.on_start_review),
                                       height=34, tag=self._t("start"))
                widgets.gap(1)

                dpg.add_text("Show", color=theme.TEXT_DIM)
                dpg.add_listbox((), tag=self._t("kinds"), width=-1, num_items=5,
                                callback=lambda *_: self._on_kind_change())
                with dpg.group(horizontal=True):
                    widgets.secondary_button(
                        "All reasons", lambda *_: self._on_show_all(),
                        width=(self.WIDTH - 30) // 2, height=24,
                        tag=self._t("all"))
                    widgets.secondary_button(
                        "Accept these", lambda *_: self._on_accept_kind(),
                        width=(self.WIDTH - 30) // 2, height=24,
                        tag=self._t("accept_kind"))
                widgets.attach_tooltip(
                    self._t("accept_kind"),
                    "Accept every unreviewed item of the reason selected "
                    "above. Items you already rejected are left alone.")

                dpg.add_checkbox(label="Hide what I have reviewed",
                                 tag=self._t("hide_done"), default_value=False,
                                 callback=lambda *_: self._on_kind_change())
                widgets.gap(1)
                dpg.add_separator()

                dpg.add_listbox((), tag=self._t("items"), width=-1,
                                num_items=14,
                                callback=lambda *_: self._on_row_click())
                with dpg.group(horizontal=True, tag=self._t("pager"), show=False):
                    widgets.secondary_button("Earlier", lambda *_: self._turn(-1),
                                             width=(self.WIDTH - 30) // 2,
                                             height=24, tag=self._t("prev_page"))
                    widgets.secondary_button("Later", lambda *_: self._turn(1),
                                             width=(self.WIDTH - 30) // 2,
                                             height=24, tag=self._t("next_page"))
                dpg.add_text("", tag=self._t("page_note"), color=theme.TEXT_FAINT)

        self._built = True
        self.refresh()

    @staticmethod
    def _call(fn: Optional[Callable], *args: Any) -> None:
        if fn is not None:
            fn(*args)

    # ── data ─────────────────────────────────────────────────────────────
    def set_queue(self, queue: review_queue.ReviewQueue, *, fps: float = 25.0) -> None:
        self.queue = queue
        self.fps = float(fps) if fps > 0 else 25.0
        self._page = 0
        self.refresh()

    # ── rendering ────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Rebuild the two lists and the progress line from the queue."""
        if not self._built:
            return

        has_items = len(self.queue) > 0
        self._show(self._t("empty"), not has_items)
        self._show(self._t("body"), has_items)
        self._set(self._t("progress"), self.queue.progress().describe())
        if not has_items:
            return

        kinds = self.queue.kinds()
        labels = [f"{self.queue.kind_label(k)}  ({n})" for k, n in kinds]
        self._kind_keys = [k for k, _ in kinds]
        if dpg.does_item_exist(self._t("kinds")):
            dpg.configure_item(self._t("kinds"), items=labels,
                               num_items=min(6, max(2, len(labels))))

        self._rows = self.queue.filtered()
        pages = max(1, (len(self._rows) + self.PAGE - 1) // self.PAGE)
        self._page = max(0, min(self._page, pages - 1))
        start = self._page * self.PAGE
        window = self._rows[start:start + self.PAGE]

        entries = []
        for item in window:
            decision = self.queue.decision(item)
            mark = {review_queue.ACCEPTED: "[ok] ",
                    review_queue.REJECTED: "[no] ",
                    review_queue.EDITED: "[ed] ",
                    review_queue.SKIPPED: "[--] "}.get(decision.state, "")
            entries.append(mark + item.describe(self.fps))
        self._window = window

        if dpg.does_item_exist(self._t("items")):
            dpg.configure_item(self._t("items"), items=entries or ["Nothing matches this filter"])

        self._show(self._t("pager"), pages > 1)
        self._set(self._t("page_note"),
                  f"Showing {start + 1}-{start + len(window)} of {len(self._rows)}"
                  if pages > 1 else "")

    # ── callbacks ────────────────────────────────────────────────────────
    def _on_kind_change(self) -> None:
        label = dpg.get_value(self._t("kinds")) if dpg.does_item_exist(self._t("kinds")) else ""
        keys = getattr(self, "_kind_keys", [])
        chosen = None
        for key, text in zip(keys, [f"{self.queue.kind_label(k)}  ({n})"
                                    for k, n in self.queue.kinds()]):
            if text == label:
                chosen = key
                break
        self.queue.set_filter(
            kinds=None if chosen is None else {chosen},
            hide_resolved=bool(dpg.get_value(self._t("hide_done")))
            if dpg.does_item_exist(self._t("hide_done")) else False)
        self._page = 0
        self.refresh()

    def _on_show_all(self) -> None:
        self.queue.clear_filter()
        if dpg.does_item_exist(self._t("kinds")):
            dpg.set_value(self._t("kinds"), "")
        self._page = 0
        self.refresh()

    def _on_accept_kind(self) -> None:
        label = dpg.get_value(self._t("kinds")) if dpg.does_item_exist(self._t("kinds")) else ""
        for key, text in zip(getattr(self, "_kind_keys", []),
                             [f"{self.queue.kind_label(k)}  ({n})"
                              for k, n in self.queue.kinds()]):
            if text == label:
                self._call(self.on_accept_kind, key)
                self.refresh()
                return

    def _on_row_click(self) -> None:
        label = dpg.get_value(self._t("items")) if dpg.does_item_exist(self._t("items")) else ""
        window = getattr(self, "_window", [])
        for item in window:
            decision = self.queue.decision(item)
            mark = {review_queue.ACCEPTED: "[ok] ",
                    review_queue.REJECTED: "[no] ",
                    review_queue.EDITED: "[ed] ",
                    review_queue.SKIPPED: "[--] "}.get(decision.state, "")
            if mark + item.describe(self.fps) == label:
                self.queue.select(item)
                self._call(self.on_select, item)
                return

    def _turn(self, direction: int) -> None:
        self._page += int(direction)
        self.refresh()


# ══════════════════════════════════════════════════════════════════════════
# The fast mode
# ══════════════════════════════════════════════════════════════════════════

class ReviewMode:
    """A stripped overlay for emptying the queue at one decision per second.

    The caller draws the video into :attr:`canvas_tag` exactly as it does in
    the annotator; this class owns everything around it and the loop range.
    """

    def __init__(
        self,
        tag: str = "review_mode",
        *,
        on_seek: Optional[Callable[[int], None]] = None,
        on_decision: Optional[Callable[[Any, str], None]] = None,
        on_edit: Optional[Callable[[Any], None]] = None,
        on_swap: Optional[Callable[[Any, bool], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ) -> None:
        self.tag = tag
        self.on_seek = on_seek
        self.on_decision = on_decision
        self.on_edit = on_edit
        self.on_swap = on_swap
        self.on_exit = on_exit

        self.queue = review_queue.ReviewQueue()
        self.fps = 25.0
        self.num_frames = 0
        self.item: Any = None
        self.loop: Tuple[int, int] = (0, 0)
        self._built = False
        self._active = False
        #: Playing by default. An ID switch cannot be judged from a still, and
        #: a human who has to press play on every one of ninety-four items has
        #: been given a worse tool than a scrub bar.
        self.playing = True
        #: Wall-clock of the last advance, so playback runs at the video's rate
        #: rather than at whatever the render loop happens to manage.
        self._last_tick = 0.0

    # ── plumbing ─────────────────────────────────────────────────────────
    def _t(self, suffix: str) -> str:
        return f"{self.tag}_{suffix}"

    @property
    def canvas_tag(self) -> str:
        """The drawlist the caller paints the frame into."""
        return self._t("canvas")

    @property
    def active(self) -> bool:
        return self._active

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

    def _key(self, action: str) -> str:
        """The key for a Review Mode action, from the ``review`` scope.

        Review Mode re-spends keys the annotator uses for something else (``E``
        draws a rotated box on the canvas and edits an item here), which is
        exactly what a separate scope in the keymap is for.
        """
        return shortcuts.label_for(action, scope="review")

    # ── construction ─────────────────────────────────────────────────────
    def build(self, *, width: int = 1100, height: int = 720) -> None:
        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)

        with dpg.window(tag=self.tag, label="Review", show=False,
                        no_collapse=True, no_scrollbar=True,
                        width=width, height=height,
                        on_close=lambda *_: self.close()):
            with dpg.group(horizontal=True):
                dpg.add_text("", tag=self._t("counter"), color=theme.TEXT_DIM)
                dpg.add_text("   ", color=theme.TEXT_FAINT)
                # The reason, as a headline. It is the one thing on this screen
                # the user must read, so it is first and it is large.
                dpg.add_text("", tag=self._t("headline"), color=theme.TEXT)
                theme.bind_h2(dpg.last_item())
            widgets.gap(1)

            dpg.add_drawlist(tag=self.canvas_tag, width=width - 40,
                             height=height - 230)

            widgets.gap(1)
            with dpg.group(horizontal=True):
                widgets.status_pill(self._t("kind"), "Review", "pending")
                dpg.add_text("", tag=self._t("loop"), color=theme.TEXT_FAINT)
                widgets.secondary_button("Pause", lambda *_: self.toggle_play(),
                                         width=90, height=24,
                                         tag=self._t("play"))
            dpg.add_text("", tag=self._t("detail"), color=theme.TEXT,
                         wrap=width - 60)
            dpg.add_text("", tag=self._t("help"), color=theme.TEXT_DIM,
                         wrap=width - 60)
            widgets.gap(1)

            with dpg.group(tag=self._t("swap_row"), horizontal=True, show=False):
                widgets.primary_button("IDs are correct",
                                       lambda *_: self._swap(False),
                                       width=220, height=36,
                                       tag=self._t("ids_ok"))
                widgets.secondary_button("Swap IDs from here",
                                         lambda *_: self._swap(True),
                                         width=220, height=36,
                                         tag=self._t("swap"))

            with dpg.group(tag=self._t("plain_row"), horizontal=True, show=True):
                widgets.primary_button("Accept", lambda *_: self.decide("accept"),
                                       width=160, height=36,
                                       tag=self._t("accept"))
                widgets.secondary_button("Reject", lambda *_: self.decide("reject"),
                                         width=160, height=36,
                                         tag=self._t("reject"))
                widgets.secondary_button("Edit", lambda *_: self.decide("edit"),
                                         width=160, height=36,
                                         tag=self._t("edit"))
                widgets.secondary_button("Skip", lambda *_: self.decide("skip"),
                                         width=120, height=36,
                                         tag=self._t("skip"))

            widgets.gap(1)
            dpg.add_text(self._legend(), tag=self._t("legend"),
                         color=theme.TEXT_FAINT)
            theme.bind_mono(dpg.last_item())

        self._built = True

    def _legend(self) -> str:
        """Generated from the keymap, so it cannot drift from the handlers.

        Permanently on screen rather than behind a help key (P10): a review
        pass is ninety-four repetitions of five keys, and a cheat sheet that
        has to be summoned is a cheat sheet nobody reads.
        """
        return "   ".join((
            f"{self._key('review.accept')} accept",
            f"{self._key('review.reject')} reject",
            f"{self._key('review.edit')} edit",
            f"{self._key('review.skip')} skip",
            f"{self._key('review.prev')} / {self._key('review.next')} back / next",
            f"{self._key('review.exit')} leave review",
        ))

    # ── lifecycle ────────────────────────────────────────────────────────
    def open(self, queue: review_queue.ReviewQueue, *, fps: float = 25.0,
             num_frames: int = 0) -> bool:
        """Enter Review Mode on the first unresolved item.

        ``False`` when the queue has nothing left, so the caller can say
        *All 94 reviewed* instead of opening an empty window.
        """
        if not self._built:
            self.build()
        self.queue = queue
        self.fps = float(fps) if fps > 0 else 25.0
        self.num_frames = int(num_frames)

        item = queue.current()
        if item is not None and queue.decision(item).done:
            item = queue.next_unresolved() or item
        if item is None:
            return False

        self._active = True
        vw, vh = widgets.viewport_size()
        self.resize(width=vw - 120, height=vh - 90)
        dpg.configure_item(self.tag, show=True)
        self._present(item)
        return True

    def resize(self, *, width: int, height: int) -> None:
        """Refit the window and the frame inside it to a new viewport.

        Review Mode is built once and lives as long as the annotator does, so
        without this it keeps whatever size the viewport had at startup: make
        the window smaller and its buttons go under the bottom edge, make it
        larger and the frame stays in a corner.
        """
        if not dpg.does_item_exist(self.tag):
            return
        width, height, x, y = widgets.fit_dialog(width, height)
        dpg.configure_item(self.tag, width=width, height=height, pos=[x, y])
        if dpg.does_item_exist(self.canvas_tag):
            dpg.configure_item(
                self.canvas_tag, width=width - 40, height=max(160, height - 230)
            )

    def close(self) -> None:
        self._active = False
        self._show(self.tag, False)
        if self.on_exit is not None:
            self.on_exit()

    # ── presentation ─────────────────────────────────────────────────────
    def _present(self, item: Any) -> None:
        self.item = item
        self.loop = context_range(item, fps=self.fps, num_frames=self.num_frames)

        kind = getattr(item, "kind", "")
        severity = float(getattr(item, "severity", 0.5))
        headline = quality.ISSUE_LABELS.get(kind, kind)
        self._set(self._t("headline"), headline)
        widgets.status_pill(self._t("kind"),
                            "Serious" if severity >= 0.66 else "Check",
                            "error" if severity >= 0.66 else "pending")
        self._set(self._t("loop"),
                  f"looping frames {self.loop[0]:,}-{self.loop[1]:,}"
                  if self.loop[1] > self.loop[0] else "")

        view = self.queue.filtered()
        position = view.index(item) + 1 if item in view else 0
        progress = self.queue.progress()
        self._set(self._t("counter"),
                  f"{position} / {len(view)}    {progress.remaining} left")

        self._set(self._t("detail"), getattr(item, "detail", ""))
        self._set(self._t("help"), quality.ISSUE_HELP.get(kind, ""))

        is_switch = kind == quality.ID_SWITCH_RISK
        self._show(self._t("swap_row"), is_switch)
        self._show(self._t("plain_row"), not is_switch)

        # Start the loop at its beginning every time, so the before-and-after
        # is seen from the start rather than from wherever the last item ended.
        self._last_tick = 0.0
        if self.on_seek is not None:
            self.on_seek(self.loop[0])

    def advance_loop(self, current: int) -> int:
        """Next frame of the loop. Call once per playback tick.

        Wrapping here rather than in the caller keeps the loop bounds in one
        place: the caller only has to ask what comes after the frame it is on.
        A frame outside the loop entirely restarts it, which is what happens
        when the user scrubs away and then lets playback resume.
        """
        lo, hi = self.loop
        nxt = int(current) + 1
        if nxt > hi or nxt < lo:
            return lo
        return nxt

    def toggle_play(self) -> None:
        self.playing = not self.playing
        if dpg.does_item_exist(self._t("play")):
            dpg.configure_item(self._t("play"),
                               label="Play" if not self.playing else "Pause")

    def due(self, now: float, fps: float) -> bool:
        """Is it time for the next loop frame?

        Rate-limited to the video's own frame rate, capped at 30 - a review
        loop that plays a 240 fps high-speed recording at 240 fps is a blur,
        and one that plays at the render loop's rate is a different speed on
        every machine.
        """
        if not (self._active and self.playing):
            return False
        interval = 1.0 / max(1.0, min(float(fps) if fps > 0 else 25.0, 30.0))
        if now - self._last_tick < interval:
            return False
        self._last_tick = now
        return True

    # ── decisions ────────────────────────────────────────────────────────
    def decide(self, verdict: str) -> None:
        """``accept`` / ``reject`` / ``edit`` / ``skip``, then move on.

        ``edit`` does **not** advance: the user is about to change something,
        and jumping to the next item would take the thing they are editing off
        the screen.
        """
        if self.item is None:
            return
        item = self.item

        if verdict == "accept":
            self.queue.accept(item)
        elif verdict == "reject":
            self.queue.reject(item)
        elif verdict == "skip":
            self.queue.skip(item)
        elif verdict == "edit":
            self.queue.edit(item)
            if self.on_edit is not None:
                self.on_edit(item)
            self.close()
            return

        if self.on_decision is not None:
            self.on_decision(item, verdict)
        self.next_item()

    def _swap(self, swapped: bool) -> None:
        if self.item is None:
            return
        item = self.item
        self.queue.swap_decision(item, swapped=swapped)
        if self.on_swap is not None:
            self.on_swap(item, swapped)
        self.next_item()

    def next_item(self) -> None:
        item = self.queue.next_unresolved()
        if item is None:
            self._finish()
            return
        self._present(item)

    def previous_item(self) -> None:
        item = self.queue.retreat()
        if item is not None:
            self._present(item)

    def _finish(self) -> None:
        progress = self.queue.progress()
        self._set(self._t("detail"),
                  f"All {progress.total:,} reviewed.")
        self._set(self._t("help"),
                  "Save to keep what you accepted, or train the model on it.")
        self._show(self._t("swap_row"), False)
        self._show(self._t("plain_row"), False)
        self.item = None

    # ── keyboard ─────────────────────────────────────────────────────────
    def handle_action(self, action: str) -> bool:
        """Route one keymap action. ``True`` when Review Mode consumed it.

        Called from the annotator's single key dispatcher, so Review Mode never
        registers handlers of its own and there is exactly one place where a
        key can be swallowed.
        """
        if not self._active:
            return False
        handler = {
            "review.accept": lambda: self.decide("accept"),
            "review.reject": lambda: self.decide("reject"),
            "review.edit": lambda: self.decide("edit"),
            "review.skip": lambda: self.decide("skip"),
            "review.next": self.next_item,
            "review.prev": self.previous_item,
            "review.exit": self.close,
            "playback.toggle": self.toggle_play,
        }.get(action)
        if handler is None:
            return False
        handler()
        return True
