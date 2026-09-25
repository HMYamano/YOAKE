"""*Activity* — one window that answers "what is it doing, and what went wrong".

    Activity                                                  [x]

    Now      Training the detector - pass 14 of 96 (round 3)
             Re-checking the candidates

    [ All ] [ Errors 2 ] [ Jobs ]        [ Copy ] [ Clear ] [ Open folder ]

    13:42:07  Round 3 started: 312 frames, up to 96 passes.
    13:44:51  ! Inference exited 1: CUDA out of memory
    13:44:51  Round 3 failed at "Training the detector".

    Run log  round-003/round.log  v                    [ Reload ]
    ...
    Epoch 14/96  loss 3.81  best 3.77
    ────────────────────────────────────────────────────────────
                                                       [ Close ]

**Why this exists.** Everything long in this app runs in a child process whose
output goes to a file in the run folder, and everything short reports through
the status line, which holds one sentence and forgets it the moment the next
one arrives. So the two questions a user actually has after a twenty-minute
round - *is it still going* and *what did it say when it stopped* - were each
answerable only by finding a folder. This window is the place they are both
answerable from, and it is one button away at all times.

**Hidden until asked for.** It is not a panel and it does not steal width from
the canvas: annotation is the job, and a log that is always on screen is a log
nobody reads and a hundred pixels nobody gets back. The pill in the top bar
carries the one-line version - what is running, or how many errors there are -
so the window only has to be opened when that line is not enough.

Two halves, and the split is deliberate:

:class:`ActivityLog`
    The ring buffer. Pure Python, no dearpygui, so the annotator can record
    into it before a viewport exists and a test can assert on what was
    recorded without one.

:class:`ActivityWindow`
    The view. It owns no state that is not on screen; everything it shows it
    reads back out of the log or off disk when it is refreshed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "INFO",
    "WARN",
    "ERROR",
    "JOB",
    "KINDS",
    "Entry",
    "ActivityLog",
    "ActivityWindow",
    "ActivityMixin",
    "classify",
    "pill_state",
    "find_run_logs",
    "tail",
]

#: The four kinds of line. ``JOB`` is what a long job says about itself when it
#: starts, finishes, fails or is stopped; ``INFO`` is everything the status line
#: said. Keeping them apart is what makes the *Jobs* filter worth having: after
#: a batch, the status line has said four hundred things and six of them are
#: the story.
INFO = "info"
WARN = "warn"
ERROR = "error"
JOB = "job"

KINDS: Tuple[str, ...] = (INFO, WARN, ERROR, JOB)

#: Marks in front of a line. ASCII, for the same reason
#: :mod:`htrtdetr.gui.phases` uses ASCII glyphs: this text is what gets pasted
#: into a bug report, and a cp932 console must be able to print it.
MARK: Dict[str, str] = {INFO: "  ", WARN: "? ", ERROR: "! ", JOB: "* "}


#: Phrases that make a status-line sentence a problem rather than a report.
#: Matched case-insensitively against the whole message.
#:
#: This is a heuristic and it is allowed to be: the status line is prose
#: written for a biologist, not a log level, and the alternative - an
#: explicit level at all two hundred call sites - is two hundred chances to
#: forget one. A message that reads like a failure is filed as one; the
#: *Jobs* filter and the pill's count are the only things that depend on it,
#: and both are recoverable by pressing *All*.
_ERROR_PHRASES: Tuple[str, ...] = (
    "could not",
    "cannot",
    "failed",
    "error",
    "no such file",
    "not found",
    "is not valid",
    "went wrong",
    "crashed",
    "out of memory",
    "traceback",
    "exited",
    "refused",
)

_WARN_PHRASES: Tuple[str, ...] = (
    "nothing",
    "no video",
    "no model",
    "no project",
    "no candidates",
    "select a",
    "first,",
    "skipped",
    "stopped",
    "turn off",
    "needs",
    "without",
)


def classify(message: str) -> str:
    """Which kind of line a status-line sentence is.

    Errors win over warnings, because "could not save: nothing selected"
    is a failure with a warning-shaped clause in it and filing it as a
    warning is how it gets missed.
    """
    low = str(message).lower()
    if any(p in low for p in _ERROR_PHRASES):
        return ERROR
    if any(p in low for p in _WARN_PHRASES):
        return WARN
    return INFO


@dataclass(frozen=True)
class Entry:
    """One line. ``at`` is wall-clock seconds, so it can be formatted late."""

    at: float
    kind: str
    text: str

    def format(self, *, mark: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.at))
        return f"{stamp}  {MARK.get(self.kind, '  ') if mark else ''}{self.text}"


class ActivityLog:
    """A bounded, thread-safe history of what the app has said.

    Thread-safe because the only other option is a rule that workers may not
    record, and a worker that fails is exactly the one whose message matters
    most. Appending under a lock costs a microsecond; the window reads a
    snapshot, so a render can never see a half-written list.

    Bounded because this runs for a working day. ``MAX`` lines is about a
    megabyte of text and covers every job of a session at the rate the status
    line actually changes - and the child processes' own logs are on disk in
    full, which is what :func:`find_run_logs` is for.
    """

    MAX = 4000

    def __init__(self, max_lines: int = MAX) -> None:
        self.max_lines = int(max_lines)
        self._entries: List[Entry] = []
        self._lock = threading.Lock()
        self._errors = 0
        self._warnings = 0
        #: Bumped on every change, so a view can skip a redraw it does not need.
        self.revision = 0
        self._last_text = ""

    # ── writing ──────────────────────────────────────────────────────────
    def record(self, text: str, kind: str = INFO) -> Optional[Entry]:
        """Append one line. Returns it, or ``None`` if it was a repeat.

        Consecutive duplicates are dropped. The status line is rewritten on
        every frame change with the same sentence - "Frame 812/108000" differs,
        but "Select a candidate first." does not - and a log that is four
        hundred copies of one sentence has hidden the error above it.
        """
        text = str(text).strip()
        if not text:
            return None
        if kind not in KINDS:
            kind = INFO
        entry = Entry(time.time(), kind, text)
        with self._lock:
            if text == self._last_text:
                return None
            self._last_text = text
            self._entries.append(entry)
            if kind == ERROR:
                self._errors += 1
            elif kind == WARN:
                self._warnings += 1
            if len(self._entries) > self.max_lines:
                # Trim in blocks. Popping one per append on a full buffer is
                # O(n) each time; dropping a tenth at once amortises it.
                del self._entries[: self.max_lines // 10]
            self.revision += 1
        return entry

    def info(self, text: str) -> None:
        self.record(text, INFO)

    def warn(self, text: str) -> None:
        self.record(text, WARN)

    def error(self, text: str) -> None:
        self.record(text, ERROR)

    def job(self, text: str) -> None:
        self.record(text, JOB)

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._errors = 0
            self._warnings = 0
            self._last_text = ""
            self.revision += 1

    # ── reading ──────────────────────────────────────────────────────────
    def entries(self, kinds: Optional[Sequence[str]] = None) -> List[Entry]:
        with self._lock:
            rows = list(self._entries)
        if kinds is None:
            return rows
        wanted = set(kinds)
        return [e for e in rows if e.kind in wanted]

    def text(self, kinds: Optional[Sequence[str]] = None, *, limit: int = 0) -> str:
        rows = self.entries(kinds)
        if limit > 0:
            rows = rows[-limit:]
        return "\n".join(e.format() for e in rows)

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def warnings(self) -> int:
        return self._warnings

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def last(self, kind: Optional[str] = None) -> Optional[Entry]:
        rows = self.entries(None if kind is None else (kind,))
        return rows[-1] if rows else None


# ══════════════════════════════════════════════════════════════════════════
# What the pill says
# ══════════════════════════════════════════════════════════════════════════
def pill_state(running: Sequence[str], log: ActivityLog) -> Tuple[str, str]:
    """``(label, pill kind)`` for the top-bar chip.

    Precedence is *what is happening now*, then *what went wrong*, then idle.
    A job in flight is the more urgent fact even when an earlier one failed:
    the failure is still in the window, and a chip that reports the past while
    the present is moving is a chip that gets ignored.
    """
    if running:
        first = running[0]
        more = f" +{len(running) - 1}" if len(running) > 1 else ""
        return (f"Activity: {first}{more}", "running")
    if log.errors:
        return (f"Activity: {log.errors} error{'s' if log.errors != 1 else ''}", "error")
    if log.warnings:
        return (
            f"Activity: {log.warnings} warning{'s' if log.warnings != 1 else ''}",
            "warn",
        )
    return ("Activity", "idle")


# ══════════════════════════════════════════════════════════════════════════
# Finding the child processes' own logs
# ══════════════════════════════════════════════════════════════════════════
#: Filenames the subsystems write, and what to call them on screen. Both are
#: opened by their writer with ``stdout=<file>`` rather than a pipe, so they
#: are the complete output of the run including the traceback that killed it.
LOG_NAMES: Dict[str, str] = {
    "round.log": "Training round",
    "inference.log": "Finding animals",
    "train.log": "Training",
}


def find_run_logs(root: Optional[Path], *, limit: int = 12) -> List[Tuple[str, Path]]:
    """``[(label, path), ...]`` for the newest run logs under a project.

    Newest first, because the question is almost always about the run that just
    ended. Nothing here reads the files - a project with a year of rounds in it
    would pay for every one of them on every refresh - so the list is cheap
    enough to rebuild whenever the window is opened.
    """
    if root is None:
        return []
    root = Path(root)
    if not root.is_dir():
        return []
    found: List[Tuple[float, str, Path]] = []
    for name, label in LOG_NAMES.items():
        for path in root.glob(f"**/{name}"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            # The run folder is what tells two rounds apart; the filename does
            # not. "Training round - round-003" beats three rows of "round.log".
            where = path.parent.name
            if where in ("stage1", "stage2", "stage3", "stage4"):
                where = f"{path.parent.parent.name}/{where}"
            found.append((mtime, f"{label} - {where}", path))
    found.sort(key=lambda row: row[0], reverse=True)
    return [(label, path) for _mtime, label, path in found[:limit]]


def tail(path: Path, *, lines: int = 400, max_bytes: int = 400_000) -> str:
    """The end of a log file, read without pulling a gigabyte into memory.

    A stalled trainer can write a progress bar for an hour. Seeking to the last
    ``max_bytes`` and keeping the last ``lines`` of that is what makes opening
    this window safe on a run that went wrong in a loop.
    """
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            if size > max_bytes:
                handle.seek(size - max_bytes)
                handle.readline()  # drop the partial line the seek landed in
            raw = handle.read()
    except OSError as exc:
        return f"(could not read {path.name}: {exc})"
    text = raw.decode("utf-8", errors="replace")
    rows = text.splitlines()
    clipped = rows[-lines:]
    note = "" if len(rows) == len(clipped) else f"(showing the last {len(clipped)} lines)\n"
    return note + "\n".join(clipped)


# ══════════════════════════════════════════════════════════════════════════
# The window
# ══════════════════════════════════════════════════════════════════════════
class ActivityWindow:
    """The view over one :class:`ActivityLog`. Safe to build twice.

    ``on_open_folder`` is handed the folder a log lives in, so the window can
    hand off to the file manager without knowing how this platform opens one.
    ``running`` is a callable rather than a stored list, because what is
    running changes without this window being told and a stale "Training..."
    is worse than no line at all.
    """

    WIDTH, HEIGHT = 860, 620

    #: Filter buttons: ``(key, label, kinds shown)``.
    FILTERS: Tuple[Tuple[str, str, Optional[Tuple[str, ...]]], ...] = (
        ("all", "All", None),
        ("problems", "Problems", (WARN, ERROR)),
        ("jobs", "Jobs", (JOB, ERROR)),
    )

    def __init__(
        self,
        log: ActivityLog,
        tag: str = "activity",
        *,
        running: Optional[Callable[[], Sequence[str]]] = None,
        project_root: Optional[Callable[[], Optional[Path]]] = None,
        on_open_folder: Optional[Callable[[Path], None]] = None,
    ) -> None:
        self.log = log
        self.tag = tag
        self.running = running or (lambda: ())
        self.project_root = project_root or (lambda: None)
        self.on_open_folder = on_open_folder
        self._built = False
        self._filter = "all"
        self._logs: List[Tuple[str, Path]] = []
        self._chosen = 0
        self._shown_revision = -1
        self._follow = True

    # ── tiny dpg helpers ─────────────────────────────────────────────────
    def _t(self, suffix: str) -> str:
        return f"{self.tag}_{suffix}"

    @staticmethod
    def _set(tag: str, value) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(tag):
            dpg.set_value(tag, value)

    @staticmethod
    def _get(tag: str, default=None):
        import dearpygui.dearpygui as dpg

        return dpg.get_value(tag) if dpg.does_item_exist(tag) else default

    # ── building ─────────────────────────────────────────────────────────
    def build(self) -> None:
        import dearpygui.dearpygui as dpg

        from . import theme, widgets

        if dpg.does_item_exist(self.tag):
            dpg.delete_item(self.tag)

        with dpg.window(
            tag=self.tag,
            label="Activity",
            show=False,
            no_collapse=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            on_close=lambda *_: self.hide(),
        ):
            # The window is deliberately **not** modal: a log you have to
            # close before you can do the thing it told you to do is a log
            # read once.
            #
            # The default footer reservation, not a tighter one. At the
            # annotator's 900x600 minimum this window is clamped to about
            # 500 px tall and its two panes are taller than that on their
            # own, so the body scrolls - which is only safe while Close is
            # genuinely outside it. 48 px was two short of that and put the
            # button below the bottom edge at the minimum size and nowhere
            # else, which is the one size nobody develops at.
            with widgets.dialog_body(self._t("body")):
                dpg.add_text("Now", color=theme.TEXT_DIM)
                dpg.add_text(
                    "Nothing is running.",
                    tag=self._t("now"),
                    color=theme.TEXT,
                    wrap=self.WIDTH - 60,
                )
                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)

                with dpg.group(horizontal=True):
                    widgets.segmented_control(
                        self._t("filter"),
                        [(key, label) for key, label, _k in self.FILTERS],
                        lambda value: self.set_filter(value),
                        default=self._filter,
                        width=90,
                    )
                    # Per button, not on the group: a dearpygui group is a
                    # layout container and is never hovered, so a tooltip
                    # parented to one is a tooltip that never appears.
                    for index, (_key, _label, _k) in enumerate(self.FILTERS):
                        widgets.attach_tooltip(
                            f'{self._t("filter")}__opt_{index}',
                            (
                                "Everything the app has said this session.",
                                "Only the warnings and the errors.",
                                "Only what the long jobs said about "
                                "themselves - started, finished, failed, "
                                "stopped.",
                            )[index],
                        )
                    dpg.add_spacer(width=12)
                    dpg.add_checkbox(
                        label="Follow",
                        tag=self._t("follow"),
                        default_value=True,
                        callback=lambda *_: self._on_follow(),
                    )
                    widgets.attach_tooltip(
                        self._t("follow"),
                        "Keep the newest line in view. Turn it off to read "
                        "back through the session without being dragged to "
                        "the bottom.",
                    )
                    dpg.add_spacer(width=12)
                    widgets.secondary_button(
                        "Copy", lambda *_: self.copy(), height=26, width=90,
                        tag=self._t("copy"),
                    )
                    widgets.attach_tooltip(
                        self._t("copy"),
                        "Put what is shown here on the clipboard, ready to "
                        "paste into a bug report.",
                    )
                    widgets.secondary_button(
                        "Clear", lambda *_: self.clear(), height=26, width=90,
                        tag=self._t("clear"),
                    )
                    widgets.attach_tooltip(
                        self._t("clear"),
                        "Empty this session's messages. The run logs on disk "
                        "are untouched.",
                    )

                widgets.gap(1)
                dpg.add_input_text(
                    tag=self._t("messages"),
                    multiline=True,
                    readonly=True,
                    default_value="",
                    width=-1,
                    height=250,
                    tracked=True,
                    track_offset=1.0,
                )

                widgets.gap(1)
                dpg.add_separator()
                widgets.gap(1)
                with dpg.group(horizontal=True):
                    dpg.add_text("Run log", color=theme.TEXT_DIM)
                    dpg.add_combo(
                        [],
                        tag=self._t("logs"),
                        default_value="",
                        width=380,
                        callback=lambda *_: self._on_log_chosen(),
                    )
                    widgets.secondary_button(
                        "Reload", lambda *_: self.refresh_run_log(), height=26,
                        width=90, tag=self._t("reload"),
                    )
                    widgets.secondary_button(
                        "Open folder", lambda *_: self.open_folder(), height=26,
                        width=120, tag=self._t("open"),
                    )
                widgets.attach_tooltip(
                    self._t("logs"),
                    "The complete output of a child process - the trainer or "
                    "the detector - exactly as it was written to the run "
                    "folder. This is where a crash leaves its traceback.",
                )
                dpg.add_input_text(
                    tag=self._t("runlog"),
                    multiline=True,
                    readonly=True,
                    default_value="No run logs in this project yet.",
                    width=-1,
                    height=180,
                )

            widgets.dialog_footer()
            with dpg.group(horizontal=True):
                widgets.secondary_button(
                    "Close", lambda *_: self.hide(), height=32, width=120,
                    tag=self._t("close"),
                )

        self._built = True

    # ── the three things it shows ────────────────────────────────────────
    def set_filter(self, key: str) -> None:
        self._filter = key if any(k == key for k, _l, _s in self.FILTERS) else "all"
        self._shown_revision = -1
        self.refresh_messages()

    def _kinds(self) -> Optional[Tuple[str, ...]]:
        for key, _label, kinds in self.FILTERS:
            if key == self._filter:
                return kinds
        return None

    def _on_follow(self) -> None:
        self._follow = bool(self._get(self._t("follow"), True))
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(self._t("messages")):
            dpg.configure_item(self._t("messages"), tracked=self._follow)

    def refresh_messages(self, *, force: bool = False) -> None:
        """Redraw the message pane if the log has moved on since last time."""
        if not self._built:
            return
        if not force and self.log.revision == self._shown_revision:
            return
        self._shown_revision = self.log.revision
        body = self.log.text(self._kinds())
        self._set(self._t("messages"), body or "Nothing to show under this filter.")

    def refresh_now(self) -> None:
        """The *Now* line: what is running, in the words the dialogs use."""
        if not self._built:
            return
        try:
            rows = list(self.running() or ())
        except Exception:  # noqa: BLE001 - a status read must not break the window
            rows = []
        self._set(self._t("now"), "\n".join(rows) if rows else "Nothing is running.")

    def refresh_run_logs(self) -> None:
        """Rediscover the run logs and keep the chosen one selected if it lives."""
        import dearpygui.dearpygui as dpg

        if not self._built:
            return
        previous = self._logs[self._chosen][1] if 0 <= self._chosen < len(self._logs) else None
        self._logs = find_run_logs(self.project_root())
        labels = [label for label, _path in self._logs]
        self._chosen = 0
        for index, (_label, path) in enumerate(self._logs):
            if previous is not None and path == previous:
                self._chosen = index
                break
        if dpg.does_item_exist(self._t("logs")):
            dpg.configure_item(self._t("logs"), items=labels)
            dpg.set_value(self._t("logs"), labels[self._chosen] if labels else "")
        self.refresh_run_log()

    def _on_log_chosen(self) -> None:
        label = self._get(self._t("logs"), "")
        for index, (name, _path) in enumerate(self._logs):
            if name == label:
                self._chosen = index
                break
        self.refresh_run_log()

    def refresh_run_log(self) -> None:
        if not self._built:
            return
        if not (0 <= self._chosen < len(self._logs)):
            self._set(self._t("runlog"), "No run logs in this project yet.")
            return
        _label, path = self._logs[self._chosen]
        self._set(self._t("runlog"), tail(path))

    def refresh(self) -> None:
        """Everything. Cheap enough for the render loop to call while open."""
        self.refresh_now()
        self.refresh_messages()

    # ── actions ──────────────────────────────────────────────────────────
    def copy(self) -> None:
        import dearpygui.dearpygui as dpg

        from . import widgets

        text = self.log.text(self._kinds())
        if 0 <= self._chosen < len(self._logs):
            label, path = self._logs[self._chosen]
            text += f"\n\n--- {label} ({path}) ---\n" + tail(path, lines=200)
        try:
            dpg.set_clipboard_text(text)
        except Exception:  # noqa: BLE001 - no clipboard on some headless setups
            widgets.toast("This build has no clipboard. Select the text instead.", "warn")
            return
        widgets.toast("Copied the activity log.", "ok")

    def clear(self) -> None:
        self.log.clear()
        self.refresh_messages(force=True)

    def open_folder(self) -> None:
        from . import widgets

        if not (0 <= self._chosen < len(self._logs)):
            widgets.toast("No run log is selected.", "warn")
            return
        _label, path = self._logs[self._chosen]
        if self.on_open_folder is not None:
            self.on_open_folder(path.parent)

    # ── window ───────────────────────────────────────────────────────────
    def show(self) -> None:
        import dearpygui.dearpygui as dpg

        from . import widgets

        if not self._built:
            self.build()
        self.refresh_now()
        self.refresh_messages(force=True)
        self.refresh_run_logs()
        widgets.place_dialog(self.tag, self.WIDTH, self.HEIGHT)
        dpg.configure_item(self.tag, show=True)
        dpg.focus_item(self.tag)

    def hide(self) -> None:
        import dearpygui.dearpygui as dpg

        if dpg.does_item_exist(self.tag):
            dpg.configure_item(self.tag, show=False)

    def toggle(self) -> None:
        if self.visible:
            self.hide()
        else:
            self.show()

    @property
    def visible(self) -> bool:
        import dearpygui.dearpygui as dpg

        return dpg.does_item_exist(self.tag) and dpg.is_item_shown(self.tag)


# ══════════════════════════════════════════════════════════════════════════
# The annotator's side of it
# ══════════════════════════════════════════════════════════════════════════
class ActivityMixin:
    """Mixed into ``AnnotationApp`` beside :class:`~.train_operations.TrainRoundMixin`.

    The pill is the button and the one-line summary at once, which is the same
    bargain the save-state and model chips make beside it in the top bar: a
    fact you can read without acting, and the way to the thing it reports on.

    Everything here reads the app's state through ``getattr`` with a default.
    This mixin is on the class before any of those attributes exist - the log
    is built first precisely so a failure during construction is recorded -
    and a window that cannot report because the thing it reports on has not
    been made yet is a window that hides the crash it exists to show.
    """

    def note_activity(self, text: str, *, failed: bool = False) -> None:
        """Record what a long job did, without saying it on the status line.

        The status line is for the sentence the user is owed *now*; this is for
        the sentence they will want in an hour, when the round that failed at
        11:40 is the reason the model is the one from Tuesday. A job's start,
        end, failure and stop go here; its per-epoch chatter stays in the run
        log on disk, which the Activity window shows beside these.
        """
        log = getattr(self, "activity_log", None)
        if log is None:
            return
        log.record(text, ERROR if failed else JOB)
        self._refresh_activity_pill()

    def running_jobs(self) -> List[str]:
        """One line per long job in flight, in the words its own dialog uses.

        Read rather than stored. A round can end while this window is open and
        a stored list would still say "Training the detector" - the exact lie
        a progress display exists to prevent.
        """
        rows: List[str] = []
        state = getattr(self, "_round", None)
        if state is not None:
            where = "Training the detector"
            if state.proc is None:
                where = "Collecting the frames"
            elif state.last_epoch >= 0:
                where = (
                    f"Training the detector - pass {state.last_epoch + 1} "
                    f"of {state.plan.epochs}"
                )
            rows.append(f"Round {state.number}: {where}")
        if getattr(self, "_smart_running", False):
            rows.append("Finding animals with the model")
        thread = getattr(self, "_batch_thread", None)
        if thread is not None and thread.is_alive() and not getattr(self, "_smart_running", False):
            rows.append("Running over the project")
        if getattr(self, "_copilot_dirty", False):
            rows.append("Re-checking the candidates")
        return rows

    def _refresh_activity_pill(self) -> None:
        import dearpygui.dearpygui as dpg

        from . import widgets

        if not dpg.does_item_exist("top_activity"):
            return
        label, kind = pill_state(self.running_jobs(), self.activity_log)
        widgets.status_pill(
            "top_activity", label, kind, callback=lambda *_: self.toggle_activity()
        )

    def toggle_activity(self) -> None:
        """Open the Activity window, or close it if it is already up."""
        self.activity.toggle()

    def show_activity(self) -> None:
        self.activity.show()

    def _tick_activity(self) -> None:
        """Keep the pill and the open window current, at human speed.

        Twice a second. The things it reports on - an epoch, a phase, an error
        - change on the order of seconds, and rebuilding a four-thousand-line
        text box at sixty hertz is the kind of cost that makes a log window
        something people close.
        """
        now = time.monotonic()
        if now - getattr(self, "_activity_polled", 0.0) < 0.5:
            return
        self._activity_polled = now
        self._refresh_activity_pill()
        if self.activity.visible:
            self.activity.refresh()
