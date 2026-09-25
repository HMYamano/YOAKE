"""
Unit tests for :mod:`htrtdetr.gui.activity` — the log behind the Activity window.

Everything here is the **dearpygui-free** half of that module: the ring buffer,
the classifier that decides what a status-line sentence is, the pill's wording,
and the two file helpers. That split is the reason the module is shaped the way
it is, so it is worth asserting: none of these tests imports dearpygui, and the
import at the top would fail if :class:`ActivityLog` ever grew a dpg dependency.

The window itself is covered by ``tests/test_gui_dialogs.py``, which opens a
viewport and checks that its buttons are reachable.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from htrtdetr.gui import activity  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════
# Purity
# ══════════════════════════════════════════════════════════════════════════

def test_the_log_half_does_not_drag_in_dearpygui():
    """Importing the module must not import the toolkit.

    The annotator records into this log before ``create_context`` has been
    called, and the log is the one thing that has to survive a failure during
    construction. A module-level ``import dearpygui`` would make the log
    unavailable in exactly the case it exists for.
    """
    import htrtdetr.gui.activity as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import dearpygui", "from dearpygui")):
            assert line.startswith("    "), (
                "dearpygui must only be imported inside a function: " + line
            )


# ══════════════════════════════════════════════════════════════════════════
# The ring buffer
# ══════════════════════════════════════════════════════════════════════════

def test_entries_keep_their_kind_and_their_order():
    log = activity.ActivityLog()
    log.info("one")
    log.warn("two")
    log.error("three")
    log.job("four")
    assert [e.text for e in log.entries()] == ["one", "two", "three", "four"]
    assert [e.kind for e in log.entries()] == ["info", "warn", "error", "job"]


def test_consecutive_duplicates_are_dropped():
    """The status line repeats itself; the log must not.

    ``Select a candidate first.`` is written on every failed press, and four
    hundred copies of it is how the error above them gets lost.
    """
    log = activity.ActivityLog()
    for _ in range(50):
        log.info("Select a candidate first.")
    assert len(log) == 1
    log.info("something else")
    log.info("Select a candidate first.")
    assert len(log) == 3, "a repeat that is not consecutive is a real event"


def test_blank_messages_are_not_recorded():
    log = activity.ActivityLog()
    log.info("")
    log.info("   ")
    assert len(log) == 0


def test_the_buffer_is_bounded():
    log = activity.ActivityLog(max_lines=100)
    for i in range(500):
        log.info(f"line {i}")
    assert len(log) <= 100
    assert log.entries()[-1].text == "line 499", "the newest line must survive"


def test_counts_survive_the_trim():
    """An error scrolled out of the buffer still happened.

    The pill's count is how a user learns to open the window at all, and a
    count that silently falls back to zero once the session is long enough
    is worse than no count.
    """
    log = activity.ActivityLog(max_lines=20)
    log.error("the one that matters")
    for i in range(200):
        log.info(f"chatter {i}")
    assert log.errors == 1
    assert all(e.kind != "error" for e in log.entries()), "it did scroll out"


def test_clear_resets_the_counts_too():
    log = activity.ActivityLog()
    log.error("bad")
    log.warn("iffy")
    log.clear()
    assert (len(log), log.errors, log.warnings) == (0, 0, 0)


def test_the_revision_moves_only_when_something_was_recorded():
    log = activity.ActivityLog()
    log.info("first")
    seen = log.revision
    log.info("first")  # duplicate
    assert log.revision == seen
    log.info("second")
    assert log.revision > seen


def test_recording_from_several_threads_loses_nothing():
    """Workers are the ones whose failures matter most, so they may record."""
    log = activity.ActivityLog(max_lines=10_000)

    def worker(base: int) -> None:
        for i in range(200):
            log.record(f"w{base}-{i}", activity.INFO)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(log) == 8 * 200


def test_filtering_returns_only_the_kinds_asked_for():
    log = activity.ActivityLog()
    log.info("chatter")
    log.job("Round 1 started")
    log.error("Round 1 failed")
    rows = log.entries((activity.JOB, activity.ERROR))
    assert [e.text for e in rows] == ["Round 1 started", "Round 1 failed"]


def test_the_formatted_line_is_ascii_marked_and_timestamped():
    log = activity.ActivityLog()
    log.error("it broke")
    line = log.entries()[0].format()
    assert "! it broke" in line
    assert line[:2].isdigit() and line[2] == ":", f"no clock in {line!r}"
    line.encode("cp932"), "the log must survive a Japanese console"


# ══════════════════════════════════════════════════════════════════════════
# Classifying a status-line sentence
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("message", [
    "Could not read those predictions: no such file",
    "Training stopped with an error (code 1)",
    "Inference exited 1: CUDA out of memory",
    "The video file was not found.",
])
def test_failures_are_classified_as_errors(message):
    assert activity.classify(message) == activity.ERROR


@pytest.mark.parametrize("message", [
    "Nothing to undo.",
    "Select a candidate first, then accept its tracklet.",
    "No model yet, so there is nothing to find animals with.",
])
def test_refusals_are_classified_as_warnings(message):
    assert activity.classify(message) == activity.WARN


@pytest.mark.parametrize("message", [
    "Frame 812/108000  |  Annotated: 312/108000 (0%)",
    "Accepted 3 candidates on this frame.",
    "Loaded: clip07.mp4  (108000 frames, 30.0 fps, 1024x1024)",
])
def test_ordinary_reports_are_info(message):
    assert activity.classify(message) == activity.INFO


def test_an_error_with_a_warning_clause_in_it_is_an_error():
    """Errors win. "could not save: nothing selected" is a failure."""
    assert activity.classify("Could not save: nothing is selected") == activity.ERROR


# ══════════════════════════════════════════════════════════════════════════
# The pill
# ══════════════════════════════════════════════════════════════════════════

def test_the_pill_is_idle_when_nothing_has_happened():
    label, kind = activity.pill_state([], activity.ActivityLog())
    assert (label, kind) == ("Activity", "idle")


def test_a_running_job_outranks_a_past_error():
    """The present beats the past: the failure is still in the window."""
    log = activity.ActivityLog()
    log.error("an earlier round failed")
    label, kind = activity.pill_state(["Round 2: Training the detector"], log)
    assert kind == "running"
    assert "Round 2" in label


def test_several_jobs_are_counted_rather_than_listed():
    label, kind = activity.pill_state(["A", "B", "C"], activity.ActivityLog())
    assert label == "Activity: A +2"
    assert kind == "running"


def test_errors_outrank_warnings_when_nothing_is_running():
    log = activity.ActivityLog()
    log.warn("iffy")
    log.error("bad")
    label, kind = activity.pill_state([], log)
    assert (label, kind) == ("Activity: 1 error", "error")


def test_the_error_count_is_pluralised():
    log = activity.ActivityLog()
    log.error("one")
    log.error("two")
    assert activity.pill_state([], log)[0] == "Activity: 2 errors"


# ══════════════════════════════════════════════════════════════════════════
# The run logs on disk
# ══════════════════════════════════════════════════════════════════════════

def test_run_logs_are_found_newest_first(tmp_path):
    old = tmp_path / "runs" / "rounds" / "round-001" / "stage1"
    new = tmp_path / "runs" / "rounds" / "round-002" / "stage1"
    for folder in (old, new):
        folder.mkdir(parents=True)
        (folder / "round.log").write_text("hello", encoding="utf-8")
    # Make the ordering unambiguous rather than trusting filesystem timing.
    import os

    os.utime(old / "round.log", (1_000_000, 1_000_000))
    os.utime(new / "round.log", (2_000_000, 2_000_000))

    found = activity.find_run_logs(tmp_path)
    assert [p for _label, p in found] == [new / "round.log", old / "round.log"]


def test_the_label_says_which_run_not_which_filename(tmp_path):
    """Three rows all called "round.log" is three rows that say nothing."""
    folder = tmp_path / "runs" / "rounds" / "round-007" / "stage1"
    folder.mkdir(parents=True)
    (folder / "round.log").write_text("x", encoding="utf-8")
    (label, _path), = activity.find_run_logs(tmp_path)
    assert "round-007" in label
    assert label.startswith("Training round")


def test_an_inference_log_is_found_and_named_for_the_job(tmp_path):
    folder = tmp_path / "analysis" / "predict" / "clip07"
    folder.mkdir(parents=True)
    (folder / "inference.log").write_text("x", encoding="utf-8")
    (label, _path), = activity.find_run_logs(tmp_path)
    assert label.startswith("Finding animals")


def test_a_missing_or_absent_project_finds_nothing(tmp_path):
    assert activity.find_run_logs(None) == []
    assert activity.find_run_logs(tmp_path / "nope") == []


def test_tail_returns_the_end_not_the_beginning(tmp_path):
    path = tmp_path / "round.log"
    path.write_text("\n".join(f"line {i}" for i in range(5000)), encoding="utf-8")
    text = activity.tail(path, lines=10)
    assert "line 4999" in text
    assert "line 0\n" not in text


def test_tail_of_a_huge_log_does_not_read_all_of_it(tmp_path):
    """A stalled trainer can write a progress bar for an hour."""
    path = tmp_path / "round.log"
    path.write_bytes(b"x" * 2_000_000 + b"\nthe end\n")
    text = activity.tail(path, max_bytes=50_000)
    assert "the end" in text
    assert len(text) < 60_000


def test_tail_of_an_unreadable_file_explains_itself(tmp_path):
    text = activity.tail(tmp_path / "gone.log")
    assert "could not read" in text.lower()


# ══════════════════════════════════════════════════════════════════════════
# The mixin's reading of what is running
# ══════════════════════════════════════════════════════════════════════════

class _Bare(activity.ActivityMixin):
    """Just the mixin, on an object with none of the app's attributes.

    The mixin is on ``AnnotationApp`` before ``__init__`` has made any of what
    it reads, and it must survive that: a crash during construction is the one
    the log exists to record.
    """


def test_running_jobs_on_a_bare_object_is_empty_rather_than_an_error():
    assert _Bare().running_jobs() == []


def test_note_activity_without_a_log_is_a_no_op():
    _Bare().note_activity("nobody is listening")  # must not raise


def test_a_round_in_flight_is_reported_with_its_pass_number():
    app = _Bare()
    app._round = type("S", (), {
        "proc": object(), "last_epoch": 13, "number": 3,
        "plan": type("P", (), {"epochs": 96})(),
    })()
    assert app.running_jobs() == ["Round 3: Training the detector - pass 14 of 96"]


def test_a_round_still_collecting_frames_says_so():
    app = _Bare()
    app._round = type("S", (), {
        "proc": None, "last_epoch": -1, "number": 1,
        "plan": type("P", (), {"epochs": 40})(),
    })()
    assert app.running_jobs() == ["Round 1: Collecting the frames"]


def test_a_pending_copilot_rebuild_is_something_the_user_can_see():
    """The deferral must be visible, or it reads as the window ignoring a click."""
    app = _Bare()
    app._copilot_dirty = True
    assert app.running_jobs() == ["Re-checking the candidates"]
