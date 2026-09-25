"""The one door from the annotator to the production training run.

The active-learning loop - train a detector on what is labelled so far, run it
back over the videos, label what it got wrong - happens inside the annotator
(``gui/train_operations.py``, ``gui/train_round.py``). This handoff is the
*other* job: the long run over finished labels, every stage, done once. It
opens the workflow window and hands it the project.

It used to be two doors, ``--page train`` and ``--page analyze``, and they were
how the annotator reached training at all. That made the loop a thing you left
the labelling window to continue, which in practice meant it was not continued.

Nothing here opens a window. The command is built by a pure function, the hub
parses it with a pure function, and the annotator's side is a mixin method
driven on a stub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from htrtdetr.gui import commands  # noqa: E402
from htrtdetr.gui.workspace_operations import WorkspaceOperations  # noqa: E402


# ── the command ──────────────────────────────────────────────────────────

def test_the_command_names_the_page_and_the_project():
    assert commands.workflow_command("train", "C:/studies/flies") == [
        "-m", "htrtdetr.gui", "--page", "train", "--project", "C:/studies/flies",
    ]


def test_an_unknown_page_is_dropped_rather_than_passed_on():
    """A window that opens on its first page beats one that refuses to open."""
    assert commands.workflow_command("teleport", "") == ["-m", "htrtdetr.gui"]


def test_every_page_the_annotator_offers_is_a_page_the_hub_knows():
    hub = pytest.importorskip("htrtdetr.gui.hub")
    for page in commands.WORKFLOW_PAGES:
        assert page in hub._PAGE_KEYS, page
    for page in WorkspaceOperations.WORKFLOW_PAGES:
        assert page in commands.WORKFLOW_PAGES, page


def test_the_annotator_offers_one_door_and_it_is_not_train():
    """Train and Analyze are operations in the labelling window now. A second
    door to them, in another window, is how the loop got abandoned."""
    assert list(WorkspaceOperations.WORKFLOW_PAGES) == ["workflow"]


# ── the hub's side ───────────────────────────────────────────────────────

def test_the_hub_reads_the_page_and_the_project():
    hub = pytest.importorskip("htrtdetr.gui.hub")
    page, project = hub.parse_argv(["--page", "analyze", "--project", "C:/x"])
    assert page == "analyze"
    assert project == Path("C:/x")


def test_the_hub_ignores_what_it_does_not_understand():
    """Someone types this command by hand. It must still open."""
    hub = pytest.importorskip("htrtdetr.gui.hub")
    assert hub.parse_argv([]) == ("", None)
    assert hub.parse_argv(["--page"]) == ("", None)
    assert hub.parse_argv(["--page", "nowhere", "--colour", "red"]) == ("", None)


# ── the annotator's side ─────────────────────────────────────────────────

class _Stub(WorkspaceOperations):
    """Just enough annotator to press the button."""

    def __init__(self, project_dir, dirty=False):
        self.project_dir = project_dir
        self._dirty = dirty
        self.saves = 0
        self.messages = []
        self.settings_tabs = []

    def save_annotation(self):
        self.saves += 1
        self._dirty = False

    def _status(self, message):
        self.messages.append(message)

    def open_settings(self, tab=""):
        self.settings_tabs.append(tab)


def _spawns(monkeypatch):
    calls = []
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: calls.append((cmd, kw)))
    return calls


def test_a_project_without_a_folder_is_told_where_to_get_one(monkeypatch):
    """The workflow window works on a project folder, so the folder is the

    precondition - and a button that silently does nothing is the failure this
    replaces."""
    calls = _spawns(monkeypatch)
    app = _Stub(None)
    app.open_workflow_page()
    assert not calls
    assert app.settings_tabs == ["project"]
    assert "Settings > Project" in app.messages[-1]


def test_unsaved_work_is_written_before_the_workflow_window_reads_it(monkeypatch):
    calls = _spawns(monkeypatch)
    app = _Stub(Path("C:/studies/flies"), dirty=True)
    app.open_workflow_page()
    assert app.saves == 1
    (cmd, _kwargs), = calls
    assert cmd[0] == sys.executable
    assert cmd[-4:-1] == ["--page", "workflow", "--project"]
    assert Path(cmd[-1]) == Path("C:/studies/flies")


def test_a_clean_project_is_not_saved_again(monkeypatch):
    calls = _spawns(monkeypatch)
    app = _Stub(Path("C:/studies/flies"))
    app.open_workflow_page()
    assert app.saves == 0
    (cmd, kwargs), = calls
    assert "workflow" in cmd
    assert kwargs["cwd"]
    assert "PYTHONPATH" in kwargs["env"]


def test_the_annotator_says_it_is_staying_open(monkeypatch):
    """The window that appears is a second window, not a replacement, and a

    user who thinks their labelling session just ended will not press it."""
    _spawns(monkeypatch)
    app = _Stub(Path("C:/studies/flies"))
    app.open_workflow_page()
    assert "stays open" in app.messages[-1]


def test_a_launch_that_fails_says_so_instead_of_pretending(monkeypatch):
    import subprocess

    def boom(cmd, **kwargs):
        raise OSError("no python here")

    monkeypatch.setattr(subprocess, "Popen", boom)
    app = _Stub(Path("C:/studies/flies"))
    app.open_workflow_page()
    assert "no python here" in app.messages[-1]
