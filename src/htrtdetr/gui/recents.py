"""
recents.py — Persistent recent-file store for the YOAKE GUI.

Stores per-field recent paths at ``~/.yoake/gui_recents.json`` so users can
quickly reuse files across sessions without navigating file dialogs every
time. Failures (missing dir, invalid JSON) fall back to an empty store.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import List, Optional

MAX_ITEMS = 10
_STORE = Path.home() / ".yoake" / "gui_recents.json"
_LOCK = Lock()


def _load() -> dict:
    try:
        return json.loads(_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    _STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get(field: str) -> List[str]:
    """Return recent paths for ``field`` in most-recent-first order."""
    with _LOCK:
        data = _load()
    items = data.get(field, [])
    return [p for p in items if isinstance(p, str) and p]


def add(field: str, path: str) -> None:
    """Prepend ``path`` to the field's history (dedup, capped at MAX_ITEMS)."""
    if not path:
        return
    with _LOCK:
        data = _load()
        items = [p for p in data.get(field, []) if p and p != path]
        items.insert(0, path)
        data[field] = items[:MAX_ITEMS]
        try:
            _save(data)
        except OSError:
            pass


#: The field that names the project folder every page works inside. It is a
#: recents list like any other, so "switch back to last month's study" is the
#: same gesture as any other recent path - but the newest entry has a second
#: job: it is what the Train and Analyze pages write their output into.
PROJECT_FIELD = "project_dir"


def active_project() -> Optional[Path]:
    """The project folder the GUI is currently working in, if it still exists.

    Returns ``None`` rather than a stale path: a project folder that has been
    moved or deleted must not silently become the destination of a training
    run. The user re-opens it, and the pages follow.
    """
    for candidate in get(PROJECT_FIELD):
        folder = Path(candidate)
        if (folder / "project.yoake.json").is_file():
            return folder
    return None


def set_active_project(path) -> None:
    """Record ``path`` (a project folder, or any file inside it) as active."""
    folder = Path(path)
    if folder.is_file():
        folder = folder.parent
    add(PROJECT_FIELD, str(folder))


def clear(field: str) -> None:
    with _LOCK:
        data = _load()
        if field in data:
            del data[field]
            try:
                _save(data)
            except OSError:
                pass
