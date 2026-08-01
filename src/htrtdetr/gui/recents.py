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
from typing import List

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


def clear(field: str) -> None:
    with _LOCK:
        data = _load()
        if field in data:
            del data[field]
            try:
                _save(data)
            except OSError:
                pass
