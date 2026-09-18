"""``yoake project`` — create, inspect and repair a project folder, headless.

The GUI is the normal way in, but three things need to work without it: a
project created on a compute node, a check that a copied folder still resolves
its videos, and the repair when it does not. They are also the smallest surface
on which the portability rules can be tested, which is why this module exists
rather than the same logic living inside a dearpygui callback.

Nothing here imports torch or dearpygui.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

USAGE = """\
yoake project <action> [key=value ...]

  new      dir=<parent> name=<project name> [videos=<folder>] [recursive=true]
           [copy=true]
           Create <parent>/<name>/ with the full folder layout and a manifest.
           copy=true also copies the videos into the project, which makes the
           folder self-contained.

  info     path=<project folder or project.yoake.json>
           Print the layout, the videos and whether each one resolves.

  collect  path=<project>
           Copy every referenced video into <project>/videos/.

  relink   path=<project> from=<folder>
           Point every missing video at a same-named file under <folder>.
           The folder is remembered for later opens.
"""


def _args(argv: Sequence[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for item in argv:
        if "=" in item:
            key, value = item.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _flag(values: Dict[str, str], key: str) -> bool:
    return str(values.get(key, "")).strip().lower() in {"1", "true", "yes", "on"}


def _print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:  # a cp932 console must not fail a save report
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace") + b"\n")


def _open(values: Dict[str, str]):
    from .annotation.project import ProjectSession

    raw = values.get("path", "")
    if not raw:
        raise SystemExit("project: path=<project folder> is required.")
    return ProjectSession.open(Path(raw))


def _new(values: Dict[str, str]) -> int:
    from .annotation.project import ProjectSession, scan_folder

    parent = values.get("dir", "")
    name = values.get("name", "")
    if not parent or not name:
        raise SystemExit("project new: dir=<parent folder> and name=<name> are required.")
    project = ProjectSession.create(Path(parent), name)
    layout = project.layout
    added = 0
    source = values.get("videos", "")
    if source:
        for path in scan_folder(Path(source), recursive=_flag(values, "recursive")):
            project.add_video(path)
            added += 1
    if _flag(values, "copy"):
        imported, failures = project.import_media()
        _print(f"Copied {imported} video(s) into {layout.videos}")
        for failure in failures:
            _print(f"  could not copy {failure}")
    project.save(layout.project_file)
    _print(f"Created {layout.root}")
    _print(f"  {added} video(s) registered")
    return 0


def _info(values: Dict[str, str]) -> int:
    from .annotation.layout import FOLDERS

    project = _open(values)
    layout = project.layout
    _print(f"Project:   {project.name}")
    _print(f"Folder:    {layout.root}")
    _print(f"Manifest:  {layout.project_file.name}")
    _print(f"Dataset:   {layout.annotations.name}")
    for _attribute, folder, description in FOLDERS:
        path = layout.root / folder
        mark = "ok     " if path.is_dir() else "missing"
        _print(f"  {folder + '/':<12} {mark}  {description}")
    if project.relocated:
        _print(f"Relocated by file name on this open: {', '.join(project.relocated)}")
    _print(f"Videos:    {len(project.videos)}")
    missing = 0
    for video in project.videos.values():
        state = "missing" if video.missing else "ok"
        missing += video.missing
        frames = len(video.annotations)
        _print(f"  [{state:>7}] {video.video_id:<24} {frames:>5} labelled frames  {video.path}")
    if missing:
        _print(f"\n{missing} video(s) not found. Try: yoake project relink path=... from=<folder>")
    return 1 if missing else 0


def _collect(values: Dict[str, str]) -> int:
    project = _open(values)
    imported, failures = project.import_media()
    project.save(project.layout.project_file)
    _print(f"Copied {imported} video(s) into {project.layout.videos}")
    for failure in failures:
        _print(f"  could not copy {failure}")
    return 1 if failures else 0


def _relink(values: Dict[str, str]) -> int:
    folder = values.get("from", "")
    if not folder:
        raise SystemExit("project relink: from=<folder to search> is required.")
    project = _open(values)
    relinked = project.relink_missing(Path(folder))
    project.save(project.layout.project_file)
    _print(f"Relinked {len(relinked)} video(s): {', '.join(relinked) or '-'}")
    still = [v.video_id for v in project.videos.values() if v.missing]
    if still:
        _print(f"Still missing: {', '.join(still)}")
    return 1 if still else 0


_ACTIONS: Dict[str, Any] = {
    "new": _new,
    "info": _info,
    "collect": _collect,
    "relink": _relink,
}


def run_project(argv: List[str]) -> None:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        _print(USAGE)
        return
    action = argv[0]
    handler = _ACTIONS.get(action)
    if handler is None:
        _print(f"yoake project: unknown action '{action}'\n")
        _print(USAGE)
        raise SystemExit(1)
    code = handler(_args(argv[1:]))
    if code:
        raise SystemExit(code)
