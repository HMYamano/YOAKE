"""The project folder: one directory that holds a study, and survives a move.

Before this module a YOAKE project was a *file* — ``annotations.json`` — and
everything else it produced landed somewhere else: training runs under the
source checkout's ``runs/``, figures under ``outputs/``, caches wherever the
caller happened to point them. Archiving a study meant remembering four
locations, and sending one to a colleague meant reconstructing them.

A project is now a **folder** with a fixed shape::

    <project>/
      project.yoake.json    the manifest: identity, videos, review state
      annotations.json      the dataset, in the training format
      sampling_log.json     which frames were chosen for labelling, and why
      README.txt            this layout, written into the folder itself
      videos/               captures imported into the project (optional)
      datasets/             exported frames and train/val/test splits
      runs/                 training runs: runs/train/stage1/weights/best.pth
      analysis/             predictions.json, overlay videos, figures
      .yoake/               snapshots the workspace owns (internal)
      .yoake_cache/         anything recomputable (internal, safe to delete)

Why a fixed shape rather than four configurable paths: a path the user has to
supply is a path they can supply *differently* in the next session, and the
result is a study whose halves no longer know about each other. The folders
above are derived from one root, so the only thing anyone has to remember is
where the project is.

Portability
-----------

**Every path written into the project is relative to the project root**, so
moving, copying or zipping the folder keeps it working, on this machine or
another one. That is only half the problem: a capture usually lives outside the
project (a NAS share, an acquisition drive), and a relative path to it breaks
the moment the project moves on its own.

:func:`resolve_media` therefore tries three things in a fixed order, and the
order is the design:

1. **Relative to the project root.** This wins first because it is the only
   answer that is right when the whole tree was copied somewhere else — a
   stale absolute path can still resolve on the old machine and quietly open
   *the wrong file*.
2. **The recorded absolute path.** Correct when the project folder alone moved
   and the capture stayed put.
3. **A search by file name**, under the project and under the folders the user
   has located files in before. Correct when both ends moved independently,
   which is what actually happens when a study is restored from a backup.

Nothing here imports ``dearpygui``, ``torch``, ``cv2`` or ``numpy``: the rules
above have to be testable from both of this project's interpreters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .cache import CACHE_DIRNAME

__all__ = [
    "PROJECT_FILE",
    "ANNOTATIONS_FILE",
    "SAMPLING_LOG_FILE",
    "README_FILE",
    "INTERNAL_DIRNAME",
    "FOLDERS",
    "ProjectLayout",
    "is_project_root",
    "find_project",
    "portable_path",
    "anchor_path",
    "resolve_media",
    "locate_media",
    "project_folder_name",
]

#: The manifest. Its presence is what makes a directory a project.
PROJECT_FILE = "project.yoake.json"
#: The dataset in the training format, always beside the manifest.
ANNOTATIONS_FILE = "annotations.json"
SAMPLING_LOG_FILE = "sampling_log.json"
README_FILE = "README.txt"
#: Workspace-owned internals. Snapshots live here; users never edit them.
INTERNAL_DIRNAME = ".yoake"

#: ``(attribute, folder name, what goes in it)`` in the order a user meets them.
#: The descriptions are the text written into the project's own README, so the
#: folder can explain itself without this source file.
FOLDERS: tuple[tuple[str, str, str], ...] = (
    (
        "videos",
        "videos",
        "Captures imported into the project. Videos left outside it are referenced in place.",
    ),
    (
        "datasets",
        "datasets",
        "Training data: exported frames and the splits built from annotations.json.",
    ),
    (
        "runs",
        "runs",
        "Training and evaluation runs, each with its own weights, results.csv and metrics.",
    ),
    (
        "analysis",
        "analysis",
        "Analysis results: predictions.json, overlay videos and figures.",
    ),
)

_README = """\
YOAKE project: {name}
{underline}

This folder is a complete YOAKE study. Move it, copy it, or hand it to someone
else - every path inside it is relative to this folder, so it keeps working.
このフォルダごと移動・コピーしても動作します (内部のパスはすべて相対パス)。

Open it with:  yoake gui   ->  1. Label  ->  Open project folder...
           or:  annotate --project "{project_file}"   (run from inside this folder)

  {annotations:<20} The dataset, in the training format.
  {manifest:<20} The workspace: videos, review state, settings.
{entries}
Internal, safe to ignore:
  {internal:<20} Snapshots of the workspace. Do not edit by hand.
  {cache:<20} Anything that can be recomputed. Deleting it only costs time.

Captures stored outside this folder are remembered by their absolute path as
well, so the project still finds them after it moves. If both ends moved, use
"Locate file" in Project Videos once: the folder you pick is remembered for the
rest of the project.
"""


@dataclass(frozen=True)
class ProjectLayout:
    """Every path a project owns, derived from its root.

    Frozen on purpose: a layout is an answer about one folder, not a mutable
    setting. To point somewhere else, make another one.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    # ── files ────────────────────────────────────────────────────────────
    @property
    def project_file(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def annotations(self) -> Path:
        return self.root / ANNOTATIONS_FILE

    @property
    def sampling_log(self) -> Path:
        return self.root / SAMPLING_LOG_FILE

    @property
    def readme(self) -> Path:
        return self.root / README_FILE

    # ── folders ──────────────────────────────────────────────────────────
    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis"

    @property
    def internal(self) -> Path:
        return self.root / INTERNAL_DIRNAME

    @property
    def snapshots(self) -> Path:
        return self.internal / "snapshots"

    @property
    def cache(self) -> Path:
        return self.root / CACHE_DIRNAME

    # Conventional sub-locations, so two panels that both want "the folder for
    # stage 2" cannot disagree about its name.
    def train_run(self, stage: int) -> Path:
        return self.runs / "train" / f"stage{int(stage)}"

    def eval_run(self, stage: int) -> Path:
        return self.runs / "val" / f"stage{int(stage)}"

    @property
    def predictions(self) -> Path:
        return self.analysis / "predict"

    @property
    def reports(self) -> Path:
        return self.analysis / "reports"

    @property
    def splits(self) -> Path:
        return self.datasets / "splits"

    @property
    def frames(self) -> Path:
        return self.datasets / "frames"

    @property
    def summary(self) -> Path:
        return self.datasets / "summary"

    # ── operations ───────────────────────────────────────────────────────
    def create(self) -> ProjectLayout:
        """Make the folder skeleton. Idempotent, and never touches content."""
        self.root.mkdir(parents=True, exist_ok=True)
        for _attribute, name, _description in FOLDERS:
            (self.root / name).mkdir(exist_ok=True)
        self.snapshots.mkdir(parents=True, exist_ok=True)
        return self

    def write_readme(self, name: str) -> None:
        """Explain the layout inside the folder, where it is found months later."""
        title = f"YOAKE project: {name}"
        entries = "".join(
            f"  {folder + '/':<20} {description}\n" for _a, folder, description in FOLDERS
        )
        text = _README.format(
            name=name,
            underline="=" * len(title),
            # The bare file name, not this folder's current path: an
            # absolute path printed into the folder is wrong the first
            # time anyone does what the paragraph above tells them to.
            project_file=PROJECT_FILE,
            annotations=ANNOTATIONS_FILE,
            manifest=PROJECT_FILE,
            entries=entries,
            internal=INTERNAL_DIRNAME + "/",
            cache=CACHE_DIRNAME + "/",
        )
        try:
            self.readme.write_text(text, encoding="utf-8")
        except OSError:
            pass  # A README is a courtesy; a read-only folder must not fail a save.

    def contains(self, path: Path | str | None) -> bool:
        if path is None:
            return False
        try:
            Path(path).expanduser().resolve().relative_to(self.root)
        except (ValueError, OSError):
            return False
        return True

    def relative(self, path: Path | str | None) -> str:
        return portable_path(path, self.root)

    def resolve(self, value: str) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def exists(self) -> bool:
        return self.project_file.is_file()


# ══════════════════════════════════════════════════════════════════════════
# path policy
# ══════════════════════════════════════════════════════════════════════════
def portable_path(path: Path | str | None, base: Path) -> str:
    """The form written to disk: relative to ``base`` whenever that is expressible."""
    if path is None or path == "":
        return ""
    try:
        return Path(os.path.relpath(Path(path), Path(base))).as_posix()
    except ValueError:  # Different Windows drives cannot have a relative path.
        return str(path)


def anchor_path(path: Path | str | None, base: Path) -> str:
    """The absolute fallback, recorded only for files *outside* the project.

    A file inside the project needs no anchor: its relative path is always
    right, and an absolute one would be a second answer that can go stale.
    """
    if path is None or path == "":
        return ""
    resolved = Path(path)
    try:
        resolved.resolve().relative_to(Path(base).resolve())
    except (ValueError, OSError):
        return str(resolved)
    return ""


def locate_media(name: str, roots: Iterable[Path], *, limit: int = 20000) -> Path | None:
    """Find a file called ``name`` under any of ``roots``. Shallow places first.

    Bounded by ``limit`` entries per root, so a project sitting next to a
    multi-terabyte capture share cannot turn "open project" into a disk scan.
    """
    if not name:
        return None
    skip = {INTERNAL_DIRNAME, CACHE_DIRNAME, "__pycache__"}
    for raw in roots:
        root = Path(raw)
        if not root.is_dir():
            continue
        for direct in (root / name, root / "videos" / name):
            if direct.is_file():
                return direct
        seen = 0
        try:
            for path in root.rglob(name):
                seen += 1
                if seen > limit:
                    break
                parts = path.relative_to(root).parts[:-1]
                if any(part in skip or part.startswith(".") for part in parts):
                    continue
                if path.is_file():
                    return path
        except OSError:
            continue
    return None


def resolve_media(
    relative: str,
    absolute: str,
    root: Path,
    *,
    search_roots: Sequence[Path] = (),
) -> tuple[Path | None, str]:
    """Find a capture again after the project, the capture, or both have moved.

    Returns ``(path, how)`` where ``how`` is ``"relative"``, ``"absolute"``,
    ``"found"`` (a name search succeeded — the caller should record the new
    location) or ``"missing"``. A missing file still comes back with the best
    guess, so the UI can name what it was looking for.
    """
    root = Path(root)
    candidates: list[tuple[Path, str]] = []
    if relative:
        near = Path(relative)
        if near.is_absolute():
            candidates.append((near, "absolute"))
        else:
            candidates.append(((root / near), "relative"))
    if absolute:
        candidates.append((Path(absolute), "absolute"))
    for candidate, how in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve(), how
        except OSError:
            continue
    found = locate_media(Path(relative or absolute).name, [root, *search_roots])
    if found is not None:
        return found.resolve(), "found"
    # Normalised, because the best guess is what the UI shows the user when it
    # asks them to find the file, and "project/../../captures/a.mp4" is not an
    # answer to "where should I look".
    guess = Path(os.path.normpath(candidates[0][0])) if candidates else None
    return guess, "missing"


# ══════════════════════════════════════════════════════════════════════════
# discovery
# ══════════════════════════════════════════════════════════════════════════
def is_project_root(path: Path | str) -> bool:
    return (Path(path) / PROJECT_FILE).is_file()


def find_project(start: Path | str) -> Path | None:
    """The nearest project manifest at or above ``start``.

    Nearest, not outermost: an exported dataset inside ``datasets/`` carries its
    own manifest, and someone who opens a file in there means *that* copy.
    """
    path = Path(start).expanduser().resolve()
    if path.is_file():
        if path.name == PROJECT_FILE:
            return path
        path = path.parent
    for folder in (path, *path.parents):
        manifest = folder / PROJECT_FILE
        if manifest.is_file():
            return manifest
    return None


def project_folder_name(name: str) -> str:
    """A usable folder name from whatever the user typed in the name box."""
    cleaned = "".join("_" if c in '<>:"/\\|?*' else c for c in str(name)).strip(" .")
    return cleaned or "untitled"
