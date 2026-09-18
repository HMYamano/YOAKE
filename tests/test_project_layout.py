"""The project folder, and the one promise it makes: it survives being moved.

Four things can move independently — the project folder, the captures, both,
or neither — and each has a different right answer. These tests drive all four
against real files rather than asserting on the stored strings, because the
stored string is not the guarantee; reopening the study is.

Torch-free and dearpygui-free except where a test says otherwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from htrtdetr.annotation.layout import (
    FOLDERS,
    PROJECT_FILE,
    ProjectLayout,
    find_project,
    is_project_root,
    portable_path,
    resolve_media,
)
from htrtdetr.annotation.objects import AnnObject
from htrtdetr.annotation.project import ProjectSession


def make_video(path: Path, payload: bytes = b"video placeholder") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


@pytest.fixture
def study(tmp_path):
    """A project outside, with its captures in a sibling folder it does not own."""
    captures = tmp_path / "captures"
    for name in ("a.mp4", "b.mp4"):
        make_video(captures / name)
    project = ProjectSession.create(tmp_path / "studies", "courtship 2026")
    for name in ("a.mp4", "b.mp4"):
        video = project.add_video(captures / name)
        video.annotations[0] = [AnnObject([1, 2, 11, 12], track_id=1)]
    project.save(project.layout.project_file)
    return project


# ══════════════════════════════════════════════════════════════════════════
# the folder
# ══════════════════════════════════════════════════════════════════════════
def test_create_makes_the_whole_layout_and_explains_itself(tmp_path):
    project = ProjectSession.create(tmp_path, "my study")
    root = project.layout.root
    assert root == (tmp_path / "my study").resolve()
    assert is_project_root(root)
    for _attribute, folder, _description in FOLDERS:
        assert (root / folder).is_dir(), folder
    assert (root / "annotations.json").is_file()
    readme = (root / "README.txt").read_text(encoding="utf-8")
    assert "my study" in readme
    # The README must stay true after a move, so it may not name its own path.
    assert str(root) not in readme
    for _a, folder, _d in FOLDERS:
        assert f"{folder}/" in readme


def test_create_refuses_to_overwrite_an_existing_project(tmp_path):
    ProjectSession.create(tmp_path, "study")
    with pytest.raises(FileExistsError):
        ProjectSession.create(tmp_path, "study")


def test_saving_an_existing_project_is_idempotent_and_keeps_content(study):
    marker = study.layout.runs / "train" / "stage1" / "results.csv"
    marker.parent.mkdir(parents=True)
    marker.write_text("epoch,loss\n", encoding="utf-8")
    study.save(study.layout.project_file)
    assert marker.read_text(encoding="utf-8") == "epoch,loss\n"


def test_find_project_returns_the_nearest_manifest(study):
    nested = study.layout.datasets / "frames"
    nested.mkdir(parents=True)
    ProjectSession.create(study.layout.datasets, "frames-export")
    assert find_project(study.layout.runs) == study.layout.project_file
    inner = study.layout.datasets / "frames-export"
    assert find_project(inner / "annotations.json") == inner / PROJECT_FILE


# ══════════════════════════════════════════════════════════════════════════
# the move
# ══════════════════════════════════════════════════════════════════════════
def test_moving_the_project_alone_keeps_external_captures(study, tmp_path):
    moved = tmp_path / "moved here"
    shutil.move(str(study.layout.root), moved)
    reopened = ProjectSession.open(moved)
    assert [v.missing for v in reopened.videos.values()] == [False, False]
    assert reopened.videos["a"].annotations[0][0].bbox == [1, 2, 11, 12]


def test_moving_the_whole_tree_keeps_working_without_the_old_paths(study, tmp_path):
    # Copy the project and the captures together, keeping their arrangement,
    # then break the originals: only the relative path can answer this.
    elsewhere = tmp_path.parent / (tmp_path.name + "_copied")
    shutil.rmtree(elsewhere, ignore_errors=True)
    shutil.copytree(tmp_path, elsewhere)
    shutil.move(str(tmp_path / "captures"), tmp_path / "captures gone")

    reopened = ProjectSession.open(elsewhere / "studies" / "courtship 2026")
    assert [v.missing for v in reopened.videos.values()] == [False, False]
    assert reopened.videos["a"].path == (elsewhere / "captures" / "a.mp4").resolve()


def test_captures_inside_the_folder_survive_a_move_to_another_root(tmp_path):
    project = ProjectSession.create(tmp_path / "here", "inside")
    layout = project.layout
    project.add_video(make_video(layout.videos / "a.mp4"))
    project.save(layout.project_file)
    raw = json.loads(layout.project_file.read_text(encoding="utf-8"))
    # Inside the folder, the relative path is the only answer, and the absolute
    # one is deliberately absent: a second answer is a second thing to go stale.
    assert raw["videos"][0]["path"] == "videos/a.mp4"
    assert raw["videos"][0]["abs_path"] == ""
    moved = tmp_path / "far" / "away" / "inside"
    moved.parent.mkdir(parents=True)
    shutil.copytree(layout.root, moved)
    shutil.rmtree(layout.root)
    reopened = ProjectSession.open(moved / PROJECT_FILE)
    assert not reopened.videos["a"].missing
    assert reopened.videos["a"].path == (moved / "videos" / "a.mp4").resolve()


def test_both_ends_moved_is_reported_then_fixed_in_one_answer(study, tmp_path):
    archive = tmp_path / "archive"
    shutil.move(str(tmp_path / "captures"), archive)
    moved = tmp_path / "relocated"
    shutil.move(str(study.layout.root), moved)

    broken = ProjectSession.open(moved)
    assert [v.video_id for v in broken.videos.values() if v.missing] == ["a", "b"]
    # A missing video keeps its annotations: the labels are the expensive part.
    assert broken.videos["a"].annotations[0]

    assert broken.relink_missing(archive) == ["a", "b"]
    assert not any(v.missing for v in broken.videos.values())
    broken.save(broken.layout.project_file)

    # The folder that answered once answers for every later open, including
    # after the captures move again inside it.
    reopened = ProjectSession.open(moved)
    assert not any(v.missing for v in reopened.videos.values())
    assert str(archive) in [str(p) for p in reopened.search_roots]


def test_a_remembered_search_root_relocates_on_the_next_open(study, tmp_path):
    archive = tmp_path / "archive"
    shutil.move(str(tmp_path / "captures"), archive)
    project = ProjectSession.open(study.layout.root)
    project.relink_missing(archive)
    project.save(project.layout.project_file)

    # The captures move again, inside the folder the user already pointed at.
    (archive / "2026-09").mkdir()
    for name in ("a.mp4", "b.mp4"):
        shutil.move(str(archive / name), archive / "2026-09" / name)

    reopened = ProjectSession.open(study.layout.root)
    assert reopened.relocated == ["a", "b"]
    assert not any(v.missing for v in reopened.videos.values())


def test_import_media_makes_the_folder_self_contained(study, tmp_path):
    imported, failures = study.import_media()
    assert (imported, failures) == (2, [])
    study.save(study.layout.project_file)
    assert sorted(p.name for p in study.layout.videos.glob("*.mp4")) == ["a.mp4", "b.mp4"]
    assert study.import_media() == (0, [])  # nothing left outside: a no-op

    shutil.rmtree(tmp_path / "captures")
    moved = tmp_path / "carried away"
    shutil.move(str(study.layout.root), moved)
    reopened = ProjectSession.open(moved)
    assert not any(v.missing for v in reopened.videos.values())


def test_import_media_never_overwrites_a_same_named_capture(tmp_path):
    project = ProjectSession.create(tmp_path, "study")
    layout = project.layout
    make_video(layout.videos / "a.mp4", b"the one already here")
    project.add_video(make_video(tmp_path / "elsewhere" / "a.mp4", b"the imported one"))
    project.import_media()
    assert (layout.videos / "a.mp4").read_bytes() == b"the one already here"
    assert (layout.videos / "a_2.mp4").read_bytes() == b"the imported one"


# ══════════════════════════════════════════════════════════════════════════
# the path policy, directly
# ══════════════════════════════════════════════════════════════════════════
def test_resolution_prefers_the_relative_answer_over_a_stale_absolute(tmp_path):
    root = tmp_path / "project"
    (root / "videos").mkdir(parents=True)
    near = make_video(root / "videos" / "a.mp4", b"the copy that travelled")
    far = make_video(tmp_path / "old" / "a.mp4", b"the file left behind")
    found, how = resolve_media("videos/a.mp4", str(far), root)
    assert (found, how) == (near.resolve(), "relative")


def test_resolution_falls_back_to_the_absolute_path_then_to_a_search(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    far = make_video(tmp_path / "captures" / "a.mp4")
    assert resolve_media("../captures/a.mp4", str(far), root)[1] == "relative"
    assert resolve_media("nowhere/a.mp4", str(far), root)[1] == "absolute"
    assert resolve_media("nowhere/a.mp4", "", root, search_roots=[far.parent])[1] == "found"
    missing, how = resolve_media("nowhere/a.mp4", "", root)
    assert how == "missing" and ".." not in str(missing)


def test_portable_path_is_posix_and_relative(tmp_path):
    assert portable_path(tmp_path / "videos" / "a.mp4", tmp_path) == "videos/a.mp4"
    assert portable_path(None, tmp_path) == ""


def test_a_manifest_without_the_new_keys_still_opens(study):
    manifest = study.layout.project_file
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    for row in raw["videos"]:
        row.pop("abs_path", None)
    raw.pop("identity", None)
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    reopened = ProjectSession.open(manifest)
    assert not any(v.missing for v in reopened.videos.values())


# ══════════════════════════════════════════════════════════════════════════
# the places the output goes
# ══════════════════════════════════════════════════════════════════════════
def test_layout_names_every_output_location_under_the_root(tmp_path):
    layout = ProjectLayout(tmp_path / "study")
    for path in (
        layout.datasets,
        layout.splits,
        layout.frames,
        layout.summary,
        layout.runs,
        layout.train_run(2),
        layout.eval_run(2),
        layout.analysis,
        layout.predictions,
        layout.reports,
        layout.cache,
        layout.snapshots,
    ):
        assert layout.contains(path), path
    assert layout.train_run(2).name == "stage2"
    assert not layout.contains(tmp_path / "somewhere else")


def test_training_and_analysis_pages_write_inside_the_project(tmp_path):
    pytest.importorskip("dearpygui.dearpygui")
    from htrtdetr.gui.analysis_panel import AnalysisPanel
    from htrtdetr.gui.training_panel import TrainingPanel

    project = ProjectSession.create(tmp_path, "study").layout.root
    repo = Path(__file__).resolve().parents[1]
    training = TrainingPanel(repo, project)
    analysis = AnalysisPanel(repo, project)
    layout = ProjectLayout(project)

    defaults = training._defaults()
    assert defaults["tr_out"] == str(layout.train_run(1))
    assert defaults["ds_split_out"] == str(layout.splits)
    assert defaults["tr_train"] == str(layout.splits / "annotations_train.json")
    assert defaults["an_out"] == str(layout.reports)
    assert analysis._default_outdir() == str(layout.predictions)
    for value in defaults.values():
        assert not value or layout.contains(value), value

    # Without a project the old behaviour is unchanged: the source checkout.
    plain = TrainingPanel(repo)
    assert plain.runs == repo / "runs"
    assert AnalysisPanel(repo)._default_outdir() == str(repo / "runs" / "predict")


# ══════════════════════════════════════════════════════════════════════════
# headless
# ══════════════════════════════════════════════════════════════════════════
def run_cli(*args, cwd: Path):
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "htrtdetr.cli", "project", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**__import__("os").environ, "PYTHONPATH": str(repo / "src")},
    )


def test_cli_creates_inspects_and_repairs_a_project(tmp_path):
    captures = tmp_path / "captures"
    for name in ("a.mp4", "b.mp4"):
        make_video(captures / name)

    created = run_cli("new", f"dir={tmp_path}", "name=study", f"videos={captures}", cwd=tmp_path)
    assert created.returncode == 0, created.stderr
    root = tmp_path / "study"
    assert is_project_root(root)

    info = run_cli("info", f"path={root}", cwd=tmp_path)
    assert info.returncode == 0, info.stdout
    assert "2" in info.stdout and "missing" not in info.stdout

    archive = tmp_path / "archive"
    shutil.move(str(captures), archive)
    broken = run_cli("info", f"path={root}", cwd=tmp_path)
    assert broken.returncode == 1
    assert "relink" in broken.stdout

    fixed = run_cli("relink", f"path={root}", f"from={archive}", cwd=tmp_path)
    assert fixed.returncode == 0, fixed.stdout
    assert run_cli("info", f"path={root}", cwd=tmp_path).returncode == 0

    collected = run_cli("collect", f"path={root}", cwd=tmp_path)
    assert collected.returncode == 0, collected.stdout
    assert sorted(p.name for p in (root / "videos").glob("*.mp4")) == ["a.mp4", "b.mp4"]
