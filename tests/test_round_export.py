"""Building a round's dataset out of the project, without damaging the project.

Two failures this file exists for, both silent:

1. *File > Save and Export Frames* rewrites the live project - it clears
   ``meta["image_root"]``, repoints every ``image_path`` at the export, and
   saves that back into the project's own ``annotations.json``. Doing that on
   every Train press would leave the user's label file pointing into a folder
   that only exists inside an export.
2. The training dataset opens ``<root>/<image_path>`` with ``strict=False`` and
   substitutes a **black image** on any failure. A wrong image root therefore
   produces a complete run with a falling loss curve and a checkpoint, trained
   on nothing. Nothing in the CLI reports it, so the round has to check before
   it spawns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from htrtdetr.gui import round_export  # noqa: E402

FRAMES, W, H = 24, 96, 72


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    folder = tmp_path_factory.mktemp("round_export")
    path = folder / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    for index in range(FRAMES):
        frame = np.full((H, W, 3), 20, np.uint8)
        cv2.rectangle(frame, (10 + index, 20), (30 + index, 50), (60, 200, 90), -1)
        writer.write(frame)
    writer.release()
    return path


def dataset(frames, image_path="WRONG/somewhere.jpg"):
    return {
        "meta": {"image_root": "a/stale/value"},
        "class_names": ["fly"],
        "videos": [
            {
                "video_id": "clip",
                "frames": [
                    {"frame_index": i, "image_path": image_path, "objects": []}
                    for i in frames
                ],
            }
        ],
    }


# ── writing the frames ───────────────────────────────────────────────────

def test_every_labelled_frame_becomes_a_jpeg(video, tmp_path):
    wanted = (0, 3, 9, 17)
    result = round_export.export_frames(
        [round_export.ExportInput("clip", video, wanted)], tmp_path
    )
    assert result.ok
    assert result.written == len(wanted)
    for index in wanted:
        assert (tmp_path / "images" / "clip" / f"{index:06d}.jpg").exists()


def test_a_second_round_reuses_what_the_first_decoded(video, tmp_path):
    """A round runs every twenty labels. Re-decoding an hour of video each
    time is the difference between a loop that gets used and one that does not."""
    first = round_export.export_frames(
        [round_export.ExportInput("clip", video, (0, 5))], tmp_path
    )
    assert first.written == 2 and first.reused == 0
    second = round_export.export_frames(
        [round_export.ExportInput("clip", video, (0, 5, 11))], tmp_path
    )
    assert second.written == 1, "only the new frame is decoded"
    assert second.reused == 2


def test_a_missing_video_is_reported_not_raised(tmp_path):
    result = round_export.export_frames(
        [round_export.ExportInput("gone", tmp_path / "nope.mp4", (1, 2))], tmp_path
    )
    assert not result.ok
    assert result.missing == [("gone", 1), ("gone", 2)]
    assert "missing" in result.failed[0]


def test_cancelling_stops_and_says_so(video, tmp_path):
    import threading

    cancel = threading.Event()
    cancel.set()
    result = round_export.export_frames(
        [round_export.ExportInput("clip", video, (0, 1, 2))], tmp_path, cancel=cancel
    )
    assert result.cancelled
    assert result.written == 0


# ── the dataset dict ─────────────────────────────────────────────────────

def test_the_paths_are_rewritten_and_the_stale_root_is_cleared():
    data = dataset([0, 4])
    round_export.rewrite_image_paths(data)
    assert data["meta"]["image_root"] == ""
    paths = [f["image_path"] for f in data["videos"][0]["frames"]]
    assert paths == ["images/clip/000000.jpg", "images/clip/000004.jpg"]


def test_rewriting_touches_only_the_dict_it_was_given():
    """The live project must come out of a round exactly as it went in."""
    data = dataset([0])
    other = dataset([0])
    round_export.rewrite_image_paths(data)
    assert other["meta"]["image_root"] == "a/stale/value"
    assert other["videos"][0]["frames"][0]["image_path"] == "WRONG/somewhere.jpg"


def test_a_frame_whose_image_failed_leaves_the_dataset():
    data = dataset([0, 1, 2])
    round_export.rewrite_image_paths(data)
    removed = round_export.drop_frames(data, [("clip", 1)])
    assert removed == 1
    kept = [f["frame_index"] for f in data["videos"][0]["frames"]]
    assert kept == [0, 2]


def test_a_video_that_loses_every_frame_leaves_the_dataset():
    data = dataset([0, 1])
    round_export.drop_frames(data, [("clip", 0), ("clip", 1)])
    assert data["videos"] == []


# ── the guard that makes the whole thing safe ────────────────────────────

def test_a_wrong_image_root_is_caught_before_anything_spawns(video, tmp_path):
    """Without this the run completes on black frames and reports success."""
    data = dataset([0, 1])
    round_export.rewrite_image_paths(data)
    assert len(round_export.missing_images(data, tmp_path)) == 2

    round_export.export_frames(
        [round_export.ExportInput("clip", video, (0, 1))], tmp_path
    )
    assert round_export.missing_images(data, tmp_path) == []


def test_a_frame_with_no_image_path_counts_as_missing():
    data = dataset([0])
    data["videos"][0]["frames"][0]["image_path"] = ""
    assert len(round_export.missing_images(data, Path("."))) == 1


def test_a_project_folder_named_in_japanese_still_exports(video, tmp_path):
    """cv2.imwrite returns False - it does not raise - for any path outside the
    active code page, so on Windows a Japanese project folder silently exported
    nothing and the round refused with "0 frames survived"."""
    folder = tmp_path / "ハエの交尾"
    folder.mkdir()
    result = round_export.export_frames(
        [round_export.ExportInput("clip", video, (0, 4))], folder
    )
    assert result.ok, result.failed
    assert result.written == 2
    assert (folder / "images" / "clip" / "000000.jpg").exists()
