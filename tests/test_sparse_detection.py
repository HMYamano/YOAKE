"""Running the detector on every Nth frame instead of all of them.

For active learning, a dense pass is waste: the sampler can only ever choose a
frame that was *scanned*, so detections on the fourteen frames between two scan
positions are computed, cached and never read. Measured on this machine at 121
frames/s, twenty one-hour videos cost 4.1 hours dense and 17 minutes at stride
15.

Sparse prediction brings three ways to be silently wrong, and there is a test
for each:

1. **The frames come back on the wrong index.** ``yoake predict`` re-indexes an
   image directory from 0, so a candidate from local index 3 belongs on real
   frame 45 and nothing in the output says so.
2. **The exported JPEG is one frame early.** ``grab()`` through the target and
   then ``retrieve()`` returns the frame the *last* grab consumed, not the
   target. It agrees at frame 0, so the error survives any test that only
   checks frame 0 or that tolerates a frame of slack. The first version of
   ``write_scan_frames`` had exactly this bug.
3. **The checks report on frames nobody ran a model on.** ``quality.analyze``
   treats every frame it is given as one it has an opinion about, so passing
   the whole video's range after a stride-15 pass declares fourteen frames in
   fifteen to be places the detector found nothing.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from htrtdetr.annotation import batch as B  # noqa: E402
from htrtdetr.annotation import quality  # noqa: E402
from htrtdetr.gui import auto_label  # noqa: E402

W, H, N = 320, 240, 200
#: Each frame states its index as the size of one white rectangle, split over
#: two axes so the signature stays coarse without running out of pixels:
#: width carries ``index % 16``, height carries ``index // 16``. A neighbouring
#: frame therefore differs by a whole 20 px column, which JPEG cannot blur
#: away - a one-row-per-index bar would put the off-by-one this file exists for
#: inside the tolerance the JPEG itself needs, and a bar that grows every frame
#: saturates before frame 30.
COL, ROW = 20, 16


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """Each frame states its own index in its pixels."""
    path = tmp_path_factory.mktemp("sparse") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (W, H))
    for index in range(N):
        frame = np.zeros((H, W, 3), np.uint8)
        cv2.rectangle(
            frame, (0, 0),
            ((index % 16 + 1) * COL, (index // 16 + 1) * ROW),
            (255, 255, 255), -1,
        )
        writer.write(frame)
    writer.release()
    return path


def _index_of(image) -> int:
    """Read the frame index back out of the pixels."""
    plane = image[:, :, 0]
    rows = int((plane.max(axis=1) > 127).sum())
    cols = int((plane.max(axis=0) > 127).sum())
    return (int(round(rows / ROW)) - 1) * 16 + (int(round(cols / COL)) - 1)


def test_the_exported_frame_is_the_frame_that_was_asked_for(video, tmp_path):
    """The off-by-one. grab-through-then-retrieve returns target-1, and it
    agrees at frame 0, which is where anyone would look first."""
    plan = auto_label.export_plan(0, -1, N, "scan", stride=15)
    folder = tmp_path / "frames"
    written = B.write_scan_frames(
        B.VideoInput(video_id="clip", path=video, num_frames=N),
        plan, folder, threading.Event(),
    )
    assert written == len(plan.frames)

    wrong = []
    for local, real in enumerate(plan.frames):
        image = cv2.imread(str(folder / f"{local:06d}.jpg"))
        assert image is not None, f"{local:06d}.jpg was not written"
        found = _index_of(image)
        if found != real:
            wrong.append((local, real, found))
    assert not wrong, f"exported the wrong frame (local, asked, got): {wrong[:5]}"


def test_the_local_numbering_starts_at_zero_and_is_contiguous(video, tmp_path):
    """predict re-indexes an image directory from 0; the files have to match
    that or the index map lines up with nothing."""
    plan = auto_label.export_plan(0, -1, N, "scan", stride=15)
    folder = tmp_path / "frames"
    B.write_scan_frames(
        B.VideoInput(video_id="clip", path=video, num_frames=N),
        plan, folder, threading.Event(),
    )
    names = sorted(p.name for p in folder.glob("*.jpg"))
    assert names == [f"{i:06d}.jpg" for i in range(len(plan.frames))]


def test_cancelling_the_export_stops_it(video, tmp_path):
    cancel = threading.Event()
    cancel.set()
    plan = auto_label.export_plan(0, -1, N, "scan", stride=15)
    with pytest.raises(B.Cancelled):
        B.write_scan_frames(
            B.VideoInput(video_id="clip", path=video, num_frames=N),
            plan, tmp_path / "frames", cancel,
        )


@pytest.mark.parametrize("failure", ["read", "encode", "grab"])
def test_a_failed_scan_frame_cannot_shift_later_predictions(monkeypatch, tmp_path, failure):
    """A gap in filenames is collapsed by the image-directory predictor."""
    class Capture:
        position = 0
        released = False

        def isOpened(self):
            return True

        def grab(self):
            self.position += 1
            return failure != "grab"

        def read(self):
            index = self.position
            self.position += 1
            if failure == "read" and index == 2:
                return False, None
            return True, np.full((8, 8, 3), index, np.uint8)

        def release(self):
            self.released = True

    cap = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: cap)
    encode = cv2.imencode

    def imencode(ext, frame):
        if failure == "encode" and frame[0, 0, 0] == 2:
            return False, None
        return encode(ext, frame)

    monkeypatch.setattr(cv2, "imencode", imencode)
    plan = auto_label.export_plan(0, -1, 6, "scan", stride=2)
    with pytest.raises(OSError, match="frame"):
        B.write_scan_frames(
            B.VideoInput(video_id="clip", path=Path("clip.mp4"), num_frames=6),
            plan, tmp_path / "frames", threading.Event(),
        )
    assert cap.released


# ── the remap ────────────────────────────────────────────────────────────

def test_candidates_are_moved_back_onto_real_frame_numbers():
    """Without this every candidate from a strided run lands on the wrong
    frame - and each one is plausible, just in the wrong place."""
    plan = auto_label.export_plan(0, -1, N, "scan", stride=15)
    local = {i: [auto_label.Candidate([0, 0, 10, 10], score=0.9)] for i in range(4)}
    remapped = auto_label.remap_frame_indices(local, plan.index_map)
    assert sorted(remapped) == [0, 15, 30, 45]


def test_a_dense_run_is_untouched_by_the_remap():
    """An empty map is the identity, so the dense path pays nothing."""
    dense = {i: [auto_label.Candidate([0, 0, 10, 10], score=0.9)] for i in range(3)}
    assert sorted(auto_label.remap_frame_indices(dense, {})) == [0, 1, 2]


# ── the checks only speak about frames that were examined ────────────────

def _candidate(score=0.9):
    return auto_label.Candidate([10, 10, 40, 40], score=score, track_id=1, class_id=0)


def test_the_checks_do_not_report_on_frames_nobody_predicted():
    """Passing range(num_frames) after a stride-15 pass declares fourteen
    frames in fifteen to be places the detector found nothing."""
    # expected_animals is what turns the count detectors on: with it set, a
    # frame holding nothing is a review item. That is the right behaviour for
    # a frame the detector looked at, and nonsense for one it never saw.
    config = quality.QualityConfig(expected_animals=1)
    candidates = {f: [_candidate()] for f in range(0, 150, 15)}

    everywhere = B.check_video(candidates, {}, config, 150)
    examined = B.check_video(candidates, {}, config, 150, frames=sorted(candidates))
    assert len(examined) < len(everywhere), (
        "the examined-frames list made no difference, so the checks are still "
        "reporting on frames that were never predicted"
    )


def test_the_default_still_covers_the_whole_video():
    """A dense run must keep reporting on every frame, including empty ones."""
    config = quality.QualityConfig(expected_animals=1)
    candidates = {0: [_candidate()]}
    default = B.check_video(candidates, {}, config, 50)
    explicit = B.check_video(candidates, {}, config, 50, frames=range(50))
    assert [i.kind for i in default] == [i.kind for i in explicit]
    assert default, "a dense run must still report the frames with nothing in them"


# ── the options that drive it ────────────────────────────────────────────

def test_detect_stride_defaults_to_dense():
    assert B.BatchOptions().detect_stride == 1


def test_the_cache_key_changes_with_the_stride():
    """Otherwise a sparse run would serve its candidates to a dense one."""
    from htrtdetr.annotation.cache import CacheKey

    def key(stride):
        return CacheKey.for_video(
            Path("v.mp4"), model="m", path="v.mp4", weights="w.pth",
            config="", threshold=0.3, device="cuda", stride=stride,
            operation="project_detection_v1",
        )

    assert key(1).digest() != key(15).digest()
