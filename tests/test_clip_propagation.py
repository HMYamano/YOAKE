"""Clip-level propagation: one export, every animal in it.

``Continue`` follows one animal from one prompt. A clip is the span Smart
Annotate already proposed and a human already agreed to look at, and it usually
holds every animal at once - so following them one at a time re-exported and
re-encoded the same frames per animal, and the encode is the whole cost of a
propagation.

Measured on this machine (RTX 5070 Ti, sam2-hiera-tiny, 40 frames, two
animals): 5.98 s for the first animal against 3.09 s for the second over the
same export, because the helper caches the inference state per ``frames_dir``.
The saving is entirely in exporting once, so what these tests pin hardest is
that every request for a clip reports the **same window**.
"""

from __future__ import annotations

import pytest

from htrtdetr.annotation.objects import AnnObject
from htrtdetr.gui import video_segment as vs


def obj(x0, y0, x1, y1, **kw):
    return AnnObject([x0, y0, x1, y1], **kw)


# ── clip_window ──────────────────────────────────────────────────────────

def test_a_clip_is_its_own_window():
    assert vs.clip_window(10, 40, 100) == (10, 40)


def test_a_backwards_clip_is_read_as_the_span_it_covers():
    """The caller is often a drag, and a drag can go either way."""
    assert vs.clip_window(40, 10, 100) == (10, 40)


def test_the_window_is_clamped_to_the_video():
    assert vs.clip_window(90, 400, 100) == (90, 99)


def test_the_window_is_clamped_to_max_window():
    first, last = vs.clip_window(0, vs.MAX_WINDOW * 3, 0)
    assert (first, last) == (0, vs.MAX_WINDOW - 1)


def test_a_negative_start_is_clamped_not_rejected():
    assert vs.clip_window(-20, 30, 100) == (0, 30)


# ── clip_prompt_frame ────────────────────────────────────────────────────

def test_the_default_prompt_is_the_middle():
    """SAM 2 goes one way from the prompt, so the middle shortens the longer run."""
    assert vs.clip_prompt_frame(10, 40, None, 100) == 25


def test_a_preferred_frame_is_honoured():
    assert vs.clip_prompt_frame(10, 40, 33, 100) == 33


def test_a_preferred_frame_outside_the_clip_is_clamped_in():
    assert vs.clip_prompt_frame(10, 40, 500, 100) == 40
    assert vs.clip_prompt_frame(10, 40, -5, 100) == 10


# ── clip_seed_frame: which frame's boxes seed the run ────────────────────

def test_the_fullest_frame_wins():
    """Seeding from a frame with one of two flies loses the other all clip.

    Nothing in the result would say why, which is what makes this the rule
    rather than "use the middle".
    """
    annotated = {12: [obj(0, 0, 5, 5)], 30: [obj(0, 0, 5, 5), obj(9, 9, 14, 14)]}
    assert vs.clip_seed_frame(10, 40, annotated, num_frames=100) == 30


def test_a_tie_goes_to_the_frame_nearest_the_middle():
    annotated = {11: [obj(0, 0, 5, 5)], 24: [obj(0, 0, 5, 5)], 39: [obj(0, 0, 5, 5)]}
    assert vs.clip_seed_frame(10, 40, annotated, num_frames=100) == 24


def test_the_preferred_frame_wins_only_when_it_is_not_beaten():
    two = [obj(0, 0, 5, 5), obj(9, 9, 14, 14)]
    annotated = {15: [obj(0, 0, 5, 5)], 30: two}
    # Looking at 15, which has fewer animals: the fuller frame wins.
    assert vs.clip_seed_frame(10, 40, annotated, preferred=15, num_frames=100) == 30
    # Looking at a frame that ties the best: stay where the human is.
    annotated[15] = list(two)
    assert vs.clip_seed_frame(10, 40, annotated, preferred=15, num_frames=100) == 15


def test_boxes_outside_the_clip_do_not_seed_it():
    annotated = {5: [obj(0, 0, 5, 5), obj(9, 9, 14, 14)], 20: [obj(0, 0, 5, 5)]}
    assert vs.clip_seed_frame(10, 40, annotated, num_frames=100) == 20


def test_no_boxes_in_the_clip_is_none_not_frame_zero():
    """A different answer, so the caller can say something different."""
    assert vs.clip_seed_frame(10, 40, {}, num_frames=100) is None
    assert vs.clip_seed_frame(10, 40, {20: []}, num_frames=100) is None


# ── prompts_from_objects ─────────────────────────────────────────────────

def test_each_box_becomes_a_box_and_a_centre_point():
    (prompt,) = vs.prompts_from_objects([obj(10, 20, 30, 60, track_id=4)])
    assert prompt.box == (10.0, 20.0, 30.0, 60.0)
    assert prompt.points == ((20.0, 40.0, 1),)
    assert prompt.track_id == 4


def test_class_and_action_ride_along():
    (prompt,) = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=1, action_id=3)])
    assert prompt.action_id == 3
    assert prompt.class_id == 0


def test_an_untracked_box_gets_an_id_that_does_not_collide():
    """The result is a tracklet, and a tracklet needs an identity to accept."""
    prompts = vs.prompts_from_objects(
        [obj(0, 0, 4, 4, track_id=1), obj(5, 5, 9, 9), obj(10, 10, 14, 14)],
        fallback_track=1,
    )
    ids = [p.track_id for p in prompts]
    assert ids[0] == 1, "an existing id must never be renumbered"
    assert len(set(ids)) == 3, f"ids collided: {ids}"
    assert 1 not in ids[1:]


# ── clip_requests: one window for all of them ────────────────────────────

def test_every_request_reports_the_same_window():
    """The whole saving is exporting once; it evaporates if these disagree."""
    prompts = vs.prompts_from_objects(
        [obj(0, 0, 4, 4, track_id=1), obj(5, 5, 9, 9, track_id=2)]
    )
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    assert len(requests) == 2
    assert {r.window(100) for r in requests} == {(10, 40)}
    assert {r.frame for r in requests} == {25}


def test_an_unpromptable_entry_is_dropped_rather_than_sent():
    """A request with no point and no box comes back empty, reading as a loss."""
    requests = vs.clip_requests(
        10, 40, [vs.ClipPrompt(track_id=1), vs.ClipPrompt(track_id=2, box=(1, 2, 3, 4))]
    )
    assert [r.track_id for r in requests] == [2]


def test_requests_carry_the_track_class_and_action_through():
    prompts = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=9, action_id=2)])
    (request,) = vs.clip_requests(10, 40, prompts, num_frames=100)
    assert (request.track_id, request.class_id, request.action_id) == (9, 0, 2)


def test_a_clip_shorter_than_the_default_window_is_not_widened():
    """Continue uses +/-100; a clip must stay the clip, or it stops being one."""
    prompts = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=1)])
    (request,) = vs.clip_requests(50, 60, prompts, num_frames=1000)
    assert request.window(1000) == (50, 60)
    assert request.before + request.after == 10


# ── describe_clip_run ────────────────────────────────────────────────────

def result(frames):
    boxes = tuple((f, [1.0, 2.0, 3.0, 4.0, 0.0], [1.0, 2.0, 3.0, 4.0]) for f in frames)
    return vs.PropagationResult(boxes=boxes, ok=True, window=len(frames))


def test_the_summary_counts_animals_before_boxes():
    prompts = vs.prompts_from_objects(
        [obj(0, 0, 4, 4, track_id=1), obj(5, 5, 9, 9, track_id=2)]
    )
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    pairs = [(requests[0], result(range(10, 41))), (requests[1], result(range(10, 41)))]
    text = vs.describe_clip_run(pairs)
    assert "2 animal(s)" in text
    assert "frames 10-40" in text
    assert "62" in text, "the box total should still be there"
    assert "candidates until you accept" in text


def test_an_animal_the_model_never_found_is_named():
    prompts = vs.prompts_from_objects(
        [obj(0, 0, 4, 4, track_id=1), obj(5, 5, 9, 9, track_id=7)]
    )
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    text = vs.describe_clip_run([(requests[0], result([10, 11])), (requests[1], None)])
    assert "Track 7" in text
    assert "1 animal(s)" in text


def test_finding_nothing_at_all_says_what_to_try():
    prompts = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=1)])
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    text = vs.describe_clip_run([(requests[0], None)])
    assert "none of the 1 animals" in text
    assert "clearly separated" in text


def test_an_empty_run_is_its_own_sentence():
    assert vs.describe_clip_run([]) == "Nothing to follow in this clip."


def test_skipped_clips_survive_to_the_end_of_the_run():
    """A clip with no box is skipped before any work starts.

    Announcing it up front means the first progress update overwrites it
    seconds later, so the count has to ride along to the final summary or it is
    not really reported.
    """
    prompts = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=1)])
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    text = vs.describe_clip_run(
        [(requests[0], result([10, 11]))], clips=2, skipped=3
    )
    assert "3 clip(s) had no box to follow" in text


def test_skipped_is_reported_even_when_nothing_was_followed():
    prompts = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=1)])
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    text = vs.describe_clip_run([(requests[0], None)], skipped=2)
    assert "2 clip(s) had no box" in text
    assert vs.describe_clip_run([], skipped=2).endswith(
        "2 clip(s) had no box to follow and were skipped."
    )


def test_multiple_clips_are_reported_as_such():
    prompts = vs.prompts_from_objects([obj(0, 0, 4, 4, track_id=1)])
    requests = vs.clip_requests(10, 40, prompts, num_frames=100)
    text = vs.describe_clip_run([(requests[0], result([10, 11]))], clips=4)
    assert "over 4 clips" in text


# ── _clip_job: the annotator's side, without an annotator ────────────────

class FakeVideo:
    def __init__(self, annotations, num_frames=200):
        self.annotations = annotations
        self.num_frames = num_frames


class FakeApp:
    """Only what _clip_job touches."""

    active_track_id = 1

    def __init__(self):
        from htrtdetr.gui.workspace_operations import WorkspaceOperations

        self._clip_job = WorkspaceOperations._clip_job.__get__(self)


def test_clip_job_builds_one_shared_window_for_every_animal():
    app = FakeApp()
    video = FakeVideo({30: [obj(0, 0, 4, 4, track_id=1), obj(5, 5, 9, 9, track_id=2)]})
    job, why = app._clip_job(video, 10, 40)
    assert why == ""
    first, last, requests = job
    assert (first, last) == (10, 40)
    assert len(requests) == 2
    assert {r.window(200) for r in requests} == {(10, 40)}
    # The prompts were read from frame 30, so that is where they anchor.
    assert {r.frame for r in requests} == {30}


def test_clip_job_reports_an_empty_clip_instead_of_running_one():
    app = FakeApp()
    job, why = app._clip_job(FakeVideo({}), 10, 40)
    assert job is None
    assert "no boxes" in why


# ── selecting a clip from the timeline ───────────────────────────────────

class FakeSelectionApp:
    """Only what _on_timeline_clip and clip_propagation_blocked touch."""

    def __init__(self, clips=(), *, use_track_ids=True, cap=object()):
        from htrtdetr import annotator as A

        self._selection = type("Sel", (), {"clips": tuple(clips)})()
        self._sel_clip = None
        self._sel_tracklet = "stale"
        self._sel_interval = "stale"
        self._sel_issue = "stale"
        self._selected = "stale"
        self.use_track_ids = use_track_ids
        self.cap = cap
        self.refreshed = 0
        self._on_timeline_clip = A.AnnotationApp._on_timeline_clip.__get__(self)
        self.clip_propagation_blocked = (
            A.AnnotationApp.clip_propagation_blocked.__get__(self)
        )

    def _refresh_sidebar(self):
        self.refreshed += 1

    def _render_canvas(self):
        pass


def test_clicking_a_clip_selects_the_real_clip_object():
    """The reason and score come from the Selection, so the identical clip has
    to be found rather than rebuilt from its two numbers."""
    from htrtdetr.annotation import sampling

    wanted = sampling.Clip(start=100, end=300, reason="interaction", score=0.8)
    app = FakeSelectionApp([sampling.Clip(start=5, end=9), wanted])
    app._on_timeline_clip((100, 300))
    assert app._sel_clip is wanted
    assert app._sel_clip.reason == "interaction"


def test_clicking_a_clip_supersedes_every_other_selection():
    from htrtdetr.annotation import sampling

    app = FakeSelectionApp([sampling.Clip(start=100, end=300)])
    app._on_timeline_clip((100, 300))
    assert app._sel_tracklet is None
    assert app._sel_interval is None
    assert app._sel_issue is None
    assert app._selected is None
    assert app.refreshed == 1


def test_a_clip_the_selection_no_longer_has_is_not_dropped():
    """The timeline only draws clips it was given, so a miss means the model
    and the selection drifted - falling through would eat the click."""
    app = FakeSelectionApp([])
    app._on_timeline_clip((100, 300))
    assert app._sel_clip is not None
    assert (app._sel_clip.start, app._sel_clip.end) == (100, 300)


# ── what blocks a clip run, and what does not ────────────────────────────

def test_track_ids_off_blocks_a_clip_run():
    app = FakeSelectionApp(use_track_ids=False)
    assert "track ID" in app.clip_propagation_blocked()


def test_no_video_blocks_a_clip_run():
    app = FakeSelectionApp(cap=None)
    assert app.clip_propagation_blocked() == "Open a video first."


def test_a_stopped_helper_does_not_block_the_button():
    """propagate_clip starts the helper on press, so locking the control over
    a condition that pressing it fixes would be wrong. Same reasoning keeps
    Continue permanently unlocked.
    """
    app = FakeSelectionApp()
    assert app.clip_propagation_blocked() == ""


@pytest.mark.live_model
def test_one_export_serves_two_animals():
    """The correctness half of the measurement: both animals get covered.

    Timing is not asserted - it is a real model on a real device and a
    threshold would be flaky - but the shared ``frames_dir`` is the thing that
    makes the second animal cheap, so this at least proves two animals can be
    followed through one export without interfering.
    """
    import shutil
    import sys
    import tempfile
    from pathlib import Path

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    from htrtdetr.gui import sidecar

    root = Path(__file__).resolve().parents[1]
    n, size = 24, 320
    folder = Path(tempfile.mkdtemp(prefix="yoake-cliptest-"))

    def centres(i):
        return (60 + i * 5, 110), (250 - i * 4, 220)

    for i in range(n):
        frame = np.full((size, size, 3), 205, np.uint8)
        for (cx, cy), angle in zip(centres(i), (20, -55)):
            cv2.ellipse(frame, (cx, cy), (30, 12), angle, 0, 360, (30, 30, 35), -1)
        cv2.imwrite(str(folder / f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    spec = sidecar.SidecarSpec(
        python=sys.executable, repo_root=str(root), backend="sam2",
        model="facebook/sam2-hiera-tiny",
    )
    client = sidecar.SidecarClient(spec)
    try:
        client.start(timeout=300)
    except sidecar.SidecarError as exc:
        pytest.skip(f"sam2 helper would not start: {exc}")

    prompt = n // 2
    try:
        coverage = []
        for obj_id, centre in enumerate(centres(prompt), 1):
            merged = []
            for backward in (False, True):
                request = client.propagate(
                    frames_dir=str(folder), first_frame=0, prompt_frame=prompt,
                    points=[[centre[0], centre[1], 1]], obj_id=obj_id,
                    backward=backward,
                )
                reply = client.wait_for(request, timeout=900)
                assert reply is not None, "the helper must answer"
                merged.append(vs.result_from_reply(reply))
            coverage.append(vs.merge_results(*merged).count)
    finally:
        client.stop()
        shutil.rmtree(folder, ignore_errors=True)

    assert all(c > n * 0.7 for c in coverage), (
        f"both animals should be followed through most of the clip, got {coverage}"
    )
